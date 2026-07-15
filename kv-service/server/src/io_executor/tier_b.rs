//! Tier B — io_uring high-performance I/O backend (Linux only).
//!
//! Design points:
//!
//! 1. **Per-device ring**: avoids cross-NVMe head-of-line blocking.
//!    A slow device does not stall SQE submission on other devices.
//!
//! 2. **Batched submit**: read_batch pushes all SQEs into the ring at once,
//!    then issues a single `submit_and_wait(N)` syscall — saves many context
//!    switches compared to N POSIX reads.
//!
//! 3. **Device routing**: pick a ring by longest-prefix match on IORequest.path.
//!    Prefixes are registered via `register_devices()` (typically the router's
//!    device list).
//!
//! 4. **Synchronous surface**: exposes blocking APIs; internally consumed by
//!    an mpsc + worker thread. Tokio callers should wrap in `tokio::task::spawn_blocking`.
//!
//! Limits:
//! - Reads/writes currently use temporary owned buffers (read into Vec, then return).
//!   Could evolve to registered buffers (IORING_REGISTER_BUFFERS) to reduce overhead.
//! - Filesystem paths only (no raw block-device io_uring NVMe passthrough).

use super::{
    log_io_batch, log_io_error, log_io_request, AlignedBuffer, IOExecutor, IORequest, IoBatchStats,
    IoLogContext,
};
use crate::error::{KVError, Result};
use crossbeam_channel as channel;
use io_uring::{opcode, types, IoUring};
use prost::bytes::Bytes;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

const DEFAULT_QUEUE_DEPTH: u32 = 256;

/// O_DIRECT / 4KB alignment — must match tier_a's AlignedBuffer value.
/// (Copied from tier_a::DIRECT_IO_ALIGN; duplicated here to avoid a cross-module pub.)
const O_DIRECT_FLAG: i32 = 0o40000;
const DIRECT_IO_ALIGN: usize = 4096;

/// Per-device worker.
struct DeviceWorker {
    #[allow(dead_code)]
    device_root: PathBuf,
    sender: channel::Sender<RingJob>,
    #[allow(dead_code)]
    handle: Option<thread::JoinHandle<()>>,
}

enum RingJob {
    ReadBatch {
        reqs: Vec<IORequest>,
        resp: channel::Sender<Vec<Result<Vec<u8>>>>,
    },
    /// O_DIRECT batched read: 8-way parallel via io_uring SQE batch + AlignedBuffer.
    /// Returns Bytes (from_owner wrapping AlignedBuffer) so upper layers can zero-copy
    /// slice sub-chunks.
    ReadAlignedBatch {
        reqs: Vec<IORequest>,
        resp: channel::Sender<Vec<Result<Bytes>>>,
    },
    WriteBatch {
        reqs: Vec<(IORequest, Bytes)>,
        resp: channel::Sender<Vec<Result<()>>>,
    },
    /// Vectored write batch: each IO item is (path, Vec<Bytes>). Internally uses one
    /// opcode::Writev SQE per item so the trait default (which concatenates) is bypassed.
    /// Dual to tier_a writev: segments from caller (put_striped_chunks) go straight to
    /// the iovec sink.
    WriteVecBatch {
        reqs: Vec<(IORequest, Vec<Bytes>)>,
        resp: channel::Sender<Vec<Result<()>>>,
    },
    /// O_DIRECT write batch (RDMA PUT data path): data is already in caller-pinned
    /// 4K-aligned memory (typically an RDMA slab extent). Uses opcode::Write SQE batch +
    /// O_DIRECT — **zero memcpy**. io_uring throttles in-flight IO by ring depth, which
    /// coexists nicely with the NVMe-oF target SQ=16 (avoids the burst that tier_a's
    /// sync pwrite could trigger, causing controller reset).
    WriteAlignedBatch {
        job_id: u64,
        queued_at: std::time::Instant,
        reqs: Vec<(IORequest, PtrWrapper, usize)>,
        resp: channel::Sender<Vec<Result<()>>>,
    },
    /// O_DIRECT read batch (RDMA GET cache-miss path): reads directly into caller-pinned
    /// 4K-aligned memory (typically an RDMA slab extent), skipping heap AlignedBuffer and
    /// per-chunk reg_mr.
    /// Fully symmetric to WriteAlignedBatch, so GET can also use the zero-registration
    /// slab fast path.
    /// ptr is *mut u8 (write target); capacity is the max writable bytes (4K-aligned,
    /// ≥ file_size). Returns Vec<Result<usize>>: actual file byte count per item (for
    /// the caller to truncate).
    ReadAlignedIntoPtrBatch {
        job_id: u64,
        queued_at: std::time::Instant,
        reqs: Vec<(IORequest, PtrWrapperMut, usize)>,
        resp: channel::Sender<Vec<Result<usize>>>,
    },
    SingleRead {
        req: IORequest,
        resp: channel::Sender<Result<Vec<u8>>>,
    },
    SingleWrite {
        path: PathBuf,
        data: Bytes,
        resp: channel::Sender<Result<()>>,
    },
    Shutdown,
}

/// Wraps `*const u8` so it can cross a `Send` channel. The caller (RDMA server
/// handle_put) holds the slab extent, guaranteeing memory stays alive until the
/// worker completes the IO. Isomorphic to tier_a's PtrWrapper.
pub(crate) struct PtrWrapper(pub *const u8);
unsafe impl Send for PtrWrapper {}

/// Same as PtrWrapper but mutable (used as the write target ptr for
/// ReadAlignedIntoPtrBatch).
pub(crate) struct PtrWrapperMut(pub *mut u8);
unsafe impl Send for PtrWrapperMut {}

pub struct TierBExecutor {
    /// Sorted by device root path for longest-prefix matching.
    /// (BTreeMap is used only for stable ordering; see route_device for lookup logic.)
    devices: Vec<Arc<DeviceWorker>>,
    /// path prefix -> device index
    prefix_index: Vec<(PathBuf, usize)>,
    job_seq: AtomicU64,
}

impl TierBExecutor {
    /// Create the executor.
    /// - `queue_depth`: SQ/CQ depth per ring
    /// - `num_devices`: expected device count (placeholder; actual devices are added
    ///   via register_device)
    pub fn new(_queue_depth: usize, _num_devices: usize) -> Result<Self> {
        Ok(Self {
            devices: Vec::new(),
            prefix_index: Vec::new(),
            job_seq: AtomicU64::new(1),
        })
    }

    /// Register a device (path prefix). Any subsequent I/O whose path falls under this
    /// prefix is routed to the corresponding ring. Must be called by StorageTier before
    /// the first I/O.
    pub fn register_device(&mut self, root: PathBuf, queue_depth: u32) -> Result<()> {
        let idx = self.devices.len();
        let (tx, rx) = channel::unbounded::<RingJob>();
        let handle = thread::Builder::new()
            .name(format!("uring-{}", idx))
            .spawn(move || ring_worker_loop(idx, rx, queue_depth))
            .map_err(|e| KVError::Internal(format!("spawn uring worker: {}", e)))?;

        let worker = Arc::new(DeviceWorker {
            device_root: root.clone(),
            sender: tx,
            handle: Some(handle),
        });
        self.devices.push(worker);
        self.prefix_index.push((root, idx));
        // Sort by prefix length descending so the longest prefix wins.
        self.prefix_index
            .sort_by(|a, b| b.0.as_os_str().len().cmp(&a.0.as_os_str().len()));
        Ok(())
    }

    fn route_device(&self, path: &Path) -> Result<&Arc<DeviceWorker>> {
        for (prefix, idx) in &self.prefix_index {
            if path.starts_with(prefix) {
                return Ok(&self.devices[*idx]);
            }
        }
        // fallback: first device (avoids panic; upper layers must ensure valid paths)
        self.devices
            .first()
            .ok_or_else(|| KVError::Internal("no devices registered in TierBExecutor".into()))
    }

    /// Group requests by device (keeping the original idx for later scatter).
    fn group_by_device<T: Clone>(&self, items: &[(T, &Path)]) -> Vec<(usize, Vec<(usize, T)>)> {
        let mut groups: BTreeMap<usize, Vec<(usize, T)>> = BTreeMap::new();
        for (orig_idx, (item, path)) in items.iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups
                .entry(device_idx)
                .or_default()
                .push((orig_idx, item.clone()));
        }
        groups.into_iter().collect()
    }
}

impl Drop for TierBExecutor {
    fn drop(&mut self) {
        for d in &self.devices {
            let _ = d.sender.send(RingJob::Shutdown);
        }
        // worker threads exit on Shutdown; JoinHandle dropped here
    }
}

/// Main loop of each per-device worker thread.
/// Owns one io_uring instance and processes RingJobs one at a time (sync mode).
fn ring_worker_loop(device_idx: usize, rx: channel::Receiver<RingJob>, queue_depth: u32) {
    let depth = if queue_depth == 0 {
        DEFAULT_QUEUE_DEPTH
    } else {
        queue_depth
    };
    let mut ring = match IoUring::new(depth) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("io_uring init failed: {}", e);
            return;
        }
    };

    while let Ok(job) = rx.recv() {
        match job {
            RingJob::SingleRead { req, resp } => {
                let r = do_read_one(&mut ring, &req);
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "read",
                    mode: "single",
                    device_id: device_idx as i64,
                    job_id: 0,
                };
                log_completed_io_batch(
                    context,
                    std::iter::once(&req),
                    std::slice::from_ref(&r),
                    |_, bytes| bytes.len(),
                );
                let _ = resp.send(r);
            }
            RingJob::SingleWrite { path, data, resp } => {
                let request = IORequest {
                    path,
                    offset: 0,
                    length: data.len(),
                };
                let r = do_write_one(&mut ring, &request.path, &data);
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "write",
                    mode: "single",
                    device_id: device_idx as i64,
                    job_id: 0,
                };
                log_completed_io_batch(
                    context,
                    std::iter::once(&request),
                    std::slice::from_ref(&r),
                    |request, _| request.length,
                );
                let _ = resp.send(r);
            }
            RingJob::ReadBatch { reqs, resp } => {
                let r = do_read_batch(&mut ring, &reqs);
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "read",
                    mode: "batch",
                    device_id: device_idx as i64,
                    job_id: 0,
                };
                log_completed_io_batch(context, reqs.iter(), &r, |_, bytes| bytes.len());
                let _ = resp.send(r);
            }
            RingJob::ReadAlignedBatch { reqs, resp } => {
                let r = do_read_aligned_batch(&mut ring, &reqs);
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "read",
                    mode: "aligned_batch",
                    device_id: device_idx as i64,
                    job_id: 0,
                };
                log_completed_io_batch(context, reqs.iter(), &r, |_, bytes| bytes.len());
                let _ = resp.send(r);
            }
            RingJob::WriteBatch { reqs, resp } => {
                let r = do_write_batch(&mut ring, &reqs);
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "write",
                    mode: "batch",
                    device_id: device_idx as i64,
                    job_id: 0,
                };
                log_completed_io_batch(
                    context,
                    reqs.iter().map(|(request, _)| request),
                    &r,
                    |request, _| request.length,
                );
                let _ = resp.send(r);
            }
            RingJob::WriteVecBatch { reqs, resp } => {
                let r = do_write_vec_batch(&mut ring, &reqs);
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "write",
                    mode: "vectored_batch",
                    device_id: device_idx as i64,
                    job_id: 0,
                };
                log_completed_io_batch(
                    context,
                    reqs.iter().map(|(request, _)| request),
                    &r,
                    |request, _| request.length,
                );
                let _ = resp.send(r);
            }
            RingJob::WriteAlignedBatch {
                job_id,
                queued_at,
                reqs,
                resp,
            } => {
                let started = std::time::Instant::now();
                let queue_wait_us = started.duration_since(queued_at).as_micros() as u64;
                let n = reqs.len();
                let requested_bytes: usize = reqs.iter().map(|request| request.2).sum();
                let r = match IoUring::new(64) {
                    Ok(mut wr) => do_write_aligned_batch(&mut wr, &reqs),
                    Err(e) => (0..n)
                        .map(|_| Err(KVError::Internal(format!("write ring init: {}", e))))
                        .collect(),
                };
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "write",
                    mode: "aligned_batch",
                    device_id: device_idx as i64,
                    job_id,
                };
                let success_count = r.iter().filter(|result| result.is_ok()).count();
                let completed_bytes = reqs
                    .iter()
                    .zip(&r)
                    .filter_map(|(request, result)| result.is_ok().then_some(request.2))
                    .sum();
                for ((request, _, bytes), result) in reqs.iter().zip(&r) {
                    match result {
                        Ok(()) => log_io_request(context, request, *bytes, *bytes),
                        Err(error) => log_io_error(context, request, *bytes, error),
                    }
                }
                log_io_batch(
                    context,
                    IoBatchStats {
                        request_count: n,
                        success_count,
                        failure_count: n.saturating_sub(success_count),
                        requested_bytes,
                        completed_bytes,
                        queue_wait_us,
                        duration_us: started.elapsed().as_micros() as u64,
                    },
                );
                let _ = resp.send(r);
            }
            RingJob::ReadAlignedIntoPtrBatch {
                job_id,
                queued_at,
                reqs,
                resp,
            } => {
                let started = std::time::Instant::now();
                let queue_wait_us = started.duration_since(queued_at).as_micros() as u64;
                let req_count = reqs.len();
                let planned_bytes: usize = reqs.iter().map(|r| r.2).sum();
                let r = do_read_aligned_into_ptr_batch(&mut ring, &reqs);
                let ok_bytes: usize = r.iter().filter_map(|x| x.as_ref().ok().copied()).sum();
                let context = IoLogContext {
                    executor: "tier_b",
                    operation: "read",
                    mode: "aligned_into_ptr_batch",
                    device_id: device_idx as i64,
                    job_id,
                };
                let success_count = r.iter().filter(|result| result.is_ok()).count();
                for ((request, _, capacity), result) in reqs.iter().zip(&r) {
                    match result {
                        Ok(bytes_read) => {
                            log_io_request(context, request, *bytes_read, *bytes_read)
                        }
                        Err(error) => log_io_error(context, request, *capacity, error),
                    }
                }
                log_io_batch(
                    context,
                    IoBatchStats {
                        request_count: req_count,
                        success_count,
                        failure_count: req_count.saturating_sub(success_count),
                        requested_bytes: planned_bytes,
                        completed_bytes: ok_bytes,
                        queue_wait_us,
                        duration_us: started.elapsed().as_micros() as u64,
                    },
                );
                let _ = resp.send(r);
            }
            RingJob::Shutdown => break,
        }
    }
}

fn log_completed_io_batch<'a, T>(
    context: IoLogContext<'_>,
    requests: impl IntoIterator<Item = &'a IORequest>,
    results: &[Result<T>],
    completed_bytes: impl Fn(&IORequest, &T) -> usize,
) {
    for (request, result) in requests.into_iter().zip(results) {
        match result {
            Ok(value) => {
                let completed = completed_bytes(request, value);
                let requested = if context.operation == "read" && request.length == 0 {
                    completed
                } else {
                    request.length
                };
                log_io_request(context, request, requested, completed);
            }
            Err(error) => log_io_error(context, request, request.length, error),
        }
    }
}

/// Open a file and return (File, length). length is used to size the read buffer.
fn open_for_read(path: &Path) -> Result<(std::fs::File, usize)> {
    let f = OpenOptions::new().read(true).open(path)?;
    let len = f.metadata()?.len() as usize;
    Ok((f, len))
}

fn open_for_write(path: &Path) -> Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?)
}

fn do_read_one(ring: &mut IoUring, req: &IORequest) -> Result<Vec<u8>> {
    let (file, total) = open_for_read(&req.path)?;
    let length = if req.length > 0 { req.length } else { total };
    let mut buf: Vec<u8> = vec![0; length];

    let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), buf.as_mut_ptr(), length as u32)
        .offset(req.offset)
        .build()
        .user_data(0);

    unsafe {
        ring.submission()
            .push(&entry)
            .map_err(|e| KVError::Internal(format!("uring push: {:?}", e)))?;
    }
    ring.submit_and_wait(1)
        .map_err(|e| KVError::Internal(format!("uring submit: {}", e)))?;

    let cqe = ring
        .completion()
        .next()
        .ok_or_else(|| KVError::Internal("uring cqe missing".into()))?;
    let n = cqe.result();
    drop(file);
    if n < 0 {
        return Err(KVError::Io(std::io::Error::from_raw_os_error(-n)));
    }
    buf.truncate(n as usize);
    Ok(buf)
}

fn do_write_one(ring: &mut IoUring, path: &Path, data: &[u8]) -> Result<()> {
    // Write to .tmp then rename for atomicity (matches Tier A).
    let tmp = path.with_extension("tmp");
    let file = open_for_write(&tmp)?;

    let entry = opcode::Write::new(
        types::Fd(file.as_raw_fd()),
        data.as_ptr(),
        data.len() as u32,
    )
    .offset(0)
    .build()
    .user_data(0);

    unsafe {
        ring.submission()
            .push(&entry)
            .map_err(|e| KVError::Internal(format!("uring push: {:?}", e)))?;
    }
    ring.submit_and_wait(1)
        .map_err(|e| KVError::Internal(format!("uring submit: {}", e)))?;

    let cqe = ring
        .completion()
        .next()
        .ok_or_else(|| KVError::Internal("uring cqe missing".into()))?;
    let n = cqe.result();
    if n < 0 {
        let _ = std::fs::remove_file(&tmp);
        return Err(KVError::Io(std::io::Error::from_raw_os_error(-n)));
    }
    // fdatasync via io_uring (optional; using std for now)
    use std::os::unix::io::AsRawFd as _;
    unsafe {
        if libc::fsync(file.as_raw_fd()) != 0 {
            return Err(KVError::Io(std::io::Error::last_os_error()));
        }
    }
    drop(file);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Batched read: all reads for a device are pushed to the ring at once and
/// completed by a single submit_and_wait(N). This is Tier B's core advantage
/// over Tier A: N syscalls collapse into 1.
fn do_read_batch(ring: &mut IoUring, reqs: &[IORequest]) -> Vec<Result<Vec<u8>>> {
    let n = reqs.len();
    let mut results: Vec<Result<Vec<u8>>> = (0..n).map(|_| Ok(Vec::new())).collect();
    let mut files: Vec<Option<std::fs::File>> = (0..n).map(|_| None).collect();
    let mut bufs: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
    let mut submitted = 0usize;

    for (i, req) in reqs.iter().enumerate() {
        match open_for_read(&req.path) {
            Ok((f, total)) => {
                let length = if req.length > 0 { req.length } else { total };
                let mut buf: Vec<u8> = vec![0; length];
                let entry =
                    opcode::Read::new(types::Fd(f.as_raw_fd()), buf.as_mut_ptr(), length as u32)
                        .offset(req.offset)
                        .build()
                        .user_data(i as u64);
                let push_res = unsafe { ring.submission().push(&entry) };
                if let Err(e) = push_res {
                    results[i] = Err(KVError::Internal(format!("uring push: {:?}", e)));
                    continue;
                }
                files[i] = Some(f);
                bufs[i] = Some(buf);
                submitted += 1;
            }
            Err(e) => {
                results[i] = Err(e);
            }
        }
    }

    if submitted == 0 {
        return results;
    }

    if let Err(e) = ring.submit_and_wait(submitted) {
        // Whole batch failed.
        for r in &mut results {
            if r.is_ok() {
                *r = Err(KVError::Internal(format!("uring submit: {}", e)));
            }
        }
        return results;
    }

    for cqe in ring.completion() {
        let i = cqe.user_data() as usize;
        if i >= n {
            continue;
        }
        let ret = cqe.result();
        if ret < 0 {
            results[i] = Err(KVError::Io(std::io::Error::from_raw_os_error(-ret)));
            bufs[i] = None;
        } else if let Some(mut buf) = bufs[i].take() {
            buf.truncate(ret as usize);
            results[i] = Ok(buf);
        }
    }

    // files Drop -> close
    results
}

/// Batched write: same one-shot submit.
fn do_write_batch(ring: &mut IoUring, reqs: &[(IORequest, Bytes)]) -> Vec<Result<()>> {
    let n = reqs.len();
    let mut results: Vec<Result<()>> = (0..n).map(|_| Ok(())).collect();
    let mut files: Vec<Option<std::fs::File>> = (0..n).map(|_| None).collect();
    let mut tmp_paths: Vec<Option<PathBuf>> = (0..n).map(|_| None).collect();
    let mut submitted = 0usize;

    for (i, (req, _data)) in reqs.iter().enumerate() {
        let tmp = req.path.with_extension("tmp");
        match open_for_write(&tmp) {
            Ok(f) => {
                files[i] = Some(f);
                tmp_paths[i] = Some(tmp);
            }
            Err(e) => {
                results[i] = Err(e);
            }
        }
    }

    for (i, (_, data)) in reqs.iter().enumerate() {
        let Some(file) = &files[i] else { continue };
        let entry = opcode::Write::new(
            types::Fd(file.as_raw_fd()),
            data.as_ptr(),
            data.len() as u32,
        )
        .offset(0)
        .build()
        .user_data(i as u64);
        let push_res = unsafe { ring.submission().push(&entry) };
        if let Err(e) = push_res {
            results[i] = Err(KVError::Internal(format!("uring push: {:?}", e)));
            files[i] = None;
            if let Some(p) = &tmp_paths[i] {
                let _ = std::fs::remove_file(p);
            }
            continue;
        }
        submitted += 1;
    }

    if submitted == 0 {
        return results;
    }

    if let Err(e) = ring.submit_and_wait(submitted) {
        for r in &mut results {
            if r.is_ok() {
                *r = Err(KVError::Internal(format!("uring submit: {}", e)));
            }
        }
        return results;
    }

    for cqe in ring.completion() {
        let i = cqe.user_data() as usize;
        if i >= n {
            continue;
        }
        let ret = cqe.result();
        if ret < 0 {
            results[i] = Err(KVError::Io(std::io::Error::from_raw_os_error(-ret)));
            if let Some(p) = &tmp_paths[i] {
                let _ = std::fs::remove_file(p);
            }
            files[i] = None;
        }
    }

    // fsync + rename for successful writes
    for i in 0..n {
        if results[i].is_err() {
            continue;
        }
        if let (Some(f), Some(tmp)) = (files[i].take(), tmp_paths[i].take()) {
            let fd = f.as_raw_fd();
            unsafe {
                if libc::fsync(fd) != 0 {
                    results[i] = Err(KVError::Io(std::io::Error::last_os_error()));
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
            }
            drop(f);
            if let Err(e) = std::fs::rename(&tmp, &reqs[i].0.path) {
                results[i] = Err(KVError::Io(e));
            }
        }
    }

    results
}

// ===== IOExecutor trait impl =====
//
// Scheduling: batched requests are grouped by device, each group is submitted
// independently to its ring (worker threads run in parallel).
// Each ring is single-consumer inside its worker thread, so we hand the whole
// batch of jobs over via one channel send.

/// Vectored write batch: each IO item (path, Vec<Bytes>) is written with a single
/// opcode::Writev SQE.
/// The kernel writes all iovec segments atomically (equivalent to a writev syscall);
/// same as tier_a's write_vectored but on the io_uring batch-submit path.
///
/// SAFETY: the iovecs array must remain valid until the CQE arrives; segments (Bytes)
/// must also remain valid (iovec.iov_base points into Bytes internals). We keep `reqs`
/// borrowed for the whole call so segments never drop; iovecs are Box<[libc::iovec]>
/// so each SQE's iovec array has a stable address.
fn do_write_vec_batch(ring: &mut IoUring, reqs: &[(IORequest, Vec<Bytes>)]) -> Vec<Result<()>> {
    let n = reqs.len();
    let mut results: Vec<Result<()>> = (0..n).map(|_| Ok(())).collect();
    let mut files: Vec<Option<std::fs::File>> = (0..n).map(|_| None).collect();
    let mut tmp_paths: Vec<Option<PathBuf>> = (0..n).map(|_| None).collect();
    // iovecs: one stable Box<[iovec]> per IO item; the SQE holds a pointer to it
    // and it must outlive the CQE.
    let mut iovecs_arr: Vec<Option<Box<[libc::iovec]>>> = (0..n).map(|_| None).collect();
    let mut submitted = 0usize;

    // Open all tmp files (same as do_write_batch).
    for (i, (req, _data)) in reqs.iter().enumerate() {
        let tmp = req.path.with_extension("tmp");
        match open_for_write(&tmp) {
            Ok(f) => {
                files[i] = Some(f);
                tmp_paths[i] = Some(tmp);
            }
            Err(e) => {
                results[i] = Err(e);
            }
        }
    }

    // Build iovec arrays and push SQEs.
    for (i, (_, segments)) in reqs.iter().enumerate() {
        let Some(file) = &files[i] else { continue };
        if segments.is_empty() {
            // Empty segments — skip (the empty file has already been created).
            continue;
        }
        // Build iovec[]; iov_base points into Bytes internals (segments live inside
        // `reqs`, borrowed for the whole function, so they never drop).
        let iovecs: Box<[libc::iovec]> = segments
            .iter()
            .map(|seg| libc::iovec {
                iov_base: seg.as_ptr() as *mut libc::c_void,
                iov_len: seg.len(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let entry = opcode::Writev::new(
            types::Fd(file.as_raw_fd()),
            iovecs.as_ptr(),
            iovecs.len() as u32,
        )
        .offset(0)
        .build()
        .user_data(i as u64);
        let push_res = unsafe { ring.submission().push(&entry) };
        if let Err(e) = push_res {
            results[i] = Err(KVError::Internal(format!("uring push: {:?}", e)));
            files[i] = None;
            if let Some(p) = &tmp_paths[i] {
                let _ = std::fs::remove_file(p);
            }
            continue;
        }
        iovecs_arr[i] = Some(iovecs);
        submitted += 1;
    }

    if submitted == 0 {
        return results;
    }

    if let Err(e) = ring.submit_and_wait(submitted) {
        for r in &mut results {
            if r.is_ok() {
                *r = Err(KVError::Internal(format!("uring submit: {}", e)));
            }
        }
        return results;
    }

    // Reap CQEs; verify ret == sum(segment lens) (writev is atomic; short write = kernel error).
    for cqe in ring.completion() {
        let i = cqe.user_data() as usize;
        if i >= n {
            continue;
        }
        let ret = cqe.result();
        if ret < 0 {
            results[i] = Err(KVError::Io(std::io::Error::from_raw_os_error(-ret)));
            if let Some(p) = &tmp_paths[i] {
                let _ = std::fs::remove_file(p);
            }
            files[i] = None;
        } else {
            // Check for short write (shouldn't happen for buffered writev in one call).
            let expected: usize = reqs[i].1.iter().map(|s| s.len()).sum();
            if (ret as usize) < expected {
                results[i] = Err(KVError::Internal(format!(
                    "writev short: wrote {} of {}",
                    ret, expected
                )));
                if let Some(p) = &tmp_paths[i] {
                    let _ = std::fs::remove_file(p);
                }
                files[i] = None;
            }
        }
    }

    // CQEs are drained; iovecs can now be freed (drop the vec entries automatically).
    drop(iovecs_arr);

    // fsync + rename for successful writes.
    for i in 0..n {
        if results[i].is_err() {
            continue;
        }
        if let (Some(f), Some(tmp)) = (files[i].take(), tmp_paths[i].take()) {
            let fd = f.as_raw_fd();
            // KV cache is recomputable, so skip fsync by default (matches tier_a;
            // set CS_SYNC_WRITES=1 to force sync).
            if std::env::var("CS_SYNC_WRITES").as_deref() == Ok("1") {
                unsafe {
                    if libc::fsync(fd) != 0 {
                        results[i] = Err(KVError::Io(std::io::Error::last_os_error()));
                        let _ = std::fs::remove_file(&tmp);
                        continue;
                    }
                }
            }
            drop(f);
            if let Err(e) = std::fs::rename(&tmp, &reqs[i].0.path) {
                results[i] = Err(KVError::Io(e));
            }
        }
    }

    results
}

/// Max bytes per single write SQE. Conservative 4MB for writes to avoid NVMe-oF
/// controller reset.
const MAX_AIO_CHUNK_WRITE: usize = 4 * 1024 * 1024;

/// Max bytes per single read SQE. Chosen to match `fio bs=4M iodepth=8`: a stripe
/// is split into many 4MB SQEs, and io_uring uses ring depth to concurrently submit
/// them to NVMe-oF, which is faster than serially processing one 60MB SQE.
///
/// Historical 64MB problem: one stripe = 1 SQE; while NVMe-oF processed that single
/// IO nothing else could run in parallel, so 4 workers queued up. With 4MB: 1 stripe
/// = 15 SQEs, 8 stripes = 120 SQEs, 4 workers × 120 = 480 SQEs. ring depth=64 can't
/// hold them all, but io_uring hands N SQEs to NVMe and immediately makes room for
/// the next batch — real parallelism.
const MAX_AIO_CHUNK_READ: usize = 4 * 1024 * 1024;

// Legacy reference (write path uses MAX_AIO_CHUNK).
const MAX_AIO_CHUNK: usize = MAX_AIO_CHUNK_WRITE;

/// O_DIRECT write batch (RDMA PUT data path) — data is already in caller-pinned
/// 4K-aligned memory; use io_uring opcode::Write for pwrite directly, **zero memcpy**.
///
/// Key differences vs tier_a `write_aligned_impl`:
/// - tier_a: 8 worker threads each issue a sync pwrite, bursting a lot of bio into
///   the NVMe driver → fills NVMe-oF SQ=16 → controller reset → tens of seconds of hang.
/// - tier_b: one ring per drive, ring depth (default 256, should tune to 8-16) strictly
///   caps in-flight SQEs. Push N IOs into SQ and submit_and_wait once; the kernel
///   throttles inside the ring, never bursts to the NVMe driver, and coexists nicely
///   with NVMe-oF SQ=16.
///
/// Flow (per IO item):
/// 1. Create O_DIRECT tmp file (`.tmp`, then rename)
/// 2. Body: aligned_down(len, 4K) pwrite directly from caller ptr
/// 3. Tail: bytes below 4K are copied into a thread_local 4K bounce buffer padded with
///    zeros, then pwrite'd
/// 4. Truncate to real len, rename .tmp → final path
///
/// SAFETY: the caller (RDMA server handle_put) must ensure:
/// - ptr..ptr+aligned_up(len, 4K) is fully readable
/// - the memory is not freed / mutated until submit_and_wait returns
/// - ptr is 4K-aligned (guaranteed by slab.alloc)
fn do_write_aligned_batch(
    ring: &mut IoUring,
    reqs: &[(IORequest, PtrWrapper, usize)],
) -> Vec<Result<()>> {
    let n = reqs.len();
    let mut results: Vec<Result<()>> = (0..n).map(|_| Ok(())).collect();
    let mut files: Vec<Option<std::fs::File>> = (0..n).map(|_| None).collect();
    let mut tmp_paths: Vec<Option<PathBuf>> = (0..n).map(|_| None).collect();
    // tail buffers: one Boxed AlignedBuffer per IO item (when there is a tail) so the
    // pointer stays stable.
    let mut tail_bufs: Vec<Option<Box<AlignedBuffer>>> = (0..n).map(|_| None).collect();

    // ===== Phase 1: open O_DIRECT tmp files =====
    for (i, (req, _ptr, _len)) in reqs.iter().enumerate() {
        if let Some(parent) = req.path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                results[i] = Err(KVError::Io(e));
                continue;
            }
        }
        let tmp = req.path.with_extension("tmp");
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(O_DIRECT_FLAG)
            .open(&tmp);
        match f {
            Ok(f) => {
                files[i] = Some(f);
                tmp_paths[i] = Some(tmp);
            }
            Err(e) if e.raw_os_error() == Some(22) => {
                // O_DIRECT not supported (container/filesystem). Fallback: buffered write.
                let slice = unsafe { std::slice::from_raw_parts(reqs[i].1 .0, reqs[i].2) };
                if let Err(e) = std::fs::write(&tmp, slice) {
                    results[i] = Err(KVError::Io(e));
                } else if let Err(e) = std::fs::rename(&tmp, &req.path) {
                    results[i] = Err(KVError::Io(e));
                }
            }
            Err(e) => results[i] = Err(KVError::Io(e)),
        }
    }

    // ===== Phase 2: verify alignment + prepare tail buffer =====
    // Collect all SQE plans (each IO split into multiple 4MB chunks + 1 tail).
    // sqe_plan[i] = Vec<(offset_in_file, ptr, len, is_tail)>
    let mut sqe_plans: Vec<Vec<(u64, *const u8, u32, bool)>> = (0..n).map(|_| Vec::new()).collect();

    for (i, (_req, ptr_wrap, len)) in reqs.iter().enumerate() {
        if files[i].is_none() {
            continue; // already fell back or failed
        }
        let ptr = ptr_wrap.0;
        let len = *len;
        if (ptr as usize) % DIRECT_IO_ALIGN != 0 {
            results[i] = Err(KVError::Internal(format!(
                "do_write_aligned_batch: ptr {:p} not 4K-aligned",
                ptr
            )));
            files[i] = None;
            continue;
        }
        let aligned_down = len & !(DIRECT_IO_ALIGN - 1);
        let tail = len - aligned_down;

        // Split body into MAX_AIO_CHUNK (4MB) pieces so io_uring throttles via ring depth.
        // Without splitting: a single 200MB SQE → kernel expands to 200 × 1MB bio and
        // slams them at the NVMe driver in one shot → SQ overflow.
        let mut off = 0usize;
        while off < aligned_down {
            let chunk_len = (aligned_down - off).min(MAX_AIO_CHUNK);
            sqe_plans[i].push((off as u64, unsafe { ptr.add(off) }, chunk_len as u32, false));
            off += chunk_len;
        }

        // Tail: bytes < 4K go into a thread_local 4K bounce buffer padded with zeros,
        // then pwrite the full 4K.
        if tail > 0 {
            let mut tbuf = Box::new(AlignedBuffer::new(DIRECT_IO_ALIGN, DIRECT_IO_ALIGN));
            unsafe {
                std::ptr::write_bytes(tbuf.as_mut_ptr(), 0, DIRECT_IO_ALIGN);
                std::ptr::copy_nonoverlapping(ptr.add(aligned_down), tbuf.as_mut_ptr(), tail);
            }
            let tail_ptr = tbuf.as_mut_ptr() as *const u8;
            sqe_plans[i].push((aligned_down as u64, tail_ptr, DIRECT_IO_ALIGN as u32, true));
            tail_bufs[i] = Some(tbuf);
        }
    }

    // ===== Phase 3: submit all SQEs in a loop of submit_and_wait =====
    // The io_uring ring depth caps in-flight SQEs. All 8 stripes of a drive go
    // through this function serially (single worker thread), but each stripe's
    // 64+ SQEs are throttled by the ring depth.
    //
    // Algorithm: gather all SQEs per IO item, then loop push + submit_and_wait
    // (each submit up to ring_capacity SQEs). ring_capacity is fixed at ring-init
    // time by `depth`.
    //
    // Key point: unlike the previous code, we don't push everything then submit;
    // we split into multiple rounds (each ring_capacity SQEs) so io_uring's
    // internal throttling actually kicks in.

    // Encode user_data: high 32 bits = io idx, low 32 bits = sqe seq within the io.
    // Used to identify errors when the CQE arrives.
    #[derive(Default)]
    struct IoStatus {
        sqe_count: u32, // planned SQE count
        completed: u32, // CQEs received
        any_error: bool,
    }
    let mut status: Vec<IoStatus> = (0..n).map(|_| IoStatus::default()).collect();
    for (i, plan) in sqe_plans.iter().enumerate() {
        status[i].sqe_count = plan.len() as u32;
    }

    // Flatten all SQEs into a (io_idx, sqe_seq, offset, ptr, len, is_tail) list.
    let mut all_sqes: Vec<(u64, u64, *const u8, u32)> = Vec::new(); // (user_data, offset, ptr, len)
    for (i, plan) in sqe_plans.iter().enumerate() {
        if files[i].is_none() {
            continue;
        }
        for (seq, (off, p, l, _is_tail)) in plan.iter().enumerate() {
            let ud = ((i as u64) << 32) | (seq as u64);
            all_sqes.push((ud, *off, *p, *l));
        }
    }

    // Actual ring capacity (from init; default 256, config lowered to 16).
    // io_uring's SQ slot count equals the depth given at init, but submit_and_wait
    // can wait on more CQEs. We conservatively push at most ring_capacity/2 per
    // batch to leave SQ headroom and reduce push retries.
    let ring_capacity = ring.params().sq_entries() as usize;
    let batch_size = ring_capacity.max(8) / 2; // at least 4, default 8 (depth=16)

    let all_files: Vec<Option<&std::fs::File>> = (0..n).map(|i| files[i].as_ref()).collect();

    let mut sqe_idx = 0;
    while sqe_idx < all_sqes.len() {
        let batch_end = (sqe_idx + batch_size).min(all_sqes.len());
        let mut pushed = 0u32;

        for (ud, off, ptr, len) in &all_sqes[sqe_idx..batch_end] {
            let i = (ud >> 32) as usize;
            let Some(file) = all_files[i] else { continue };
            if status[i].any_error {
                continue;
            }
            let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), *ptr, *len)
                .offset(*off)
                .build()
                .user_data(*ud);
            let push_res = unsafe { ring.submission().push(&entry) };
            if let Err(e) = push_res {
                status[i].any_error = true;
                results[i] = Err(KVError::Internal(format!("uring push: {:?}", e)));
                continue;
            }
            pushed += 1;
        }

        if pushed == 0 {
            sqe_idx = batch_end;
            continue;
        }

        if let Err(e) = ring.submit_and_wait(pushed as usize) {
            for (i, r) in results.iter_mut().enumerate() {
                if r.is_ok() && all_files[i].is_some() {
                    *r = Err(KVError::Internal(format!("uring submit: {}", e)));
                    status[i].any_error = true;
                }
            }
            return results;
        }

        // Reap CQEs.
        for cqe in ring.completion() {
            let ud = cqe.user_data();
            let i = (ud >> 32) as usize;
            if i >= n {
                continue;
            }
            let ret = cqe.result();
            if ret < 0 {
                status[i].any_error = true;
                results[i] = Err(KVError::Io(std::io::Error::from_raw_os_error(-ret)));
            }
            status[i].completed += 1;
        }

        sqe_idx = batch_end;
    }

    // Drop the file references (later code needs to take ownership).
    drop(all_files);

    // ===== Phase 4: truncate tail + rename =====
    for i in 0..n {
        if results[i].is_err() {
            if let Some(p) = &tmp_paths[i] {
                let _ = std::fs::remove_file(p);
            }
            continue;
        }
        let Some(file) = files[i].take() else {
            continue;
        };
        let Some(tmp) = tmp_paths[i].take() else {
            continue;
        };
        let (_req, _ptr, len) = &reqs[i];
        let aligned_down = len & !(DIRECT_IO_ALIGN - 1);
        let tail = len - aligned_down;

        if std::env::var("CS_SYNC_WRITES").as_deref() == Ok("1") {
            unsafe {
                if libc::fsync(file.as_raw_fd()) != 0 {
                    results[i] = Err(KVError::Io(std::io::Error::last_os_error()));
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
            }
        }
        drop(file);

        if tail > 0 {
            let f2 = match OpenOptions::new().write(true).open(&tmp) {
                Ok(f) => f,
                Err(e) => {
                    results[i] = Err(KVError::Io(e));
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
            };
            if let Err(e) = f2.set_len(*len as u64) {
                results[i] = Err(KVError::Io(e));
                let _ = std::fs::remove_file(&tmp);
                continue;
            }
        }
        if let Err(e) = std::fs::rename(&tmp, &reqs[i].0.path) {
            results[i] = Err(KVError::Io(e));
        }
    }

    drop(tail_bufs);

    results
}

/// O_DIRECT batch read (into caller-pinned 4K-aligned memory) — symmetric to
/// do_write_aligned_batch.
/// **Zero memcpy + zero reg_mr**: server slab.alloc(N) → read NVMe directly into
/// the slab ptr → RDMA WRITE using the slab's pre-registered MR (GET cache-miss
/// fast path).
///
/// Differences vs do_read_aligned_batch:
/// - Old: per-IO Box<AlignedBuffer> (heap) → read into heap → wrap as Bytes →
///        caller (serve_get_fallback) still needs a per-chunk ibv_reg_mr
///        (~33ms of serial syscalls).
/// - New: read straight into a caller-supplied ptr (in slab, already pre-registered)
///        → zero reg_mr → caller can RDMA WRITE with the slab lkey.
///
/// Key: chunked 4MB SQEs (same as the write path) let io_uring throttle by ring depth.
///
/// SAFETY (caller must guarantee):
/// - each ptr..ptr+capacity is a fully writable 4K-aligned region, capacity ≥ file_size
/// - memory is not freed / mutated until the worker completes
/// - actual file_size is returned via Vec<Result<usize>> so the caller can truncate
fn do_read_aligned_into_ptr_batch(
    ring: &mut IoUring,
    reqs: &[(IORequest, PtrWrapperMut, usize)],
) -> Vec<Result<usize>> {
    let n = reqs.len();
    let mut results: Vec<Result<usize>> = (0..n).map(|_| Ok(0)).collect();
    let mut files: Vec<Option<std::fs::File>> = (0..n).map(|_| None).collect();
    let mut file_sizes: Vec<usize> = vec![0; n];

    // sqe_plans[i] = Vec<(offset_in_file, ptr_in_buf, chunk_len)>
    let mut sqe_plans: Vec<Vec<(u64, *mut u8, u32)>> = (0..n).map(|_| Vec::new()).collect();

    // ===== Phase 1: open O_DIRECT + check alignment + plan chunked SQEs =====
    for (i, (req, ptr_wrap, capacity)) in reqs.iter().enumerate() {
        let dst_ptr = ptr_wrap.0;
        let capacity = *capacity;
        if (dst_ptr as usize) % DIRECT_IO_ALIGN != 0 {
            results[i] = Err(KVError::Internal(format!(
                "read_aligned_into_ptr: ptr {:p} not 4K-aligned",
                dst_ptr
            )));
            continue;
        }
        if capacity % DIRECT_IO_ALIGN != 0 {
            results[i] = Err(KVError::Internal(format!(
                "read_aligned_into_ptr: capacity {} not 4K-aligned",
                capacity
            )));
            continue;
        }
        match open_for_direct_read(&req.path) {
            Ok((Some(f), file_size)) => {
                if file_size == 0 {
                    results[i] = Ok(0);
                    continue;
                }
                let aligned_len = (file_size + DIRECT_IO_ALIGN - 1) & !(DIRECT_IO_ALIGN - 1);
                if capacity < aligned_len {
                    results[i] = Err(KVError::Internal(format!(
                        "read_aligned_into_ptr: capacity {} < aligned file_size {}",
                        capacity, aligned_len
                    )));
                    continue;
                }
                // Split into 4MB chunked Read SQEs (same as write; lets ring depth throttle).
                let mut off = 0usize;
                while off < aligned_len {
                    let chunk_len = (aligned_len - off).min(MAX_AIO_CHUNK_READ);
                    let chunk_ptr = unsafe { dst_ptr.add(off) };
                    sqe_plans[i].push((off as u64, chunk_ptr, chunk_len as u32));
                    off += chunk_len;
                }
                files[i] = Some(f);
                file_sizes[i] = file_size;
            }
            Ok((None, _file_size)) => {
                // O_DIRECT not supported (container/filesystem): fall back to buffered sync
                // read; use ordinary read then memcpy into ptr (slow path; rarely triggered).
                let f = match OpenOptions::new().read(true).open(&req.path) {
                    Ok(f) => f,
                    Err(e) => {
                        results[i] = Err(KVError::Io(e));
                        continue;
                    }
                };
                let file_size = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
                if file_size > capacity {
                    results[i] = Err(KVError::Internal(format!(
                        "fallback read: file_size {} > capacity {}",
                        file_size, capacity
                    )));
                    continue;
                }
                let mut buf = vec![0u8; file_size];
                use std::io::Read as _;
                let mut f_mut = f;
                match f_mut.read_exact(&mut buf) {
                    Ok(()) => {
                        unsafe {
                            std::ptr::copy_nonoverlapping(buf.as_ptr(), dst_ptr, file_size);
                        }
                        results[i] = Ok(file_size);
                    }
                    Err(e) => results[i] = Err(KVError::Io(e)),
                }
            }
            Err(e) => {
                results[i] = Err(e);
            }
        }
    }

    // ===== Phase 2: gather all SQEs and submit in batches =====
    let mut all_sqes: Vec<(u64, u64, *mut u8, u32)> = Vec::new();
    for (i, plan) in sqe_plans.iter().enumerate() {
        if files[i].is_none() {
            continue;
        }
        for (seq, (off, ptr, len)) in plan.iter().enumerate() {
            let ud = ((i as u64) << 32) | (seq as u64);
            all_sqes.push((ud, *off, *ptr, *len));
        }
    }

    if all_sqes.is_empty() {
        return results;
    }

    let ring_capacity = ring.params().sq_entries() as usize;
    let batch_size = ring_capacity.max(8) / 2;

    let mut any_error: Vec<bool> = vec![false; n];
    let mut total_read: Vec<usize> = vec![0; n];

    let mut sqe_idx = 0;
    while sqe_idx < all_sqes.len() {
        let batch_end = (sqe_idx + batch_size).min(all_sqes.len());
        let mut pushed = 0u32;

        for (ud, off, ptr, len) in &all_sqes[sqe_idx..batch_end] {
            let i = (ud >> 32) as usize;
            if any_error[i] {
                continue;
            }
            let Some(file) = &files[i] else { continue };
            let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), *ptr, *len)
                .offset(*off)
                .build()
                .user_data(*ud);
            let push_res = unsafe { ring.submission().push(&entry) };
            if let Err(e) = push_res {
                any_error[i] = true;
                results[i] = Err(KVError::Internal(format!("uring push read_into: {:?}", e)));
                continue;
            }
            pushed += 1;
        }

        if pushed == 0 {
            sqe_idx = batch_end;
            continue;
        }

        if let Err(e) = ring.submit_and_wait(pushed as usize) {
            for (i, r) in results.iter_mut().enumerate() {
                if !any_error[i] && files[i].is_some() {
                    *r = Err(KVError::Internal(format!("uring submit read_into: {}", e)));
                    any_error[i] = true;
                }
            }
            return results;
        }

        for cqe in ring.completion() {
            let ud = cqe.user_data();
            let i = (ud >> 32) as usize;
            if i >= n {
                continue;
            }
            let ret = cqe.result();
            if ret < 0 {
                any_error[i] = true;
                results[i] = Err(KVError::Io(std::io::Error::from_raw_os_error(-ret)));
            } else {
                total_read[i] += ret as usize;
            }
        }

        sqe_idx = batch_end;
    }

    // ===== Phase 3: fill in return values (real file_size, not 4K padding) =====
    for i in 0..n {
        if !any_error[i] && files[i].is_some() {
            // total_read[i] may be ≥ file_size (0-padded to 4K); take the real file_size.
            results[i] = Ok(file_sizes[i].min(total_read[i]));
        }
    }

    results
}

/// Open a file for O_DIRECT read and get file_size; the fallback path returns
/// (None, file_size) when the file doesn't support O_DIRECT, telling the caller to
/// fall back to buffered.
fn open_for_direct_read(path: &Path) -> Result<(Option<std::fs::File>, usize)> {
    let meta = std::fs::metadata(path)?;
    let size = meta.len() as usize;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT_FLAG)
        .open(path);
    match f {
        Ok(f) => Ok((Some(f), size)),
        // EINVAL: filesystem / mount option doesn't support O_DIRECT, fall back.
        Err(e) if e.raw_os_error() == Some(22) => Ok((None, size)),
        Err(e) => Err(KVError::Io(e)),
    }
}

/// O_DIRECT batch read: per-segment AlignedBuffer + opcode::Read SQE, single
/// submit_and_wait to reap all CQEs.
/// Equivalent to do_read_batch but uses aligned buffers + O_DIRECT to skip the page cache.
/// Returns zero-copy Bytes (AlignedBuffer wrapped via from_owner).
///
/// **Key design (mirrors the write path)**: a single 60MB file → split into many 4MB
/// Read SQEs so io_uring strictly throttles in-flight read IO by ring depth. Otherwise
/// one 60MB SQE → kernel expands to 60 × 1MB bio submitted all at once → NVMe-oF target
/// SQ=16 gets loaded, and per-drive read BW drops from the fio limit of 0.89 GB/s to
/// 0.67 GB/s (measured).
fn do_read_aligned_batch(ring: &mut IoUring, reqs: &[IORequest]) -> Vec<Result<Bytes>> {
    let n = reqs.len();
    let mut results: Vec<Result<Bytes>> = (0..n).map(|_| Ok(Bytes::new())).collect();
    let mut files: Vec<Option<std::fs::File>> = (0..n).map(|_| None).collect();
    // bufs keep stable pointers (io_uring SQE holds a ptr); an IO item's chunked
    // Reads share one Box<AlignedBuffer>.
    let mut bufs: Vec<Option<Box<AlignedBuffer>>> = (0..n).map(|_| None).collect();
    let mut sizes: Vec<usize> = vec![0; n];
    // Number of CQEs to wait for per IO (= number of chunked SQEs).
    let mut expected_cqes: Vec<u32> = vec![0; n];

    // ===== Phase 1: open + alloc buffer + plan SQEs =====
    // sqe_plans[i] = Vec<(offset_in_file, ptr_in_buf, chunk_len)>
    let mut sqe_plans: Vec<Vec<(u64, *mut u8, u32)>> = (0..n).map(|_| Vec::new()).collect();

    for (i, req) in reqs.iter().enumerate() {
        match open_for_direct_read(&req.path) {
            Ok((Some(f), file_size)) => {
                if file_size == 0 {
                    results[i] = Ok(Bytes::new());
                    continue;
                }
                let aligned_len = (file_size + DIRECT_IO_ALIGN - 1) & !(DIRECT_IO_ALIGN - 1);
                let mut buf = Box::new(AlignedBuffer::new(aligned_len, DIRECT_IO_ALIGN));
                let buf_base = buf.as_mut_ptr();

                // Split into Read SQEs of MAX_AIO_CHUNK (4MB).
                let mut off = 0usize;
                while off < aligned_len {
                    let chunk_len = (aligned_len - off).min(MAX_AIO_CHUNK_READ);
                    let chunk_ptr = unsafe { buf_base.add(off) };
                    sqe_plans[i].push((off as u64, chunk_ptr, chunk_len as u32));
                    off += chunk_len;
                }

                files[i] = Some(f);
                bufs[i] = Some(buf);
                sizes[i] = file_size;
                expected_cqes[i] = sqe_plans[i].len() as u32;
            }
            Ok((None, _file_size)) => {
                // O_DIRECT not supported: fall back to buffered (use the old do_read_one).
                let r = do_read_one(ring, req).map(Bytes::from);
                results[i] = r;
            }
            Err(e) => {
                results[i] = Err(e);
            }
        }
    }

    // ===== Phase 2: gather all SQEs and submit in batches =====
    // Same as the write path: at most ring_capacity/2 SQEs per batch so ring
    // throttling really kicks in.
    let mut all_sqes: Vec<(u64, u64, *mut u8, u32)> = Vec::new(); // (user_data, offset, ptr, len)
    for (i, plan) in sqe_plans.iter().enumerate() {
        if files[i].is_none() {
            continue;
        }
        for (seq, (off, ptr, len)) in plan.iter().enumerate() {
            let ud = ((i as u64) << 32) | (seq as u64);
            all_sqes.push((ud, *off, *ptr, *len));
        }
    }

    if all_sqes.is_empty() {
        return results;
    }

    let ring_capacity = ring.params().sq_entries() as usize;
    let batch_size = ring_capacity.max(8) / 2; // at least 4, default 8 (depth=16)

    // Track whether each IO has already hit an error (skip its remaining SQEs on
    // short read / error).
    let mut any_error: Vec<bool> = vec![false; n];
    // Total CQE bytes received (used later for set_len).
    let mut total_read: Vec<usize> = vec![0; n];

    let mut sqe_idx = 0;
    while sqe_idx < all_sqes.len() {
        let batch_end = (sqe_idx + batch_size).min(all_sqes.len());
        let mut pushed = 0u32;

        for (ud, off, ptr, len) in &all_sqes[sqe_idx..batch_end] {
            let i = (ud >> 32) as usize;
            if any_error[i] {
                continue;
            }
            let Some(file) = &files[i] else { continue };
            let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), *ptr, *len)
                .offset(*off)
                .build()
                .user_data(*ud);
            let push_res = unsafe { ring.submission().push(&entry) };
            if let Err(e) = push_res {
                any_error[i] = true;
                results[i] = Err(KVError::Internal(format!("uring push read: {:?}", e)));
                bufs[i] = None;
                continue;
            }
            pushed += 1;
        }

        if pushed == 0 {
            sqe_idx = batch_end;
            continue;
        }

        if let Err(e) = ring.submit_and_wait(pushed as usize) {
            for (i, r) in results.iter_mut().enumerate() {
                if !any_error[i] && bufs[i].is_some() {
                    *r = Err(KVError::Internal(format!("uring submit read: {}", e)));
                    any_error[i] = true;
                    bufs[i] = None;
                }
            }
            return results;
        }

        for cqe in ring.completion() {
            let ud = cqe.user_data();
            let i = (ud >> 32) as usize;
            if i >= n {
                continue;
            }
            let ret = cqe.result();
            if ret < 0 {
                any_error[i] = true;
                results[i] = Err(KVError::Io(std::io::Error::from_raw_os_error(-ret)));
                bufs[i] = None;
            } else {
                total_read[i] += ret as usize;
            }
        }

        sqe_idx = batch_end;
    }

    // ===== Phase 3: wrap successfully-read buffers as Bytes =====
    for i in 0..n {
        if any_error[i] {
            continue;
        }
        if let Some(buf) = bufs[i].take() {
            let mut owned = *buf;
            // sizes[i] is the true file size; total_read[i] includes 0-padding to 4K.
            owned.set_len(sizes[i].min(total_read[i]));
            results[i] = Ok(Bytes::from_owner(owned));
        }
    }

    // files Drop -> close.
    results
}

impl IOExecutor for TierBExecutor {
    fn read_file(&self, path: &Path) -> Result<Vec<u8>> {
        let worker = self.route_device(path)?;
        let (tx, rx) = channel::bounded(1);
        worker
            .sender
            .send(RingJob::SingleRead {
                req: IORequest {
                    path: path.to_path_buf(),
                    offset: 0,
                    length: 0,
                },
                resp: tx,
            })
            .map_err(|_| KVError::Internal("worker shut down".into()))?;
        rx.recv()
            .map_err(|_| KVError::Internal("worker no response".into()))?
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> Result<()> {
        let worker = self.route_device(path)?;
        let (tx, rx) = channel::bounded(1);
        worker
            .sender
            .send(RingJob::SingleWrite {
                path: path.to_path_buf(),
                // Cold path for single write_file; copy into Bytes only to match the
                // RingJob field type.
                data: Bytes::copy_from_slice(data),
                resp: tx,
            })
            .map_err(|_| KVError::Internal("worker shut down".into()))?;
        rx.recv()
            .map_err(|_| KVError::Internal("worker no response".into()))?
    }

    fn delete_file(&self, path: &Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    fn file_exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn gds_available(&self) -> bool {
        #[cfg(feature = "gds")]
        {
            crate::gds::is_available()
        }
        #[cfg(not(feature = "gds"))]
        {
            false
        }
    }

    #[cfg(feature = "gds")]
    fn read_to_gpu(
        &self,
        path: &Path,
        file_offset: u64,
        buf: &mut crate::gds::GpuBuffer,
        size: usize,
    ) -> Result<usize> {
        use std::fs::OpenOptions;
        let f = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| KVError::Internal(format!("open {} (GDS): {}", path.display(), e)))?;
        let h = crate::gds::GpuFileHandle::register(f)?;
        h.pread(buf, file_offset, size)
    }

    #[cfg(feature = "gds")]
    fn write_from_gpu(
        &self,
        path: &Path,
        file_offset: u64,
        buf: &crate::gds::GpuBuffer,
        size: usize,
    ) -> Result<usize> {
        use std::fs::OpenOptions;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| KVError::Internal(format!("open {} (GDS): {}", path.display(), e)))?;
        let h = crate::gds::GpuFileHandle::register(f)?;
        h.pwrite(buf, file_offset, size)
    }

    fn read_batch(&self, requests: &[IORequest]) -> Vec<Result<Vec<u8>>> {
        // Group by device.
        let items: Vec<(IORequest, &Path)> = requests
            .iter()
            .map(|r| {
                (
                    IORequest {
                        path: r.path.clone(),
                        offset: r.offset,
                        length: r.length,
                    },
                    r.path.as_path(),
                )
            })
            .collect();
        let groups = self.group_by_device(&items);

        let mut results: Vec<Option<Result<Vec<u8>>>> = (0..requests.len()).map(|_| None).collect();
        let mut receivers: Vec<(Vec<usize>, channel::Receiver<Vec<Result<Vec<u8>>>>)> = Vec::new();

        for (device_idx, group) in groups {
            let (tx, rx) = channel::bounded(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<IORequest> = group.into_iter().map(|(_, r)| r).collect();
            let worker = &self.devices[device_idx];
            if worker
                .sender
                .send(RingJob::ReadBatch { reqs, resp: tx })
                .is_err()
            {
                for i in &orig_indices {
                    results[*i] = Some(Err(KVError::Internal("worker shut down".into())));
                }
                continue;
            }
            receivers.push((orig_indices, rx));
        }

        for (orig_indices, rx) in receivers {
            match rx.recv() {
                Ok(batch_results) => {
                    for (orig_idx, r) in orig_indices.into_iter().zip(batch_results.into_iter()) {
                        results[orig_idx] = Some(r);
                    }
                }
                Err(_) => {
                    for i in orig_indices {
                        results[i] = Some(Err(KVError::Internal("worker no response".into())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".into()))))
            .collect()
    }

    fn write_batch(&self, requests: Vec<(IORequest, Bytes)>) -> Vec<Result<()>> {
        let n = requests.len();
        // Group by device (move data into each group without cloning); device index is
        // decided by path-prefix match.
        let mut groups: BTreeMap<usize, Vec<(usize, (IORequest, Bytes))>> = BTreeMap::new();
        for (orig_idx, (r, d)) in requests.into_iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| r.path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups
                .entry(device_idx)
                .or_default()
                .push((orig_idx, (r, d)));
        }

        let mut results: Vec<Option<Result<()>>> = (0..n).map(|_| None).collect();
        let mut receivers: Vec<(Vec<usize>, channel::Receiver<Vec<Result<()>>>)> = Vec::new();

        for (device_idx, group) in groups {
            let (tx, rx) = channel::bounded(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<(IORequest, Bytes)> = group.into_iter().map(|(_, p)| p).collect();
            let worker = &self.devices[device_idx];
            if worker
                .sender
                .send(RingJob::WriteBatch { reqs, resp: tx })
                .is_err()
            {
                for i in &orig_indices {
                    results[*i] = Some(Err(KVError::Internal("worker shut down".into())));
                }
                continue;
            }
            receivers.push((orig_indices, rx));
        }

        for (orig_indices, rx) in receivers {
            match rx.recv() {
                Ok(batch_results) => {
                    for (orig_idx, r) in orig_indices.into_iter().zip(batch_results.into_iter()) {
                        results[orig_idx] = Some(r);
                    }
                }
                Err(_) => {
                    for i in orig_indices {
                        results[i] = Some(Err(KVError::Internal("worker no response".into())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".into()))))
            .collect()
    }

    /// Vectored write override: group by device, one ring per group receiving N
    /// opcode::Writev SQEs.
    /// Dual to tier_a's write_vectored — avoids the trait-default fallback that
    /// concatenates (concatenating 480MB triggers page-fault first-touch and drops
    /// PUT from 2.6 → 1.2 GB/s).
    fn write_batch_vectored(&self, requests: Vec<(IORequest, Vec<Bytes>)>) -> Vec<Result<()>> {
        let n = requests.len();
        let mut groups: BTreeMap<usize, Vec<(usize, (IORequest, Vec<Bytes>))>> = BTreeMap::new();
        for (orig_idx, (r, segs)) in requests.into_iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| r.path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups
                .entry(device_idx)
                .or_default()
                .push((orig_idx, (r, segs)));
        }

        let mut results: Vec<Option<Result<()>>> = (0..n).map(|_| None).collect();
        let mut receivers: Vec<(Vec<usize>, channel::Receiver<Vec<Result<()>>>)> = Vec::new();

        for (device_idx, group) in groups {
            let (tx, rx) = channel::bounded(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<(IORequest, Vec<Bytes>)> = group.into_iter().map(|(_, p)| p).collect();
            let worker = &self.devices[device_idx];
            if worker
                .sender
                .send(RingJob::WriteVecBatch { reqs, resp: tx })
                .is_err()
            {
                for i in &orig_indices {
                    results[*i] = Some(Err(KVError::Internal("worker shut down".into())));
                }
                continue;
            }
            receivers.push((orig_indices, rx));
        }

        for (orig_indices, rx) in receivers {
            match rx.recv() {
                Ok(batch_results) => {
                    for (orig_idx, r) in orig_indices.into_iter().zip(batch_results.into_iter()) {
                        results[orig_idx] = Some(r);
                    }
                }
                Err(_) => {
                    for i in orig_indices {
                        results[i] = Some(Err(KVError::Internal("worker no response".into())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".into()))))
            .collect()
    }

    /// O_DIRECT batched write (RDMA PUT data path) — dual to tier_a's method of the
    /// same name but on io_uring.
    /// Grouped by device and sent to each ring worker; each ring does a single
    /// submit_and_wait, and io_uring auto-throttles in-flight SQE count by ring depth,
    /// coexisting nicely with the NVMe-oF target SQ.
    ///
    /// Key advantage vs tier_a sync pwrite:
    /// - tier_a's 8 threads each issue sync pwrite; the concurrent in-flight bio count
    ///   is uncontrolled and can burst-overflow NVMe-oF SQ=16 → controller reset →
    ///   tens of seconds of hang.
    /// - tier_b's ring depth is a strict upper bound; configured ≤16 it guarantees
    ///   NVMe-oF SQ congestion is never triggered.
    fn write_aligned_batch(&self, requests: Vec<(IORequest, *const u8, usize)>) -> Vec<Result<()>> {
        let n = requests.len();
        // Group by device (path-prefix match).
        let mut groups: BTreeMap<usize, Vec<(usize, (IORequest, PtrWrapper, usize))>> =
            BTreeMap::new();
        for (orig_idx, (req, ptr, len)) in requests.into_iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| req.path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups
                .entry(device_idx)
                .or_default()
                .push((orig_idx, (req, PtrWrapper(ptr), len)));
        }

        let mut results: Vec<Option<Result<()>>> = (0..n).map(|_| None).collect();
        let mut receivers: Vec<(Vec<usize>, channel::Receiver<Vec<Result<()>>>)> = Vec::new();

        for (device_idx, group) in groups {
            let (tx, rx) = channel::bounded(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<(IORequest, PtrWrapper, usize)> =
                group.into_iter().map(|(_, p)| p).collect();
            let worker = &self.devices[device_idx];
            let job_id = self.job_seq.fetch_add(1, Ordering::Relaxed);
            if worker
                .sender
                .send(RingJob::WriteAlignedBatch {
                    job_id,
                    queued_at: std::time::Instant::now(),
                    reqs,
                    resp: tx,
                })
                .is_err()
            {
                for i in &orig_indices {
                    results[*i] = Some(Err(KVError::Internal("worker shut down".into())));
                }
                continue;
            }
            receivers.push((orig_indices, rx));
        }

        for (orig_indices, rx) in receivers {
            match rx.recv() {
                Ok(batch_results) => {
                    for (orig_idx, r) in orig_indices.into_iter().zip(batch_results.into_iter()) {
                        results[orig_idx] = Some(r);
                    }
                }
                Err(_) => {
                    for i in orig_indices {
                        results[i] = Some(Err(KVError::Internal("worker no response".into())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".into()))))
            .collect()
    }

    /// O_DIRECT batched read into caller ptr (RDMA GET cache-miss path) — symmetric
    /// to write_aligned_batch.
    /// Puts cache-miss GET on a zero reg_mr path too: storage reads directly into the
    /// slab, and the subsequent RDMA WRITE can use the slab's pre-registered lkey,
    /// eliminating the 8 per-chunk ibv_reg_mr calls of the old serve_get_fallback
    /// (~33ms of serial syscalls, 16% of GET latency).
    fn read_aligned_into_ptr_batch(
        &self,
        requests: Vec<(IORequest, *mut u8, usize)>,
    ) -> Vec<Result<usize>> {
        let n = requests.len();
        let mut groups: BTreeMap<usize, Vec<(usize, (IORequest, PtrWrapperMut, usize))>> =
            BTreeMap::new();
        for (orig_idx, (req, ptr, cap)) in requests.into_iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| req.path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups
                .entry(device_idx)
                .or_default()
                .push((orig_idx, (req, PtrWrapperMut(ptr), cap)));
        }

        let mut results: Vec<Option<Result<usize>>> = (0..n).map(|_| None).collect();
        let mut receivers: Vec<(Vec<usize>, channel::Receiver<Vec<Result<usize>>>)> = Vec::new();

        for (device_idx, group) in groups {
            let (tx, rx) = channel::bounded(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<(IORequest, PtrWrapperMut, usize)> =
                group.into_iter().map(|(_, p)| p).collect();
            let worker = &self.devices[device_idx];
            let job_id = self.job_seq.fetch_add(1, Ordering::Relaxed);
            if worker
                .sender
                .send(RingJob::ReadAlignedIntoPtrBatch {
                    job_id,
                    queued_at: std::time::Instant::now(),
                    reqs,
                    resp: tx,
                })
                .is_err()
            {
                for i in &orig_indices {
                    results[*i] = Some(Err(KVError::Internal("worker shut down".into())));
                }
                continue;
            }
            receivers.push((orig_indices, rx));
        }

        for (orig_indices, rx) in receivers {
            match rx.recv() {
                Ok(batch_results) => {
                    for (orig_idx, r) in orig_indices.into_iter().zip(batch_results.into_iter()) {
                        results[orig_idx] = Some(r);
                    }
                }
                Err(_) => {
                    for i in orig_indices {
                        results[i] = Some(Err(KVError::Internal("worker no response".into())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".into()))))
            .collect()
    }

    /// O_DIRECT streaming read (RDMA GET pipeline) — pushes completion events as
    /// soon as each IO finishes.
    /// Comparison with the batch version:
    /// - batch: collects all device batch_results and returns a Vec only after all done
    /// - stream: each device worker pushes all its IO-completion events into a shared
    ///   tx as soon as it finishes
    ///
    /// Note the current granularity is **per-device** (not per-IO — a device's N IOs
    /// only push after they all complete):
    /// - Our RDMA GET path is 8 stripes = 8 IOs = each on a different device (path
    ///   prefix decides) → each device has 1 IO → push on completion is equivalent to
    ///   per-IO.
    /// - If a device holds multiple IOs, they still complete serially inside the batch
    ///   (chunked SQEs within the batch impl), and once batch_results is done all IO
    ///   events are pushed in one shot (gain is still "parallel across devices").
    fn read_aligned_into_ptr_stream(
        &self,
        requests: Vec<(IORequest, *mut u8, usize)>,
    ) -> channel::Receiver<(usize, Result<usize>)> {
        let n = requests.len();
        let (tx, rx) = channel::unbounded::<(usize, Result<usize>)>();

        // Group by device (same as the batch impl).
        let mut groups: BTreeMap<usize, Vec<(usize, (IORequest, PtrWrapperMut, usize))>> =
            BTreeMap::new();
        for (orig_idx, (req, ptr, cap)) in requests.into_iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| req.path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups
                .entry(device_idx)
                .or_default()
                .push((orig_idx, (req, PtrWrapperMut(ptr), cap)));
        }

        if groups.is_empty() {
            return rx;
        }

        // Each device group spawns its own forwarder thread:
        // take batch_results from the device worker → immediately push each IO's
        // completion into the shared tx. So the order events reach tx matches the
        // order devices finish.
        for (device_idx, group) in groups {
            let (resp_tx, resp_rx) = channel::bounded::<Vec<Result<usize>>>(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<(IORequest, PtrWrapperMut, usize)> =
                group.into_iter().map(|(_, p)| p).collect();
            let worker = &self.devices[device_idx];
            let job_id = self.job_seq.fetch_add(1, Ordering::Relaxed);
            if worker
                .sender
                .send(RingJob::ReadAlignedIntoPtrBatch {
                    job_id,
                    queued_at: std::time::Instant::now(),
                    reqs,
                    resp: resp_tx,
                })
                .is_err()
            {
                for i in orig_indices {
                    let _ = tx.send((i, Err(KVError::Internal("worker shut down".into()))));
                }
                continue;
            }
            // forwarder: separate thread waits on the device worker and pushes the
            // result into the shared tx (uses std thread because device worker
            // rx.recv() is blocking).
            let tx_clone = tx.clone();
            std::thread::Builder::new()
                .name(format!("stream-fwd-{}", device_idx))
                .spawn(move || match resp_rx.recv() {
                    Ok(batch_results) => {
                        for (orig_idx, r) in orig_indices.into_iter().zip(batch_results) {
                            let _ = tx_clone.send((orig_idx, r));
                        }
                    }
                    Err(_) => {
                        for i in orig_indices {
                            let _ = tx_clone
                                .send((i, Err(KVError::Internal("worker no response".into()))));
                        }
                    }
                })
                .expect("spawn stream forwarder");
        }

        // Drop the original tx → rx naturally closes once all forwarder threads exit.
        // (Note: forwarders hold tx_clone; the channel closes when the last one exits.)
        drop(tx);
        let _ = n; // suppress unused warning
        rx
    }

    /// O_DIRECT batched read override: group by device, one ring per group receiving
    /// N opcode::Read SQEs, single submit_and_wait(N). Compared to tier_a's ThreadPool
    /// POSIX, io_uring saves N syscalls (though the current 8-device × 1-chunk shape
    /// has only 1 SQE per group, so gains are limited; benefit grows once stripe_chunk
    /// shrinks).
    fn read_aligned_batch(&self, requests: &[IORequest]) -> Vec<Result<Bytes>> {
        let n = requests.len();
        // Group by device.
        let mut groups: BTreeMap<usize, Vec<(usize, IORequest)>> = BTreeMap::new();
        for (orig_idx, r) in requests.iter().enumerate() {
            let device_idx = self
                .prefix_index
                .iter()
                .find(|(p, _)| r.path.starts_with(p))
                .map(|(_, i)| *i)
                .unwrap_or(0);
            groups.entry(device_idx).or_default().push((
                orig_idx,
                IORequest {
                    path: r.path.clone(),
                    offset: r.offset,
                    length: r.length,
                },
            ));
        }

        let mut results: Vec<Option<Result<Bytes>>> = (0..n).map(|_| None).collect();
        let mut receivers: Vec<(Vec<usize>, channel::Receiver<Vec<Result<Bytes>>>)> = Vec::new();

        for (device_idx, group) in groups {
            let (tx, rx) = channel::bounded(1);
            let orig_indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
            let reqs: Vec<IORequest> = group.into_iter().map(|(_, r)| r).collect();
            let worker = &self.devices[device_idx];
            if worker
                .sender
                .send(RingJob::ReadAlignedBatch { reqs, resp: tx })
                .is_err()
            {
                for i in &orig_indices {
                    results[*i] = Some(Err(KVError::Internal("worker shut down".into())));
                }
                continue;
            }
            receivers.push((orig_indices, rx));
        }

        for (orig_indices, rx) in receivers {
            match rx.recv() {
                Ok(batch_results) => {
                    for (orig_idx, r) in orig_indices.into_iter().zip(batch_results.into_iter()) {
                        results[orig_idx] = Some(r);
                    }
                }
                Err(_) => {
                    for i in orig_indices {
                        results[i] = Some(Err(KVError::Internal("worker no response".into())));
                    }
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".into()))))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_executor(tmp: &TempDir, n_devices: usize) -> TierBExecutor {
        let mut exec = TierBExecutor::new(64, n_devices).unwrap();
        for i in 0..n_devices {
            let root = tmp.path().join(format!("nvme{}", i));
            std::fs::create_dir_all(&root).unwrap();
            exec.register_device(root, 64).unwrap();
        }
        exec
    }

    #[test]
    fn single_write_read() {
        let tmp = TempDir::new().unwrap();
        let exec = setup_executor(&tmp, 1);
        let path = tmp.path().join("nvme0/file.bin");
        exec.write_file(&path, b"hello uring").unwrap();
        let data = exec.read_file(&path).unwrap();
        assert_eq!(data, b"hello uring");
    }

    #[test]
    fn batch_across_devices() {
        let tmp = TempDir::new().unwrap();
        let exec = setup_executor(&tmp, 4);

        // Write 4 files across 4 devices = 16 total.
        let mut write_reqs: Vec<(IORequest, Bytes)> = Vec::new();
        for dev in 0..4 {
            for i in 0..4 {
                let p = tmp.path().join(format!("nvme{}/f{}.bin", dev, i));
                write_reqs.push((
                    IORequest {
                        path: p,
                        offset: 0,
                        length: 0,
                    },
                    Bytes::from(format!("d{}-{}", dev, i).into_bytes()),
                ));
            }
        }
        let w = exec.write_batch(write_reqs.clone());
        for r in &w {
            r.as_ref().unwrap();
        }

        // Batch read back.
        let read_reqs: Vec<IORequest> = write_reqs
            .iter()
            .map(|(r, _)| IORequest {
                path: r.path.clone(),
                offset: 0,
                length: 0,
            })
            .collect();
        let res = exec.read_batch(&read_reqs);
        for (i, r) in res.into_iter().enumerate() {
            let want = &write_reqs[i].1;
            assert_eq!(r.unwrap().as_slice(), want.as_ref());
        }
    }
}
