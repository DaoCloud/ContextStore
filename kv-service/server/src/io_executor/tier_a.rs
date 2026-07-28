//! Tier A — ThreadPool + POSIX read/write
//!
//! Cross-platform baseline implementation, suitable for the Phase 1 MVP.
//! On Linux, ReadAligned uses O_DIRECT to bypass the page cache and read NVMe-oF
//! directly (3.3 → 6+ GB/s).

use super::{
    log_io_batch, log_io_error, AlignedBuffer, IOExecutor, IORequest, IoBatchStats, IoLogContext,
};
use crate::error::{KVError, Result};
use crossbeam_channel as channel;
use prost::bytes::Bytes;
use std::path::Path;
use std::thread;

/// Linux O_DIRECT flag value (x86_64 / aarch64 = 0o40000); fully equivalent to libc::O_DIRECT.
/// Inlined as a constant to avoid depending on libc (CLAUDE.md: don't modify Cargo.toml deps).
#[cfg(target_os = "linux")]
const O_DIRECT_FLAG: i32 = 0o40000;
#[cfg(not(target_os = "linux"))]
const O_DIRECT_FLAG: i32 = 0;

/// Alignment required by O_DIRECT (Linux standard page size = filesystem block size = NVMe physical sector).
pub const DIRECT_IO_ALIGN: usize = 4096;

pub struct TierAExecutor {
    workers: Vec<thread::JoinHandle<()>>,
    sender: channel::Sender<Job>,
}

enum Job {
    Read {
        req: IORequest,
        resp: channel::Sender<Result<Vec<u8>>>,
    },
    Write {
        req: IORequest,
        data: Bytes,
        resp: channel::Sender<Result<()>>,
    },
    /// Vectored write: use writev in a single syscall to avoid the page-fault first-touch
    /// hit that happens when concatenating into a single buffer.
    /// Each segment is an independent Bytes (a refcount view over the gRPC framework's buffer);
    /// no new buffer is allocated overall.
    WriteVec {
        req: IORequest,
        segments: Vec<Bytes>,
        resp: channel::Sender<Result<()>>,
    },
    /// O_DIRECT write: data is already in caller-pinned 4K-aligned memory (RDMA slab extent),
    /// pwritten directly with **zero memcpy**. Caller guarantees ptr lifetime until resp returns.
    ///
    /// Uses PtrWrapper so the raw pointer can cross the Send boundary (channel requires Send).
    /// The worker thread never mutates the memory, only reads it via pwrite, so Send is safe
    /// (assuming the caller does not mutate during our read).
    WriteAligned {
        req: IORequest,
        ptr: PtrWrapper,
        len: usize,
        resp: channel::Sender<Result<()>>,
    },
    /// O_DIRECT read: bypass the page cache with an aligned buffer + pread O_DIRECT.
    /// resp returns Bytes (AlignedBuffer wrapped zero-copy via from_owner); upper layers can
    /// slice freely without alignment constraints.
    ReadAligned {
        path: std::path::PathBuf,
        resp: channel::Sender<Result<Bytes>>,
    },
    Shutdown,
}

/// Wraps `*const u8` so it can cross a `Send` channel. Ownership stays with the caller; we only read.
/// Safety precondition: caller guarantees the memory ptr points to is not freed or mutated
/// until join returns.
struct PtrWrapper(*const u8);
unsafe impl Send for PtrWrapper {}

impl TierAExecutor {
    pub fn new(num_workers: usize) -> Self {
        let (tx, rx) = channel::unbounded::<Job>();

        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            // crossbeam Receiver is MPMC; clone it per worker and each recv directly.
            // No need for Arc<Mutex<Receiver>> — that's the std::sync::mpsc Receiver pattern.
            // crossbeam is lock-free internally; wrapping in a Mutex would make all workers
            // contend for the same lock and serialize them heavily.
            // (Measured: before the fix, 8-chunk read through the thread pool couldn't hit
            //  8×850MB/s aggregate for exactly this reason.)
            let rx = rx.clone();
            workers.push(thread::spawn(move || loop {
                let job = rx.recv();
                match job {
                    Ok(Job::Read { req, resp }) => {
                        let _ = resp.send(read_file_impl(&req));
                    }
                    Ok(Job::Write { req, data, resp }) => {
                        let _ = resp.send(write_file_impl(&req, &data));
                    }
                    Ok(Job::WriteVec {
                        req,
                        segments,
                        resp,
                    }) => {
                        let _ = resp.send(write_vec_impl(&req, &segments));
                    }
                    Ok(Job::WriteAligned {
                        req,
                        ptr,
                        len,
                        resp,
                    }) => {
                        let result = write_aligned_impl(&req, ptr.0, len);
                        if let Err(error) = &result {
                            log_io_error(
                                IoLogContext {
                                    executor: "tier_a",
                                    operation: "write",
                                    mode: "aligned_batch",
                                    device_id: -1,
                                    job_id: 0,
                                },
                                &req,
                                len,
                                error,
                            );
                        }
                        let _ = resp.send(result);
                    }
                    Ok(Job::ReadAligned { path, resp }) => {
                        let _ = resp.send(read_aligned_impl(&path));
                    }
                    Ok(Job::Shutdown) | Err(_) => break,
                }
            }));
        }

        Self {
            workers,
            sender: tx,
        }
    }
}

impl Drop for TierAExecutor {
    fn drop(&mut self) {
        for _ in 0..self.workers.len() {
            let _ = self.sender.send(Job::Shutdown);
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn read_file_impl(req: &IORequest) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(&req.path)?;
    if req.offset > 0 {
        f.seek(SeekFrom::Start(req.offset))?;
    }
    let mut buf = if req.length > 0 {
        Vec::with_capacity(req.length)
    } else {
        let len = f.metadata()?.len() as usize;
        Vec::with_capacity(len)
    };
    if req.length > 0 {
        buf.resize(req.length, 0);
        f.read_exact(&mut buf)?;
    } else {
        f.read_to_end(&mut buf)?;
    }
    Ok(buf)
}

fn write_file_impl(req: &IORequest, data: &[u8]) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = req.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write to a temp file then rename atomically, avoiding half-written state.
    let tmp = req.path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(data)?;
        // KV cache is recomputable data, so losing cache on crash is acceptable → skip
        // per-write fsync (major speedup on the hot path).
        // Set CS_SYNC_WRITES=1 to restore strict fsync semantics.
        if std::env::var("CS_SYNC_WRITES").as_deref() == Ok("1") {
            f.sync_data()?;
        }
    }
    std::fs::rename(&tmp, &req.path)?;
    Ok(())
}

/// Vectored write: use write_vectored to flush all segments in a single syscall — zero merging,
/// zero copy. The kernel's writev limit is UIO_MAXIOV=1024 segments; 240 segments is well below that.
fn write_vec_impl(req: &IORequest, segments: &[Bytes]) -> Result<()> {
    use std::os::unix::fs::{FileExt, OpenOptionsExt};

    let t0 = std::time::Instant::now();
    if let Some(parent) = req.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = req.path.with_extension("tmp");

    let total: usize = segments.iter().map(|s| s.len()).sum();
    let aligned_len = (total + DIRECT_IO_ALIGN - 1) & !(DIRECT_IO_ALIGN - 1);

    let f_direct = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT_FLAG)
        .open(&tmp);
    let t_open = t0.elapsed();

    match f_direct {
        Ok(f) => {
            // thread_local buffer reuse: avoids alloc'ing 64MB + first-touch page fault
            // on every call (measured: 88ms!). Each worker holds one buffer that grows
            // dynamically (grow-only, amortizing alloc cost).
            // Key: on first creation the whole block must be memset (prefaulted); otherwise
            // demand-paging turns subsequent copy_nonoverlapping into ~16k page faults = 255ms.
            // After prefault, reuse costs 7ms.
            thread_local! {
                static BUF: std::cell::RefCell<AlignedBuffer> = {
                    let mut b = AlignedBuffer::new(64 * 1024 * 1024, DIRECT_IO_ALIGN);
                    // prefault: touch every page so the kernel allocates physical memory
                    unsafe { std::ptr::write_bytes(b.as_mut_ptr(), 0, b.capacity()); }
                    std::cell::RefCell::new(b)
                };
            }
            BUF.with(|cell| -> Result<()> {
                let mut buf = cell.borrow_mut();
                // Reallocate if the current buf isn't big enough (rare; most stripes are 64MB).
                if buf.capacity() < aligned_len {
                    *buf = AlignedBuffer::new(aligned_len, DIRECT_IO_ALIGN);
                }
                unsafe {
                    let mut dst = buf.as_mut_ptr();
                    for seg in segments {
                        std::ptr::copy_nonoverlapping(seg.as_ptr(), dst, seg.len());
                        dst = dst.add(seg.len());
                    }
                    // zero-fill padding if needed (allocation is already zeroed)
                    if aligned_len > total {
                        std::ptr::write_bytes(buf.as_mut_ptr().add(total), 0, aligned_len - total);
                    }
                }
                let t_memcpy = t0.elapsed();
                let slice = unsafe { std::slice::from_raw_parts(buf.as_mut_ptr(), aligned_len) };
                let mut written = 0;
                while written < aligned_len {
                    let n = f.write_at(&slice[written..], written as u64)
                        .map_err(KVError::Io)?;
                    if n == 0 {
                        return Err(KVError::Internal("O_DIRECT pwrite returned 0".into()));
                    }
                    written += n;
                }
                let t_write = t0.elapsed();
                if std::env::var("CS_SYNC_WRITES").as_deref() == Ok("1") {
                    f.sync_data()?;
                }
                drop(f);
                if aligned_len != total {
                    let f2 = std::fs::OpenOptions::new().write(true).open(&tmp)?;
                    f2.set_len(total as u64)?;
                }
                let t_rename_start = t0.elapsed();
                std::fs::rename(&tmp, &req.path)?;
                let t_end = t0.elapsed();
                tracing::trace!(
                    "WV_PERF bytes={} open={}us memcpy={}us pwrite={}us trunc={}us rename={}us total={}ms",
                    total,
                    t_open.as_micros(),
                    (t_memcpy - t_open).as_micros(),
                    (t_write - t_memcpy).as_micros(),
                    (t_rename_start - t_write).as_micros(),
                    (t_end - t_rename_start).as_micros(),
                    t_end.as_millis(),
                );
                Ok(())
            })?;
            return Ok(());
        }
        Err(e) if e.raw_os_error() == Some(22) => {
            return write_vec_impl_buffered(req, segments, &tmp);
        }
        Err(e) => return Err(KVError::Io(e)),
    }
}

/// O_DIRECT write: data is already in 4K-aligned memory (RDMA slab extent) — pwrite directly,
/// zero memcpy.
///
/// Key difference from write_vec_impl: allocates nothing, copies nothing — uses the ptr
/// directly as the pwrite source. That's why the caller must pin the memory and 4K-align it
/// (O_DIRECT hard requirement).
///
/// Flow:
/// 1. Compute aligned_len = round_up(len, 4096).
/// 2. open tmp with O_DIRECT.
/// 3. If aligned_len > len: is it legal to just use caller's ptr[0..aligned_len]?
///    Answer: **no**. Caller only guarantees ptr..ptr+len is valid data; ptr+len..ptr+aligned_len
///    may be out of bounds / contain sensitive data. Must pwrite in two steps:
///    - pwrite(ptr[0..aligned_down(len)]) — all a 4K multiple
///    - tail: use a thread-local 4K bounce buffer; memcpy the trailing <4K into it and pwrite.
///    Simplified variant: if len isn't a 4K multiple, fall back to buffered (no correctness loss;
///    in practice all 480MB stripes are 64MB integer 4K multiples, so the tail path almost
///    never triggers).
/// 4. truncate to len (drop the padding).
/// 5. rename.
fn write_aligned_impl(req: &IORequest, ptr: *const u8, len: usize) -> Result<()> {
    use std::os::unix::fs::{FileExt, OpenOptionsExt};

    let t0 = std::time::Instant::now();
    if let Some(parent) = req.path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = req.path.with_extension("tmp");

    // Must be 4K aligned: caller (slab) guarantees ptr is 4K aligned; len is not necessarily a 4K multiple.
    if (ptr as usize) % DIRECT_IO_ALIGN != 0 {
        return Err(KVError::Internal(format!(
            "write_aligned_impl: ptr {:p} not 4K-aligned",
            ptr
        )));
    }

    // len is a 4K multiple → single pwrite covers everything (most common path; all 64MB stripes).
    // len is not a 4K multiple → tail bounce buffer.
    let aligned_down = len & !(DIRECT_IO_ALIGN - 1);
    let tail = len - aligned_down;

    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT_FLAG)
        .open(&tmp);
    let t_open = t0.elapsed();

    match f {
        Ok(f) => {
            // Main body: aligned_down bytes pwritten directly from caller's ptr (zero memcpy).
            if aligned_down > 0 {
                let main_slice = unsafe { std::slice::from_raw_parts(ptr, aligned_down) };
                let mut written = 0;
                while written < aligned_down {
                    let n = f
                        .write_at(&main_slice[written..], written as u64)
                        .map_err(KVError::Io)?;
                    if n == 0 {
                        return Err(KVError::Internal("O_DIRECT pwrite returned 0".into()));
                    }
                    written += n;
                }
            }
            // tail: sub-4K trailing bytes go through a thread-local 4K bounce buffer
            // (memset 0 + copy actual data), pwrite 4K, then truncate(len) to drop padding.
            if tail > 0 {
                thread_local! {
                    static TAIL_BUF: std::cell::RefCell<AlignedBuffer> =
                        std::cell::RefCell::new(
                            AlignedBuffer::new(DIRECT_IO_ALIGN, DIRECT_IO_ALIGN)
                        );
                }
                TAIL_BUF.with(|cell| -> Result<()> {
                    let mut buf = cell.borrow_mut();
                    unsafe {
                        std::ptr::write_bytes(buf.as_mut_ptr(), 0, DIRECT_IO_ALIGN);
                        std::ptr::copy_nonoverlapping(
                            ptr.add(aligned_down),
                            buf.as_mut_ptr(),
                            tail,
                        );
                    }
                    let slice =
                        unsafe { std::slice::from_raw_parts(buf.as_mut_ptr(), DIRECT_IO_ALIGN) };
                    let mut written = 0;
                    while written < DIRECT_IO_ALIGN {
                        let n = f
                            .write_at(&slice[written..], (aligned_down + written) as u64)
                            .map_err(KVError::Io)?;
                        if n == 0 {
                            return Err(KVError::Internal(
                                "O_DIRECT pwrite tail returned 0".into(),
                            ));
                        }
                        written += n;
                    }
                    Ok(())
                })?;
            }
            let t_write = t0.elapsed();

            if std::env::var("CS_SYNC_WRITES").as_deref() == Ok("1") {
                f.sync_data()?;
            }
            drop(f);

            // Truncate away the tail padding: one set_len(len).
            if tail > 0 {
                let f2 = std::fs::OpenOptions::new().write(true).open(&tmp)?;
                f2.set_len(len as u64)?;
            }
            std::fs::rename(&tmp, &req.path)?;
            let t_end = t0.elapsed();
            tracing::trace!(
                "WA_PERF bytes={} open={}us pwrite={}us total={}ms tail={}",
                len,
                t_open.as_micros(),
                (t_write - t_open).as_micros(),
                t_end.as_millis(),
                tail,
            );
            Ok(())
        }
        Err(e) if e.raw_os_error() == Some(22) => {
            // O_DIRECT unsupported: fallback to buffered (slow path; only triggers in rare dev envs).
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            std::fs::write(&tmp, slice).map_err(KVError::Io)?;
            std::fs::rename(&tmp, &req.path)?;
            Ok(())
        }
        Err(e) => Err(KVError::Io(e)),
    }
}

/// Fallback: buffered writev (when O_DIRECT is unavailable). Same as the original impl.
fn write_vec_impl_buffered(
    req: &IORequest,
    segments: &[Bytes],
    tmp: &std::path::Path,
) -> Result<()> {
    use std::io::{IoSlice, Write};
    let mut f = std::fs::File::create(tmp)?;
    let mut start = 0usize;
    let mut offset = 0usize;
    while start < segments.len() {
        let mut bufs: Vec<IoSlice<'_>> = Vec::with_capacity(segments.len() - start);
        bufs.push(IoSlice::new(&segments[start].as_ref()[offset..]));
        for s in &segments[start + 1..] {
            bufs.push(IoSlice::new(s.as_ref()));
        }
        let n = f.write_vectored(&bufs)?;
        if n == 0 {
            return Err(KVError::Internal("writev returned 0 bytes".into()));
        }
        let mut remaining = n;
        let first_avail = segments[start].len() - offset;
        if remaining < first_avail {
            offset += remaining;
        } else {
            remaining -= first_avail;
            start += 1;
            offset = 0;
            while start < segments.len() && remaining >= segments[start].len() {
                remaining -= segments[start].len();
                start += 1;
            }
            if remaining > 0 && start < segments.len() {
                offset = remaining;
            }
        }
    }
    if std::env::var("CS_SYNC_WRITES").as_deref() == Ok("1") {
        f.sync_data()?;
    }
    drop(f);
    std::fs::rename(tmp, &req.path)?;
    Ok(())
}

/// O_DIRECT full-file read: bypass the page cache, read NVMe(-oF) directly.
/// - alignment: 4KB (DIRECT_IO_ALIGN); file offset 0, len typically ≤ 64MB (one stripe chunk).
/// - tail handling: if file_size isn't a 4K multiple, O_DIRECT-read up to the full 4K multiple
///   (this reads past file_size into zero-fill, but XFS/ext4 permits it — the kernel returns
///   file_size bytes and 0-pads to 4K), then set_len(file_size) to trim to the real length.
///   This avoids falling back to buffered for the last segment.
///
/// Non-5.x Linux kernels / non-Linux platforms: fall back to buffered read (read_file_impl + Bytes::from).
#[cfg(target_os = "linux")]
fn read_aligned_impl(path: &Path) -> Result<Bytes> {
    use std::os::unix::fs::{FileExt, OpenOptionsExt};

    let t0 = std::time::Instant::now();
    let file_size = std::fs::metadata(path)?.len() as usize;
    if file_size == 0 {
        return Ok(Bytes::new());
    }
    let aligned_len = (file_size + DIRECT_IO_ALIGN - 1) & !(DIRECT_IO_ALIGN - 1);
    let mut buf = AlignedBuffer::new(aligned_len, DIRECT_IO_ALIGN);
    let t_alloc = t0.elapsed();

    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT_FLAG)
        .open(path);
    let t_open = t0.elapsed();
    let f = match f {
        Ok(f) => f,
        Err(e) if e.raw_os_error() == Some(22) => {
            return read_buffered_to_bytes(path, file_size);
        }
        Err(e) => return Err(KVError::Io(e)),
    };

    let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr(), aligned_len) };
    let mut total = 0usize;
    while total < aligned_len {
        match f.read_at(&mut slice[total..], total as u64) {
            Ok(0) => break,
            Ok(n) => {
                total += n;
                if total < aligned_len && total % DIRECT_IO_ALIGN != 0 {
                    drop(f);
                    return read_buffered_to_bytes(path, file_size);
                }
            }
            Err(e) => return Err(KVError::Io(e)),
        }
        if total >= file_size {
            break;
        }
    }
    let t_read = t0.elapsed();
    buf.set_len(file_size);
    let res = Bytes::from_owner(buf);
    let t_end = t0.elapsed();
    tracing::trace!(
        "IO_BREAK alloc={}us open={}us read={}us bytes_wrap={}us total={}us bytes={}",
        t_alloc.as_micros(),
        (t_open - t_alloc).as_micros(),
        (t_read - t_open).as_micros(),
        (t_end - t_read).as_micros(),
        t_end.as_micros(),
        file_size,
    );
    Ok(res)
}

/// Fallback for non-Linux platforms / when O_DIRECT is unsupported.
fn read_buffered_to_bytes(path: &Path, file_size: usize) -> Result<Bytes> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::with_capacity(file_size);
    f.read_to_end(&mut buf)?;
    Ok(Bytes::from(buf))
}

#[cfg(not(target_os = "linux"))]
fn read_aligned_impl(path: &Path) -> Result<Bytes> {
    let file_size = std::fs::metadata(path)?.len() as usize;
    read_buffered_to_bytes(path, file_size)
}

impl IOExecutor for TierAExecutor {
    fn read_file(&self, path: &Path) -> Result<Vec<u8>> {
        let req = IORequest {
            path: path.to_path_buf(),
            offset: 0,
            length: 0,
        };
        let (tx, rx) = channel::bounded(1);
        self.sender
            .send(Job::Read { req, resp: tx })
            .map_err(|_| KVError::Internal("executor closed".to_string()))?;
        rx.recv()
            .map_err(|_| KVError::Internal("worker response failed".to_string()))?
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> Result<()> {
        let req = IORequest {
            path: path.to_path_buf(),
            offset: 0,
            length: data.len(),
        };
        let (tx, rx) = channel::bounded(1);
        self.sender
            .send(Job::Write {
                req,
                // Single write_file is a cold path (write_batch is the hot path); copying into
                // Bytes just aligns the field type.
                data: Bytes::copy_from_slice(data),
                resp: tx,
            })
            .map_err(|_| KVError::Internal("executor closed".to_string()))?;
        rx.recv()
            .map_err(|_| KVError::Internal("worker response failed".to_string()))?
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

    fn read_batch(&self, requests: &[IORequest]) -> Vec<Result<Vec<u8>>> {
        let mut recvs = Vec::with_capacity(requests.len());
        for req in requests {
            let (tx, rx) = channel::bounded(1);
            if self
                .sender
                .send(Job::Read {
                    req: IORequest {
                        path: req.path.clone(),
                        offset: req.offset,
                        length: req.length,
                    },
                    resp: tx,
                })
                .is_err()
            {
                recvs.push(None);
                continue;
            }
            recvs.push(Some(rx));
        }
        recvs
            .into_iter()
            .map(|opt| match opt {
                Some(rx) => rx
                    .recv()
                    .map_err(|_| KVError::Internal("worker response failed".to_string()))?,
                None => Err(KVError::Internal("executor closed".to_string())),
            })
            .collect()
    }

    fn write_batch(&self, requests: Vec<(IORequest, Bytes)>) -> Vec<Result<()>> {
        let mut recvs = Vec::with_capacity(requests.len());
        for (req, data) in requests {
            let (tx, rx) = channel::bounded(1);
            if self
                .sender
                .send(Job::Write {
                    req,
                    data,
                    resp: tx,
                })
                .is_err()
            {
                recvs.push(None);
                continue;
            }
            recvs.push(Some(rx));
        }
        recvs
            .into_iter()
            .map(|opt| match opt {
                Some(rx) => rx
                    .recv()
                    .map_err(|_| KVError::Internal("worker response failed".to_string()))?,
                None => Err(KVError::Internal("executor closed".to_string())),
            })
            .collect()
    }

    /// Fast path: multi-segment Bytes are not concatenated; they're flushed via writev in a
    /// single syscall. See trait::write_batch_vectored for usage.
    fn write_batch_vectored(&self, requests: Vec<(IORequest, Vec<Bytes>)>) -> Vec<Result<()>> {
        let mut recvs = Vec::with_capacity(requests.len());
        for (req, segments) in requests {
            let (tx, rx) = channel::bounded(1);
            if self
                .sender
                .send(Job::WriteVec {
                    req,
                    segments,
                    resp: tx,
                })
                .is_err()
            {
                recvs.push(None);
                continue;
            }
            recvs.push(Some(rx));
        }
        recvs
            .into_iter()
            .map(|opt| match opt {
                Some(rx) => rx
                    .recv()
                    .map_err(|_| KVError::Internal("worker response failed".to_string()))?,
                None => Err(KVError::Internal("executor closed".to_string())),
            })
            .collect()
    }

    /// O_DIRECT batched write — pwrites directly from caller-pinned 4K-aligned memory,
    /// **zero memcpy**. 8 stripes run concurrently across 8 workers, each with its own
    /// pwrite syscall. See trait::write_aligned_batch for usage.
    fn write_aligned_batch(&self, requests: Vec<(IORequest, *const u8, usize)>) -> Vec<Result<()>> {
        let started = std::time::Instant::now();
        let request_count = requests.len();
        let requested_bytes: usize = requests.iter().map(|request| request.2).sum();
        let mut recvs = Vec::with_capacity(requests.len());
        for (req, ptr, len) in requests {
            let (tx, rx) = channel::bounded(1);
            if self
                .sender
                .send(Job::WriteAligned {
                    req,
                    ptr: PtrWrapper(ptr),
                    len,
                    resp: tx,
                })
                .is_err()
            {
                recvs.push((len, None));
                continue;
            }
            recvs.push((len, Some(rx)));
        }
        let mut completed_bytes = 0usize;
        let results: Vec<Result<()>> = recvs
            .into_iter()
            .map(|(bytes, opt)| {
                let result = match opt {
                    Some(rx) => rx
                        .recv()
                        .map_err(|_| KVError::Internal("worker response failed".to_string()))
                        .and_then(|result| result),
                    None => Err(KVError::Internal("executor closed".to_string())),
                };
                if result.is_ok() {
                    completed_bytes += bytes;
                }
                result
            })
            .collect();
        let success_count = results.iter().filter(|result| result.is_ok()).count();
        log_io_batch(
            IoLogContext {
                executor: "tier_a",
                operation: "write",
                mode: "aligned_batch",
                device_id: -1,
                job_id: 0,
            },
            IoBatchStats {
                request_count,
                success_count,
                failure_count: request_count.saturating_sub(success_count),
                requested_bytes,
                completed_bytes,
                queue_wait_us: 0,
                duration_us: started.elapsed().as_micros() as u64,
            },
        );
        results
    }

    /// O_DIRECT batched read: 8 segments run in parallel, each with its own aligned buffer +
    /// pread O_DIRECT, returning zero-copy Bytes. See trait::read_aligned_batch for usage.
    fn read_aligned_batch(&self, requests: &[IORequest]) -> Vec<Result<Bytes>> {
        let started = std::time::Instant::now();
        let request_count = requests.len();
        let requested_bytes: usize = requests.iter().map(|request| request.length).sum();
        let mut recvs = Vec::with_capacity(requests.len());
        for req in requests {
            let (tx, rx) = channel::bounded(1);
            if self
                .sender
                .send(Job::ReadAligned {
                    path: req.path.clone(),
                    resp: tx,
                })
                .is_err()
            {
                recvs.push(None);
                continue;
            }
            recvs.push(Some(rx));
        }
        let results: Vec<Result<Bytes>> = recvs
            .into_iter()
            .map(|opt| match opt {
                Some(rx) => rx
                    .recv()
                    .map_err(|_| KVError::Internal("worker response failed".to_string()))?,
                None => Err(KVError::Internal("executor closed".to_string())),
            })
            .collect();
        let success_count = results.iter().filter(|result| result.is_ok()).count();
        let completed_bytes = results
            .iter()
            .filter_map(|result| result.as_ref().ok().map(Bytes::len))
            .sum();
        log_io_batch(
            IoLogContext {
                executor: "tier_a",
                operation: "read",
                mode: "aligned_batch",
                device_id: -1,
                job_id: 0,
            },
            IoBatchStats {
                request_count,
                success_count,
                failure_count: request_count.saturating_sub(success_count),
                requested_bytes,
                completed_bytes,
                queue_wait_us: 0,
                duration_us: started.elapsed().as_micros() as u64,
            },
        );
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let exec = TierAExecutor::new(4);
        let path = tmp.path().join("test.bin");
        exec.write_file(&path, b"hello world").unwrap();
        let data = exec.read_file(&path).unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn batch_read_parallel() {
        let tmp = TempDir::new().unwrap();
        let exec = TierAExecutor::new(8);
        let mut reqs = Vec::new();
        for i in 0..16 {
            let p = tmp.path().join(format!("f{}.bin", i));
            exec.write_file(&p, format!("data-{}", i).as_bytes())
                .unwrap();
            reqs.push(IORequest {
                path: p,
                offset: 0,
                length: 0,
            });
        }
        let results = exec.read_batch(&reqs);
        assert_eq!(results.len(), 16);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.as_ref().unwrap(), format!("data-{}", i).as_bytes());
        }
    }
}
