//! I/O executor
//!
//! Three-tier architecture (see design doc §5):
//! - Tier A: ThreadPool + POSIX read/write   (cross-platform, simple and reliable)
//! - Tier B: io_uring                         (Linux high-performance, batched submission) [feature]
//! - Tier C: SPDK userspace NVMe              (extreme performance) [feature]
//!
//! Phase 1 only implements Tier A.

mod tier_a;
mod aligned_buffer;

pub use aligned_buffer::AlignedBuffer;

#[cfg(all(feature = "io-uring", target_os = "linux"))]
mod tier_b;

#[cfg(feature = "spdk")]
mod tier_c;

use crate::config::Config;
use crate::error::Result;
use prost::bytes::{Bytes, BytesMut};
use std::path::Path;
use std::sync::Arc;

pub use tier_a::TierAExecutor;

/// A single I/O request.
#[derive(Debug, Clone)]
pub struct IORequest {
    pub path: std::path::PathBuf,
    pub offset: u64,
    pub length: usize,
}

/// Abstract I/O executor.
pub trait IOExecutor: Send + Sync {
    fn read_file(&self, path: &Path) -> Result<Vec<u8>>;
    fn write_file(&self, path: &Path, data: &[u8]) -> Result<()>;
    fn delete_file(&self, path: &Path) -> Result<()>;
    fn file_exists(&self, path: &Path) -> bool;

    /// Batched read — the key parallel interface.
    fn read_batch(&self, requests: &[IORequest]) -> Vec<Result<Vec<u8>>>;
    /// Batched write — `data` is a `Bytes` (Arc-refcounted), letting multiple chunks share the
    /// same underlying buffer in zero-copy slice scenarios.
    /// (Previous version used `Vec<u8>`; a 480MB striped write would trigger 8 chunk.to_vec calls
    ///  = ~600ms of pure allocation overhead. After switching to Bytes, put_striped uses
    ///  data.slice(start..end) to split into 8 segments — all just refcount bumps.)
    fn write_batch(&self, requests: Vec<(IORequest, Bytes)>) -> Vec<Result<()>>;

    /// Vectored batched write — each IO item's data is **multiple Bytes segments**,
    /// flushed to disk in a single `writev` syscall.
    ///
    /// Purpose: the 240 × 2MB Bytes accumulated by put_stream are no longer merged into a
    /// single 480MB block (which would trigger page-fault first-touch writes at 0.6-0.8 GB/s);
    /// instead, 32 chunks aligned to stripe boundaries are handed to writev as `IoSlice`s,
    /// and the kernel flushes all segments to the file in one go — zero merging, zero copy.
    ///
    /// Default implementation: falls back to `write_batch` (concatenates all segments per IO item);
    /// tier_a provides the writev fast path.
    fn write_batch_vectored(&self, requests: Vec<(IORequest, Vec<Bytes>)>) -> Vec<Result<()>> {
        // Default degradation: concatenate multiple Bytes segments into one (triggers 480MB
        // allocation; keeps behavior compatible with older callers). tier_a overrides this
        // fast path with IoSlice + write_all_vectored.
        let flat: Vec<(IORequest, Bytes)> = requests
            .into_iter()
            .map(|(req, segments)| {
                let total: usize = segments.iter().map(|s| s.len()).sum();
                let mut buf = BytesMut::with_capacity(total);
                for s in &segments {
                    buf.extend_from_slice(s);
                }
                (req, buf.freeze())
            })
            .collect();
        self.write_batch(flat)
    }

    /// O_DIRECT batched read — bypasses the page cache, reads NVMe-oF / local disks directly,
    /// with 4KB-aligned buffers.
    ///
    /// Purpose: on the GET hot path through storage_tier::read_striped_chunks, use O_DIRECT
    /// to bypass the ~3.3 GB/s buffered-I/O ceiling (fio measures O_DIRECT at 8 GB/s vs
    /// buffered at 3.3 GB/s).
    ///
    /// The returned `Bytes` is already an owned `AlignedBuffer` zero-copy-wrapped via
    /// `Bytes::from_owner`. Upper layers can `Bytes::slice()` sub-chunks directly for
    /// tonic encoding (slice does not require alignment).
    ///
    /// Default implementation: falls back to `read_batch` (buffered, converts Vec<u8> to Bytes);
    /// tier_a provides the O_DIRECT fast path.
    fn read_aligned_batch(&self, requests: &[IORequest]) -> Vec<Result<Bytes>> {
        // Default buffered path; tier_a overrides to go through O_DIRECT.
        self.read_batch(requests)
            .into_iter()
            .map(|r| r.map(Bytes::from))
            .collect()
    }

    /// O_DIRECT batched write — data is already in caller-pinned 4K-aligned memory (typically:
    /// RDMA slab extent), workers pwrite O_DIRECT directly, **zero memcpy**.
    ///
    /// Each item `(req, ptr, len)`:
    /// - req.path: file path (temp `.tmp` then rename, preserving atomicity as in write_vec_impl)
    /// - ptr: 4K-aligned host memory pointer (caller guarantees lifetime until all workers finish)
    /// - len: valid bytes for this stripe (write_vec_impl internally rounds up to 4K padding+truncate)
    ///
    /// # Safety (caller responsibility)
    /// - ptr..ptr+aligned_up(len, 4096) must be entirely readable
    /// - Must not free / mutate this memory before join returns
    /// - ptr must be 4K-aligned (slab's SLAB_ALIGN=4096 already satisfies this)
    ///
    /// Default implementation: no safe buffered fallback (ptr has no lifetime); returns Err
    /// directly. tier_a overrides with the actual implementation.
    fn write_aligned_batch(
        &self,
        _requests: Vec<(IORequest, *const u8, usize)>,
    ) -> Vec<Result<()>> {
        // No reasonable fallback — Err lets the caller degrade gracefully rather than panicking.
        // If some executor hasn't implemented this, the caller should use write_batch_vectored
        // (the copy path) instead.
        vec![Err(crate::error::KVError::Internal(
            "write_aligned_batch not implemented by this executor".into(),
        ))]
    }

    /// O_DIRECT batched read (into caller-pinned 4K-aligned memory) — symmetric to
    /// `write_aligned_batch`. Used on the RDMA GET cache-miss path: server slab.alloc(N) →
    /// read NVMe directly into slab ptr → send RDMA WRITE using slab's pre-registered MR
    /// (**zero reg_mr**).
    ///
    /// Each item `(req, ptr, capacity)`:
    /// - req.path: file path
    /// - ptr: 4K-aligned destination memory pointer (caller holds slab extent)
    /// - capacity: maximum bytes writable into ptr (must be ≥ file size; 4K-aligned)
    ///
    /// Returns Vec<Result<usize>>: bytes actually read per item (== file size, not counting
    /// 4K padding).
    ///
    /// # Safety (caller responsibility)
    /// - ptr..ptr+capacity must be entirely writable
    /// - Must not free / mutate / share before join returns
    /// - ptr must be 4K-aligned
    /// - capacity must be 4K-aligned and >= file_size
    ///
    /// Default implementation: same as write_aligned_batch — no reasonable fallback, returns Err.
    fn read_aligned_into_ptr_batch(
        &self,
        _requests: Vec<(IORequest, *mut u8, usize)>,
    ) -> Vec<Result<usize>> {
        vec![Err(crate::error::KVError::Internal(
            "read_aligned_into_ptr_batch not implemented by this executor".into(),
        ))]
    }

    /// O_DIRECT streaming read — accepts N IO requests like `read_aligned_into_ptr_batch`,
    /// but pushes a completion event on the channel **as soon as each one finishes**
    /// (without waiting for the others).
    ///
    /// Purpose: storage+RDMA **pipeline** on the RDMA GET cache-miss path.
    /// The existing batch interface must wait for all stripes to complete before returning →
    /// only then can the server post RDMA WRITEs. The streaming interface lets the server
    /// post an RDMA WRITE as soon as it receives one segment, truly overlapping storage
    /// with RDMA (expected to reduce ~RDMA WRITE time, i.e. max(storage,RDMA) rather than sum).
    ///
    /// Returns:
    /// - `Receiver<(usize, Result<usize>)>`: each recv yields one IO completion event
    ///   - first usize: index into the requests array (so the server knows which stripe)
    ///   - Result<usize>: bytes actually read for that IO
    /// - After all IOs complete the channel closes naturally, and recv returns Err.
    ///
    /// # Safety
    /// Same as the batch version: caller guarantees ptr is not freed/mutated until all
    /// events have been received.
    ///
    /// Default implementation: falls back to the batch version (waits synchronously for all
    /// completions, then serially sends completion events). Lets executors that haven't
    /// implemented streaming still run, just without the pipeline advantage.
    fn read_aligned_into_ptr_stream(
        &self,
        requests: Vec<(IORequest, *mut u8, usize)>,
    ) -> crossbeam_channel::Receiver<(usize, Result<usize>)> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let results = self.read_aligned_into_ptr_batch(requests);
        for (i, r) in results.into_iter().enumerate() {
            let _ = tx.send((i, r));
        }
        rx
    }

    /// Returns whether the current executor supports GPU direct DMA (GDS).
    /// Defaults to false; only true when Tier B + gds feature + runtime probe all succeed.
    fn gds_available(&self) -> bool {
        false
    }

    /// GDS read: NVMe → the specified GPU buffer (zero-copy).
    /// Not implemented by default; goes through cuFile under Tier B + gds feature.
    #[cfg(feature = "gds")]
    fn read_to_gpu(
        &self,
        _path: &Path,
        _file_offset: u64,
        _buf: &mut crate::gds::GpuBuffer,
        _size: usize,
    ) -> Result<usize> {
        Err(crate::error::KVError::Internal(
            "executor does not support GDS read".into(),
        ))
    }

    /// GDS write: GPU → NVMe (zero-copy).
    #[cfg(feature = "gds")]
    fn write_from_gpu(
        &self,
        _path: &Path,
        _file_offset: u64,
        _buf: &crate::gds::GpuBuffer,
        _size: usize,
    ) -> Result<usize> {
        Err(crate::error::KVError::Internal(
            "executor does not support GDS write".into(),
        ))
    }
}

/// Create the appropriate executor based on config.
pub fn create_executor(config: &Config) -> Result<Arc<dyn IOExecutor>> {
    match config.io_executor.kind.as_str() {
        "tier_a" => Ok(Arc::new(TierAExecutor::new(config.io_executor.thread_pool_size))),
        #[cfg(all(feature = "io-uring", target_os = "linux"))]
        "tier_b" => {
            let mut exec = tier_b::TierBExecutor::new(
                config.io_executor.io_uring_depth,
                config.storage.devices.len().max(1),
            )?;
            // Register a dedicated ring per storage device (avoids head-of-line blocking across devices).
            for dev in &config.storage.devices {
                let root = dev.join(&config.storage.data_subdir);
                std::fs::create_dir_all(&root).ok();
                exec.register_device(root, config.io_executor.io_uring_depth as u32)?;
            }
            Ok(Arc::new(exec))
        }
        #[cfg(feature = "spdk")]
        "tier_c" => Ok(Arc::new(tier_c::TierCExecutor::new(config)?)),
        other => Err(crate::error::KVError::Config(format!(
            "unknown or unenabled io_executor.kind: {} (does it need a feature flag?)",
            other
        ))),
    }
}
