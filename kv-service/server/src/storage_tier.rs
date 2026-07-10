//! Storage Tier (L2) — persistent storage
//!
//! Responsibilities:
//! - Use the router to decide the physical location of a key.
//! - Use io_executor to perform the actual I/O.
//! - Maintain block metadata (via MetadataService).
//! - Auto-stripe large values: values over the threshold are split into chunks distributed across
//!   multiple devices.

use crate::config::Config;
use crate::error::{KVError, Result};
use crate::io_executor::{create_executor, IOExecutor, IORequest};
use crate::metadata::{BlockMeta, MetadataService, StripingInfo};
use crate::router::{ObjectKey, ShardRouter};
use dashmap::DashMap;
use parking_lot::Mutex;
use prost::bytes::{Bytes, BytesMut};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info};
use twox_hash::xxh3::hash64;

pub struct StorageTier {
    router: Arc<ShardRouter>,
    executor: Arc<dyn IOExecutor>,
    metadata: Arc<MetadataService>,
    write_locks: DashMap<String, Arc<Mutex<()>>>,
    striping_threshold: u64,
    striping_chunk_size: u64,

    // ===== Stats (exported to Prometheus) =====
    pub reads_total: AtomicU64,
    pub writes_total: AtomicU64,
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,
    pub striped_writes: AtomicU64,
    pub device_read_bytes_total: Arc<Vec<AtomicU64>>,
}

impl StorageTier {
    pub fn new(config: &Config, router: Arc<ShardRouter>) -> Result<Self> {
        let executor = create_executor(config)?;
        let metadata = Arc::new(MetadataService::new(config)?);
        let num_devices = router.num_devices();
        // Create the root directory for each device.
        for device in router.devices() {
            let root = device.join(&config.storage.data_subdir).join("data");
            std::fs::create_dir_all(&root).ok();
        }
        Ok(Self {
            router,
            executor,
            metadata,
            write_locks: DashMap::new(),
            striping_threshold: config.storage.striping_threshold,
            striping_chunk_size: config.storage.striping_chunk_size.max(1),
            reads_total: AtomicU64::new(0),
            writes_total: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            striped_writes: AtomicU64::new(0),
            device_read_bytes_total: Arc::new(
                (0..num_devices).map(|_| AtomicU64::new(0)).collect(),
            ),
        })
    }

    pub fn metadata(&self) -> Arc<MetadataService> {
        self.metadata.clone()
    }

    pub fn router(&self) -> Arc<ShardRouter> {
        self.router.clone()
    }

    pub fn striping_threshold(&self) -> u64 {
        self.striping_threshold
    }

    pub fn striping_chunk_size(&self) -> u64 {
        self.striping_chunk_size
    }

    pub fn device_read_bytes(&self, device_id: usize) -> u64 {
        self.device_read_bytes_total
            .get(device_id)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn record_device_read(&self, device_id: usize, bytes: u64) {
        if let Some(counter) = self.device_read_bytes_total.get(device_id) {
            counter.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    fn meta_path_or_route(&self, key: &ObjectKey, meta: &BlockMeta) -> PathBuf {
        if meta.file_path.is_empty() {
            self.router.key_to_path(key)
        } else {
            PathBuf::from(&meta.file_path)
        }
    }

    fn meta_device_or_route(&self, key: &ObjectKey, meta: &BlockMeta) -> usize {
        let device_id = meta.device_id as usize;
        if device_id < self.router.num_devices() {
            device_id
        } else {
            self.router.route(key)
        }
    }

    fn content_etag(key: &ObjectKey, generation: u64, size: u64, created_at: i64) -> String {
        let seed = format!(
            "{}|generation={}|size={}|created_at={}",
            key.to_string_key(),
            generation,
            size,
            created_at
        );
        format!("{:016x}", hash64(seed.as_bytes()))
    }

    fn object_handle(
        key: &ObjectKey,
        generation: u64,
        layout_version: u64,
        size: u64,
        created_at: i64,
    ) -> String {
        let seed = format!(
            "{}|generation={}|layout={}|size={}|created_at={}",
            key.to_string_key(),
            generation,
            layout_version,
            size,
            created_at
        );
        format!(
            "ctxobj-v1-{}-g{}-l{}-{:016x}",
            key.object_digest(),
            generation,
            layout_version,
            hash64(seed.as_bytes())
        )
    }

    fn key_write_lock(&self, key: &ObjectKey) -> Arc<Mutex<()>> {
        let str_key = key.to_string_key();
        self.write_locks
            .entry(str_key)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    pub fn prepare_write_meta(
        &self,
        key: &ObjectKey,
        mut meta: BlockMeta,
        size: u64,
    ) -> Result<BlockMeta> {
        let generation = self.metadata.next_generation(&key.to_string_key())?;
        let layout_version = 1;
        meta.size = size;
        meta.object_generation = generation;
        meta.layout_version = layout_version;
        meta.object_handle =
            Self::object_handle(key, generation, layout_version, size, meta.created_at);
        meta.content_etag = Self::content_etag(key, generation, meta.size, meta.created_at);
        Ok(meta)
    }

    fn ensure_managed_path(&self, path: &std::path::Path) -> Result<()> {
        let managed = self
            .router
            .devices()
            .iter()
            .any(|device| path.starts_with(device));
        if !managed {
            return Err(KVError::InvalidArgument(format!(
                "storage_handle outside managed devices: {}",
                path.display()
            )));
        }
        Ok(())
    }

    /// Data-node internal API: write an object stripe with a determined generation/layout.
    pub fn put_placement_chunk(
        &self,
        key: &ObjectKey,
        stripe_index: usize,
        generation: u64,
        layout_version: u64,
        data: Bytes,
    ) -> Result<(u32, String)> {
        let device_id = self.router.chunk_device(key, stripe_index);
        let path = self.router.chunk_versioned_path(
            key,
            stripe_index,
            device_id,
            generation,
            layout_version,
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        self.executor.write_file(&path, &data)?;
        self.writes_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok((device_id as u32, path.to_string_lossy().to_string()))
    }

    /// Validate that a placement chunk handle matches the descriptor identity and router layout.
    pub fn validate_placement_chunk_handle(
        &self,
        key: &ObjectKey,
        stripe_index: usize,
        generation: u64,
        layout_version: u64,
        device_id: u32,
        storage_handle: &str,
    ) -> Result<()> {
        let path = PathBuf::from(storage_handle);
        self.ensure_managed_path(&path)?;
        let device_id = device_id as usize;
        if device_id >= self.router.num_devices() {
            return Err(KVError::InvalidArgument(format!(
                "placement chunk device out of range: {}",
                device_id
            )));
        }
        let expected = self.router.chunk_versioned_path(
            key,
            stripe_index,
            device_id,
            generation,
            layout_version,
        );
        if path != expected {
            return Err(KVError::InvalidArgument(format!(
                "placement chunk handle does not match descriptor: {}",
                path.display()
            )));
        }
        Ok(())
    }

    /// Data-node internal API: read a stripe by a storage_handle previously returned by the server.
    pub fn read_placement_chunk(
        &self,
        storage_handle: &str,
        expected_len: u64,
    ) -> Result<Option<Bytes>> {
        let path = PathBuf::from(storage_handle);
        self.ensure_managed_path(&path)?;
        if !self.executor.file_exists(&path) {
            return Ok(None);
        }
        let data = self.executor.read_file(&path)?;
        if expected_len > 0 && data.len() as u64 != expected_len {
            return Err(KVError::Internal(format!(
                "placement chunk length mismatch: expected {} got {} ({})",
                expected_len,
                data.len(),
                path.display()
            )));
        }
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(Some(Bytes::from(data)))
    }

    /// Data-node internal API: delete a placement chunk, used for object deletion or write-failure
    /// rollback.
    pub fn delete_placement_chunk(&self, storage_handle: &str) -> Result<bool> {
        let path = PathBuf::from(storage_handle);
        self.ensure_managed_path(&path)?;
        let existed = self.executor.file_exists(&path);
        self.executor.delete_file(&path)?;
        Ok(existed)
    }

    /// Whether a value should be striped.
    fn should_stripe(&self, len: usize) -> bool {
        self.striping_threshold > 0
            && (len as u64) > self.striping_threshold
            && self.router.num_devices() > 1
    }

    // ===== Single-entry =====

    pub fn get(&self, key: &ObjectKey) -> Result<Option<(Bytes, BlockMeta)>> {
        let str_key = key.to_string_key();
        let meta = match self.metadata.get_block(&str_key)? {
            Some(m) => m,
            None => return Ok(None),
        };

        // Striped path: read all chunks in parallel and concat.
        if let Some(stripe) = &meta.striping {
            let data = self.read_striped(key, stripe)?;
            self.reads_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_read
                .fetch_add(data.len() as u64, Ordering::Relaxed);
            return Ok(Some((Bytes::from(data), meta)));
        }

        let path = self.meta_path_or_route(key, &meta);
        if !self.executor.file_exists(&path) {
            self.metadata.delete_block(&str_key)?;
            return Ok(None);
        }
        debug!("L2 GET {} -> {}", str_key, path.display());
        let data = self.executor.read_file(&path)?;
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.record_device_read(self.meta_device_or_route(key, &meta), data.len() as u64);
        // Vec<u8> → Bytes is a zero-copy handoff (Bytes::from takes over the Vec's allocation).
        Ok(Some((Bytes::from(data), meta)))
    }

    pub fn put(&self, key: &ObjectKey, data: Bytes, meta: BlockMeta) -> Result<()> {
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock();

        if self.should_stripe(data.len()) {
            return self.put_striped(key, data, meta);
        }

        let mut meta = self.prepare_write_meta(key, meta, data.len() as u64)?;
        let device_id = self.router.route(key);
        let path = self.router.key_to_versioned_path(
            key,
            device_id,
            meta.object_generation,
            meta.layout_version,
        );
        // Ensure the parent directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        debug!(
            "L2 PUT {} -> {} ({} bytes)",
            key.to_string_key(),
            path.display(),
            data.len()
        );
        let nbytes = data.len();
        // Non-striping single path: write_file takes &[u8]; data is only borrowed here, no copy.
        self.executor.write_file(&path, &data)?;
        meta.file_path = path.to_string_lossy().to_string();
        meta.size = nbytes as u64;
        meta.device_id = device_id as u32;
        meta.striping = None;
        self.metadata.put_block(&key.to_string_key(), &meta)?;
        self.writes_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(nbytes as u64, Ordering::Relaxed);
        Ok(())
    }

    /// PUT-if-absent: check metadata and write under the same per-key lock.
    pub fn put_if_absent(&self, key: &ObjectKey, data: Bytes, meta: BlockMeta) -> Result<bool> {
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock();

        if self.metadata.get_block(&key.to_string_key())?.is_some() {
            return Ok(false);
        }
        if self.should_stripe(data.len()) {
            let total = data.len();
            return self.put_striped_chunks_impl(key, vec![data], total, meta, true);
        }

        let mut meta = self.prepare_write_meta(key, meta, data.len() as u64)?;
        let device_id = self.router.route(key);
        let path = self.router.key_to_versioned_path(
            key,
            device_id,
            meta.object_generation,
            meta.layout_version,
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let nbytes = data.len();
        self.executor.write_file(&path, &data)?;
        meta.file_path = path.to_string_lossy().to_string();
        meta.size = nbytes as u64;
        meta.device_id = device_id as u32;
        meta.striping = None;
        if !self
            .metadata
            .put_block_if_absent(&key.to_string_key(), &meta)?
        {
            let _ = self.executor.delete_file(&path);
            return Ok(false);
        }
        self.writes_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(nbytes as u64, Ordering::Relaxed);
        Ok(true)
    }

    /// Multi-segment PUT — zero-copy passthrough: caller supplies a set of `Bytes` (typical: 240 2MB
    /// chunks accumulated by put_stream). On the striping path, segments are rebucketed on
    /// `striping_chunk_size` boundaries into N groups, each `writev`'d to a single disk file in one
    /// syscall. **Never concatenated into a single large Bytes**, so we avoid the 480MB anonymous
    /// mmap first-touch (~780ms page-fault write).
    ///
    /// Small values that skip striping: fall back to concatenating a single copy here (they are
    /// small, so the copy is negligible).
    pub fn put_chunks(&self, key: &ObjectKey, segments: Vec<Bytes>, meta: BlockMeta) -> Result<()> {
        let total: usize = segments.iter().map(|s| s.len()).sum();
        if !self.should_stripe(total) {
            // Small value: concat and take the old path (copy overhead is negligible).
            let mut buf = BytesMut::with_capacity(total);
            for s in &segments {
                buf.extend_from_slice(s);
            }
            return self.put(key, buf.freeze(), meta);
        }
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock();
        self.put_striped_chunks(key, segments, total, meta)
    }

    /// Multi-segment PUT-if-absent. The existence check and write share one per-key lock, so
    /// concurrent stream writes for the same object cannot overwrite each other.
    pub fn put_chunks_if_absent(
        &self,
        key: &ObjectKey,
        segments: Vec<Bytes>,
        meta: BlockMeta,
    ) -> Result<bool> {
        let total: usize = segments.iter().map(|s| s.len()).sum();
        if !self.should_stripe(total) {
            let mut buf = BytesMut::with_capacity(total);
            for s in &segments {
                buf.extend_from_slice(s);
            }
            return self.put_if_absent(key, buf.freeze(), meta);
        }
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock();
        if self.metadata.get_block(&key.to_string_key())?.is_some() {
            return Ok(false);
        }
        self.put_striped_chunks_impl(key, segments, total, meta, true)
    }

    /// **RDMA PUT data-plane only** — data is already in caller-pinned 4K-aligned memory (an RDMA
    /// slab extent). Directly O_DIRECT pwrite N stripes across N NVMes, **zero memcpy, zero concat**.
    ///
    /// Key difference vs put_chunks: put_chunks goes through write_vec_impl which internally memcpy's
    /// 240 segments into a thread_local AlignedBuffer (138–255ms/64MB bug) and then pwrites.
    /// put_from_ptr skips memcpy entirely; ptr is the pwrite source (slab is already 4K-aligned +
    /// pinned).
    ///
    /// Parameters:
    /// - ptr: 4K-aligned start address (slab extent base)
    /// - total: valid data length in bytes
    /// - meta: BlockMeta (file_path / size / device_id are overwritten by this fn)
    ///
    /// # Safety
    /// The caller must guarantee:
    /// - ptr..ptr+total is not freed / mutated before every worker's pwrite completes.
    /// - ptr is 4K-aligned (slab.alloc already guarantees this).
    /// - total > 0 (empty values should not reach here; they should take the put path).
    ///
    /// Small values that skip striping: they still take the striped path here (the RDMA PUT is
    /// designed for large values; small values are faster on the gRPC path anyway).
    pub fn put_from_ptr(
        &self,
        key: &ObjectKey,
        ptr: *const u8,
        total: usize,
        meta: BlockMeta,
    ) -> Result<()> {
        if total == 0 {
            return Err(KVError::Internal("put_from_ptr: empty data".into()));
        }
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock();
        // The RDMA PUT path only serves large values; small values never take this route. But as a
        // fallback: no stripe → single-file pwrite (via write_aligned_batch with one item).
        if !self.should_stripe(total) {
            let mut meta = self.prepare_write_meta(key, meta, total as u64)?;
            let device_id = self.router.route(key);
            let path = self.router.key_to_versioned_path(
                key,
                device_id,
                meta.object_generation,
                meta.layout_version,
            );
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let req = IORequest {
                path: path.clone(),
                offset: 0,
                length: total,
            };
            let results = self.executor.write_aligned_batch(vec![(req, ptr, total)]);
            results.into_iter().next().unwrap_or_else(|| {
                Err(KVError::Internal("write_aligned_batch empty result".into()))
            })?;
            meta.file_path = path.to_string_lossy().to_string();
            meta.size = total as u64;
            meta.device_id = device_id as u32;
            meta.striping = None;
            self.metadata.put_block(&key.to_string_key(), &meta)?;
            self.writes_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_written
                .fetch_add(total as u64, Ordering::Relaxed);
            return Ok(());
        }
        self.put_from_ptr_striped(key, ptr, total, meta)
    }

    /// Internal: put_from_ptr implementation for the striping path.
    ///
    /// Key point: ptr[stripe_idx * chunk_size .. (stripe_idx+1) * chunk_size] is directly the pwrite
    /// source for that stripe; no memory split is needed. 8 stripes → 8 workers each pwrite
    /// (writes_aligned_batch is internally parallel).
    fn put_from_ptr_striped(
        &self,
        key: &ObjectKey,
        ptr: *const u8,
        total: usize,
        meta: BlockMeta,
    ) -> Result<()> {
        let mut meta = self.prepare_write_meta(key, meta, total as u64)?;
        let chunk_size = self.striping_chunk_size as usize;
        let n_stripes = total.div_ceil(chunk_size);

        let mut chunk_devices: Vec<u32> = Vec::with_capacity(n_stripes);
        let mut chunk_paths: Vec<String> = Vec::with_capacity(n_stripes);
        let mut io_items: Vec<(IORequest, *const u8, usize)> = Vec::with_capacity(n_stripes);

        for i in 0..n_stripes {
            let stripe_start = i * chunk_size;
            let stripe_end = (stripe_start + chunk_size).min(total);
            let stripe_len = stripe_end - stripe_start;
            let dev_id = self.router.chunk_device(key, i);
            let path = self.router.chunk_versioned_path(
                key,
                i,
                dev_id,
                meta.object_generation,
                meta.layout_version,
            );
            // Ensure the parent directory exists (subdirectories may not exist on the first PUT for a
            // new namespace/object).
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            chunk_devices.push(dev_id as u32);
            chunk_paths.push(path.to_string_lossy().to_string());
            // Key point: slab ptr + offset, no memory copy. SLAB_ALIGN=4096 guarantees sub-segments
            // are also 4K-aligned (chunk_size default 64MB is an integer multiple of 4K).
            let stripe_ptr = unsafe { ptr.add(stripe_start) };
            io_items.push((
                IORequest {
                    path,
                    offset: 0,
                    length: stripe_len,
                },
                stripe_ptr,
                stripe_len,
            ));
        }

        debug!(
            "L2 PUT striped(ptr) {} -> {} stripes ({} bytes total) ZERO-COPY",
            key.to_string_key(),
            n_stripes,
            total
        );

        let results = self.executor.write_aligned_batch(io_items);
        for (i, r) in results.into_iter().enumerate() {
            if let Err(e) = r {
                // Roll back the stripe files already written.
                for path in &chunk_paths[..i] {
                    let _ = self.executor.delete_file(std::path::Path::new(path));
                }
                return Err(e);
            }
        }

        meta.size = total as u64;
        meta.device_id = chunk_devices[0];
        meta.file_path = String::new();
        meta.striping = Some(StripingInfo {
            chunk_size: self.striping_chunk_size,
            chunk_devices,
            chunk_paths,
            total_size: total as u64,
            chunk_locations: Vec::new(),
        });
        self.metadata.put_block(&key.to_string_key(), &meta)?;
        self.writes_total.fetch_add(1, Ordering::Relaxed);
        self.striped_writes.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(total as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn delete(&self, key: &ObjectKey) -> Result<bool> {
        let str_key = key.to_string_key();
        let meta = self.metadata.get_block(&str_key)?;
        let existed = meta.is_some();

        if let Some(meta) = meta {
            if let Some(stripe) = &meta.striping {
                for path in &stripe.chunk_paths {
                    let _ = self.executor.delete_file(std::path::Path::new(path));
                }
            } else {
                let path = self.meta_path_or_route(key, &meta);
                self.executor.delete_file(&path)?;
            }
        }
        self.metadata.delete_block(&str_key)?;
        Ok(existed)
    }

    pub fn exists(&self, key: &ObjectKey) -> Result<bool> {
        self.metadata.exists_block(&key.to_string_key())
    }

    // ===== GDS direct read/write (zero-copy) =====
    // Striping and compression are not supported on this path yet; callers should check should_stripe
    // and return an error early.
    #[cfg(feature = "gds")]
    pub fn get_to_gpu(
        &self,
        key: &ObjectKey,
        gpu_buf: &mut crate::gds::GpuBuffer,
    ) -> Result<Option<(usize, BlockMeta)>> {
        let str_key = key.to_string_key();
        let meta = match self.metadata.get_block(&str_key)? {
            Some(m) => m,
            None => return Ok(None),
        };
        if meta.striping.is_some() {
            return Err(crate::error::KVError::InvalidArgument(
                "GDS path does not support striped values yet (TODO: multi-chunk DMA)".into(),
            ));
        }
        let path = self.meta_path_or_route(key, &meta);
        if !self.executor.file_exists(&path) {
            self.metadata.delete_block(&str_key)?;
            return Ok(None);
        }
        let size = meta.size as usize;
        let n = self.executor.read_to_gpu(&path, 0, gpu_buf, size)?;
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
        self.record_device_read(self.meta_device_or_route(key, &meta), n as u64);
        Ok(Some((n, meta)))
    }

    #[cfg(feature = "gds")]
    pub fn put_from_gpu(
        &self,
        key: &ObjectKey,
        gpu_buf: &crate::gds::GpuBuffer,
        size: usize,
        meta: BlockMeta,
    ) -> Result<()> {
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock();

        if self.should_stripe(size) {
            return Err(crate::error::KVError::InvalidArgument(
                "GDS path does not support striped values yet".into(),
            ));
        }
        let mut meta = self.prepare_write_meta(key, meta, size as u64)?;
        let device_id = self.router.route(key);
        let path = self.router.key_to_versioned_path(
            key,
            device_id,
            meta.object_generation,
            meta.layout_version,
        );
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let n = self.executor.write_from_gpu(&path, 0, gpu_buf, size)?;
        meta.size = n as u64;
        meta.file_path = path.to_string_lossy().to_string();
        meta.device_id = device_id as u32;
        self.metadata.put_block(&key.to_string_key(), &meta)?;
        self.writes_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_written.fetch_add(n as u64, Ordering::Relaxed);
        Ok(())
    }

    // ===== Striping path =====

    fn put_striped(&self, key: &ObjectKey, data: Bytes, meta: BlockMeta) -> Result<()> {
        let total = data.len();
        let chunk_size = self.striping_chunk_size as usize;
        let n = (total + chunk_size - 1) / chunk_size;

        // Slice the single Bytes into n Bytes (Arc refcount, zero-copy), then reuse put_striped_chunks.
        let mut segments: Vec<Bytes> = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(total);
            segments.push(data.slice(start..end));
        }
        // Each segment is already a complete stripe; the rebucketing inside put_striped_chunks
        // degenerates to identity here (each segment length == chunk_size).
        self.put_striped_chunks(key, segments, total, meta)
    }

    /// Multi-segment stripe write — rebucket segments on `chunk_size` boundaries into N stripes,
    /// each stripe writev-ing multiple Bytes in one syscall (zero concat, zero copy).
    ///
    /// Key point: when caller-supplied segments (typical: 240 2MB chunks) cross a stripe boundary,
    /// use `Bytes::slice` to split it in two (refcount, no copy) and assign each half to the adjacent
    /// two stripes.
    fn put_striped_chunks(
        &self,
        key: &ObjectKey,
        segments: Vec<Bytes>,
        total: usize,
        meta: BlockMeta,
    ) -> Result<()> {
        self.put_striped_chunks_impl(key, segments, total, meta, false)
            .map(|_| ())
    }

    fn put_striped_chunks_impl(
        &self,
        key: &ObjectKey,
        segments: Vec<Bytes>,
        total: usize,
        meta: BlockMeta,
        if_absent: bool,
    ) -> Result<bool> {
        let mut meta = self.prepare_write_meta(key, meta, total as u64)?;
        let chunk_size = self.striping_chunk_size as usize;
        let n_stripes = (total + chunk_size - 1) / chunk_size;

        let mut chunk_devices: Vec<u32> = Vec::with_capacity(n_stripes);
        let mut chunk_paths: Vec<String> = Vec::with_capacity(n_stripes);

        // Rebucket segments on chunk_size boundaries: each stripe = Vec<Bytes>.
        // Time complexity O(segments + stripes); only Bytes::slice (Arc bump), no large buffer
        // allocation.
        let mut stripes: Vec<Vec<Bytes>> = (0..n_stripes).map(|_| Vec::new()).collect();
        let mut cur_segment = 0usize;
        let mut cur_offset = 0usize; // Bytes already consumed within the current segment.
        for stripe_idx in 0..n_stripes {
            let stripe_start = stripe_idx * chunk_size;
            let stripe_end = (stripe_start + chunk_size).min(total);
            let mut filled = stripe_start;
            while filled < stripe_end && cur_segment < segments.len() {
                let seg = &segments[cur_segment];
                let seg_remaining = seg.len() - cur_offset;
                let stripe_need = stripe_end - filled;
                let take = seg_remaining.min(stripe_need);
                if take == seg.len() && cur_offset == 0 {
                    // Whole segment belongs to this stripe (zero-copy, just clone the Arc).
                    stripes[stripe_idx].push(seg.clone());
                } else {
                    stripes[stripe_idx].push(seg.slice(cur_offset..cur_offset + take));
                }
                filled += take;
                cur_offset += take;
                if cur_offset == seg.len() {
                    cur_segment += 1;
                    cur_offset = 0;
                }
            }
        }

        // Build batch write requests: each stripe → one file on one device (writev writes multiple
        // Bytes).
        let mut io_items: Vec<(IORequest, Vec<Bytes>)> = Vec::with_capacity(n_stripes);
        for (i, stripe_segs) in stripes.into_iter().enumerate() {
            let stripe_len: usize = stripe_segs.iter().map(|s| s.len()).sum();
            let dev_id = self.router.chunk_device(key, i);
            let path = self.router.chunk_versioned_path(
                key,
                i,
                dev_id,
                meta.object_generation,
                meta.layout_version,
            );
            // Ensure the parent directory exists.
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            chunk_devices.push(dev_id as u32);
            chunk_paths.push(path.to_string_lossy().to_string());
            io_items.push((
                IORequest {
                    path,
                    offset: 0,
                    length: stripe_len,
                },
                stripe_segs,
            ));
        }

        debug!(
            "L2 PUT striped(chunks) {} -> {} stripes ({} bytes total)",
            key.to_string_key(),
            n_stripes,
            total
        );

        let results = self.executor.write_batch_vectored(io_items);
        for (i, r) in results.into_iter().enumerate() {
            if let Err(e) = r {
                // Roll back the stripe files already written.
                for path in &chunk_paths[..i] {
                    let _ = self.executor.delete_file(std::path::Path::new(path));
                }
                return Err(e);
            }
        }

        meta.size = total as u64;
        meta.device_id = chunk_devices[0];
        meta.file_path = String::new();
        meta.striping = Some(StripingInfo {
            chunk_size: self.striping_chunk_size,
            chunk_devices,
            chunk_paths,
            total_size: total as u64,
            chunk_locations: Vec::new(),
        });
        let committed = if if_absent {
            self.metadata
                .put_block_if_absent(&key.to_string_key(), &meta)?
        } else {
            self.metadata.put_block(&key.to_string_key(), &meta)?;
            true
        };
        if !committed {
            if let Some(stripe) = &meta.striping {
                for path in &stripe.chunk_paths {
                    let _ = self.executor.delete_file(std::path::Path::new(path));
                }
            }
            return Ok(false);
        }
        self.writes_total.fetch_add(1, Ordering::Relaxed);
        self.striped_writes.fetch_add(1, Ordering::Relaxed);
        self.bytes_written
            .fetch_add(total as u64, Ordering::Relaxed);
        Ok(true)
    }

    fn read_striped(&self, key: &ObjectKey, stripe: &StripingInfo) -> Result<Vec<u8>> {
        // Backwards compat for the old API: go through read_striped_chunks and concat to Vec<u8>
        // (this concat is a necessary copy because old callers expect a single buffer; hot paths like
        // get_stream should switch to read_striped_chunks passthrough).
        let segments = self.read_striped_chunks(key, stripe)?;
        let total: usize = segments.iter().map(|s| s.len()).sum();
        let mut out = Vec::with_capacity(total);
        for s in &segments {
            out.extend_from_slice(s);
        }
        Ok(out)
    }

    /// Multi-segment read — zero-copy passthrough: parallel read_aligned_batch on 8 segments
    /// (O_DIRECT, bypassing page cache), returning Vec<Bytes>. Each Bytes wraps a 4KB-aligned
    /// AlignedBuffer via from_owner; upper-layer Bytes::slice for sub-chunks does not require
    /// alignment, and encoding to gRPC does not copy.
    ///
    /// Use: after get_stream gets this Vec<Bytes>, **each segment is streamed to the client as a
    /// DataChunk** without concatenating a 480MB block anywhere (the old read_striped concat's
    /// ~600ms page-fault write has been eliminated).
    fn read_striped_chunks(&self, _key: &ObjectKey, stripe: &StripingInfo) -> Result<Vec<Bytes>> {
        let reqs: Vec<IORequest> = stripe
            .chunk_paths
            .iter()
            .map(|p| IORequest {
                path: std::path::PathBuf::from(p),
                offset: 0,
                length: 0,
            })
            .collect();

        // O_DIRECT takes read_aligned_batch (tier_a override); buffered fallback takes the trait
        // default.
        let results = self.executor.read_aligned_batch(&reqs);
        let mut out: Vec<Bytes> = Vec::with_capacity(results.len());
        for (i, r) in results.into_iter().enumerate() {
            let chunk =
                r.map_err(|e| KVError::Internal(format!("read striped chunk {}: {}", i, e)))?;
            if let Some(device_id) = stripe.chunk_devices.get(i) {
                self.record_device_read(*device_id as usize, chunk.len() as u64);
            }
            out.push(chunk);
        }
        Ok(out)
    }

    /// Read an object using metadata supplied by the caller; do not re-lookup by logical key.
    ///
    /// This is the core path for descriptor reads: the server first validates the descriptor with
    /// metadata, then reads data using the physical layout pointed to by the same metadata, avoiding
    /// reading a different version by logical key between validation and read.
    pub fn get_with_meta(
        &self,
        key: &ObjectKey,
        meta: &BlockMeta,
    ) -> Result<Option<(Bytes, BlockMeta)>> {
        if let Some(stripe) = &meta.striping {
            let data = self.read_striped(key, stripe)?;
            self.reads_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_read
                .fetch_add(data.len() as u64, Ordering::Relaxed);
            return Ok(Some((Bytes::from(data), meta.clone())));
        }

        let path = self.meta_path_or_route(key, meta);
        if !self.executor.file_exists(&path) {
            return Ok(None);
        }
        let data = self.executor.read_file(&path)?;
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.record_device_read(self.meta_device_or_route(key, meta), data.len() as u64);
        Ok(Some((Bytes::from(data), meta.clone())))
    }

    /// Multi-segment descriptor read; does not re-lookup by logical key.
    pub fn get_chunks_with_meta(
        &self,
        key: &ObjectKey,
        meta: &BlockMeta,
    ) -> Result<Option<(Vec<Bytes>, BlockMeta)>> {
        if let Some(stripe) = &meta.striping {
            let segments = self.read_striped_chunks(key, stripe)?;
            let total: u64 = segments.iter().map(|s| s.len() as u64).sum();
            self.reads_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(total, Ordering::Relaxed);
            return Ok(Some((segments, meta.clone())));
        }

        let path = self.meta_path_or_route(key, meta);
        if !self.executor.file_exists(&path) {
            return Ok(None);
        }
        let data = self.executor.read_file(&path)?;
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.record_device_read(self.meta_device_or_route(key, meta), data.len() as u64);
        Ok(Some((vec![Bytes::from(data)], meta.clone())))
    }

    /// **RDMA GET data-plane only** — read directly into caller-pinned 4K-aligned memory (typical:
    /// slab extent), skipping the heap AlignedBuffer intermediary and per-chunk reg_mr.
    ///
    /// Differs from read_striped_chunks:
    /// - Old: read NVMe → heap AlignedBuffer × 8 → return Vec<Bytes> → caller serve_get_fallback
    ///   does 8 ibv_reg_mr calls (~33ms serial) + post_write × 8.
    /// - New: read NVMe → write slab ptr directly (pre-registered MR) → caller uses slab.lkey with a
    ///   single post_write, **saving 33ms of reg_mr overhead**.
    ///
    /// Parameters:
    /// - key: used for metadata lookup to get striping_info
    /// - ptr: start address of the SlabExtent obtained via slab.alloc(meta.size) (4K aligned)
    /// - capacity: max writable bytes at ptr (≥ size, 4K aligned; slab.alloc already rounds up).
    ///
    /// Returns: (actual bytes, BlockMeta) | None (key does not exist)
    ///
    /// # Safety (caller's responsibility)
    /// - ptr..ptr+capacity is entirely writable.
    /// - Not freed / mutated / shared with other workers before this function returns.
    /// - Actual bytes written are indicated by the return value (typically equal to meta.size).
    pub fn get_into_ptr(
        &self,
        key: &ObjectKey,
        ptr: *mut u8,
        capacity: usize,
    ) -> Result<Option<(usize, BlockMeta)>> {
        let str_key = key.to_string_key();
        let meta = match self.metadata.get_block(&str_key)? {
            Some(m) => m,
            None => return Ok(None),
        };

        // Non-striped single file.
        let Some(stripe) = &meta.striping else {
            let path = self.meta_path_or_route(key, &meta);
            if !self.executor.file_exists(&path) {
                self.metadata.delete_block(&str_key)?;
                return Ok(None);
            }
            let req = IORequest {
                path,
                offset: 0,
                length: 0,
            };
            let results = self
                .executor
                .read_aligned_into_ptr_batch(vec![(req, ptr, capacity)]);
            let bytes_read = results
                .into_iter()
                .next()
                .unwrap_or_else(|| Err(KVError::Internal("empty result".into())))?;
            self.reads_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_read
                .fetch_add(bytes_read as u64, Ordering::Relaxed);
            self.record_device_read(self.meta_device_or_route(key, &meta), bytes_read as u64);
            return Ok(Some((bytes_read, meta)));
        };

        // striped: read 8 segments in parallel into different offsets of ptr.
        let chunk_size = stripe.chunk_size as usize;
        let total = stripe.total_size as usize;
        if capacity < total {
            return Err(KVError::Internal(format!(
                "get_into_ptr: capacity {} < total {}",
                capacity, total
            )));
        }

        let n_stripes = stripe.chunk_paths.len();
        let mut reqs: Vec<(IORequest, *mut u8, usize)> = Vec::with_capacity(n_stripes);
        for (i, p) in stripe.chunk_paths.iter().enumerate() {
            let stripe_offset = i * chunk_size;
            let stripe_end = ((i + 1) * chunk_size).min(total);
            let stripe_len = stripe_end - stripe_offset;
            // 4K-aligned round-up length for this stripe (used for capacity checks; tier_b
            // recomputes internally).
            let aligned_stripe = (stripe_len + 4095) & !4095;
            let stripe_ptr = unsafe { ptr.add(stripe_offset) };
            // Capacity for this stripe = capacity - stripe_offset (but at least aligned_stripe).
            // Simplified: use chunk_size (matches the PUT path, each stripe is at most chunk_size).
            let stripe_cap = (capacity - stripe_offset).min((chunk_size + 4095) & !4095);
            if stripe_cap < aligned_stripe {
                return Err(KVError::Internal(format!(
                    "stripe {} cap {} < aligned {}",
                    i, stripe_cap, aligned_stripe
                )));
            }
            reqs.push((
                IORequest {
                    path: std::path::PathBuf::from(p),
                    offset: 0,
                    length: 0,
                },
                stripe_ptr,
                stripe_cap,
            ));
        }

        // Read 8 offsets of ptr in parallel (tier_b groups by device across different rings).
        let results = self.executor.read_aligned_into_ptr_batch(reqs);
        let mut total_read = 0usize;
        for (i, r) in results.into_iter().enumerate() {
            let n =
                r.map_err(|e| KVError::Internal(format!("read stripe {} into ptr: {}", i, e)))?;
            if let Some(device_id) = stripe.chunk_devices.get(i) {
                self.record_device_read(*device_id as usize, n as u64);
            }
            total_read += n;
        }
        if total_read != total {
            // Not fatal (the kernel may 0-pad to 4K), but log it.
            tracing::warn!(
                "get_into_ptr: read {} bytes != expected {} (key={})",
                total_read,
                total,
                str_key
            );
        }
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(total as u64, Ordering::Relaxed);
        Ok(Some((total, meta)))
    }

    /// **RDMA GET data-plane pipeline version** — like `get_into_ptr`, reads directly into ptr, but
    /// streams per-stripe completion events over a channel (instead of waiting for all stripes to
    /// complete).
    ///
    /// Use: as soon as the server receives a stripe completion event, it can post the corresponding
    /// RDMA WRITE, so storage read and RDMA WRITE truly overlap (vs `get_into_ptr` which is serial).
    ///
    /// Returns:
    /// - Ok(Some((meta, Receiver))): contains BlockMeta + completion event channel
    ///   - Each recv yields (stripe_idx, stripe_offset, stripe_len, Result<bytes_read>)
    ///   - stripe_idx: 0..n_stripes
    ///   - stripe_offset: byte offset within the whole value (used to compute dst_addr)
    ///   - stripe_len: bytes for this stripe (used for post_write length)
    ///   - After all stripes complete, the channel closes and recv returns Err.
    /// - Ok(None): key does not exist
    /// - Err(e): metadata error, etc.
    ///
    /// # Safety
    /// - ptr..ptr+capacity must not be freed/mutated before every recv completes.
    /// - Non-striped (single file) returns a single event (stripe_idx=0, offset=0).
    ///
    /// Non-striped takes the old path (get_into_ptr single IO) since single files have no split
    /// meaning.
    pub fn get_into_ptr_stream(
        &self,
        key: &ObjectKey,
        ptr: *mut u8,
        capacity: usize,
    ) -> Result<
        Option<(
            BlockMeta,
            crossbeam_channel::Receiver<(usize, usize, usize, Result<usize>)>,
        )>,
    > {
        let str_key = key.to_string_key();
        let meta = match self.metadata.get_block(&str_key)? {
            Some(m) => m,
            None => return Ok(None),
        };
        self.get_into_ptr_stream_with_meta(key, &meta, ptr, capacity)
    }

    /// RDMA pipeline read path with metadata already resolved.
    ///
    /// The descriptor/RDMA path first rebuilds the physical layout from the descriptor and then
    /// calls this method to read, avoiding a re-lookup by logical key on the data plane that could
    /// hit a different version.
    pub fn get_into_ptr_stream_with_meta(
        &self,
        key: &ObjectKey,
        meta: &BlockMeta,
        ptr: *mut u8,
        capacity: usize,
    ) -> Result<
        Option<(
            BlockMeta,
            crossbeam_channel::Receiver<(usize, usize, usize, Result<usize>)>,
        )>,
    > {
        let str_key = key.to_string_key();
        let meta = meta.clone();
        // Non-striped: fall back to the old get_into_ptr (single IO, sync read),
        // then wrap it in a single-event channel to keep a uniform interface.
        let Some(stripe) = &meta.striping else {
            let path = self.meta_path_or_route(key, &meta);
            if !self.executor.file_exists(&path) {
                return Ok(None);
            }
            let total = meta.size as usize;
            let req = IORequest {
                path,
                offset: 0,
                length: 0,
            };
            info!(
                "IO_PATTERN key={} mode=single total={} request_count=1 offset=0 length=0 capacity={}",
                str_key,
                total,
                capacity,
            );
            let rx = self
                .executor
                .read_aligned_into_ptr_stream(vec![(req, ptr, capacity)]);
            // Wrap in a channel to add stripe_offset info.
            let (tx, rx_out) = crossbeam_channel::unbounded();
            let device_id = self.meta_device_or_route(key, &meta);
            let device_read_bytes = self.device_read_bytes_total.clone();
            std::thread::spawn(move || {
                if let Ok((_idx, result)) = rx.recv() {
                    if let Ok(n) = result.as_ref() {
                        if let Some(counter) = device_read_bytes.get(device_id) {
                            counter.fetch_add(*n as u64, Ordering::Relaxed);
                        }
                    }
                    let _ = tx.send((0usize, 0usize, total, result));
                }
            });
            self.reads_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(total as u64, Ordering::Relaxed);
            return Ok(Some((meta, rx_out)));
        };

        // striped: prepare N IORequests reading into N offsets of ptr.
        let chunk_size = stripe.chunk_size as usize;
        let total = stripe.total_size as usize;
        if capacity < total {
            return Err(KVError::Internal(format!(
                "get_into_ptr_stream: capacity {} < total {}",
                capacity, total
            )));
        }

        let n_stripes = stripe.chunk_paths.len();
        let mut reqs: Vec<(IORequest, *mut u8, usize)> = Vec::with_capacity(n_stripes);
        // Record each stripe's (offset_in_value, len) for event push.
        let mut stripe_meta: Vec<(usize, usize)> = Vec::with_capacity(n_stripes);
        let mut stripe_log: Vec<String> = Vec::with_capacity(n_stripes);
        let mut device_summary: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
        for (i, p) in stripe.chunk_paths.iter().enumerate() {
            let stripe_offset = i * chunk_size;
            let stripe_end = ((i + 1) * chunk_size).min(total);
            let stripe_len = stripe_end - stripe_offset;
            let device_id = stripe.chunk_devices.get(i).copied().unwrap_or(u32::MAX);
            let aligned_stripe = (stripe_len + 4095) & !4095;
            let stripe_ptr = unsafe { ptr.add(stripe_offset) };
            let stripe_cap = (capacity - stripe_offset).min((chunk_size + 4095) & !4095);
            if stripe_cap < aligned_stripe {
                return Err(KVError::Internal(format!(
                    "stripe {} cap {} < aligned {}",
                    i, stripe_cap, aligned_stripe
                )));
            }
            reqs.push((
                IORequest {
                    path: std::path::PathBuf::from(p),
                    offset: 0,
                    length: 0,
                },
                stripe_ptr,
                stripe_cap,
            ));
            stripe_meta.push((stripe_offset, stripe_len));
            let entry = device_summary.entry(device_id).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += stripe_len;
            stripe_log.push(format!(
                "{}:dev{}:off{}:len{}:{}",
                i, device_id, stripe_offset, stripe_len, p,
            ));
        }
        let device_summary_log = device_summary
            .iter()
            .map(|(device_id, (count, bytes))| {
                format!("dev{}:{}stripes:{}B", device_id, count, bytes)
            })
            .collect::<Vec<_>>()
            .join(",");
        let min_stripe_len = stripe_meta.iter().map(|(_, len)| *len).min().unwrap_or(0);
        let max_stripe_len = stripe_meta.iter().map(|(_, len)| *len).max().unwrap_or(0);
        info!(
            concat!(
                "IO_PATTERN key={} mode=striped total={} chunk_size={} n_stripes={} ",
                "device_count={} min_stripe_len={} max_stripe_len={} device_summary=[{}]"
            ),
            str_key,
            total,
            chunk_size,
            n_stripes,
            device_summary.len(),
            min_stripe_len,
            max_stripe_len,
            device_summary_log,
        );
        tracing::info!(
            "STORAGE_GET_STREAM key={} total={} chunk_size={} n_stripes={} stripe_map=[{}]",
            str_key,
            total,
            chunk_size,
            n_stripes,
            stripe_log.join(","),
        );

        // Call executor's stream API to get an (idx, Result) event stream.
        let raw_rx = self.executor.read_aligned_into_ptr_stream(reqs);

        // Add a translation layer converting (idx, Result<usize>) into (idx, offset, len,
        // Result<usize>). Use a spawn to forward without blocking.
        let (tx_out, rx_out) = crossbeam_channel::unbounded();
        let device_read_bytes = self.device_read_bytes_total.clone();
        let chunk_devices = stripe.chunk_devices.clone();
        std::thread::spawn(move || {
            while let Ok((idx, result)) = raw_rx.recv() {
                if idx < stripe_meta.len() {
                    if let Ok(n) = result.as_ref() {
                        if let Some(device_id) = chunk_devices.get(idx) {
                            if let Some(counter) = device_read_bytes.get(*device_id as usize) {
                                counter.fetch_add(*n as u64, Ordering::Relaxed);
                            }
                        }
                    }
                    let (off, len) = stripe_meta[idx];
                    let _ = tx_out.send((idx, off, len, result));
                }
            }
        });

        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(total as u64, Ordering::Relaxed);
        Ok(Some((meta, rx_out)))
    }

    /// Multi-segment GET — zero-copy passthrough: striped keys return Vec<Bytes> directly;
    /// non-striped single files return a single-element Vec. The caller (get_stream) streams the
    /// Bytes segments to the client without concatenating.
    pub fn get_chunks(&self, key: &ObjectKey) -> Result<Option<(Vec<Bytes>, BlockMeta)>> {
        let str_key = key.to_string_key();
        let meta = match self.metadata.get_block(&str_key)? {
            Some(m) => m,
            None => return Ok(None),
        };

        if let Some(stripe) = &meta.striping {
            let segments = self.read_striped_chunks(key, stripe)?;
            let total: u64 = segments.iter().map(|s| s.len() as u64).sum();
            self.reads_total.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(total, Ordering::Relaxed);
            return Ok(Some((segments, meta)));
        }

        let path = self.meta_path_or_route(key, &meta);
        if !self.executor.file_exists(&path) {
            self.metadata.delete_block(&str_key)?;
            return Ok(None);
        }
        let data = self.executor.read_file(&path)?;
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.record_device_read(self.meta_device_or_route(key, &meta), data.len() as u64);
        Ok(Some((vec![Bytes::from(data)], meta)))
    }

    // ===== Batch (parallel) =====

    /// Batch read — grouped by device and executed in parallel.
    pub fn get_batch(&self, keys: &[ObjectKey]) -> Vec<Result<Option<(Bytes, BlockMeta)>>> {
        // 1. Collect each key's metadata and path and build IORequest.
        //    striped keys take a separate path (inline read_striped, not going through batch submit).
        let mut results: Vec<Option<Result<Option<(Bytes, BlockMeta)>>>> =
            (0..keys.len()).map(|_| None).collect();
        let mut io_reqs: Vec<(usize, IORequest, BlockMeta, usize)> = Vec::new();

        for (idx, key) in keys.iter().enumerate() {
            let str_key = key.to_string_key();
            match self.metadata.get_block(&str_key) {
                Ok(Some(meta)) => {
                    if meta.striping.is_some() {
                        // striped: handled separately (internally already a parallel read_batch).
                        let stripe = meta.striping.as_ref().unwrap().clone();
                        match self.read_striped(key, &stripe) {
                            Ok(data) => {
                                self.reads_total.fetch_add(1, Ordering::Relaxed);
                                self.bytes_read
                                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                                results[idx] = Some(Ok(Some((Bytes::from(data), meta))));
                            }
                            Err(e) => results[idx] = Some(Err(e)),
                        }
                        continue;
                    }
                    let path = self.meta_path_or_route(key, &meta);
                    let device_id = self.meta_device_or_route(key, &meta);
                    io_reqs.push((
                        idx,
                        IORequest {
                            path,
                            offset: 0,
                            length: 0,
                        },
                        meta,
                        device_id,
                    ));
                }
                Ok(None) => {
                    results[idx] = Some(Ok(None));
                }
                Err(e) => {
                    results[idx] = Some(Err(e));
                }
            }
        }

        // 2. BatchOptimizer: group by (device → sorted paths) to reduce random I/O.
        //    Router already implicitly routes key→device; here we sort by path to exploit sequential
        //    reads.
        io_reqs.sort_by(|a, b| a.1.path.cmp(&b.1.path));

        // 3. Submit I/O in batch (executor is internally parallel).
        let reqs: Vec<IORequest> = io_reqs
            .iter()
            .map(|(_, r, _, _)| IORequest {
                path: r.path.clone(),
                offset: r.offset,
                length: r.length,
            })
            .collect();
        let read_results = self.executor.read_batch(&reqs);

        // 4. Fill results back in.
        for ((idx, _, meta, device_id), data_res) in
            io_reqs.into_iter().zip(read_results.into_iter())
        {
            match data_res {
                Ok(data) => {
                    self.reads_total.fetch_add(1, Ordering::Relaxed);
                    self.bytes_read
                        .fetch_add(data.len() as u64, Ordering::Relaxed);
                    self.record_device_read(device_id, data.len() as u64);
                    // Vec<u8> → Bytes zero-copy handoff.
                    results[idx] = Some(Ok(Some((Bytes::from(data), meta))));
                }
                Err(e) => results[idx] = Some(Err(e)),
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(KVError::Internal("missing slot".to_string()))))
            .collect()
    }

    /// Batch write.
    pub fn put_batch(&self, items: Vec<(ObjectKey, Bytes, BlockMeta)>) -> Vec<Result<()>> {
        items
            .into_iter()
            .map(|(key, data, meta)| self.put(&key, data, meta))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(dir: &std::path::Path) -> Config {
        test_config_with_devices(dir, 2)
    }

    fn test_config_with_devices(dir: &std::path::Path, n_devices: usize) -> Config {
        let mut cfg = Config::default();
        cfg.storage.devices = (0..n_devices)
            .map(|i| dir.join(format!("nvme{}", i)))
            .collect();
        cfg.metadata.redis_url = format!("memory://storage-tier-{}", dir.display());
        cfg
    }

    fn mk_meta() -> BlockMeta {
        BlockMeta {
            device_id: 0,
            file_path: String::new(),
            size: 0,
            object_handle: String::new(),
            object_generation: 1,
            content_etag: String::new(),
            layout_version: 1,
            created_at: 0,
            last_accessed_at: 0,
            ttl_seconds: 0,
            num_tokens: 128,
            num_layers: 1,
            dtype: "bfloat16".to_string(),
            compressed: false,
            striping: None,
        }
    }

    fn flatten_segments(segments: &[Bytes]) -> Vec<u8> {
        segments
            .iter()
            .flat_map(|segment| segment.iter().copied())
            .collect()
    }

    #[test]
    fn put_get_delete_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cfg = test_config(tmp.path());
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "abcdef/layer_0".into(),
        };
        st.put(&key, Bytes::from_static(b"hello"), mk_meta())
            .unwrap();
        let (data, meta) = st.get(&key).unwrap().unwrap();
        assert_eq!(data.as_ref(), b"hello");
        assert_eq!(meta.object_generation, 1);
        assert!(!meta.content_etag.is_empty());
        assert_eq!(meta.layout_version, 1);
        let first_etag = meta.content_etag.clone();

        st.put(&key, Bytes::from_static(b"hello-again"), mk_meta())
            .unwrap();
        let (data, meta) = st.get(&key).unwrap().unwrap();
        assert_eq!(data.as_ref(), b"hello-again");
        assert_eq!(meta.object_generation, 2);
        assert_ne!(meta.content_etag, first_etag);
        assert_eq!(meta.layout_version, 1);

        assert!(st.exists(&key).unwrap());
        assert!(st.delete(&key).unwrap());
        assert!(!st.exists(&key).unwrap());
    }

    #[test]
    fn put_if_absent_does_not_overwrite_existing_object() {
        let tmp = TempDir::new().unwrap();
        let cfg = test_config(tmp.path());
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "immutable/object".into(),
        };

        assert!(st
            .put_if_absent(&key, Bytes::from_static(b"first"), mk_meta())
            .unwrap());
        let first_meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();

        assert!(!st
            .put_if_absent(&key, Bytes::from_static(b"second"), mk_meta())
            .unwrap());
        let (data, meta) = st.get(&key).unwrap().unwrap();

        assert_eq!(data.as_ref(), b"first");
        assert_eq!(meta.object_generation, first_meta.object_generation);
        assert_eq!(meta.content_etag, first_meta.content_etag);
        assert_eq!(meta.file_path, first_meta.file_path);
    }

    #[test]
    fn put_chunks_if_absent_does_not_overwrite_striped_object() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.storage.striping_threshold = 8;
        cfg.storage.striping_chunk_size = 4;
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "immutable/striped".into(),
        };

        assert!(st
            .put_chunks_if_absent(
                &key,
                vec![Bytes::from_static(b"abcd"), Bytes::from_static(b"efghij")],
                mk_meta(),
            )
            .unwrap());
        let first_meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();

        assert!(!st
            .put_chunks_if_absent(
                &key,
                vec![Bytes::from_static(b"ZZZZ"), Bytes::from_static(b"YYYYYY")],
                mk_meta(),
            )
            .unwrap());
        let (segments, meta) = st.get_chunks(&key).unwrap().unwrap();

        assert_eq!(flatten_segments(&segments), b"abcdefghij");
        assert_eq!(meta.object_generation, first_meta.object_generation);
        assert_eq!(meta.content_etag, first_meta.content_etag);
        assert_eq!(
            meta.striping.unwrap().chunk_paths,
            first_meta.striping.unwrap().chunk_paths
        );
    }

    #[test]
    fn non_striped_get_uses_metadata_path_after_device_count_changes() {
        let tmp = TempDir::new().unwrap();
        let old_cfg = test_config_with_devices(tmp.path(), 2);
        let new_cfg = test_config_with_devices(tmp.path(), 3);
        let old_router = ShardRouter::new(&old_cfg).unwrap();
        let new_router = ShardRouter::new(&new_cfg).unwrap();
        let key = (0..1024)
            .map(|i| ObjectKey {
                namespace: "test".into(),
                object_key: format!("remap/object_{}", i),
            })
            .find(|k| old_router.route(k) != new_router.route(k))
            .expect("should find key remapped by device-count change");

        let old_path = {
            let router = Arc::new(ShardRouter::new(&old_cfg).unwrap());
            let st = StorageTier::new(&old_cfg, router).unwrap();
            st.put(&key, Bytes::from_static(b"stable-path"), mk_meta())
                .unwrap();
            st.metadata
                .get_block(&key.to_string_key())
                .unwrap()
                .unwrap()
                .file_path
        };

        let router = Arc::new(ShardRouter::new(&new_cfg).unwrap());
        let st = StorageTier::new(&new_cfg, router).unwrap();
        assert_ne!(
            std::path::PathBuf::from(&old_path),
            st.router.key_to_path(&key),
            "test key must route to a different path after expansion"
        );

        let (data, meta) = st.get(&key).unwrap().unwrap();
        assert_eq!(data.as_ref(), b"stable-path");
        assert_eq!(meta.file_path, old_path);
    }

    #[test]
    fn overwrite_writes_new_physical_path_without_mutating_old_file() {
        let tmp = TempDir::new().unwrap();
        let cfg = test_config(tmp.path());
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "overwrite/object".into(),
        };

        st.put(&key, Bytes::from_static(b"old-value"), mk_meta())
            .unwrap();
        let old_meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();
        let old_path = std::path::PathBuf::from(&old_meta.file_path);
        assert!(old_path.exists());

        st.put(&key, Bytes::from_static(b"new-value"), mk_meta())
            .unwrap();
        let new_meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();

        assert_eq!(new_meta.object_generation, old_meta.object_generation + 1);
        assert_ne!(new_meta.object_handle, old_meta.object_handle);
        assert_ne!(new_meta.file_path, old_meta.file_path);
        assert_eq!(std::fs::read(&old_path).unwrap(), b"old-value");

        let (old_data, old_read_meta) = st.get_with_meta(&key, &old_meta).unwrap().unwrap();
        assert_eq!(old_data.as_ref(), b"old-value");
        assert_eq!(old_read_meta.object_generation, old_meta.object_generation);

        let (new_data, _) = st.get(&key).unwrap().unwrap();
        assert_eq!(new_data.as_ref(), b"new-value");
    }

    #[test]
    fn striped_overwrite_uses_new_chunk_paths() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.storage.striping_threshold = 16;
        cfg.storage.striping_chunk_size = 8;
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "overwrite/striped".into(),
        };

        st.put(&key, Bytes::from_static(b"abcdefghijklmnopq"), mk_meta())
            .unwrap();
        let old_meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();
        let old_paths = old_meta.striping.as_ref().unwrap().chunk_paths.clone();

        st.put(&key, Bytes::from_static(b"ABCDEFGHIJKLMNOPQ"), mk_meta())
            .unwrap();
        let new_meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();
        let new_paths = new_meta.striping.as_ref().unwrap().chunk_paths.clone();

        assert_eq!(new_meta.object_generation, old_meta.object_generation + 1);
        assert_ne!(new_paths, old_paths);
        assert!(old_paths.iter().all(|p| std::path::Path::new(p).exists()));

        let (old_segments, _) = st.get_chunks_with_meta(&key, &old_meta).unwrap().unwrap();
        let old_joined = old_segments
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(old_joined.as_slice(), b"abcdefghijklmnopq");
    }

    #[test]
    fn batch_distributes_to_multiple_devices() {
        let tmp = TempDir::new().unwrap();
        let cfg = test_config(tmp.path());
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let items: Vec<_> = (0..8)
            .map(|i| {
                (
                    ObjectKey {
                        namespace: "test".into(),
                        object_key: format!("abcdef/layer_{}", i),
                    },
                    Bytes::from(format!("data-{}", i).into_bytes()),
                    mk_meta(),
                )
            })
            .collect();
        let keys: Vec<_> = items.iter().map(|(k, _, _)| k.clone()).collect();
        let results = st.put_batch(items);
        for r in &results {
            r.as_ref().unwrap();
        }
        let got = st.get_batch(&keys);
        for (i, r) in got.into_iter().enumerate() {
            let (d, _) = r.unwrap().unwrap();
            assert_eq!(d.as_ref(), format!("data-{}", i).as_bytes());
        }
    }

    #[test]
    fn large_value_is_striped() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        // Small threshold for easier testing.
        cfg.storage.striping_threshold = 1024; // 1KB
        cfg.storage.striping_chunk_size = 512; // 512B per chunk
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let st = StorageTier::new(&cfg, router).unwrap();
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "ab/huge".into(),
        };
        // 2 KB data → should be split into 4 chunks.
        let data: Vec<u8> = (0..2048u32).map(|i| (i & 0xff) as u8).collect();
        st.put(&key, Bytes::from(data.clone()), mk_meta()).unwrap();

        // Metadata should contain striping info.
        let meta = st
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .unwrap();
        let stripe = meta.striping.as_ref().expect("should be striped");
        assert_eq!(stripe.chunk_paths.len(), 4);
        assert_eq!(stripe.total_size, 2048);

        // Read back the full data and verify.
        let (got, _) = st.get(&key).unwrap().unwrap();
        assert_eq!(got.as_ref(), data.as_slice());

        // delete should clean up all chunks.
        st.delete(&key).unwrap();
        for p in &stripe.chunk_paths {
            assert!(
                !std::path::Path::new(p).exists(),
                "chunk {} still exists",
                p
            );
        }
    }
}
