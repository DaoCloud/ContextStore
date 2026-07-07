//! Memory Tier (L1) — LRU memory cache + write-through
//!
//! Phase 1 uses plain Vec<u8> as buffer.
//! Phase 2 can switch to a pinned memory pool (use_pinned_memory=true).

use crate::config::Config;
use crate::error::Result;
use crate::metadata::BlockMeta;
use crate::router::ObjectKey;
use crate::storage_tier::StorageTier;
use lru::LruCache;
use parking_lot::Mutex;
use prost::bytes::Bytes;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(feature = "rdma")]
use crate::rdma::slab::{RdmaSlab, SlabExtent, SlabExtentBytes, SlabPlacement};
#[cfg(feature = "rdma")]
use once_cell::sync::OnceCell;

struct CacheEntry {
    data: Bytes,
    meta: BlockMeta,
}

/// Multi-segment cache entry (for put_chunks/get_chunks). Each Bytes segment is an independent zero-copy slice.
struct ChunksCacheEntry {
    segments: Vec<Bytes>,
    meta: BlockMeta,
    total_size: usize,
    /// When slab-backed, holds the extent (also serves as the "is slab-backed" flag). RDMA GET takes the
    /// zero-registration fast path; None means data lives on the heap (segments are AlignedBuffer-backed
    /// Bytes) and falls back to the per-chunk path.
    /// When the entry is LRU-evicted, this field drops with it; the last Arc drop triggers slab reclamation
    /// (pure RAII).
    #[cfg(feature = "rdma")]
    slab: Option<Arc<SlabExtent>>,
}

pub struct MemoryTier {
    capacity_bytes: usize,
    cache: Mutex<LruCache<String, CacheEntry>>,
    /// Multi-segment chunks cache: used by the put_chunks/get_chunks paths. Shares the capacity_bytes
    /// budget with `cache`.
    chunks_cache: Mutex<LruCache<String, ChunksCacheEntry>>,
    current_size: AtomicU64,
    storage: Arc<StorageTier>,
    // Stats
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    /// Pre-registered RDMA slab, late-injected by `rdma::server::run_server` via `set_rdma_slab` after
    /// startup. When not injected (slab creation failed / rdma feature off), all paths fall back to
    /// heap Bytes, and behavior is unchanged from today.
    #[cfg(feature = "rdma")]
    rdma_slab: OnceCell<Arc<RdmaSlab>>,
}

impl MemoryTier {
    pub fn new(config: &Config, storage: Arc<StorageTier>) -> Self {
        let capacity_bytes = config.memory_tier.capacity_mb * 1024 * 1024;
        // Use NonZeroUsize::MAX as the entry-count limit; the actual capacity is controlled in bytes.
        Self {
            capacity_bytes,
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(1_000_000).unwrap())),
            chunks_cache: Mutex::new(LruCache::new(NonZeroUsize::new(1_000_000).unwrap())),
            current_size: AtomicU64::new(0),
            storage,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            #[cfg(feature = "rdma")]
            rdma_slab: OnceCell::new(),
        }
    }

    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.current_size.load(Ordering::Relaxed),
        )
    }

    fn meta_identity_matches(actual: &BlockMeta, expected: &BlockMeta) -> bool {
        actual.object_handle == expected.object_handle
            && actual.object_generation == expected.object_generation
            && actual.layout_version == expected.layout_version
            && actual.content_etag == expected.content_etag
            && actual.size == expected.size
    }

    /// Read an object with pre-validated metadata. Only returns the cache when the L1 version identity
    /// fully matches.
    pub fn get_with_meta(
        &self,
        key: &ObjectKey,
        expected: &BlockMeta,
    ) -> Result<Option<(Bytes, BlockMeta)>> {
        let str_key = key.to_string_key();
        {
            let mut cache = self.cache.lock();
            if let Some(entry) = cache.get(&str_key) {
                if Self::meta_identity_matches(&entry.meta, expected) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some((entry.data.clone(), entry.meta.clone())));
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);

        match self.storage.get_with_meta(key, expected)? {
            Some((data, meta)) => {
                self.cache_insert(str_key, data.clone(), meta.clone());
                Ok(Some((data, meta)))
            }
            None => Ok(None),
        }
    }

    /// Read a multi-segment object with pre-validated metadata. Only returns the cache when the L1
    /// chunks-cache version matches.
    pub fn get_chunks_with_meta(
        &self,
        key: &ObjectKey,
        expected: &BlockMeta,
    ) -> Result<Option<(Vec<Bytes>, BlockMeta)>> {
        let str_key = key.to_string_key();
        {
            let mut cache = self.chunks_cache.lock();
            if let Some(entry) = cache.get(&str_key) {
                if Self::meta_identity_matches(&entry.meta, expected) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    let segments = entry.segments.iter().cloned().collect();
                    return Ok(Some((segments, entry.meta.clone())));
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);

        match self.storage.get_chunks_with_meta(key, expected)? {
            Some((segments, meta)) => {
                let total_size: usize = segments.iter().map(|b| b.len()).sum();
                self.chunks_cache_insert(str_key, segments.clone(), meta.clone(), total_size);
                Ok(Some((segments, meta)))
            }
            None => Ok(None),
        }
    }

    /// Multi-segment GET — zero-copy passthrough: check L1 chunks_cache first; on hit return
    /// (Bytes::clone = Arc refcount), on miss fall through to L2 storage and fill the result back into
    /// L1 (same write-through cache semantics as single get).
    ///
    /// Use: after get_stream fetches it, each segment is streamed to the client as one DataChunk with
    /// no concat anywhere.
    pub fn get_chunks(&self, key: &ObjectKey) -> Result<Option<(Vec<Bytes>, BlockMeta)>> {
        let str_key = key.to_string_key();
        {
            let mut cache = self.chunks_cache.lock();
            if let Some(entry) = cache.get(&str_key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                // Each Bytes::clone is an Arc::clone; underlying data is not copied.
                let segments = entry.segments.iter().cloned().collect();
                return Ok(Some((segments, entry.meta.clone())));
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);

        match self.storage.get_chunks(key)? {
            Some((segments, meta)) => {
                // Fill L1 cache (write-back on read)
                let total_size: usize = segments.iter().map(|b| b.len()).sum();
                self.chunks_cache_insert(str_key, segments.clone(), meta.clone(), total_size);
                Ok(Some((segments, meta)))
            }
            None => Ok(None),
        }
    }

    /// GET: check L1 first, fall through to L2 on miss.
    pub fn get(&self, key: &ObjectKey) -> Result<Option<(Bytes, BlockMeta)>> {
        let str_key = key.to_string_key();
        {
            let mut cache = self.cache.lock();
            if let Some(entry) = cache.get(&str_key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                // Bytes::clone is just an Arc refcount bump; underlying data is not copied.
                return Ok(Some((entry.data.clone(), entry.meta.clone())));
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);

        match self.storage.get(key)? {
            Some((data, meta)) => {
                self.cache_insert(str_key, data.clone(), meta.clone());
                Ok(Some((data, meta)))
            }
            None => Ok(None),
        }
    }

    /// PUT: write-through (persist to L2, then write L1).
    pub fn put(&self, key: &ObjectKey, data: Bytes, meta: BlockMeta) -> Result<()> {
        let str_key = key.to_string_key();
        let fallback_meta = meta.clone();
        // Persist to L2 (internal put_striped uses data.slice() for zero-copy slicing).
        self.storage.put(key, data.clone(), meta)?;
        let stored_meta = self
            .storage
            .metadata()
            .get_block(&str_key)?
            .unwrap_or(fallback_meta);
        // After a successful write, clear every L1 representation of the same object to avoid
        // single/chunks/slab caches returning stale values across interfaces on overwrite.
        self.invalidate_l1_entry(&str_key);
        // L1 cache; data.clone is an Arc::clone, no copy.
        self.cache_insert(str_key, data, stored_meta);
        Ok(())
    }

    /// Multi-segment PUT — zero-copy passthrough: caller supplies N Bytes segments (typical: 240 2MB
    /// chunks accumulated by put_stream, each one a refcount view from the gRPC framework).
    /// **Does not concat into a single large Bytes** before L2. Since LruCache requires a single Bytes
    /// per entry, L1 **skips caching** here (a large value would be evicted from L1 quickly anyway,
    /// so the cost of not caching is small).
    pub fn put_chunks(&self, key: &ObjectKey, segments: Vec<Bytes>, meta: BlockMeta) -> Result<()> {
        let str_key = key.to_string_key();
        let fallback_meta = meta.clone();
        self.storage.put_chunks(key, segments.clone(), meta)?;
        let stored_meta = self
            .storage
            .metadata()
            .get_block(&str_key)?
            .unwrap_or(fallback_meta);
        self.invalidate_l1_entry(&str_key);
        // Multi-segment also goes into L1 (Bytes::clone = Arc refcount, zero-copy).
        let total_size: usize = segments.iter().map(|b| b.len()).sum();
        self.chunks_cache_insert(str_key, segments, stored_meta, total_size);
        Ok(())
    }

    pub fn delete(&self, key: &ObjectKey) -> Result<bool> {
        let str_key = key.to_string_key();
        self.invalidate_l1_entry(&str_key);
        self.storage.delete(key)
    }

    pub fn invalidate(&self, key: &ObjectKey) {
        self.invalidate_l1_entry(&key.to_string_key());
    }

    pub fn exists(&self, key: &ObjectKey) -> Result<bool> {
        let str_key = key.to_string_key();
        if self.cache.lock().contains(&str_key) {
            return Ok(true);
        }
        if self.chunks_cache.lock().contains(&str_key) {
            return Ok(true);
        }
        self.storage.exists(key)
    }

    // ===== GDS direct read/write =====
    // L1 cache only holds host-side Vec<u8>; the GDS path passes straight through to L2, not polluting
    // the cache. (A GPU-side second-level cache could be added later, but for now vLLM has its own KV
    // block pool.)
    #[cfg(feature = "gds")]
    pub fn get_to_gpu(
        &self,
        key: &ObjectKey,
        gpu_buf: &mut crate::gds::GpuBuffer,
    ) -> Result<Option<(usize, BlockMeta)>> {
        self.storage.get_to_gpu(key, gpu_buf)
    }

    #[cfg(feature = "gds")]
    pub fn put_from_gpu(
        &self,
        key: &ObjectKey,
        gpu_buf: &crate::gds::GpuBuffer,
        size: usize,
        meta: BlockMeta,
    ) -> Result<()> {
        self.storage.put_from_gpu(key, gpu_buf, size, meta)?;
        self.invalidate_l1_entry(&key.to_string_key());
        Ok(())
    }

    // ===== Batch =====

    pub fn get_batch(&self, keys: &[ObjectKey]) -> Vec<Result<Option<(Bytes, BlockMeta)>>> {
        let mut results: Vec<Option<Result<Option<(Bytes, BlockMeta)>>>> =
            (0..keys.len()).map(|_| None).collect();
        let mut to_fetch: Vec<(usize, ObjectKey)> = Vec::new();

        {
            let mut cache = self.cache.lock();
            for (i, k) in keys.iter().enumerate() {
                let sk = k.to_string_key();
                if let Some(e) = cache.get(&sk) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    results[i] = Some(Ok(Some((e.data.clone(), e.meta.clone()))));
                } else {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    to_fetch.push((i, k.clone()));
                }
            }
        }

        if !to_fetch.is_empty() {
            let l2_keys: Vec<ObjectKey> = to_fetch.iter().map(|(_, k)| k.clone()).collect();
            let l2_res = self.storage.get_batch(&l2_keys);
            for ((idx, key), res) in to_fetch.into_iter().zip(l2_res.into_iter()) {
                match res {
                    Ok(Some((data, meta))) => {
                        self.cache_insert(key.to_string_key(), data.clone(), meta.clone());
                        results[idx] = Some(Ok(Some((data, meta))));
                    }
                    Ok(None) => results[idx] = Some(Ok(None)),
                    Err(e) => results[idx] = Some(Err(e)),
                }
            }
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Ok(None)))
            .collect()
    }

    pub fn put_batch(&self, items: Vec<(ObjectKey, Bytes, BlockMeta)>) -> Vec<Result<()>> {
        // Persist to L2 first; Bytes::clone is an Arc refcount bump, no data copy.
        let items_for_l2: Vec<_> = items
            .iter()
            .map(|(k, d, m)| (k.clone(), d.clone(), m.clone()))
            .collect();
        let l2_results = self.storage.put_batch(items_for_l2);
        // Successful writes go into L1.
        for ((k, d, m), r) in items.into_iter().zip(l2_results.iter()) {
            if r.is_ok() {
                let str_key = k.to_string_key();
                let stored_meta = self
                    .storage
                    .metadata()
                    .get_block(&str_key)
                    .ok()
                    .flatten()
                    .unwrap_or(m);
                self.invalidate_l1_entry(&str_key);
                self.cache_insert(str_key, d, stored_meta);
            }
        }
        l2_results
    }

    fn invalidate_l1_entry(&self, key: &str) {
        {
            let mut cache = self.cache.lock();
            if let Some(entry) = cache.pop(key) {
                self.current_size
                    .fetch_sub(entry.data.len() as u64, Ordering::Relaxed);
            }
        }
        {
            let mut cache = self.chunks_cache.lock();
            if let Some(entry) = cache.pop(key) {
                self.current_size
                    .fetch_sub(entry.total_size as u64, Ordering::Relaxed);
            }
        }
    }

    fn cache_insert(&self, key: String, data: Bytes, meta: BlockMeta) {
        let size = data.len() as u64;
        if size > self.capacity_bytes as u64 {
            // Single value exceeds L1 capacity; skip caching. The write path explicitly invalidates
            // before the call, and the read path leaves the existing cache unchanged.
            return;
        }
        self.invalidate_l1_entry(&key);
        let mut cache = self.cache.lock();
        // Evict until there is enough space.
        while self.current_size.load(Ordering::Relaxed) + size > self.capacity_bytes as u64
            && !cache.is_empty()
        {
            if let Some((_, evicted)) = cache.pop_lru() {
                self.current_size
                    .fetch_sub(evicted.data.len() as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        if let Some(old) = cache.put(key, CacheEntry { data, meta }) {
            self.current_size
                .fetch_sub(old.data.len() as u64, Ordering::Relaxed);
        }
        self.current_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Multi-segment cache insert. Shares the capacity_bytes budget with cache_insert (single
    /// current_size counter), but eviction runs LRU only within chunks_cache (cross-cache eviction is
    /// complex with small benefit).
    fn chunks_cache_insert(
        &self,
        key: String,
        segments: Vec<Bytes>,
        meta: BlockMeta,
        total_size: usize,
    ) {
        // L1 fully disabled (capacity=0) or single entry over capacity: return directly, no work.
        // Must check before maybe_slab_backed, otherwise at capacity=0 we would still slab.alloc +
        // copy_in 480MB into the slab, then return-drop → a net 80ms+ of wasted memcpy.
        if total_size as u64 > self.capacity_bytes as u64 {
            return;
        }
        self.invalidate_l1_entry(&key);

        // Try moving data into the pre-registered slab: on success, segments become zero-copy views
        // into the slab, and the original heap Bytes (AlignedBuffer) drop → data lives only in the
        // slab (1× memory). alloc fails / no slab → keep the heap segments (slab: None) and fall
        // back to per-chunk.
        #[cfg(feature = "rdma")]
        let (segments, slab) = self.maybe_slab_backed(segments, total_size);

        let size = total_size as u64;
        let mut cache = self.chunks_cache.lock();
        // Evict chunks_cache until there is space (current_size covers the sum of both caches).
        while self.current_size.load(Ordering::Relaxed) + size > self.capacity_bytes as u64
            && !cache.is_empty()
        {
            if let Some((_, evicted)) = cache.pop_lru() {
                self.current_size
                    .fetch_sub(evicted.total_size as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        let entry = ChunksCacheEntry {
            segments,
            meta,
            total_size,
            #[cfg(feature = "rdma")]
            slab,
        };
        if let Some(old) = cache.put(key, entry) {
            self.current_size
                .fetch_sub(old.total_size as u64, Ordering::Relaxed);
        }
        self.current_size.fetch_add(size, Ordering::Relaxed);
    }

    /// If the slab is available and `alloc` succeeds, memcpy segments into a `SlabExtent` and return
    /// zero-copy view segments pointing into the slab plus `Arc<SlabExtent>`; otherwise return the
    /// segments unchanged + None.
    #[cfg(feature = "rdma")]
    fn maybe_slab_backed(
        &self,
        segments: Vec<Bytes>,
        total_size: usize,
    ) -> (Vec<Bytes>, Option<Arc<SlabExtent>>) {
        let Some(slab) = self.rdma_slab.get() else {
            return (segments, None);
        };
        let Some(mut ext) = slab.alloc(total_size) else {
            // slab full / fragmented: gracefully fall back to the heap path.
            return (segments, None);
        };
        // Fill exclusively before wrapping in Arc (statically guaranteed no concurrent readers).
        ext.copy_in(&segments);
        let arc = Arc::new(ext);
        // One owner holds the whole slab region, then slice out views matching each original segment's
        // length. All views share the same Arc refcount → any surviving gRPC clone keeps the whole
        // extent alive.
        let whole = Bytes::from_owner(SlabExtentBytes(arc.clone()));
        let mut views = Vec::with_capacity(segments.len());
        let mut off = 0usize;
        for seg in &segments {
            let len = seg.len();
            views.push(whole.slice(off..off + len));
            off += len;
        }
        // The original heap segments drop here → AlignedBuffer freed, data lives only in the slab.
        (views, Some(arc))
    }

    /// RDMA GET fast path: if the key hits and is slab-backed, return a `SlabPlacement` containing a
    /// `SlabView` (source address + lkey + len) and an `Arc<SlabExtent>` pin. Heap-backed hit or miss
    /// returns None.
    ///
    /// `nic_idx`: the NIC index of the listener the caller (server.rs handle_client) is on.
    /// Pass 0 for single-NIC deployments; for multi-NIC, the server binds each listener to a nic_idx
    /// at startup, and the same slab region is registered in every NIC's PD, so we return the view
    /// for the matching lkey.
    ///
    /// `_pin` must be held by the caller until the RDMA WRITE's `poll_n` completes, to prevent the
    /// entry from being evicted mid-flight, the extent reclaimed, and `copy_in` overwriting memory
    /// the NIC is still reading (use-after-free).
    #[cfg(feature = "rdma")]
    pub fn get_chunks_slab(&self, key: &ObjectKey, nic_idx: usize) -> Option<SlabPlacement> {
        let str_key = key.to_string_key();
        let mut cache = self.chunks_cache.lock();
        let entry = cache.get(&str_key)?; // LRU touch
        let arc = entry.slab.clone()?; // None → heap-backed; caller falls back to per-chunk.
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(SlabPlacement {
            view: arc.view(nic_idx),
            meta: entry.meta.clone(),
            _pin: arc,
        })
    }

    /// Late-inject the pre-registered slab at startup. Called once by `rdma::server::run_server`
    /// before the accept loop. If already set, returns Err (containing the passed-in slab); the
    /// caller can ignore it.
    #[cfg(feature = "rdma")]
    pub fn set_rdma_slab(&self, slab: Arc<RdmaSlab>) -> std::result::Result<(), Arc<RdmaSlab>> {
        self.rdma_slab.set(slab)
    }

    /// Get the injected slab (for the PUT data-plane to alloc an extent). Returns None before
    /// startup / on injection failure.
    #[cfg(feature = "rdma")]
    pub fn rdma_slab_get(&self) -> Option<Arc<RdmaSlab>> {
        self.rdma_slab.get().cloned()
    }

    /// Actively evict chunks_cache to free slab space. Called by `handle_put` when `slab.alloc` fails.
    ///
    /// Keeps pop_lru-ing until: (a) cache is empty, or (b) enough `need` bytes (plus slack) have been
    /// freed. Release path: cache pop_lru → ChunksCacheEntry drop → SlabExtent Arc refcount decrement
    /// → if this is the last Arc, SlabExtent::drop triggers free → returned to the slab free-list.
    ///
    /// `slab_capacity`: total slab capacity, a defensive parameter (avoids infinite loops; in
    /// practice we pop at most until the cache is empty).
    #[cfg(feature = "rdma")]
    pub fn evict_chunks_cache_to_free(&self, need: usize, _slab_capacity: usize) {
        let need_u64 = need as u64;
        let mut cache = self.chunks_cache.lock();
        let mut freed: u64 = 0;
        let mut count = 0;
        while freed < need_u64 && !cache.is_empty() {
            if let Some((_, evicted)) = cache.pop_lru() {
                let s = evicted.total_size as u64;
                self.current_size.fetch_sub(s, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
                freed += s;
                count += 1;
            } else {
                break;
            }
        }
        if count > 0 {
            tracing::debug!(
                "chunks_cache LRU evicted {} entries ({} MB) to free for new PUT ({} MB needed)",
                count,
                freed / (1024 * 1024),
                need_u64 / (1024 * 1024),
            );
        }
    }

    /// **RDMA PUT data-plane only**: inject a `SlabExtent` that already holds data directly into
    /// chunks_cache, so subsequent GETs hit the L1 slab fast path (zero reg_mr, ~11 GB/s).
    ///
    /// Differs from `chunks_cache_insert`:
    /// - Input is an extent already holding data (caller=handle_put holds it); no re-alloc / copy_in.
    /// - **Forced insertion** (not subject to capacity_bytes=0), because PUT already paid for the slab
    ///   memory; not caching would waste it. As long as LRU pop_lru releases the extent → slab
    ///   auto-reclaims, and total slab usage is bounded by total slab capacity.
    /// - No multi-segment segments (the PUT path has no sub-chunk concept); packed as a single
    ///   SlabExtentBytes view.
    ///
    /// Call order (inside handle_put): pwrite completes → this call → resp ok.
    /// Subsequent GETs take `get_chunks_slab` and get a SlabView directly, zero reg_mr, big reduction
    /// in GET latency.
    #[cfg(feature = "rdma")]
    pub fn insert_chunks_from_slab(&self, key: String, extent: Arc<SlabExtent>, meta: BlockMeta) {
        let total_size = extent.len();
        if total_size == 0 {
            return;
        }
        // RDMA PUT bypasses MemoryTier::put_chunks and lands directly on StorageTier; before
        // injecting into the slab cache we must still purge both single/chunks stale representations.
        self.invalidate_l1_entry(&key);
        // Wrap as a single-segment Bytes view (slab-backed, refcount holds the extent) so the gRPC
        // GET path can also use it.
        let whole = Bytes::from_owner(SlabExtentBytes(extent.clone()));
        let segments = vec![whole];

        let size = total_size as u64;
        let mut cache = self.chunks_cache.lock();
        // Cap: max(capacity_bytes, total slab capacity). When capacity_bytes=0 (L1 disabled), we
        // still need the slab capacity as a safety net; otherwise cache accumulates monotonically →
        // slab alloc is guaranteed to be exhausted → subsequent PUTs are rejected by the server
        // (slab full). Slab capacity is the real physical upper bound, and should be respected even
        // when L1 is disabled.
        let cache_budget = {
            let slab_cap = self
                .rdma_slab
                .get()
                .map(|s| s.capacity() as u64)
                .unwrap_or(0);
            (self.capacity_bytes as u64).max(slab_cap)
        };
        // LRU-evict until there is space. Only when cache_budget=0 (no slab and no L1) do we skip
        // eviction entirely (fallback).
        while cache_budget > 0
            && self.current_size.load(Ordering::Relaxed) + size > cache_budget
            && !cache.is_empty()
        {
            if let Some((_, evicted)) = cache.pop_lru() {
                self.current_size
                    .fetch_sub(evicted.total_size as u64, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        let entry = ChunksCacheEntry {
            segments,
            meta,
            total_size,
            slab: Some(extent),
        };
        if let Some(old) = cache.put(key, entry) {
            self.current_size
                .fetch_sub(old.total_size as u64, Ordering::Relaxed);
        }
        self.current_size.fetch_add(size, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::ShardRouter;
    use tempfile::TempDir;

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

    fn make_tier(dir: &std::path::Path) -> MemoryTier {
        let mut cfg = Config::default();
        cfg.storage.devices = vec![dir.join("nvme0")];
        cfg.metadata.rocksdb_path = dir.join("meta");
        cfg.memory_tier.capacity_mb = 1; // 1MB cap
        let router = Arc::new(ShardRouter::new(&cfg).unwrap());
        let storage = Arc::new(StorageTier::new(&cfg, router).unwrap());
        MemoryTier::new(&cfg, storage)
    }

    fn flatten_segments(segments: &[Bytes]) -> Vec<u8> {
        segments
            .iter()
            .flat_map(|segment| segment.iter().copied())
            .collect()
    }

    #[test]
    fn cache_hit_after_put() {
        let tmp = TempDir::new().unwrap();
        let tier = make_tier(tmp.path());
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "abcdef/layer_0".into(),
        };
        tier.put(&key, Bytes::from_static(b"hello"), mk_meta())
            .unwrap();
        let (data, _) = tier.get(&key).unwrap().unwrap();
        assert_eq!(data, b"hello".as_ref());
        let (h, m, _, _) = tier.stats();
        assert_eq!(h, 1);
        assert_eq!(m, 0);
    }

    #[test]
    fn single_put_invalidates_chunks_cache_for_same_object() {
        let tmp = TempDir::new().unwrap();
        let tier = make_tier(tmp.path());
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "object".into(),
        };

        tier.put_chunks(&key, vec![Bytes::from_static(b"old")], mk_meta())
            .unwrap();
        let (old_segments, _) = tier.get_chunks(&key).unwrap().unwrap();
        assert_eq!(flatten_segments(&old_segments), b"old");

        tier.put(&key, Bytes::from_static(b"new"), mk_meta())
            .unwrap();

        let (segments, _) = tier.get_chunks(&key).unwrap().unwrap();
        assert_eq!(flatten_segments(&segments), b"new");
        let (data, _) = tier.get(&key).unwrap().unwrap();
        assert_eq!(data.as_ref(), b"new");
    }

    #[test]
    fn chunk_put_invalidates_single_cache_for_same_object() {
        let tmp = TempDir::new().unwrap();
        let tier = make_tier(tmp.path());
        let key = ObjectKey {
            namespace: "test".into(),
            object_key: "object".into(),
        };

        tier.put(&key, Bytes::from_static(b"old"), mk_meta())
            .unwrap();
        let (old_data, _) = tier.get(&key).unwrap().unwrap();
        assert_eq!(old_data.as_ref(), b"old");

        tier.put_chunks(
            &key,
            vec![Bytes::from_static(b"ne"), Bytes::from_static(b"w")],
            mk_meta(),
        )
        .unwrap();

        let (data, _) = tier.get(&key).unwrap().unwrap();
        assert_eq!(data.as_ref(), b"new");
        let (segments, _) = tier.get_chunks(&key).unwrap().unwrap();
        assert_eq!(flatten_segments(&segments), b"new");
    }
}
