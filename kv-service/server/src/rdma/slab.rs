//! RDMA Slab — one `ibv_reg_mr` at startup registers a large block of memory; the GET hot
//! path slices from it with zero registration.
//!
//! ## Motivation
//!
//! The original GET path did a temporary `register_mr_raw` (= `ibv_reg_mr` syscall) per
//! striping chunk — up to 32 reg + 32 dereg per GET, capping single-worker bandwidth at
//! 0.63 GB/s (vs. the 12 GB/s hardware ceiling). The slab pre-registers ~8GB at startup;
//! after chunks_cache data is memcpy'd into it, GET issues an RDMA WRITE with the slab's lkey
//! + (base+offset), for **zero reg syscalls**; because slab memory is contiguous, 32 WRITEs
//! coalesce into 1.
//!
//! ## Memory model (1× zero-copy view)
//!
//! On insert the data is stored in the slab only once; the gRPC path's `segments` become
//! `Bytes::from_owner` slices pointing into the slab, sharing the same memory with RDMA.
//! Extent reclamation is purely RAII: the space is returned only when the last
//! `Arc<SlabExtent>` drops (once the cache entry field + all outstanding gRPC `Bytes` clones
//! are released).
//!
//! ## Allocator
//!
//! Address-ordered best-fit free-list with immediate left/right coalescing. Metadata lives
//! out-of-band (two BTrees), not as boundary tags in the registered region. alloc/free only
//! happens on insert/evict (not on the GET hot path).

use crate::metadata::BlockMeta;
use crate::rdma::context::{MemRegion, RdmaContext};
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use prost::bytes::Bytes;
use rdma_sys::ibv_access_flags;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::Arc;

/// Extent alignment granularity. All offsets/capacities in the slab are integer multiples of
/// this, guaranteeing 4KB alignment (leaves room for the v2 "O_DIRECT read directly into slab"
/// path: O_DIRECT requires 4KB-aligned buffers).
const SLAB_ALIGN: usize = 4096;

/// Alignment when attempting huge pages (2MB). The slab total size is rounded up to this so
/// `MAP_HUGETLB` won't EINVAL.
const HUGE_ALIGN: usize = 2 * 1024 * 1024;

#[inline]
fn round_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// RDMA WRITE source descriptor (Copy): points to a pre-registered region inside the slab.
#[derive(Clone, Copy, Debug)]
pub struct SlabView {
    pub addr: u64,
    pub lkey: u32,
    pub len: u64,
}

/// Slab usage stats (metrics / debug).
#[derive(Clone, Copy, Debug)]
pub struct SlabStats {
    pub total: u64,
    pub used: u64,
    pub free: u64,
    pub high_watermark: u64,
}

// ===================== mmap backing =====================

/// Underlying slab memory: anonymous mmap (optional huge pages); munmap on Drop.
struct SlabBacking {
    ptr: NonNull<u8>,
    len: usize,
}

impl SlabBacking {
    /// Allocate `len` bytes of anonymous memory. Try `MAP_HUGETLB` first; fall back to normal
    /// pages on failure (no huge pages reserved).
    fn new(len: usize) -> Result<Self> {
        unsafe {
            let do_mmap = |extra: i32| {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | extra,
                    -1,
                    0,
                )
            };
            let mut p = do_mmap(libc::MAP_HUGETLB);
            let mut huge = true;
            if p == libc::MAP_FAILED {
                huge = false;
                p = do_mmap(0);
            }
            if p == libc::MAP_FAILED {
                return Err(anyhow!(
                    "mmap {} bytes failed: {}",
                    len,
                    std::io::Error::last_os_error()
                ));
            }
            let ptr = NonNull::new(p as *mut u8).ok_or_else(|| anyhow!("mmap returned null"))?;
            tracing::info!(
                "SlabBacking: mmap {} bytes at {:p} (huge_pages={})",
                len,
                ptr.as_ptr(),
                huge
            );
            Ok(Self { ptr, len })
        }
    }
}

impl Drop for SlabBacking {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
    }
}

// ===================== free-list allocator =====================

/// Address-ordered best-fit free-list. Sizes/offsets are in bytes and are all `SLAB_ALIGN` multiples.
struct FreeList {
    /// offset -> len, for O(log n) neighbor lookup during coalesce.
    by_offset: BTreeMap<u64, u64>,
    /// (len, offset), for best-fit queries (range: smallest len >= size).
    by_size: BTreeSet<(u64, u64)>,
    bytes_free: u64,
    bytes_used: u64,
    high_watermark: u64,
    total: u64,
}

impl FreeList {
    fn new(total: u64) -> Self {
        let mut by_offset = BTreeMap::new();
        let mut by_size = BTreeSet::new();
        by_offset.insert(0u64, total);
        by_size.insert((total, 0u64));
        Self {
            by_offset,
            by_size,
            bytes_free: total,
            bytes_used: 0,
            high_watermark: 0,
            total,
        }
    }

    #[inline]
    fn insert_block(&mut self, offset: u64, len: u64) {
        debug_assert!(len > 0);
        self.by_offset.insert(offset, len);
        self.by_size.insert((len, offset));
    }

    #[inline]
    fn remove_block(&mut self, offset: u64, len: u64) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(len, offset));
    }

    /// Best-fit allocate `size` bytes (caller has already rounded to SLAB_ALIGN). Returns
    /// offset, or None if full.
    fn alloc(&mut self, size: u64) -> Option<u64> {
        debug_assert_eq!(size % SLAB_ALIGN as u64, 0);
        // Smallest (len, offset) with len >= size; ties broken by lowest offset.
        let (blk_len, blk_off) = self.by_size.range((size, 0)..).next().copied()?;
        self.remove_block(blk_off, blk_len);
        if blk_len > size {
            // split: take head [blk_off, blk_off+size); put remainder back on the free-list.
            self.insert_block(blk_off + size, blk_len - size);
        }
        self.bytes_free -= size;
        self.bytes_used += size;
        if self.bytes_used > self.high_watermark {
            self.high_watermark = self.bytes_used;
        }
        Some(blk_off)
    }

    /// Return [offset, offset+cap); immediately coalesces with adjacent free blocks.
    fn free(&mut self, offset: u64, cap: u64) {
        debug_assert_eq!(cap % SLAB_ALIGN as u64, 0);
        self.bytes_free += cap;
        self.bytes_used -= cap;

        let mut start = offset;
        let mut end = offset + cap;

        // Look up left/right neighbors first (copy out owned values so we finish the immutable
        // borrow on by_offset), then perform mutable remove/insert to avoid borrow conflicts.
        // Right neighbor: a free block starting exactly at `end`.
        let right = self.by_offset.get(&end).copied();
        // Left neighbor: a free block that ends at `start` (the largest one with offset < start).
        let left = self
            .by_offset
            .range(..start)
            .next_back()
            .map(|(&off, &len)| (off, len));

        if let Some(rlen) = right {
            self.remove_block(end, rlen);
            end += rlen;
        }
        if let Some((loff, llen)) = left {
            if loff + llen == start {
                self.remove_block(loff, llen);
                start = loff;
            }
        }
        self.insert_block(start, end - start);
    }

    fn stats(&self) -> SlabStats {
        SlabStats {
            total: self.total,
            used: self.bytes_used,
            free: self.bytes_free,
            high_watermark: self.high_watermark,
        }
    }
}

// ===================== Slab shared core =====================

/// Slab shared state. Shared between `RdmaSlab` and every `SlabExtent` via `Arc`.
///
/// # Field Drop order (critical)
/// Rust drops in declaration order: `mrs` MUST be declared **before** `backing` / `_ctxs` so
/// that Drop performs `ibv_dereg_mr` first, then munmaps memory / releases the PD; otherwise
/// the NIC could translate already-unmapped pages (UB).
///
/// # Multi-NIC support
/// The same host backing is `ibv_reg_mr`'d once per PD across N NICs, yielding N (lkey,
/// MemRegion) tuples. `mrs[i]` corresponds to `_ctxs[i]`. server.rs handle_client picks
/// nic_idx based on which listener it's on; `view(nic_idx)` returns the SlabView with the
/// matching lkey. One copy of data; hot path is lock-free.
struct SlabInner {
    /// One MR per NIC (covering the entire backing). Dropped first so `ibv_dereg_mr` runs
    /// before backing munmap and PD release. `mrs[i]` corresponds to `lkeys[i]` and `_ctxs[i]`.
    #[allow(dead_code)]
    mrs: Vec<MemRegion>,
    /// Underlying mmap memory (RAII, Drop→munmap). Held only for the Drop side effect (base is cached separately).
    #[allow(dead_code)]
    backing: SlabBacking,
    /// Hold every RdmaContext so PDs outlive the MRs (MRs belong to a PD). Held only for lifetime.
    #[allow(dead_code)]
    _ctxs: Vec<Arc<RdmaContext>>,
    /// Cached base pointer (== backing.ptr) to avoid indirection on the hot path.
    base: NonNull<u8>,
    /// Total slab size in bytes.
    len: usize,
    /// Per-NIC lkey; index = nic_idx, matches the order of `mrs`/`_ctxs`.
    /// Read-only on the GET hot path, unchanged after startup, lock-free.
    lkeys: Vec<u32>,
    /// Per-NIC rkey (RDMA remote access key). The client needs this rkey to RDMA WRITE into
    /// memory registered in the corresponding NIC's PD. Used by the PUT data path (server
    /// returns dst_rkey to the client).
    rkeys: Vec<u32>,
    /// Allocator (only holds the lock on insert/evict, not on the GET hot path).
    alloc: Mutex<FreeList>,
}

// SlabInner holds a raw NonNull<u8> pointer, requiring manual assertions. The underlying mmap
// memory supports multi-threaded reads, and the allocator has a Mutex.
unsafe impl Send for SlabInner {}
unsafe impl Sync for SlabInner {}

/// Pre-registered RDMA slab. `Arc`-clones are cheap; multiple client threads share the same registered region.
#[derive(Clone)]
pub struct RdmaSlab {
    inner: Arc<SlabInner>,
}

impl RdmaSlab {
    /// Register a slab of `size_bytes` (rounded up to 2MB). `ibv_reg_mr` is called once per
    /// `RdmaContext`, yielding N (lkey, MemRegion) tuples. One copy of data, shared host
    /// backing across NICs.
    ///
    /// On failure (mmap ENOMEM / reg_mr RLIMIT_MEMLOCK / any per-NIC registration failure),
    /// returns Err; the caller should warn and degrade (don't set the slab; every GET uses
    /// per-chunk fallback).
    pub fn new(ctxs: &[Arc<RdmaContext>], size_bytes: usize) -> Result<Self> {
        if ctxs.is_empty() {
            return Err(anyhow!("RdmaSlab::new: empty ctx list"));
        }
        let len = round_up(size_bytes.max(HUGE_ALIGN), HUGE_ALIGN);
        let backing = SlabBacking::new(len)?;
        let base = backing.ptr;

        // Register the same memory once per PD. **LOCAL_WRITE + REMOTE_WRITE**:
        // - LOCAL_WRITE: on RDMA GET, the server WRITEs slab contents to the client (source side).
        // - REMOTE_WRITE: on RDMA PUT, the client RDMA WRITEs into the slab (server is the
        //   WRITE target); the client uses the rkey returned by the server — without this
        //   flag it would get REM_ACCESS_ERR.
        let mut mrs: Vec<MemRegion> = Vec::with_capacity(ctxs.len());
        let mut lkeys: Vec<u32> = Vec::with_capacity(ctxs.len());
        let mut rkeys: Vec<u32> = Vec::with_capacity(ctxs.len());
        for (i, ctx) in ctxs.iter().enumerate() {
            let mr = unsafe {
                ctx.register_mr_raw(
                    base.as_ptr(),
                    len,
                    ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                        | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0,
                )
                .map_err(|e| anyhow!("RdmaSlab reg_mr nic_idx={}: {}", i, e))?
            };
            lkeys.push(mr.lkey);
            rkeys.push(mr.rkey);
            mrs.push(mr);
        }

        tracing::info!(
            "RdmaSlab registered: {} bytes ({} MB) on {} NIC(s), lkeys={:?} rkeys={:?}",
            len,
            len / (1024 * 1024),
            ctxs.len(),
            lkeys.iter().map(|k| format!("0x{:x}", k)).collect::<Vec<_>>(),
            rkeys.iter().map(|k| format!("0x{:x}", k)).collect::<Vec<_>>(),
        );

        Ok(Self {
            inner: Arc::new(SlabInner {
                mrs,
                backing,
                _ctxs: ctxs.to_vec(),
                base,
                len,
                lkeys,
                rkeys,
                alloc: Mutex::new(FreeList::new(len as u64)),
            }),
        })
    }

    /// Number of registered NICs (= exclusive upper bound on nic_idx).
    pub fn num_nics(&self) -> usize {
        self.inner.lkeys.len()
    }

    /// Allocate an extent that can hold `size` logical bytes (capacity rounded up to
    /// SLAB_ALIGN). Returns None if the slab is full (no large enough contiguous free block);
    /// caller should fall back to the heap path.
    pub fn alloc(&self, size: usize) -> Option<SlabExtent> {
        if size == 0 {
            return None;
        }
        let cap = round_up(size, SLAB_ALIGN) as u64;
        let offset = self.inner.alloc.lock().alloc(cap)?;
        Some(SlabExtent {
            slab: self.inner.clone(),
            offset,
            len: size as u64,
            capacity: cap,
        })
    }

    pub fn stats(&self) -> SlabStats {
        self.inner.alloc.lock().stats()
    }

    /// Total slab capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.inner.len
    }
}

// ===================== SlabExtent (RAII) =====================

/// A region allocated within the slab. On Drop returns `[offset, offset+capacity)` to the free-list.
///
/// Shared via `Arc<SlabExtent>`: the cache entry holds one, each gRPC `Bytes` view holds one
/// via the `SlabExtentBytes` owner. Only the last Arc drop actually reclaims — no
/// use-after-free.
pub struct SlabExtent {
    slab: Arc<SlabInner>,
    offset: u64,
    /// Logical length (== bytes actually written, used for as_ref / WRITE length).
    len: u64,
    /// Actual reservation (rounded up to SLAB_ALIGN); returned on Drop.
    capacity: u64,
}

impl SlabExtent {
    /// memcpy multiple `Bytes` segments into this extent in order.
    ///
    /// Takes `&mut self`: must be called BEFORE the extent is wrapped in an `Arc` (shared),
    /// which statically guarantees there are no concurrent readers (NIC WRITE source / gRPC
    /// reader), so filling and reading do not race.
    pub fn copy_in(&mut self, segments: &[Bytes]) {
        let base = unsafe { self.slab.base.as_ptr().add(self.offset as usize) };
        let mut off = 0usize;
        for seg in segments {
            if seg.is_empty() {
                continue;
            }
            debug_assert!(off + seg.len() <= self.len as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(seg.as_ptr(), base.add(off), seg.len());
            }
            off += seg.len();
        }
        debug_assert_eq!(off as u64, self.len);
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.slab.base.as_ptr().add(self.offset as usize) }
    }

    /// The number of bytes actually reserved (capacity rounded up to 4K, not logical len).
    /// Used by the RDMA GET cache-miss path (storage.get_into_ptr needs capacity for its 4K alignment check).
    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.capacity as usize
    }

    /// Absolute address of this segment in the process address space (= base + offset). Used for `ibv_sge.addr`.
    #[inline]
    pub fn addr(&self) -> u64 {
        self.slab.base.as_ptr() as u64 + self.offset
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// lkey for the given NIC. 0-indexed, matches the ctxs order in `RdmaSlab::new`.
    /// Panics if nic_idx >= num_nics (config error at startup — fail-fast is safer than picking the wrong NIC).
    #[inline]
    pub fn lkey(&self, nic_idx: usize) -> u32 {
        self.slab.lkeys[nic_idx]
    }

    /// rkey for the given NIC. PUT data path: the server hands this rkey to the client, and
    /// the client uses it as the remote access key for its RDMA WRITE.
    #[inline]
    pub fn rkey(&self, nic_idx: usize) -> u32 {
        self.slab.rkeys[nic_idx]
    }

    /// Return the SlabView for the given NIC (same addr, different lkey).
    #[inline]
    pub fn view(&self, nic_idx: usize) -> SlabView {
        SlabView {
            addr: self.addr(),
            lkey: self.lkey(nic_idx),
            len: self.len,
        }
    }
}

impl Drop for SlabExtent {
    fn drop(&mut self) {
        self.slab.alloc.lock().free(self.offset, self.capacity);
    }
}

// ===================== Bytes owner adapter =====================

/// Lets `Arc<SlabExtent>` act as an owner for `Bytes::from_owner`, zero-copy wrapping slab memory into `Bytes`.
///
/// `Bytes::from_owner` requires owner: `AsRef<[u8]> + Send + 'static`.
/// `Arc<SlabExtent>` satisfies Send (SlabInner is manually Send+Sync) and 'static (no borrows).
pub struct SlabExtentBytes(pub Arc<SlabExtent>);

impl AsRef<[u8]> for SlabExtentBytes {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: extent's memory is valid and read-only for the lifetime of this Arc; len is the valid length written by copy_in.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), self.0.len()) }
    }
}

// ===================== GET path return type =====================

/// The return of `MemoryTier::get_chunks_slab`: provides both the `SlabView` (Copy) for RDMA
/// WRITE, and via `_pin` holds an `Arc<SlabExtent>` to prevent eviction/reclamation during
/// WRITE + poll (UAF).
pub struct SlabPlacement {
    pub view: SlabView,
    pub meta: BlockMeta,
    /// Only there to extend the extent's lifetime past poll_n; caller drops it to release the pin.
    pub _pin: Arc<SlabExtent>,
}

// ===================== Unit tests (pure free-list logic, no RDMA hardware required) =====================

#[cfg(test)]
mod tests {
    use super::*;

    const K: u64 = SLAB_ALIGN as u64; // 4096

    #[test]
    fn round_up_to_align() {
        assert_eq!(round_up(1, 4096), 4096);
        assert_eq!(round_up(4096, 4096), 4096);
        assert_eq!(round_up(4097, 4096), 8192);
        assert_eq!(round_up(0, 4096), 0);
    }

    #[test]
    fn alloc_split_and_offset() {
        let mut fl = FreeList::new(16 * K);
        // First alloc slices from offset 0.
        let a = fl.alloc(4 * K).unwrap();
        assert_eq!(a, 0);
        // Remainder [4K, 16K) remains; the next best-fit takes it.
        let b = fl.alloc(4 * K).unwrap();
        assert_eq!(b, 4 * K);
        let s = fl.stats();
        assert_eq!(s.used, 8 * K);
        assert_eq!(s.free, 8 * K);
        assert_eq!(s.high_watermark, 8 * K);
    }

    #[test]
    fn best_fit_picks_smallest_sufficient() {
        let mut fl = FreeList::new(100 * K);
        // Manufacture fragmentation: split 3 regions and free the middle, forming holes of different sizes.
        let a = fl.alloc(10 * K).unwrap(); // [0,10K)
        let b = fl.alloc(20 * K).unwrap(); // [10K,30K)
        let _c = fl.alloc(5 * K).unwrap(); // [30K,35K), leftover [35K,100K)
        fl.free(a, 10 * K); // hole 1: [0,10K) len=10K
        fl.free(b, 20 * K); // hole 2: [10K,30K) len=20K (not adjacent to hole 1? adjacent! coalesces into [0,30K)=30K)
                            // Actually: a and b are adjacent, free coalesces into [0,30K).
        // Current free blocks: [0,30K)=30K, [35K,100K)=65K.
        // Request 8K: best-fit should pick the smaller 30K block (not 65K), returning offset 0.
        let d = fl.alloc(8 * K).unwrap();
        assert_eq!(d, 0);
    }

    #[test]
    fn free_coalesce_right() {
        let mut fl = FreeList::new(30 * K);
        let a = fl.alloc(10 * K).unwrap(); // [0,10K)
        let _b = fl.alloc(10 * K).unwrap(); // [10K,20K), leftover [20K,30K)
        // Free a [0,10K): right neighbor is the occupied _b (not in free-list); no left neighbor. No coalesce.
        fl.free(a, 10 * K);
        // Free now: [0,10K), [20K,30K). Request 10K hits [0,10K).
        assert_eq!(fl.alloc(10 * K).unwrap(), 0);
    }

    #[test]
    fn free_coalesce_both_sides() {
        let mut fl = FreeList::new(30 * K);
        let a = fl.alloc(10 * K).unwrap(); // [0,10K)
        let b = fl.alloc(10 * K).unwrap(); // [10K,20K)
        let _c = fl.alloc(10 * K).unwrap(); // [20K,30K)
        // Free both ends first, then the middle → middle free should coalesce both sides, restoring the whole [0,30K).
        fl.free(a, 10 * K); // [0,10K) free
        fl.free(_c, 10 * K); // [20K,30K) free
        fl.free(b, 10 * K); // [10K,20K) free → coalesces into [0,30K)
        // Being able to allocate the whole 30K in one shot proves full coalescing.
        assert_eq!(fl.alloc(30 * K).unwrap(), 0);
        assert_eq!(fl.stats().free, 0);
    }

    #[test]
    fn alloc_full_returns_none() {
        let mut fl = FreeList::new(8 * K);
        assert!(fl.alloc(8 * K).is_some());
        // No space left.
        assert!(fl.alloc(K).is_none());
    }

    #[test]
    fn alloc_too_big_returns_none() {
        let mut fl = FreeList::new(8 * K);
        // Single request exceeds total capacity.
        assert!(fl.alloc(16 * K).is_none());
        // But <= capacity still works.
        assert!(fl.alloc(8 * K).is_some());
    }

    #[test]
    fn fragmentation_then_recover() {
        let mut fl = FreeList::new(40 * K);
        let blocks: Vec<u64> = (0..4).map(|_| fl.alloc(10 * K).unwrap()).collect();
        assert_eq!(fl.stats().free, 0);
        // Free every other block (10K holes wedged between occupied blocks, no adjacency, no coalesce).
        fl.free(blocks[0], 10 * K);
        fl.free(blocks[2], 10 * K);
        // Two independent 10K holes: a 20K request cannot fit (no contiguous 20K).
        assert!(fl.alloc(20 * K).is_none());
        // 10K fits.
        assert!(fl.alloc(10 * K).is_some());
        // After releasing all, coalescing should give back the whole block.
        fl.free(blocks[1], 10 * K);
        fl.free(blocks[3], 10 * K);
        // At this point the earlier alloc for blocks[0] has taken one hole; remainder should coalesce. Freeing it:
        // (Simplified: just verify the free total is correct.)
        assert_eq!(fl.stats().used, 10 * K);
    }
}
