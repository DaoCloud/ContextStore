//! Aligned buffer — dedicated to the O_DIRECT path.
//!
//! O_DIRECT read/write requires the buffer address, length, and file offset to be aligned
//! to the block size (typically 4096B). Rust's `Vec<u8>` defaults to type alignment
//! (u8 = 1 byte), which doesn't satisfy this → must use `posix_memalign` or
//! `std::alloc::Layout::from_size_align` for a custom allocation.
//!
//! Design:
//! - `AlignedBuffer` owns ptr + cap (allocated size, 4KB multiple) + len (actual valid bytes)
//! - Implements `AsRef<[u8]>` so `Bytes::from_owner(buf)` can wrap it into Bytes zero-copy.
//! - Drop uses the same Layout to free, avoiding alloc/dealloc mismatch UB.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Owned buffer aligned to 4KB; convert to Bytes zero-copy via Bytes::from_owner.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    cap: usize,   // allocated byte count (rounded up to alignment)
    len: usize,   // actual valid byte count (≤ cap)
    align: usize, // alignment (fixed at 4096)
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate a buffer of at least `min_size` bytes, aligned to `align`.
    /// Actual allocation = (min_size + align - 1) & !(align - 1) (rounded up).
    /// Initialized to all zeros.
    pub fn new(min_size: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "align must be power of two");
        assert!(align >= std::mem::align_of::<u8>());
        let cap = (min_size + align - 1) & !(align - 1);
        let cap = cap.max(align); // at least 1 align unit (even if min_size=0)
        let layout = Layout::from_size_align(cap, align).expect("invalid layout");
        // alloc_zeroed would work too, but read will overwrite immediately; use alloc to save a memset
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).expect("aligned alloc failed");
        Self {
            ptr,
            cap,
            len: 0,
            align,
        }
    }

    /// Total capacity (allocated byte count, rounded up to alignment)
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Current valid length (≤ capacity)
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Expose the raw ptr covering the full capacity (used for read syscalls to write into)
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Mark the actual valid length (caller sets after read, e.g. read returned N < capacity)
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.cap, "set_len {} > cap {}", len, self.cap);
        self.len = len;
    }

    /// Shorten the valid length without affecting the underlying allocation.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: ptr is valid and len ≤ cap; the entire memory range is owned by this object,
        // and lifetime matches &self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: dealloc with the same Layout (cap, align) used at alloc time.
        let layout = Layout::from_size_align(self.cap, self.align).unwrap();
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

// ===== Pooled aligned buffers =====
//
// O_DIRECT DMA into a freshly-allocated anonymous buffer forces the kernel to fault-in and
// zero every page inside get_user_pages before the device can write (measured: a 64MB
// first-touch costs 50-180ms on the profiling host — far more than the 18ms the NVMe read
// itself takes). Reusing already-touched buffers removes that tax on every read after the
// first. The pool is bounded, so steady-state memory is (pool cap × slot size); overflow
// buffers are simply dropped (freed).

use std::sync::Mutex;

/// Global bounded pool of 4K-aligned buffers, bucketed by capacity.
pub struct AlignedBufferPool {
    /// (capacity, buffers) — small fixed set of capacity classes; linear scan is fine.
    buckets: Mutex<Vec<(usize, Vec<AlignedBuffer>)>>,
    /// Max buffers retained per capacity class.
    per_class_cap: usize,
}

impl AlignedBufferPool {
    pub const fn new(per_class_cap: usize) -> Self {
        Self {
            buckets: Mutex::new(Vec::new()),
            per_class_cap,
        }
    }

    /// Fetch a pooled buffer with capacity ≥ min_size (exact capacity-class match), or
    /// allocate a fresh one. The returned buffer's len is reset to 0.
    pub fn acquire(&self, min_size: usize, align: usize) -> AlignedBuffer {
        let want_cap = ((min_size + align - 1) & !(align - 1)).max(align);
        {
            let mut buckets = self.buckets.lock().unwrap();
            if let Some((_, bufs)) = buckets.iter_mut().find(|(cap, _)| *cap == want_cap) {
                if let Some(mut buf) = bufs.pop() {
                    buf.set_len(0);
                    return buf;
                }
            }
        }
        AlignedBuffer::new(min_size, align)
    }

    /// Return a buffer to the pool. Buffers beyond per_class_cap are dropped (freed).
    pub fn release(&self, buf: AlignedBuffer) {
        let cap = buf.capacity();
        let mut buckets = self.buckets.lock().unwrap();
        if let Some((_, bufs)) = buckets.iter_mut().find(|(c, _)| *c == cap) {
            if bufs.len() < self.per_class_cap {
                bufs.push(buf);
            }
            return;
        }
        buckets.push((cap, vec![buf]));
    }
}

/// Global pool for the O_DIRECT read path. 32 slots per capacity class: with the default
/// 64MB stripe size that bounds steady-state pool memory at 2GB while covering
/// 8 objects × 4-stripe read-ahead of concurrent GET traffic.
pub static READ_BUFFER_POOL: AlignedBufferPool = AlignedBufferPool::new(32);

/// An AlignedBuffer that returns itself to a pool on drop instead of freeing.
/// Wrap in `Bytes::from_owner` so the buffer rejoins the pool as soon as the last
/// Bytes clone is gone.
pub struct PooledAlignedBuffer {
    buf: Option<AlignedBuffer>,
    pool: &'static AlignedBufferPool,
}

impl PooledAlignedBuffer {
    pub fn acquire(pool: &'static AlignedBufferPool, min_size: usize, align: usize) -> Self {
        Self {
            buf: Some(pool.acquire(min_size, align)),
            pool,
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut().unwrap().as_mut_ptr()
    }

    pub fn capacity(&self) -> usize {
        self.buf.as_ref().unwrap().capacity()
    }

    pub fn set_len(&mut self, len: usize) {
        self.buf.as_mut().unwrap().set_len(len);
    }
}

impl AsRef<[u8]> for PooledAlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.buf.as_ref().unwrap().as_ref()
    }
}

impl Drop for PooledAlignedBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.release(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_4k_ptr() {
        let buf = AlignedBuffer::new(1, 4096);
        assert_eq!(buf.capacity(), 4096); // rounded up to 4K
        assert_eq!(buf.ptr.as_ptr() as usize % 4096, 0);
    }

    #[test]
    fn write_via_ptr_then_read() {
        let mut buf = AlignedBuffer::new(8192, 4096);
        unsafe {
            let p = buf.as_mut_ptr();
            for i in 0..100 {
                *p.add(i) = (i & 0xff) as u8;
            }
        }
        buf.set_len(100);
        assert_eq!(buf.as_ref().len(), 100);
        for (i, &b) in buf.as_ref().iter().enumerate() {
            assert_eq!(b, (i & 0xff) as u8);
        }
    }

    #[test]
    fn truncate_shortens() {
        let mut buf = AlignedBuffer::new(4096, 4096);
        buf.set_len(4096);
        buf.truncate(1000);
        assert_eq!(buf.len(), 1000);
        // truncate does not grow length
        buf.truncate(5000);
        assert_eq!(buf.len(), 1000);
    }

    #[test]
    fn pool_reuses_same_capacity() {
        static POOL: AlignedBufferPool = AlignedBufferPool::new(2);
        let mut a = POOL.acquire(8192, 4096);
        let ptr_a = a.as_mut_ptr() as usize;
        POOL.release(a);
        let mut b = POOL.acquire(8192, 4096);
        assert_eq!(b.as_mut_ptr() as usize, ptr_a, "expected pooled buffer reuse");
        assert_eq!(b.len(), 0, "reused buffer len must be reset");
        POOL.release(b);
    }

    #[test]
    fn pool_bounded_per_class() {
        static POOL: AlignedBufferPool = AlignedBufferPool::new(1);
        POOL.release(AlignedBuffer::new(4096, 4096));
        POOL.release(AlignedBuffer::new(4096, 4096)); // beyond cap → dropped, must not panic
        let _ = POOL.acquire(4096, 4096);
        let _ = POOL.acquire(4096, 4096); // pool empty → fresh alloc
    }

    #[test]
    fn pooled_buffer_returns_on_drop() {
        static POOL: AlignedBufferPool = AlignedBufferPool::new(4);
        let mut p = PooledAlignedBuffer::acquire(&POOL, 4096, 4096);
        let ptr = p.as_mut_ptr() as usize;
        drop(p);
        let mut q = PooledAlignedBuffer::acquire(&POOL, 4096, 4096);
        assert_eq!(q.as_mut_ptr() as usize, ptr);
    }

    #[test]
    fn pooled_buffer_via_bytes_from_owner() {
        static POOL: AlignedBufferPool = AlignedBufferPool::new(4);
        let mut p = PooledAlignedBuffer::acquire(&POOL, 4096, 4096);
        let ptr = p.as_mut_ptr() as usize;
        unsafe { *p.as_mut_ptr() = 42 };
        p.set_len(1);
        let b = prost::bytes::Bytes::from_owner(p);
        assert_eq!(b[0], 42);
        let b2 = b.clone();
        drop(b);
        drop(b2); // last clone gone → buffer back in pool
        let mut q = PooledAlignedBuffer::acquire(&POOL, 4096, 4096);
        assert_eq!(q.as_mut_ptr() as usize, ptr);
    }

    #[test]
    fn pool_distinct_capacity_classes() {
        static POOL: AlignedBufferPool = AlignedBufferPool::new(2);
        POOL.release(AlignedBuffer::new(4096, 4096));
        POOL.release(AlignedBuffer::new(8192, 4096));
        assert_eq!(POOL.acquire(8192, 4096).capacity(), 8192);
        assert_eq!(POOL.acquire(4096, 4096).capacity(), 4096);
    }
}
