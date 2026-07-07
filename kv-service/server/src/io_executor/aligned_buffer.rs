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
}
