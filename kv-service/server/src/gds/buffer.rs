//! GpuBuffer — RAII GPU memory abstraction
//!
//! Two sources:
//! 1. `GpuBuffer::alloc(n)` — self-managed cudaMalloc + cuFileBufRegister (cudaFree + deregister on drop)
//! 2. `GpuBuffer::borrow(ptr, len)` — IPC mapping from vLLM (used in Phase 2)
//!
//! Synchronous H2D / D2H copies go through cudaMemcpy (fallback path when not using GDS).

use super::driver::{cuda_free_raw, cuda_malloc_raw, GdsDriver};
use super::ffi::*;
use crate::error::{KVError, Result};
use std::os::raw::c_void;
enum Ownership {
    Owned,
    Borrowed,
    /// Mapped in via cuIpcOpenMemHandle; requires cuIpcCloseMemHandle on drop
    IpcImported,
}

pub struct GpuBuffer {
    ptr: *mut c_void,
    len: usize,
    registered: bool,
    ownership: Ownership,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Allocate + register. Rolls back on failure.
    pub fn alloc(len: usize) -> Result<Self> {
        let ptr = cuda_malloc_raw(len)?;
        let mut buf = Self {
            ptr,
            len,
            registered: false,
            ownership: Ownership::Owned,
        };
        buf.register()?;
        Ok(buf)
    }

    /// Borrows external GPU memory (e.g. result of cuIpcOpenMemHandle). Does not free, but does register.
    /// # Safety
    /// Caller guarantees `ptr` points to device memory with at least `len` valid bytes.
    pub unsafe fn borrow(ptr: *mut c_void, len: usize) -> Result<Self> {
        let mut buf = Self {
            ptr,
            len,
            registered: false,
            ownership: Ownership::Borrowed,
        };
        buf.register()?;
        Ok(buf)
    }

    /// Maps a client-process GPU buffer via CUDA IPC handle.
    /// `flags` recommended value: `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1`.
    ///
    /// Note: client and server must be on the same host, and the server must have already
    /// setDevice'd to the correct GPU before this call.
    pub fn from_ipc_handle(handle_bytes: &[u8], len: usize) -> Result<Self> {
        if handle_bytes.len() != CU_IPC_HANDLE_MAX_SIZE {
            return Err(KVError::InvalidArgument(format!(
                "IPC handle must be {} bytes, got {}",
                CU_IPC_HANDLE_MAX_SIZE,
                handle_bytes.len()
            )));
        }
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        let mut handle = CUipcMemHandle {
            reserved: [0u8; CU_IPC_HANDLE_MAX_SIZE],
        };
        handle.reserved.copy_from_slice(handle_bytes);

        let mut dptr: CUdeviceptr = 0;
        let cu = unsafe {
            (d.cu_ipc_open_mem_handle)(
                &mut dptr,
                handle,
                CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
            )
        };
        if cu != 0 {
            return Err(super::cuda_err("cuIpcOpenMemHandle", cu));
        }
        let ptr = dptr as *mut c_void;
        let mut buf = Self {
            ptr,
            len,
            registered: false,
            ownership: Ownership::IpcImported,
        };
        // Register for cuFile direct DMA
        if let Err(e) = buf.register() {
            // Registration failed; close IPC before returning
            unsafe { (d.cu_ipc_close_mem_handle)(dptr) };
            buf.ownership = Ownership::Borrowed; // prevent double-close on drop
            return Err(e);
        }
        Ok(buf)
    }

    fn register(&mut self) -> Result<()> {
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        let err = unsafe { (d.cu_file_buf_register)(self.ptr, self.len, 0) };
        if err.err != 0 {
            return Err(super::cufile_err("cuFileBufRegister", err.err));
        }
        self.registered = true;
        Ok(())
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// device → host copy (cudaMemcpy). For non-GDS fallback / tests.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        let n = dst.len().min(self.len);
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        let err = unsafe {
            (d.cuda_memcpy)(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                n,
                cudaMemcpyDeviceToHost,
            )
        };
        if err != 0 {
            return Err(super::cuda_err("cudaMemcpy D2H", err));
        }
        Ok(())
    }

    /// host → device copy
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        let n = src.len().min(self.len);
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        let err = unsafe {
            (d.cuda_memcpy)(
                self.ptr,
                src.as_ptr() as *const c_void,
                n,
                cudaMemcpyHostToDevice,
            )
        };
        if err != 0 {
            return Err(super::cuda_err("cudaMemcpy H2D", err));
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.registered {
            if let Some(d) = GdsDriver::get() {
                unsafe {
                    (d.cu_file_buf_deregister)(self.ptr);
                }
            }
        }
        match self.ownership {
            Ownership::Owned if !self.ptr.is_null() => cuda_free_raw(self.ptr),
            Ownership::IpcImported if !self.ptr.is_null() => {
                if let Some(d) = GdsDriver::get() {
                    unsafe { (d.cu_ipc_close_mem_handle)(self.ptr as CUdeviceptr) };
                }
            }
            _ => {}
        }
    }
}
