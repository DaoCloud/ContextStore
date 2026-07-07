//! GpuFileHandle — RAII cuFile file handle
//!
//! Usage:
//!   let fd = std::fs::File::open(path)?;
//!   let h = GpuFileHandle::register(fd)?;
//!   h.pread(&mut gpu_buf, file_offset, size)?;     // DMA → GPU
//!   h.pwrite(&gpu_buf, file_offset, size)?;        // GPU → DMA → NVMe

use super::buffer::GpuBuffer;
use super::driver::GdsDriver;
use super::ffi::*;
use crate::error::{KVError, Result};
use std::fs::File;
use std::os::raw::c_void;
use std::os::unix::io::{AsRawFd, IntoRawFd};

pub struct GpuFileHandle {
    handle: CUfileHandle_t,
    fd: i32,
}

unsafe impl Send for GpuFileHandle {}
unsafe impl Sync for GpuFileHandle {}

impl GpuFileHandle {
    /// Takes ownership of File (holds fd internally, closes fd on drop)
    pub fn register(file: File) -> Result<Self> {
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        let fd = file.into_raw_fd();
        let mut descr = CUfileDescr_t {
            handle_type: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            fd,
            _pad: [0; 4],
            fs_ops: std::ptr::null_mut(),
        };
        let mut handle: CUfileHandle_t = std::ptr::null_mut();
        let err = unsafe { (d.cu_file_handle_register)(&mut handle, &mut descr) };
        if err.err != 0 {
            // Registration failed; close fd ourselves
            unsafe { libc::close(fd) };
            return Err(super::cufile_err("cuFileHandleRegister", err.err));
        }
        Ok(Self { handle, fd })
    }

    /// Borrows a File reference (no ownership transfer). Caller must ensure File outlives GpuFileHandle.
    pub fn register_borrowed(file: &File) -> Result<Self> {
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        // fd here is only used as a registration parameter; descr can be destroyed after register returns.
        // Borrowing is therefore safe, but we must not close fd on drop → use fd=-1 as a placeholder.
        let mut descr = CUfileDescr_t {
            handle_type: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            fd: file.as_raw_fd(),
            _pad: [0; 4],
            fs_ops: std::ptr::null_mut(),
        };
        let mut handle: CUfileHandle_t = std::ptr::null_mut();
        let err = unsafe { (d.cu_file_handle_register)(&mut handle, &mut descr) };
        if err.err != 0 {
            return Err(super::cufile_err("cuFileHandleRegister", err.err));
        }
        Ok(Self { handle, fd: -1 })
    }

    /// GDS read: NVMe → GPU
    pub fn pread(&self, buf: &mut GpuBuffer, file_offset: u64, size: usize) -> Result<usize> {
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        if size > buf.len() {
            return Err(KVError::Internal(format!(
                "GpuFileHandle::pread size {} > buf {}",
                size,
                buf.len()
            )));
        }
        let n = unsafe {
            (d.cu_file_read)(
                self.handle,
                buf.as_ptr() as *mut c_void,
                size,
                file_offset as i64,
                0,
            )
        };
        if n < 0 {
            return Err(super::cufile_err("cuFileRead", n as i32));
        }
        Ok(n as usize)
    }

    /// GDS write: GPU → NVMe
    pub fn pwrite(&self, buf: &GpuBuffer, file_offset: u64, size: usize) -> Result<usize> {
        let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not init".into()))?;
        if size > buf.len() {
            return Err(KVError::Internal(format!(
                "GpuFileHandle::pwrite size {} > buf {}",
                size,
                buf.len()
            )));
        }
        let n = unsafe {
            (d.cu_file_write)(
                self.handle,
                buf.as_ptr() as *const c_void,
                size,
                file_offset as i64,
                0,
            )
        };
        if n < 0 {
            return Err(super::cufile_err("cuFileWrite", n as i32));
        }
        Ok(n as usize)
    }
}

impl Drop for GpuFileHandle {
    fn drop(&mut self) {
        if let Some(d) = GdsDriver::get() {
            unsafe { (d.cu_file_handle_deregister)(self.handle) };
        }
        if self.fd >= 0 {
            unsafe { libc::close(self.fd) };
        }
    }
}
