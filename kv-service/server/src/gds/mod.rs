//! GDS (GPUDirect Storage) integration
//!
//! Loads libcufile.so + libcudart.so at runtime via libloading;
//! no CUDA / cufile header dependency at compile time.
//!
//! Architecture:
//! - `ffi`     : hand-written minimal FFI bindings (cuFile + a few required CUDA runtime APIs)
//! - `driver`  : global singleton holding dynamic library handles, manages cuFile driver open/close
//! - `handle`  : `GpuFileHandle` (RAII wrapper around CUfileHandle_t)
//! - `buffer`  : `GpuBuffer` abstraction (owned / borrowed GPU memory)
//!
//! Design notes:
//! 1. On startup, `GdsDriver::init()` probes for libcufile.so; on failure it only warns (no panic)
//! 2. `is_available()` lets callers decide whether to take the GDS path
//! 3. All cuFile calls dispatch through the driver singleton to avoid duplicate loading

#![cfg(feature = "gds")]

pub mod buffer;
pub mod driver;
pub mod ffi;
pub mod handle;

pub use buffer::GpuBuffer;
pub use driver::{is_available, GdsDriver};
pub use handle::GpuFileHandle;

use crate::error::{KVError, Result};

/// Unified error conversion
pub(crate) fn cufile_err(op: &str, code: i32) -> KVError {
    KVError::Internal(format!("cuFile {} failed: status={}", op, code))
}

pub(crate) fn cuda_err(op: &str, code: i32) -> KVError {
    KVError::Internal(format!("CUDA {} failed: code={}", op, code))
}

pub(crate) fn map_result(op: &'static str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(cufile_err(op, status))
    }
}
