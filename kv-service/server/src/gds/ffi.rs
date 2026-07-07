//! Hand-written minimal FFI bindings
//!
//! Corresponds to:
//! - `cufile.h` (NVIDIA GDS, typically at /usr/local/cuda/gds/include/)
//! - `cuda_runtime.h` (CUDA runtime)
//!
//! Only declares the symbols and types actually used, avoiding a bindgen build dependency.
//! ABI aligned with CUDA 12.x (cuFile 1.6+).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// ===================== CUDA Runtime =====================

pub type cudaError_t = c_int;
pub const cudaSuccess: cudaError_t = 0;

/// `cudaMemcpyKind`
pub type cudaMemcpyKind = c_int;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;

pub type CudaMalloc = unsafe extern "C" fn(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
pub type CudaFree = unsafe extern "C" fn(devPtr: *mut c_void) -> cudaError_t;
pub type CudaMemcpy = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub type CudaSetDevice = unsafe extern "C" fn(device: c_int) -> cudaError_t;
pub type CudaDeviceSynchronize = unsafe extern "C" fn() -> cudaError_t;
pub type CudaGetErrorString = unsafe extern "C" fn(err: cudaError_t) -> *const c_char;

// ===================== cuFile =====================

/// cuFile status code (CUfileOpError); 0 = SUCCESS
pub type CUfileOpError = c_int;
pub const CU_FILE_SUCCESS: CUfileOpError = 0;

/// Opaque handle. Real definition in cufile.h:
///   typedef struct CUfileHandle * CUfileHandle_t;
pub type CUfileHandle_t = *mut c_void;

/// `CUfileFileHandleType`: 1 = POSIX FD
pub type CUfileFileHandleType = c_int;
pub const CU_FILE_HANDLE_TYPE_OPAQUE_FD: CUfileFileHandleType = 1;

/// `CUfileDescr_t`. Real ABI:
///   struct CUfileDescr_t {
///       CUfileFileHandleType type;
///       union { int fd; void* handle; } handle;
///       CUfileFSOps_t* fs_ops;  // may be NULL
///   };
/// We only use the POSIX fd form, represented here as a fixed-layout struct.
#[repr(C)]
pub struct CUfileDescr_t {
    pub handle_type: CUfileFileHandleType,
    pub fd: c_int,
    pub _pad: [u8; 4], // align to 8-byte union
    pub fs_ops: *mut c_void,
}

/// `CUfileError_t`. Real struct has err / cu_err fields; we flatten to a single i64.
/// Since libcufile only uses the low 32 bits of err on most return paths, we take only the low 32 bits.
#[repr(C)]
pub struct CUfileError_t {
    pub err: c_int,
    pub cu_err: c_int,
}

pub type CuFileDriverOpen = unsafe extern "C" fn() -> CUfileError_t;
pub type CuFileDriverClose = unsafe extern "C" fn() -> CUfileError_t;

pub type CuFileHandleRegister =
    unsafe extern "C" fn(handle: *mut CUfileHandle_t, descr: *mut CUfileDescr_t) -> CUfileError_t;
pub type CuFileHandleDeregister = unsafe extern "C" fn(handle: CUfileHandle_t);

pub type CuFileBufRegister =
    unsafe extern "C" fn(devPtr: *const c_void, length: usize, flags: c_int) -> CUfileError_t;
pub type CuFileBufDeregister = unsafe extern "C" fn(devPtr: *const c_void) -> CUfileError_t;

/// `ssize_t cuFileRead(CUfileHandle_t fh, void *bufPtr_base,
///                     size_t size, off_t file_offset, off_t bufPtr_offset);`
/// Returns actual bytes (>=0) or a negative error code
pub type CuFileRead = unsafe extern "C" fn(
    fh: CUfileHandle_t,
    buf: *mut c_void,
    size: usize,
    file_offset: i64,
    buf_offset: i64,
) -> isize;

pub type CuFileWrite = unsafe extern "C" fn(
    fh: CUfileHandle_t,
    buf: *const c_void,
    size: usize,
    file_offset: i64,
    buf_offset: i64,
) -> isize;

// ===================== CUDA Driver API (libcuda.so) =====================
// IPC goes through the driver API, not the runtime. We only use the 4 below.

pub type CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

/// CUdeviceptr on 64-bit platforms = u64
pub type CUdeviceptr = u64;

/// `CU_IPC_HANDLE_MAX_SIZE = 64`
pub const CU_IPC_HANDLE_MAX_SIZE: usize = 64;

/// `CUipcMemHandle = struct { char reserved[64]; }`
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUipcMemHandle {
    pub reserved: [u8; CU_IPC_HANDLE_MAX_SIZE],
}

/// `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1`
pub const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: c_int = 1;

pub type CuInit = unsafe extern "C" fn(flags: c_int) -> CUresult;

pub type CuIpcGetMemHandle =
    unsafe extern "C" fn(handle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult;

pub type CuIpcOpenMemHandle = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    handle: CUipcMemHandle,
    flags: c_int,
) -> CUresult;

pub type CuIpcCloseMemHandle = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;

pub type CuGetErrorString =
    unsafe extern "C" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult;

