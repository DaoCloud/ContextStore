//! GdsDriver — global singleton holding libcufile.so / libcudart.so handles
//!
//! On startup, `init()` probes for library availability:
//! - Locate and dlopen libcufile.so → cuFileDriverOpen
//! - Locate and dlopen libcudart.so → prepare cudaMalloc etc. function pointers
//!
//! If any step fails, `is_available()` returns false and callers should fall back to a non-GDS path.

use super::ffi::*;
use super::map_result;
use crate::error::{KVError, Result};
use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;
use std::os::raw::c_int;
use std::os::raw::c_void;
use std::sync::Mutex;
use tracing::{info, warn};

static DRIVER: OnceCell<Option<GdsDriver>> = OnceCell::new();

/// `Send + Sync`: all fn pointers are passed transparently as *const c_void; calls are thread-safe (per cuFile docs)
pub struct GdsDriver {
    _cufile: Library,
    _cudart: Library,
    _cuda: Library,

    pub cu_file_driver_open: CuFileDriverOpen,
    pub cu_file_driver_close: CuFileDriverClose,
    pub cu_file_handle_register: CuFileHandleRegister,
    pub cu_file_handle_deregister: CuFileHandleDeregister,
    pub cu_file_buf_register: CuFileBufRegister,
    pub cu_file_buf_deregister: CuFileBufDeregister,
    pub cu_file_read: CuFileRead,
    pub cu_file_write: CuFileWrite,

    pub cuda_malloc: CudaMalloc,
    pub cuda_free: CudaFree,
    pub cuda_memcpy: CudaMemcpy,
    pub cuda_set_device: CudaSetDevice,
    pub cuda_device_synchronize: CudaDeviceSynchronize,

    // CUDA Driver API (libcuda.so) for IPC
    pub cu_init: CuInit,
    pub cu_ipc_get_mem_handle: CuIpcGetMemHandle,
    pub cu_ipc_open_mem_handle: CuIpcOpenMemHandle,
    pub cu_ipc_close_mem_handle: CuIpcCloseMemHandle,

    /// driver_open was called successfully; needs driver_close on drop
    open_guard: Mutex<bool>,
}

unsafe impl Send for GdsDriver {}
unsafe impl Sync for GdsDriver {}

const CUFILE_SONAMES: &[&str] = &[
    "libcufile.so",
    "libcufile.so.0",
    "libcufile.so.1",
    "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
    "/usr/local/cuda/lib64/libcufile.so",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cufile/lib/libcufile.so.0",
];

const CUDART_SONAMES: &[&str] = &[
    "libcudart.so",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "/usr/local/cuda/lib64/libcudart.so",
];

const CUDA_SONAMES: &[&str] = &[
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
];

fn try_load(candidates: &[&str]) -> Option<Library> {
    for name in candidates {
        match unsafe { Library::new(name) } {
            Ok(lib) => {
                info!("GDS: loaded {} successfully", name);
                return Some(lib);
            }
            Err(_) => continue,
        }
    }
    None
}

impl GdsDriver {
    /// Probes and initialises the global driver. Idempotent across multiple calls; does not panic on failure.
    /// Returns Ok(true) if available, Ok(false) if not (already logged a warn), Err on load-stage exceptions.
    pub fn init() -> Result<bool> {
        let cell = DRIVER.get_or_init(|| {
            let cufile = match try_load(CUFILE_SONAMES) {
                Some(l) => l,
                None => {
                    warn!("GDS: libcufile.so not found, GDS path disabled (falling back to pread/pwrite + cudaMemcpy)");
                    return None;
                }
            };
            let cudart = match try_load(CUDART_SONAMES) {
                Some(l) => l,
                None => {
                    warn!("GDS: libcudart.so not found, GDS path disabled");
                    return None;
                }
            };
            let cuda = match try_load(CUDA_SONAMES) {
                Some(l) => l,
                None => {
                    warn!("GDS: libcuda.so (driver API) not found, CUDA IPC path disabled");
                    return None;
                }
            };

            match build(cufile, cudart, cuda) {
                Ok(d) => Some(d),
                Err(e) => {
                    warn!("GDS: symbol resolution failed ({}), GDS path disabled", e);
                    None
                }
            }
        });

        match cell {
            Some(d) => {
                d.ensure_driver_open()?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn get() -> Option<&'static GdsDriver> {
        DRIVER.get().and_then(|opt| opt.as_ref())
    }

    fn ensure_driver_open(&self) -> Result<()> {
        let mut g = self.open_guard.lock().unwrap();
        if *g {
            return Ok(());
        }
        // cuInit(0) must be called once before any driver API call (including IPC)
        let cu = unsafe { (self.cu_init)(0) };
        if cu != 0 {
            return Err(super::cuda_err("cuInit", cu));
        }
        let err = unsafe { (self.cu_file_driver_open)() };
        map_result("cuFileDriverOpen", err.err)?;
        info!("GDS: cuInit + cuFileDriverOpen succeeded");
        *g = true;
        Ok(())
    }
}

fn build(cufile: Library, cudart: Library, cuda: Library) -> Result<GdsDriver> {
    unsafe {
        macro_rules! sym {
            ($lib:expr, $name:literal, $t:ty) => {{
                let s: Symbol<$t> = $lib
                    .get($name)
                    .map_err(|e| KVError::Internal(format!("dlsym {}: {}", stringify!($name), e)))?;
                *s.into_raw()
            }};
        }
        Ok(GdsDriver {
            cu_file_driver_open: sym!(cufile, b"cuFileDriverOpen", CuFileDriverOpen),
            cu_file_driver_close: sym!(cufile, b"cuFileDriverClose", CuFileDriverClose),
            cu_file_handle_register: sym!(cufile, b"cuFileHandleRegister", CuFileHandleRegister),
            cu_file_handle_deregister: sym!(cufile, b"cuFileHandleDeregister", CuFileHandleDeregister),
            cu_file_buf_register: sym!(cufile, b"cuFileBufRegister", CuFileBufRegister),
            cu_file_buf_deregister: sym!(cufile, b"cuFileBufDeregister", CuFileBufDeregister),
            cu_file_read: sym!(cufile, b"cuFileRead", CuFileRead),
            cu_file_write: sym!(cufile, b"cuFileWrite", CuFileWrite),

            cuda_malloc: sym!(cudart, b"cudaMalloc", CudaMalloc),
            cuda_free: sym!(cudart, b"cudaFree", CudaFree),
            cuda_memcpy: sym!(cudart, b"cudaMemcpy", CudaMemcpy),
            cuda_set_device: sym!(cudart, b"cudaSetDevice", CudaSetDevice),
            cuda_device_synchronize: sym!(cudart, b"cudaDeviceSynchronize", CudaDeviceSynchronize),

            cu_init: sym!(cuda, b"cuInit", CuInit),
            cu_ipc_get_mem_handle: sym!(cuda, b"cuIpcGetMemHandle", CuIpcGetMemHandle),
            cu_ipc_open_mem_handle: sym!(cuda, b"cuIpcOpenMemHandle_v2", CuIpcOpenMemHandle),
            cu_ipc_close_mem_handle: sym!(cuda, b"cuIpcCloseMemHandle", CuIpcCloseMemHandle),

            _cufile: cufile,
            _cudart: cudart,
            _cuda: cuda,
            open_guard: Mutex::new(false),
        })
    }
}

impl Drop for GdsDriver {
    fn drop(&mut self) {
        let g = self.open_guard.lock().unwrap();
        if *g {
            unsafe {
                (self.cu_file_driver_close)();
            }
        }
    }
}

/// Caller check for whether GDS is actually usable (libcufile loaded + driver_open succeeded)
pub fn is_available() -> bool {
    GdsDriver::get().is_some()
}

/// Sets the current thread's CUDA device (must be called before cudaMalloc)
pub fn set_device(device_ordinal: c_int) -> Result<()> {
    let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not initialised".into()))?;
    let err = unsafe { (d.cuda_set_device)(device_ordinal) };
    if err != 0 {
        return Err(super::cuda_err("cudaSetDevice", err));
    }
    Ok(())
}

/// Internal helper: allocate GPU memory via the CUDA runtime
pub(crate) fn cuda_malloc_raw(size: usize) -> Result<*mut c_void> {
    let d = GdsDriver::get().ok_or_else(|| KVError::Internal("GDS not initialised".into()))?;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let err = unsafe { (d.cuda_malloc)(&mut ptr, size) };
    if err != 0 {
        return Err(super::cuda_err("cudaMalloc", err));
    }
    Ok(ptr)
}

pub(crate) fn cuda_free_raw(ptr: *mut c_void) {
    if let Some(d) = GdsDriver::get() {
        unsafe {
            (d.cuda_free)(ptr);
        }
    }
}
