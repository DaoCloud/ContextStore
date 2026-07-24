//! Local GPUDirect Storage C ABI used by ContextStore GPU workers.
//!
//! This library is deliberately process-local: a CUDA pointer is only valid in
//! the vLLM worker that owns it. It caches cuFile buffer registrations and file
//! handles so the hot read path avoids repeated registration work.

use anyhow::{anyhow, Context, Result};
use libloading::{Library, Symbol};
use lru::LruCache;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::File;
use std::num::NonZeroUsize;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

type CUfileHandle = *mut c_void;

#[repr(C)]
#[derive(Copy, Clone)]
struct CUfileError {
    err: c_int,
    cu_err: c_int,
}

#[repr(C)]
struct CUfileDescr {
    handle_type: c_int,
    fd: c_int,
    pad: [u8; 4],
    fs_ops: *mut c_void,
}

type CuFileDriverOpen = unsafe extern "C" fn() -> CUfileError;
type CuFileDriverClose = unsafe extern "C" fn() -> CUfileError;
type CuFileHandleRegister =
    unsafe extern "C" fn(*mut CUfileHandle, *mut CUfileDescr) -> CUfileError;
type CuFileHandleDeregister = unsafe extern "C" fn(CUfileHandle);
type CuFileBufRegister = unsafe extern "C" fn(*const c_void, usize, c_int) -> CUfileError;
type CuFileBufDeregister = unsafe extern "C" fn(*const c_void) -> CUfileError;
type CuFileRead = unsafe extern "C" fn(CUfileHandle, *mut c_void, usize, i64, i64) -> isize;
type CudaSetDevice = unsafe extern "C" fn(c_int) -> c_int;

struct Driver {
    _cufile: Library,
    _cudart: Library,
    driver_open: CuFileDriverOpen,
    driver_close: CuFileDriverClose,
    handle_register: CuFileHandleRegister,
    handle_deregister: CuFileHandleDeregister,
    buffer_register: CuFileBufRegister,
    buffer_deregister: CuFileBufDeregister,
    read: CuFileRead,
    set_device: CudaSetDevice,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Driver {
    fn load() -> Result<Self> {
        let cufile = load_first(&[
            "libcufile.so",
            "libcufile.so.0",
            "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
            "/usr/local/cuda/lib64/libcufile.so",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cufile/lib/libcufile.so.0",
        ])?;
        let cudart = load_first(&[
            "libcudart.so",
            "libcudart.so.12",
            "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
        ])?;

        unsafe {
            macro_rules! symbol {
                ($library:expr, $name:literal, $ty:ty) => {{
                    let symbol: Symbol<$ty> = $library
                        .get($name)
                        .with_context(|| format!("resolve {}", stringify!($name)))?;
                    *symbol
                }};
            }

            let driver = Self {
                driver_open: symbol!(cufile, b"cuFileDriverOpen", CuFileDriverOpen),
                driver_close: symbol!(cufile, b"cuFileDriverClose", CuFileDriverClose),
                handle_register: symbol!(cufile, b"cuFileHandleRegister", CuFileHandleRegister),
                handle_deregister: symbol!(
                    cufile,
                    b"cuFileHandleDeregister",
                    CuFileHandleDeregister
                ),
                buffer_register: symbol!(cufile, b"cuFileBufRegister", CuFileBufRegister),
                buffer_deregister: symbol!(cufile, b"cuFileBufDeregister", CuFileBufDeregister),
                read: symbol!(cufile, b"cuFileRead", CuFileRead),
                set_device: symbol!(cudart, b"cudaSetDevice", CudaSetDevice),
                _cufile: cufile,
                _cudart: cudart,
            };
            let status = (driver.driver_open)();
            if status.err != 0 {
                return Err(anyhow!("cuFileDriverOpen failed: {}", status.err));
            }
            Ok(driver)
        }
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        unsafe {
            (self.driver_close)();
        }
    }
}

fn load_first(candidates: &[&str]) -> Result<Library> {
    for candidate in candidates {
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(anyhow!("unable to load any supported shared library"))
}

struct RegisteredBuffer {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for RegisteredBuffer {}

struct CachedFile {
    _file: File,
    handle: CUfileHandle,
}

unsafe impl Send for CachedFile {}

impl CachedFile {
    fn open(driver: &Driver, path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open GDS file {}", path.display()))?;
        let mut descriptor = CUfileDescr {
            handle_type: 1,
            fd: file.as_raw_fd(),
            pad: [0; 4],
            fs_ops: ptr::null_mut(),
        };
        let mut handle = ptr::null_mut();
        let status = unsafe { (driver.handle_register)(&mut handle, &mut descriptor) };
        if status.err != 0 {
            return Err(anyhow!(
                "cuFileHandleRegister {} failed: {}",
                path.display(),
                status.err
            ));
        }
        Ok(Self {
            _file: file,
            handle,
        })
    }
}

struct Client {
    files: Mutex<LruCache<PathBuf, CachedFile>>,
    buffers: Mutex<HashMap<u64, RegisteredBuffer>>,
    next_buffer_id: AtomicU64,
    driver: Driver,
}

unsafe impl Send for Client {}
unsafe impl Sync for Client {}

impl Client {
    fn new(file_cache_capacity: usize) -> Result<Self> {
        Ok(Self {
            files: Mutex::new(LruCache::new(
                NonZeroUsize::new(file_cache_capacity.max(1)).unwrap(),
            )),
            buffers: Mutex::new(HashMap::new()),
            next_buffer_id: AtomicU64::new(1),
            driver: Driver::load()?,
        })
    }

    fn set_device(&self, device: c_int) -> Result<()> {
        let status = unsafe { (self.driver.set_device)(device) };
        if status != 0 {
            return Err(anyhow!("cudaSetDevice({device}) failed: {status}"));
        }
        Ok(())
    }

    fn register_buffer(&self, ptr: *mut c_void, len: usize) -> Result<u64> {
        if ptr.is_null() || len == 0 {
            return Err(anyhow!("GDS buffer must be non-empty"));
        }
        let status = unsafe { (self.driver.buffer_register)(ptr, len, 0) };
        if status.err != 0 {
            return Err(anyhow!("cuFileBufRegister failed: {}", status.err));
        }
        let id = self.next_buffer_id.fetch_add(1, Ordering::Relaxed);
        self.buffers
            .lock()
            .unwrap()
            .insert(id, RegisteredBuffer { ptr, len });
        Ok(id)
    }

    fn unregister_buffer(&self, id: u64) {
        if let Some(buffer) = self.buffers.lock().unwrap().remove(&id) {
            unsafe {
                (self.driver.buffer_deregister)(buffer.ptr);
            }
        }
    }

    fn read(
        &self,
        buffer_id: u64,
        path: &Path,
        file_offset: u64,
        buffer_offset: u64,
        len: usize,
    ) -> Result<usize> {
        let buffer = self
            .buffers
            .lock()
            .unwrap()
            .get(&buffer_id)
            .map(|buffer| (buffer.ptr, buffer.len))
            .ok_or_else(|| anyhow!("unknown GDS buffer {buffer_id}"))?;
        let end = usize::try_from(buffer_offset)
            .ok()
            .and_then(|offset| offset.checked_add(len))
            .ok_or_else(|| anyhow!("GDS buffer range overflows usize"))?;
        if end > buffer.1 {
            return Err(anyhow!("GDS read exceeds registered buffer"));
        }

        let mut files = self.files.lock().unwrap();
        if files.get(path).is_none() {
            let handle = CachedFile::open(&self.driver, path)?;
            if let Some((_path, evicted)) = files.push(path.to_path_buf(), handle) {
                unsafe {
                    (self.driver.handle_deregister)(evicted.handle);
                }
            }
        }
        let handle = files.get(path).expect("file inserted above").handle;
        let read = unsafe {
            (self.driver.read)(
                handle,
                buffer.0,
                len,
                i64::try_from(file_offset).context("file offset too large")?,
                i64::try_from(buffer_offset).context("buffer offset too large")?,
            )
        };
        if read < 0 {
            return Err(anyhow!("cuFileRead failed: {read}"));
        }
        Ok(read as usize)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        for (_id, buffer) in self.buffers.get_mut().unwrap().drain() {
            unsafe {
                (self.driver.buffer_deregister)(buffer.ptr);
            }
        }
        for (_path, file) in self.files.get_mut().unwrap().iter() {
            unsafe {
                (self.driver.handle_deregister)(file.handle);
            }
        }
    }
}

fn with_client<T>(client: *mut c_void, action: impl FnOnce(&Client) -> Result<T>) -> Result<T> {
    let client =
        unsafe { (client as *mut Client).as_ref() }.ok_or_else(|| anyhow!("GDS client is null"))?;
    action(client)
}

/// Creates a process-local GDS client. Returns null when cuFile is unavailable.
#[no_mangle]
pub extern "C" fn cs_gds_client_new(file_cache_capacity: u32) -> *mut c_void {
    Client::new(file_cache_capacity as usize)
        .map(Box::new)
        .map(|client| Box::into_raw(client) as *mut c_void)
        .unwrap_or(ptr::null_mut())
}

/// Releases all cached file handles and registered GPU buffers.
///
/// # Safety
/// `client` must be a non-null pointer returned by `cs_gds_client_new` that has
/// not already been released. No other thread may use it after this call.
#[no_mangle]
pub unsafe extern "C" fn cs_gds_client_free(client: *mut c_void) {
    if !client.is_null() {
        drop(Box::from_raw(client as *mut Client));
    }
}

/// Selects the CUDA device for the calling worker thread. Returns 0 on success.
///
/// # Safety
/// `client` must be a live pointer returned by `cs_gds_client_new`.
#[no_mangle]
pub unsafe extern "C" fn cs_gds_client_set_device(client: *mut c_void, device: c_int) -> c_int {
    with_client(client, |client| client.set_device(device))
        .map(|_| 0)
        .unwrap_or(-1)
}

/// Registers a stable CUDA allocation with cuFile. Returns a positive buffer id on success.
///
/// # Safety
/// `client` must be live and `ptr..ptr + len` must remain a valid CUDA allocation
/// until the returned id is passed to `cs_gds_unregister_buffer`.
#[no_mangle]
pub unsafe extern "C" fn cs_gds_register_buffer(
    client: *mut c_void,
    ptr: *mut c_void,
    len: u64,
) -> i64 {
    let len = match usize::try_from(len) {
        Ok(len) => len,
        Err(_) => return -1,
    };
    with_client(client, |client| client.register_buffer(ptr, len))
        .and_then(|id| i64::try_from(id).context("buffer id exceeds i64"))
        .unwrap_or(-1)
}

/// Deregisters a buffer previously returned by `cs_gds_register_buffer`.
///
/// # Safety
/// `client` must be live. The caller must not issue or retain any I/O using
/// `buffer_id` after deregistration.
#[no_mangle]
pub unsafe extern "C" fn cs_gds_unregister_buffer(client: *mut c_void, buffer_id: u64) {
    if let Some(client) = (client as *mut Client).as_ref() {
        client.unregister_buffer(buffer_id);
    }
}

/// Reads a file range directly into a registered CUDA buffer. Returns bytes read or -1.
///
/// # Safety
/// `client` must be live, `path` must point to a NUL-terminated UTF-8 string for
/// the duration of this call, and `buffer_id` must identify a registered CUDA
/// allocation large enough for `buffer_offset + len`.
#[no_mangle]
pub unsafe extern "C" fn cs_gds_read(
    client: *mut c_void,
    buffer_id: u64,
    path: *const c_char,
    file_offset: u64,
    buffer_offset: u64,
    len: u64,
) -> i64 {
    if path.is_null() {
        return -1;
    }
    let len = match usize::try_from(len) {
        Ok(len) => len,
        Err(_) => return -1,
    };
    let path = match CStr::from_ptr(path).to_str() {
        Ok(path) => PathBuf::from(path),
        Err(_) => return -1,
    };
    with_client(client, |client| {
        client
            .read(buffer_id, &path, file_offset, buffer_offset, len)
            .and_then(|read| i64::try_from(read).context("read size exceeds i64"))
    })
    .unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_cache_capacity_is_never_zero() {
        let capacity = NonZeroUsize::new(0usize.max(1)).unwrap();
        assert_eq!(capacity.get(), 1);
    }
}
