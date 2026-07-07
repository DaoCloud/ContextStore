//! ContextStore NIXL client C ABI.
//!
//! This cdylib is loaded via `dlopen` by the `CONTEXTSTORE` NIXL backend plugin.
//! When the `rdma` feature is enabled, NIXL's DRAM descriptor pointer is registered
//! as an RDMA external buffer, and the data plane goes through `kv-service/rdma-ffi`:
//!
//! - PUT: client RDMA-WRITEs into the server slab; the server flushes it directly
//!   to NVMe from the slab.
//! - GET: server RDMA-WRITEs into the DRAM buffer supplied by NIXL.
//!
//! When RDMA is not enabled or `rdma_enabled=false`, we retain the gRPC fallback for
//! local smoke tests.

use contextstore_client_rs::KvClient;
use libc::{c_char, c_int, c_void, size_t};
#[cfg(feature = "rdma")]
use std::collections::HashMap;
use std::ffi::CStr;
#[cfg(feature = "rdma")]
use std::ffi::CString;
use std::ptr;
use std::slice;
use std::sync::Mutex;
use tokio::runtime::Runtime;

#[repr(C)]
pub struct CsNixlClientConfig {
    endpoint: *const c_char,
    model_id: *const c_char,
    namespace_name: *const c_char,
    rdma_server_addr: *const c_char,
    rdma_enabled: c_int,
}

enum Transport {
    Grpc {
        rt: Runtime,
        client: Mutex<KvClient>,
    },
    #[cfg(feature = "rdma")]
    Rdma { state: Mutex<RdmaState> },
}

#[cfg(feature = "rdma")]
struct RdmaState {
    client: *mut c_void,
    regions: HashMap<(usize, u64), i32>,
}

struct ClientHandle {
    transport: Transport,
    namespace_name: String,
}

unsafe impl Send for ClientHandle {}
unsafe impl Sync for ClientHandle {}

#[cfg(feature = "rdma")]
unsafe impl Send for RdmaState {}

fn cstr(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn bool_env(name: &str, fallback: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(fallback)
}

#[cfg(feature = "rdma")]
fn env_u8(name: &str, fallback: u8) -> u8 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(fallback)
}

#[cfg(feature = "rdma")]
fn env_u64(name: &str, fallback: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(fallback)
}

fn grpc_endpoint(raw: String) -> String {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw
    } else {
        format!("http://{}", raw)
    }
}

fn kv_object_key(key: &str) -> &str {
    key
}

fn canonical_string_key(namespace: &str, object_key: &str) -> String {
    format!("{}:{}{}", namespace.as_bytes().len(), namespace, object_key)
}

#[cfg(feature = "rdma")]
fn rdma_key(handle: &ClientHandle, key: &str) -> Option<CString> {
    let object_key = kv_object_key(key);
    CString::new(canonical_string_key(&handle.namespace_name, object_key)).ok()
}

fn open_grpc(endpoint: String) -> Result<Transport, c_int> {
    let rt = Runtime::new().map_err(|_| -10)?;
    let client = rt
        .block_on(KvClient::connect(grpc_endpoint(endpoint)))
        .map_err(|_| -11)?;
    Ok(Transport::Grpc {
        rt,
        client: Mutex::new(client),
    })
}

#[cfg(feature = "rdma")]
fn open_rdma(server_addr: String) -> Result<Transport, c_int> {
    let device = std::env::var("CS_NIXL_RDMA_DEVICE").unwrap_or_else(|_| "mlx5_0".to_string());
    let port = env_u8("CS_NIXL_RDMA_PORT", 1);
    let gid_index = env_u8("CS_NIXL_RDMA_GID_INDEX", 3);
    let internal_buffer_mb = env_u64("CS_NIXL_RDMA_INTERNAL_BUFFER_MB", 64);
    let device = CString::new(device).map_err(|_| -20)?;
    let server_addr = CString::new(server_addr).map_err(|_| -21)?;
    let client = unsafe {
        contextstore_rdma_ffi::cs_rdma_client_new(
            device.as_ptr(),
            port,
            gid_index,
            internal_buffer_mb * 1024 * 1024,
        )
    };
    if client.is_null() {
        return Err(-22);
    }
    let rc = unsafe { contextstore_rdma_ffi::cs_rdma_client_connect(client, server_addr.as_ptr()) };
    if rc != 0 {
        unsafe { contextstore_rdma_ffi::cs_rdma_client_free(client) };
        return Err(-23);
    }
    Ok(Transport::Rdma {
        state: Mutex::new(RdmaState {
            client,
            regions: HashMap::new(),
        }),
    })
}

#[cfg(feature = "rdma")]
fn rdma_region_for(state: &mut RdmaState, ptr: *const u8, len: u64) -> Result<i32, c_int> {
    let key = (ptr as usize, len);
    if let Some(region_id) = state.regions.get(&key) {
        return Ok(*region_id);
    }
    let region_id = unsafe {
        contextstore_rdma_ffi::cs_rdma_client_register_external_buffer(state.client, ptr, len)
    };
    if region_id < 0 {
        return Err(-6);
    }
    state.regions.insert(key, region_id);
    Ok(region_id)
}

fn open_transport(cfg: &CsNixlClientConfig) -> Result<Transport, c_int> {
    if cfg.rdma_enabled != 0 {
        #[cfg(feature = "rdma")]
        {
            let addr = cstr(cfg.rdma_server_addr);
            if !addr.is_empty() {
                match open_rdma(addr) {
                    Ok(t) => return Ok(t),
                    Err(e) if bool_env("CS_NIXL_RDMA_FALLBACK_GRPC", false) => {
                        eprintln!(
                            "[contextstore-nixl-client] RDMA open failed rc={}, falling back to gRPC",
                            e
                        );
                    }
                    Err(e) => return Err(e),
                }
            } else if !bool_env("CS_NIXL_RDMA_FALLBACK_GRPC", false) {
                return Err(-24);
            }
        }
        #[cfg(not(feature = "rdma"))]
        {
            if !bool_env("CS_NIXL_RDMA_FALLBACK_GRPC", false) {
                return Err(-25);
            }
        }
    }
    open_grpc(cstr(cfg.endpoint))
}

#[no_mangle]
pub extern "C" fn cs_nixl_client_open(
    config: *const CsNixlClientConfig,
    out_client: *mut *mut c_void,
) -> c_int {
    if config.is_null() || out_client.is_null() {
        return -1;
    }
    let cfg = unsafe { &*config };
    let model_id = cstr(cfg.model_id);
    let namespace_name = cstr(cfg.namespace_name);
    let namespace_name = if namespace_name.is_empty() {
        if model_id.is_empty() {
            "nixl".to_string()
        } else {
            model_id
        }
    } else {
        namespace_name
    };
    let transport = match open_transport(cfg) {
        Ok(t) => t,
        Err(rc) => return rc,
    };
    let handle = Box::new(ClientHandle {
        transport,
        namespace_name,
    });
    unsafe {
        *out_client = Box::into_raw(handle) as *mut c_void;
    }
    0
}

#[no_mangle]
pub extern "C" fn cs_nixl_client_close(client: *mut c_void) {
    if client.is_null() {
        return;
    }
    let handle = unsafe { Box::from_raw(client as *mut ClientHandle) };
    match handle.transport {
        #[cfg(feature = "rdma")]
        Transport::Rdma { state } => {
            if let Ok(mut state) = state.lock() {
                state.regions.clear();
                unsafe {
                    contextstore_rdma_ffi::cs_rdma_client_free(state.client);
                }
                state.client = ptr::null_mut();
            }
        }
        Transport::Grpc { .. } => {}
    }
}

#[no_mangle]
pub extern "C" fn cs_nixl_client_put(
    client: *mut c_void,
    key: *const c_char,
    data: *const c_void,
    len: size_t,
    offset: u64,
) -> c_int {
    if client.is_null() || key.is_null() || (data.is_null() && len != 0) {
        return -1;
    }
    let handle = unsafe { &*(client as *mut ClientHandle) };
    let key = cstr(key);
    match &handle.transport {
        Transport::Grpc { rt, client } => {
            if offset != 0 {
                return -2;
            }
            let object_key = kv_object_key(&key);
            let bytes = if len == 0 {
                Vec::new()
            } else {
                unsafe { slice::from_raw_parts(data as *const u8, len as usize) }.to_vec()
            };
            let mut guard = match client.lock() {
                Ok(g) => g,
                Err(_) => return -3,
            };
            match rt.block_on(guard.put(&handle.namespace_name, object_key, bytes)) {
                Ok(true) => 0,
                _ => -4,
            }
        }
        #[cfg(feature = "rdma")]
        Transport::Rdma { state } => {
            if offset != 0 {
                return -2;
            }
            let Some(key) = rdma_key(handle, &key) else {
                return -5;
            };
            let mut state = match state.lock() {
                Ok(s) => s,
                Err(_) => return -3,
            };
            let region_id = match rdma_region_for(&mut state, data as *const u8, len as u64) {
                Ok(id) => id,
                Err(rc) => return rc,
            };
            let rc = unsafe {
                contextstore_rdma_ffi::cs_rdma_client_put(
                    state.client,
                    region_id,
                    key.as_ptr(),
                    0,
                    len as u64,
                )
            };
            if rc == 0 {
                0
            } else {
                -7
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nixl_obj_key_is_passed_through() {
        let raw = "prefix_96|__combined__";
        assert_eq!(kv_object_key(raw), raw);
    }

    #[test]
    fn canonical_key_uses_namespace_byte_length() {
        assert_eq!(
            canonical_string_key("ns", "prefix_96|__combined__"),
            "2:nsprefix_96|__combined__"
        );
        assert_eq!(canonical_string_key("tenant", "obj"), "6:tenantobj");
    }
}

#[no_mangle]
pub extern "C" fn cs_nixl_client_get(
    client: *mut c_void,
    key: *const c_char,
    data: *mut c_void,
    len: size_t,
    offset: u64,
) -> c_int {
    if client.is_null() || key.is_null() || (data.is_null() && len != 0) {
        return -1;
    }
    let handle = unsafe { &*(client as *mut ClientHandle) };
    let key = cstr(key);
    match &handle.transport {
        Transport::Grpc { rt, client } => {
            if offset != 0 {
                return -2;
            }
            let object_key = kv_object_key(&key);
            let mut guard = match client.lock() {
                Ok(g) => g,
                Err(_) => return -3,
            };
            let value = match rt.block_on(guard.get(&handle.namespace_name, object_key)) {
                Ok(Some(value)) => value,
                Ok(None) => return -4,
                Err(_) => return -5,
            };
            if value.len() != len as usize {
                return -6;
            }
            if len != 0 {
                unsafe {
                    ptr::copy_nonoverlapping(value.as_ptr(), data as *mut u8, len as usize);
                }
            }
            0
        }
        #[cfg(feature = "rdma")]
        Transport::Rdma { state } => {
            if offset != 0 {
                return -2;
            }
            let Some(key) = rdma_key(handle, &key) else {
                return -7;
            };
            let mut state = match state.lock() {
                Ok(s) => s,
                Err(_) => return -3,
            };
            let region_id = match rdma_region_for(&mut state, data as *const u8, len as u64) {
                Ok(id) => id,
                Err(_) => return -8,
            };
            let n = unsafe {
                contextstore_rdma_ffi::cs_rdma_client_get_into(
                    state.client,
                    region_id,
                    key.as_ptr(),
                    0,
                )
            };
            if n == len as i64 {
                0
            } else if n == 0 {
                -9
            } else {
                -10
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn cs_nixl_client_exists(
    client: *mut c_void,
    key: *const c_char,
    size: *mut u64,
    found: *mut c_int,
) -> c_int {
    if client.is_null() || key.is_null() || size.is_null() || found.is_null() {
        return -1;
    }
    let handle = unsafe { &*(client as *mut ClientHandle) };
    let key = cstr(key);
    match &handle.transport {
        Transport::Grpc { rt, client } => {
            let object_key = kv_object_key(&key);
            let mut guard = match client.lock() {
                Ok(g) => g,
                Err(_) => return -2,
            };
            match rt.block_on(guard.exists(&handle.namespace_name, object_key)) {
                Ok(exists) => unsafe {
                    *size = 0;
                    *found = if exists { 1 } else { 0 };
                    0
                },
                Err(_) => -3,
            }
        }
        #[cfg(feature = "rdma")]
        Transport::Rdma { .. } => {
            // The RDMA wire protocol currently has no metadata-only exists. Return
            // unknown; the caller can just READ.
            unsafe {
                *size = 0;
                *found = 0;
            }
            0
        }
    }
}
