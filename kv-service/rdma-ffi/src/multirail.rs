//! Multi-rail C ABI: exposes `contextstore_client_rs::multirail::MultiRailClient`
//! to Python ctypes, so a Python KVConnector Worker can read one striped
//! object in parallel over several local RDMA devices.
//!
//! ```c
//! // Create a reader over N rails. rail_specs[i] = "device[:port[:gid[:weight[:mtu]]]]".
//! // io_timeout_ms bounds each operation (and connect). NULL on error.
//! void* cs_mr_new(const char* const* rail_specs, uint32_t rail_count, uint64_t io_timeout_ms);
//!
//! // Object identity + placement, mirroring the gRPC ObjectDescriptor /
//! // PlacementChunk the Python side obtained from LookupObject.
//! typedef struct {
//!     const char* namespace;
//!     const char* object_key;
//!     const char* object_handle;
//!     uint64_t    object_generation;
//!     const char* content_etag;
//!     uint64_t    layout_version;
//!     uint64_t    size;
//!     uint32_t    is_striped;
//!     uint32_t    stripe_count;
//!     uint64_t    chunk_size;
//! } CsMrDescriptor;
//! typedef struct {
//!     uint32_t    stripe_index;
//!     const char* rdma_endpoint;
//!     uint64_t    offset;
//!     uint64_t    length;
//!     const char* checksum;   // optional xxh3-64 lowercase hex; NULL/"" skips
//! } CsMrChunk;
//!
//! // Read the whole object into buffer (server RDMA-WRITEs each stripe at
//! // buffer + stripe_index * chunk_size). sticky=1 keeps per-rail registrations
//! // cached across calls — the buffer must then be a long-lived pinned pool
//! // region that outlives the reader and is never freed/reused otherwise.
//! // Returns bytes read (>=0) or -1; human-readable error text in err_buf.
//! int64_t cs_mr_read(void* reader,
//!                    const CsMrDescriptor* descriptor,
//!                    const CsMrChunk* chunks, uint32_t chunk_count,
//!                    uint8_t* buffer, uint64_t buffer_len,
//!                    int32_t sticky,
//!                    char* err_buf, uint32_t err_buf_len);
//!
//! // Per-rail stats snapshot; returns the number of rails written to `out`.
//! typedef struct {
//!     uint32_t index, healthy, cooldown_ms;
//!     uint64_t requests_ok, requests_err, bytes_read, timeouts;
//!     uint64_t connections_created, connections_quiesced;
//!     uint64_t inflight_requests, inflight_bytes;
//!     char     device[64];
//!     char     topology[160];
//! } CsMrRailStats;
//! int32_t cs_mr_rail_stats(void* reader, CsMrRailStats* out, uint32_t max);
//!
//! void cs_mr_free(void* reader);
//! ```

use contextstore_client_rs::multirail::{
    MultiRailClient, MultiRailError, RailConfig, RailLimits, RailSelectPolicy,
};
use contextstore_client_rs::pb;
use std::ffi::{c_char, CStr};
use std::os::raw::{c_int, c_void};
use std::time::Duration;

/// Opaque reader handle body.
pub struct MrReader {
    client: MultiRailClient,
}

#[repr(C)]
pub struct CsMrDescriptor {
    pub namespace: *const c_char,
    pub object_key: *const c_char,
    pub object_handle: *const c_char,
    pub object_generation: u64,
    pub content_etag: *const c_char,
    pub layout_version: u64,
    pub size: u64,
    pub is_striped: u32,
    pub stripe_count: u32,
    pub chunk_size: u64,
}

#[repr(C)]
pub struct CsMrChunk {
    pub stripe_index: u32,
    pub rdma_endpoint: *const c_char,
    pub offset: u64,
    pub length: u64,
    pub checksum: *const c_char,
}

#[repr(C)]
pub struct CsMrRailStats {
    pub index: u32,
    pub healthy: u32,
    pub cooldown_ms: u32,
    pub requests_ok: u64,
    pub requests_err: u64,
    pub bytes_read: u64,
    pub timeouts: u64,
    pub connections_created: u64,
    pub connections_quiesced: u64,
    pub inflight_requests: u64,
    pub inflight_bytes: u64,
    pub latency_avg_us: u64,
    pub latency_max_us: u64,
    pub registered_bytes: u64,
    pub device: [c_char; 64],
    pub topology: [c_char; 160],
}

fn write_err(err_buf: *mut c_char, err_buf_len: u32, message: &str) {
    if err_buf.is_null() || err_buf_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let capacity = err_buf_len as usize - 1;
    let len = bytes.len().min(capacity);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), err_buf as *mut u8, len);
        *err_buf.add(len) = 0;
    }
}

unsafe fn cstr<'a>(ptr: *const c_char) -> Result<&'a str, MultiRailError> {
    if ptr.is_null() {
        return Err(MultiRailError::InvalidPlacement("null string".into()));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| MultiRailError::InvalidPlacement("non-utf8 string in descriptor".into()))
}

#[no_mangle]
pub unsafe extern "C" fn cs_mr_new(
    rail_specs: *const *const c_char,
    rail_count: u32,
    io_timeout_ms: u64,
) -> *mut c_void {
    let mut rails = Vec::with_capacity(rail_count as usize);
    for index in 0..rail_count as usize {
        let spec = match rail_specs
            .add(index)
            .read()
            .as_ref()
            .and_then(|p| CStr::from_ptr(p).to_str().ok())
        {
            Some(spec) => spec,
            None => return std::ptr::null_mut(),
        };
        match RailConfig::parse(spec) {
            Some(config) => rails.push(config),
            None => return std::ptr::null_mut(),
        }
    }
    let limits = RailLimits {
        io_timeout: Duration::from_millis(io_timeout_ms.max(1)),
        ..RailLimits::default()
    };
    match MultiRailClient::new(rails) {
        Ok(client) => Box::into_raw(Box::new(MrReader {
            client: client
                .with_limits(limits)
                .with_policy(RailSelectPolicy::LeastLoaded),
        })) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

fn to_placement(
    descriptor: &CsMrDescriptor,
    chunks: &[CsMrChunk],
) -> Result<(pb::ObjectDescriptor, Vec<pb::PlacementChunk>), MultiRailError> {
    unsafe {
        let descriptor = pb::ObjectDescriptor {
            key: Some(pb::ObjectKey {
                namespace: cstr(descriptor.namespace)?.to_string(),
                object_key: cstr(descriptor.object_key)?.to_string(),
            }),
            object_handle: cstr(descriptor.object_handle)?.to_string(),
            object_generation: descriptor.object_generation,
            content_etag: cstr(descriptor.content_etag)?.to_string(),
            layout_version: descriptor.layout_version,
            size: descriptor.size,
            is_striped: descriptor.is_striped != 0,
            stripe_count: descriptor.stripe_count,
            chunk_size: descriptor.chunk_size,
        };
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            out.push(pb::PlacementChunk {
                stripe_index: chunk.stripe_index,
                node_id: String::new(),
                grpc_endpoint: String::new(),
                rdma_endpoint: cstr(chunk.rdma_endpoint)?.to_string(),
                device_id: 0,
                storage_handle: String::new(),
                offset: chunk.offset,
                length: chunk.length,
                checksum: if chunk.checksum.is_null() {
                    String::new()
                } else {
                    cstr(chunk.checksum)?.to_string()
                },
            });
        }
        Ok((descriptor, out))
    }
}

#[no_mangle]
pub unsafe extern "C" fn cs_mr_read(
    reader: *mut c_void,
    descriptor: *const CsMrDescriptor,
    chunks: *const CsMrChunk,
    chunk_count: u32,
    buffer: *mut u8,
    buffer_len: u64,
    sticky: c_int,
    err_buf: *mut c_char,
    err_buf_len: u32,
) -> i64 {
    let reader = match (reader as *mut MrReader).as_mut() {
        Some(reader) => reader,
        None => {
            write_err(err_buf, err_buf_len, "cs_mr_read: null reader");
            return -1;
        }
    };
    let descriptor = match descriptor.as_ref() {
        Some(descriptor) => descriptor,
        None => {
            write_err(err_buf, err_buf_len, "cs_mr_read: null descriptor");
            return -1;
        }
    };
    if chunks.is_null() || chunk_count == 0 || buffer.is_null() {
        write_err(err_buf, err_buf_len, "cs_mr_read: null chunks/buffer");
        return -1;
    }
    let chunk_slice = std::slice::from_raw_parts(chunks, chunk_count as usize);
    let (pb_descriptor, pb_chunks) = match to_placement(descriptor, chunk_slice) {
        Ok(value) => value,
        Err(error) => {
            write_err(err_buf, err_buf_len, &error.to_string());
            return -1;
        }
    };
    let result = if sticky != 0 {
        reader.client.read_object_into_raw(
            &pb_descriptor,
            &pb_chunks,
            buffer,
            buffer_len as usize,
            true,
        )
    } else {
        // SAFETY: the caller guarantees buffer..buffer+len stays valid and
        // unmoved for the duration of the call (it is synchronous).
        reader.client.read_object_into_raw(
            &pb_descriptor,
            &pb_chunks,
            buffer,
            buffer_len as usize,
            false,
        )
    };
    match result {
        Ok(bytes) => bytes as i64,
        Err(error) => {
            write_err(err_buf, err_buf_len, &error.to_string());
            -1
        }
    }
}

/// Copy a Rust string into a fixed-size C buffer (truncating, always
/// NUL-terminated).
fn fill_str_field(dst: &mut [c_char], value: &str) {
    dst[0] = 0;
    let bytes = value.as_bytes();
    let len = bytes.len().min(dst.len() - 1);
    for (index, byte) in bytes[..len].iter().enumerate() {
        dst[index] = *byte as c_char;
    }
    dst[len] = 0;
}

#[no_mangle]
pub unsafe extern "C" fn cs_mr_rail_stats(
    reader: *mut c_void,
    out: *mut CsMrRailStats,
    max: u32,
) -> i32 {
    let reader = match (reader as *mut MrReader).as_ref() {
        Some(reader) => reader,
        None => return -1,
    };
    if out.is_null() || max == 0 {
        return -1;
    }
    let snapshots = reader.client.rails_snapshot();
    let count = snapshots.len().min(max as usize);
    for (index, snapshot) in snapshots[..count].iter().enumerate() {
        let dst = out.add(index);
        *dst = CsMrRailStats {
            index: snapshot.index as u32,
            healthy: u32::from(snapshot.healthy),
            cooldown_ms: snapshot.cooldown_ms_remaining as u32,
            requests_ok: snapshot.requests_ok,
            requests_err: snapshot.requests_err,
            bytes_read: snapshot.bytes_read,
            timeouts: snapshot.timeouts,
            connections_created: snapshot.connections_created,
            connections_quiesced: snapshot.connections_quiesced,
            inflight_requests: snapshot.inflight_requests,
            inflight_bytes: snapshot.inflight_bytes,
            latency_avg_us: snapshot.latency_avg_us,
            latency_max_us: snapshot.latency_max_us,
            registered_bytes: snapshot.registered_bytes,
            device: [0; 64],
            topology: [0; 160],
        };
        fill_str_field(&mut (*dst).device, &snapshot.device);
        fill_str_field(&mut (*dst).topology, &snapshot.topology.to_string());
    }
    count as i32
}

#[no_mangle]
pub unsafe extern "C" fn cs_mr_free(reader: *mut c_void) {
    if !reader.is_null() {
        drop(Box::from_raw(reader as *mut MrReader));
    }
}
