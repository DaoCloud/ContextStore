//! Native RDMA data-plane client for ContextStore KVService.
//!
//! The control plane is a small TCP protocol and the data plane uses an RC
//! queue pair with RDMA WRITE operations. Reads are server-initiated WRITEs
//! into a caller-owned, registered host buffer; writes copy a registered
//! caller buffer into a server-owned slab and commit it over the control
//! connection. This is deliberately synchronous: callers that run on an async
//! runtime should invoke it from a dedicated blocking worker.
//!
//! `RegisteredBuffer` carries the borrow of the caller's buffer, so the memory
//! cannot be released while its memory region is registered with the NIC.

use crate::pb;
use anyhow::{anyhow, Context, Result};
use rdma_sys::*;
use std::ffi::{c_void, CStr};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::net::TcpStream;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MSG_HELLO: u8 = 1;
const MSG_GET_REQ: u8 = 2;
const MSG_GET_RESP: u8 = 3;
const MSG_PUT_REQ: u8 = 4;
const MSG_PUT_READY: u8 = 5;
const MSG_PUT_COMMIT: u8 = 6;
const MSG_PUT_RESP: u8 = 7;
const MSG_GET_DESCRIPTOR_REQ: u8 = 8;
const MSG_PUT_IF_ABSENT_REQ: u8 = 9;
const MSG_BYE: u8 = 99;
const PUT_RESULT_FAILED: u8 = 0;
const PUT_RESULT_STORED: u8 = 1;
const PUT_RESULT_EXISTS: u8 = 2;
const CQ_DEPTH: i32 = 128;
const MAX_WRITE_BYTES: u64 = 1 << 30;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(30);

/// RDMA connection parameters for a ContextStore data node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RdmaClientConfig {
    /// TCP endpoint of the ContextStore RDMA listener, e.g. `10.0.0.10:50053`.
    pub endpoint: String,
    /// Verbs device name, e.g. `mlx5_0`.
    pub device: String,
    /// HCA port number.
    pub port: u8,
    /// GID index used to construct the RoCE address handle.
    pub gid_index: u8,
}

impl RdmaClientConfig {
    /// Build a configuration with the common RoCE v2 defaults.
    pub fn new(endpoint: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            device: device.into(),
            port: 1,
            gid_index: 3,
        }
    }

    /// Set the HCA port number.
    pub fn with_port(mut self, port: u8) -> Self {
        self.port = port;
        self
    }

    /// Set the GID index.
    pub fn with_gid_index(mut self, gid_index: u8) -> Self {
        self.gid_index = gid_index;
        self
    }
}

/// Result of an RDMA read. `None` means the object did not exist.
pub type RdmaReadResult = Option<usize>;

/// A connected ContextStore RDMA client.
///
/// Every method takes `&mut self` because a connection serializes its TCP
/// control messages. Create one connection per concurrent transfer worker.
pub struct RdmaClient {
    resources: Arc<RdmaResources>,
    qp: NonNull<ibv_qp>,
    stream: TcpStream,
}

// SAFETY: A client is exclusively accessed through `&mut self`; libibverbs
// QPs and their owning resources may be used from a different thread, and
// TcpStream is Send. This lets callers pool established QPs across workers.
unsafe impl Send for RdmaClient {}

/// A registered, caller-owned host buffer.
///
/// The buffer must remain registered for the full RDMA operation. The lifetime
/// parameter and the private marker enforce that requirement for the safe API.
pub struct RegisteredBuffer<'a> {
    // Keeps the PD/CQ/device alive until the MR is deregistered in Drop.
    _resources: Arc<RdmaResources>,
    mr: NonNull<ibv_mr>,
    ptr: NonNull<u8>,
    len: usize,
    _buffer: PhantomData<&'a mut [u8]>,
}

struct RdmaResources {
    context: NonNull<ibv_context>,
    pd: NonNull<ibv_pd>,
    cq: NonNull<ibv_cq>,
    local_gid: ibv_gid,
}

unsafe impl Send for RdmaResources {}
unsafe impl Sync for RdmaResources {}

#[derive(Clone, Copy)]
struct QpInfo {
    qpn: u32,
    psn: u32,
    gid: ibv_gid,
}

impl QpInfo {
    fn to_bytes(self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[0..4].copy_from_slice(&self.qpn.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.psn.to_le_bytes());
        unsafe {
            bytes[8..24].copy_from_slice(&self.gid.raw[..]);
        }
        bytes
    }

    fn from_bytes(bytes: &[u8; 24]) -> Self {
        let qpn = u32::from_le_bytes(bytes[0..4].try_into().expect("fixed qpn length"));
        let psn = u32::from_le_bytes(bytes[4..8].try_into().expect("fixed psn length"));
        let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
        unsafe {
            gid.raw[..].copy_from_slice(&bytes[8..24]);
        }
        Self { qpn, psn, gid }
    }
}

impl RdmaResources {
    fn open(config: &RdmaClientConfig) -> Result<Self> {
        unsafe {
            let mut count = 0i32;
            let devices = ibv_get_device_list(&mut count);
            if devices.is_null() || count == 0 {
                return Err(anyhow!("no RDMA device found"));
            }

            let mut selected = ptr::null_mut();
            for index in 0..count {
                let device = *devices.offset(index as isize);
                let name = CStr::from_ptr(ibv_get_device_name(device)).to_string_lossy();
                if name == config.device {
                    selected = device;
                    break;
                }
            }
            if selected.is_null() {
                ibv_free_device_list(devices);
                return Err(anyhow!("RDMA device '{}' not found", config.device));
            }

            let context = NonNull::new(ibv_open_device(selected))
                .ok_or_else(|| anyhow!("ibv_open_device failed"))?;
            ibv_free_device_list(devices);

            let pd = match NonNull::new(ibv_alloc_pd(context.as_ptr())) {
                Some(pd) => pd,
                None => {
                    ibv_close_device(context.as_ptr());
                    return Err(anyhow!("ibv_alloc_pd failed"));
                }
            };
            let cq = match NonNull::new(ibv_create_cq(
                context.as_ptr(),
                CQ_DEPTH,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            )) {
                Some(cq) => cq,
                None => {
                    ibv_dealloc_pd(pd.as_ptr());
                    ibv_close_device(context.as_ptr());
                    return Err(anyhow!("ibv_create_cq failed"));
                }
            };

            let mut local_gid: ibv_gid = std::mem::zeroed();
            let rc = ibv_query_gid(
                context.as_ptr(),
                config.port,
                config.gid_index as i32,
                &mut local_gid,
            );
            if rc != 0 {
                ibv_destroy_cq(cq.as_ptr());
                ibv_dealloc_pd(pd.as_ptr());
                ibv_close_device(context.as_ptr());
                return Err(anyhow!("ibv_query_gid failed: rc={rc}"));
            }

            Ok(Self {
                context,
                pd,
                cq,
                local_gid,
            })
        }
    }
}

impl Drop for RdmaResources {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_cq(self.cq.as_ptr());
            ibv_dealloc_pd(self.pd.as_ptr());
            ibv_close_device(self.context.as_ptr());
        }
    }
}

impl RdmaClient {
    /// Open the configured HCA, connect the TCP control channel, and establish
    /// an RC queue pair with the ContextStore server.
    pub fn connect(config: RdmaClientConfig) -> Result<Self> {
        let resources = Arc::new(RdmaResources::open(&config)?);
        let qp = create_qp(&resources)?;
        if let Err(error) = transition_qp_to_init(qp, config.port) {
            unsafe { ibv_destroy_qp(qp.as_ptr()) };
            return Err(error);
        }

        let result = (|| -> Result<TcpStream> {
            let mut stream = TcpStream::connect(&config.endpoint)
                .with_context(|| format!("connect RDMA control endpoint {}", config.endpoint))?;
            let local = QpInfo {
                qpn: unsafe { (*qp.as_ptr()).qp_num },
                psn: random_psn(),
                gid: resources.local_gid,
            };
            write_hello(&mut stream, local)?;
            let remote = read_hello(&mut stream)?;
            transition_qp_to_rtr(qp, &remote, config.port, config.gid_index)?;
            transition_qp_to_rts(qp, local.psn)?;
            Ok(stream)
        })();

        match result {
            Ok(stream) => Ok(Self {
                resources,
                qp,
                stream,
            }),
            Err(error) => {
                unsafe { ibv_destroy_qp(qp.as_ptr()) };
                Err(error)
            }
        }
    }

    /// Register a mutable host buffer for use as an RDMA read target or write
    /// source. Registration may pin pageable memory, so callers should reuse
    /// buffers and prefer pre-pinned allocations for the data path.
    pub fn register_buffer<'a>(&self, buffer: &'a mut [u8]) -> Result<RegisteredBuffer<'a>> {
        let ptr = NonNull::new(buffer.as_mut_ptr())
            .ok_or_else(|| anyhow!("cannot register an empty RDMA buffer"))?;
        self.register_parts(ptr, buffer.len())
    }

    /// Register caller-owned memory whose lifetime cannot be expressed as a
    /// Rust slice, such as a buffer received through an FFI boundary.
    ///
    /// # Safety
    /// `ptr..ptr + len` must be valid writable memory and must remain alive,
    /// unmoved, and exclusively available to the RDMA operation until the
    /// returned [`RegisteredBuffer`] is dropped. The caller must not release
    /// or reuse the memory while the NIC can still access it.
    pub unsafe fn register_raw_buffer(
        &self,
        ptr: *mut u8,
        len: usize,
    ) -> Result<RegisteredBuffer<'static>> {
        let ptr = NonNull::new(ptr).ok_or_else(|| anyhow!("cannot register a null RDMA buffer"))?;
        self.register_parts(ptr, len)
    }

    fn register_parts<'a>(&self, ptr: NonNull<u8>, len: usize) -> Result<RegisteredBuffer<'a>> {
        if len == 0 {
            return Err(anyhow!("cannot register an empty RDMA buffer"));
        }
        let mr = unsafe {
            NonNull::new(ibv_reg_mr(
                self.resources.pd.as_ptr(),
                ptr.as_ptr() as *mut c_void,
                len,
                access_flags(),
            ))
        }
        .ok_or_else(|| {
            anyhow!(
                "ibv_reg_mr for {} bytes failed: {}",
                len,
                std::io::Error::last_os_error()
            )
        })?;
        Ok(RegisteredBuffer {
            _resources: Arc::clone(&self.resources),
            mr,
            ptr,
            len,
            _buffer: PhantomData,
        })
    }

    /// Read an object directly into `buffer[offset..]`. `Ok(None)` means the
    /// object is absent; `Ok(Some(bytes))` means that many bytes were written.
    pub fn get_into(
        &mut self,
        namespace: &str,
        object_key: &str,
        buffer: &RegisteredBuffer<'_>,
        offset: usize,
    ) -> Result<RdmaReadResult> {
        self.get_key_into(&canonical_key(namespace, object_key), buffer, offset)
    }

    /// Read a validated descriptor directly into `buffer[offset..]`. Use a
    /// descriptor from `KvClient::lookup_object`; a stale descriptor is
    /// rejected by the server rather than returning unrelated bytes.
    pub fn get_descriptor_into(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        buffer: &RegisteredBuffer<'_>,
        offset: usize,
    ) -> Result<RdmaReadResult> {
        let key = descriptor
            .key
            .as_ref()
            .ok_or_else(|| anyhow!("RDMA descriptor is missing its object key"))?;
        let (dst_addr, rkey, available) = buffer.destination(offset)?;
        let request = build_descriptor_get_request(
            &canonical_key(&key.namespace, &key.object_key),
            descriptor,
            dst_addr,
            rkey,
            available as u64,
        )?;
        self.stream.write_all(&request)?;
        self.stream.flush()?;
        read_get_response(&mut self.stream)
    }

    /// Write `buffer[offset..offset + size]` through the RDMA PUT data path.
    /// The operation only completes after the server confirms that it has
    /// persisted the slab contents to its storage tier.
    pub fn put_from(
        &mut self,
        namespace: &str,
        object_key: &str,
        buffer: &RegisteredBuffer<'_>,
        offset: usize,
        size: usize,
    ) -> Result<()> {
        if !self.put_key_from(
            &canonical_key(namespace, object_key),
            buffer,
            offset,
            size,
            false,
        )? {
            return Err(anyhow!("server rejected RDMA PUT"));
        }
        Ok(())
    }

    /// Immutably write `buffer[offset..offset + size]`. Returns `Ok(true)`
    /// when this client stored the object and `Ok(false)` when it already
    /// existed. The server uses the same metadata `SET NX` contract as the
    /// gRPC `put_if_absent` API.
    pub fn put_if_absent_from(
        &mut self,
        namespace: &str,
        object_key: &str,
        buffer: &RegisteredBuffer<'_>,
        offset: usize,
        size: usize,
    ) -> Result<bool> {
        self.put_key_from(
            &canonical_key(namespace, object_key),
            buffer,
            offset,
            size,
            true,
        )
    }

    fn put_key_from(
        &mut self,
        key: &str,
        buffer: &RegisteredBuffer<'_>,
        offset: usize,
        size: usize,
        if_not_exists: bool,
    ) -> Result<bool> {
        let (src_addr, lkey) = buffer.source(offset, size)?;
        let message = if if_not_exists {
            MSG_PUT_IF_ABSENT_REQ
        } else {
            MSG_PUT_REQ
        };
        let request = build_put_request(message, key, size as u64)?;
        self.stream.write_all(&request)?;
        self.stream.flush()?;

        let ready = read_put_ready(&mut self.stream)?;
        if !ready.ok {
            let result = read_put_result(&mut self.stream)?;
            if if_not_exists && result == PUT_RESULT_EXISTS {
                return Ok(false);
            }
            return Err(anyhow!("server rejected RDMA PUT before data transfer"));
        }

        self.write_remote(src_addr, lkey, ready.addr, ready.rkey, size)?;
        self.stream.write_all(&[MSG_PUT_COMMIT])?;
        self.stream.flush()?;
        match read_put_result(&mut self.stream)? {
            PUT_RESULT_STORED => Ok(true),
            PUT_RESULT_EXISTS if if_not_exists => Ok(false),
            PUT_RESULT_FAILED => Err(anyhow!("server failed to persist RDMA PUT")),
            result => Err(anyhow!("invalid RDMA PUT result code {result}")),
        }
    }

    fn get_key_into(
        &mut self,
        key: &str,
        buffer: &RegisteredBuffer<'_>,
        offset: usize,
    ) -> Result<RdmaReadResult> {
        let (dst_addr, rkey, available) = buffer.destination(offset)?;
        let request = build_get_request(key, dst_addr, rkey, available as u64)?;
        self.stream.write_all(&request)?;
        self.stream.flush()?;
        read_get_response(&mut self.stream)
    }

    fn write_remote(
        &self,
        src_addr: u64,
        src_lkey: u32,
        dst_addr: u64,
        dst_rkey: u32,
        size: usize,
    ) -> Result<()> {
        let total = size as u64;
        let mut transferred = 0u64;
        let mut index = 0u64;
        let writes = total.div_ceil(MAX_WRITE_BYTES);
        while transferred < total {
            let len = (total - transferred).min(MAX_WRITE_BYTES);
            let write = RemoteWrite {
                id: index,
                src_addr: src_addr + transferred,
                src_lkey,
                dst_addr: dst_addr + transferred,
                dst_rkey,
                len: len as u32,
                signaled: index + 1 == writes,
            };
            post_write(self.qp, &write)?;
            transferred += len;
            index += 1;
        }
        poll_completion(self.resources.cq)
    }
}

impl Drop for RdmaClient {
    fn drop(&mut self) {
        let _ = self.stream.write_all(&[MSG_BYE]);
        let _ = self.stream.flush();
        unsafe {
            ibv_destroy_qp(self.qp.as_ptr());
        }
    }
}

impl<'a> RegisteredBuffer<'a> {
    /// Buffer capacity in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this registered buffer has no capacity.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the caller-owned buffer after an RDMA operation completes.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutably view the caller-owned buffer. Do not mutate it while an RDMA
    /// operation is in flight on another connection.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn destination(&self, offset: usize) -> Result<(u64, u32, usize)> {
        let available = self.len.checked_sub(offset).ok_or_else(|| {
            anyhow!(
                "RDMA destination offset {offset} exceeds buffer length {}",
                self.len
            )
        })?;
        if available == 0 {
            return Err(anyhow!("RDMA destination has no remaining capacity"));
        }
        Ok((
            self.ptr.as_ptr() as u64 + offset as u64,
            unsafe { (*self.mr.as_ptr()).rkey },
            available,
        ))
    }

    fn source(&self, offset: usize, size: usize) -> Result<(u64, u32)> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("RDMA source range overflows usize"))?;
        if end > self.len {
            return Err(anyhow!(
                "RDMA source range {offset}..{end} exceeds buffer length {}",
                self.len
            ));
        }
        Ok((self.ptr.as_ptr() as u64 + offset as u64, unsafe {
            (*self.mr.as_ptr()).lkey
        }))
    }
}

impl Drop for RegisteredBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            ibv_dereg_mr(self.mr.as_ptr());
        }
    }
}

/// Return the canonical internal key consumed by the ContextStore RDMA server.
/// The namespace length makes the namespace/object-key boundary unambiguous.
pub fn canonical_key(namespace: &str, object_key: &str) -> String {
    format!("{}:{namespace}{object_key}", namespace.len())
}

fn access_flags() -> i32 {
    (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
        | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
        | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as i32
}

fn random_psn() -> u32 {
    (SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos() as u32)
        .unwrap_or(0))
        & 0x00ff_ffff
}

fn create_qp(resources: &RdmaResources) -> Result<NonNull<ibv_qp>> {
    let mut attr = ibv_qp_init_attr {
        qp_context: ptr::null_mut(),
        send_cq: resources.cq.as_ptr(),
        recv_cq: resources.cq.as_ptr(),
        srq: ptr::null_mut(),
        cap: ibv_qp_cap {
            max_send_wr: CQ_DEPTH as u32,
            max_recv_wr: CQ_DEPTH as u32,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
        },
        qp_type: ibv_qp_type::IBV_QPT_RC,
        sq_sig_all: 0,
    };
    unsafe {
        NonNull::new(ibv_create_qp(resources.pd.as_ptr(), &mut attr))
            .ok_or_else(|| anyhow!("ibv_create_qp failed: {}", std::io::Error::last_os_error()))
    }
}

fn transition_qp_to_init(qp: NonNull<ibv_qp>, port: u8) -> Result<()> {
    unsafe {
        let mut attr: ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = port;
        attr.qp_access_flags = access_flags() as u32;
        let mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
            | ibv_qp_attr_mask::IBV_QP_PORT
            | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
        let rc = ibv_modify_qp(qp.as_ptr(), &mut attr, mask.0 as i32);
        if rc != 0 {
            return Err(anyhow!("modify QP to INIT failed: rc={rc}"));
        }
    }
    Ok(())
}

fn transition_qp_to_rtr(
    qp: NonNull<ibv_qp>,
    remote: &QpInfo,
    port: u8,
    gid_index: u8,
) -> Result<()> {
    unsafe {
        let mut attr: ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
        attr.path_mtu = ibv_mtu::IBV_MTU_1024;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = port;
        attr.ah_attr.grh.dgid = remote.gid;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.sgid_index = gid_index;
        let mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_AV
            | ibv_qp_attr_mask::IBV_QP_PATH_MTU
            | ibv_qp_attr_mask::IBV_QP_DEST_QPN
            | ibv_qp_attr_mask::IBV_QP_RQ_PSN
            | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
            | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
        let rc = ibv_modify_qp(qp.as_ptr(), &mut attr, mask.0 as i32);
        if rc != 0 {
            return Err(anyhow!("modify QP to RTR failed: rc={rc}"));
        }
    }
    Ok(())
}

fn transition_qp_to_rts(qp: NonNull<ibv_qp>, psn: u32) -> Result<()> {
    unsafe {
        let mut attr: ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = psn;
        attr.max_rd_atomic = 1;
        let mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_TIMEOUT
            | ibv_qp_attr_mask::IBV_QP_RETRY_CNT
            | ibv_qp_attr_mask::IBV_QP_RNR_RETRY
            | ibv_qp_attr_mask::IBV_QP_SQ_PSN
            | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;
        let rc = ibv_modify_qp(qp.as_ptr(), &mut attr, mask.0 as i32);
        if rc != 0 {
            return Err(anyhow!("modify QP to RTS failed: rc={rc}"));
        }
    }
    Ok(())
}

fn write_hello(stream: &mut TcpStream, info: QpInfo) -> Result<()> {
    stream.write_all(&[MSG_HELLO])?;
    stream.write_all(&info.to_bytes())?;
    stream.flush()?;
    Ok(())
}

fn read_hello(stream: &mut TcpStream) -> Result<QpInfo> {
    let mut tag = [0u8; 1];
    stream.read_exact(&mut tag)?;
    if tag[0] != MSG_HELLO {
        return Err(anyhow!("expected RDMA HELLO, got {}", tag[0]));
    }
    let mut payload = [0u8; 24];
    stream.read_exact(&mut payload)?;
    Ok(QpInfo::from_bytes(&payload))
}

fn build_get_request(key: &str, addr: u64, rkey: u32, max_size: u64) -> Result<Vec<u8>> {
    let mut request = Vec::with_capacity(1 + 2 + key.len() + 8 + 4 + 8);
    request.push(MSG_GET_REQ);
    push_string(&mut request, key, "key")?;
    request.extend_from_slice(&addr.to_le_bytes());
    request.extend_from_slice(&rkey.to_le_bytes());
    request.extend_from_slice(&max_size.to_le_bytes());
    Ok(request)
}

fn build_descriptor_get_request(
    key: &str,
    descriptor: &pb::ObjectDescriptor,
    addr: u64,
    rkey: u32,
    max_size: u64,
) -> Result<Vec<u8>> {
    let mut request = Vec::with_capacity(128 + key.len() + descriptor.object_handle.len());
    request.push(MSG_GET_DESCRIPTOR_REQ);
    push_string(&mut request, key, "key")?;
    push_string(&mut request, &descriptor.object_handle, "object_handle")?;
    request.extend_from_slice(&descriptor.object_generation.to_le_bytes());
    push_string(&mut request, &descriptor.content_etag, "content_etag")?;
    request.extend_from_slice(&descriptor.layout_version.to_le_bytes());
    request.extend_from_slice(&descriptor.size.to_le_bytes());
    request.push(u8::from(descriptor.is_striped));
    request.extend_from_slice(&descriptor.stripe_count.to_le_bytes());
    request.extend_from_slice(&descriptor.chunk_size.to_le_bytes());
    request.extend_from_slice(&addr.to_le_bytes());
    request.extend_from_slice(&rkey.to_le_bytes());
    request.extend_from_slice(&max_size.to_le_bytes());
    Ok(request)
}

fn build_put_request(message: u8, key: &str, size: u64) -> Result<Vec<u8>> {
    let mut request = Vec::with_capacity(1 + 2 + key.len() + 8);
    request.push(message);
    push_string(&mut request, key, "key")?;
    request.extend_from_slice(&size.to_le_bytes());
    Ok(request)
}

fn push_string(target: &mut Vec<u8>, value: &str, field: &str) -> Result<()> {
    let bytes = value.as_bytes();
    let length = u16::try_from(bytes.len())
        .map_err(|_| anyhow!("RDMA {field} exceeds {} bytes", u16::MAX))?;
    target.extend_from_slice(&length.to_le_bytes());
    target.extend_from_slice(bytes);
    Ok(())
}

fn read_get_response(stream: &mut TcpStream) -> Result<RdmaReadResult> {
    let mut tag = [0u8; 1];
    stream.read_exact(&mut tag)?;
    if tag[0] != MSG_GET_RESP {
        return Err(anyhow!("expected RDMA GET response, got {}", tag[0]));
    }
    let mut body = [0u8; 13];
    stream.read_exact(&mut body)?;
    if body[0] == 0 {
        return Ok(None);
    }
    let bytes = u64::from_le_bytes(body[1..9].try_into().expect("fixed get response length"));
    let bytes = usize::try_from(bytes).map_err(|_| anyhow!("RDMA read size exceeds usize"))?;
    Ok(Some(bytes))
}

struct PutReady {
    ok: bool,
    addr: u64,
    rkey: u32,
}

struct RemoteWrite {
    id: u64,
    src_addr: u64,
    src_lkey: u32,
    dst_addr: u64,
    dst_rkey: u32,
    len: u32,
    signaled: bool,
}

fn read_put_ready(stream: &mut TcpStream) -> Result<PutReady> {
    let mut tag = [0u8; 1];
    stream.read_exact(&mut tag)?;
    if tag[0] != MSG_PUT_READY {
        return Err(anyhow!("expected RDMA PUT ready, got {}", tag[0]));
    }
    let mut body = [0u8; 13];
    stream.read_exact(&mut body)?;
    Ok(PutReady {
        ok: body[0] != 0,
        addr: u64::from_le_bytes(body[1..9].try_into().expect("fixed put ready address")),
        rkey: u32::from_le_bytes(body[9..13].try_into().expect("fixed put ready rkey")),
    })
}

fn read_put_result(stream: &mut TcpStream) -> Result<u8> {
    let mut tag = [0u8; 1];
    stream.read_exact(&mut tag)?;
    if tag[0] != MSG_PUT_RESP {
        return Err(anyhow!("expected RDMA PUT response, got {}", tag[0]));
    }
    let mut body = [0u8; 1];
    stream.read_exact(&mut body)?;
    Ok(body[0])
}

fn post_write(qp: NonNull<ibv_qp>, write: &RemoteWrite) -> Result<()> {
    unsafe {
        let mut sge = ibv_sge {
            addr: write.src_addr,
            length: write.len,
            lkey: write.src_lkey,
        };
        let mut wr: ibv_send_wr = std::mem::zeroed();
        wr.wr_id = write.id;
        wr.sg_list = &mut sge;
        wr.num_sge = 1;
        wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
        wr.send_flags = if write.signaled {
            ibv_send_flags::IBV_SEND_SIGNALED.0
        } else {
            0
        };
        wr.wr.rdma.remote_addr = write.dst_addr;
        wr.wr.rdma.rkey = write.dst_rkey;
        let mut bad_wr = ptr::null_mut();
        let rc = ibv_post_send(qp.as_ptr(), &mut wr, &mut bad_wr);
        if rc != 0 {
            return Err(anyhow!("ibv_post_send RDMA WRITE failed: rc={rc}"));
        }
    }
    Ok(())
}

fn poll_completion(cq: NonNull<ibv_cq>) -> Result<()> {
    unsafe {
        let mut completion: ibv_wc = std::mem::zeroed();
        let started = Instant::now();
        loop {
            let received = ibv_poll_cq(cq.as_ptr(), 1, &mut completion);
            if received < 0 {
                return Err(anyhow!("ibv_poll_cq failed"));
            }
            if received > 0 {
                if completion.status != ibv_wc_status::IBV_WC_SUCCESS {
                    return Err(anyhow!(
                        "RDMA WRITE completion failed: wr_id={} status={}",
                        completion.wr_id,
                        completion.status
                    ));
                }
                return Ok(());
            }
            if started.elapsed() >= COMPLETION_TIMEOUT {
                return Err(anyhow!("RDMA WRITE completion timed out"));
            }
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_key_preserves_namespace_boundary() {
        assert_eq!(canonical_key("redhare", "block-1"), "7:redhareblock-1");
        assert_eq!(canonical_key("", "block-1"), "0:block-1");
    }

    #[test]
    fn descriptor_request_contains_identity_and_destination() {
        let descriptor = pb::ObjectDescriptor {
            key: None,
            object_handle: "handle-1".into(),
            object_generation: 7,
            content_etag: "etag-7".into(),
            layout_version: 3,
            size: 4096,
            is_striped: true,
            stripe_count: 2,
            chunk_size: 2048,
        };
        let request =
            build_descriptor_get_request("7:redhareblock-1", &descriptor, 9, 10, 11).unwrap();
        assert_eq!(request[0], MSG_GET_DESCRIPTOR_REQ);
        assert!(request
            .windows(b"handle-1".len())
            .any(|window| window == b"handle-1"));
        assert!(request
            .windows(b"etag-7".len())
            .any(|window| window == b"etag-7"));
        assert!(request.ends_with(&11u64.to_le_bytes()));
    }

    #[test]
    fn request_rejects_oversized_wire_string() {
        let key = "x".repeat(u16::MAX as usize + 1);
        assert!(build_get_request(&key, 1, 2, 3).is_err());
        assert!(build_put_request(MSG_PUT_REQ, &key, 3).is_err());
    }

    #[test]
    fn config_uses_roce_defaults() {
        let config = RdmaClientConfig::new("10.0.0.1:50053", "mlx5_0");
        assert_eq!(config.port, 1);
        assert_eq!(config.gid_index, 3);
    }
}
