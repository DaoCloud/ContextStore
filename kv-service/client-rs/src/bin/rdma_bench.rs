//! RDMA Bench Client — directly measure RDMA tier bandwidth (bypasses gRPC).
//!
//! Flow:
//! 1. TCP connect to server (default 127.0.0.1:50053)
//! 2. Exchange QP info, transition INIT→RTR→RTS
//! 3. Register a local buffer as an MR
//! 4. Send GetReq; the server RDMA-WRITEs chunks_cache data into the local MR
//! 5. Measure throughput
//!
//! Note: the server's chunks_cache must already contain data (run cs-bench --combined
//! first to PUT).

use anyhow::{anyhow, Result};
use clap::Parser;
use contextstore_client_rs::rdma::{canonical_key, RdmaClient, RdmaClientConfig};
use contextstore_client_rs::KvClient;
use rdma_sys::*;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::ptr::{self, NonNull};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Args {
    /// Server TCP address (RDMA control plane)
    #[arg(long, default_value = "127.0.0.1:50053")]
    server: String,

    /// Benchmark direction: `get` (default) reads an existing object;
    /// `put` writes `--put-mb` of patterned data per iteration over the
    /// RDMA data plane (PutReq -> server slab grant -> RDMA WRITE ->
    /// commit -> server pwrite), giving the write-path counterpart of
    /// the read benchmark. `put` requires --namespace/--object-key and
    /// is single-endpoint only (not valid with --coordinator).
    #[arg(long, default_value = "get")]
    mode: String,

    /// PUT mode: payload size in MB per iteration.
    #[arg(long, default_value_t = 480)]
    put_mb: usize,

    /// PUT mode: TTL seconds stamped on stored objects (0 = no expiry).
    #[arg(long, default_value_t = 0)]
    ttl_seconds: i64,

    /// Multi-endpoint GET: read through the SGE (scatter destination) wire
    /// path, logically splitting the destination buffer into this many
    /// segments. 0 = use the classic single-destination stripe GET.
    /// Validates the tag-15 server path; data lands at identical offsets so
    /// data_ok checks still hold.
    #[arg(long, default_value_t = 0)]
    sge_segments: usize,

    /// gRPC coordinator used to discover and read all RDMA placement endpoints.
    ///
    /// When set, the benchmark groups stripes by their owning node and issues
    /// concurrent stripe-subset GETs instead of reading the whole object from
    /// --server. Requires --namespace and --object-key.
    #[arg(long, requires_all = ["namespace", "object_key"])]
    coordinator: Option<String>,

    /// HCA device name (e.g. mlx5_0)
    #[arg(long, default_value = "mlx5_0")]
    device: String,

    /// HCA port number
    #[arg(long, default_value_t = 1u8)]
    port: u8,

    /// GID index (RoCE v2 IPv4 mapped = 3 typically)
    #[arg(long, default_value_t = 3u8)]
    gid_index: u8,

    /// Canonical object key to GET: <namespace_byte_len>:<namespace><object_key>.
    #[arg(long, conflicts_with_all = ["namespace", "object_key"])]
    key: Option<String>,

    /// Logical namespace. Use together with --object-key for a readable selector.
    #[arg(long, requires = "object_key")]
    namespace: Option<String>,

    /// Logical object key within --namespace.
    #[arg(long, requires = "namespace")]
    object_key: Option<String>,

    /// Buffer size in MB (client side recv buffer)
    #[arg(long, default_value_t = 512usize)]
    buf_mb: usize,

    /// Number of iterations
    #[arg(long, default_value_t = 5usize)]
    iters: usize,

    /// Clear the complete receive buffer before every request.
    ///
    /// Disabled by default because clearing multi-GiB buffers between requests lowers
    /// the sustained disk duty cycle while being outside the timed RDMA transfer.
    #[arg(long, default_value_t = false)]
    clear_buffer: bool,
}

const MSG_HELLO: u8 = 1;
const MSG_GET_REQ: u8 = 2;
const MSG_GET_RESP: u8 = 3;

#[derive(Clone, Copy)]
struct QpInfo {
    qpn: u32,
    psn: u32,
    gid: ibv_gid,
}

impl QpInfo {
    fn to_bytes(self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&self.qpn.to_le_bytes());
        buf[4..8].copy_from_slice(&self.psn.to_le_bytes());
        unsafe {
            buf[8..24].copy_from_slice(&self.gid.raw[..]);
        }
        buf
    }
    fn from_bytes(buf: &[u8; 24]) -> Self {
        let qpn = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let psn = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
        unsafe {
            gid.raw[..].copy_from_slice(&buf[8..24]);
        }
        Self { qpn, psn, gid }
    }
}

fn read_exact(s: &mut TcpStream, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf)?;
    Ok(buf)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.mode == "put" {
        return run_put(&args);
    }
    if args.mode != "get" {
        return Err(anyhow!("unknown --mode {} (expected get|put)", args.mode));
    }
    let object_key = resolve_object_key(&args)?;
    if let Some(coordinator) = &args.coordinator {
        return run_multi_endpoint(&args, coordinator);
    }
    run_single_endpoint(&args, &object_key)
}

/// RDMA PUT benchmark: each iteration writes a fresh object (`<object_key>-i<N>`)
/// so the server takes the full write path every time (slab grant + client RDMA
/// WRITE + commit + striped O_DIRECT pwrite) instead of dedup-skipping.
fn run_put(args: &Args) -> Result<()> {
    if args.coordinator.is_some() {
        return Err(anyhow!("--mode put does not support --coordinator (single endpoint only)"));
    }
    let namespace = args
        .namespace
        .as_deref()
        .ok_or_else(|| anyhow!("--mode put requires --namespace"))?;
    let object_key = args
        .object_key
        .as_deref()
        .ok_or_else(|| anyhow!("--mode put requires --object-key"))?;

    let size = args.put_mb * 1024 * 1024;
    let config = RdmaClientConfig::new(&args.server, &args.device)
        .with_port(args.port)
        .with_gid_index(args.gid_index);
    let mut client = RdmaClient::connect(config)?;

    // Patterned payload: byte i = i % 251 (prime, detects offset shifts).
    let mut payload = vec![0u8; size];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let t_reg = Instant::now();
    let buffer = client.register_buffer(&mut payload)?;
    println!(
        "[client] PUT source MR registered: {} MB in {:.1}ms",
        args.put_mb,
        t_reg.elapsed().as_secs_f64() * 1000.0
    );

    let mut times = Vec::with_capacity(args.iters);
    for iter in 1..=args.iters {
        let key = format!("{object_key}-i{iter}");
        let t0 = Instant::now();
        client.put_from_with_ttl(namespace, &key, &buffer, 0, size, args.ttl_seconds)?;
        let dt = t0.elapsed();
        times.push(dt);
        let gbps = size as f64 * 8.0 / dt.as_secs_f64() / 1e9;
        println!(
            "[client] iter {iter}: bytes={size} time={:.2}ms BW={:.2} Gbps = {:.2} GB/s key={key}",
            dt.as_secs_f64() * 1000.0,
            gbps,
            size as f64 / dt.as_secs_f64() / 1e9,
        );
    }

    times.sort();
    let med = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let bw = |d: &Duration| size as f64 / d.as_secs_f64() / 1e9;
    println!();
    println!(
        "[summary] mode=rdma_put iters={} bytes_per_iter={} ({} MB)",
        args.iters, size, args.put_mb
    );
    println!(
        "  latency  min={:.2}ms med={:.2}ms max={:.2}ms",
        min.as_secs_f64() * 1000.0,
        med.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0
    );
    println!(
        "  BW       max={:.2} GB/s med={:.2} GB/s min={:.2} GB/s",
        bw(&min),
        bw(&med),
        bw(&max)
    );
    Ok(())
}

fn run_single_endpoint(args: &Args, object_key: &str) -> Result<()> {
    let buf_size = args.buf_mb * 1024 * 1024;

    // ===== 1. Open HCA + PD + CQ + register MR =====
    unsafe {
        let mut num = 0i32;
        let dev_list = ibv_get_device_list(&mut num);
        if dev_list.is_null() {
            return Err(anyhow!("no RDMA device"));
        }
        let mut dev_ptr: *mut ibv_device = ptr::null_mut();
        for i in 0..num {
            let d = *dev_list.offset(i as isize);
            let name = CStr::from_ptr(ibv_get_device_name(d)).to_string_lossy();
            if name == args.device {
                dev_ptr = d;
                break;
            }
        }
        if dev_ptr.is_null() {
            ibv_free_device_list(dev_list);
            return Err(anyhow!("device {} not found", args.device));
        }

        let ctx = NonNull::new(ibv_open_device(dev_ptr))
            .ok_or_else(|| anyhow!("ibv_open_device failed"))?;
        ibv_free_device_list(dev_list);
        let pd = NonNull::new(ibv_alloc_pd(ctx.as_ptr()))
            .ok_or_else(|| anyhow!("ibv_alloc_pd failed"))?;
        let cq = NonNull::new(ibv_create_cq(
            ctx.as_ptr(),
            128,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
        ))
        .ok_or_else(|| anyhow!("ibv_create_cq failed"))?;

        let mut local_gid: ibv_gid = std::mem::zeroed();
        let rc = ibv_query_gid(
            ctx.as_ptr(),
            args.port,
            args.gid_index as i32,
            &mut local_gid,
        );
        if rc != 0 {
            return Err(anyhow!("ibv_query_gid failed: {}", rc));
        }

        // Allocate + register buffer
        let layout = std::alloc::Layout::from_size_align(buf_size, 4096)?;
        let buf_ptr = std::alloc::alloc_zeroed(layout);
        if buf_ptr.is_null() {
            return Err(anyhow!("alloc failed"));
        }
        let mr = NonNull::new(ibv_reg_mr(
            pd.as_ptr(),
            buf_ptr as *mut std::ffi::c_void,
            buf_size,
            (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as i32,
        ))
        .ok_or_else(|| anyhow!("ibv_reg_mr failed: {}", std::io::Error::last_os_error()))?;
        let local_addr = buf_ptr as u64;
        let local_rkey = (*mr.as_ptr()).rkey;
        println!(
            "[client] MR registered: addr=0x{:x} len={} rkey=0x{:x}",
            local_addr, buf_size, local_rkey
        );

        // ===== 2. Create QP, INIT =====
        let mut qp_attr = ibv_qp_init_attr {
            qp_context: ptr::null_mut(),
            send_cq: cq.as_ptr(),
            recv_cq: cq.as_ptr(),
            srq: ptr::null_mut(),
            cap: ibv_qp_cap {
                max_send_wr: 128,
                max_recv_wr: 128,
                max_send_sge: 4,
                max_recv_sge: 4,
                max_inline_data: 0,
            },
            qp_type: ibv_qp_type::IBV_QPT_RC,
            sq_sig_all: 0,
        };
        let qp = NonNull::new(ibv_create_qp(pd.as_ptr(), &mut qp_attr))
            .ok_or_else(|| anyhow!("ibv_create_qp failed"))?;
        let local_qpn = (*qp.as_ptr()).qp_num;
        let local_psn = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(0))
            & 0xFFFFFF;

        // to_init
        let mut attr: ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = args.port;
        attr.qp_access_flags = (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
            | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
            | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as i32
            as u32;
        let mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
            | ibv_qp_attr_mask::IBV_QP_PORT
            | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
        let rc = ibv_modify_qp(qp.as_ptr(), &mut attr, mask.0 as i32);
        if rc != 0 {
            return Err(anyhow!("modify_qp INIT: {}", rc));
        }

        // ===== 3. TCP connect + exchange QP info =====
        let mut stream = TcpStream::connect(&args.server)?;
        let local_qp_info = QpInfo {
            qpn: local_qpn,
            psn: local_psn,
            gid: local_gid,
        };

        // Send first (server receives first).
        let mut hello = Vec::with_capacity(25);
        hello.push(MSG_HELLO);
        hello.extend_from_slice(&local_qp_info.to_bytes());
        stream.write_all(&hello)?;
        stream.flush()?;

        // recv server hello
        let tag = read_exact(&mut stream, 1)?[0];
        if tag != MSG_HELLO {
            return Err(anyhow!("expected HELLO, got {}", tag));
        }
        let body = read_exact(&mut stream, 24)?;
        let arr: [u8; 24] = body.try_into().unwrap();
        let remote = QpInfo::from_bytes(&arr);
        println!("[client] remote qpn={} psn=0x{:x}", remote.qpn, remote.psn);

        // ===== 4. to_rtr + to_rts =====
        let mut attr: ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
        attr.path_mtu = ibv_mtu::IBV_MTU_1024;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = 0;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = args.port;
        attr.ah_attr.grh.dgid = remote.gid;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.sgid_index = args.gid_index;
        attr.ah_attr.grh.traffic_class = 0;
        let mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_AV
            | ibv_qp_attr_mask::IBV_QP_PATH_MTU
            | ibv_qp_attr_mask::IBV_QP_DEST_QPN
            | ibv_qp_attr_mask::IBV_QP_RQ_PSN
            | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
            | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
        let rc = ibv_modify_qp(qp.as_ptr(), &mut attr, mask.0 as i32);
        if rc != 0 {
            return Err(anyhow!(
                "modify_qp RTR: {} errno={}",
                rc,
                std::io::Error::last_os_error()
            ));
        }
        let mut attr: ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = local_psn;
        attr.max_rd_atomic = 1;
        let mask = ibv_qp_attr_mask::IBV_QP_STATE
            | ibv_qp_attr_mask::IBV_QP_TIMEOUT
            | ibv_qp_attr_mask::IBV_QP_RETRY_CNT
            | ibv_qp_attr_mask::IBV_QP_RNR_RETRY
            | ibv_qp_attr_mask::IBV_QP_SQ_PSN
            | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;
        let rc = ibv_modify_qp(qp.as_ptr(), &mut attr, mask.0 as i32);
        if rc != 0 {
            return Err(anyhow!("modify_qp RTS: {}", rc));
        }

        println!(
            "[client] QP established. starting benchmark (validation={})...",
            if args.clear_buffer {
                "full_buffer_clear"
            } else {
                "prefix_sentinel"
            }
        );

        // ===== 5. Run N GETs =====
        let mut latencies = Vec::new();
        let mut last_bytes = 0u64;
        for i in 0..args.iters {
            // A small sentinel proves the RDMA WRITE updated the target without adding a
            // multi-GiB host-memory memset between requests. The full clear remains useful
            // for diagnostics but must be explicitly requested.
            if args.clear_buffer {
                std::ptr::write_bytes(buf_ptr, 0u8, buf_size);
            } else {
                std::ptr::write_bytes(buf_ptr, 0xff, 64.min(buf_size));
            }

            let t0 = Instant::now();

            // send GetReq
            let key_bytes = object_key.as_bytes();
            let mut req = Vec::with_capacity(1 + 2 + key_bytes.len() + 8 + 4 + 8);
            req.push(MSG_GET_REQ);
            req.extend_from_slice(&(key_bytes.len() as u16).to_le_bytes());
            req.extend_from_slice(key_bytes);
            req.extend_from_slice(&local_addr.to_le_bytes());
            req.extend_from_slice(&local_rkey.to_le_bytes());
            req.extend_from_slice(&(buf_size as u64).to_le_bytes());
            stream.write_all(&req)?;
            stream.flush()?;

            // recv GetResp
            let tag = read_exact(&mut stream, 1)?[0];
            if tag != MSG_GET_RESP {
                return Err(anyhow!("expected GET_RESP, got {}", tag));
            }
            let body = read_exact(&mut stream, 1 + 8 + 4)?;
            let found = body[0] != 0;
            let bytes_written = u64::from_le_bytes(body[1..9].try_into().unwrap());
            let num_chunks = u32::from_le_bytes(body[9..13].try_into().unwrap());
            let dt = t0.elapsed();

            if !found {
                return Err(anyhow!(
                    "key '{}' not found in server cache. Run cs-bench first to PUT",
                    object_key
                ));
            }
            latencies.push(dt);
            last_bytes = bytes_written;

            // Verify data: cs-bench combined fills with the (i % 251) pattern.
            // Look at the first few bytes.
            let head = std::slice::from_raw_parts(buf_ptr, 16.min(bytes_written as usize));
            let expected: Vec<u8> = (0..head.len()).map(|i| (i % 251) as u8).collect();
            let matches = head == expected.as_slice();
            let bw_gbps = (bytes_written as f64 * 8.0) / dt.as_secs_f64() / 1e9;
            let bw_gb = (bytes_written as f64) / dt.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
            println!(
                "[client] iter {}: bytes={} chunks={} time={:.2}ms BW={:.2} Gbps = {:.2} GB/s data_ok={} head={:02x?}",
                i + 1,
                bytes_written,
                num_chunks,
                dt.as_secs_f64() * 1000.0,
                bw_gbps,
                bw_gb,
                matches,
                head,
            );
        }

        // summary
        latencies.sort();
        let min = latencies.first().unwrap();
        let med = &latencies[latencies.len() / 2];
        let max = latencies.last().unwrap();
        let bw_min = (last_bytes as f64) / max.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        let bw_med = (last_bytes as f64) / med.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        let bw_max = (last_bytes as f64) / min.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!(
            "\n[summary] iters={} bytes_per_iter={} ({} MB)",
            args.iters,
            last_bytes,
            last_bytes / (1024 * 1024)
        );
        println!(
            "  latency  min={:.2}ms med={:.2}ms max={:.2}ms",
            min.as_secs_f64() * 1000.0,
            med.as_secs_f64() * 1000.0,
            max.as_secs_f64() * 1000.0
        );
        println!(
            "  BW       max={:.2} GB/s med={:.2} GB/s min={:.2} GB/s",
            bw_max, bw_med, bw_min
        );

        // send BYE (best effort)
        let _ = stream.write_all(&[99u8]);

        // cleanup
        ibv_dereg_mr(mr.as_ptr());
        std::alloc::dealloc(buf_ptr, layout);
        ibv_destroy_qp(qp.as_ptr());
        ibv_destroy_cq(cq.as_ptr());
        ibv_dealloc_pd(pd.as_ptr());
        ibv_close_device(ctx.as_ptr());
    }
    Ok(())
}

enum WorkerCommand {
    Read,
    Stop,
}

enum WorkerEvent {
    Ready {
        endpoint: String,
        result: std::result::Result<(), String>,
    },
    ReadDone {
        endpoint: String,
        result: std::result::Result<usize, String>,
    },
}

struct AlignedBuffer {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    fn new(size: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(size, 4096)?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(anyhow!("failed to allocate {size} byte receive buffer"));
        }
        Ok(Self { ptr, layout })
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

fn run_multi_endpoint(args: &Args, coordinator: &str) -> Result<()> {
    let namespace = args
        .namespace
        .as_deref()
        .ok_or_else(|| anyhow!("--coordinator requires --namespace"))?;
    let object_key = args
        .object_key
        .as_deref()
        .ok_or_else(|| anyhow!("--coordinator requires --object-key"))?;
    let coordinator = if coordinator.starts_with("http://") || coordinator.starts_with("https://") {
        coordinator.to_string()
    } else {
        format!("http://{coordinator}")
    };

    let runtime = tokio::runtime::Runtime::new()?;
    let lookup = runtime.block_on(async {
        let mut client = KvClient::connect(coordinator)
            .await
            .map_err(|error| anyhow!(error.to_string()))?;
        let lookup = client.lookup_object(namespace, object_key).await?;
        Ok::<_, anyhow::Error>(lookup)
    })?;
    let lookup = lookup.ok_or_else(|| anyhow!("object not found: {namespace}/{object_key}"))?;
    let placement = lookup
        .placement
        .ok_or_else(|| anyhow!("object lookup returned no placement"))?;
    if placement.chunks.is_empty() {
        return Err(anyhow!("object placement contains no chunks"));
    }

    let mut endpoint_stripes: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for chunk in &placement.chunks {
        if chunk.rdma_endpoint.is_empty() {
            return Err(anyhow!(
                "stripe {} has no RDMA endpoint in placement",
                chunk.stripe_index
            ));
        }
        endpoint_stripes
            .entry(chunk.rdma_endpoint.clone())
            .or_default()
            .push(chunk.stripe_index);
    }
    for stripes in endpoint_stripes.values_mut() {
        stripes.sort_unstable();
        stripes.dedup();
    }

    let buf_size = args.buf_mb * 1024 * 1024;
    let sge_segments = args.sge_segments;
    let object_size = usize::try_from(lookup.descriptor.size)
        .map_err(|_| anyhow!("object size does not fit usize"))?;
    if buf_size < object_size {
        return Err(anyhow!(
            "client buffer too small: {} bytes for {} byte object",
            buf_size,
            object_size
        ));
    }
    let buffer = AlignedBuffer::new(buf_size)?;
    let buffer_addr = buffer.ptr as usize;
    let endpoint_count = endpoint_stripes.len();
    let stripe_count = placement.chunks.len();

    println!(
        "[client] multi-endpoint placement: endpoints={} stripes={} bytes={}",
        endpoint_count, stripe_count, lookup.descriptor.size
    );
    for (endpoint, stripes) in &endpoint_stripes {
        println!("[client] endpoint {endpoint}: {} stripe(s)", stripes.len());
    }

    let (event_tx, event_rx) = mpsc::channel();
    let mut workers = Vec::with_capacity(endpoint_count);
    for (endpoint, stripes) in endpoint_stripes {
        let (command_tx, command_rx) = mpsc::channel();
        let worker_events = event_tx.clone();
        let worker_endpoint = endpoint.clone();
        let descriptor = lookup.descriptor.clone();
        let device = args.device.clone();
        let port = args.port;
        let gid_index = args.gid_index;
        let handle = thread::spawn(move || {
            let setup = (|| -> Result<_> {
                let config = RdmaClientConfig::new(&worker_endpoint, device)
                    .with_port(port)
                    .with_gid_index(gid_index);
                let client = RdmaClient::connect(config)?;
                // SAFETY: the allocation remains alive until all workers are stopped.
                // Each endpoint writes only the disjoint stripes assigned to it.
                let registered =
                    unsafe { client.register_raw_buffer(buffer_addr as *mut u8, buf_size)? };
                Ok((client, registered))
            })();
            let (mut client, registered) = match setup {
                Ok(resources) => {
                    let _ = worker_events.send(WorkerEvent::Ready {
                        endpoint: worker_endpoint.clone(),
                        result: Ok(()),
                    });
                    resources
                }
                Err(error) => {
                    let _ = worker_events.send(WorkerEvent::Ready {
                        endpoint: worker_endpoint,
                        result: Err(error.to_string()),
                    });
                    return;
                }
            };

            while let Ok(command) = command_rx.recv() {
                match command {
                    WorkerCommand::Read => {
                        let result = if sge_segments > 0 {
                            // 把整个目标 buffer 均分为 N 段 (同一 MR, 不同偏移),
                            // 走 tag-15 SGE 路径; server 按段映射逐段 WRITE.
                            let view = registered.view();
                            let seg_len = (buf_size / sge_segments).max(1) as u64;
                            let mut segments = Vec::with_capacity(sge_segments);
                            let mut off = 0u64;
                            let (base, rkey, total) =
                                (view.addr(), view.rkey(), buf_size as u64);
                            while off < total {
                                let n = seg_len.min(total - off);
                                segments.push((base + off, rkey, n));
                                off += n;
                            }
                            client
                                .get_descriptor_stripes_sge(&descriptor, &stripes, &segments)
                                .map(|bytes| bytes.unwrap_or(0))
                                .map_err(|error| error.to_string())
                        } else {
                            client
                                .get_descriptor_stripes_into(&descriptor, &stripes, &registered, 0)
                                .map(|bytes| bytes.unwrap_or(0))
                                .map_err(|error| error.to_string())
                        };
                        let _ = worker_events.send(WorkerEvent::ReadDone {
                            endpoint: worker_endpoint.clone(),
                            result,
                        });
                    }
                    WorkerCommand::Stop => break,
                }
            }
        });
        workers.push((endpoint, command_tx, handle));
    }
    drop(event_tx);

    let mut setup_error = None;
    for _ in 0..endpoint_count {
        match event_rx.recv()? {
            WorkerEvent::Ready {
                endpoint,
                result: Ok(()),
            } => println!("[client] RDMA endpoint ready: {endpoint}"),
            WorkerEvent::Ready {
                endpoint,
                result: Err(error),
            } => setup_error = Some(anyhow!("RDMA endpoint {endpoint} setup failed: {error}")),
            WorkerEvent::ReadDone { .. } => {
                setup_error = Some(anyhow!("worker returned data before setup completed"))
            }
        }
    }

    let benchmark_result = if let Some(error) = setup_error {
        Err(error)
    } else {
        run_multi_endpoint_iterations(
            args,
            &workers,
            &event_rx,
            &buffer,
            object_size,
            stripe_count,
        )
    };

    for (_, command_tx, _) in &workers {
        let _ = command_tx.send(WorkerCommand::Stop);
    }
    for (endpoint, _, handle) in workers {
        handle
            .join()
            .map_err(|_| anyhow!("RDMA endpoint worker panicked: {endpoint}"))?;
    }
    benchmark_result
}

fn run_multi_endpoint_iterations(
    args: &Args,
    workers: &[(String, mpsc::Sender<WorkerCommand>, thread::JoinHandle<()>)],
    event_rx: &mpsc::Receiver<WorkerEvent>,
    buffer: &AlignedBuffer,
    object_size: usize,
    stripe_count: usize,
) -> Result<()> {
    let mut latencies = Vec::with_capacity(args.iters);
    for iteration in 0..args.iters {
        unsafe {
            if args.clear_buffer {
                std::ptr::write_bytes(buffer.ptr, 0, buffer.layout.size());
            } else {
                std::ptr::write_bytes(buffer.ptr, 0xff, 64.min(buffer.layout.size()));
            }
        }

        let started = Instant::now();
        for (endpoint, command_tx, _) in workers {
            command_tx
                .send(WorkerCommand::Read)
                .map_err(|_| anyhow!("RDMA endpoint worker stopped: {endpoint}"))?;
        }

        let mut bytes_written = 0usize;
        for _ in workers {
            match event_rx.recv()? {
                WorkerEvent::ReadDone {
                    endpoint: _,
                    result: Ok(bytes),
                } => bytes_written += bytes,
                WorkerEvent::ReadDone {
                    endpoint,
                    result: Err(error),
                } => return Err(anyhow!("RDMA endpoint {endpoint} read failed: {error}")),
                WorkerEvent::Ready { endpoint, .. } => {
                    return Err(anyhow!("duplicate ready event from {endpoint}"))
                }
            }
        }
        let elapsed = started.elapsed();
        if bytes_written != object_size {
            return Err(anyhow!(
                "partial multi-endpoint read: {bytes_written} of {object_size} bytes"
            ));
        }
        latencies.push(elapsed);

        let head = unsafe { std::slice::from_raw_parts(buffer.ptr, 16.min(object_size)) };
        let expected: Vec<u8> = (0..head.len()).map(|index| (index % 251) as u8).collect();
        let data_ok = head == expected.as_slice();
        let gib_per_second =
            bytes_written as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        let gbits_per_second = bytes_written as f64 * 8.0 / elapsed.as_secs_f64() / 1e9;
        println!(
            "[client] iter {}: bytes={} stripes={} endpoints={} time={:.2}ms BW={:.2} Gbps = {:.3} GiB/s data_ok={} head={:02x?}",
            iteration + 1,
            bytes_written,
            stripe_count,
            workers.len(),
            elapsed.as_secs_f64() * 1000.0,
            gbits_per_second,
            gib_per_second,
            data_ok,
            head
        );
        if !data_ok {
            return Err(anyhow!("received data failed prefix validation"));
        }
    }

    latencies.sort_unstable();
    let min = latencies
        .first()
        .ok_or_else(|| anyhow!("--iters must be greater than zero"))?;
    let median = latencies[latencies.len() / 2];
    let max = latencies.last().expect("latencies is not empty");
    let gib = object_size as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "\n[summary] mode=multi_endpoint_rdma iters={} bytes_per_iter={} endpoints={} stripes={}",
        args.iters,
        object_size,
        workers.len(),
        stripe_count
    );
    println!(
        "  latency  min={:.2}ms med={:.2}ms max={:.2}ms",
        min.as_secs_f64() * 1000.0,
        median.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0
    );
    println!(
        "  BW       max={:.3} GiB/s med={:.3} GiB/s min={:.3} GiB/s",
        gib / min.as_secs_f64(),
        gib / median.as_secs_f64(),
        gib / max.as_secs_f64()
    );
    Ok(())
}

fn resolve_object_key(args: &Args) -> Result<String> {
    match (&args.key, &args.namespace, &args.object_key) {
        (Some(key), None, None) => {
            validate_canonical_key(key)?;
            Ok(key.clone())
        }
        (None, Some(namespace), Some(object_key)) => Ok(canonical_key(namespace, object_key)),
        (None, None, None) => Ok(canonical_key("rust-bench", "comb0/__combined__")),
        _ => Err(anyhow!(
            "provide --key <canonical-key> or both --namespace <namespace> and --object-key <object-key>"
        )),
    }
}

fn validate_canonical_key(key: &str) -> Result<()> {
    let Some((namespace_length, remainder)) = key.split_once(':') else {
        return Err(anyhow!(
            "invalid --key {key:?}: expected <namespace_byte_len>:<namespace><object_key>; use --namespace and --object-key for a readable selector"
        ));
    };
    let namespace_length: usize = namespace_length.parse().map_err(|_| {
        anyhow!(
            "invalid --key {key:?}: namespace byte length must be a decimal integer; use --namespace and --object-key for a readable selector"
        )
    })?;
    if namespace_length > remainder.len() {
        return Err(anyhow!(
            "invalid --key {key:?}: namespace byte length exceeds the remaining key bytes; use --namespace and --object-key for a readable selector"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readable_selector_encodes_canonical_key() {
        let args = Args {
            key: None,
            namespace: Some("rust-bench".to_string()),
            object_key: Some("rdma-checksum0/__combined__".to_string()),
            server: "127.0.0.1:50053".to_string(),
            coordinator: None,
            device: "mlx5_0".to_string(),
            port: 1,
            gid_index: 3,
            buf_mb: 512,
            iters: 5,
            clear_buffer: false,
        };
        assert_eq!(
            resolve_object_key(&args).unwrap(),
            "10:rust-benchrdma-checksum0/__combined__"
        );
    }

    #[test]
    fn raw_object_key_explains_readable_selector() {
        let error = validate_canonical_key("rdma-checksum0/__combined__").unwrap_err();
        assert!(error.to_string().contains("--namespace and --object-key"));
    }
}
