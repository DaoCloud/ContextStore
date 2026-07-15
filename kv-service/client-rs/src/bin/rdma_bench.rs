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
use rdma_sys::*;
use std::ffi::CStr;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::ptr::{self, NonNull};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// Server TCP address (RDMA control plane)
    #[arg(long, default_value = "127.0.0.1:50053")]
    server: String,

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
    /// Default matches what cs-bench --combined writes: rust-bench / comb0/__combined__.
    #[arg(long, default_value = "10:rust-benchcomb0/__combined__")]
    key: String,

    /// Buffer size in MB (client side recv buffer)
    #[arg(long, default_value_t = 512usize)]
    buf_mb: usize,

    /// Number of iterations
    #[arg(long, default_value_t = 5usize)]
    iters: usize,
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

        println!("[client] QP established. starting benchmark...");

        // ===== 5. Run N GETs =====
        let mut latencies = Vec::new();
        let mut last_bytes = 0u64;
        for i in 0..args.iters {
            // Zero the buffer so we can verify the server really wrote to it.
            std::ptr::write_bytes(buf_ptr, 0u8, buf_size);

            let t0 = Instant::now();

            // send GetReq
            let key_bytes = args.key.as_bytes();
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
                    args.key
                ));
            }
            latencies.push(dt);
            last_bytes = bytes_written;

            // Verify data: cs-bench combined fills with the (i % 251) pattern.
            // Look at the first few bytes.
            let head = std::slice::from_raw_parts(buf_ptr, 16.min(bytes_written as usize));
            let expected: Vec<u8> = (0..16).map(|i| (i % 251) as u8).collect();
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
