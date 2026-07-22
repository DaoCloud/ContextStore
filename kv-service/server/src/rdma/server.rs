//! RDMA Server — listens on a TCP control connection, one client per connection, pushes data with RDMA WRITE in the background.
//!
//! Coexists with the existing gRPC tier: both accept requests in parallel. The RDMA tier uses a separate TCP port (default 50053).
//!
//! ## Simplifying assumptions (PoC)
//! - Single HCA (mlx5_0)
//! - Single-threaded server (one thread per TCP connection)
//! - Each GET temporarily `reg_mr`s the server-side chunks (can be cached later)
//!
//! ## Path
//! ```text
//! client TCP connect → spawn thread → exchange QP info → loop:
//!   recv GetReq → ctx.memory.get_chunks(key) → reg server chunks as MR
//!     → for each chunk: post_write(...) to client buffer
//!     → poll completion → send GetResp
//! ```

use crate::metadata::{BlockMeta, StripingInfo};
use crate::rdma::context::RdmaContext;
use crate::rdma::qp::RcQp;
use crate::rdma::slab::{SlabExtent, SlabPlacement};
use crate::rdma::wire::{
    self, DescriptorGetReqMsg, GetRespMsg, PutReadyMsg, PutRespMsg, MSG_GET_DESCRIPTOR_REQ,
    MSG_GET_REQ, MSG_PUT_COMMIT, MSG_PUT_IF_ABSENT_REQ, MSG_PUT_REQ, PUT_RESULT_EXISTS,
    PUT_RESULT_FAILED, PUT_RESULT_STORED,
};
use crate::router::ObjectKey;
use crate::KVServiceContext;
use anyhow::{anyhow, Result};
use rdma_sys::ibv_access_flags;
use std::net::{TcpListener, TcpStream};
use std::ptr::NonNull;
use std::sync::Arc;
use std::thread;

/// Max bytes per single RDMA WRITE. `ibv_sge.length` is u32, and the NIC has a max_message_size
/// (commonly 1-2GiB). We cap at 1GiB; oversized values are split into multiple WRITEs from the
/// same extent (still zero-registration).
const MAX_WRITE_BYTES: u64 = 1024 * 1024 * 1024;

/// Config for a single NIC (a host may have multiple NICs, each with its own listener + RdmaContext, sharing one slab).
#[derive(Clone, Debug)]
pub struct RdmaDeviceConfig {
    pub device_name: String,
    pub port_num: u8,
    pub gid_index: u8,
    pub tcp_listen: String, // e.g. "0.0.0.0:50053"
}

/// Config. `devices.len()` determines the NIC count: 1 = single NIC (backwards-compatible);
/// >1 = multi-NIC fan-out — the same slab is `reg_mr`'d once per NIC's PD, listeners share the
/// same slab data.
#[derive(Clone, Debug)]
pub struct RdmaServerConfig {
    pub devices: Vec<RdmaDeviceConfig>,
    /// Pre-registered slab size (MB). Recommended >= ~1.5-2× memory_tier.capacity_mb to
    /// absorb fragmentation. 0 = disable slab (all GETs use per-chunk fallback).
    pub rdma_slab_size_mb: usize,
}

impl Default for RdmaServerConfig {
    fn default() -> Self {
        Self {
            devices: vec![RdmaDeviceConfig {
                device_name: "mlx5_0".to_string(),
                port_num: 1,
                gid_index: 3, // RoCE v2 IPv4-mapped
                tcp_listen: "0.0.0.0:50053".to_string(),
            }],
            rdma_slab_size_mb: 8192, // 8GB, 2× headroom for default 4GB chunks_cache
        }
    }
}

/// Launch the RDMA server. With multiple NICs, blocks the current thread until all listeners exit.
///
/// Design:
/// 1. Open N RdmaContexts sequentially (per-NIC PD/CQ/GID).
/// 2. `RdmaSlab::new(&all_ctxs)` `reg_mr`s the same host backing once per PD.
///    One copy of the data, `lkeys[i]` corresponds to `ctxs[i]`.
/// 3. `set_rdma_slab` injects into MemoryTier (slab insert/get_chunks_slab share the same slab).
/// 4. Start one listener thread per NIC, passing `nic_idx` to `handle_client`; subsequent RDMA
///    WRITEs use `extent.view(nic_idx)` to fetch the matching lkey.
pub fn run_server(ctx: Arc<KVServiceContext>, cfg: RdmaServerConfig) -> Result<()> {
    if cfg.devices.is_empty() {
        return Err(anyhow!("RdmaServerConfig.devices is empty"));
    }

    // ===== 1. Open the RdmaContext for every NIC =====
    let mut rdma_ctxs: Vec<Arc<RdmaContext>> = Vec::with_capacity(cfg.devices.len());
    for d in &cfg.devices {
        let c = Arc::new(RdmaContext::open(&d.device_name, d.port_num, d.gid_index)?);
        tracing::info!(
            "opened NIC {}: dev={} port={} gid_index={}",
            rdma_ctxs.len(),
            d.device_name,
            d.port_num,
            d.gid_index
        );
        rdma_ctxs.push(c);
    }

    // ===== 2. Pre-register the slab (once, shared host backing across all NICs) =====
    if cfg.rdma_slab_size_mb > 0 {
        match crate::rdma::slab::RdmaSlab::new(&rdma_ctxs, cfg.rdma_slab_size_mb * 1024 * 1024) {
            Ok(slab) => {
                let n_nics = slab.num_nics();
                if ctx.memory.set_rdma_slab(Arc::new(slab)).is_err() {
                    tracing::warn!("RDMA slab already set (run_server called twice?)");
                } else {
                    tracing::info!(
                        "RDMA slab injected into MemoryTier ({} MB, {} NIC{})",
                        cfg.rdma_slab_size_mb,
                        n_nics,
                        if n_nics == 1 { "" } else { "s" },
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    "RDMA slab registration failed ({}); falling back to per-GET reg_mr. \
                     Check `ulimit -l` / CAP_IPC_LOCK.",
                    e
                );
            }
        }
    } else {
        tracing::info!("RDMA slab disabled (rdma_slab_size_mb=0); using per-GET reg_mr path");
    }

    // ===== 3. Spawn a listener thread per NIC =====
    // With N>1, the main thread runs the first NIC (blocking); others are spawned
    // (daemon-style, ending when the process exits).
    // With N=1, behavior is equivalent to the old implementation.
    let n_nics = cfg.devices.len();
    let mut listener_threads = Vec::with_capacity(n_nics.saturating_sub(1));
    for nic_idx in 1..n_nics {
        let kv_ctx = ctx.clone();
        let rdma = rdma_ctxs[nic_idx].clone();
        let d = cfg.devices[nic_idx].clone();
        let h = thread::Builder::new()
            .name(format!("rdma-listener-{}", nic_idx))
            .spawn(move || {
                if let Err(e) = run_listener(kv_ctx, rdma, d, nic_idx) {
                    tracing::error!("RDMA listener nic_idx={} exited with error: {}", nic_idx, e);
                }
            })
            .map_err(|e| anyhow!("spawn listener nic_idx={}: {}", nic_idx, e))?;
        listener_threads.push(h);
    }

    // Main thread runs nic_idx=0
    let main_result = run_listener(ctx, rdma_ctxs[0].clone(), cfg.devices[0].clone(), 0);
    // When the main exits, the other listeners should end too (TcpListener::incoming blocks
    // and in practice won't exit on its own; the join here is for completeness — the OS reaps
    // them when the process exits.)
    for h in listener_threads {
        let _ = h.join();
    }
    main_result
}

/// Single-NIC listener loop: accept TCP, spawn a `handle_client` thread per client.
fn run_listener(
    ctx: Arc<KVServiceContext>,
    rdma_ctx: Arc<RdmaContext>,
    cfg: RdmaDeviceConfig,
    nic_idx: usize,
) -> Result<()> {
    let listener = TcpListener::bind(&cfg.tcp_listen)
        .map_err(|e| anyhow!("RDMA tcp bind {} failed: {}", cfg.tcp_listen, e))?;
    tracing::info!(
        "RDMA server listening on {} (nic_idx={} dev={} port={} gid_index={})",
        cfg.tcp_listen,
        nic_idx,
        cfg.device_name,
        cfg.port_num,
        cfg.gid_index
    );

    for stream_res in listener.incoming() {
        match stream_res {
            Ok(stream) => {
                let kv_ctx = ctx.clone();
                let rdma = rdma_ctx.clone();
                let port_num = cfg.port_num;
                let gid_index = cfg.gid_index;
                thread::spawn(move || {
                    let peer = stream
                        .peer_addr()
                        .map(|a| a.to_string())
                        .unwrap_or_default();
                    tracing::info!("RDMA client connected: {} (nic_idx={})", peer, nic_idx);
                    #[cfg(feature = "metrics")]
                    if let Some(metrics) = &kv_ctx.metrics {
                        metrics.change_rdma_connections(&format!("nic{}", nic_idx), 1);
                    }
                    let result =
                        handle_client(stream, kv_ctx.clone(), rdma, port_num, gid_index, nic_idx);
                    #[cfg(feature = "metrics")]
                    if let Some(metrics) = &kv_ctx.metrics {
                        metrics.change_rdma_connections(&format!("nic{}", nic_idx), -1);
                    }
                    if let Err(e) = result {
                        #[cfg(feature = "metrics")]
                        if let Some(metrics) = &kv_ctx.metrics {
                            metrics.record_rdma_error(&format!("nic{}", nic_idx), "disconnect");
                        }
                        tracing::warn!(
                            "RDMA client {} (nic_idx={}) disconnected: {}",
                            peer,
                            nic_idx,
                            e
                        );
                    }
                });
            }
            Err(e) => {
                tracing::warn!("accept error nic_idx={}: {}", nic_idx, e);
            }
        }
    }
    Ok(())
}

fn handle_client(
    mut stream: TcpStream,
    kv_ctx: Arc<KVServiceContext>,
    rdma: Arc<RdmaContext>,
    port_num: u8,
    gid_index: u8,
    nic_idx: usize,
) -> Result<()> {
    // ===== 1. Create a per-client CQ (avoids CQ-sharing races when multiple clients poll) =====
    let client_cq = unsafe {
        let cq_raw = rdma_sys::ibv_create_cq(
            rdma.ctx.as_ptr(),
            256,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
        NonNull::new(cq_raw).ok_or_else(|| anyhow!("ibv_create_cq for client failed"))?
    };
    // RAII: destroys this client's CQ on any `return` from inside the loop (the original code
    // ran cleanup after the loop, but the loop only exits via `return` — the cleanup never
    // executed → CQ leak). Guard must be declared BEFORE the qp so drop order is
    // qp(destroy_qp) → cq(destroy_cq), satisfying the verbs requirement (QP must be destroyed
    // before its CQ).
    let _cq_guard = CqGuard(client_cq);

    // ===== 2. Build the QP and exchange info (QP is bound to the per-client CQ) =====
    let qp = RcQp::new(&rdma, client_cq)?;
    qp.to_init(port_num)?;

    // Receive the client's hello first (avoids deadlock: client connects first, server recvs first)
    let remote = wire::recv_hello(&mut stream)?;
    wire::send_hello(&mut stream, &qp.local)?;

    qp.to_rtr(&remote, port_num, gid_index)?;
    qp.to_rts()?;
    tracing::info!(
        "RDMA QP established: local_qpn={} remote_qpn={}",
        qp.local.qpn,
        remote.qpn
    );

    // ===== 2. Serve-request loop =====
    loop {
        let tag_buf = match wire::read_exact(&mut stream, 1) {
            Ok(b) => b,
            Err(_) => {
                tracing::debug!("client closed");
                return Ok(());
            }
        };
        let tag = tag_buf[0];

        // Feed the tag back to recv_get_req: simplified — no putback, just branch on it
        if tag == 99 {
            // BYE
            tracing::debug!("client BYE");
            return Ok(());
        }
        // ===== PUT data path =====
        if tag == MSG_PUT_REQ || tag == MSG_PUT_IF_ABSENT_REQ {
            handle_put(&mut stream, &kv_ctx, nic_idx, tag == MSG_PUT_IF_ABSENT_REQ)?;
            continue;
        }
        if tag == MSG_GET_DESCRIPTOR_REQ {
            handle_descriptor_get(&mut stream, &kv_ctx, &rdma, &qp, client_cq, nic_idx)?;
            continue;
        }
        if tag != MSG_GET_REQ {
            return Err(anyhow!("unexpected tag in main loop: {}", tag));
        }

        // Re-read the get_req body (we already consumed the tag, reconstruct manually)
        let t_recv_start = std::time::Instant::now();
        let key_len_b = wire::read_exact(&mut stream, 2)?;
        let key_len = u16::from_le_bytes([key_len_b[0], key_len_b[1]]) as usize;
        let key_bytes = wire::read_exact(&mut stream, key_len)?;
        let key = String::from_utf8(key_bytes).map_err(|e| anyhow!("key utf8: {}", e))?;
        let dst_addr = u64::from_le_bytes(wire::read_exact(&mut stream, 8)?.try_into().unwrap());
        let dst_rkey = u32::from_le_bytes(wire::read_exact(&mut stream, 4)?.try_into().unwrap());
        let max_size = u64::from_le_bytes(wire::read_exact(&mut stream, 8)?.try_into().unwrap());
        let t_recv_done = std::time::Instant::now();

        // ===== 3. Look up chunks_cache =====
        let kv_key = parse_string_key(&key)?;
        let t_lookup_done = std::time::Instant::now();

        // **DIAGNOSTIC TOGGLE**: CS_FORCE_DISK_READ=1 forces skipping slab cache and going down
        // the real disk-read path. Used for benchmarking real 8-disk stripe read performance
        // by excluding cache-hit interference.
        let force_disk_read = std::env::var("CS_FORCE_DISK_READ").ok().as_deref() == Some("1");

        // Fast path: slab-backed entry → single lkey, zero registration, contiguous data can
        // be coalesced into a (near) single WRITE.
        // Fallback: heap-backed / no slab injected → per-chunk temporary `reg_mr` (preserves
        // original behavior, correctness safety net).
        let mut slab_post_us: u64 = 0;
        let mut slab_poll_us: u64 = 0;
        let mut slab_hit = false;
        let mut fb_storage_get_us: u64 = 0;
        let mut fb_reg_post_us: u64 = 0;
        let mut fb_poll_us: u64 = 0;
        let mut fb_n_chunks: usize = 0;
        let cache_lookup = if force_disk_read {
            #[cfg(feature = "metrics")]
            if let Some(metrics) = &kv_ctx.metrics {
                metrics.record_force_disk_read();
            }
            None // Force skip cache, go straight to storage read
        } else {
            kv_ctx.memory.get_chunks_slab(&kv_key, nic_idx)
        };
        let (found, bytes_written, num_chunks) = match cache_lookup {
            Some(placement) => {
                slab_hit = true;
                #[cfg(feature = "metrics")]
                if let Some(metrics) = &kv_ctx.metrics {
                    metrics.record_cache_hit("slab");
                }
                let (f, b, c, post_us, poll_us) =
                    serve_get_slab(&qp, client_cq, &placement, dst_addr, dst_rkey, max_size)?;
                slab_post_us = post_us;
                slab_poll_us = poll_us;
                // Explicitly release the pin (drop placement) AFTER poll completes; prevents
                // reclamation while the NIC is still reading.
                drop(placement);
                (f, b, c)
            }
            None => {
                // ===== cache miss path =====
                // Optimization: slab.alloc → storage.get_into_ptr (zero intermediate buffer)
                //               → slab path posts RDMA WRITE (zero reg_mr). Perfectly
                //               symmetric with the PUT path.
                // On failure, fall back to the old serve_get_fallback (per-chunk reg_mr,
                // compatibility safety net).
                let t_storage_start = std::time::Instant::now();
                let slab_path_result = try_serve_get_via_slab(
                    &kv_ctx, &qp, client_cq, &kv_key, dst_addr, dst_rkey, max_size, nic_idx,
                );
                let storage_get_us = t_storage_start.elapsed().as_micros() as u64;
                fb_storage_get_us = storage_get_us;

                match slab_path_result {
                    Ok(Some((bytes, post_us, poll_us))) => {
                        // Slab path succeeded: note reg_post=0 (zero reg_mr!)
                        fb_reg_post_us = 0;
                        fb_poll_us = poll_us;
                        fb_n_chunks = 1; // slab path uses a single WRITE
                        slab_post_us = post_us; // reuse the slab_post_us field for logging
                        tracing::debug!("RDMA GET SLAB-MISS-FAST key={} bytes={}", key, bytes);
                        (true, bytes, 1u32)
                    }
                    Ok(None) => {
                        // Key does not exist
                        tracing::debug!("RDMA GET MISS key={}", key);
                        (false, 0u64, 0u32)
                    }
                    Err(e) => {
                        // Slab path failed (slab full / I/O error) → fall back to the old path
                        tracing::warn!(
                            "RDMA GET slab fast path failed, fallback to per-chunk reg_mr: {}",
                            e
                        );
                        #[cfg(feature = "metrics")]
                        if let Some(metrics) = &kv_ctx.metrics {
                            metrics.record_fallback(
                                "rdma_slab",
                                "rdma_per_chunk",
                                "slab_fast_path_failed",
                            );
                        }
                        let storage_result = kv_ctx.memory.get_chunks(&kv_key);
                        match storage_result {
                            Ok(Some((segments, _meta))) => {
                                fb_n_chunks = segments.len();
                                let (f, b, c, reg_post_us, poll_us) = serve_get_fallback(
                                    &rdma, &qp, client_cq, &segments, dst_addr, dst_rkey, max_size,
                                )?;
                                fb_reg_post_us = reg_post_us;
                                fb_poll_us = poll_us;
                                (f, b, c)
                            }
                            Ok(None) => (false, 0u64, 0u32),
                            Err(e2) => {
                                tracing::warn!("chunks_cache get error: {}", e2);
                                (false, 0u64, 0u32)
                            }
                        }
                    }
                }
            }
        };
        let t_serve_done = std::time::Instant::now();

        if found {
            tracing::debug!(
                target: "contextstore_server::storage_io",
                event = "rdma_get_complete",
                status = "ok",
                source = if slab_hit { "memory_tier" } else { "storage_tier" },
                bytes = bytes_written,
                chunks = num_chunks,
                force_disk_read,
            );
        }

        // ===== 5. Send response =====
        wire::send_get_resp(
            &mut stream,
            &GetRespMsg {
                found,
                bytes_written,
                num_chunks,
            },
        )?;
        let t_send_done = std::time::Instant::now();

        #[cfg(feature = "metrics")]
        if let Some(metrics) = &kv_ctx.metrics {
            if bytes_written > 0 {
                let nic = format!("nic{}", nic_idx);
                metrics.record_rdma_bytes(&nic, "tx", bytes_written);
                let transfer_us = if slab_poll_us > 0 {
                    slab_poll_us
                } else {
                    fb_poll_us
                };
                if transfer_us > 0 {
                    metrics.record_rdma_transfer_duration(
                        &nic,
                        "tx",
                        transfer_us as f64 / 1_000_000.0,
                    );
                }
            }
        }

        // Diagnostic: per-GET stage breakdown (trace level, only emitted with
        // RUST_LOG=contextstore_server::rdma=trace).
        // recv_us = TCP read req; lookup_us = chunks_cache lookup; post/poll = RDMA WR
        // submit/complete; send_us = TCP write resp. Used to locate where each part of the
        // 297ms get_into observed on the vLLM side is spent.
        // Format: PERF gid=<gid> bytes=<N> recv_us=X lookup_us=X post_us=X poll_us=X send_us=X total_us=X slab=<bool>
        if slab_hit && bytes_written > 0 {
            tracing::trace!(
                "PERF bytes={} recv_us={} lookup_us={} post_us={} poll_us={} send_us={} total_us={} slab=true",
                bytes_written,
                t_recv_done.duration_since(t_recv_start).as_micros(),
                t_lookup_done.duration_since(t_recv_done).as_micros(),
                slab_post_us,
                slab_poll_us,
                t_send_done.duration_since(t_serve_done).as_micros(),
                t_send_done.duration_since(t_recv_start).as_micros(),
            );
        } else if !slab_hit && bytes_written > 0 {
            // fallback path breakdown: storage_get (disk IO) + reg_post (8× ibv_reg_mr + post_write) + poll (RDMA WRITE complete)
            tracing::trace!(
                "PERF bytes={} recv_us={} lookup_us={} storage_get_us={} reg_post_us={} poll_us={} send_us={} total_us={} slab=false n_chunks={}",
                bytes_written,
                t_recv_done.duration_since(t_recv_start).as_micros(),
                t_lookup_done.duration_since(t_recv_done).as_micros(),
                fb_storage_get_us,
                fb_reg_post_us,
                fb_poll_us,
                t_send_done.duration_since(t_serve_done).as_micros(),
                t_send_done.duration_since(t_recv_start).as_micros(),
                fb_n_chunks,
            );
        }
    }
    // The loop only exits via an inner `return`; the CQ is destroyed by `_cq_guard` on exit via RAII.
}

/// RAII guard holding the per-client CQ; `ibv_destroy_cq` on Drop. Ensures the CQ is released
/// (no leak) whenever `handle_client` exits (client BYE / protocol error / I/O error).
struct CqGuard(NonNull<rdma_sys::ibv_cq>);

impl Drop for CqGuard {
    fn drop(&mut self) {
        unsafe {
            rdma_sys::ibv_destroy_cq(self.0.as_ptr());
        }
    }
}

/// Slab fast path: the data lives in a single pre-registered region, so we post an RDMA WRITE
/// using the slab's lkey with **zero reg syscalls**. The data is contiguous — typically a
/// single WRITE suffices; only when `len` exceeds `MAX_WRITE_BYTES` do we split into multiple
/// WRITEs from the same extent.
///
/// Returns `(found, bytes_written, num_chunks, post_us, poll_us)`:
/// - post_us: from entry to all WRs submitted (including metadata checks; usually < 1ms)
/// - poll_us: time spent in `poll_n` waiting for NIC completion (dominates actual RDMA WRITE time)
///
/// Used to diagnose the NIC-time vs. other-overhead split within the 297ms get_into observed on the vLLM side.
fn serve_get_slab(
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    placement: &SlabPlacement,
    dst_addr: u64,
    dst_rkey: u32,
    max_size: u64,
) -> Result<(bool, u64, u32, u64, u64)> {
    let total = placement.view.len;
    if total > max_size {
        tracing::warn!("client buf too small: total={} max={}", total, max_size);
        return Ok((false, 0, 0, 0, 0));
    }
    if total == 0 {
        return Ok((true, 0, 0, 0, 0));
    }

    let src_base = placement.view.addr;
    let lkey = placement.view.lkey;
    // Split by MAX_WRITE_BYTES (most values are < 1GiB, so a single WRITE is emitted).
    let n_writes = total.div_ceil(MAX_WRITE_BYTES);
    let mut offset: u64 = 0;
    let mut idx: u64 = 0;

    let t_post_start = std::time::Instant::now();
    while offset < total {
        let len = (total - offset).min(MAX_WRITE_BYTES);
        let signaled = idx + 1 == n_writes; // Only signal on the last WRITE (RC guarantees prior completions)
        qp.post_write(
            idx,
            src_base + offset,
            lkey,
            dst_addr + offset,
            dst_rkey,
            len as u32,
            signaled,
        )?;
        offset += len;
        idx += 1;
    }
    let t_poll_start = std::time::Instant::now();
    RcQp::poll_n(client_cq, 1)?;
    let t_poll_done = std::time::Instant::now();

    let post_us = t_poll_start.duration_since(t_post_start).as_micros() as u64;
    let poll_us = t_poll_done.duration_since(t_poll_start).as_micros() as u64;
    // The chunk count is display-only for the client; the slab path has coalesced,
    // and we report the number of splits.
    Ok((true, total, n_writes as u32, post_us, poll_us))
}

/// Fallback path: heap-backed entry, temporarily `register_mr_raw` (= ibv_reg_mr) each chunk
/// then WRITE. Preserves the pre-slab behavior as a compiled-in safety net (used when the
/// slab is not injected / is full).
fn serve_get_fallback(
    rdma: &RdmaContext,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    segments: &[prost::bytes::Bytes],
    dst_addr: u64,
    dst_rkey: u32,
    max_size: u64,
) -> Result<(bool, u64, u32, u64, u64)> {
    let total_size: u64 = segments.iter().map(|b| b.len() as u64).sum();
    if total_size > max_size {
        tracing::warn!(
            "client buf too small: total={} max={}",
            total_size,
            max_size
        );
        return Ok((false, 0, 0, 0, 0));
    }
    let n = segments.len();
    let mut offset: u64 = 0;
    // Hold the MR until poll completes (drop = dereg).
    let mut mrs = Vec::with_capacity(n);
    let t_reg_post_start = std::time::Instant::now();
    for (i, seg) in segments.iter().enumerate() {
        if seg.is_empty() {
            continue;
        }
        // RDMA WRITE source side only needs LOCAL access; the LOCAL_WRITE flag matches the slab path convention.
        let mr = unsafe {
            rdma.register_mr_raw(
                seg.as_ptr() as *mut u8,
                seg.len(),
                ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0,
            )?
        };
        let signaled = i + 1 == n; // Only signal on the last one
        qp.post_write(
            i as u64,
            mr.addr,
            mr.lkey,
            dst_addr + offset,
            dst_rkey,
            seg.len() as u32,
            signaled,
        )?;
        offset += seg.len() as u64;
        mrs.push(mr);
    }
    let t_poll_start = std::time::Instant::now();
    // Wait for the last WRITE to complete (RC guarantees prior ones did too).
    RcQp::poll_n(client_cq, 1)?;
    let t_poll_done = std::time::Instant::now();
    // At this point mrs drop and dereg.
    let reg_post_us = t_poll_start.duration_since(t_reg_post_start).as_micros() as u64;
    let poll_us = t_poll_done.duration_since(t_poll_start).as_micros() as u64;
    Ok((true, total_size, n as u32, reg_post_us, poll_us))
}

/// **RDMA GET cache-miss fast path** — perfectly symmetric with the PUT path (zero reg_mr).
///
/// Flow:
/// 1. Query metadata for BlockMeta (get the real size + striping info).
/// 2. slab.alloc(size) → SlabExtent (4K aligned, pre-registered MR).
/// 3. storage.get_into_ptr(extent.ptr) → O_DIRECT pread straight into slab (zero intermediate buffer).
/// 4. Post RDMA WRITE using the slab's pre-registered lkey (zero reg_mr!).
/// 5. **insert_chunks_from_slab injects into L1 cache** → subsequent GETs hit the slab fast path immediately.
///
/// Difference vs. the old serve_get_fallback:
/// - Old: read into heap × 8 → 8× ibv_reg_mr (~33ms) → 8 SQEs post_write → poll
/// - New: read into slab → 0 reg_mr → 1 SQE post_write → poll
///
/// Returns:
/// - Ok(Some((bytes, post_us, poll_us))): RDMA WRITE succeeded
/// - Ok(None): key does not exist (metadata miss)
/// - Err(e): slab full / I/O error / protocol error → caller falls back to fallback
fn try_serve_get_via_slab(
    kv_ctx: &Arc<KVServiceContext>,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    kv_key: &ObjectKey,
    dst_addr: u64,
    dst_rkey: u32,
    max_size: u64,
    nic_idx: usize,
) -> Result<Option<(u64, u64, u64)>> {
    // 1. Look up metadata to get size
    let meta = match kv_ctx.metadata.get_block(&kv_key.to_string_key())? {
        Some(m) => m,
        None => return Ok(None),
    };
    try_serve_get_via_slab_with_meta(
        kv_ctx, qp, client_cq, kv_key, &meta, dst_addr, dst_rkey, max_size, nic_idx,
    )
}

fn try_serve_get_via_slab_with_meta(
    kv_ctx: &Arc<KVServiceContext>,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    kv_key: &ObjectKey,
    meta: &BlockMeta,
    dst_addr: u64,
    dst_rkey: u32,
    max_size: u64,
    nic_idx: usize,
) -> Result<Option<(u64, u64, u64)>> {
    // ===== Stage timers =====
    let t_total_start = std::time::Instant::now();
    // wall-clock origin (ms since boot), used to align timelines across workers
    let wall_start_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);

    let size = meta.size as usize;
    if size == 0 {
        return Ok(Some((0, 0, 0)));
    }
    if size as u64 > max_size {
        return Err(anyhow!(
            "client buf too small: size={} max={}",
            size,
            max_size
        ));
    }
    let t_meta = t_total_start.elapsed().as_micros() as u64;

    // 2. slab.alloc(size); on failure evict then retry (same pattern as handle_put: multi-round + exponential evict)
    let t_alloc_start = std::time::Instant::now();
    let slab = kv_ctx
        .memory
        .rdma_slab_get()
        .ok_or_else(|| anyhow!("rdma slab not set"))?;
    let extent = match slab.alloc(size) {
        Some(e) => e,
        None => {
            // Multi-round retry: alloc fail → evict 2× size → alloc again; up to 5 rounds
            let slab_cap = slab.capacity();
            let mut found = None;
            let mut evict_mult = 2usize;
            for _retry in 0..5 {
                let evict_target = (size * evict_mult).min(slab_cap);
                kv_ctx
                    .memory
                    .evict_chunks_cache_to_free(evict_target, slab_cap);
                if let Some(e) = slab.alloc(size) {
                    found = Some(e);
                    break;
                }
                evict_mult *= 2;
            }
            found.ok_or_else(|| anyhow!("slab full after 5 evict rounds (need {} bytes)", size))?
        }
    };
    let t_alloc = t_alloc_start.elapsed().as_micros() as u64;

    let extent_ptr = extent.as_ptr() as *mut u8;
    let extent_cap = extent.capacity_bytes(); // 4K aligned, ≥ size

    // 3. **Pipeline**: storage stream kicks off N-stripe pread; on each completion, immediately post the RDMA WRITE.
    let t_stream_start = std::time::Instant::now();
    let stream_result = kv_ctx
        .storage
        .get_into_ptr_stream_with_meta(kv_key, meta, extent_ptr, extent_cap);
    let stream_rx = match stream_result {
        Ok(Some((_real_meta, rx))) => rx,
        Ok(None) => return Ok(None),
        Err(e) => return Err(anyhow!("storage.get_into_ptr_stream: {}", e)),
    };
    let t_stream_setup = t_stream_start.elapsed().as_micros() as u64;

    // 4. Consume each stripe-completion event, immediately posting the RDMA WRITE for that segment
    let view = extent.view(nic_idx);
    let t_post_start = std::time::Instant::now();
    let mut n_writes_posted = 0u64;
    let mut last_wr_id = 0u64;
    let mut had_error: Option<String> = None;
    // Track each stripe's completion time relative to stream_start along with its stripe_idx
    let mut stripe_finish_times_us: Vec<(u64, usize)> = Vec::with_capacity(8);

    while let Ok((stripe_idx, offset_in_value, stripe_len, result)) = stream_rx.recv() {
        let t_now_us = t_stream_start.elapsed().as_micros() as u64;
        stripe_finish_times_us.push((t_now_us, stripe_idx));
        match result {
            Ok(_bytes_read) => {
                qp.post_write(
                    stripe_idx as u64,
                    view.addr + offset_in_value as u64,
                    view.lkey,
                    dst_addr + offset_in_value as u64,
                    dst_rkey,
                    stripe_len as u32,
                    true, // signaled
                )?;
                n_writes_posted += 1;
                last_wr_id = stripe_idx as u64;
            }
            Err(e) => {
                had_error = Some(format!("{}", e));
                break;
            }
        }
    }
    let t_post_done = t_post_start.elapsed().as_micros() as u64;

    if let Some(e) = had_error {
        return Err(anyhow!("stripe read failed: {}", e));
    }
    if n_writes_posted == 0 {
        return Err(anyhow!("no stripes posted"));
    }

    // 5. Poll all N WRITE completions (each is signaled)
    let t_poll_start = std::time::Instant::now();
    RcQp::poll_n(client_cq, n_writes_posted as usize)?;
    let t_poll_done = std::time::Instant::now();
    let _ = last_wr_id;

    let post_us = t_post_done;
    let poll_us = t_poll_done.duration_since(t_poll_start).as_micros() as u64;

    // 6. Inject into chunks_cache (slab-backed) so subsequent GETs cache-hit.
    // **DIAGNOSTIC TOGGLE**: with CS_FORCE_DISK_READ=1 we skip injection, so the next GET
    // still takes the cache-miss path.
    let force_disk_read = std::env::var("CS_FORCE_DISK_READ").ok().as_deref() == Some("1");
    let t_inject_start = std::time::Instant::now();
    let extent_arc = Arc::new(extent);
    if !force_disk_read {
        kv_ctx
            .memory
            .insert_chunks_from_slab(kv_key.to_string_key(), extent_arc, meta.clone());
    }
    let t_inject = t_inject_start.elapsed().as_micros() as u64;

    // **DETAILED TRACE**: break down every stage to see where time goes
    // **DETAILED TRACE**: break down every stage to see where time goes
    // Emit [(ms@stripe_idx)...] so we can spot which stripes (= which disks) are slow
    let stripe_finish_str = stripe_finish_times_us
        .iter()
        .map(|(t, idx)| format!("{}@{}", t / 1000, idx))
        .collect::<Vec<_>>()
        .join(",");
    tracing::info!(
        "MISS_DETAIL wall_us={} key={} bytes={} meta_us={} alloc_us={} stream_setup_us={} \
         post_done_us={} poll_us={} inject_us={} stripe_finish_ms=[{}] n_writes={}",
        wall_start_us,
        kv_key.to_string_key(),
        size,
        t_meta,
        t_alloc,
        t_stream_setup,
        post_us,
        poll_us,
        t_inject,
        stripe_finish_str,
        n_writes_posted,
    );

    Ok(Some((size as u64, post_us, poll_us)))
}

/// Parse the client-provided canonical string key back into an ObjectKey.
fn parse_string_key(s: &str) -> Result<ObjectKey> {
    ObjectKey::from_string_key(s).map_err(|e| anyhow!("invalid key format: {} ({})", s, e))
}

fn meta_matches_descriptor(meta: &BlockMeta, req: &DescriptorGetReqMsg) -> bool {
    meta.object_handle == req.object_handle
        && meta.object_generation == req.object_generation
        && meta.content_etag == req.content_etag
        && meta.layout_version == req.layout_version
        && meta.size == req.size
}

fn descriptor_meta_from_req(
    kv_ctx: &Arc<KVServiceContext>,
    key: &ObjectKey,
    req: &DescriptorGetReqMsg,
) -> Result<BlockMeta> {
    if req.object_handle.is_empty() {
        return Err(anyhow!("descriptor missing object_handle"));
    }
    if req.object_generation == 0 || req.layout_version == 0 {
        return Err(anyhow!(
            "descriptor has invalid version: generation={} layout={}",
            req.object_generation,
            req.layout_version
        ));
    }

    let mut meta = BlockMeta {
        device_id: kv_ctx.router.route(key) as u32,
        file_path: String::new(),
        size: req.size,
        object_handle: req.object_handle.clone(),
        object_generation: req.object_generation,
        content_etag: req.content_etag.clone(),
        layout_version: req.layout_version,
        created_at: 0,
        last_accessed_at: 0,
        ttl_seconds: 0,
        num_tokens: 0,
        num_layers: 0,
        dtype: "bytes".to_string(),
        compressed: false,
        striping: None,
    };

    if req.is_striped {
        if req.stripe_count == 0 || req.chunk_size == 0 {
            return Err(anyhow!("striped descriptor missing stripe layout"));
        }
        let mut chunk_devices = Vec::with_capacity(req.stripe_count as usize);
        let mut chunk_paths = Vec::with_capacity(req.stripe_count as usize);
        for i in 0..req.stripe_count as usize {
            let dev_id = kv_ctx.router.chunk_device(key, i);
            let path = kv_ctx.router.chunk_versioned_path(
                key,
                i,
                dev_id,
                req.object_generation,
                req.layout_version,
            );
            chunk_devices.push(dev_id as u32);
            chunk_paths.push(path.to_string_lossy().to_string());
        }
        meta.device_id = chunk_devices[0];
        meta.striping = Some(StripingInfo {
            chunk_size: req.chunk_size,
            chunk_devices,
            chunk_paths,
            total_size: req.size,
            chunk_locations: Vec::new(),
        });
    } else {
        let device_id = kv_ctx.router.route(key);
        meta.device_id = device_id as u32;
        meta.file_path = kv_ctx
            .router
            .key_to_versioned_path(key, device_id, req.object_generation, req.layout_version)
            .to_string_lossy()
            .to_string();
    }

    Ok(meta)
}

fn handle_descriptor_get(
    stream: &mut TcpStream,
    kv_ctx: &Arc<KVServiceContext>,
    rdma: &Arc<RdmaContext>,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    nic_idx: usize,
) -> Result<()> {
    let req = wire::recv_descriptor_get_req_body(stream)?;
    let kv_key = parse_string_key(&req.key)?;
    let descriptor_meta = descriptor_meta_from_req(kv_ctx, &kv_key, &req)?;

    let active_meta = match kv_ctx.metadata.get_block(&kv_key.to_string_key())? {
        Some(meta) => meta,
        None => {
            wire::send_get_resp(
                stream,
                &GetRespMsg {
                    found: false,
                    bytes_written: 0,
                    num_chunks: 0,
                },
            )?;
            return Ok(());
        }
    };
    if !meta_matches_descriptor(&active_meta, &req) {
        wire::send_get_resp(
            stream,
            &GetRespMsg {
                found: false,
                bytes_written: 0,
                num_chunks: 0,
            },
        )?;
        return Ok(());
    }

    let force_disk_read = std::env::var("CS_FORCE_DISK_READ").ok().as_deref() == Some("1");
    let cache_lookup = if force_disk_read {
        None
    } else {
        kv_ctx.memory.get_chunks_slab(&kv_key, nic_idx)
    };

    let (found, bytes_written, num_chunks) = match cache_lookup {
        Some(placement) if meta_matches_descriptor(&placement.meta, &req) => {
            let (found, bytes, chunks, _post_us, _poll_us) = serve_get_slab(
                qp,
                client_cq,
                &placement,
                req.dst_addr,
                req.dst_rkey,
                req.max_size,
            )?;
            drop(placement);
            (found, bytes, chunks)
        }
        _ => match try_serve_get_via_slab_with_meta(
            kv_ctx,
            qp,
            client_cq,
            &kv_key,
            &descriptor_meta,
            req.dst_addr,
            req.dst_rkey,
            req.max_size,
            nic_idx,
        ) {
            Ok(Some((bytes, _post_us, _poll_us))) => (true, bytes, 1u32),
            Ok(None) => (false, 0u64, 0u32),
            Err(e) => {
                tracing::warn!(
                    "RDMA descriptor GET slab path failed, fallback to per-chunk reg_mr: {}",
                    e
                );
                match kv_ctx
                    .storage
                    .get_chunks_with_meta(&kv_key, &descriptor_meta)
                {
                    Ok(Some((segments, _meta))) => {
                        let (found, bytes, chunks, _reg_post_us, _poll_us) = serve_get_fallback(
                            rdma,
                            qp,
                            client_cq,
                            &segments,
                            req.dst_addr,
                            req.dst_rkey,
                            req.max_size,
                        )?;
                        (found, bytes, chunks)
                    }
                    Ok(None) => (false, 0u64, 0u32),
                    Err(e2) => {
                        tracing::warn!("RDMA descriptor GET fallback failed: {}", e2);
                        (false, 0u64, 0u32)
                    }
                }
            }
        },
    };

    wire::send_get_resp(
        stream,
        &GetRespMsg {
            found,
            bytes_written,
            num_chunks,
        },
    )?;
    Ok(())
}

/// Handle a PUT request (tag MSG_PUT_REQ has already been consumed by the caller).
///
/// Flow:
/// 1. recv PutReq {key, size}
/// 2. slab.alloc(size) → SlabExtent
/// 3. send PutReady {ok, dst_addr, dst_rkey} (tells the client where to WRITE)
/// 4. recv PutCommit (tag-only; the client confirms RDMA WRITE completed)
/// 5. storage.put_from_ptr(extent.as_ptr(), size) — pwrite from slab straight to NVMe (zero memcpy)
/// 6. send PutResp {ok}
///
/// Key point: `extent` is held on the stack (let extent), not placed in chunks_cache (avoids
/// introducing GET-path complexity); when the fn returns Drop returns it to the slab. Reads for
/// this key subsequently go through the L2 storage path (already on NVMe).
///
/// Note: does not write L1 chunks_cache. If PUT-then-immediate-GET hit rate matters, we could
/// synchronously inject L1 via memory_tier.put_chunks_from_slab after put_from_ptr completes
/// (not in this iteration; let GETs naturally miss → L2 → L1).
fn handle_put(
    stream: &mut TcpStream,
    kv_ctx: &Arc<KVServiceContext>,
    nic_idx: usize,
    if_not_exists: bool,
) -> Result<()> {
    let t_recv_start = std::time::Instant::now();
    let put_req = wire::recv_put_req_body(stream)?;
    let t_recv_done = std::time::Instant::now();

    let size = put_req.size as usize;
    if size == 0 || size > 4 * 1024 * 1024 * 1024 {
        // 0 bytes / >4GB: reject (avoid huge allocations causing slab fragmentation)
        tracing::warn!("RDMA PUT rejected: invalid size {}", size);
        wire::send_put_ready(
            stream,
            &PutReadyMsg {
                ok: false,
                dst_addr: 0,
                dst_rkey: 0,
            },
        )?;
        wire::send_put_resp(stream, &PutRespMsg { ok: false })?;
        return Ok(());
    }

    // ===== 1. Allocate destination memory from the slab =====
    let extent: SlabExtent = {
        let slab_opt = kv_ctx.memory.rdma_slab_get();
        let try_alloc = || -> Option<SlabExtent> { slab_opt.as_ref().and_then(|s| s.alloc(size)) };
        // Multi-round retry: alloc fail → evict 2× size → alloc again; up to 5 rounds
        // (cumulative evict 10× size).
        // A single round of eviction may fall short due to slab fragmentation: pop_lru may
        // release non-contiguous extents so best-fit still fails.
        // Multi-round + exponential evict volume: after cumulatively releasing 10× size and
        // still failing, there's truly no hope.
        let extent_opt = try_alloc().or_else(|| {
            if let Some(slab) = slab_opt.as_ref() {
                let slab_cap = slab.capacity();
                let mut evict_mult = 2usize;
                for retry in 0..5 {
                    let evict_target = (size * evict_mult).min(slab_cap);
                    kv_ctx
                        .memory
                        .evict_chunks_cache_to_free(evict_target, slab_cap);
                    if let Some(e) = try_alloc() {
                        if retry > 0 {
                            tracing::debug!(
                                "RDMA PUT slab alloc succeeded after {} evict rounds (evict_mult={})",
                                retry + 1, evict_mult
                            );
                        }
                        return Some(e);
                    }
                    evict_mult *= 2;
                }
                None
            } else {
                None
            }
        });
        match extent_opt {
            Some(e) => e,
            None => {
                tracing::warn!("RDMA PUT rejected: slab full or not set (size={})", size);
                wire::send_put_ready(
                    stream,
                    &PutReadyMsg {
                        ok: false,
                        dst_addr: 0,
                        dst_rkey: 0,
                    },
                )?;
                // We don't send PUT_RESP; per protocol the client sees PutReadyMsg.ok=false
                // and bails out (no COMMIT).
                // To guard against protocol drift where the client blocks on
                // recv_put_resp, still send resp{ok=false}.
                wire::send_put_resp(stream, &PutRespMsg { ok: false })?;
                return Ok(());
            }
        }
    };
    let dst_addr = extent.addr();
    let dst_rkey = extent.rkey(nic_idx);
    let t_alloc_done = std::time::Instant::now();

    // ===== 2. Tell the client the destination address; wait for it to send COMMIT after its RDMA WRITE completes =====
    wire::send_put_ready(
        stream,
        &PutReadyMsg {
            ok: true,
            dst_addr,
            dst_rkey,
        },
    )?;

    // ===== 3. Block on COMMIT (client has already polled its WRITE to completion) =====
    let commit_tag = wire::read_exact(stream, 1)?[0];
    if commit_tag != MSG_PUT_COMMIT {
        return Err(anyhow!(
            "expected MSG_PUT_COMMIT={}, got {}",
            MSG_PUT_COMMIT,
            commit_tag
        ));
    }
    let t_commit_done = std::time::Instant::now();

    // ===== 4. pwrite O_DIRECT from slab straight to NVMe (zero memcpy) =====
    let kv_key = parse_string_key(&put_req.key)?;
    let meta = crate::metadata::BlockMeta {
        device_id: 0, // put_from_ptr will overwrite
        file_path: String::new(),
        size: 0,
        object_handle: String::new(),
        object_generation: 1,
        content_etag: String::new(),
        layout_version: 1,
        created_at: chrono::Utc::now().timestamp(),
        last_accessed_at: chrono::Utc::now().timestamp(),
        ttl_seconds: 0,
        num_tokens: 0,
        num_layers: 1,
        dtype: "uint8".to_string(),
        compressed: false,
        striping: None,
    };
    let put_result = if if_not_exists {
        kv_ctx
            .storage
            .put_from_ptr_if_absent(&kv_key, extent.as_ptr(), size, meta)
    } else {
        kv_ctx
            .storage
            .put_from_ptr(&kv_key, extent.as_ptr(), size, meta)
            .map(|_| true)
    };
    let t_disk_done = std::time::Instant::now();

    // Disk write failed → resp ok=false; extent drops, returning to slab.
    let result_code = match put_result {
        Ok(true) => PUT_RESULT_STORED,
        Ok(false) => PUT_RESULT_EXISTS,
        Err(e) => {
            tracing::warn!("RDMA PUT pwrite failed key={}: {}", put_req.key, e);
            PUT_RESULT_FAILED
        }
    };
    let ok = result_code == PUT_RESULT_STORED;

    // ===== 4.5 Inject into L1 chunks_cache (slab-backed) =====
    // Lets subsequent GETs take the slab fast path (~11 GB/s) instead of the fallback
    // storage_tier (~0.3 GB/s).
    // Failing to inject is not a PUT failure (data is on disk; only follow-on GET performance
    // is affected).
    // Must happen before resp ok: once the client sees resp, an immediate GET must hit.
    let mut cache_inject_us: u64 = 0;
    if ok {
        let t_inject_start = std::time::Instant::now();
        // Re-fetch BlockMeta from metadata (with correct striping/file_path) so that when a
        // GET hits cache, valid meta is available. One metadata fetch is a single-key metadata
        // query, microsecond-scale overhead.
        let real_meta_opt = kv_ctx.metadata.get_block(&put_req.key);
        match real_meta_opt {
            Ok(Some(real_meta)) => {
                let extent_arc = Arc::new(extent);
                kv_ctx
                    .memory
                    .insert_chunks_from_slab(put_req.key.clone(), extent_arc, real_meta);
                cache_inject_us = t_inject_start.elapsed().as_micros() as u64;
            }
            Ok(None) => {
                tracing::warn!(
                    "RDMA PUT meta lookup miss after disk write key={}",
                    put_req.key
                );
                // Explicit drop returns extent to slab (no cache injection)
                drop(extent);
            }
            Err(e) => {
                tracing::warn!("RDMA PUT meta lookup err key={}: {}", put_req.key, e);
                drop(extent);
            }
        }
    } else {
        // Failure path: explicitly drop extent to return it to slab
        drop(extent);
    }

    // ===== 5. Send final ok =====
    if if_not_exists {
        wire::send_put_result(stream, result_code)?;
    } else {
        wire::send_put_resp(stream, &PutRespMsg { ok })?;
    }
    let t_resp_done = std::time::Instant::now();

    // PUT_PERF diagnostic (trace level, only emitted with RUST_LOG=contextstore_server::rdma=trace).
    // recv: TCP put_req; alloc: slab.alloc; wait_commit: network RTT + client RDMA WRITE;
    // disk: storage.put_from_ptr (= 8-stripe O_DIRECT pwrite); inject: L1 chunks_cache insert;
    // send_resp: send final ok.
    tracing::trace!(
        "PUT_PERF bytes={} recv_us={} alloc_us={} wait_commit_us={} disk_us={} inject_us={} resp_us={} total_us={} ok={}",
        size,
        t_recv_done.duration_since(t_recv_start).as_micros(),
        t_alloc_done.duration_since(t_recv_done).as_micros(),
        t_commit_done.duration_since(t_alloc_done).as_micros(),
        t_disk_done.duration_since(t_commit_done).as_micros(),
        cache_inject_us,
        t_resp_done.duration_since(t_disk_done).as_micros(),
        t_resp_done.duration_since(t_recv_start).as_micros(),
        ok,
    );

    // extent already moved into cache (success) or explicitly dropped (failure); nothing to do here
    Ok(())
}
