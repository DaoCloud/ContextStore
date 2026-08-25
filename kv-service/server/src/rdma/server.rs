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
    self, DescriptorGetReqMsg, GetRespMsg, PutReadyMsg, PutRespMsg, PutStripeLocation,
    PutStripesRespMsg, MSG_GET_DESCRIPTOR_REQ, MSG_GET_DESCRIPTOR_STRIPES_REQ,
    MSG_GET_DESCRIPTOR_STRIPES_SGE_REQ, MSG_GET_REQ,
    MSG_PUT_COMMIT, MSG_PUT_IF_ABSENT_REQ, MSG_PUT_IF_ABSENT_WITH_OPTIONS_REQ, MSG_PUT_REQ,
    MSG_PUT_STRIPES_REQ, MSG_PUT_WITH_OPTIONS_REQ, PUT_RESULT_EXISTS, PUT_RESULT_FAILED,
    PUT_RESULT_STORED,
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
        if tag == MSG_PUT_REQ
            || tag == MSG_PUT_IF_ABSENT_REQ
            || tag == MSG_PUT_WITH_OPTIONS_REQ
            || tag == MSG_PUT_IF_ABSENT_WITH_OPTIONS_REQ
        {
            let if_not_exists =
                tag == MSG_PUT_IF_ABSENT_REQ || tag == MSG_PUT_IF_ABSENT_WITH_OPTIONS_REQ;
            let with_options =
                tag == MSG_PUT_WITH_OPTIONS_REQ || tag == MSG_PUT_IF_ABSENT_WITH_OPTIONS_REQ;
            handle_put(&mut stream, &kv_ctx, nic_idx, if_not_exists, with_options)?;
            continue;
        }
        if tag == MSG_PUT_STRIPES_REQ {
            handle_put_stripes(&mut stream, &kv_ctx, nic_idx)?;
            continue;
        }
        if tag == MSG_GET_DESCRIPTOR_REQ
            || tag == MSG_GET_DESCRIPTOR_STRIPES_REQ
            || tag == MSG_GET_DESCRIPTOR_STRIPES_SGE_REQ
        {
            handle_descriptor_get(
                &mut stream,
                &kv_ctx,
                &rdma,
                &qp,
                client_cq,
                nic_idx,
                tag != MSG_GET_DESCRIPTOR_REQ,
                tag == MSG_GET_DESCRIPTOR_STRIPES_SGE_REQ,
            )?;
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
                let active_meta = kv_ctx.metadata.get_block(&kv_key.to_string_key())?;
                let slab_path_result = match active_meta {
                    Some(meta) if placement_spans_multiple_nodes(&meta) => {
                        tracing::warn!(
                            event = "rdma_get_multi_endpoint_required",
                            key = %kv_key.to_string_key(),
                            "legacy single-endpoint RDMA GET rejected for distributed placement; use placement lookup and stripe-subset GET"
                        );
                        Ok(None)
                    }
                    Some(meta) if meta.is_expired() => {
                        kv_ctx.storage.delete_if_expired(&kv_key, &meta)?;
                        Ok(None)
                    }
                    Some(meta) => try_serve_get_via_slab_with_meta(
                        &kv_ctx, &qp, client_cq, &kv_key, &meta, dst_addr, dst_rkey, max_size,
                        nic_idx,
                    ),
                    None => Ok(None),
                };
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
/// 1. slab.alloc(size) → SlabExtent (4K aligned, pre-registered MR).
/// 2. storage.get_into_ptr(extent.ptr) → O_DIRECT pread straight into slab (zero intermediate buffer).
/// 3. Post RDMA WRITE using the slab's pre-registered lkey (zero reg_mr!).
/// 4. **insert_chunks_from_slab injects into L1 cache** → subsequent GETs hit the slab fast path immediately.
///
/// Difference vs. the old serve_get_fallback:
/// - Old: read into heap × 8 → 8× ibv_reg_mr (~33ms) → 8 SQEs post_write → poll
/// - New: read into slab → 0 reg_mr → 1 SQE post_write → poll
///
/// Returns:
/// - Ok(Some((bytes, post_us, poll_us))): RDMA WRITE succeeded
/// - Ok(None): the supplied metadata became unavailable or expired
/// - Err(e): slab full / I/O error / protocol error → caller falls back to fallback
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

    // 4. Consume each stream completion and immediately post its RDMA WRITE. Keep the
    // number of outstanding WRs bounded so a larger object or more devices cannot exhaust
    // the QP send queue.
    const RDMA_WRITE_COMPLETION_WINDOW: usize = RcQp::MAX_SEND_WR / 2;
    let view = extent.view(nic_idx);
    let t_post_start = std::time::Instant::now();
    let mut n_writes_posted = 0u64;
    let mut outstanding_writes = 0usize;
    let mut poll_us = 0u64;
    let mut had_error: Option<String> = None;
    let mut first_stream_completion_us: Option<u64> = None;
    let mut last_stream_completion_us = 0u64;

    while let Ok((stripe_idx, offset_in_value, stripe_len, result)) = stream_rx.recv() {
        let t_now_us = t_stream_start.elapsed().as_micros() as u64;
        first_stream_completion_us.get_or_insert(t_now_us);
        last_stream_completion_us = t_now_us;
        match result {
            Ok(bytes_read) if bytes_read == stripe_len && had_error.is_none() => {
                if let Err(error) = qp.post_write(
                    stripe_idx as u64,
                    view.addr + offset_in_value as u64,
                    view.lkey,
                    dst_addr + offset_in_value as u64,
                    dst_rkey,
                    stripe_len as u32,
                    true, // signaled
                ) {
                    had_error = Some(format!(
                        "post RDMA write for stripe {}: {}",
                        stripe_idx, error
                    ));
                } else {
                    n_writes_posted += 1;
                    outstanding_writes += 1;
                    if outstanding_writes == RDMA_WRITE_COMPLETION_WINDOW {
                        let poll_start = std::time::Instant::now();
                        if let Err(error) = RcQp::poll_n(client_cq, outstanding_writes) {
                            had_error =
                                Some(format!("poll RDMA write completion window: {}", error));
                        }
                        poll_us += poll_start.elapsed().as_micros() as u64;
                        outstanding_writes = 0;
                    }
                }
            }
            Ok(bytes_read) if bytes_read != stripe_len && had_error.is_none() => {
                had_error = Some(format!(
                    "stripe {} short read: expected {} bytes, got {}",
                    stripe_idx, stripe_len, bytes_read
                ));
            }
            Ok(_) => {}
            Err(e) if had_error.is_none() => {
                had_error = Some(format!("stripe {} read failed: {}", stripe_idx, e));
            }
            Err(_) => {}
        }
    }
    let t_post_done = t_post_start.elapsed().as_micros() as u64;

    // 5. Drain every posted WRITE before returning, including when a later stripe
    // failed. The slab extent backs in-flight RNIC DMA and must not be released early.
    let poll_result = if outstanding_writes > 0 {
        let poll_start = std::time::Instant::now();
        let result = RcQp::poll_n(client_cq, outstanding_writes);
        poll_us += poll_start.elapsed().as_micros() as u64;
        result
    } else {
        Ok(())
    };

    let post_us = t_post_done;

    if let Some(error) = had_error {
        return Err(anyhow!("RDMA GET stream failed: {}", error));
    }
    poll_result?;
    if n_writes_posted == 0 {
        return Err(anyhow!("no stripes posted"));
    }

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

    tracing::info!(
        "MISS_DETAIL wall_us={} key={} bytes={} meta_us={} alloc_us={} stream_setup_us={} \
         first_stream_completion_us={} last_stream_completion_us={} post_done_us={} poll_us={} \
         inject_us={} n_writes={}",
        wall_start_us,
        kv_key.to_string_key(),
        size,
        t_meta,
        t_alloc,
        t_stream_setup,
        first_stream_completion_us.unwrap_or(0),
        last_stream_completion_us,
        post_us,
        poll_us,
        t_inject,
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
            chunk_checksums: Vec::new(),
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

fn chunk_is_local(ctx: &KVServiceContext, location: &crate::metadata::ChunkLocation) -> bool {
    let local_node_id = std::env::var("CS_NODE_ID")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| ctx.config.cluster.node_id.clone());
    let local_grpc_endpoint = std::env::var("CS_GRPC_ADVERTISE")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| ctx.config.cluster.grpc_advertise.clone());
    let local_rdma_endpoint = std::env::var("CS_RDMA_ADVERTISE")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| ctx.config.cluster.rdma_advertise.clone());

    (!local_node_id.is_empty() && location.node_id == local_node_id)
        || (!local_grpc_endpoint.is_empty() && location.grpc_endpoint == local_grpc_endpoint)
        || (!local_rdma_endpoint.is_empty() && location.rdma_endpoint == local_rdma_endpoint)
}

fn placement_spans_multiple_nodes(meta: &BlockMeta) -> bool {
    let Some(striping) = meta.striping.as_ref() else {
        return false;
    };
    let mut first_identity: Option<&str> = None;
    for location in &striping.chunk_locations {
        let identity = if !location.node_id.is_empty() {
            location.node_id.as_str()
        } else if !location.rdma_endpoint.is_empty() {
            location.rdma_endpoint.as_str()
        } else {
            location.grpc_endpoint.as_str()
        };
        if identity.is_empty() {
            continue;
        }
        match first_identity {
            Some(first) if first != identity => return true,
            None => first_identity = Some(identity),
            _ => {}
        }
    }
    false
}

fn allocate_subset_staging(kv_ctx: &KVServiceContext, size: usize) -> Result<SlabExtent> {
    let slab = kv_ctx
        .memory
        .rdma_slab_get()
        .ok_or_else(|| anyhow!("RDMA slab is unavailable"))?;
    if size > slab.capacity() {
        return Err(anyhow!(
            "RDMA subset staging requires {} bytes but slab capacity is {}",
            size,
            slab.capacity()
        ));
    }
    if let Some(extent) = slab.alloc(size) {
        return Ok(extent);
    }

    let mut evict_multiplier = 2usize;
    for _ in 0..5 {
        let evict_target = size.saturating_mul(evict_multiplier).min(slab.capacity());
        kv_ctx
            .memory
            .evict_chunks_cache_to_free(evict_target, slab.capacity());
        if let Some(extent) = slab.alloc(size) {
            return Ok(extent);
        }
        evict_multiplier = evict_multiplier.saturating_mul(2);
    }
    Err(anyhow!(
        "RDMA subset staging allocation failed after cache eviction (need {} bytes)",
        size
    ))
}


/// 把对象字节区间 [object_offset, object_offset+length) 映射到 scatter 段表.
/// 段表按序覆盖对象字节范围; 一个区间可能跨多个段, 产出多条 (dst_addr, rkey, len).
/// 返回 Err 当区间超出段表覆盖范围.
fn map_range_to_segments(
    segments: &[(u64, u32, u64)],
    object_offset: u64,
    length: u64,
) -> anyhow::Result<Vec<(u64, u32, u64)>> {
    let mut out = Vec::new();
    let mut remaining = length;
    let mut cursor = object_offset;
    let mut seg_base = 0u64; // 当前段在对象字节空间的起点
    for (addr, rkey, seg_len) in segments {
        let seg_end = seg_base + seg_len;
        if cursor < seg_end && remaining > 0 {
            let in_seg = cursor - seg_base;
            let n = remaining.min(seg_end - cursor);
            out.push((addr + in_seg, *rkey, n));
            cursor += n;
            remaining -= n;
        }
        seg_base = seg_end;
        if remaining == 0 {
            break;
        }
    }
    if remaining > 0 {
        return Err(anyhow!(
            "scatter segments cover {} bytes but range needs {}+{}",
            seg_base,
            object_offset,
            length
        ));
    }
    Ok(out)
}

fn serve_get_stripes_fallback(
    kv_ctx: &Arc<KVServiceContext>,
    rdma: &Arc<RdmaContext>,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    striping: &StripingInfo,
    req: &DescriptorGetReqMsg,
) -> Result<(bool, u64, u32)> {
    let indices = req
        .stripes
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    let segments = match kv_ctx.storage.read_striped_chunks_at(striping, &indices) {
        Ok(segments) => segments,
        Err(error) => {
            tracing::warn!(error = %error, "stripe-subset fallback read failed");
            return Ok((false, 0, 0));
        }
    };

    let mut total = 0u64;
    let mut mrs = Vec::with_capacity(segments.len());
    for (write_index, (stripe_index, segment)) in segments.iter().enumerate() {
        if segment.is_empty() {
            continue;
        }
        let mr = unsafe {
            rdma.register_mr_raw(
                segment.as_ptr() as *mut u8,
                segment.len(),
                ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0,
            )?
        };
        qp.post_write(
            write_index as u64,
            mr.addr,
            mr.lkey,
            req.dst_addr + *stripe_index as u64 * striping.chunk_size,
            req.dst_rkey,
            segment.len() as u32,
            write_index + 1 == segments.len(),
        )?;
        total += segment.len() as u64;
        mrs.push(mr);
    }
    if !mrs.is_empty() {
        RcQp::poll_n(client_cq, 1)?;
    }
    Ok((true, total, segments.len() as u32))
}

/// Serve a stripe-subset descriptor GET (tag 12): read each requested stripe
/// from local storage and RDMA-WRITE it at `dst_addr + index * chunk_size`.
///
/// Only stripes present on this node's devices can be served — a request for
/// a stripe this node does not hold fails the whole request (found=false) so
/// the client falls back to the gRPC path rather than receiving a hole.
fn serve_get_stripes(
    kv_ctx: &Arc<KVServiceContext>,
    rdma: &Arc<RdmaContext>,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    kv_key: &ObjectKey,
    active_meta: &crate::metadata::BlockMeta,
    req: &DescriptorGetReqMsg,
    nic_idx: usize,
) -> Result<(bool, u64, u32)> {
    let Some(striping) = active_meta.striping.as_ref() else {
        tracing::warn!("stripe-subset GET on a non-striped object");
        return Ok((false, 0, 0));
    };
    let chunk_size = striping.chunk_size;
    if chunk_size == 0 {
        return Ok((false, 0, 0));
    }
    let locations_are_complete = striping.chunk_locations.len() == striping.chunk_paths.len();
    let mut staged_bytes = 0usize;
    // Validate ownership, indices, and the destination window before any I/O.
    for (position, &idx) in req.stripes.iter().enumerate() {
        if req.stripes[..position].contains(&idx) {
            tracing::warn!(
                stripe_index = idx,
                "stripe-subset GET contains a duplicate index"
            );
            return Ok((false, 0, 0));
        }
        let idx = idx as usize;
        if idx >= striping.chunk_paths.len() {
            tracing::warn!("stripe-subset GET: index {} out of range", idx);
            return Ok((false, 0, 0));
        }
        let end = (idx as u64) * chunk_size
            + chunk_size.min(striping.total_size - (idx as u64) * chunk_size);
        if end > req.max_size {
            tracing::warn!(
                "stripe-subset GET: stripe {} ends at {} beyond client window {}",
                idx,
                end,
                req.max_size
            );
            return Ok((false, 0, 0));
        }
        if locations_are_complete && !chunk_is_local(kv_ctx, &striping.chunk_locations[idx]) {
            tracing::warn!(
                event = "rdma_get_stripes_wrong_endpoint",
                key = %kv_key.to_string_key(),
                stripe_index = idx,
                owner_node = %striping.chunk_locations[idx].node_id,
                owner_endpoint = %striping.chunk_locations[idx].rdma_endpoint,
                "stripe-subset GET was sent to a node that does not own the stripe"
            );
            return Ok((false, 0, 0));
        }
        let stripe_offset = idx as u64 * chunk_size;
        staged_bytes = staged_bytes
            .checked_add(chunk_size.min(striping.total_size.saturating_sub(stripe_offset)) as usize)
            .ok_or_else(|| anyhow!("stripe subset staging size overflow"))?;
    }

    let total_start = std::time::Instant::now();
    let alloc_start = std::time::Instant::now();
    let extent = match allocate_subset_staging(kv_ctx, staged_bytes) {
        Ok(extent) => extent,
        Err(error) => {
            tracing::warn!(
                key = %kv_key.to_string_key(),
                error = %error,
                "RDMA stripe-subset slab path unavailable; using registered-buffer fallback"
            );
            return serve_get_stripes_fallback(kv_ctx, rdma, qp, client_cq, striping, req);
        }
    };
    let alloc_us = alloc_start.elapsed().as_micros() as u64;

    let indices = req
        .stripes
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    let stream_start = std::time::Instant::now();
    let stream = match kv_ctx.storage.read_striped_subset_into_ptr_stream(
        striping,
        &indices,
        extent.as_ptr() as *mut u8,
        extent.capacity_bytes(),
    ) {
        Ok(stream) => stream,
        Err(error) => {
            tracing::warn!(
                "stripe-subset GET: read failed for key {}: {}",
                kv_key.to_string_key(),
                error
            );
            return Ok((false, 0, 0));
        }
    };
    let stream_setup_us = stream_start.elapsed().as_micros() as u64;
    let view = extent.view(nic_idx);
    const COMPLETION_WINDOW: usize = RcQp::MAX_SEND_WR / 2;
    let mut outstanding = 0usize;
    let mut writes_posted = 0usize;
    let mut total = 0u64;
    let mut poll_us = 0u64;
    let mut first_io_us = None;
    let mut last_io_us = 0u64;
    let mut first_error = None;

    while let Ok((stripe_index, source_offset, object_offset, length, result)) = stream.recv() {
        let completion_us = stream_start.elapsed().as_micros() as u64;
        first_io_us.get_or_insert(completion_us);
        last_io_us = completion_us;
        match result {
            Ok(bytes) if bytes == length && first_error.is_none() => {
                // tag-15 (SGE): 数据区间按段表映射, 可能拆成多条 WRITE;
                // tag-12: 单一连续目标, 等价于一个覆盖全对象的段.
                let targets: Vec<(u64, u32, u64)> = if req.dst_segments.is_empty() {
                    vec![(req.dst_addr + object_offset as u64, req.dst_rkey, length as u64)]
                } else {
                    match map_range_to_segments(
                        &req.dst_segments,
                        object_offset as u64,
                        length as u64,
                    ) {
                        Ok(t) => t,
                        Err(error) => {
                            first_error = Some(format!(
                                "map stripe {stripe_index} to scatter segments: {error}"
                            ));
                            continue;
                        }
                    }
                };
                let mut src = view.addr + source_offset as u64;
                let mut post_failed = false;
                for (dst, rkey, n) in &targets {
                    if let Err(error) = qp.post_write(
                        writes_posted as u64,
                        src,
                        view.lkey,
                        *dst,
                        *rkey,
                        *n as u32,
                        true,
                    ) {
                        first_error = Some(format!(
                            "post RDMA write for stripe {stripe_index}: {error}"
                        ));
                        post_failed = true;
                        break;
                    }
                    src += n;
                    writes_posted += 1;
                    outstanding += 1;
                }
                if !post_failed {
                    total += bytes as u64;
                    // 机会式非阻塞收割: 每次 post 后顺手清已完成的 CQE.
                    // 旧逻辑凑满窗口才 poll_n(64) 阻塞等全部完成, 期间不消费盘 IO
                    // 完成事件 → 流水线断流. 现在仅当 SQ 接近满时才阻塞等 1 个腾位.
                    let poll_start = std::time::Instant::now();
                    match RcQp::poll_available(client_cq, outstanding) {
                        Ok(n) => outstanding -= n,
                        Err(error) => {
                            first_error =
                                Some(format!("poll RDMA completion window: {error}"))
                        }
                    }
                    while first_error.is_none() && outstanding >= COMPLETION_WINDOW {
                        match RcQp::poll_n(client_cq, 1) {
                            Ok(()) => outstanding -= 1,
                            Err(error) => {
                                first_error =
                                    Some(format!("poll RDMA completion window: {error}"))
                            }
                        }
                    }
                    poll_us += poll_start.elapsed().as_micros() as u64;
                }
            }
            Ok(bytes) if first_error.is_none() => {
                first_error = Some(format!(
                    "stripe {stripe_index} short read: expected {length}, got {bytes}"
                ));
            }
            Err(error) if first_error.is_none() => {
                first_error = Some(format!("stripe {stripe_index} read failed: {error}"));
            }
            _ => {}
        }
    }

    if outstanding > 0 {
        let poll_start = std::time::Instant::now();
        let poll_result = RcQp::poll_n(client_cq, outstanding);
        poll_us += poll_start.elapsed().as_micros() as u64;
        if let Err(error) = poll_result {
            first_error.get_or_insert_with(|| format!("poll final RDMA completions: {error}"));
        }
    }
    let total_us = total_start.elapsed().as_micros() as u64;
    tracing::info!(
        "RDMA_SUBSET_DETAIL key={} bytes={} stripes={} requests={} alloc_us={} stream_setup_us={} first_io_us={} last_io_us={} poll_us={} total_us={}",
        kv_key.to_string_key(),
        total,
        req.stripes.len(),
        writes_posted,
        alloc_us,
        stream_setup_us,
        first_io_us.unwrap_or(0),
        last_io_us,
        poll_us,
        total_us,
    );

    if let Some(error) = first_error {
        tracing::warn!(
            key = %kv_key.to_string_key(),
            error = %error,
            "RDMA stripe-subset stream failed without deleting object metadata"
        );
        return Ok((false, 0, 0));
    }
    Ok((true, total, req.stripes.len() as u32))
}

fn handle_descriptor_get(
    stream: &mut TcpStream,
    kv_ctx: &Arc<KVServiceContext>,
    rdma: &Arc<RdmaContext>,
    qp: &RcQp,
    client_cq: NonNull<rdma_sys::ibv_cq>,
    nic_idx: usize,
    with_stripes: bool,
    with_segments: bool,
) -> Result<()> {
    let req = wire::recv_descriptor_get_req_body_ext(stream, with_stripes, with_segments)?;
    let kv_key = parse_string_key(&req.key)?;
    // Validate descriptor layout fields before consulting the authoritative metadata. The actual
    // metadata is used for I/O so optional per-stripe checksums cannot be omitted by a client.
    let _ = descriptor_meta_from_req(kv_ctx, &kv_key, &req)?;

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
    if active_meta.is_expired() {
        kv_ctx.storage.delete_if_expired(&kv_key, &active_meta)?;
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

    // ===== Stripe-subset path (tag 12): serve only the requested stripes =====
    //
    // Multi-endpoint direct reads: the client splits an object's stripes by
    // owning node (from the gRPC placement) and asks each node for just its
    // stripes. Every stripe lands at dst_addr + index * chunk_size, so N
    // nodes fill disjoint regions of one client buffer concurrently with no
    // coordinator forwarding.
    if !req.stripes.is_empty() {
        let (found, bytes_written, num_chunks) = serve_get_stripes(
            kv_ctx,
            rdma,
            qp,
            client_cq,
            &kv_key,
            &active_meta,
            &req,
            nic_idx,
        )?;
        wire::send_get_resp(
            stream,
            &GetRespMsg {
                found,
                bytes_written,
                num_chunks,
            },
        )?;
        if found {
            tracing::debug!(
                target: "contextstore_server::storage_io",
                event = "rdma_get_stripes_complete",
                status = "ok",
                bytes = bytes_written,
                stripes = num_chunks,
            );
        }
        return Ok(());
    }

    // Legacy descriptor GET asks one endpoint to serve the complete object. A distributed
    // placement must be split by endpoint first; attempting a local full read would interpret
    // remote stripes as missing files and could incorrectly invalidate otherwise valid metadata.
    if placement_spans_multiple_nodes(&active_meta) {
        tracing::warn!(
            event = "rdma_get_multi_endpoint_required",
            key = %kv_key.to_string_key(),
            "legacy single-endpoint RDMA GET rejected for distributed placement; use placement lookup and stripe-subset GET"
        );
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
            &active_meta,
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
                match kv_ctx.storage.get_chunks_with_meta(&kv_key, &active_meta) {
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

/// Handle a stripe-subset PUT (tag MSG_PUT_STRIPES_REQ already consumed).
///
/// Multi-endpoint direct writes: the coordinator prepared the object identity
/// (generation / layout) via gRPC PrepareDistributedPut; the client pushes
/// this node's stripes back-to-back into one slab extent over RDMA, and this
/// handler pwrites each stripe as a placement chunk, answering with the
/// per-stripe storage handles the coordinator needs for its commit.
///
/// No metadata is written here — commit happens exactly once, at the
/// coordinator, after every node acknowledged its stripes. A crashed client
/// leaves only unreferenced chunk files (reaped by rollback/TTL), never a
/// visible half-object.
fn handle_put_stripes(
    stream: &mut TcpStream,
    kv_ctx: &Arc<KVServiceContext>,
    nic_idx: usize,
) -> Result<()> {
    let req = wire::recv_put_stripes_req_body(stream)?;
    let total: u64 = req.stripes.iter().map(|(_, len)| *len).sum();
    let size = total as usize;
    let fail = |stream: &mut TcpStream| -> Result<()> {
        wire::send_put_ready(
            stream,
            &PutReadyMsg {
                ok: false,
                dst_addr: 0,
                dst_rkey: 0,
            },
        )?;
        wire::send_put_stripes_resp(
            stream,
            &PutStripesRespMsg {
                ok: false,
                stripes: Vec::new(),
            },
        )?;
        Ok(())
    };
    if req.stripes.is_empty() || size == 0 || size > 4 * 1024 * 1024 * 1024 {
        tracing::warn!("RDMA stripe PUT rejected: invalid size {}", size);
        return fail(stream);
    }
    for (idx, len) in &req.stripes {
        if *len == 0 || *len > req.chunk_size {
            tracing::warn!(
                "RDMA stripe PUT rejected: stripe {} has bad len {}",
                idx,
                len
            );
            return fail(stream);
        }
    }

    // Slab extent sized for just this node's stripes (packed back-to-back).
    let extent: SlabExtent = {
        let slab_opt = kv_ctx.memory.rdma_slab_get();
        match slab_opt.as_ref().and_then(|s| s.alloc(size)) {
            Some(e) => e,
            None => {
                tracing::warn!("RDMA stripe PUT rejected: slab full (size={})", size);
                return fail(stream);
            }
        }
    };
    wire::send_put_ready(
        stream,
        &PutReadyMsg {
            ok: true,
            dst_addr: extent.addr(),
            dst_rkey: extent.rkey(nic_idx),
        },
    )?;

    // Client RDMA-WRITEs, then commits.
    let commit_tag = wire::read_exact(stream, 1)?[0];
    if commit_tag != MSG_PUT_COMMIT {
        return Err(anyhow!(
            "expected MSG_PUT_COMMIT={}, got {}",
            MSG_PUT_COMMIT,
            commit_tag
        ));
    }

    // pwrite each stripe from its slab offset as a placement chunk.
    let kv_key = parse_string_key(&req.key)?;
    let mut locations = Vec::with_capacity(req.stripes.len());
    let mut offset: usize = 0;
    let mut write_err = false;
    for (idx, len) in &req.stripes {
        let stripe =
            unsafe { std::slice::from_raw_parts(extent.as_ptr().add(offset), *len as usize) };
        offset += *len as usize;
        // Device rotation uses this node's local ordinal among its own
        // stripes (mirrors the gRPC placement path).
        let device_stripe_index = locations.len();
        match kv_ctx.storage.put_placement_chunk(
            &kv_key,
            *idx as usize,
            device_stripe_index,
            req.object_generation,
            req.layout_version,
            prost::bytes::Bytes::copy_from_slice(stripe),
        ) {
            Ok((device_id, storage_handle, checksum)) => locations.push(PutStripeLocation {
                stripe_index: *idx,
                device_id,
                storage_handle,
                checksum,
            }),
            Err(e) => {
                tracing::warn!("RDMA stripe PUT: stripe {} write failed: {}", idx, e);
                write_err = true;
                break;
            }
        }
    }
    if write_err {
        // Roll back stripes already written on this node.
        for loc in &locations {
            let _ = kv_ctx.storage.delete_placement_chunk(&loc.storage_handle);
        }
        wire::send_put_stripes_resp(
            stream,
            &PutStripesRespMsg {
                ok: false,
                stripes: Vec::new(),
            },
        )?;
        return Ok(());
    }
    tracing::debug!(
        target: "contextstore_server::storage_io",
        event = "rdma_put_stripes_complete",
        status = "ok",
        bytes = total,
        stripes = locations.len(),
    );
    wire::send_put_stripes_resp(
        stream,
        &PutStripesRespMsg {
            ok: true,
            stripes: locations,
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
    with_options: bool,
) -> Result<()> {
    let t_recv_start = std::time::Instant::now();
    let put_req = wire::recv_put_req_body(stream, with_options)?;
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
        // TTL from the with-options wire body (0 = no expiry); ContextStore's
        // TTL lifecycle reaps expired objects the same way the gRPC path does.
        ttl_seconds: put_req.ttl_seconds.max(0),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ChunkLocation;

    fn striped_meta(node_ids: &[&str]) -> BlockMeta {
        let chunk_locations = node_ids
            .iter()
            .enumerate()
            .map(|(stripe_index, node_id)| ChunkLocation {
                stripe_index: stripe_index as u32,
                node_id: (*node_id).to_string(),
                grpc_endpoint: format!("{node_id}:50051"),
                rdma_endpoint: format!("{node_id}:50053"),
                device_id: (stripe_index % 2) as u32,
                storage_handle: format!("/data/chunk{stripe_index}.bin"),
                offset: stripe_index as u64 * 64 * 1024 * 1024,
                length: 64 * 1024 * 1024,
                checksum: String::new(),
            })
            .collect::<Vec<_>>();
        BlockMeta {
            device_id: 0,
            file_path: String::new(),
            size: node_ids.len() as u64 * 64 * 1024 * 1024,
            object_handle: "test-object".to_string(),
            object_generation: 1,
            content_etag: "test-etag".to_string(),
            layout_version: 1,
            created_at: 0,
            last_accessed_at: 0,
            ttl_seconds: 0,
            num_tokens: 0,
            num_layers: 0,
            dtype: "bytes".to_string(),
            compressed: false,
            striping: Some(StripingInfo {
                chunk_size: 64 * 1024 * 1024,
                chunk_devices: (0..node_ids.len())
                    .map(|index| (index % 2) as u32)
                    .collect(),
                chunk_paths: (0..node_ids.len())
                    .map(|index| format!("/data/chunk{index}.bin"))
                    .collect(),
                total_size: node_ids.len() as u64 * 64 * 1024 * 1024,
                chunk_locations,
                chunk_checksums: Vec::new(),
            }),
        }
    }

    #[test]
    fn distributed_placement_requires_multi_endpoint_get() {
        assert!(placement_spans_multiple_nodes(&striped_meta(&[
            "worker01", "worker02", "worker01", "worker02"
        ])));
    }

    #[test]
    fn local_striped_placement_allows_single_endpoint_get() {
        assert!(!placement_spans_multiple_nodes(&striped_meta(&[
            "worker01", "worker01", "worker01", "worker01"
        ])));
    }
}

#[cfg(test)]
mod sge_tests {
    use super::map_range_to_segments;

    #[test]
    fn range_within_one_segment() {
        let segs = [(0x1000, 7, 100u64), (0x9000, 8, 100)];
        let m = map_range_to_segments(&segs, 10, 50).unwrap();
        assert_eq!(m, vec![(0x1000 + 10, 7, 50)]);
    }

    #[test]
    fn range_spans_segments() {
        let segs = [(0x1000, 7, 100u64), (0x9000, 8, 100)];
        let m = map_range_to_segments(&segs, 80, 50).unwrap();
        assert_eq!(m, vec![(0x1000 + 80, 7, 20), (0x9000, 8, 30)]);
    }

    #[test]
    fn range_beyond_coverage_errors() {
        let segs = [(0x1000, 7, 100u64)];
        assert!(map_range_to_segments(&segs, 80, 50).is_err());
    }

    #[test]
    fn exact_tail_fit() {
        let segs = [(0x1000, 7, 64u64), (0x9000, 8, 64)];
        let m = map_range_to_segments(&segs, 64, 64).unwrap();
        assert_eq!(m, vec![(0x9000, 8, 64)]);
    }
}
