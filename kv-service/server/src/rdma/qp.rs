//! Reliable Connection QP — one-to-one connection with a client; use ibv_post_send WRITE after handshake

use crate::rdma::context::RdmaContext;
use anyhow::{anyhow, Result};
use rdma_sys::*;
use std::ptr::{self, NonNull};

/// An RC QP (Reliable Connection Queue Pair).
///
/// Creation flow:
/// 1. `RcQp::new(ctx)` — create QP in RESET state
/// 2. `qp.to_init(...)` — transition to INIT
/// 3. `qp.to_rtr(remote)` — after receiving remote QP info, transition to Ready-To-Receive
/// 4. `qp.to_rts(...)` — transition to Ready-To-Send
/// 5. `qp.post_write(...)` — actual operation
pub struct RcQp {
    qp: NonNull<ibv_qp>,
    /// Local QP info, sent to remote over the control plane
    pub local: QpInfo,
}

unsafe impl Send for RcQp {}
unsafe impl Sync for RcQp {}

/// QP metadata, exchanged over the TCP control plane
#[derive(Clone, Copy)]
pub struct QpInfo {
    pub qpn: u32,
    pub psn: u32,
    pub gid: ibv_gid,
}

impl std::fmt::Debug for QpInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            f.debug_struct("QpInfo")
                .field("qpn", &self.qpn)
                .field("psn", &format!("0x{:x}", self.psn))
                .field("gid", &format!("{:02x?}", &self.gid.raw[..]))
                .finish()
        }
    }
}

impl QpInfo {
    /// Serialize to 24 bytes: 4(qpn) + 4(psn) + 16(gid)
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&self.qpn.to_le_bytes());
        buf[4..8].copy_from_slice(&self.psn.to_le_bytes());
        unsafe { buf[8..24].copy_from_slice(&self.gid.raw[..]); }
        buf
    }

    pub fn from_bytes(buf: &[u8; 24]) -> Self {
        let qpn = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let psn = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
        unsafe { gid.raw[..].copy_from_slice(&buf[8..24]); }
        Self { qpn, psn, gid }
    }
}

impl RcQp {
    /// Create an RC QP in RESET state. Follow with to_init/to_rtr/to_rts.
    ///
    /// `cq`: dedicated CQ (per-client) to avoid races when multiple client threads poll the same CQ concurrently.
    pub fn new(ctx: &RdmaContext, cq: NonNull<ibv_cq>) -> Result<Self> {
        unsafe {
            let mut attr = ibv_qp_init_attr {
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
                sq_sig_all: 0, // do not signal every wr; caller controls explicitly
            };
            let qp_raw = ibv_create_qp(ctx.pd.as_ptr(), &mut attr);
            let qp = NonNull::new(qp_raw)
                .ok_or_else(|| anyhow!("ibv_create_qp failed: {}", std::io::Error::last_os_error()))?;

            // Pick a random PSN (Packet Serial Number). Cryptographic randomness not required.
            // Use the low 24 bits of the timestamp.
            let psn = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(0))
                & 0xFFFFFF;

            let local = QpInfo {
                qpn: (*qp_raw).qp_num,
                psn,
                gid: ctx.local_gid,
            };

            tracing::info!("RcQp created: qpn={} psn=0x{:x}", local.qpn, local.psn);

            Ok(Self { qp, local })
        }
    }

    /// Transition to INIT. Set port and access flags.
    pub fn to_init(&self, port_num: u8) -> Result<()> {
        unsafe {
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
            attr.pkey_index = 0;
            attr.port_num = port_num;
            // Allow remote WRITE/READ on our MR (direction here is server WRITE to client, but symmetric permissions ease debugging)
            attr.qp_access_flags = (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as i32 as u32;

            let mask = ibv_qp_attr_mask::IBV_QP_STATE
                | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
                | ibv_qp_attr_mask::IBV_QP_PORT
                | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
            let rc = ibv_modify_qp(self.qp.as_ptr(), &mut attr, mask.0 as i32);
            if rc != 0 {
                return Err(anyhow!("modify_qp -> INIT failed: rc={} errno={}", rc, std::io::Error::last_os_error()));
            }
            Ok(())
        }
    }

    /// Transition to RTR (Ready-To-Receive). Requires remote QP info + path MTU + GID index.
    pub fn to_rtr(&self, remote: &QpInfo, port_num: u8, gid_index: u8) -> Result<()> {
        unsafe {
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
            attr.path_mtu = ibv_mtu::IBV_MTU_1024; // align with hardware active_mtu
            attr.dest_qp_num = remote.qpn;
            attr.rq_psn = remote.psn;
            attr.max_dest_rd_atomic = 1;
            attr.min_rnr_timer = 12;

            // AH (address handle) — RoCE v2 uses GID-based addressing
            attr.ah_attr.is_global = 1;
            attr.ah_attr.dlid = 0;
            attr.ah_attr.sl = 0;
            attr.ah_attr.src_path_bits = 0;
            attr.ah_attr.port_num = port_num;
            attr.ah_attr.grh.dgid = remote.gid;
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.sgid_index = gid_index;
            attr.ah_attr.grh.traffic_class = 0;

            let mask = ibv_qp_attr_mask::IBV_QP_STATE
                | ibv_qp_attr_mask::IBV_QP_AV
                | ibv_qp_attr_mask::IBV_QP_PATH_MTU
                | ibv_qp_attr_mask::IBV_QP_DEST_QPN
                | ibv_qp_attr_mask::IBV_QP_RQ_PSN
                | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
                | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
            let rc = ibv_modify_qp(self.qp.as_ptr(), &mut attr, mask.0 as i32);
            if rc != 0 {
                return Err(anyhow!("modify_qp -> RTR failed: rc={} errno={}", rc, std::io::Error::last_os_error()));
            }
            Ok(())
        }
    }

    /// Transition to RTS (Ready-To-Send). After this, post_send is allowed.
    pub fn to_rts(&self) -> Result<()> {
        unsafe {
            let mut attr: ibv_qp_attr = std::mem::zeroed();
            attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
            attr.timeout = 14;
            attr.retry_cnt = 7;
            attr.rnr_retry = 7;
            attr.sq_psn = self.local.psn;
            attr.max_rd_atomic = 1;

            let mask = ibv_qp_attr_mask::IBV_QP_STATE
                | ibv_qp_attr_mask::IBV_QP_TIMEOUT
                | ibv_qp_attr_mask::IBV_QP_RETRY_CNT
                | ibv_qp_attr_mask::IBV_QP_RNR_RETRY
                | ibv_qp_attr_mask::IBV_QP_SQ_PSN
                | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;
            let rc = ibv_modify_qp(self.qp.as_ptr(), &mut attr, mask.0 as i32);
            if rc != 0 {
                return Err(anyhow!("modify_qp -> RTS failed: rc={} errno={}", rc, std::io::Error::last_os_error()));
            }
            Ok(())
        }
    }

    /// Post an RDMA WRITE: copy local [local_addr, local_addr+len) to remote [remote_addr, remote_addr+len).
    ///
    /// `wr_id`: user-defined id returned in the CQE, used to correlate requests.
    /// `signaled`: whether to generate a CQE (batched WRITEs typically only signal the last one).
    pub fn post_write(
        &self,
        wr_id: u64,
        local_addr: u64,
        local_lkey: u32,
        remote_addr: u64,
        remote_rkey: u32,
        len: u32,
        signaled: bool,
    ) -> Result<()> {
        unsafe {
            let mut sge = ibv_sge {
                addr: local_addr,
                length: len,
                lkey: local_lkey,
            };
            let mut wr: ibv_send_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
            wr.send_flags = if signaled {
                ibv_send_flags::IBV_SEND_SIGNALED.0
            } else {
                0
            };
            wr.wr.rdma.remote_addr = remote_addr;
            wr.wr.rdma.rkey = remote_rkey;

            let mut bad_wr: *mut ibv_send_wr = ptr::null_mut();
            let rc = ibv_post_send(self.qp.as_ptr(), &mut wr, &mut bad_wr);
            if rc != 0 {
                return Err(anyhow!("ibv_post_send failed: rc={}", rc));
            }
            Ok(())
        }
    }

    /// Wait for N work completions. Simple busy poll for PoC use.
    pub fn poll_n(cq: NonNull<ibv_cq>, expected: usize) -> Result<()> {
        unsafe {
            // ibv_wc does not implement Clone; use push instead of vec!.
            let mut wcs: Vec<ibv_wc> = Vec::with_capacity(expected.max(1));
            for _ in 0..expected.max(1) {
                wcs.push(std::mem::zeroed());
            }
            let mut got = 0;
            let start = std::time::Instant::now();
            while got < expected {
                let n = ibv_poll_cq(
                    cq.as_ptr(),
                    (expected - got) as i32,
                    wcs[got..].as_mut_ptr(),
                );
                if n < 0 {
                    return Err(anyhow!("ibv_poll_cq error"));
                }
                if n > 0 {
                    for i in got..(got + n as usize) {
                        if wcs[i].status != ibv_wc_status::IBV_WC_SUCCESS {
                            return Err(anyhow!(
                                "WR {} failed: status={} ({})",
                                wcs[i].wr_id,
                                wcs[i].status,
                                wc_status_str(wcs[i].status),
                            ));
                        }
                    }
                    got += n as usize;
                }
                if start.elapsed().as_secs() > 30 {
                    return Err(anyhow!("poll_n timeout after 30s, got {}/{}", got, expected));
                }
            }
            Ok(())
        }
    }
}

fn wc_status_str(status: u32) -> &'static str {
    match status {
        0 => "SUCCESS",
        1 => "LOC_LEN_ERR",
        2 => "LOC_QP_OP_ERR",
        4 => "LOC_PROT_ERR",
        5 => "WR_FLUSH_ERR",
        10 => "REM_ACCESS_ERR",
        12 => "REM_OP_ERR",
        13 => "RETRY_EXC_ERR",
        14 => "RNR_RETRY_EXC_ERR",
        _ => "OTHER",
    }
}

impl Drop for RcQp {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_qp(self.qp.as_ptr());
        }
    }
}
