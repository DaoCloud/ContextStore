//! RDMA tier — bypasses gRPC, uses verbs RC WRITE to push chunks_cache directly to the client
//!
//! ## Architecture
//!
//! ```text
//! Control plane (TCP)              Data plane (RDMA verbs RC WRITE)
//! ─────────────────────           ─────────────────────────────────
//! Client TCP connect      →       Server accept
//!   send QP info (gid/qpn/psn)   send back QP info
//!   QP transition INIT→RTR→RTS    QP transition INIT→RTR→RTS
//!
//!   send REQ {key, dst_addr, dst_rkey, max_size}  →   look up chunks_cache
//!                                                      for chunk in chunks:
//!                                                         post_send(WRITE, src=cache_buf, dst=client_buf+offset)
//!                                                      poll completion
//!                                                      send back {bytes_written}
//!                                                      ←
//!   recv {bytes_written}, data already in client buffer
//! ```
//!
//! ## Key design decisions
//!
//! - **RC (Reliable Connection)**: WRITE is one-sided; the client does not need to actively recv
//! - **TCP control channel**: simple, no extra RDMA control protocol required
//! - **Server-initiated WRITE push**: client does not poll; the server sends a TCP ack when done
//! - **Server-side MR**: register the chunks_cache Bytes as an MR (note: registered once per cache insert, off the hot path)
//!   * Note: this version is simplified — temporarily registers single chunks without MR caching (TODO: cache)
//! - **Client-side MR**: the client's pinned host buffer is registered as an MR; server WRITE lands directly in it
//!
//! ## Future work
//! - GPUDirect RDMA: switch the client-side MR to GPU memory (requires nvidia_peermem to be loaded)
//! - MR pool: register on chunks_cache insert to avoid ibv_reg_mr overhead on every GET

#[cfg(feature = "rdma")]
pub mod context;
#[cfg(feature = "rdma")]
pub mod qp;
#[cfg(feature = "rdma")]
pub mod server;
#[cfg(feature = "rdma")]
pub mod slab;
#[cfg(feature = "rdma")]
pub mod wire;

#[cfg(not(feature = "rdma"))]
pub mod stub {
    //! When the rdma feature is off, provide a stub so upstream code still compiles.
    use anyhow::Result;

    pub struct RdmaTier;

    impl RdmaTier {
        pub fn new_disabled() -> Result<Self> {
            anyhow::bail!("RDMA tier not compiled in. Rebuild with --features rdma")
        }
    }
}
