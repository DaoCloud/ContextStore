//! RDMA Context — manages HCA device, PD, CQ, and other shared resources

use anyhow::{anyhow, Result};
use rdma_sys::*;
use std::ffi::CStr;
use std::ptr::{self, NonNull};

/// RDMA context for a single HCA (device + PD + CQ).
///
/// A server process usually uses one HCA. Different client connections share the same context but hold independent QPs.
///
/// # Safety
/// Internally holds a raw verbs handle (NonNull); deallocated on Drop.
pub struct RdmaContext {
    /// libibverbs device context
    pub ctx: NonNull<ibv_context>,
    /// Protection domain — all MRs/QPs belong to this PD
    pub pd: NonNull<ibv_pd>,
    /// Completion queue (shared; multiple QPs poll the same CQ)
    pub cq: NonNull<ibv_cq>,
    /// HCA port (typically 1)
    pub port_num: u8,
    /// GID index (RoCE v2 is usually 3)
    pub gid_index: u8,
    /// Local GID, exchanged with the client over the control plane
    pub local_gid: ibv_gid,
}

// raw pointer + NonNull do not auto-implement Send/Sync, but libibverbs is internally thread-safe
// (sharing PD/CQ across threads is standard verbs behavior). Assert it manually.
unsafe impl Send for RdmaContext {}
unsafe impl Sync for RdmaContext {}

impl RdmaContext {
    /// Open the HCA by device name (e.g. "mlx5_0").
    ///
    /// `gid_index` is usually 3 (RoCE v2 IPv4-mapped GID; see `show_gids` output).
    pub fn open(device_name: &str, port_num: u8, gid_index: u8) -> Result<Self> {
        unsafe {
            // ===== 1. find the device =====
            let mut num = 0i32;
            let dev_list = ibv_get_device_list(&mut num);
            if dev_list.is_null() || num == 0 {
                return Err(anyhow!("no RDMA device found"));
            }

            // Iterate the list to find the device whose name matches
            let mut found_dev: *mut ibv_device = ptr::null_mut();
            for i in 0..num {
                let d = *dev_list.offset(i as isize);
                let name_ptr = ibv_get_device_name(d);
                let name = CStr::from_ptr(name_ptr).to_string_lossy();
                if name == device_name {
                    found_dev = d;
                    break;
                }
            }

            if found_dev.is_null() {
                ibv_free_device_list(dev_list);
                return Err(anyhow!("RDMA device '{}' not found", device_name));
            }

            // ===== 2. open the device context =====
            let ctx_raw = ibv_open_device(found_dev);
            ibv_free_device_list(dev_list); // list can be freed, but the device handle stays open
            let ctx = NonNull::new(ctx_raw)
                .ok_or_else(|| anyhow!("ibv_open_device failed: {}", std::io::Error::last_os_error()))?;

            // ===== 3. allocate PD =====
            let pd_raw = ibv_alloc_pd(ctx.as_ptr());
            let pd = NonNull::new(pd_raw).ok_or_else(|| {
                ibv_close_device(ctx.as_ptr());
                anyhow!("ibv_alloc_pd failed")
            })?;

            // ===== 4. create CQ =====
            // cq depth 1024 is enough for a single server: a GET has at most 32 chunks
            let cq_raw = ibv_create_cq(ctx.as_ptr(), 1024, ptr::null_mut(), ptr::null_mut(), 0);
            let cq = NonNull::new(cq_raw).ok_or_else(|| {
                ibv_dealloc_pd(pd.as_ptr());
                ibv_close_device(ctx.as_ptr());
                anyhow!("ibv_create_cq failed")
            })?;

            // ===== 5. query local GID =====
            let mut gid: ibv_gid = std::mem::zeroed();
            let rc = ibv_query_gid(ctx.as_ptr(), port_num, gid_index as i32, &mut gid);
            if rc != 0 {
                ibv_destroy_cq(cq.as_ptr());
                ibv_dealloc_pd(pd.as_ptr());
                ibv_close_device(ctx.as_ptr());
                return Err(anyhow!("ibv_query_gid failed: rc={}", rc));
            }

            tracing::info!(
                "RdmaContext: opened device={} port={} gid_index={} gid={:02x?}",
                device_name, port_num, gid_index,
                &gid.raw[..]
            );

            Ok(Self { ctx, pd, cq, port_num, gid_index, local_gid: gid })
        }
    }

    /// Register a memory region as an MR for RDMA WRITE/READ.
    ///
    /// The returned MR must be explicitly dropped / `ibv_dereg_mr`; this wraps it in an RAII guard.
    pub fn register_mr(&self, buf: &mut [u8], access: u32) -> Result<MemRegion> {
        unsafe {
            let mr_raw = ibv_reg_mr(
                self.pd.as_ptr(),
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                buf.len(),
                access as i32,
            );
            let mr = NonNull::new(mr_raw)
                .ok_or_else(|| anyhow!("ibv_reg_mr failed: {}", std::io::Error::last_os_error()))?;
            Ok(MemRegion {
                mr,
                addr: buf.as_ptr() as u64,
                length: buf.len() as u64,
                lkey: (*mr_raw).lkey,
                rkey: (*mr_raw).rkey,
            })
        }
    }

    /// Register a raw pointer (bypasses `&mut [u8]` borrow checking). Useful for Bytes-internal buffers.
    ///
    /// # Safety
    /// Caller must ensure ptr/len point at valid memory that outlives the MR.
    pub unsafe fn register_mr_raw(
        &self,
        ptr: *mut u8,
        len: usize,
        access: u32,
    ) -> Result<MemRegion> {
        let mr_raw = ibv_reg_mr(
            self.pd.as_ptr(),
            ptr as *mut std::ffi::c_void,
            len,
            access as i32,
        );
        let mr = NonNull::new(mr_raw)
            .ok_or_else(|| anyhow!("ibv_reg_mr failed: {}", std::io::Error::last_os_error()))?;
        Ok(MemRegion {
            mr,
            addr: ptr as u64,
            length: len as u64,
            lkey: (*mr_raw).lkey,
            rkey: (*mr_raw).rkey,
        })
    }
}

impl Drop for RdmaContext {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_cq(self.cq.as_ptr());
            ibv_dealloc_pd(self.pd.as_ptr());
            ibv_close_device(self.ctx.as_ptr());
        }
    }
}

/// RAII wrapper for libibverbs MR (memory region).
pub struct MemRegion {
    mr: NonNull<ibv_mr>,
    pub addr: u64,
    pub length: u64,
    pub lkey: u32,
    pub rkey: u32,
}

unsafe impl Send for MemRegion {}
unsafe impl Sync for MemRegion {}

impl MemRegion {
    pub fn as_ptr(&self) -> *mut ibv_mr {
        self.mr.as_ptr()
    }
}

impl Drop for MemRegion {
    fn drop(&mut self) {
        unsafe {
            ibv_dereg_mr(self.mr.as_ptr());
        }
    }
}
