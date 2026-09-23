//! Mock transport implementing [`multi_rail::RailReader`] for hardware-free
//! functional and fault-injection testing of the multi-rail read path.
//!
//! Each `MockRailClient` is one rail with a configurable bandwidth. The shared
//! `MockStore` holds object bytes in-memory, so a multi-rail read exercises
//! the real scheduler / aggregator / consistency logic without any RDMA NIC.

use crate::multi_rail::{RailReader, RailRegistration};
use crate::pb;
use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Shared in-memory object store used by all mock rails.
#[derive(Default)]
pub struct MockStore {
    pub objects: std::collections::HashMap<String, Vec<u8>>,
}

/// Fault injection for the use-after-free hazard.
///
/// When `late_write_after_cancel` is set, after a successful read this rail
/// schedules a *late* write that targets `victim_region` of the buffer, guarded
/// by `epoch` / `live`. The late writer only proceeds when `live == epoch`
/// (i.e. the buffer is still live for that read); bumping `live` (simulating a
/// cancel / free) makes the guard skip the write, which is exactly the
/// defense against a late RDMA WRITE corrupting a reused buffer.
#[derive(Clone, Default)]
pub struct MockFault {
    pub late_write_after_cancel: bool,
    pub epoch: Option<Arc<AtomicU64>>,
    pub live: Option<Arc<AtomicU64>>,
    pub victim_region: Option<(usize, usize)>,
    pub garbage: u8,
}

pub struct MockRailClient {
    rail_id: String,
    bandwidth_bps: f64,
    store: Arc<Mutex<MockStore>>,
    fault: MockFault,
}

impl MockRailClient {
    pub fn new(rail_id: impl Into<String>, bandwidth_bps: f64, store: Arc<Mutex<MockStore>>) -> Self {
        Self {
            rail_id: rail_id.into(),
            bandwidth_bps,
            store,
            fault: MockFault::default(),
        }
    }

    pub fn with_fault(mut self, fault: MockFault) -> Self {
        self.fault = fault;
        self
    }
}

impl RailReader for MockRailClient {
    fn rail_id(&self) -> String {
        self.rail_id.clone()
    }

    unsafe fn register(&mut self, base: *mut u8, len: usize) -> Result<RailRegistration> {
        Ok(RailRegistration {
            addr: base as u64,
            rkey: 0,
            len,
        })
    }

    fn read_stripes(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        stripes: &[u32],
        segments: &[(u64, u32, u64)],
    ) -> Result<usize> {
        let canon = canonical_key(descriptor);
        let store = self.store.lock().unwrap();
        let data = store
            .objects
            .get(&canon)
            .ok_or_else(|| anyhow::anyhow!("mock object missing: {canon}"))?;
        let chunk = descriptor.chunk_size as usize;
        let mut copied = 0usize;
        for (k, &si) in stripes.iter().enumerate() {
            let off = (si as usize) * chunk;
            let l = chunk.min(data.len().saturating_sub(off));
            if l == 0 {
                continue;
            }
            let dst = segments[k].0 as *mut u8;
            std::ptr::copy_nonoverlapping(data[off..off + l].as_ptr(), dst, l);
            copied += l;
            if self.bandwidth_bps > 0.0 {
                let secs = (l as f64 / self.bandwidth_bps).min(0.2);
                if secs > 0.0 {
                    thread::sleep(Duration::from_secs_f64(secs));
                }
            }
        }
        drop(store);

        // Fault injection: a late write after cancel, guarded by the epoch.
        if self.fault.late_write_after_cancel {
            if let (Some(epoch), Some(live), Some((voff, vlen))) =
                (&self.fault.epoch, &self.fault.live, self.fault.victim_region)
            {
                let captured = epoch.load(Ordering::SeqCst);
                let garbage = self.fault.garbage;
                let base = segments.first().map(|s| s.0 as usize).unwrap_or(0);
                let live = Arc::clone(live);
                thread::spawn(move || {
                    thread::sleep(Duration::from_millis(20));
                    // Epoch guard: only corrupt if the buffer is still live for
                    // THIS read. A cancel/free bumps `live`, so the write skips.
                    if live.load(Ordering::SeqCst) == captured {
                        unsafe {
                            std::ptr::write_bytes((base + voff) as *mut u8, garbage, vlen);
                        }
                    }
                });
            }
        }
        Ok(copied)
    }
}

/// Mirror of `RdmaClient::canonical_key` so mock objects key consistently.
fn canonical_key(d: &pb::ObjectDescriptor) -> String {
    match &d.key {
        Some(k) => format!("{}:{}{}", k.namespace.len(), k.namespace, k.object_key),
        None => String::new(),
    }
}
