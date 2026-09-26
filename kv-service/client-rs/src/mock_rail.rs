//! Mock transport implementing [`multi_rail::RailReader`] for hardware-free
//! functional and fault-injection testing of the multi-rail read path.
//!
//! Each `MockRailClient` is one rail with a configurable bandwidth. The shared
//! `MockStore` holds object bytes in-memory, so a multi-rail read exercises
//! the real scheduler / aggregator / consistency logic without any RDMA NIC.

use crate::multi_rail::{RailReader, RailRegistration};
use crate::pb;
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Shared in-memory object store used by all mock rails.
///
/// Objects are held behind `Arc<Vec<u8>>` and treated as **immutable** for the
/// duration of a read. That lets a rail clone the Arc under the lock and then do
/// all byte-copying / bandwidth simulation *outside* the lock, so multiple rails
/// truly transfer in parallel — which is what makes the mock a valid capacity
/// model of independent NICs rather than a serial pipe.
#[derive(Default)]
pub struct MockStore {
    pub objects: std::collections::HashMap<String, Arc<Vec<u8>>>,
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
    /// Silent data corruption: flip one byte of this stripe in transit so its
    /// checksum no longer matches the server's expectation. Exercises the
    /// client's post-aggregation consistency layer.
    pub corrupt_stripe: Option<u32>,
    /// Simulated stalled / hung rail (seconds). Exercises the reader's timeout
    /// path — the sleep happens outside the store lock, like bandwidth sim.
    pub stall_secs: f64,
    /// Simulated disconnect: `read_stripes` returns an error immediately.
    pub fail: bool,
    /// Under-delivery: copy only `n` bytes of stripe 0 then report success,
    /// leaving the rest of the region untouched. Exercises the consistency layer
    /// catching a rail that claims success but moves too little data.
    pub short_read: Option<usize>,
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
        // Fault injection: simulated disconnect. A real NIC/QP going down must
        // surface as an error so the aggregator fails the whole read safely
        // (it must never return partial or stale data to the caller).
        if self.fault.fail {
            return Err(anyhow::anyhow!("injected rail disconnect (fault injection)"));
        }

        let canon = canonical_key(descriptor);
        let chunk = descriptor.chunk_size as usize;

        // Phase 1 — clone the object Arc under the lock, then release it.
        // The store is *shared* across rails, so the lock must be held only for
        // this O(1) Arc clone. All byte movement and bandwidth simulation happen
        // below, outside the lock, so rails genuinely run in parallel. (Holding
        // the lock across the sleep would serialize every rail and destroy the
        // whole point of multi-rail — the earlier version had exactly that bug.)
        let obj: Arc<Vec<u8>> = {
            let store = self.store.lock().unwrap();
            store
                .objects
                .get(&canon)
                .ok_or_else(|| anyhow::anyhow!("mock object missing: {canon}"))?
                .clone()
        };

        // Fault injection: simulated stalled / hung rail (exercises the reader's
        // timeout path). The sleep happens outside the lock, like bandwidth sim.
        if self.fault.stall_secs > 0.0 {
            thread::sleep(Duration::from_secs_f64(self.fault.stall_secs));
        }

        // Phase 2 — copy stripes to their destination regions outside the lock.
        let mut copied = 0usize;
        let mut total_secs = 0f64;
        for (k, &si) in stripes.iter().enumerate() {
            let off = (si as usize) * chunk;
            let l = chunk.min(obj.len().saturating_sub(off));
            if l == 0 {
                continue;
            }
            // Fault injection: under-delivery. Copy fewer bytes than the stripe
            // needs (the trailing bytes of the region are left untouched),
            // simulating a rail that reports success but fails to move all its
            // data. The client's post-aggregation checksum must catch this and
            // fail the whole read rather than returning a truncated object.
            let copy_len = if let Some(n) = self.fault.short_read {
                if k == 0 {
                    l.min(n)
                } else {
                    l
                }
            } else {
                l
            };
            let dst = segments[k].0 as *mut u8;
            // SAFETY: the caller guarantees `segments[k].0` points to a writable
            // region of at least `l` bytes for this rail's registration, and the
            // region is disjoint across stripes (the scheduler assigns each stripe
            // to exactly one rail at its own object offset).
            unsafe {
                std::ptr::copy_nonoverlapping(obj[off..off + copy_len].as_ptr(), dst, copy_len);
            }

            // Fault injection: silent data corruption. Flip one byte of this
            // stripe in transit so its checksum no longer matches the server's
            // expectation; the consistency layer must reject the read.
            if self.fault.corrupt_stripe == Some(si) {
                unsafe {
                    let b = *dst;
                    *dst = b.wrapping_add(1);
                }
            }

            copied += copy_len;
            if self.bandwidth_bps > 0.0 {
                total_secs += copy_len as f64 / self.bandwidth_bps;
            }
        }

        // Phase 3 — simulate this rail's transfer time, still outside the lock.
        if total_secs > 0.0 {
            // Cap each rail's simulated transfer so a pathological object size
            // can't hang the benchmark; the cap is far above the sizes we use.
            thread::sleep(Duration::from_secs_f64(total_secs.min(5.0)));
        }

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
