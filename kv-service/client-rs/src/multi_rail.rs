//! Multi-rail parallel read layer for ContextStore KVService.
//!
//! A "rail" is one independent network path (an RDMA NIC + QP + CQ, or a Mock
//! transport). `MultiRailReader` partitions an object's stripes across the
//! available rails, issues one stripe-subset GET per rail concurrently, and
//! aggregates the completions into a single object read.
//!
//! Design invariants (required by the competition brief):
//! - The on-disk stripe layout (`StripingInfo`) is NEVER touched.
//! - The upper read interface (`LookupObject` / `ReadByDescriptor`) is NEVER
//!   touched; this layer only replaces the RDMA fast path used when a
//!   `PlacementDescriptor` is present.
//! - Single-rail (one NIC) deployments keep working: with one rail the
//!   scheduler is a pass-through and behavior is identical to today.

use crate::pb;
use anyhow::{anyhow, Result};
use std::thread;
use std::time::Instant;

/// Rail-specific registration of the caller's destination buffer.
///
/// For real RDMA this carries the local address + rkey of THIS rail's memory
/// region (the same physical buffer is registered once per rail, each rail
/// yielding a different rkey). For the Mock transport `rkey` is unused.
#[derive(Clone, Copy)]
pub struct RailRegistration {
    pub addr: u64,
    pub rkey: u32,
    pub len: usize,
}

/// One independent network path capable of moving a stripe subset.
pub trait RailReader: Send {
    /// Stable identifier, e.g. the RDMA device name (`mlx5_0`) or `mock-0`.
    fn rail_id(&self) -> String;

    /// Register `base..base+len` for this rail. For real RDMA this pins the
    /// region for the read's duration; the caller must keep `base` valid until
    /// every rail's `read_stripes` has returned. This is the use-after-free
    /// guard: never unregister / recycle a buffer while a transfer may still be
    /// in flight (a late RDMA WRITE could otherwise land in a reused region).
    ///
    /// # Safety
    /// `base..base+len` must be valid, writable, and alive until all rails
    /// finish. The region may be written by the NIC / Mock during the read.
    unsafe fn register(&mut self, base: *mut u8, len: usize) -> Result<RailRegistration>;

    /// Move `stripes` of `descriptor` into the registered destination regions.
    /// `segments` are `(addr, rkey, len)` for THIS rail's registration, one per
    /// stripe, in object-stripe order. Returns bytes the rail claims to have
    /// written (0 = object absent).
    fn read_stripes(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        stripes: &[u32],
        segments: &[(u64, u32, u64)],
    ) -> Result<usize>;
}

/// The stripe subset assigned to one rail.
#[derive(Clone, Debug, Default)]
pub struct StripePlan {
    pub stripes: Vec<u32>,
    pub offsets: Vec<usize>,
    pub lens: Vec<usize>,
}

/// Outcome of one rail's transfer.
#[derive(Clone, Debug, Default)]
pub struct RailOutcome {
    pub bytes: usize,
    pub ok: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct RailTimed {
    outcome: RailOutcome,
    elapsed: std::time::Duration,
}

/// Per-object multi-rail read statistics, used for bottleneck attribution.
#[derive(Clone, Debug)]
pub struct RailReadStats {
    pub rail_count: usize,
    pub total_bytes: u64,
    pub object_ms: f64,
    pub schedule_ms: f64,
    pub per_rail_ms: Vec<f64>,
    pub verify_ok: bool,
}

impl RailReadStats {
    /// Coarse bottleneck classifier for the report / demo.
    ///
    /// NOTE: on Soft-RoCE / Mock this is a *model* of where the bottleneck
    /// sits, not a hardware microbenchmark. The honest claim is that multi-rail
    /// removes the single-NIC bandwidth cap; it does NOT prove hardware
    /// aggregate bandwidth, which requires real RDMA NICs.
    pub fn bottleneck(&self) -> &'static str {
        let max_rail = self.per_rail_ms.iter().cloned().fold(0.0_f64, f64::max);
        if max_rail <= 0.0 {
            return "unknown";
        }
        if self.schedule_ms > max_rail * 0.5 {
            "software: scheduling / serialization on the client"
        } else if (self.object_ms - max_rail).abs() < max_rail * 0.25 {
            "network: single-NIC bandwidth (multi-rail aggregates rails in parallel)"
        } else {
            "storage: disk / JBOF aggregate bandwidth below rail aggregate"
        }
    }
}

/// Partition `stripe_count` stripes across `rail_count` rails.
///
/// `locality[i]` is the preferred rail for stripe `i` (e.g. the rail whose NIC
/// reaches the node that owns the stripe); `None` falls back to round-robin.
/// Dynamic least-loaded balancing can be layered on top by the caller.
pub fn plan_stripes(
    total_size: u64,
    chunk_size: u64,
    stripe_count: u32,
    rail_count: usize,
    locality: &[Option<usize>],
) -> Vec<StripePlan> {
    let mut plans = vec![StripePlan::default(); rail_count.max(1)];
    if rail_count == 0 {
        return plans;
    }
    let mut rr = 0usize;
    for i in 0..stripe_count as usize {
        let rail = locality
            .get(i)
            .copied()
            .flatten()
            .filter(|r| *r < rail_count)
            .unwrap_or_else(|| {
                let r = rr;
                rr = (rr + 1) % rail_count;
                r
            });
        let offset = i as u64 * chunk_size;
        let len = chunk_size.min(total_size.saturating_sub(offset)) as usize;
        plans[rail].stripes.push(i as u32);
        plans[rail].offsets.push(offset as usize);
        plans[rail].lens.push(len);
    }
    plans
}

/// Owns the live rail connections and performs multi-rail reads.
pub struct MultiRailReader {
    rails: Vec<Box<dyn RailReader>>,
}

impl MultiRailReader {
    pub fn new(rails: Vec<Box<dyn RailReader>>) -> Self {
        Self { rails }
    }

    /// Number of rails (1 = single-NIC compatible pass-through behavior).
    pub fn rail_count(&self) -> usize {
        self.rails.len()
    }

    /// Read a striped object into `buffer` across all rails.
    ///
    /// `checksums[i]` (when `Some`) is the expected xxh3-64 of stripe `i`,
    /// taken from `PlacementDescriptor.chunks[i].checksum`; used for the
    /// consistency layer after aggregation.
    pub fn read(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        checksums: &[Option<String>],
        buffer: &mut [u8],
    ) -> Result<RailReadStats> {
        let rail_count = self.rails.len();
        if rail_count == 0 {
            return Err(anyhow!("no rails configured"));
        }
        let plan_start = Instant::now();
        let stripe_count = descriptor.stripe_count as usize;
        let locality = vec![None; stripe_count];
        let plans =
            plan_stripes(descriptor.size, descriptor.chunk_size, descriptor.stripe_count, rail_count, &locality);
        let schedule_ms = plan_start.elapsed().as_secs_f64() * 1000.0;

        let ptr = buffer.as_mut_ptr();
        let len = buffer.len();
        let schedule_start = Instant::now();

        let mut rails = std::mem::take(&mut self.rails);
        let mut handles = Vec::with_capacity(rail_count);
        thread::scope(|s| {
            for (i, rail) in rails.iter_mut().enumerate() {
                let plan = &plans[i];
                let desc = descriptor;
                // Raw pointers are !Send; pass the address as `usize` so the
                // closure stays Send for `thread::scope`. Cast back inside.
                let base = ptr as usize;
                let blen = len;
                handles.push(s.spawn(move |_| {
                    let t0 = Instant::now();
                    let outcome = if plan.stripes.is_empty() {
                        // No stripes assigned to this rail: nothing to transfer.
                        RailOutcome::default()
                    } else {
                        unsafe {
                            match rail.register(base as *mut u8, blen) {
                                Ok(reg) => {
                                    let mut segments = Vec::with_capacity(plan.stripes.len());
                                    for k in 0..plan.stripes.len() {
                                        segments.push((
                                            reg.addr + plan.offsets[k] as u64,
                                            reg.rkey,
                                            plan.lens[k] as u64,
                                        ));
                                    }
                                    match rail.read_stripes(desc, &plan.stripes, &segments) {
                                        Ok(bytes) => RailOutcome { bytes, ok: true, error: None },
                                        Err(e) => RailOutcome {
                                            bytes: 0,
                                            ok: false,
                                            error: Some(e.to_string()),
                                        },
                                    }
                                }
                                Err(e) => RailOutcome {
                                    bytes: 0,
                                    ok: false,
                                    error: Some(e.to_string()),
                                },
                            }
                        }
                    };
                    RailTimed { outcome, elapsed: t0.elapsed() }
                }));
            }
        });

        let mut per_rail = vec![RailTimed::default(); rail_count];
        for (i, h) in handles.into_iter().enumerate() {
            per_rail[i] = h.join().expect("rail thread panicked");
        }
        self.rails = rails;

        // Consistency layer: verify per-stripe checksum after aggregation.
        let mut verify_ok = true;
        for plan in &plans {
            for k in 0..plan.stripes.len() {
                let si = plan.stripes[k] as usize;
                if let Some(expected) = checksums.get(si).and_then(|c| c.as_ref()) {
                    let off = plan.offsets[k];
                    let l = plan.lens[k];
                    let got = stripe_checksum(&buffer[off..off + l]);
                    if &got != expected {
                        verify_ok = false;
                    }
                }
            }
        }

        let object_ms = schedule_start.elapsed().as_secs_f64() * 1000.0;
        let total_bytes: u64 = per_rail.iter().map(|p| p.outcome.bytes as u64).sum();
        let per_rail_ms: Vec<f64> = per_rail.iter().map(|p| p.elapsed.as_secs_f64() * 1000.0).collect();

        let stats = RailReadStats {
            rail_count,
            total_bytes,
            object_ms,
            schedule_ms,
            per_rail_ms,
            verify_ok,
        };

        if !verify_ok {
            return Err(anyhow!("stripe checksum mismatch after multi-rail aggregation"));
        }
        if per_rail.iter().any(|p| !p.outcome.ok) {
            return Err(anyhow!("one or more rails failed during multi-rail read"));
        }
        Ok(stats)
    }

    /// Recover the rail connections for reuse.
    pub fn into_rails(self) -> Vec<Box<dyn RailReader>> {
        self.rails
    }
}

/// `RailManager` builds and owns the rail set (real RDMA devices or Mock).
pub struct RailManager {
    rails: Vec<Box<dyn RailReader>>,
}

impl RailManager {
    pub fn new(rails: Vec<Box<dyn RailReader>>) -> Self {
        Self { rails }
    }

    /// True when only a single NIC / path is available (backward compatible).
    pub fn is_single(&self) -> bool {
        self.rails.len() <= 1
    }

    pub fn rail_count(&self) -> usize {
        self.rails.len()
    }

    /// Hand the rails to a `MultiRailReader` for one or more reads.
    pub fn reader(&mut self) -> MultiRailReader {
        MultiRailReader::new(std::mem::take(&mut self.rails))
    }

    /// Reclaim rails after reading.
    pub fn reclaim(&mut self, reader: MultiRailReader) {
        self.rails = reader.into_rails();
    }
}

/// Lightweight stripe checksum. TODO(compete): replace with xxh3-64 to match
/// server `StripingInfo.chunk_checksums` so client-side verification is exact.
pub fn stripe_checksum(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}
