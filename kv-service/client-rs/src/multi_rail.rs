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
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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
    /// Wall time of the parallel transfer phase only (no checksum verification).
    pub object_ms: f64,
    /// Time spent building the stripe→rail plan.
    pub schedule_ms: f64,
    /// Per-rail transfer time (each rail runs in parallel).
    pub per_rail_ms: Vec<f64>,
    /// Time spent on post-aggregation stripe checksum verification.
    pub verify_ms: f64,
    /// Aggregate goodput the transfer phase achieved, in MB/s.
    pub transfer_mbps: f64,
    pub verify_ok: bool,
}

impl RailReadStats {
    /// Coarse bottleneck classifier for the report / demo.
    ///
    /// Attributes the read's wall time to one of: client scheduling, the
    /// slowest single rail (i.e. network / single-NIC bandwidth), post-read
    /// verification, or storage behind the rails.
    ///
    /// NOTE: on Soft-RoCE / Mock this is a *model* of where the bottleneck
    /// sits, not a hardware microbenchmark. The honest claim is that multi-rail
    /// removes the single-NIC bandwidth cap; it does NOT prove hardware
    /// aggregate bandwidth, which requires real RDMA NICs.
    pub fn bottleneck(&self) -> &'static str {
        let max_rail = self.per_rail_ms.iter().cloned().fold(0.0_f64, f64::max);
        let span = self.object_ms.max(max_rail).max(1e-9);

        // Verification dominating the read is a client-side software cost.
        if self.verify_ms > span * 0.5 {
            return "software: post-read stripe verification (client CPU)";
        }
        // Scheduling overhead dominating means the client is the bottleneck.
        if self.schedule_ms > span * 0.5 {
            return "software: scheduling / serialization on the client";
        }
        // The slowest rail tracks the whole transfer → that rail *is* the limit.
        // With N rails each carrying 1/N of the object, this is the network /
        // single-path cap that multi-rail is designed to remove.
        if max_rail > 0.0 && self.object_ms <= max_rail * 1.25 {
            "network: slowest rail caps the object (multi-rail spreads load across rails)"
        } else {
            "storage: backend (disk / JBOF) slower than the rail aggregate"
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

/// Options controlling the robustness behavior of a multi-rail read.
///
/// All fields default to "permissive" so [`MultiRailReader::read`] (which uses
/// [`ReadOptions::default`]) is a drop-in for the original 3-argument API.
#[derive(Clone, Debug, Default)]
pub struct ReadOptions {
    /// Per-read deadline. When `Some`, the read fails safely (without touching
    /// the caller's buffer) if not all rails complete in time.
    pub timeout: Option<Duration>,
    /// Expected `object_generation` observed at lookup. When `Some`, a read
    /// whose descriptor generation differs is rejected as a stale read.
    pub expected_generation: Option<u64>,
    /// Per-read inflight budget in bytes. When `Some(n)` with `n > 0`, an object
    /// larger than `n` is rejected with a backpressure error.
    pub max_inflight_bytes: Option<u64>,
}

/// Perform one rail's stripe transfer. Extracted so both the scoped (no-deadline)
/// path and the spawned (deadline) path share identical transfer + fault logic.
///
/// # Safety
/// `base..base+total_len` must be valid, writable, and alive for the duration of
/// the call; segments point into disjoint sub-ranges of it.
unsafe fn do_rail_read(
    rail: &mut Box<dyn RailReader>,
    descriptor: &pb::ObjectDescriptor,
    plan: &StripePlan,
    base: *mut u8,
    total_len: usize,
) -> RailOutcome {
    if plan.stripes.is_empty() {
        return RailOutcome::default();
    }
    match rail.register(base, total_len) {
        Ok(reg) => {
            let mut segments = Vec::with_capacity(plan.stripes.len());
            for k in 0..plan.stripes.len() {
                segments.push((
                    reg.addr + plan.offsets[k] as u64,
                    reg.rkey,
                    plan.lens[k] as u64,
                ));
            }
            match rail.read_stripes(descriptor, &plan.stripes, &segments) {
                Ok(bytes) => RailOutcome {
                    bytes,
                    ok: true,
                    error: None,
                },
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

/// Verify per-stripe checksums and assemble [`RailReadStats`].
///
/// Shared by both read paths. Returns an error (without leaking partial data to
/// the caller) when any checksum mismatches or any rail failed.
fn aggregate(
    plans: &[StripePlan],
    per_rail: &[RailTimed],
    buffer: &[u8],
    checksums: &[Option<String>],
    object_ms: f64,
    schedule_ms: f64,
) -> Result<RailReadStats> {
    let total_bytes: u64 = per_rail.iter().map(|p| p.outcome.bytes as u64).sum();
    let per_rail_ms: Vec<f64> = per_rail
        .iter()
        .map(|p| p.elapsed.as_secs_f64() * 1000.0)
        .collect();

    // Consistency layer: verify per-stripe checksum after aggregation.
    let verify_start = Instant::now();
    let mut verify_ok = true;
    for plan in plans {
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
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    let transfer_mbps = if object_ms > 0.0 {
        (total_bytes as f64) / (object_ms / 1000.0) / (1024.0 * 1024.0)
    } else {
        0.0
    };

    let stats = RailReadStats {
        rail_count: per_rail.len(),
        total_bytes,
        object_ms,
        schedule_ms,
        per_rail_ms,
        verify_ms,
        transfer_mbps,
        verify_ok,
    };

    if !verify_ok {
        return Err(anyhow!(
            "stripe checksum mismatch after multi-rail aggregation"
        ));
    }
    if per_rail.iter().any(|p| !p.outcome.ok) {
        let details: Vec<String> = per_rail
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.outcome.ok)
            .map(|(i, p)| {
                format!(
                    "rail {i}: {}",
                    p.outcome.error.as_deref().unwrap_or("unknown error")
                )
            })
            .collect();
        return Err(anyhow!(
            "one or more rails failed during multi-rail read: {}",
            details.join("; ")
        ));
    }
    Ok(stats)
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
    /// Read a striped object into `buffer` across all rails.
    ///
    /// Convenience wrapper around [`MultiRailReader::read_ex`] with default
    /// options (no timeout, no generation guard, no inflight budget). Kept for
    /// backward compatibility with `cs-mock-bench` and the existing tests.
    pub fn read(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        checksums: &[Option<String>],
        buffer: &mut [u8],
    ) -> Result<RailReadStats> {
        self.read_ex(descriptor, checksums, buffer, ReadOptions::default())
    }

    /// Read with explicit robustness controls.
    ///
    /// `opts.timeout` adds a per-read deadline (the read fails safely instead of
    /// hanging when a rail stalls; the caller's buffer is never touched on
    /// timeout). `opts.expected_generation` rejects a read whose descriptor
    /// generation drifted from the one observed at lookup (stale-read guard).
    /// `opts.max_inflight_bytes` enforces a per-read inflight budget and rejects
    /// oversized objects with a backpressure error instead of risking OOM.
    pub fn read_ex(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        checksums: &[Option<String>],
        buffer: &mut [u8],
        opts: ReadOptions,
    ) -> Result<RailReadStats> {
        let rail_count = self.rails.len();
        if rail_count == 0 {
            return Err(anyhow!("no rails configured"));
        }

        // Resource backpressure: reject objects larger than the configured
        // inflight budget before spending any work. A multi-rail read pins
        // `size` bytes of registered memory, so an unbounded object can OOM a
        // client; we fail fast and let the caller fall back to chunked or
        // single-rail reads. This is the inflight-budget guard described in the contest brief.
        if let Some(max_inflight) = opts.max_inflight_bytes {
            if max_inflight > 0 && descriptor.size > max_inflight {
                return Err(anyhow!(
                    "object size {} exceeds multi-rail inflight budget {} bytes (backpressure)",
                    descriptor.size,
                    max_inflight
                ));
            }
        }

        // Stale-read guard: if the descriptor's generation no longer matches the
        // one observed at lookup time, the placement may have changed under us.
        // Refuse to read rather than risk returning mismatched stripe data.
        if let Some(expected) = opts.expected_generation {
            if descriptor.object_generation != expected {
                return Err(anyhow!(
                    "object generation mismatch: expected {} but descriptor has {} (stale read rejected)",
                    expected, descriptor.object_generation
                ));
            }
        }

        if let Some(timeout) = opts.timeout {
            return self.read_with_deadline(descriptor, checksums, buffer, timeout);
        }

        // ---- Normal parallel path (no deadline) ----
        let plan_start = Instant::now();
        let stripe_count = descriptor.stripe_count as usize;
        let locality = vec![None; stripe_count];
        let plans = plan_stripes(
            descriptor.size,
            descriptor.chunk_size,
            descriptor.stripe_count,
            rail_count,
            &locality,
        );
        let schedule_ms = plan_start.elapsed().as_secs_f64() * 1000.0;

        let ptr = buffer.as_mut_ptr();
        let len = buffer.len();
        let schedule_start = Instant::now();

        let mut rails = std::mem::take(&mut self.rails);
        // All handles are created AND joined inside the scope closure: a
        // `ScopedJoinHandle` borrows the scope, so it must not outlive it.
        let per_rail: Vec<RailTimed> = thread::scope(|s| {
            let mut handles = Vec::with_capacity(rail_count);
            for (i, rail) in rails.iter_mut().enumerate() {
                let plan = &plans[i];
                let desc = descriptor;
                // Raw pointers are !Send; pass the address as `usize` so the
                // closure stays Send for `thread::scope`. Cast back inside.
                let base = ptr as usize;
                let blen = len;
                handles.push(s.spawn(move || {
                    let t0 = Instant::now();
                    let outcome = if plan.stripes.is_empty() {
                        // No stripes assigned to this rail: nothing to transfer.
                        RailOutcome::default()
                    } else {
                        unsafe { do_rail_read(rail, desc, plan, base as *mut u8, blen) }
                    };
                    RailTimed {
                        outcome,
                        elapsed: t0.elapsed(),
                    }
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("rail thread panicked"))
                .collect()
        });
        self.rails = rails;

        // Transfer phase wall time: measures the parallel rails only, so the
        // bottleneck classifier is not polluted by post-read verification.
        let object_ms = schedule_start.elapsed().as_secs_f64() * 1000.0;
        aggregate(&plans, &per_rail, buffer, checksums, object_ms, schedule_ms)
    }

    /// Deadline-capable read. Spawns one thread per rail, each writing into a
    /// private internal buffer (owned via `Arc`), and completes only when every
    /// rail signals. If the deadline passes first, the read returns an error
    /// **without touching the caller's buffer** — the in-flight rail threads
    /// keep their `Arc` clones, so the internal destination stays alive and no
    /// use-after-free can occur. This is what makes a multi-rail read safe under
    /// a single-rail stall (the timeout/disconnect robustness requirement).
    ///
    /// NOTE: a timeout-capable read *consumes* the rails (they are owned by the
    /// rail threads and dropped when those finish); rebuild the reader for a
    /// retry. The caller's `buffer` is only written after a clean, verified
    /// success.
    fn read_with_deadline(
        &mut self,
        descriptor: &pb::ObjectDescriptor,
        checksums: &[Option<String>],
        buffer: &mut [u8],
        timeout: Duration,
    ) -> Result<RailReadStats> {
        let rail_count = self.rails.len();
        let plan_start = Instant::now();
        let stripe_count = descriptor.stripe_count as usize;
        let locality = vec![None; stripe_count];
        let plans = plan_stripes(
            descriptor.size,
            descriptor.chunk_size,
            descriptor.stripe_count,
            rail_count,
            &locality,
        );
        let schedule_ms = plan_start.elapsed().as_secs_f64() * 1000.0;
        let schedule_start = Instant::now();

        let size = descriptor.size as usize;
        let shared: Arc<Vec<u8>> = Arc::new(vec![0u8; size]);
        let base = shared.as_ptr() as *mut u8;

        let results = Arc::new(Mutex::new(vec![None; rail_count]));
        let (tx, rx) = mpsc::channel::<usize>();
        let mut handles = Vec::with_capacity(rail_count);

        let rails = std::mem::take(&mut self.rails);
        for (i, mut rail) in rails.into_iter().enumerate() {
            let plan = plans[i].clone();
            let desc = descriptor.clone();
            let _buf_arc = Arc::clone(&shared);
            let res_slot = Arc::clone(&results);
            let tx = tx.clone();
            let base_usize = base as usize;
            let h = thread::spawn(move || {
                let t0 = Instant::now();
                let outcome =
                    unsafe { do_rail_read(&mut rail, &desc, &plan, base_usize as *mut u8, size) };
                res_slot.lock().unwrap()[i] = Some(RailTimed {
                    outcome,
                    elapsed: t0.elapsed(),
                });
                let _ = tx.send(i);
            });
            handles.push(h);
        }
        drop(tx); // channel closes once all rail threads (and this) drop

        let mut completed = 0usize;
        let deadline = Instant::now() + timeout;
        let timed_out = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(remaining) {
                Ok(_) => {
                    completed += 1;
                    if completed == rail_count {
                        break false;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break true,
                Err(mpsc::RecvTimeoutError::Disconnected) => break completed < rail_count,
            }
        };

        if timed_out {
            // Safe abandonment: each rail thread still holds an `Arc` clone of
            // `shared`, so the destination buffer stays valid until those
            // threads finish (they will, in the background). The caller's
            // `buffer` is never written on a timeout. The rails are consumed by
            // the detached threads and freed when they complete; rebuild the
            // reader for a retry.
            drop(handles);
            self.rails = Vec::new();
            return Err(anyhow!(
                "multi-rail read timed out after {:.0?} ({}/{} rails completed)",
                timeout,
                completed,
                rail_count
            ));
        }

        // All rails completed within the deadline: join (cheap now) and aggregate.
        for h in handles {
            let _ = h.join();
        }
        let per_rail = results
            .lock()
            .unwrap()
            .iter()
            .map(|o| o.clone().unwrap())
            .collect::<Vec<_>>();
        let object_ms = schedule_start.elapsed().as_secs_f64() * 1000.0;
        let stats = aggregate(
            &plans,
            &per_rail,
            &shared[..],
            checksums,
            object_ms,
            schedule_ms,
        )?;
        // Only copy to the caller's buffer after a clean, verified success.
        let n = size.min(buffer.len());
        buffer[..n].copy_from_slice(&shared[..n]);
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

/// Stripe checksum, byte-for-byte compatible with the server's
/// `StripingInfo.chunk_checksums` (see `kv-service/server/src/storage_tier.rs`
/// `checksum_bytes`: `format!("{:016x}", twox_hash::xxh3::hash64(data))`).
///
/// The client computes the same xxh3-64 digest, so the multi-rail consistency
/// layer verifies against the *server's* checksums rather than a private
/// algorithm — this is what makes cross-path data verification meaningful.
pub fn stripe_checksum(data: &[u8]) -> String {
    format!("{:016x}", twox_hash::xxh3::hash64(data))
}
