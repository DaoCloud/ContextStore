//! `cs-mock-bench` — a bandwidth capacity-model benchmark for multi-rail reads
//! (**no RDMA hardware required**).
//!
//! Feeds `MultiRailReader` with `MockRailClient`s (each rail has a configurable
//! bandwidth) and sweeps the rail count 1..=N, printing the "N-rail aggregate
//! bandwidth vs single rail" curve and the bottleneck attribution.
//!
//! This is the strongest defense-grade material: it uses a configurable
//! per-rail-bandwidth capacity model to demonstrate that multi-rail stacks the
//! per-NIC bandwidth ceiling, while the contest brief acknowledges that
//! Soft-RoCE / Mock can only prove functional and failure semantics, not the
//! hardware aggregate bandwidth — so we honestly label it a *model*.
//!
//! Run on any machine with Rust (including native Windows):
//! ```text
//! cargo run --bin cs-mock-bench
//! OBJ_SIZE=$((128*1024*1024)) cargo run --bin cs-mock-bench   # 128 MiB object
//! ```
//! Default features; does not depend on libibverbs / GPU.

use contextstore_client_rs::mock_rail::{MockRailClient, MockStore};
use contextstore_client_rs::multi_rail::{RailManager, RailReadStats, RailReader};
use contextstore_client_rs::pb;
use std::sync::{Arc, Mutex};

const NS: &str = "bench";
const KEY: &str = "multi-rail-object";

/// Object key format that must stay identical to `mock_rail::canonical_key`.
fn canon(ns: &str, key: &str) -> String {
    format!("{}:{}{}", ns.len(), ns, key)
}

fn main() {
    // ---- Tunable parameters (overridable via environment variables) ----
    let object_size: u64 = std::env::var("OBJ_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64 * 1024 * 1024); // 64 MiB
    let chunk_size: u64 = std::env::var("CHUNK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4 * 1024 * 1024); // 4 MiB / stripe by default; 1 MiB shows finer scaling
    let max_rails: usize = std::env::var("MAX_RAILS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    // Per-rail bandwidth: models the throughput ceiling of one Soft-RoCE soft NIC
    // (here 1 Gbps ≈ 125 MB/s). Shrink/grow it and the curve slope follows —
    // this is the "capacity model" knob.
    let per_rail_bps: f64 = std::env::var("PER_RAIL_MBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|mbps| mbps * 1024.0 * 1024.0)
        .unwrap_or(125.0 * 1024.0 * 1024.0);

    let stripe_count = object_size.div_ceil(chunk_size) as u32;

    // ---- Build the mock object (shared store; every rail reads the same data) ----
    let data = Arc::new(vec![0xABu8; object_size as usize]);
    let store = Arc::new(Mutex::new(MockStore::default()));
    store
        .lock()
        .unwrap()
        .objects
        .insert(canon(NS, KEY), Arc::clone(&data));

    let descriptor = pb::ObjectDescriptor {
        key: Some(pb::ObjectKey {
            namespace: NS.to_string(),
            object_key: KEY.to_string(),
        }),
        object_handle: String::new(),
        object_generation: 1,
        content_etag: String::new(),
        layout_version: 1,
        size: object_size,
        is_striped: true,
        stripe_count,
        chunk_size,
    };

    // Consistency layer: precompute each stripe's checksum, hand it to read(),
    // and compare stripe-by-stripe after aggregation.
    let mut checksums: Vec<Option<String>> = Vec::with_capacity(stripe_count as usize);
    for i in 0..stripe_count as usize {
        let off = (i as u64) * chunk_size;
        let l = chunk_size.min(object_size - off) as usize;
        checksums.push(Some(contextstore_client_rs::multi_rail::stripe_checksum(
            &data[off as usize..off as usize + l],
        )));
    }

    println!("=== ContextStore multi-rail read · Mock bandwidth capacity model ===");
    println!(
        "object={} MiB  chunk={} MiB  stripes={}  per_rail={:.0} MB/s",
        object_size / 1024 / 1024,
        chunk_size / 1024 / 1024,
        stripe_count,
        per_rail_bps / 1024.0 / 1024.0
    );
    println!(
        "{:<6} {:<11} {:<11} {:<11} {:<11} {:<9}  bottleneck",
        "rails", "agg_MB/s", "bal_MB/s", "theo_MB/s", "obj_ms", "speedup"
    );

    for r in 1..=max_rails {
        // Build r rails (same mock bandwidth each), hand them to RailManager.
        let mut rails: Vec<Box<dyn RailReader>> = Vec::with_capacity(r);
        for i in 0..r {
            rails.push(Box::new(MockRailClient::new(
                format!("mock-{i}"),
                per_rail_bps,
                store.clone(),
            )) as Box<dyn RailReader>);
        }
        let mut manager = RailManager::new(rails);
        let mut reader = manager.reader();
        let mut buf = vec![0u8; object_size as usize];

        // Warm-up read (discarded): first-touch page faults on `buf` would
        // otherwise be charged to the measured run and flatten the curve.
        let _ = reader.read(&descriptor, &checksums, &mut buf).ok();
        buf.iter_mut().for_each(|b| *b = 0);

        // Timed read. Use the transfer-phase goodput reported by the reader
        // (excludes checksum verification) so the number reflects rail bandwidth.
        let stats: RailReadStats = reader
            .read(&descriptor, &checksums, &mut buf)
            .unwrap_or_else(|e| panic!("rail_count={r} read failed: {e}"));
        manager.reclaim(reader);

        let agg_mbps = stats.transfer_mbps;
        // Balanced (ideal) aggregate: each rail carries ceil/floor(stripes/r)
        // stripes, so the *slowest* rail sets the object time. With indivisible
        // stripe counts this is slightly below `r × per_rail` — the gap is the
        // stripe-quantization effect, not a scheduling defect.
        let max_stripes_on_a_rail = (stripe_count as usize).div_ceil(r);
        let bal_mbps = per_rail_bps * (stripe_count as f64)
            / (max_stripes_on_a_rail as f64)
            / (1024.0 * 1024.0);
        let theo_mbps = per_rail_bps * (r as f64) / (1024.0 * 1024.0);
        let speedup = agg_mbps / (per_rail_bps / (1024.0 * 1024.0));
        println!(
            "{:<6} {:<11.1} {:<11.1} {:<11.1} {:<11.2} {:<9.2}x  {}",
            r,
            agg_mbps,
            bal_mbps,
            theo_mbps,
            stats.object_ms,
            speedup,
            stats.bottleneck()
        );
    }

    println!(
        "\nNote: agg = measured transfer-phase throughput; bal = ideal value under stripe quantization\n\
         (the slowest rail carries ceil(stripes/rails) stripes, so it sits slightly below theo when rails\n\
         do not divide stripes evenly); theo = r x per-rail bandwidth ideal upper bound. speedup is 1.00x at 1 rail.\n\
         Observation: as rails go 1..={max_rails}, agg approaches bal and the bottleneck should read network;\n\
         if agg stalls while the bottleneck flips to storage, the backend disk/JBOF aggregate bandwidth hit its ceiling first.\n\
         Caveat: this is a capacity model, not a hardware measurement; real multi-rail aggregate bandwidth needs physical RDMA NICs (a known contest limitation).",
        max_rails = max_rails
    );
}
