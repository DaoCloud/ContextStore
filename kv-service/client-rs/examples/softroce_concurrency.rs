//! `softroce_concurrency` — concurrency and resource-overhead sweep on top of
//! the multi-rail read path (contest deliverable e: "different object sizes
//! and concurrency levels ... CPU, memory, registered memory, inflight bytes").
//!
//! This complements `softroce_dual_rail` (which proves *correctness* of one
//! read) by sweeping **object size** and **concurrency** and reporting resource
//! consumption alongside throughput, so both trends are observable end to end
//! on Soft-RoCE.
//!
//! Topology / prerequisites are identical to `softroce_dual_rail` (see
//! `docs/wsl2-softroce-setup.md`): rxe0@192.168.96.110:50053,
//! rxe1@192.168.96.111:50054.
//!
//! Run:
//! ```text
//! cargo run -p contextstore-client-rs --features rdma --release \
//!   --example softroce_concurrency
//! ```
//!
//! Environment knobs (all optional):
//!   CS_GRPC            gRPC endpoint        (default http://127.0.0.1:50051)
//!   CS_RAIL_ENDPOINTS  RDMA control endpoints
//!   CS_RAIL_DEVICES    RDMA device per rail (default rxe0,rxe1)
//!   CS_RAIL_GID        GID index            (default 1 = RXE)
//!   CS_DEMO_MIB        single object size in MiB (used when CS_DEMO_MIBS unset)
//!   CS_DEMO_MIBS       comma list of object sizes to sweep (default 32,64,128,256)
//!   CS_CONC            comma list of worker counts to sweep (default 1,2)
//!   CS_ITERS           timed iterations per (size, mode, concurrency) (default 2)
//!   CS_MAX_WORKERS     hard cap on workers to bound client memory (default 2)
//!
//! Output: one table per object size, rows per (mode, concurrency) with
//! aggregate goodput, read-latency percentiles, and sampled process resources
//! (RSS, registered memory accounted by the client, inflight bytes).

use anyhow::{anyhow, ensure, Result};
use contextstore_client_rs::multi_rail::{
    stripe_checksum, RailManager, RailReader, ReadOptions,
};
use contextstore_client_rs::rdma::{RdmaClient, RdmaClientConfig};
use contextstore_client_rs::KvClient;
use std::sync::Arc;
use std::time::Instant;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_list(key: &str, default: &str) -> Vec<String> {
    env_or(key, default)
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Resident set size of this process in MiB, read from /proc (Linux only).
fn rss_mib() -> f64 {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: f64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
    }
    f64::NAN
}

struct Point {
    size_mib: usize,
    mode: &'static str,
    conc: usize,
    agg_mbps: f64,
    p50_ms: f64,
    p95_ms: f64,
    bytes: u64,
    rss_mib: f64,
    reg_mib: f64,
    inflight_mib: f64,
    all_verified: bool,
}

/// One worker: its own rail set and its own destination buffer. Under `conc`
/// workers we run `conc` of these concurrently, each performing `iters`
/// timed reads and reporting its per-read latencies.
fn run_worker(
    n_rails: usize,
    desc: &contextstore_client_rs::pb::ObjectDescriptor,
    checksums: &[Option<String>],
    expected: Arc<Vec<u8>>,
    iters: usize,
    endpoints: &[String],
    devices: &[String],
    gid: u8,
) -> Result<(Vec<f64>, u64, bool)> {
    let mut rails: Vec<Box<dyn RailReader>> = Vec::with_capacity(n_rails);
    for i in 0..n_rails {
        let config = RdmaClientConfig::new(endpoints[i].clone(), devices[i].clone())
            .with_gid_index(gid);
        rails.push(Box::new(RdmaClient::connect(config)?));
    }
    let mut mgr = RailManager::new(rails);
    let mut reader = mgr.reader();

    let mut lat = Vec::with_capacity(iters);
    let mut bytes = 0u64;
    let mut ok = true;
    for it in 0..iters {
        // A fresh destination buffer per iteration on purpose: this is the
        // shape that used to trip the address-keyed MR cache (the allocator
        // hands back the same address, a stale MR still pins the old physical
        // pages, and the read comes back all zeros). The default read path
        // now registers afresh per read, so this must reconstruct correctly.
        let mut buf = vec![0u8; desc.size as usize];
        let t = Instant::now();
        // Capture the result instead of `?` so a checksum failure still lets us
        // diagnose the assembled buffer (which rail's stripes are wrong / zero).
        let res = reader.read_ex(desc, checksums, &mut buf, ReadOptions::default());
        let elapsed = t.elapsed().as_secs_f64() * 1000.0;
        match res {
            Ok(_) => {
                lat.push(elapsed);
                bytes += desc.size;
                if buf != *expected {
                    ok = false;
                }
            }
            Err(e) => {
                // Diagnose: per-stripe correctness against locally derived
                // checksums, and which rail owns each wrong stripe.
                let chunk = desc.chunk_size as usize;
                let mut bad = Vec::new();
                for i in 0..desc.stripe_count as usize {
                    let start = i * chunk;
                    let end = ((i + 1) * chunk).min(buf.len());
                    let got = stripe_checksum(&buf[start..end]);
                    let want = checksums
                        .get(i)
                        .and_then(|c| c.as_ref())
                        .cloned()
                        .unwrap_or_default();
                    let all_zero = buf[start..end].iter().all(|&b| b == 0);
                    let matches_bytes = buf[start..end] == expected[start..end];
                    if got != want || !matches_bytes {
                        bad.push(format!(
                            "stripe {i} (rail{}, all_zero={all_zero}, bytes_match={matches_bytes})",
                            i % n_rails
                        ));
                    }
                }
                eprintln!(
                    "[diag] iter {it} {n_rails}-rail read failed: {e}\n\
                     [diag] stripes={} chunk={} size={} wrong={}/{} -> {:?}",
                    desc.stripe_count,
                    chunk,
                    desc.size,
                    bad.len(),
                    desc.stripe_count,
                    bad
                );
                mgr.reclaim(reader);
                return Err(e);
            }
        }
    }
    mgr.reclaim(reader);
    Ok((lat, bytes, ok))
}

fn percentile(v: &mut Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[idx]
}

fn measure(
    size_mib: usize,
    mode: &'static str,
    n_rails: usize,
    conc: usize,
    iters: usize,
    desc: &contextstore_client_rs::pb::ObjectDescriptor,
    checksums: &[Option<String>],
    expected: Arc<Vec<u8>>,
    endpoints: &[String],
    devices: &[String],
    gid: u8,
) -> Result<Point> {
    let rss_before = rss_mib();
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(conc);
    for _ in 0..conc {
        let desc = desc.clone();
        let checksums = checksums.to_vec();
        let expected = expected.clone();
        let eps = endpoints.to_vec();
        let devs = devices.to_vec();
        handles.push(std::thread::spawn(move || {
            run_worker(n_rails, &desc, &checksums, expected, iters, &eps, &devs, gid)
        }));
    }
    let mut all_lat = Vec::new();
    let mut bytes = 0u64;
    let mut ok = true;
    for h in handles {
        let (lat, b, o) = h.join().map_err(|_| anyhow!("worker thread panicked"))??;
        all_lat.extend(lat);
        bytes += b;
        ok &= o;
    }
    let wall = t0.elapsed().as_secs_f64();
    let rss_peak = rss_mib();

    let agg_mbps = (bytes as f64 / (1024.0 * 1024.0)) / wall;
    let p50 = percentile(&mut all_lat.clone(), 0.50);
    let p95 = percentile(&mut all_lat.clone(), 0.95);

    // Registered memory accounted by the client: each worker registers one
    // destination buffer per rail (pinned until the read joins).
    let reg_mib = (desc.size as f64 / (1024.0 * 1024.0)) * (conc as f64) * (n_rails as f64);
    // Inflight bytes at the read call = per-worker destination buffer.
    let inflight_mib = (desc.size as f64 / (1024.0 * 1024.0)) * conc as f64;

    Ok(Point {
        size_mib,
        mode,
        conc,
        agg_mbps,
        p50_ms: p50,
        p95_ms: p95,
        bytes,
        rss_mib: rss_peak.max(rss_before),
        reg_mib,
        inflight_mib,
        all_verified: ok,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let grpc = env_or("CS_GRPC", "http://127.0.0.1:50051");
    let endpoints = env_list(
        "CS_RAIL_ENDPOINTS",
        "192.168.96.110:50053,192.168.96.111:50054",
    );
    let devices = env_list("CS_RAIL_DEVICES", "rxe0,rxe1");
    let gid: u8 = env_or("CS_RAIL_GID", "1").parse()?;

    // Object-size sweep: CS_DEMO_MIBS wins, else the single CS_DEMO_MIB, else default.
    let size_spec: Vec<String> = if std::env::var("CS_DEMO_MIBS").is_ok() {
        env_list("CS_DEMO_MIBS", "32,64,128,256")
    } else if let Ok(single) = std::env::var("CS_DEMO_MIB") {
        vec![single]
    } else {
        vec![
            "32".to_string(),
            "64".to_string(),
            "128".to_string(),
            "256".to_string(),
        ]
    };
    let mibs: Vec<usize> = size_spec.iter().map(|s| s.parse().unwrap_or(64)).collect();

    let max_workers: usize = env_or("CS_MAX_WORKERS", "2").parse()?;
    let mut concs: Vec<usize> = env_list("CS_CONC", "1,2")
        .iter()
        .map(|s| s.parse().unwrap_or(1))
        .collect();
    for c in concs.iter_mut() {
        if *c > max_workers {
            println!(
                "[warn] concurrency {c} exceeds CS_MAX_WORKERS={max_workers}; clamping \
                 (client memory = workers x rails x object size)"
            );
            *c = max_workers;
        }
    }
    let iters: usize = env_or("CS_ITERS", "2").parse()?;

    ensure!(
        endpoints.len() == devices.len() && !endpoints.is_empty(),
        "CS_RAIL_ENDPOINTS and CS_RAIL_DEVICES must be non-empty and equal length"
    );

    let mut kv = KvClient::connect(grpc.clone())
        .await
        .map_err(|e| anyhow!("gRPC connect {grpc}: {e}"))?;
    ensure!(
        kv.health().await.unwrap_or(false),
        "server health check failed at {grpc}"
    );

    println!(
        "[setup] sizes = {:?} MiB, concurrency = {:?}, iters = {}, gid = {}",
        mibs, concs, iters, gid
    );
    println!(
        "[setup] rails: {} devices on {} endpoints",
        devices.len(),
        endpoints.len()
    );
    println!(
        "[setup] client memory bound: workers x rails x object_size \
         (concurrency capped at CS_MAX_WORKERS={})",
        max_workers
    );

    let n_rails_max = endpoints.len();
    let mut all_rows: Vec<Point> = Vec::new();

    for &mib in &mibs {
        let size = mib * 1024 * 1024;
        let mut data = vec![0u8; size];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        // Each size gets its own object so descriptors and stripe counts are independent.
        let key = format!("obj-{mib}mib");
        kv.put("multirail-conc", &key, data.clone()).await?;
        let lookup = kv
            .lookup_object("multirail-conc", &key)
            .await?
            .ok_or_else(|| anyhow!("object {key} not found right after PUT"))?;
        let desc = lookup.descriptor;
        ensure!(
            desc.is_striped && desc.stripe_count >= 2,
            "object {key} is not striped (stripe_count={})",
            desc.stripe_count
        );

        let chunk = desc.chunk_size as usize;
        let checksums: Vec<Option<String>> = (0..desc.stripe_count as usize)
            .map(|i| {
                let start = i * chunk;
                let end = ((i + 1) * chunk).min(data.len());
                Some(stripe_checksum(&data[start..end]))
            })
            .collect();
        let expected = Arc::new(data.clone());

        println!();
        println!(
            "=== object {mib} MiB: {} stripes x {} KiB ===",
            desc.stripe_count,
            desc.chunk_size / 1024
        );

        let mut rows: Vec<Point> = Vec::new();
        // Modes: always single (1 rail); plus dual only when a second rail exists.
        let rail_opts: Vec<usize> = if n_rails_max >= 2 {
            vec![1, n_rails_max]
        } else {
            vec![1]
        };
        for &conc in &concs {
            for &n_rails in &rail_opts {
                let mode = if n_rails == 1 { "single" } else { "dual" };
                match measure(
                    mib, mode, n_rails, conc, iters, &desc, &checksums, expected.clone(),
                    &endpoints, &devices, gid,
                ) {
                    Ok(p) => {
                        println!(
                            "[{mib:>4} MiB {:>6} conc={:<2}] agg={:8.1} MB/s  p50={:7.2}ms  p95={:7.2}ms  bytes={}  verify_ok={}",
                            mode, p.conc, p.agg_mbps, p.p50_ms, p.p95_ms, p.bytes, p.all_verified
                        );
                        rows.push(p);
                    }
                    Err(e) => {
                        // Keep sweeping: a single failing point should not abort
                        // the whole table, and the [diag] lines above carry the
                        // per-stripe root cause.
                        println!(
                            "[{mib:>4} MiB {:>6} conc={:<2}] FAILED: {e}",
                            mode, conc
                        );
                    }
                }
            }
        }

        println!();
        println!("--- {mib} MiB: concurrency & resource table ---");
        println!(
            "{:<8} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "mode", "conc", "agg_MB/s", "p50_ms", "p95_ms", "RSS_MiB", "reg_MiB", "infl_MiB", "verify"
        );
        for r in &rows {
            println!(
                "{:<8} {:>4} {:>10.1} {:>10.2} {:>10.2} {:>10.0} {:>10.0} {:>10.0} {:>8}",
                r.mode,
                r.conc,
                r.agg_mbps,
                r.p50_ms,
                r.p95_ms,
                r.rss_mib,
                r.reg_mib,
                r.inflight_mib,
                r.all_verified
            );
        }

        // Per-size: dual-rail speedup at conc=1, and concurrency trend per mode.
        if let (Some(s1), Some(d1)) = (
            rows.iter().find(|r| r.mode == "single" && r.conc == 1),
            rows.iter().find(|r| r.mode == "dual" && r.conc == 1),
        ) {
            println!(
                "[trend] {mib} MiB conc=1 dual/single goodput speedup: {:.2}x",
                d1.agg_mbps / s1.agg_mbps.max(1e-9)
            );
        }
        for mode in ["single", "dual"] {
            let base = rows.iter().find(|r| r.mode == mode && r.conc == 1);
            let top = rows.iter().filter(|r| r.mode == mode).max_by_key(|r| r.conc);
            if let (Some(b), Some(t)) = (base, top) {
                if t.conc > 1 {
                    println!(
                        "[trend] {mib} MiB {mode} concurrency {}->{}: {:.2}x aggregate goodput",
                        b.conc,
                        t.conc,
                        t.agg_mbps / b.agg_mbps.max(1e-9)
                    );
                }
            }
        }

        all_rows.extend(rows);

        // Free the big client-side buffers before the next (larger) size.
        drop(expected);
    }

    // Cross-size summary: per-rail goodput should be roughly size-independent
    // (this is the "extension trend across object sizes" the contest asks for).
    println!();
    println!("=== summary: dual-rail aggregate goodput vs object size (conc=1) ===");
    println!("{:>10} {:>12} {:>12} {:>10}", "size_MiB", "single_MB/s", "dual_MB/s", "speedup");
    for &mib in &mibs {
        let s = all_rows.iter().find(|r| r.mode == "single" && r.conc == 1 && r.size_mib == mib);
        let d = all_rows.iter().find(|r| r.mode == "dual" && r.conc == 1 && r.size_mib == mib);
        if let (Some(s), Some(d)) = (s, d) {
            println!(
                "{:>10} {:>12.1} {:>12.1} {:>9.2}x",
                mib,
                s.agg_mbps,
                d.agg_mbps,
                d.agg_mbps / s.agg_mbps.max(1e-9)
            );
        }
    }

    // Fair-comparison statement expected by the contest rules.
    println!();
    println!(
        "[note] Same stripe layout, same server and same client process for every row; \
         only the object size and concurrency vary. Soft-RoCE shares the host CPU, so this \
         measures scheduling/concurrency behaviour and software overhead, NOT physical NIC \
         aggregate bandwidth. Registered memory and inflight bytes are client-side accounting \
         derived from the read calls, not HCA counters."
    );

    Ok(())
}