//! Rust client benchmark — compares against the Python client (0.32 / 0.61 GB/s).
//!
//! Usage: cs-bench --endpoint http://127.0.0.1:50051 --layers 80 --layer-mb 6 --concurrency 8

use clap::Parser;
use contextstore_client_rs::{pb, KvClient};
use prost::bytes::Bytes;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:50051")]
    endpoint: String,
    #[arg(long, default_value_t = 80)]
    layers: usize,
    #[arg(long, default_value_t = 6)]
    layer_mb: usize,
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
    /// Simulate a realistic vLLM combined write: pack all layers into a single large
    /// value and PUT once (triggers server-side striping).
    #[arg(long, default_value_t = false)]
    combined: bool,
    /// In combined mode, write this many distinct prefixes consecutively (simulates
    /// multiple requests).
    #[arg(long, default_value_t = 8)]
    combined_count: usize,
    /// In combined mode, use streaming (split into small messages) to bypass the
    /// 480MB single-message decode wall.
    #[arg(long, default_value_t = false)]
    stream: bool,
    /// Streaming chunk size (MB).
    #[arg(long, default_value_t = 2)]
    stream_chunk_mb: usize,
    /// Prefix base used in combined mode (lets multiple clients run in parallel
    /// without colliding).
    #[arg(long, default_value = "comb")]
    prefix_base: String,
    /// Only run PUT (skip GET).
    #[arg(long, default_value_t = false)]
    only_put: bool,
    /// Only run GET (assumes data is already written; skips PUT).
    #[arg(long, default_value_t = false)]
    only_get: bool,
    /// Zero-copy passthrough: use put_stream_chunks / get_stream_chunks (Vec<Bytes>),
    /// without concatenating into a 480MB Vec<u8> on the client. Expected to break
    /// the 0.4 GB/s single-connection ceiling.
    #[arg(long, default_value_t = false)]
    bytes_pass: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let total_mb = args.layers * args.layer_mb;
    let payload_len = args.layer_mb * 1024 * 1024;
    // Pre-generate one deterministic payload (avoids a rand dependency); each layer
    // reuses the same block.
    let payload: Vec<u8> = (0..payload_len).map(|i| (i % 251) as u8).collect();

    let mut c = KvClient::connect(args.endpoint.clone()).await?;
    assert!(c.health().await?, "server not serving");

    // ===== Simulate a realistic vLLM combined write (single large value, triggers striping) =====
    if args.combined {
        let combined_mb = args.layers * args.layer_mb;
        let combined_len = combined_mb * 1024 * 1024;
        let big: Vec<u8> = (0..combined_len).map(|i| (i % 251) as u8).collect();
        let chunk_bytes = args.stream_chunk_mb * 1024 * 1024;
        println!(
            "== Rust combined bench: {} prefixes x {} MB (single value, simulates __combined__ striping), concurrency={}, mode={} ==",
            args.combined_count,
            combined_mb,
            args.concurrency,
            if args.stream {
                format!("stream(chunk={}MB)", args.stream_chunk_mb)
            } else {
                "single-message".to_string()
            }
        );

        // PUT: write multiple large combined values concurrently.
        let prefix_base = args.prefix_base.clone();
        let do_put = !args.only_get;
        let do_get = !args.only_put;
        let bytes_pass = args.bytes_pass;
        // In bytes_pass mode, split `big` once into N Bytes segments (Arc-refcounted);
        // each spawn shares the same underlying buffer.
        let big_bytes: Bytes = Bytes::from(big.clone());
        if do_put {
            let t0 = Instant::now();
            let mut handles = Vec::new();
            let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
            for p in 0..args.combined_count {
                // Each task opens its own connection — avoids sharing a single tonic gRPC
                // channel (which caps at HTTP/2 single-connection throughput). Measured
                // (2026-06-11): c.clone() shared a single connection and only reached
                // 0.85 GB/s under 8-way concurrency; independent connects break through.
                let endpoint = args.endpoint.clone();
                let data = big.clone();
                let big_b = big_bytes.clone();
                let sem = sem.clone();
                let stream = args.stream;
                let pb_name = prefix_base.clone();
                let bp = bytes_pass;
                let csz = chunk_bytes;
                handles.push(tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    let mut cc = KvClient::connect(endpoint)
                        .await
                        .map_err(|e| tonic::Status::unavailable(format!("connect: {}", e)))?;
                    if bp && stream {
                        // Zero-copy passthrough: split big_b into N Bytes segments
                        // (Arc bump, no memcpy).
                        let total = big_b.len();
                        let n = total.div_ceil(csz);
                        let mut segs: Vec<Bytes> = Vec::with_capacity(n);
                        for i in 0..n {
                            let s = i * csz;
                            let e = (s + csz).min(total);
                            segs.push(big_b.slice(s..e));
                        }
                        cc.put_stream_chunks(
                            "rust-bench",
                            &format!("{}{}/__combined__", pb_name, p),
                            segs,
                        )
                        .await
                    } else if stream {
                        cc.put_stream(
                            "rust-bench",
                            &format!("{}{}/__combined__", pb_name, p),
                            data,
                            csz,
                        )
                        .await
                    } else {
                        cc.put(
                            "rust-bench",
                            &format!("{}{}/__combined__", pb_name, p),
                            data,
                        )
                        .await
                    }
                }));
            }
            for h in handles {
                h.await??;
            }
            let dt = t0.elapsed().as_secs_f64();
            let tot = (args.combined_count * combined_mb) as f64;
            println!(
                "[PUT combined    ] {} x {}MB = {:.0}MB  {:7.1}ms  {:.2} GB/s",
                args.combined_count,
                combined_mb,
                tot,
                dt * 1000.0,
                tot / dt / 1024.0
            );
        }

        // GET: concurrent reads (triggers parallel striped reads).
        if do_get {
            let t0 = Instant::now();
            let mut handles = Vec::new();
            let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
            for p in 0..args.combined_count {
                let mut cc = c.clone();
                let sem = sem.clone();
                let stream = args.stream;
                let pb_name = prefix_base.clone();
                let bp = bytes_pass;
                handles.push(tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    if bp && stream {
                        // Zero-copy passthrough: returns Vec<Bytes>; client only totals
                        // the bytes without concatenating.
                        match cc
                            .get_stream_chunks(
                                "rust-bench",
                                &format!("{}{}/__combined__", pb_name, p),
                            )
                            .await
                        {
                            Ok(Some(segs)) => Ok::<Option<usize>, tonic::Status>(Some(
                                segs.iter().map(|s| s.len()).sum(),
                            )),
                            Ok(None) => Ok(None),
                            Err(e) => Err(e),
                        }
                    } else if stream {
                        cc.get_stream("rust-bench", &format!("{}{}/__combined__", pb_name, p))
                            .await
                            .map(|opt| opt.map(|v| v.len()))
                    } else {
                        cc.get("rust-bench", &format!("{}{}/__combined__", pb_name, p))
                            .await
                            .map(|opt| opt.map(|v| v.len()))
                    }
                }));
            }
            let mut got = 0.0;
            for h in handles {
                if let Some(n) = h.await?? {
                    got += n as f64 / 1024.0 / 1024.0;
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "[GET combined    ] {:.0}MB  {:7.1}ms  {:.2} GB/s",
                got,
                dt * 1000.0,
                got / dt / 1024.0
            );
        }
        return Ok(());
    }

    println!(
        "== Rust client bench: {} layers x {} MB = {} MB, concurrency={} ==",
        args.layers, args.layer_mb, total_mb, args.concurrency
    );

    // ---------- PUT: serial single-request ----------
    let t0 = Instant::now();
    for i in 0..args.layers {
        c.put("rust-bench", &format!("serial/L{}", i), payload.clone())
            .await?;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[PUT serial      ] {:7.1}ms  {:.2} GB/s",
        dt * 1000.0,
        total_mb as f64 / dt / 1024.0
    );

    // ---------- PUT: concurrent single-request ----------
    let t0 = Instant::now();
    let mut handles = Vec::new();
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    for i in 0..args.layers {
        let mut cc = c.clone();
        let p = payload.clone();
        let sem = sem.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            cc.put("rust-bench", &format!("conc/L{}", i), p).await
        }));
    }
    for h in handles {
        h.await??;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[PUT concurrent  ] {:7.1}ms  {:.2} GB/s  (concurrency={})",
        dt * 1000.0,
        total_mb as f64 / dt / 1024.0,
        args.concurrency
    );

    // ---------- PUT: batch (single RPC) ----------
    let items: Vec<(pb::ObjectKey, Vec<u8>)> = (0..args.layers)
        .map(|i| {
            (
                KvClient::make_key("rust-bench", &format!("batch/L{}", i)),
                payload.clone(),
            )
        })
        .collect();
    let t0 = Instant::now();
    let ok = c.put_batch(items).await?;
    let dt = t0.elapsed().as_secs_f64();
    assert!(ok.iter().all(|b| *b));
    println!(
        "[PUT batch(1 RPC)] {:7.1}ms  {:.2} GB/s",
        dt * 1000.0,
        total_mb as f64 / dt / 1024.0
    );

    // ---------- GET: concurrent single-request ----------
    let t0 = Instant::now();
    let mut handles = Vec::new();
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    for i in 0..args.layers {
        let mut cc = c.clone();
        let sem = sem.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            cc.get("rust-bench", &format!("conc/L{}", i)).await
        }));
    }
    let mut got_mb = 0.0;
    for h in handles {
        if let Some(d) = h.await?? {
            got_mb += d.len() as f64 / 1024.0 / 1024.0;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[GET concurrent  ] {:7.1}ms  {:.2} GB/s  ({:.0} MB)",
        dt * 1000.0,
        got_mb / dt / 1024.0,
        got_mb
    );

    // ---------- GET: batch ----------
    let keys: Vec<pb::ObjectKey> = (0..args.layers)
        .map(|i| KvClient::make_key("rust-bench", &format!("batch/L{}", i)))
        .collect();
    let t0 = Instant::now();
    let res = c.get_batch(keys).await?;
    let dt = t0.elapsed().as_secs_f64();
    let gmb: f64 = res
        .iter()
        .filter_map(|r| r.as_ref())
        .map(|d| d.len() as f64 / 1024.0 / 1024.0)
        .sum();
    println!(
        "[GET batch(1 RPC)] {:7.1}ms  {:.2} GB/s  ({:.0} MB)",
        dt * 1000.0,
        gmb / dt / 1024.0,
        gmb
    );

    Ok(())
}
