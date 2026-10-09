//! Multi-rail benchmark / functional driver for the ContextStore client.
//!
//! Reads one striped object through N local RDMA devices in parallel and
//! reports per-rail throughput, error, and in-flight metrics, plus the
//! aggregate bandwidth and (optionally) a single-rail baseline for the
//! multi-rail speedup ratio.
//!
//! Example (two Soft-RoCE rails against a two-NIC server):
//! ```text
//! cs-multirail-bench --coordinator http://10.0.0.1:50051 \
//!   --namespace bench --object-key obj1 --put-size-mb 1024 --verify \
//!   --rails "rxe0,rxe1" --iters 5 --baseline
//! ```

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use contextstore_client_rs::multirail::{
    MultiRailClient, RailConfig, RailLimits, RailSelectPolicy,
};
use contextstore_client_rs::{KvClient, ObjectLookup};
use prost::bytes::Bytes;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Multi-rail RDMA read benchmark")]
struct Args {
    /// gRPC coordinator endpoint, e.g. http://10.0.0.1:50051
    #[arg(long)]
    coordinator: String,
    #[arg(long)]
    namespace: String,
    #[arg(long)]
    object_key: String,
    /// Comma-separated rail specs: device[:port[:gid[:weight]]]
    #[arg(long, value_delimiter = ',')]
    rails: Vec<String>,
    /// Explicit rail↔endpoint pinning, e.g. `rxe0=10.0.0.2:50054,rxe1=10.0.0.1:50053`
    /// (rail-optimized fabrics / cross-wired testbeds). Multiple endpoints per
    /// rail separated by `;`.
    #[arg(long, value_delimiter = ',')]
    pin: Vec<String>,
    /// Rewrite per-stripe RDMA endpoints round-robin across this list (single
    /// node exposing several NIC listeners; every listener serves every stripe
    /// of the node). Enables true multi-rail reads against one storage node.
    #[arg(long, value_delimiter = ',')]
    alternate_endpoints: Vec<String>,
    /// Stripe→rail policy: least-loaded | endpoint-affinity | rr
    #[arg(long, default_value = "least-loaded")]
    policy: String,
    /// RC path MTU for all rails (bytes). 4096 on jumbo-frame fabrics.
    #[arg(long, default_value = "1024")]
    qp_mtu: u16,
    /// Keep per-rail registrations cached across iterations (pinned-buffer
    /// fast path; skips ~ibv_reg_mr of the whole buffer every read).
    #[arg(long, default_value_t = false)]
    sticky: bool,
    /// Max stripes per task: splits an endpoint's stripes over several
    /// connections per rail (intra-rail concurrency). 0 = one task per
    /// (rail, endpoint).
    #[arg(long, default_value = "0")]
    task_max_stripes: usize,
    /// GRH hop limit for all rails (routed RoCE needs more than 1).
    #[arg(long, default_value = "1")]
    hop_limit: u8,
    /// Endpoint rewrite map for reachability indirection, e.g.
    /// `10.0.0.2:50053=127.0.0.1:15053`: placement endpoints on the left are
    /// dialed via the right (RDMA GIDs are exchanged in-band and unaffected).
    #[arg(long, value_delimiter = ',')]
    endpoint_map: Vec<String>,
    #[arg(long, default_value = "5")]
    iters: usize,
    /// Destination buffer size in MiB (>= object size).
    #[arg(long, default_value = "1024")]
    buf_mb: usize,
    /// Seed the object with a deterministic pattern first (needed for --verify).
    #[arg(long)]
    put_size_mb: Option<usize>,
    /// Verify every byte of every read against the deterministic pattern.
    #[arg(long, default_value_t = true)]
    verify: bool,
    /// Also run a single-rail baseline for the speedup ratio.
    #[arg(long, default_value_t = true)]
    baseline: bool,
    #[arg(long, default_value = "30")]
    io_timeout_secs: u64,
}

/// Deterministic object pattern: each 8-byte word derived from its index.
fn pattern_word(word_index: u64) -> u64 {
    word_index
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(23)
        ^ word_index
}

// chunks_exact keeps a single implementation shared with the verifier below;
// as_chunks_mut's tuple API would obscure it.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn fill_pattern(buffer: &mut [u8]) {
    for (index, chunk) in buffer.chunks_exact_mut(8).enumerate() {
        chunk.copy_from_slice(&pattern_word(index as u64).to_le_bytes());
    }
    let words = buffer.len() / 8 * 8;
    for (i, byte) in buffer[words..].iter_mut().enumerate() {
        *byte = pattern_word(words as u64 / 8 + i as u64) as u8;
    }
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn verify_pattern(buffer: &[u8], label: &str) -> Result<()> {
    for (index, chunk) in buffer.chunks_exact(8).enumerate() {
        let expected = pattern_word(index as u64).to_le_bytes();
        if chunk != expected {
            let first_bad = index * 8;
            return Err(anyhow!(
                "{label}: verification failed at byte {first_bad} (0x{:x?} != 0x{:x?})",
                &chunk[..4],
                &expected[..4]
            ));
        }
    }
    Ok(())
}

struct AlignedBuffer {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
}

// The pointer is only dereferenced while the struct is alive; the multi-rail
// read keeps it registered and quiesced within that window.
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(len: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len, 4096)?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(anyhow!("failed to allocate {len} byte buffer"));
        }
        Ok(Self { ptr, layout, len })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

fn seed_object(args: &Args, runtime: &tokio::runtime::Runtime, size_mb: usize) -> Result<()> {
    let size = size_mb * 1024 * 1024;
    let mut buffer = AlignedBuffer::new(size)?;
    fill_pattern(buffer.as_mut());
    runtime.block_on(async {
        let mut client = KvClient::connect(format_coordinator(&args.coordinator))
            .await
            .map_err(|error| anyhow!(error.to_string()))?;
        let big = Bytes::from(buffer.as_mut().to_vec());
        let chunk = 4 * 1024 * 1024;
        let mut segments = Vec::new();
        for offset in (0..size).step_by(chunk) {
            segments.push(big.slice(offset..(offset + chunk).min(size)));
        }
        client
            .put_stream_chunks(&args.namespace, &args.object_key, segments)
            .await
            .map_err(|error| anyhow!(error.to_string()))?;
        Ok::<_, anyhow::Error>(())
    })?;
    println!("[seed] wrote {size} byte object via gRPC");
    Ok(())
}

fn format_coordinator(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("http://{url}")
    }
}

fn lookup(args: &Args, runtime: &tokio::runtime::Runtime) -> Result<ObjectLookup> {
    runtime
        .block_on(async {
            let mut client = KvClient::connect(format_coordinator(&args.coordinator))
                .await
                .map_err(|error| anyhow!(error.to_string()))?;
            client
                .lookup_object(&args.namespace, &args.object_key)
                .await
                .map_err(|error| anyhow!(error.to_string()))
        })?
        .ok_or_else(|| anyhow!("object not found: {}/{}", args.namespace, args.object_key))
}

fn run_client(
    args: &Args,
    lookup: &ObjectLookup,
    rails: Vec<RailConfig>,
    label: &str,
) -> Result<(f64, usize)> {
    let policy = RailSelectPolicy::parse(&args.policy)
        .with_context(|| format!("unknown policy '{}'", args.policy))?;
    let limits = RailLimits {
        io_timeout: std::time::Duration::from_secs(args.io_timeout_secs),
        task_max_stripes: args.task_max_stripes,
        ..RailLimits::default()
    };
    let client = MultiRailClient::new(rails)?
        .with_limits(limits)
        .with_policy(policy);
    let mut buffer = AlignedBuffer::new(args.buf_mb * 1024 * 1024)?;
    let object_size = usize::try_from(lookup.descriptor.size)?;

    println!(
        "[{label}] rails={} policy={:?} object={}B stripes={} chunk={}B",
        client.rail_count(),
        policy,
        object_size,
        lookup.descriptor.stripe_count,
        lookup.descriptor.chunk_size,
    );
    for snapshot in client.rails_snapshot() {
        println!("[{label}]   {snapshot}");
    }

    let mut latencies = Vec::with_capacity(args.iters);
    let mut last_len = 0usize;
    for iteration in 0..args.iters {
        // Poison the buffer so a missing stripe cannot slip through.
        buffer.as_mut().iter_mut().for_each(|b| *b = 0xA5);
        let started = Instant::now();
        let bytes = if args.sticky {
            // SAFETY: the AlignedBuffer outlives the client and is never
            // freed or reused while reads run.
            unsafe {
                client.read_object_into_raw(
                    &lookup.descriptor,
                    &lookup
                        .placement
                        .as_ref()
                        .map(|p| p.chunks.clone())
                        .unwrap_or_default(),
                    buffer.ptr,
                    buffer.len,
                    true,
                )
            }
        } else {
            client.read_lookup_into(lookup, buffer.as_mut())
        }
        .with_context(|| format!("[{label}] iteration {iteration} failed"))?;
        latencies.push(started.elapsed());
        last_len = bytes;
        if bytes != object_size {
            return Err(anyhow!(
                "[{label}] iteration {iteration}: got {bytes} bytes, expected {object_size}"
            ));
        }
        if args.verify {
            verify_pattern(
                &buffer.as_mut()[..object_size],
                &format!("{label}#{iteration}"),
            )?;
        }
    }

    for snapshot in client.rails_snapshot() {
        println!("[{label}]   {snapshot}");
    }
    let total: f64 = latencies.iter().map(|d| d.as_secs_f64()).sum();
    let best = latencies
        .iter()
        .map(|d| d.as_secs_f64())
        .fold(f64::INFINITY, f64::min);
    let gbps = object_size as f64 / 1024f64.powi(3) / (total / latencies.len() as f64);
    println!(
        "[{label}] avg={:.3}s best={:.3}s avg_bw={:.3} GiB/s over {} iterations",
        total / latencies.len() as f64,
        best,
        gbps,
        latencies.len()
    );
    Ok((gbps, last_len))
}

fn parse_pins(specs: &[String]) -> HashMap<String, Vec<String>> {
    let mut pins = HashMap::new();
    for spec in specs {
        if let Some((device, targets)) = spec.split_once('=') {
            pins.insert(
                device.trim().to_string(),
                targets
                    .split(';')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect(),
            );
        }
    }
    pins
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.rails.is_empty() {
        return Err(anyhow!("--rails is required, e.g. --rails rxe0,rxe1"));
    }
    let runtime = tokio::runtime::Runtime::new()?;

    if let Some(size_mb) = args.put_size_mb {
        seed_object(&args, &runtime, size_mb)?;
    }
    let mut lookup = lookup(&args, &runtime)?;
    if lookup.placement.is_none() {
        return Err(anyhow!(
            "lookup returned no placement (is the object striped?)"
        ));
    }
    let original_lookup = lookup.clone();
    // Reachability indirection: rewrite placement endpoints through the map
    // (the RDMA GID exchange travels inside the control stream, so the data
    // path is unaffected by the TCP detour).
    let endpoint_map: HashMap<String, String> = args
        .endpoint_map
        .iter()
        .filter_map(|spec| spec.split_once('='))
        .map(|(from, to)| (from.trim().to_string(), to.trim().to_string()))
        .collect();
    if !endpoint_map.is_empty() {
        if let Some(placement) = lookup.placement.as_mut() {
            for chunk in placement.chunks.iter_mut() {
                if let Some(to) = endpoint_map.get(&chunk.rdma_endpoint) {
                    chunk.rdma_endpoint = to.clone();
                }
            }
        }
    }
    if !args.alternate_endpoints.is_empty() {
        // Spread the node's stripes over its listeners so several rails can
        // carry the object concurrently.
        let endpoints = &args.alternate_endpoints;
        if let Some(placement) = lookup.placement.as_mut() {
            for (index, chunk) in placement.chunks.iter_mut().enumerate() {
                chunk.rdma_endpoint = endpoints[index % endpoints.len()].clone();
            }
        }
    }

    let pins = parse_pins(&args.pin);
    let rails: Vec<RailConfig> = args
        .rails
        .iter()
        .map(|spec| RailConfig::parse(spec).ok_or_else(|| anyhow!("bad rail spec '{spec}'")))
        .map(|rail| {
            rail.map(|mut rail| {
                if let Some(endpoints) = pins.get(&rail.device) {
                    rail.endpoints = endpoints.clone();
                }
                if args.qp_mtu != 1024 {
                    rail.mtu = args.qp_mtu;
                }
                if args.hop_limit != 1 {
                    rail.hop_limit = args.hop_limit;
                }
                rail
            })
        })
        .collect::<Result<_>>()?;

    // Multi-rail run.
    let (multi_gbps, bytes) = run_client(&args, &lookup, rails.clone(), "multi")?;

    // Optional single-rail baseline for the speedup ratio (unpinned, reading
    // the original placement: one rail must reach the advertised endpoint).
    if args.baseline && rails.len() > 1 {
        let mut baseline_rail = rails[0].clone();
        baseline_rail.endpoints.clear();
        let (single_gbps, single_bytes) =
            run_client(&args, &original_lookup, vec![baseline_rail], "single")?;
        if single_bytes != bytes {
            return Err(anyhow!("single/multi rail byte counts differ"));
        }
        println!(
            "[speedup] single={:.3} GiB/s multi({})={:.3} GiB/s ratio={:.2}x",
            single_gbps,
            rails.len(),
            multi_gbps,
            multi_gbps / single_gbps
        );
    }
    Ok(())
}
