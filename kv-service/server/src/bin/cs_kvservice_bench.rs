//! cs-kvservice-bench: six KVService performance and correctness benchmarks.
//!
//! Fixed coverage of six cases:
//! 1. MemoryTier single put + L1 get
//! 2. MemoryTier chunks put + L1 get_chunks
//! 3. MemoryTier single -> chunks overwrite consistency
//! 4. MemoryTier chunks -> single overwrite consistency
//! 5. StorageTier disk write throughput
//! 6. StorageTier disk read throughput
//!
//! By default it only prints results; passing threshold arguments turns it into a performance gate on a fixed machine.

use anyhow::{bail, Result};
use clap::Parser;
use contextstore_server::config::Config;
use contextstore_server::memory_tier::MemoryTier;
use contextstore_server::metadata::BlockMeta;
use contextstore_server::router::{ObjectKey, ShardRouter};
use contextstore_server::storage_tier::StorageTier;
use prost::bytes::Bytes;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
struct Args {
    /// Temporary benchmark data directory; when unset, a unique directory under /tmp is used
    #[arg(long)]
    root: Option<PathBuf>,
    /// Server config used for disk read/write tests; when unset a temporary single-disk config is used
    #[arg(long)]
    storage_config: Option<PathBuf>,
    /// Keep the temporary data directory after the run for troubleshooting
    #[arg(long, default_value_t = false)]
    keep_data: bool,
    /// Try to drop the page cache before the read test; requires Linux root, only warns on failure
    #[arg(long, default_value_t = false)]
    drop_caches: bool,
    /// L1 capacity (MB)
    #[arg(long, default_value_t = 512)]
    memory_mb: usize,
    /// Regular object size (bytes)
    #[arg(long, default_value_t = 4096)]
    single_bytes: usize,
    /// Total size of the chunks object (KB)
    #[arg(long, default_value_t = 64)]
    chunk_total_kb: usize,
    /// Number of segments in the chunks object
    #[arg(long, default_value_t = 8)]
    chunk_count: usize,
    /// Iteration count for the MemoryTier single case
    #[arg(long, default_value_t = 5000)]
    single_iters: usize,
    /// Iteration count for the MemoryTier chunks case
    #[arg(long, default_value_t = 2000)]
    chunk_iters: usize,
    /// Iteration count for the MemoryTier cross-interface overwrite cases
    #[arg(long, default_value_t = 2000)]
    cross_iters: usize,
    /// Disk single-object size (MB)
    #[arg(long, default_value_t = 128)]
    storage_size_mb: usize,
    /// Number of disk objects
    #[arg(long, default_value_t = 4)]
    storage_iters: usize,
    /// Disk concurrency
    #[arg(long, default_value_t = 4)]
    storage_concurrency: usize,
    /// Per-segment Bytes size for disk writes (MB)
    #[arg(long, default_value_t = 2)]
    storage_chunk_mb: usize,
    /// Maximum allowed us/op for single put+get; 0 disables the check
    #[arg(long, default_value_t = 0.0)]
    max_single_us: f64,
    /// Maximum allowed us/op for chunks put+get; 0 disables the check
    #[arg(long, default_value_t = 0.0)]
    max_chunks_us: f64,
    /// Maximum allowed us/op for the single -> chunks overwrite; 0 disables the check
    #[arg(long, default_value_t = 0.0)]
    max_cross_single_to_chunks_us: f64,
    /// Maximum allowed us/op for the chunks -> single overwrite; 0 disables the check
    #[arg(long, default_value_t = 0.0)]
    max_cross_chunks_to_single_us: f64,
    /// Minimum allowed disk write throughput (GiB/s); 0 disables the check
    #[arg(long, default_value_t = 0.0)]
    min_storage_write_gib: f64,
    /// Minimum allowed disk read throughput (GiB/s); 0 disables the check
    #[arg(long, default_value_t = 0.0)]
    min_storage_read_gib: f64,
}

#[derive(Clone, Copy)]
enum Threshold {
    MaxUs(f64),
    MinGib(f64),
}

struct BenchResult {
    name: &'static str,
    iters: usize,
    elapsed: Duration,
    bytes_per_op: usize,
    threshold: Threshold,
}

impl BenchResult {
    fn us_per_op(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1_000_000.0 / self.iters as f64
    }

    fn gib_per_sec(&self) -> f64 {
        let total_bytes = self.bytes_per_op as f64 * self.iters as f64;
        total_bytes / self.elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0
    }

    fn print(&self) {
        println!(
            "{:<30} {:>8} ops  {:>9.2} us/op  {:>7.2} GiB/s  {:>8.2} ms",
            self.name,
            self.iters,
            self.us_per_op(),
            self.gib_per_sec(),
            self.elapsed.as_secs_f64() * 1000.0,
        );
    }

    fn check(&self) -> Result<()> {
        match self.threshold {
            Threshold::MaxUs(max_us) if max_us > 0.0 && self.us_per_op() > max_us => {
                bail!(
                    "{} exceeded threshold: {:.2} us/op > {:.2} us/op",
                    self.name,
                    self.us_per_op(),
                    max_us
                );
            }
            Threshold::MinGib(min_gib) if min_gib > 0.0 && self.gib_per_sec() < min_gib => {
                bail!(
                    "{} below threshold: {:.2} GiB/s < {:.2} GiB/s",
                    self.name,
                    self.gib_per_sec(),
                    min_gib
                );
            }
            _ => {}
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let root = args.root.clone().unwrap_or_else(unique_root);
    if root.exists() {
        std::fs::remove_dir_all(&root)?;
    }
    std::fs::create_dir_all(root.join("memory").join("nvme0"))?;
    std::fs::create_dir_all(root.join("storage").join("nvme0"))?;

    let memory_tier = make_memory_tier(&args, &root.join("memory"))?;
    let single = Bytes::from(vec![7u8; args.single_bytes]);
    let chunks = make_chunks(args.chunk_total_kb * 1024, args.chunk_count)?;
    warmup_memory_tier(&memory_tier, &single, &chunks, 128)?;

    let storage = make_storage_tier(&args, &root.join("storage"))?;
    let storage_size = args.storage_size_mb * 1024 * 1024;
    let storage_segments = make_storage_segments(storage_size, args.storage_chunk_mb * 1024 * 1024);

    println!(
        "[kvservice-bench] root={} memory={}MB single={}B chunks={}B/{}segs storage={}MBx{} conc={}",
        root.display(),
        args.memory_mb,
        single.len(),
        chunks_len(&chunks),
        chunks.len(),
        args.storage_size_mb,
        args.storage_iters,
        args.storage_concurrency,
    );

    let mut results = Vec::with_capacity(6);
    results.push(bench_single_put_get(
        &memory_tier,
        &single,
        args.single_iters,
        args.max_single_us,
    )?);
    results.push(bench_chunks_put_get(
        &memory_tier,
        &chunks,
        args.chunk_iters,
        args.max_chunks_us,
    )?);
    results.push(bench_cross_single_to_chunks(
        &memory_tier,
        &single,
        &chunks,
        args.cross_iters,
        args.max_cross_single_to_chunks_us,
    )?);
    results.push(bench_cross_chunks_to_single(
        &memory_tier,
        &single,
        &chunks,
        args.cross_iters,
        args.max_cross_chunks_to_single_us,
    )?);
    results.push(bench_storage_write(
        storage.clone(),
        &storage_segments,
        args.storage_iters,
        args.storage_concurrency,
        args.min_storage_write_gib,
    )?);
    maybe_drop_caches(args.drop_caches);
    results.push(bench_storage_read(
        storage,
        storage_size,
        args.storage_iters,
        args.storage_concurrency,
        args.min_storage_read_gib,
    )?);

    println!();
    for result in &results {
        result.print();
    }
    for result in &results {
        result.check()?;
    }

    let (hits, misses, evictions, l1_bytes) = memory_tier.stats();
    println!(
        "\n[stats] memory_hits={} memory_misses={} memory_evictions={} memory_l1_size={} bytes",
        hits, misses, evictions, l1_bytes
    );

    if !args.keep_data && args.storage_config.is_none() {
        std::fs::remove_dir_all(&root)?;
    } else if !args.keep_data {
        std::fs::remove_dir_all(root.join("memory")).ok();
        std::fs::remove_dir_all(root.join("storage")).ok();
        std::fs::remove_dir_all(&root).ok();
    }
    println!("[kvservice-bench] PASS");
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.single_bytes == 0 {
        bail!("--single-bytes must be > 0");
    }
    if args.chunk_total_kb == 0 {
        bail!("--chunk-total-kb must be > 0");
    }
    if args.chunk_count == 0 {
        bail!("--chunk-count must be > 0");
    }
    if args.chunk_total_kb * 1024 < args.chunk_count {
        bail!("chunk total size must be >= chunk count");
    }
    if args.single_iters == 0
        || args.chunk_iters == 0
        || args.cross_iters == 0
        || args.storage_iters == 0
        || args.storage_concurrency == 0
    {
        bail!("iteration and concurrency arguments must be > 0");
    }
    if args.storage_size_mb == 0 || args.storage_chunk_mb == 0 {
        bail!("storage size arguments must be > 0");
    }
    Ok(())
}

fn unique_root() -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "contextstore-kvservice-bench-{}-{}",
        std::process::id(),
        now
    ))
}

fn make_memory_tier(args: &Args, root: &Path) -> Result<MemoryTier> {
    let mut cfg = Config::default();
    cfg.storage.devices = vec![root.join("nvme0")];
    cfg.storage.striping_threshold = 0;
    cfg.metadata.rocksdb_path = root.join("meta");
    cfg.memory_tier.capacity_mb = args.memory_mb;
    let router = Arc::new(ShardRouter::new(&cfg)?);
    let storage = Arc::new(StorageTier::new(&cfg, router)?);
    Ok(MemoryTier::new(&cfg, storage))
}

fn make_storage_tier(args: &Args, root: &Path) -> Result<Arc<StorageTier>> {
    let mut cfg = if let Some(path) = &args.storage_config {
        Config::from_file(path)?
    } else {
        let mut cfg = Config::default();
        cfg.storage.devices = vec![root.join("nvme0")];
        cfg.storage.striping_threshold = 256 * 1024 * 1024;
        cfg.storage.striping_chunk_size = 64 * 1024 * 1024;
        cfg.metadata.rocksdb_path = root.join("meta");
        cfg
    };
    if cfg.storage.devices.len() == 1 && args.storage_size_mb < 256 {
        cfg.storage.striping_threshold = 0;
    }
    let router = Arc::new(ShardRouter::new(&cfg)?);
    Ok(Arc::new(StorageTier::new(&cfg, router)?))
}

fn make_key(range: &str, idx: usize) -> ObjectKey {
    ObjectKey {
        namespace: "kvservice-bench".to_string(),
        object_key: format!("{}/object_{:08}", range, idx),
    }
}

fn make_meta(size: usize) -> BlockMeta {
    BlockMeta {
        device_id: 0,
        file_path: String::new(),
        size: size as u64,
        object_handle: String::new(),
        object_generation: 1,
        content_etag: String::new(),
        layout_version: 1,
        created_at: 0,
        last_accessed_at: 0,
        ttl_seconds: 0,
        num_tokens: 128,
        num_layers: 1,
        dtype: "uint8".to_string(),
        compressed: false,
        striping: None,
    }
}

fn make_chunks(total: usize, count: usize) -> Result<Vec<Bytes>> {
    let payload = Bytes::from((0..total).map(|i| (i % 251) as u8).collect::<Vec<_>>());
    let base = total / count;
    let mut offset = 0usize;
    let mut chunks = Vec::with_capacity(count);
    for i in 0..count {
        let len = if i + 1 == count { total - offset } else { base };
        chunks.push(payload.slice(offset..offset + len));
        offset += len;
    }
    Ok(chunks)
}

fn make_storage_segments(size: usize, chunk: usize) -> Vec<Bytes> {
    let payload = Bytes::from((0..size).map(|i| (i % 251) as u8).collect::<Vec<_>>());
    let mut segments = Vec::with_capacity(size.div_ceil(chunk));
    let mut offset = 0usize;
    while offset < size {
        let end = (offset + chunk).min(size);
        segments.push(payload.slice(offset..end));
        offset = end;
    }
    segments
}

fn chunks_len(chunks: &[Bytes]) -> usize {
    chunks.iter().map(|chunk| chunk.len()).sum()
}

fn warmup_memory_tier(
    tier: &MemoryTier,
    single: &Bytes,
    chunks: &[Bytes],
    iters: usize,
) -> Result<()> {
    let single_meta = make_meta(single.len());
    let chunks_meta = make_meta(chunks_len(chunks));
    for i in 0..iters {
        let single_key = make_key("warmup-single", i);
        tier.put(&single_key, single.clone(), single_meta.clone())?;
        let _ = tier.get(&single_key)?;

        let chunks_key = make_key("warmup-chunks", i);
        tier.put_chunks(&chunks_key, chunks.to_vec(), chunks_meta.clone())?;
        let _ = tier.get_chunks(&chunks_key)?;
    }
    Ok(())
}

fn bench_single_put_get(
    tier: &MemoryTier,
    single: &Bytes,
    iters: usize,
    threshold_us: f64,
) -> Result<BenchResult> {
    let meta = make_meta(single.len());
    let start = Instant::now();
    for i in 0..iters {
        let key = make_key("single-put-get", i);
        tier.put(&key, single.clone(), meta.clone())?;
        let (data, _) = tier
            .get(&key)?
            .ok_or_else(|| anyhow::anyhow!("single get returned None"))?;
        if data.len() != single.len() {
            bail!(
                "single get size mismatch: {} != {}",
                data.len(),
                single.len()
            );
        }
    }
    Ok(BenchResult {
        name: "01 single put+L1 get",
        iters,
        elapsed: start.elapsed(),
        bytes_per_op: single.len(),
        threshold: Threshold::MaxUs(threshold_us),
    })
}

fn bench_chunks_put_get(
    tier: &MemoryTier,
    chunks: &[Bytes],
    iters: usize,
    threshold_us: f64,
) -> Result<BenchResult> {
    let total = chunks_len(chunks);
    let meta = make_meta(total);
    let start = Instant::now();
    for i in 0..iters {
        let key = make_key("chunks-put-get", i);
        tier.put_chunks(&key, chunks.to_vec(), meta.clone())?;
        let (segments, _) = tier
            .get_chunks(&key)?
            .ok_or_else(|| anyhow::anyhow!("chunks get returned None"))?;
        let got = chunks_len(&segments);
        if got != total {
            bail!("chunks get size mismatch: {} != {}", got, total);
        }
    }
    Ok(BenchResult {
        name: "02 chunks put+L1 get",
        iters,
        elapsed: start.elapsed(),
        bytes_per_op: total,
        threshold: Threshold::MaxUs(threshold_us),
    })
}

fn bench_cross_single_to_chunks(
    tier: &MemoryTier,
    single: &Bytes,
    chunks: &[Bytes],
    iters: usize,
    threshold_us: f64,
) -> Result<BenchResult> {
    let single_meta = make_meta(single.len());
    let chunks_total = chunks_len(chunks);
    let chunks_meta = make_meta(chunks_total);
    let start = Instant::now();
    for i in 0..iters {
        let key = make_key("cross-single-to-chunks", i);
        tier.put(&key, single.clone(), single_meta.clone())?;
        tier.put_chunks(&key, chunks.to_vec(), chunks_meta.clone())?;

        let (data, _) = tier
            .get(&key)?
            .ok_or_else(|| anyhow::anyhow!("cross get returned None"))?;
        if data.len() != chunks_total {
            bail!(
                "single->chunks stale read: get returned {} bytes, expected {} bytes",
                data.len(),
                chunks_total
            );
        }
    }
    Ok(BenchResult {
        name: "03 single->chunks",
        iters,
        elapsed: start.elapsed(),
        bytes_per_op: chunks_total,
        threshold: Threshold::MaxUs(threshold_us),
    })
}

fn bench_cross_chunks_to_single(
    tier: &MemoryTier,
    single: &Bytes,
    chunks: &[Bytes],
    iters: usize,
    threshold_us: f64,
) -> Result<BenchResult> {
    let single_meta = make_meta(single.len());
    let chunks_total = chunks_len(chunks);
    let chunks_meta = make_meta(chunks_total);
    let start = Instant::now();
    for i in 0..iters {
        let key = make_key("cross-chunks-to-single", i);
        tier.put_chunks(&key, chunks.to_vec(), chunks_meta.clone())?;
        tier.put(&key, single.clone(), single_meta.clone())?;

        let (segments, _) = tier
            .get_chunks(&key)?
            .ok_or_else(|| anyhow::anyhow!("cross get_chunks returned None"))?;
        let got = chunks_len(&segments);
        if got != single.len() {
            bail!(
                "chunks->single stale read: get_chunks returned {} bytes, expected {} bytes",
                got,
                single.len()
            );
        }
    }
    Ok(BenchResult {
        name: "04 chunks->single",
        iters,
        elapsed: start.elapsed(),
        bytes_per_op: single.len(),
        threshold: Threshold::MaxUs(threshold_us),
    })
}

fn bench_storage_write(
    storage: Arc<StorageTier>,
    segments: &[Bytes],
    iters: usize,
    concurrency: usize,
    min_gib: f64,
) -> Result<BenchResult> {
    let total = chunks_len(segments);
    let meta = make_meta(total);
    let start = Instant::now();
    for batch_start in (0..iters).step_by(concurrency) {
        let batch_end = (batch_start + concurrency).min(iters);
        let mut handles = Vec::with_capacity(batch_end - batch_start);
        for i in batch_start..batch_end {
            let storage = storage.clone();
            let segments = segments.to_vec();
            let meta = meta.clone();
            handles.push(std::thread::spawn(move || -> Result<()> {
                let key = make_key("storage-write", i);
                storage.put_chunks(&key, segments, meta)?;
                Ok(())
            }));
        }
        join_all(handles)?;
    }
    Ok(BenchResult {
        name: "05 storage write",
        iters,
        elapsed: start.elapsed(),
        bytes_per_op: total,
        threshold: Threshold::MinGib(min_gib),
    })
}

fn bench_storage_read(
    storage: Arc<StorageTier>,
    size: usize,
    iters: usize,
    concurrency: usize,
    min_gib: f64,
) -> Result<BenchResult> {
    let start = Instant::now();
    for batch_start in (0..iters).step_by(concurrency) {
        let batch_end = (batch_start + concurrency).min(iters);
        let mut handles = Vec::with_capacity(batch_end - batch_start);
        for i in batch_start..batch_end {
            let storage = storage.clone();
            handles.push(std::thread::spawn(move || -> Result<()> {
                let key = make_key("storage-write", i);
                let (segments, _) = storage
                    .get_chunks(&key)?
                    .ok_or_else(|| anyhow::anyhow!("storage read None"))?;
                let got = chunks_len(&segments);
                if got != size {
                    bail!("storage read size mismatch: {} != {}", got, size);
                }
                Ok(())
            }));
        }
        join_all(handles)?;
    }
    Ok(BenchResult {
        name: "06 storage read",
        iters,
        elapsed: start.elapsed(),
        bytes_per_op: size,
        threshold: Threshold::MinGib(min_gib),
    })
}

fn join_all(handles: Vec<std::thread::JoinHandle<Result<()>>>) -> Result<()> {
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("benchmark worker panicked"))??;
    }
    Ok(())
}

fn maybe_drop_caches(enabled: bool) {
    if !enabled {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::process::Command::new("sh")
            .args(["-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
            .status();
        if !matches!(status, Ok(s) if s.success()) {
            eprintln!("[warn] drop page cache failed; storage read may hit page cache");
        }
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("[warn] drop page cache is only supported on Linux");
}
