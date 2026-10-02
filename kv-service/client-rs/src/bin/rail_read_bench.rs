//! Compare one Worker reading a striped object through one or more RDMA rails.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use contextstore_client_rs::rail_read::{LocalRailPath, RailLimits, RailReader, RailRoute};
use contextstore_client_rs::rdma::RdmaClientConfig;
use contextstore_client_rs::KvClient;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "Read one ContextStore object over independently configured RDMA rails")]
struct Args {
    #[arg(long)]
    coordinator: String,
    #[arg(long)]
    namespace: String,
    #[arg(long)]
    object_key: String,
    /// Repeat: id,local_device,advertised_endpoint,listener,port,gid_index[,weight].
    #[arg(long = "rail", conflicts_with = "local_rails")]
    rails: Vec<String>,
    /// Repeat: id,fabric_id,local_device,port,gid_index[,weight].
    /// Remote listener is discovered from the object's PlacementDescriptor.
    #[arg(long = "local-rail", conflicts_with = "rails")]
    local_rails: Vec<String>,
    /// State which real Verbs environment produced these measurements.
    #[arg(long, value_parser = ["physical", "soft-roce"])]
    environment: String,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    /// Concurrent object reads within this Worker, sharing one RailReader.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
}

fn parse_route(spec: &str) -> Result<RailRoute> {
    let fields: Vec<_> = spec.split(',').map(str::trim).collect();
    if !(6..=7).contains(&fields.len()) || fields[..4].iter().any(|field| field.is_empty()) {
        return Err(anyhow!(
            "rail spec must be id,device,advertised_endpoint,listener,port,gid_index[,weight]"
        ));
    }
    let port = fields[4].parse::<u8>().context("rail port is not a u8")?;
    let gid = fields[5]
        .parse::<u8>()
        .context("rail GID index is not a u8")?;
    let weight = fields
        .get(6)
        .map(|value| value.parse::<u32>().context("rail weight is not a u32"))
        .transpose()?
        .unwrap_or(1);
    if weight == 0 {
        return Err(anyhow!("rail weight must be positive"));
    }
    Ok(RailRoute::new(
        fields[0],
        fields[2],
        RdmaClientConfig::new(fields[3], fields[1])
            .with_port(port)
            .with_gid_index(gid),
    )
    .with_weight(weight))
}

fn parse_local_path(spec: &str) -> Result<LocalRailPath> {
    let fields: Vec<_> = spec.split(',').map(str::trim).collect();
    if !(5..=6).contains(&fields.len()) || fields[..3].iter().any(|field| field.is_empty()) {
        return Err(anyhow!(
            "local rail spec must be id,fabric_id,device,port,gid_index[,weight]"
        ));
    }
    let port = fields[3]
        .parse::<u8>()
        .context("local rail port is not a u8")?;
    if port == 0 {
        return Err(anyhow!("local rail port must be positive"));
    }
    let gid = fields[4]
        .parse::<u8>()
        .context("local rail GID index is not a u8")?;
    let weight = fields
        .get(5)
        .map(|value| {
            value
                .parse::<u32>()
                .context("local rail weight is not a u32")
        })
        .transpose()?
        .unwrap_or(1);
    if weight == 0 {
        return Err(anyhow!("local rail weight must be positive"));
    }
    Ok(LocalRailPath::new(fields[0], fields[1], fields[2])
        .with_port(port)
        .with_gid_index(gid)
        .with_weight(weight))
}

#[cfg(target_os = "linux")]
fn process_usage() -> (u64, u64, u64) {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(status, 0, "getrusage failed");
    let micros = |time: libc::timeval| time.tv_sec as u64 * 1_000_000 + time.tv_usec as u64;
    (
        micros(usage.ru_utime),
        micros(usage.ru_stime),
        usage.ru_maxrss as u64,
    )
}

#[cfg(not(target_os = "linux"))]
fn process_usage() -> (u64, u64, u64) {
    (0, 0, 0)
}

async fn run_concurrent(
    args: &Args,
    client: &KvClient,
    reader: Arc<RailReader>,
    size: usize,
    warmup_bytes: &[u64],
) -> Result<()> {
    let mut request_times = Vec::<Duration>::new();
    let mut batch_times = Vec::<Duration>::new();
    let mut cpu_user_us = 0u64;
    let mut cpu_system_us = 0u64;
    let mut expected_hash = None;
    for batch in 0..args.iterations {
        let destinations = (0..args.concurrency)
            .map(|_| vec![0xA5; size])
            .collect::<Vec<_>>();
        let before_cpu = process_usage();
        let batch_started = Instant::now();
        let mut handles = Vec::with_capacity(args.concurrency);
        for (worker, mut destination) in destinations.into_iter().enumerate() {
            let mut worker_client = client.clone();
            let worker_reader = Arc::clone(&reader);
            let namespace = args.namespace.clone();
            let object_key = args.object_key.clone();
            handles.push(tokio::spawn(async move {
                let started = Instant::now();
                let result = worker_client
                    .read_multi_rail_into(
                        worker_reader,
                        &namespace,
                        &object_key,
                        &mut destination,
                        None,
                    )
                    .await;
                (worker, result, destination, started.elapsed())
            }));
        }
        let mut first_error = None;
        for handle in handles {
            let (worker, result, destination, elapsed) = handle.await?;
            match result {
                Ok(Some(bytes)) if bytes == size => {
                    let checksum = twox_hash::xxh3::hash64(&destination);
                    if expected_hash.is_some_and(|expected| expected != checksum) {
                        first_error.get_or_insert_with(|| anyhow!("object hash changed"));
                    }
                    expected_hash.get_or_insert(checksum);
                    println!(
                        "sample,environment={},rails={},concurrency={},batch={},worker={},bytes={},latency_us={},xxh3={checksum:016x}",
                        args.environment,
                        reader.snapshots().len(),
                        args.concurrency,
                        batch + 1,
                        worker,
                        bytes,
                        elapsed.as_micros()
                    );
                    request_times.push(elapsed);
                }
                Ok(Some(bytes)) => {
                    first_error.get_or_insert_with(|| anyhow!("short read: {bytes} of {size}"));
                }
                Ok(None) => {
                    first_error.get_or_insert_with(|| anyhow!("object disappeared"));
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        let wall = batch_started.elapsed();
        let after_cpu = process_usage();
        cpu_user_us += after_cpu.0 - before_cpu.0;
        cpu_system_us += after_cpu.1 - before_cpu.1;
        if let Some(error) = first_error {
            return Err(error);
        }
        println!(
            "batch,environment={},rails={},concurrency={},iteration={},wall_us={}",
            args.environment,
            reader.snapshots().len(),
            args.concurrency,
            batch + 1,
            wall.as_micros()
        );
        batch_times.push(wall);
    }
    request_times.sort_unstable();
    batch_times.sort_unstable();
    let total_wall_seconds: f64 = batch_times.iter().map(Duration::as_secs_f64).sum();
    let total_bytes = size
        .checked_mul(args.concurrency)
        .and_then(|bytes| bytes.checked_mul(args.iterations))
        .ok_or_else(|| anyhow!("benchmark byte count overflow"))?;
    let aggregate_gib_per_s = total_bytes as f64 / total_wall_seconds / 1024f64.powi(3);
    let rail_bytes: Vec<_> = reader
        .snapshots()
        .iter()
        .zip(warmup_bytes)
        .map(|(snapshot, warmup)| snapshot.bytes - warmup)
        .collect();
    println!(
        "summary_concurrent,environment={},rails={},concurrency={},bytes_per_read={},batches={},median_request_us={},median_batch_us={},aggregate_gib_per_s={aggregate_gib_per_s:.3},cpu_user_us={},cpu_system_us={},peak_rss_kb={},rail_bytes={rail_bytes:?}",
        args.environment,
        rail_bytes.len(),
        args.concurrency,
        size,
        args.iterations,
        request_times[request_times.len() / 2].as_micros(),
        batch_times[batch_times.len() / 2].as_micros(),
        cpu_user_us,
        cpu_system_us,
        process_usage().2
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.iterations == 0 || !(1..=8).contains(&args.concurrency) {
        return Err(anyhow!(
            "--iterations must be positive and --concurrency must be 1..=8"
        ));
    }
    if args.rails.is_empty() == args.local_rails.is_empty() {
        return Err(anyhow!(
            "provide either --rail for explicit listeners or --local-rail for discovered listeners"
        ));
    }
    let explicit_routes = args
        .rails
        .iter()
        .map(|spec| parse_route(spec))
        .collect::<Result<Vec<_>>>()?;
    let local_paths = args
        .local_rails
        .iter()
        .map(|spec| parse_local_path(spec))
        .collect::<Result<Vec<_>>>()?;
    let endpoint =
        if args.coordinator.starts_with("http://") || args.coordinator.starts_with("https://") {
            args.coordinator.clone()
        } else {
            format!("http://{}", args.coordinator)
        };
    let mut client = KvClient::connect(endpoint)
        .await
        .map_err(|error| anyhow!(error.to_string()))?;
    let lookup = client
        .lookup_object(&args.namespace, &args.object_key)
        .await?
        .ok_or_else(|| anyhow!("object not found"))?;
    let reader = Arc::new(if local_paths.is_empty() {
        RailReader::new(explicit_routes, RailLimits::default())?
    } else {
        let placement = lookup
            .placement
            .as_ref()
            .ok_or_else(|| anyhow!("object lookup returned no placement"))?;
        RailReader::discover_from_placement(placement, &local_paths, RailLimits::default())?
    });
    let size = usize::try_from(lookup.descriptor.size)?;
    let mut destination = vec![0u8; size];
    for route in reader.routes() {
        println!(
            "route,id={},device={},port={},gid={},weight={},advertised={},listener={}",
            route.id,
            route.connection.device,
            route.connection.port,
            route.connection.gid_index,
            route.weight,
            route.advertised_endpoint,
            route.connection.endpoint
        );
    }
    for _ in 0..args.warmup {
        client
            .read_multi_rail_into(
                Arc::clone(&reader),
                &args.namespace,
                &args.object_key,
                &mut destination,
                None,
            )
            .await?
            .ok_or_else(|| anyhow!("object disappeared during warmup"))?;
    }
    let warmup_bytes: Vec<_> = reader.snapshots().iter().map(|rail| rail.bytes).collect();
    if args.concurrency > 1 {
        return run_concurrent(&args, &client, reader, size, &warmup_bytes).await;
    }
    let mut times = Vec::with_capacity(args.iterations);
    let mut cpu_user_us = 0u64;
    let mut cpu_system_us = 0u64;
    for iteration in 0..args.iterations {
        destination.fill(0xA5);
        let before_cpu = process_usage();
        let started = Instant::now();
        let bytes = client
            .read_multi_rail_into(
                Arc::clone(&reader),
                &args.namespace,
                &args.object_key,
                &mut destination,
                None,
            )
            .await?
            .ok_or_else(|| anyhow!("object disappeared during benchmark"))?;
        let elapsed = started.elapsed();
        let after_cpu = process_usage();
        cpu_user_us += after_cpu.0 - before_cpu.0;
        cpu_system_us += after_cpu.1 - before_cpu.1;
        if bytes != size {
            return Err(anyhow!("short read: {bytes} of {size} bytes"));
        }
        let checksum = twox_hash::xxh3::hash64(&destination);
        let gib_per_s = bytes as f64 / elapsed.as_secs_f64() / 1024f64.powi(3);
        println!(
            "sample,environment={},rails={},iteration={},bytes={},latency_us={},gib_per_s={gib_per_s:.3},xxh3={checksum:016x}",
            args.environment,
            reader.routes().len(),
            iteration + 1,
            bytes,
            elapsed.as_micros(),
        );
        times.push(elapsed);
    }
    times.sort_unstable();
    let median = times[times.len() / 2];
    let rail_bytes: Vec<_> = reader
        .snapshots()
        .iter()
        .zip(warmup_bytes)
        .map(|(snapshot, warmup)| snapshot.bytes - warmup)
        .collect();
    println!(
        "summary,environment={},rails={},bytes_per_iter={},iters={},median_us={},cpu_user_us={},cpu_system_us={},peak_rss_kb={},rail_bytes={rail_bytes:?}",
        args.environment,
        reader.routes().len(),
        size,
        args.iterations,
        median.as_micros(),
        cpu_user_us,
        cpu_system_us,
        process_usage().2
    );
    for snapshot in reader.snapshots() {
        let completed = snapshot.reads_ok + snapshot.reads_err;
        let average_us = snapshot.duration_us / completed.max(1);
        println!(
            "rail_stats,id={},device={},listener={},numa={:?},pci={:?},healthy={},cooldown_ms={},reads_ok={},reads_err={},bytes={},avg_us={},inflight_requests={},inflight_bytes={},peak_inflight_bytes={},registered_bytes_reserved={},peak_registered_bytes={}",
            snapshot.id,
            snapshot.device,
            snapshot.listener,
            snapshot.topology.numa_node,
            snapshot.topology.pci_bdf,
            snapshot.healthy,
            snapshot.cooldown_ms,
            snapshot.reads_ok,
            snapshot.reads_err,
            snapshot.bytes,
            average_us,
            snapshot.inflight_requests,
            snapshot.inflight_bytes,
            snapshot.peak_inflight_bytes,
            snapshot.registered_bytes,
            snapshot.peak_registered_bytes
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rail_spec_keeps_advertised_owner_separate_from_listener() {
        let route = parse_route("r1,mlx5_1,10.0.0.1:50053,10.0.1.1:50054,1,3").unwrap();
        assert_eq!(route.id, "r1");
        assert_eq!(route.advertised_endpoint, "10.0.0.1:50053");
        assert_eq!(route.connection.endpoint, "10.0.1.1:50054");
        assert_eq!(route.connection.device, "mlx5_1");
        assert_eq!(route.connection.port, 1);
        assert_eq!(route.connection.gid_index, 3);
    }

    #[test]
    fn rail_spec_accepts_a_capacity_weight() {
        let route = parse_route("r1,mlx5_1,10.0.0.1:50053,10.0.1.1:50054,1,3,4").unwrap();
        assert_eq!(route.weight, 4);
    }

    #[test]
    fn local_rail_spec_names_a_fabric_without_hardcoding_a_listener() {
        assert!(parse_local_path("r1,fabric-b,mlx5_1,1,3,4").is_ok());
        assert!(parse_local_path("r1,,mlx5_1,1,3").is_err());
        assert!(parse_local_path("r1,fabric-b,mlx5_1,0,3").is_err());
    }
}
