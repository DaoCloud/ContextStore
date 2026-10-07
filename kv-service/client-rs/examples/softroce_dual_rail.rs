//! `softroce_dual_rail` — end-to-end multi-rail read over two Soft-RoCE paths
//! in WSL2, against a real ContextStore server.
//!
//! Topology (see `wsl2-softroce-setup.md` and `setup-wsl2-rxe.sh`):
//!   - rxe0 bound to veth0 (192.168.96.110), rxe1 bound to veth1 (192.168.96.111)
//!   - one server process, gRPC on :50051, RDMA control channels on
//!     :50053 (rxe0) and :50054 (rxe1) via
//!     `CS_RDMA_DEVICES=rxe0:0.0.0.0:50053:1,rxe1:0.0.0.0:50054:1`
//!
//! Run:
//! ```text
//! cargo run -p contextstore-client-rs --features rdma --example softroce_dual_rail
//! ```
//!
//! What it does:
//!   1. PUTs a deterministic object through the gRPC control plane;
//!   2. `lookup_object` for a fresh descriptor (generation / layout included);
//!   3. derives the expected per-stripe xxh3-64 checksums locally (same digest
//!      the server stores in `chunk_checksums`);
//!   4. reads the object back twice — once single-rail (rxe0), once dual-rail
//!      (rxe0+rxe1) — verifying bytes and checksums both times;
//!   5. prints transfer goodput, per-rail timing, bottleneck attribution and
//!      the dual-rail speedup.
//!
//! Environment knobs (all optional):
//!   CS_GRPC            gRPC endpoint            (default http://127.0.0.1:50051)
//!   CS_RAIL_ENDPOINTS  RDMA control endpoints   (default 192.168.96.110:50053,192.168.96.111:50054)
//!   CS_RAIL_DEVICES    RDMA device per rail     (default rxe0,rxe1)
//!   CS_RAIL_GID        GID index (RXE=1, mlx5 host RoCEv2=3)  (default 1)
//!   CS_DEMO_MIB        object size in MiB       (default 64)
//!   CS_DEMO_NS / CS_DEMO_KEY  object identity   (default multirail-demo / obj)

use anyhow::{anyhow, ensure, Result};
use contextstore_client_rs::multi_rail::{stripe_checksum, RailManager, RailReader, RailReadStats};
use contextstore_client_rs::rdma::RdmaClient;
use contextstore_client_rs::KvClient;

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

fn print_stats(label: &str, s: &RailReadStats) {
    println!(
        "[{label}] rails={} bytes={} transfer={:.1} MB/s obj_ms={:.2} per_rail_ms={:?} verify_ms={:.2} verify_ok={}",
        s.rail_count, s.total_bytes, s.transfer_mbps, s.object_ms, s.per_rail_ms, s.verify_ms, s.verify_ok
    );
    println!("[{label}] bottleneck: {}", s.bottleneck());
}

#[tokio::main]
async fn main() -> Result<()> {
    let grpc = env_or("CS_GRPC", "http://127.0.0.1:50051");
    // NOTE: RDMA control endpoints must be the veth IPs the rxe devices are bound to
    // (setup-wsl2-rxe.sh: rxe0->veth0=192.168.96.110, rxe1->veth1=192.168.96.111).
    // 127.0.0.1 would resolve RDMA-CM to loopback, where no rxe device exists.
    let endpoints = env_list(
        "CS_RAIL_ENDPOINTS",
        "192.168.96.110:50053,192.168.96.111:50054",
    );
    let devices = env_list("CS_RAIL_DEVICES", "rxe0,rxe1");
    let gid: u8 = env_or("CS_RAIL_GID", "1").parse()?;
    let mib: usize = env_or("CS_DEMO_MIB", "64").parse()?;
    let ns = env_or("CS_DEMO_NS", "multirail-demo");
    let key = env_or("CS_DEMO_KEY", "obj");

    ensure!(
        endpoints.len() == devices.len() && !endpoints.is_empty(),
        "CS_RAIL_ENDPOINTS and CS_RAIL_DEVICES must be non-empty and equal length"
    );

    // 1. Deterministic payload so a corrupted read is detectable byte-for-byte.
    let size = mib * 1024 * 1024;
    let mut data = vec![0u8; size];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }

    // 2. PUT through the gRPC control plane.
    let mut kv = KvClient::connect(grpc.clone())
        .await
        .map_err(|e| anyhow!("gRPC connect {grpc}: {e}"))?;
    ensure!(
        kv.health().await.unwrap_or(false),
        "server health check failed at {grpc}"
    );
    let t_put = std::time::Instant::now();
    kv.put(&ns, &key, data.clone()).await?;
    println!(
        "[setup] PUT {ns}/{key} ({} MiB) via gRPC in {:.2?}",
        mib,
        t_put.elapsed()
    );

    // 3. Fresh descriptor from the control plane (generation / layout on the wire).
    let lookup = kv
        .lookup_object(&ns, &key)
        .await?
        .ok_or_else(|| anyhow!("object {ns}/{key} not found right after PUT"))?;
    let desc = lookup.descriptor;
    println!(
        "[setup] descriptor: size={} stripes={} chunk={} generation={} layout_v={}",
        desc.size, desc.stripe_count, desc.chunk_size, desc.object_generation, desc.layout_version
    );
    ensure!(
        desc.is_striped && desc.stripe_count >= 2,
        "object is not striped (stripe_count={}); multi-rail needs >= 2 stripes",
        desc.stripe_count
    );

    // 4. Expected per-stripe xxh3-64 (identical to the server's chunk_checksums).
    let chunk = desc.chunk_size as usize;
    let checksums: Vec<Option<String>> = (0..desc.stripe_count as usize)
        .map(|i| {
            let start = i * chunk;
            let end = ((i + 1) * chunk).min(data.len());
            Some(stripe_checksum(&data[start..end]))
        })
        .collect();

    // Rail factory: one RdmaClient per (endpoint, device) pair.
    let build_rails = |n: usize| -> Vec<Box<dyn RailReader>> {
        (0..n)
            .map(|i| {
                Box::new(
                    RdmaClient::new(endpoints[i].clone(), devices[i].clone()).with_gid_index(gid),
                ) as Box<dyn RailReader>
            })
            .collect()
    };

    // 5. Single-rail baseline (rxe0 only).
    let mut buf1 = vec![0u8; desc.size as usize];
    let stats1 = {
        let mut mgr = RailManager::new(build_rails(1));
        let mut reader = mgr.reader();
        let s = reader.read(&desc, &checksums, &mut buf1)?;
        mgr.reclaim(reader);
        s
    };
    ensure!(
        buf1 == data,
        "single-rail read content mismatch (corruption detected)"
    );
    print_stats("single-rail", &stats1);

    // 6. Multi-rail read across all configured rails.
    let n_rails = endpoints.len();
    let mut buf2 = vec![0u8; desc.size as usize];
    let stats2 = {
        let mut mgr = RailManager::new(build_rails(n_rails));
        println!(
            "[dual-rail] assembled multi-rail reader from {} independent RDMA paths ({})",
            mgr.rail_count(),
            devices.join("/")
        );
        let mut reader = mgr.reader();
        let s = reader.read(&desc, &checksums, &mut buf2)?;
        mgr.reclaim(reader);
        s
    };
    ensure!(
        buf2 == data,
        "multi-rail read content mismatch (corruption detected)"
    );
    print_stats("dual-rail", &stats2);

    // 7. Verdict.
    let speedup = stats1.object_ms / stats2.object_ms.max(1e-9);
    println!(
        "[result] dual-rail vs single-rail transfer-phase speedup: {:.2}x ({} rails, Soft-RoCE; not a hardware-NIC claim)",
        speedup, n_rails
    );
    Ok(())
}
