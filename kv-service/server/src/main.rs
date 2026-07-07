//! ContextStore KV Service — server entry point
//!
//! Starts the gRPC service and handles KV operation requests.

use clap::Parser;
#[cfg(feature = "metrics")]
use contextstore_server::metrics::{serve_metrics, Metrics};
use contextstore_server::{
    api::generated::contextstore::kv::v1::kv_service_server::KvServiceServer, api::KVServiceImpl,
    config::Config, KVServiceContext,
};
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "contextstore-server")]
#[command(about = "ContextStore KV Service - JBOF-optimized LLM KV cache storage", long_about = None)]
struct Cli {
    /// Path to the config file
    #[arg(short, long, default_value = "configs/server.toml")]
    config: PathBuf,

    /// Log level (trace/debug/info/warn/error)
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("contextstore_server={}", cli.log_level)));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    info!(
        "starting ContextStore KV Service v{}...",
        env!("CARGO_PKG_VERSION")
    );
    info!("config file: {}", cli.config.display());

    // Load config
    let config = Config::from_file(&cli.config)?;
    info!(
        "loaded config: {} storage device(s), L1 capacity {} MB, I/O executor = {}",
        config.storage.devices.len(),
        config.memory_tier.capacity_mb,
        config.io_executor.kind,
    );

    // Optional: initialize GDS (only when compiled with --features gds and enabled in config)
    #[cfg(feature = "gds")]
    if config.gds.enabled {
        match contextstore_server::gds::GdsDriver::init() {
            Ok(true) => {
                info!("GDS: enabled, cuFile driver ready");
                if config.gds.device >= 0 {
                    if let Err(e) = contextstore_server::gds::driver::set_device(config.gds.device)
                    {
                        warn!("GDS: cudaSetDevice({}) failed: {}", config.gds.device, e);
                    }
                }
            }
            Ok(false) => warn!("GDS: runtime probe failed, falling back to non-GDS path"),
            Err(e) => warn!("GDS: initialization error: {}", e),
        }
    }

    #[cfg(feature = "metrics")]
    let metrics = if config.metrics.enabled {
        Some(Arc::new(Metrics::new()))
    } else {
        None
    };

    // Build the service context
    #[cfg(feature = "metrics")]
    let ctx = Arc::new(KVServiceContext::new_with_metrics(
        config.clone(),
        metrics.clone(),
    )?);
    #[cfg(not(feature = "metrics"))]
    let ctx = Arc::new(KVServiceContext::new(config.clone())?);

    // Optional: start the Prometheus exporter
    if config.metrics.enabled {
        #[cfg(feature = "metrics")]
        {
            let addr = config.metrics.listen.parse()?;
            let ctx_m = ctx.clone();
            let m = metrics
                .clone()
                .expect("metrics.enabled=true should initialize Metrics");
            tokio::spawn(async move {
                if let Err(e) = serve_metrics(addr, ctx_m, m).await {
                    warn!("Prometheus exporter exited: {}", e);
                }
            });
        }
        #[cfg(not(feature = "metrics"))]
        warn!(
            "metrics.enabled=true, but this binary was built without the metrics feature; Prometheus exporter not started"
        );
    }

    // Start the gRPC server
    let addr = config.api.listen.parse()?;
    info!("gRPC listening on: {}", addr);

    // Optional: start the RDMA tier server (bypasses gRPC for bulk transfer)
    #[cfg(feature = "rdma")]
    {
        use contextstore_server::rdma::server::{RdmaDeviceConfig, RdmaServerConfig};
        let mut cfg = RdmaServerConfig::default();

        // Debug knob: env var overrides slab size (e.g. CS_RDMA_SLAB_MB=0 forces the fallback path).
        if let Ok(v) = std::env::var("CS_RDMA_SLAB_MB") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.rdma_slab_size_mb = n;
                info!("CS_RDMA_SLAB_MB override rdma_slab_size_mb={}", n);
            }
        }

        // Multi-NIC support: configured via CS_RDMA_DEVICES (comma-separated list of
        // `device:tcp_listen` items), e.g.
        //   CS_RDMA_DEVICES=mlx5_0:0.0.0.0:50053,mlx5_1:0.0.0.0:50054
        // Parse errors fall back to default (single NIC mlx5_0). port_num=1, gid_index=3
        // stay at their defaults; each NIC gets its own listener but shares the same slab
        // (each PD does its own reg_mr).
        if let Ok(v) = std::env::var("CS_RDMA_DEVICES") {
            let mut devs = Vec::new();
            let mut parse_err: Option<String> = None;
            for item in v.split(',') {
                let item = item.trim();
                if item.is_empty() {
                    continue;
                }
                // Format: device:host:port, e.g. mlx5_0:0.0.0.0:50053
                let parts: Vec<&str> = item.splitn(2, ':').collect();
                if parts.len() != 2 {
                    parse_err = Some(format!("bad item '{}', expect dev:host:port", item));
                    break;
                }
                devs.push(RdmaDeviceConfig {
                    device_name: parts[0].to_string(),
                    port_num: 1,
                    gid_index: 3,
                    tcp_listen: parts[1].to_string(),
                });
            }
            if let Some(e) = parse_err {
                warn!(
                    "CS_RDMA_DEVICES parse error ({}); using default single-NIC",
                    e
                );
            } else if !devs.is_empty() {
                info!(
                    "CS_RDMA_DEVICES override: {} NIC(s) {:?}",
                    devs.len(),
                    devs.iter()
                        .map(|d| format!("{}@{}", d.device_name, d.tcp_listen))
                        .collect::<Vec<_>>()
                );
                cfg.devices = devs;
            }
        }

        info!(
            "RDMA tier enabled: {} NIC(s), primary listener={}",
            cfg.devices.len(),
            cfg.devices[0].tcp_listen
        );
        let ctx_r = ctx.clone();
        std::thread::spawn(move || {
            if let Err(e) = contextstore_server::rdma::server::run_server(ctx_r, cfg) {
                warn!("RDMA server exited: {}", e);
            }
        });
    }

    let svc = KVServiceImpl::new_shared(ctx);
    // Large KV payloads: a single layer can reach several MB and a batch several
    // hundred MB. Raise tonic's default 4MB limit to 2GiB.
    let kv_server = KvServiceServer::new(svc)
        .max_decoding_message_size(2 * 1024 * 1024 * 1024)
        .max_encoding_message_size(2 * 1024 * 1024 * 1024);
    // HTTP/2 flow-control tuning: the ~64KB default forces large-value transfers to
    // wait on frequent WINDOW_UPDATEs, capping loopback throughput at ~0.2 GB/s.
    // Bump the stream window to 64MB and the connection window to 128MB, paired with
    // a 16MB max frame size, so large chunks can be pushed through in one go.
    Server::builder()
        .initial_stream_window_size(Some(64 * 1024 * 1024))
        .initial_connection_window_size(Some(128 * 1024 * 1024))
        .max_frame_size(Some(16 * 1024 * 1024 - 1))
        .add_service(kv_server)
        .serve(addr)
        .await?;

    Ok(())
}
