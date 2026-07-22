//! Prometheus metrics exporter
//!
//! Enable with: `cargo build --features metrics`
//!
//! Exported metrics:
//! - L1 cache hits/misses/evictions/size
//! - L2 reads/writes/bytes/striped_writes
//! - Per-device capacity, I/O, and utilization
//! - Redis metadata operations
//! - RDMA transfer and connection state
//!
//! The HTTP exporter listens on its own port (default 9090) and is started by main.rs.

#[cfg(feature = "metrics")]
pub use enabled::*;

#[cfg(not(feature = "metrics"))]
pub use disabled::*;

// ===================== enabled =====================
#[cfg(feature = "metrics")]
mod enabled {
    use crate::KVServiceContext;
    use hyper::service::{make_service_fn, service_fn};
    use hyper::{Body, Request, Response, Server, StatusCode};
    use prometheus::{
        Encoder, GaugeVec, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
        IntGaugeVec, Opts, Registry, TextEncoder,
    };
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use tracing::info;

    pub struct Metrics {
        pub registry: Registry,

        pub kvservice_up: IntGauge,
        pub kvservice_build_info: GaugeVec,
        pub kvservice_request_total: IntCounterVec,
        pub kvservice_request_duration_seconds: HistogramVec,
        pub kvservice_storage_io_total: IntCounterVec,
        pub kvservice_storage_io_bytes_total: IntCounterVec,
        pub kvservice_storage_io_duration_seconds: HistogramVec,
        pub kvservice_storage_io_inflight: IntGaugeVec,
        pub kvservice_storage_io_errors_total: IntCounterVec,
        pub kvservice_metadata_operations_total: IntCounterVec,
        pub kvservice_metadata_operation_duration_seconds: HistogramVec,
        pub kvservice_metadata_reconnect_total: IntCounter,
        pub kvservice_rdma_transfer_duration_seconds: HistogramVec,
        pub kvservice_cache_hit_total: IntCounterVec,
        pub kvservice_force_disk_read_total: IntCounter,
        pub kvservice_fallback_total: IntCounterVec,
        pub kvservice_nvme_read_bytes_total: IntCounterVec,
        pub kvservice_rdma_bytes_total: IntCounterVec,
        pub kvservice_rdma_errors_total: IntCounterVec,
        pub kvservice_rdma_connections: IntGaugeVec,

        pub l1_hits: IntCounter,
        pub l1_misses: IntCounter,
        pub l1_evictions: IntCounter,
        pub l1_size_bytes: IntGauge,

        pub l2_reads: IntCounter,
        pub l2_writes: IntCounter,
        pub l2_bytes_read: IntCounter,
        pub l2_bytes_written: IntCounter,
        pub l2_striped_writes: IntCounter,

        pub device_used_bytes: GaugeVec,
    }

    impl Default for Metrics {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Metrics {
        pub fn new() -> Self {
            let r = Registry::new();

            let kvservice_up = IntGauge::new("kvservice_up", "KVService liveness").unwrap();
            kvservice_up.set(1);
            let kvservice_build_info = GaugeVec::new(
                Opts::new("kvservice_build_info", "KVService build metadata"),
                &["version", "commit", "features"],
            )
            .unwrap();
            kvservice_build_info
                .with_label_values(&[
                    env!("CARGO_PKG_VERSION"),
                    option_env!("GIT_COMMIT").unwrap_or("unknown"),
                    feature_flags(),
                ])
                .set(1.0);
            let kvservice_request_total = IntCounterVec::new(
                Opts::new("kvservice_request_total", "KVService request count"),
                &["op", "status"],
            )
            .unwrap();
            let kvservice_request_duration_seconds = HistogramVec::new(
                HistogramOpts::new(
                    "kvservice_request_duration_seconds",
                    "KVService request duration",
                )
                .buckets(duration_buckets()),
                &["op"],
            )
            .unwrap();
            let kvservice_storage_io_total = IntCounterVec::new(
                Opts::new(
                    "kvservice_storage_io_total",
                    "Physical storage I/O operations",
                ),
                &["operation", "device", "executor", "mode", "status"],
            )
            .unwrap();
            let kvservice_storage_io_bytes_total = IntCounterVec::new(
                Opts::new(
                    "kvservice_storage_io_bytes_total",
                    "Physical storage I/O bytes",
                ),
                &["operation", "device", "executor", "mode"],
            )
            .unwrap();
            let kvservice_storage_io_duration_seconds = HistogramVec::new(
                HistogramOpts::new(
                    "kvservice_storage_io_duration_seconds",
                    "Physical storage I/O duration",
                )
                .buckets(duration_buckets()),
                &["operation", "device", "executor", "mode"],
            )
            .unwrap();
            let kvservice_storage_io_inflight = IntGaugeVec::new(
                Opts::new(
                    "kvservice_storage_io_inflight",
                    "In-flight physical storage I/O",
                ),
                &["operation", "device", "executor", "mode"],
            )
            .unwrap();
            let kvservice_storage_io_errors_total = IntCounterVec::new(
                Opts::new(
                    "kvservice_storage_io_errors_total",
                    "Physical storage I/O errors",
                ),
                &["operation", "device", "executor", "mode", "reason"],
            )
            .unwrap();
            let kvservice_metadata_operations_total = IntCounterVec::new(
                Opts::new(
                    "kvservice_metadata_operations_total",
                    "Redis metadata operations",
                ),
                &["operation", "status"],
            )
            .unwrap();
            let kvservice_metadata_operation_duration_seconds = HistogramVec::new(
                HistogramOpts::new(
                    "kvservice_metadata_operation_duration_seconds",
                    "Redis metadata operation duration",
                )
                .buckets(duration_buckets()),
                &["operation"],
            )
            .unwrap();
            let kvservice_metadata_reconnect_total = IntCounter::new(
                "kvservice_metadata_reconnect_total",
                "Redis metadata reconnect attempts",
            )
            .unwrap();
            let kvservice_rdma_transfer_duration_seconds = HistogramVec::new(
                HistogramOpts::new(
                    "kvservice_rdma_transfer_duration_seconds",
                    "KVService RDMA transfer duration",
                )
                .buckets(duration_buckets()),
                &["nic", "direction"],
            )
            .unwrap();
            let kvservice_cache_hit_total = IntCounterVec::new(
                Opts::new("kvservice_cache_hit_total", "KVService cache hits"),
                &["tier"],
            )
            .unwrap();
            let kvservice_force_disk_read_total = IntCounter::new(
                "kvservice_force_disk_read_total",
                "Forced disk-read requests",
            )
            .unwrap();
            let kvservice_fallback_total = IntCounterVec::new(
                Opts::new("kvservice_fallback_total", "KVService fallback count"),
                &["from", "to", "reason"],
            )
            .unwrap();
            let kvservice_nvme_read_bytes_total = IntCounterVec::new(
                Opts::new(
                    "kvservice_nvme_read_bytes_total",
                    "KVService per-device NVMe read bytes",
                ),
                &["device"],
            )
            .unwrap();
            let kvservice_rdma_bytes_total = IntCounterVec::new(
                Opts::new("kvservice_rdma_bytes_total", "KVService RDMA bytes"),
                &["nic", "direction"],
            )
            .unwrap();
            let kvservice_rdma_errors_total = IntCounterVec::new(
                Opts::new("kvservice_rdma_errors_total", "KVService RDMA errors"),
                &["nic", "type"],
            )
            .unwrap();
            let kvservice_rdma_connections = IntGaugeVec::new(
                Opts::new(
                    "kvservice_rdma_connections",
                    "Active KVService RDMA client connections",
                ),
                &["nic"],
            )
            .unwrap();

            let l1_hits = IntCounter::new("l1_cache_hits_total", "L1 cache hits").unwrap();
            let l1_misses = IntCounter::new("l1_cache_misses_total", "L1 cache misses").unwrap();
            let l1_evictions = IntCounter::new("l1_cache_evictions_total", "L1 evictions").unwrap();
            let l1_size_bytes = IntGauge::new("l1_cache_size_bytes", "L1 size").unwrap();

            let l2_reads = IntCounter::new("l2_reads_total", "L2 reads").unwrap();
            let l2_writes = IntCounter::new("l2_writes_total", "L2 writes").unwrap();
            let l2_bytes_read = IntCounter::new("l2_bytes_read_total", "L2 bytes read").unwrap();
            let l2_bytes_written =
                IntCounter::new("l2_bytes_written_total", "L2 bytes written").unwrap();
            let l2_striped_writes =
                IntCounter::new("l2_striped_writes_total", "L2 striped writes").unwrap();

            let device_used_bytes = GaugeVec::new(
                Opts::new("device_used_bytes", "Used bytes per device"),
                &["device"],
            )
            .unwrap();
            r.register(Box::new(kvservice_up.clone())).unwrap();
            r.register(Box::new(kvservice_build_info.clone())).unwrap();
            r.register(Box::new(kvservice_request_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_request_duration_seconds.clone()))
                .unwrap();
            r.register(Box::new(kvservice_storage_io_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_storage_io_bytes_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_storage_io_duration_seconds.clone()))
                .unwrap();
            r.register(Box::new(kvservice_storage_io_inflight.clone()))
                .unwrap();
            r.register(Box::new(kvservice_storage_io_errors_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_metadata_operations_total.clone()))
                .unwrap();
            r.register(Box::new(
                kvservice_metadata_operation_duration_seconds.clone(),
            ))
            .unwrap();
            r.register(Box::new(kvservice_metadata_reconnect_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_rdma_transfer_duration_seconds.clone()))
                .unwrap();
            r.register(Box::new(kvservice_cache_hit_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_force_disk_read_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_fallback_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_nvme_read_bytes_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_rdma_bytes_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_rdma_errors_total.clone()))
                .unwrap();
            r.register(Box::new(kvservice_rdma_connections.clone()))
                .unwrap();
            r.register(Box::new(l1_hits.clone())).unwrap();
            r.register(Box::new(l1_misses.clone())).unwrap();
            r.register(Box::new(l1_evictions.clone())).unwrap();
            r.register(Box::new(l1_size_bytes.clone())).unwrap();
            r.register(Box::new(l2_reads.clone())).unwrap();
            r.register(Box::new(l2_writes.clone())).unwrap();
            r.register(Box::new(l2_bytes_read.clone())).unwrap();
            r.register(Box::new(l2_bytes_written.clone())).unwrap();
            r.register(Box::new(l2_striped_writes.clone())).unwrap();
            r.register(Box::new(device_used_bytes.clone())).unwrap();
            Self {
                registry: r,
                kvservice_up,
                kvservice_build_info,
                kvservice_request_total,
                kvservice_request_duration_seconds,
                kvservice_storage_io_total,
                kvservice_storage_io_bytes_total,
                kvservice_storage_io_duration_seconds,
                kvservice_storage_io_inflight,
                kvservice_storage_io_errors_total,
                kvservice_metadata_operations_total,
                kvservice_metadata_operation_duration_seconds,
                kvservice_metadata_reconnect_total,
                kvservice_rdma_transfer_duration_seconds,
                kvservice_cache_hit_total,
                kvservice_force_disk_read_total,
                kvservice_fallback_total,
                kvservice_nvme_read_bytes_total,
                kvservice_rdma_bytes_total,
                kvservice_rdma_errors_total,
                kvservice_rdma_connections,
                l1_hits,
                l1_misses,
                l1_evictions,
                l1_size_bytes,
                l2_reads,
                l2_writes,
                l2_bytes_read,
                l2_bytes_written,
                l2_striped_writes,
                device_used_bytes,
            }
        }

        /// Sync atomic counters from ctx into the Prometheus metrics
        pub fn refresh(&self, ctx: &KVServiceContext) {
            let (hits, misses, evictions, size) = ctx.memory.stats();
            let l1_hit = self.kvservice_cache_hit_total.with_label_values(&["l1"]);
            l1_hit.reset();
            l1_hit.inc_by(hits);
            self.l1_hits.reset();
            self.l1_hits.inc_by(hits);
            self.l1_misses.reset();
            self.l1_misses.inc_by(misses);
            self.l1_evictions.reset();
            self.l1_evictions.inc_by(evictions);
            self.l1_size_bytes.set(size as i64);

            let st = ctx.storage.as_ref();
            self.l2_reads.reset();
            self.l2_reads.inc_by(st.reads_total.load(Ordering::Relaxed));
            self.l2_writes.reset();
            self.l2_writes
                .inc_by(st.writes_total.load(Ordering::Relaxed));
            self.l2_bytes_read.reset();
            self.l2_bytes_read
                .inc_by(st.bytes_read.load(Ordering::Relaxed));
            self.l2_bytes_written.reset();
            self.l2_bytes_written
                .inc_by(st.bytes_written.load(Ordering::Relaxed));
            self.l2_striped_writes.reset();
            self.l2_striped_writes
                .inc_by(st.striped_writes.load(Ordering::Relaxed));

            for (i, _) in ctx.router.devices().iter().enumerate() {
                let device = format!("nvme{}", i);
                let read_bytes = ctx.storage.device_read_bytes(i);
                let read_counter = self
                    .kvservice_nvme_read_bytes_total
                    .with_label_values(&[&device]);
                read_counter.reset();
                read_counter.inc_by(read_bytes);
                let used_bytes = ctx.storage.device_used_bytes(i);
                self.device_used_bytes
                    .with_label_values(&[&device])
                    .set(used_bytes as f64);
            }
        }

        pub fn record_request(&self, op: &str, status: &str, duration_seconds: f64) {
            self.kvservice_request_total
                .with_label_values(&[op, status])
                .inc();
            self.kvservice_request_duration_seconds
                .with_label_values(&[op])
                .observe(duration_seconds);
        }

        pub fn storage_io_started(
            &self,
            operation: &str,
            device: &str,
            executor: &str,
            mode: &str,
        ) {
            self.kvservice_storage_io_inflight
                .with_label_values(&[operation, device, executor, mode])
                .inc();
        }

        #[allow(clippy::too_many_arguments)]
        pub fn record_storage_io(
            &self,
            operation: &str,
            device: &str,
            executor: &str,
            mode: &str,
            status: &str,
            bytes: u64,
            duration_seconds: f64,
        ) {
            let labels = &[operation, device, executor, mode];
            self.kvservice_storage_io_inflight
                .with_label_values(labels)
                .dec();
            self.kvservice_storage_io_total
                .with_label_values(&[operation, device, executor, mode, status])
                .inc();
            self.kvservice_storage_io_duration_seconds
                .with_label_values(labels)
                .observe(duration_seconds);
            if status == "ok" && bytes > 0 {
                self.kvservice_storage_io_bytes_total
                    .with_label_values(labels)
                    .inc_by(bytes);
            }
        }

        pub fn record_storage_io_error(
            &self,
            operation: &str,
            device: &str,
            executor: &str,
            mode: &str,
            reason: &str,
        ) {
            self.kvservice_storage_io_errors_total
                .with_label_values(&[operation, device, executor, mode, reason])
                .inc();
        }

        pub fn record_storage_io_bytes(
            &self,
            operation: &str,
            device: &str,
            executor: &str,
            mode: &str,
            bytes: u64,
        ) {
            if bytes > 0 {
                self.kvservice_storage_io_bytes_total
                    .with_label_values(&[operation, device, executor, mode])
                    .inc_by(bytes);
            }
        }

        pub fn record_metadata_operation(
            &self,
            operation: &str,
            status: &str,
            duration_seconds: f64,
        ) {
            self.kvservice_metadata_operations_total
                .with_label_values(&[operation, status])
                .inc();
            self.kvservice_metadata_operation_duration_seconds
                .with_label_values(&[operation])
                .observe(duration_seconds);
        }

        pub fn record_metadata_reconnect(&self) {
            self.kvservice_metadata_reconnect_total.inc();
        }

        pub fn record_cache_hit(&self, tier: &str) {
            self.kvservice_cache_hit_total
                .with_label_values(&[tier])
                .inc();
        }

        pub fn record_force_disk_read(&self) {
            self.kvservice_force_disk_read_total.inc();
        }

        pub fn record_fallback(&self, from: &str, to: &str, reason: &str) {
            self.kvservice_fallback_total
                .with_label_values(&[from, to, reason])
                .inc();
        }

        pub fn record_rdma_bytes(&self, nic: &str, direction: &str, bytes: u64) {
            self.kvservice_rdma_bytes_total
                .with_label_values(&[nic, direction])
                .inc_by(bytes);
        }

        pub fn record_rdma_transfer_duration(
            &self,
            nic: &str,
            direction: &str,
            duration_seconds: f64,
        ) {
            self.kvservice_rdma_transfer_duration_seconds
                .with_label_values(&[nic, direction])
                .observe(duration_seconds);
        }

        pub fn record_rdma_error(&self, nic: &str, error_type: &str) {
            self.kvservice_rdma_errors_total
                .with_label_values(&[nic, error_type])
                .inc();
        }

        pub fn change_rdma_connections(&self, nic: &str, delta: i64) {
            self.kvservice_rdma_connections
                .with_label_values(&[nic])
                .add(delta);
        }
    }

    /// Render Prometheus text-format metrics; shared by the HTTP handler and unit tests.
    pub fn render_metrics(ctx: &KVServiceContext, metrics: &Metrics) -> anyhow::Result<Vec<u8>> {
        metrics.refresh(ctx);
        let mut buf = Vec::new();
        let encoder = TextEncoder::new();
        let mf = metrics.registry.gather();
        encoder.encode(&mf, &mut buf)?;
        Ok(buf)
    }

    /// Start the HTTP exporter
    pub async fn serve_metrics(
        addr: SocketAddr,
        ctx: Arc<KVServiceContext>,
        metrics: Arc<Metrics>,
    ) -> anyhow::Result<()> {
        info!("Prometheus exporter listening on {}", addr);

        let make_svc = make_service_fn(move |_conn| {
            let metrics = metrics.clone();
            let ctx = ctx.clone();
            async move {
                Ok::<_, Infallible>(service_fn(move |req: Request<Body>| {
                    let metrics = metrics.clone();
                    let ctx = ctx.clone();
                    async move { Ok::<_, Infallible>(handle(req, ctx, metrics)) }
                }))
            }
        });

        Server::bind(&addr).serve(make_svc).await?;
        Ok(())
    }

    fn handle(
        req: Request<Body>,
        ctx: Arc<KVServiceContext>,
        metrics: Arc<Metrics>,
    ) -> Response<Body> {
        match req.uri().path() {
            "/metrics" => {
                let encoder = TextEncoder::new();
                match render_metrics(&ctx, &metrics) {
                    Ok(buf) => Response::builder()
                        .header("Content-Type", encoder.format_type())
                        .body(Body::from(buf))
                        .unwrap(),
                    Err(e) => Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::from(e.to_string()))
                        .unwrap(),
                }
            }
            "/health" => Response::new(Body::from("ok")),
            _ => Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::empty())
                .unwrap(),
        }
    }

    fn duration_buckets() -> Vec<f64> {
        vec![
            0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0,
        ]
    }

    fn feature_flags() -> &'static str {
        if cfg!(feature = "rdma") && cfg!(feature = "io-uring") && cfg!(feature = "gds") {
            "metrics,rdma,io-uring,gds"
        } else if cfg!(feature = "rdma") && cfg!(feature = "io-uring") {
            "metrics,rdma,io-uring"
        } else if cfg!(feature = "rdma") {
            "metrics,rdma"
        } else if cfg!(feature = "io-uring") && cfg!(feature = "gds") {
            "metrics,io-uring,gds"
        } else if cfg!(feature = "io-uring") {
            "metrics,io-uring"
        } else {
            "metrics"
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::config::Config;
        use crate::KVServiceContext;
        use prost::bytes::Bytes;
        use std::sync::Arc;
        use tempfile::TempDir;

        fn make_test_config(tmp: &TempDir) -> Config {
            let mut config = Config::default();
            config.storage.devices = vec![tmp.path().join("nvme0"), tmp.path().join("nvme1")];
            config.metadata.redis_url = format!("memory://metrics-{}", tmp.path().display());
            config.memory_tier.capacity_mb = 1;
            config
        }

        #[test]
        fn render_metrics_exports_contextstore_counters() {
            let tmp = TempDir::new().unwrap();
            let metrics = Arc::new(Metrics::new());
            let ctx =
                KVServiceContext::new_with_metrics(make_test_config(&tmp), Some(metrics.clone()))
                    .unwrap();
            let key = crate::router::ObjectKey {
                namespace: "metrics-test".to_string(),
                object_key: "abcdef".to_string(),
            };
            let meta = crate::metadata::BlockMeta {
                device_id: 0,
                file_path: String::new(),
                size: 0,
                object_handle: String::new(),
                object_generation: 1,
                content_etag: String::new(),
                layout_version: 1,
                created_at: 0,
                last_accessed_at: 0,
                ttl_seconds: 0,
                num_tokens: 0,
                num_layers: 0,
                dtype: "bfloat16".to_string(),
                compressed: false,
                striping: None,
            };
            ctx.memory
                .put(&key, Bytes::from_static(b"abc"), meta)
                .unwrap();
            ctx.memory.get(&key).unwrap();
            metrics.record_metadata_operation("get_block", "ok", 0.001);
            metrics.record_metadata_reconnect();
            metrics.change_rdma_connections("nic0", 1);

            let body = render_metrics(&ctx, &metrics).unwrap();
            let text = String::from_utf8(body).unwrap();

            assert!(text.contains("l1_cache_hits_total 1"));
            assert!(text.contains("l2_writes_total 1"));
            assert!(text.contains("l2_bytes_written_total 3"));
            assert!(text.contains("device_used_bytes{device=\"nvme0\"} 3"));
            assert!(text.contains("kvservice_up 1"));
            assert!(text.contains("kvservice_cache_hit_total{tier=\"l1\"} 1"));
            assert!(text.contains("kvservice_nvme_read_bytes_total{device=\"nvme0\"} 0"));
            assert!(text.contains(
                "kvservice_storage_io_total{device=\"nvme0\",executor=\"tier_a\",mode=\"buffered\",operation=\"write\",status=\"ok\"} 1"
            ));
            assert!(text.contains(
                "kvservice_storage_io_bytes_total{device=\"nvme0\",executor=\"tier_a\",mode=\"buffered\",operation=\"write\"} 3"
            ));
            assert!(text.contains(
                "kvservice_metadata_operations_total{operation=\"get_block\",status=\"ok\"} 1"
            ));
            assert!(text.contains("kvservice_metadata_reconnect_total 1"));
            assert!(text.contains("kvservice_rdma_connections{nic=\"nic0\"} 1"));
        }
    }
}

// ===================== disabled stub =====================
#[cfg(not(feature = "metrics"))]
mod disabled {
    use crate::KVServiceContext;
    use std::net::SocketAddr;
    use std::sync::Arc;

    pub struct Metrics;
    impl Default for Metrics {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Metrics {
        pub fn new() -> Self {
            Self
        }
        pub fn refresh(&self, _ctx: &KVServiceContext) {}
        pub fn record_request(&self, _op: &str, _status: &str, _duration_seconds: f64) {}
        pub fn storage_io_started(
            &self,
            _operation: &str,
            _device: &str,
            _executor: &str,
            _mode: &str,
        ) {
        }
        pub fn record_storage_io(
            &self,
            _operation: &str,
            _device: &str,
            _executor: &str,
            _mode: &str,
            _status: &str,
            _bytes: u64,
            _duration_seconds: f64,
        ) {
        }
        pub fn record_storage_io_error(
            &self,
            _operation: &str,
            _device: &str,
            _executor: &str,
            _mode: &str,
            _reason: &str,
        ) {
        }
        pub fn record_storage_io_bytes(
            &self,
            _operation: &str,
            _device: &str,
            _executor: &str,
            _mode: &str,
            _bytes: u64,
        ) {
        }
        pub fn record_metadata_operation(
            &self,
            _operation: &str,
            _status: &str,
            _duration_seconds: f64,
        ) {
        }
        pub fn record_metadata_reconnect(&self) {}
        pub fn record_cache_hit(&self, _tier: &str) {}
        pub fn record_force_disk_read(&self) {}
        pub fn record_fallback(&self, _from: &str, _to: &str, _reason: &str) {}
        pub fn record_rdma_bytes(&self, _nic: &str, _direction: &str, _bytes: u64) {}
        pub fn record_rdma_transfer_duration(
            &self,
            _nic: &str,
            _direction: &str,
            _duration_seconds: f64,
        ) {
        }
        pub fn record_rdma_error(&self, _nic: &str, _error_type: &str) {}
        pub fn change_rdma_connections(&self, _nic: &str, _delta: i64) {}
    }

    pub async fn serve_metrics(
        _addr: SocketAddr,
        _ctx: Arc<KVServiceContext>,
        _m: Arc<Metrics>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}
