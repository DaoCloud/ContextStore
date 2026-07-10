//! Configuration loading (TOML)
//!
//! Corresponds to configs/server.toml

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::{KVError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub api: ApiConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub memory_tier: MemoryTierConfig,
    #[serde(default)]
    pub io_executor: IoExecutorConfig,
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub metadata: MetadataConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
    #[serde(default)]
    pub gds: GdsConfig,
}

// ===== Cluster / Placement =====
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClusterNodeConfig {
    /// Stable node ID. When empty the endpoint is used as fallback.
    #[serde(default)]
    pub node_id: String,
    /// gRPC data plane/control plane endpoint, e.g. "10.0.0.11:50051".
    pub grpc_endpoint: String,
    /// Optional RDMA endpoint, e.g. "10.0.0.11:18515".
    #[serde(default)]
    pub rdma_endpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClusterConfig {
    /// This node's ID; if empty, read CS_NODE_ID, and fall back to "local" if still empty.
    #[serde(default)]
    pub node_id: String,
    /// This node's outward gRPC endpoint; if empty, read CS_GRPC_ADVERTISE, and fall back to api.listen.
    #[serde(default)]
    pub grpc_advertise: String,
    /// This node's outward RDMA endpoint; if empty, read CS_RDMA_ADVERTISE.
    #[serde(default)]
    pub rdma_advertise: String,
    /// KVService data nodes eligible for object stripe placement.
    ///
    /// Empty means single-node mode; cross-node placement is enabled only when
    /// more than one data node is configured.
    #[serde(default)]
    pub data_nodes: Vec<ClusterNodeConfig>,
}

// ===== API =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub listen: String,
    pub max_connections: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:50051".to_string(),
            max_connections: 1000,
        }
    }
}

// ===== Storage =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Mount directory (or raw device path) of each NVMe device
    pub devices: Vec<PathBuf>,
    /// Data subdirectory name (under each device)
    pub data_subdir: String,
    /// Internal striping threshold for large values (bytes). 0 = disabled.
    /// Values above this threshold are split into `striping_chunk_size` chunks
    /// distributed across devices, matching ZFS/Lustre single-file parallelism.
    #[serde(default = "default_striping_threshold")]
    pub striping_threshold: u64,
    /// Size of each chunk when striping (bytes)
    #[serde(default = "default_striping_chunk_size")]
    pub striping_chunk_size: u64,
}

fn default_striping_threshold() -> u64 {
    256 * 1024 * 1024 // 256 MB
}

fn default_striping_chunk_size() -> u64 {
    64 * 1024 * 1024 // 64 MB
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            devices: vec![PathBuf::from("./data/nvme0")],
            data_subdir: "contextstore".to_string(),
            striping_threshold: default_striping_threshold(),
            striping_chunk_size: default_striping_chunk_size(),
        }
    }
}

// ===== Memory Tier (L1) =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTierConfig {
    /// L1 capacity (MB)
    pub capacity_mb: usize,
    /// Slab size (MB)
    pub slab_size_mb: usize,
    /// Whether to use pinned memory (requires CUDA environment)
    pub use_pinned_memory: bool,
}

impl Default for MemoryTierConfig {
    fn default() -> Self {
        Self {
            capacity_mb: 4096, // 4GB
            slab_size_mb: 64,
            use_pinned_memory: false,
        }
    }
}

// ===== I/O Executor =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoExecutorConfig {
    /// "tier_a" (ThreadPool) | "tier_b" (io_uring) | "tier_c" (SPDK)
    pub kind: String,
    pub thread_pool_size: usize,
    pub io_uring_depth: usize,
}

impl Default for IoExecutorConfig {
    fn default() -> Self {
        Self {
            kind: "tier_a".to_string(),
            thread_pool_size: 32,
            io_uring_depth: 256,
        }
    }
}

// ===== Router =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// "object_hash"
    pub strategy: String,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            strategy: "object_hash".to_string(),
        }
    }
}

// ===== Metadata =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataConfig {
    pub redis_url: String,
    #[serde(default = "default_redis_key_prefix")]
    pub redis_key_prefix: String,
    #[serde(default = "default_redis_connect_timeout_ms")]
    pub redis_connect_timeout_ms: u64,
    #[serde(default = "default_redis_command_timeout_ms")]
    pub redis_command_timeout_ms: u64,
}

fn default_redis_key_prefix() -> String {
    "contextstore:metadata:".to_string()
}

fn default_redis_connect_timeout_ms() -> u64 {
    1000
}

fn default_redis_command_timeout_ms() -> u64 {
    1000
}

impl Default for MetadataConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://127.0.0.1:6379/".to_string(),
            redis_key_prefix: default_redis_key_prefix(),
            redis_connect_timeout_ms: default_redis_connect_timeout_ms(),
            redis_command_timeout_ms: default_redis_command_timeout_ms(),
        }
    }
}

// ===== Metrics =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable the Prometheus HTTP exporter (requires feature=metrics at build time)
    pub enabled: bool,
    /// Listen address
    pub listen: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen: "0.0.0.0:9090".to_string(),
        }
    }
}

// ===== GDS (GPUDirect Storage) =====
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdsConfig {
    /// Whether to load libcufile.so + cuFileDriverOpen at startup.
    /// Failures are logged as warnings (not panics); callers fall back to pread/pwrite.
    pub enabled: bool,
    /// CUDA device ordinal (-1 = do not force setDevice)
    pub device: i32,
}

impl Default for GdsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device: -1,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api: Default::default(),
            storage: Default::default(),
            memory_tier: Default::default(),
            io_executor: Default::default(),
            router: Default::default(),
            cluster: Default::default(),
            metadata: Default::default(),
            metrics: Default::default(),
            gds: Default::default(),
        }
    }
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            KVError::Config(format!(
                "failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| KVError::Config(format!("failed to parse TOML: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.storage.devices.is_empty() {
            return Err(KVError::Config(
                "at least one storage device must be configured".to_string(),
            ));
        }
        for node in &self.cluster.data_nodes {
            if node.grpc_endpoint.trim().is_empty() {
                return Err(KVError::Config(
                    "cluster.data_nodes.grpc_endpoint must not be empty".to_string(),
                ));
            }
        }
        match self.router.strategy.as_str() {
            "object_hash" => {}
            other => {
                return Err(KVError::Config(format!(
                    "unknown router strategy: {}",
                    other
                )))
            }
        }
        match self.io_executor.kind.as_str() {
            "tier_a" | "tier_b" | "tier_c" => {}
            other => {
                return Err(KVError::Config(format!(
                    "unknown io_executor.kind: {}",
                    other
                )))
            }
        }
        if self.metadata.redis_url.trim().is_empty() {
            return Err(KVError::Config(
                "metadata.redis_url must not be empty".to_string(),
            ));
        }
        #[cfg(not(test))]
        if self.metadata.redis_url.starts_with("memory://") {
            return Err(KVError::Config(
                "metadata.redis_url memory:// is only available in unit tests".to_string(),
            ));
        }
        if self.metadata.redis_key_prefix.trim().is_empty() {
            return Err(KVError::Config(
                "metadata.redis_key_prefix must not be empty".to_string(),
            ));
        }
        if self.metadata.redis_connect_timeout_ms == 0 {
            return Err(KVError::Config(
                "metadata.redis_connect_timeout_ms must be greater than 0".to_string(),
            ));
        }
        if self.metadata.redis_command_timeout_ms == 0 {
            return Err(KVError::Config(
                "metadata.redis_command_timeout_ms must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}
