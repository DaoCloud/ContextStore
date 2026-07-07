//! ContextStore KV Service — Library entry point
//!
//! Module layout:
//! - `api`         : gRPC interface layer
//! - `router`      : request routing and sharding
//! - `memory_tier` : L1 in-memory cache
//! - `storage_tier`: L2 persistent storage
//! - `io_executor` : I/O executor (Tier A/B/C)
//! - `metadata`    : Prefix Index + Block Allocator
//! - `config`      : configuration loading

pub mod api;
pub mod config;
pub mod error;
#[cfg(feature = "gds")]
pub mod gds;
pub mod io_executor;
pub mod memory_tier;
pub mod metadata;
pub mod metrics;
pub mod rdma;
pub mod router;
pub mod storage_tier;

use std::sync::Arc;

use crate::config::Config;
use crate::memory_tier::MemoryTier;
use crate::metadata::MetadataService;
#[cfg(feature = "metrics")]
use crate::metrics::Metrics;
use crate::router::ShardRouter;
use crate::storage_tier::StorageTier;

/// Service runtime context, shared by gRPC handlers
pub struct KVServiceContext {
    pub config: Config,
    pub router: Arc<ShardRouter>,
    pub memory: Arc<MemoryTier>,
    pub storage: Arc<StorageTier>,
    pub metadata: Arc<MetadataService>,
    #[cfg(feature = "metrics")]
    pub metrics: Option<Arc<Metrics>>,
}

impl KVServiceContext {
    #[cfg(not(feature = "metrics"))]
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let router = Arc::new(ShardRouter::new(&config)?);
        let storage = Arc::new(StorageTier::new(&config, router.clone())?);
        let metadata = storage.metadata();
        let memory = Arc::new(MemoryTier::new(&config, storage.clone()));

        Ok(Self {
            config,
            router,
            memory,
            storage,
            metadata,
        })
    }

    #[cfg(feature = "metrics")]
    pub fn new(config: Config) -> anyhow::Result<Self> {
        Self::new_with_metrics(config, None)
    }

    #[cfg(feature = "metrics")]
    pub fn new_with_metrics(config: Config, metrics: Option<Arc<Metrics>>) -> anyhow::Result<Self> {
        let router = Arc::new(ShardRouter::new(&config)?);
        let storage = Arc::new(StorageTier::new(&config, router.clone())?);
        let metadata = storage.metadata();
        let memory = Arc::new(MemoryTier::new(&config, storage.clone()));

        Ok(Self {
            config,
            router,
            memory,
            storage,
            metadata,
            metrics,
        })
    }
}
