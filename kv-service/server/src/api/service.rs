//! gRPC service handler implementation

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use futures::{future::join_all, Stream};
use prost::bytes::{Bytes, BytesMut};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinSet;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use super::generated::contextstore::kv::v1 as pb;
use crate::config::ClusterNodeConfig;
use crate::metadata::{BlockMeta, ChunkLocation, StripingInfo};
use crate::router::ObjectKey as InternalKey;
use crate::KVServiceContext;
use twox_hash::xxh3::hash64;

#[derive(Debug, Clone, PartialEq, Eq)]
struct DataNode {
    node_id: String,
    grpc_endpoint: String,
    rdma_endpoint: String,
}

const DATA_NODE_MAX_MESSAGE_SIZE: usize = 2 * 1024 * 1024 * 1024;
const DATA_NODE_STREAM_WINDOW_SIZE: u32 = 64 * 1024 * 1024;
const DATA_NODE_CONNECTION_WINDOW_SIZE: u32 = 128 * 1024 * 1024;

/// Connect to a data node with the same large-payload HTTP/2 settings as the public API.
async fn connect_data_node(
    endpoint: &str,
) -> Result<pb::kv_service_client::KvServiceClient<Channel>, Status> {
    let channel = Channel::from_shared(grpc_uri(endpoint))
        .map_err(|err| Status::unavailable(format!("invalid data node endpoint: {err}")))?
        .initial_stream_window_size(Some(DATA_NODE_STREAM_WINDOW_SIZE))
        .initial_connection_window_size(Some(DATA_NODE_CONNECTION_WINDOW_SIZE))
        .connect()
        .await
        .map_err(|err| Status::unavailable(format!("connect data node: {err}")))?;
    Ok(pb::kv_service_client::KvServiceClient::new(channel)
        .max_decoding_message_size(DATA_NODE_MAX_MESSAGE_SIZE)
        .max_encoding_message_size(DATA_NODE_MAX_MESSAGE_SIZE))
}

struct ChunkWriteResult {
    location: ChunkLocation,
    is_local: bool,
    connect_elapsed: Duration,
    transfer_elapsed: Duration,
    total_elapsed: Duration,
}

/// Timing/counters for the streaming distributed PUT pipeline.
#[derive(Default)]
struct StreamPutStats {
    local_stripes: usize,
    remote_stripes: usize,
    slowest_stripe: Duration,
    receive_elapsed: Duration,
    backpressure_wait: Duration,
    drain_wait: Duration,
    metadata_elapsed: Duration,
}

pub struct KVServiceImpl {
    ctx: Arc<KVServiceContext>,
    write_locks: Arc<DashMap<String, Arc<AsyncMutex<()>>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{ChunkLocation, StripingInfo};

    fn ctx_with_nodes(nodes: Vec<ClusterNodeConfig>) -> KVServiceContext {
        ctx_with_local_and_nodes("coordinator", "127.0.0.1:50051", nodes)
    }

    fn ctx_with_local_and_nodes(
        node_id: &str,
        grpc_advertise: &str,
        nodes: Vec<ClusterNodeConfig>,
    ) -> KVServiceContext {
        let mut cfg = crate::config::Config::default();
        cfg.cluster.node_id = node_id.to_string();
        cfg.cluster.grpc_advertise = grpc_advertise.to_string();
        cfg.cluster.data_nodes = nodes;
        cfg.metadata.redis_url = format!(
            "memory://api-service-placement-{}-{}",
            node_id,
            cfg.cluster.data_nodes.len(),
        );
        KVServiceContext::new(cfg).unwrap()
    }

    fn data_node(id: &str, grpc: &str) -> ClusterNodeConfig {
        ClusterNodeConfig {
            node_id: id.to_string(),
            grpc_endpoint: grpc.to_string(),
            rdma_endpoint: String::new(),
        }
    }

    fn meta() -> BlockMeta {
        BlockMeta {
            device_id: 0,
            file_path: "/tmp/object.bin".to_string(),
            size: 128,
            object_handle: "handle-2".to_string(),
            object_generation: 2,
            content_etag: "etag-2".to_string(),
            layout_version: 1,
            created_at: 0,
            last_accessed_at: 0,
            ttl_seconds: 0,
            num_tokens: 16,
            num_layers: 1,
            dtype: "bfloat16".to_string(),
            compressed: false,
            striping: None,
        }
    }

    fn key() -> InternalKey {
        InternalKey {
            namespace: "test".to_string(),
            object_key: "object".to_string(),
        }
    }

    #[test]
    fn descriptor_contains_version_identity() {
        let meta = meta();
        let desc = descriptor_from_meta(&key(), &meta);

        assert_eq!(desc.object_generation, 2);
        assert_eq!(desc.content_etag, "etag-2");
        assert_eq!(desc.layout_version, 1);
        assert_eq!(desc.size, 128);
        assert!(!desc.object_handle.is_empty());
        validate_descriptor(&desc, &meta).unwrap();
    }

    #[test]
    fn descriptor_validation_rejects_stale_generation() {
        let meta = meta();
        let mut desc = descriptor_from_meta(&key(), &meta);
        desc.object_generation = 1;

        let err = validate_descriptor(&desc, &meta).unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn placement_uses_materialized_chunk_locations() {
        let mut cfg = crate::config::Config::default();
        cfg.cluster.node_id = "coordinator".to_string();
        cfg.cluster.grpc_advertise = "127.0.0.1:50051".to_string();
        cfg.metadata.redis_url = "memory://api-service-placement".to_string();
        let ctx = KVServiceContext::new(cfg).unwrap();
        let mut meta = meta();
        meta.size = 12;
        meta.file_path.clear();
        meta.striping = Some(StripingInfo {
            chunk_size: 6,
            chunk_devices: vec![0, 0],
            chunk_paths: vec!["/tmp/a".to_string(), "/tmp/b".to_string()],
            total_size: 12,
            chunk_locations: vec![
                ChunkLocation {
                    stripe_index: 0,
                    node_id: "node-a".to_string(),
                    grpc_endpoint: "10.0.0.1:50051".to_string(),
                    rdma_endpoint: String::new(),
                    device_id: 0,
                    storage_handle: "/tmp/a".to_string(),
                    offset: 0,
                    length: 6,
                    checksum: "checksum-a".to_string(),
                },
                ChunkLocation {
                    stripe_index: 1,
                    node_id: "node-b".to_string(),
                    grpc_endpoint: "10.0.0.2:50051".to_string(),
                    rdma_endpoint: String::new(),
                    device_id: 0,
                    storage_handle: "/tmp/b".to_string(),
                    offset: 6,
                    length: 6,
                    checksum: "checksum-b".to_string(),
                },
            ],
            chunk_checksums: vec!["checksum-a".to_string(), "checksum-b".to_string()],
        });

        let placement = placement_from_meta(&ctx, &key(), &meta);
        assert_eq!(placement.chunks.len(), 2);
        assert_eq!(placement.chunks[0].node_id, "node-a");
        assert_eq!(placement.chunks[1].grpc_endpoint, "10.0.0.2:50051");
        assert_eq!(placement.chunks[0].checksum, "checksum-a");
    }

    #[test]
    fn placement_epoch_changes_when_topology_changes() {
        let ctx_a = ctx_with_nodes(vec![
            data_node("node-a", "10.0.0.1:50051"),
            data_node("node-b", "10.0.0.2:50051"),
        ]);
        let ctx_b = ctx_with_nodes(vec![
            data_node("node-a", "10.0.0.1:50051"),
            data_node("node-c", "10.0.0.3:50051"),
        ]);

        let meta = meta();
        let placement_a = placement_from_meta(&ctx_a, &key(), &meta);
        let placement_b = placement_from_meta(&ctx_b, &key(), &meta);

        assert_ne!(placement_a.placement_epoch, placement_b.placement_epoch);
        assert_ne!(placement_a.layout_hash, placement_b.layout_hash);
    }

    #[test]
    fn placement_epoch_is_stable_across_local_nodes() {
        let nodes = vec![
            data_node("node-a", "10.0.0.1:50051"),
            data_node("node-b", "10.0.0.2:50051"),
        ];
        let ctx_a = ctx_with_local_and_nodes("node-a", "10.0.0.1:50051", nodes.clone());
        let ctx_b = ctx_with_local_and_nodes("node-b", "10.0.0.2:50051", nodes);

        assert_eq!(placement_epoch(&ctx_a), placement_epoch(&ctx_b));
    }

    #[test]
    fn local_stripe_indices_rotate_devices_per_data_node() {
        let nodes = vec![
            data_node("node-a", "10.0.0.1:50051"),
            data_node("node-b", "10.0.0.2:50051"),
        ];
        let ctx_a = ctx_with_local_and_nodes("node-a", "10.0.0.1:50051", nodes.clone());
        let ctx_b = ctx_with_local_and_nodes("node-b", "10.0.0.2:50051", nodes);
        let key = key();
        let mut node_a_seen = 0;
        let mut node_b_seen = 0;

        for stripe_index in 0..8 {
            let node = select_data_node(&ctx_a, &key, stripe_index);
            let (ctx, seen) = if node.node_id == "node-a" {
                (&ctx_a, &mut node_a_seen)
            } else {
                (&ctx_b, &mut node_b_seen)
            };
            assert_eq!(
                local_device_stripe_index(ctx, &key, stripe_index),
                Some(*seen)
            );
            *seen += 1;
        }
    }

    #[test]
    fn local_stripe_indices_handle_duplicate_local_data_nodes() {
        let nodes = vec![
            data_node("node-a", "10.0.0.1:50051"),
            data_node("node-a", "10.0.0.1:50051"),
        ];
        let ctx = ctx_with_local_and_nodes("node-a", "10.0.0.1:50051", nodes);

        for stripe_index in 0..8 {
            assert_eq!(
                local_device_stripe_index(&ctx, &key(), stripe_index),
                Some(stripe_index)
            );
        }
    }

    #[test]
    fn placement_validation_accepts_current_descriptor() {
        let ctx = ctx_with_nodes(vec![data_node("node-a", "10.0.0.1:50051")]);
        let meta = meta();
        let placement = placement_from_meta(&ctx, &key(), &meta);

        validate_placement_descriptor(&ctx, &key(), &meta, Some(&placement)).unwrap();
    }

    #[test]
    fn placement_validation_rejects_stale_epoch() {
        let ctx = ctx_with_nodes(vec![data_node("node-a", "10.0.0.1:50051")]);
        let meta = meta();
        let mut placement = placement_from_meta(&ctx, &key(), &meta);
        placement.placement_epoch = placement.placement_epoch.wrapping_add(1);

        let err = validate_placement_descriptor(&ctx, &key(), &meta, Some(&placement)).unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn meta_from_pb_applies_put_options_ttl() {
        let options = pb::PutOptions {
            ttl_seconds: 30,
            if_not_exists: false,
            compression: pb::CompressionType::None as i32,
        };
        let meta = meta_from_pb(None, Some(&options));

        assert_eq!(meta.ttl_seconds, 30);
    }

    #[test]
    fn meta_from_pb_clamps_negative_ttl_to_disabled() {
        let options = pb::PutOptions {
            ttl_seconds: -1,
            if_not_exists: false,
            compression: pb::CompressionType::None as i32,
        };
        let meta = meta_from_pb(None, Some(&options));

        assert_eq!(meta.ttl_seconds, 0);
    }

    #[tokio::test]
    async fn metadata_get_live_purges_expired_metadata() {
        let ctx = ctx_with_nodes(Vec::new());
        let key = key();
        let mut expired = meta();
        expired.ttl_seconds = 1;
        ctx.metadata
            .put_block(&key.to_string_key(), &expired)
            .unwrap();

        let service = KVServiceImpl::new(ctx);
        assert!(service.metadata_get_live(&key).await.unwrap().is_none());
        assert!(service
            .ctx
            .metadata
            .get_block(&key.to_string_key())
            .unwrap()
            .is_none());
    }
}

impl KVServiceImpl {
    pub fn new(ctx: KVServiceContext) -> Self {
        Self {
            ctx: Arc::new(ctx),
            write_locks: Arc::new(DashMap::new()),
        }
    }

    pub fn new_shared(ctx: Arc<KVServiceContext>) -> Self {
        Self {
            ctx,
            write_locks: Arc::new(DashMap::new()),
        }
    }

    fn record_request<T>(
        &self,
        op: &str,
        start: Instant,
        result: &Result<Response<T>, Status>,
        ok_status: &str,
    ) {
        #[cfg(feature = "metrics")]
        if let Some(metrics) = &self.ctx.metrics {
            let status = match result {
                Ok(_) => ok_status,
                Err(status) => status.code().description(),
            };
            metrics.record_request(op, status, start.elapsed().as_secs_f64());
        }
        #[cfg(not(feature = "metrics"))]
        {
            let _ = (op, start, result, ok_status);
        }
    }

    fn should_use_distributed_placement(&self, len: usize) -> bool {
        distributed_placement_enabled(&self.ctx, len)
    }

    fn key_write_lock(&self, key: &InternalKey) -> Arc<AsyncMutex<()>> {
        let str_key = key.to_string_key();
        self.write_locks
            .entry(str_key)
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    fn meta_identity_matches(actual: &BlockMeta, expected: &BlockMeta) -> bool {
        actual.object_handle == expected.object_handle
            && actual.object_generation == expected.object_generation
            && actual.layout_version == expected.layout_version
            && actual.content_etag == expected.content_etag
            && actual.size == expected.size
    }

    async fn metadata_get_live(&self, key: &InternalKey) -> Result<Option<BlockMeta>, Status> {
        let metadata = self.ctx.metadata.clone();
        let str_key = key.to_string_key();
        let meta = tokio::task::spawn_blocking(move || metadata.get_block(&str_key))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        let Some(meta) = meta else {
            return Ok(None);
        };
        if !meta.is_expired() {
            return Ok(Some(meta));
        }
        if self.purge_expired_object(key, &meta).await? {
            return Ok(None);
        }

        let metadata = self.ctx.metadata.clone();
        let str_key = key.to_string_key();
        tokio::task::spawn_blocking(move || metadata.get_block(&str_key))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)
            .map(|current| current.filter(|current| !current.is_expired()))
    }

    async fn purge_expired_object(
        &self,
        key: &InternalKey,
        meta: &BlockMeta,
    ) -> Result<bool, Status> {
        if !meta.is_expired() {
            return Ok(false);
        }
        let write_lock = self.key_write_lock(key);
        let _guard = write_lock.lock().await;
        self.purge_expired_object_locked(key, meta).await
    }

    async fn purge_expired_object_locked(
        &self,
        key: &InternalKey,
        meta: &BlockMeta,
    ) -> Result<bool, Status> {
        if !meta.is_expired() {
            return Ok(false);
        }
        let placement = placement_from_meta(&self.ctx, key, meta);
        if self.placement_has_remote_chunks(&placement) {
            let metadata = self.ctx.metadata.clone();
            let str_key = key.to_string_key();
            let expected = meta.clone();
            let should_delete =
                tokio::task::spawn_blocking(move || -> Result<bool, crate::error::KVError> {
                    let Some(current) = metadata.get_block(&str_key)? else {
                        return Ok(false);
                    };
                    Ok(current.is_expired() && Self::meta_identity_matches(&current, &expected))
                })
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            if !should_delete {
                return Ok(false);
            }

            let metadata = self.ctx.metadata.clone();
            let str_key = key.to_string_key();
            let expected = meta.clone();
            let deleted = tokio::task::spawn_blocking(move || {
                metadata.delete_block_if_matches(&str_key, &expected)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
            if !deleted {
                return Ok(false);
            }
            self.ctx.memory.invalidate(key);
            self.delete_distributed_chunks(placement).await?;
            return Ok(true);
        }

        let storage = self.ctx.storage.clone();
        let key = key.clone();
        let meta = meta.clone();
        tokio::task::spawn_blocking(move || storage.delete_if_expired(&key, &meta))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)
    }

    fn make_distributed_descriptor(
        &self,
        key: &InternalKey,
        meta: &BlockMeta,
        stripe_count: usize,
        chunk_size: u64,
    ) -> pb::ObjectDescriptor {
        let mut desc = descriptor_from_meta(key, meta);
        desc.is_striped = true;
        desc.stripe_count = stripe_count as u32;
        desc.chunk_size = chunk_size;
        desc
    }

    fn flatten_segments(segments: Vec<Bytes>) -> Bytes {
        if segments.len() == 1 {
            return segments.into_iter().next().unwrap_or_else(Bytes::new);
        }
        let total: usize = segments.iter().map(|s| s.len()).sum();
        let mut buf = BytesMut::with_capacity(total);
        for seg in segments {
            buf.extend_from_slice(&seg);
        }
        buf.freeze()
    }

    async fn put_chunk_on_node(
        ctx: Arc<KVServiceContext>,
        node: DataNode,
        key: InternalKey,
        descriptor: pb::ObjectDescriptor,
        stripe_index: usize,
        offset: u64,
        chunk_size: u64,
        total_size: u64,
        data: Bytes,
    ) -> Result<ChunkWriteResult, Status> {
        let chunk_start = Instant::now();
        let data_len = data.len() as u64;
        if is_local_node(&ctx, &node) {
            let device_stripe_index = local_device_stripe_index(&ctx, &key, stripe_index)
                .ok_or_else(|| Status::failed_precondition("stripe assigned to a different data node"))?;
            let key_for_write = key.clone();
            let storage = ctx.storage.clone();
            let generation = descriptor.object_generation;
            let layout_version = descriptor.layout_version;
            let (device_id, storage_handle, checksum) = tokio::task::spawn_blocking(move || {
                storage.put_placement_chunk(
                    &key_for_write,
                    stripe_index,
                    device_stripe_index,
                    generation,
                    layout_version,
                    data,
                )
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
            let total_elapsed = chunk_start.elapsed();
            tracing::debug!(
                event = "grpc_placement_chunk_write",
                status = "ok",
                target = "local",
                stripe_index,
                bytes = data_len,
                storage_us = total_elapsed.as_micros(),
            );
            return Ok(ChunkWriteResult {
                location: ChunkLocation {
                    stripe_index: stripe_index as u32,
                    node_id: node.node_id,
                    grpc_endpoint: node.grpc_endpoint,
                    rdma_endpoint: node.rdma_endpoint,
                    device_id,
                    storage_handle,
                    offset,
                    length: data_len,
                    checksum,
                },
                is_local: true,
                connect_elapsed: Duration::ZERO,
                transfer_elapsed: Duration::ZERO,
                total_elapsed,
            });
        }

        let connect_start = Instant::now();
        let mut client = connect_data_node(&node.grpc_endpoint)
            .await
            .map_err(|status| {
                Status::unavailable(format!(
                    "connect data node {}: {}",
                    node.node_id,
                    status.message()
                ))
            })?;
        let connect_elapsed = connect_start.elapsed();
        let transfer_start = Instant::now();
        let resp = client
            .put_placement_chunk(pb::PutPlacementChunkRequest {
                key: Some(internal_key_to_pb(&key)),
                descriptor: Some(descriptor),
                stripe_index: stripe_index as u32,
                chunk_size,
                total_size,
                data,
            })
            .await
            .map_err(|e| Status::unavailable(format!("put chunk to {}: {}", node.node_id, e)))?
            .into_inner();
        let transfer_elapsed = transfer_start.elapsed();
        if !resp.success {
            return Err(Status::internal(format!(
                "data node {} rejected placement chunk",
                node.node_id
            )));
        }
        let chunk = resp
            .chunk
            .ok_or_else(|| Status::internal("missing placement chunk in response"))?;
        let total_elapsed = chunk_start.elapsed();
        tracing::debug!(
            event = "grpc_placement_chunk_write",
            status = "ok",
            target = "remote",
            stripe_index,
            bytes = data_len,
            connect_us = connect_elapsed.as_micros(),
            transfer_us = transfer_elapsed.as_micros(),
            total_us = total_elapsed.as_micros(),
        );
        Ok(ChunkWriteResult {
            location: pb_chunk_to_location(&chunk),
            is_local: false,
            connect_elapsed,
            transfer_elapsed,
            total_elapsed,
        })
    }

    async fn put_distributed_bytes_impl(
        &self,
        key: InternalKey,
        data: Bytes,
        meta: BlockMeta,
        if_absent: bool,
    ) -> Result<bool, Status> {
        let operation_start = Instant::now();
        let total = data.len();
        let chunk_size = self.ctx.storage.striping_chunk_size().max(1) as usize;
        let stripe_count = total.div_ceil(chunk_size);
        let prepare_start = Instant::now();
        let prepared_meta = self
            .ctx
            .storage
            .prepare_write_meta(&key, meta, total as u64)
            .map_err(Status::from)?;
        let prepare_elapsed = prepare_start.elapsed();
        let descriptor =
            self.make_distributed_descriptor(&key, &prepared_meta, stripe_count, chunk_size as u64);

        let mut tasks = Vec::with_capacity(stripe_count);
        for stripe_index in 0..stripe_count {
            let start = stripe_index * chunk_size;
            let end = (start + chunk_size).min(total);
            let chunk = data.slice(start..end);
            let node = select_data_node(&self.ctx, &key, stripe_index);
            tasks.push(Self::put_chunk_on_node(
                self.ctx.clone(),
                node,
                key.clone(),
                descriptor.clone(),
                stripe_index,
                start as u64,
                chunk_size as u64,
                total as u64,
                chunk,
            ));
        }
        let stripe_write_start = Instant::now();
        let mut locations = Vec::with_capacity(stripe_count);
        let mut local_stripes = 0usize;
        let mut remote_stripes = 0usize;
        let mut slowest_stripe = Duration::ZERO;
        let mut remote_connect_total = Duration::ZERO;
        let mut remote_transfer_total = Duration::ZERO;
        for result in join_all(tasks).await {
            let result = result?;
            if result.is_local {
                local_stripes += 1;
            } else {
                remote_stripes += 1;
                remote_connect_total += result.connect_elapsed;
                remote_transfer_total += result.transfer_elapsed;
            }
            slowest_stripe = slowest_stripe.max(result.total_elapsed);
            locations.push(result.location);
        }
        let stripe_write_elapsed = stripe_write_start.elapsed();
        locations.sort_by_key(|loc| loc.stripe_index);

        let rollback_chunks = locations
            .iter()
            .map(|loc| chunk_location_to_pb(&key, loc))
            .collect::<Vec<_>>();
        if self.ctx.storage.verify_stripe_checksums()
            && locations.iter().any(|location| location.checksum.is_empty())
        {
            for chunk in rollback_chunks {
                let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
            }
            return Err(Status::failed_precondition(
                "stripe integrity is enabled but a data node did not return a checksum",
            ));
        }
        let chunk_devices = locations.iter().map(|loc| loc.device_id).collect();
        let chunk_paths = locations
            .iter()
            .map(|loc| loc.storage_handle.clone())
            .collect();
        let chunk_checksums = locations
            .iter()
            .map(|loc| loc.checksum.clone())
            .collect();
        let mut committed = prepared_meta;
        committed.size = total as u64;
        committed.file_path = String::new();
        committed.device_id = locations.first().map(|loc| loc.device_id).unwrap_or(0);
        committed.striping = Some(StripingInfo {
            chunk_size: chunk_size as u64,
            chunk_devices,
            chunk_paths,
            total_size: total as u64,
            chunk_locations: locations,
            chunk_checksums,
        });
        self.ctx.memory.invalidate(&key);
        let metadata = self.ctx.metadata.clone();
        let str_key = key.to_string_key();
        let committed_meta = committed.clone();
        let metadata_start = Instant::now();
        let committed_result = tokio::task::spawn_blocking(move || {
            if if_absent {
                metadata.put_block_if_absent(&str_key, &committed_meta)
            } else {
                metadata.put_block(&str_key, &committed_meta).map(|_| true)
            }
        })
        .await;
        let committed = match committed_result {
            Ok(Ok(committed)) => committed,
            Ok(Err(e)) => {
                for chunk in rollback_chunks {
                    let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
                }
                return Err(Status::from(e));
            }
            Err(e) => {
                for chunk in rollback_chunks {
                    let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
                }
                return Err(Status::internal(e.to_string()));
            }
        };
        let metadata_elapsed = metadata_start.elapsed();
        if !committed {
            for chunk in rollback_chunks {
                let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
            }
            return Ok(false);
        }
        tracing::debug!(
            event = "grpc_distributed_put",
            status = "ok",
            bytes = total,
            stripe_count,
            local_stripes,
            remote_stripes,
            prepare_us = prepare_elapsed.as_micros(),
            stripe_write_us = stripe_write_elapsed.as_micros(),
            slowest_stripe_us = slowest_stripe.as_micros(),
            remote_connect_total_us = remote_connect_total.as_micros(),
            remote_transfer_total_us = remote_transfer_total.as_micros(),
            metadata_us = metadata_elapsed.as_micros(),
            total_us = operation_start.elapsed().as_micros(),
        );
        Ok(true)
    }

    async fn put_distributed_bytes(
        &self,
        key: InternalKey,
        data: Bytes,
        meta: BlockMeta,
    ) -> Result<(), Status> {
        let write_lock = self.key_write_lock(&key);
        let _guard = write_lock.lock().await;
        self.put_distributed_bytes_impl(key, data, meta, false)
            .await
            .map(|_| ())
    }

    async fn put_distributed_bytes_if_absent(
        &self,
        key: InternalKey,
        data: Bytes,
        meta: BlockMeta,
    ) -> Result<bool, Status> {
        let write_lock = self.key_write_lock(&key);
        let _guard = write_lock.lock().await;

        let metadata = self.ctx.metadata.clone();
        let str_key = key.to_string_key();
        let existing = tokio::task::spawn_blocking(move || metadata.get_block(&str_key))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        if let Some(existing) = existing {
            if !existing.is_expired() {
                return Ok(false);
            }
            self.purge_expired_object_locked(&key, &existing).await?;
        }

        self.put_distributed_bytes_impl(key, data, meta, true).await
    }

    /// Streaming stripe write task: local stripes go through the vectored (writev) path with the
    /// segments as-is — no per-stripe concat; remote stripes flatten only their own 64MB before
    /// the wire call (unavoidable: protobuf field is a single `bytes`).
    async fn put_stripe_segments_on_node(
        ctx: Arc<KVServiceContext>,
        node: DataNode,
        key: InternalKey,
        descriptor: pb::ObjectDescriptor,
        stripe_index: usize,
        offset: u64,
        chunk_size: u64,
        total_size: u64,
        segments: Vec<Bytes>,
    ) -> Result<ChunkWriteResult, Status> {
        if is_local_node(&ctx, &node) {
            let chunk_start = Instant::now();
            let data_len: u64 = segments.iter().map(|s| s.len() as u64).sum();
            let device_stripe_index = local_device_stripe_index(&ctx, &key, stripe_index)
                .ok_or_else(|| Status::failed_precondition("stripe assigned to a different data node"))?;
            let key_for_write = key.clone();
            let storage = ctx.storage.clone();
            let generation = descriptor.object_generation;
            let layout_version = descriptor.layout_version;
            let (device_id, storage_handle, checksum) = tokio::task::spawn_blocking(move || {
                storage.put_placement_chunk_segments(
                    &key_for_write,
                    stripe_index,
                    device_stripe_index,
                    generation,
                    layout_version,
                    segments,
                )
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
            let total_elapsed = chunk_start.elapsed();
            tracing::debug!(
                event = "grpc_placement_chunk_write",
                status = "ok",
                target = "local_streamed",
                stripe_index,
                bytes = data_len,
                storage_us = total_elapsed.as_micros(),
            );
            return Ok(ChunkWriteResult {
                location: ChunkLocation {
                    stripe_index: stripe_index as u32,
                    node_id: node.node_id,
                    grpc_endpoint: node.grpc_endpoint,
                    rdma_endpoint: node.rdma_endpoint,
                    device_id,
                    storage_handle,
                    offset,
                    length: data_len,
                    checksum,
                },
                is_local: true,
                connect_elapsed: Duration::ZERO,
                transfer_elapsed: Duration::ZERO,
                total_elapsed,
            });
        }
        // Remote node: the wire format needs one contiguous Bytes — flatten just this stripe.
        let data = Self::flatten_segments(segments);
        Self::put_chunk_on_node(
            ctx,
            node,
            key,
            descriptor,
            stripe_index,
            offset,
            chunk_size,
            total_size,
            data,
        )
        .await
    }

    /// Distributed streaming PUT pipeline: receive gRPC chunks, aggregate on stripe boundaries,
    /// and dispatch each stripe's disk write as soon as it fills — receive / stripe aggregation /
    /// dual-NVMe I/O all overlap. Memory bound is `MAX_INFLIGHT_STRIPES` stripes, not the full
    /// object. Metadata is committed once after every stripe succeeds; on failure all written
    /// stripes are rolled back (same semantics as `put_distributed_bytes_impl`).
    #[allow(clippy::too_many_arguments)]
    async fn put_distributed_stream_impl(
        &self,
        stream: &mut tonic::Streaming<pb::PutChunk>,
        key: InternalKey,
        meta: BlockMeta,
        declared_total: usize,
        first_data: Bytes,
        first_is_last: bool,
        if_absent: bool,
    ) -> Result<(bool, StreamPutStats), Status> {
        const MAX_INFLIGHT_STRIPES: usize = 4;

        let operation_start = Instant::now();
        let chunk_size = self.ctx.storage.striping_chunk_size().max(1) as usize;
        let stripe_count = declared_total.div_ceil(chunk_size);

        let prepare_start = Instant::now();
        let prepared_meta = self
            .ctx
            .storage
            .prepare_write_meta(&key, meta, declared_total as u64)
            .map_err(Status::from)?;
        let prepare_elapsed = prepare_start.elapsed();
        let descriptor = self.make_distributed_descriptor(
            &key,
            &prepared_meta,
            stripe_count,
            chunk_size as u64,
        );

        let mut inflight: JoinSet<Result<ChunkWriteResult, Status>> = JoinSet::new();
        let mut locations: Vec<ChunkLocation> = Vec::with_capacity(stripe_count);
        let mut stats = StreamPutStats::default();
        let mut first_err: Option<Status> = None;

        let mut cur_segments: Vec<Bytes> = Vec::new();
        let mut cur_filled: usize = 0;
        let mut next_stripe: usize = 0;
        let mut received_total: usize = 0;

        // Drain one completed stripe write, folding its outcome into locations / first_err.
        fn absorb(
            joined: Result<Result<ChunkWriteResult, Status>, tokio::task::JoinError>,
            locations: &mut Vec<ChunkLocation>,
            stats: &mut StreamPutStats,
            first_err: &mut Option<Status>,
        ) {
            match joined {
                Ok(Ok(result)) => {
                    if result.is_local {
                        stats.local_stripes += 1;
                    } else {
                        stats.remote_stripes += 1;
                    }
                    stats.slowest_stripe = stats.slowest_stripe.max(result.total_elapsed);
                    locations.push(result.location);
                }
                Ok(Err(status)) => {
                    if first_err.is_none() {
                        *first_err = Some(status);
                    }
                }
                Err(join_err) => {
                    if first_err.is_none() {
                        *first_err = Some(Status::internal(join_err.to_string()));
                    }
                }
            }
        }

        macro_rules! dispatch_stripe {
            ($segments:expr, $len:expr) => {{
                let stripe_index = next_stripe;
                #[allow(unused_assignments)]
                {
                    next_stripe += 1;
                }
                let node = select_data_node(&self.ctx, &key, stripe_index);
                let offset = (stripe_index * chunk_size) as u64;
                // Backpressure: keep at most MAX_INFLIGHT_STRIPES buffered + writing.
                let wait_start = Instant::now();
                while inflight.len() >= MAX_INFLIGHT_STRIPES {
                    if let Some(joined) = inflight.join_next().await {
                        absorb(joined, &mut locations, &mut stats, &mut first_err);
                    } else {
                        break;
                    }
                }
                stats.backpressure_wait += wait_start.elapsed();
                if first_err.is_none() {
                    inflight.spawn(Self::put_stripe_segments_on_node(
                        self.ctx.clone(),
                        node,
                        key.clone(),
                        descriptor.clone(),
                        stripe_index,
                        offset,
                        chunk_size as u64,
                        declared_total as u64,
                        $segments,
                    ));
                }
                let _ = $len;
            }};
        }

        // Feed one gRPC chunk into the stripe aggregator, dispatching every stripe it completes.
        macro_rules! feed {
            ($data:expr) => {{
                let mut data: Bytes = $data;
                received_total += data.len();
                while !data.is_empty() {
                    let need = chunk_size - cur_filled;
                    if data.len() < need {
                        cur_filled += data.len();
                        cur_segments.push(data);
                        break;
                    }
                    let head = data.split_to(need);
                    cur_segments.push(head);
                    let segments = std::mem::take(&mut cur_segments);
                    cur_filled = 0;
                    dispatch_stripe!(segments, chunk_size);
                }
            }};
        }

        let receive_start = Instant::now();
        feed!(first_data);
        let mut saw_last = first_is_last;
        while !saw_last && first_err.is_none() {
            match stream.next().await {
                Some(chunk) => {
                    let chunk = chunk?;
                    saw_last = chunk.is_last;
                    feed!(chunk.data);
                }
                None => break,
            }
        }
        // Flush the trailing partial stripe.
        if first_err.is_none() && !cur_segments.is_empty() {
            let segments = std::mem::take(&mut cur_segments);
            let len = cur_filled;
            dispatch_stripe!(segments, len);
        }
        stats.receive_elapsed = receive_start.elapsed();

        // Drain remaining in-flight stripe writes.
        let drain_start = Instant::now();
        while let Some(joined) = inflight.join_next().await {
            absorb(joined, &mut locations, &mut stats, &mut first_err);
        }
        stats.drain_wait = drain_start.elapsed();

        if first_err.is_none() && received_total != declared_total {
            first_err = Some(Status::invalid_argument(format!(
                "stream length mismatch: declared {declared_total} received {received_total}"
            )));
        }

        if let Some(status) = first_err {
            // Roll back every stripe that made it to disk.
            for loc in &locations {
                let chunk = chunk_location_to_pb(&key, loc);
                let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
            }
            return Err(status);
        }

        locations.sort_by_key(|loc| loc.stripe_index);
        let rollback_chunks = locations
            .iter()
            .map(|loc| chunk_location_to_pb(&key, loc))
            .collect::<Vec<_>>();
        let chunk_devices = locations.iter().map(|loc| loc.device_id).collect();
        let chunk_paths = locations
            .iter()
            .map(|loc| loc.storage_handle.clone())
            .collect();
        let chunk_checksums = locations
            .iter()
            .map(|loc| loc.checksum.clone())
            .collect();
        let mut committed_meta = prepared_meta;
        committed_meta.size = declared_total as u64;
        committed_meta.file_path = String::new();
        committed_meta.device_id = locations.first().map(|loc| loc.device_id).unwrap_or(0);
        committed_meta.striping = Some(StripingInfo {
            chunk_size: chunk_size as u64,
            chunk_devices,
            chunk_paths,
            total_size: declared_total as u64,
            chunk_locations: locations,
            chunk_checksums,
        });
        self.ctx.memory.invalidate(&key);
        let metadata = self.ctx.metadata.clone();
        let str_key = key.to_string_key();
        let meta_for_commit = committed_meta.clone();
        let metadata_start = Instant::now();
        let committed = tokio::task::spawn_blocking(move || {
            if if_absent {
                metadata.put_block_if_absent(&str_key, &meta_for_commit)
            } else {
                metadata.put_block(&str_key, &meta_for_commit).map(|_| true)
            }
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(Status::from)?;
        stats.metadata_elapsed = metadata_start.elapsed();
        if !committed {
            for chunk in rollback_chunks {
                let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
            }
            return Ok((false, stats));
        }
        tracing::debug!(
            event = "grpc_distributed_put",
            status = "ok",
            mode = "streamed",
            bytes = declared_total,
            stripe_count,
            local_stripes = stats.local_stripes,
            remote_stripes = stats.remote_stripes,
            prepare_us = prepare_elapsed.as_micros(),
            receive_us = stats.receive_elapsed.as_micros(),
            backpressure_wait_us = stats.backpressure_wait.as_micros(),
            drain_wait_us = stats.drain_wait.as_micros(),
            slowest_stripe_us = stats.slowest_stripe.as_micros(),
            metadata_us = stats.metadata_elapsed.as_micros(),
            total_us = operation_start.elapsed().as_micros(),
        );
        Ok((true, stats))
    }

    fn placement_has_remote_chunks(&self, placement: &pb::PlacementDescriptor) -> bool {
        placement.chunks.iter().any(|chunk| {
            let node = DataNode {
                node_id: chunk.node_id.clone(),
                grpc_endpoint: chunk.grpc_endpoint.clone(),
                rdma_endpoint: chunk.rdma_endpoint.clone(),
            };
            !is_local_node(&self.ctx, &node)
        })
    }

    async fn read_chunk_from_placement(
        ctx: Arc<KVServiceContext>,
        descriptor: pb::ObjectDescriptor,
        placement: pb::PlacementDescriptor,
        chunk: pb::PlacementChunk,
    ) -> Result<(u32, Bytes), Status> {
        let node = DataNode {
            node_id: chunk.node_id.clone(),
            grpc_endpoint: chunk.grpc_endpoint.clone(),
            rdma_endpoint: chunk.rdma_endpoint.clone(),
        };
        if is_local_node(&ctx, &node) {
            let storage = ctx.storage.clone();
            let handle = chunk.storage_handle.clone();
            let expected_len = chunk.length;
            let expected_checksum = descriptor.is_striped.then(|| chunk.checksum.clone());
            let stripe_index = chunk.stripe_index;
            let data = tokio::task::spawn_blocking(move || {
                storage.read_placement_chunk(&handle, expected_len, expected_checksum.as_deref())
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?
            .ok_or_else(|| Status::not_found("placement chunk not found"))?;
            return Ok((stripe_index, data));
        }

        let mut client = connect_data_node(&node.grpc_endpoint)
            .await
            .map_err(|status| {
                Status::unavailable(format!(
                    "connect data node {}: {}",
                    node.node_id,
                    status.message()
                ))
            })?;
        let mut stream = client
            .read_placement_chunk(pb::ReadPlacementChunkRequest {
                descriptor: Some(descriptor),
                chunk: Some(chunk.clone()),
                placement: Some(placement),
            })
            .await
            .map_err(|e| Status::unavailable(format!("read chunk from {}: {}", node.node_id, e)))?
            .into_inner();
        let mut parts = Vec::new();
        while let Some(part) = stream
            .message()
            .await
            .map_err(|e| Status::unavailable(format!("read chunk stream: {}", e)))?
        {
            parts.push(part.data);
            if part.is_last {
                break;
            }
        }
        Ok((chunk.stripe_index, Self::flatten_segments(parts)))
    }

    async fn read_chunks_by_placement(
        &self,
        descriptor: pb::ObjectDescriptor,
        placement: pb::PlacementDescriptor,
    ) -> Result<Vec<Bytes>, Status> {
        let mut validation_placement = placement.clone();
        validation_placement.chunks.clear();
        let mut tasks = Vec::with_capacity(placement.chunks.len());
        for chunk in &placement.chunks {
            tasks.push(Self::read_chunk_from_placement(
                self.ctx.clone(),
                descriptor.clone(),
                validation_placement.clone(),
                chunk.clone(),
            ));
        }
        let mut indexed = Vec::with_capacity(tasks.len());
        for result in join_all(tasks).await {
            indexed.push(result?);
        }
        indexed.sort_by_key(|(idx, _)| *idx);
        Ok(indexed.into_iter().map(|(_, data)| data).collect())
    }

    async fn delete_chunk_from_placement(
        ctx: Arc<KVServiceContext>,
        chunk: pb::PlacementChunk,
    ) -> Result<(), Status> {
        let node = DataNode {
            node_id: chunk.node_id.clone(),
            grpc_endpoint: chunk.grpc_endpoint.clone(),
            rdma_endpoint: chunk.rdma_endpoint.clone(),
        };
        if is_local_node(&ctx, &node) {
            let storage = ctx.storage.clone();
            let handle = chunk.storage_handle.clone();
            tokio::task::spawn_blocking(move || storage.delete_placement_chunk(&handle))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            return Ok(());
        }

        let mut client = connect_data_node(&node.grpc_endpoint)
            .await
            .map_err(|status| {
                Status::unavailable(format!(
                    "connect data node {}: {}",
                    node.node_id,
                    status.message()
                ))
            })?;
        client
            .delete_placement_chunk(pb::DeletePlacementChunkRequest { chunk: Some(chunk) })
            .await
            .map_err(|e| {
                Status::unavailable(format!("delete chunk from {}: {}", node.node_id, e))
            })?;
        Ok(())
    }

    async fn delete_distributed_chunks(
        &self,
        placement: pb::PlacementDescriptor,
    ) -> Result<(), Status> {
        let mut tasks = Vec::with_capacity(placement.chunks.len());
        for chunk in placement.chunks {
            tasks.push(Self::delete_chunk_from_placement(self.ctx.clone(), chunk));
        }
        for result in join_all(tasks).await {
            result?;
        }
        Ok(())
    }
}

fn pb_key_to_internal(k: &pb::ObjectKey) -> InternalKey {
    InternalKey {
        namespace: k.namespace.clone(),
        object_key: k.object_key.clone(),
    }
}

fn internal_key_to_pb(k: &InternalKey) -> pb::ObjectKey {
    pb::ObjectKey {
        namespace: k.namespace.clone(),
        object_key: k.object_key.clone(),
    }
}

fn meta_from_pb(m: Option<&pb::KvMetadata>, options: Option<&pb::PutOptions>) -> BlockMeta {
    let now = chrono::Utc::now().timestamp();
    let ttl_seconds = options
        .map(|opts| opts.ttl_seconds.max(0))
        .unwrap_or_default();
    match m {
        Some(m) => BlockMeta {
            device_id: 0,
            file_path: String::new(),
            size: 0,
            object_handle: String::new(),
            object_generation: 1,
            content_etag: String::new(),
            layout_version: 1,
            created_at: if m.created_at > 0 { m.created_at } else { now },
            last_accessed_at: now,
            ttl_seconds,
            num_tokens: m.num_tokens,
            num_layers: m.num_layers,
            dtype: m.dtype.clone(),
            compressed: m.compressed,
            striping: None,
        },
        None => BlockMeta {
            device_id: 0,
            file_path: String::new(),
            size: 0,
            object_handle: String::new(),
            object_generation: 1,
            content_etag: String::new(),
            layout_version: 1,
            created_at: now,
            last_accessed_at: now,
            ttl_seconds,
            num_tokens: 0,
            num_layers: 0,
            dtype: "bfloat16".to_string(),
            compressed: false,
            striping: None,
        },
    }
}

fn meta_to_pb(m: &BlockMeta) -> pb::KvMetadata {
    pb::KvMetadata {
        num_tokens: m.num_tokens,
        num_layers: m.num_layers,
        dtype: m.dtype.clone(),
        shape: vec![],
        compressed: m.compressed,
        compression_level: 0,
        created_at: m.created_at,
        last_accessed_at: m.last_accessed_at,
    }
}

fn put_options_if_not_exists(options: Option<&pb::PutOptions>) -> bool {
    options.map(|opts| opts.if_not_exists).unwrap_or(false)
}

fn grpc_uri(endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("http://{}", endpoint)
    }
}

fn local_node(ctx: &KVServiceContext) -> DataNode {
    let node_id = std::env::var("CS_NODE_ID")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| {
            (!ctx.config.cluster.node_id.is_empty()).then(|| ctx.config.cluster.node_id.clone())
        })
        .unwrap_or_else(|| "local".to_string());
    let grpc_endpoint = std::env::var("CS_GRPC_ADVERTISE")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| {
            (!ctx.config.cluster.grpc_advertise.is_empty())
                .then(|| ctx.config.cluster.grpc_advertise.clone())
        })
        .unwrap_or_else(|| ctx.config.api.listen.clone());
    let rdma_endpoint = std::env::var("CS_RDMA_ADVERTISE")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| {
            (!ctx.config.cluster.rdma_advertise.is_empty())
                .then(|| ctx.config.cluster.rdma_advertise.clone())
        })
        .unwrap_or_default();
    DataNode {
        node_id,
        grpc_endpoint,
        rdma_endpoint,
    }
}

fn configured_data_nodes(ctx: &KVServiceContext) -> Vec<DataNode> {
    if ctx.config.cluster.data_nodes.is_empty() {
        return vec![local_node(ctx)];
    }
    ctx.config
        .cluster
        .data_nodes
        .iter()
        .map(|n: &ClusterNodeConfig| DataNode {
            node_id: if n.node_id.is_empty() {
                n.grpc_endpoint.clone()
            } else {
                n.node_id.clone()
            },
            grpc_endpoint: n.grpc_endpoint.clone(),
            rdma_endpoint: n.rdma_endpoint.clone(),
        })
        .collect()
}

fn is_local_node(ctx: &KVServiceContext, node: &DataNode) -> bool {
    let local = local_node(ctx);
    node.node_id == local.node_id || node.grpc_endpoint == local.grpc_endpoint
}

fn distributed_placement_enabled(ctx: &KVServiceContext, len: usize) -> bool {
    let threshold = ctx.storage.striping_threshold();
    threshold > 0 && len as u64 > threshold && configured_data_nodes(ctx).len() > 1
}

fn select_data_node(ctx: &KVServiceContext, key: &InternalKey, stripe_index: usize) -> DataNode {
    let nodes = configured_data_nodes(ctx);
    let base = (hash64(key.to_string_key().as_bytes()) as usize) % nodes.len();
    nodes[(base + stripe_index) % nodes.len()].clone()
}

/// Return a stripe's zero-based ordinal among stripes assigned to this data node.
///
/// Data-node placement alternates globally across nodes. Selecting a local device
/// from the global stripe index would pin each node to one device whenever the
/// node and device counts share a factor. The local ordinal preserves the same
/// deterministic routing while rotating disks independently on every node, and
/// duplicate local data-node entries retain the same rotation as a single entry.
fn local_device_stripe_index(
    ctx: &KVServiceContext,
    key: &InternalKey,
    stripe_index: usize,
) -> Option<usize> {
    let nodes = configured_data_nodes(ctx);
    let local_count = nodes.iter().filter(|node| is_local_node(ctx, node)).count();
    if local_count == 0 {
        return None;
    }
    let base = (hash64(key.to_string_key().as_bytes()) as usize) % nodes.len();
    let slot = (base + stripe_index) % nodes.len();
    if !is_local_node(ctx, &nodes[slot]) {
        return None;
    }

    // Count local assignments before this stripe by whole node-list cycles, then
    // count the remaining partial cycle. This also handles duplicate local entries.
    let full_cycles = stripe_index / nodes.len();
    let remainder = stripe_index % nodes.len();
    let local_in_partial_cycle = (0..=remainder)
        .filter(|offset| is_local_node(ctx, &nodes[(base + offset) % nodes.len()]))
        .count();
    Some(full_cycles * local_count + local_in_partial_cycle - 1)
}

fn placement_policy_id(ctx: &KVServiceContext) -> String {
    format!("{}_v1", ctx.config.router.strategy)
}

fn placement_epoch(ctx: &KVServiceContext) -> u64 {
    let mut seed = format!(
        "policy={}|striping_threshold={}|striping_chunk_size={}|devices={}",
        placement_policy_id(ctx),
        ctx.storage.striping_threshold(),
        ctx.storage.striping_chunk_size(),
        ctx.storage.router().num_devices()
    );
    for (idx, node) in configured_data_nodes(ctx).iter().enumerate() {
        seed.push_str(&format!(
            "|node{}={}:{}:{}",
            idx, node.node_id, node.grpc_endpoint, node.rdma_endpoint
        ));
    }
    hash64(seed.as_bytes()).max(1)
}

fn chunk_location_to_pb(key: &InternalKey, loc: &ChunkLocation) -> pb::PlacementChunk {
    let _ = key;
    pb::PlacementChunk {
        stripe_index: loc.stripe_index,
        node_id: loc.node_id.clone(),
        grpc_endpoint: loc.grpc_endpoint.clone(),
        rdma_endpoint: loc.rdma_endpoint.clone(),
        device_id: loc.device_id,
        storage_handle: loc.storage_handle.clone(),
        offset: loc.offset,
        length: loc.length,
        checksum: loc.checksum.clone(),
    }
}

fn pb_chunk_to_location(chunk: &pb::PlacementChunk) -> ChunkLocation {
    ChunkLocation {
        stripe_index: chunk.stripe_index,
        node_id: chunk.node_id.clone(),
        grpc_endpoint: chunk.grpc_endpoint.clone(),
        rdma_endpoint: chunk.rdma_endpoint.clone(),
        device_id: chunk.device_id,
        storage_handle: chunk.storage_handle.clone(),
        offset: chunk.offset,
        length: chunk.length,
        checksum: chunk.checksum.clone(),
    }
}

fn object_handle(key: &InternalKey, meta: &BlockMeta) -> String {
    if !meta.object_handle.is_empty() {
        return meta.object_handle.clone();
    }
    format!(
        "v1:{}:g{}:l{}",
        key.to_string_key(),
        meta.object_generation,
        meta.layout_version
    )
}

fn descriptor_from_meta(key: &InternalKey, meta: &BlockMeta) -> pb::ObjectDescriptor {
    let (is_striped, stripe_count, chunk_size) = match &meta.striping {
        Some(stripe) => (true, stripe.chunk_paths.len() as u32, stripe.chunk_size),
        None => (false, 0, 0),
    };
    pb::ObjectDescriptor {
        key: Some(internal_key_to_pb(key)),
        object_handle: object_handle(key, meta),
        object_generation: meta.object_generation,
        content_etag: meta.content_etag.clone(),
        layout_version: meta.layout_version,
        size: meta.size,
        is_striped,
        stripe_count,
        chunk_size,
    }
}

fn placement_from_meta(
    ctx: &KVServiceContext,
    key: &InternalKey,
    meta: &BlockMeta,
) -> pb::PlacementDescriptor {
    let local = local_node(ctx);
    let placement_epoch = placement_epoch(ctx);
    let placement_policy_id = placement_policy_id(ctx);

    let chunks = match &meta.striping {
        Some(stripe) if stripe.chunk_locations.len() == stripe.chunk_paths.len() => stripe
            .chunk_locations
            .iter()
            .map(|loc| chunk_location_to_pb(key, loc))
            .collect(),
        Some(stripe) => stripe
            .chunk_paths
            .iter()
            .enumerate()
            .map(|(idx, path)| {
                let offset = idx as u64 * stripe.chunk_size;
                let length = stripe
                    .total_size
                    .saturating_sub(offset)
                    .min(stripe.chunk_size);
                pb::PlacementChunk {
                    stripe_index: idx as u32,
                    node_id: local.node_id.clone(),
                    grpc_endpoint: local.grpc_endpoint.clone(),
                    rdma_endpoint: local.rdma_endpoint.clone(),
                    device_id: stripe.chunk_devices.get(idx).copied().unwrap_or(0),
                    storage_handle: path.clone(),
                    offset,
                    length,
                    checksum: stripe.chunk_checksums.get(idx).cloned().unwrap_or_default(),
                }
            })
            .collect(),
        None => vec![pb::PlacementChunk {
            stripe_index: 0,
            node_id: local.node_id.clone(),
            grpc_endpoint: local.grpc_endpoint.clone(),
            rdma_endpoint: local.rdma_endpoint.clone(),
            device_id: meta.device_id,
            storage_handle: meta.file_path.clone(),
            offset: 0,
            length: meta.size,
            checksum: String::new(),
        }],
    };

    let mut hash_seed = format!(
        "{}|g{}|l{}|epoch{}|{}|{}",
        key.to_string_key(),
        meta.object_generation,
        meta.layout_version,
        placement_epoch,
        placement_policy_id,
        chunks.len()
    );
    for chunk in &chunks {
        hash_seed.push_str(&format!(
            "|{}:{}:{}:{}:{}",
            chunk.stripe_index, chunk.node_id, chunk.device_id, chunk.offset, chunk.storage_handle
        ));
    }

    pb::PlacementDescriptor {
        key: Some(internal_key_to_pb(key)),
        placement_epoch,
        placement_policy_id,
        layout_hash: format!("{:016x}", hash64(hash_seed.as_bytes())),
        primary_node_id: local.node_id,
        primary_grpc_endpoint: local.grpc_endpoint,
        primary_rdma_endpoint: local.rdma_endpoint,
        chunks,
    }
}

fn key_from_descriptor(desc: &pb::ObjectDescriptor) -> Result<InternalKey, Status> {
    let key = desc
        .key
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("descriptor missing key"))?;
    Ok(pb_key_to_internal(key))
}

fn validate_descriptor(desc: &pb::ObjectDescriptor, meta: &BlockMeta) -> Result<(), Status> {
    if desc.object_generation != meta.object_generation
        || desc.content_etag != meta.content_etag
        || desc.layout_version != meta.layout_version
        || desc.size != meta.size
        || desc.object_handle != meta.object_handle
    {
        return Err(Status::failed_precondition("stale descriptor"));
    }
    Ok(())
}

fn validate_placement_descriptor(
    ctx: &KVServiceContext,
    key: &InternalKey,
    meta: &BlockMeta,
    placement: Option<&pb::PlacementDescriptor>,
) -> Result<(), Status> {
    let Some(placement) = placement else {
        return Ok(());
    };
    let Some(placement_key) = placement.key.as_ref() else {
        return Err(Status::invalid_argument("placement descriptor missing key"));
    };
    if pb_key_to_internal(placement_key) != *key {
        return Err(Status::failed_precondition("stale placement descriptor"));
    }

    let expected = placement_from_meta(ctx, key, meta);
    if placement.placement_epoch != expected.placement_epoch
        || placement.placement_policy_id != expected.placement_policy_id
        || placement.layout_hash != expected.layout_hash
    {
        return Err(Status::failed_precondition("stale placement descriptor"));
    }
    Ok(())
}

#[tonic::async_trait]
impl pb::kv_service_server::KvService for KVServiceImpl {
    // ===== Health / Stats =====
    async fn health(
        &self,
        _req: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        Ok(Response::new(pb::HealthResponse {
            status: pb::health_response::ServingStatus::Serving as i32,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }

    async fn stats(
        &self,
        _req: Request<pb::StatsRequest>,
    ) -> Result<Response<pb::StatsResponse>, Status> {
        let (hits, misses, _evic, size) = self.ctx.memory.stats();
        Ok(Response::new(pb::StatsResponse {
            l1_cache_hits: hits as i64,
            l1_cache_misses: misses as i64,
            l1_cache_size_bytes: size as i64,
            l2_reads_total: 0,
            l2_writes_total: 0,
            l2_bytes_read: 0,
            l2_bytes_written: 0,
            metadata_entries: 0,
            devices: vec![],
        }))
    }

    // ===== Single ops =====
    async fn get(&self, req: Request<pb::GetRequest>) -> Result<Response<pb::GetResponse>, Status> {
        let start = Instant::now();
        let result = async {
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            let meta = self.metadata_get_live(&internal).await?;
            if let Some(meta) = meta.as_ref() {
                let placement = placement_from_meta(&self.ctx, &internal, meta);
                if self.placement_has_remote_chunks(&placement) {
                    let descriptor = descriptor_from_meta(&internal, meta);
                    let chunks = self.read_chunks_by_placement(descriptor, placement).await?;
                    let data = Self::flatten_segments(chunks);
                    return Ok(Response::new(pb::GetResponse {
                        data,
                        metadata: Some(meta_to_pb(meta)),
                        found: true,
                    }));
                }
            }
            let ctx = self.ctx.clone();
            let res = tokio::task::spawn_blocking(move || ctx.memory.get(&internal))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            match res {
                Some((data, meta)) => Ok(Response::new(pb::GetResponse {
                    // pb::GetResponse.data is Bytes (build.rs enables bytes(["."]))
                    data,
                    metadata: Some(meta_to_pb(&meta)),
                    found: true,
                })),
                None => Ok(Response::new(pb::GetResponse {
                    data: Bytes::new(),
                    metadata: None,
                    found: false,
                })),
            }
        }
        .await;
        let ok_status = if result.as_ref().map(|r| r.get_ref().found).unwrap_or(false) {
            "ok"
        } else {
            "not_found"
        };
        self.record_request("get", start, &result, ok_status);
        result
    }

    async fn put(&self, req: Request<pb::PutRequest>) -> Result<Response<pb::PutResponse>, Status> {
        let start = Instant::now();
        let result = async {
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            let meta = meta_from_pb(req.metadata.as_ref(), req.options.as_ref());
            let if_not_exists = put_options_if_not_exists(req.options.as_ref());
            // pb::PutRequest.data is Bytes (a buffer reference handed over by the gRPC framework, no copy)
            let data: Bytes = req.data;
            if self.should_use_distributed_placement(data.len()) {
                let inserted = if if_not_exists {
                    self.put_distributed_bytes_if_absent(internal, data, meta)
                        .await?
                } else {
                    self.put_distributed_bytes(internal, data, meta).await?;
                    true
                };
                return Ok(Response::new(pb::PutResponse {
                    success: inserted,
                    message: if inserted {
                        String::new()
                    } else {
                        "already exists".to_string()
                    },
                }));
            }
            let ctx = self.ctx.clone();
            let inserted = tokio::task::spawn_blocking(move || {
                if if_not_exists {
                    ctx.memory.put_if_absent(&internal, data, meta)
                } else {
                    ctx.memory.put(&internal, data, meta)?;
                    Ok(true)
                }
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
            Ok(Response::new(pb::PutResponse {
                success: inserted,
                message: if inserted {
                    String::new()
                } else {
                    "already exists".to_string()
                },
            }))
        }
        .await;
        self.record_request("put", start, &result, "ok");
        result
    }

    async fn delete(
        &self,
        req: Request<pb::DeleteRequest>,
    ) -> Result<Response<pb::DeleteResponse>, Status> {
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let internal = pb_key_to_internal(&key);
        let str_key = internal.to_string_key();
        let meta_ctx = self.ctx.clone();
        let meta = tokio::task::spawn_blocking(move || meta_ctx.metadata.get_block(&str_key))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        if let Some(meta) = meta.as_ref() {
            let placement = placement_from_meta(&self.ctx, &internal, meta);
            if self.placement_has_remote_chunks(&placement) {
                self.delete_distributed_chunks(placement).await?;
                self.ctx.memory.invalidate(&internal);
                let metadata = self.ctx.metadata.clone();
                let str_key = internal.to_string_key();
                let expected = meta.clone();
                tokio::task::spawn_blocking(move || {
                    metadata.delete_block_if_matches(&str_key, &expected)
                })
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
                return Ok(Response::new(pb::DeleteResponse { success: true }));
            }
        }
        let ctx = self.ctx.clone();
        let ok = tokio::task::spawn_blocking(move || ctx.memory.delete(&internal))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        Ok(Response::new(pb::DeleteResponse { success: ok }))
    }

    async fn exists(
        &self,
        req: Request<pb::ExistsRequest>,
    ) -> Result<Response<pb::ExistsResponse>, Status> {
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let internal = pb_key_to_internal(&key);
        let ctx = self.ctx.clone();
        let ok = tokio::task::spawn_blocking(move || ctx.memory.exists(&internal))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        Ok(Response::new(pb::ExistsResponse { exists: ok }))
    }

    async fn lookup_object(
        &self,
        req: Request<pb::LookupObjectRequest>,
    ) -> Result<Response<pb::LookupObjectResponse>, Status> {
        let start = Instant::now();
        let result = async {
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            let meta = self.metadata_get_live(&internal).await?;
            let descriptor = meta.as_ref().map(|m| descriptor_from_meta(&internal, m));
            let placement = meta
                .as_ref()
                .map(|m| placement_from_meta(&self.ctx, &internal, m));
            Ok(Response::new(pb::LookupObjectResponse {
                found: descriptor.is_some(),
                descriptor,
                placement,
            }))
        }
        .await;
        let ok_status = if result.as_ref().map(|r| r.get_ref().found).unwrap_or(false) {
            "ok"
        } else {
            "not_found"
        };
        self.record_request("lookup_object", start, &result, ok_status);
        result
    }

    async fn read_by_descriptor(
        &self,
        req: Request<pb::ReadByDescriptorRequest>,
    ) -> Result<Response<pb::DataReadResponse>, Status> {
        let start = Instant::now();
        let result = async {
            let req = req.into_inner();
            let requested_placement = req.placement;
            let descriptor = req
                .descriptor
                .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
            let internal = key_from_descriptor(&descriptor)?;
            let active_meta = self.metadata_get_live(&internal).await?;
            let Some(active_meta) = active_meta else {
                return Ok(Response::new(pb::DataReadResponse {
                    found: false,
                    data: Bytes::new(),
                    metadata: None,
                    descriptor: None,
                    placement: None,
                }));
            };
            validate_descriptor(&descriptor, &active_meta)?;
            validate_placement_descriptor(
                &self.ctx,
                &internal,
                &active_meta,
                requested_placement.as_ref(),
            )?;
            let placement = placement_from_meta(&self.ctx, &internal, &active_meta);
            if self.placement_has_remote_chunks(&placement) {
                let chunks = self
                    .read_chunks_by_placement(descriptor.clone(), placement.clone())
                    .await?;
                let data = Self::flatten_segments(chunks);
                let fresh = descriptor_from_meta(&internal, &active_meta);
                return Ok(Response::new(pb::DataReadResponse {
                    found: true,
                    data,
                    metadata: Some(meta_to_pb(&active_meta)),
                    descriptor: Some(fresh),
                    placement: Some(placement),
                }));
            }
            let layout_meta = active_meta.clone();
            let read_ctx = self.ctx.clone();
            let read_key = internal.clone();
            let res = tokio::task::spawn_blocking(
                move || -> Result<Option<(Bytes, BlockMeta)>, Status> {
                    read_ctx
                        .storage
                        .get_with_meta(&read_key, &layout_meta)
                        .map_err(Status::from)
                },
            )
            .await
            .map_err(|e| Status::internal(e.to_string()))??;
            match res {
                Some((data, _layout_meta)) => {
                    let fresh = descriptor_from_meta(&internal, &active_meta);
                    Ok(Response::new(pb::DataReadResponse {
                        found: true,
                        data,
                        metadata: Some(meta_to_pb(&active_meta)),
                        descriptor: Some(fresh),
                        placement: Some(placement),
                    }))
                }
                None => Ok(Response::new(pb::DataReadResponse {
                    found: false,
                    data: Bytes::new(),
                    metadata: None,
                    descriptor: None,
                    placement: None,
                })),
            }
        }
        .await;
        let ok_status = if result.as_ref().map(|r| r.get_ref().found).unwrap_or(false) {
            "ok"
        } else {
            "not_found"
        };
        self.record_request("read_by_descriptor", start, &result, ok_status);
        result
    }

    /// Multi-endpoint direct writes, phase 1: reserve a striped layout.
    ///
    /// Returns the prepared object identity (generation / handle / etag) and a
    /// placement mapping every stripe to its owning node (rdma_endpoint
    /// filled), so the client can push each node's stripes over that node's
    /// own RDMA connection. Nothing is visible until CommitDistributedPut.
    async fn prepare_distributed_put(
        &self,
        req: Request<pb::PrepareDistributedPutRequest>,
    ) -> Result<Response<pb::PrepareDistributedPutResponse>, Status> {
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let internal = pb_key_to_internal(&key);
        let size = req.size as usize;
        let reject = |message: &str| {
            Ok(Response::new(pb::PrepareDistributedPutResponse {
                accepted: false,
                message: message.to_string(),
                descriptor: None,
                placement: None,
            }))
        };
        if size == 0 {
            return reject("empty object");
        }
        if !self.should_use_distributed_placement(size) {
            return reject("below striping threshold or single data node");
        }
        let if_not_exists = put_options_if_not_exists(req.options.as_ref());
        if if_not_exists && self.metadata_get_live(&internal).await?.is_some() {
            return reject("already exists");
        }

        let meta = meta_from_pb(req.metadata.as_ref(), req.options.as_ref());
        let chunk_size = self.ctx.storage.striping_chunk_size().max(1) as usize;
        let stripe_count = size.div_ceil(chunk_size);
        let prepared = self
            .ctx
            .storage
            .prepare_write_meta(&internal, meta, size as u64)
            .map_err(Status::from)?;
        let descriptor =
            self.make_distributed_descriptor(&internal, &prepared, stripe_count, chunk_size as u64);

        // Stripe -> owning node, same deterministic routing the gRPC path uses.
        let chunks = (0..stripe_count)
            .map(|stripe_index| {
                let node = select_data_node(&self.ctx, &internal, stripe_index);
                let start = (stripe_index * chunk_size) as u64;
                let length = ((start + chunk_size as u64).min(size as u64)) - start;
                pb::PlacementChunk {
                    stripe_index: stripe_index as u32,
                    node_id: node.node_id,
                    grpc_endpoint: node.grpc_endpoint,
                    rdma_endpoint: node.rdma_endpoint,
                    device_id: 0,
                    storage_handle: String::new(),
                    offset: start,
                    length,
                    checksum: String::new(),
                }
            })
            .collect();
        let local = local_node(&self.ctx);
        let placement = pb::PlacementDescriptor {
            key: Some(key),
            placement_epoch: placement_epoch(&self.ctx),
            placement_policy_id: placement_policy_id(&self.ctx),
            layout_hash: String::new(),
            primary_node_id: local.node_id,
            primary_grpc_endpoint: local.grpc_endpoint,
            primary_rdma_endpoint: local.rdma_endpoint,
            chunks,
        };
        Ok(Response::new(pb::PrepareDistributedPutResponse {
            accepted: true,
            message: String::new(),
            descriptor: Some(descriptor),
            placement: Some(placement),
        }))
    }

    /// Multi-endpoint direct writes, phase 3: publish the assembled metadata.
    ///
    /// The client supplies the per-stripe locations returned by each node's
    /// stripe-subset PUT. The commit is atomic: on an if_not_exists race loss
    /// every written stripe is rolled back on its owning node.
    async fn commit_distributed_put(
        &self,
        req: Request<pb::CommitDistributedPutRequest>,
    ) -> Result<Response<pb::CommitDistributedPutResponse>, Status> {
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let internal = pb_key_to_internal(&key);
        let descriptor = req
            .descriptor
            .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
        if req.chunks.is_empty() {
            return Err(Status::invalid_argument("no stripe locations"));
        }
        let stripe_count = descriptor.stripe_count as usize;
        let mut chunks = req.chunks;
        chunks.sort_by_key(|c| c.stripe_index);
        if chunks.len() != stripe_count
            || chunks
                .iter()
                .enumerate()
                .any(|(i, c)| c.stripe_index as usize != i)
        {
            return Err(Status::invalid_argument(
                "stripe locations do not cover the prepared layout",
            ));
        }
        if self.ctx.storage.verify_stripe_checksums()
            && chunks.iter().any(|c| c.checksum.is_empty())
        {
            return Err(Status::failed_precondition(
                "stripe integrity is enabled but a stripe location lacks a checksum",
            ));
        }

        let if_not_exists = put_options_if_not_exists(req.options.as_ref());
        let total: u64 = chunks.iter().map(|c| c.length).sum();
        let locations: Vec<ChunkLocation> = chunks.iter().map(pb_chunk_to_location).collect();
        let rollback_chunks: Vec<pb::PlacementChunk> = chunks.clone();

        let mut committed_meta = meta_from_pb(req.metadata.as_ref(), req.options.as_ref());
        committed_meta.size = total;
        committed_meta.object_generation = descriptor.object_generation;
        committed_meta.layout_version = descriptor.layout_version;
        committed_meta.object_handle = descriptor.object_handle.clone();
        committed_meta.content_etag = descriptor.content_etag.clone();
        committed_meta.file_path = String::new();
        committed_meta.device_id = locations.first().map(|l| l.device_id).unwrap_or(0);
        committed_meta.striping = Some(StripingInfo {
            chunk_size: descriptor.chunk_size,
            chunk_devices: locations.iter().map(|l| l.device_id).collect(),
            chunk_paths: locations.iter().map(|l| l.storage_handle.clone()).collect(),
            total_size: total,
            chunk_locations: locations,
            chunk_checksums: chunks.iter().map(|c| c.checksum.clone()).collect(),
        });

        self.ctx.memory.invalidate(&internal);
        let metadata = self.ctx.metadata.clone();
        let str_key = internal.to_string_key();
        let meta_for_commit = committed_meta;
        let committed = tokio::task::spawn_blocking(move || {
            if if_not_exists {
                metadata.put_block_if_absent(&str_key, &meta_for_commit)
            } else {
                metadata.put_block(&str_key, &meta_for_commit).map(|_| true)
            }
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(Status::from)?;
        if !committed {
            for chunk in rollback_chunks {
                let mut chunk = chunk;
                chunk.node_id.clear(); // resolved via endpoint below
                let _ = Self::delete_chunk_from_placement(self.ctx.clone(), chunk).await;
            }
            return Ok(Response::new(pb::CommitDistributedPutResponse {
                committed: false,
                message: "already exists".to_string(),
            }));
        }
        Ok(Response::new(pb::CommitDistributedPutResponse {
            committed: true,
            message: String::new(),
        }))
    }

    async fn put_placement_chunk(
        &self,
        req: Request<pb::PutPlacementChunkRequest>,
    ) -> Result<Response<pb::PutPlacementChunkResponse>, Status> {
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let descriptor = req
            .descriptor
            .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
        let internal = pb_key_to_internal(&key);
        let data_len = req.data.len() as u64;
        let stripe_index_u32 = req.stripe_index;
        let stripe_index = stripe_index_u32 as usize;
        let device_stripe_index = local_device_stripe_index(&self.ctx, &internal, stripe_index)
            .ok_or_else(|| Status::failed_precondition("stripe assigned to a different data node"))?;
        let offset = stripe_index as u64 * req.chunk_size;
        let local = local_node(&self.ctx);
        let ctx = self.ctx.clone();
        let (device_id, storage_handle, checksum) = tokio::task::spawn_blocking(move || {
            ctx.storage.put_placement_chunk(
                &internal,
                stripe_index,
                device_stripe_index,
                descriptor.object_generation,
                descriptor.layout_version,
                req.data,
            )
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(Status::from)?;
        Ok(Response::new(pb::PutPlacementChunkResponse {
            success: true,
            chunk: Some(pb::PlacementChunk {
                stripe_index: stripe_index_u32,
                node_id: local.node_id,
                grpc_endpoint: local.grpc_endpoint,
                rdma_endpoint: local.rdma_endpoint,
                device_id,
                storage_handle,
                offset,
                length: data_len,
                checksum,
            }),
        }))
    }

    async fn read_placement_chunk(
        &self,
        req: Request<pb::ReadPlacementChunkRequest>,
    ) -> Result<Response<Self::ReadPlacementChunkStream>, Status> {
        let req = req.into_inner();
        let descriptor = req
            .descriptor
            .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
        let internal = key_from_descriptor(&descriptor)?;
        if let Some(placement) = req.placement.as_ref() {
            let active_meta = self
                .metadata_get_live(&internal)
                .await?
                .ok_or_else(|| Status::not_found("key not found"))?;
            validate_descriptor(&descriptor, &active_meta)?;
            validate_placement_descriptor(&self.ctx, &internal, &active_meta, Some(placement))?;
        }
        let chunk = req
            .chunk
            .ok_or_else(|| Status::invalid_argument("missing placement chunk"))?;
        let storage = self.ctx.storage.clone();
        let handle = chunk.storage_handle.clone();
        let expected_len = chunk.length;
        let expected_checksum = descriptor.is_striped.then(|| chunk.checksum.clone());
        storage
            .validate_placement_chunk_handle(
                &internal,
                chunk.stripe_index as usize,
                descriptor.object_generation,
                descriptor.layout_version,
                chunk.device_id,
                &handle,
            )
            .map_err(Status::from)?;
        let data = tokio::task::spawn_blocking(move || {
            storage.read_placement_chunk(&handle, expected_len, expected_checksum.as_deref())
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(Status::from)?
        .ok_or_else(|| Status::not_found("placement chunk not found"))?;

        const SUB_CHUNK: usize = 4 * 1024 * 1024;
        let mut chunks = Vec::new();
        let n_sub = data.len().div_ceil(SUB_CHUNK);
        for i in 0..n_sub {
            let start = i * SUB_CHUNK;
            let end = (start + SUB_CHUNK).min(data.len());
            chunks.push(pb::DataChunk {
                data: data.slice(start..end),
                offset: chunk.offset as i64 + start as i64,
                total_size: chunk.length as i64,
                is_last: false,
            });
        }
        if let Some(last) = chunks.last_mut() {
            last.is_last = true;
        }
        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Ok(Response::new(
            Box::pin(stream) as Self::ReadPlacementChunkStream
        ))
    }

    async fn delete_placement_chunk(
        &self,
        req: Request<pb::DeletePlacementChunkRequest>,
    ) -> Result<Response<pb::DeletePlacementChunkResponse>, Status> {
        let req = req.into_inner();
        let chunk = req
            .chunk
            .ok_or_else(|| Status::invalid_argument("missing placement chunk"))?;
        let storage = self.ctx.storage.clone();
        let handle = chunk.storage_handle.clone();
        let existed = tokio::task::spawn_blocking(move || storage.delete_placement_chunk(&handle))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        Ok(Response::new(pb::DeletePlacementChunkResponse {
            success: existed,
        }))
    }

    // ===== Batch =====
    async fn get_batch(
        &self,
        req: Request<pb::GetBatchRequest>,
    ) -> Result<Response<pb::GetBatchResponse>, Status> {
        let req = req.into_inner();
        let keys: Vec<InternalKey> = req.keys.iter().map(pb_key_to_internal).collect();
        let ctx = self.ctx.clone();
        let results = tokio::task::spawn_blocking(move || ctx.memory.get_batch(&keys))
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let pb_results: Vec<pb::GetResponse> = results
            .into_iter()
            .map(|r| match r {
                Ok(Some((d, m))) => pb::GetResponse {
                    data: d,
                    metadata: Some(meta_to_pb(&m)),
                    found: true,
                },
                _ => pb::GetResponse {
                    data: Bytes::new(),
                    metadata: None,
                    found: false,
                },
            })
            .collect();
        Ok(Response::new(pb::GetBatchResponse {
            results: pb_results,
        }))
    }

    async fn put_batch(
        &self,
        req: Request<pb::PutBatchRequest>,
    ) -> Result<Response<pb::PutBatchResponse>, Status> {
        let req = req.into_inner();
        // item.data is Bytes (a refcounted view over the gRPC framework buffer)
        let mut items: Vec<(InternalKey, Bytes, BlockMeta, bool)> =
            Vec::with_capacity(req.items.len());
        let mut has_if_not_exists = false;
        for item in req.items {
            let k = item
                .key
                .ok_or_else(|| Status::invalid_argument("missing key in batch item"))?;
            let m = meta_from_pb(item.metadata.as_ref(), item.options.as_ref());
            let if_not_exists = put_options_if_not_exists(item.options.as_ref());
            has_if_not_exists |= if_not_exists;
            items.push((pb_key_to_internal(&k), item.data, m, if_not_exists));
        }
        let ctx = self.ctx.clone();
        let success = if has_if_not_exists {
            tokio::task::spawn_blocking(move || {
                items
                    .into_iter()
                    .map(|(key, data, meta, if_not_exists)| {
                        if if_not_exists {
                            ctx.memory.put_if_absent(&key, data, meta)
                        } else {
                            ctx.memory.put(&key, data, meta).map(|_| true)
                        }
                        .unwrap_or(false)
                    })
                    .collect::<Vec<bool>>()
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        } else {
            let batch_items = items
                .into_iter()
                .map(|(key, data, meta, _)| (key, data, meta))
                .collect();
            let results = tokio::task::spawn_blocking(move || ctx.memory.put_batch(batch_items))
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            results.iter().map(|r| r.is_ok()).collect()
        };
        Ok(Response::new(pb::PutBatchResponse { success }))
    }

    // ===== Stream =====
    type GetStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::DataChunk, Status>> + Send + 'static>>;
    type ReadByDescriptorStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::DescriptorDataChunk, Status>> + Send + 'static>>;
    type ReadPlacementChunkStream =
        Pin<Box<dyn Stream<Item = Result<pb::DataChunk, Status>> + Send + 'static>>;

    async fn get_stream(
        &self,
        req: Request<pb::GetRequest>,
    ) -> Result<Response<Self::GetStreamStream>, Status> {
        let start = Instant::now();
        let result = async {
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            let meta = self.metadata_get_live(&internal).await?;
            if let Some(meta) = meta.as_ref() {
                let placement = placement_from_meta(&self.ctx, &internal, meta);
                if self.placement_has_remote_chunks(&placement) {
                    let descriptor = descriptor_from_meta(&internal, meta);
                    let segments = self.read_chunks_by_placement(descriptor, placement).await?;
                    let total: i64 = segments.iter().map(|s| s.len() as i64).sum();
                    const SUB_CHUNK: usize = 4 * 1024 * 1024;
                    let mut chunks: Vec<pb::DataChunk> = Vec::new();
                    let mut running_offset: i64 = 0;
                    for seg in segments {
                        let seg_len = seg.len();
                        let n_sub = seg_len.div_ceil(SUB_CHUNK);
                        for i in 0..n_sub {
                            let start = i * SUB_CHUNK;
                            let end = (start + SUB_CHUNK).min(seg_len);
                            chunks.push(pb::DataChunk {
                                data: seg.slice(start..end),
                                offset: running_offset + start as i64,
                                total_size: total,
                                is_last: false,
                            });
                        }
                        running_offset += seg_len as i64;
                    }
                    if let Some(last) = chunks.last_mut() {
                        last.is_last = true;
                    }
                    let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
                    return Ok(Response::new(Box::pin(stream) as Self::GetStreamStream));
                }
            }
            let ctx = self.ctx.clone();
            // L1 hit: serve straight from the chunks cache (Arc clones, zero-copy).
            if let Some((segments, _meta)) = ctx.memory.peek_chunks(&internal) {
                let total: i64 = segments.iter().map(|s| s.len() as i64).sum();
                const SUB_CHUNK: usize = 4 * 1024 * 1024;
                let mut chunks: Vec<pb::DataChunk> = Vec::new();
                let mut running_offset: i64 = 0;
                for seg in segments {
                    let seg_len = seg.len();
                    let n_sub = seg_len.div_ceil(SUB_CHUNK);
                    for i in 0..n_sub {
                        let start = i * SUB_CHUNK;
                        let end = (start + SUB_CHUNK).min(seg_len);
                        chunks.push(pb::DataChunk {
                            data: seg.slice(start..end),
                            offset: running_offset + start as i64,
                            total_size: total,
                            is_last: false,
                        });
                    }
                    running_offset += seg_len as i64;
                }
                if let Some(last) = chunks.last_mut() {
                    last.is_last = true;
                }
                let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
                return Ok(Response::new(Box::pin(stream) as Self::GetStreamStream));
            }

            // Local striped object, L1 miss: streaming stripe pipeline. Read stripes with
            // bounded read-ahead and stream sub-chunks as each stripe lands, so disk reads of
            // stripe N+1..N+RA overlap with encoding/sending of stripe N. The old path waited
            // for ALL stripes (read_striped_chunks barrier) before sending the first byte.
            if let Some(meta) = meta.filter(|m| m.striping.is_some()) {
                let stripe_info = meta.striping.clone().unwrap();
                let stripe_count = stripe_info.chunk_paths.len();
                let total: i64 = stripe_info.total_size as i64;
                const SUB_CHUNK: usize = 4 * 1024 * 1024;
                const READ_AHEAD: usize = 4;
                let (tx, rx) = tokio::sync::mpsc::channel::<Result<pb::DataChunk, Status>>(64);
                let read_start = Instant::now();
                tokio::spawn(async move {
                    let stripe_info = Arc::new(stripe_info);
                    let chunk_size = stripe_info.chunk_size as usize;
                    let mut pending: std::collections::VecDeque<
                        tokio::task::JoinHandle<Result<Bytes, Status>>,
                    > = std::collections::VecDeque::new();
                    let mut next_read = 0usize;
                    let mut next_send = 0usize;
                    let mut sent_bytes: u64 = 0;
                    'outer: while next_send < stripe_count {
                        while next_read < stripe_count && pending.len() < READ_AHEAD {
                            let storage = ctx.storage.clone();
                            let info = stripe_info.clone();
                            let idx = next_read;
                            pending.push_back(tokio::task::spawn_blocking(move || {
                                storage
                                    .read_striped_chunk_at(&info, idx)
                                    .map_err(Status::from)
                            }));
                            next_read += 1;
                        }
                        let handle = match pending.pop_front() {
                            Some(h) => h,
                            None => break,
                        };
                        let seg = match handle.await {
                            Ok(Ok(seg)) => seg,
                            Ok(Err(status)) => {
                                let _ = tx.send(Err(status)).await;
                                return;
                            }
                            Err(e) => {
                                let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                                return;
                            }
                        };
                        let base = (next_send * chunk_size) as i64;
                        let seg_len = seg.len();
                        let n_sub = seg_len.div_ceil(SUB_CHUNK);
                        for i in 0..n_sub {
                            let start = i * SUB_CHUNK;
                            let end = (start + SUB_CHUNK).min(seg_len);
                            sent_bytes += (end - start) as u64;
                            let is_last =
                                next_send + 1 == stripe_count && i + 1 == n_sub;
                            let chunk = pb::DataChunk {
                                data: seg.slice(start..end),
                                offset: base + start as i64,
                                total_size: total,
                                is_last,
                            };
                            if tx.send(Ok(chunk)).await.is_err() {
                                break 'outer; // client went away
                            }
                        }
                        next_send += 1;
                    }
                    tracing::debug!(
                        event = "grpc_get_stream_pipeline",
                        status = "ok",
                        bytes = sent_bytes,
                        stripe_count,
                        total_us = read_start.elapsed().as_micros(),
                    );
                });
                let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                return Ok(Response::new(Box::pin(stream) as Self::GetStreamStream));
            }

            // Non-striped (small) object: old buffered path.
            let opt = tokio::task::spawn_blocking(move || ctx.memory.get_chunks(&internal))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;

            let (segments, _meta) = opt.ok_or_else(|| Status::not_found("key not found"))?;
            let total: i64 = segments.iter().map(|s| s.len() as i64).sum();

            // Split each 64MB stripe segment further with Bytes::slice into multiple ~4MB chunks (zero-copy Arc bump).
            // Reason: an oversized DataChunk (64MB) hits tonic encoder's single-large-message
            // wall (same story as the PUT-side observation that 240×2MB beats 8×60MB). Fine-grained + zero-copy = optimal.
            const SUB_CHUNK: usize = 4 * 1024 * 1024;
            let mut chunks: Vec<pb::DataChunk> = Vec::new();
            let mut running_offset: i64 = 0;
            for seg in segments {
                let seg_len = seg.len();
                let n_sub = seg_len.div_ceil(SUB_CHUNK);
                for i in 0..n_sub {
                    let start = i * SUB_CHUNK;
                    let end = (start + SUB_CHUNK).min(seg_len);
                    let sub_offset = running_offset + start as i64;
                    chunks.push(pb::DataChunk {
                        data: seg.slice(start..end),
                        offset: sub_offset,
                        total_size: total,
                        is_last: false,
                    });
                }
                running_offset += seg_len as i64;
            }
            if let Some(last) = chunks.last_mut() {
                last.is_last = true;
            }
            let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
            Ok(Response::new(Box::pin(stream) as Self::GetStreamStream))
        }
        .await;
        self.record_request("get_stream", start, &result, "ok");
        result
    }

    async fn read_by_descriptor_stream(
        &self,
        req: Request<pb::ReadByDescriptorRequest>,
    ) -> Result<Response<Self::ReadByDescriptorStreamStream>, Status> {
        let start = Instant::now();
        let result = async {
            let req = req.into_inner();
            let requested_placement = req.placement;
            let descriptor = req
                .descriptor
                .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
            let internal = key_from_descriptor(&descriptor)?;
            let active_meta = self.metadata_get_live(&internal).await?;
            let active_meta = active_meta.ok_or_else(|| Status::not_found("key not found"))?;
            validate_descriptor(&descriptor, &active_meta)?;
            validate_placement_descriptor(
                &self.ctx,
                &internal,
                &active_meta,
                requested_placement.as_ref(),
            )?;
            let fresh_descriptor = descriptor_from_meta(&internal, &active_meta);
            let fresh_placement = placement_from_meta(&self.ctx, &internal, &active_meta);
            if self.placement_has_remote_chunks(&fresh_placement) {
                let segments = self
                    .read_chunks_by_placement(descriptor.clone(), fresh_placement.clone())
                    .await?;
                let total: i64 = segments.iter().map(|s| s.len() as i64).sum();
                const SUB_CHUNK: usize = 4 * 1024 * 1024;
                let mut chunks: Vec<pb::DescriptorDataChunk> = Vec::new();
                let mut running_offset: i64 = 0;
                let mut first = true;
                for seg in segments {
                    let seg_len = seg.len();
                    let n_sub = seg_len.div_ceil(SUB_CHUNK);
                    for i in 0..n_sub {
                        let start = i * SUB_CHUNK;
                        let end = (start + SUB_CHUNK).min(seg_len);
                        chunks.push(pb::DescriptorDataChunk {
                            data: seg.slice(start..end),
                            offset: running_offset + start as i64,
                            total_size: total,
                            is_last: false,
                            descriptor: if first {
                                Some(fresh_descriptor.clone())
                            } else {
                                None
                            },
                            placement: if first {
                                first = false;
                                Some(fresh_placement.clone())
                            } else {
                                None
                            },
                        });
                    }
                    running_offset += seg_len as i64;
                }
                if let Some(last) = chunks.last_mut() {
                    last.is_last = true;
                }
                let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
                return Ok(Response::new(
                    Box::pin(stream) as Self::ReadByDescriptorStreamStream
                ));
            }

            let layout_meta = active_meta.clone();
            let read_ctx = self.ctx.clone();
            let read_key = internal.clone();
            let opt = tokio::task::spawn_blocking(
                move || -> Result<Option<(Vec<Bytes>, BlockMeta)>, Status> {
                    read_ctx
                        .storage
                        .get_chunks_with_meta(&read_key, &layout_meta)
                        .map_err(Status::from)
                },
            )
            .await
            .map_err(|e| Status::internal(e.to_string()))??;
            let (segments, _layout_meta) = opt.ok_or_else(|| Status::not_found("key not found"))?;
            let total: i64 = segments.iter().map(|s| s.len() as i64).sum();

            const SUB_CHUNK: usize = 4 * 1024 * 1024;
            let mut chunks: Vec<pb::DescriptorDataChunk> = Vec::new();
            let mut running_offset: i64 = 0;
            let mut first = true;
            for seg in segments {
                let seg_len = seg.len();
                let n_sub = seg_len.div_ceil(SUB_CHUNK);
                for i in 0..n_sub {
                    let start = i * SUB_CHUNK;
                    let end = (start + SUB_CHUNK).min(seg_len);
                    let sub_offset = running_offset + start as i64;
                    chunks.push(pb::DescriptorDataChunk {
                        data: seg.slice(start..end),
                        offset: sub_offset,
                        total_size: total,
                        is_last: false,
                        descriptor: if first {
                            Some(fresh_descriptor.clone())
                        } else {
                            None
                        },
                        placement: if first {
                            first = false;
                            Some(fresh_placement.clone())
                        } else {
                            None
                        },
                    });
                }
                running_offset += seg_len as i64;
            }
            if let Some(last) = chunks.last_mut() {
                last.is_last = true;
            }
            let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
            Ok(Response::new(
                Box::pin(stream) as Self::ReadByDescriptorStreamStream
            ))
        }
        .await;
        self.record_request("read_by_descriptor_stream", start, &result, "ok");
        result
    }

    async fn put_stream(
        &self,
        req: Request<tonic::Streaming<pb::PutChunk>>,
    ) -> Result<Response<pb::PutResponse>, Status> {
        let request_start = Instant::now();
        let t0 = std::time::Instant::now();
        let mut stream = req.into_inner();

        // Pull the first chunk eagerly: it carries key/meta/options/total_size, and total_size
        // decides whether we can enter the streaming stripe pipeline without buffering the
        // whole object.
        let first = match stream.next().await {
            Some(chunk) => chunk?,
            None => {
                let result = Err(Status::invalid_argument("empty stream"));
                self.record_request("put_stream", request_start, &result, "ok");
                return result;
            }
        };
        let first_chunk_elapsed = t0.elapsed();
        let key = first
            .key
            .ok_or_else(|| Status::invalid_argument("first chunk must include key"))?;
        let internal = pb_key_to_internal(&key);
        let m = meta_from_pb(first.metadata.as_ref(), first.options.as_ref());
        let if_not_exists = put_options_if_not_exists(first.options.as_ref());
        let declared_total = first.total_size;

        // Streaming stripe pipeline: declared size is known and large enough for distributed
        // placement — receive, stripe aggregation, and disk I/O overlap; memory is bounded by
        // in-flight stripes instead of the full object. flatten_segments is never called for
        // local stripes.
        if declared_total > 0 && self.should_use_distributed_placement(declared_total as usize) {
            let write_lock = self.key_write_lock(&internal);
            let _guard = write_lock.lock().await;
            if if_not_exists {
                // TTL-aware existence gate, same semantics as put_distributed_bytes_if_absent:
                // a live object refuses the write; an expired one is purged then overwritten.
                let metadata = self.ctx.metadata.clone();
                let str_key = internal.to_string_key();
                let existing =
                    tokio::task::spawn_blocking(move || metadata.get_block(&str_key))
                        .await
                        .map_err(|e| Status::internal(e.to_string()))?
                        .map_err(Status::from)?;
                if let Some(existing) = existing {
                    if !existing.is_expired() {
                        let result = Ok(Response::new(pb::PutResponse {
                            success: false,
                            message: "already exists".to_string(),
                        }));
                        self.record_request("put_stream", request_start, &result, "ok");
                        return result;
                    }
                    self.purge_expired_object_locked(&internal, &existing).await?;
                }
            }
            let first_is_last = first.is_last;
            let outcome = self
                .put_distributed_stream_impl(
                    &mut stream,
                    internal,
                    m,
                    declared_total as usize,
                    first.data,
                    first_is_last,
                    if_not_exists,
                )
                .await;
            let t_total = t0.elapsed();
            let (inserted, stats) = match outcome {
                Ok(v) => v,
                Err(status) => {
                    tracing::debug!(
                        event = "grpc_put_stream",
                        status = "error",
                        mode = "distributed_streamed",
                        declared_bytes = declared_total,
                        total_us = t_total.as_micros(),
                        error = %status,
                    );
                    let result = Err(status);
                    self.record_request("put_stream", request_start, &result, "ok");
                    return result;
                }
            };
            tracing::debug!(
                event = "grpc_put_stream",
                status = "ok",
                mode = "distributed_streamed",
                bytes = declared_total,
                first_chunk_us = first_chunk_elapsed.as_micros(),
                receive_us = stats.receive_elapsed.as_micros(),
                backpressure_wait_us = stats.backpressure_wait.as_micros(),
                drain_wait_us = stats.drain_wait.as_micros(),
                slowest_stripe_us = stats.slowest_stripe.as_micros(),
                metadata_us = stats.metadata_elapsed.as_micros(),
                total_us = t_total.as_micros(),
                throughput_gib_s =
                    declared_total as f64 / t_total.as_secs_f64() / 1_073_741_824.0,
            );
            let result = Ok(Response::new(pb::PutResponse {
                success: inserted,
                message: if inserted {
                    String::new()
                } else {
                    "already exists".to_string()
                },
            }));
            self.record_request("put_stream", request_start, &result, "ok");
            return result;
        }

        // Buffered path: unknown declared size or small object — accumulate segments as before.
        let mut segments: Vec<Bytes> = Vec::new();
        if declared_total > 0 {
            segments.reserve(((declared_total as usize) / (2 * 1024 * 1024)).max(8));
        }
        let mut saw_last = first.is_last;
        segments.push(first.data);
        while !saw_last {
            match stream.next().await {
                Some(chunk) => {
                    let chunk = chunk?;
                    saw_last = chunk.is_last;
                    segments.push(chunk.data);
                }
                None => break,
            }
        }
        let t_recv_done = t0.elapsed();

        let ctx = self.ctx.clone();
        let total_bytes: usize = segments.iter().map(|s| s.len()).sum();
        let n_segs = segments.len();
        tracing::debug!(
            event = "grpc_put_stream_receive",
            status = "ok",
            bytes = total_bytes,
            chunk_count = n_segs,
            declared_bytes = declared_total,
            first_chunk_us = first_chunk_elapsed.as_micros(),
            receive_us = t_recv_done.as_micros(),
        );
        // A client that never declared total_size can still cross the placement threshold once
        // fully received — keep the old flatten + distributed path for that case.
        if self.should_use_distributed_placement(total_bytes) {
            let flatten_start = Instant::now();
            let data = Self::flatten_segments(segments);
            let flatten_elapsed = flatten_start.elapsed();
            let placement_start = Instant::now();
            let inserted = if if_not_exists {
                self.put_distributed_bytes_if_absent(internal, data, m)
                    .await
            } else {
                self.put_distributed_bytes(internal, data, m)
                    .await
                    .map(|_| true)
            };
            let placement_elapsed = placement_start.elapsed();
            let inserted = match inserted {
                Ok(inserted) => inserted,
                Err(status) => {
                    tracing::debug!(
                        event = "grpc_put_stream",
                        status = "error",
                        mode = "distributed",
                        bytes = total_bytes,
                        chunk_count = n_segs,
                        receive_us = t_recv_done.as_micros(),
                        flatten_us = flatten_elapsed.as_micros(),
                        placement_us = placement_elapsed.as_micros(),
                        error = %status,
                    );
                    let result = Err(status);
                    self.record_request("put_stream", request_start, &result, "ok");
                    return result;
                }
            };
            let t_total = t0.elapsed();
            tracing::debug!(
                event = "grpc_put_stream",
                status = "ok",
                mode = "distributed",
                bytes = total_bytes,
                chunk_count = n_segs,
                receive_us = t_recv_done.as_micros(),
                flatten_us = flatten_elapsed.as_micros(),
                placement_us = placement_elapsed.as_micros(),
                total_us = t_total.as_micros(),
                throughput_gib_s = total_bytes as f64 / t_total.as_secs_f64() / 1_073_741_824.0,
            );
            let result = Ok(Response::new(pb::PutResponse {
                success: inserted,
                message: if inserted {
                    String::new()
                } else {
                    "already exists".to_string()
                },
            }));
            self.record_request("put_stream", request_start, &result, "ok");
            return result;
        }
        // Pass-through put_chunks: no concatenation; the storage layer rebuckets on stripe boundaries and flushes via writev
        let storage_start = Instant::now();
        let inserted = tokio::task::spawn_blocking(move || {
            if if_not_exists {
                ctx.memory.put_chunks_if_absent(&internal, segments, m)
            } else {
                ctx.memory.put_chunks(&internal, segments, m)?;
                Ok(true)
            }
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(Status::from);
        let storage_elapsed = storage_start.elapsed();
        let inserted = match inserted {
            Ok(inserted) => inserted,
            Err(status) => {
                tracing::debug!(
                    event = "grpc_put_stream",
                    status = "error",
                    mode = "local",
                    bytes = total_bytes,
                    chunk_count = n_segs,
                    receive_us = t_recv_done.as_micros(),
                    storage_us = storage_elapsed.as_micros(),
                    error = %status,
                );
                let result = Err(status);
                self.record_request("put_stream", request_start, &result, "ok");
                return result;
            }
        };
        let t_total = t0.elapsed();
        tracing::debug!(
            event = "grpc_put_stream",
            status = "ok",
            mode = "local",
            bytes = total_bytes,
            chunk_count = n_segs,
            receive_us = t_recv_done.as_micros(),
            storage_us = storage_elapsed.as_micros(),
            total_us = t_total.as_micros(),
            throughput_gib_s = total_bytes as f64 / t_total.as_secs_f64() / 1_073_741_824.0,
        );
        let _ = declared_total; // silence unused
        let result = Ok(Response::new(pb::PutResponse {
            success: inserted,
            message: if inserted {
                String::new()
            } else {
                "already exists".to_string()
            },
        }));
        self.record_request("put_stream", request_start, &result, "ok");
        result
    }

    // ===== GPU zero-copy (GDS + CUDA IPC) =====
    async fn get_to_gpu(
        &self,
        req: Request<pb::GetToGpuRequest>,
    ) -> Result<Response<pb::GetToGpuResponse>, Status> {
        #[cfg(not(feature = "gds"))]
        {
            let _ = req;
            return Err(Status::unimplemented(
                "GDS path not compiled (rebuild with --features gds)",
            ));
        }
        #[cfg(feature = "gds")]
        {
            if !crate::gds::is_available() {
                return Err(Status::failed_precondition(
                    "GDS runtime not available (libcufile missing or driver_open failed)",
                ));
            }
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            if req.ipc_handle.len() != 64 {
                return Err(Status::invalid_argument("ipc_handle must be 64 bytes"));
            }
            let handle_bytes = req.ipc_handle.clone();
            let buf_size = req.buf_size as usize;
            let device = req.gpu_device;
            let ctx = self.ctx.clone();

            let res = tokio::task::spawn_blocking(move || -> Result<_, crate::error::KVError> {
                if device >= 0 {
                    crate::gds::driver::set_device(device)?;
                }
                let mut gpu_buf = crate::gds::GpuBuffer::from_ipc_handle(&handle_bytes, buf_size)?;
                ctx.memory.get_to_gpu(&internal, &mut gpu_buf)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;

            match res {
                Some((n, meta)) => Ok(Response::new(pb::GetToGpuResponse {
                    found: true,
                    bytes_read: n as u64,
                    metadata: Some(meta_to_pb(&meta)),
                })),
                None => Ok(Response::new(pb::GetToGpuResponse {
                    found: false,
                    bytes_read: 0,
                    metadata: None,
                })),
            }
        }
    }

    async fn put_from_gpu(
        &self,
        req: Request<pb::PutFromGpuRequest>,
    ) -> Result<Response<pb::PutResponse>, Status> {
        #[cfg(not(feature = "gds"))]
        {
            let _ = req;
            return Err(Status::unimplemented(
                "GDS path not compiled (rebuild with --features gds)",
            ));
        }
        #[cfg(feature = "gds")]
        {
            if !crate::gds::is_available() {
                return Err(Status::failed_precondition("GDS runtime not available"));
            }
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            if req.ipc_handle.len() != 64 {
                return Err(Status::invalid_argument("ipc_handle must be 64 bytes"));
            }
            let handle_bytes = req.ipc_handle.clone();
            let buf_size = req.buf_size as usize;
            let device = req.gpu_device;
            let meta = meta_from_pb(req.metadata.as_ref(), None);
            let ctx = self.ctx.clone();

            tokio::task::spawn_blocking(move || -> Result<(), crate::error::KVError> {
                if device >= 0 {
                    crate::gds::driver::set_device(device)?;
                }
                let gpu_buf = crate::gds::GpuBuffer::from_ipc_handle(&handle_bytes, buf_size)?;
                ctx.memory.put_from_gpu(&internal, &gpu_buf, buf_size, meta)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;

            Ok(Response::new(pb::PutResponse {
                success: true,
                message: String::new(),
            }))
        }
    }
}
