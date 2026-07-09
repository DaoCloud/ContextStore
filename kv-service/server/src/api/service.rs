//! gRPC service handler implementation

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use futures::{future::join_all, Stream, StreamExt};
use prost::bytes::{Bytes, BytesMut};
use tokio::sync::Mutex as AsyncMutex;
use tonic::metadata::{MetadataMap, MetadataValue};
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use super::generated::contextstore::kv::v1 as pb;
use crate::config::ClusterNodeConfig;
use crate::metadata::{BlockMeta, ChunkLocation, StripingInfo};
use crate::router::ObjectKey as InternalKey;
use crate::KVServiceContext;
use twox_hash::xxh3::hash64;

const FORWARDED_METADATA_KEY: &str = "x-cs-forwarded";

#[derive(Debug, Clone, PartialEq, Eq)]
struct DataNode {
    node_id: String,
    grpc_endpoint: String,
    rdma_endpoint: String,
}

pub struct KVServiceImpl {
    ctx: Arc<KVServiceContext>,
    local_node: DataNode,
    remote_clients: Arc<DashMap<String, pb::kv_service_client::KvServiceClient<Channel>>>,
    write_locks: Arc<DashMap<String, Arc<AsyncMutex<()>>>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{ChunkLocation, StripingInfo};

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
        let tmp = tempfile::TempDir::new().unwrap();
        cfg.metadata.rocksdb_path = tmp.path().join("meta");
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
                },
            ],
        });

        let placement = placement_from_meta(&ctx, &key(), &meta);
        assert_eq!(placement.chunks.len(), 2);
        assert_eq!(placement.chunks[0].node_id, "node-a");
        assert_eq!(placement.chunks[1].grpc_endpoint, "10.0.0.2:50051");
    }

    #[test]
    fn metadata_owner_matches_primary_data_node() {
        let mut cfg = crate::config::Config::default();
        cfg.cluster.node_id = "node-a".to_string();
        cfg.cluster.grpc_advertise = "10.0.0.1:50051".to_string();
        cfg.cluster.data_nodes = vec![
            ClusterNodeConfig {
                node_id: "node-a".to_string(),
                grpc_endpoint: "10.0.0.1:50051".to_string(),
                rdma_endpoint: String::new(),
            },
            ClusterNodeConfig {
                node_id: "node-b".to_string(),
                grpc_endpoint: "10.0.0.2:50051".to_string(),
                rdma_endpoint: String::new(),
            },
            ClusterNodeConfig {
                node_id: "node-c".to_string(),
                grpc_endpoint: "10.0.0.3:50051".to_string(),
                rdma_endpoint: String::new(),
            },
        ];
        let tmp = tempfile::TempDir::new().unwrap();
        cfg.metadata.rocksdb_path = tmp.path().join("meta");
        let ctx = KVServiceContext::new(cfg).unwrap();

        let owner = select_metadata_owner(&ctx, &key());
        let primary = select_data_node(&ctx, &key(), 0);

        assert_eq!(owner, primary);
    }

    #[test]
    fn canonical_local_node_matches_data_node_by_node_id() {
        let mut cfg = crate::config::Config::default();
        cfg.cluster.node_id = "node-a".to_string();
        cfg.cluster.grpc_advertise = "0.0.0.0:50051".to_string();
        cfg.cluster.data_nodes = vec![
            ClusterNodeConfig {
                node_id: "node-a".to_string(),
                grpc_endpoint: "10.0.0.1:50051".to_string(),
                rdma_endpoint: "10.0.0.1:18515".to_string(),
            },
            ClusterNodeConfig {
                node_id: "node-b".to_string(),
                grpc_endpoint: "10.0.0.2:50051".to_string(),
                rdma_endpoint: String::new(),
            },
        ];
        let tmp = tempfile::TempDir::new().unwrap();
        cfg.metadata.rocksdb_path = tmp.path().join("meta");
        let ctx = KVServiceContext::new(cfg).unwrap();

        let local = canonical_local_node(&ctx);

        assert_eq!(local.node_id, "node-a");
        assert_eq!(local.grpc_endpoint, "10.0.0.1:50051");
    }

    #[test]
    fn forwarded_request_preserves_metadata_and_sets_marker() {
        let mut metadata = MetadataMap::new();
        metadata.insert("x-trace-id", MetadataValue::from_static("trace-1"));

        let request = forwarded_request(
            &metadata,
            pb::GetRequest {
                key: Some(internal_key_to_pb(&key())),
            },
        );

        assert!(request_was_forwarded(request.metadata()));
        assert_eq!(
            request.metadata().get("x-trace-id").unwrap().to_str().unwrap(),
            "trace-1"
        );
        assert!(request.get_ref().key.is_some());
    }

    #[test]
    fn push_node_group_coalesces_identical_nodes() {
        let node_a = DataNode {
            node_id: "node-a".to_string(),
            grpc_endpoint: "10.0.0.1:50051".to_string(),
            rdma_endpoint: String::new(),
        };
        let node_b = DataNode {
            node_id: "node-b".to_string(),
            grpc_endpoint: "10.0.0.2:50051".to_string(),
            rdma_endpoint: String::new(),
        };
        let mut groups: Vec<(DataNode, Vec<usize>)> = Vec::new();

        push_node_group(&mut groups, node_a.clone(), 0);
        push_node_group(&mut groups, node_b, 1);
        push_node_group(&mut groups, node_a, 2);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].1, vec![0, 2]);
        assert_eq!(groups[1].1, vec![1]);
    }

    #[test]
    fn batch_degrade_helpers_update_only_selected_indexes() {
        let mut get_results = vec![
            Some(pb::GetResponse {
                data: Bytes::from_static(b"a"),
                metadata: None,
                found: true,
            }),
            None,
            Some(pb::GetResponse {
                data: Bytes::from_static(b"c"),
                metadata: None,
                found: true,
            }),
        ];
        fill_missing_get_results(&mut get_results, vec![1]);

        assert!(get_results[0].as_ref().unwrap().found);
        assert!(!get_results[1].as_ref().unwrap().found);
        assert!(get_results[2].as_ref().unwrap().found);

        let mut put_success = vec![true, true, true, true];
        fill_failed_put_results(&mut put_success, vec![1, 3]);

        assert_eq!(put_success, vec![true, false, true, false]);
    }

    #[test]
    fn put_stream_chunks_require_final_marker() {
        let first = pb::PutChunk {
            key: Some(internal_key_to_pb(&key())),
            data: Bytes::from_static(b"a"),
            metadata: None,
            options: None,
            offset: 0,
            total_size: 2,
            is_last: false,
        };
        let final_chunk = pb::PutChunk {
            key: None,
            data: Bytes::from_static(b"b"),
            metadata: None,
            options: None,
            offset: 1,
            total_size: 2,
            is_last: true,
        };

        assert!(!put_stream_chunks_have_final(&[first.clone()]));
        assert!(put_stream_chunks_have_final(&[first, final_chunk]));
    }
}

impl KVServiceImpl {
    pub fn new(ctx: KVServiceContext) -> Self {
        Self::new_shared(Arc::new(ctx))
    }

    pub fn new_shared(ctx: Arc<KVServiceContext>) -> Self {
        let local_node = canonical_local_node(&ctx);
        Self {
            ctx,
            local_node,
            remote_clients: Arc::new(DashMap::new()),
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

    async fn owner_client_for_key(
        &self,
        key: &InternalKey,
        metadata: &MetadataMap,
    ) -> Result<Option<(DataNode, pb::kv_service_client::KvServiceClient<Channel>)>, Status> {
        if request_was_forwarded(metadata) {
            return Ok(None);
        }
        let owner = select_metadata_owner(&self.ctx, key);
        if is_local_node(&self.local_node, &owner) {
            return Ok(None);
        }
        let client = Self::cached_client_for_node(self.remote_clients.clone(), &owner).await?;
        Ok(Some((owner, client)))
    }

    async fn cached_client_for_node(
        remote_clients: Arc<DashMap<String, pb::kv_service_client::KvServiceClient<Channel>>>,
        node: &DataNode,
    ) -> Result<pb::kv_service_client::KvServiceClient<Channel>, Status> {
        if let Some(client) = remote_clients.get(&node.grpc_endpoint) {
            return Ok(client.value().clone());
        }

        let client =
            pb::kv_service_client::KvServiceClient::connect(grpc_uri(&node.grpc_endpoint))
                .await
                .map_err(|e| {
                    Status::unavailable(format!(
                        "connect metadata owner {}: {}",
                        node.node_id, e
                    ))
                })?;
        remote_clients.insert(node.grpc_endpoint.clone(), client.clone());
        Ok(client)
    }

    fn evict_remote_client(&self, node: &DataNode) {
        self.remote_clients.remove(&node.grpc_endpoint);
    }

    fn key_write_lock(&self, key: &InternalKey) -> Arc<AsyncMutex<()>> {
        let str_key = key.to_string_key();
        self.write_locks
            .entry(str_key)
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    async fn metadata_exists(&self, key: &InternalKey) -> Result<bool, Status> {
        let metadata = self.ctx.metadata.clone();
        let str_key = key.to_string_key();
        tokio::task::spawn_blocking(move || metadata.get_block(&str_key).map(|m| m.is_some()))
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
        local_node: DataNode,
        remote_clients: Arc<DashMap<String, pb::kv_service_client::KvServiceClient<Channel>>>,
        node: DataNode,
        key: InternalKey,
        descriptor: pb::ObjectDescriptor,
        stripe_index: usize,
        offset: u64,
        chunk_size: u64,
        total_size: u64,
        data: Bytes,
    ) -> Result<ChunkLocation, Status> {
        if is_local_node(&local_node, &node) {
            let key_for_write = key.clone();
            let data_len = data.len() as u64;
            let storage = ctx.storage.clone();
            let generation = descriptor.object_generation;
            let layout_version = descriptor.layout_version;
            let (device_id, storage_handle) = tokio::task::spawn_blocking(move || {
                storage.put_placement_chunk(
                    &key_for_write,
                    stripe_index,
                    generation,
                    layout_version,
                    data,
                )
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
            return Ok(ChunkLocation {
                stripe_index: stripe_index as u32,
                node_id: node.node_id,
                grpc_endpoint: node.grpc_endpoint,
                rdma_endpoint: node.rdma_endpoint,
                device_id,
                storage_handle,
                offset,
                length: data_len,
            });
        }

        let mut client = Self::cached_client_for_node(remote_clients, &node).await?;
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
        if !resp.success {
            return Err(Status::internal(format!(
                "data node {} rejected placement chunk",
                node.node_id
            )));
        }
        let chunk = resp
            .chunk
            .ok_or_else(|| Status::internal("missing placement chunk in response"))?;
        Ok(pb_chunk_to_location(&chunk))
    }

    async fn put_distributed_bytes_impl(
        &self,
        key: InternalKey,
        data: Bytes,
        meta: BlockMeta,
    ) -> Result<(), Status> {
        let total = data.len();
        let chunk_size = self.ctx.storage.striping_chunk_size().max(1) as usize;
        let stripe_count = total.div_ceil(chunk_size);
        let prepared_meta = self
            .ctx
            .storage
            .prepare_write_meta(&key, meta, total as u64)
            .map_err(Status::from)?;
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
                self.local_node.clone(),
                self.remote_clients.clone(),
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
        let mut locations = Vec::with_capacity(stripe_count);
        for result in join_all(tasks).await {
            locations.push(result?);
        }
        locations.sort_by_key(|loc| loc.stripe_index);

        let chunk_devices = locations.iter().map(|loc| loc.device_id).collect();
        let chunk_paths = locations
            .iter()
            .map(|loc| loc.storage_handle.clone())
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
        });
        self.ctx.memory.invalidate(&key);
        self.ctx
            .metadata
            .put_block(&key.to_string_key(), &committed)
            .map_err(Status::from)?;
        Ok(())
    }

    async fn put_distributed_bytes(
        &self,
        key: InternalKey,
        data: Bytes,
        meta: BlockMeta,
    ) -> Result<(), Status> {
        let write_lock = self.key_write_lock(&key);
        let _guard = write_lock.lock().await;
        self.put_distributed_bytes_impl(key, data, meta).await
    }

    async fn put_distributed_bytes_if_absent(
        &self,
        key: InternalKey,
        data: Bytes,
        meta: BlockMeta,
    ) -> Result<bool, Status> {
        let write_lock = self.key_write_lock(&key);
        let _guard = write_lock.lock().await;

        if self.metadata_exists(&key).await? {
            return Ok(false);
        }

        self.put_distributed_bytes_impl(key, data, meta).await?;
        Ok(true)
    }

    fn placement_has_remote_chunks(&self, placement: &pb::PlacementDescriptor) -> bool {
        placement.chunks.iter().any(|chunk| {
            let node = DataNode {
                node_id: chunk.node_id.clone(),
                grpc_endpoint: chunk.grpc_endpoint.clone(),
                rdma_endpoint: chunk.rdma_endpoint.clone(),
            };
            !is_local_node(&self.local_node, &node)
        })
    }

    async fn read_chunk_from_placement(
        ctx: Arc<KVServiceContext>,
        local_node: DataNode,
        remote_clients: Arc<DashMap<String, pb::kv_service_client::KvServiceClient<Channel>>>,
        descriptor: pb::ObjectDescriptor,
        chunk: pb::PlacementChunk,
    ) -> Result<(u32, Bytes), Status> {
        let node = DataNode {
            node_id: chunk.node_id.clone(),
            grpc_endpoint: chunk.grpc_endpoint.clone(),
            rdma_endpoint: chunk.rdma_endpoint.clone(),
        };
        if is_local_node(&local_node, &node) {
            let storage = ctx.storage.clone();
            let handle = chunk.storage_handle.clone();
            let expected_len = chunk.length;
            let stripe_index = chunk.stripe_index;
            let data = tokio::task::spawn_blocking(move || {
                storage.read_placement_chunk(&handle, expected_len)
            })
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?
            .ok_or_else(|| Status::not_found("placement chunk not found"))?;
            return Ok((stripe_index, data));
        }

        let mut client = Self::cached_client_for_node(remote_clients, &node).await?;
        let mut stream = client
            .read_placement_chunk(pb::ReadPlacementChunkRequest {
                descriptor: Some(descriptor),
                chunk: Some(chunk.clone()),
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
        let mut tasks = Vec::with_capacity(placement.chunks.len());
        for chunk in placement.chunks {
            tasks.push(Self::read_chunk_from_placement(
                self.ctx.clone(),
                self.local_node.clone(),
                self.remote_clients.clone(),
                descriptor.clone(),
                chunk,
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
        local_node: DataNode,
        remote_clients: Arc<DashMap<String, pb::kv_service_client::KvServiceClient<Channel>>>,
        chunk: pb::PlacementChunk,
    ) -> Result<(), Status> {
        let node = DataNode {
            node_id: chunk.node_id.clone(),
            grpc_endpoint: chunk.grpc_endpoint.clone(),
            rdma_endpoint: chunk.rdma_endpoint.clone(),
        };
        if is_local_node(&local_node, &node) {
            let storage = ctx.storage.clone();
            let handle = chunk.storage_handle.clone();
            tokio::task::spawn_blocking(move || storage.delete_placement_chunk(&handle))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            return Ok(());
        }

        let mut client = Self::cached_client_for_node(remote_clients, &node).await?;
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
            tasks.push(Self::delete_chunk_from_placement(
                self.ctx.clone(),
                self.local_node.clone(),
                self.remote_clients.clone(),
                chunk,
            ));
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

fn meta_from_pb(m: Option<&pb::KvMetadata>) -> BlockMeta {
    let now = chrono::Utc::now().timestamp();
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
            ttl_seconds: 0,
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
            ttl_seconds: 0,
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

fn request_was_forwarded(metadata: &MetadataMap) -> bool {
    metadata.contains_key(FORWARDED_METADATA_KEY)
}

fn forwarded_request<T>(metadata: &MetadataMap, message: T) -> Request<T> {
    let mut request = Request::new(message);
    *request.metadata_mut() = metadata.clone();
    request.metadata_mut().insert(
        FORWARDED_METADATA_KEY,
        MetadataValue::from_static("1"),
    );
    request
}

fn missing_get_response() -> pb::GetResponse {
    pb::GetResponse {
        data: Bytes::new(),
        metadata: None,
        found: false,
    }
}

fn fill_missing_get_results(results: &mut [Option<pb::GetResponse>], indexes: Vec<usize>) {
    for idx in indexes {
        results[idx] = Some(missing_get_response());
    }
}

fn fill_failed_put_results(success: &mut [bool], indexes: Vec<usize>) {
    for idx in indexes {
        success[idx] = false;
    }
}

fn put_stream_chunks_have_final(chunks: &[pb::PutChunk]) -> bool {
    chunks.last().map(|chunk| chunk.is_last).unwrap_or(false)
}

fn memory_get_result_to_pb(
    result: crate::error::Result<Option<(Bytes, BlockMeta)>>,
) -> pb::GetResponse {
    match result {
        Ok(Some((data, meta))) => pb::GetResponse {
            data,
            metadata: Some(meta_to_pb(&meta)),
            found: true,
        },
        _ => missing_get_response(),
    }
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

fn canonical_local_node(ctx: &KVServiceContext) -> DataNode {
    let configured = local_node(ctx);
    if ctx.config.cluster.data_nodes.is_empty() {
        return configured;
    }

    for node in configured_data_nodes(ctx) {
        let node_id_matches =
            !configured.node_id.is_empty() && configured.node_id.as_str() == node.node_id.as_str();
        let endpoint_matches = !configured.grpc_endpoint.is_empty()
            && configured.grpc_endpoint.as_str() == node.grpc_endpoint.as_str();
        if node_id_matches || endpoint_matches {
            return node;
        }
    }

    configured
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

fn is_local_node(local: &DataNode, node: &DataNode) -> bool {
    node.node_id.as_str() == local.node_id.as_str()
        || node.grpc_endpoint.as_str() == local.grpc_endpoint.as_str()
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

fn select_metadata_owner(ctx: &KVServiceContext, key: &InternalKey) -> DataNode {
    select_data_node(ctx, key, 0)
}

fn push_node_group<T>(groups: &mut Vec<(DataNode, Vec<T>)>, node: DataNode, item: T) {
    for (existing, items) in groups.iter_mut() {
        if PartialEq::eq(&*existing, &node) {
            items.push(item);
            return;
        }
    }
    groups.push((node, vec![item]));
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

#[cfg(test)]
fn placement_from_meta(
    ctx: &KVServiceContext,
    key: &InternalKey,
    meta: &BlockMeta,
) -> pb::PlacementDescriptor {
    let local = local_node(ctx);
    placement_from_meta_with_local(ctx, &local, key, meta)
}

fn placement_from_meta_with_local(
    ctx: &KVServiceContext,
    local: &DataNode,
    key: &InternalKey,
    meta: &BlockMeta,
) -> pb::PlacementDescriptor {
    let placement_policy_id = format!("{}_v1", ctx.config.router.strategy);

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
        }],
    };

    let mut hash_seed = format!(
        "{}|g{}|l{}|{}|{}",
        key.to_string_key(),
        meta.object_generation,
        meta.layout_version,
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
        placement_epoch: 1,
        placement_policy_id,
        layout_hash: format!("{:016x}", hash64(hash_seed.as_bytes())),
        primary_node_id: local.node_id.clone(),
        primary_grpc_endpoint: local.grpc_endpoint.clone(),
        primary_rdma_endpoint: local.rdma_endpoint.clone(),
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
            let metadata = req.metadata().clone();
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            if let Some((owner, mut client)) =
                self.owner_client_for_key(&internal, &metadata).await?
            {
                let response = client
                    .get(forwarded_request(&metadata, pb::GetRequest { key: Some(key) }))
                    .await;
                if response.is_err() {
                    self.evict_remote_client(&owner);
                }
                return response;
            }
            let str_key = internal.to_string_key();
            let meta_ctx = self.ctx.clone();
            let meta = tokio::task::spawn_blocking(move || meta_ctx.metadata.get_block(&str_key))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            if let Some(meta) = meta.as_ref() {
                let placement =
                    placement_from_meta_with_local(&self.ctx, &self.local_node, &internal, meta);
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
            let metadata = req.metadata().clone();
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            // pb::PutRequest.data is Bytes (a buffer reference handed over by the gRPC framework, no copy)
            let data: Bytes = req.data;
            if let Some((owner, mut client)) =
                self.owner_client_for_key(&internal, &metadata).await?
            {
                let response = client
                    .put(forwarded_request(
                        &metadata,
                        pb::PutRequest {
                            key: Some(key),
                            data,
                            metadata: req.metadata,
                            options: req.options,
                        },
                    ))
                    .await;
                if response.is_err() {
                    self.evict_remote_client(&owner);
                }
                return response;
            }
            let meta = meta_from_pb(req.metadata.as_ref());
            let if_not_exists = put_options_if_not_exists(req.options.as_ref());
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
        let metadata = req.metadata().clone();
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let internal = pb_key_to_internal(&key);
        if let Some((owner, mut client)) =
            self.owner_client_for_key(&internal, &metadata).await?
        {
            let response = client
                .delete(forwarded_request(&metadata, pb::DeleteRequest { key: Some(key) }))
                .await;
            if response.is_err() {
                self.evict_remote_client(&owner);
            }
            return response;
        }
        let str_key = internal.to_string_key();
        let meta_ctx = self.ctx.clone();
        let meta = tokio::task::spawn_blocking(move || meta_ctx.metadata.get_block(&str_key))
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)?;
        if let Some(meta) = meta.as_ref() {
            let placement =
                placement_from_meta_with_local(&self.ctx, &self.local_node, &internal, meta);
            if self.placement_has_remote_chunks(&placement) {
                self.delete_distributed_chunks(placement).await?;
                self.ctx.memory.invalidate(&internal);
                self.ctx
                    .metadata
                    .delete_block(&internal.to_string_key())
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
        let metadata = req.metadata().clone();
        let req = req.into_inner();
        let key = req
            .key
            .ok_or_else(|| Status::invalid_argument("missing key"))?;
        let internal = pb_key_to_internal(&key);
        if let Some((owner, mut client)) =
            self.owner_client_for_key(&internal, &metadata).await?
        {
            let response = client
                .exists(forwarded_request(&metadata, pb::ExistsRequest { key: Some(key) }))
                .await;
            if response.is_err() {
                self.evict_remote_client(&owner);
            }
            return response;
        }
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
            let metadata = req.metadata().clone();
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            if let Some((owner, mut client)) =
                self.owner_client_for_key(&internal, &metadata).await?
            {
                let response = client
                    .lookup_object(forwarded_request(
                        &metadata,
                        pb::LookupObjectRequest { key: Some(key) },
                    ))
                    .await;
                if response.is_err() {
                    self.evict_remote_client(&owner);
                }
                return response;
            }
            let str_key = internal.to_string_key();
            let ctx = self.ctx.clone();
            let meta = tokio::task::spawn_blocking(move || ctx.metadata.get_block(&str_key))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            let descriptor = meta.as_ref().map(|m| descriptor_from_meta(&internal, m));
            let placement = meta
                .as_ref()
                .map(|m| {
                    placement_from_meta_with_local(&self.ctx, &self.local_node, &internal, m)
                });
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
            let metadata = req.metadata().clone();
            let req = req.into_inner();
            let descriptor = req
                .descriptor
                .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
            let internal = key_from_descriptor(&descriptor)?;
            if let Some((owner, mut client)) =
                self.owner_client_for_key(&internal, &metadata).await?
            {
                let response = client
                    .read_by_descriptor(forwarded_request(
                        &metadata,
                        pb::ReadByDescriptorRequest {
                            descriptor: Some(descriptor),
                            placement: req.placement,
                        },
                    ))
                    .await;
                if response.is_err() {
                    self.evict_remote_client(&owner);
                }
                return response;
            }
            let str_key = internal.to_string_key();
            let meta_ctx = self.ctx.clone();
            let meta_task =
                tokio::task::spawn_blocking(move || meta_ctx.metadata.get_block(&str_key));
            let active_meta = meta_task
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
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
            let placement = placement_from_meta_with_local(
                &self.ctx,
                &self.local_node,
                &internal,
                &active_meta,
            );
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
        let offset = stripe_index as u64 * req.chunk_size;
        let local = self.local_node.clone();
        let ctx = self.ctx.clone();
        let (device_id, storage_handle) = tokio::task::spawn_blocking(move || {
            ctx.storage.put_placement_chunk(
                &internal,
                stripe_index,
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
        let chunk = req
            .chunk
            .ok_or_else(|| Status::invalid_argument("missing placement chunk"))?;
        let storage = self.ctx.storage.clone();
        let handle = chunk.storage_handle.clone();
        let expected_len = chunk.length;
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
            storage.read_placement_chunk(&handle, expected_len)
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
        let metadata = req.metadata().clone();
        let was_forwarded = request_was_forwarded(&metadata);
        let req = req.into_inner();
        let mut local_keys: Vec<(usize, InternalKey)> = Vec::new();
        let mut remote_groups: Vec<(DataNode, Vec<(usize, pb::ObjectKey)>)> = Vec::new();
        let total = req.keys.len();

        for (idx, key) in req.keys.into_iter().enumerate() {
            let internal = pb_key_to_internal(&key);
            let owner = select_metadata_owner(&self.ctx, &internal);
            if was_forwarded || is_local_node(&self.local_node, &owner) {
                local_keys.push((idx, internal));
            } else {
                push_node_group(&mut remote_groups, owner, (idx, key));
            }
        }

        let mut results: Vec<Option<pb::GetResponse>> = (0..total).map(|_| None).collect();

        if !local_keys.is_empty() {
            let batch_keys: Vec<InternalKey> =
                local_keys.iter().map(|(_, key)| key.clone()).collect();
            let ctx = self.ctx.clone();
            let local_results =
                tokio::task::spawn_blocking(move || ctx.memory.get_batch(&batch_keys))
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
            for ((idx, _), result) in local_keys.into_iter().zip(local_results.into_iter()) {
                results[idx] = Some(memory_get_result_to_pb(result));
            }
        }

        let mut remote_tasks = Vec::with_capacity(remote_groups.len());
        for (node, entries) in remote_groups {
            let remote_clients = self.remote_clients.clone();
            let metadata = metadata.clone();
            let indexes: Vec<usize> = entries.iter().map(|(idx, _)| *idx).collect();
            let keys: Vec<pb::ObjectKey> = entries.into_iter().map(|(_, key)| key).collect();
            remote_tasks.push(async move {
                let outcome = match Self::cached_client_for_node(remote_clients, &node).await {
                    Ok(mut client) => client
                        .get_batch(forwarded_request(&metadata, pb::GetBatchRequest { keys }))
                        .await
                        .map_err(|e| {
                            Status::unavailable(format!("get batch from {}: {}", node.node_id, e))
                        })
                        .and_then(|response| {
                            let response = response.into_inner();
                            if response.results.len() != indexes.len() {
                                return Err(Status::internal(format!(
                                    "metadata owner {} returned {} get results for {} keys",
                                    node.node_id,
                                    response.results.len(),
                                    indexes.len()
                                )));
                            }
                            Ok(response.results)
                        }),
                    Err(status) => Err(status),
                };
                (node, indexes, outcome)
            });
        }
        for (node, indexes, outcome) in join_all(remote_tasks).await {
            match outcome {
                Ok(response_results) => {
                    for (idx, result) in indexes.into_iter().zip(response_results.into_iter()) {
                        results[idx] = Some(result);
                    }
                }
                Err(status) => {
                    self.evict_remote_client(&node);
                    tracing::warn!(
                        "metadata owner {} failed get batch for {} keys: {}",
                        node.node_id,
                        indexes.len(),
                        status
                    );
                    fill_missing_get_results(&mut results, indexes);
                }
            }
        }

        Ok(Response::new(pb::GetBatchResponse {
            results: results
                .into_iter()
                .map(|result| result.unwrap_or_else(missing_get_response))
                .collect(),
        }))
    }

    async fn put_batch(
        &self,
        req: Request<pb::PutBatchRequest>,
    ) -> Result<Response<pb::PutBatchResponse>, Status> {
        let metadata = req.metadata().clone();
        let was_forwarded = request_was_forwarded(&metadata);
        let req = req.into_inner();
        // item.data is Bytes (a refcounted view over the gRPC framework buffer)
        let mut local_items: Vec<(usize, InternalKey, Bytes, BlockMeta, bool)> =
            Vec::with_capacity(req.items.len());
        let mut remote_groups: Vec<(DataNode, Vec<(usize, pb::PutRequest)>)> = Vec::new();
        let mut has_if_not_exists = false;
        let total = req.items.len();

        for (idx, item) in req.items.into_iter().enumerate() {
            let key = item
                .key
                .clone()
                .ok_or_else(|| Status::invalid_argument("missing key in batch item"))?;
            let internal = pb_key_to_internal(&key);
            let owner = select_metadata_owner(&self.ctx, &internal);
            if was_forwarded || is_local_node(&self.local_node, &owner) {
                let meta = meta_from_pb(item.metadata.as_ref());
                let if_not_exists = put_options_if_not_exists(item.options.as_ref());
                has_if_not_exists |= if_not_exists;
                local_items.push((idx, internal, item.data, meta, if_not_exists));
            } else {
                push_node_group(&mut remote_groups, owner, (idx, item));
            }
        }

        let mut success = vec![false; total];

        if !local_items.is_empty() {
            let ctx = self.ctx.clone();
            let indexed_success = if has_if_not_exists {
                tokio::task::spawn_blocking(move || {
                    local_items
                        .into_iter()
                        .map(|(idx, key, data, meta, if_not_exists)| {
                            let ok = if if_not_exists {
                                ctx.memory.put_if_absent(&key, data, meta)
                            } else {
                                ctx.memory.put(&key, data, meta).map(|_| true)
                            }
                            .unwrap_or(false);
                            (idx, ok)
                        })
                        .collect::<Vec<(usize, bool)>>()
                })
                .await
                .map_err(|e| Status::internal(e.to_string()))?
            } else {
                let indexes: Vec<usize> = local_items
                    .iter()
                    .map(|(idx, _, _, _, _)| *idx)
                    .collect();
                let batch_items = local_items
                    .into_iter()
                    .map(|(_, key, data, meta, _)| (key, data, meta))
                    .collect();
                let results =
                    tokio::task::spawn_blocking(move || ctx.memory.put_batch(batch_items))
                        .await
                        .map_err(|e| Status::internal(e.to_string()))?;
                indexes
                    .into_iter()
                    .zip(results.into_iter().map(|result| result.is_ok()))
                    .collect()
            };
            for (idx, ok) in indexed_success {
                success[idx] = ok;
            }
        }

        let mut remote_tasks = Vec::with_capacity(remote_groups.len());
        for (node, entries) in remote_groups {
            let remote_clients = self.remote_clients.clone();
            let metadata = metadata.clone();
            let indexes: Vec<usize> = entries.iter().map(|(idx, _)| *idx).collect();
            let items: Vec<pb::PutRequest> =
                entries.into_iter().map(|(_, item)| item).collect();
            remote_tasks.push(async move {
                let outcome = match Self::cached_client_for_node(remote_clients, &node).await {
                    Ok(mut client) => client
                        .put_batch(forwarded_request(&metadata, pb::PutBatchRequest { items }))
                        .await
                        .map_err(|e| {
                            Status::unavailable(format!("put batch to {}: {}", node.node_id, e))
                        })
                        .and_then(|response| {
                            let response = response.into_inner();
                            if response.success.len() != indexes.len() {
                                return Err(Status::internal(format!(
                                    "metadata owner {} returned {} put results for {} items",
                                    node.node_id,
                                    response.success.len(),
                                    indexes.len()
                                )));
                            }
                            Ok(response.success)
                        }),
                    Err(status) => Err(status),
                };
                (node, indexes, outcome)
            });
        }
        for (node, indexes, outcome) in join_all(remote_tasks).await {
            match outcome {
                Ok(response_success) => {
                    for (idx, ok) in indexes.into_iter().zip(response_success.into_iter()) {
                        success[idx] = ok;
                    }
                }
                Err(status) => {
                    self.evict_remote_client(&node);
                    tracing::warn!(
                        "metadata owner {} failed put batch for {} items: {}",
                        node.node_id,
                        indexes.len(),
                        status
                    );
                    fill_failed_put_results(&mut success, indexes);
                }
            }
        }

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
            let metadata = req.metadata().clone();
            let req = req.into_inner();
            let key = req
                .key
                .ok_or_else(|| Status::invalid_argument("missing key"))?;
            let internal = pb_key_to_internal(&key);
            if let Some((owner, mut client)) =
                self.owner_client_for_key(&internal, &metadata).await?
            {
                let response = client
                    .get_stream(forwarded_request(&metadata, pb::GetRequest { key: Some(key) }))
                    .await;
                if response.is_err() {
                    self.evict_remote_client(&owner);
                }
                let response = response?;
                return Ok(Response::new(
                    Box::pin(response.into_inner()) as Self::GetStreamStream
                ));
            }
            let str_key = internal.to_string_key();
            let meta_ctx = self.ctx.clone();
            let meta = tokio::task::spawn_blocking(move || meta_ctx.metadata.get_block(&str_key))
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            if let Some(meta) = meta.as_ref() {
                let placement =
                    placement_from_meta_with_local(&self.ctx, &self.local_node, &internal, meta);
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
            // Grab Vec<Bytes> directly (typically 8 × 64MB segments), zero-copy throughout — no more 480MB concatenation.
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
            let metadata = req.metadata().clone();
            let req = req.into_inner();
            let descriptor = req
                .descriptor
                .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
            let internal = key_from_descriptor(&descriptor)?;
            if let Some((owner, mut client)) =
                self.owner_client_for_key(&internal, &metadata).await?
            {
                let response = client
                    .read_by_descriptor_stream(forwarded_request(
                        &metadata,
                        pb::ReadByDescriptorRequest {
                            descriptor: Some(descriptor),
                            placement: req.placement,
                        },
                    ))
                    .await;
                if response.is_err() {
                    self.evict_remote_client(&owner);
                }
                let response = response?;
                return Ok(Response::new(
                    Box::pin(response.into_inner()) as Self::ReadByDescriptorStreamStream
                ));
            }
            let str_key = internal.to_string_key();
            let meta_ctx = self.ctx.clone();
            let meta_task =
                tokio::task::spawn_blocking(move || meta_ctx.metadata.get_block(&str_key));
            let active_meta = meta_task
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            let active_meta = active_meta.ok_or_else(|| Status::not_found("key not found"))?;
            validate_descriptor(&descriptor, &active_meta)?;
            let fresh_descriptor = descriptor_from_meta(&internal, &active_meta);
            let fresh_placement = placement_from_meta_with_local(
                &self.ctx,
                &self.local_node,
                &internal,
                &active_meta,
            );
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
        let metadata = req.metadata().clone();
        let mut inbound = req.into_inner();
        let first_chunk = match inbound.next().await {
            Some(chunk) => chunk?,
            None => {
                let result = Err(Status::invalid_argument("empty stream"));
                self.record_request("put_stream", request_start, &result, "ok");
                return result;
            }
        };
        let key = first_chunk
            .key
            .clone()
            .ok_or_else(|| Status::invalid_argument("first chunk must include key"))?;
        let internal = pb_key_to_internal(&key);
        if let Some((owner, mut client)) =
            self.owner_client_for_key(&internal, &metadata).await?
        {
            let (tx, rx) = tokio::sync::mpsc::channel(16);
            tokio::spawn(async move {
                if tx.send(first_chunk).await.is_err() {
                    return;
                }
                while let Some(chunk) = inbound.next().await {
                    match chunk {
                        Ok(chunk) => {
                            if tx.send(chunk).await.is_err() {
                                return;
                            }
                        }
                        Err(status) => {
                            tracing::warn!("PUT stream forwarding stopped: {}", status);
                            return;
                        }
                    }
                }
            });
            let forwarded_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let result = client
                .put_stream(forwarded_request(&metadata, forwarded_stream))
                .await;
            if result.is_err() {
                self.evict_remote_client(&owner);
            }
            self.record_request("put_stream", request_start, &result, "ok");
            return result;
        }
        let meta_opt = first_chunk.metadata.clone();
        let options_opt = first_chunk.options.clone();
        let declared_total = first_chunk.total_size;
        let first_is_last = first_chunk.is_last;
        let mut chunks: Vec<pb::PutChunk> = Vec::new();
        if declared_total > 0 {
            chunks.reserve(((declared_total as usize) / (2 * 1024 * 1024)).max(8));
        }
        chunks.push(first_chunk);
        if !first_is_last {
            while let Some(chunk) = inbound.next().await {
                let chunk = chunk?;
                let is_last = chunk.is_last;
                chunks.push(chunk);
                if is_last {
                    break;
                }
            }
        }
        if !put_stream_chunks_have_final(&chunks) {
            let result = Err(Status::invalid_argument("PUT stream missing final chunk"));
            self.record_request("put_stream", request_start, &result, "ok");
            return result;
        }
        let t_recv_done = t0.elapsed();
        let m = meta_from_pb(meta_opt.as_ref());
        let if_not_exists = put_options_if_not_exists(options_opt.as_ref());
        let ctx = self.ctx.clone();
        // chunk.data is Bytes (a refcounted view decoded by gRPC, zero-copy)
        let segments: Vec<Bytes> = chunks.into_iter().map(|chunk| chunk.data).collect();
        let total_bytes: usize = segments.iter().map(|s| s.len()).sum();
        let n_segs = segments.len();
        if self.should_use_distributed_placement(total_bytes) {
            let data = Self::flatten_segments(segments);
            let inserted = if if_not_exists {
                self.put_distributed_bytes_if_absent(internal, data, m)
                    .await?
            } else {
                self.put_distributed_bytes(internal, data, m).await?;
                true
            };
            let t_total = t0.elapsed();
            tracing::trace!(
                "PUT_PERF bytes={} n_segs={} recv_ms={} put_ms={} total_ms={} BW={:.2}GB/s mode=distributed",
                total_bytes,
                n_segs,
                t_recv_done.as_millis(),
                (t_total - t_recv_done).as_millis(),
                t_total.as_millis(),
                total_bytes as f64 / t_total.as_secs_f64() / 1_073_741_824.0,
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
        .map_err(Status::from)?;
        let t_total = t0.elapsed();
        tracing::trace!(
            "PUT_PERF bytes={} n_segs={} recv_ms={} put_ms={} total_ms={} BW={:.2}GB/s",
            total_bytes,
            n_segs,
            t_recv_done.as_millis(),
            (t_total - t_recv_done).as_millis(),
            t_total.as_millis(),
            total_bytes as f64 / t_total.as_secs_f64() / 1_073_741_824.0,
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
            let meta = meta_from_pb(req.metadata.as_ref());
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
