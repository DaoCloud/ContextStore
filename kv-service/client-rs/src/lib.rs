//! ContextStore KV Service — Rust client SDK.
//!
//! Mirrors the Python `contextstore_client.KVClient` but uses native Rust + tonic,
//! bypassing the ~30ms per-RPC protocol/interpreter overhead of the Python gRPC stack.
//!
//! **Zero-copy interfaces**: `get_stream_chunks` / `put_stream_chunks` operate on
//! `Vec<Bytes>` directly, avoiding the ~0.4 GB/s single-connection ceiling caused by
//! concatenating 480MB on the client side. Instead, the gRPC framework's inbound
//! buffer view is handed straight to the caller.

pub mod pb {
    tonic::include_proto!("contextstore.kv.v1");
}

use pb::kv_service_client::KvServiceClient;
use prost::bytes::Bytes;
use tonic::transport::Channel;

const TWO_GIB: usize = 2 * 1024 * 1024 * 1024;

/// Result of a KVService object lookup. Upper-layer tier clients can keep the
/// descriptor identity in their cached object reference, then read through
/// `read_by_descriptor_stream_chunks`.
#[derive(Clone, Debug)]
pub struct ObjectLookup {
    /// Server-generated object identity, including generation / etag / layout version.
    pub descriptor: pb::ObjectDescriptor,
    /// Actual object placement. The first gRPC descriptor-read path can pass it back to
    /// the server without interpreting it.
    pub placement: Option<pb::PlacementDescriptor>,
}

/// Descriptor-read result, preserving the fresh descriptor / placement returned by the server.
#[derive(Clone, Debug)]
pub struct DescriptorReadChunks {
    /// Data segments returned in offset order.
    pub segments: Vec<Bytes>,
    /// Fresh descriptor returned by the server; callers may refresh cached object references with it.
    pub descriptor: pb::ObjectDescriptor,
    /// Fresh placement returned by the server; later RDMA/GDS paths can reuse it.
    pub placement: Option<pb::PlacementDescriptor>,
}

/// Rust gRPC client wrapper for the KV Service.
#[derive(Clone)]
pub struct KvClient {
    inner: KvServiceClient<Channel>,
}

impl KvClient {
    /// Connect to a server. `endpoint` looks like "http://127.0.0.1:50051".
    pub async fn connect(endpoint: String) -> Result<Self, Box<dyn std::error::Error>> {
        // Large messages: raise decode/encode limits to 2 GiB, matching server main.rs.
        // Widen the HTTP/2 flow-control window (default ~64KB); otherwise large-value
        // transfers get bottlenecked by WINDOW_UPDATE round trips.
        let channel = Channel::from_shared(endpoint)?
            .initial_stream_window_size(Some(64 * 1024 * 1024))
            .initial_connection_window_size(Some(128 * 1024 * 1024))
            .connect()
            .await?;
        let inner = KvServiceClient::new(channel)
            .max_decoding_message_size(TWO_GIB)
            .max_encoding_message_size(TWO_GIB);
        Ok(Self { inner })
    }

    pub async fn health(&mut self) -> Result<bool, tonic::Status> {
        let resp = self.inner.health(pb::HealthRequest {}).await?;
        Ok(resp.into_inner().status == pb::health_response::ServingStatus::Serving as i32)
    }

    fn key(namespace: &str, object_key: &str) -> pb::ObjectKey {
        pb::ObjectKey {
            namespace: namespace.to_string(),
            object_key: object_key.to_string(),
        }
    }

    /// Construct immutable write options: the server returns `Ok(false)` if the object
    /// already exists and does not overwrite it.
    pub fn put_options_if_absent() -> pb::PutOptions {
        pb::PutOptions {
            ttl_seconds: 0,
            if_not_exists: true,
            compression: pb::CompressionType::None as i32,
        }
    }

    /// Single PUT. `data` is moved directly into the request; Vec<u8> → Bytes is a
    /// zero-copy takeover.
    pub async fn put(
        &mut self,
        namespace: &str,
        object_key: &str,
        data: Vec<u8>,
    ) -> Result<bool, tonic::Status> {
        self.put_with_options(namespace, object_key, data, None)
            .await
    }

    /// Single PUT with `PutOptions`. `Ok(false)` means the server explicitly rejected
    /// the write, typically because `if_not_exists=true` and the object already exists.
    pub async fn put_with_options(
        &mut self,
        namespace: &str,
        object_key: &str,
        data: Vec<u8>,
        options: Option<pb::PutOptions>,
    ) -> Result<bool, tonic::Status> {
        let req = pb::PutRequest {
            key: Some(Self::key(namespace, object_key)),
            data: Bytes::from(data),
            metadata: None,
            options,
        };
        let resp = self.inner.put(req).await?;
        Ok(resp.into_inner().success)
    }

    /// Single immutable PUT. Returns `Ok(false)` when the object already exists.
    pub async fn put_if_absent(
        &mut self,
        namespace: &str,
        object_key: &str,
        data: Vec<u8>,
    ) -> Result<bool, tonic::Status> {
        self.put_with_options(
            namespace,
            object_key,
            data,
            Some(Self::put_options_if_absent()),
        )
        .await
    }

    /// Single GET, returns `data` (None = not found). Bytes → Vec<u8> is a zero-copy
    /// takeover (Bytes::into fallback path).
    pub async fn get(
        &mut self,
        namespace: &str,
        object_key: &str,
    ) -> Result<Option<Vec<u8>>, tonic::Status> {
        let req = pb::GetRequest {
            key: Some(Self::key(namespace, object_key)),
        };
        let resp = self.inner.get(req).await?.into_inner();
        if resp.found {
            // resp.data is Bytes (a refcounted view owned by the gRPC framework); converting
            // to Vec<u8> is a move when we're the sole owner, otherwise it falls back to a copy.
            Ok(Some(resp.data.to_vec()))
        } else {
            Ok(None)
        }
    }

    /// Single Exists check — only tests object existence, does not fetch the value.
    pub async fn exists(
        &mut self,
        namespace: &str,
        object_key: &str,
    ) -> Result<bool, tonic::Status> {
        let req = pb::ExistsRequest {
            key: Some(Self::key(namespace, object_key)),
        };
        let resp = self.inner.exists(req).await?;
        Ok(resp.into_inner().exists)
    }

    /// Delete an object. `false` means the object was absent or no underlying bytes were removed.
    pub async fn delete(
        &mut self,
        namespace: &str,
        object_key: &str,
    ) -> Result<bool, tonic::Status> {
        let req = pb::DeleteRequest {
            key: Some(Self::key(namespace, object_key)),
        };
        let resp = self.inner.delete(req).await?;
        Ok(resp.into_inner().success)
    }

    /// Look up object descriptor and actual placement without reading the value.
    pub async fn lookup_object(
        &mut self,
        namespace: &str,
        object_key: &str,
    ) -> Result<Option<ObjectLookup>, tonic::Status> {
        let req = pb::LookupObjectRequest {
            key: Some(Self::key(namespace, object_key)),
        };
        let resp = self.inner.lookup_object(req).await?.into_inner();
        if !resp.found {
            return Ok(None);
        }
        let Some(descriptor) = resp.descriptor else {
            return Ok(None);
        };
        Ok(Some(ObjectLookup {
            descriptor,
            placement: resp.placement,
        }))
    }

    /// Batch PUT (single RPC; server writes in parallel).
    pub async fn put_batch(
        &mut self,
        items: Vec<(pb::ObjectKey, Vec<u8>)>,
    ) -> Result<Vec<bool>, tonic::Status> {
        let pb_items: Vec<pb::PutRequest> = items
            .into_iter()
            .map(|(key, data)| pb::PutRequest {
                key: Some(key),
                data: Bytes::from(data),
                metadata: None,
                options: None,
            })
            .collect();
        let resp = self
            .inner
            .put_batch(pb::PutBatchRequest { items: pb_items })
            .await?;
        Ok(resp.into_inner().success)
    }

    /// Batch GET (single RPC).
    pub async fn get_batch(
        &mut self,
        keys: Vec<pb::ObjectKey>,
    ) -> Result<Vec<Option<Vec<u8>>>, tonic::Status> {
        let resp = self
            .inner
            .get_batch(pb::GetBatchRequest { keys })
            .await?
            .into_inner();
        Ok(resp
            .results
            .into_iter()
            .map(|r| if r.found { Some(r.data.to_vec()) } else { None })
            .collect())
    }

    pub fn make_key(namespace: &str, object_key: &str) -> pb::ObjectKey {
        Self::key(namespace, object_key)
    }

    // ===== Streaming (bypasses the large single-message codec wall) =====
    // Split a large value into N smaller PutChunk messages sent as a stream. Each chunk
    // is an independent small protobuf message that goes through the prost fast path,
    // and the server accumulates the small buffers as they arrive. Mirrors the Python
    // put_objects_stream.

    /// Streaming PUT: split a large value into multiple small PutChunk messages,
    /// bypassing the slow-path decoder for 480MB single messages. `chunk_size` is the
    /// per-chunk byte cap (1–4 MB recommended).
    pub async fn put_stream(
        &mut self,
        namespace: &str,
        object_key: &str,
        data: Vec<u8>,
        chunk_size: usize,
    ) -> Result<bool, tonic::Status> {
        self.put_stream_with_options(namespace, object_key, data, chunk_size, None)
            .await
    }

    /// Streaming PUT with `PutOptions`. Options are sent only on the first PutChunk.
    pub async fn put_stream_with_options(
        &mut self,
        namespace: &str,
        object_key: &str,
        data: Vec<u8>,
        chunk_size: usize,
        options: Option<pb::PutOptions>,
    ) -> Result<bool, tonic::Status> {
        // Vec<u8> → Bytes is a zero-copy takeover; then slice into N Bytes (Arc-refcounted)
        // and delegate to the chunks variant.
        let total = data.len();
        let big = Bytes::from(data);
        let chunk_size = chunk_size.max(1);
        let n = if total == 0 {
            1
        } else {
            total.div_ceil(chunk_size)
        };
        let mut segments: Vec<Bytes> = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(total);
            segments.push(big.slice(start..end));
        }
        self.put_stream_chunks_with_options(namespace, object_key, segments, options)
            .await
    }

    /// Streaming immutable PUT. Returns `Ok(false)` when the object already exists.
    pub async fn put_stream_if_absent(
        &mut self,
        namespace: &str,
        object_key: &str,
        data: Vec<u8>,
        chunk_size: usize,
    ) -> Result<bool, tonic::Status> {
        self.put_stream_with_options(
            namespace,
            object_key,
            data,
            chunk_size,
            Some(Self::put_options_if_absent()),
        )
        .await
    }

    /// Streaming PUT passthrough (zero-copy): caller has already split the data into N
    /// Bytes segments (typical case: Python-side layer tensors each pointing at a pinned
    /// memory pool). Sends them directly, without another Vec → Bytes → slice conversion
    /// on the client.
    pub async fn put_stream_chunks(
        &mut self,
        namespace: &str,
        object_key: &str,
        segments: Vec<Bytes>,
    ) -> Result<bool, tonic::Status> {
        self.put_stream_chunks_with_options(namespace, object_key, segments, None)
            .await
    }

    /// Streaming PUT passthrough with `PutOptions`. Options are sent only on the first chunk.
    pub async fn put_stream_chunks_with_options(
        &mut self,
        namespace: &str,
        object_key: &str,
        mut segments: Vec<Bytes>,
        options: Option<pb::PutOptions>,
    ) -> Result<bool, tonic::Status> {
        if segments.is_empty() {
            segments.push(Bytes::new());
        }
        let key = Self::key(namespace, object_key);
        let total: usize = segments.iter().map(|s| s.len()).sum();
        let n_chunks = segments.len().max(1);
        let mut chunks: Vec<pb::PutChunk> = Vec::with_capacity(n_chunks);
        let mut running_offset = 0usize;
        for (i, seg) in segments.into_iter().enumerate() {
            let is_first = i == 0;
            let key_field = if is_first { Some(key.clone()) } else { None };
            let seg_len = seg.len();
            chunks.push(pb::PutChunk {
                key: key_field,
                data: seg,
                offset: running_offset as i64,
                is_last: i + 1 == n_chunks,
                metadata: None,
                total_size: if is_first { total as i64 } else { 0 },
                options: if is_first { options.clone() } else { None },
            });
            running_offset += seg_len;
        }
        let outbound = tokio_stream::iter(chunks);
        let resp = self.inner.put_stream(outbound).await?;
        Ok(resp.into_inner().success)
    }

    /// Streaming immutable PUT passthrough. Returns `Ok(false)` when the object already exists.
    pub async fn put_stream_chunks_if_absent(
        &mut self,
        namespace: &str,
        object_key: &str,
        segments: Vec<Bytes>,
    ) -> Result<bool, tonic::Status> {
        self.put_stream_chunks_with_options(
            namespace,
            object_key,
            segments,
            Some(Self::put_options_if_absent()),
        )
        .await
    }

    /// Streaming GET: consume the DataChunk stream and concatenate into a Vec<u8>
    /// (compatible with callers that expect a single buffer).
    /// Note: concatenating 480MB triggers page-fault first-touch writes, capping a
    /// single connection at ~0.4 GB/s. For high-throughput cases use
    /// `get_stream_chunks` (zero-copy, returns Vec<Bytes>).
    pub async fn get_stream(
        &mut self,
        namespace: &str,
        object_key: &str,
    ) -> Result<Option<Vec<u8>>, tonic::Status> {
        match self.get_stream_chunks(namespace, object_key).await? {
            None => Ok(None),
            Some(segments) => {
                let total: usize = segments.iter().map(|s| s.len()).sum();
                let mut buf = Vec::with_capacity(total);
                for s in &segments {
                    buf.extend_from_slice(s);
                }
                Ok(Some(buf))
            }
        }
    }

    /// Streaming GET passthrough (zero-copy): returns a Vec<Bytes> segment list without
    /// concatenating on the client. Each Bytes segment is an Arc-refcounted view of
    /// the gRPC inbound buffer; callers can:
    ///   - iterate over segments and do GPU H2D copies directly (no concat needed when
    ///     object boundaries align)
    ///   - extend into a large Vec<u8> themselves (equivalent to falling back to
    ///     get_stream). Measured: single-connection 0.4 → ~1.5 GB/s; multi-connection
    ///     aggregate approaches the server-side raw-read ceiling.
    pub async fn get_stream_chunks(
        &mut self,
        namespace: &str,
        object_key: &str,
    ) -> Result<Option<Vec<Bytes>>, tonic::Status> {
        use tokio_stream::StreamExt;
        let req = pb::GetRequest {
            key: Some(Self::key(namespace, object_key)),
        };
        let mut stream = match self.inner.get_stream(req).await {
            Ok(s) => s.into_inner(),
            Err(status) if status.code() == tonic::Code::NotFound => return Ok(None),
            Err(status) => return Err(status),
        };
        let mut segments: Vec<Bytes> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            // chunk.data is Bytes (build.rs bytes(["."])), so push is a zero-copy handoff.
            // Capacity estimate: use the first chunk's total_size to guess the segment
            // count (assumes ~4MB sub-chunks; even if off, a few Vec reallocs cost far
            // less than the 480MB payload itself).
            if segments.is_empty() && chunk.total_size > 0 {
                let est = ((chunk.total_size as usize) / (4 * 1024 * 1024)).max(8);
                segments.reserve(est);
            }
            segments.push(chunk.data);
            if chunk.is_last {
                break;
            }
        }
        Ok(Some(segments))
    }

    /// Read an object using a client-held descriptor. The server validates generation / etag /
    /// layout first. A stale descriptor returns `FAILED_PRECONDITION`; callers should lookup again.
    pub async fn read_by_descriptor_stream_chunks(
        &mut self,
        descriptor: pb::ObjectDescriptor,
        placement: Option<pb::PlacementDescriptor>,
    ) -> Result<Option<DescriptorReadChunks>, tonic::Status> {
        use tokio_stream::StreamExt;

        let req = pb::ReadByDescriptorRequest {
            descriptor: Some(descriptor.clone()),
            placement: placement.clone(),
        };
        let mut stream = match self.inner.read_by_descriptor_stream(req).await {
            Ok(s) => s.into_inner(),
            Err(status) if status.code() == tonic::Code::NotFound => return Ok(None),
            Err(status) => return Err(status),
        };

        let mut segments = Vec::new();
        let mut fresh_descriptor: Option<pb::ObjectDescriptor> = None;
        let mut fresh_placement: Option<pb::PlacementDescriptor> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if fresh_descriptor.is_none() {
                fresh_descriptor = chunk.descriptor;
            }
            if fresh_placement.is_none() {
                fresh_placement = chunk.placement;
            }
            segments.push(chunk.data);
            if chunk.is_last {
                break;
            }
        }

        let descriptor = fresh_descriptor.unwrap_or(descriptor);
        Ok(Some(DescriptorReadChunks {
            segments,
            descriptor,
            placement: fresh_placement.or(placement),
        }))
    }
}
