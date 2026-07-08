from __future__ import annotations

"""ContextStore KV Service - Python client."""

import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterable, Iterator

import grpc

from .types import (
    DataReadResult,
    HealthStatus,
    KVMetadata,
    ObjectDescriptor,
    ObjectKey,
    ObjectLookupResult,
    PlacementChunk,
    PlacementDescriptor,
    PutOptions,
    StatsSnapshot,
)


def _get_ipc_handle(tensor) -> tuple[bytes, int, int]:
    """
    Extract a CUDA IPC mem handle from a CUDA tensor.

    Returns:
        (handle_bytes_64, nbytes, device_ordinal)

    Raises:
        RuntimeError: tensor is not on GPU or is not contiguous.
    """
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError("get_to_gpu/put_from_gpu requires torch") from e

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("gpu_tensor must be a torch.Tensor")
    if not tensor.is_cuda:
        raise RuntimeError("gpu_tensor must be on a CUDA device")
    if not tensor.is_contiguous():
        raise RuntimeError("gpu_tensor must be contiguous (call .contiguous())")

    storage = tensor.untyped_storage()
    # _share_cuda_() returns:
    #   (device, handle_bytes, storage_size_bytes, storage_offset_bytes,
    #    ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
    share = storage._share_cuda_()
    device = int(share[0])
    handle = bytes(share[1])
    if len(handle) != 64:
        raise RuntimeError(
            f"unexpected CUDA IPC handle length: {len(handle)} (expected 64)"
        )
    nbytes = int(tensor.nbytes)
    return handle, nbytes, device


# Make the generated _pb module importable via relative import
_HERE = os.path.dirname(os.path.abspath(__file__))
_PB_DIR = os.path.join(_HERE, "_pb")
if _PB_DIR not in sys.path:
    sys.path.insert(0, _PB_DIR)

try:
    from contextstore.kvservice_client._pb import kv_service_pb2 as pb  # type: ignore
    from contextstore.kvservice_client._pb import kv_service_pb2_grpc as pb_grpc  # type: ignore
except ImportError:  # pragma: no cover
    raise ImportError(
        "generated protobuf code not found. Run first: "
        "make proto-python (under the kv-service/ directory)"
    )


class KVClient:
    """gRPC client wrapper for ContextStore KV Service."""

    def __init__(
        self,
        endpoint: str = "localhost:50051",
        timeout_ms: int = 5000,
        max_message_mb: int = 2048,
        descriptor_cache_capacity: int = 4096,
    ):
        self.endpoint = endpoint
        self.timeout = timeout_ms / 1000.0
        self._timeout_ms = timeout_ms
        self._max_message_mb = max_message_mb
        self._descriptor_cache_capacity = max(0, descriptor_cache_capacity)
        self._descriptor_cache: OrderedDict[str, ObjectLookupResult] = OrderedDict()
        self._data_node_clients: dict[str, KVClient] = {}
        self._descriptor_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cs-desc",
        )
        self._chunk_executor = ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="cs-chunk",
        )
        max_message_bytes = min(max_message_mb * 1024 * 1024, 2_147_483_647)
        options = [
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
            # Relax HTTP/2 frame/window limits to improve single-stream throughput:
            # default keepalive is off, initial window is 64KB -> streaming large messages
            # frequently blocks on the flow-control window.
            # Set the window to 64MB so long streams are not throttled by flow control.
            ("grpc.http2.max_frame_size", 16 * 1024 * 1024),  # 16MB frames (default 16KB)
            ("grpc.http2.bdp_probe", 1),  # automatic BDP probing to tune the window
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),  # avoid channel pool sharing (KVServiceBackend already creates N independent clients internally)
        ]
        self._channel = grpc.insecure_channel(endpoint, options=options)
        self._stub = pb_grpc.KVServiceStub(self._channel)

    def close(self) -> None:
        self._descriptor_executor.shutdown(wait=False, cancel_futures=True)
        self._chunk_executor.shutdown(wait=False, cancel_futures=True)
        for client in self._data_node_clients.values():
            client.close()
        self._data_node_clients.clear()
        self._channel.close()

    def __enter__(self) -> "KVClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ===== Helpers =====
    @staticmethod
    def _to_pb_key(k: ObjectKey) -> "pb.ObjectKey":
        return pb.ObjectKey(
            namespace=k.namespace,
            object_key=k.object_key,
        )

    @staticmethod
    def _to_pb_meta(m: KVMetadata | None) -> "pb.KVMetadata | None":
        if m is None:
            return None
        return pb.KVMetadata(
            num_tokens=m.num_tokens,
            num_layers=m.num_layers,
            dtype=m.dtype,
            shape=m.shape,
            compressed=m.compressed,
            compression_level=m.compression_level,
            created_at=m.created_at,
            last_accessed_at=m.last_accessed_at,
        )

    @staticmethod
    def _from_pb_meta(m: "pb.KVMetadata") -> KVMetadata:
        return KVMetadata(
            num_tokens=m.num_tokens,
            num_layers=m.num_layers,
            dtype=m.dtype,
            shape=list(m.shape),
            compressed=m.compressed,
            compression_level=m.compression_level,
            created_at=m.created_at,
            last_accessed_at=m.last_accessed_at,
        )

    @staticmethod
    def _to_pb_descriptor(d: ObjectDescriptor) -> "pb.ObjectDescriptor":
        return pb.ObjectDescriptor(
            key=KVClient._to_pb_key(d.key),
            object_handle=d.object_handle,
            object_generation=d.object_generation,
            content_etag=d.content_etag,
            layout_version=d.layout_version,
            size=d.size,
            is_striped=d.is_striped,
            stripe_count=d.stripe_count,
            chunk_size=d.chunk_size,
        )

    @staticmethod
    def _from_pb_descriptor(d: "pb.ObjectDescriptor") -> ObjectDescriptor:
        return ObjectDescriptor(
            key=ObjectKey(
                namespace=d.key.namespace,
                object_key=d.key.object_key,
            ),
            object_handle=d.object_handle,
            object_generation=d.object_generation,
            content_etag=d.content_etag,
            layout_version=d.layout_version,
            size=d.size,
            is_striped=d.is_striped,
            stripe_count=d.stripe_count,
            chunk_size=d.chunk_size,
        )

    @staticmethod
    def _to_pb_placement(p: PlacementDescriptor | None) -> "pb.PlacementDescriptor | None":
        if p is None:
            return None
        return pb.PlacementDescriptor(
            key=KVClient._to_pb_key(p.key),
            placement_epoch=p.placement_epoch,
            placement_policy_id=p.placement_policy_id,
            layout_hash=p.layout_hash,
            primary_node_id=p.primary_node_id,
            primary_grpc_endpoint=p.primary_grpc_endpoint,
            primary_rdma_endpoint=p.primary_rdma_endpoint,
            chunks=[
                pb.PlacementChunk(
                    stripe_index=chunk.stripe_index,
                    node_id=chunk.node_id,
                    grpc_endpoint=chunk.grpc_endpoint,
                    rdma_endpoint=chunk.rdma_endpoint,
                    device_id=chunk.device_id,
                    storage_handle=chunk.storage_handle,
                    offset=chunk.offset,
                    length=chunk.length,
                )
                for chunk in p.chunks
            ],
        )

    @staticmethod
    def _to_pb_chunk(chunk: PlacementChunk) -> "pb.PlacementChunk":
        return pb.PlacementChunk(
            stripe_index=chunk.stripe_index,
            node_id=chunk.node_id,
            grpc_endpoint=chunk.grpc_endpoint,
            rdma_endpoint=chunk.rdma_endpoint,
            device_id=chunk.device_id,
            storage_handle=chunk.storage_handle,
            offset=chunk.offset,
            length=chunk.length,
        )

    @staticmethod
    def _from_pb_placement(p: "pb.PlacementDescriptor") -> PlacementDescriptor:
        return PlacementDescriptor(
            key=ObjectKey(
                namespace=p.key.namespace,
                object_key=p.key.object_key,
            ),
            placement_epoch=p.placement_epoch,
            placement_policy_id=p.placement_policy_id,
            layout_hash=p.layout_hash,
            primary_node_id=p.primary_node_id,
            primary_grpc_endpoint=p.primary_grpc_endpoint,
            primary_rdma_endpoint=p.primary_rdma_endpoint,
            chunks=[
                PlacementChunk(
                    stripe_index=chunk.stripe_index,
                    node_id=chunk.node_id,
                    grpc_endpoint=chunk.grpc_endpoint,
                    rdma_endpoint=chunk.rdma_endpoint,
                    device_id=chunk.device_id,
                    storage_handle=chunk.storage_handle,
                    offset=chunk.offset,
                    length=chunk.length,
                )
                for chunk in p.chunks
            ],
        )

    @staticmethod
    def _descriptor_cache_key(key: ObjectKey) -> str:
        return key.to_string()

    @staticmethod
    def _same_descriptor_identity(a: ObjectDescriptor, b: ObjectDescriptor) -> bool:
        return (
            a.object_handle == b.object_handle
            and a.object_generation == b.object_generation
            and a.content_etag == b.content_etag
            and a.layout_version == b.layout_version
            and a.size == b.size
        )

    @staticmethod
    def _same_content_identity(a: ObjectDescriptor, b: ObjectDescriptor) -> bool:
        return (
            a.object_generation == b.object_generation
            and a.content_etag == b.content_etag
            and a.size == b.size
        )

    def _cached_lookup(self, key: ObjectKey) -> ObjectLookupResult | None:
        if self._descriptor_cache_capacity == 0:
            return None
        cache_key = self._descriptor_cache_key(key)
        result = self._descriptor_cache.get(cache_key)
        if result is None:
            return None
        self._descriptor_cache.move_to_end(cache_key)
        return result

    def _cached_descriptor(self, key: ObjectKey) -> ObjectDescriptor | None:
        result = self._cached_lookup(key)
        return result.descriptor if result is not None else None

    def _cache_descriptor(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None = None,
    ) -> None:
        if self._descriptor_cache_capacity == 0:
            return
        cache_key = self._descriptor_cache_key(descriptor.key)
        if placement is None:
            cached = self._descriptor_cache.get(cache_key)
            if cached is not None:
                placement = cached.placement
        self._descriptor_cache[cache_key] = ObjectLookupResult(
            descriptor=descriptor,
            placement=placement,
        )
        self._descriptor_cache.move_to_end(cache_key)
        while len(self._descriptor_cache) > self._descriptor_cache_capacity:
            self._descriptor_cache.popitem(last=False)

    def _cache_lookup(self, result: ObjectLookupResult) -> None:
        self._cache_descriptor(result.descriptor, result.placement)

    def _evict_descriptor(self, key: ObjectKey) -> None:
        self._descriptor_cache.pop(self._descriptor_cache_key(key), None)

    def _data_client_for_placement(
        self,
        placement: PlacementDescriptor | None,
    ) -> "KVClient":
        """Pick a data-node client based on placement; return self when there is no placement or the endpoint matches this node."""
        if placement is None or not placement.primary_grpc_endpoint:
            return self
        endpoint = placement.primary_grpc_endpoint
        if endpoint == self.endpoint:
            return self
        client = self._data_node_clients.get(endpoint)
        if client is None:
            client = KVClient(
                endpoint=endpoint,
                timeout_ms=self._timeout_ms,
                max_message_mb=self._max_message_mb,
                descriptor_cache_capacity=0,
            )
            self._data_node_clients[endpoint] = client
        return client

    def _data_client_for_endpoint(self, endpoint: str) -> "KVClient":
        if not endpoint or endpoint == self.endpoint:
            return self
        client = self._data_node_clients.get(endpoint)
        if client is None:
            client = KVClient(
                endpoint=endpoint,
                timeout_ms=self._timeout_ms,
                max_message_mb=self._max_message_mb,
                descriptor_cache_capacity=0,
            )
            self._data_node_clients[endpoint] = client
        return client

    @staticmethod
    def _placement_requires_chunk_fanout(placement: PlacementDescriptor | None) -> bool:
        if placement is None or len(placement.chunks) <= 1:
            return False
        endpoints = {
            chunk.grpc_endpoint or placement.primary_grpc_endpoint
            for chunk in placement.chunks
        }
        return len(endpoints) > 1

    def _read_placement_chunk_bytes(
        self,
        descriptor: ObjectDescriptor,
        chunk: PlacementChunk,
        fallback_endpoint: str,
    ) -> tuple[int, bytes]:
        endpoint = chunk.grpc_endpoint or fallback_endpoint
        client = self._data_client_for_endpoint(endpoint)
        stream = client._stub.ReadPlacementChunk(
            pb.ReadPlacementChunkRequest(
                descriptor=self._to_pb_descriptor(descriptor),
                chunk=self._to_pb_chunk(chunk),
            ),
            timeout=self.timeout,
        )
        parts: list[bytes] = []
        for part in stream:
            parts.append(part.data)
            if part.is_last:
                break
        return chunk.stripe_index, b"".join(parts)

    def _read_chunks_by_placement(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor,
    ) -> list[bytes]:
        futures = [
            self._chunk_executor.submit(
                self._read_placement_chunk_bytes,
                descriptor,
                chunk,
                placement.primary_grpc_endpoint,
            )
            for chunk in placement.chunks
        ]
        indexed = [future.result() for future in futures]
        indexed.sort(key=lambda item: item[0])
        return [data for _, data in indexed]

    def _read_by_descriptor_via_placement(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None,
    ) -> DataReadResult | None:
        client = self._data_client_for_placement(placement)
        return client.read_by_descriptor(descriptor, placement)

    def _read_stream_by_descriptor_via_placement(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None,
    ) -> tuple[list[bytes], ObjectDescriptor, PlacementDescriptor | None] | None:
        client = self._data_client_for_placement(placement)
        return client.read_by_descriptor_stream_chunks(descriptor, placement)

    # ===== Health / Stats =====
    def health(self) -> HealthStatus:
        resp = self._stub.Health(pb.HealthRequest(), timeout=self.timeout)
        status_name = pb.HealthResponse.ServingStatus.Name(resp.status)
        return HealthStatus(
            status=status_name,
            version=resp.version,
            is_serving=resp.status == pb.HealthResponse.SERVING,
        )

    def stats(self) -> StatsSnapshot:
        resp = self._stub.Stats(pb.StatsRequest(), timeout=self.timeout)
        return StatsSnapshot(
            l1_cache_hits=resp.l1_cache_hits,
            l1_cache_misses=resp.l1_cache_misses,
            l1_cache_size_bytes=resp.l1_cache_size_bytes,
            l2_reads_total=resp.l2_reads_total,
            l2_writes_total=resp.l2_writes_total,
            l2_bytes_read=resp.l2_bytes_read,
            l2_bytes_written=resp.l2_bytes_written,
            metadata_entries=resp.metadata_entries,
        )

    # ===== Single-object =====
    def get(self, key: ObjectKey) -> tuple[bytes, KVMetadata] | None:
        resp = self._stub.Get(
            pb.GetRequest(key=self._to_pb_key(key)),
            timeout=self.timeout,
        )
        if not resp.found:
            return None
        return resp.data, self._from_pb_meta(resp.metadata)

    def put(
        self,
        key: ObjectKey,
        data: bytes,
        metadata: KVMetadata | None = None,
        options: PutOptions | None = None,
    ) -> bool:
        pb_opts = None
        if options:
            pb_opts = pb.PutOptions(
                ttl_seconds=options.ttl_seconds,
                if_not_exists=options.if_not_exists,
                compression=pb.CompressionType.Value(options.compression),
            )
        resp = self._stub.Put(
            pb.PutRequest(
                key=self._to_pb_key(key),
                data=data,
                metadata=self._to_pb_meta(metadata),
                options=pb_opts,
            ),
            timeout=self.timeout,
        )
        if resp.success:
            self._evict_descriptor(key)
        return resp.success

    def delete(self, key: ObjectKey) -> bool:
        resp = self._stub.Delete(
            pb.DeleteRequest(key=self._to_pb_key(key)),
            timeout=self.timeout,
        )
        if resp.success:
            self._evict_descriptor(key)
        return resp.success

    def exists(self, key: ObjectKey) -> bool:
        resp = self._stub.Exists(
            pb.ExistsRequest(key=self._to_pb_key(key)),
            timeout=self.timeout,
        )
        return resp.exists

    def lookup_object_with_placement(self, key: ObjectKey) -> ObjectLookupResult | None:
        """Query only the object descriptor and its actual placement; do not read the value."""
        resp = self._stub.LookupObject(
            pb.LookupObjectRequest(key=self._to_pb_key(key)),
            timeout=self.timeout,
        )
        if not resp.found:
            return None
        placement = None
        if resp.HasField("placement"):
            placement = self._from_pb_placement(resp.placement)
        return ObjectLookupResult(
            descriptor=self._from_pb_descriptor(resp.descriptor),
            placement=placement,
        )

    def lookup_object(self, key: ObjectKey) -> ObjectDescriptor | None:
        """Query only the object descriptor; do not read the value."""
        result = self.lookup_object_with_placement(key)
        return result.descriptor if result is not None else None

    def read_by_descriptor(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None = None,
    ) -> DataReadResult | None:
        """Read an object by descriptor. gRPC returns FAILED_PRECONDITION when the descriptor is stale."""
        pb_placement = self._to_pb_placement(placement)
        req = pb.ReadByDescriptorRequest(
            descriptor=self._to_pb_descriptor(descriptor),
        )
        if pb_placement is not None:
            req.placement.CopyFrom(pb_placement)
        resp = self._stub.ReadByDescriptor(
            req,
            timeout=self.timeout,
        )
        if not resp.found:
            return None
        fresh_placement = None
        if resp.HasField("placement"):
            fresh_placement = self._from_pb_placement(resp.placement)
        return DataReadResult(
            data=resp.data,
            metadata=self._from_pb_meta(resp.metadata),
            descriptor=self._from_pb_descriptor(resp.descriptor),
            placement=fresh_placement,
        )

    # ===== Batch (server-side parallel I/O) =====
    def get_batch(
        self, keys: list[ObjectKey]
    ) -> list[tuple[bytes, KVMetadata] | None]:
        req = pb.GetBatchRequest(keys=[self._to_pb_key(k) for k in keys])
        resp = self._stub.GetBatch(req, timeout=self.timeout)
        out: list[tuple[bytes, KVMetadata] | None] = []
        for r in resp.results:
            if r.found:
                out.append((r.data, self._from_pb_meta(r.metadata)))
            else:
                out.append(None)
        return out

    def put_batch(
        self,
        items: list[tuple[ObjectKey, bytes, KVMetadata | None]],
    ) -> list[bool]:
        pb_items = []
        for k, d, m in items:
            pb_items.append(
                pb.PutRequest(
                    key=self._to_pb_key(k),
                    data=d,
                    metadata=self._to_pb_meta(m),
                )
            )
        resp = self._stub.PutBatch(
            pb.PutBatchRequest(items=pb_items),
            timeout=self.timeout,
        )
        results = list(resp.success)
        for (key, _, _), ok in zip(items, results):
            if ok:
                self._evict_descriptor(key)
        return results

    # ===== Stream (large value) =====
    def get_stream(self, key: ObjectKey, chunk_buffer: int = 4 * 1024 * 1024) -> bytes:
        stream = self._stub.GetStream(
            pb.GetRequest(key=self._to_pb_key(key)),
            timeout=self.timeout,
        )
        chunks = []
        for ch in stream:
            chunks.append(ch.data)
            if ch.is_last:
                break
        return b"".join(chunks)

    def get_stream_chunks(self, key: ObjectKey) -> list[bytes]:
        """Streaming GET pass-through: return the chunk list without concatenating into a single bytes.

        Intended for callers that want to handle chunk boundaries themselves (e.g. the
        vLLM connector writes each chunk back into a GPU block without needing a
        480MB intermediate concatenation buffer).

        Raises grpc.RpcError(NOT_FOUND) if the key does not exist — caller must catch it.
        """
        stream = self._stub.GetStream(
            pb.GetRequest(key=self._to_pb_key(key)),
            timeout=self.timeout,
        )
        chunks: list[bytes] = []
        for ch in stream:
            chunks.append(ch.data)
            if ch.is_last:
                break
        return chunks

    def read_by_descriptor_stream_chunks(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None = None,
    ) -> tuple[list[bytes], ObjectDescriptor, PlacementDescriptor | None] | None:
        """Stream-read an object by descriptor; return the chunk list and the server's fresh descriptor."""
        if self._placement_requires_chunk_fanout(placement):
            assert placement is not None
            return self._read_chunks_by_placement(descriptor, placement), descriptor, placement

        pb_placement = self._to_pb_placement(placement)
        req = pb.ReadByDescriptorRequest(
            descriptor=self._to_pb_descriptor(descriptor),
        )
        if pb_placement is not None:
            req.placement.CopyFrom(pb_placement)
        stream = self._stub.ReadByDescriptorStream(
            req,
            timeout=self.timeout,
        )
        chunks: list[bytes] = []
        fresh: ObjectDescriptor | None = None
        fresh_placement: PlacementDescriptor | None = None
        for ch in stream:
            if fresh is None and ch.HasField("descriptor"):
                fresh = self._from_pb_descriptor(ch.descriptor)
            if fresh_placement is None and ch.HasField("placement"):
                fresh_placement = self._from_pb_placement(ch.placement)
            chunks.append(ch.data)
            if ch.is_last:
                break
        if fresh is None:
            fresh = descriptor
        if fresh_placement is None:
            fresh_placement = placement
        return chunks, fresh, fresh_placement

    def get_stream_chunks_cached(
        self,
        key: ObjectKey,
        strict_validation: bool = True,
    ) -> list[bytes] | None:
        """Streaming GET with descriptor cache.

        On cache miss, first LookupObject then ReadByDescriptor; on cache hit,
        fire LookupObject and ReadByDescriptor concurrently. Content identity must
        match before returning; if the content version changed, automatically re-read
        with the latest descriptor.
        """
        cached = self._cached_lookup(key)
        if cached is None:
            return self._read_chunks_after_lookup(key)

        lookup_future = self._descriptor_executor.submit(self.lookup_object_with_placement, key)
        read_future = self._descriptor_executor.submit(
            self._read_stream_by_descriptor_via_placement,
            cached.descriptor,
            cached.placement,
        )

        lookup_error: Exception | None = None
        fresh_lookup: ObjectLookupResult | None = None
        try:
            fresh_lookup = lookup_future.result()
        except Exception as exc:  # pragma: no cover - transport specific
            lookup_error = exc

        read_error: Exception | None = None
        read_result: tuple[list[bytes], ObjectDescriptor, PlacementDescriptor | None] | None = None
        try:
            read_result = read_future.result()
        except Exception as exc:
            read_error = exc

        if fresh_lookup is None:
            if lookup_error is not None:
                if strict_validation:
                    raise lookup_error
                if read_result is not None:
                    chunks, read_descriptor, read_placement = read_result
                    self._cache_descriptor(read_descriptor, read_placement)
                    return chunks
            self._evict_descriptor(key)
            return None

        self._cache_lookup(fresh_lookup)
        fresh_descriptor = fresh_lookup.descriptor

        if read_error is not None:
            if (
                isinstance(read_error, grpc.RpcError)
                and read_error.code()
                in (grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.NOT_FOUND)
            ):
                return self._read_chunks_by_fresh_lookup(fresh_lookup)
            raise read_error

        if read_result is None:
            return self._read_chunks_by_fresh_lookup(fresh_lookup)

        chunks, read_descriptor, _read_placement = read_result
        if self._same_descriptor_identity(read_descriptor, fresh_descriptor):
            return chunks

        if self._same_content_identity(read_descriptor, fresh_descriptor):
            # Layout-only change: data read with the old descriptor is still valid; refresh cache to the latest layout.
            self._cache_lookup(fresh_lookup)
            return chunks

        # Content version changed; must re-read with the latest descriptor.
        return self._read_chunks_by_fresh_lookup(fresh_lookup)

    def get_cached(self, key: ObjectKey, strict_validation: bool = True) -> DataReadResult | None:
        """Single-object GET with descriptor cache."""
        cached = self._cached_lookup(key)
        if cached is None:
            lookup = self.lookup_object_with_placement(key)
            if lookup is None:
                self._evict_descriptor(key)
                return None
            self._cache_lookup(lookup)
            return self._read_by_fresh_lookup(lookup)

        lookup_future = self._descriptor_executor.submit(self.lookup_object_with_placement, key)
        read_future = self._descriptor_executor.submit(
            self._read_by_descriptor_via_placement,
            cached.descriptor,
            cached.placement,
        )

        lookup_error: Exception | None = None
        fresh_lookup: ObjectLookupResult | None = None
        try:
            fresh_lookup = lookup_future.result()
        except Exception as exc:  # pragma: no cover - transport specific
            lookup_error = exc

        read_error: Exception | None = None
        read_result: DataReadResult | None = None
        try:
            read_result = read_future.result()
        except Exception as exc:
            read_error = exc

        if fresh_lookup is None:
            if lookup_error is not None:
                if strict_validation:
                    raise lookup_error
                if read_result is not None:
                    self._cache_descriptor(read_result.descriptor, read_result.placement)
                    return read_result
            self._evict_descriptor(key)
            return None

        self._cache_lookup(fresh_lookup)
        fresh_descriptor = fresh_lookup.descriptor

        if read_error is not None:
            if (
                isinstance(read_error, grpc.RpcError)
                and read_error.code()
                in (grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.NOT_FOUND)
            ):
                return self._read_by_fresh_lookup(fresh_lookup)
            raise read_error

        if read_result is None:
            return self._read_by_fresh_lookup(fresh_lookup)

        if self._same_descriptor_identity(read_result.descriptor, fresh_descriptor):
            return read_result

        if self._same_content_identity(read_result.descriptor, fresh_descriptor):
            self._cache_lookup(fresh_lookup)
            return read_result

        return self._read_by_fresh_lookup(fresh_lookup)

    def _read_chunks_after_lookup(self, key: ObjectKey) -> list[bytes] | None:
        lookup = self.lookup_object_with_placement(key)
        if lookup is None:
            self._evict_descriptor(key)
            return None
        self._cache_lookup(lookup)
        return self._read_chunks_by_fresh_lookup(lookup)

    def _read_chunks_by_fresh_lookup(
        self,
        lookup: ObjectLookupResult,
    ) -> list[bytes] | None:
        return self._read_chunks_by_fresh_descriptor(lookup.descriptor, lookup.placement)

    def _read_chunks_by_fresh_descriptor(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None = None,
    ) -> list[bytes] | None:
        result = self._read_stream_by_descriptor_via_placement(descriptor, placement)
        if result is None:
            self._evict_descriptor(descriptor.key)
            return None
        chunks, fresh, fresh_placement = result
        self._cache_descriptor(fresh, fresh_placement)
        return chunks

    def _read_by_fresh_lookup(
        self,
        lookup: ObjectLookupResult,
    ) -> DataReadResult | None:
        return self._read_by_fresh_descriptor(lookup.descriptor, lookup.placement)

    def _read_by_fresh_descriptor(
        self,
        descriptor: ObjectDescriptor,
        placement: PlacementDescriptor | None = None,
    ) -> DataReadResult | None:
        result = self._read_by_descriptor_via_placement(descriptor, placement)
        if result is None:
            self._evict_descriptor(descriptor.key)
            return None
        self._cache_descriptor(result.descriptor, result.placement)
        return result

    def put_stream(
        self,
        key: ObjectKey,
        data_iter: Iterable[bytes],
        metadata: KVMetadata | None = None,
    ) -> bool:
        def request_iter() -> Iterator["pb.PutChunk"]:
            first = True
            offset = 0
            data_list = list(data_iter)
            for i, chunk in enumerate(data_list):
                is_last = i == len(data_list) - 1
                req = pb.PutChunk(
                    data=chunk,
                    offset=offset,
                    is_last=is_last,
                )
                if first:
                    req.key.CopyFrom(self._to_pb_key(key))
                    pb_meta = self._to_pb_meta(metadata)
                    if pb_meta is not None:
                        req.metadata.CopyFrom(pb_meta)
                    first = False
                offset += len(chunk)
                yield req

        resp = self._stub.PutStream(request_iter(), timeout=self.timeout)
        if resp.success:
            self._evict_descriptor(key)
        return resp.success

    # ===== Multi-object streaming (large-object optimization path) =====
    # Split multiple objects into independent streaming RPCs, and further chunk each
    # object into small protobuf messages, bypassing upb's slow path for large messages
    # caused by stuffing the whole batch into a single huge message.
    @staticmethod
    def _chunk_bytes(data: bytes, chunk_size: int) -> Iterator[bytes]:
        if not data:
            yield b""
            return
        for off in range(0, len(data), chunk_size):
            yield data[off : off + chunk_size]

    def put_objects_stream(
        self,
        items: list[tuple[ObjectKey, bytes, KVMetadata | None]],
        chunk_size: int = 1024 * 1024,
        max_workers: int = 8,
    ) -> list[bool]:
        """Parallel streaming write of multiple objects. Each object is chunked and sent via PutStream, avoiding a single huge message.

        Args:
            items: [(object_key, data, meta), ...].
            chunk_size: Byte cap per PutChunk (default 1MB).
            max_workers: Number of concurrent streams.
        Returns:
            A bool list the same length as items (whether each object succeeded).
        """
        def _one(key: ObjectKey, data: bytes, meta: KVMetadata | None) -> bool:
            return self.put_stream(key, self._chunk_bytes(data, chunk_size), meta)

        if max_workers <= 1 or len(items) <= 1:
            return [_one(k, d, m) for k, d, m in items]

        from concurrent.futures import ThreadPoolExecutor

        results: list[bool] = [False] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(_one, k, d, m): i
                for i, (k, d, m) in enumerate(items)
            }
            for fut, i in futs.items():
                results[i] = fut.result()
        return results

    def get_objects_stream(
        self,
        keys: list[ObjectKey],
        max_workers: int = 8,
    ) -> list[bytes]:
        """Parallel streaming read of multiple objects. Each object is received via GetStream as a DataChunk stream and then concatenated.

        Returns:
            A bytes list the same length as keys.
        """
        def _one(key: ObjectKey) -> bytes:
            return self.get_stream(key)

        if max_workers <= 1 or len(keys) <= 1:
            return [_one(k) for k in keys]

        from concurrent.futures import ThreadPoolExecutor

        results: list[bytes] = [b""] * len(keys)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_one, k): i for i, k in enumerate(keys)}
            for fut, i in futs.items():
                results[i] = fut.result()
        return results

    # ===== GPU zero-copy (GDS + CUDA IPC) =====
    # Requires a torch + CUDA environment; only effective when the client and server run on the same host and the same GPU.
    def get_to_gpu(
        self,
        key: ObjectKey,
        gpu_tensor: "torch.Tensor",  # noqa: F821
    ) -> tuple[int, KVMetadata] | None:
        """
        DMA data from NVMe directly into the GPU buffer occupied by ``gpu_tensor``.

        Args:
            gpu_tensor: A pre-allocated CUDA tensor (must be contiguous + on GPU).
                       The server writes data into nbytes bytes starting at tensor.data_ptr().

        Returns:
            (bytes_read, metadata), or None if the key does not exist.
        """
        handle, size, device = _get_ipc_handle(gpu_tensor)
        resp = self._stub.GetToGpu(
            pb.GetToGpuRequest(
                key=self._to_pb_key(key),
                ipc_handle=handle,
                gpu_device=device,
                buf_size=size,
                buf_offset=0,
            ),
            timeout=self.timeout,
        )
        if not resp.found:
            return None
        return int(resp.bytes_read), self._from_pb_meta(resp.metadata)

    def put_from_gpu(
        self,
        key: ObjectKey,
        gpu_tensor: "torch.Tensor",  # noqa: F821
        metadata: KVMetadata | None = None,
    ) -> bool:
        """
        DMA the contents of ``gpu_tensor`` directly from GPU into NVMe (zero-copy).
        """
        handle, size, device = _get_ipc_handle(gpu_tensor)
        resp = self._stub.PutFromGpu(
            pb.PutFromGpuRequest(
                key=self._to_pb_key(key),
                ipc_handle=handle,
                gpu_device=device,
                buf_size=size,
                buf_offset=0,
                metadata=self._to_pb_meta(metadata),
            ),
            timeout=self.timeout,
        )
        return resp.success

    @contextmanager
    def session(self):
        """Context manager (equivalent to close(); convenient for use with the with-statement)."""
        try:
            yield self
        finally:
            self.close()
