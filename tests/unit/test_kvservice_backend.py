from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from contextstore.storage.kvservice import KVServiceBackend


class _FakeObjectKey:
    def __init__(self, namespace: str, object_key: str) -> None:
        self.namespace = namespace
        self.object_key = object_key

    def to_string(self) -> str:
        return f"{len(self.namespace.encode())}:{self.namespace}{self.object_key}"


class _FakeClient:
    def __init__(self, chunks: list[bytes] | None) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, str, str]] = []

    def get_stream_chunks(self, kv_key: _FakeObjectKey) -> list[bytes] | None:
        self.calls.append(("get_stream_chunks", kv_key.namespace, kv_key.object_key))
        return self._chunks

    def exists(self, kv_key: _FakeObjectKey) -> bool:
        self.calls.append(("exists", kv_key.namespace, kv_key.object_key))
        return self._chunks is not None


class _FakeLookupClient(_FakeClient):
    def lookup_object(self, kv_key: _FakeObjectKey) -> object | None:
        self.calls.append(("lookup_object", kv_key.namespace, kv_key.object_key))
        return object() if self._chunks is not None else None


class _FakeDescriptor:
    def __init__(self, handle: str, generation: int = 1, size: int = 4) -> None:
        self.object_handle = handle
        self.object_generation = generation
        self.content_etag = f"etag-{handle}"
        self.layout_version = 1
        self.size = size
        self.is_striped = False
        self.stripe_count = 0
        self.chunk_size = 0


class _FakeDescriptorLookupClient(_FakeClient):
    def __init__(self, descriptors: list[_FakeDescriptor | None]) -> None:
        super().__init__([b"data"])
        self._descriptors = descriptors

    def lookup_object(self, kv_key: _FakeObjectKey) -> _FakeDescriptor | None:
        self.calls.append(("lookup_object", kv_key.namespace, kv_key.object_key))
        if not self._descriptors:
            return None
        return self._descriptors.pop(0)


class _FakeRdmaClient:
    def __init__(self) -> None:
        self.next_region_id = 10
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []
        self.closed = False

    def register_external_buffer(self, ptr: int, size: int) -> int:
        self.registered.append((ptr, size))
        region_id = self.next_region_id
        self.next_region_id += 1
        return region_id

    def unregister_external_buffer(self, region_id: int) -> None:
        self.unregistered.append(region_id)

    def close(self) -> None:
        self.closed = True


class _FakeRdmaZcClient(_FakeRdmaClient):
    def __init__(self, values: list[int | Exception]) -> None:
        super().__init__()
        self.values = values
        self.get_into_calls: list[tuple[int, str, int]] = []

    def get_into(self, region_id: int, key: str, offset: int) -> int:
        self.get_into_calls.append((region_id, key, offset))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class _FakeRdmaDescriptorZcClient(_FakeRdmaZcClient):
    supports_descriptor_get = True

    def __init__(self, values: list[int | Exception]) -> None:
        super().__init__(values)
        self.get_descriptor_into_calls: list[tuple[int, str, str, int]] = []

    def get_descriptor_into(
        self,
        region_id: int,
        key: str,
        descriptor: _FakeDescriptor,
        offset: int,
    ) -> int:
        self.get_descriptor_into_calls.append(
            (region_id, key, descriptor.object_handle, offset)
        )
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class _FakeRdmaBufferClient(_FakeRdmaClient):
    supports_descriptor_get = True

    def __init__(self, values: list[int | Exception], data: bytes = b"data") -> None:
        super().__init__()
        self.values = values
        self.data = data
        self.get_calls: list[str] = []
        self.get_descriptor_calls: list[tuple[str, str]] = []

    def get(self, key: str) -> int:
        self.get_calls.append(key)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def get_descriptor(self, key: str, descriptor: _FakeDescriptor) -> int:
        self.get_descriptor_calls.append((key, descriptor.object_handle))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def buffer_view(self, length: int) -> bytes:
        return self.data[:length]


class _FakePlacementChunk:
    def __init__(self, storage_handle: str, offset: int, length: int) -> None:
        self.storage_handle = storage_handle
        self.offset = offset
        self.length = length


class _FakePlacement:
    def __init__(self, chunks: list[_FakePlacementChunk]) -> None:
        self.chunks = chunks


class _FakeLookup:
    def __init__(self, size: int, chunks: list[_FakePlacementChunk]) -> None:
        self.descriptor = _FakeDescriptor("shared", size=size)
        self.placement = _FakePlacement(chunks)


class _FakeSharedGds:
    def __init__(self, result: int) -> None:
        self.result = result
        self.calls: list[tuple[int, int, int, list[tuple[object, int, int]]]] = []

    def read_into(
        self,
        ptr: int,
        size: int,
        device: int,
        segments: list[tuple[object, int, int]],
    ) -> int:
        self.calls.append((ptr, size, device, segments))
        return self.result


def _make_backend(client: _FakeClient) -> KVServiceBackend:
    backend = object.__new__(KVServiceBackend)
    backend._rdma_enabled = True
    backend._rdma_fallback_to_grpc = True
    backend._parallel = 1
    backend._model_id = "test-model"
    backend._client = client
    return backend


def _make_rdma_backend(client: _FakeRdmaClient) -> KVServiceBackend:
    backend = object.__new__(KVServiceBackend)
    backend._parallel = 1
    backend._get_or_create_rdma_client_pool = lambda: [client]  # type: ignore[method-assign]
    return backend


def _make_reconnect_backend(
    first_client: _FakeRdmaZcClient,
    second_client: _FakeRdmaZcClient,
) -> KVServiceBackend:
    backend = object.__new__(KVServiceBackend)
    backend._rdma_enabled = True
    backend._parallel = 1
    backend._model_id = "test-model"
    backend._rdma_lock = __import__("threading").Lock()
    backend._rdma_client_pool = [first_client]
    backend._rdma_region_cache = {(100, 1024): 10}
    backend._rdma_put_region_cache = {"old": 99}
    backend._object_key = (  # type: ignore[method-assign]
        lambda key, layer_name: _FakeObjectKey(
            backend._model_id,
            KVServiceBackend._encode_object_key(key, layer_name),
        )
    )

    def _get_pool() -> list[_FakeRdmaZcClient]:
        if not backend._rdma_client_pool:
            backend._rdma_client_pool = [second_client]
        return backend._rdma_client_pool

    backend._get_or_create_rdma_client_pool = _get_pool  # type: ignore[method-assign]
    return backend


def _make_descriptor_rdma_backend(
    rdma_client: _FakeRdmaClient,
    lookup_client: _FakeClient,
) -> KVServiceBackend:
    backend = object.__new__(KVServiceBackend)
    backend._rdma_enabled = True
    backend._rdma_fallback_to_grpc = True
    backend._parallel = 1
    backend._model_id = "test-model"
    backend._client = lookup_client
    backend._executor = ThreadPoolExecutor(max_workers=1)
    backend._get_or_create_rdma_client_pool = lambda: [rdma_client]  # type: ignore[method-assign]
    backend._object_key = (  # type: ignore[method-assign]
        lambda key, layer_name: _FakeObjectKey(
            backend._model_id,
            KVServiceBackend._encode_object_key(key, layer_name),
        )
    )
    return backend


def _make_shared_gds_backend(lookup: _FakeLookup | None, client: _FakeSharedGds) -> KVServiceBackend:
    backend = object.__new__(KVServiceBackend)
    backend._model_id = "test-model"
    backend._shared_gds_enabled = True
    backend._shared_gds_server_root = Path("/server/data")
    backend._shared_gds_mount_root = Path("/lustre/contextstore")
    backend._shared_gds_min_bytes = 1
    backend._shared_gds_client = client
    backend._shared_gds_lock = threading.Lock()
    backend._client = types.SimpleNamespace(
        lookup_object_with_placement=lambda key: lookup,
    )
    backend._object_key = lambda key, layer: _FakeObjectKey(  # type: ignore[method-assign]
        "test-model", f"{key}:{layer}"
    )
    return backend


def test_ensure_rdma_region_keeps_multiple_buffers_registered() -> None:
    client = _FakeRdmaClient()
    backend = _make_rdma_backend(client)

    first = backend.ensure_rdma_region(100, 1024)
    second = backend.ensure_rdma_region(200, 2048)
    first_again = backend.ensure_rdma_region(100, 1024)

    assert (first, second, first_again) == (10, 11, 10)
    assert client.registered == [(100, 1024), (200, 2048)]
    assert client.unregistered == []


def test_ensure_rdma_region_evicts_when_cache_limit_is_reached(monkeypatch: Any) -> None:
    monkeypatch.setenv("CS_RDMA_REGION_CACHE_LIMIT", "1")
    client = _FakeRdmaClient()
    backend = _make_rdma_backend(client)

    first = backend.ensure_rdma_region(100, 1024)
    second = backend.ensure_rdma_region(200, 2048)

    assert (first, second) == (10, 11)
    assert client.registered == [(100, 1024), (200, 2048)]
    assert client.unregistered == [10]


def test_reset_rdma_client_pool_closes_clients_and_clears_region_caches() -> None:
    client = _FakeRdmaClient()
    backend = object.__new__(KVServiceBackend)
    backend._rdma_lock = __import__("threading").Lock()
    backend._rdma_client_pool = [client]
    backend._rdma_region_cache = {(100, 1024): 10}
    backend._rdma_put_region_cache = {"old": 99}

    backend._reset_rdma_client_pool("unit-test")

    assert client.closed
    assert backend._rdma_client_pool == []
    assert backend._rdma_region_cache == {}
    assert backend._rdma_put_region_cache == {}


def test_get_chunks_into_reconnects_and_reregisters_region() -> None:
    first = _FakeRdmaZcClient([RuntimeError("stale qp")])
    second = _FakeRdmaZcClient([4096])
    backend = _make_reconnect_backend(first, second)

    assert backend.get_chunks_into("prefix", "layer", 10, 64) == 4096

    assert first.closed
    assert first.get_into_calls == [(10, "10:test-model6:prefixlayer", 64)]
    assert second.registered == [(100, 1024)]
    assert second.get_into_calls == [(10, "10:test-model6:prefixlayer", 64)]


def test_get_chunks_into_retries_with_fresh_descriptor_on_miss() -> None:
    lookup_client = _FakeDescriptorLookupClient(
        [_FakeDescriptor("old", generation=1), _FakeDescriptor("new", generation=2)]
    )
    rdma_client = _FakeRdmaDescriptorZcClient([0, 4096])
    backend = _make_descriptor_rdma_backend(rdma_client, lookup_client)
    backend._rdma_region_cache = {(100, 1024): 10}

    try:
        assert backend.get_chunks_into("prefix", "layer", 10, 64) == 4096
    finally:
        backend._executor.shutdown(wait=False)

    assert rdma_client.get_descriptor_into_calls == [
        (10, "10:test-model6:prefixlayer", "old", 64),
        (10, "10:test-model6:prefixlayer", "new", 64),
    ]
    assert rdma_client.get_into_calls == []


def test_get_chunks_rdma_uses_descriptor_get_for_internal_buffer() -> None:
    lookup_client = _FakeDescriptorLookupClient([_FakeDescriptor("current")])
    rdma_client = _FakeRdmaBufferClient([4], data=b"data")
    backend = _make_descriptor_rdma_backend(rdma_client, lookup_client)

    try:
        segments, had_error = backend._get_chunks_rdma("prefix", "layer")
    finally:
        backend._executor.shutdown(wait=False)

    assert had_error is False
    assert segments == [b"data"]
    assert rdma_client.get_descriptor_calls == [
        ("10:test-model6:prefixlayer", "current")
    ]
    assert rdma_client.get_calls == []


def test_get_chunks_falls_back_to_grpc_on_rdma_error(monkeypatch: Any) -> None:
    client = _FakeClient([b"grpc-data"])
    backend = _make_backend(client)
    backend._get_chunks_rdma = lambda key, layer_name: (None, True)  # type: ignore[method-assign]

    client_mod = types.ModuleType("contextstore.kvservice_client")
    client_mod.ObjectKey = _FakeObjectKey
    monkeypatch.setitem(sys.modules, "contextstore.kvservice_client", client_mod)

    assert backend.get_chunks("prefix", "layer") == [b"grpc-data"]
    assert client.calls == [("get_stream_chunks", "test-model", "6:prefixlayer")]


def test_probe_layer_exists_prefers_lookup_object(monkeypatch: Any) -> None:
    client = _FakeLookupClient([b"grpc-data"])
    backend = _make_backend(client)

    client_mod = types.ModuleType("contextstore.kvservice_client")
    client_mod.ObjectKey = _FakeObjectKey
    monkeypatch.setitem(sys.modules, "contextstore.kvservice_client", client_mod)

    assert backend.probe_layer_exists("prefix", "layer") is True
    assert client.calls == [("lookup_object", "test-model", "6:prefixlayer")]


def test_get_chunks_returns_miss_without_grpc_fallback(monkeypatch: Any) -> None:
    client = _FakeClient([b"grpc-data"])
    backend = _make_backend(client)
    backend._get_chunks_rdma = lambda key, layer_name: (None, False)  # type: ignore[method-assign]

    client_mod = types.ModuleType("contextstore.kvservice_client")
    client_mod.ObjectKey = _FakeObjectKey
    monkeypatch.setitem(sys.modules, "contextstore.kvservice_client", client_mod)

    assert backend.get_chunks("prefix", "layer") is None
    assert client.calls == []


def test_get_chunks_raises_when_grpc_fallback_disabled(monkeypatch: Any) -> None:
    client = _FakeClient([b"grpc-data"])
    backend = _make_backend(client)
    backend._rdma_fallback_to_grpc = False
    backend._get_chunks_rdma = lambda key, layer_name: (None, True)  # type: ignore[method-assign]

    client_mod = types.ModuleType("contextstore.kvservice_client")
    client_mod.ObjectKey = _FakeObjectKey
    monkeypatch.setitem(sys.modules, "contextstore.kvservice_client", client_mod)

    try:
        backend.get_chunks("prefix", "layer")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "rdma_fallback_to_grpc is disabled" in str(exc)
    assert client.calls == []


def test_put_chunks_raises_when_grpc_fallback_disabled(monkeypatch: Any) -> None:
    client = _FakeClient([b"grpc-data"])
    backend = _make_backend(client)
    backend._rdma_fallback_to_grpc = False
    backend._try_rdma_put = lambda key, layer_name, segments, total: False  # type: ignore[method-assign]
    client_mod = types.ModuleType("contextstore.kvservice_client")
    client_mod.ObjectKey = _FakeObjectKey
    monkeypatch.setitem(sys.modules, "contextstore.kvservice_client", client_mod)

    try:
        backend.put_chunks("prefix", "layer", [b"x" * (5 * 1024 * 1024)])
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "RDMA put_chunks failed" in str(exc)


def test_shared_gds_maps_single_placement_into_local_mount() -> None:
    lookup = _FakeLookup(6, [_FakePlacementChunk("/server/data/a/object.bin", 0, 6)])
    gds = _FakeSharedGds(6)
    backend = _make_shared_gds_backend(lookup, gds)

    assert backend.get_chunks_to_gpu("prefix", "layer", 0x1000, 64, 6, 0) == 6
    assert gds.calls == [
        (
            0x1000,
            64,
            0,
            [(Path("/lustre/contextstore/a/object.bin"), 0, 6)],
        )
    ]


def test_shared_gds_preserves_stripe_offsets() -> None:
    lookup = _FakeLookup(
        8,
        [
            _FakePlacementChunk("/server/data/a/stripe-1", 4, 4),
            _FakePlacementChunk("/server/data/a/stripe-0", 0, 4),
        ],
    )
    gds = _FakeSharedGds(8)
    backend = _make_shared_gds_backend(lookup, gds)

    assert backend.get_chunks_to_gpu("prefix", "layer", 0x1000, 64, 8, 1) == 8
    assert gds.calls[0][3] == [
        (Path("/lustre/contextstore/a/stripe-0"), 0, 4),
        (Path("/lustre/contextstore/a/stripe-1"), 4, 4),
    ]


def test_shared_gds_rejects_handle_outside_configured_server_root() -> None:
    lookup = _FakeLookup(4, [_FakePlacementChunk("/other/data/object.bin", 0, 4)])
    gds = _FakeSharedGds(4)
    backend = _make_shared_gds_backend(lookup, gds)

    assert backend.get_chunks_to_gpu("prefix", "layer", 0x1000, 64, 4, 0) is None
    assert gds.calls == []
