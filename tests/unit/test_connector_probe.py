from __future__ import annotations

import importlib
import sys
import types
from typing import Any


def _install_connector_import_stubs(monkeypatch: Any) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.uint8 = "uint8"
    torch_mod.long = "long"
    cuda_mod = types.SimpleNamespace(
        Stream=lambda *args, **kwargs: None,
        stream=lambda stream: _NullContext(),
        current_stream=lambda: None,
    )
    torch_mod.cuda = cuda_mod
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    base_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class FakeKVConnectorBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._kv_transfer_config = types.SimpleNamespace(kv_connector_extra_config={})

        def bind_connector_metadata(self, meta: Any) -> None:
            self._connector_metadata = meta

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

    class FakeKVConnectorMetadata:
        ...

    class FakeKVConnectorRole:
        SCHEDULER = "scheduler"
        WORKER = "worker"

    base_mod.KVConnectorBase_V1 = FakeKVConnectorBase
    base_mod.KVConnectorMetadata = FakeKVConnectorMetadata
    base_mod.KVConnectorRole = FakeKVConnectorRole

    module_names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _FakeStorage:
    def __init__(self, existing_layers: set[str], parallel: int = 1) -> None:
        self._parallel = parallel
        self.checked: list[tuple[str, str]] = []
        self._existing_layers = existing_layers

    def probe_layer_exists(self, key: str, layer_name: str) -> bool:
        self.checked.append((key, layer_name))
        return layer_name in self._existing_layers


class _FakeIndex:
    def __init__(self, matched: int = 0) -> None:
        self._matched = matched

    def lookup_prefix(self, _token_ids: list[int]) -> int:
        return self._matched


class _FakeEngine:
    def __init__(self, storage: _FakeStorage) -> None:
        self.storage = storage
        self.index = _FakeIndex()
        self.registered: list[int] = []

    def lookup(self, _token_ids: list[int]) -> int:
        return 0

    def register_prefix(self, _token_ids: list[int], num_tokens: int) -> None:
        self.registered.append(num_tokens)
        self.index._matched = num_tokens


class _StrictZerocopyStorage:
    def __init__(self) -> None:
        self.get_chunks_calls = 0

    def supports_zerocopy(self) -> bool:
        return True

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        return 1

    def get_chunks_into(self, key: str, layer_name: str, region_id: int, offset: int) -> int | None:
        return None

    def get_chunks(self, key: str, layer_name: str) -> list[bytes] | None:
        self.get_chunks_calls += 1
        return [b"unexpected"]


class _FakePinned:
    def __init__(self, size: int = 128) -> None:
        self._size = size
        self.numpy_calls = 0

    def data_ptr(self) -> int:
        return 123456

    def numel(self) -> int:
        return self._size

    def numpy(self) -> "_FakePinned":
        self.numpy_calls += 1
        return self

    def tobytes(self) -> bytes:
        return b"unexpected"


class _FakeTensor:
    dtype = "float16"


class _StrictRdmaPutStorage:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.regions: list[tuple[int, int]] = []
        self.put_from_calls: list[tuple[str, str, int, int, int]] = []
        self.put_chunks_calls = 0

    def supports_rdma_put(self) -> bool:
        return True

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        self.regions.append((ptr, size))
        return 7

    def put_chunks_from(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
        size: int,
    ) -> bool:
        self.put_from_calls.append((key, layer_name, region_id, offset, size))
        return self.ok

    def put_chunks(self, key: str, layer_name: str, segments: list[bytes], meta: Any) -> None:
        self.put_chunks_calls += 1


def _load_connector(monkeypatch: Any) -> Any:
    _install_connector_import_stubs(monkeypatch)
    sys.modules.pop("contextstore.connector", None)
    return importlib.import_module("contextstore.connector")


def test_probe_layer_names_cover_all_tp_ranks(monkeypatch: Any) -> None:
    connector = _load_connector(monkeypatch)
    config = connector.ContextStoreConfig(model_id="m", block_size=16)
    engine = _FakeEngine(_FakeStorage(existing_layers=set(), parallel=8))
    scheduler = connector._SchedulerImpl(config, engine, tp_size=4)

    assert scheduler._probe_layer_names(2) == [
        "blocks_2_tp0_p0of8",
        "blocks_2_tp1_p0of8",
        "blocks_2_tp2_p0of8",
        "blocks_2_tp3_p0of8",
    ]


def test_probe_requires_every_tp_rank_before_hit(monkeypatch: Any) -> None:
    connector = _load_connector(monkeypatch)
    config = connector.ContextStoreConfig(model_id="m", block_size=16)
    storage = _FakeStorage(existing_layers={"blocks_2_tp0"}, parallel=1)
    engine = _FakeEngine(storage)
    scheduler = connector._SchedulerImpl(config, engine, tp_size=4)
    request = types.SimpleNamespace(prompt_token_ids=list(range(32)))

    matched, async_load = scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0)

    assert matched == 0
    assert async_load is False
    expected_key = connector._prefix_key("m", list(range(32)), 32)
    assert storage.checked == [
        (expected_key, "blocks_2_tp0"),
        (expected_key, "blocks_2_tp1"),
        (expected_key, "blocks_2_tp2"),
        (expected_key, "blocks_2_tp3"),
    ]
    assert engine.registered == []


def test_worker_zerocopy_failure_raises_when_grpc_fallback_disabled(monkeypatch: Any) -> None:
    connector = _load_connector(monkeypatch)
    config = connector.ContextStoreConfig(
        model_id="m",
        block_size=16,
        rdma_enabled=True,
        rdma_fallback_to_grpc=False,
    )
    storage = _StrictZerocopyStorage()
    engine = types.SimpleNamespace(storage=storage)
    worker = connector._WorkerImpl(config, engine)
    worker._make_layer_name = lambda spec: "blocks_1_tp0"  # type: ignore[method-assign]
    worker._load_blocks_from_service_zerocopy = (  # type: ignore[method-assign]
        lambda spec, layer_name, t_start, wall_start_ns: False
    )
    spec = connector._Spec(req_id="req-1", prefix_key="abcdef0123456789", gpu_block_ids=[1], num_tokens=16)

    try:
        worker._load_blocks_from_service(spec)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "rdma_fallback_to_grpc is disabled" in str(exc)
    assert storage.get_chunks_calls == 0


def test_worker_store_uses_zerocopy_rdma_put(monkeypatch: Any) -> None:
    connector = _load_connector(monkeypatch)
    storage = _StrictRdmaPutStorage(ok=True)
    worker = object.__new__(connector._WorkerImpl)
    worker._cs_config = connector.ContextStoreConfig(model_id="m", block_size=16)
    worker._engine = types.SimpleNamespace(storage=storage)
    worker._layer_names = ["layer0"]
    worker._kv_caches = {"layer0": _FakeTensor()}
    pinned = _FakePinned(size=128)
    worker._spec_byte_layout = lambda n_blocks: ([64], [0, 64], 64)  # type: ignore[method-assign]
    worker._acquire_pinned_from_pool = lambda total: pinned  # type: ignore[method-assign]
    worker._release_pinned_to_pool = lambda buf: None  # type: ignore[method-assign]
    worker._copy_spec_to_pinned = lambda spec, pinned_buf: (pinned, 64, 1.5)  # type: ignore[method-assign]
    worker._make_layer_name = lambda spec: "blocks_1_tp0"  # type: ignore[method-assign]
    spec = connector._Spec(req_id="req-1", prefix_key="abcdef0123456789", gpu_block_ids=[1], num_tokens=16)

    worker._store_blocks_to_service(spec)

    assert storage.regions == [(123456, 128)]
    assert storage.put_from_calls == [("abcdef0123456789", "blocks_1_tp0", 7, 0, 64)]
    assert storage.put_chunks_calls == 0
    assert pinned.numpy_calls == 0


def test_worker_store_strict_zerocopy_put_failure_does_not_fallback(monkeypatch: Any) -> None:
    connector = _load_connector(monkeypatch)
    storage = _StrictRdmaPutStorage(ok=False)
    worker = object.__new__(connector._WorkerImpl)
    worker._cs_config = connector.ContextStoreConfig(
        model_id="m",
        block_size=16,
        rdma_fallback_to_grpc=False,
    )
    worker._engine = types.SimpleNamespace(storage=storage)
    worker._layer_names = ["layer0"]
    worker._kv_caches = {"layer0": _FakeTensor()}
    pinned = _FakePinned(size=128)
    worker._spec_byte_layout = lambda n_blocks: ([64], [0, 64], 64)  # type: ignore[method-assign]
    worker._acquire_pinned_from_pool = lambda total: pinned  # type: ignore[method-assign]
    worker._release_pinned_to_pool = lambda buf: None  # type: ignore[method-assign]
    worker._copy_spec_to_pinned = lambda spec, pinned_buf: (pinned, 64, 1.5)  # type: ignore[method-assign]
    worker._make_layer_name = lambda spec: "blocks_1_tp0"  # type: ignore[method-assign]
    spec = connector._Spec(req_id="req-1", prefix_key="abcdef0123456789", gpu_block_ids=[1], num_tokens=16)

    try:
        worker._store_blocks_to_service(spec)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "RDMA put_chunks_from failed" in str(exc)
    assert storage.put_chunks_calls == 0
    assert pinned.numpy_calls == 0
