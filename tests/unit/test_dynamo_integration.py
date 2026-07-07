from __future__ import annotations

import importlib
import sys
import types
from typing import Any


def _install_vllm_stubs(monkeypatch: Any) -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.uint8 = "uint8"
    torch_mod.long = "long"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    base_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class FakeKVConnectorBase:
        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None) -> None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._role = role
            self._kv_cache_config = kv_cache_config

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

    for name in [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)


def _reload_connector_module(monkeypatch: Any) -> Any:
    _install_vllm_stubs(monkeypatch)
    for name in [
        "contextstore.connector",
        "contextstore.integrations.dynamo",
        "contextstore.integrations.dynamo.connector",
    ]:
        sys.modules.pop(name, None)
    return importlib.import_module("contextstore.integrations.dynamo.connector")


def test_dynamo_connector_class_is_contextstore_plugin(monkeypatch: Any) -> None:
    module = _reload_connector_module(monkeypatch)
    contextstore_connector = importlib.import_module("contextstore.connector")

    assert issubclass(module.DynamoConnector, contextstore_connector.ContextStoreConnector)


def test_normalize_extra_config_accepts_dynamo_dotted_keys(monkeypatch: Any) -> None:
    module = _reload_connector_module(monkeypatch)

    normalized = module.normalize_extra_config(
        {
            "contextstore.endpoint": "127.0.0.1:50051",
            "contextstore.model_id": "qwen",
            "contextstore.parallel_channels": 8,
            "contextstore.rdma_enabled": True,
        }
    )

    assert normalized["kv_service_endpoint"] == "127.0.0.1:50051"
    assert normalized["model_id"] == "qwen"
    assert normalized["kv_service_parallel_channels"] == 8
    assert normalized["rdma_enabled"] is True


def test_normalize_extra_config_accepts_nested_contextstore_block(monkeypatch: Any) -> None:
    module = _reload_connector_module(monkeypatch)

    normalized = module.normalize_extra_config(
        {
            "contextstore": {
                "endpoints": "10.0.0.1:50051,10.0.0.2:50051",
                "timeout_ms": 1000,
            },
            "kv_service_timeout_ms": 2000,
        }
    )

    assert normalized["kv_service_endpoints"] == [
        "10.0.0.1:50051",
        "10.0.0.2:50051",
    ]
    assert normalized["kv_service_timeout_ms"] == 2000


def test_dynamo_connector_mutates_vllm_extra_config_before_super(monkeypatch: Any) -> None:
    module = _reload_connector_module(monkeypatch)
    engine_mod = importlib.import_module("contextstore.connector")

    class FakeEngine:
        def __init__(self, config: Any) -> None:
            self.config = config

    monkeypatch.setattr(engine_mod, "ContextStoreEngine", FakeEngine)

    kv_transfer_config = types.SimpleNamespace(
        kv_connector_extra_config={
            "contextstore.endpoint": "127.0.0.1:50051",
            "contextstore.model_id": "qwen",
        }
    )
    vllm_config = types.SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=types.SimpleNamespace(block_size=16),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
    )

    connector = module.DynamoConnector(vllm_config, role="scheduler")

    assert connector._cs_config.kv_service_endpoint == "127.0.0.1:50051"
    assert connector._cs_config.model_id == "qwen"
