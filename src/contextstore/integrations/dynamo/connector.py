from __future__ import annotations

"""Dynamo/vLLM KVConnector plugin entry point for ContextStore.

This module is loaded through vLLM's standard external connector mechanism:

    --kv-transfer-config '{"kv_connector":"DynamoConnector",
      "kv_connector_module_path":"contextstore.integrations.dynamo.connector",
      "kv_role":"kv_both"}'

The class name intentionally matches Dynamo's KVBM connector name so Dynamo's
vLLM integration recognizes it as a KVBM-compatible connector, while the
implementation reuses ContextStore's existing vLLM connector.
"""

from typing import Any

from contextstore.connector import ContextStoreConnector


_DOTTED_ALIASES = {
    "contextstore.endpoint": "kv_service_endpoint",
    "contextstore.endpoints": "kv_service_endpoints",
    "contextstore.model_id": "model_id",
    "contextstore.parallel_channels": "kv_service_parallel_channels",
    "contextstore.chunk_size_mb": "kv_service_chunk_size_mb",
    "contextstore.timeout_ms": "kv_service_timeout_ms",
    "contextstore.host_memory_capacity_gb": "host_memory_capacity_gb",
    "contextstore.prefix_index_url": "prefix_index_url",
    "contextstore.max_capacity_gb": "max_capacity_gb",
    "contextstore.storage_path": "storage_path",
    "contextstore.rdma_enabled": "rdma_enabled",
    "contextstore.rdma_server_addr": "rdma_server_addr",
    "contextstore.rdma_device": "rdma_device",
    "contextstore.rdma_port": "rdma_port",
    "contextstore.rdma_gid_index": "rdma_gid_index",
    "contextstore.rdma_buf_size_mb": "rdma_buf_size_mb",
    "contextstore.rdma_fallback_to_grpc": "rdma_fallback_to_grpc",
}

_NESTED_ALIASES = {
    "endpoint": "kv_service_endpoint",
    "endpoints": "kv_service_endpoints",
    "model_id": "model_id",
    "parallel_channels": "kv_service_parallel_channels",
    "chunk_size_mb": "kv_service_chunk_size_mb",
    "timeout_ms": "kv_service_timeout_ms",
    "host_memory_capacity_gb": "host_memory_capacity_gb",
    "prefix_index_url": "prefix_index_url",
    "max_capacity_gb": "max_capacity_gb",
    "storage_path": "storage_path",
    "rdma_enabled": "rdma_enabled",
    "rdma_server_addr": "rdma_server_addr",
    "rdma_device": "rdma_device",
    "rdma_port": "rdma_port",
    "rdma_gid_index": "rdma_gid_index",
    "rdma_buf_size_mb": "rdma_buf_size_mb",
    "rdma_fallback_to_grpc": "rdma_fallback_to_grpc",
}


def _normalize_endpoints(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if value is not None and str(value).strip() else []


def normalize_extra_config(extra_config: dict[str, Any] | None) -> dict[str, Any]:
    """Return ContextStore-native config with Dynamo-friendly aliases expanded."""
    normalized = dict(extra_config or {})

    nested = normalized.get("contextstore")
    if isinstance(nested, dict):
        for key, target in _NESTED_ALIASES.items():
            if key in nested and target not in normalized:
                normalized[target] = nested[key]

    for key, target in _DOTTED_ALIASES.items():
        if key in normalized and target not in normalized:
            normalized[target] = normalized[key]

    if "kv_service_endpoints" in normalized:
        normalized["kv_service_endpoints"] = _normalize_endpoints(
            normalized["kv_service_endpoints"]
        )

    if "kv_service_endpoint" not in normalized:
        endpoints = normalized.get("kv_service_endpoints")
        if isinstance(endpoints, list) and len(endpoints) == 1:
            normalized["kv_service_endpoint"] = endpoints[0]

    return normalized


class DynamoConnector(ContextStoreConnector):
    """ContextStore connector exposed under Dynamo's KVBM connector name."""

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None) -> None:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        if kv_transfer_config is not None:
            kv_transfer_config.kv_connector_extra_config = normalize_extra_config(
                getattr(kv_transfer_config, "kv_connector_extra_config", None)
            )
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )


__all__ = ["DynamoConnector", "normalize_extra_config"]

