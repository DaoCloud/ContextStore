from __future__ import annotations

import time
from typing import Sequence

import torch

from contextstore.core.codec import KVCodec, NoOpCodec
from contextstore.core.config import ContextStoreConfig
from contextstore.core.metrics import ContextStoreMetrics
from contextstore.index.prefix_index import PrefixIndex
from contextstore.storage.base import BlockMeta, StorageBackend
from contextstore.storage.local import LocalStorageBackend
from contextstore.storage.memory import MemoryStorageBackend


def _slice_csv(csv: str, n: int, idx: int) -> str:
    """Slice a comma-separated multi-node config (e.g. rdma_server_addr) to pick element idx in endpoint order.

    When element count == n, take one per endpoint (one NIC endpoint per child); when element
    count == 1, all children share the same value; otherwise fall back to idx modulo. The returned
    single element may still contain commas (further split inside a child to represent that child's
    multiple NICs), which does not conflict with KVServiceBackend's comma parsing (that only applies
    to rdma_server_addr/rdma_device).
    """
    parts = [s.strip() for s in csv.split(",") if s.strip()]
    if not parts:
        return csv
    if len(parts) == n:
        return parts[idx]
    if len(parts) == 1:
        return parts[0]
    return parts[idx % len(parts)]


class ContextStoreEngine:
    def __init__(self, config: ContextStoreConfig, storage: StorageBackend | None = None, prefix_index=None):
        self._config = config

        if prefix_index is not None:
            self._index = prefix_index
        elif config.prefix_index_url:
            from contextstore.index.prefix_index_redis import RedisPrefixIndex
            self._index = RedisPrefixIndex(
                model_id=config.model_id,
                block_size=config.block_size,
                redis_url=config.prefix_index_url,
            )
        else:
            self._index = PrefixIndex(
                model_id=config.model_id,
                block_size=config.block_size,
            )

        if storage is not None:
            self._storage = storage
        else:
            endpoints = list(config.kv_service_endpoints) or (
                [config.kv_service_endpoint] if config.kv_service_endpoint else []
            )
            if len(endpoints) >= 2:
                # ===== Dual-node HA: one independent child KVServiceBackend per endpoint =====
                from contextstore.storage.ha_kvservice import HaKVServiceBackend
                from contextstore.storage.kvservice import KVServiceBackend

                children = [
                    KVServiceBackend(
                        endpoint=ep,
                        model_id=config.model_id or "default",
                        chunk_size_mb=config.kv_service_chunk_size_mb,
                        # HA children use a shorter timeout: when one node is down, PROBE/health
                        # trials fail fast instead of blocking on the default 30s.
                        timeout_ms=getattr(config, "ha_child_timeout_ms", 2000),
                        parallel_channels=config.kv_service_parallel_channels,
                        rdma_enabled=getattr(config, "rdma_enabled", False),
                        rdma_server_addr=_slice_csv(
                            getattr(config, "rdma_server_addr", "127.0.0.1:50053"),
                            len(endpoints), i,
                        ),
                        rdma_device=_slice_csv(
                            getattr(config, "rdma_device", "mlx5_0"), len(endpoints), i,
                        ),
                        rdma_port=getattr(config, "rdma_port", 1),
                        rdma_gid_index=getattr(config, "rdma_gid_index", 3),
                        rdma_buf_size_mb=getattr(config, "rdma_buf_size_mb", 512),
                        rdma_fallback_to_grpc=getattr(config, "rdma_fallback_to_grpc", True),
                        shared_gds_enabled=getattr(config, "shared_gds_enabled", False),
                        shared_gds_server_root=getattr(config, "shared_gds_server_root", ""),
                        shared_gds_mount_root=getattr(config, "shared_gds_mount_root", ""),
                        shared_gds_min_bytes=getattr(config, "shared_gds_min_bytes", 1024 * 1024),
                        shared_gds_file_cache_capacity=getattr(config, "shared_gds_file_cache_capacity", 128),
                        shared_gds_buffer_cache_capacity=getattr(config, "shared_gds_buffer_cache_capacity", 2),
                        shared_gds_library_path=getattr(config, "shared_gds_library_path", ""),
                    )
                    for i, ep in enumerate(endpoints)
                ]
                # zero-copy / rdma-put capability is derived from config (rdma_enabled + parallel==1),
                # not from a live AND across active children, to avoid a single flapping node
                # shutting down the entire RDMA path.
                rdma_zc = (
                    getattr(config, "rdma_enabled", False)
                    and config.kv_service_parallel_channels == 1
                )
                l2: StorageBackend = HaKVServiceBackend(
                    children,
                    cooldown_s=getattr(config, "ha_breaker_cooldown_s", 5.0),
                    supports_zerocopy_hint=rdma_zc,
                    supports_rdma_put_hint=rdma_zc,
                )
            elif len(endpoints) == 1:
                # L2 = remote Rust KV Service (gRPC). Highest priority: when an endpoint is
                # explicitly configured, it replaces the local Sharded/Local backend.
                # L1 (HostMemoryBackend) is still governed by host_memory_capacity_gb.
                from contextstore.storage.kvservice import KVServiceBackend
                l2 = KVServiceBackend(
                    endpoint=endpoints[0],
                    model_id=config.model_id or "default",
                    chunk_size_mb=config.kv_service_chunk_size_mb,
                    timeout_ms=config.kv_service_timeout_ms,
                    parallel_channels=config.kv_service_parallel_channels,
                    rdma_enabled=getattr(config, "rdma_enabled", False),
                    rdma_server_addr=getattr(config, "rdma_server_addr", "127.0.0.1:50053"),
                    rdma_device=getattr(config, "rdma_device", "mlx5_0"),
                    rdma_port=getattr(config, "rdma_port", 1),
                    rdma_gid_index=getattr(config, "rdma_gid_index", 3),
                    rdma_buf_size_mb=getattr(config, "rdma_buf_size_mb", 512),
                    rdma_fallback_to_grpc=getattr(config, "rdma_fallback_to_grpc", True),
                    shared_gds_enabled=getattr(config, "shared_gds_enabled", False),
                    shared_gds_server_root=getattr(config, "shared_gds_server_root", ""),
                    shared_gds_mount_root=getattr(config, "shared_gds_mount_root", ""),
                    shared_gds_min_bytes=getattr(config, "shared_gds_min_bytes", 1024 * 1024),
                    shared_gds_file_cache_capacity=getattr(config, "shared_gds_file_cache_capacity", 128),
                    shared_gds_buffer_cache_capacity=getattr(config, "shared_gds_buffer_cache_capacity", 2),
                    shared_gds_library_path=getattr(config, "shared_gds_library_path", ""),
                )
            elif config.sharded_device_paths and len(config.sharded_device_paths) > 1:
                from contextstore.storage.sharded import ShardedStorageBackend
                l2 = ShardedStorageBackend(
                    device_paths=config.sharded_device_paths,
                    max_capacity_bytes_per_device=config.max_capacity_bytes // len(config.sharded_device_paths),
                    stripe_policy=config.sharded_stripe_policy,
                    io_depth=config.sharded_io_depth,
                )
            else:
                l2 = LocalStorageBackend(
                    storage_path=config.storage_path,
                    max_capacity_bytes=config.max_capacity_bytes,
                )
            if config.host_memory_capacity_gb > 0:
                from contextstore.storage.host_memory import HostMemoryBackend
                self._storage = HostMemoryBackend(
                    wrapped=l2,
                    max_capacity_bytes=int(config.host_memory_capacity_gb * 1024**3),
                )
            else:
                self._storage = l2

        if config.enable_compression:
            self._codec: KVCodec | NoOpCodec = KVCodec(level=config.compression_level)
        else:
            self._codec = NoOpCodec()
        self._metrics = ContextStoreMetrics()

    @property
    def index(self) -> PrefixIndex:
        return self._index

    @property
    def storage(self) -> StorageBackend:
        return self._storage

    @property
    def metrics(self) -> ContextStoreMetrics:
        return self._metrics

    def lookup(self, token_ids: Sequence[int]) -> int:
        matched = self._index.lookup_prefix(token_ids)
        self._metrics.record_request()
        if matched > 0:
            self._metrics.record_hit(matched)
        else:
            self._metrics.record_miss(len(token_ids))
        return matched

    def save_layer(
        self,
        block_keys: list[str],
        layer_name: str,
        kv_data: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_ids: Sequence[int],
    ) -> None:
        t0 = time.perf_counter()
        total_bytes = 0
        num_tokens_per_block = self._config.block_size

        all_slots = slot_mapping[:len(block_keys) * num_tokens_per_block]
        if kv_data.dim() >= 3 and kv_data.shape[0] == 2:
            all_kv_cpu = kv_data[:, all_slots, ...].cpu()
        else:
            all_kv_cpu = kv_data[all_slots, ...].cpu()

        for i, key in enumerate(block_keys):
            start = i * num_tokens_per_block
            end = min(start + num_tokens_per_block, all_slots.shape[0])
            if all_kv_cpu.dim() >= 3 and all_kv_cpu.shape[0] == 2:
                block_kv = all_kv_cpu[:, start:end, ...]
            else:
                block_kv = all_kv_cpu[start:end, ...]
            data_bytes = self._codec.encode_to_bytes(block_kv)
            total_bytes += len(data_bytes)
            meta = BlockMeta(
                num_tokens=end - start,
                num_layers=1,
                dtype=str(kv_data.dtype),
                shape=list(block_kv.shape),
                compressed=self._config.enable_compression,
                compression_level=self._config.compression_level,
            )
            self._storage.put(key, layer_name, data_bytes, meta)
        elapsed = (time.perf_counter() - t0) * 1000
        self._metrics.record_save(total_bytes, elapsed)

    def load_layer(
        self,
        block_keys: list[str],
        layer_name: str,
        target_kv: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        t0 = time.perf_counter()
        total_bytes = 0
        num_tokens_per_block = self._config.block_size

        raw_chunks = self._storage.get_parallel(block_keys, layer_name)
        valid_chunks: list[bytes] = []
        for chunk in raw_chunks:
            if chunk is None:
                break
            valid_chunks.append(chunk)
            total_bytes += len(chunk)

        if not valid_chunks:
            elapsed = (time.perf_counter() - t0) * 1000
            self._metrics.record_load(0, elapsed)
            return

        total_tokens = len(valid_chunks) * num_tokens_per_block
        decoded_blocks = [self._codec.decode_from_bytes(chunk) for chunk in valid_chunks]
        all_kv_cpu = torch.cat(decoded_blocks, dim=-2 if decoded_blocks[0].dim() >= 3 and decoded_blocks[0].shape[0] == 2 else 0)

        total_slots = min(total_tokens, slot_mapping.shape[0])
        slots = slot_mapping[:total_slots]

        is_contiguous = total_slots > 0 and slots[0].item() == 0 and (total_slots == 1 or slots[-1].item() == total_slots - 1)

        all_kv_gpu = all_kv_cpu.to(target_kv.device, non_blocking=True)

        if is_contiguous:
            if target_kv.dim() >= 3 and target_kv.shape[0] == 2:
                target_kv[:, :total_slots, ...] = all_kv_gpu
            else:
                target_kv[:total_slots, ...] = all_kv_gpu
        else:
            if target_kv.dim() >= 3 and target_kv.shape[0] == 2:
                target_kv[:, slots, ...] = all_kv_gpu
            else:
                target_kv[slots, ...] = all_kv_gpu

        elapsed = (time.perf_counter() - t0) * 1000
        self._metrics.record_load(total_bytes, elapsed)

    def save_layer_bulk(
        self,
        prefix_key: str,
        layer_name: str,
        kv_data: torch.Tensor,
        num_tokens: int,
    ) -> None:
        t0 = time.perf_counter()
        if kv_data.dim() >= 3 and kv_data.shape[0] == 2:
            kv_cpu = kv_data[:, :num_tokens, ...].cpu()
        else:
            kv_cpu = kv_data[:num_tokens, ...].cpu()
        data_bytes = self._codec.encode_to_bytes(kv_cpu)
        meta = BlockMeta(
            num_tokens=num_tokens,
            num_layers=1,
            dtype=str(kv_data.dtype),
            shape=list(kv_cpu.shape),
            compressed=self._config.enable_compression,
            compression_level=self._config.compression_level,
        )
        if hasattr(self._storage, 'put_tensor'):
            self._storage.put_tensor(prefix_key, layer_name, kv_cpu, data_bytes, meta)
        else:
            self._storage.put(prefix_key, layer_name, data_bytes, meta)
        elapsed = (time.perf_counter() - t0) * 1000
        self._metrics.record_save(len(data_bytes), elapsed)

    def load_layer_bulk(
        self,
        prefix_key: str,
        layer_name: str,
        target_kv: torch.Tensor,
        num_tokens: int,
    ) -> bool:
        t0 = time.perf_counter()

        kv_cpu = None
        if hasattr(self._storage, 'get_tensor'):
            kv_cpu = self._storage.get_tensor(prefix_key, layer_name)

        if kv_cpu is not None:
            kv_gpu = kv_cpu.to(target_kv.device, non_blocking=True)
            nbytes = kv_cpu.numel() * kv_cpu.element_size()
        else:
            data_bytes = self._storage.get(prefix_key, layer_name)
            if data_bytes is None:
                self._metrics.record_load(0, (time.perf_counter() - t0) * 1000)
                return False
            kv_cpu = self._codec.decode_from_bytes(data_bytes)
            kv_gpu = kv_cpu.to(target_kv.device, non_blocking=True)
            nbytes = len(data_bytes)

        if target_kv.dim() >= 3 and target_kv.shape[0] == 2:
            target_kv[:, :num_tokens, ...] = kv_gpu
        else:
            target_kv[:num_tokens, ...] = kv_gpu
        elapsed = (time.perf_counter() - t0) * 1000
        self._metrics.record_load(nbytes, elapsed)
        return True

    def register_prefix(self, token_ids: Sequence[int], num_tokens: int) -> str:
        return self._index.register_prefix(token_ids, num_tokens)
