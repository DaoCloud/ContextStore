from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from contextstore.storage.base import BlockMeta, StorageBackend
from contextstore.storage.local import LocalStorageBackend


class ShardedStorageBackend(StorageBackend):
    """Multi-NVMe sharded storage with parallel I/O for bandwidth aggregation.

    Distributes KV blocks across multiple storage paths (NVMe devices) and
    issues parallel reads/writes to aggregate bandwidth beyond single-device limits.
    """

    def __init__(
        self,
        device_paths: list[str],
        max_capacity_bytes_per_device: int,
        stripe_policy: str = "block",
        io_depth: int = 4,
    ):
        if not device_paths:
            raise ValueError("device_paths must contain at least one path")
        self._shards: list[LocalStorageBackend] = [
            LocalStorageBackend(
                storage_path=path,
                max_capacity_bytes=max_capacity_bytes_per_device,
            )
            for path in device_paths
        ]
        self._num_shards = len(self._shards)
        self._stripe_policy = stripe_policy
        self._io_depth = io_depth
        self._pool = ThreadPoolExecutor(
            max_workers=self._num_shards * io_depth,
            thread_name_prefix="sharded_io",
        )
        self._lock = threading.Lock()

    def _shard_for_key(self, key: str, layer_name: str = "") -> int:
        if self._stripe_policy == "layer" and layer_name:
            h = int(hashlib.md5(layer_name.encode()).hexdigest(), 16)
            return h % self._num_shards
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return h % self._num_shards

    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        shard_id = self._shard_for_key(key, layer_name)
        self._shards[shard_id].put(key, layer_name, data, meta)

    def put_parallel(
        self,
        items: list[tuple[str, str, bytes, BlockMeta | None]],
    ) -> None:
        futures = []
        for key, layer_name, data, meta in items:
            shard_id = self._shard_for_key(key, layer_name)
            fut = self._pool.submit(self._shards[shard_id].put, key, layer_name, data, meta)
            futures.append(fut)
        for fut in as_completed(futures):
            fut.result()

    def get(self, key: str, layer_name: str) -> bytes | None:
        shard_id = self._shard_for_key(key, layer_name)
        return self._shards[shard_id].get(key, layer_name)

    def get_parallel(self, keys: list[str], layer_name: str) -> list[bytes | None]:
        if not keys:
            return []
        futures_map: dict[int, Any] = {}
        for i, key in enumerate(keys):
            shard_id = self._shard_for_key(key, layer_name)
            fut = self._pool.submit(self._shards[shard_id].get, key, layer_name)
            futures_map[i] = fut
        results: list[bytes | None] = [None] * len(keys)
        for i, fut in futures_map.items():
            results[i] = fut.result()
        return results

    def exists(self, key: str) -> bool:
        shard_id = self._shard_for_key(key)
        return self._shards[shard_id].exists(key)

    def delete(self, key: str) -> None:
        for shard in self._shards:
            shard.delete(key)

    def get_meta(self, key: str) -> BlockMeta | None:
        shard_id = self._shard_for_key(key)
        return self._shards[shard_id].get_meta(key)

    def capacity_usage(self) -> tuple[int, int]:
        total_used = 0
        total_cap = 0
        for shard in self._shards:
            used, cap = shard.capacity_usage()
            total_used += used
            total_cap += cap
        return total_used, total_cap

    def list_keys(self) -> list[str]:
        all_keys: set[str] = set()
        for shard in self._shards:
            all_keys.update(shard.list_keys())
        return list(all_keys)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
