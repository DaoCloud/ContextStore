from __future__ import annotations

import threading
from collections import OrderedDict

import torch

from contextstore.storage.base import BlockMeta, StorageBackend


class HostMemoryBackend(StorageBackend):
    def __init__(self, wrapped: StorageBackend, max_capacity_bytes: int):
        self._wrapped = wrapped
        self._max_capacity = max_capacity_bytes
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._tensor_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self._meta_cache: dict[str, BlockMeta] = {}
        self._used_bytes = 0
        self._lock = threading.Lock()

    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        self._wrapped.put(key, layer_name, data, meta)
        with self._lock:
            cache_key = (key, layer_name)
            if cache_key in self._cache:
                self._used_bytes -= len(self._cache[cache_key])
                self._cache.move_to_end(cache_key)
            self._cache[cache_key] = data
            self._used_bytes += len(data)
            self._tensor_cache.pop(cache_key, None)
            if meta is not None:
                self._meta_cache[key] = meta
            self._evict_if_needed()

    def put_tensor(self, key: str, layer_name: str, tensor: torch.Tensor, data: bytes, meta: BlockMeta | None = None) -> None:
        self._wrapped.put(key, layer_name, data, meta)
        pinned = tensor.pin_memory() if not tensor.is_pinned() else tensor
        with self._lock:
            cache_key = (key, layer_name)
            nbytes = pinned.numel() * pinned.element_size()
            if cache_key in self._cache:
                self._used_bytes -= len(self._cache[cache_key])
            elif cache_key in self._tensor_cache:
                old_t = self._tensor_cache[cache_key]
                self._used_bytes -= old_t.numel() * old_t.element_size()
            self._tensor_cache[cache_key] = pinned
            self._tensor_cache.move_to_end(cache_key)
            self._cache.pop(cache_key, None)
            self._used_bytes += nbytes
            if meta is not None:
                self._meta_cache[key] = meta
            self._evict_if_needed()

    def get_tensor(self, key: str, layer_name: str) -> torch.Tensor | None:
        with self._lock:
            cache_key = (key, layer_name)
            if cache_key in self._tensor_cache:
                self._tensor_cache.move_to_end(cache_key)
                return self._tensor_cache[cache_key]
        return None

    def get(self, key: str, layer_name: str) -> bytes | None:
        with self._lock:
            cache_key = (key, layer_name)
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
        data = self._wrapped.get(key, layer_name)
        if data is not None:
            with self._lock:
                cache_key = (key, layer_name)
                self._cache[cache_key] = data
                self._cache.move_to_end(cache_key)
                self._used_bytes += len(data)
                self._evict_if_needed()
        return data

    def exists(self, key: str) -> bool:
        with self._lock:
            for (k, _) in self._cache:
                if k == key:
                    return True
            for (k, _) in self._tensor_cache:
                if k == key:
                    return True
        return self._wrapped.exists(key)

    def delete(self, key: str) -> None:
        with self._lock:
            to_remove = [ck for ck in self._cache if ck[0] == key]
            for ck in to_remove:
                self._used_bytes -= len(self._cache[ck])
                del self._cache[ck]
            to_remove_t = [ck for ck in self._tensor_cache if ck[0] == key]
            for ck in to_remove_t:
                t = self._tensor_cache[ck]
                self._used_bytes -= t.numel() * t.element_size()
                del self._tensor_cache[ck]
            self._meta_cache.pop(key, None)
        self._wrapped.delete(key)

    def get_meta(self, key: str) -> BlockMeta | None:
        with self._lock:
            if key in self._meta_cache:
                return self._meta_cache[key]
        return self._wrapped.get_meta(key)

    def capacity_usage(self) -> tuple[int, int]:
        with self._lock:
            return self._used_bytes, self._max_capacity

    def list_keys(self) -> list[str]:
        with self._lock:
            cached_keys = {k for (k, _) in self._cache}
            cached_keys.update(k for (k, _) in self._tensor_cache)
        wrapped_keys = set(self._wrapped.list_keys())
        return list(cached_keys | wrapped_keys)

    def get_parallel(self, keys: list[str], layer_name: str) -> list[bytes | None]:
        results: list[bytes | None] = [None] * len(keys)
        miss_indices: list[int] = []
        miss_keys: list[str] = []

        with self._lock:
            for i, key in enumerate(keys):
                cache_key = (key, layer_name)
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    results[i] = self._cache[cache_key]
                else:
                    miss_indices.append(i)
                    miss_keys.append(key)

        if miss_keys:
            fetched = self._wrapped.get_parallel(miss_keys, layer_name)
            with self._lock:
                for idx, data in zip(miss_indices, fetched):
                    results[idx] = data
                    if data is not None:
                        cache_key = (miss_keys[miss_indices.index(idx)], layer_name)
                        self._cache[cache_key] = data
                        self._cache.move_to_end(cache_key)
                        self._used_bytes += len(data)
                self._evict_if_needed()

        return results

    def put_chunks(
        self,
        key: str,
        layer_name: str,
        segments: list[bytes],
        meta: BlockMeta | None = None,
    ) -> None:
        """Pass-through interface for large segmented writes, avoiding the L1 wrapper breaking the KVService parallel/RDMA path."""
        if hasattr(self._wrapped, "put_chunks"):
            self._wrapped.put_chunks(key, layer_name, segments, meta)
            return
        self.put(key, layer_name, b"".join(segments), meta)

    def get_chunks(self, key: str, layer_name: str) -> list[bytes] | None:
        """Pass-through interface for large segmented reads, preserving compatibility with KVServiceBackend extensions."""
        if hasattr(self._wrapped, "get_chunks"):
            return self._wrapped.get_chunks(key, layer_name)
        data = self.get(key, layer_name)
        return [data] if data is not None else None

    def supports_zerocopy(self) -> bool:
        if not hasattr(self._wrapped, "supports_zerocopy"):
            return False
        return bool(self._wrapped.supports_zerocopy())

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        if not hasattr(self._wrapped, "ensure_rdma_region"):
            raise RuntimeError("wrapped storage does not support RDMA region registration")
        return int(self._wrapped.ensure_rdma_region(ptr, size))

    def get_chunks_into(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
    ) -> int | None:
        if not hasattr(self._wrapped, "get_chunks_into"):
            return None
        return self._wrapped.get_chunks_into(key, layer_name, region_id, offset)

    def supports_rdma_put(self) -> bool:
        if not hasattr(self._wrapped, "supports_rdma_put"):
            return False
        return bool(self._wrapped.supports_rdma_put())

    def put_chunks_from(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
        size: int,
    ) -> bool:
        if not hasattr(self._wrapped, "put_chunks_from"):
            return False
        return bool(self._wrapped.put_chunks_from(key, layer_name, region_id, offset, size))

    def _evict_if_needed(self) -> None:
        while self._used_bytes > self._max_capacity:
            if self._cache:
                _, evicted_data = self._cache.popitem(last=False)
                self._used_bytes -= len(evicted_data)
            elif self._tensor_cache:
                _, evicted_t = self._tensor_cache.popitem(last=False)
                self._used_bytes -= evicted_t.numel() * evicted_t.element_size()
            else:
                break
