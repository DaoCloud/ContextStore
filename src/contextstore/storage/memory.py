from __future__ import annotations

from contextstore.storage.base import BlockMeta, StorageBackend


class MemoryStorageBackend(StorageBackend):
    def __init__(self, max_capacity_bytes: int = 1024**3):
        self._store: dict[str, dict[str, bytes]] = {}
        self._meta: dict[str, BlockMeta] = {}
        self._max_capacity = max_capacity_bytes
        self._used_bytes = 0

    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        if key not in self._store:
            self._store[key] = {}
        old_size = len(self._store[key].get(layer_name, b""))
        self._store[key][layer_name] = data
        self._used_bytes += len(data) - old_size
        if meta is not None:
            self._meta[key] = meta

    def get(self, key: str, layer_name: str) -> bytes | None:
        layers = self._store.get(key)
        if layers is None:
            return None
        return layers.get(layer_name)

    def exists(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str) -> None:
        if key in self._store:
            for data in self._store[key].values():
                self._used_bytes -= len(data)
            del self._store[key]
        self._meta.pop(key, None)

    def get_meta(self, key: str) -> BlockMeta | None:
        return self._meta.get(key)

    def capacity_usage(self) -> tuple[int, int]:
        return self._used_bytes, self._max_capacity

    def list_keys(self) -> list[str]:
        return list(self._store.keys())
