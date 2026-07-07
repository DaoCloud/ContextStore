from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BlockMeta:
    num_tokens: int
    num_layers: int
    dtype: str
    shape: list[int]
    compressed: bool = False
    compression_level: int = 0


class StorageBackend(ABC):
    @abstractmethod
    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        ...

    @abstractmethod
    def get(self, key: str, layer_name: str) -> bytes | None:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def get_meta(self, key: str) -> BlockMeta | None:
        ...

    @abstractmethod
    def capacity_usage(self) -> tuple[int, int]:
        """Returns (used_bytes, total_capacity_bytes)."""
        ...

    def list_keys(self) -> list[str]:
        return []

    def get_parallel(self, keys: list[str], layer_name: str) -> list[bytes | None]:
        """Batch read multiple keys. Override for true parallel I/O."""
        return [self.get(key, layer_name) for key in keys]

    def put_parallel(
        self,
        items: list[tuple[str, str, bytes, BlockMeta | None]],
    ) -> None:
        """Batch write multiple items. Override for true parallel I/O."""
        for key, layer_name, data, meta in items:
            self.put(key, layer_name, data, meta)
