from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from contextstore.storage.base import BlockMeta, StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, storage_path: str, max_capacity_bytes: int):
        self._root = Path(storage_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_capacity = max_capacity_bytes

    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        block_dir = self._root / key
        block_dir.mkdir(parents=True, exist_ok=True)
        layer_path = block_dir / f"{layer_name}.bin"
        layer_path.write_bytes(data)
        if meta is not None:
            meta_path = block_dir / "meta.json"
            meta_path.write_text(json.dumps({
                "num_tokens": meta.num_tokens,
                "num_layers": meta.num_layers,
                "dtype": meta.dtype,
                "shape": meta.shape,
                "compressed": meta.compressed,
                "compression_level": meta.compression_level,
            }))

    def get(self, key: str, layer_name: str) -> bytes | None:
        layer_path = self._root / key / f"{layer_name}.bin"
        if not layer_path.exists():
            return None
        return layer_path.read_bytes()

    def exists(self, key: str) -> bool:
        return (self._root / key).is_dir()

    def delete(self, key: str) -> None:
        block_dir = self._root / key
        if block_dir.exists():
            shutil.rmtree(block_dir)

    def get_meta(self, key: str) -> BlockMeta | None:
        meta_path = self._root / key / "meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return BlockMeta(**data)

    def capacity_usage(self) -> tuple[int, int]:
        used = sum(f.stat().st_size for f in self._root.rglob("*") if f.is_file())
        return used, self._max_capacity

    def list_keys(self) -> list[str]:
        if not self._root.exists():
            return []
        return [d.name for d in self._root.iterdir() if d.is_dir()]
