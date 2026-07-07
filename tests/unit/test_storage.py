from __future__ import annotations

import os
import tempfile

import pytest
from contextstore.storage.base import BlockMeta
from contextstore.storage.local import LocalStorageBackend
from contextstore.storage.memory import MemoryStorageBackend


class TestMemoryStorageBackend:
    def setup_method(self):
        self.backend = MemoryStorageBackend(max_capacity_bytes=1024 * 1024)

    def test_put_and_get(self):
        self.backend.put("key1", "layer_0", b"hello", None)
        assert self.backend.get("key1", "layer_0") == b"hello"

    def test_get_nonexistent(self):
        assert self.backend.get("nope", "layer_0") is None

    def test_exists(self):
        assert not self.backend.exists("key1")
        self.backend.put("key1", "layer_0", b"data")
        assert self.backend.exists("key1")

    def test_delete(self):
        self.backend.put("key1", "layer_0", b"data")
        self.backend.delete("key1")
        assert not self.backend.exists("key1")
        assert self.backend.get("key1", "layer_0") is None

    def test_capacity_tracking(self):
        self.backend.put("key1", "layer_0", b"x" * 100)
        used, total = self.backend.capacity_usage()
        assert used == 100
        assert total == 1024 * 1024

    def test_multiple_layers(self):
        self.backend.put("key1", "layer_0", b"aaa")
        self.backend.put("key1", "layer_1", b"bbb")
        assert self.backend.get("key1", "layer_0") == b"aaa"
        assert self.backend.get("key1", "layer_1") == b"bbb"

    def test_meta(self):
        meta = BlockMeta(num_tokens=16, num_layers=1, dtype="float16", shape=[2, 16, 64])
        self.backend.put("key1", "layer_0", b"data", meta)
        retrieved = self.backend.get_meta("key1")
        assert retrieved is not None
        assert retrieved.num_tokens == 16

    def test_list_keys(self):
        self.backend.put("k1", "l0", b"a")
        self.backend.put("k2", "l0", b"b")
        keys = self.backend.list_keys()
        assert set(keys) == {"k1", "k2"}


class TestLocalStorageBackend:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.backend = LocalStorageBackend(
            storage_path=self.tmpdir,
            max_capacity_bytes=10 * 1024 * 1024,
        )

    def test_put_and_get(self):
        self.backend.put("block_abc", "layer_0", b"tensor_data")
        result = self.backend.get("block_abc", "layer_0")
        assert result == b"tensor_data"

    def test_exists(self):
        assert not self.backend.exists("block_abc")
        self.backend.put("block_abc", "layer_0", b"data")
        assert self.backend.exists("block_abc")

    def test_delete(self):
        self.backend.put("block_abc", "layer_0", b"data")
        self.backend.delete("block_abc")
        assert not self.backend.exists("block_abc")

    def test_meta_persistence(self):
        meta = BlockMeta(
            num_tokens=16, num_layers=1, dtype="float16",
            shape=[2, 16, 64], compressed=True, compression_level=1,
        )
        self.backend.put("block_abc", "layer_0", b"data", meta)
        retrieved = self.backend.get_meta("block_abc")
        assert retrieved is not None
        assert retrieved.compressed is True
        assert retrieved.compression_level == 1

    def test_list_keys(self):
        self.backend.put("k1", "l0", b"a")
        self.backend.put("k2", "l0", b"b")
        keys = self.backend.list_keys()
        assert set(keys) == {"k1", "k2"}
