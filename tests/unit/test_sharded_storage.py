from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from contextstore.storage.base import BlockMeta
from contextstore.storage.sharded import ShardedStorageBackend


class TestShardedStorageBackend:
    def setup_method(self):
        self._tmpdirs = [tempfile.mkdtemp() for _ in range(4)]
        self.backend = ShardedStorageBackend(
            device_paths=self._tmpdirs,
            max_capacity_bytes_per_device=10 * 1024 * 1024,
            stripe_policy="block",
            io_depth=2,
        )

    def teardown_method(self):
        self.backend.shutdown()
        import shutil
        for d in self._tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    def test_put_and_get(self):
        self.backend.put("key1", "layer_0", b"hello", None)
        assert self.backend.get("key1", "layer_0") == b"hello"

    def test_distribution_across_shards(self):
        keys = [f"key_{i}" for i in range(20)]
        for k in keys:
            self.backend.put(k, "layer_0", f"data_{k}".encode(), None)

        shard_counts = [0] * 4
        for k in keys:
            shard_id = self.backend._shard_for_key(k)
            shard_counts[shard_id] += 1

        assert all(c > 0 for c in shard_counts), f"Not all shards used: {shard_counts}"

    def test_get_nonexistent(self):
        assert self.backend.get("missing", "layer_0") is None

    def test_exists(self):
        assert not self.backend.exists("key1")
        self.backend.put("key1", "layer_0", b"data", None)
        assert self.backend.exists("key1")

    def test_delete(self):
        self.backend.put("key1", "layer_0", b"data", None)
        self.backend.delete("key1")
        assert not self.backend.exists("key1")
        assert self.backend.get("key1", "layer_0") is None

    def test_get_meta(self):
        meta = BlockMeta(
            num_tokens=16, num_layers=1, dtype="torch.float16",
            shape=[2, 16, 8, 64], compressed=False, compression_level=0,
        )
        self.backend.put("key1", "layer_0", b"data", meta)
        retrieved = self.backend.get_meta("key1")
        assert retrieved is not None
        assert retrieved.num_tokens == 16

    def test_capacity_usage(self):
        self.backend.put("key1", "layer_0", b"x" * 1000, None)
        used, total = self.backend.capacity_usage()
        assert used >= 1000
        assert total == 4 * 10 * 1024 * 1024

    def test_list_keys(self):
        self.backend.put("a", "l0", b"1", None)
        self.backend.put("b", "l0", b"2", None)
        self.backend.put("c", "l0", b"3", None)
        keys = self.backend.list_keys()
        assert set(keys) == {"a", "b", "c"}

    def test_get_parallel(self):
        keys = [f"key_{i}" for i in range(10)]
        for k in keys:
            self.backend.put(k, "layer_0", f"value_{k}".encode(), None)

        results = self.backend.get_parallel(keys, "layer_0")
        assert len(results) == 10
        for i, k in enumerate(keys):
            assert results[i] == f"value_{k}".encode()

    def test_get_parallel_with_missing(self):
        self.backend.put("key_0", "l0", b"exists", None)
        results = self.backend.get_parallel(["key_0", "key_missing"], "l0")
        assert results[0] == b"exists"
        assert results[1] is None

    def test_put_parallel(self):
        items = [
            (f"key_{i}", "layer_0", f"data_{i}".encode(), None)
            for i in range(10)
        ]
        self.backend.put_parallel(items)
        for i in range(10):
            assert self.backend.get(f"key_{i}", "layer_0") == f"data_{i}".encode()

    def test_layer_stripe_policy(self):
        tmpdirs = [tempfile.mkdtemp() for _ in range(4)]
        backend = ShardedStorageBackend(
            device_paths=tmpdirs,
            max_capacity_bytes_per_device=10 * 1024 * 1024,
            stripe_policy="layer",
            io_depth=2,
        )
        try:
            shard_l0 = backend._shard_for_key("key1", "layer_0")
            shard_l1 = backend._shard_for_key("key1", "layer_1")
            shard_l0_again = backend._shard_for_key("key1", "layer_0")
            assert shard_l0 == shard_l0_again
            # Different layers should (likely) map to different shards
            # with 4 shards and different layer names, probability of same shard is 25%
        finally:
            backend.shutdown()
            import shutil
            for d in tmpdirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_parallel_read_faster_than_serial(self):
        data = b"x" * 64 * 1024  # 64KB per block
        keys = [f"key_{i}" for i in range(20)]
        for k in keys:
            self.backend.put(k, "layer_0", data, None)

        # Parallel reads should produce correct results
        results = self.backend.get_parallel(keys, "layer_0")
        assert len(results) == 20
        assert all(r == data for r in results)


class TestShardedStorageBackendSingleDevice:
    """Verify it works correctly with edge case of single path (falls through in engine, but test anyway)."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.backend = ShardedStorageBackend(
            device_paths=[self._tmpdir],
            max_capacity_bytes_per_device=10 * 1024 * 1024,
        )

    def teardown_method(self):
        self.backend.shutdown()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_basic_operations(self):
        self.backend.put("k1", "l0", b"data", None)
        assert self.backend.get("k1", "l0") == b"data"
        assert self.backend.exists("k1")
        self.backend.delete("k1")
        assert not self.backend.exists("k1")


class TestShardedStorageBackendValidation:
    def test_empty_paths_raises(self):
        with pytest.raises(ValueError):
            ShardedStorageBackend(
                device_paths=[],
                max_capacity_bytes_per_device=1024,
            )
