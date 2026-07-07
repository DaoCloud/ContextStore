from __future__ import annotations

import pytest

from contextstore.kvbm import KVBMBlockKey, KVBMBlockMetadata, KVBMContextStoreBackend
from contextstore.storage.memory import MemoryStorageBackend


class TestKVBMContextStoreBackend:
    def setup_method(self) -> None:
        self.storage = MemoryStorageBackend(max_capacity_bytes=1024 * 1024)
        self.backend = KVBMContextStoreBackend(self.storage)
        self.key = KVBMBlockKey(
            namespace="tenant/a",
            model_id="qwen/32b",
            sequence_hash="abc123",
            prefix_hash="pref/456",
            tp_rank=2,
            block_index=7,
        )

    def test_storage_key_is_stable_and_path_safe(self) -> None:
        storage_key = self.key.storage_key()

        assert storage_key == self.key.storage_key()
        assert "/" not in storage_key
        assert "tenant%2Fa" in storage_key
        assert "qwen%2F32b" in storage_key

    def test_put_get_exists_and_delete_block(self) -> None:
        self.backend.put_block(self.key, b"kv-block")

        assert self.backend.exists_block(self.key)
        assert self.backend.get_block(self.key) == b"kv-block"

        self.backend.delete_block(self.key)

        assert not self.backend.exists_block(self.key)
        assert self.backend.get_block(self.key) is None

    def test_put_block_accepts_memoryview(self) -> None:
        payload = memoryview(bytearray(b"abcdef"))[1:5]

        self.backend.put_block(self.key, payload)

        assert self.backend.get_block(self.key) == b"bcde"

    def test_metadata_round_trip(self) -> None:
        metadata = KVBMBlockMetadata(
            num_tokens=16,
            num_layers=1,
            dtype="bfloat16",
            shape=[2, 16, 8, 128],
        )

        self.backend.put_block(self.key, b"data", metadata)
        retrieved = self.backend.get_block_metadata(self.key)

        assert retrieved is not None
        assert retrieved.num_tokens == 16
        assert retrieved.dtype == "bfloat16"
        assert retrieved.shape == [2, 16, 8, 128]

    def test_read_block_into_target_buffer(self) -> None:
        target = bytearray(8)
        self.backend.put_block(self.key, b"abcd")

        n = self.backend.read_block_into(self.key, target)

        assert n == 4
        assert target == bytearray(b"abcd\x00\x00\x00\x00")

    def test_read_block_into_returns_none_for_miss(self) -> None:
        assert self.backend.read_block_into(self.key, bytearray(8)) is None

    def test_read_block_into_rejects_small_target(self) -> None:
        self.backend.put_block(self.key, b"abcd")

        with pytest.raises(ValueError, match="target buffer too small"):
            self.backend.read_block_into(self.key, bytearray(2))

    def test_batch_put_and_get_blocks(self) -> None:
        other = KVBMBlockKey(
            model_id="qwen/32b",
            sequence_hash="def456",
            block_index=8,
            tp_rank=2,
        )

        self.backend.put_blocks([
            (self.key, b"one", None),
            (other, b"two", None),
        ])

        assert self.backend.get_blocks([self.key, other]) == [b"one", b"two"]

    def test_get_block_into_region_uses_backend_fast_path(self) -> None:
        calls: list[tuple[str, str, int, int]] = []

        def get_chunks_into(key: str, layer_name: str, region_id: int, offset: int) -> int:
            calls.append((key, layer_name, region_id, offset))
            return 123

        self.storage.get_chunks_into = get_chunks_into  # type: ignore[attr-defined]

        assert self.backend.get_block_into_region(self.key, 9, 64) == 123
        assert calls == [(self.key.storage_key(), "__combined__", 9, 64)]
