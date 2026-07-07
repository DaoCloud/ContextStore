from __future__ import annotations

from contextstore.storage.base import BlockMeta
from contextstore.storage.host_memory import HostMemoryBackend
from contextstore.storage.memory import MemoryStorageBackend


class DelegatingStorageBackend(MemoryStorageBackend):
    def __init__(self) -> None:
        super().__init__()
        self.put_chunks_calls: list[tuple[str, str, list[bytes]]] = []
        self.registered_regions: list[tuple[int, int]] = []

    def put_chunks(
        self,
        key: str,
        layer_name: str,
        segments: list[bytes],
        meta: BlockMeta | None = None,
    ) -> None:
        self.put_chunks_calls.append((key, layer_name, segments))
        self.put(key, layer_name, b"".join(segments), meta)

    def get_chunks(self, key: str, layer_name: str) -> list[bytes] | None:
        data = self.get(key, layer_name)
        return [data] if data is not None else None

    def supports_zerocopy(self) -> bool:
        return True

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        self.registered_regions.append((ptr, size))
        return 7

    def get_chunks_into(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
    ) -> int | None:
        return len(key) + len(layer_name) + region_id + offset

    def supports_rdma_put(self) -> bool:
        return True

    def put_chunks_from(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
        size: int,
    ) -> bool:
        return (
            key == "k1"
            and layer_name == "l0"
            and region_id == 7
            and offset == 0
            and size == 10
        )


class TestHostMemoryBackend:
    def setup_method(self):
        self.disk = MemoryStorageBackend(max_capacity_bytes=10 * 1024 * 1024)
        self.backend = HostMemoryBackend(wrapped=self.disk, max_capacity_bytes=1024)

    def test_put_and_get(self):
        self.backend.put("k1", "layer_0", b"hello")
        assert self.backend.get("k1", "layer_0") == b"hello"

    def test_write_through(self):
        self.backend.put("k1", "layer_0", b"data")
        assert self.disk.get("k1", "layer_0") == b"data"

    def test_l1_hit_skips_disk(self):
        self.backend.put("k1", "layer_0", b"original")
        self.disk.put("k1", "layer_0", b"modified_on_disk")
        assert self.backend.get("k1", "layer_0") == b"original"

    def test_l1_miss_fills_from_disk(self):
        self.disk.put("k1", "layer_0", b"from_disk")
        result = self.backend.get("k1", "layer_0")
        assert result == b"from_disk"
        self.disk.put("k1", "layer_0", b"changed")
        assert self.backend.get("k1", "layer_0") == b"from_disk"

    def test_lru_eviction(self):
        self.backend.put("k1", "layer_0", b"x" * 600)
        self.backend.put("k2", "layer_0", b"y" * 600)
        used, _ = self.backend.capacity_usage()
        assert used <= 1024
        assert self.disk.get("k1", "layer_0") == b"x" * 600

    def test_lru_order(self):
        self.backend.put("k1", "l0", b"x" * 400)
        self.backend.put("k2", "l0", b"y" * 400)
        self.backend.get("k1", "l0")
        self.backend.put("k3", "l0", b"z" * 400)
        used, _ = self.backend.capacity_usage()
        assert used <= 1024

    def test_exists(self):
        assert not self.backend.exists("k1")
        self.backend.put("k1", "layer_0", b"data")
        assert self.backend.exists("k1")

    def test_exists_only_on_disk(self):
        self.disk.put("k1", "layer_0", b"data")
        assert self.backend.exists("k1")

    def test_delete(self):
        self.backend.put("k1", "layer_0", b"data")
        self.backend.delete("k1")
        assert not self.backend.exists("k1")
        assert self.backend.get("k1", "layer_0") is None

    def test_get_nonexistent(self):
        assert self.backend.get("nope", "layer_0") is None

    def test_meta(self):
        meta = BlockMeta(num_tokens=16, num_layers=1, dtype="float16", shape=[2, 16, 64])
        self.backend.put("k1", "layer_0", b"data", meta)
        retrieved = self.backend.get_meta("k1")
        assert retrieved is not None
        assert retrieved.num_tokens == 16

    def test_list_keys(self):
        self.backend.put("k1", "l0", b"a")
        self.disk.put("k2", "l0", b"b")
        keys = self.backend.list_keys()
        assert set(keys) == {"k1", "k2"}

    def test_capacity_tracking(self):
        self.backend.put("k1", "l0", b"x" * 100)
        used, total = self.backend.capacity_usage()
        assert used == 100
        assert total == 1024

    def test_multiple_layers(self):
        self.backend.put("k1", "l0", b"aaa")
        self.backend.put("k1", "l1", b"bbb")
        assert self.backend.get("k1", "l0") == b"aaa"
        assert self.backend.get("k1", "l1") == b"bbb"

    def test_delegates_chunk_and_rdma_extensions(self):
        wrapped = DelegatingStorageBackend()
        backend = HostMemoryBackend(wrapped=wrapped, max_capacity_bytes=1024)

        backend.put_chunks("k1", "l0", [b"aa", b"bb"])

        assert wrapped.put_chunks_calls == [("k1", "l0", [b"aa", b"bb"])]
        assert backend.get_chunks("k1", "l0") == [b"aabb"]
        assert backend.supports_zerocopy() is True
        assert backend.ensure_rdma_region(123, 456) == 7
        assert wrapped.registered_regions == [(123, 456)]
        assert backend.get_chunks_into("k1", "l0", 7, 3) == 14
        assert backend.supports_rdma_put() is True
        assert backend.put_chunks_from("k1", "l0", 7, 0, 10) is True
