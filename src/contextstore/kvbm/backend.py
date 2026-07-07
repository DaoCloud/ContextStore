from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from contextstore.storage.base import BlockMeta, StorageBackend
from contextstore.storage.kvservice import KVServiceBackend


_DEFAULT_LAYER_NAME = "__combined__"


def _escape_component(value: str) -> str:
    return quote(value, safe="")


@dataclass(frozen=True)
class KVBMBlockKey:
    """Stable block identity supplied by KVBM/Dynamo."""

    model_id: str
    sequence_hash: str
    block_index: int
    tp_rank: int = 0
    namespace: str = "default"
    prefix_hash: str = ""
    layout_version: str = "v1"

    def storage_key(self) -> str:
        """Return a filesystem-safe ContextStore key for this KVBM block."""
        parts = [
            "kvbm",
            self.layout_version,
            self.namespace,
            self.model_id,
            self.sequence_hash,
            self.prefix_hash or "-",
            f"tp{self.tp_rank}",
            f"b{self.block_index}",
        ]
        return ":".join(_escape_component(part) for part in parts)


@dataclass
class KVBMBlockMetadata:
    """Metadata carried with a KVBM block object."""

    num_tokens: int
    dtype: str
    shape: list[int]
    num_layers: int = 1
    compressed: bool = False
    compression_level: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_block_meta(self) -> BlockMeta:
        return BlockMeta(
            num_tokens=self.num_tokens,
            num_layers=self.num_layers,
            dtype=self.dtype,
            shape=self.shape,
            compressed=self.compressed,
            compression_level=self.compression_level,
        )

    @classmethod
    def from_block_meta(cls, meta: BlockMeta) -> KVBMBlockMetadata:
        return cls(
            num_tokens=meta.num_tokens,
            num_layers=meta.num_layers,
            dtype=meta.dtype,
            shape=list(meta.shape),
            compressed=meta.compressed,
            compression_level=meta.compression_level,
        )


class KVBMContextStoreBackend:
    """KVBM-compatible storage adapter backed by ContextStore.

    This class intentionally depends only on ContextStore's StorageBackend
    contract. A future Dynamo/KVBM integration can wrap it behind the exact
    NIXL/KVBM interface without changing KVService storage semantics.
    """

    def __init__(
        self,
        storage: StorageBackend,
        layer_name: str = _DEFAULT_LAYER_NAME,
    ) -> None:
        self._storage = storage
        self._layer_name = layer_name

    @classmethod
    def from_kvservice(
        cls,
        endpoint: str,
        model_id: str,
        *,
        layer_name: str = _DEFAULT_LAYER_NAME,
        chunk_size_mb: int = 2,
        timeout_ms: int = 30000,
        parallel_channels: int = 1,
        rdma_enabled: bool = False,
        rdma_server_addr: str = "127.0.0.1:50053",
        rdma_device: str = "mlx5_0",
        rdma_port: int = 1,
        rdma_gid_index: int = 3,
        rdma_buf_size_mb: int = 512,
        rdma_fallback_to_grpc: bool = True,
    ) -> KVBMContextStoreBackend:
        storage = KVServiceBackend(
            endpoint=endpoint,
            model_id=model_id,
            chunk_size_mb=chunk_size_mb,
            timeout_ms=timeout_ms,
            parallel_channels=parallel_channels,
            rdma_enabled=rdma_enabled,
            rdma_server_addr=rdma_server_addr,
            rdma_device=rdma_device,
            rdma_port=rdma_port,
            rdma_gid_index=rdma_gid_index,
            rdma_buf_size_mb=rdma_buf_size_mb,
            rdma_fallback_to_grpc=rdma_fallback_to_grpc,
        )
        return cls(storage=storage, layer_name=layer_name)

    @property
    def storage(self) -> StorageBackend:
        return self._storage

    @property
    def layer_name(self) -> str:
        return self._layer_name

    def put_block(
        self,
        key: KVBMBlockKey,
        data: bytes | bytearray | memoryview,
        metadata: KVBMBlockMetadata | None = None,
    ) -> None:
        payload = bytes(data)
        meta = metadata.to_block_meta() if metadata is not None else None
        storage_key = key.storage_key()
        put_chunks = getattr(self._storage, "put_chunks", None)
        if callable(put_chunks):
            put_chunks(storage_key, self._layer_name, [payload], meta)
            return
        self._storage.put(storage_key, self._layer_name, payload, meta)

    def get_block(self, key: KVBMBlockKey) -> bytes | None:
        storage_key = key.storage_key()
        get_chunks = getattr(self._storage, "get_chunks", None)
        if callable(get_chunks):
            chunks = get_chunks(storage_key, self._layer_name)
            if chunks is None:
                return None
            return b"".join(chunks)
        return self._storage.get(storage_key, self._layer_name)

    def read_block_into(
        self,
        key: KVBMBlockKey,
        target: bytearray | memoryview,
    ) -> int | None:
        data = self.get_block(key)
        if data is None:
            return None
        view = memoryview(target)
        if len(view) < len(data):
            raise ValueError(
                f"target buffer too small: need {len(data)} bytes, got {len(view)}"
            )
        view[: len(data)] = data
        return len(data)

    def get_block_into_region(
        self,
        key: KVBMBlockKey,
        region_id: int,
        offset: int = 0,
    ) -> int | None:
        """Read directly into a registered RDMA region when the backend supports it."""
        get_chunks_into = getattr(self._storage, "get_chunks_into", None)
        if not callable(get_chunks_into):
            return None
        return get_chunks_into(key.storage_key(), self._layer_name, region_id, offset)

    def exists_block(self, key: KVBMBlockKey) -> bool:
        return self._storage.exists(key.storage_key())

    def delete_block(self, key: KVBMBlockKey) -> None:
        self._storage.delete(key.storage_key())

    def get_block_metadata(self, key: KVBMBlockKey) -> KVBMBlockMetadata | None:
        meta = self._storage.get_meta(key.storage_key())
        if meta is None:
            return None
        return KVBMBlockMetadata.from_block_meta(meta)

    def put_blocks(
        self,
        items: list[tuple[KVBMBlockKey, bytes | bytearray | memoryview, KVBMBlockMetadata | None]],
    ) -> None:
        for key, data, metadata in items:
            self.put_block(key, data, metadata)

    def get_blocks(self, keys: list[KVBMBlockKey]) -> list[bytes | None]:
        return [self.get_block(key) for key in keys]
