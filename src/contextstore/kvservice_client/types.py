from __future__ import annotations

"""ContextStore KV Service - public data types."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectKey:
    """Opaque object key understood only by the caller."""

    namespace: str
    object_key: str

    def to_string(self) -> str:
        return f"{len(self.namespace.encode())}:{self.namespace}{self.object_key}"


@dataclass
class KVMetadata:
    """Metadata for a KV object."""

    num_tokens: int = 0
    num_layers: int = 0
    dtype: str = "bfloat16"
    shape: list[int] = field(default_factory=list)
    compressed: bool = False
    compression_level: int = 0
    created_at: int = 0
    last_accessed_at: int = 0


@dataclass
class PutOptions:
    """Options for PUT operations."""

    ttl_seconds: int = 0
    if_not_exists: bool = False
    compression: str = "NONE"  # NONE / INT8 / INT4


@dataclass
class ObjectDescriptor:
    """Object descriptor returned by KVService. Treated as an opaque read handle by the client."""

    key: ObjectKey
    object_handle: str = ""
    object_generation: int = 0
    content_etag: str = ""
    layout_version: int = 0
    size: int = 0
    is_striped: bool = False
    stripe_count: int = 0
    chunk_size: int = 0


@dataclass
class PlacementChunk:
    """Actual placement of a single physical shard of an object."""

    stripe_index: int
    node_id: str
    grpc_endpoint: str
    rdma_endpoint: str
    device_id: int
    storage_handle: str
    offset: int
    length: int


@dataclass
class PlacementDescriptor:
    """Actual placement fixed by the server at write time; used by the client to pick a data node."""

    key: ObjectKey
    placement_epoch: int = 1
    placement_policy_id: str = "object_hash_v1"
    layout_hash: str = ""
    primary_node_id: str = ""
    primary_grpc_endpoint: str = ""
    primary_rdma_endpoint: str = ""
    chunks: list[PlacementChunk] = field(default_factory=list)


@dataclass
class ObjectLookupResult:
    """Full return value of LookupObject."""

    descriptor: ObjectDescriptor
    placement: PlacementDescriptor | None = None


@dataclass
class DataReadResult:
    """Return value of ReadByDescriptor."""

    data: bytes
    metadata: KVMetadata
    descriptor: ObjectDescriptor
    placement: PlacementDescriptor | None = None


@dataclass
class HealthStatus:
    status: str
    version: str
    is_serving: bool


@dataclass
class StatsSnapshot:
    l1_cache_hits: int
    l1_cache_misses: int
    l1_cache_size_bytes: int
    l2_reads_total: int
    l2_writes_total: int
    l2_bytes_read: int
    l2_bytes_written: int
    metadata_entries: int
