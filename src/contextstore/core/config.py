from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextStoreConfig:
    storage_path: str = "/tmp/contextstore"
    block_size: int = 16
    enable_compression: bool = True
    compression_level: int = 1
    max_capacity_gb: float = 10.0
    num_io_workers: int = 4
    enable_layerwise_streaming: bool = True
    model_id: str = ""
    prefix_index_url: str = ""
    host_memory_capacity_gb: float = 2.0
    sharded_device_paths: list[str] | None = None
    sharded_stripe_policy: str = "block"
    sharded_io_depth: int = 4
    # Remote KV Service routing (enabled when non-empty): "host:port" e.g. "127.0.0.1:50051"
    # When enabled, L2 uses gRPC to call the remote Rust KV Service, replacing LocalStorageBackend / ShardedStorageBackend.
    # L1 (HostMemoryBackend) is still wrapped based on host_memory_capacity_gb.
    kv_service_endpoint: str = ""
    kv_service_chunk_size_mb: int = 2
    kv_service_timeout_ms: int = 30000
    # ===== Dual-node High Availability (HA) =====
    # When >=2 endpoints are configured, L2 uses HaKVServiceBackend: one independent child
    # KVServiceBackend per endpoint (each corresponding to one storage server), sharded via
    # md5(key)%N with no replication. When one node goes down, its ~1/N keys become cache
    # misses and get recomputed; the rest continue to hit normally, so inference requests don't fail.
    # Single endpoint (this field empty, using kv_service_endpoint) = original single-node
    # behavior, byte-for-byte identical.
    # Multi-NIC: rdma_server_addr / rdma_device are comma-separated, sliced across children in
    #   this list's order; each slice may itself contain commas to represent that child's
    #   multiple NICs (split internally within the child; no conflict).
    kv_service_endpoints: list[str] = field(default_factory=list)
    # Circuit breaker cooldown after a child backend goes down: within the window, requests to
    # that child skip the network and are marked as miss directly, avoiding the 30s gRPC timeout
    # dragging down the scheduler hot path; a single trial probe is issued at the end of the window
    # to detect recovery.
    ha_breaker_cooldown_s: float = 5.0
    # gRPC timeout (ms) for child KVServiceBackends in HA multi-node mode: use a shorter value so
    # that PROBE/health trial calls fail quickly when a node is down, instead of blocking on the
    # default 30s.
    ha_child_timeout_ms: int = 2000
    # Number of concurrent channels: KVServiceBackend maintains N independent KVClients over
    # separate HTTP/2 connections; put_chunks/get_chunks fan out segments across N channels in
    # parallel. Used to break past the ~180 MB/s single-stream transport ceiling; the greatest
    # gain is when it aligns with the server's 8-disk striping.
    # Default 8 aligns with the KV Service's 8 disks; 1 = degrade to single connection (backwards compatible).
    kv_service_parallel_channels: int = 8

    # ===== RDMA tier (optional) =====
    # When enabled, get_chunks prefers the RDMA tier (server must be compiled with --features rdma
    # and start the RDMA server); on failure, automatically falls back to gRPC. Requires:
    # - libcontextstore_rdma_ffi.so (cargo build --release in kv-service/rdma-ffi)
    # - libibverbs1 + ConnectX-x HCA
    # - Server's RDMA TCP listening port reachable
    rdma_enabled: bool = False
    rdma_server_addr: str = "127.0.0.1:50053"
    rdma_device: str = "mlx5_0"
    rdma_port: int = 1
    rdma_gid_index: int = 3  # 3 = RoCE v2 IPv4-mapped (check with show_gids)
    rdma_buf_size_mb: int = 512  # RDMA recv buffer per worker (take the max KV size)
    # When False, RDMA transport errors no longer auto-fall-back to gRPC, but raise directly
    # to expose the issue. Suitable for performance diagnosis/stress-testing phases, to avoid
    # "RDMA is actually broken but results just look a bit slow".
    rdma_fallback_to_grpc: bool = True
    # In the Worker register_kv_caches phase, prewarm the RDMA client and pre-register the MR
    # for the LOAD pinned buffer. This moves the connection/reg_mr fixed overhead of the first
    # cache-hit LOAD forward into the vLLM startup phase.
    rdma_prewarm_load_region: bool = True

    # ===== Shared filesystem GDS (optional) =====
    # GDS runs in the GPU worker process. The KVService remains the metadata and
    # write authority while the worker reads a versioned placement through its local
    # mount of the same Lustre/WekaFS/GPFS namespace.
    shared_gds_enabled: bool = False
    shared_gds_server_root: str = ""
    shared_gds_mount_root: str = ""
    shared_gds_min_bytes: int = 1024 * 1024
    shared_gds_file_cache_capacity: int = 128
    shared_gds_buffer_cache_capacity: int = 2
    shared_gds_staging_max_mb: int = 1024
    shared_gds_library_path: str = ""

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> ContextStoreConfig:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in extra_config.items() if k in known_fields}
        return cls(**filtered)

    @property
    def max_capacity_bytes(self) -> int:
        return int(self.max_capacity_gb * 1024**3)
