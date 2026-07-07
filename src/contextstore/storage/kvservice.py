from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from contextstore.storage.base import BlockMeta, StorageBackend

logger = logging.getLogger(__name__)

_DEFAULT_RDMA_REGION_CACHE_LIMIT = 8


def _append_rdma_perf_log(line: str) -> None:
    """Write to RDMA diagnostic log, sharing the same file as the connector perf log."""
    if os.environ.get("CS_PERF_LOG", "0") != "1":
        return
    path = os.environ.get("CS_PERF_LOG_PATH", "/tmp/cs_zc_perf.log")
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


class KVServiceBackend(StorageBackend):
    """StorageBackend adapter that talks to a remote Rust KV Service via gRPC.

    Maps StorageBackend's (key, layer_name, data) to KV Service's
    ObjectKey(namespace=model_id, object_key=<encoded key/layer>). The prefix/layer
    are semantics of this adapter only; KVService itself sees an opaque object key.

    Uses the streaming put_stream/get_stream RPCs (rather than single-message
    put/get) because vLLM combined-value writes are typically ~480MB and a single
    protobuf message hits the server's single-message decode ceiling (~0.17 GB/s).
    Streaming measures ~2 GB/s PUT and ~1.5 GB/s aggregate GET across connections.

    Lifecycle: long-lived per-process connection (gRPC channel reuse); no reconnect
    on every put/get.

    **Multi-lane parallelism (parallel_channels > 1)**: maintain N independent
    KVClients (each with its own channel + HTTP/2 connection) and run N concurrent
    put_stream/get_stream calls. This breaks the single-stream transport ceiling
    (~180 MB/s) — measured 8-way concurrency reaches ~2.6 GB/s aggregate, letting
    the server's 8-disk striping shine.

    Partitioning strategy: put_chunks splits segments contiguously into N parts,
    each with its own layer_name suffix (`_p{i}of{N}`); get_chunks fetches the N
    parts concurrently in the same order and concatenates.
    """

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        chunk_size_mb: int = 2,
        timeout_ms: int = 30000,
        max_message_mb: int = 2047,  # avoid the int32 overflow at max_message_mb=2048
        parallel_channels: int = 8,  # N independent gRPC channels (aligned with the server's 8 disks)
        # ===== RDMA (optional) =====
        # When all of these are set, get_chunks prefers the RDMA tier (the server
        # must be built with --features rdma and start the RDMA server).
        rdma_enabled: bool = False,
        rdma_server_addr: str = "127.0.0.1:50053",
        rdma_device: str = "mlx5_0",
        rdma_port: int = 1,
        rdma_gid_index: int = 3,
        rdma_buf_size_mb: int = 512,  # per-RDMA-client buffer (sized for the max possible KV)
        rdma_fallback_to_grpc: bool = True,
    ) -> None:
        # Lazy import: avoid a hard dependency on grpcio for deployments that don't enable KVService
        from contextstore.kvservice_client import KVClient

        self._endpoint = endpoint
        self._model_id = model_id
        self._chunk_size = max(chunk_size_mb, 1) * 1024 * 1024
        self._parallel = max(parallel_channels, 1)
        # RDMA configuration
        self._rdma_enabled = rdma_enabled
        # Multi-NIC support: rdma_server_addr and rdma_device accept comma-separated lists,
        # e.g. server="127.0.0.1:50053,127.0.0.1:50054" device="mlx5_0,mlx5_1".
        # The two lists must have equal length (one endpoint + one device per NIC). A single
        # value means single NIC, behavior unchanged.
        # NIC selection is deferred to _get_or_create_rdma_client_pool (after worker fork)
        # using pid hashing, because __init__ runs during the EngineCore stage when all
        # workers share the same pid.
        self._rdma_server_addrs: list[str] = [s.strip() for s in rdma_server_addr.split(',') if s.strip()]
        self._rdma_devices: list[str] = [s.strip() for s in rdma_device.split(',') if s.strip()]
        if len(self._rdma_devices) == 1 and len(self._rdma_server_addrs) > 1:
            self._rdma_devices = self._rdma_devices * len(self._rdma_server_addrs)
        if len(self._rdma_server_addrs) != len(self._rdma_devices):
            raise ValueError(
                f"rdma_server_addr and rdma_device counts differ: "
                f"{len(self._rdma_server_addrs)} addrs vs {len(self._rdma_devices)} devs"
            )
        # Legacy fields: _rdma_server_addr/_rdma_device will be set to this worker's chosen
        # NIC after lazy init. During __init__ we fill list[0] for startup logger.warning
        # messages.
        self._rdma_server_addr = self._rdma_server_addrs[0]
        self._rdma_device = self._rdma_devices[0]
        self._rdma_nic_idx = -1  # not yet selected; decided during lazy init
        self._rdma_port = rdma_port
        self._rdma_gid_index = rdma_gid_index
        self._rdma_buf_size_mb = rdma_buf_size_mb
        self._rdma_fallback_to_grpc = rdma_fallback_to_grpc
        # Lazy RDMA client pool creation (connect only on first get_chunks).
        # Each worker process connects independently (worker fork must happen first,
        # otherwise RDMA QP state is lost).
        # pool size = self._parallel: aligned with the sub-chunk splitting parallelism so each
        # client has its own buffer and can GET concurrently without overwriting each other.
        self._rdma_client_pool: list[Any] = []
        self._rdma_lock = threading.Lock()

        # Primary client (used for low-bandwidth paths: health / stats / exists / delete)
        self._client = KVClient(
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            max_message_mb=max_message_mb,
        )
        # Concurrent client pool: N independent channels, each with its own HTTP/2 connection.
        self._client_pool: list[Any] = [
            KVClient(
                endpoint=endpoint,
                timeout_ms=timeout_ms,
                max_message_mb=max_message_mb,
            )
            for _ in range(self._parallel)
        ]
        # Reuse an executor to avoid spawning a new thread on every put/get.
        self._executor = ThreadPoolExecutor(
            max_workers=self._parallel,
            thread_name_prefix="kvservice-parallel",
        )

        # Health check (surface connection problems early); on failure downgrade to warning
        # instead of raising, so vLLM startup is not blocked.
        try:
            h = self._client.health()
            # WARNING level so it shows in vllm serve default log
            logger.warning(
                "[CS KVService] connected to %s (serving=%s, version=%s, parallel=%d, rdma=%s)",
                endpoint, h.is_serving, h.version, self._parallel,
                "ON" if self._rdma_enabled else "OFF",
            )
        except Exception as e:
            logger.warning("KVServiceBackend health check failed: %s", e)

        # Warm up the channel pool: gRPC channels are lazy-connected by default and the first
        # RPC pays 100-500ms for the HTTP/2 handshake. Send a health() concurrently on every
        # pool client to establish connections. This does not block vLLM startup (fire-and-forget
        # in the thread pool).
        def _warmup_one(client: Any) -> None:
            try:
                client.health()
            except Exception:
                pass

        warmup_futs = [self._executor.submit(_warmup_one, c) for c in self._client_pool]
        for f in warmup_futs:
            try:
                f.result(timeout=5.0)
            except Exception:
                pass
        logger.info("KVServiceBackend warmed up %d channels", self._parallel)

    # ===== StorageBackend required methods =====

    @staticmethod
    def _encode_object_key(key: str, layer_name: str) -> str:
        """Encode StorageBackend's binary key into a KVService opaque object_key."""
        return f"{len(key.encode())}:{key}{layer_name}"

    def _object_key(self, key: str, layer_name: str) -> Any:
        from contextstore.kvservice_client import ObjectKey

        return ObjectKey(
            namespace=self._model_id or "default",
            object_key=self._encode_object_key(key, layer_name),
        )

    def _rdma_string_key(self, key: str, layer_name: str) -> str:
        return self._object_key(key, layer_name).to_string()

    def _get_or_create_rdma_client_pool(self):
        """Lazily create N RDMA clients per worker process (each with its own buffer + QP)
        for concurrent sub-chunk GETs.

        Must connect after worker fork (otherwise RDMA QP state is lost). Per-client buffer =
        total buf // N, keeping the pinned-memory budget the same as with a single client
        (avoids blowing past the container RLIMIT_MEMLOCK).

        Multi-NIC: picks a NIC by pid hash (ensuring an even distribution across workers);
        all pool clients within a worker share the same NIC.
        """
        if self._rdma_client_pool:
            return self._rdma_client_pool
        with self._rdma_lock:
            if self._rdma_client_pool:
                return self._rdma_client_pool
            try:
                from contextstore.storage.rdma_client import RdmaClient
                # For multi-NIC, pick NIC via pid % N (worker pids are unique after fork).
                # For single NIC always pick 0. All pool clients of the same worker share
                # one NIC (avoids QP-across-NICs complexity).
                if self._rdma_nic_idx < 0:
                    n_nics = len(self._rdma_server_addrs)
                    self._rdma_nic_idx = os.getpid() % n_nics
                    self._rdma_server_addr = self._rdma_server_addrs[self._rdma_nic_idx]
                    self._rdma_device = self._rdma_devices[self._rdma_nic_idx]
                    if n_nics > 1:
                        logger.warning(
                            "[CS RDMA] multi-NIC: pid=%d chose nic_idx=%d (server=%s dev=%s), total %d NICs",
                            os.getpid(), self._rdma_nic_idx,
                            self._rdma_server_addr, self._rdma_device, n_nics,
                        )
                # Shrink per-client buffer to max(total_buf // N, 256MB). 256MB is a bit tight
                # for the extreme case of a single sub-chunk (~270MB) at 32K tokens 32B-TP4,
                # but with the default buf=2048 and N=8 it's exactly 256MB; at buf=4096 it's
                # 512MB, comfortably enough. At N=1 each client = the full buf size, matching
                # legacy behavior.
                per_client_mb = max(self._rdma_buf_size_mb // self._parallel, 256)
                pool: list[Any] = []
                for i in range(self._parallel):
                    c = RdmaClient(
                        server_addr=self._rdma_server_addr,
                        device=self._rdma_device,
                        port=self._rdma_port,
                        gid_index=self._rdma_gid_index,
                        buf_size_mb=per_client_mb,
                    )
                    c.connect()
                    pool.append(c)
                self._rdma_client_pool = pool
                logger.warning(
                    "[CS RDMA] connected pool: server=%s dev=%s clients=%d buf_each=%dMB pid=%d nic_idx=%d",
                    self._rdma_server_addr, self._rdma_device,
                    self._parallel, per_client_mb, os.getpid(), self._rdma_nic_idx,
                )
                return pool
            except Exception as e:
                logger.error("[CS RDMA] init failed: %s", e)
                if self._rdma_fallback_to_grpc:
                    self._rdma_enabled = False  # permanently disable only when fallback is allowed
                return None

    def _reset_rdma_client_pool(self, reason: str) -> None:
        """Close the old RDMA connections and clear the MR cache; the next access will
        rebuild the connection.

        After a kvservice restart, both the old QPs and the TCP control channel become
        invalid; external region_ids are also bound to the old client handle, so they must
        be discarded together with the client pool.
        """
        with self._rdma_lock:
            pool = self._rdma_client_pool
            self._rdma_client_pool = []
            self._rdma_region_cache = {}
            self._rdma_put_region_cache = {}
        for client in pool:
            try:
                client.close()
            except Exception:
                pass
        logger.warning("[CS RDMA] reset client pool pid=%d reason=%s", os.getpid(), reason)

    def _find_rdma_region(self, region_id: int) -> tuple[int, int] | None:
        """Recover the caller's pinned buffer pointer and size given an old region_id."""
        cache = getattr(self, "_rdma_region_cache", None)
        if not cache:
            return None
        for (ptr, size), rid in cache.items():
            if rid == region_id:
                return ptr, size
        return None

    # ===== Plan-A zero-copy: connector pre-pins a host buffer, RDMA WRITE lands directly in it =====
    # Idea: the connector takes the spec, computes total_bytes, calls ensure_rdma_region once to
    # register the pinned buffer; then per-layer it calls get_chunks_into(offset) for a direct RDMA
    # write; subsequent H2D reads straight from the pinned region.
    # Skips the ctypes.string_at (GIL-holding bulk memcpy) + bytearray secondary copy on the
    # original get_chunks path.

    def supports_zerocopy(self) -> bool:
        """Whether the zero-copy path is usable (RDMA enabled + N=1 + ffi supports get_into)."""
        if not self._rdma_enabled or self._parallel != 1:
            return False
        pool = self._get_or_create_rdma_client_pool()
        return pool is not None and pool[0].supports_zerocopy

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        """Register a connector-owned pinned buffer as an RDMA MR. Returns region_id.

        Repeated calls with the same (ptr, size) return the cached region_id; the first call
        triggers ibv_reg_mr (~80ms/2GB). A single worker may hold both a LOAD buffer and a
        STORE pool buffer concurrently, so multiple regions must be cached; otherwise
        alternating load/store would repeatedly unregister/register and introduce
        second-scale latency at 16K/32K.
        """
        pool = self._get_or_create_rdma_client_pool()
        if pool is None or self._parallel != 1:
            raise RuntimeError("ensure_rdma_region only supported with rdma_enabled + parallel=1")
        client = pool[0]
        # Use (ptr, size) as the cache key; check whether re-registration is required.
        cache = getattr(self, "_rdma_region_cache", None)
        if cache is None:
            cache = {}
            self._rdma_region_cache = cache
        key = (ptr, size)
        if key in cache:
            return cache[key]

        try:
            cache_limit = int(
                os.environ.get("CS_RDMA_REGION_CACHE_LIMIT", str(_DEFAULT_RDMA_REGION_CACHE_LIMIT))
            )
        except ValueError:
            cache_limit = _DEFAULT_RDMA_REGION_CACHE_LIMIT
        cache_limit = max(cache_limit, 1)
        while len(cache) >= cache_limit:
            old_key, old_rid = next(iter(cache.items()))
            try:
                client.unregister_external_buffer(old_rid)
            except Exception:
                pass
            cache.pop(old_key, None)

        rid = client.register_external_buffer(ptr, size)
        cache[key] = rid
        logger.warning(
            "[CS RDMA] registered pinned buffer ptr=0x%x size=%dMB region_id=%d pid=%d",
            ptr, size // (1024 * 1024), rid, os.getpid(),
        )
        return rid

    def get_chunks_into(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
    ) -> int | None:
        """RDMA GET; server WRITEs into a pre-registered region (starting at offset).
        Returns bytes written; 0 = miss; None = not supported.

        Only available with RDMA enabled + parallel=1. Skips the chunks → bytes → numpy
        copy chain. On failure (RDMA error) returns None and the caller should fall back
        to get_chunks.
        """
        if not self._rdma_enabled or self._parallel != 1:
            return None
        pool = self._get_or_create_rdma_client_pool()
        if pool is None:
            return None
        full_key = self._rdma_string_key(key, layer_name)
        current_region_id = region_id
        region = self._find_rdma_region(region_id)
        descriptor = self._lookup_descriptor_for_rdma(key, layer_name)
        for attempt in range(2):
            client = pool[0]
            attempt_start_ns = time.time_ns()
            attempt_start = time.perf_counter()
            logger.info(
                "CS storage RDMA_GET_REQUEST key=%s layer=%s full_key=%s "
                "attempt=%d endpoint=%s device=%s region=%d offset=%d parallel=%d",
                key[:8],
                layer_name,
                full_key,
                attempt + 1,
                getattr(self, "_rdma_server_addr", "unknown"),
                getattr(self, "_rdma_device", "unknown"),
                current_region_id,
                offset,
                self._parallel,
            )
            try:
                if descriptor is not None and getattr(client, "supports_descriptor_get", False):
                    n = client.get_descriptor_into(current_region_id, full_key, descriptor, offset)
                    if n == 0:
                        fresh = self._lookup_descriptor_for_rdma(key, layer_name)
                        if (
                            fresh is not None
                            and self._descriptor_identity(fresh)
                            != self._descriptor_identity(descriptor)
                        ):
                            descriptor = fresh
                            n = client.get_descriptor_into(
                                current_region_id,
                                full_key,
                                descriptor,
                                offset,
                            )
                else:
                    n = client.get_into(current_region_id, full_key, offset)
                attempt_end_ns = time.time_ns()
                duration_ms = (attempt_end_ns - attempt_start_ns) / 1_000_000
                mib_s = n / max(duration_ms / 1000.0, 1e-9) / (1024 * 1024)
                _append_rdma_perf_log(
                    "RDMA_GET_ATTEMPT "
                    f"pid={os.getpid()} key={key[:8]} layer={layer_name} "
                    f"attempt={attempt + 1} ok=1 bytes={n} "
                    f"start_ns={attempt_start_ns} end_ns={attempt_end_ns} "
                    f"duration_ms={duration_ms:.3f} mib_s={mib_s:.2f} "
                    f"endpoint={getattr(self, '_rdma_server_addr', 'unknown')} "
                    f"device={getattr(self, '_rdma_device', 'unknown')}"
                )
                logger.info(
                    "CS storage RDMA_GET_DONE key=%s layer=%s attempt=%d "
                    "bytes=%d duration_ms=%.3f mib_s=%.2f",
                    key[:8],
                    layer_name,
                    attempt + 1,
                    n,
                    duration_ms,
                    mib_s,
                )
                return n
            except Exception as e:
                attempt_end_ns = time.time_ns()
                _append_rdma_perf_log(
                    "RDMA_GET_ATTEMPT "
                    f"pid={os.getpid()} key={key[:8]} layer={layer_name} "
                    f"attempt={attempt + 1} ok=0 bytes=0 "
                    f"start_ns={attempt_start_ns} end_ns={attempt_end_ns} "
                    f"duration_ms={(attempt_end_ns - attempt_start_ns) / 1_000_000:.3f} "
                    f"error={type(e).__name__}"
                )
                logger.warning(
                    "get_chunks_into rdma error key=%s layer=%s attempt=%d duration_ms=%.3f: %s",
                    key[:8], layer_name, attempt + 1,
                    (time.perf_counter() - attempt_start) * 1000,
                    e,
                )
                if attempt > 0:
                    return None
                self._reset_rdma_client_pool(f"get_into {type(e).__name__}")
                if region is None:
                    return None
                try:
                    current_region_id = self.ensure_rdma_region(region[0], region[1])
                    pool = self._get_or_create_rdma_client_pool()
                    if pool is None:
                        return None
                except Exception as reconnect_error:
                    logger.warning(
                        "get_chunks_into rdma reconnect failed key=%s layer=%s: %s",
                        key[:8], layer_name, reconnect_error,
                    )
                    return None
        return None

    # ===== PUT data plane (RDMA fast path) =====

    def supports_rdma_put(self) -> bool:
        """Whether the RDMA PUT path is usable (rdma enabled + parallel=1 + ffi supports put)."""
        if not self._rdma_enabled or self._parallel != 1:
            return False
        pool = self._get_or_create_rdma_client_pool()
        return pool is not None and getattr(pool[0], "supports_put", False)

    def put_chunks_from(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
        size: int,
    ) -> bool:
        """**Zero-copy PUT**: data already sits in a connector-owned pinned buffer; RDMA
        WRITE it to the server.

        Fully symmetric with get_chunks_into. Suitable once the connector is refactored
        to a "D2H into pinned buffer, then call this" zero-copy PUT path. Today put_chunks
        goes through _try_rdma_put with an assembly step; this method is reserved for the
        truly zero-copy path.

        Returns:
            True = success; False = RDMA unavailable / failure (caller should fall back to
            gRPC put_chunks).
        """
        if not self.supports_rdma_put():
            return False
        pool = self._get_or_create_rdma_client_pool()
        if pool is None:
            return False
        full_key = self._rdma_string_key(key, layer_name)
        current_region_id = region_id
        region = self._find_rdma_region(region_id)
        for attempt in range(2):
            client = pool[0]
            try:
                client.put(current_region_id, full_key, offset, size)
                return True
            except Exception as e:
                logger.warning(
                    "put_chunks_from rdma error key=%s layer=%s attempt=%d: %s",
                    key[:8], layer_name, attempt + 1, e,
                )
                if attempt > 0:
                    return False
                self._reset_rdma_client_pool(f"put_from {type(e).__name__}")
                if region is None:
                    return False
                try:
                    current_region_id = self.ensure_rdma_region(region[0], region[1])
                    pool = self._get_or_create_rdma_client_pool()
                    if pool is None:
                        return False
                except Exception as reconnect_error:
                    logger.warning(
                        "put_chunks_from rdma reconnect failed key=%s layer=%s: %s",
                        key[:8], layer_name, reconnect_error,
                    )
                    return False
        return False

    def _try_rdma_put(
        self,
        key: str,
        layer_name: str,
        segments: list[bytes],
        total: int,
    ) -> bool:
        """RDMA PUT path: assemble segments into one pinned buffer, then RDMA WRITE to the server.

        The current implementation needs one assembly (segments → buffer memcpy), but is
        ~10x faster than the gRPC path because:
        - Skips ~680ms of gRPC HTTP/2 + tonic decode
        - Skips the 138ms thread_local memcpy bug on the server
        - Server writes to disk zero-memcpy O_DIRECT (new put_from_ptr)

        Budget: one 480MB memcpy ~600ms; gRPC path ~850ms; net ~250ms savings.
        Once the connector D2Hs directly into a pinned buffer, put_chunks_from will skip
        this memcpy entirely.

        Returns:
            True = RDMA PUT succeeded, caller can return directly.
            False = RDMA unavailable / failure, caller should fall back to gRPC put_stream.
        """
        for attempt in range(2):
            if not self.supports_rdma_put():
                return False
            pool = self._get_or_create_rdma_client_pool()
            if pool is None:
                return False
            client = pool[0]
            # Reuse the GET path buffer (the built-in buffer allocated by cs_rdma_client_new).
            # Check the buffer has enough capacity (per-client buf_size_mb, default 512MB).
            try:
                buffer_ptr = client.buffer_ptr()
                # buffer_size from lib
                buf_size = self._client_buffer_size(client)
                if total > buf_size:
                    logger.warning(
                        "[CS RDMA PUT] total %d > client buffer %d, fallback to gRPC",
                        total, buf_size,
                    )
                    return False
                # Reuse the GET built-in buffer as the PUT source; register it as an external
                # region (rkey used bidirectionally).
                # Cache region_id: reuse a single region for the same (ptr, size) to avoid
                # paying ~80ms reg_mr every time.
                cache_key = ("__internal_put_buf__", buffer_ptr, buf_size)
                cache = getattr(self, "_rdma_put_region_cache", None)
                if cache is None:
                    cache = {}
                    self._rdma_put_region_cache = cache
                region_id = cache.get(cache_key)
                if region_id is None:
                    region_id = client.register_external_buffer(buffer_ptr, buf_size)
                    cache[cache_key] = region_id
                    logger.warning(
                        "[CS RDMA PUT] registered internal buffer ptr=0x%x size=%dMB region_id=%d",
                        buffer_ptr, buf_size // (1024 * 1024), region_id,
                    )
                # Assemble segments → buffer (one memcpy)
                self._copy_segments_to_buffer(buffer_ptr, segments)
                # Push to server
                full_key = self._rdma_string_key(key, layer_name)
                client.put(region_id, full_key, 0, total)
                return True
            except Exception as e:
                logger.warning(
                    "[CS RDMA PUT] failed key=%s layer=%s attempt=%d: %s",
                    key[:8] if key else "", layer_name, attempt + 1, e,
                )
                if attempt > 0:
                    return False
                self._reset_rdma_client_pool(f"try_put {type(e).__name__}")
        return False

    @staticmethod
    def _client_buffer_size(client: Any) -> int:
        """Get the built-in buffer size (bytes) of an RdmaClient."""
        # The buf_size field on the RdmaClient instance
        return getattr(client, "buf_size", 0)

    @staticmethod
    def _copy_segments_to_buffer(buffer_ptr: int, segments: list[bytes]) -> None:
        """Sequentially memcpy list[bytes] into memory starting at buffer_ptr.

        Measured: ctypes.memmove is 250ms+ cold start (Python bytes trigger src page faults)
        and 50ms warm; the numpy.frombuffer path is a steady 47ms with no cold start
        (zero-copy view, does not fault the src memory). Total 480MB PUT end-to-end goes
        from 167ms → ~130ms (~22% improvement).

        # numpy import is deferred to inside the function to avoid an import-time dependency
        # on numpy (not really an issue because vLLM already ships numpy).
        """
        try:
            import numpy as np
            # Wrap the whole buffer as a numpy uint8 view (zero-copy, holds the memory the
            # ptr points to).
            # Note: from_address does not take ownership; the buffer is pinned memory managed
            # by RdmaClient.
            total = sum(len(s) for s in segments)
            buf_arr_type = ctypes.c_uint8 * total
            buf_view = np.frombuffer(buf_arr_type.from_address(buffer_ptr), dtype=np.uint8)
            offset = 0
            for seg in segments:
                n = len(seg)
                if n == 0:
                    continue
                # numpy.frombuffer(bytes) is a zero-copy view; slice assignment triggers an
                # optimized memcpy.
                src_view = np.frombuffer(seg, dtype=np.uint8)
                buf_view[offset:offset + n] = src_view
                offset += n
        except ImportError:
            # Fallback: ctypes.memmove (slow path, only when numpy is not available)
            ptr = buffer_ptr
            for seg in segments:
                n = len(seg)
                if n == 0:
                    continue
                ctypes.memmove(ptr, seg, n)
                ptr += n

    def _get_chunks_rdma(
        self,
        key: str,
        layer_name: str,
    ) -> tuple[list[bytes] | None, bool]:
        """RDMA fast path: RDMA WRITE the server-side chunks_cache into the client buffer in one shot.

        Note: the RDMA server requires the key to be the canonical ObjectKey string:
        `<namespace_byte_len>:<namespace><object_key>`.

        **Concurrency optimization (C2)**: when parallel>1, N sub-chunks (`_pNofM` suffix) are
        fetched concurrently using N independent RDMA clients (executor pool). Replaces the
        original serial for-loop over a single client, dropping latency from ~N× to ~1× plus
        balancing overhead.

        Returns:
            (segments, had_error)
            - segments is not None: RDMA hit
            - segments is None and had_error is False: confirmed real miss
            - segments is None and had_error is True: RDMA transport error, caller should
              fall back to gRPC in the same call
        """
        N = self._parallel
        # Match the gRPC path: when parallel>1 the layer name has a _p{i}of{N} suffix.
        if N == 1:
            layer_parts = [layer_name]
        else:
            layer_parts = [f"{layer_name}_p{i}of{N}" for i in range(N)]

        for attempt in range(2):
            pool = self._get_or_create_rdma_client_pool()
            if pool is None:
                return None, True

            def _get_one(idx: int, lp: str) -> tuple[int, bytes | None, str | None]:
                """Single sub-chunk GET. Returns (idx, data_or_None, error_str_or_None).

                data=None and error=None → part miss (overall miss).
                data=b'' (empty) → sentinel, skip (matches gRPC put_chunks behavior).
                """
                client = pool[idx]
                full_key = self._rdma_string_key(key, lp)
                descriptor = self._lookup_descriptor_for_rdma(key, lp)
                try:
                    if descriptor is not None and getattr(client, "supports_descriptor_get", False):
                        n = client.get_descriptor(full_key, descriptor)
                        if n == 0:
                            fresh = self._lookup_descriptor_for_rdma(key, lp)
                            if (
                                fresh is not None
                                and self._descriptor_identity(fresh)
                                != self._descriptor_identity(descriptor)
                            ):
                                n = client.get_descriptor(full_key, fresh)
                    else:
                        n = client.get(full_key)
                except Exception as e:
                    return (idx, None, f"RDMA get part {lp} error: {e}")
                if n == 0:
                    # part miss → overall miss (matches gRPC behavior)
                    return (idx, None, None)
                if n == 1:
                    # sentinel, skip
                    return (idx, b"", None)
                # Copy out bytes immediately (string_at does a memcpy). Must complete before
                # the next GET overwrites the buffer, but each client has its own buffer so
                # threads don't overwrite each other.
                data = client.buffer_view(n)
                return (idx, data, None)

            # Submit N GETs concurrently. At N=1 the executor still spawns a task; the tiny
            # overhead is not worth optimizing away.
            futs = [self._executor.submit(_get_one, i, lp) for i, lp in enumerate(layer_parts)]
            # Collect results in idx order to keep segments in the same order as the N
            # sub-chunk concatenation order.
            results: list[tuple[int, bytes | None, str | None]] = [None] * N  # type: ignore
            for f in futs:
                idx, data, err = f.result()
                results[idx] = (idx, data, err)

            had_transport_error = False
            for idx, data, err in results:
                if err is not None:
                    had_transport_error = True
                    logger.warning(
                        "RDMA get_chunks part error attempt=%d: %s",
                        attempt + 1, err,
                    )
                    break
                if data is None:
                    # part miss → overall miss (matches original logic)
                    return None, False

            if had_transport_error:
                if attempt > 0:
                    return None, True
                self._reset_rdma_client_pool("get_chunks transport error")
                continue

            merged: list[bytes] = []
            for idx, data, _err in results:
                assert data is not None  # already excluded by the part-miss check above
                if data == b"":
                    continue  # sentinel
                merged.append(data)
            return (merged if merged else None), False

        return None, True

    def put(
        self,
        key: str,
        layer_name: str,
        data: bytes,
        meta: BlockMeta | None = None,
    ) -> None:
        kv_key = self._object_key(key, layer_name)
        kv_meta = self._to_kv_metadata(meta)
        # Streaming: split data into chunk_size pieces and iterate through PutStream RPC
        chunks_iter = self._chunk_iter(data, self._chunk_size)
        ok = self._client.put_stream(kv_key, chunks_iter, kv_meta)
        if not ok:
            raise RuntimeError(
                f"KVService put_stream failed: key={key} layer={layer_name} bytes={len(data)}"
            )

    def get(self, key: str, layer_name: str) -> bytes | None:
        kv_key = self._object_key(key, layer_name)
        try:
            get_cached = getattr(self._client, "get_cached", None)
            if get_cached is not None:
                result = get_cached(kv_key)
                data = result.data if result is not None else None
            else:
                data = self._client.get_stream(kv_key, chunk_buffer=self._chunk_size)
        except Exception as e:
            # NotFound mapping: get_stream raises RpcError internally for NotFound; map to None
            import grpc  # type: ignore

            if isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            logger.warning(
                "KVService get_stream error for key=%s layer=%s: %s",
                key, layer_name, e,
            )
            return None
        return data if data else None

    def exists(self, key: str) -> bool:
        # KV Service has no "list layers by prefix" API; we use the "__combined__" layer as
        # an existence probe (matches the storage.exists(prefix_key) call semantics in
        # connector.py, since the vLLM connector only writes a single __combined__ layer).
        kv_key = self._object_key(key, "__combined__")
        try:
            return self._client.exists(kv_key)
        except Exception as e:
            logger.warning("KVService exists error for key=%s: %s", key, e)
            return False

    def probe_layer_exists(self, key: str, layer_name: str) -> bool:
        """Check whether the given StorageBackend key/layer has been committed to KVService.

        This is the vLLM Scheduler's remote prefix probe entry point. Callers only pass
        StorageBackend-level key/layer; encoding of namespace and opaque object_key is
        this backend's responsibility, so Connector need not depend directly on the
        KVService wire key format.
        """
        kv_key = self._object_key(key, layer_name)
        try:
            lookup = getattr(self._client, "lookup_object", None)
            if lookup is not None:
                return lookup(kv_key) is not None
            return self._client.exists(kv_key)
        except Exception as e:
            logger.warning(
                "KVService probe_layer_exists error for key=%s layer=%s: %s",
                key, layer_name, e,
            )
            return False

    def lookup_object(self, key: str, layer_name: str) -> Any | None:
        """Return the KVService descriptor for the given StorageBackend key/layer."""
        kv_key = self._object_key(key, layer_name)
        try:
            lookup = getattr(self._client, "lookup_object", None)
            if lookup is None:
                return None
            return lookup(kv_key)
        except Exception as e:
            logger.warning(
                "KVService lookup_object error for key=%s layer=%s: %s",
                key, layer_name, e,
            )
            return None

    def _lookup_descriptor_for_rdma(self, key: str, layer_name: str) -> Any | None:
        """Metadata lookup used by the RDMA descriptor GET; falls back to key-based RDMA on failure."""
        try:
            lookup = getattr(self._client, "lookup_object", None)
            if lookup is None:
                return None
            return lookup(self._object_key(key, layer_name))
        except Exception as e:
            logger.warning(
                "KVService RDMA descriptor lookup error for key=%s layer=%s: %s",
                key[:8],
                layer_name,
                e,
            )
            return None

    @staticmethod
    def _descriptor_identity(descriptor: Any) -> tuple[Any, Any, Any, Any, Any]:
        """Descriptor version identity; used to check whether the local cache is stale."""
        return (
            getattr(descriptor, "object_handle", ""),
            getattr(descriptor, "object_generation", 0),
            getattr(descriptor, "content_etag", ""),
            getattr(descriptor, "layout_version", 0),
            getattr(descriptor, "size", 0),
        )

    def delete(self, key: str) -> None:
        # Same layer convention as exists; if we support multi-layer in the future,
        # scan each layer_name.
        kv_key = self._object_key(key, "__combined__")
        try:
            self._client.delete(kv_key)
        except Exception as e:
            logger.warning("KVService delete error for key=%s: %s", key, e)

    def get_meta(self, key: str) -> BlockMeta | None:
        kv_key = self._object_key(key, "__combined__")
        try:
            # A single-RPC way for KV Service to fetch meta: use the get path and throw
            # away the data (server single-get includes the metadata field); but here we
            # want meta without data — get_meta doesn't exist.
            # Fallback: call get, take (data, meta), discard data. For vLLM callers,
            # get_meta typically happens on the exists/lookup path — a cold path where
            # the small overhead is acceptable.
            result = self._client.get(kv_key)
            if result is None:
                return None
            _data, kv_meta = result
            return BlockMeta(
                num_tokens=kv_meta.num_tokens,
                num_layers=kv_meta.num_layers,
                dtype=kv_meta.dtype,
                shape=list(kv_meta.shape),
                compressed=kv_meta.compressed,
                compression_level=kv_meta.compression_level,
            )
        except Exception as e:
            logger.warning("KVService get_meta error for key=%s: %s", key, e)
            return None

    def capacity_usage(self) -> tuple[int, int]:
        # Remote KV Service capacity is managed by the server; here we return the server
        # stats' L1 cache size + 0 as an approximation. Real persistent-tier capacity lives
        # in the server's router/storage_tier.
        try:
            stats = self._client.stats()
            # No server total-capacity field; return (used, MAX_INT64 as placeholder)
            return stats.l1_cache_size_bytes, (1 << 63) - 1
        except Exception:
            return 0, (1 << 63) - 1

    # ===== Multi-segment direct interfaces (used by the vLLM connector) =====
    # Accept/return list[bytes] directly instead of assembling into one large bytes
    # inside the backend.
    # Purpose: on connector get_finished, serialize N GPU blocks and PUT them in one
    # streaming put_chunks; on load, get_chunks fetches the corresponding chunks and
    # writes back to GPU block-by-block.

    def put_chunks(
        self,
        key: str,
        layer_name: str,
        segments: list[bytes],
        meta: BlockMeta | None = None,
    ) -> None:
        """Streaming PUT of multiple bytes segments (no assembly). Equivalent to put but
        the caller has already split the data.

        **Multi-lane parallel**: split segments contiguously into `self._parallel` parts;
        each part uses an independent KVClient/channel to concurrently put_stream into a
        different layer `{layer_name}_p{i}of{N}`. This breaks the ~180 MB/s single-stream
        transport ceiling so the server's 8-disk striping shines.

        When parallel=1 this degrades to a single-stream path (backwards compatible).

        **RDMA fast path** (rdma_enabled + parallel=1 + supports_put): skip gRPC and
        assemble segments into a pinned host buffer then RDMA WRITE to the server; the
        server writes zero-memcpy O_DIRECT to disk. Applies to large values (>=
        striping_threshold); small values still go through gRPC.
        """
        n_segs = len(segments)
        # Important: N must equal self._parallel (fixed), not degrade because segments is small.
        # Otherwise connector-side prefix_probe using a fixed _p0of{parallel} won't find the
        # actual _p0of{N<parallel} layer.
        # If segments < parallel, pad the trailing parts with empty segments so layer names
        # stay stable.
        N = self._parallel
        kv_meta = self._to_kv_metadata(meta)

        # ===== RDMA PUT fast path (parallel=1 + supports_put + large value) =====
        if N == 1 and self._rdma_enabled:
            total = sum(len(s) for s in segments)
            # Only large values go through RDMA (the one-time assembly cost is smaller than
            # the gRPC path). The threshold aligns with the server-side striping_threshold
            # (default 8MB); here we hard-code 4MB — a bit conservative to avoid small
            # layers going through gRPC.
            if total >= 4 * 1024 * 1024 and self._try_rdma_put(key, layer_name, segments, total):
                return
            if total >= 4 * 1024 * 1024 and not self._rdma_fallback_to_grpc:
                raise RuntimeError(
                    f"RDMA put_chunks failed for key={key[:8]} layer={layer_name} bytes={total} "
                    "and rdma_fallback_to_grpc is disabled"
                )
            # RDMA unavailable or failed → continue on the gRPC path

        if N == 1:
            # Fast path: skip the executor, use a single client directly
            kv_key = self._object_key(key, layer_name)
            ok = self._client.put_stream(kv_key, iter(segments), kv_meta)
            if not ok:
                total = sum(len(s) for s in segments)
                raise RuntimeError(
                    f"KVService put_chunks failed: key={key} layer={layer_name} segs={n_segs} bytes={total}"
                )
            return

        # Multi-lane parallel: split contiguously into N parts, each lane on its own channel
        # Partition strategy: as even as possible, remainder distributed to the front
        sizes = self._partition_count(n_segs, N)  # e.g. n_segs=13 N=4 → [4,3,3,3]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)

        def _put_one(part_idx: int) -> bool:
            start = offsets[part_idx]
            end = offsets[part_idx + 1]
            part_segments = segments[start:end]
            # Important: even if this part has no real segments (segments < N), write an
            # empty part; otherwise the GET side treats the missing part as a miss → the
            # whole prefix misses.
            # Use a 1-byte sentinel (server-side chunks_cache accepts an empty list but
            # RocksDB metadata requires at least one segment). This keeps layer names
            # stable so probe/load work.
            if not part_segments:
                part_segments = [b"\x00"]
            part_layer = f"{layer_name}_p{part_idx}of{N}"
            kv_key = self._object_key(key, part_layer)
            # meta only on part 0 (save serialization; the other parts carry the same meta
            # but the server doesn't rely on it)
            m = kv_meta if part_idx == 0 else None
            return self._client_pool[part_idx].put_stream(kv_key, iter(part_segments), m)

        # Submit N lanes concurrently
        futures = [self._executor.submit(_put_one, i) for i in range(N)]
        results = [f.result() for f in futures]
        if not all(results):
            failed = [i for i, ok in enumerate(results) if not ok]
            raise RuntimeError(
                f"KVService put_chunks parallel failed: key={key} layer={layer_name} "
                f"N={N} failed_parts={failed}"
            )

    def get_chunks(self, key: str, layer_name: str) -> list[bytes] | None:
        """Streaming GET of multiple bytes segments (no assembly). Returns None if the key
        doesn't exist.

        **Multi-lane parallel**: symmetric with put_chunks; concurrently fetch data from
        N `{layer_name}_p{i}of{N}` layers then concatenate back into the original segments
        list (order preserved: part0+part1+...+partN-1).

        When N == 1 falls back to a direct GET on the original layer_name (keeps the
        single-channel path).

        **RDMA fast path**: when rdma_enabled=True prefer the RDMA tier; fall back to
        gRPC on failure.
        """
        # Entry marker (debug; using print because the vllm worker mutes our logger)
        import sys
        print(f"[CS GET_CHUNKS] enter key={key[:8]} layer={layer_name} rdma_enabled={self._rdma_enabled} pid={os.getpid()}", file=sys.stderr, flush=True)

        # ===== RDMA fast path =====
        if self._rdma_enabled:
            t0 = __import__('time').perf_counter()
            result, had_error = self._get_chunks_rdma(key, layer_name)
            dt = (__import__('time').perf_counter() - t0) * 1000.0
            n_total = sum(len(b) for b in result) if result else 0
            if result is not None:
                msg = (
                    f"[CS RDMA] get_chunks key={key[:8]} layer={layer_name} "
                    f"parts={len(result)} bytes={n_total} time={dt:.1f}ms "
                    f"BW={n_total / max(dt/1000.0, 1e-6) / (1024*1024*1024):.2f}GB/s "
                    f"pid={os.getpid()}\n"
                )
                logger.warning(msg.strip())
                # Back up to a persistent log (vllm worker stderr may be swallowed)
                try:
                    log_path = os.environ.get("CS_RDMA_LOG", "/data/cs-logs/rdma_bw.log")
                    with open(log_path, "a") as f:
                        f.write(msg)
                except Exception:
                    pass
            # RDMA hit / confirmed real miss: return directly.
            # Only transport errors continue to this call's gRPC fallback, avoiding
            # swallowing a transient RDMA failure as a miss.
            if result is not None or not had_error:
                return result
            if not self._rdma_fallback_to_grpc:
                raise RuntimeError(
                    f"RDMA get_chunks transport error for key={key[:8]} layer={layer_name} "
                    "and rdma_fallback_to_grpc is disabled"
                )
            # Otherwise fall through to gRPC

        N = self._parallel

        if N == 1:
            kv_key = self._object_key(key, layer_name)
            try:
                cached_get = getattr(self._client, "get_stream_chunks_cached", None)
                if cached_get is not None:
                    chunks = cached_get(kv_key)
                else:
                    chunks = self._client.get_stream_chunks(kv_key)
            except Exception as e:
                import grpc  # type: ignore

                if isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.NOT_FOUND:
                    return None
                logger.warning("KVService get_chunks error key=%s: %s", key, e)
                return None
            return chunks if chunks else None

        # Multi-lane parallel: N concurrent GETs
        def _get_one(part_idx: int) -> list[bytes] | None:
            part_layer = f"{layer_name}_p{part_idx}of{N}"
            kv_key = self._object_key(key, part_layer)
            try:
                client = self._client_pool[part_idx]
                cached_get = getattr(client, "get_stream_chunks_cached", None)
                if cached_get is not None:
                    return cached_get(kv_key)
                return client.get_stream_chunks(kv_key)
            except Exception as e:
                import grpc  # type: ignore

                if isinstance(e, grpc.RpcError) and e.code() == grpc.StatusCode.NOT_FOUND:
                    return None
                logger.warning("KVService get_chunks(part %d) error key=%s: %s", part_idx, key, e)
                return None

        futures = [self._executor.submit(_get_one, i) for i in range(N)]
        parts = [f.result() for f in futures]
        # If any part is None (not found), treat overall as miss
        if any(p is None for p in parts):
            return None
        # Concatenate in part order back into a complete segments list. Skip 1-byte sentinels
        # (empty-part placeholders).
        merged: list[bytes] = []
        for p in parts:
            for seg in p:
                if len(seg) == 1 and seg == b"\x00":
                    continue  # sentinel placeholder, skip
                merged.append(seg)
        return merged if merged else None

    @staticmethod
    def _partition_count(total: int, N: int) -> list[int]:
        """Split total items into N parts, as even as possible, with the remainder added
        to the front.
        Example: total=13 N=4 → [4,3,3,3]; total=8 N=8 → [1,1,1,1,1,1,1,1]
        """
        if N <= 1:
            return [total]
        base = total // N
        extra = total % N
        return [base + 1 if i < extra else base for i in range(N)]

    # ===== Helpers =====

    @staticmethod
    def _chunk_iter(data: bytes, chunk_size: int):
        if not data:
            yield b""
            return
        for off in range(0, len(data), chunk_size):
            yield data[off:off + chunk_size]

    @staticmethod
    def _to_kv_metadata(meta: BlockMeta | None) -> Any | None:
        if meta is None:
            return None
        from contextstore.kvservice_client import KVMetadata

        return KVMetadata(
            num_tokens=meta.num_tokens,
            num_layers=meta.num_layers,
            dtype=meta.dtype,
            shape=meta.shape,
            compressed=meta.compressed,
            compression_level=meta.compression_level,
            created_at=0,
            last_accessed_at=0,
        )
