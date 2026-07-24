from __future__ import annotations

"""ContextStore vLLM v1 KV Connector (CUDA-compatible).

Design:
1) ContextStoreConnector is a router: splits into scheduler / worker implementations
   based on KVConnectorRole.
2) Scheduler side (SchedulerImpl): maintains the PrefixIndex; marks load/store tasks
   in build_connector_meta; bind_gpu_block_pool holds a reference but the first
   version does not block eviction.
3) Worker side (WorkerImpl): register_kv_caches receives dict[layer_name, tensor];
   save_kv_layer / wait_for_save / start_load_kv / wait_for_layer_load are all no-ops
   (never invoked on CUDA); get_finished is the real trigger point for GPU↔KVService
   transfers.

Why we do not rely on save_kv_layer:
- Under the vLLM CUDA platform opaque_attention_op()=True → use_direct_call=False
- Attention goes through torch.ops.vllm.unified_attention_with_output (custom op)
- This bypasses the @maybe_transfer_kv_layer decorator → save_kv_layer is never called
- Same pattern as vLLM's SimpleCPUOffloadConnector

Data format:
- Key = (model_id, prefix_hash, layer_name="block") + suffix block_idx
- Each block contains the KV of every layer; vLLM block_size tokens, multi-layer
  K/V dumped as one segment
- Measured on Qwen2.5-0.5B: ~200KB/block; KVService streaming handles this easily

First-version limitations:
- Synchronous PUT/GET (blocking inside get_finished); can be moved to cuda stream +
  background thread later
- tensor_parallel_size=1; TP>1 requires cross-worker coordination
- Does not use GDS / CUDA IPC (supported by the Rust server but client wiring is
  complex)
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from contextstore.core.config import ContextStoreConfig
from contextstore.core.engine import ContextStoreEngine

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

logger = logging.getLogger(__name__)


def _append_perf_log(line: str) -> None:
    """Optionally write a diagnostic performance log line. Disabled by default to
    avoid hot-path file I/O."""
    if not _perf_log_enabled():
        return
    path = os.environ.get("CS_PERF_LOG_PATH", "/tmp/cs_zc_perf.log")
    try:
        with open(path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _perf_log_enabled() -> bool:
    return os.environ.get("CS_PERF_LOG", "0") == "1"


def _format_perf_time(ns: int) -> str:
    """Format epoch_ns as a log-friendly local timestamp."""
    sec = ns // 1_000_000_000
    nsec = ns % 1_000_000_000
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(sec))}.{nsec:09d}"


def _prefix_key(model_id: str, token_ids: list[int], num_tokens: int) -> str:
    payload = f"{model_id}:{','.join(str(t) for t in token_ids[:num_tokens])}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ===== Serialization helpers: support bfloat16 (which numpy does not handle directly) =====
# Approach: for bit widths numpy does not support, first view as an integer type of the
# same width (uint16/uint8/...). tensor.untyped_storage() + element_size() gives the
# byte count; reconstruction uses ctypes or torch.frombuffer.


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """CPU 1D contiguous tensor → bytes. Supports bfloat16/float16/float32, etc.

    Implementation: use view(torch.uint8) to reinterpret any dtype as a byte stream,
    then `numpy().tobytes()` (numpy uint8 is a zero-copy view + a single memcpy).

    Using bytes(t.untyped_storage()) is extremely slow on PyTorch 2.x (measured 2s
    for 917KB; likely goes through the Python iter __bytes__ path). view-as-uint8 +
    numpy is 1000+× faster.
    """
    assert t.is_cpu and t.is_contiguous() and t.dim() == 1
    return t.view(torch.uint8).numpy().tobytes()


def _bytes_to_tensor(data: bytes, dtype: torch.dtype, expected_numel: int) -> torch.Tensor:
    """bytes → CPU 1D tensor of dtype. expected_numel validates the length matches."""
    # torch.frombuffer creates a view directly from bytes; bfloat16 is supported too.
    # Note: data must be bytes-like; torch.frombuffer does not copy (lifetime is
    # bound to the buffer). For safety (data comes from a KVService stream and may
    # be GC'd), copy it once.
    expected_bytes = expected_numel * torch.tensor([], dtype=dtype).element_size()
    if len(data) != expected_bytes:
        raise RuntimeError(
            f"_bytes_to_tensor size mismatch: got {len(data)}B expected {expected_bytes}B "
            f"({expected_numel} × {dtype}={torch.tensor([], dtype=dtype).element_size()}B)"
        )
    # bytearray is writable; torch.frombuffer requires a writable buffer.
    return torch.frombuffer(bytearray(data), dtype=dtype)


# ===== Metadata is delivered from scheduler to worker via vLLM's pickle RPC =====


@dataclass
class _Spec:
    """Description of a single load or store task (scheduler→worker)."""
    req_id: str
    prefix_key: str
    gpu_block_ids: list[int]
    # Used on the load path: the prefix that the worker uses to line up with vLLM's
    # own slot_mapping (unused in this version; block-level passthrough).
    num_tokens: int = 0


@dataclass
class ContextStoreMeta(KVConnectorMetadata):
    """Scheduler → Worker metadata."""
    block_size: int = 16
    store_specs: list[_Spec] = field(default_factory=list)
    load_specs: list[_Spec] = field(default_factory=list)


# ===== Scheduler-side implementation =====


class _SchedulerImpl:
    def __init__(self, cs_config: ContextStoreConfig, engine: ContextStoreEngine, tp_size: int = 1):
        self._cs_config = cs_config
        self._engine = engine
        self._tp_size = tp_size  # scheduler needs tp_size to compute the probe layer name
        # request_id → (request, blocks, num_external_tokens); resolved into a LoadSpec in build_meta
        self._pending_loads: dict[str, tuple[Any, Any, int]] = {}
        # GPU block pool reference (held for future eviction control; first version
        # takes no action).
        self._gpu_block_pool: "BlockPool | None" = None
        # req_id -> prompt token ids / accumulated block ids. Under chunked prefill
        # the same request allocates blocks across multiple steps, so we cannot
        # rely on the first batch of blocks in scheduled_new_reqs alone.
        self._store_prompt_tokens: dict[str, list[int]] = {}
        self._store_block_ids: dict[str, list[int]] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        perf_enabled = _perf_log_enabled()
        wall_start_ns = time.time_ns() if perf_enabled else 0
        t_start = time.perf_counter() if perf_enabled else 0.0
        token_ids = list(request.prompt_token_ids or [])
        matched = self._engine.lookup(token_ids)
        lookup_ms = (time.perf_counter() - t_start) * 1000 if perf_enabled else 0.0

        # ===== Cross-process PrefixIndex rebuild + intra-process length-growth probing =====
        # Original design: on a fresh process the PrefixIndex is empty and lookup
        # returns 0; PROBE populates the local index from the remote side.
        # Bug fix (2026-06-10): within the same process, when prompt length grows
        # (e.g. 10k → 16k → 28k), the local trie hits at the longest length registered
        # by the first PROBE, subsequent longer prompts skip PROBE, and although the
        # server holds a longer entry (after R1 STORE finishes) the connector only
        # LOADs the shorter length.
        # Fix: change the trigger from `matched <= num_computed_tokens` to
        # `matched < full_cacheable_len`. As long as local hits are shorter than
        # the fully cacheable prompt length, PROBE the remote for a longer entry.
        # If PROBE fails (exists=False), matched is unchanged → partial hit + prefill,
        # so consistency is preserved (see I1/I2/I3 in the docs).
        # Cost: one extra exists RPC on the hot path (< 1ms in practice); after a hit
        # the entry is registered locally and later requests with the same prompt
        # hit locally without PROBE. Only paid once for the first occurrence of a
        # new length.
        block_size = self._cs_config.block_size
        full_cacheable = (len(token_ids) // block_size) * block_size
        probe_storage = self._probe_storage_backend()
        probed = False
        found = False
        prefix_key = ""
        missing_layers: list[str] = []
        if matched < full_cacheable and probe_storage is not None:
            num_blocks = full_cacheable // block_size
            if num_blocks > 0:
                num_cacheable = full_cacheable
                prefix_key = _prefix_key(self._cs_config.model_id, token_ids, num_cacheable)
                # Probe: the actual storage layer names get a _p{i}of{N} suffix from
                # put_chunks parallel writes. Probing part0 alone is enough to
                # confirm a rank's entry has been written and its metadata
                # committed; when TP>1 every rank's shard must exist, otherwise
                # the scheduler could declare HIT after only tp0 finished, and
                # the tp1/2/3 workers would LOAD MISS and then zero their KV.
                probe_layers = self._probe_layer_names(num_blocks)
                # The connector only passes the StorageBackend-level (prefix_key,
                # layer_name). KVService's (namespace, opaque object_key) mapping
                # must stay inside the backend.
                probe_fn = getattr(probe_storage, "probe_layer_exists")
                try:
                    probed = True
                    probe_start = time.perf_counter() if perf_enabled else 0.0
                    for probe_layer in probe_layers:
                        found_layer = probe_fn(prefix_key, probe_layer)
                        if not found_layer:
                            missing_layers.append(probe_layer)
                    found = not missing_layers
                    if perf_enabled:
                        probe_end_ns = time.time_ns()
                        _append_perf_log(
                            "SCHED_PROBE "
                            f"ts={_format_perf_time(probe_end_ns)} start_ns={wall_start_ns} "
                            f"end_ns={probe_end_ns} req={request.request_id[:16]} "
                            f"key={prefix_key[:8]} tokens={len(token_ids)} "
                            f"matched_before={matched} full={full_cacheable} "
                            f"layers={len(probe_layers)} found={int(found)} "
                            f"missing={len(missing_layers)} "
                            f"duration_ms={(time.perf_counter() - probe_start) * 1000:.3f}"
                        )
                    if found:
                        # Remote has it! Register into the local PrefixIndex so
                        # subsequent identical prompts skip the remote lookup.
                        self._engine.register_prefix(token_ids, num_cacheable)
                        matched = num_cacheable
                        logger.info(
                            "CS sched prefix_probe HIT: tokens=%d matched=%d "
                            "prefix=%s layers=%s (cross-process)",
                            len(token_ids), matched, prefix_key[:8], probe_layers,
                        )
                    else:
                        logger.info(
                            "CS sched prefix_probe MISS: tokens=%d prefix=%s missing_layers=%s",
                            len(token_ids), prefix_key[:8], missing_layers,
                        )
                except Exception as e:
                    logger.warning("CS sched prefix_probe error: %s", e)

        if matched <= num_computed_tokens:
            if perf_enabled:
                wall_end_ns = time.time_ns()
                _append_perf_log(
                    "SCHED_MATCH "
                    f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} "
                    f"end_ns={wall_end_ns} req={request.request_id[:16]} "
                    f"tokens={len(token_ids)} computed={num_computed_tokens} "
                    f"matched={matched} external=0 probed={int(probed)} found={int(found)} "
                    f"key={prefix_key[:8]} missing={len(missing_layers)} "
                    f"lookup_ms={lookup_ms:.3f} duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
                )
            return 0, False
        # vLLM requires at least 1 token to be handled by prefill (sched.schedule()
        # asserts num_new_tokens > 0). If the prompt length is an exact multiple of
        # block_size and every token is cached, leave the last token for prefill.
        # This is the same workaround lmcache and peers use.
        total = len(token_ids)
        max_match = total - num_computed_tokens - 1
        n_external = matched - num_computed_tokens
        if n_external > max_match:
            n_external = max_match
        if n_external <= 0:
            if perf_enabled:
                wall_end_ns = time.time_ns()
                _append_perf_log(
                    "SCHED_MATCH "
                    f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} "
                    f"end_ns={wall_end_ns} req={request.request_id[:16]} "
                    f"tokens={len(token_ids)} computed={num_computed_tokens} "
                    f"matched={matched} external=0 probed={int(probed)} found={int(found)} "
                    f"key={prefix_key[:8]} missing={len(missing_layers)} "
                    f"lookup_ms={lookup_ms:.3f} duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
                )
            return 0, False
        if perf_enabled:
            wall_end_ns = time.time_ns()
            _append_perf_log(
                "SCHED_MATCH "
                f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} "
                f"end_ns={wall_end_ns} req={request.request_id[:16]} "
                f"tokens={len(token_ids)} computed={num_computed_tokens} "
                f"matched={matched} external={n_external} probed={int(probed)} found={int(found)} "
                f"key={prefix_key[:8]} missing={len(missing_layers)} "
                f"lookup_ms={lookup_ms:.3f} duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
            )
        return n_external, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._pending_loads[request.request_id] = (request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> ContextStoreMeta:
        perf_enabled = _perf_log_enabled()
        wall_start_ns = time.time_ns() if perf_enabled else 0
        t_start = time.perf_counter() if perf_enabled else 0.0
        block_size = self._cs_config.block_size
        meta = ContextStoreMeta(block_size=block_size)

        # ---- LOAD specs ----
        for req_id, (request, blocks, num_ext) in self._pending_loads.items():
            token_ids = list(request.prompt_token_ids or [])
            block_ids_tuple = blocks.get_block_ids()
            block_ids = list(block_ids_tuple[0]) if block_ids_tuple else []
            if not block_ids:
                continue
            matched = self._engine.index.lookup_prefix(token_ids)
            if matched <= 0:
                continue
            num_blocks_to_load = min(matched // block_size, len(block_ids))
            if num_blocks_to_load <= 0:
                continue
            prefix_key = _prefix_key(self._cs_config.model_id, token_ids, num_blocks_to_load * block_size)
            meta.load_specs.append(_Spec(
                req_id=req_id,
                prefix_key=prefix_key,
                gpu_block_ids=block_ids[:num_blocks_to_load],
                num_tokens=num_blocks_to_load * block_size,
            ))
            logger.info(
                "CS sched LOAD_SPEC req=%s matched_tokens=%d block_size=%d "
                "available_gpu_blocks=%d load_blocks=%d load_tokens=%d "
                "prefix=%s gpu_block_first=%s gpu_block_last=%s",
                req_id[:8],
                matched,
                block_size,
                len(block_ids),
                num_blocks_to_load,
                num_blocks_to_load * block_size,
                prefix_key[:8],
                block_ids[0] if block_ids else "NA",
                block_ids[num_blocks_to_load - 1] if num_blocks_to_load else "NA",
            )

        # ---- STORE tracking (new requests + incremental blocks for cached requests) ----
        n_new = len(scheduler_output.scheduled_new_reqs)
        if n_new > 0:
            logger.info("CS sched build_meta: scheduled_new_reqs=%d", n_new)
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            if token_ids:
                self._store_prompt_tokens[new_req.req_id] = token_ids
            if new_req.block_ids and new_req.block_ids[0]:
                self._store_block_ids[new_req.req_id] = list(new_req.block_ids[0])

        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_blocks in zip(cached.req_ids, cached.new_block_ids):
            if new_blocks is None or not new_blocks or not new_blocks[0]:
                continue
            if req_id in cached.resumed_req_ids or req_id not in self._store_block_ids:
                self._store_block_ids[req_id] = list(new_blocks[0])
            else:
                self._store_block_ids[req_id].extend(new_blocks[0])

        # ---- STORE specs (every active request in this step carries its latest full block list) ----
        active_req_ids: set[str] = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        active_req_ids.update(cached.req_ids)
        for req_id in active_req_ids:
            token_ids = self._store_prompt_tokens.get(req_id)
            block_ids = self._store_block_ids.get(req_id)
            if not token_ids:
                continue
            if not block_ids:
                logger.info("CS sched STORE skip req=%s reason=no_block_ids", req_id[:8])
                continue

            num_tokens = len(token_ids)
            num_blocks = num_tokens // block_size
            if num_blocks <= 0:
                logger.info(
                    "CS sched STORE skip req=%s reason=too_few_tokens tokens=%d bs=%d",
                    req_id[:8], num_tokens, block_size,
                )
                continue
            if len(block_ids) < num_blocks:
                logger.info(
                    "CS sched STORE defer req=%s have_blocks=%d need_blocks=%d",
                    req_id[:8], len(block_ids), num_blocks,
                )
                continue

            num_cacheable = num_blocks * block_size
            prefix_key = _prefix_key(self._cs_config.model_id, token_ids, num_cacheable)
            already_have = self._engine.index.lookup_prefix(token_ids) >= num_cacheable
            if already_have:
                logger.info(
                    "CS sched STORE skip req=%s reason=index_already_has prefix=%s",
                    req_id[:8], prefix_key[:8],
                )
                continue

            meta.store_specs.append(_Spec(
                req_id=req_id,
                prefix_key=prefix_key,
                gpu_block_ids=block_ids[:num_blocks],
                num_tokens=num_cacheable,
            ))
            logger.info(
                "CS sched STORE req=%s tokens=%d blocks=%d prefix=%s",
                req_id[:8], num_tokens, num_blocks, prefix_key[:8],
            )

        self._pending_loads.clear()
        if meta.store_specs or meta.load_specs:
            logger.info("CS sched meta: %d store, %d load",
                        len(meta.store_specs), len(meta.load_specs))
        if perf_enabled and (meta.store_specs or meta.load_specs):
            wall_end_ns = time.time_ns()
            load_keys = ",".join(spec.prefix_key[:8] for spec in meta.load_specs)
            store_keys = ",".join(spec.prefix_key[:8] for spec in meta.store_specs)
            _append_perf_log(
                "SCHED_BUILD_META "
                f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} end_ns={wall_end_ns} "
                f"load={len(meta.load_specs)} store={len(meta.store_specs)} "
                f"load_keys={load_keys} store_keys={store_keys} "
                f"duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
            )
        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        # Do NOT register_prefix here: worker STORE is asynchronous and may not have
        # finished yet. If we register, later requests with the same prompt would
        # scheduler.lookup_prefix() hit → issue a LOAD spec → worker get_chunks miss
        # (server has not finished the write) → serve stale GPU data → wrong output.
        # Correct approach: scheduler always goes through prefix_probe (query server
        # exists — the authoritative check). server only reports storage.exists()==true
        # after PUT completes, which naturally guarantees consistency.
        if block_ids:
            self._store_block_ids[request.request_id] = list(block_ids)
        return False, None

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        # First version only holds the reference; later we can call pool.touch()
        # to prevent eviction until store completes.
        self._gpu_block_pool = gpu_block_pool

    def _probe_layer_names(self, num_blocks: int) -> list[str]:
        """Return the storage layer names the scheduler must probe.

        TP=1 checks the key with no rank suffix; TP>1 must check every rank's shard.
        For parallel writes we only probe part0 because put_chunks writes a sentinel
        for empty parts, so part0 existing means that rank's PUT has committed
        metadata.
        """
        probe_storage = self._probe_storage_backend()
        parallel = getattr(probe_storage, "_parallel", 1)
        tp_size = max(self._tp_size, 1)
        ranks: list[int | None] = [None] if tp_size == 1 else list(range(tp_size))
        layers: list[str] = []
        for rank in ranks:
            base = f"blocks_{num_blocks}"
            if rank is not None:
                base = f"{base}_tp{rank}"
            if parallel > 1:
                base = f"{base}_p0of{parallel}"
            layers.append(base)
        return layers

    def _probe_storage_backend(self) -> Any | None:
        """Return the actual backend that supports KVService layer-level exists.

        ContextStoreEngine may wrap KVServiceBackend inside HostMemoryBackend. The
        scheduler's cross-process PROBE must reach the real KVService client;
        plain local backends have no remote layer-level exists semantics and skip
        PROBE.
        """
        storage = self._engine.storage
        while storage is not None:
            # Both the single-node KVServiceBackend and the HA backend expose the
            # same backend-level probe interface. The connector does not touch
            # backend._client or construct KVService ObjectKeys.
            if hasattr(storage, "probe_layer_exists"):
                return storage
            storage = getattr(storage, "_wrapped", None)
        return None


# ===== Worker-side implementation =====


class _WorkerImpl:
    """The real execution end of KV transfer.

    Timing (aligned with vLLM v1 + lmcache):

    forward step N:
      bind_connector_metadata(meta)        ← scheduler passes this step's LOAD/STORE specs
      start_load_kv(forward_ctx)           ← **synchronously** pull KV from KVService to GPU
                                              so the attention data vLLM is about to
                                              use is correct
      [model forward runs attention]       ← by now GPU KV is populated
      wait_for_save()                       ← no-op (save_kv_layer is not called on CUDA)
      get_finished(finished_req_ids):
        - Async enqueue STORE specs (thread pool runs in background, returns immediately)
        - Returns (sent_now, recv_now):
          * sent_now = _pending_done_sent ∩ finished_req_ids
            (STORE enqueued in this or an earlier step has finished and vLLM has
             told us the req is finished, so ack back to vLLM to free the blocks)
          * recv_now = None (LOAD is done synchronously in start_load_kv; no async
             reporting needed)
      clear_connector_metadata()

    Correctness notes:
    - The old implementation put LOAD in get_finished (after forward); vLLM ran
      attention on uninitialized GPU data and produced tokens inconsistent with
      the "real prefix + continuation" (verified: ContextStore RUN 1 and RUN 2
      differed, while lmcache stayed consistent).
    - The new implementation moves LOAD to start_load_kv so attention sees the
      correct data and RUN 2 output matches RUN 1.

    Async STORE protocol gotcha:
    - Every req_id in finished_sending must satisfy req.is_finished() (vLLM's
      scheduler asserts this).
    - So we cannot ack immediately after enqueue; we ack only when STORE actually
      completes AND vLLM has added the req to finished_req_ids. We also need to
      keep GPU blocks from being evicted before STORE finishes (TODO: hook into
      BlockPool ref count).
    """

    def __init__(self, cs_config: ContextStoreConfig, engine: ContextStoreEngine):
        self._cs_config = cs_config
        self._engine = engine
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._current_meta: ContextStoreMeta | None = None
        # Async STORE: runs on a thread pool, get_finished returns immediately.
        # Flow: _store_blocks_to_service → submit → future done callback adds req_id
        # to _pending_done_sent → get_finished returns _pending_done_sent ∩ finished_req_ids.
        #
        # max_workers=4 aligns with _pinned_pool_size so D2H/PUT across specs can
        # actually pipeline (the old max_workers=2 + single pinned buffer was
        # implicitly serial with no concurrency benefit).
        # Note: actual concurrency depends on pool_size (default 1 → effectively
        # serial); multiple workers only run in parallel when the pool is large.
        import concurrent.futures
        self._store_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("CS_STORE_WORKERS", "2")),
            thread_name_prefix="cs-store-async",
        )
        # In-flight STORE (req_id → Future); used by wait_for_save to block-wait.
        self._inflight_stores: dict[str, concurrent.futures.Future] = {}
        # req_ids whose STORE has finished (ack after vLLM reports them in finished_req_ids).
        import threading
        self._done_lock = threading.Lock()
        self._pending_done_sent: set[str] = set()
        # Remember which prefix_keys have already been enqueued (avoid duplicate stores for the same req).
        self._stored_keys: set[str] = set()
        # req_id -> latest complete store spec. Under chunked prefill we need to
        # keep this across steps and only submit when the request truly finishes.
        self._store_specs_by_req: dict[str, _Spec] = {}
        # ===== Async D2H / H2D =====
        # Dedicated cuda stream so GPU↔CPU memcpy truly runs in parallel with
        # model forward (rather than competing on the default stream); only
        # effective together with pinned host memory + non_blocking=True.
        self._copy_stream: torch.cuda.Stream | None = None
        # Reusable pinned buffer pool: per-spec size varies but most KV share dtype,
        # so grow on demand and reuse a large buffer to avoid cudaHostAlloc every
        # call (~1ms × dozens of calls = tens of ms wasted).
        #
        # **STORE pipeline design (v2, 2026-06-11)**:
        # Use a buffer pool (N independent pinned buffers) instead of a single
        # buffer so store_executor's multiple workers can truly pipeline (instead
        # of contending for one buffer and serializing).
        # - Each _store_blocks_to_service worker acquires a buf from queue.Queue,
        #   uses it, returns it.
        # - max_workers=2 + pool_size=2: spec_A D2H holds buf_0 while spec_B uses
        #   buf_1 concurrently.
        # - D2H (~50ms for a large spec) + put_chunks (~100ms) overlap across specs.
        # - With a single buffer, 2 workers contend and serialize implicitly, no
        #   real concurrency gain.
        #
        # _pinned_buf is kept for backward compat (LOAD path still uses a single
        # buffer); STORE uses _pinned_pool.
        self._pinned_buf: torch.Tensor | None = None  # LOAD path single reusable buffer
        # Shared-filesystem GDS staging allocation. It is GPU-resident and registered
        # once with cuFile, so LOAD avoids both host copies and per-request GPU allocs.
        self._gds_staging: torch.Tensor | None = None
        import queue
        self._pinned_pool: queue.Queue[torch.Tensor] = queue.Queue()
        # Default 1: matches the single-buffer behavior and keeps total pinned
        # memory unchanged (vLLM TP × 4 worker each has its own connector; pool
        # size=4 → 4×4=16 buffers of 1.5GB = 24GB pinned). Setting 1 still routes
        # through the pool interface, letting multiple workers serialize on one
        # buffer, matching the previous max_workers=2 + single buffer effective
        # concurrency.
        # To get real multi-buffer concurrency, set 2-4 but monitor total pinned
        # memory usage.
        self._pinned_pool_size = int(os.environ.get("CS_PINNED_POOL_SIZE", "1"))
        # TP rank (determined in register_kv_caches); used as the suffix for
        # STORE/LOAD keys.
        # When TP>1, each worker's KV tensor is only that rank's shard (different
        # head slice), so keys must distinguish rank — otherwise 4 workers would
        # read the same key and write into different GPUs, a correctness bug.
        self._tp_rank: int = 0
        self._tp_size: int = 1

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())
        if self._layer_names:
            sample = kv_caches[self._layer_names[0]]
            # Create the dedicated cuda stream (used for D2H/H2D) once we know the device.
            if sample.is_cuda and self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=sample.device)
            # Fetch TP rank (torch.distributed is initialized by now).
            try:
                from vllm.distributed.parallel_state import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                )
                self._tp_rank = get_tensor_model_parallel_rank()
                self._tp_size = get_tensor_model_parallel_world_size()
            except Exception:
                self._tp_rank = 0
                self._tp_size = 1

            # ===== Pre-warm pinned buffer + cuda stream =====
            # First LOAD pays cudaHostAlloc(150MB) + cuda stream startup ~ 300ms.
            # Allocating here at register time keeps that overhead off the first LOAD.
            # Capacity is estimated from the sample tensor's per-layer size ×
            # num_layers × 1.2 safety factor; _ensure_pinned still grows if the
            # actual nbytes exceeds this.
            try:
                # Conservative estimate: per_block_numel * num_blocks_max * num_layers * elem_size.
                # We do not know num_blocks_max, so use the sample tensor's second
                # dim (the num_blocks slot) fully populated.
                if sample.dim() >= 4 and sample.shape[0] == 2:
                    per_block_numel = sample[:, 0, ...].numel()
                else:
                    per_block_numel = sample[0].numel()
                # Cap: estimate from the current worker's KV cache block slots and
                # then apply a 2GiB ceiling to control pinned memory. The old value
                # of 256 blocks caused 16k prompts (1000 blocks) to re-cudaHostAlloc
                # on first LOAD and charge the ~1.2s pin cost to TTFT.
                if sample.dim() >= 4 and sample.shape[0] == 2:
                    est_max_blocks = int(sample.shape[1])
                else:
                    est_max_blocks = int(sample.shape[0])
                est_bytes = per_block_numel * est_max_blocks * len(self._layer_names) * sample.element_size()
                # Cap to 2 GiB to prevent OOM (5 worker × 2GB = 10GB; the container
                # cgroup may cap at 32-128GB and once you add RDMA buffers / server
                # L1 / vLLM heap it can blow up. In real scenarios _ensure_pinned
                # still grows on demand; this is only warmup preallocation.)
                est_bytes = min(est_bytes, 2 * 1024 * 1024 * 1024)
                self._ensure_pinned(est_bytes)
                gds_prewarm_bytes = min(
                    est_bytes,
                    max(self._cs_config.shared_gds_staging_max_mb, 1) * 1024 * 1024,
                )
                self._prewarm_shared_gds_buffer(gds_prewarm_bytes, sample.device)
                # Warm the stream: a tiny copy triggers lazy init.
                if self._copy_stream is not None:
                    with torch.cuda.stream(self._copy_stream):
                        warm = torch.empty(1024, dtype=torch.uint8, device=sample.device)
                        warm_cpu = self._pinned_buf[:1024]
                        warm.copy_(warm_cpu, non_blocking=True)
                    self._copy_stream.synchronize()
                self._prewarm_load_rdma_region()
                logger.info(
                    "CS worker pre-warmed: pinned=%d bytes copy_stream=ready",
                    est_bytes,
                )
            except Exception as e:
                logger.warning("CS worker pre-warm failed: %s", e)

            logger.info(
                "CS worker register_kv_caches: layers=%d sample=%s dtype=%s device=%s "
                "copy_stream=%s tp_rank=%d/%d",
                len(self._layer_names), list(sample.shape), sample.dtype, sample.device,
                self._copy_stream is not None, self._tp_rank, self._tp_size,
            )

    def _prewarm_load_rdma_region(self) -> None:
        """Pre-warm the LOAD RDMA connection and external MR so the first cache
        hit does not charge the fixed cost to TTFT."""
        if not self._cs_config.rdma_enabled or not self._cs_config.rdma_prewarm_load_region:
            return
        if self._pinned_buf is None:
            return
        storage = self._engine.storage
        if not (
            hasattr(storage, "supports_zerocopy")
            and hasattr(storage, "ensure_rdma_region")
        ):
            return
        t0 = time.perf_counter()
        try:
            if not storage.supports_zerocopy():
                return
            t_support = time.perf_counter()
            region_id = storage.ensure_rdma_region(
                self._pinned_buf.data_ptr(), self._pinned_buf.numel()
            )
            t_region = time.perf_counter()
            logger.warning(
                "CS worker pre-warmed LOAD RDMA region: region_id=%d size=%dMB "
                "support_ms=%.1f region_ms=%.1f total_ms=%.1f",
                region_id,
                self._pinned_buf.numel() // (1024 * 1024),
                (t_support - t0) * 1000,
                (t_region - t_support) * 1000,
                (t_region - t0) * 1000,
            )
        except Exception as e:
            logger.warning("CS worker LOAD RDMA pre-warm failed: %s", e)

    def _prewarm_shared_gds_buffer(self, size: int, device: torch.device) -> None:
        """Allocate and register the GDS staging buffer outside the first LOAD."""
        storage = self._engine.storage
        if not (
            hasattr(storage, "supports_shared_gds")
            and hasattr(storage, "prepare_shared_gds_buffer")
            and storage.supports_shared_gds()
        ):
            return
        staging = self._ensure_gds_staging(size, device)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        storage.prepare_shared_gds_buffer(staging.data_ptr(), staging.numel(), device_index)

    def _ensure_gds_staging(self, size: int, device: torch.device) -> torch.Tensor:
        current = self._gds_staging
        if current is not None and current.device == device and current.numel() >= size:
            return current
        self._gds_staging = torch.empty(size, dtype=torch.uint8, device=device)
        return self._gds_staging

    def bind_connector_metadata(self, meta: KVConnectorMetadata) -> None:
        perf_enabled = _perf_log_enabled()
        wall_start_ns = time.time_ns() if perf_enabled else 0
        t_start = time.perf_counter() if perf_enabled else 0.0
        if isinstance(meta, ContextStoreMeta):
            self._current_meta = meta
            if meta.store_specs or meta.load_specs:
                logger.info("CS worker bind_meta: store=%d load=%d",
                            len(meta.store_specs), len(meta.load_specs))
                if perf_enabled:
                    for spec in meta.store_specs:
                        _append_perf_log(
                            "BIND_META "
                            f"ts={_format_perf_time(wall_start_ns)} ns={wall_start_ns} "
                            f"pid={os.getpid()} req={spec.req_id[:16]} key={spec.prefix_key[:8]} "
                            f"kind=store blocks={len(spec.gpu_block_ids)} tokens={spec.num_tokens}"
                        )
                    for spec in meta.load_specs:
                        _append_perf_log(
                            "BIND_META "
                            f"ts={_format_perf_time(wall_start_ns)} ns={wall_start_ns} "
                            f"pid={os.getpid()} req={spec.req_id[:16]} key={spec.prefix_key[:8]} "
                            f"kind=load blocks={len(spec.gpu_block_ids)} tokens={spec.num_tokens}"
                        )
            for spec in meta.store_specs:
                self._store_specs_by_req[spec.req_id] = spec
            if perf_enabled and (meta.store_specs or meta.load_specs):
                wall_end_ns = time.time_ns()
                _append_perf_log(
                    "BIND_META_DONE "
                    f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} end_ns={wall_end_ns} "
                    f"pid={os.getpid()} store={len(meta.store_specs)} load={len(meta.load_specs)} "
                    f"duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
                )
        else:
            self._current_meta = None

    def clear_connector_metadata(self) -> None:
        self._current_meta = None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Called before model forward; **synchronously** pulls the KV for LOAD
        specs from KVService to GPU.

        vLLM v1 call order on CUDA:
          bind_connector_metadata → start_load_kv → [model forward] → wait_for_save → get_finished

        Data must be ready here, otherwise forward runs attention against
        stale/uninitialized GPU data and the generated tokens differ (measured:
        old implementation RUN 2 output ≠ RUN 1).

        LOAD cannot be async: vLLM forward cannot wait (once forward starts, LOAD
        finishing later in the step is too late). A strict async LOAD would
        require get_num_new_matched_tokens to return async_load=True so vLLM
        pauses the request one step — larger change, not done yet.
        """
        meta = self._current_meta
        if meta is None or not meta.load_specs:
            return
        perf_enabled = _perf_log_enabled()
        wall_start_ns = time.time_ns() if perf_enabled else 0
        t_start = time.perf_counter() if perf_enabled else 0.0
        reqs = ",".join(spec.req_id[:8] for spec in meta.load_specs)
        if perf_enabled:
            _append_perf_log(
                "START_LOAD_KV "
                f"ts={_format_perf_time(wall_start_ns)} start_ns={wall_start_ns} "
                f"pid={os.getpid()} reqs={reqs} count={len(meta.load_specs)}"
            )
        ok_count = 0
        for spec in meta.load_specs:
            try:
                self._load_blocks_from_service(spec)
                ok_count += 1
            except Exception as e:
                logger.error("CS worker LOAD failed req=%s: %s", spec.req_id[:8], e, exc_info=True)
        if perf_enabled:
            wall_end_ns = time.time_ns()
            _append_perf_log(
                "START_LOAD_KV_DONE "
                f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} end_ns={wall_end_ns} "
                f"pid={os.getpid()} reqs={reqs} count={len(meta.load_specs)} ok={ok_count} "
                f"duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Called after forward. Two things happen here:

        1. Async-enqueue STORE specs (thread pool runs GPU→CPU serialization + gRPC PUT).
        2. Report completed STOREs: _pending_done_sent ∩ finished_req_ids.

        Protocol:
        - finished_sending: pending sends carried across generation steps; returned
          only when the req_id appears in finished_req_ids (vLLM marks these reqs
          is_finished, so ack is safe); vLLM will then free the corresponding GPU
          blocks.
        - finished_recving: LOAD completed synchronously in start_load_kv, no need
          to report.
        """
        perf_enabled = _perf_log_enabled()
        wall_start_ns = time.time_ns() if perf_enabled else 0
        t_start = time.perf_counter() if perf_enabled else 0.0
        if perf_enabled and finished_req_ids:
            _append_perf_log(
                "GET_FINISHED "
                f"ts={_format_perf_time(wall_start_ns)} start_ns={wall_start_ns} "
                f"pid={os.getpid()} finished={','.join(sorted(r[:8] for r in finished_req_ids))}"
            )
        enqueued = 0
        # ---- Async-enqueue STORE specs ----
        if finished_req_ids:
            for req_id in finished_req_ids:
                spec = self._store_specs_by_req.get(req_id)
                if spec is None:
                    continue
                if spec.prefix_key in self._stored_keys:
                    # Already stored (vLLM occasionally re-schedules the same prefix); skip.
                    continue
                self._stored_keys.add(spec.prefix_key)
                # Submit to executor; the done callback moves req_id into _pending_done_sent.
                fut = self._store_executor.submit(self._store_blocks_to_service, spec)
                self._inflight_stores[req_id] = fut
                enqueued += 1

                def _done_cb(f, rid=req_id):
                    try:
                        f.result()  # raises on failure
                        with self._done_lock:
                            self._pending_done_sent.add(rid)
                    except Exception as e:
                        logger.error("CS worker async STORE failed req=%s: %s", rid[:8], e, exc_info=True)
                        # Add failed ones too, to avoid inflight deadlock; vLLM does not retry KV.
                        with self._done_lock:
                            self._pending_done_sent.add(rid)
                    finally:
                        self._inflight_stores.pop(rid, None)
                        self._store_specs_by_req.pop(rid, None)

                fut.add_done_callback(_done_cb)
                logger.info("CS worker STORE async enqueued req=%s prefix=%s blocks=%d",
                            req_id[:8], spec.prefix_key[:8], len(spec.gpu_block_ids))

        # ---- Report completed STOREs ----
        # Protocol warning: vLLM scheduler asserts `req_id in self.requests` (sched.py
        # L2149), i.e. every element of finished_sending must be in this step's
        # finished_req_ids and vLLM has not yet freed the req. If STORE completes
        # asynchronously after vLLM frees the req, our ack crashes.
        #
        # Temporary approach (matching SimpleCPUOffloadConnector): always return
        # None for finished_sending.
        # Risk: in theory vLLM may free GPU blocks while STORE is still running,
        # so a race condition could let STORE grab evicted/reused data. Measured:
        # under single-prompt workloads vLLM does not evict (GPU memory is
        # abundant), so the race does not trigger.
        # Long-term correct approach: hook bind_gpu_block_pool, ref-count the
        # blocks until STORE completes then dec-ref (like lmcache's
        # offloading_manager).
        with self._done_lock:
            if self._pending_done_sent:
                # Trim completed ones to keep the set from growing without bound.
                self._pending_done_sent &= finished_req_ids  # keep only the not-yet-final ones
                done_count = len(self._pending_done_sent)
                if done_count:
                    logger.debug("CS worker %d STORE completed but not acked (vLLM free)", done_count)

        if perf_enabled and (finished_req_ids or enqueued):
            wall_end_ns = time.time_ns()
            _append_perf_log(
                "GET_FINISHED_DONE "
                f"ts={_format_perf_time(wall_end_ns)} start_ns={wall_start_ns} end_ns={wall_end_ns} "
                f"pid={os.getpid()} enqueued={enqueued} "
                f"duration_ms={(time.perf_counter() - t_start) * 1000:.3f}"
            )

        return None, None

    def _make_layer_name(self, spec: _Spec) -> str:
        """Build the layer_name used by KV Service storage, including block count
        and TP rank suffix.

        Under TP>1 each worker's KV tensor is a different shard (different head
        slice), so each shard must live under a distinct key; on LOAD each rank
        reads its own key.
        Under TP=1 no suffix is added (backward compatible with older data).
        """
        num_blocks = spec.num_tokens // self._cs_config.block_size
        base = f"blocks_{num_blocks}"
        if self._tp_size > 1:
            return f"{base}_tp{self._tp_rank}"
        return base

    def _ensure_pinned(self, nbytes: int) -> torch.Tensor:
        """Reuse / grow the pinned buffer (uint8 view). The returned tensor has at
        least nbytes bytes.

        **Note**: this is the single buffer used on the LOAD path (LOAD serial
        worker thread, single buffer is enough). The STORE path uses
        `_acquire_pinned_from_pool` / `_release_pinned_to_pool` for multi-worker
        concurrency.
        """
        if self._pinned_buf is None or self._pinned_buf.numel() < nbytes:
            # Grow to 1.25× to leave headroom and avoid frequent realloc.
            target = max(nbytes, int(nbytes * 1.25))
            # Ensure the old buffer has no references before dropping (Python GC
            # collects; cudaHostFree runs automatically).
            self._pinned_buf = torch.empty(target, dtype=torch.uint8, pin_memory=True)
        return self._pinned_buf[:nbytes]

    def _acquire_pinned_from_pool(self, nbytes: int, timeout: float = 30.0) -> torch.Tensor:
        """Borrow a pinned buffer from the pool (blocks until a buffer is returned).

        Pool design (STORE pipeline):
        - Starts empty; lazily creates up to N buffers (`_pinned_pool_size`,
          aligned with store_executor).
        - Worker calls acquire → uses a buffer → calls release to return it.
        - Multiple workers actually parallelize: each has its own buffer, so D2H
          and PUT overlap across specs.
        - When the buffer is too small it is reallocated to nbytes × 1.25 (same
          policy as _ensure_pinned).
        - Must call release inside try/finally, otherwise the pool exhausts and
          later workers block forever.

        Returns: torch.Tensor (uint8 view, nbytes long)
        Raises:  RuntimeError when the pool has N buffers and none are returned
                 within the timeout (30s)
        """
        import queue
        # First check whether we can still allocate up to N — create a new buf
        # directly (avoids waiting 30s on get(timeout=30) when the pool is empty).
        # _pool_lock guards _pool_alloc_count against concurrent acquire.
        if not hasattr(self, "_pool_lock"):
            import threading as _th
            self._pool_lock = _th.Lock()
        if not hasattr(self, "_pool_alloc_count"):
            self._pool_alloc_count = 0
        with self._pool_lock:
            cur = self._pool_alloc_count
            if cur < self._pinned_pool_size:
                self._pool_alloc_count = cur + 1
                target = max(nbytes, int(nbytes * 1.25))
                return torch.empty(target, dtype=torch.uint8, pin_memory=True)
        # N buffers already allocated; must wait for someone to return one.
        try:
            buf = self._pinned_pool.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"pinned pool exhausted ({self._pinned_pool_size} bufs, none returned in {timeout}s); "
                f"check _store_blocks_to_service for missing release"
            )
        # Grow (existing buffer is too small).
        if buf.numel() < nbytes:
            target = max(nbytes, int(nbytes * 1.25))
            del buf
            buf = torch.empty(target, dtype=torch.uint8, pin_memory=True)
        return buf

    def _release_pinned_to_pool(self, buf: torch.Tensor) -> None:
        """Return the pinned buffer to the pool. Must be called from try/finally."""
        try:
            self._pinned_pool.put_nowait(buf)
        except Exception:
            # Pool is full (should not happen; safety net). Drop the buffer and let GC reclaim.
            pass

    def _spec_byte_layout(self, n_blocks: int) -> tuple[list[int], list[int], int]:
        """Return (per_layer_byte_size, per_layer_offset, total_bytes) for n_blocks per layer."""
        if not self._layer_names:
            return [], [], 0
        per_layer_sizes: list[int] = []
        sample = self._kv_caches[self._layer_names[0]]
        elem_size = sample.element_size()
        for layer_name in self._layer_names:
            t = self._kv_caches[layer_name]
            # Element count per block: if (2, num_blocks, ...) use t[:, 0].numel(); else t[0].numel().
            if t.dim() >= 4 and t.shape[0] == 2:
                per_block_numel = t[:, 0, ...].numel()
            else:
                per_block_numel = t[0].numel()
            per_layer_sizes.append(per_block_numel * n_blocks * elem_size)
        offsets = [0]
        for s in per_layer_sizes:
            offsets.append(offsets[-1] + s)
        total = offsets[-1]
        return per_layer_sizes, offsets, total

    def _copy_spec_to_pinned(
        self,
        spec: _Spec,
        pinned_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        """Bulk-move the spec's GPU KV into a pinned buffer; return the valid view /
        byte count / D2H elapsed time.

        Key points:
        - Use `index_select(gpu_block_ids)` to gather all needed blocks along the
          block dim in one shot.
        - Runs on the dedicated cuda stream + pinned host buffer + non_blocking=True.
        - The whole spec (97 blocks × 28 layers) does just 1 D2H + 1 stream sync.
        - vs. the old 2716 synchronous .cpu() calls.

        Layout: N blocks of layer 0 contiguous → N blocks of layer 1 → ..., dual to
        _deserialize_spec_into.
        """
        n_blocks = len(spec.gpu_block_ids)
        if n_blocks == 0 or not self._layer_names:
            empty = pinned_buf[:0] if pinned_buf is not None else torch.empty(0, dtype=torch.uint8)
            return empty, 0, 0.0
        sample = self._kv_caches[self._layer_names[0]]
        device = sample.device

        per_layer_sizes, offsets, total_bytes = self._spec_byte_layout(n_blocks)
        # Prefer the caller-provided buffer; otherwise fall back to the shared single buf (compat).
        if pinned_buf is not None:
            assert pinned_buf.numel() >= total_bytes, (
                f"pinned_buf too small: {pinned_buf.numel()} < {total_bytes}"
            )
            pinned = pinned_buf[:total_bytes]
        else:
            pinned = self._ensure_pinned(total_bytes)
        # Block index tensor on GPU, built once and shared across all layers.
        idx = torch.tensor(spec.gpu_block_ids, dtype=torch.long, device=device)

        t0 = time.perf_counter()
        # Run on the dedicated stream so the default stream is not blocked (model
        # forward can continue).
        # But because we enqueue STORE after get_finished, forward has already
        # finished — the stream mainly lets D2H overlap with the next step's
        # forward (RUN N STORE concurrent with RUN N+1 prefill).
        stream = self._copy_stream
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream())

        with ctx:
            offset = 0
            for layer_name, seg_size in zip(self._layer_names, per_layer_sizes):
                t = self._kv_caches[layer_name]
                if t.dim() >= 4 and t.shape[0] == 2:
                    # (2, num_blocks, ...) → gather along dim=1
                    gathered = t.index_select(1, idx).contiguous()  # (2, n_blocks, ...)
                else:
                    gathered = t.index_select(0, idx).contiguous()  # (n_blocks, ...)
                # Flatten + async copy into the pinned slice.
                src = gathered.view(torch.uint8).view(-1)
                # Use offset indexing (avoid another .index_select).
                dst = pinned[offset:offset + seg_size]
                dst.copy_(src, non_blocking=True)
                offset += seg_size

        # Wait for the copy to complete (required; otherwise tobytes reads pinned
        # data before it has landed).
        if stream is not None:
            stream.synchronize()
        t_d2h = time.perf_counter()
        return pinned, total_bytes, (t_d2h - t0) * 1000

    def _serialize_spec(self, spec: _Spec, pinned_buf: torch.Tensor | None = None) -> bytes:
        """**Batched** GPU→CPU transfer + serialization (replaces per-block per-layer
        synchronous .cpu()).

        Args:
            pinned_buf: Optional caller-provided pinned buffer (for the STORE pipeline
                        pool). None falls back to self._ensure_pinned (shared single
                        buffer).
        """
        pinned, total_bytes, d2h_ms = self._copy_spec_to_pinned(spec, pinned_buf)
        if total_bytes == 0:
            return b""
        t_d2h = time.perf_counter()
        # pinned uint8 → bytes (zero-copy view + one memcpy)
        data = pinned.numpy().tobytes()
        t_bytes = time.perf_counter()
        logger.info(
            "CS serialize_spec blocks=%d bytes=%d batched_d2h=%.1fms tobytes=%.1fms",
            len(spec.gpu_block_ids), total_bytes, d2h_ms, (t_bytes - t_d2h) * 1000,
        )
        return data

    def _deserialize_spec_into(self, spec: _Spec, data: bytes) -> None:
        """**Batched** deserialize + CPU→GPU write-back (replaces per-block per-layer
        synchronous .to(device)).

        Key points:
        - bytes → pinned CPU tensor → single async H2D into a GPU temp buffer.
        - Uses `index_copy_` to scatter into the target layer tensor along the block dim.
        - Just 1 H2D + 1 stream sync.
        """
        # Wrap into a single-chunk list and delegate to the chunks path.
        return self._deserialize_spec_into_chunks(spec, [data], len(data))

    def _deserialize_spec_into_chunks(self, spec: _Spec, chunks: list[bytes], total_bytes: int) -> None:
        """Deserialize from a list[bytes] directly to GPU.

        Compared to _deserialize_spec_into(bytes):
        - Skips the intermediate b"".join(chunks) alloc/copy.
        - Skips the second bytearray(data) alloc/copy (frombuffer needs writable).
        - Uses numpy.frombuffer (zero-copy view) to copy directly into the pinned slice.
        """
        import numpy as np
        n_blocks = len(spec.gpu_block_ids)
        if n_blocks == 0 or not self._layer_names or not chunks or total_bytes == 0:
            return
        sample = self._kv_caches[self._layer_names[0]]
        device = sample.device
        dtype = sample.dtype

        per_layer_sizes, offsets, expected_bytes = self._spec_byte_layout(n_blocks)
        if total_bytes != expected_bytes:
            raise RuntimeError(
                f"deserialize_spec_into_chunks size mismatch: got {total_bytes} expected {expected_bytes} "
                f"(n_blocks={n_blocks}, layers={len(self._layer_names)})"
            )

        t0 = time.perf_counter()
        pinned = self._ensure_pinned(total_bytes)
        # pinned is a torch uint8 tensor; take a numpy view (zero copy).
        pinned_np = pinned.numpy()
        # Copy each chunk directly into the pinned numpy view (no intermediate buffer).
        cursor = 0
        for ch in chunks:
            n = len(ch)
            if n == 0:
                continue
            # bytes → numpy view (zero copy), then copy into the writable pinned slice.
            # Note: np.frombuffer(bytes) returns a read-only view; copy into the writable pinned slice.
            src_view = np.frombuffer(ch, dtype=np.uint8)
            pinned_np[cursor:cursor + n] = src_view
            cursor += n
        t_to_pinned = time.perf_counter()

        idx = torch.tensor(spec.gpu_block_ids, dtype=torch.long, device=device)
        stream = self._copy_stream
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream())

        with ctx:
            offset = 0
            for layer_name, seg_size in zip(self._layer_names, per_layer_sizes):
                t = self._kv_caches[layer_name]
                seg_cpu = pinned[offset:offset + seg_size]
                if t.dim() >= 4 and t.shape[0] == 2:
                    per_block_shape = list(t.shape[2:])
                    full_shape = [2, n_blocks] + per_block_shape
                else:
                    per_block_shape = list(t.shape[1:])
                    full_shape = [n_blocks] + per_block_shape

                gpu_tmp = torch.empty(full_shape, dtype=dtype, device=device)
                gpu_tmp_flat = gpu_tmp.view(torch.uint8).view(-1)
                gpu_tmp_flat.copy_(seg_cpu, non_blocking=True)

                if t.dim() >= 4 and t.shape[0] == 2:
                    t.index_copy_(1, idx, gpu_tmp)
                else:
                    t.index_copy_(0, idx, gpu_tmp)
                offset += seg_size

        if stream is not None:
            stream.synchronize()
        t_h2d = time.perf_counter()
        logger.info(
            "CS deserialize_spec_chunks blocks=%d bytes=%d nchunks=%d to_pinned=%.1fms h2d_scatter=%.1fms",
            n_blocks, total_bytes, len(chunks),
            (t_to_pinned - t0) * 1000, (t_h2d - t_to_pinned) * 1000,
        )

    # ===== block ↔ KVService conversion (old per-block interface, kept as fallback) =====

    def _serialize_block(self, gpu_block_id: int) -> bytes:
        """Concatenate the gpu_block_id-th block across all layers into a single
        bytes blob.

        Layout: the K/V dim sits on dim 0 (common FlashAttn NHD is
        (2, num_blocks, block_size, H, D)) or a plain (num_blocks, block_size, H*2, D)
        etc. We simply take bytes via tensor.contiguous().

        All layers are concatenated in the order of self._layer_names.
        For dtypes numpy does not directly support (bfloat16 / float8 etc.), first
        view as an integer type of the same width and then tobytes.
        """
        t0 = time.perf_counter()
        pieces: list[torch.Tensor] = []
        for layer_name in self._layer_names:
            t = self._kv_caches[layer_name]
            # Slice the gpu_block_id-th block; when dim 0 is 2 (K/V), slice dim 1.
            if t.dim() >= 4 and t.shape[0] == 2:
                # (2, num_blocks, ...) → block is dim 1
                blk = t[:, gpu_block_id, ...].contiguous()
            else:
                # (num_blocks, ...) → block is dim 0
                blk = t[gpu_block_id].contiguous()
            pieces.append(blk.cpu())
        t_gather = time.perf_counter()
        # After gathering on CPU, convert to bytes (first version is sync; later
        # can pin_memory + non_blocking).
        flat = torch.cat([p.view(-1) for p in pieces], dim=0)
        t_cat = time.perf_counter()
        data = _tensor_to_bytes(flat)
        t_bytes = time.perf_counter()
        # Log every call (debug).
        logger.info(
            "CS serialize block=%d (%d layers, %d bytes): gather_cpu=%.1fms cat=%.1fms tobytes=%.1fms",
            gpu_block_id, len(self._layer_names), len(data),
            (t_gather - t0) * 1000,
            (t_cat - t_gather) * 1000,
            (t_bytes - t_cat) * 1000,
        )
        return data

    def _deserialize_block_into(self, gpu_block_id: int, data: bytes) -> None:
        """Inverse of _serialize_block: deserialize and write back the gpu_block_id-th
        block for every layer."""
        if not self._layer_names:
            return
        sample = self._kv_caches[self._layer_names[0]]
        dtype = sample.dtype
        device = sample.device
        # Derive per-layer numel.
        sizes = []
        for layer_name in self._layer_names:
            t = self._kv_caches[layer_name]
            if t.dim() >= 4 and t.shape[0] == 2:
                sizes.append(t[:, 0, ...].numel())
            else:
                sizes.append(t[0].numel())
        total = sum(sizes)
        # bytes → CPU tensor (aligned with dtype), then H2D.
        cpu_flat = _bytes_to_tensor(data, dtype, total)
        gpu_flat = cpu_flat.to(device, non_blocking=True)
        offset = 0
        for layer_name, sz in zip(self._layer_names, sizes):
            t = self._kv_caches[layer_name]
            chunk = gpu_flat[offset:offset + sz]
            if t.dim() >= 4 and t.shape[0] == 2:
                # restore (2, per_block_shape...)
                target_shape = t[:, gpu_block_id, ...].shape
                t[:, gpu_block_id, ...] = chunk.view(target_shape)
            else:
                target_shape = t[gpu_block_id].shape
                t[gpu_block_id] = chunk.view(target_shape)
            offset += sz

    def _store_blocks_to_service(self, spec: _Spec) -> None:
        """Serialize every block in spec.gpu_block_ids and PUT to KVService.

        New path: use _serialize_spec to do a batched D2H (replaces per-block
        per-layer synchronous .cpu()).
        The old per-block concat segments is replaced with a single bytes blob →
        put_chunks splits into N parallel channels (inside KVServiceBackend).

        **STORE pipeline (v2)**: borrow a buffer from `_pinned_pool` (instead of
        the global shared `_pinned_buf`) so store_executor's multiple workers can
        truly parallelize — spec_A uses buf_0 for D2H while spec_B uses buf_1
        concurrently.
        try/finally guarantees the buf is always returned to the pool (otherwise
        pool exhausts → later specs block forever).
        """
        wall_start_ns = time.time_ns()
        t_start = time.perf_counter()
        # Compute how many bytes the spec needs and borrow a buffer from the pool.
        n_blocks = len(spec.gpu_block_ids)
        if n_blocks == 0 or not self._layer_names:
            return
        _, _, total_bytes = self._spec_byte_layout(n_blocks)
        pinned_buf = self._acquire_pinned_from_pool(total_bytes)
        try:
            # One-shot batched D2H serializing the whole spec (using pool buf, not shared).
            pinned, total_bytes, d2h_ms = self._copy_spec_to_pinned(spec, pinned_buf=pinned_buf)
            t_serialize = time.perf_counter()

            layer_name = self._make_layer_name(spec)
            from contextstore.storage.base import BlockMeta
            meta = BlockMeta(
                num_tokens=spec.num_tokens,
                num_layers=len(self._layer_names),
                dtype=str(self._kv_caches[self._layer_names[0]].dtype) if self._layer_names else "",
                shape=[],
                compressed=False,
                compression_level=0,
            )
            storage = self._engine.storage
            if (
                hasattr(storage, "supports_rdma_put")
                and storage.supports_rdma_put()
                and hasattr(storage, "ensure_rdma_region")
                and hasattr(storage, "put_chunks_from")
            ):
                try:
                    region_id = storage.ensure_rdma_region(pinned_buf.data_ptr(), pinned_buf.numel())
                    if storage.put_chunks_from(spec.prefix_key, layer_name, region_id, 0, total_bytes):
                        wall_end_ns = time.time_ns()
                        t_put = time.perf_counter()
                        if _perf_log_enabled():
                            _append_perf_log(
                                "STORE_ZC "
                                f"ts={_format_perf_time(wall_end_ns)} "
                                f"start_ns={wall_start_ns} end_ns={wall_end_ns} "
                                f"pid={os.getpid()} req={spec.req_id[:16]} key={spec.prefix_key[:8]} "
                                f"layer={layer_name} blocks={len(spec.gpu_block_ids)} "
                                f"bytes={total_bytes} d2h_ms={d2h_ms:.3f} "
                                f"put_ms={(t_put - t_serialize) * 1000:.3f} "
                                f"total_ms={(t_put - t_start) * 1000:.3f}"
                            )
                        logger.info(
                            "CS worker STORE ZC req=%s prefix=%s blocks=%d bytes=%d "
                            "d2h=%.1fms put=%.1fms",
                            spec.req_id[:8], spec.prefix_key[:8], len(spec.gpu_block_ids),
                            total_bytes, d2h_ms, (t_put - t_serialize) * 1000,
                        )
                        return
                    if not self._cs_config.rdma_fallback_to_grpc:
                        raise RuntimeError(
                            f"RDMA put_chunks_from failed for key={spec.prefix_key[:8]} "
                            f"layer={layer_name} bytes={total_bytes} "
                            "and rdma_fallback_to_grpc is disabled"
                        )
                    logger.warning("[CS RDMA PUT] put_chunks_from failed, fallback to bytes path")
                except Exception as e:
                    if not self._cs_config.rdma_fallback_to_grpc:
                        raise
                    logger.warning("[CS RDMA PUT] put_chunks_from error: %s, fallback to bytes path", e)

            # Non-RDMA path or when fallback is allowed: build bytes from the
            # already-filled pinned buffer and take the old path.
            big_blob = pinned.numpy().tobytes()
            # Split into segments for put_chunks (when parallel_channels=8, split into 8 parallel channels).
            # Segment size: ≤ 4MB each, aligned with server-side sub_chunk to
            # exploit streaming receive.
            seg_size = 4 * 1024 * 1024
            segments: list[bytes] = [big_blob[i:i + seg_size] for i in range(0, len(big_blob), seg_size)]

            if hasattr(self._engine.storage, "put_chunks"):
                self._engine.storage.put_chunks(spec.prefix_key, layer_name, segments, meta)
            else:
                self._engine.storage.put(spec.prefix_key, layer_name, big_blob, meta)
            t_put = time.perf_counter()
            logger.info(
                "CS worker STORE req=%s prefix=%s blocks=%d bytes=%d "
                "d2h=%.1fms tobytes=%.1fms put=%.1fms",
                spec.req_id[:8], spec.prefix_key[:8], len(spec.gpu_block_ids), len(big_blob), d2h_ms,
                (t_serialize - t_start) * 1000, (t_put - t_serialize) * 1000,
            )
        finally:
            # Must return the pinned buffer; otherwise the pool exhausts.
            self._release_pinned_to_pool(pinned_buf)

    def _load_blocks_from_service(self, spec: _Spec) -> None:
        """Fetch the blocks for spec.prefix_key from KVService and write back to GPU.

        New path: chunks (list[bytes]) → copy directly into pinned (skipping the
        b''.join + bytearray double copy) → 1 H2D + index_copy_ scatter.

        ★ Zero-copy fast path (plan A): if the storage supports zero-copy (RDMA +
          parallel=1 + new ffi interface), use `_load_blocks_from_service_zerocopy`
          — RDMA WRITE goes directly into the pinned buffer, skipping the two
          GIL-bound copies (string_at ~560ms/625MB + numpy memcpy ~50ms).
        """
        # Write directly to a file — vLLM redirects stderr/stdout and swallows print/logger.
        try:
            with open("/tmp/cs_worker_load.log", "a") as f:
                f.write(f"[CS WORKER LOAD] enter req={spec.req_id[:8]} prefix={spec.prefix_key[:8]} blocks={len(spec.gpu_block_ids)} pid={os.getpid()}\n")
        except Exception:
            pass
        wall_start_ns = time.time_ns()
        t_start = time.perf_counter()
        layer_name = self._make_layer_name(spec)

        # ===== Shared-filesystem GDS fast path =====
        storage = self._engine.storage
        if (
            hasattr(storage, "supports_shared_gds")
            and storage.supports_shared_gds()
            and hasattr(storage, "get_chunks_to_gpu")
        ):
            if self._load_blocks_from_shared_gds(spec, layer_name):
                return

        # ===== Zero-copy fast path =====
        storage = self._engine.storage
        # DEBUG: log the path selection once via logger.warning (file writes can silently
        # fail; logger is more reliable).
        if not getattr(self, "_zc_debug_logged", False):
            self._zc_debug_logged = True
            has_supp = hasattr(storage, "supports_zerocopy")
            supp_val = storage.supports_zerocopy() if has_supp else "N/A"
            has_ens = hasattr(storage, "ensure_rdma_region")
            has_get = hasattr(storage, "get_chunks_into")
            logger.warning(
                "[CS ZC DEBUG] storage=%s has_supports_zerocopy=%s val=%s "
                "has_ensure_rdma_region=%s has_get_chunks_into=%s",
                type(storage).__name__, has_supp, supp_val, has_ens, has_get,
            )
        if (
            hasattr(storage, "supports_zerocopy")
            and storage.supports_zerocopy()
            and hasattr(storage, "ensure_rdma_region")
            and hasattr(storage, "get_chunks_into")
        ):
            ok = self._load_blocks_from_service_zerocopy(
                spec,
                layer_name,
                t_start,
                wall_start_ns,
            )
            if ok:
                return
            if not self._cs_config.rdma_fallback_to_grpc:
                raise RuntimeError(
                    "CS zerocopy RDMA load failed and rdma_fallback_to_grpc is disabled "
                    f"(prefix={spec.prefix_key[:8]} layer={layer_name})"
                )
            logger.warning("[CS ZC] zerocopy path failed, fallback to list[bytes]")
            # Failure → fall through to the list[bytes] path.

        if hasattr(storage, "get_chunks"):
            chunks = storage.get_chunks(spec.prefix_key, layer_name)
        else:
            data = storage.get(spec.prefix_key, layer_name)
            chunks = [data] if data else None
        t_get = time.perf_counter()
        if not chunks:
            # ★ LOAD MISS is a correctness hazard!
            # The scheduler decided to skip prefill based on a prefix_probe HIT,
            # but the worker cannot fetch the KV → attention uses stale /
            # uninitialized GPU data → wrong output.
            # Defence: zero-fill the corresponding GPU blocks (avoid reading
            # residual data from a previous inference). This degrades output
            # (these tokens see all-zero attention) but is better than dirty data.
            # Long-term fix: scheduler receives a worker-miss signal → rolls back
            # the cache hit and does a cold prefill.
            logger.error(
                "CS worker LOAD MISS req=%s prefix=%s blocks=%d — zero-filling GPU blocks "
                "to prevent dirty attention (output will be degraded for these tokens)",
                spec.req_id[:8], spec.prefix_key[:8], len(spec.gpu_block_ids),
            )
            self._zero_fill_gpu_blocks(spec)
            return
        total_bytes = sum(len(c) for c in chunks)
        # Copy chunks directly into pinned (saves the intermediate b"".join alloc).
        self._deserialize_spec_into_chunks(spec, chunks, total_bytes)
        t_write = time.perf_counter()
        logger.info(
            "CS worker LOAD req=%s prefix=%s blocks=%d bytes=%d get=%.1fms write_gpu=%.1fms",
            spec.req_id[:8], spec.prefix_key[:8], len(spec.gpu_block_ids), total_bytes,
            (t_get - t_start) * 1000, (t_write - t_get) * 1000,
        )

    def _load_blocks_from_service_zerocopy(
        self,
        spec: _Spec,
        layer_name: str,
        t_start: float,
        wall_start_ns: int,
    ) -> bool:
        """RDMA WRITE directly into the pinned host buffer, then H2D scatter into all
        GPU KV cache layers.

        Skips the two GIL-bound memcpies of the old path (ctypes.string_at +
        numpy copy into pinned).
        Returns True on success (caller should return immediately); False on
        failure (caller should fall back to the old path).

        Note: a spec's storage key (e.g. blocks_625_tp0) contains the concatenated
        KV data of every vLLM model layer (the full 625MB). With N=1 the whole
        layer is fetched in one RDMA GET, written into pinned, then scattered per
        per-layer offset into GPU's self._kv_caches[model_layer_name].
        The layer_name parameter is the storage key (not the vLLM model layer name).
        """
        t_zc_entry = time.perf_counter()
        n_blocks = len(spec.gpu_block_ids)
        if n_blocks == 0 or not self._layer_names:
            return False

        # Compute the layout of every vLLM layer inside pinned + total bytes
        # (matching _deserialize_spec_into_chunks).
        per_layer_sizes, _offsets, total_bytes = self._spec_byte_layout(n_blocks)
        t_layout = time.perf_counter()
        if total_bytes == 0:
            return False

        # Acquire the pinned buffer and register with RDMA (first call ~80ms
        # reg_mr; subsequent calls reuse).
        pinned = self._ensure_pinned(total_bytes)
        t_pin = time.perf_counter()
        storage = self._engine.storage
        try:
            region_id = storage.ensure_rdma_region(
                self._pinned_buf.data_ptr(), self._pinned_buf.numel()
            )
        except Exception as e:
            logger.warning("[CS ZC] ensure_rdma_region failed: %s", e)
            return False
        t_region = time.perf_counter()

        # RDMA WRITE the whole spec into pinned[0:total_bytes]
        wall_get_start_ns = time.time_ns()
        t_reg = t_region
        bytes_per_vllm_block = total_bytes // n_blocks if n_blocks else 0
        logger.info(
            "CS worker READ_REQUEST req=%s prefix=%s storage_key=%s "
            "vllm_blocks=%d block_size=%d tokens=%d expected_bytes=%d "
            "bytes_per_vllm_block=%d rdma_region=%d region_offset=%d",
            spec.req_id[:8],
            spec.prefix_key[:8],
            layer_name,
            n_blocks,
            self._cs_config.block_size,
            spec.num_tokens,
            total_bytes,
            bytes_per_vllm_block,
            region_id,
            0,
        )
        if _perf_log_enabled():
            _append_perf_log(
                "READ_REQUEST "
                f"ts={_format_perf_time(wall_get_start_ns)} "
                f"pid={os.getpid()} req={spec.req_id[:16]} key={spec.prefix_key[:8]} "
                f"layer={layer_name} vllm_blocks={n_blocks} "
                f"block_size={self._cs_config.block_size} tokens={spec.num_tokens} "
                f"expected_bytes={total_bytes} bytes_per_vllm_block={bytes_per_vllm_block} "
                f"rdma_region={region_id} region_offset=0"
            )
        n = storage.get_chunks_into(spec.prefix_key, layer_name, region_id, 0)
        wall_get_end_ns = time.time_ns()
        t_get = time.perf_counter()
        if n is None:
            logger.warning("[CS ZC] get_chunks_into None (RDMA error)")
            return False
        if n <= 0:
            logger.warning("[CS ZC] get_chunks_into miss n=%d key=%s storage_key=%s",
                           n, spec.prefix_key[:8], layer_name)
            return False
        if n != total_bytes:
            logger.warning(
                "[CS ZC] bytes mismatch: got %d expected %d (key=%s storage_key=%s) "
                "→ server entry size != expected, fallback",
                n, total_bytes, spec.prefix_key[:8], layer_name,
            )
            return False

        # ===== H2D: pinned → each vLLM layer's GPU KV cache =====
        sample = self._kv_caches[self._layer_names[0]]
        device = sample.device
        dtype = sample.dtype
        idx = torch.tensor(spec.gpu_block_ids, dtype=torch.long, device=device)
        stream = self._copy_stream
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream())

        with ctx:
            offset = 0
            for vllm_layer_name, seg_size in zip(self._layer_names, per_layer_sizes):
                t = self._kv_caches[vllm_layer_name]
                seg_cpu = pinned[offset:offset + seg_size]
                if t.dim() >= 4 and t.shape[0] == 2:
                    per_block_shape = list(t.shape[2:])
                    full_shape = [2, n_blocks] + per_block_shape
                else:
                    per_block_shape = list(t.shape[1:])
                    full_shape = [n_blocks] + per_block_shape
                gpu_tmp = torch.empty(full_shape, dtype=dtype, device=device)
                gpu_tmp_flat = gpu_tmp.view(torch.uint8).view(-1)
                gpu_tmp_flat.copy_(seg_cpu, non_blocking=True)
                if t.dim() >= 4 and t.shape[0] == 2:
                    t.index_copy_(1, idx, gpu_tmp)
                else:
                    t.index_copy_(0, idx, gpu_tmp)
                offset += seg_size
        if stream is not None:
            stream.synchronize()
        wall_end_ns = time.time_ns()
        t_h2d = time.perf_counter()
        if _perf_log_enabled():
            _append_perf_log(
                "LOAD_ZC "
                f"ts={_format_perf_time(wall_end_ns)} "
                f"start_ns={wall_start_ns} get_start_ns={wall_get_start_ns} "
                f"get_end_ns={wall_get_end_ns} end_ns={wall_end_ns} "
                f"pid={os.getpid()} req={spec.req_id[:16]} key={spec.prefix_key[:8]} "
                f"layer={layer_name} blocks={n_blocks} bytes={n} "
                f"pre_zc_ms={(t_zc_entry - t_start) * 1000:.3f} "
                f"layout_ms={(t_layout - t_zc_entry) * 1000:.3f} "
                f"pin_ms={(t_pin - t_layout) * 1000:.3f} "
                f"region_ms={(t_region - t_pin) * 1000:.3f} "
                f"setup_ms={(t_reg - t_start) * 1000:.3f} "
                f"get_ms={(t_get - t_reg) * 1000:.3f} "
                f"h2d_ms={(t_h2d - t_get) * 1000:.3f} "
                f"total_ms={(t_h2d - t_start) * 1000:.3f}"
            )
        logger.info(
            "CS worker LOAD-ZC req=%s prefix=%s blocks=%d storage_key=%s bytes=%d "
            "pre_zc=%.1fms layout=%.1fms pin=%.1fms region=%.1fms "
            "setup=%.1fms get_into=%.1fms h2d=%.1fms total=%.1fms",
            spec.req_id[:8], spec.prefix_key[:8], n_blocks, layer_name, n,
            (t_zc_entry - t_start) * 1000,
            (t_layout - t_zc_entry) * 1000,
            (t_pin - t_layout) * 1000,
            (t_region - t_pin) * 1000,
            (t_reg - t_start) * 1000, (t_get - t_reg) * 1000,
            (t_h2d - t_get) * 1000, (t_h2d - t_start) * 1000,
        )
        return True

    def _load_blocks_from_shared_gds(self, spec: _Spec, layer_name: str) -> bool:
        """Load a combined value through local cuFile into reusable GPU staging memory."""
        n_blocks = len(spec.gpu_block_ids)
        if n_blocks == 0 or not self._layer_names:
            return False
        per_layer_sizes, _offsets, total_bytes = self._spec_byte_layout(n_blocks)
        if total_bytes == 0:
            return False
        sample = self._kv_caches[self._layer_names[0]]
        if not sample.is_cuda:
            return False
        staging = self._ensure_gds_staging(total_bytes, sample.device)
        device_index = (
            sample.device.index if sample.device.index is not None else torch.cuda.current_device()
        )
        storage = self._engine.storage
        try:
            transferred = storage.get_chunks_to_gpu(
                spec.prefix_key,
                layer_name,
                staging.data_ptr(),
                staging.numel(),
                total_bytes,
                device_index,
            )
            if transferred != total_bytes:
                return False

            idx = torch.tensor(spec.gpu_block_ids, dtype=torch.long, device=sample.device)
            stream = self._copy_stream
            ctx = (
                torch.cuda.stream(stream)
                if stream is not None
                else torch.cuda.stream(torch.cuda.current_stream())
            )
            with ctx:
                offset = 0
                for vllm_layer_name, segment_size in zip(self._layer_names, per_layer_sizes):
                    target = self._kv_caches[vllm_layer_name]
                    if target.dim() >= 4 and target.shape[0] == 2:
                        shape = [2, n_blocks, *target.shape[2:]]
                    else:
                        shape = [n_blocks, *target.shape[1:]]
                    source = staging[offset : offset + segment_size].view(target.dtype).view(shape)
                    if target.dim() >= 4 and target.shape[0] == 2:
                        target.index_copy_(1, idx, source)
                    else:
                        target.index_copy_(0, idx, source)
                    offset += segment_size
            if stream is not None:
                stream.synchronize()
            return True
        except Exception as exc:
            logger.warning("shared GDS load failed: %s", exc)
            return False

    def _zero_fill_gpu_blocks(self, spec: _Spec) -> None:
        """Zero out every layer's KV block corresponding to spec.gpu_block_ids.
        Emergency defence on LOAD miss (to prevent attention from reading dirty data)."""
        if not self._layer_names or not spec.gpu_block_ids:
            return
        try:
            idx = torch.tensor(spec.gpu_block_ids, dtype=torch.long,
                               device=self._kv_caches[self._layer_names[0]].device)
            for layer_name in self._layer_names:
                t = self._kv_caches[layer_name]
                if t.dim() >= 4 and t.shape[0] == 2:
                    # (2, num_blocks, ...) → zero via index_fill_ along dim=1
                    zero = torch.zeros(2, len(spec.gpu_block_ids), *t.shape[2:],
                                       dtype=t.dtype, device=t.device)
                    t.index_copy_(1, idx, zero)
                else:
                    zero = torch.zeros(len(spec.gpu_block_ids), *t.shape[1:],
                                       dtype=t.dtype, device=t.device)
                    t.index_copy_(0, idx, zero)
        except Exception as e:
            logger.error("CS worker zero_fill failed: %s", e)


# ===== Top-level Router =====


class ContextStoreConnector(KVConnectorBase_V1):
    """Router connector: scheduler/worker each hold an inner impl."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        self._cs_config = ContextStoreConfig.from_extra_config(extra)
        self._cs_config.block_size = vllm_config.cache_config.block_size
        self._engine = ContextStoreEngine(self._cs_config)

        # The scheduler also needs TP size to compute the probe layer name (when
        # TP>1, STORE appends the _tp0 suffix and PROBE must match). Missing this
        # previously caused PROBE to always MISS under TP=4.
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        self._role = role
        if role == KVConnectorRole.SCHEDULER:
            self._sched: _SchedulerImpl | None = _SchedulerImpl(self._cs_config, self._engine, tp_size=tp_size)
            self._worker: _WorkerImpl | None = None
        else:
            self._sched = None
            self._worker = _WorkerImpl(self._cs_config, self._engine)

    # ===== Scheduler side (direct forwarding) =====

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self._sched is None:
            return 0, False
        return self._sched.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if self._sched is not None:
            self._sched.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        if self._sched is None:
            return ContextStoreMeta(block_size=self._cs_config.block_size)
        return self._sched.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._sched is None:
            return False, None
        return self._sched.request_finished(request, block_ids)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self._sched is not None:
            self._sched.bind_gpu_block_pool(gpu_block_pool)

    # ===== Worker side — hooks valid on CUDA =====

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, meta: KVConnectorMetadata) -> None:
        # Must call super first to maintain the base class's _connector_metadata state
        # (some code paths still go through _get_connector_metadata to fetch data).
        super().bind_connector_metadata(meta)
        if self._worker is not None:
            self._worker.bind_connector_metadata(meta)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self._worker is not None:
            self._worker.clear_connector_metadata()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._worker is None:
            return None, None
        return self._worker.get_finished(finished_req_ids)

    # ===== Worker side — active path on CUDA =====

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Called before forward; **synchronously** pulls the KV for LOAD specs to GPU.
        Must complete before attention runs, otherwise attention uses uninitialized data."""
        if self._worker is not None:
            self._worker.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass  # Current synchronous load finishes in start_load_kv, no per-layer wait needed.

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        pass  # CUDA bypass; store runs on an async ThreadPool, enqueued from get_finished.

    def wait_for_save(self) -> None:
        pass  # STORE is fire-and-track-completion asynchronous; no blocking here.
