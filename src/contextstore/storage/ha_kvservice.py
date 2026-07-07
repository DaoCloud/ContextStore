from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Callable

from contextstore.storage.base import BlockMeta, StorageBackend
from contextstore.storage.kvservice import KVServiceBackend

logger = logging.getLogger(__name__)


def _route(key: str, n: int) -> int:
    """Pick a child backend by md5(key) % n.

    Same scheme as ShardedStorageBackend._shard_for_key (sharded.py), ensuring the
    scheduler side (using prefix_key) and the worker side (using spec.prefix_key) route
    the same prefix to the same server.
    """
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return h % n


class _CircuitBreaker:
    """Circuit breaker for a single child backend. Three states CLOSED / OPEN / HALF_OPEN, guarded by threading.Lock.

    Purpose: when a storage server dies, gRPC calls will block until the 30s timeout. PROBE is on the
    scheduler hot path (<1ms budget) and cannot tolerate such blocking. The breaker flips to OPEN
    immediately after the first exception from a child; within the cooldown_s window all further calls
    to that child are rejected without hitting the network (direct miss); after the window, one
    HALF_OPEN trial is admitted — success flips back to CLOSED, failure re-opens.

    Tripping is triggered only by "call raised an exception" (gRPC timeout / connection failure).
    RDMA get_chunks_into returning None is a normal gRPC-fallback signal, not a failure, and the
    caller is responsible for not feeding it into record(False).

    The clock is injected via a constructor parameter (defaults to time.monotonic) so unit tests
    can drive cooldown with a fake clock.
    """

    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def __init__(self, cooldown_s: float = 5.0, clock: Callable[[], float] | None = None) -> None:
        self._cooldown = cooldown_s
        self._clock = clock or time.monotonic
        self._state = self._CLOSED
        self._opened_at = 0.0
        self._trial_inflight = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Whether to admit one call. OPEN and not past cooldown → False (reject without network).

        HALF_OPEN admits only one trial (trial_inflight prevents thundering herd); other concurrent
        calls are treated as OPEN.
        """
        with self._lock:
            if self._state == self._CLOSED:
                return True
            if self._state == self._OPEN:
                if self._clock() - self._opened_at >= self._cooldown:
                    # Enter half-open; admit this call as the trial
                    self._state = self._HALF_OPEN
                    self._trial_inflight = True
                    return True
                return False
            # HALF_OPEN: reject if a trial is in flight, otherwise admit one
            if self._trial_inflight:
                return False
            self._trial_inflight = True
            return True

    def record(self, ok: bool) -> None:
        """Record one call result. Success → CLOSED; failure → OPEN and start the cooldown timer."""
        with self._lock:
            if ok:
                self._state = self._CLOSED
                self._trial_inflight = False
            else:
                self._state = self._OPEN
                self._opened_at = self._clock()
                self._trial_inflight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


class HaKVServiceBackend(StorageBackend):
    """High-availability KV Service backend: holds N independent child KVServiceBackends (one per
    storage server).

    All per-key calls are routed to a fixed child by md5(key) % N; when a child is unreachable the
    call degrades gracefully (reads return None / exists returns False / writes swallow exceptions)
    and never propagates upward, letting the inference request treat "storage unavailable" as an
    ordinary cache miss → prefill recompute.

    Placement policy: pure sharding, no replication. If one server dies, its ~1/N of keys all miss
    and recompute while the rest hit normally; total capacity = sum of children. Intra-disk
    redundancy (ZFS/RAIDZ) is managed per server; this layer is unaware.

    Circuit breaking: one _CircuitBreaker per child isolates dead children and avoids 30s gRPC
    timeouts stalling the scheduler hot path.

    Deliberately does not expose a single `_client` attribute (unlike single-node KVServiceBackend);
    instead exposes `probe_layer_exists` so the connector's _probe_storage_backend takes the
    HA-specific per-prefix-routed probe branch.
    """

    def __init__(
        self,
        children: list[KVServiceBackend],
        *,
        cooldown_s: float = 5.0,
        supports_zerocopy_hint: bool | None = None,
        supports_rdma_put_hint: bool | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not children:
            raise ValueError("HaKVServiceBackend requires at least one child backend")
        self._children = children
        self._n = len(children)
        self._breakers = [_CircuitBreaker(cooldown_s=cooldown_s, clock=clock) for _ in children]
        # _probe_layer_names reads this (aligned with the put_chunks multi-way suffixes). All children share the same _parallel.
        self._parallel = getattr(children[0], "_parallel", 1)

        # RDMA region: HA assigns an opaque ha_id and caches (ptr, size); never caches the child
        # region_id (after a child errors and resets, its rid is reassigned, so we must always
        # re-translate idempotently).
        self._regions: dict[int, tuple[int, int]] = {}
        self._region_ids: dict[tuple[int, int], int] = {}
        self._next_region_id = 1
        self._region_lock = threading.Lock()

        # zero-copy / rdma-put capability: derived from config and cached, not from real-time AND of live children
        # (to avoid one flaky child shutting down the whole RDMA path). None = fall back to AND of children.
        self._zc_hint = supports_zerocopy_hint
        self._rdma_put_hint = supports_rdma_put_hint

    # ===== Internal routing helpers =====

    def _call(self, idx: int, fn: Callable[..., Any], *args: Any, miss: Any) -> Any:
        """Circuit-breaker gate + try/except wrapper for one child call.

        Rejected by breaker (child already tripped) or the call raised → return miss (None / False);
        never propagates upward.
        """
        breaker = self._breakers[idx]
        if not breaker.allow():
            return miss
        try:
            result = fn(*args)
            breaker.record(True)
            return result
        except Exception as e:
            breaker.record(False)
            logger.warning(
                "[CS HA] child %d call %s failed, degrade to miss: %s",
                idx, getattr(fn, "__name__", "?"), e,
            )
            return miss

    # ===== StorageBackend mandatory surface (all routed by key) =====

    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        idx = _route(key, self._n)
        # Swallow write exceptions: losing one store = one future miss + recompute; does not break
        # consistency (connector only register_prefix on successful store, with server-side exists
        # as the source of truth for hits).
        self._call(idx, self._children[idx].put, key, layer_name, data, meta, miss=None)

    def get(self, key: str, layer_name: str) -> bytes | None:
        idx = _route(key, self._n)
        return self._call(idx, self._children[idx].get, key, layer_name, miss=None)

    def exists(self, key: str) -> bool:
        idx = _route(key, self._n)
        return bool(self._call(idx, self._children[idx].exists, key, miss=False))

    def delete(self, key: str) -> None:
        idx = _route(key, self._n)
        self._call(idx, self._children[idx].delete, key, miss=None)

    def get_meta(self, key: str) -> BlockMeta | None:
        idx = _route(key, self._n)
        return self._call(idx, self._children[idx].get_meta, key, miss=None)

    def capacity_usage(self) -> tuple[int, int]:
        """Sum of healthy children; tripped/erroring children contribute (0, 0)."""
        total_used = 0
        total_cap = 0
        for idx, child in enumerate(self._children):
            res = self._call(idx, child.capacity_usage, miss=None)
            if res is not None:
                used, cap = res
                total_used += used
                total_cap += cap
        return total_used, total_cap

    def list_keys(self) -> list[str]:
        """Union of healthy children; tripped/erroring children are skipped."""
        all_keys: set[str] = set()
        for idx, child in enumerate(self._children):
            res = self._call(idx, child.list_keys, miss=None)
            if res:
                all_keys.update(res)
        return list(all_keys)

    def get_parallel(self, keys: list[str], layer_name: str) -> list[bytes | None]:
        """Group keys by _route, batch-fetch per child, scatter back to original order; positions
        for dead children remain None."""
        if not keys:
            return []
        results: list[bytes | None] = [None] * len(keys)
        # idx -> (original index, key)
        groups: dict[int, list[tuple[int, str]]] = {}
        for i, key in enumerate(keys):
            groups.setdefault(_route(key, self._n), []).append((i, key))
        for idx, items in groups.items():
            sub_keys = [k for _, k in items]
            fetched = self._call(
                idx, self._children[idx].get_parallel, sub_keys, layer_name, miss=None
            )
            if fetched is None:
                continue  # child dead: leave this group's slots as None
            for (orig_i, _), data in zip(items, fetched):
                results[orig_i] = data
        return results

    # ===== Multi-segment passthrough interface (used by connector) =====

    def put_chunks(
        self,
        key: str,
        layer_name: str,
        segments: list[bytes],
        meta: BlockMeta | None = None,
    ) -> None:
        idx = _route(key, self._n)
        # Child put_chunks failure raises; swallow and degrade (same semantics as put).
        self._call(idx, self._children[idx].put_chunks, key, layer_name, segments, meta, miss=None)

    def get_chunks(self, key: str, layer_name: str) -> list[bytes] | None:
        idx = _route(key, self._n)
        return self._call(idx, self._children[idx].get_chunks, key, layer_name, miss=None)

    # ===== PROBE entry point (called by connector scheduler) =====

    def probe_layer_exists(self, key: str, layer_name: str) -> bool:
        """Layer-level existence probe using StorageBackend key routing.

        All probe_layer calls for one request share the same storage key → routed to the same child,
        aligned with that prefix's data/RDMA placement. Child tripped or raised → return False
        (PROBE MISS); the connector leaves matched unchanged → goes to prefill, preserving
        consistency.
        """
        idx = _route(key, self._n)
        return bool(
            self._call(
                idx,
                self._children[idx].probe_layer_exists,
                key,
                layer_name,
                miss=False,
            )
        )

    # ===== RDMA zero-copy path =====

    def supports_zerocopy(self) -> bool:
        if self._zc_hint is not None:
            return self._zc_hint
        return all(
            hasattr(c, "supports_zerocopy") and bool(c.supports_zerocopy())
            for c in self._children
        )

    def supports_rdma_put(self) -> bool:
        if self._rdma_put_hint is not None:
            return self._rdma_put_hint
        return all(
            hasattr(c, "supports_rdma_put") and bool(c.supports_rdma_put())
            for c in self._children
        )

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        """Register a connector pinned buffer as an RDMA MR and return an opaque ha_region_id.

        Calls ensure_rdma_region(ptr, size) on each healthy child to pre-pay the reg_mr (children
        idempotently cache by (ptr, size) internally). Only the (ptr, size) ↔ ha_id mapping is
        cached — child region_ids are NOT cached because a child's rid changes after a reset, so
        each get/put must re-translate idempotently via child.ensure_rdma_region.
        """
        with self._region_lock:
            cached = self._region_ids.get((ptr, size))
            if cached is not None:
                ha_id = cached
            else:
                ha_id = self._next_region_id
                self._next_region_id += 1
                self._regions[ha_id] = (ptr, size)
                self._region_ids[(ptr, size)] = ha_id
        # best-effort pre-registration on each healthy child (failure is non-fatal; get/put will
        # retry translation)
        for idx, child in enumerate(self._children):
            if not hasattr(child, "ensure_rdma_region"):
                continue
            self._call(idx, child.ensure_rdma_region, ptr, size, miss=None)
        return ha_id

    def _translate_region(self, idx: int, ha_region_id: int) -> int | None:
        """Translate ha_region_id to the child's current region_id (idempotent; survives child pool reset)."""
        with self._region_lock:
            ptr_size = self._regions.get(ha_region_id)
        if ptr_size is None:
            return None
        child = self._children[idx]
        if not hasattr(child, "ensure_rdma_region"):
            return None
        return self._call(idx, child.ensure_rdma_region, ptr_size[0], ptr_size[1], miss=None)

    def get_chunks_into(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
    ) -> int | None:
        idx = _route(key, self._n)
        if not self._breakers[idx].allow():
            return None
        child_rid = self._translate_region(idx, region_id)
        if child_rid is None:
            return None
        child = self._children[idx]
        if not hasattr(child, "get_chunks_into"):
            return None
        # Note: a None return from child get_chunks_into is the normal "RDMA unavailable/failed,
        # please fall back to gRPC" signal, not a child-backend failure → do not feed record(False).
        # Real exceptions inside the child are already swallowed to None.
        try:
            n = child.get_chunks_into(key, layer_name, child_rid, offset)
            self._breakers[idx].record(True)
            return n
        except Exception as e:
            self._breakers[idx].record(False)
            logger.warning("[CS HA] child %d get_chunks_into failed: %s", idx, e)
            return None

    def put_chunks_from(
        self,
        key: str,
        layer_name: str,
        region_id: int,
        offset: int,
        size: int,
    ) -> bool:
        idx = _route(key, self._n)
        if not self._breakers[idx].allow():
            return False
        child_rid = self._translate_region(idx, region_id)
        if child_rid is None:
            return False
        child = self._children[idx]
        if not hasattr(child, "put_chunks_from"):
            return False
        try:
            ok = bool(child.put_chunks_from(key, layer_name, child_rid, offset, size))
            self._breakers[idx].record(True)
            return ok
        except Exception as e:
            self._breakers[idx].record(False)
            logger.warning("[CS HA] child %d put_chunks_from failed: %s", idx, e)
            return False
