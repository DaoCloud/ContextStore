from __future__ import annotations

from typing import Any

from contextstore.storage.base import BlockMeta, StorageBackend
from contextstore.storage.ha_kvservice import (
    HaKVServiceBackend,
    _CircuitBreaker,
    _route,
)


# ===== Test doubles =====


class _FakeKVClient:
    """Mocks contextstore.kvservice_client.KVClient with layer-level exists (used by PROBE)."""

    def __init__(self) -> None:
        self.exists_calls = 0


class _FakeChild(StorageBackend):
    """Mocks a single KVServiceBackend: full StorageBackend surface + _client + _parallel + the RDMA quartet.

    down=True simulates a downed server, all calls raise. RDMA uses an in-memory bytearray as the
    remote peer; region_id increments, and bump_rid() simulates rid reassignment after a child pool
    reset (verifying HA does not cache child rids).
    """

    def __init__(self, parallel: int = 1) -> None:
        self._store: dict[tuple[str, str], bytes] = {}
        self._down = {"down": False}
        self._client = _FakeKVClient()
        self._parallel = parallel
        # Call counters (used to verify circuit breaker suppresses network calls)
        self.get_calls = 0
        self.put_calls = 0
        # RDMA region: (ptr,size) -> rid
        self._regions: dict[tuple[int, int], int] = {}
        self._next_rid = 100
        self._rdma_backing: dict[tuple[str, str], bytes] = {}

    # --- Fault injection ---
    def set_down(self, down: bool) -> None:
        self._down["down"] = down

    def bump_rid(self) -> None:
        """Simulate pool reset: clear the region map so the next ensure allocates a new rid."""
        self._regions.clear()
        self._next_rid += 50

    def _check(self) -> None:
        if self._down["down"]:
            raise ConnectionError("server down")

    # --- StorageBackend core ---
    def put(self, key: str, layer_name: str, data: bytes, meta: BlockMeta | None = None) -> None:
        self.put_calls += 1
        self._check()
        self._store[(key, layer_name)] = data

    def get(self, key: str, layer_name: str) -> bytes | None:
        self.get_calls += 1
        self._check()
        return self._store.get((key, layer_name))

    def exists(self, key: str) -> bool:
        self._check()
        return any(k == key for (k, _) in self._store)

    def probe_layer_exists(self, key: str, layer_name: str) -> bool:
        self._client.exists_calls += 1
        self._check()
        return (key, layer_name) in self._store

    def delete(self, key: str) -> None:
        self._check()
        for ck in [ck for ck in self._store if ck[0] == key]:
            del self._store[ck]

    def get_meta(self, key: str) -> BlockMeta | None:
        self._check()
        return None

    def capacity_usage(self) -> tuple[int, int]:
        self._check()
        used = sum(len(v) for v in self._store.values())
        return used, 1000

    def list_keys(self) -> list[str]:
        self._check()
        return list({k for (k, _) in self._store})

    def get_parallel(self, keys: list[str], layer_name: str) -> list[bytes | None]:
        self._check()
        return [self._store.get((k, layer_name)) for k in keys]

    # --- Multi-segment passthrough ---
    def put_chunks(
        self, key: str, layer_name: str, segments: list[bytes], meta: BlockMeta | None = None
    ) -> None:
        self.put_calls += 1
        self._check()
        self._store[(key, layer_name)] = b"".join(segments)

    def get_chunks(self, key: str, layer_name: str) -> list[bytes] | None:
        self.get_calls += 1
        self._check()
        data = self._store.get((key, layer_name))
        return [data] if data is not None else None

    # --- RDMA quartet ---
    def supports_zerocopy(self) -> bool:
        return True

    def supports_rdma_put(self) -> bool:
        return True

    def ensure_rdma_region(self, ptr: int, size: int) -> int:
        self._check()
        k = (ptr, size)
        if k not in self._regions:
            self._regions[k] = self._next_rid
            self._next_rid += 1
        return self._regions[k]

    def get_chunks_into(self, key: str, layer_name: str, region_id: int, offset: int) -> int | None:
        self._check()
        data = self._rdma_backing.get((key, layer_name))
        if data is None:
            return None
        return len(data)

    def put_chunks_from(
        self, key: str, layer_name: str, region_id: int, offset: int, size: int
    ) -> bool:
        self._check()
        self._rdma_backing[(key, layer_name)] = b"x" * size
        return True


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _key_for_bucket(bucket: int, n: int = 2) -> str:
    """Find a key that routes to the given bucket."""
    i = 0
    while True:
        k = f"prefix-{i:04d}"
        if _route(k, n) == bucket:
            return k
        i += 1


def _make_ha(parallel: int = 1, **kwargs: Any) -> tuple[HaKVServiceBackend, _FakeChild, _FakeChild]:
    a = _FakeChild(parallel=parallel)
    b = _FakeChild(parallel=parallel)
    ha = HaKVServiceBackend([a, b], **kwargs)
    return ha, a, b


# ===== 1. Route determinism + balance =====


def test_route_determinism_and_balance() -> None:
    # Same key is stable
    assert _route("abc", 2) == _route("abc", 2)
    # Roughly balanced: 1000 keys give each bucket a reasonable share
    counts = [0, 0]
    for i in range(1000):
        counts[_route(f"k{i}", 2)] += 1
    assert counts[0] > 300 and counts[1] > 300


# ===== 2. put/get round-trip through the same child =====


def test_put_get_roundtrip_routes_to_same_child() -> None:
    ha, a, b = _make_ha()
    key_a = _key_for_bucket(0)
    ha.put(key_a, "blocks_1", b"hello")
    assert a._store[(key_a, "blocks_1")] == b"hello"
    assert (key_a, "blocks_1") not in b._store
    assert ha.get(key_a, "blocks_1") == b"hello"


# ===== 3. All probe_layer calls for one prefix land on the same child =====


def test_probe_routes_one_prefix_to_one_child() -> None:
    ha, a, b = _make_ha()
    key_b = _key_for_bucket(1)
    # Write multiple layers into b
    for layer in ["blocks_5_tp0", "blocks_5_tp1", "blocks_5_tp2"]:
        b._store[(key_b, layer)] = b"d"
    for layer in ["blocks_5_tp0", "blocks_5_tp1", "blocks_5_tp2"]:
        assert ha.probe_layer_exists(key_b, layer) is True
    # a was never probed
    assert a._client.exists_calls == 0
    assert b._client.exists_calls == 3


# ===== 4. One server down → graceful degradation to miss, the other stays healthy =====


def test_one_server_down_degrades_to_miss() -> None:
    ha, a, b = _make_ha()
    key_a = _key_for_bucket(0)  # routes to a
    key_b = _key_for_bucket(1)  # routes to b
    ha.put(key_a, "L", b"aaa")
    ha.put(key_b, "L", b"bbb")

    b.set_down(True)

    # b's keys all degrade
    assert ha.get(key_b, "L") is None
    assert ha.exists(key_b) is False
    ha.put(key_b, "L", b"new")  # does not raise
    assert ha.probe_layer_exists(key_b, "L") is False
    # a's keys still work (using a, unaffected by b's breaker)
    assert ha.get(key_a, "L") == b"aaa"
    assert ha.exists(key_a) is True


# ===== 5. Breaker OPEN → no network → recovers after cooldown =====


def test_circuit_breaker_opens_and_recovers() -> None:
    clock = _FakeClock()
    ha, a, b = _make_ha(cooldown_s=5.0, clock=clock)
    key_b = _key_for_bucket(1)
    ha.put(key_b, "L", b"data")  # write while b is healthy

    b.set_down(True)
    # First failure → breaker OPEN
    assert ha.get(key_b, "L") is None
    calls_after_trip = b.get_calls
    # While OPEN: further calls should not actually hit b (no network)
    for _ in range(5):
        assert ha.get(key_b, "L") is None
    assert b.get_calls == calls_after_trip  # call counter unchanged

    # Not past cooldown, still rejected
    clock.advance(3.0)
    assert ha.get(key_b, "L") is None
    assert b.get_calls == calls_after_trip

    # Cooldown elapsed + b recovers → HALF_OPEN trial succeeds → CLOSED
    clock.advance(3.0)
    b.set_down(False)
    assert ha.get(key_b, "L") == b"data"
    assert b.get_calls == calls_after_trip + 1
    assert ha._breakers[1].state == _CircuitBreaker._CLOSED


# ===== 6. RDMA region fan-out: registration hits both children, translation survives rid changes =====


def test_rdma_region_fanout() -> None:
    ha, a, b = _make_ha()
    ha_id = ha.ensure_rdma_region(0x1000, 4096)
    # Both children were pre-registered
    assert (0x1000, 4096) in a._regions
    assert (0x1000, 4096) in b._regions
    # ha_id is opaque (not equal to any child's rid)
    assert ha_id == 1

    key_a = _key_for_bucket(0)
    # PUT via RDMA: written into a's rdma_backing
    assert ha.put_chunks_from(key_a, "L", ha_id, 0, 64) is True
    assert (key_a, "L") in a._rdma_backing
    # GET via RDMA returns byte count
    assert ha.get_chunks_into(key_a, "L", ha_id, 0) == 64

    # Simulate child pool reset → rid changes; HA idempotently re-translates and still works
    a.bump_rid()
    assert ha.put_chunks_from(key_a, "L", ha_id, 0, 128) is True
    assert ha.get_chunks_into(key_a, "L", ha_id, 0) == 128


# ===== 7. Aggregate methods behave correctly when one server is down =====


def test_aggregates_with_one_down() -> None:
    ha, a, b = _make_ha()
    key_a = _key_for_bucket(0)
    key_b = _key_for_bucket(1)
    ha.put(key_a, "L", b"aa")     # 2 bytes -> a
    ha.put(key_b, "L", b"bbbb")   # 4 bytes -> b

    # All healthy: sum / union
    used, cap = ha.capacity_usage()
    assert used == 6 and cap == 2000
    assert set(ha.list_keys()) == {key_a, key_b}
    gp = ha.get_parallel([key_a, key_b], "L")
    assert gp == [b"aa", b"bbbb"]

    # b down: capacity only from a, list_keys only a's, b's position in get_parallel is None
    b.set_down(True)
    used, cap = ha.capacity_usage()
    assert used == 2 and cap == 1000
    assert set(ha.list_keys()) == {key_a}
    gp = ha.get_parallel([key_a, key_b], "L")
    assert gp == [b"aa", None]


# ===== 8. Connector PROBE bypass: always via probe_layer_exists =====


def test_connector_probe_backend_selection() -> None:
    from contextstore.connector import _SchedulerImpl
    from contextstore.storage.host_memory import HostMemoryBackend

    # HA wrapped by HostMemoryBackend: _probe_storage_backend should see through L1 and find HA
    ha, a, b = _make_ha()
    l1 = HostMemoryBackend(wrapped=ha, max_capacity_bytes=1024)

    class _FakeEngine:
        def __init__(self, storage: Any) -> None:
            self.storage = storage

    sched = object.__new__(_SchedulerImpl)
    sched._engine = _FakeEngine(l1)
    found = sched._probe_storage_backend()
    assert found is ha
    assert hasattr(found, "probe_layer_exists")

    # Single-node child also exposes layer-level probe via probe_layer_exists; does not expose _client to the connector.
    sched2 = object.__new__(_SchedulerImpl)
    sched2._engine = _FakeEngine(a)
    found2 = sched2._probe_storage_backend()
    assert found2 is a
    assert hasattr(found2, "probe_layer_exists")


# ===== 9. Engine wiring: 2 endpoints → HA, 1 endpoint → single-node =====


def test_engine_wiring_selects_ha_for_multi_endpoint(monkeypatch: Any) -> None:
    import contextstore.storage.kvservice as kvmod
    from contextstore.core.config import ContextStoreConfig
    from contextstore.core.engine import ContextStoreEngine
    from contextstore.storage.ha_kvservice import HaKVServiceBackend

    # Use a stub KVServiceBackend to avoid real gRPC connections
    class _StubKVBackend(StorageBackend):
        def __init__(self, endpoint: str, **kw: Any) -> None:
            self.endpoint = endpoint
            self._parallel = kw.get("parallel_channels", 8)

        def put(self, *a: Any, **k: Any) -> None: ...
        def get(self, *a: Any, **k: Any) -> bytes | None: return None
        def exists(self, *a: Any, **k: Any) -> bool: return False
        def delete(self, *a: Any, **k: Any) -> None: ...
        def get_meta(self, *a: Any, **k: Any) -> BlockMeta | None: return None
        def capacity_usage(self) -> tuple[int, int]: return (0, 0)

    monkeypatch.setattr(kvmod, "KVServiceBackend", _StubKVBackend)

    # 2 endpoints → HaKVServiceBackend (disable L1 wrapper so we can assert type directly)
    cfg2 = ContextStoreConfig(
        kv_service_endpoints=["A:50051", "B:50051"],
        host_memory_capacity_gb=0,
    )
    eng2 = ContextStoreEngine(cfg2)
    assert isinstance(eng2.storage, HaKVServiceBackend)

    # 1 endpoint → single-node KVServiceBackend (stub here)
    cfg1 = ContextStoreConfig(
        kv_service_endpoint="A:50051",
        host_memory_capacity_gb=0,
    )
    eng1 = ContextStoreEngine(cfg1)
    assert isinstance(eng1.storage, _StubKVBackend)
