from __future__ import annotations

"""Integration tests for ContextStore KV Service.

Server must be started first:
  cd kv-service/
  make server && ./server/target/release/contextstore-server --config configs/server.toml
"""

import os

import pytest

from contextstore.kvservice_client import KVClient, KVMetadata, ObjectKey


ENDPOINT = os.environ.get("CS_ENDPOINT", "localhost:50051")


@pytest.fixture(scope="module")
def client():
    c = KVClient(ENDPOINT)
    try:
        h = c.health()
        if not h.is_serving:
            pytest.skip(f"server not serving: {h}")
    except Exception as e:
        pytest.skip(f"server unreachable at {ENDPOINT}: {e}")
    yield c
    c.close()


def test_health(client):
    h = client.health()
    assert h.is_serving
    assert h.version


def test_single_roundtrip(client):
    key = ObjectKey("test-model", "abcdef0123456789/layer_0")
    client.put(key, b"hello world", KVMetadata(num_tokens=4, dtype="bfloat16"))
    got = client.get(key)
    assert got is not None
    data, meta = got
    assert data == b"hello world"
    assert meta.dtype == "bfloat16"
    assert client.exists(key)
    assert client.delete(key)
    assert not client.exists(key)


def test_batch_parallel(client):
    keys = [ObjectKey("test-model", f"batchtest/layer_{i}") for i in range(8)]
    items = [(k, f"data-{i}".encode(), KVMetadata(num_tokens=i)) for i, k in enumerate(keys)]

    successes = client.put_batch(items)
    assert all(successes)

    results = client.get_batch(keys)
    assert len(results) == 8
    for i, r in enumerate(results):
        assert r is not None
        assert r[0] == f"data-{i}".encode()

    for k in keys:
        client.delete(k)


def test_stream_large_value(client):
    key = ObjectKey("test-model", "streamtest/layer_huge")
    big = b"x" * (8 * 1024 * 1024)  # 8MB
    client.put(key, big, KVMetadata(num_tokens=1024))

    got = client.get_stream(key)
    assert len(got) == len(big)
    assert got == big

    client.delete(key)


def test_namespace_isolation(client):
    key_a = ObjectKey("tenant-a", "same-object")
    key_b = ObjectKey("tenant-b", "same-object")

    client.put(key_a, b"a")
    client.put(key_b, b"b")

    assert client.get(key_a)[0] == b"a"
    assert client.get(key_b)[0] == b"b"
    client.delete(key_a)
    client.delete(key_b)
