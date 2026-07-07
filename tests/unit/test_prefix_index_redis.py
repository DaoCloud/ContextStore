from __future__ import annotations

import pytest

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from contextstore.index.prefix_index_redis import RedisPrefixIndex


@pytest.fixture
def fake_redis_server():
    if not HAS_FAKEREDIS:
        pytest.skip("fakeredis not installed")
    return fakeredis.FakeServer()


@pytest.fixture
def redis_index(fake_redis_server):
    index = RedisPrefixIndex(
        model_id="test-model",
        block_size=4,
        redis_url="redis://localhost:6379/0",
    )
    index._client = fakeredis.FakeRedis(server=fake_redis_server, decode_responses=True)
    return index


@pytest.fixture
def redis_index_b(fake_redis_server):
    index = RedisPrefixIndex(
        model_id="test-model",
        block_size=4,
        redis_url="redis://localhost:6379/0",
    )
    index._client = fakeredis.FakeRedis(server=fake_redis_server, decode_responses=True)
    return index


class TestRedisPrefixIndex:
    def test_empty_index_returns_zero(self, redis_index):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        assert redis_index.lookup_prefix(token_ids) == 0

    def test_register_and_lookup(self, redis_index):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        keys = redis_index.register_blocks(token_ids, num_tokens=8)
        assert len(keys) == 2
        matched = redis_index.lookup_prefix(token_ids)
        assert matched == 8

    def test_partial_match(self, redis_index):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        redis_index.register_blocks(token_ids, num_tokens=4)
        matched = redis_index.lookup_prefix(token_ids)
        assert matched == 4

    def test_prefix_semantics(self, redis_index):
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens_b = [1, 2, 3, 4, 9, 10, 11, 12]
        redis_index.register_blocks(tokens_a, num_tokens=8)
        matched = redis_index.lookup_prefix(tokens_b)
        assert matched == 4

    def test_different_prefix_no_match(self, redis_index):
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        redis_index.register_blocks(tokens_a, num_tokens=4)
        matched = redis_index.lookup_prefix(tokens_b)
        assert matched == 0

    def test_remove_blocks(self, redis_index):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        keys = redis_index.register_blocks(token_ids, num_tokens=8)
        redis_index.remove_blocks(keys[:1])
        matched = redis_index.lookup_prefix(token_ids)
        assert matched == 0

    def test_contains(self, redis_index):
        token_ids = [1, 2, 3, 4]
        keys = redis_index.register_blocks(token_ids, num_tokens=4)
        assert redis_index.contains(keys[0])
        assert not redis_index.contains("nonexistent")

    def test_num_registered_blocks(self, redis_index):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        redis_index.register_blocks(token_ids, num_tokens=12)
        assert redis_index.num_registered_blocks == 3

    def test_cross_process_visibility(self, redis_index, redis_index_b):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        redis_index.register_blocks(token_ids, num_tokens=8)
        matched = redis_index_b.lookup_prefix(token_ids)
        assert matched == 8

    def test_pd_disagg_simulation(self, redis_index, redis_index_b):
        token_ids = [10, 20, 30, 40, 50, 60, 70, 80]
        redis_index.register_blocks(token_ids, num_tokens=8)
        matched = redis_index_b.lookup_prefix(token_ids)
        assert matched == 8
        keys = redis_index_b.register_blocks(token_ids, num_tokens=0)
        assert keys == []

    def test_short_sequence(self, redis_index):
        token_ids = [1, 2, 3]
        matched = redis_index.lookup_prefix(token_ids)
        assert matched == 0
