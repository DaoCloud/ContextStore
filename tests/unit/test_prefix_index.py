from __future__ import annotations

import pytest
from contextstore.index.prefix_index import PrefixIndex


class TestPrefixIndex:
    def setup_method(self):
        self.index = PrefixIndex(model_id="test-model", block_size=4)

    def test_empty_index_returns_zero(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        assert self.index.lookup_prefix(token_ids) == 0

    def test_register_and_lookup(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        keys = self.index.register_blocks(token_ids, num_tokens=8)
        assert len(keys) == 2
        matched = self.index.lookup_prefix(token_ids)
        assert matched == 8

    def test_partial_match(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.index.register_blocks(token_ids, num_tokens=4)
        matched = self.index.lookup_prefix(token_ids)
        assert matched == 4

    def test_prefix_semantics(self):
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens_b = [1, 2, 3, 4, 9, 10, 11, 12]
        self.index.register_blocks(tokens_a, num_tokens=8)
        matched = self.index.lookup_prefix(tokens_b)
        assert matched == 4

    def test_different_prefix_no_match(self):
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        self.index.register_blocks(tokens_a, num_tokens=4)
        matched = self.index.lookup_prefix(tokens_b)
        assert matched == 0

    def test_remove_blocks(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        keys = self.index.register_blocks(token_ids, num_tokens=8)
        self.index.remove_blocks(keys[:1])
        matched = self.index.lookup_prefix(token_ids)
        assert matched == 0

    def test_short_sequence_no_full_block(self):
        token_ids = [1, 2, 3]
        matched = self.index.lookup_prefix(token_ids)
        assert matched == 0

    def test_model_isolation(self):
        index_a = PrefixIndex(model_id="model-a", block_size=4)
        index_b = PrefixIndex(model_id="model-b", block_size=4)
        token_ids = [1, 2, 3, 4]
        index_a.register_blocks(token_ids, num_tokens=4)
        key_a = index_a.compute_block_key(token_ids, 0)
        key_b = index_b.compute_block_key(token_ids, 0)
        assert key_a != key_b

    def test_num_registered_blocks(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.index.register_blocks(token_ids, num_tokens=12)
        assert self.index.num_registered_blocks == 3

    def test_register_prefix_supports_block_partial_match(self):
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens_b = [1, 2, 3, 4, 9, 10, 11, 12]
        self.index.register_prefix(tokens_a, num_tokens=8)
        assert self.index.lookup_prefix(tokens_b) == 4

    def test_register_prefix_preserves_block_key_membership(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.index.register_prefix(token_ids, num_tokens=8)
        assert self.index.num_registered_blocks == 2
        assert self.index.contains(self.index.compute_block_key(token_ids, 0))
        assert self.index.contains(self.index.compute_block_key(token_ids, 1))
