from __future__ import annotations

import pytest
import torch
from contextstore.metadata import ContextStoreMetadata, RequestMeta


class TestRequestMeta:
    def test_make_slot_mapping(self):
        block_ids = [0, 2, 5]
        block_size = 4
        num_tokens = 10
        slot_mapping = RequestMeta.make_slot_mapping(block_ids, block_size, num_tokens)
        assert slot_mapping.shape[0] == num_tokens
        assert slot_mapping[0].item() == 0
        assert slot_mapping[4].item() == 8
        assert slot_mapping[8].item() == 20

    def test_slot_mapping_alignment(self):
        block_ids = [1]
        block_size = 4
        num_tokens = 3
        slot_mapping = RequestMeta.make_slot_mapping(block_ids, block_size, num_tokens)
        assert slot_mapping.shape[0] == 3
        assert slot_mapping[0].item() == 4
        assert slot_mapping[2].item() == 6


class TestContextStoreMetadata:
    def test_add_load_request(self):
        meta = ContextStoreMetadata(block_size=4)
        meta.add_load_request(
            request_id="req-1",
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_ids=[0, 1],
            block_size=4,
            block_keys=["key_a", "key_b"],
            matched_prefix_len=8,
        )
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is False
        assert meta.requests[0].matched_prefix_len == 8
        assert meta.requests[0].slot_mapping.shape[0] == 8

    def test_add_store_request(self):
        meta = ContextStoreMetadata(block_size=4)
        meta.add_store_request(
            request_id="req-2",
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            block_ids=[3, 4],
            block_size=4,
            block_keys=["key_c", "key_d"],
        )
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True
