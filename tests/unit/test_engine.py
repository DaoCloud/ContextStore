from __future__ import annotations

import torch
from contextstore.core.config import ContextStoreConfig
from contextstore.core.engine import ContextStoreEngine
from contextstore.storage.memory import MemoryStorageBackend


class TestContextStoreEngine:
    def setup_method(self):
        self.config = ContextStoreConfig(
            storage_path="/tmp/test_cs",
            block_size=4,
            enable_compression=False,
            model_id="test-model",
        )
        self.storage = MemoryStorageBackend()
        self.engine = ContextStoreEngine(self.config, storage=self.storage)

    def test_lookup_miss(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        assert self.engine.lookup(token_ids) == 0

    def test_save_and_load_roundtrip(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.engine.register_prefix(token_ids, num_tokens=8)
        block_keys = self.engine.index.register_blocks(token_ids, num_tokens=8)
        assert len(block_keys) == 2

        kv_data = torch.randn(2, 8, 64)
        slot_mapping = torch.arange(8)
        for layer_name in ["layer_0", "layer_1"]:
            self.engine.save_layer(block_keys, layer_name, kv_data, slot_mapping, token_ids)

        matched = self.engine.lookup(token_ids)
        assert matched == 8

        target_kv = torch.zeros(2, 8, 64)
        self.engine.load_layer(block_keys, "layer_0", target_kv, slot_mapping)
        assert not torch.all(target_kv == 0)

    def test_prefix_sharing(self):
        tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
        tokens_b = [1, 2, 3, 4, 9, 10, 11, 12]
        self.engine.register_prefix(tokens_a, num_tokens=8)
        block_keys = self.engine.index.register_blocks(tokens_a, num_tokens=8)

        kv_data = torch.randn(2, 8, 64)
        slot_mapping = torch.arange(8)
        self.engine.save_layer(block_keys, "layer_0", kv_data, slot_mapping, tokens_a)

        matched = self.engine.lookup(tokens_b)
        assert matched == 4

    def test_metrics_tracking(self):
        token_ids = [1, 2, 3, 4]
        self.engine.lookup(token_ids)
        assert self.engine.metrics.cache_misses == 1

        self.engine.register_prefix(token_ids, num_tokens=4)
        kv_data = torch.randn(2, 4, 64)
        slot_mapping = torch.arange(4)
        self.engine.save_layer(
            self.engine.index.register_blocks(token_ids, 4),
            "layer_0", kv_data, slot_mapping, token_ids,
        )

        self.engine.lookup(token_ids)
        assert self.engine.metrics.cache_hits >= 1

    def test_with_compression(self):
        config = ContextStoreConfig(
            block_size=4,
            enable_compression=True,
            compression_level=1,
            model_id="test-model",
        )
        storage = MemoryStorageBackend()
        engine = ContextStoreEngine(config, storage=storage)
        token_ids = [1, 2, 3, 4]
        engine.register_prefix(token_ids, num_tokens=4)
        block_keys = engine.index.register_blocks(token_ids, num_tokens=4)
        kv_data = torch.randn(2, 4, 64, dtype=torch.float16)
        slot_mapping = torch.arange(4)
        engine.save_layer(block_keys, "layer_0", kv_data, slot_mapping, token_ids)

        target_kv = torch.zeros(2, 4, 64, dtype=torch.float16)
        engine.load_layer(block_keys, "layer_0", target_kv, slot_mapping)
        max_error = (target_kv.float() - kv_data.float()).abs().max().item()
        assert max_error < 0.1
