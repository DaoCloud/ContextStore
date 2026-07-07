from __future__ import annotations

import time

import pytest
import torch
from contextstore.core.codec import KVCodec, NoOpCodec


class TestKVCodecBinary:
    def setup_method(self):
        self.codec = KVCodec(level=1)

    def test_roundtrip_float16(self):
        t = torch.randn(2, 16, 64, dtype=torch.float16)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert restored.shape == t.shape
        assert restored.dtype == torch.float16
        max_err = (restored.float() - t.float()).abs().max().item()
        assert max_err < 0.1

    def test_roundtrip_float32(self):
        t = torch.randn(2, 8, 128, dtype=torch.float32)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert restored.shape == t.shape
        assert restored.dtype == torch.float32
        max_err = (restored - t).abs().max().item()
        assert max_err < 0.05

    def test_roundtrip_bfloat16(self):
        t = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert restored.shape == t.shape
        assert restored.dtype == torch.bfloat16

    def test_compression_ratio(self):
        t = torch.randn(2, 16, 128, dtype=torch.float16)
        data = self.codec.encode_to_bytes(t)
        raw_size = t.numel() * 2
        assert len(data) < raw_size

    def test_performance(self):
        t = torch.randn(2, 16, 128, dtype=torch.float16)
        # Warmup
        for _ in range(10):
            self.codec.encode_to_bytes(t)
        t0 = time.perf_counter()
        for _ in range(100):
            data = self.codec.encode_to_bytes(t)
        encode_ms = (time.perf_counter() - t0) * 1000 / 100

        for _ in range(10):
            self.codec.decode_from_bytes(data)
        t0 = time.perf_counter()
        for _ in range(100):
            self.codec.decode_from_bytes(data)
        decode_ms = (time.perf_counter() - t0) * 1000 / 100

        assert encode_ms < 50.0, f"Encode too slow: {encode_ms:.2f}ms"
        assert decode_ms < 50.0, f"Decode too slow: {decode_ms:.2f}ms"


class TestNoOpCodecBinary:
    def setup_method(self):
        self.codec = NoOpCodec()

    def test_roundtrip_float16(self):
        t = torch.randn(2, 16, 64, dtype=torch.float16)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert torch.equal(t, restored)

    def test_roundtrip_float32(self):
        t = torch.randn(4, 32, dtype=torch.float32)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert torch.equal(t, restored)

    def test_roundtrip_bfloat16(self):
        t = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert torch.equal(t, restored)

    def test_1d_tensor(self):
        t = torch.randn(128, dtype=torch.float16)
        data = self.codec.encode_to_bytes(t)
        restored = self.codec.decode_from_bytes(data)
        assert torch.equal(t, restored)

    def test_performance(self):
        t = torch.randn(2, 16, 128, dtype=torch.float16)
        # Warmup
        for _ in range(10):
            self.codec.encode_to_bytes(t)
        t0 = time.perf_counter()
        for _ in range(100):
            data = self.codec.encode_to_bytes(t)
        encode_ms = (time.perf_counter() - t0) * 1000 / 100

        for _ in range(10):
            self.codec.decode_from_bytes(data)
        t0 = time.perf_counter()
        for _ in range(100):
            self.codec.decode_from_bytes(data)
        decode_ms = (time.perf_counter() - t0) * 1000 / 100

        assert encode_ms < 50.0, f"Encode too slow: {encode_ms:.2f}ms"
        assert decode_ms < 50.0, f"Decode too slow: {decode_ms:.2f}ms"
