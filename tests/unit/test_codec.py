from __future__ import annotations

import pytest
import torch
from contextstore.core.codec import KVCodec, NoOpCodec, EncodedKV


class TestKVCodec:
    def setup_method(self):
        self.codec = KVCodec(level=1)

    def test_encode_decode_roundtrip(self):
        tensor = torch.randn(2, 32, 128, dtype=torch.float16)
        encoded = self.codec.encode(tensor)
        decoded = self.codec.decode(encoded)
        assert decoded.shape == tensor.shape
        assert decoded.dtype == tensor.dtype
        max_error = (decoded.float() - tensor.float()).abs().max().item()
        assert max_error < 0.1

    def test_compression_ratio(self):
        tensor = torch.randn(2, 64, 128, dtype=torch.float16)
        original_bytes = tensor.nelement() * tensor.element_size()
        encoded = self.codec.encode(tensor)
        compressed_bytes = encoded.quantized.nelement() * 1 + encoded.scales.nelement() * 2 + encoded.zero_points.nelement() * 2
        ratio = original_bytes / compressed_bytes
        assert ratio > 1.5

    def test_encode_decode_bytes_roundtrip(self):
        tensor = torch.randn(2, 16, 64, dtype=torch.float16)
        data = self.codec.encode_to_bytes(tensor)
        assert isinstance(data, bytes)
        decoded = self.codec.decode_from_bytes(data)
        assert decoded.shape == tensor.shape
        max_error = (decoded.float() - tensor.float()).abs().max().item()
        assert max_error < 0.1

    def test_preserves_shape(self):
        for shape in [(32, 64), (2, 16, 128), (4, 8, 16, 32)]:
            tensor = torch.randn(*shape, dtype=torch.float16)
            encoded = self.codec.encode(tensor)
            decoded = self.codec.decode(encoded)
            assert decoded.shape == tensor.shape


class TestNoOpCodec:
    def setup_method(self):
        self.codec = NoOpCodec()

    def test_roundtrip_exact(self):
        tensor = torch.randn(2, 16, 64, dtype=torch.float16)
        data = self.codec.encode_to_bytes(tensor)
        decoded = self.codec.decode_from_bytes(data)
        assert torch.equal(tensor, decoded)
