from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

_DTYPE_TO_ID = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2, torch.uint8: 3}
_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}

_MAGIC_KVCODEC = 0xC5010000
_MAGIC_NOOP = 0xC5000000


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    try:
        return t.numpy().tobytes()
    except (RuntimeError, TypeError):
        return bytes(t.untyped_storage())


def _bytes_to_tensor(data: bytes, dtype: torch.dtype, shape: list[int], offset: int = 0, length: int = 0) -> torch.Tensor:
    if length == 0:
        length = len(data) - offset
    # Single-copy path using numpy (much faster for large tensors)
    try:
        import numpy as np
        np_dtype = {
            torch.float16: np.float16,
            torch.bfloat16: np.uint16,
            torch.float32: np.float32,
            torch.uint8: np.uint8,
            torch.int16: np.int16,
        }.get(dtype)
        if np_dtype is not None:
            arr = np.frombuffer(data, dtype=np_dtype, count=length // np.dtype(np_dtype).itemsize, offset=offset)
            t = torch.from_numpy(arr.copy()).reshape(shape)
            if dtype == torch.bfloat16:
                t = t.view(torch.bfloat16)
            return t
    except (ImportError, RuntimeError):
        pass
    # Fallback: bytearray + clone (slower but always works)
    buf = bytearray(data[offset:offset + length])
    t = torch.frombuffer(buf, dtype=dtype).reshape(shape)
    return t.clone()


@dataclass
class EncodedKV:
    quantized: torch.Tensor  # uint8
    scales: torch.Tensor     # float16, per-channel
    zero_points: torch.Tensor  # float16, per-channel
    original_dtype: torch.dtype
    original_shape: list[int]


class KVCodec:
    def __init__(self, level: int = 1):
        if level not in (1,):
            raise ValueError(f"Unsupported compression level: {level}. Only level 1 (INT8) is supported.")
        self._level = level

    def encode(self, kv_tensor: torch.Tensor) -> EncodedKV:
        original_shape = list(kv_tensor.shape)
        original_dtype = kv_tensor.dtype
        flat = kv_tensor.float().reshape(-1, kv_tensor.shape[-1])
        ch_min = flat.min(dim=0).values
        ch_max = flat.max(dim=0).values
        scales = (ch_max - ch_min) / 255.0
        scales = scales.clamp(min=1e-8)
        zero_points = ch_min
        quantized = ((flat - zero_points) / scales).round().clamp(0, 255).to(torch.uint8)
        return EncodedKV(
            quantized=quantized,
            scales=scales.half(),
            zero_points=zero_points.half(),
            original_dtype=original_dtype,
            original_shape=original_shape,
        )

    def decode(self, encoded: EncodedKV) -> torch.Tensor:
        flat = encoded.quantized.float()
        scales = encoded.scales.float()
        zero_points = encoded.zero_points.float()
        decoded = flat * scales + zero_points
        decoded = decoded.reshape(encoded.original_shape)
        return decoded.to(encoded.original_dtype)

    def encode_to_bytes(self, kv_tensor: torch.Tensor) -> bytes:
        encoded = self.encode(kv_tensor)
        dtype_id = _DTYPE_TO_ID.get(encoded.original_dtype, 2)
        ndim = len(encoded.original_shape)
        numel = encoded.quantized.numel()
        last_dim = encoded.scales.numel()

        header = struct.pack(
            f"<IBB{ndim}III",
            _MAGIC_KVCODEC,
            dtype_id,
            ndim,
            *encoded.original_shape,
            numel,
            last_dim,
        )
        q_bytes = _tensor_to_bytes(encoded.quantized)
        s_bytes = _tensor_to_bytes(encoded.scales)
        z_bytes = _tensor_to_bytes(encoded.zero_points)
        return header + q_bytes + s_bytes + z_bytes

    def decode_from_bytes(self, data: bytes) -> torch.Tensor:
        offset = 0
        magic, dtype_id, ndim = struct.unpack_from("<IBB", data, offset)
        offset += 6
        shape = list(struct.unpack_from(f"<{ndim}I", data, offset))
        offset += ndim * 4
        numel, last_dim = struct.unpack_from("<II", data, offset)
        offset += 8

        q_nbytes = numel
        s_nbytes = last_dim * 2
        z_nbytes = last_dim * 2

        quantized = _bytes_to_tensor(data, torch.uint8, [-1, last_dim], offset, q_nbytes)
        offset += q_nbytes
        scales = _bytes_to_tensor(data, torch.float16, [last_dim], offset, s_nbytes)
        offset += s_nbytes
        zero_points = _bytes_to_tensor(data, torch.float16, [last_dim], offset, z_nbytes)

        encoded = EncodedKV(
            quantized=quantized,
            scales=scales,
            zero_points=zero_points,
            original_dtype=_ID_TO_DTYPE.get(dtype_id, torch.float32),
            original_shape=shape,
        )
        return self.decode(encoded)


class NoOpCodec:
    def encode_to_bytes(self, kv_tensor: torch.Tensor) -> bytes:
        dtype_id = _DTYPE_TO_ID.get(kv_tensor.dtype, 2)
        ndim = kv_tensor.dim()
        shape = list(kv_tensor.shape)
        header = struct.pack(
            f"<IBB{ndim}I",
            _MAGIC_NOOP,
            dtype_id,
            ndim,
            *shape,
        )
        t = kv_tensor.contiguous()
        if kv_tensor.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        raw = _tensor_to_bytes(t)
        return header + raw

    def decode_from_bytes(self, data: bytes) -> torch.Tensor:
        offset = 0
        magic, dtype_id, ndim = struct.unpack_from("<IBB", data, offset)
        offset += 6
        shape = list(struct.unpack_from(f"<{ndim}I", data, offset))
        offset += ndim * 4

        original_dtype = _ID_TO_DTYPE.get(dtype_id, torch.float32)
        storage_dtype = torch.int16 if original_dtype == torch.bfloat16 else original_dtype
        numel = 1
        for s in shape:
            numel *= s
        elem_size = torch.tensor([], dtype=storage_dtype).element_size()
        nbytes = numel * elem_size

        t = _bytes_to_tensor(data, storage_dtype, shape, offset, nbytes)
        if original_dtype == torch.bfloat16:
            t = t.view(torch.bfloat16)
        return t
