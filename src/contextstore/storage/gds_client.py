from __future__ import annotations

import ctypes
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


class GdsUnavailableError(RuntimeError):
    """Raised when the local cuFile client cannot be initialized or used."""


class LocalGdsClient:
    """Process-local cuFile client with bounded CUDA-buffer registrations.

    The caller owns CUDA allocations. Keeping registrations keyed by `(ptr, size)`
    moves cuFileBufRegister out of the repeated KV LOAD path.
    """

    def __init__(
        self,
        library_path: str = "",
        file_cache_capacity: int = 128,
        buffer_cache_capacity: int = 2,
    ) -> None:
        self._library = self._load_library(library_path)
        self._configure_symbols()
        self._client = self._library.cs_gds_client_new(max(file_cache_capacity, 1))
        if not self._client:
            raise GdsUnavailableError("unable to initialize local GPUDirect Storage client")
        self._buffer_cache_capacity = max(buffer_cache_capacity, 1)
        self._buffers: OrderedDict[tuple[int, int], int] = OrderedDict()

    @staticmethod
    def _load_library(library_path: str) -> ctypes.CDLL:
        candidates: list[Path] = []
        if library_path:
            candidates.append(Path(library_path))
        env_path = os.environ.get("CONTEXTSTORE_GDS_FFI_PATH", "")
        if env_path:
            candidates.append(Path(env_path))
        repo_root = Path(__file__).resolve().parents[3]
        for extension in ("so", "dylib"):
            candidates.append(
                repo_root
                / "kv-service"
                / "gds-ffi"
                / "target"
                / "release"
                / f"libcontextstore_gds_ffi.{extension}"
            )
        for candidate in candidates:
            if candidate.is_file():
                return ctypes.CDLL(str(candidate))
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise GdsUnavailableError(f"contextstore GDS FFI library not found: {searched}")

    def _configure_symbols(self) -> None:
        self._library.cs_gds_client_new.argtypes = [ctypes.c_uint32]
        self._library.cs_gds_client_new.restype = ctypes.c_void_p
        self._library.cs_gds_client_free.argtypes = [ctypes.c_void_p]
        self._library.cs_gds_client_free.restype = None
        self._library.cs_gds_client_set_device.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._library.cs_gds_client_set_device.restype = ctypes.c_int
        self._library.cs_gds_register_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self._library.cs_gds_register_buffer.restype = ctypes.c_int64
        self._library.cs_gds_unregister_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._library.cs_gds_unregister_buffer.restype = None
        self._library.cs_gds_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self._library.cs_gds_read.restype = ctypes.c_int64

    def _region_for(self, ptr: int, size: int, device: int) -> int:
        key = (ptr, size)
        cached = self._buffers.get(key)
        if cached is not None:
            self._buffers.move_to_end(key)
            return cached
        if self._library.cs_gds_client_set_device(self._client, device) != 0:
            raise GdsUnavailableError(f"unable to select CUDA device {device} for GDS")
        region_id = int(
            self._library.cs_gds_register_buffer(
                self._client,
                ctypes.c_void_p(ptr),
                ctypes.c_uint64(size),
            )
        )
        if region_id <= 0:
            raise GdsUnavailableError("cuFileBufRegister failed for GPU staging buffer")
        while len(self._buffers) >= self._buffer_cache_capacity:
            _, evicted = self._buffers.popitem(last=False)
            self._library.cs_gds_unregister_buffer(self._client, ctypes.c_uint64(evicted))
        self._buffers[key] = region_id
        return region_id

    def read_into(
        self,
        ptr: int,
        size: int,
        device: int,
        segments: Iterable[tuple[Path, int, int]],
    ) -> int:
        """Read `(path, destination_offset, length)` segments into one CUDA allocation."""
        region_id = self._region_for(ptr, size, device)
        total = 0
        for path, destination_offset, length in segments:
            if destination_offset < 0 or length <= 0 or destination_offset + length > size:
                raise GdsUnavailableError("invalid GDS destination range")
            transferred = int(
                self._library.cs_gds_read(
                    self._client,
                    ctypes.c_uint64(region_id),
                    os.fsencode(path),
                    ctypes.c_uint64(0),
                    ctypes.c_uint64(destination_offset),
                    ctypes.c_uint64(length),
                )
            )
            if transferred != length:
                raise GdsUnavailableError(
                    f"cuFileRead failed or returned a short read for {path}: {transferred}/{length}"
                )
            total += transferred
        return total

    def prepare_buffer(self, ptr: int, size: int, device: int) -> None:
        """Register a reusable CUDA allocation before it enters the LOAD hot path."""
        self._region_for(ptr, size, device)

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if not client:
            return
        for region_id in self._buffers.values():
            self._library.cs_gds_unregister_buffer(self._client, ctypes.c_uint64(region_id))
        self._buffers.clear()
        self._library.cs_gds_client_free(client)
        self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
