"""Multi-rail RDMA reader for ContextStore (Python ctypes binding).

A single Worker reads one striped object in parallel over several local RDMA
devices ("rails"). The Rust side (:mod:`contextstore_rdma_ffi` ``cs_mr_*`` C
ABI) manages per-rail QPs/CQs/memory regions, health + cooldown, byte-exact
stripe balancing, backpressure, integrity verification (byte counts, chunk
counts, optional per-stripe xxh3-64), and late-write protection: a failed or
cancelled read quiesces every participating connection before returning.

Typical use with the gRPC client::

    from contextstore.kvservice_client.client import KVClient
    from contextstore.storage.multirail_client import MultiRailReader

    grpc = KVClient("http://10.0.0.1:50051")
    reader = MultiRailReader(["mlx5_0", "mlx5_1"])          # local HCAs
    buffer = pinned_pool_region(size)                        # long-lived

    lookup = grpc.lookup_object("ns", "key")                 # descriptor+placement
    n = reader.read_into(buffer, lookup, sticky=True)        # all rails in parallel
    for stats in reader.rail_stats():
        print(stats)                                        # per-rail metrics

``sticky=True`` keeps the per-rail registrations cached across reads; the
buffer must then be a long-lived pinned pool region that outlives the reader
and is never freed or reused for non-RDMA purposes while it is open.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Sequence

from .rdma_client import _find_lib

_ERR_LEN = 512


class _CsMrDescriptor(ctypes.Structure):
    _fields_ = [
        ("namespace", ctypes.c_char_p),
        ("object_key", ctypes.c_char_p),
        ("object_handle", ctypes.c_char_p),
        ("object_generation", ctypes.c_uint64),
        ("content_etag", ctypes.c_char_p),
        ("layout_version", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("is_striped", ctypes.c_uint32),
        ("stripe_count", ctypes.c_uint32),
        ("chunk_size", ctypes.c_uint64),
    ]


class _CsMrChunk(ctypes.Structure):
    _fields_ = [
        ("stripe_index", ctypes.c_uint32),
        ("rdma_endpoint", ctypes.c_char_p),
        ("offset", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("checksum", ctypes.c_char_p),
    ]


class _CsMrRailStats(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("healthy", ctypes.c_uint32),
        ("cooldown_ms", ctypes.c_uint32),
        ("requests_ok", ctypes.c_uint64),
        ("requests_err", ctypes.c_uint64),
        ("bytes_read", ctypes.c_uint64),
        ("timeouts", ctypes.c_uint64),
        ("connections_created", ctypes.c_uint64),
        ("connections_quiesced", ctypes.c_uint64),
        ("inflight_requests", ctypes.c_uint64),
        ("inflight_bytes", ctypes.c_uint64),
        ("latency_avg_us", ctypes.c_uint64),
        ("latency_max_us", ctypes.c_uint64),
        ("registered_bytes", ctypes.c_uint64),
        ("device", ctypes.c_char * 64),
        ("topology", ctypes.c_char * 160),
    ]


@dataclass
class RailStats:
    index: int
    device: str
    topology: str
    healthy: bool
    cooldown_ms: int
    requests_ok: int
    requests_err: int
    bytes_read: int
    timeouts: int
    connections_created: int
    connections_quiesced: int
    inflight_requests: int
    inflight_bytes: int
    latency_avg_us: int
    latency_max_us: int
    registered_bytes: int

    def __str__(self) -> str:
        return (
            f"rail[{self.index}] {self.device} {self.topology} "
            f"healthy={self.healthy} cooldown={self.cooldown_ms}ms "
            f"ok={self.requests_ok} err={self.requests_err} tmo={self.timeouts} "
            f"read={self.bytes_read >> 20}MiB conns=+{self.connections_created}"
            f"/~{self.connections_quiesced} "
            f"inflight=(req:{self.inflight_requests},B:{self.inflight_bytes}) "
            f"lat=(avg:{self.latency_avg_us}us,max:{self.latency_max_us}us) "
            f"reg={self.registered_bytes >> 20}MiB"
        )


class MultiRailError(RuntimeError):
    """Raised when a multi-rail read fails; text comes from the typed Rust error."""


class MultiRailReader:
    """Read one striped object across several local RDMA devices."""

    def __init__(
        self,
        rails: Sequence[str],
        io_timeout_ms: int = 30_000,
        lib_path: str | None = None,
    ) -> None:
        """``rails`` entries are ``device[:port[:gid[:weight[:mtu]]]]`` specs."""
        if not rails:
            raise ValueError("at least one rail is required")
        self._lib = ctypes.CDLL(lib_path or _find_lib(), use_errno=True)
        self._setup_prototypes()
        specs = [ctypes.c_char_p(r.encode()) for r in rails]
        array = (ctypes.c_char_p * len(specs))(*specs)
        self._handle = self._lib.cs_mr_new(array, len(specs), io_timeout_ms)
        if not self._handle:
            raise MultiRailError(
                f"cs_mr_new failed (rails={list(rails)}); check device names/GIDs"
            )

    def _setup_prototypes(self) -> None:
        lib = self._lib
        lib.cs_mr_new.restype = ctypes.c_void_p
        lib.cs_mr_new.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint32,
            ctypes.c_uint64,
        ]
        lib.cs_mr_read.restype = ctypes.c_int64
        lib.cs_mr_read.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CsMrDescriptor),
            ctypes.POINTER(_CsMrChunk),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        lib.cs_mr_rail_stats.restype = ctypes.c_int32
        lib.cs_mr_rail_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CsMrRailStats),
            ctypes.c_uint32,
        ]
        lib.cs_mr_free.argtypes = [ctypes.c_void_p]
        lib.cs_mr_free.restype = None

    def read_into(
        self,
        buffer,  # ctypes buffer / memoryview-compatible address
        lookup,
        sticky: bool = False,
        buffer_addr: int | None = None,
        buffer_len: int | None = None,
    ) -> int:
        """Read the looked-up object into ``buffer`` across all healthy rails.

        ``lookup`` is a gRPC ``LookupObjectResponse``-style object exposing
        ``.descriptor`` (with namespace/object_key via ``.key``, handle,
        generation, etag, layout, size, striping fields) and ``.placement``
        (with ``.chunks``). Returns the number of bytes placed in the buffer.
        """
        descriptor = lookup.descriptor
        placement = getattr(lookup, "placement", None)
        if placement is None or not placement.chunks:
            raise MultiRailError("lookup carried no placement chunks")
        key = getattr(descriptor, "key", None)
        if key is None:
            raise MultiRailError("descriptor is missing its object key")

        addr = buffer_addr if buffer_addr is not None else ctypes.addressof(buffer)
        length = buffer_len if buffer_len is not None else ctypes.sizeof(buffer)

        c_desc = _CsMrDescriptor(
            namespace=key.namespace.encode(),
            object_key=key.object_key.encode(),
            object_handle=descriptor.object_handle.encode(),
            object_generation=descriptor.object_generation,
            content_etag=descriptor.content_etag.encode(),
            layout_version=descriptor.layout_version,
            size=descriptor.size,
            is_striped=1 if descriptor.is_striped else 0,
            stripe_count=descriptor.stripe_count,
            chunk_size=descriptor.chunk_size,
        )
        c_chunks = [
            _CsMrChunk(
                stripe_index=chunk.stripe_index,
                rdma_endpoint=chunk.rdma_endpoint.encode(),
                offset=chunk.offset,
                length=chunk.length,
                checksum=(chunk.checksum or "").encode() or None,
            )
            for chunk in placement.chunks
        ]
        array = (_CsMrChunk * len(c_chunks))(*c_chunks)
        err = ctypes.create_string_buffer(_ERR_LEN)
        result = self._lib.cs_mr_read(
            self._handle,
            ctypes.byref(c_desc),
            array,
            len(c_chunks),
            ctypes.c_void_p(addr),
            length,
            1 if sticky else 0,
            err,
            _ERR_LEN,
        )
        if result < 0:
            raise MultiRailError(err.value.decode(errors="replace") or "cs_mr_read failed")
        return int(result)

    def rail_stats(self) -> list[RailStats]:
        """Snapshot of per-rail health, throughput, error and in-flight stats."""
        max_rails = 16
        array = (_CsMrRailStats * max_rails)()
        count = self._lib.cs_mr_rail_stats(self._handle, array, max_rails)
        if count < 0:
            raise MultiRailError("cs_mr_rail_stats failed")
        stats = []
        for i in range(count):
            entry = array[i]
            stats.append(
                RailStats(
                    index=entry.index,
                    device=entry.device.decode(errors="replace"),
                    topology=entry.topology.decode(errors="replace"),
                    healthy=bool(entry.healthy),
                    cooldown_ms=entry.cooldown_ms,
                    requests_ok=entry.requests_ok,
                    requests_err=entry.requests_err,
                    bytes_read=entry.bytes_read,
                    timeouts=entry.timeouts,
                    connections_created=entry.connections_created,
                    connections_quiesced=entry.connections_quiesced,
                    inflight_requests=entry.inflight_requests,
                    inflight_bytes=entry.inflight_bytes,
                    latency_avg_us=entry.latency_avg_us,
                    latency_max_us=entry.latency_max_us,
                    registered_bytes=entry.registered_bytes,
                )
            )
        return stats

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.cs_mr_free(self._handle)
            self._handle = None

    def __del__(self):  # noqa: D105
        try:
            self.close()
        except Exception:
            pass
