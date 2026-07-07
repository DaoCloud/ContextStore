from __future__ import annotations

"""Python RDMA client — ctypes wrapper for contextstore-rdma-ffi (Rust cdylib).

Usage:
    from contextstore.storage.rdma_client import RdmaClient

    client = RdmaClient(
        server_addr="127.0.0.1:50053",
        device="mlx5_0", port=1, gid_index=3,
        buf_size_mb=512,
    )
    client.connect()
    n = client.get("rust-bench:rdma_test0:__combined__")
    if n > 0:
        # n bytes have been RDMA WRITE'd into client.buffer (zero-copy view)
        data: bytes = client.buffer_view(n)
    client.close()
"""

import ctypes
import ctypes.util
import logging
import os
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


def _find_lib() -> str:
    """Search for contextstore_rdma_ffi.so. Prefer env var, then cargo target, then system path."""
    # 1) explicit env var
    env_path = os.environ.get("CONTEXTSTORE_RDMA_LIB")
    if env_path and Path(env_path).is_file():
        return env_path
    # 2) common cargo target paths (persistent / remote environments)
    candidates = [
        # Persistent PVC paths (preferred, survive container restarts)
        Path("/data/cs-build/rdma-ffi-target/release/libcontextstore_rdma_ffi.so"),
        Path("/data/cs-build/target/release/libcontextstore_rdma_ffi.so"),
        # Legacy transient cargo target (compatibility)
        Path("/root/cs-build/rdma-ffi-target/release/libcontextstore_rdma_ffi.so"),
        Path("/root/cs-build/target/release/libcontextstore_rdma_ffi.so"),
        # workspace-relative (for dev)
        Path(__file__).parent.parent.parent.parent
        / "kv-service/rdma-ffi/target/release/libcontextstore_rdma_ffi.so",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    # 3) system search
    found = ctypes.util.find_library("contextstore_rdma_ffi")
    if found:
        return found
    raise FileNotFoundError(
        "libcontextstore_rdma_ffi.so not found. Build it with: "
        "cd kv-service/rdma-ffi && cargo build --release. "
        "Or set env CONTEXTSTORE_RDMA_LIB=/path/to/libcontextstore_rdma_ffi.so"
    )


class RdmaClient:
    """Single-connection RDMA client. **Not thread-safe** (internal buffer is shared). Multi-threaded use requires one client per thread."""

    def __init__(
        self,
        server_addr: str,
        device: str = "mlx5_0",
        port: int = 1,
        gid_index: int = 3,
        buf_size_mb: int = 512,
    ) -> None:
        self.server_addr = server_addr
        self.device = device
        self.port = port
        self.gid_index = gid_index
        self.buf_size = buf_size_mb * 1024 * 1024

        lib_path = _find_lib()
        logger.info("RdmaClient: loading lib from %s", lib_path)
        self._lib = ctypes.CDLL(lib_path, use_errno=True)
        self._setup_prototypes()

        # Create client (handle is an opaque void*)
        self._handle = self._lib.cs_rdma_client_new(
            device.encode(),
            ctypes.c_uint8(port),
            ctypes.c_uint8(gid_index),
            ctypes.c_uint64(self.buf_size),
        )
        if not self._handle:
            raise RuntimeError(
                f"cs_rdma_client_new failed (device={device}, buf_size={self.buf_size})"
            )
        self._connected = False
        self._lock = Lock()  # Guards get() calls (server responses match requests 1:1, cannot interleave)

    def _setup_prototypes(self) -> None:
        L = self._lib
        L.cs_rdma_client_new.restype = ctypes.c_void_p
        L.cs_rdma_client_new.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint64,
        ]
        L.cs_rdma_client_connect.restype = ctypes.c_int
        L.cs_rdma_client_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.cs_rdma_client_get.restype = ctypes.c_int64
        L.cs_rdma_client_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.cs_rdma_client_buffer.restype = ctypes.POINTER(ctypes.c_uint8)
        L.cs_rdma_client_buffer.argtypes = [ctypes.c_void_p]
        L.cs_rdma_client_buffer_size.restype = ctypes.c_uint64
        L.cs_rdma_client_buffer_size.argtypes = [ctypes.c_void_p]
        L.cs_rdma_client_free.restype = None
        L.cs_rdma_client_free.argtypes = [ctypes.c_void_p]
        # ===== Plan A zero-copy interface (added by rdma-ffi, absent in older .so) =====
        self._has_zerocopy = hasattr(L, "cs_rdma_client_register_external_buffer")
        if self._has_zerocopy:
            L.cs_rdma_client_register_external_buffer.restype = ctypes.c_int32
            L.cs_rdma_client_register_external_buffer.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64
            ]
            L.cs_rdma_client_get_into.restype = ctypes.c_int64
            L.cs_rdma_client_get_into.argtypes = [
                ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_uint64
            ]
            L.cs_rdma_client_unregister_external_buffer.restype = ctypes.c_int32
            L.cs_rdma_client_unregister_external_buffer.argtypes = [
                ctypes.c_void_p, ctypes.c_int32
            ]
        self._has_descriptor_get_into = hasattr(L, "cs_rdma_client_get_descriptor_into")
        if self._has_descriptor_get_into:
            L.cs_rdma_client_get_descriptor_into.restype = ctypes.c_int64
            L.cs_rdma_client_get_descriptor_into.argtypes = [
                ctypes.c_void_p,  # client
                ctypes.c_int32,   # region_id
                ctypes.c_char_p,  # key
                ctypes.c_char_p,  # object_handle
                ctypes.c_uint64,  # object_generation
                ctypes.c_char_p,  # content_etag
                ctypes.c_uint64,  # layout_version
                ctypes.c_uint64,  # size
                ctypes.c_uint8,   # is_striped
                ctypes.c_uint32,  # stripe_count
                ctypes.c_uint64,  # chunk_size
                ctypes.c_uint64,  # offset
            ]
        self._has_descriptor_get = hasattr(L, "cs_rdma_client_get_descriptor")
        if self._has_descriptor_get:
            L.cs_rdma_client_get_descriptor.restype = ctypes.c_int64
            L.cs_rdma_client_get_descriptor.argtypes = [
                ctypes.c_void_p,  # client
                ctypes.c_char_p,  # key
                ctypes.c_char_p,  # object_handle
                ctypes.c_uint64,  # object_generation
                ctypes.c_char_p,  # content_etag
                ctypes.c_uint64,  # layout_version
                ctypes.c_uint64,  # size
                ctypes.c_uint8,   # is_striped
                ctypes.c_uint32,  # stripe_count
                ctypes.c_uint64,  # chunk_size
            ]
        # ===== PUT data plane (further additions in rdma-ffi, absent in older .so) =====
        self._has_put = hasattr(L, "cs_rdma_client_put")
        if self._has_put:
            L.cs_rdma_client_put.restype = ctypes.c_int32
            L.cs_rdma_client_put.argtypes = [
                ctypes.c_void_p,  # client
                ctypes.c_int32,   # region_id
                ctypes.c_char_p,  # key
                ctypes.c_uint64,  # offset
                ctypes.c_uint64,  # size
            ]

    def connect(self) -> None:
        if self._connected:
            return
        rc = self._lib.cs_rdma_client_connect(self._handle, self.server_addr.encode())
        if rc != 0:
            raise RuntimeError(f"cs_rdma_client_connect failed: rc={rc}")
        self._connected = True
        logger.info("RdmaClient connected to %s", self.server_addr)

    def get(self, key: str) -> int:
        """Send GET and block until server WRITE completes. Returns bytes written; 0 = miss; exception = error.

        Written data starts at offset 0 of the buffer, length equals the return value.
        """
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        with self._lock:
            n = self._lib.cs_rdma_client_get(self._handle, key.encode())
        if n < 0:
            raise RuntimeError(f"cs_rdma_client_get failed: rc={n}")
        return int(n)

    def buffer_ptr(self) -> int:
        """Returns the buffer start address (int). Advanced usage (numpy / torch frombuffer)."""
        ptr = self._lib.cs_rdma_client_buffer(self._handle)
        return ctypes.cast(ptr, ctypes.c_void_p).value or 0

    def buffer_view(self, length: int) -> bytes:
        """Copy `length` bytes out of the buffer (memcpy). Simple but incurs one copy.

        Uses ctypes.string_at, slightly faster than bytes(memoryview(...)).
        """
        ptr = self._lib.cs_rdma_client_buffer(self._handle)
        return ctypes.string_at(ptr, length)

    def buffer_memview(self, length: int) -> memoryview:
        """Zero-copy memoryview into the buffer. Warning: don't hold this view while server rewrites the buffer!"""
        ptr = self._lib.cs_rdma_client_buffer(self._handle)
        # cast as c_uint8 array
        arr_type = ctypes.c_uint8 * length
        arr = arr_type.from_address(ctypes.addressof(ptr.contents))
        return memoryview(arr)

    # ===== Plan A zero-copy interface =====
    @property
    def supports_zerocopy(self) -> bool:
        """Whether the rdma-ffi .so provides the zero-copy interface (register_external + get_into)."""
        return self._has_zerocopy

    def register_external_buffer(self, ptr: int, size: int) -> int:
        """Register a caller-owned pinned host buffer as an RDMA MR. Returns region_id (>=0).

        Args:
            ptr: memory start address (int, typically from `numpy.ndarray.ctypes.data` or `posix_memalign`)
            size: buffer size in bytes

        Caller must ensure:
        - Memory is 4KB page-aligned (recommend `posix_memalign(align=4096)`)
        - Already mlock'd or cudaHostAlloc'd, else reg_mr triggers implicit pin (slow, consumes RLIMIT_MEMLOCK)
        - Lifetime ≥ until unregister call

        Typical usage (numpy + mlock):
            arr = np.zeros(N, dtype=np.uint8)  # Note: default is not page-aligned
            # Or use posix_memalign for a page-aligned block
            region_id = client.register_external_buffer(arr.ctypes.data, arr.nbytes)
        """
        if not self._has_zerocopy:
            raise RuntimeError("older rdma-ffi does not support zero-copy interface, please rebuild")
        ptr_u8 = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8))
        rid = self._lib.cs_rdma_client_register_external_buffer(
            self._handle, ptr_u8, ctypes.c_uint64(size)
        )
        if rid < 0:
            raise RuntimeError(f"register_external_buffer failed: rc={rid}")
        return int(rid)

    def get_into(self, region_id: int, key: str, offset: int = 0) -> int:
        """RDMA GET; server WRITE goes directly to buffer at `region_id` starting at `offset`.

        Returns bytes written; 0 = miss; exception = error.

        Zero-copy: after data is written, Python can directly numpy/torch frombuffer view the memory without memcpy or GIL hold.
        """
        if not self._has_zerocopy:
            raise RuntimeError("older rdma-ffi does not support zero-copy interface, please rebuild")
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        with self._lock:
            n = self._lib.cs_rdma_client_get_into(
                self._handle, ctypes.c_int32(region_id), key.encode(), ctypes.c_uint64(offset)
            )
        if n < 0:
            raise RuntimeError(f"get_into failed: rc={n}")
        return int(n)

    @property
    def supports_descriptor_get(self) -> bool:
        """Whether the rdma-ffi .so provides the descriptor GET interface."""
        return self._has_descriptor_get and self._has_descriptor_get_into

    def get_descriptor_into(self, region_id: int, key: str, descriptor, offset: int = 0) -> int:
        """RDMA descriptor GET; server writes the descriptor-specified version into the external buffer."""
        if not self._has_descriptor_get_into:
            raise RuntimeError("older rdma-ffi does not support descriptor GET interface, please rebuild")
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        with self._lock:
            n = self._lib.cs_rdma_client_get_descriptor_into(
                self._handle,
                ctypes.c_int32(region_id),
                key.encode(),
                descriptor.object_handle.encode(),
                ctypes.c_uint64(descriptor.object_generation),
                descriptor.content_etag.encode(),
                ctypes.c_uint64(descriptor.layout_version),
                ctypes.c_uint64(descriptor.size),
                ctypes.c_uint8(1 if descriptor.is_striped else 0),
                ctypes.c_uint32(descriptor.stripe_count),
                ctypes.c_uint64(descriptor.chunk_size),
                ctypes.c_uint64(offset),
            )
        if n < 0:
            raise RuntimeError(f"get_descriptor_into failed: rc={n}")
        return int(n)

    def get_descriptor(self, key: str, descriptor) -> int:
        """RDMA descriptor GET; server writes the descriptor-specified version into the client's built-in buffer."""
        if not self._has_descriptor_get:
            raise RuntimeError("older rdma-ffi does not support descriptor GET interface, please rebuild")
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        with self._lock:
            n = self._lib.cs_rdma_client_get_descriptor(
                self._handle,
                key.encode(),
                descriptor.object_handle.encode(),
                ctypes.c_uint64(descriptor.object_generation),
                descriptor.content_etag.encode(),
                ctypes.c_uint64(descriptor.layout_version),
                ctypes.c_uint64(descriptor.size),
                ctypes.c_uint8(1 if descriptor.is_striped else 0),
                ctypes.c_uint32(descriptor.stripe_count),
                ctypes.c_uint64(descriptor.chunk_size),
            )
        if n < 0:
            raise RuntimeError(f"get_descriptor failed: rc={n}")
        return int(n)

    def unregister_external_buffer(self, region_id: int) -> None:
        """Unregister (dereg_mr). Caller is responsible for freeing memory."""
        if not self._has_zerocopy:
            return
        rc = self._lib.cs_rdma_client_unregister_external_buffer(
            self._handle, ctypes.c_int32(region_id)
        )
        if rc != 0:
            logger.warning("unregister_external_buffer rc=%d (region_id=%d)", rc, region_id)

    # ===== PUT data plane =====
    @property
    def supports_put(self) -> bool:
        """Whether the rdma-ffi .so provides the PUT interface."""
        return self._has_put

    def put(self, region_id: int, key: str, offset: int, size: int) -> None:
        """RDMA PUT: push [offset, offset+size) from the pre-registered buffer to the server; returns after server persists.

        Args:
            region_id: region id returned by register_external_buffer
            key: ObjectKey canonical string, format "<namespace_byte_len>:<namespace><object_key>"
            offset: offset within the buffer
            size: bytes to push

        Raises:
            RuntimeError: older rdma-ffi does not support / not connected / RDMA WRITE failed / server persist failed
        """
        if not self._has_put:
            raise RuntimeError("older rdma-ffi does not support PUT interface, please rebuild")
        if not self._connected:
            raise RuntimeError("not connected — call connect() first")
        with self._lock:
            rc = self._lib.cs_rdma_client_put(
                self._handle,
                ctypes.c_int32(region_id),
                key.encode(),
                ctypes.c_uint64(offset),
                ctypes.c_uint64(size),
            )
        if rc != 0:
            raise RuntimeError(f"cs_rdma_client_put failed: rc={rc}")

    def close(self) -> None:
        if self._handle:
            self._lib.cs_rdma_client_free(self._handle)
            self._handle = None
            self._connected = False

    def __enter__(self) -> "RdmaClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
