from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Any


class AlignedBuffer:
    def __init__(self, size: int, alignment: int = 4096) -> None:
        self.size = size
        self._libc = ctypes.CDLL(None)
        self._ptr = ctypes.c_void_p()
        rc = self._libc.posix_memalign(ctypes.byref(self._ptr), alignment, size)
        if rc != 0 or not self._ptr.value:
            raise MemoryError(f"posix_memalign failed: rc={rc}")
        ctypes.memset(self._ptr, 0, size)

    @property
    def addr(self) -> int:
        value = self._ptr.value
        if value is None:
            raise RuntimeError("buffer already freed")
        return int(value)

    def fill_pattern(self, seed: int) -> None:
        array_t = ctypes.c_uint8 * self.size
        view = array_t.from_address(self.addr)
        for i in range(self.size):
            view[i] = (i * 131 + seed) & 0xFF

    def zero(self) -> None:
        ctypes.memset(self.addr, 0, self.size)

    def to_bytes(self) -> bytes:
        return ctypes.string_at(self.addr, self.size)

    def close(self) -> None:
        if self._ptr.value:
            self._libc.free(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> AlignedBuffer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class Result:
    size: int
    write_ms: float
    read_ms: float

    @property
    def write_mbps(self) -> float:
        return self.size / (1024 * 1024) / (self.write_ms / 1000)

    @property
    def read_mbps(self) -> float:
        return self.size / (1024 * 1024) / (self.read_ms / 1000)


def parse_size(value: str) -> int:
    raw = value.strip().lower()
    scale = 1
    if raw.endswith("k"):
        scale = 1024
        raw = raw[:-1]
    elif raw.endswith("m"):
        scale = 1024 * 1024
        raw = raw[:-1]
    elif raw.endswith("g"):
        scale = 1024 * 1024 * 1024
        raw = raw[:-1]
    return int(raw) * scale


def wait_done(agent: Any, handle: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    status = agent.transfer(handle)
    while status == "PROC":
        if time.monotonic() > deadline:
            raise TimeoutError("NIXL transfer timed out")
        time.sleep(0.001)
        status = agent.check_xfer_state(handle)
    if status != "DONE":
        raise RuntimeError(f"NIXL transfer failed: {status}")


def run_once(
    agent: Any,
    size: int,
    key: str,
    timeout_s: float,
    src: AlignedBuffer,
    dst: AlignedBuffer,
) -> Result:
    local_reg = None
    remote_reg = None
    write_handle = None
    read_handle = None
    local_write_side = None
    local_read_side = None
    remote_side = None
    try:
        src.fill_pattern(seed=len(key))
        dst.zero()
        object_dev_id = (zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF) + 1

        local_reg = agent.register_memory(
            [(src.addr, size, 0, ""), (dst.addr, size, 0, "")],
            "DRAM",
            ["CONTEXTSTORE"],
        )
        remote_reg = agent.register_memory(
            [(0, size, object_dev_id, key)],
            "OBJ",
            ["CONTEXTSTORE"],
        )

        local_write = agent.get_xfer_descs([(src.addr, size, 0)], "DRAM")
        local_read = agent.get_xfer_descs([(dst.addr, size, 0)], "DRAM")
        remote = agent.get_xfer_descs([(0, size, object_dev_id)], "OBJ")

        local_write_side = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local_write,
            backends=["CONTEXTSTORE"],
        )
        local_read_side = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local_read,
            backends=["CONTEXTSTORE"],
        )
        remote_side = agent.prep_xfer_dlist(
            agent.name,
            remote,
            backends=["CONTEXTSTORE"],
        )

        write_handle = agent.make_prepped_xfer(
            "WRITE",
            local_write_side,
            [0],
            remote_side,
            [0],
            backends=["CONTEXTSTORE"],
        )
        start = time.perf_counter()
        wait_done(agent, write_handle, timeout_s)
        write_ms = (time.perf_counter() - start) * 1000

        read_handle = agent.make_prepped_xfer(
            "READ",
            local_read_side,
            [0],
            remote_side,
            [0],
            backends=["CONTEXTSTORE"],
        )
        start = time.perf_counter()
        wait_done(agent, read_handle, timeout_s)
        read_ms = (time.perf_counter() - start) * 1000

        if src.to_bytes() != dst.to_bytes():
            raise RuntimeError(f"roundtrip payload mismatch for key={key}")
        return Result(size=size, write_ms=write_ms, read_ms=read_ms)
    finally:
        for handle in (read_handle, write_handle):
            if handle is not None:
                agent.release_xfer_handle(handle)
        for handle in (remote_side, local_read_side, local_write_side):
            if handle is not None:
                agent.release_dlist_handle(handle)
        if remote_reg is not None:
            agent.deregister_memory(remote_reg, ["CONTEXTSTORE"])
        if local_reg is not None:
            agent.deregister_memory(local_reg, ["CONTEXTSTORE"])


def backend_params(args: argparse.Namespace) -> dict[str, str]:
    params: dict[str, str] = {}
    if args.mode == "file":
        params["file_root"] = args.file_root
        return params

    params.update(
        {
            "client_library": args.client_library,
            "endpoint": args.endpoint,
            "namespace": args.namespace,
        }
    )
    if args.mode == "rdma":
        params["rdma_enabled"] = "true"
        params["rdma_server_addr"] = args.rdma_server_addr
    return params


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CONTEXTSTORE NIXL backend roundtrip tests."
    )
    parser.add_argument("--mode", choices=("file", "grpc", "rdma"), default="file")
    parser.add_argument(
        "--size",
        action="append",
        default=None,
        help="payload size, e.g. 1m",
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--file-root", default="/tmp/contextstore-nixl-objects")
    parser.add_argument("--client-library", default="")
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--rdma-server-addr", default="127.0.0.1:50053")
    parser.add_argument("--namespace", default="nixl-roundtrip")
    args = parser.parse_args()

    if args.mode != "file" and not args.client_library:
        parser.error("--client-library is required for grpc/rdma mode")
    if "NIXL_PLUGIN_DIR" not in os.environ:
        parser.error(
            "NIXL_PLUGIN_DIR must include the directory containing "
            "libplugin_CONTEXTSTORE.so"
        )

    import nixl._api as nixl_api

    config = nixl_api.nixl_agent_config(backends=[])
    agent = nixl_api.nixl_agent("contextstore-nixl-roundtrip", config)
    if "CONTEXTSTORE" not in agent.get_plugin_list():
        raise RuntimeError(
            f"CONTEXTSTORE plugin not found; plugins={agent.get_plugin_list()}"
        )
    agent.create_backend("CONTEXTSTORE", backend_params(args))

    sizes = [parse_size(v) for v in (args.size or ["1m"])]
    for size in sizes:
        results = []
        with AlignedBuffer(size) as src, AlignedBuffer(size) as dst:
            for i in range(args.iters):
                key = f"{args.namespace}-{args.mode}-{size}-{i}|__combined__"
                result = run_once(agent, size, key, args.timeout_s, src, dst)
                results.append(result)
                print(
                    f"size={size} iter={i} write={result.write_ms:.3f}ms "
                    f"({result.write_mbps:.1f} MB/s) read={result.read_ms:.3f}ms "
                    f"({result.read_mbps:.1f} MB/s)",
                    flush=True,
                )
        if len(results) > 1:
            write_ms = statistics.median(r.write_ms for r in results)
            read_ms = statistics.median(r.read_ms for r in results)
            summary = Result(size=size, write_ms=write_ms, read_ms=read_ms)
            print(
                f"median size={size} write={summary.write_ms:.3f}ms "
                f"({summary.write_mbps:.1f} MB/s) read={summary.read_ms:.3f}ms "
                f"({summary.read_mbps:.1f} MB/s)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
