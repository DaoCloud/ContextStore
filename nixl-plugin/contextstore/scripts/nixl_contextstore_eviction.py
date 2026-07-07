from __future__ import annotations

import argparse
import ctypes
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

    def fill_byte(self, value: int) -> None:
        ctypes.memset(self.addr, value & 0xFF, self.size)

    def zero(self) -> None:
        ctypes.memset(self.addr, 0, self.size)

    def sampled_byte_matches(self, expected: int, stride: int = 4096) -> bool:
        byte = expected & 0xFF
        view = (ctypes.c_uint8 * self.size).from_address(self.addr)
        for offset in range(0, self.size, stride):
            if view[offset] != byte:
                return False
        return self.size == 0 or view[self.size - 1] == byte

    def close(self) -> None:
        if self._ptr.value:
            self._libc.free(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __enter__(self) -> AlignedBuffer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class TransferResult:
    phase: str
    key: str
    size: int
    elapsed_ms: float

    @property
    def mbps(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return self.size / (1024 * 1024) / (self.elapsed_ms / 1000)


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


def phase(name: str) -> None:
    print(f"PHASE name={name} ts={time.time():.9f} ns={time.time_ns()}", flush=True)


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


def object_dev_id(key: str) -> int:
    return (zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF) + 1


def put_object(
    agent: Any,
    key: str,
    buffer: AlignedBuffer,
    timeout_s: float,
) -> TransferResult:
    local_reg = None
    remote_reg = None
    local_side = None
    remote_side = None
    handle = None
    size = buffer.size
    dev_id = object_dev_id(key)
    try:
        local_reg = agent.register_memory(
            [(buffer.addr, size, 0, "")],
            "DRAM",
            ["CONTEXTSTORE"],
        )
        remote_reg = agent.register_memory(
            [(0, size, dev_id, key)],
            "OBJ",
            ["CONTEXTSTORE"],
        )
        local = agent.get_xfer_descs([(buffer.addr, size, 0)], "DRAM")
        remote = agent.get_xfer_descs([(0, size, dev_id)], "OBJ")
        local_side = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local,
            backends=["CONTEXTSTORE"],
        )
        remote_side = agent.prep_xfer_dlist(
            agent.name,
            remote,
            backends=["CONTEXTSTORE"],
        )
        handle = agent.make_prepped_xfer(
            "WRITE",
            local_side,
            [0],
            remote_side,
            [0],
            backends=["CONTEXTSTORE"],
        )
        start = time.perf_counter()
        wait_done(agent, handle, timeout_s)
        return TransferResult(
            phase="put",
            key=key,
            size=size,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    finally:
        if handle is not None:
            agent.release_xfer_handle(handle)
        for side in (remote_side, local_side):
            if side is not None:
                agent.release_dlist_handle(side)
        if remote_reg is not None:
            agent.deregister_memory(remote_reg, ["CONTEXTSTORE"])
        if local_reg is not None:
            agent.deregister_memory(local_reg, ["CONTEXTSTORE"])


def get_object(
    agent: Any,
    key: str,
    buffer: AlignedBuffer,
    timeout_s: float,
) -> TransferResult:
    local_reg = None
    remote_reg = None
    local_side = None
    remote_side = None
    handle = None
    size = buffer.size
    dev_id = object_dev_id(key)
    try:
        local_reg = agent.register_memory(
            [(buffer.addr, size, 0, "")],
            "DRAM",
            ["CONTEXTSTORE"],
        )
        remote_reg = agent.register_memory(
            [(0, size, dev_id, key)],
            "OBJ",
            ["CONTEXTSTORE"],
        )
        local = agent.get_xfer_descs([(buffer.addr, size, 0)], "DRAM")
        remote = agent.get_xfer_descs([(0, size, dev_id)], "OBJ")
        local_side = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local,
            backends=["CONTEXTSTORE"],
        )
        remote_side = agent.prep_xfer_dlist(
            agent.name,
            remote,
            backends=["CONTEXTSTORE"],
        )
        handle = agent.make_prepped_xfer(
            "READ",
            local_side,
            [0],
            remote_side,
            [0],
            backends=["CONTEXTSTORE"],
        )
        start = time.perf_counter()
        wait_done(agent, handle, timeout_s)
        return TransferResult(
            phase="get",
            key=key,
            size=size,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    finally:
        if handle is not None:
            agent.release_xfer_handle(handle)
        for side in (remote_side, local_side):
            if side is not None:
                agent.release_dlist_handle(side)
        if remote_reg is not None:
            agent.deregister_memory(remote_reg, ["CONTEXTSTORE"])
        if local_reg is not None:
            agent.deregister_memory(local_reg, ["CONTEXTSTORE"])


def print_result(result: TransferResult, tag: str, verified: bool | None = None) -> None:
    suffix = "" if verified is None else f" verified={int(verified)}"
    print(
        f"NIXL_RESULT tag={tag} phase={result.phase} key={result.key} "
        f"size={result.size} elapsed_ms={result.elapsed_ms:.3f} "
        f"mbps={result.mbps:.1f}{suffix}",
        flush=True,
    )


def backend_params(args: argparse.Namespace) -> dict[str, str]:
    params = {
        "client_library": args.client_library,
        "endpoint": args.endpoint,
        "namespace": args.namespace,
    }
    if args.mode == "rdma":
        params["rdma_enabled"] = "true"
        params["rdma_server_addr"] = args.rdma_server_addr
    return params


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a ContextStore NIXL slab-eviction disk-read case."
    )
    parser.add_argument("--mode", choices=("grpc", "rdma"), default="rdma")
    parser.add_argument("--client-library", required=True)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--rdma-server-addr", default="127.0.0.1:50053")
    parser.add_argument("--namespace", default="nixl-eviction")
    parser.add_argument("--object-size", default="512m")
    parser.add_argument("--evict-count", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    import nixl._api as nixl_api

    object_size = parse_size(args.object_size)
    base_key = f"{args.namespace}-base|__combined__"
    evict_keys = [
        f"{args.namespace}-evict-{index}|__combined__"
        for index in range(1, args.evict_count + 1)
    ]

    config = nixl_api.nixl_agent_config(backends=[])
    agent = nixl_api.nixl_agent("contextstore-nixl-eviction", config)
    if "CONTEXTSTORE" not in agent.get_plugin_list():
        raise RuntimeError(f"CONTEXTSTORE plugin not found: {agent.get_plugin_list()}")
    agent.create_backend("CONTEXTSTORE", backend_params(args))

    write_results: list[TransferResult] = []
    with AlignedBuffer(object_size) as buffer:
        base_byte = object_dev_id(base_key) & 0xFF
        phase("base_put_start")
        buffer.fill_byte(base_byte)
        base_put = put_object(agent, base_key, buffer, args.timeout_s)
        write_results.append(base_put)
        print_result(base_put, "base_put")
        phase("base_put_end")

        for index, key in enumerate(evict_keys, start=1):
            evict_byte = object_dev_id(key) & 0xFF
            phase(f"evict_{index}_put_start")
            buffer.fill_byte(evict_byte)
            result = put_object(agent, key, buffer, args.timeout_s)
            write_results.append(result)
            print_result(result, f"evict_{index}_put")
            phase(f"evict_{index}_put_end")

        phase("reload_get_start")
        buffer.zero()
        reload_get = get_object(agent, base_key, buffer, args.timeout_s)
        verified = buffer.sampled_byte_matches(base_byte)
        print_result(reload_get, "reload_get", verified=verified)
        phase("reload_get_end")

    write_median_ms = statistics.median(result.elapsed_ms for result in write_results)
    write_median_mbps = statistics.median(result.mbps for result in write_results)
    print(
        f"NIXL_SUMMARY object_size={object_size} evict_count={args.evict_count} "
        f"write_median_ms={write_median_ms:.3f} write_median_mbps={write_median_mbps:.1f} "
        f"reload_ms={reload_get.elapsed_ms:.3f} reload_mbps={reload_get.mbps:.1f} "
        f"verified={int(verified)} base_key={base_key}",
        flush=True,
    )
    return 0 if verified else 2


if __name__ == "__main__":
    sys.exit(main())
