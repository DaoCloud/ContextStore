from __future__ import annotations

"""ContextStore KV Service — performance benchmark

Usage:
  python benchmarks/run_benchmark.py --endpoint localhost:50051
"""

import argparse
import os
import time

from contextstore.kvservice_client import KVClient, KVMetadata, ObjectKey


def bench_put(client: KVClient, n_layers: int, layer_size_mb: int):
    data = os.urandom(layer_size_mb * 1024 * 1024)
    meta = KVMetadata(num_tokens=32768, num_layers=n_layers, dtype="bfloat16")

    t0 = time.perf_counter()
    items = [
        (ObjectKey("bench-model", f"benchprefix/layer_{i}"), data, meta)
        for i in range(n_layers)
    ]
    success = client.put_batch(items)
    elapsed = time.perf_counter() - t0
    total_mb = n_layers * layer_size_mb
    print(
        f"[PUT BATCH] {n_layers} layers × {layer_size_mb}MB = {total_mb}MB "
        f"in {elapsed*1000:.1f}ms  → {total_mb/elapsed/1024:.2f} GB/s"
    )
    assert all(success), "some puts failed"


def bench_get(client: KVClient, n_layers: int, layer_size_mb: int):
    keys = [
        ObjectKey("bench-model", f"benchprefix/layer_{i}")
        for i in range(n_layers)
    ]
    # L1 hit
    t0 = time.perf_counter()
    results = client.get_batch(keys)
    elapsed = time.perf_counter() - t0
    total_mb = sum(len(r[0]) for r in results if r is not None) / 1024 / 1024
    print(
        f"[GET BATCH L1] {n_layers} layers → {total_mb:.0f}MB in {elapsed*1000:.1f}ms "
        f"→ {total_mb/elapsed/1024:.2f} GB/s"
    )


def cleanup(client: KVClient, n_layers: int):
    for i in range(n_layers):
        client.delete(ObjectKey("bench-model", f"benchprefix/layer_{i}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="localhost:50051")
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--layer-size-mb", type=int, default=6)
    args = parser.parse_args()

    print(f"==> Connecting to {args.endpoint}")
    with KVClient(args.endpoint, max_message_mb=2048) as client:
        print(f"==> Health: {client.health()}")
        print(f"==> Benchmark: {args.layers} layers × {args.layer_size_mb}MB")
        try:
            bench_put(client, args.layers, args.layer_size_mb)
            bench_get(client, args.layers, args.layer_size_mb)
            bench_get(client, args.layers, args.layer_size_mb)  # Second L1 hit
        finally:
            cleanup(client, args.layers)


if __name__ == "__main__":
    main()
