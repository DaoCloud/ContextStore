from __future__ import annotations

# Collect software-only rail scheduling samples from the Rust Mock test.
# Example: python kv-service/benchmarks/collect_rail_mock.py --sizes 64,256,512
# This measures memory copies and scheduling, not RDMA or disk bandwidth.

import argparse
import csv
import json
import os
import platform
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "kv-service/client-rs/Cargo.toml"
SAMPLES = re.compile(r"mock_samples_us=\[([^]]+)\]")
SUMMARY = re.compile(
    r"mock_only,rails=(\d+),size_bytes=(\d+),iters=(\d+),"
    r"avg_us=(\d+),gib_per_s=([0-9.]+),cpu_user_us=(\d+),"
    r"cpu_system_us=(\d+),peak_rss_kb=(\d+),rail_bytes=\[([^]]+)\]"
)


def read_command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect software-only rail Mock samples")
    parser.add_argument("--sizes", default="64,256,512", help="Comma-separated MiB sizes")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-prefix", default="rail-mock")
    parser.add_argument("--environment-note", default="")
    parser.add_argument(
        "--source-commit", help="Source commit when running a copied tree without .git"
    )
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or any(value <= 0 for value in sizes):
        parser.error("--sizes must contain positive MiB values")
    if args.trials <= 0 or args.iterations <= 0:
        parser.error("--trials and --iterations must be positive")

    summary_rows: list[dict[str, int | float]] = []
    sample_rows: list[dict[str, int]] = []
    for trial in range(1, args.trials + 1):
        for size in sizes:
            for rails in (1, 2):
                env = os.environ.copy()
                env.update(
                    CS_RAIL_MOCK_MIB=str(size),
                    CS_RAIL_MOCK_ITERS=str(args.iterations),
                    CS_RAIL_MOCK_RAILS=str(rails),
                )
                command = [
                    "cargo",
                    "test",
                    "--manifest-path",
                    str(MANIFEST),
                    "--release",
                    "--features",
                    "rdma",
                    "--lib",
                    "software_only_mock_benchmark",
                    "--",
                    "--ignored",
                    "--nocapture",
                ]
                output = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=True,
                ).stdout
                samples = SAMPLES.search(output)
                summary = SUMMARY.search(output)
                if samples is None or summary is None:
                    raise RuntimeError(f"Mock benchmark output lacks samples: {output[-2000:]}")
                values = [int(value) for value in samples.group(1).split(",")]
                _, size_bytes, iterations, avg_us, gib, user_us, system_us, rss_kb, rail_text = (
                    summary.groups()
                )
                rail_bytes = [int(value) for value in rail_text.split(",")]
                if len(values) != args.iterations or int(size_bytes) != size * 1024 * 1024:
                    raise RuntimeError("Mock benchmark sample count or object size disagrees")
                summary_rows.append(
                    dict(
                        trial=trial,
                        size_mib=size,
                        rails=rails,
                        iterations=int(iterations),
                        mean_read_us=int(avg_us),
                        effective_gib_s=float(gib),
                        cpu_user_us=int(user_us),
                        cpu_system_us=int(system_us),
                        peak_rss_kb=int(rss_kb),
                        rail0_bytes=rail_bytes[0],
                        rail1_bytes=rail_bytes[1] if rails == 2 else 0,
                    )
                )
                sample_rows.extend(
                    dict(trial=trial, size_mib=size, rails=rails, iteration=index, read_us=value)
                    for index, value in enumerate(values, 1)
                )
                print(f"trial={trial} size={size}MiB rails={rails} avg_us={avg_us}", flush=True)

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / f"{args.output_prefix}-summary.csv", summary_rows)
    write_csv(output_dir / f"{args.output_prefix}-samples.csv", sample_rows)
    metadata = {
        "environment": "Mock software-only; no RDMA Verbs or disk I/O",
        "environment_note": args.environment_note,
        "git_commit": args.source_commit or read_command("git", "rev-parse", "HEAD"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "rustc": read_command("rustc", "--version"),
        "sizes_mib": sizes,
        "trials": args.trials,
        "iterations": args.iterations,
        "stripe_mib": 4,
        "warmup_reads": 1,
    }
    (output_dir / f"{args.output_prefix}-environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
