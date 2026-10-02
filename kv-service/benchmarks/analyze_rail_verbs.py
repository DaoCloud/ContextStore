from __future__ import annotations

# Summarize checked-in real-Verbs rail samples without inferential claims.

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def percentile_nearest_rank(values: list[int], percent: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]


def summarize_pair(results: Path, prefix: str) -> list[dict[str, object]]:
    runs = read_csv(results / f"{prefix}-summary.csv")
    samples = read_csv(results / f"{prefix}-samples.csv")
    paired = []
    sizes = sorted({int(run["size_mib"]) for run in runs})
    concurrent = "concurrency" in runs[0]
    for size in sizes:
        levels = (
            sorted({int(run["concurrency"]) for run in runs if int(run["size_mib"]) == size})
            if concurrent
            else [1]
        )
        for concurrency in levels:
            results_by_rail = {}
            for rails in (1, 2):
                selected = [
                    run
                    for run in runs
                    if int(run["size_mib"]) == size
                    and int(run["rails"]) == rails
                    and int(run.get("concurrency", 1)) == concurrency
                ]
                sampled = [
                    int(row["latency_us"])
                    for row in samples
                    if int(row["size_mib"]) == size
                    and int(row["rails"]) == rails
                    and int(row.get("concurrency", 1)) == concurrency
                ]
                if not selected or not sampled:
                    raise ValueError(f"missing size={size} concurrency={concurrency} rail={rails}")
                hashes = {run["xxh3"] for run in selected}
                if len(hashes) != 1:
                    raise ValueError("object hash changed across runs")
                if rails == 2 and not all(
                    int(run["rail0_bytes"]) > 0 and int(run["rail1_bytes"]) > 0 for run in selected
                ):
                    raise ValueError("one rail transferred no payload")
                median_us = statistics.median(
                    int(run["median_request_us" if concurrent else "median_us"]) for run in selected
                )
                cpu_per_read_ms = statistics.median(
                    (int(run["cpu_user_us"]) + int(run["cpu_system_us"]))
                    / (int(run["batches" if concurrent else "iterations"]) * concurrency)
                    / 1000
                    for run in selected
                )
                throughput = (
                    statistics.median(float(run["aggregate_gib_per_s"]) for run in selected)
                    if concurrent
                    else (size / 1024) / (median_us / 1_000_000)
                )
                results_by_rail[rails] = dict(
                    rails=rails,
                    median_request_ms=median_us / 1000,
                    p95_request_ms=percentile_nearest_rank(sampled, 0.95) / 1000,
                    aggregate_gib_per_s=throughput,
                    cpu_ms_per_read=cpu_per_read_ms,
                    peak_rss_mib=statistics.median(
                        int(run["peak_rss_kb"]) / 1024 for run in selected
                    ),
                    object_xxh3=next(iter(hashes)),
                    runs=len(selected),
                    raw_request_samples=len(sampled),
                )
            if results_by_rail[1]["object_xxh3"] != results_by_rail[2]["object_xxh3"]:
                raise ValueError("single- and dual-rail payload hashes differ")
            speedup = (
                results_by_rail[2]["aggregate_gib_per_s"]
                / results_by_rail[1]["aggregate_gib_per_s"]
            )
            paired.append(
                dict(
                    size_mib=size,
                    concurrency=concurrency,
                    single=results_by_rail[1],
                    dual=results_by_rail[2],
                    aggregate_throughput_ratio=speedup,
                    two_rail_scaling_efficiency=speedup / 2,
                )
            )
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real-Verbs rail samples")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = {
        "interpretation": "Descriptive paired measurements; no statistical confidence interval or HCA offload claim",
        "groups": {prefix: summarize_pair(args.results, prefix) for prefix in args.prefix},
    }
    rendered = json.dumps(analysis, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
