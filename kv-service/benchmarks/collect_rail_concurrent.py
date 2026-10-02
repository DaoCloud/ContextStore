from __future__ import annotations

# Measure concurrent reads within one Rust Worker/RailReader over real Verbs.
# Compare one/two rails only within the same object size and concurrency cell.

import argparse
import csv
import json
import platform
import re
import subprocess
from pathlib import Path


SAMPLE = re.compile(
    r"^sample,environment=([^,]+),rails=(\d+),concurrency=(\d+),"
    r"batch=(\d+),worker=(\d+),bytes=(\d+),latency_us=(\d+),xxh3=([0-9a-f]+)$",
    re.MULTILINE,
)
BATCH = re.compile(
    r"^batch,environment=([^,]+),rails=(\d+),concurrency=(\d+)," r"iteration=(\d+),wall_us=(\d+)$",
    re.MULTILINE,
)
SUMMARY = re.compile(
    r"^summary_concurrent,environment=([^,]+),rails=(\d+),concurrency=(\d+),"
    r"bytes_per_read=(\d+),batches=(\d+),median_request_us=(\d+),"
    r"median_batch_us=(\d+),aggregate_gib_per_s=([0-9.]+),"
    r"cpu_user_us=(\d+),cpu_system_us=(\d+),peak_rss_kb=(\d+),"
    r"rail_bytes=\[([^]]+)\]$",
    re.MULTILINE,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect concurrent real-Verbs rail reads")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--environment", required=True, choices=["physical", "soft-roce"])
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--object", action="append", required=True, help="SIZE_MIB:OBJECT_KEY")
    parser.add_argument("--rail", action="append", required=True)
    parser.add_argument("--concurrency", action="append", type=int, required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", default="rail-verbs-concurrent")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if len(args.rail) != 2 or args.trials <= 0 or args.batches <= 0:
        parser.error("two --rail entries and positive trials/batches are required")
    if any(value < 2 or value > 8 for value in args.concurrency):
        parser.error("concurrency values must be in 2..=8")
    objects = []
    for spec in args.object:
        size_text, separator, key = spec.partition(":")
        if not separator or not key or not size_text.isdigit() or int(size_text) <= 0:
            parser.error("each --object must be SIZE_MIB:OBJECT_KEY")
        objects.append((int(size_text), key))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    expected_hashes: dict[int, str] = {}
    for trial in range(1, args.trials + 1):
        for size_mib, key in objects:
            for concurrency in args.concurrency:
                for rails in (1, 2) if trial % 2 else (2, 1):
                    command = [
                        str(args.binary),
                        "--environment",
                        args.environment,
                        "--coordinator",
                        args.coordinator,
                        "--namespace",
                        args.namespace,
                        "--object-key",
                        key,
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.batches),
                        "--concurrency",
                        str(concurrency),
                    ]
                    for rail in args.rail[:rails]:
                        command.extend(["--rail", rail])
                    output = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=True,
                    ).stdout
                    sample_matches = SAMPLE.findall(output)
                    batch_matches = BATCH.findall(output)
                    summary_match = SUMMARY.search(output)
                    if (
                        len(sample_matches) != args.batches * concurrency
                        or len(batch_matches) != args.batches
                        or summary_match is None
                    ):
                        raise RuntimeError(f"incomplete concurrent output: {output[-2000:]}")
                    hashes = {row[7] for row in sample_matches}
                    if len(hashes) != 1:
                        raise RuntimeError("concurrent readers returned different objects")
                    checksum = hashes.pop()
                    if size_mib in expected_hashes and expected_hashes[size_mib] != checksum:
                        raise RuntimeError("one- and two-rail hashes disagree")
                    expected_hashes[size_mib] = checksum
                    for _, _, _, batch, worker, byte_count, latency_us, _ in sample_matches:
                        if int(byte_count) != size_mib * 1024 * 1024:
                            raise RuntimeError("unexpected object size")
                        samples.append(
                            dict(
                                trial=trial,
                                size_mib=size_mib,
                                rails=rails,
                                concurrency=concurrency,
                                batch=int(batch),
                                worker=int(worker),
                                bytes=int(byte_count),
                                latency_us=int(latency_us),
                                xxh3=checksum,
                            )
                        )
                    for _, _, _, iteration, wall_us in batch_matches:
                        batches.append(
                            dict(
                                trial=trial,
                                size_mib=size_mib,
                                rails=rails,
                                concurrency=concurrency,
                                batch=int(iteration),
                                wall_us=int(wall_us),
                            )
                        )
                    (
                        _,
                        _,
                        _,
                        _,
                        _,
                        median_request,
                        median_batch,
                        throughput,
                        cpu_user,
                        cpu_system,
                        rss,
                        rail_text,
                    ) = summary_match.groups()
                    rail_bytes = [int(value) for value in rail_text.split(",")]
                    if len(rail_bytes) != rails or sum(rail_bytes) != (
                        size_mib * 1024 * 1024 * args.batches * concurrency
                    ):
                        raise RuntimeError("per-rail byte counters disagree")
                    if any(value == 0 for value in rail_bytes):
                        raise RuntimeError("one configured rail transferred no bytes")
                    summaries.append(
                        dict(
                            trial=trial,
                            size_mib=size_mib,
                            rails=rails,
                            concurrency=concurrency,
                            batches=args.batches,
                            median_request_us=int(median_request),
                            median_batch_us=int(median_batch),
                            aggregate_gib_per_s=float(throughput),
                            cpu_user_us=int(cpu_user),
                            cpu_system_us=int(cpu_system),
                            peak_rss_kb=int(rss),
                            rail0_bytes=rail_bytes[0],
                            rail1_bytes=rail_bytes[1] if rails == 2 else 0,
                            xxh3=checksum,
                        )
                    )
                    print(
                        f"trial={trial} size={size_mib}MiB concurrency={concurrency} "
                        f"rails={rails} aggregate_gib_s={throughput}",
                        flush=True,
                    )

    write_csv(args.output_dir / f"{args.output_prefix}-summary.csv", summaries)
    write_csv(args.output_dir / f"{args.output_prefix}-samples.csv", samples)
    write_csv(args.output_dir / f"{args.output_prefix}-batches.csv", batches)
    environment = {
        "environment": args.environment,
        "source_commit": args.source_commit,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "notes": args.notes,
        "objects_mib": [size for size, _ in objects],
        "concurrency": args.concurrency,
        "trials": args.trials,
        "batches_per_trial": args.batches,
        "warmup_reads": args.warmup,
        "coordinator": args.coordinator,
        "rails": args.rail,
    }
    (args.output_dir / f"{args.output_prefix}-environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
