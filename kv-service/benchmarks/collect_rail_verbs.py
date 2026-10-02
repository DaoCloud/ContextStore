from __future__ import annotations

# Capture paired single/dual-rail real Verbs reads without changing object layout.
# The environment label must say whether the devices are physical HCA or RXE.

import argparse
import csv
import json
import platform
import re
import subprocess
from pathlib import Path


SAMPLE = re.compile(
    r"^sample,environment=([^,]+),rails=(\d+),iteration=(\d+),"
    r"bytes=(\d+),latency_us=(\d+),gib_per_s=([0-9.]+),xxh3=([0-9a-f]+)$",
    re.MULTILINE,
)
SUMMARY = re.compile(
    r"^summary,environment=([^,]+),rails=(\d+),bytes_per_iter=(\d+),"
    r"iters=(\d+),median_us=(\d+),cpu_user_us=(\d+),cpu_system_us=(\d+),"
    r"peak_rss_kb=(\d+),rail_bytes=\[([^]]+)\]$",
    re.MULTILINE,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paired real-Verbs rail reads")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--environment", required=True, choices=["physical", "soft-roce"])
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--object", action="append", required=True, help="SIZE_MIB:OBJECT_KEY")
    parser.add_argument("--rail", action="append", required=True, help="SDK rail specification")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", default="rail-verbs")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if len(args.rail) != 2 or args.trials <= 0 or args.iterations <= 0:
        parser.error("two --rail entries and positive trials/iterations are required")
    objects = []
    for spec in args.object:
        size_text, separator, key = spec.partition(":")
        if not separator or not key or not size_text.isdigit() or int(size_text) <= 0:
            parser.error("each --object must be SIZE_MIB:OBJECT_KEY")
        objects.append((int(size_text), key))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    expected_hashes: dict[int, str] = {}
    for trial in range(1, args.trials + 1):
        for size_mib, key in objects:
            # Alternate order to reduce a systematic warm-cache advantage.
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
                    str(args.iterations),
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
                summary_match = SUMMARY.search(output)
                if len(sample_matches) != args.iterations or summary_match is None:
                    raise RuntimeError(f"incomplete benchmark output: {output[-2000:]}")
                run_hashes = {row[6] for row in sample_matches}
                if len(run_hashes) != 1:
                    raise RuntimeError("object hash changed within one benchmark run")
                checksum = run_hashes.pop()
                if size_mib in expected_hashes and expected_hashes[size_mib] != checksum:
                    raise RuntimeError("single- and dual-rail hashes disagree")
                expected_hashes[size_mib] = checksum
                for _, _, iteration, byte_count, latency_us, gib_per_s, _ in sample_matches:
                    if int(byte_count) != size_mib * 1024 * 1024:
                        raise RuntimeError("benchmark returned an unexpected object size")
                    samples.append(
                        dict(
                            trial=trial,
                            size_mib=size_mib,
                            rails=rails,
                            iteration=int(iteration),
                            bytes=int(byte_count),
                            latency_us=int(latency_us),
                            gib_per_s=float(gib_per_s),
                            xxh3=checksum,
                        )
                    )
                _, _, _, _, median_us, cpu_user, cpu_system, rss, rail_text = summary_match.groups()
                rail_bytes = [int(value) for value in rail_text.split(",")]
                if (
                    len(rail_bytes) != rails
                    or sum(rail_bytes) != size_mib * 1024 * 1024 * args.iterations
                ):
                    raise RuntimeError("per-rail counters disagree with total completed bytes")
                if any(value == 0 for value in rail_bytes):
                    raise RuntimeError("one configured rail transferred no object bytes")
                summaries.append(
                    dict(
                        trial=trial,
                        size_mib=size_mib,
                        rails=rails,
                        iterations=args.iterations,
                        median_us=int(median_us),
                        cpu_user_us=int(cpu_user),
                        cpu_system_us=int(cpu_system),
                        peak_rss_kb=int(rss),
                        rail0_bytes=rail_bytes[0],
                        rail1_bytes=rail_bytes[1] if rails == 2 else 0,
                        xxh3=checksum,
                    )
                )
                print(
                    f"trial={trial} size={size_mib}MiB rails={rails} median_us={median_us}",
                    flush=True,
                )

    write_csv(args.output_dir / f"{args.output_prefix}-summary.csv", summaries)
    write_csv(args.output_dir / f"{args.output_prefix}-samples.csv", samples)
    environment = {
        "environment": args.environment,
        "source_commit": args.source_commit,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "notes": args.notes,
        "objects_mib": [size for size, _ in objects],
        "trials": args.trials,
        "iterations_per_trial": args.iterations,
        "warmup_reads": args.warmup,
        "coordinator": args.coordinator,
        "rails": args.rail,
    }
    (args.output_dir / f"{args.output_prefix}-environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
