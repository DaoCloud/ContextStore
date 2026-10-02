from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(os.environ.get("CS_GUEST_ROOT", "/home/railtest"))
BACKUP = ROOT / "evidence/softroce-stripe1.backup"
MANIFEST = ROOT / "evidence/softroce-stripe1-corruption.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stripe_path(namespace: str, key: str) -> Path:
    output = subprocess.check_output(
        [
            str(ROOT / "bin/cs-meta"),
            "--config",
            str(ROOT / "evidence/server.toml"),
            "--json",
            "get",
            "--namespace",
            namespace,
            "--object-key",
            key,
        ],
        text=True,
    )
    record = json.loads(output)
    path = Path(record["metadata"]["striping"]["chunk_paths"][1]).resolve()
    if not path.is_relative_to(ROOT / "data") or not path.name.endswith(".chunk1.bin"):
        raise ValueError(f"unexpected test stripe path: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["inject", "restore"])
    parser.add_argument("--namespace", default="rust-bench")
    parser.add_argument("--object-key", default="rxetest0/__combined__")
    args = parser.parse_args()
    target = stripe_path(args.namespace, args.object_key)
    if args.phase == "inject":
        if BACKUP.exists():
            raise RuntimeError("backup exists; restore before another injection")
        shutil.copy2(target, BACKUP)
        original = sha256(target)
        with target.open("r+b") as handle:
            first = handle.read(1)
            if not first:
                raise RuntimeError("target stripe is empty")
            handle.seek(0)
            handle.write(bytes([first[0] ^ 0xFF]))
            handle.flush()
            os.fsync(handle.fileno())
        corrupted = sha256(target)
        MANIFEST.write_text(
            json.dumps(
                {
                    "namespace": args.namespace,
                    "object_key": args.object_key,
                    "stripe_index": 1,
                    "path": str(target),
                    "original_sha256": original,
                    "corrupted_sha256": corrupted,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"injected stripe=1 original={original} corrupted={corrupted}")
    else:
        if not BACKUP.exists():
            raise RuntimeError("backup is missing")
        shutil.copy2(BACKUP, target)
        expected = json.loads(MANIFEST.read_text())["original_sha256"]
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"restore mismatch: {actual} != {expected}")
        BACKUP.unlink()
        print(f"restored stripe=1 sha256={actual}")


if __name__ == "__main__":
    main()
