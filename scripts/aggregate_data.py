"""Deterministically aggregate benign candump files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAME = "aggregate_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(output_file: Path, overwrite: bool) -> None:
    output_dir = output_file.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return

    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite"
        )
    allowed = {output_file.name, MANIFEST_NAME}
    unexpected = sorted(path.name for path in existing if path.name not in allowed)
    if unexpected:
        raise FileExistsError(
            "Refusing to overwrite a directory with unrelated entries: "
            + ", ".join(unexpected)
        )
    for path in existing:
        path.unlink()


def aggregate_files(
    source_dir: Path, output_file: Path, overwrite: bool = False
) -> dict:
    """Aggregate sorted source files and return a relative-path manifest."""
    source_dir = source_dir.resolve()
    output_file = output_file.resolve()
    if output_file == source_dir or source_dir in output_file.parents:
        raise ValueError(
            "Aggregate output must be entirely outside the source tree"
        )
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    files = sorted(
        source_dir.rglob("*.log"),
        key=lambda path: path.relative_to(source_dir).as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"No .log files found in {source_dir}")
    _prepare_output(output_file, overwrite)
    entries = []
    with output_file.open("wb") as output:
        for source in files:
            digest = hashlib.sha256()
            source_bytes = 0
            last_byte = b""
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    source_bytes += len(chunk)
                    last_byte = chunk[-1:]
            if source_bytes and last_byte != b"\n":
                output.write(b"\n")
            entries.append(
                {
                    "source_relative_path": source.relative_to(source_dir).as_posix(),
                    "source_sha256": digest.hexdigest(),
                    "source_bytes": source_bytes,
                }
            )

    manifest = {
        "schema_version": 1,
        "source_count": len(entries),
        "sources": entries,
        "output_file": output_file.name,
        "output_sha256": _sha256(output_file),
    }
    manifest_path = output_file.parent / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset/CAN-MIRGU(train)/Benign"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/HCRL_dataset/train_aggregated.log"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = aggregate_files(args.source_dir, args.output_file, args.overwrite)
    print(
        f"[INFO] aggregated {manifest['source_count']} files -> {args.output_file}"
    )


if __name__ == "__main__":
    main()
