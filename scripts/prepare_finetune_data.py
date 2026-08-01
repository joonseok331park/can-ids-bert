"""Create deterministic file-level train, validation, and test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from uuid import uuid4

from core.classes import CLASS_NAMES


SPLIT_NAMES = ("train", "validation", "test")
MANIFEST_NAME = "split_manifest.json"
MANIFEST_SCHEMA_VERSION = 2


def _sorted_logs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        directory.rglob("*.log"),
        key=lambda path: path.relative_to(directory).as_posix().casefold(),
    )


def classify_source_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    """Classify source files and reject ambiguous real-attack filenames.

    Benign files come from ``Benign``. Real-attack filenames must contain
    ``dos``, ``fuzz``, or ``malfunction``. Files under ``Masquerade_attacks``
    and ``Suspension_attacks`` are explicitly mapped to Malfunction.
    """
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    classified: Dict[str, List[Path]] = {name: [] for name in CLASS_NAMES}
    classified["Benign"].extend(_sorted_logs(dataset_dir / "Benign"))

    unclassified: list[Path] = []
    for path in _sorted_logs(dataset_dir / "Attack" / "Real_attacks"):
        filename = path.name.casefold()
        matches = []
        if "dos" in filename:
            matches.append("DoS")
        if "fuzz" in filename:
            matches.append("Fuzzy")
        if "malfunction" in filename:
            matches.append("Malfunction")
        if len(matches) != 1:
            unclassified.append(path)
        else:
            classified[matches[0]].append(path)

    for directory_name in ("Masquerade_attacks", "Suspension_attacks"):
        classified["Malfunction"].extend(
            _sorted_logs(dataset_dir / "Attack" / directory_name)
        )

    if unclassified:
        relative = [
            path.relative_to(dataset_dir).as_posix() for path in unclassified
        ]
        raise ValueError(
            "Unclassified attack files; rename or extend the documented rule: "
            + ", ".join(relative)
        )

    for class_name in CLASS_NAMES:
        classified[class_name] = sorted(
            classified[class_name],
            key=lambda path: path.relative_to(dataset_dir).as_posix().casefold(),
        )
    return classified


def split_files(
    files: Iterable[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split at file level while keeping every split non-empty."""
    files = sorted((Path(path) for path in files), key=lambda path: path.as_posix())
    if len(files) < 3:
        raise ValueError(
            f"At least 3 source files per class are required; found {len(files)}"
        )
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and leave a positive test ratio")

    shuffled = files.copy()
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * train_ratio))
    validation_count = max(1, int(len(shuffled) * val_ratio))
    while len(shuffled) - train_count - validation_count < 1:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("Unable to keep every split non-empty")

    train = shuffled[:train_count]
    validation = shuffled[train_count : train_count + validation_count]
    test = shuffled[train_count + validation_count :]
    split_sets = [set(group) for group in (train, validation, test)]
    if any(
        split_sets[left] & split_sets[right]
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    ):
        raise AssertionError("A source file appears in more than one split")
    return train, validation, test


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_target(entry: Mapping[str, Any]) -> PurePosixPath:
    class_name = entry.get("class")
    split_name = entry.get("split")
    raw_target = entry.get("target_relative_path")
    if class_name not in CLASS_NAMES or split_name not in SPLIT_NAMES:
        raise ValueError("Split manifest contains an invalid class or split")
    if not isinstance(raw_target, str):
        raise ValueError("Split manifest target_relative_path must be a string")

    target = PurePosixPath(raw_target)
    if (
        target.is_absolute()
        or ".." in target.parts
        or target.as_posix() != raw_target
        or len(target.parts) != 2
        or target.parts[0] != split_name
        or target.suffix.casefold() != ".log"
        or not target.name.casefold().startswith(f"{class_name}_".casefold())
    ):
        raise ValueError(f"Unsafe split manifest target: {raw_target!r}")
    return target


def _validate_manifest_owned_tree(output_dir: Path) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: a valid {MANIFEST_NAME} is required"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: invalid {MANIFEST_NAME}: {error}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: manifest root must be an object"
        )
    if manifest.get("schema_version") not in {1, MANIFEST_SCHEMA_VERSION}:
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: unsupported manifest schema"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: manifest entries are missing"
        )

    expected_files = {MANIFEST_NAME}
    expected_dirs: set[str] = set()
    try:
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Split manifest entry must be an object")
            target = _manifest_target(entry)
            target_name = target.as_posix()
            if target_name in expected_files:
                raise ValueError(f"Duplicate split manifest target: {target_name}")
            expected_files.add(target_name)
            parent = target.parent
            while parent != PurePosixPath("."):
                expected_dirs.add(parent.as_posix())
                parent = parent.parent
    except ValueError as error:
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: invalid {MANIFEST_NAME}: {error}"
        ) from error

    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in output_dir.rglob("*"):
        relative = path.relative_to(output_dir).as_posix()
        if path.is_dir() and not path.is_symlink():
            actual_dirs.add(relative)
        else:
            actual_files.add(relative)

    unknown_files = sorted(actual_files - expected_files)
    missing_files = sorted(expected_files - actual_files)
    unknown_dirs = sorted(actual_dirs - expected_dirs)
    if unknown_files or missing_files or unknown_dirs:
        details = []
        if unknown_files:
            details.append("unknown files=" + ", ".join(unknown_files))
        if missing_files:
            details.append("missing files=" + ", ".join(missing_files))
        if unknown_dirs:
            details.append("unknown directories=" + ", ".join(unknown_dirs))
        raise FileExistsError(
            f"Cannot overwrite {output_dir}: existing tree is not manifest-owned ("
            + "; ".join(details)
            + ")"
        )


def _validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite"
        )
    if existing:
        _validate_manifest_owned_tree(output_dir)


def _sibling_work_dir(output_dir: Path, label: str) -> Path:
    while True:
        candidate = output_dir.with_name(
            f".{output_dir.name}.{label}-{uuid4().hex}"
        )
        if not candidate.exists():
            return candidate


def _replace_with_staging(output_dir: Path, staging_dir: Path) -> None:
    if not output_dir.exists():
        staging_dir.rename(output_dir)
        return

    backup_dir = _sibling_work_dir(output_dir, "backup")
    output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except Exception:
        backup_dir.rename(output_dir)
        raise
    shutil.rmtree(backup_dir)


def _materialize(source: Path, target: Path, link_mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if link_mode == "symlink":
        os.symlink(source.resolve(), target)
        return "symlink"
    try:
        os.symlink(source.resolve(), target)
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def prepare_splits(
    dataset_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
    link_mode: str = "auto",
) -> dict:
    """Prepare split files and write ``split_manifest.json``."""
    if link_mode not in {"auto", "copy", "symlink"}:
        raise ValueError("link_mode must be auto, copy, or symlink")
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the source dataset")

    classified = classify_source_files(dataset_dir)
    planned: dict[str, dict[str, list[Path]]] = {}
    class_seeds: dict[str, int] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        files = classified[class_name]
        if len(files) < 3:
            raise ValueError(
                f"Class {class_name} requires at least 3 files; found {len(files)}"
            )
        effective_seed = seed + class_index
        class_seeds[class_name] = effective_seed
        groups = split_files(files, train_ratio, val_ratio, effective_seed)
        planned[class_name] = dict(zip(SPLIT_NAMES, groups))

    every_source: list[Path] = [
        source
        for class_splits in planned.values()
        for sources in class_splits.values()
        for source in sources
    ]
    if len(every_source) != len(set(every_source)):
        raise AssertionError("A source file appears in more than one split")
    for class_name, class_splits in planned.items():
        for split_name in SPLIT_NAMES:
            if not class_splits[split_name]:
                raise AssertionError(
                    f"Class {class_name} is absent from split {split_name}"
                )

    _validate_output_dir(output_dir, overwrite)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = _sibling_work_dir(output_dir, "staging")
    staging_dir.mkdir()
    try:
        entries = []
        for class_name in CLASS_NAMES:
            for split_name in SPLIT_NAMES:
                for index, source in enumerate(
                    planned[class_name][split_name], start=1
                ):
                    target_name = f"{class_name}_{index:03d}_{source.stem}.log"
                    target = staging_dir / split_name / target_name
                    materialization = _materialize(source, target, link_mode)
                    entries.append(
                        {
                            "class": class_name,
                            "split": split_name,
                            "source_relative_path": source.relative_to(
                                dataset_dir
                            ).as_posix(),
                            "target_relative_path": target.relative_to(
                                staging_dir
                            ).as_posix(),
                            "materialization": materialization,
                            "source_sha256": _sha256(source),
                        }
                    )

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "seed": seed,
            "seed_derivation": "base seed + CLASS_NAMES index",
            "class_seeds": class_seeds,
            "ratios": {
                "train": train_ratio,
                "validation": val_ratio,
                "test": 1.0 - train_ratio - val_ratio,
            },
            "classification_rules": {
                "Benign": "files under Benign",
                "DoS": "Real_attacks filename contains dos",
                "Fuzzy": "Real_attacks filename contains fuzz",
                "Malfunction": (
                    "Real_attacks filename contains malfunction, or file is under "
                    "Masquerade_attacks/Suspension_attacks"
                ),
            },
            "entries": entries,
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_with_staging(output_dir, staging_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("dataset/CAN-MIRGU(train)")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/finetune_data")
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--link-mode", choices=("auto", "copy", "symlink"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_splits(
        args.dataset_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.overwrite,
        args.link_mode,
    )
    print(f"[INFO] wrote {len(manifest['entries'])} split entries")


if __name__ == "__main__":
    main()
