import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.aggregate_data import aggregate_files
from scripts.prepare_finetune_data import (
    classify_source_files,
    prepare_splits,
    split_files,
)


def write_log(path, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"(1.000000) can0 123#{marker:02X}\n", encoding="utf-8"
    )


def build_dataset(root, count=3):
    for index in range(count):
        write_log(root / "Benign" / f"Day_1/benign_{index}.log", index)
        write_log(
            root / "Attack/Real_attacks" / f"dos_{index}.log", 16 + index
        )
        write_log(
            root / "Attack/Real_attacks" / f"fuzz_{index}.log", 32 + index
        )
        write_log(
            root / "Attack/Real_attacks" / f"malfunction_{index}.log",
            48 + index,
        )


class DataPreparationTests(unittest.TestCase):
    def test_aggregate_is_sorted_and_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            write_log(source / "z.log", 2)
            write_log(source / "a.log", 1)
            output = root / "output/train.log"
            manifest = aggregate_files(source, output)
            self.assertEqual(
                [entry["source_relative_path"] for entry in manifest["sources"]],
                ["a.log", "z.log"],
            )
            self.assertEqual(output.read_text(encoding="utf-8").count("\n"), 2)
            with self.assertRaises(FileExistsError):
                aggregate_files(source, output)
            second = aggregate_files(source, output, overwrite=True)
            self.assertEqual(second["output_sha256"], manifest["output_sha256"])
            on_disk = json.loads(
                (output.parent / "aggregate_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(on_disk, second)

    def test_aggregate_rejects_resolved_output_inside_source_before_mutation(self):
        for label in ("equal", "nested", "normalized"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source"
                write_log(source / "input.log", 1)
                before = (source / "input.log").read_bytes()
                if label == "equal":
                    output = source
                elif label == "nested":
                    output = source / "generated" / "aggregate.log"
                    output.parent.mkdir()
                    output.write_text("must survive\n", encoding="utf-8")
                else:
                    (source / "generated").mkdir()
                    output = source / "generated" / ".." / "aggregate.log"

                with self.assertRaisesRegex(ValueError, "outside the source tree"):
                    aggregate_files(source, output, overwrite=True)

                self.assertEqual((source / "input.log").read_bytes(), before)
                if label == "nested":
                    self.assertEqual(
                        output.read_text(encoding="utf-8"), "must survive\n"
                    )
                elif label == "normalized":
                    self.assertFalse(output.resolve().exists())

    def test_three_files_per_class_produces_one_file_in_every_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "prepared"
            build_dataset(dataset, count=3)
            manifest = prepare_splits(
                dataset, output, seed=7, link_mode="copy"
            )

            self.assertEqual(len(manifest["entries"]), 12)
            observed = {
                (entry["class"], entry["split"])
                for entry in manifest["entries"]
            }
            expected = {
                (class_name, split)
                for class_name in ("Benign", "DoS", "Fuzzy", "Malfunction")
                for split in ("train", "validation", "test")
            }
            self.assertEqual(observed, expected)
            sources = [
                entry["source_relative_path"] for entry in manifest["entries"]
            ]
            self.assertEqual(len(sources), len(set(sources)))
            self.assertTrue(
                all(len(entry["source_sha256"]) == 64 for entry in manifest["entries"])
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["class_seeds"],
                {"Benign": 7, "DoS": 8, "Fuzzy": 9, "Malfunction": 10},
            )
            self.assertEqual(
                manifest["seed_derivation"], "base seed + CLASS_NAMES index"
            )

            repeated = prepare_splits(
                dataset,
                output,
                seed=7,
                overwrite=True,
                link_mode="copy",
            )
            self.assertEqual(repeated, manifest)

    def test_prepare_splits_rejects_overlap_in_both_directions_before_mutation(self):
        for label in ("equal", "output-child", "source-child"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if label == "source-child":
                    output = root / "container"
                    dataset = output / "dataset"
                else:
                    dataset = root / "dataset"
                    output = (
                        dataset
                        if label == "equal"
                        else dataset / "generated" / ".." / "prepared"
                    )
                build_dataset(dataset)
                before = {
                    path.relative_to(dataset).as_posix(): path.read_bytes()
                    for path in dataset.rglob("*")
                    if path.is_file()
                }

                with self.assertRaisesRegex(ValueError, "must be disjoint"):
                    prepare_splits(
                        dataset,
                        output,
                        overwrite=True,
                        link_mode="copy",
                    )

                after = {
                    path.relative_to(dataset).as_posix(): path.read_bytes()
                    for path in dataset.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(after, before)
                self.assertFalse((output / "split_manifest.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "prepared"
            build_dataset(dataset)
            manifest = prepare_splits(dataset, output, link_mode="copy")
            self.assertEqual(len(manifest["entries"]), 12)
            self.assertTrue((output / "split_manifest.json").is_file())

    def test_nonempty_output_requires_overwrite_and_preserves_unknown_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "prepared"
            build_dataset(dataset)
            output.mkdir()
            (output / "keep.txt").write_text("user file", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_splits(dataset, output, link_mode="copy")
            with self.assertRaisesRegex(FileExistsError, "valid split_manifest"):
                prepare_splits(
                    dataset, output, overwrite=True, link_mode="copy"
                )
            self.assertEqual(
                (output / "keep.txt").read_text(encoding="utf-8"), "user file"
            )

    def test_overwrite_rejects_nested_unknown_and_missing_manifest_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "prepared"
            build_dataset(dataset)
            manifest = prepare_splits(dataset, output, link_mode="copy")

            keep = output / "train" / "nested" / "keep.txt"
            keep.parent.mkdir()
            keep.write_text("user file", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not manifest-owned"):
                prepare_splits(
                    dataset, output, overwrite=True, link_mode="copy"
                )
            self.assertEqual(keep.read_text(encoding="utf-8"), "user file")

            keep.unlink()
            keep.parent.rmdir()
            missing = output / manifest["entries"][0]["target_relative_path"]
            missing.unlink()
            with self.assertRaisesRegex(FileExistsError, "missing files"):
                prepare_splits(
                    dataset, output, overwrite=True, link_mode="copy"
                )
            self.assertFalse(missing.exists())

    def test_generation_failure_preserves_existing_output_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "prepared"
            build_dataset(dataset)
            prepare_splits(dataset, output, link_mode="copy")
            before = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }

            with patch(
                "scripts.prepare_finetune_data._materialize",
                side_effect=RuntimeError("simulated generation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated generation"):
                    prepare_splits(
                        dataset, output, overwrite=True, link_mode="copy"
                    )

            after = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertEqual(
                list(root.glob(".prepared.staging-*")), [], "staging tree leaked"
            )

    def test_small_class_and_unknown_attack_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output = root / "prepared"
            build_dataset(dataset, count=2)
            with self.assertRaisesRegex(ValueError, "at least 3 files"):
                prepare_splits(dataset, output, link_mode="copy")
            self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            build_dataset(dataset)
            write_log(dataset / "Attack/Real_attacks/mystery.log", 99)
            with self.assertRaisesRegex(ValueError, "Unclassified attack files"):
                classify_source_files(dataset)

    def test_split_files_rejects_fewer_than_three(self):
        with self.assertRaisesRegex(ValueError, "At least 3"):
            split_files([Path("a"), Path("b")])


if __name__ == "__main__":
    unittest.main()
