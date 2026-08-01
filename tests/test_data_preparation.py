import json
import tempfile
import unittest
from pathlib import Path

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

            repeated = prepare_splits(
                dataset,
                output,
                seed=7,
                overwrite=True,
                link_mode="copy",
            )
            self.assertEqual(repeated, manifest)

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
            with self.assertRaisesRegex(FileExistsError, "unrelated entries"):
                prepare_splits(
                    dataset, output, overwrite=True, link_mode="copy"
                )
            self.assertEqual(
                (output / "keep.txt").read_text(encoding="utf-8"), "user file"
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
