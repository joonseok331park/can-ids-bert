import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from core.classification_dataset import ClassificationDataset
from core.tokenizer import CANTokenizer
from scripts.finetune import (
    calculate_class_weights,
    compute_confusion_matrix,
    compute_metrics,
    evaluate_model,
)
from utils.data_loader import load_classification_data


def frame_line(timestamp, can_id="123", payload="0011"):
    return f"({timestamp:.6f}) can0 {can_id}#{payload}\n"


def write_frames(path, count, can_id="123"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(frame_line(index + 1, can_id) for index in range(count)),
        encoding="utf-8",
    )


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


class ClassificationIntegrityTests(unittest.TestCase):
    def tokenizer(self):
        tokenizer = CANTokenizer()
        tokenizer.add_can_ids(["123", "456"])
        return tokenizer

    def test_sequences_never_cross_source_file_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            write_frames(data_dir / "Benign_b.log", 2, "123")
            write_frames(data_dir / "DoS_a.log", 2, "456")
            dataset = ClassificationDataset(data_dir, self.tokenizer(), seq_len=18)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.labels, [0, 1])
        self.assertEqual(
            [item["source_file"] for item in dataset.sequence_sources],
            ["Benign_b.log", "DoS_a.log"],
        )
        self.assertEqual(
            [(item["start_line"], item["end_line"]) for item in dataset.sequence_sources],
            [(1, 2), (1, 2)],
        )

    def test_loader_sorts_paths_and_retains_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            write_frames(data_dir / "Benign_z.log", 1, "456")
            write_frames(data_dir / "Benign_a.log", 1, "123")
            frame = load_classification_data(data_dir)
        self.assertEqual(
            frame["SourceFile"].tolist(), ["Benign_a.log", "Benign_z.log"]
        )
        self.assertEqual(frame["SourceLine"].tolist(), [1, 1])
        self.assertEqual(frame["Label"].tolist(), [0, 0])
        self.assertEqual(frame["CAN_ID"].tolist(), ["123", "456"])

    def test_loader_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No .log files"):
                load_classification_data(Path(tmp))

    def test_unclassified_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            write_frames(data_dir / "unknown.log", 2)
            with self.assertRaisesRegex(ValueError, "Unclassified file"):
                load_classification_data(data_dir)

    def test_malformed_only_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "Benign_bad.log").write_text("invalid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No valid candump frames"):
                load_classification_data(data_dir)

    def test_short_file_error_skip_and_all_skipped_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            write_frames(data_dir / "Benign_short.log", 1)
            write_frames(data_dir / "DoS_long.log", 2)
            with self.assertRaisesRegex(ValueError, "at least 2 are required"):
                ClassificationDataset(
                    data_dir,
                    self.tokenizer(),
                    seq_len=18,
                    short_file_policy="error",
                )
            skipped = ClassificationDataset(
                data_dir,
                self.tokenizer(),
                seq_len=18,
                short_file_policy="skip",
            )
            self.assertEqual(skipped.skipped_short_files, ["Benign_short.log"])
            self.assertEqual(skipped.labels, [1])

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            write_frames(data_dir / "Benign_short.log", 1)
            with self.assertRaisesRegex(ValueError, "No classification sequences"):
                ClassificationDataset(
                    data_dir,
                    self.tokenizer(),
                    seq_len=18,
                    short_file_policy="skip",
                )

    def test_missing_training_class_is_rejected(self):
        dataset = SimpleNamespace(labels=[0, 1, 2, 2])
        with self.assertRaisesRegex(ValueError, "Malfunction"):
            calculate_class_weights(dataset)

    def test_class_weights_always_have_four_entries(self):
        dataset = SimpleNamespace(labels=[0, 1, 2, 3, 3])
        weights = calculate_class_weights(dataset)
        self.assertEqual(tuple(weights.shape), (4,))

    def test_metrics_keep_fixed_names_for_sparse_labels(self):
        metrics = compute_metrics([0, 2], [0, 2])
        self.assertEqual(metrics["precision_Benign"], 1.0)
        self.assertEqual(metrics["precision_DoS"], 0.0)
        self.assertEqual(metrics["precision_Fuzzy"], 1.0)
        self.assertEqual(metrics["precision_Malfunction"], 0.0)
        matrix = compute_confusion_matrix([0, 2], [0, 0])
        self.assertEqual(matrix.shape, (4, 4))

    def test_single_class_prediction_and_empty_input(self):
        metrics = compute_metrics([0, 1, 2, 3], [0, 0, 0, 0])
        self.assertIn("f1_Malfunction", metrics)
        with self.assertRaisesRegex(ValueError, "empty target"):
            compute_metrics([], [])

    def test_empty_evaluation_loader_is_rejected(self):
        loader = DataLoader(EmptyDataset(), batch_size=2)
        with self.assertRaisesRegex(ValueError, "dataloader is empty"):
            evaluate_model(torch.nn.Identity(), loader, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
