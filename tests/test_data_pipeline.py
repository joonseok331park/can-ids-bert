import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.dataset import MLMDataset
from core.tokenizer import CANSequencer, CANTokenizer, SpecialToken
from scripts.build_vocab import _collect_unique_ids
from utils.data_loader import _parse, load_can_data, parse_candump_line


class CandumpParserTests(unittest.TestCase):
    def test_valid_classic_and_extended_frames(self):
        classic = _parse("(1613599955.394625) can0 0c8#0011Aaff")
        self.assertEqual(classic["CAN_ID"], "0C8")
        self.assertEqual(classic["Data"], ["00", "11", "AA", "FF"])
        self.assertEqual(classic["DLC"], 4)

        extended = _parse("(1.000000) vcan0 1ABCDEFF#")
        self.assertEqual(extended["CAN_ID"], "1ABCDEFF")
        self.assertEqual(extended["Data"], [])

    def test_malformed_frames_are_rejected(self):
        malformed = [
            "(1.000000) can0 123#0",
            "(1.000000) can0 123#000000000000000000",
            "(1.000000) can0 20000000#00",
            "(1.000000) can0 123#00 trailing",
            "not candump",
        ]
        for line in malformed:
            with self.subTest(line=line):
                self.assertIsNone(_parse(line))

    def test_file_loader_ignores_invalid_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.log"
            path.write_text(
                "(1.000000) can0 123#0011\ninvalid\n",
                encoding="utf-8",
            )
            frame = load_can_data(path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["CAN_ID"], "123")
        self.assertEqual(
            frame.attrs["parse_stats"],
            {"total_lines": 2, "valid_lines": 1, "rejected_lines": 1},
        )

    def test_lowercase_uses_the_same_canonical_parser_for_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.log"
            path.write_text(
                "(1.000000) can0 0a#00ff\n"
                "(2.000000) can0 20000000#00\n"
                "trailing garbage\n",
                encoding="utf-8",
            )
            ids, stats = _collect_unique_ids(path)
        self.assertEqual(ids, {"0A"})
        self.assertEqual(stats.valid_lines, 1)
        self.assertEqual(stats.rejected_lines, 2)
        self.assertEqual(parse_candump_line("(1.0) can0 0a#00ff")["CAN_ID"], "0A")

    def test_all_invalid_files_fail_for_loader_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.log"
            path.write_text("invalid\n(1.0) can0 20000000#00\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No valid candump frames"):
                load_can_data(path)
            with self.assertRaisesRegex(ValueError, "No valid candump frames"):
                _collect_unique_ids(path)


class TokenizerAndDatasetTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "CAN_ID": ["123", "456"],
                "Data": [["00", "AA"], ["10", "20", "30"]],
            }
        )

    def test_vocab_and_sequence_are_deterministic(self):
        tokenizer = CANTokenizer()
        tokenizer.build_vocab(self.frame)
        first = CANSequencer(tokenizer, seq_len=18).transform(self.frame)
        second = CANSequencer(tokenizer, seq_len=18).transform(self.frame)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(first[0]), 18)

    def test_vocab_orders_can_ids_numerically(self):
        frame = pd.DataFrame(
            {"CAN_ID": ["00F", "001", "00A"], "Data": [[], [], []]}
        )
        tokenizer = CANTokenizer()
        tokenizer.build_vocab(frame)
        tokens = [
            str(tokenizer.id_offset + value)
            for value in (0x001, 0x00A, 0x00F)
        ]
        assigned = [tokenizer.token_to_id[token] for token in tokens]
        self.assertEqual(assigned, sorted(assigned))

    def test_public_add_can_ids_deduplicates_and_validates(self):
        tokenizer = CANTokenizer()
        added = tokenizer.add_can_ids(["00a", "A", "001"])
        self.assertEqual(added, 2)
        with self.assertRaises(ValueError):
            tokenizer.add_can_ids(["20000000"])
        with self.assertRaises(ValueError):
            tokenizer.add_can_ids(["not-hex"])

    def test_payload_validation(self):
        tokenizer = CANTokenizer()
        sequencer = CANSequencer(tokenizer)
        with self.assertRaises(ValueError):
            sequencer._frame_to_tokens("123", ["0"])
        with self.assertRaises(ValueError):
            sequencer._frame_to_tokens("123", ["00"] * 9)

    def test_mlm_dataset_handles_only_special_tokens(self):
        tokenizer = CANTokenizer()
        sequence = [tokenizer.token_to_id[SpecialToken.PAD]] * 4
        item = MLMDataset([sequence], tokenizer)[0]
        self.assertEqual(item["input_ids"].tolist(), sequence)
        self.assertEqual(item["labels"].tolist(), [-100] * 4)


if __name__ == "__main__":
    unittest.main()
