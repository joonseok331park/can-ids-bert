import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig

from models.teacher import CANBertForMaskedLM
from models.teacher_classifier import CANBertForClassification
from scripts.finetune import (
    _finetune_checkpoint_payload,
    _load_pretrained_bert,
)
from scripts.pretrain import _unwrap_model


def tiny_config():
    return BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )


class CheckpointTests(unittest.TestCase):
    def test_compiled_model_unwrap_has_portable_keys(self):
        model = CANBertForMaskedLM(tiny_config())
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is not available")
        compiled = torch.compile(model)
        self.assertIs(_unwrap_model(compiled), model)
        self.assertTrue(_unwrap_model(compiled).state_dict())
        self.assertFalse(
            any(key.startswith("_orig_mod.") for key in _unwrap_model(compiled).state_dict())
        )

    def test_complete_teacher_body_loads_with_wrapper_prefixes(self):
        teacher = CANBertForMaskedLM(tiny_config())
        classifier = CANBertForClassification(tiny_config())
        wrapped_state = {
            f"module._orig_mod.{key}": value
            for key, value in teacher.state_dict().items()
        }
        loaded = _load_pretrained_bert(classifier, wrapped_state)
        self.assertEqual(loaded, len(classifier.bert.state_dict()))

    def test_incomplete_teacher_body_is_rejected(self):
        classifier = CANBertForClassification(tiny_config())
        with self.assertRaises(RuntimeError):
            _load_pretrained_bert(classifier, {"cls.bias": torch.zeros(32)})

    def test_finetune_checkpoint_uses_plain_config_data(self):
        config = tiny_config()
        model = CANBertForClassification(config)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = _finetune_checkpoint_payload(model, optimizer, 2, 0.5, config)
        self.assertIsInstance(payload["config"], dict)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(payload, path)
            restored = torch.load(path)
        self.assertEqual(restored["epoch"], 2)
        self.assertEqual(restored["config"]["hidden_size"], 16)


if __name__ == "__main__":
    unittest.main()
