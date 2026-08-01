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
from scripts.pretrain import (
    CHECKPOINT_SCHEMA_VERSION,
    IncompatibleCheckpointError,
    _build_grad_scaler,
    _load_checkpoint,
    _save_checkpoint,
)


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

    def _training_parts(self):
        model = CANBertForMaskedLM(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 - min(step, 10) / 10
        )
        scaler = _build_grad_scaler(torch.device("cpu"))
        return model, optimizer, scheduler, scaler

    def test_versioned_checkpoint_round_trip_restores_full_state(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        input_ids = torch.randint(0, 32, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        loss = model(input_ids, attention_mask, labels)[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        training_config = {"updates_per_epoch": 3, "total_epochs": 2}

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=3,
                out_dir=Path(tmp),
                rank=0,
                training_config=training_config,
            )
            restored_model, restored_optimizer, restored_scheduler, restored_scaler = (
                self._training_parts()
            )
            state = _load_checkpoint(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_scaler,
                path,
                torch.device("cpu"),
                rank=0,
                expected_training_config=training_config,
            )

        self.assertEqual(state.mode, "true-resume")
        self.assertEqual(state.start_epoch, 1)
        self.assertEqual(state.global_optimizer_step, 3)
        self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored_model.state_dict()[key]))
        self.assertTrue(restored_optimizer.state)
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for state_values in restored_optimizer.state.values()
                for tensor in state_values.values()
                if torch.is_tensor(tensor)
            )
        )

    def test_legacy_checkpoint_is_an_explicit_warm_start(self):
        source, _, _, _ = self._training_parts()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "epoch": 99,
                    "model": source.state_dict(),
                    "optim_state": {"legacy": True},
                },
                path,
            )
            model, optimizer, scheduler, scaler = self._training_parts()
            state = _load_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                path,
                torch.device("cpu"),
                rank=0,
            )
        self.assertEqual(state.mode, "legacy-warm-start")
        self.assertEqual(state.start_epoch, 0)
        self.assertEqual(state.global_optimizer_step, 0)

    def test_unknown_versioned_schema_is_not_silently_warm_started(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "future.pt"
            torch.save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION + 1,
                    "model_state_dict": model.state_dict(),
                },
                path,
            )
            with self.assertRaises(IncompatibleCheckpointError):
                _load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    path,
                    torch.device("cpu"),
                    rank=0,
                )

    def test_schedule_mismatch_is_rejected(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=0,
                out_dir=Path(tmp),
                rank=0,
                training_config={"updates_per_epoch": 3},
            )
            with self.assertRaisesRegex(
                IncompatibleCheckpointError, "Training schedule is incompatible"
            ):
                _load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    path,
                    torch.device("cpu"),
                    rank=0,
                    expected_training_config={"updates_per_epoch": 4},
                )

    def test_shape_compatible_architecture_mismatch_is_rejected(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=0,
                out_dir=Path(tmp),
                rank=0,
                training_config={},
            )
            payload = torch.load(path)
            payload["model_config"]["num_attention_heads"] = 4
            torch.save(payload, path)
            with self.assertRaisesRegex(
                IncompatibleCheckpointError, "Model architecture is incompatible"
            ):
                _load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    path,
                    torch.device("cpu"),
                    rank=0,
                )


if __name__ == "__main__":
    unittest.main()
