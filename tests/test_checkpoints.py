import hashlib
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig

from models.teacher import CANBertForMaskedLM
from models.teacher_classifier import CANBertForClassification
from scripts.finetune import (
    IncompatiblePretrainedCheckpointError,
    _finetune_checkpoint_payload,
    _load_finetune_pretrained_checkpoint,
    _load_pretrained_bert,
)
from scripts.pretrain import _unwrap_model
from scripts.pretrain import (
    CHECKPOINT_SCHEMA_VERSION,
    IncompatibleCheckpointError,
    _build_grad_scaler,
    _load_checkpoint,
    _save_checkpoint,
    _training_config,
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


def full_training_config(**overrides):
    config = {
        "microbatches_per_epoch": 2,
        "gradient_accumulation_steps": 2,
        "updates_per_epoch": 1,
        "total_epochs": 2,
        "total_optimizer_steps": 2,
        "warmup_steps": 0,
        "learning_rate": 0.001,
        "mask_prob": 0.45,
        "batch_size": 2,
        "seed": 42,
        "world_size": 1,
        "amp_enabled": False,
        "seq_len": 16,
        "dataset_type": "candump",
        "vocab_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
    }
    config.update(overrides)
    return config


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

    def _pretraining_payload(self, vocab_sha256, config=None):
        config = config or tiny_config()
        teacher = CANBertForMaskedLM(config)
        model_config = config.to_dict()
        model_config.setdefault(
            "position_embedding_type",
            getattr(config, "position_embedding_type", "absolute"),
        )
        return teacher, {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "can-bert-pretrain",
            "model_state_dict": teacher.state_dict(),
            "model_config": model_config,
            "training_config": {"vocab_sha256": vocab_sha256},
        }

    def test_matching_versioned_pretraining_checkpoint_loads_for_finetuning(self):
        vocab_bytes = b'{"tokens": ["A", "B"]}\n'
        vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
        teacher, payload = self._pretraining_payload(vocab_sha256)
        classifier = CANBertForClassification(tiny_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab_path = root / "vocab.json"
            checkpoint_path = root / "pretrained.pt"
            vocab_path.write_bytes(vocab_bytes)
            torch.save(payload, checkpoint_path)
            loaded = _load_finetune_pretrained_checkpoint(
                classifier, checkpoint_path, vocab_path
            )

        self.assertEqual(loaded, len(classifier.bert.state_dict()))
        for key, value in teacher.bert.state_dict().items():
            self.assertTrue(torch.equal(value, classifier.bert.state_dict()[key]))

    def test_finetuning_rejects_same_size_vocabulary_with_different_hash(self):
        checkpoint_vocab = b'{"token": "A"}\n'
        current_vocab = b'{"token": "B"}\n'
        self.assertEqual(len(checkpoint_vocab), len(current_vocab))
        _, payload = self._pretraining_payload(
            hashlib.sha256(checkpoint_vocab).hexdigest()
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab_path = root / "vocab.json"
            checkpoint_path = root / "pretrained.pt"
            vocab_path.write_bytes(current_vocab)
            torch.save(payload, checkpoint_path)
            with self.assertRaisesRegex(
                IncompatiblePretrainedCheckpointError, "vocab_sha256"
            ):
                _load_finetune_pretrained_checkpoint(
                    CANBertForClassification(tiny_config()),
                    checkpoint_path,
                    vocab_path,
                )

    def test_finetuning_rejects_forward_semantic_model_config_mismatch(self):
        vocab_bytes = b"vocabulary"
        vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
        for field, incompatible_value in (
            ("hidden_act", "relu"),
            ("attention_probs_dropout_prob", 0.25),
        ):
            with self.subTest(field=field):
                _, payload = self._pretraining_payload(vocab_sha256)
                payload["model_config"][field] = incompatible_value
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    vocab_path = root / "vocab.json"
                    checkpoint_path = root / "pretrained.pt"
                    vocab_path.write_bytes(vocab_bytes)
                    torch.save(payload, checkpoint_path)
                    with self.assertRaisesRegex(
                        IncompatiblePretrainedCheckpointError, field
                    ):
                        _load_finetune_pretrained_checkpoint(
                            CANBertForClassification(tiny_config()),
                            checkpoint_path,
                            vocab_path,
                        )

    def test_finetuning_rejects_wrong_checkpoint_type_and_schema(self):
        vocab_bytes = b"vocabulary"
        vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
        for field, value, message in (
            ("checkpoint_type", "can-bert-finetune", "checkpoint type"),
            (
                "schema_version",
                CHECKPOINT_SCHEMA_VERSION + 1,
                "Unsupported pretrained checkpoint schema",
            ),
        ):
            with self.subTest(field=field):
                _, payload = self._pretraining_payload(vocab_sha256)
                payload[field] = value
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    vocab_path = root / "vocab.json"
                    checkpoint_path = root / "pretrained.pt"
                    vocab_path.write_bytes(vocab_bytes)
                    torch.save(payload, checkpoint_path)
                    with self.assertRaisesRegex(
                        IncompatiblePretrainedCheckpointError, message
                    ):
                        _load_finetune_pretrained_checkpoint(
                            CANBertForClassification(tiny_config()),
                            checkpoint_path,
                            vocab_path,
                        )

    def test_finetuning_rejects_missing_or_malformed_vocabulary_hash(self):
        vocab_bytes = b"vocabulary"
        valid_hash = hashlib.sha256(vocab_bytes).hexdigest()
        for label, training_config in (
            ("missing", {}),
            ("malformed", {"vocab_sha256": "A" * 64}),
        ):
            with self.subTest(label=label):
                _, payload = self._pretraining_payload(valid_hash)
                payload["training_config"] = training_config
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    vocab_path = root / "vocab.json"
                    checkpoint_path = root / "pretrained.pt"
                    vocab_path.write_bytes(vocab_bytes)
                    torch.save(payload, checkpoint_path)
                    with self.assertRaisesRegex(
                        IncompatiblePretrainedCheckpointError,
                        "missing or invalid vocab_sha256",
                    ):
                        _load_finetune_pretrained_checkpoint(
                            CANBertForClassification(tiny_config()),
                            checkpoint_path,
                            vocab_path,
                        )

    def test_schema_less_checkpoint_is_rejected_by_normal_finetuning_option(self):
        teacher = CANBertForMaskedLM(tiny_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "legacy.pt"
            vocab_path = root / "vocab.json"
            vocab_path.write_text("{}\n", encoding="utf-8")
            torch.save({"model": teacher.state_dict()}, checkpoint_path)
            with self.assertRaisesRegex(
                IncompatiblePretrainedCheckpointError,
                "--legacy_pretrained_checkpoint",
            ):
                _load_finetune_pretrained_checkpoint(
                    CANBertForClassification(tiny_config()),
                    checkpoint_path,
                    vocab_path,
                )

    def test_explicit_legacy_finetuning_checkpoint_loads_with_warning(self):
        teacher = CANBertForMaskedLM(tiny_config())
        classifier = CANBertForClassification(tiny_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "legacy.pt"
            vocab_path = root / "vocab.json"
            vocab_path.write_text("{}\n", encoding="utf-8")
            torch.save({"model": teacher.state_dict()}, checkpoint_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = _load_finetune_pretrained_checkpoint(
                    classifier,
                    checkpoint_path,
                    vocab_path,
                    legacy=True,
                )
            self.assertTrue(
                any(issubclass(item.category, RuntimeWarning) for item in caught)
            )
            warning_text = " ".join(str(item.message) for item in caught).casefold()
            for concept in (
                "cannot verify vocabulary semantic compatibility",
                "weights-only initialization",
                "separately confirm",
                "same vocabulary lineage",
            ):
                self.assertIn(concept, warning_text)

        self.assertEqual(loaded, len(classifier.bert.state_dict()))
        for key, value in teacher.bert.state_dict().items():
            self.assertTrue(torch.equal(value, classifier.bert.state_dict()[key]))

    def test_explicit_legacy_finetuning_rejects_any_schema_before_weight_load(self):
        vocab_bytes = b"current vocabulary"
        current_vocab_sha256 = hashlib.sha256(vocab_bytes).hexdigest()
        mismatched_vocab_sha256 = "0" * 64
        self.assertNotEqual(current_vocab_sha256, mismatched_vocab_sha256)
        teacher, versioned_payload = self._pretraining_payload(
            mismatched_vocab_sha256
        )
        versioned_payload["checkpoint_type"] = "wrong-checkpoint-type"
        future_payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION + 1,
            "model": teacher.state_dict(),
        }

        for label, payload in (
            ("v2-wrong-type-and-vocab", versioned_payload),
            ("future-schema", future_payload),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint_path = root / "metadata-bearing.pt"
                vocab_path = root / "vocab.json"
                vocab_path.write_bytes(vocab_bytes)
                torch.save(payload, checkpoint_path)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with patch(
                        "scripts.finetune._load_pretrained_bert"
                    ) as load_weights:
                        with self.assertRaisesRegex(
                            IncompatiblePretrainedCheckpointError,
                            "--pretrained_checkpoint",
                        ):
                            _load_finetune_pretrained_checkpoint(
                                CANBertForClassification(tiny_config()),
                                checkpoint_path,
                                vocab_path,
                                legacy=True,
                            )
                        load_weights.assert_not_called()
                self.assertEqual(caught, [])

    def test_explicit_legacy_finetuning_rejects_key_and_shape_mismatch(self):
        teacher = CANBertForMaskedLM(tiny_config())
        state = teacher.state_dict()
        body_key = next(key for key in state if key.startswith("bert."))
        incompatible_states = {}
        missing_key_state = dict(state)
        missing_key_state.pop(body_key)
        incompatible_states["missing keys"] = missing_key_state
        shape_mismatch_state = dict(state)
        shape_mismatch_state[body_key] = state[body_key].reshape(-1)[:1]
        incompatible_states["shape mismatches"] = shape_mismatch_state

        for message, incompatible_state in incompatible_states.items():
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    checkpoint_path = root / "legacy.pt"
                    vocab_path = root / "vocab.json"
                    vocab_path.write_text("{}\n", encoding="utf-8")
                    torch.save({"model": incompatible_state}, checkpoint_path)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with self.assertRaisesRegex(
                            IncompatiblePretrainedCheckpointError, message
                        ):
                            _load_finetune_pretrained_checkpoint(
                                CANBertForClassification(tiny_config()),
                                checkpoint_path,
                                vocab_path,
                                legacy=True,
                            )
                    self.assertTrue(
                        any(
                            issubclass(item.category, RuntimeWarning)
                            for item in caught
                        )
                    )

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

    def _advance_one_update(self, model, optimizer, scheduler):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def test_training_config_hashes_vocab_and_dataset_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab_path = root / "vocab.json"
            data_path = root / "train.log"
            vocab_bytes = b'{"tokens": ["A", "B"]}\n'
            data_bytes = b"(1.000000) can0 123#00\n"
            vocab_path.write_bytes(vocab_bytes)
            data_path.write_bytes(data_bytes)
            args = SimpleNamespace(
                gradient_accumulation_steps=2,
                epochs=3,
                warmup_steps=1,
                learning_rate=0.001,
                mask_prob=0.45,
                batch_size=2,
                seed=42,
                seq_len=16,
                dataset_type="candump",
                vocab_path=str(vocab_path),
                data_path=str(data_path),
            )
            config = _training_config(
                args,
                microbatches_per_epoch=5,
                world_size=2,
                scaler=_build_grad_scaler(torch.device("cpu")),
            )

        self.assertEqual(config["updates_per_epoch"], 3)
        self.assertEqual(config["total_optimizer_steps"], 9)
        self.assertEqual(
            config["vocab_sha256"], hashlib.sha256(vocab_bytes).hexdigest()
        )
        self.assertEqual(
            config["dataset_sha256"], hashlib.sha256(data_bytes).hexdigest()
        )
        self.assertEqual(config["world_size"], 2)
        self.assertFalse(config["amp_enabled"])

    def test_versioned_checkpoint_round_trip_restores_full_state(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        self._advance_one_update(model, optimizer, scheduler)
        training_config = full_training_config()

        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=1,
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
        self.assertEqual(state.global_optimizer_step, 1)
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

        self._advance_one_update(model, optimizer, scheduler)
        self._advance_one_update(
            restored_model, restored_optimizer, restored_scheduler
        )
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual(restored_scheduler.last_epoch, 2)
        self.assertEqual(scheduler.get_last_lr(), restored_scheduler.get_last_lr())
        for source, restored in zip(
            model.parameters(), restored_model.parameters(), strict=True
        ):
            torch.testing.assert_close(source, restored)

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
        self._advance_one_update(model, optimizer, scheduler)
        training_config = full_training_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=1,
                out_dir=Path(tmp),
                rank=0,
                training_config=training_config,
            )
            with self.assertRaisesRegex(
                IncompatibleCheckpointError, "Training configuration is incompatible"
            ):
                _load_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    path,
                    torch.device("cpu"),
                    rank=0,
                    expected_training_config=full_training_config(
                        updates_per_epoch=2, total_optimizer_steps=4
                    ),
                )

    def test_scheduler_and_global_step_mismatch_is_rejected(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        training_config = full_training_config()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                IncompatibleCheckpointError, "Scheduler state is incompatible"
            ):
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=0,
                    global_optimizer_step=1,
                    out_dir=Path(tmp),
                    rank=0,
                    training_config=training_config,
                )

            self._advance_one_update(model, optimizer, scheduler)
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=1,
                out_dir=Path(tmp),
                rank=0,
                training_config=training_config,
            )
            payload = torch.load(path, map_location="cpu")
            payload["global_optimizer_step"] = 2
            torch.save(payload, path)

            restored = self._training_parts()
            with self.assertRaisesRegex(
                IncompatibleCheckpointError, "Scheduler state is incompatible"
            ):
                _load_checkpoint(
                    *restored,
                    path,
                    torch.device("cpu"),
                    rank=0,
                    expected_training_config=training_config,
                )

    def test_resume_rejects_changed_or_missing_content_hashes(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        self._advance_one_update(model, optimizer, scheduler)
        training_config = full_training_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=1,
                out_dir=Path(tmp),
                rank=0,
                training_config=training_config,
            )
            payload = torch.load(path, map_location="cpu")

            for hash_key in ("vocab_sha256", "dataset_sha256"):
                with self.subTest(hash_key=hash_key):
                    expected = full_training_config(**{hash_key: "c" * 64})
                    restored = self._training_parts()
                    with self.assertRaisesRegex(
                        IncompatibleCheckpointError, hash_key
                    ):
                        _load_checkpoint(
                            *restored,
                            path,
                            torch.device("cpu"),
                            rank=0,
                            expected_training_config=expected,
                        )

            missing_payload = dict(payload)
            missing_payload["training_config"] = dict(
                payload["training_config"]
            )
            missing_payload["training_config"].pop("dataset_sha256")
            missing_path = Path(tmp) / "missing-hash.pt"
            torch.save(missing_payload, missing_path)
            restored = self._training_parts()
            with self.assertRaisesRegex(
                IncompatibleCheckpointError, "Training configuration is incomplete"
            ):
                _load_checkpoint(
                    *restored,
                    missing_path,
                    torch.device("cpu"),
                    rank=0,
                    expected_training_config=training_config,
                )

    def test_versioned_config_fields_require_mappings(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        self._advance_one_update(model, optimizer, scheduler)
        training_config = full_training_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=1,
                out_dir=Path(tmp),
                rank=0,
                training_config=training_config,
            )
            payload = torch.load(path, map_location="cpu")
            for field in ("training_config", "model_config"):
                with self.subTest(field=field):
                    malformed = dict(payload)
                    malformed[field] = []
                    malformed_path = Path(tmp) / f"malformed-{field}.pt"
                    torch.save(malformed, malformed_path)
                    restored = self._training_parts()
                    with self.assertRaisesRegex(
                        IncompatibleCheckpointError, "must be a mapping"
                    ):
                        _load_checkpoint(
                            *restored,
                            malformed_path,
                            torch.device("cpu"),
                            rank=0,
                            expected_training_config=training_config,
                        )

            restored = self._training_parts()
            with self.assertRaisesRegex(
                IncompatibleCheckpointError,
                "Expected training configuration must be a mapping",
            ):
                _load_checkpoint(
                    *restored,
                    path,
                    torch.device("cpu"),
                    rank=0,
                    expected_training_config=[],
                )

    def test_shape_compatible_architecture_mismatch_is_rejected(self):
        model, optimizer, scheduler, scaler = self._training_parts()
        self._advance_one_update(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=0,
                global_optimizer_step=1,
                out_dir=Path(tmp),
                rank=0,
                training_config=full_training_config(),
            )
            payload = torch.load(path)
            payload["model_config"]["hidden_act"] = "relu"
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
