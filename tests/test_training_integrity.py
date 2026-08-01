import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.pretrain import (
    _build_grad_scaler,
    _optimizer_updates_per_epoch,
    _train_epoch,
)


class MicrobatchDataset(Dataset):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([float(index + 1)]),
            "labels": torch.tensor([0.0]),
        }


class LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels):
        loss = ((self.weight * input_ids.mean()) - labels.mean()).pow(2)
        return (loss,)


class NoSyncLossModel(LossModel):
    def __init__(self):
        super().__init__()
        self.no_sync_calls = 0

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.step_calls = 0
        self.zero_grad_flags = []

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.zero_grad_flags.append(set_to_none)
        return super().zero_grad(set_to_none=set_to_none)


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def get_last_lr(self):
        return [0.0]


class DisabledScaler:
    def __init__(self):
        self.scale_value = 1.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def get_scale(self):
        return self.scale_value


class OverflowOnceScaler(DisabledScaler):
    def __init__(self):
        super().__init__()
        self.attempt = 0
        self.overflowed = False

    def step(self, optimizer):
        self.attempt += 1
        self.overflowed = self.attempt == 1
        if not self.overflowed:
            optimizer.step()

    def update(self):
        if self.overflowed:
            self.scale_value /= 2


class TrainingIntegrityTests(unittest.TestCase):
    def _run(self, microbatches, accumulation, model=None, scaler=None):
        model = model or LossModel()
        loader = DataLoader(MicrobatchDataset(microbatches), batch_size=1)
        optimizer = CountingSGD(model.parameters(), lr=0.0)
        scheduler = CountingScheduler()
        with patch("scripts.pretrain.wandb.log"):
            result = _train_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                scaler or DisabledScaler(),
                torch.device("cpu"),
                epoch=0,
                end_epoch=1,
                rank=0,
                gradient_accumulation_steps=accumulation,
                global_optimizer_step=10,
            )
        return result, model, optimizer, scheduler

    def test_update_count_uses_ceiling_and_steps_remainder(self):
        result, _, optimizer, scheduler = self._run(5, 2)
        self.assertEqual(result.microbatches, 5)
        self.assertEqual(result.optimizer_steps, 3)
        self.assertEqual(result.global_optimizer_step, 13)
        self.assertEqual(optimizer.step_calls, 3)
        self.assertEqual(scheduler.step_calls, 3)
        self.assertEqual(optimizer.zero_grad_flags, [True, True, True, True])
        self.assertEqual(result.average_raw_loss, 11.0)

    def test_accumulation_one_updates_every_microbatch(self):
        result, _, optimizer, scheduler = self._run(4, 1)
        self.assertEqual(result.optimizer_steps, 4)
        self.assertEqual(optimizer.step_calls, 4)
        self.assertEqual(scheduler.step_calls, 4)

    def test_non_update_microbatches_use_no_sync(self):
        model = NoSyncLossModel()
        result, model, _, _ = self._run(5, 2, model=model)
        self.assertEqual(result.optimizer_steps, 3)
        self.assertEqual(model.no_sync_calls, 2)

    def test_overflow_does_not_advance_scheduler_or_global_step(self):
        result, _, optimizer, scheduler = self._run(
            2, 1, scaler=OverflowOnceScaler()
        )
        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(result.skipped_optimizer_steps, 1)
        self.assertEqual(result.global_optimizer_step, 11)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(scheduler.step_calls, 1)

    def test_cpu_scaler_is_a_disabled_scaler_not_none(self):
        scaler = _build_grad_scaler(torch.device("cpu"))
        self.assertIsNotNone(scaler)
        self.assertFalse(scaler.is_enabled())

    def test_optimizer_step_budget_uses_ceiling(self):
        self.assertEqual(_optimizer_updates_per_epoch(5, 2), 3)
        self.assertEqual(_optimizer_updates_per_epoch(4, 2), 2)
        self.assertEqual(_optimizer_updates_per_epoch(4, 1), 4)


if __name__ == "__main__":
    unittest.main()
