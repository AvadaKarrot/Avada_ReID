import logging
import tempfile
import unittest
from pathlib import Path

import torch

from engine.checkpoint import TrainingCheckpointer, TrainingState
from processor.runtime import period_due, save_epoch_state


class TrainingCheckpointTest(unittest.TestCase):
    def _components(self):
        model = torch.nn.Linear(3, 2)
        center = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer_center = torch.optim.SGD(center.parameters(), lr=0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.1
        )
        return model, center, optimizer, optimizer_center, scheduler

    def test_best_epoch_keeps_earlier_checkpoint_on_tie(self):
        state = TrainingState()
        self.assertTrue(state.update_best(0.4, 10))
        self.assertFalse(state.update_best(0.4, 20))
        self.assertEqual(state.best_epoch, 10)

    def test_complete_checkpoint_restores_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            components = self._components()
            model, center, optimizer, optimizer_center, scheduler = components
            checkpointer = TrainingCheckpointer(
                directory,
                model=model,
                model_name="reid",
                center_criterion=center,
                optimizer=optimizer,
                optimizer_center=optimizer_center,
                scheduler=scheduler,
            )
            state = TrainingState(epoch=4, best_mAP=0.35, best_epoch=4)
            expected_weight = model.weight.detach().clone()
            checkpoint_path = checkpointer.save_epoch(
                4, state, {"mAP": 0.35}
            )

            with torch.no_grad():
                model.weight.zero_()
            optimizer.param_groups[0]["lr"] = 9.0

            restored = checkpointer.resume(checkpoint_path)
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
            self.assertEqual(restored.epoch, 4)
            self.assertEqual(restored.best_epoch, 4)
            self.assertAlmostEqual(restored.best_mAP, 0.35)

            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
            )
            for key in (
                "model",
                "objective",
                "center_criterion",
                "optimizer",
                "optimizer_center",
                "scheduler",
                "epoch",
                "best_mAP",
                "best_epoch",
                "metrics",
            ):
                self.assertIn(key, payload)

    def test_evaluated_epochs_are_saved_by_eval_period(self):
        with tempfile.TemporaryDirectory() as directory:
            model, center, optimizer, optimizer_center, scheduler = (
                self._components()
            )
            checkpointer = TrainingCheckpointer(
                directory,
                model=model,
                model_name="reid",
                center_criterion=center,
                optimizer=optimizer,
                optimizer_center=optimizer_center,
                scheduler=scheduler,
            )
            state = TrainingState()
            maps = {2: 0.1, 4: 0.3, 6: 0.2}
            logger = logging.getLogger("checkpoint-test")

            for epoch in range(1, 7):
                metrics = (
                    {"mAP": maps[epoch]}
                    if period_due(epoch, 2)
                    else None
                )
                save_epoch_state(
                    checkpointer=checkpointer,
                    state=state,
                    epoch=epoch,
                    max_epochs=6,
                    checkpoint_period=3,
                    metrics=metrics,
                    logger=logger,
                )

            output = Path(directory)
            saved_epochs = {
                int(path.stem.removeprefix("reid_epoch"))
                for path in output.glob("reid_epoch*.pth")
            }
            self.assertEqual(saved_epochs, {2, 3, 4, 6})
            self.assertTrue(checkpointer.best_path.is_file())
            self.assertTrue(checkpointer.last_path.is_file())
            self.assertTrue(checkpointer.latest_path.is_file())
            self.assertEqual(state.best_epoch, 4)
            self.assertAlmostEqual(state.best_mAP, 0.3)

    def test_resume_rejects_weights_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model, center, optimizer, optimizer_center, scheduler = (
                self._components()
            )
            checkpointer = TrainingCheckpointer(
                directory,
                model=model,
                model_name="reid",
                center_criterion=center,
                optimizer=optimizer,
                optimizer_center=optimizer_center,
                scheduler=scheduler,
            )
            weights_only = Path(directory) / "weights_only.pth"
            torch.save({"model": model.state_dict(), "epoch": 2}, weights_only)
            with self.assertRaisesRegex(
                ValueError, "complete training checkpoint"
            ):
                checkpointer.resume(weights_only)


if __name__ == "__main__":
    unittest.main()
