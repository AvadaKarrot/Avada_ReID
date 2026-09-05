import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from engine.trainer import Trainer


class ImageOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, images):
        raw = self.projection(images)
        return SimpleNamespace(
            raw_feature=raw,
            embedding=raw,
            logits=raw,
        )


class CaptionRecordingObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.seen_captions = None

    def forward(self, outputs, batch):
        self.seen_captions = batch.get("captions")
        total = outputs.logits.square().mean() * self.scale
        return {"total": total}


class CountingModel(ImageOnlyModel):
    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, images):
        self.forward_count += 1
        return super().forward(images)


class ConstantEvaluator:
    def evaluate_loader(self, model, loader, num_query):
        self.last_num_query = num_query
        return {"mAP": 0.25, "rank1": 0.5}


class UnifiedTrainerTest(unittest.TestCase):
    def test_memory_smoke_stops_without_writing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            model = CountingModel()
            objective = CaptionRecordingObjective()
            optimizer = torch.optim.SGD(
                list(model.parameters()) + list(objective.parameters()),
                lr=0.1,
            )
            trainer = Trainer(
                model=model,
                objective=objective,
                optimizer=optimizer,
                device="cpu",
                output_dir=directory,
            )
            batch = {
                "images": torch.ones(2, 2),
                "pids": torch.tensor([0, 1]),
            }
            state = trainer.fit(
                train_loader=[batch, batch, batch],
                max_epochs=1,
                max_iterations_per_epoch=1,
                save_checkpoints=False,
            )
            self.assertEqual(model.forward_count, 1)
            self.assertEqual(state.epoch, 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_captions_reach_objective_but_not_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ImageOnlyModel()
            objective = CaptionRecordingObjective()
            optimizer = torch.optim.SGD(
                list(model.parameters()) + list(objective.parameters()),
                lr=0.1,
            )
            trainer = Trainer(
                model=model,
                objective=objective,
                optimizer=optimizer,
                evaluator=ConstantEvaluator(),
                device="cpu",
                output_dir=directory,
                model_name="contract",
                log_period=10,
            )
            batch = {
                "images": torch.ones(2, 2),
                "pids": torch.tensor([0, 1]),
                "camids": torch.tensor([0, 1]),
                "captions": ("first", "second"),
                "caption_mask": torch.tensor([True, True]),
            }
            state = trainer.fit(
                train_loader=[batch],
                max_epochs=2,
                checkpoint_period=2,
                eval_period=1,
                validation={"loader": [], "num_query": 1},
            )

            self.assertEqual(
                objective.seen_captions, ("first", "second")
            )
            self.assertEqual(state.epoch, 2)
            self.assertEqual(state.best_epoch, 1)
            output = Path(directory)
            self.assertTrue((output / "contract_epoch1.pth").is_file())
            self.assertTrue((output / "contract_epoch2.pth").is_file())
            self.assertTrue((output / "model_best.pth.tar").is_file())
            self.assertTrue((output / "model_last.pth.tar").is_file())
            payload = torch.load(
                output / "checkpoint_latest.pth.tar",
                map_location="cpu",
            )
            self.assertIn("objective", payload)
            self.assertIn("scaler", payload)

            resumed_model = ImageOnlyModel()
            resumed_objective = CaptionRecordingObjective()
            resumed_optimizer = torch.optim.SGD(
                list(resumed_model.parameters())
                + list(resumed_objective.parameters()),
                lr=0.1,
            )
            resumed_trainer = Trainer(
                model=resumed_model,
                objective=resumed_objective,
                optimizer=resumed_optimizer,
                evaluator=ConstantEvaluator(),
                device="cpu",
                output_dir=directory,
                model_name="contract",
            )
            resumed_state = resumed_trainer.fit(
                train_loader=[batch],
                max_epochs=3,
                checkpoint_period=2,
                eval_period=1,
                validation={"loader": [], "num_query": 1},
                resume=True,
            )
            self.assertEqual(resumed_state.epoch, 3)
            self.assertEqual(resumed_state.best_epoch, 1)
            self.assertTrue((output / "contract_epoch3.pth").is_file())


if __name__ == "__main__":
    unittest.main()
