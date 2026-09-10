import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

try:
    from config import cfg
except ModuleNotFoundError as error:
    if error.name != "yacs":
        raise
    cfg = None

from objectives.build import build_caption_objective
from objectives.losses.caption_alignment import CaptionAlignmentObjective
from objectives.losses.caption_sigmoid import CaptionSigmoidObjective
from utils.config_validation import validate_training_config


class FakeEncoder(nn.Module):
    output_dim = 768

    def forward(self, captions):
        return torch.ones(len(captions), 768)


@unittest.skipIf(cfg is None, "yacs required")
class SigmoidConfigTest(unittest.TestCase):
    def test_builder_switches_and_invalid_values(self):
        base = cfg.clone()
        base.merge_from_file(str(Path(__file__).resolve().parents[1] /
            "configs/experiments/base/siglip2_naflex_caption_pid.yml"))
        with patch("objectives.build.SigLIP2TextEncoder", return_value=FakeEncoder()):
            self.assertIs(type(build_caption_objective(base, 768)), CaptionAlignmentObjective)
            for mode in ("instance", "pid"):
                for reduction in ("anchor", "pair_mean", "balanced"):
                    variant = base.clone()
                    variant.OBJECTIVE.CAPTION.LOSS_TYPE = "sigmoid"
                    variant.OBJECTIVE.CAPTION.POSITIVE_MODE = mode
                    variant.OBJECTIVE.CAPTION.SIGMOID_REDUCTION = reduction
                    validate_training_config(variant)
                    objective = build_caption_objective(variant, 768)
                    self.assertIsInstance(objective, CaptionSigmoidObjective)
                    self.assertEqual(objective.positive_mode, mode)
                    self.assertEqual(objective.sigmoid_reduction, reduction)
            for key, value in (("LOSS_TYPE", "bad"), ("SIGMOID_REDUCTION", "bad")):
                variant = base.clone()
                setattr(variant.OBJECTIVE.CAPTION, key, value)
                with self.assertRaises(ValueError):
                    validate_training_config(variant)


if __name__ == "__main__":
    unittest.main()
