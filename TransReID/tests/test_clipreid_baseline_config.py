import unittest
from pathlib import Path

try:
    from config import cfg as default_cfg
except ModuleNotFoundError as error:
    if error.name != "yacs":
        raise
    default_cfg = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "clip_market_to_msmt17.yml"
)


@unittest.skipIf(default_cfg is None, "yacs is not installed")
class ClipReIDBaselineConfigTest(unittest.TestCase):
    def setUp(self):
        self.cfg = default_cfg.clone()
        self.cfg.merge_from_file(str(CONFIG_PATH))

    def test_market_is_the_only_training_source(self):
        self.assertEqual(self.cfg.DATASETS.SOURCES, "market1501")

    def test_msmt17_is_the_only_target(self):
        self.assertEqual(self.cfg.DATASETS.TARGETS, "msmt17")

    def test_caption_and_prompt_paths_are_disabled(self):
        self.assertFalse(self.cfg.MODEL.CAPTION)
        self.assertFalse(self.cfg.MODEL.PROMPT)
        self.assertFalse(self.cfg.OBJECTIVE.CAPTION.ENABLED)
        self.assertEqual(self.cfg.OBJECTIVE.CAPTION.WEIGHT, 0.0)
        self.assertFalse(self.cfg.COOP.COOP_PROMPT)
        self.assertFalse(self.cfg.MAPLE.MAPLE_PROMPT)

    def test_legacy_batch_contract_does_not_enable_view_labels(self):
        self.assertFalse(self.cfg.MODEL.SIE_VIEW)

    def test_amp_starts_at_the_verified_scale(self):
        self.assertTrue(self.cfg.SOLVER.AMP_ENABLED)
        self.assertEqual(self.cfg.SOLVER.AMP_INIT_SCALE, 512.0)


if __name__ == "__main__":
    unittest.main()
