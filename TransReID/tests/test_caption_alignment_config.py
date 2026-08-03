import unittest
from pathlib import Path

try:
    from config import cfg as default_cfg
except ModuleNotFoundError as error:
    if error.name != "yacs":
        raise
    default_cfg = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_ONLY_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "clip_market_to_msmt_image_only.yml"
)
CAPTION_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "clip_market_to_msmt_caption_alignment_v2_4.yml"
)


@unittest.skipIf(default_cfg is None, "yacs is not installed")
class CaptionAlignmentConfigTest(unittest.TestCase):
    def setUp(self):
        self.image_cfg = default_cfg.clone()
        self.image_cfg.merge_from_file(str(IMAGE_ONLY_CONFIG))
        self.caption_cfg = default_cfg.clone()
        self.caption_cfg.merge_from_file(str(CAPTION_CONFIG))

    def test_only_caption_objective_and_output_path_change(self):
        image_dict = self.image_cfg.clone()
        caption_dict = self.caption_cfg.clone()
        image_dict.defrost()
        caption_dict.defrost()
        image_dict.OBJECTIVE.CAPTION = caption_dict.OBJECTIVE.CAPTION
        image_dict.OUTPUT_DIR = caption_dict.OUTPUT_DIR
        self.assertEqual(image_dict, caption_dict)

    def test_global_native_clip_alignment_is_explicit(self):
        caption = self.caption_cfg.OBJECTIVE.CAPTION
        self.assertTrue(caption.ENABLED)
        self.assertEqual(caption.FEATURE_LEVEL, "global")
        self.assertFalse(caption.USE_PROJECTION)
        self.assertEqual(caption.SELECTION, "first")
        self.assertFalse(caption.TEXT_TRAINABLE)
        self.assertEqual(caption.POSITIVE_MODE, "pid")


if __name__ == "__main__":
    unittest.main()
