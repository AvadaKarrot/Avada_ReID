import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.config_validation import validate_training_config

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
    / "base"
    / "clip_vit_b16_image_only.yml"
)
CAPTION_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "base"
    / "clip_vit_b16_caption_pid.yml"
)
SIGLIP2_IMAGE_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "siglip2_multibranch_image_only.yml"
)
SIGLIP2_CAPTION_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "siglip2_multibranch_caption_alignment.yml"
)
SIGLIP2_NAFLEX_IMAGE_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "base"
    / "siglip2_naflex_image_only.yml"
)
SIGLIP2_NAFLEX_CAPTION_CONFIG = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "base"
    / "siglip2_naflex_caption_pid.yml"
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

    def test_siglip2_caption_only_changes_objective_and_output(self):
        image_cfg = default_cfg.clone()
        image_cfg.merge_from_file(str(SIGLIP2_IMAGE_CONFIG))
        caption_cfg = default_cfg.clone()
        caption_cfg.merge_from_file(str(SIGLIP2_CAPTION_CONFIG))
        image_cfg.defrost()
        caption_cfg.defrost()
        image_cfg.OBJECTIVE.CAPTION = caption_cfg.OBJECTIVE.CAPTION
        image_cfg.OUTPUT_DIR = caption_cfg.OUTPUT_DIR
        self.assertEqual(image_cfg, caption_cfg)

    def test_siglip2_native_multibranch_caption_is_valid(self):
        caption_cfg = default_cfg.clone()
        caption_cfg.merge_from_file(str(SIGLIP2_CAPTION_CONFIG))
        validate_training_config(caption_cfg)
        caption = caption_cfg.OBJECTIVE.CAPTION
        self.assertEqual(caption.TEXT_ENCODER, "siglip2_native")
        self.assertFalse(caption.USE_PROJECTION)
        self.assertEqual(caption.PROJECTION_DIM, 768)

    def test_siglip2_naflex_pair_differs_only_by_caption_and_output(self):
        image_cfg = default_cfg.clone()
        image_cfg.merge_from_file(str(SIGLIP2_NAFLEX_IMAGE_CONFIG))
        caption_cfg = default_cfg.clone()
        caption_cfg.merge_from_file(str(SIGLIP2_NAFLEX_CAPTION_CONFIG))
        image_cfg.defrost()
        caption_cfg.defrost()
        image_cfg.OBJECTIVE.CAPTION = caption_cfg.OBJECTIVE.CAPTION
        image_cfg.OUTPUT_DIR = caption_cfg.OUTPUT_DIR
        self.assertEqual(image_cfg, caption_cfg)

    def test_siglip2_naflex_config_uses_native_variable_shape_contract(self):
        caption_cfg = default_cfg.clone()
        caption_cfg.merge_from_file(str(SIGLIP2_NAFLEX_CAPTION_CONFIG))
        validate_training_config(caption_cfg)
        self.assertEqual(
            caption_cfg.MODEL.BACKBONE.NAME,
            "siglip2_base_patch16_naflex",
        )
        self.assertFalse(
            caption_cfg.MODEL.BACKBONE.STATIC_POSITION_EMBEDDING
        )
        self.assertEqual(
            caption_cfg.MODEL.BACKBONE.NAFLEX_MAX_NUM_PATCHES,
            128,
        )
        self.assertEqual(
            list(caption_cfg.DATASETS.TRANSFORMS),
            ["random_flip", "color_jitter"],
        )


class CaptionAlignmentValidationTest(unittest.TestCase):
    @staticmethod
    def _config(backbone, text_encoder):
        return SimpleNamespace(
            MODEL=SimpleNamespace(
                CAPTION=False,
                BACKBONE=SimpleNamespace(NAME=backbone),
                HEAD=SimpleNamespace(TYPE="multibranch_parity"),
                SIE_CAMERA=False,
                SIE_VIEW=False,
            ),
            OBJECTIVE=SimpleNamespace(
                CAPTION=SimpleNamespace(
                    ENABLED=True,
                    FILE="captions.jsonl",
                    WEIGHT=0.1,
                    FEATURE_LEVEL="global",
                    POSITIVE_MODE="pid",
                    TEXT_ENCODER=text_encoder,
                )
            ),
        )

    def test_dinov3_multibranch_caption_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supported only"):
            validate_training_config(
                self._config("dinov3_vitb16", "siglip2_native")
            )

    def test_siglip2_with_legacy_clip_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "siglip2_native"):
            validate_training_config(
                self._config("siglip2_base_patch16", "clip_legacy")
            )


if __name__ == "__main__":
    unittest.main()
