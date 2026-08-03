import unittest
from types import SimpleNamespace

from utils.config_validation import validate_training_config


def make_cfg(
    *,
    legacy=False,
    enabled=False,
    file="",
    weight=0.0,
    feature_level="global",
):
    return SimpleNamespace(
        MODEL=SimpleNamespace(CAPTION=legacy),
        OBJECTIVE=SimpleNamespace(
            CAPTION=SimpleNamespace(
                ENABLED=enabled,
                FILE=file,
                WEIGHT=weight,
                FEATURE_LEVEL=feature_level,
            )
        ),
    )


class TrainConfigTest(unittest.TestCase):
    def test_image_only_config_is_valid(self):
        validate_training_config(make_cfg())

    def test_legacy_caption_switch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "MODEL.CAPTION"):
            validate_training_config(make_cfg(legacy=True))

    def test_caption_requires_file_and_positive_weight(self):
        with self.assertRaisesRegex(ValueError, "CAPTION.FILE"):
            validate_training_config(make_cfg(enabled=True, weight=1.0))
        with self.assertRaisesRegex(ValueError, "CAPTION.WEIGHT"):
            validate_training_config(
                make_cfg(enabled=True, file="captions.jsonl")
            )
        validate_training_config(
            make_cfg(
                enabled=True,
                file="captions.jsonl",
                weight=0.5,
            )
        )

    def test_token_level_alignment_is_a_separate_experiment(self):
        with self.assertRaisesRegex(ValueError, "FEATURE_LEVEL"):
            validate_training_config(
                make_cfg(
                    enabled=True,
                    file="captions.jsonl",
                    weight=0.1,
                    feature_level="token",
                )
            )


if __name__ == "__main__":
    unittest.main()
