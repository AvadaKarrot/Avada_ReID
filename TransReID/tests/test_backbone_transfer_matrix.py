import unittest
from pathlib import Path

from utils.backbone_transfer_matrix import (
    BACKBONE_CONFIGS,
    BACKBONE_METHOD_CONFIGS,
    EXPECTED_DIRECTIONS,
    config_overrides,
    load_backbone_transfer_matrix,
)


MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "backbone_transfer_matrix_s1.json"
)

SIGLIP2_FINAL_MAP_ONLY_IMAGE_ONLY_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "siglip2_final_map_only_image_only_s1_30ep.json"
)


class BackboneTransferMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load_backbone_transfer_matrix(MATRIX_PATH)

    def test_complete_cross_product(self):
        self.assertEqual(len(self.matrix["runs"]), 30)
        observed = {
            (
                run["source"],
                run["target"],
                run["backbone"],
                run.get("method", "image_only"),
            )
            for run in self.matrix["runs"]
        }
        expected = {
            (source, target, backbone, method)
            for source, target in EXPECTED_DIRECTIONS
            for backbone, methods in BACKBONE_METHOD_CONFIGS.items()
            for method in methods
        }
        self.assertEqual(observed, expected)

    def test_s1_schedule_and_isolated_outputs(self):
        solver = self.matrix["solver_overrides"]
        self.assertEqual(solver["MAX_EPOCHS"], 30)
        self.assertEqual(solver["WARMUP_ITERS"], 5)
        self.assertEqual(solver["STEPS"], [5, 20])
        self.assertEqual(solver["EVAL_PERIOD"], 5)
        self.assertEqual(solver["CHECKPOINT_PERIOD"], 5)
        outputs = [run["output_dir"] for run in self.matrix["runs"]]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_overrides_keep_target_image_only_protocol(self):
        run = self.matrix["runs"][0]
        overrides = config_overrides(self.matrix, run)
        joined = " ".join(overrides)
        self.assertIn("DATASETS.SOURCES", joined)
        self.assertIn("DATASETS.TARGETS", joined)
        self.assertIn("DATASETS.COMBINEALL False", joined)

    def test_siglip2_final_map_only_image_only_scoped_matrix(self):
        matrix = load_backbone_transfer_matrix(
            SIGLIP2_FINAL_MAP_ONLY_IMAGE_ONLY_MATRIX_PATH
        )
        self.assertEqual(len(matrix["runs"]), 6)
        self.assertEqual(
            {(run["source"], run["target"]) for run in matrix["runs"]},
            EXPECTED_DIRECTIONS,
        )
        self.assertEqual(
            {(run["backbone"], run["method"]) for run in matrix["runs"]},
            {("siglip2", "image_only")},
        )
        overrides = config_overrides(matrix, matrix["runs"][0])
        joined = " ".join(overrides)
        self.assertIn("MODEL.HEAD.TYPE siglip2_native_pooler", joined)
        self.assertIn("MODEL.BACKBONE.STATIC_POSITION_EMBEDDING True", joined)
        self.assertIn("MODEL.BACKBONE.PENULTIMATE_MAP_POOLER False", joined)


if __name__ == "__main__":
    unittest.main()
