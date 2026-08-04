import unittest
from pathlib import Path

try:
    from config import cfg as default_cfg
except ModuleNotFoundError as error:
    if error.name != "yacs":
        raise
    default_cfg = None

from utils.protocol2_matrix import config_overrides, load_protocol2_matrix


PROJECT_DIR = Path(__file__).resolve().parents[1]
MATRIX_PATH_60 = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "protocol2_clip_matrix.json"
)
MATRIX_PATH_30 = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "protocol2_clip_matrix_30ep.json"
)
MATRIX_PATH_SIGLIP2_30 = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "protocol2_siglip2_matrix_30ep.json"
)


class Protocol2MatrixStructureTest(unittest.TestCase):
    def test_complete_matrix_loads(self):
        for path in (MATRIX_PATH_60, MATRIX_PATH_30, MATRIX_PATH_SIGLIP2_30):
            matrix = load_protocol2_matrix(path)
            self.assertFalse(matrix["combineall"])
            self.assertEqual(len(matrix["runs"]), 6)


@unittest.skipIf(default_cfg is None, "yacs is not installed")
class Protocol2ResolvedConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrices = [
            load_protocol2_matrix(MATRIX_PATH_60),
            load_protocol2_matrix(MATRIX_PATH_30),
            load_protocol2_matrix(MATRIX_PATH_SIGLIP2_30),
        ]

    def _resolve(self, matrix, run):
        current = default_cfg.clone()
        current.merge_from_file(str(PROJECT_DIR / run["base_config"]))
        current.merge_from_list(config_overrides(matrix, run))
        return current

    def test_each_pair_differs_only_by_caption_and_output(self):
        for matrix in self.matrices:
            grouped = {}
            for run in matrix["runs"]:
                key = (tuple(run["sources"]), run["target"])
                grouped.setdefault(key, {})[run["method"]] = self._resolve(
                    matrix, run
                )

            for configs in grouped.values():
                image = configs["image_only"].clone()
                caption = configs["caption_alignment"].clone()
                image.defrost()
                caption.defrost()
                image.OBJECTIVE.CAPTION = caption.OBJECTIVE.CAPTION
                image.OUTPUT_DIR = caption.OUTPUT_DIR
                self.assertEqual(image, caption)

    def test_protocol2_never_combines_target_splits(self):
        matrix = self.matrices[0]
        for run in matrix["runs"]:
            current = self._resolve(matrix, run)
            self.assertFalse(current.DATASETS.COMBINEALL)
            self.assertEqual(current.SOLVER.SEED, 1234)
            self.assertEqual(current.SOLVER.MAX_EPOCHS, 60)
            self.assertEqual(current.SOLVER.EVAL_PERIOD, 10)

    def test_30_epoch_schedule_is_proportionally_compressed(self):
        matrix = self.matrices[1]
        for run in matrix["runs"]:
            current = self._resolve(matrix, run)
            self.assertEqual(current.SOLVER.SEED, 1234)
            self.assertEqual(current.SOLVER.MAX_EPOCHS, 30)
            self.assertEqual(current.SOLVER.BASE_LR, 0.000005)
            self.assertEqual(current.SOLVER.WARMUP_ITERS, 5)
            self.assertEqual(tuple(current.SOLVER.STEPS), (15, 25))
            self.assertEqual(current.SOLVER.EVAL_PERIOD, 5)
            self.assertEqual(current.SOLVER.CHECKPOINT_PERIOD, 5)

    def test_siglip2_matrix_uses_siglip2_configs_and_s1_schedule(self):
        matrix = load_protocol2_matrix(MATRIX_PATH_SIGLIP2_30)
        self.assertEqual(matrix["backbone"], "siglip2")
        self.assertEqual(
            matrix["model_overrides"],
            {
                "HEAD.TYPE": "siglip2_native_pooler",
                "BACKBONE.STATIC_POSITION_EMBEDDING": True,
            },
        )
        self.assertEqual(
            matrix["method_configs"]["image_only"],
            "configs/experiments/siglip2_multibranch_image_only.yml",
        )
        for run in matrix["runs"]:
            current = self._resolve(matrix, run)
            self.assertEqual(current.MODEL.BACKBONE.NAME, "siglip2_base_patch16")
            self.assertEqual(current.MODEL.HEAD.TYPE, "siglip2_native_pooler")
            self.assertTrue(
                current.MODEL.BACKBONE.STATIC_POSITION_EMBEDDING
            )
            self.assertEqual(current.SOLVER.MAX_EPOCHS, 30)
            self.assertEqual(tuple(current.SOLVER.STEPS), (5, 20))
            self.assertEqual(current.SOLVER.EVAL_PERIOD, 5)


if __name__ == "__main__":
    unittest.main()
