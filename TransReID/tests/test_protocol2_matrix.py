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
MATRIX_PATH = (
    PROJECT_DIR
    / "configs"
    / "experiments"
    / "protocol2_clip_matrix.json"
)


class Protocol2MatrixStructureTest(unittest.TestCase):
    def test_complete_matrix_loads(self):
        matrix = load_protocol2_matrix(MATRIX_PATH)
        self.assertFalse(matrix["combineall"])
        self.assertEqual(len(matrix["runs"]), 6)


@unittest.skipIf(default_cfg is None, "yacs is not installed")
class Protocol2ResolvedConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = load_protocol2_matrix(MATRIX_PATH)

    def _resolve(self, run):
        current = default_cfg.clone()
        current.merge_from_file(str(PROJECT_DIR / run["base_config"]))
        current.merge_from_list(config_overrides(self.matrix, run))
        return current

    def test_each_pair_differs_only_by_caption_and_output(self):
        grouped = {}
        for run in self.matrix["runs"]:
            key = (tuple(run["sources"]), run["target"])
            grouped.setdefault(key, {})[run["method"]] = self._resolve(run)

        for configs in grouped.values():
            image = configs["image_only"].clone()
            caption = configs["caption_alignment"].clone()
            image.defrost()
            caption.defrost()
            image.OBJECTIVE.CAPTION = caption.OBJECTIVE.CAPTION
            image.OUTPUT_DIR = caption.OUTPUT_DIR
            self.assertEqual(image, caption)

    def test_protocol2_never_combines_target_splits(self):
        for run in self.matrix["runs"]:
            current = self._resolve(run)
            self.assertFalse(current.DATASETS.COMBINEALL)
            self.assertEqual(current.SOLVER.SEED, 1234)
            self.assertEqual(current.SOLVER.MAX_EPOCHS, 60)
            self.assertEqual(current.SOLVER.EVAL_PERIOD, 10)


if __name__ == "__main__":
    unittest.main()
