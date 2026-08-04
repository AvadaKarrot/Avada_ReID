import tempfile
import unittest
from pathlib import Path

from tools.run_gated_protocol2_matrix import best_map_from_log


class GatedProtocol2MatrixTest(unittest.TestCase):
    def test_best_map_uses_maximum_observed_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "epoch=5 target_mAP=0.35 best_mAP=0.35 best_epoch=5\n"
                "epoch=10 target_mAP=0.41 best_mAP=0.41 best_epoch=10\n"
                "epoch=30 target_mAP=0.39 best_mAP=0.41 best_epoch=10\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(best_map_from_log(path), 0.41)

    def test_missing_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("no metric\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "No best_mAP"):
                best_map_from_log(path)


if __name__ == "__main__":
    unittest.main()
