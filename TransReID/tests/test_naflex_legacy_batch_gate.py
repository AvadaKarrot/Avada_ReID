import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from tools.run_naflex_legacy_bs256_gate import build_command
from utils.naflex_legacy_batch_gate import load_gate, validate_gate


PROJECT_DIR = Path(__file__).resolve().parents[1]
GATE_PATH = PROJECT_DIR / "configs/experiments/gates/naflex_m_to_ms_legacy_bs256_60ep.json"


class LegacyBatchGateTest(unittest.TestCase):
    def test_gate_contract_and_order(self):
        gate = load_gate(GATE_PATH)
        self.assertEqual(gate["global_batch_size"], 256)
        self.assertEqual(gate["world_size"], 2)
        self.assertEqual([run["name"] for run in gate["runs"]], [
            "image_only", "caption_pid", "caption_codebook_delayed",
            "caption_codebook_relation",
        ])

    def test_command_is_two_rank_global_256_and_runtime_relocatable(self):
        gate = load_gate(GATE_PATH)
        root = PurePosixPath("/srv/reid-runtime")
        run = gate["runs"][3]
        command = build_command(gate, run, root)
        joined = " ".join(command)
        self.assertIn("--nproc_per_node=2", command)
        self.assertIn("SOLVER.IMS_PER_BATCH 256", joined)
        self.assertIn("SOLVER.MAX_EPOCHS 60", joined)
        self.assertIn("OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 6", joined)
        self.assertIn("OBJECTIVE.ATTRIBUTE_RELATION.START_EPOCH 6", joined)
        self.assertIn(str(root / "precomputed/attribute_codebooks/market1501_pr1/codebook.pt"), command)

    def test_rejects_schedule_drift(self):
        gate = load_gate(GATE_PATH)
        altered = json.loads(json.dumps(gate))
        altered["solver_overrides"]["MAX_EPOCHS"] = 30
        with self.assertRaisesRegex(ValueError, "MAX_EPOCHS"):
            validate_gate(altered)


if __name__ == "__main__":
    unittest.main()
