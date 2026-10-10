import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RevealedDiagnosticCliTest(unittest.TestCase):
    def test_revealed_mode_requires_expected_masks(self):
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluation/run_hf_cutpoint_oracle.py"),
                "--model", "/missing-model",
                "--model-label", "test",
                "--manifest", str(ROOT / "datasets/dac_spatial_l2res_blind.jsonl"),
                "--output-dir", "/tmp/not-created",
                "--allow-revealed-diagnostic",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("requires --expected-masks", process.stderr)


if __name__ == "__main__":
    unittest.main()
