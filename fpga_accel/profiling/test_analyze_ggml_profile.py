import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_ggml_profile.py")


class ProfileAnalysisTest(unittest.TestCase):
    def test_stage_aggregation_and_roofline(self):
        log = """
inference stage = text_prefill
GGML_GRAPH_PROFILE_BEGIN graph=7 nodes=2 threads=4
GGML_OP_PROFILE graph=7 node=0 op=MUL_MAT name=Qcur-0 out_type=f32 out_ne=4x2x1x1 src0=w src0_type=q8_0 src0_ne=8x4x1x1 src1=x src1_type=f32 src1_ne=8x2x1x1 src2=- src2_type=- src2_ne=0x0x0x0 elapsed_us=100 flops=128 bytes=64
GGML_OP_PROFILE graph=7 node=1 op=SWIGLU name=ffn_swiglu-0 out_type=f32 out_ne=4x2x1x1 src0=a src0_type=f32 src0_ne=4x2x1x1 src1=b src1_type=f32 src1_ne=4x2x1x1 src2=- src2_type=- src2_ne=0x0x0x0 elapsed_us=20 flops=0 bytes=96
GGML_OP_PROFILE graph=7 node=2 op=FLASH_ATTN_EXT name=attn out_type=f32 out_ne=8x2x4x1 src0=q src0_type=f32 src0_ne=8x4x2x1 src1=k src1_type=f16 src1_ne=8x4x2x1 src2=v src2_type=f16 src2_ne=8x4x2x1 elapsed_us=10 flops=0 bytes=256
GGML_GRAPH_PROFILE_END graph=7 status=0
""".strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "profile.log"
            output_json = root / "summary.json"
            output_md = root / "summary.md"
            source.write_text(log + "\n", encoding="utf-8")
            result = subprocess.run(
                ["python3", str(SCRIPT), str(source), "--json", str(output_json),
                 "--markdown", str(output_md), "--peak-gflops", "10", "--bandwidth-gbps", "2"],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("PROFILE_ANALYSIS_PASS", result.stdout)
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["stage_time_ms"]["text_prefill"], 0.13)
            self.assertEqual(summary["operators"][0]["op"], "MUL_MAT")
            self.assertEqual(summary["operators"][0]["roofline_bound"], "memory")
            attention = next(row for row in summary["operators"] if row["op"] == "FLASH_ATTN_EXT")
            self.assertEqual(attention["flops"], 4 * 8 * 4 * 4 * 2)
            self.assertIn("text_prefill", output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
