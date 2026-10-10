#!/usr/bin/env python3
"""Summarize directed control RTL and generated DATAFLOW structure evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
RUN_ROOT = ROOT / "artifacts/compute_rtl_smoke"
FINAL_RUN = RUN_ROOT / "20260829T132655Z"
RTL_ROOT = ROOT / "build/q8q8_pe16_dataflow_prj/solution1/syn/verilog"
DATAFLOW_RTL = RTL_ROOT / "q8q8_gemm_p_anonymous_namespace_process_n_tile_dataflow.v"
COMPUTE_RTL = RTL_ROOT / "q8q8_gemm_p_anonymous_namespace_compute_cached_tile4.v"
OUTPUT_JSON = ROOT / "artifacts/pe16_dataflow_rtl_summary.json"
OUTPUT_MD = WORKSPACE / "reports/2026-08-29-zcu102-q8q8-pe16-dataflow-rtl.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> dict[str, object]:
    text = "\n".join(
        item.read_text(encoding="utf-8", errors="replace")
        for item in sorted(path.glob("*.log"))
    )
    if "RTL_CONTROL_RESULT=PASS" in text:
        status = "PASS"
    elif "Module <q8q8_gemm_mac_muladd" in text:
        status = "FAIL_MISSING_ADDRESS_MAC_DEPENDENCY"
    elif "unexpected exception" in text:
        status = "FAIL_XSIM_SANDBOX_RUNTIME"
    else:
        status = "INCOMPLETE"
    finish = re.search(r"\$finish called at time\s*:\s*(\d+) ps", text)
    return {
        "run_id": path.name,
        "path": str(path),
        "status": status,
        "finish_time_ps": int(finish.group(1)) if finish else None,
        "log_sha256": {
            item.name: sha256(item) for item in sorted(path.glob("*.log"))
        },
    }


def main() -> int:
    for output in (OUTPUT_JSON, OUTPUT_MD):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite record: {output}")
    runs = [classify(path) for path in sorted(RUN_ROOT.iterdir()) if path.is_dir()]
    final = classify(FINAL_RUN)
    if final["status"] != "PASS":
        raise RuntimeError("final directed RTL run did not pass")

    text = DATAFLOW_RTL.read_text(encoding="utf-8", errors="replace")
    requirements = {
        "load_process": "q8q8_gemm_p_anonymous_namespace_load_activation_stream " in text,
        "compute_process": "q8q8_gemm_p_anonymous_namespace_compute_streamed_tiles " in text,
        "store_process": "q8q8_gemm_p_anonymous_namespace_store_output_stream " in text,
        "activation_q_fifo_256x4864": "q8q8_gemm_fifo_w256_d4864_A activation_q_stream_U" in text,
        "activation_scale_fifo_32x4864": "q8q8_gemm_fifo_w32_d4864_A activation_scale_stream_U" in text,
        "output_fifo_128x256": "q8q8_gemm_fifo_w128_d256_A output_stream_U" in text,
        "load_to_activation_q_fifo": (
            ".if_write(p_anonymous_namespace_load_activation_stream_U0_activation_q_stream_write)" in text
        ),
        "activation_q_fifo_to_compute": (
            ".if_read(p_anonymous_namespace_compute_streamed_tiles_U0_activation_q_stream_read)" in text
        ),
        "compute_to_output_fifo": (
            ".if_write(p_anonymous_namespace_compute_streamed_tiles_U0_output_stream_write)" in text
        ),
        "output_fifo_to_store": (
            ".if_read(p_anonymous_namespace_store_output_stream_U0_output_stream_read)" in text
        ),
    }
    structure_pass = all(requirements.values())
    decision = (
        "DIRECTED_RTL_AND_DATAFLOW_STRUCTURE_PASS / EXPORT_IP_NEXT"
        if structure_pass else "RTL_STRUCTURE_FAIL / STOP"
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "directed_compute_control_rtl": {
            "status": final["status"],
            "shape": {"m": 3, "n": 64, "k": 64, "k_blocks": 2},
            "banks": 16,
            "writes_per_bank": 24,
            "valid_addresses_per_bank": 12,
            "writes_per_valid_address": 2,
            "accumulator_initialization": "fused into kb=0 write",
            "arithmetic": "behavioral zero-data stubs; control/address validation only",
            "final_run": final,
        },
        "generated_dataflow_structure": {
            "status": "PASS" if structure_pass else "FAIL",
            "requirements": requirements,
            "rtl": str(DATAFLOW_RTL),
            "rtl_sha256": sha256(DATAFLOW_RTL),
        },
        "attempt_history": runs,
        "limitations": [
            "Numerical arithmetic is validated by the 9-case C testbench, not this RTL smoke test.",
            "Top-level AXI RTL cosim was not retried because the frozen PE16 baseline showed about 30.7 GiB generated models and OOM risk.",
            "Generated DATAFLOW connectivity is statically checked; AXI back-pressure behavior awaits system implementation and board tests.",
            "No bitstream was generated and the board was not accessed.",
        ],
        "input_sha256": {
            "compute_rtl": sha256(COMPUTE_RTL),
            "testbench": sha256(ROOT / "tb/tb_compute_control_dataflow_rtl.sv"),
            "address_stubs": sha256(ROOT / "tb/address_control_stubs.sv"),
            "runner": sha256(ROOT / "scripts/run_compute_rtl_smoke.sh"),
        },
        "output_json": str(OUTPUT_JSON),
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md = f"""# ZCU102 Q8×Q8 PE16 DATAFLOW RTL门控

生成时间：{summary['generated_at']}
状态：`{decision}`

## 定向XSim

- HLS生成的融合初始化计算控制RTL通过：16个PE bank各写24次，覆盖12个合法地址，每地址写2次。
- 测试shape为`M=3,N=64,K=64`；完成时间{final['finish_time_ps']} ps。
- 算术单元使用行为stub，本项证明启动/完成握手、bank选择、地址遍历和`kb=0`直接写；数值由9组C testbench覆盖。
- 沙箱内XSim会在时间仿真入口发生工具内部异常；相同snapshot在沙箱外运行通过。失败和通过目录均保留。

## DATAFLOW生成结构

- RTL中同时存在load、compute、store三个独立process。
- 激活Q FIFO：256 bit × 4864；激活scale FIFO：32 bit × 4864；输出FIFO：128 bit × 256。
- 静态连接检查确认`load → activation FIFO → compute → output FIFO → store`完整。

## 证据边界

没有重试已知会产生约30.7 GiB模型并有OOM风险的顶层AXI cosim；本阶段也没有导出bitstream或访问板卡。下一步先导出独立HLS IP，再进入Vivado布局布线门控。

机器可读证据：`{OUTPUT_JSON}`。
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"DECISION={decision}")
    print(f"JSON={OUTPUT_JSON}")
    print(f"MARKDOWN={OUTPUT_MD}")
    return 0 if structure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
