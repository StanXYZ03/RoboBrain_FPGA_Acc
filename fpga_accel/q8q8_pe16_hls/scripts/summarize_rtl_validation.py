#!/usr/bin/env python3
"""Summarize PE16 RTL validation without overstating top-level cosim."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
RUN_ROOT = ROOT / "artifacts/compute_rtl_smoke"
COSIM_LOG = ROOT / "artifacts/cosim_small.log"
COSIM_SIM = ROOT / "build/q8q8_pe16_cosim_prj/solution1/sim/verilog"
OUTPUT_JSON = ROOT / "artifacts/pe16_rtl_validation_summary.json"
OUTPUT_MD = WORKSPACE / "reports/2026-08-27-zcu102-q8q8-pe16-rtl-validation.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_run(path: Path) -> dict[str, Any]:
    compile_log = path / "compile.log"
    elaborate_log = path / "elaborate.log"
    simulation_log = path / "simulation.log"
    texts = {
        "compile": compile_log.read_text(errors="replace")
        if compile_log.exists()
        else "",
        "elaborate": elaborate_log.read_text(errors="replace")
        if elaborate_log.exists()
        else "",
        "simulation": simulation_log.read_text(errors="replace")
        if simulation_log.exists()
        else "",
    }
    merged = "\n".join(texts.values())
    if "RTL_CONTROL_RESULT=PASS" in merged:
        status = "PASS"
    elif "RTL_CONTROL_RESULT=FAIL reason=timeout" in merged:
        status = "FAIL_HANDSHAKE_TIMEOUT"
    elif "flow_control_loop_pipe_sequential_init> not found" in merged:
        status = "FAIL_MISSING_FLOW_CONTROL_RTL"
    else:
        status = "INCOMPLETE"
    finish_match = re.search(r"\$finish called at time\s*:\s*(\d+) ps", merged)
    return {
        "run_id": path.name,
        "path": str(path),
        "status": status,
        "finish_time_ps": int(finish_match.group(1)) if finish_match else None,
        "log_sha256": {
            item.name: sha256(item)
            for item in (compile_log, elaborate_log, simulation_log)
            if item.exists()
        },
    }


def main() -> int:
    runs = [classify_run(path) for path in sorted(RUN_ROOT.iterdir()) if path.is_dir()]
    passed = [run for run in runs if run["status"] == "PASS"]
    if not passed:
        raise RuntimeError("no passing compute RTL control run found")
    final_run = passed[-1]

    cosim_text = COSIM_LOG.read_text(errors="replace")
    c_case_names = set(
        re.findall(
            r"^TB_CASE name=([^ ]+) .* result=PASS$",
            cosim_text,
            re.MULTILINE,
        )
    )
    axi_files = sorted(COSIM_SIM.glob("AESL_axi_master_*.v"))
    axi_bytes = sum(path.stat().st_size for path in axi_files)
    sim_bytes = sum(
        path.stat().st_size for path in COSIM_SIM.rglob("*") if path.is_file()
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "PE16 numerical C test, stopped top-level AXI cosim, and directed "
            "HLS-generated compute-controller RTL simulation"
        ),
        "numerical_csim": {
            "status": "PASS",
            "full_capacity_cases": 7,
            "rtl_smoke_c_cases_before_cosim": len(c_case_names),
            "max_abs_error": 2.50339508e-05,
        },
        "top_level_axi_cosim": {
            "status": "STOPPED_RESOURCE_RISK",
            "rtl_result_available": False,
            "generated_axi_model_count": len(axi_files),
            "generated_axi_model_bytes": axi_bytes,
            "simulation_directory_bytes": sim_bytes,
            "observed_xelab_rss_kib_after_first_axi_model": 20515048,
            "reason_zh": (
                "Vitis HLS 2023.2生成约30 GiB AXI仿真Verilog；xelab仅解析首个"
                "适配器时RSS已约20.5 GiB。为避免整机OOM主动停止，不能记为RTL通过。"
            ),
            "log": str(COSIM_LOG),
            "log_sha256": sha256(COSIM_LOG),
        },
        "directed_compute_control_rtl": {
            "status": "PASS",
            "run": final_run,
            "test_shape": {"m": 3, "n": 64, "k": 64, "k_blocks": 2},
            "pe_banks": 16,
            "writes_per_bank": 24,
            "valid_addresses_per_bank": 12,
            "writes_per_valid_address": 2,
            "arithmetic_model": "zero-data behavioral stubs",
            "validated": [
                "HLS-generated pipeline control",
                "M-row interleaved work traversal",
                "16-bank accumulator addressing",
                "valid-row gating",
                "ap_ctrl_hs start-ready-done handshake",
            ],
            "not_validated": [
                "RTL floating-point/integer numerical arithmetic",
                "top-level AXI protocol behavior",
                "board performance",
            ],
        },
        "attempt_history": runs,
        "decision": (
            "COMPUTE_CONTROL_RTL_PASS / TOP_AXI_COSIM_RESOURCE_STOP / "
            "BITSTREAM_GENERATION_NEXT"
        ),
        "output_json": str(OUTPUT_JSON),
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    gib = sim_bytes / 1024**3
    md = f"""# ZCU102 Q8×Q8 PE16 RTL验证报告

生成时间：{result['generated_at']}
状态：`{result['decision']}`

## 结论边界

- 全尺寸C数值测试7/7通过，最大绝对误差`2.50339508e-05`。
- Vitis HLS顶层AXI cosim没有完成RTL阶段，不能记为通过。
- 生成的`compute_cached_tile`控制RTL完成定向XSim；算术单元使用零数据行为stub，因此本项只证明控制、地址和有效位调度。
- PE16板端数值和性能尚未测量。

## 顶层cosim工具限制

- HLS的C侧RTL smoke 5/5通过，小容量综合保持II=1。
- Vitis HLS 2023.2生成{len(axi_files)}个AXI主端口模型，仿真目录约{gib:.2f} GiB。
- `xelab`仅解析首个模型时RSS采样已约20.5 GiB；为避免OOM主动停止。
- 原始日志：`{COSIM_LOG}`。

## 定向计算控制RTL

- 测试shape：`M=3, N=64, K=64`，16个PE bank。
- 每个bank写24次，覆盖12个合法地址，每个地址写2次。
- XSim完成时间：{final_run['finish_time_ps']} ps。
- 最终原始目录：`{final_run['path']}`。
- 前两次失败分别用于补齐flow-control RTL依赖和修正`ap_start/ap_ready`握手，日志保留且未覆盖。

## 决策

PE16的新M-row交错控制及16-bank累加地址映射通过生成RTL的定向验证；数值正确性由C testbench覆盖。顶层AXI cosim因工具资源风险停止。结合独立完整系统报告已经通过的200 MHz routed门控，下一步可以生成bitstream，再以板端self-test补足真实数值与性能证据；不得将本报告写成PE16板端通过。

机器可读证据：`{OUTPUT_JSON}`。
"""
    OUTPUT_MD.write_text(md.replace("`", chr(96)), encoding="utf-8")
    print(f"JSON={OUTPUT_JSON}")
    print(f"MARKDOWN={OUTPUT_MD}")
    print(f"FINAL_RTL_RUN={final_run['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
