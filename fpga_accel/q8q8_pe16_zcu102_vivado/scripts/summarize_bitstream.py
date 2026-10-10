#!/usr/bin/env python3
"""Freeze PE16 bitstream-generation evidence without accessing the board."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT_JSON = ARTIFACTS / "pe16_bitstream_summary.json"
OUTPUT_MD = WORKSPACE / "reports/2026-08-28-zcu102-q8q8-pe16-bitstream.md"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def violation_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    total_match = re.search(r"Violations found:\s*(\d+)", text)
    if total_match is None:
        raise ValueError(f"missing DRC violation count: {path}")
    result = {"total": int(total_match.group(1)), "Warning": 0,
              "Advisory": 0, "Error": 0}
    for severity, count in re.findall(
        r"^\|\s*[A-Z0-9-]+\s*\|\s*([A-Za-z]+)\s*\|.*\|\s*(\d+)\s*\|$",
        text,
        flags=re.MULTILINE,
    ):
        result[severity] = result.get(severity, 0) + int(count)
    return result


def main() -> int:
    summary_path = ARTIFACTS / "implementation_summary.txt"
    bit_path = ARTIFACTS / "q8q8_pe16_zcu102.bit"
    xsa_path = ARTIFACTS / "q8q8_pe16_zcu102.xsa"
    dcp_path = ARTIFACTS / "q8q8_pe16_zcu102_routed.dcp"
    timing_path = ARTIFACTS / "impl_timing_summary.rpt"
    route_path = ARTIFACTS / "impl_route_status.rpt"
    drc_path = ARTIFACTS / "impl_drc.rpt"
    methodology_path = ARTIFACTS / "impl_methodology.rpt"
    required = (
        summary_path,
        bit_path,
        xsa_path,
        dcp_path,
        timing_path,
        route_path,
        drc_path,
        methodology_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifacts: {missing}")

    values = key_values(summary_path)
    with zipfile.ZipFile(xsa_path) as archive:
        names = archive.namelist()
        embedded_bit = archive.read("q8q8_pe16_zcu102.bit")

    bit_hash = sha256_file(bit_path)
    embedded_hash = sha256_bytes(embedded_bit)
    timing_text = timing_path.read_text(encoding="utf-8")
    route_text = route_path.read_text(encoding="utf-8")
    drc = violation_counts(drc_path)
    methodology = violation_counts(methodology_path)
    checks = {
        "summary_says_generated": values.get("BITSTREAM_GENERATED") == "YES",
        "summary_says_board_not_programmed": values.get("BOARD_PROGRAMMED") == "NO",
        "timing_pass": (
            float(values["WNS_NS"]) >= 0.0
            and float(values["WHS_NS"]) >= 0.0
            and "All user specified timing constraints are met." in timing_text
        ),
        "route_pass": (
            int(values["UNROUTED_NETS"]) == 0
            and "# of nets with routing errors.......... :           0" in route_text
        ),
        "drc_zero_error": drc.get("Error", 0) == 0,
        "xsa_contains_bit": "q8q8_pe16_zcu102.bit" in names,
        "xsa_bit_matches_standalone": bit_hash == embedded_hash,
        "output_axi_128": int(values["OUTPUT_AXI_WIDTH"]) == 128,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "offline PE16 bitstream/XSA integrity; no board access",
        "decision": (
            "PE16_BITSTREAM_PASS / BOARD_AUTHORIZATION_REQUIRED"
            if passed
            else "PE16_BITSTREAM_FAIL / BOARD_PROGRAMMING_FORBIDDEN"
        ),
        "checks": checks,
        "implementation": {
            "target_mhz": float(values["TARGET_MHZ"]),
            "actual_mhz": float(values["ACTUAL_MHZ"]),
            "wns_ns": float(values["WNS_NS"]),
            "whs_ns": float(values["WHS_NS"]),
            "unrouted_nets": int(values["UNROUTED_NETS"]),
            "output_axi_width": int(values["OUTPUT_AXI_WIDTH"]),
            "board_programmed": values["BOARD_PROGRAMMED"],
        },
        "drc": drc,
        "methodology": methodology,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in required
        },
        "xsa": {
            "members": len(names),
            "embedded_bit_bytes": len(embedded_bit),
            "embedded_bit_sha256": embedded_hash,
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ZCU102 Q8×Q8 PE16 bitstream离线验收",
        "",
        f"生成时间：{result['generated_at']}",
        f"状态：`{result['decision']}`",
        "",
        "## 结论",
        "",
        "- PE16完整系统已生成独立bitstream和含bitstream的XSA。",
        "- 本阶段只做文件生成与完整性检查；未连接、复位或烧写ZCU102。",
        "- 这不证明PE16板端数值正确或获得任何实测加速。",
        "",
        "## 门控结果",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if state else 'FAIL'} |"
        for name, state in checks.items()
    )
    lines += [
        "",
        "## 实现与产物",
        "",
        f"- 频率：{values['ACTUAL_MHZ']} MHz；WNS={values['WNS_NS']} ns；"
        f"WHS={values['WHS_NS']} ns；未布线网络={values['UNROUTED_NETS']}。",
        f"- DRC：Error {drc.get('Error', 0)}，Warning {drc.get('Warning', 0)}，"
        f"Advisory {drc.get('Advisory', 0)}。",
        f"- bitstream：{bit_path.stat().st_size} bytes，SHA256=`{bit_hash}`。",
        f"- XSA：{xsa_path.stat().st_size} bytes，SHA256=`{sha256_file(xsa_path)}`。",
        f"- XSA内嵌bit与独立bit：{'一致' if bit_hash == embedded_hash else '不一致'}。",
        "",
        "## 下一门",
        "",
        "先用PE16 IP重新核对PS runtime寄存器映射并冻结板测包；获得明确上板授权后，"
        "才可临时配置PL并依次运行确定性self-test、language_prefill、vision_encoder和"
        "decoder_gemv microbenchmark。",
        "",
        f"机器可读证据：`{OUTPUT_JSON}`。",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "checks": checks},
                     ensure_ascii=False))
    print(f"JSON={OUTPUT_JSON}")
    print(f"MARKDOWN={OUTPUT_MD}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
