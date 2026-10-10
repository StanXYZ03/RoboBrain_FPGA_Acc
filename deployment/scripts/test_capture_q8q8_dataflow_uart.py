#!/usr/bin/env python3
"""PTY-only test of the DATAFLOW UART capture/command state machine."""

from __future__ import annotations

import os
import pty
import select
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "deployment/scripts/capture_q8q8_dataflow_uart.py"


def main() -> int:
    master, slave = pty.openpty()
    device = os.ttyname(slave)
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "uart.log"
        process = subprocess.Popen(
            [str(CAPTURE), "--device", device, "--log", str(log),
             "--boot-timeout", "5", "--measure-timeout", "5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.1)
        os.write(master, b"boot\r\nRAM_LINUX_PL_SELFTEST_HOLD uart_shell=enabled\r\n")
        command = b""
        deadline = time.monotonic() + 3
        while b"/opt/robobrain/q8q8-dataflow-measure\r" not in command:
            assert time.monotonic() < deadline, command
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                command += os.read(master, 4096)
        os.write(master, b"DATAFLOW_MEASURE_BEGIN protocol=1 dry_run=0\r\n")
        os.write(master, b"DATAFLOW_MEASURE_PASS protocol=1\r\n")
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        assert b"DATAFLOW_UART_CAPTURE_PASS protocol=1" in stdout
        evidence = log.read_bytes()
        assert b"RAM_LINUX_PL_SELFTEST_HOLD" in evidence
        assert b"DATAFLOW_MEASURE_PASS protocol=1" in evidence
    os.close(master)
    os.close(slave)
    print("DATAFLOW_UART_CAPTURE_TEST_PASS command_after_boot=1 append_only=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

