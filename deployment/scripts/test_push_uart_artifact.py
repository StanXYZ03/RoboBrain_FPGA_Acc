#!/usr/bin/env python3
"""Pure-host checks for the UART artifact framing and safety boundary."""

from __future__ import annotations

import base64
import gzip
import importlib.util
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deployment/scripts/push_uart_artifact.py"
SPEC = importlib.util.spec_from_file_location("push_uart_artifact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "payload"
        data = bytes(range(256)) * 19
        source.write_bytes(data)
        payload, digest, count = MODULE.make_payload(source, "/tmp/test-bin")
        assert count == len(data)
        assert len(digest) == 64
        begin = payload.index(b"\n", payload.index(b"RB_UART_EOF") + 11) + 1
        end = payload.index(b"\nRB_UART_EOF\n", begin)
        encoded = payload[begin:end].replace(b"\n", b"")
        assert gzip.decompress(base64.b64decode(encoded)) == data
        assert b"> /tmp/test-bin" in payload
        assert b"sha256sum /tmp/test-bin" in payload
        forbidden = (b"mmc", b"flashcp", b"mtd", b"qspi", b"/dev/sd")
        assert not any(token in payload.lower() for token in forbidden)
        help_text = subprocess.check_output(["python3", str(SCRIPT), "--help"], text=True)
        assert "--pass-marker" in help_text
    print("UART_PUSH_ARTIFACT_TEST_PASS roundtrip=1 ram_only=1 append_only=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
