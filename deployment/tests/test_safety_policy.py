from __future__ import annotations

import re
import unittest
from pathlib import Path


BOARD = Path(__file__).resolve().parents[1] / "board"


class SafetyPolicyTests(unittest.TestCase):
    def test_board_scripts_contain_no_sd_mutation_commands(self) -> None:
        forbidden = (
            r"\bmkfs(?:\.|\s)", r"\bfdisk\b", r"\bparted\b",
            r"\bsaveenv\b", r"\bdd\s+[^\n]*\bof=", r"\bmount\s+[^\n]*\brw\b",
        )
        content = "\n".join(path.read_text(encoding="utf-8") for path in BOARD.glob("*.sh"))
        for pattern in forbidden:
            self.assertIsNone(re.search(pattern, content), pattern)

    def test_guard_uses_kernel_ro_full_hash_and_write_counters(self) -> None:
        guard = (BOARD / "readonly_sd_guard.sh").read_text(encoding="utf-8")
        final = (BOARD / "verify_sd_unchanged.sh").read_text(encoding="utf-8")
        self.assertIn("blockdev --setro", guard)
        self.assertIn('sha256sum "$MMC_DEVICE"', guard)
        self.assertIn("write_sectors", guard)
        self.assertIn("media_sha256", final)
        self.assertIn("write_counters_unchanged=1", final)

    def test_vision_server_loads_cpu_mmproj_with_image_context(self) -> None:
        start = (BOARD / "start_text_server.sh").read_text(encoding="utf-8")
        self.assertIn(
            '--mmproj "$RUNTIME_DIR/RoboBrain2.5-4B.mmproj-Q8_0.gguf"',
            start,
        )
        self.assertIn("--no-mmproj-offload", start)
        self.assertIn("--image-min-tokens 1024", start)
        self.assertIn("--image-max-tokens 1024", start)
        self.assertIn("--batch-size 2048", start)
        self.assertIn("--ubatch-size 512", start)
        self.assertIn("--metrics", start)
        self.assertIn("--slots", start)
        self.assertIn("--perf", start)
        self.assertIn("--log-timestamps", start)
        self.assertNotIn("--ctx-size 128", start)

    def test_no_uart_boot_streams_logs_before_sd_hash(self) -> None:
        boot = (BOARD / "boot_entrypoint.sh").read_text(encoding="utf-8")
        self.assertLess(boot.index("early_network.sh"), boot.index("readonly_sd_guard.sh"))
        self.assertLess(boot.index("nc \"$HOST_IP\""), boot.index("readonly_sd_guard.sh"))

    def test_final_attestation_is_read_only(self) -> None:
        server = (BOARD / "final_attestation_server.sh").read_text(encoding="utf-8")
        self.assertIn("verify_sd_unchanged.sh", server)
        self.assertIn('"$CONTROL_ROOT/cgi-bin/runtime"', server)
        self.assertIn("tail -n 200 /dev/shm/robobrain-boot.log", server)
        self.assertNotRegex(server, r"\b(mkfs|fdisk|parted|saveenv)\b")

    def test_board_network_uses_proven_no_sudo_subnet(self) -> None:
        early = (BOARD / "early_network.sh").read_text(encoding="utf-8")
        self.assertIn("172.21.127.254/17", early)


if __name__ == "__main__":
    unittest.main()
