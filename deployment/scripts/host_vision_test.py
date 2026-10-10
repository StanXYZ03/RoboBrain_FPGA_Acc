#!/usr/bin/env python3
"""Send one image over Ethernet to a llama.cpp OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import mimetypes
import queue
import re
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


POINT_RE = re.compile(r"\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)")
DEFAULT_PROMPT = (
    "the affordance area for holding the cup. Please provide its 2D "
    "coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], "
    "where x and y are normalized integer coordinates from 0 to 1000."
)
STAGE_MARKERS = (
    "decoding image batch",
    "image decoded",
    "slice encoded",
    "prompt eval",
    "prompt processing",
    "eval time",
)


class BoundHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, interface: str = "", **kwargs):
        self.interface = interface
        super().__init__(*args, **kwargs)

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        if self.interface:
            self.sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BINDTODEVICE,
                self.interface.encode("ascii") + b"\0",
            )
        self.sock.connect((self.host, self.port))


def request_json(
    url: str,
    payload: dict | None = None,
    *,
    timeout: float,
    interface: str,
) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError(f"only HTTP URLs are supported: {url}")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    connection = BoundHTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=timeout,
        interface=interface,
    )
    try:
        connection.request(
            "POST" if data is not None else "GET",
            parsed.path or "/",
            body=data,
            headers=headers,
        )
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise OSError(f"HTTP {response.status}: {body[:500]!r}")
        return json.loads(body.decode("utf-8"))
    finally:
        connection.close()


def request_text(url: str, *, timeout: float, interface: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError(f"only HTTP URLs are supported: {url}")
    connection = BoundHTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=timeout,
        interface=interface,
    )
    try:
        connection.request("GET", parsed.path or "/")
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise OSError(f"HTTP {response.status}: {body[:500]!r}")
        return body.decode("utf-8", errors="replace")
    finally:
        connection.close()


def image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"empty image: {path}")
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("image is larger than the 5 MiB acceptance-test limit")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"unsupported image type: {mime or path.suffix}")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_points(content: str) -> list[tuple[int, int]]:
    points = [(int(x), int(y)) for x, y in POINT_RE.findall(content)]
    if any(not (0 <= x <= 1000 and 0 <= y <= 1000) for x, y in points):
        raise RuntimeError(f"coordinate outside normalized 0..1000 range: {points}")
    return points


def read_log_observations(path: Path, offset: int) -> tuple[int, list[str]]:
    if not path.is_file():
        return offset, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(offset)
        text = stream.read()
        new_offset = stream.tell()
    lines = [
        line.strip() for line in text.splitlines()
        if any(marker in line.lower() for marker in STAGE_MARKERS)
    ]
    return new_offset, lines


def extract_stage_lines(text: str) -> list[str]:
    return [
        line.strip() for line in text.splitlines()
        if any(marker in line.lower() for marker in STAGE_MARKERS)
    ]


def parse_prometheus(text: str) -> dict[str, float]:
    metrics = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        name, value = line.split(None, 1)
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def request_worker(result_queue: queue.Queue, *args, **kwargs) -> None:
    try:
        result_queue.put((True, request_json(*args, **kwargs)))
    except BaseException as exc:  # Preserve the original request exception.
        result_queue.put((False, exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default="http://192.168.50.2:8080")
    parser.add_argument("--interface", default="enp4s0")
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--monitor-seconds", type=int, default=15)
    parser.add_argument(
        "--status-url",
        default="http://192.168.50.2:9001/cgi-bin/runtime",
    )
    parser.add_argument(
        "--board-log",
        type=Path,
        default=Path("deployment/reports/board-runtime.log"),
    )
    parser.add_argument("--require-point", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()

    if not args.image.is_file():
        raise ValueError(f"image does not exist: {args.image}")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.monitor_seconds < 1:
        raise ValueError("--monitor-seconds must be positive")

    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            health = request_json(
                args.base_url + "/health",
                timeout=5,
                interface=args.interface,
            )
            if health.get("status") == "ok":
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("vision server did not become healthy")
        time.sleep(2)

    payload = {
        "model": "RoboBrain2.5-4B.Q8_0.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri(args.image)}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    log_offset = args.board_log.stat().st_size if args.board_log.is_file() else 0
    observations = []
    seen_stages = set()
    result_queue: queue.Queue = queue.Queue(maxsize=1)
    request_thread = threading.Thread(
        target=request_worker,
        args=(result_queue, args.base_url + "/v1/chat/completions", payload),
        kwargs={"timeout": args.timeout_seconds, "interface": args.interface},
        daemon=True,
    )
    started = time.monotonic()
    request_thread.start()
    print(
        f"VISION_REQUEST_START image={args.image} monitor_seconds={args.monitor_seconds}",
        flush=True,
    )
    while True:
        try:
            succeeded, value = result_queue.get(timeout=args.monitor_seconds)
            break
        except queue.Empty:
            elapsed_now = round(time.monotonic() - started, 3)
            try:
                current_health = request_json(
                    args.base_url + "/health",
                    timeout=5,
                    interface=args.interface,
                ).get("status", "unknown")
            except (OSError, ValueError) as exc:
                current_health = f"unavailable:{type(exc).__name__}"
            log_offset, new_stage_lines = read_log_observations(
                args.board_log, log_offset
            )
            try:
                runtime_text = request_text(
                    args.status_url,
                    timeout=5,
                    interface=args.interface,
                )
                for line in extract_stage_lines(runtime_text):
                    if line not in seen_stages:
                        new_stage_lines.append(line)
            except (OSError, ValueError) as exc:
                runtime_text = f"unavailable:{type(exc).__name__}"
            new_stage_lines = [
                line for line in new_stage_lines if line not in seen_stages
            ]
            seen_stages.update(new_stage_lines)
            observation = {
                "elapsed_seconds": elapsed_now,
                "health": current_health,
                "stages": new_stage_lines,
                "runtime_status": runtime_text.splitlines()[:4],
            }
            observations.append(observation)
            latest = new_stage_lines[-1] if new_stage_lines else "no-new-stage"
            print(
                f"VISION_PROGRESS elapsed={elapsed_now}s health={current_health} "
                f"stage={latest}",
                flush=True,
            )
    if not succeeded:
        raise value
    response = value
    elapsed = round(time.monotonic() - started, 3)
    log_offset, final_stage_lines = read_log_observations(args.board_log, log_offset)
    if final_stage_lines:
        observations.append(
            {
                "elapsed_seconds": elapsed,
                "health": "ok",
                "stages": final_stage_lines,
            }
        )
    try:
        metrics = parse_prometheus(
            request_text(
                args.base_url + "/metrics",
                timeout=10,
                interface=args.interface,
            )
        )
    except (OSError, ValueError):
        metrics = {}
    try:
        content = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"malformed response: {response}") from exc
    if not content:
        raise RuntimeError("empty vision response")
    points = parse_points(content)
    if args.require_point and not points:
        raise RuntimeError(f"response contains no '(x, y)' point: {content!r}")

    report = {
        "status": "PASS",
        "health": health,
        "image": str(args.image.resolve()),
        "prompt": args.prompt,
        "response": content,
        "points": points,
        "elapsed_seconds": elapsed,
        "usage": response.get("usage", {}),
        "observations": observations,
        "metrics": metrics,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.result:
        args.result.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("VISION_TEST_PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
