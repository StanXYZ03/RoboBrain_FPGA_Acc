#!/usr/bin/env python3
"""Deterministic remote text acceptance test for llama-server."""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit


PROMPTS = (
    "Reply with only the word OK.",
    "用两个汉字回答：天空通常是什么颜色？",
    "What is 1+1? Answer with only the number.",
)


class DeviceBoundHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, interface: str, **kwargs):
        self.interface = interface
        super().__init__(*args, **kwargs)

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
            self.interface.encode("ascii") + b"\0",
        )
        self.sock.connect((self.host, self.port))


def request_json(
    url: str, payload: dict | None = None, timeout: float = 30,
    interface: str = "enp4s0",
) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError(f"only numeric HTTP URLs are supported: {url}")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    connection = DeviceBoundHTTPConnection(
        parsed.hostname, parsed.port or 80, timeout=timeout, interface=interface
    )
    try:
        connection.request(
            "POST" if data is not None else "GET",
            parsed.path or "/", body=data, headers=headers,
        )
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise OSError(f"HTTP {response.status}: {body[:200]!r}")
        return json.loads(body.decode("utf-8"))
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://172.21.127.254:8080")
    parser.add_argument("--interface", default="enp4s0")
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            health = request_json(
                args.base_url + "/health", timeout=5, interface=args.interface
            )
            if health.get("status") == "ok":
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            print("FAIL: llama-server did not become healthy", file=sys.stderr)
            return 1
        time.sleep(2)

    records = []
    for prompt in PROMPTS + (PROMPTS[0],):
        started = time.monotonic()
        response = request_json(
            args.base_url + "/v1/chat/completions",
            {
                "model": "RoboBrain2.5-4B.Q8_0.gguf",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
            },
            timeout=1800,
            interface=args.interface,
        )
        try:
            content = response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"malformed response: {response}") from exc
        if not content:
            raise RuntimeError(f"empty response for {prompt!r}")
        record = {
            "prompt": prompt, "response": content,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "usage": response.get("usage", {}),
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        records.append(record)
    if records[0]["response"] != records[-1]["response"]:
        raise RuntimeError("temperature=0 repeat was not deterministic")
    report = {"health": health, "tests": records, "status": "PASS"}
    if args.result:
        args.result.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("CHAT_TEST_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
