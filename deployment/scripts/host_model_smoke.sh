#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/deployment/models}"
LOCK="${MODEL_LOCK:-$MODEL_DIR/model.lock.json}"
SERVER="${HOST_LLAMA_SERVER:-$ROOT/deployment/artifacts/host/llama-server}"
REPORT_DIR="${REPORT_DIR:-$ROOT/deployment/reports}"
PORT="${HOST_TEST_PORT:-18080}"

python3 "$ROOT/deployment/scripts/model_gate.py" verify --directory "$MODEL_DIR" --lock "$LOCK"
[ -x "$SERVER" ] || { echo "missing host llama-server: $SERVER" >&2; exit 1; }
mkdir -p "$REPORT_DIR"
"$SERVER" -m "$MODEL_DIR/RoboBrain2.5-4B.Q8_0.gguf" \
    --ctx-size 128 --batch-size 8 --ubatch-size 8 \
    --threads 4 --threads-batch 4 --parallel 1 --jinja \
    --host 127.0.0.1 --port "$PORT" > "$REPORT_DIR/host-llama-server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT INT TERM
python3 "$ROOT/deployment/scripts/host_chat_test.py" \
    --base-url "http://127.0.0.1:$PORT" --wait-seconds 900 \
    --result "$REPORT_DIR/host-chat.json"
echo "HOST_MODEL_SMOKE_PASS report=$REPORT_DIR/host-chat.json"
