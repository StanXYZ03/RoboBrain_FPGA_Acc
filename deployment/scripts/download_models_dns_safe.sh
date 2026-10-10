#!/bin/sh
# Download pinned artifacts without changing system DNS. Content authenticity is
# enforced by exact Hugging Face LFS SHA-256 values in config/model.json.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT/deployment/models}"
POLICY="${MODEL_POLICY:-$ROOT/deployment/config/model.json}"
HF_DOMAIN="huggingface.co"
CDN_DOMAIN="us.aws.cdn.hf.co"

json_value() {
    role="$1"; key="$2"
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(next(x for x in p["files"] if x["role"]==sys.argv[2])[sys.argv[3]])' \
        "$POLICY" "$role" "$key"
}

resolve_doh() {
    domain="$1"
    curl -L --fail -4 --connect-timeout 15 --max-time 60 --silent --show-error \
        "https://dns.google/resolve?name=$domain&type=A" |
        python3 -c 'import json,sys; r=json.load(sys.stdin); print(next(x["data"] for x in r.get("Answer",[]) if x.get("type")==1))'
}

revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$POLICY")"
repo="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_id"])' "$POLICY")"
hf_ip="$(resolve_doh "$HF_DOMAIN")"
cdn_ip="$(resolve_doh "$CDN_DOMAIN")"
mkdir -p "$MODEL_DIR"
echo "DNS_SAFE_RESOLVE $HF_DOMAIN=$hf_ip $CDN_DOMAIN=$cdn_ip"

for role in model mmproj; do
    name="$(json_value "$role" name)"
    expected_bytes="$(json_value "$role" bytes)"
    expected_sha="$(json_value "$role" sha256)"
    final="$MODEL_DIR/$name"
    part="$final.part"
    if [ ! -f "$final" ]; then
        curl -L --fail -4 \
            --connect-to "$HF_DOMAIN:443:$hf_ip:443" \
            --connect-to "$CDN_DOMAIN:443:$cdn_ip:443" \
            --connect-timeout 30 --retry 8 --retry-delay 5 \
            --continue-at - --output "$part" \
            "https://$HF_DOMAIN/$repo/resolve/$revision/$name?download=true"
        actual_bytes="$(wc -c < "$part")"
        [ "$actual_bytes" = "$expected_bytes" ] || {
            echo "DOWNLOAD_FAIL $name bytes=$actual_bytes expected=$expected_bytes" >&2
            exit 1
        }
        actual_sha="$(sha256sum "$part" | awk '{print $1}')"
        [ "$actual_sha" = "$expected_sha" ] || {
            echo "DOWNLOAD_FAIL $name sha256=$actual_sha expected=$expected_sha" >&2
            exit 1
        }
        mv "$part" "$final"
    fi
    echo "DOWNLOAD_PRESENT $name"
done

python3 "$ROOT/deployment/scripts/model_gate.py" verify \
    --policy "$POLICY" --directory "$MODEL_DIR"
echo "DNS_SAFE_DOWNLOAD_PASS directory=$MODEL_DIR revision=$revision"
