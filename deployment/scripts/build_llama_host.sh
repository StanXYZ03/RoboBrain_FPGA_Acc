#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SOURCE_DIR="${LLAMA_SOURCE_DIR:-$ROOT/third_party/llama.cpp}"
BUILD_DIR="${LLAMA_HOST_BUILD_DIR:-$ROOT/build/llama-host}"
OUTPUT_DIR="${LLAMA_HOST_OUTPUT_DIR:-$ROOT/deployment/artifacts/host}"
LLAMA_COMMIT="${LLAMA_COMMIT:-178a6c44937154dc4c4eff0d166f4a044c4fceba}"
VITIS_ROOT="${VITIS_ROOT:-/home/xilinx/Vitis/2023.2}"
CMAKE="${CMAKE:-$VITIS_ROOT/tps/lnx64/cmake-3.24.2/bin/cmake}"
CMAKE_LD_DIR="$VITIS_ROOT/tps/lnx64/cmake-3.24.2/libs/Ubuntu"

[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$LLAMA_COMMIT" ] || {
    echo "llama.cpp checkout is not pinned to $LLAMA_COMMIT" >&2; exit 1;
}
LD_LIBRARY_PATH="$CMAKE_LD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CMAKE" -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_SERVER=ON
LD_LIBRARY_PATH="$CMAKE_LD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CMAKE" --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-4}" --target llama-server llama-cli
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_DIR/bin/llama-server" "$OUTPUT_DIR/llama-server"
cp "$BUILD_DIR/bin/llama-cli" "$OUTPUT_DIR/llama-cli"
strip "$OUTPUT_DIR/llama-server" "$OUTPUT_DIR/llama-cli"
"$OUTPUT_DIR/llama-server" --version
(cd "$OUTPUT_DIR" && sha256sum llama-server llama-cli > SHA256SUMS)
printf '%s\n' "$LLAMA_COMMIT" > "$OUTPUT_DIR/LLAMA_COMMIT"
echo "HOST_BUILD_PASS output=$OUTPUT_DIR commit=$LLAMA_COMMIT"
