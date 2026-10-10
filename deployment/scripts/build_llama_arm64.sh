#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
SOURCE_DIR="${LLAMA_SOURCE_DIR:-$ROOT/third_party/llama.cpp}"
BUILD_DIR="${LLAMA_BUILD_DIR:-$ROOT/build/llama-arm64}"
OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-$ROOT/deployment/artifacts/arm64}"
LLAMA_COMMIT="${LLAMA_COMMIT:-178a6c44937154dc4c4eff0d166f4a044c4fceba}"
VITIS_ROOT="${VITIS_ROOT:-/home/xilinx/Vitis/2023.2}"
CMAKE="${CMAKE:-$VITIS_ROOT/tps/lnx64/cmake-3.24.2/bin/cmake}"
CMAKE_LD_DIR="$VITIS_ROOT/tps/lnx64/cmake-3.24.2/libs/Ubuntu"
TOOLCHAIN="$VITIS_ROOT/gnu/aarch64/lin/aarch64-linux/bin"
CC="$TOOLCHAIN/aarch64-linux-gnu-gcc"
CXX="$TOOLCHAIN/aarch64-linux-gnu-g++"
SYSROOT="$($CC -print-sysroot)"
STRIP="$TOOLCHAIN/aarch64-linux-gnu-strip"

[ -x "$CMAKE" ] || { echo "missing cmake: $CMAKE" >&2; exit 1; }
[ -x "$CC" ] && [ -x "$CXX" ] || { echo "missing Vitis AArch64 compiler" >&2; exit 1; }
[ -d "$SOURCE_DIR/.git" ] || { echo "missing llama.cpp checkout: $SOURCE_DIR" >&2; exit 1; }
actual_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[ "$actual_commit" = "$LLAMA_COMMIT" ] || {
    echo "llama.cpp commit mismatch: $actual_commit != $LLAMA_COMMIT" >&2
    exit 1
}

LD_LIBRARY_PATH="$CMAKE_LD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CMAKE" -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_SYSROOT="$SYSROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -mcpu=cortex-a53" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -mcpu=cortex-a53" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8-a \
    -DGGML_OPENMP=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_LLAMAFILE=OFF \
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
"$STRIP" "$OUTPUT_DIR/llama-server" "$OUTPUT_DIR/llama-cli"
file "$OUTPUT_DIR/llama-server" | grep -q 'ARM aarch64' || {
    echo "built server is not AArch64" >&2; exit 1;
}
if readelf -d "$OUTPUT_DIR/llama-server" | grep -E 'NEEDED.*(libllama|libggml)'; then
    echo "llama/ggml shared-library dependency is forbidden" >&2
    exit 1
fi
(cd "$OUTPUT_DIR" && sha256sum llama-server llama-cli > SHA256SUMS)
printf '%s\n' "$LLAMA_COMMIT" > "$OUTPUT_DIR/LLAMA_COMMIT"
QEMU="$VITIS_ROOT/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin/qemu-aarch64"
if [ -x "$QEMU" ]; then
    "$QEMU" -L "$SYSROOT" "$OUTPUT_DIR/llama-server" --version
else
    echo "warning: qemu-aarch64 unavailable; on-host ARM64 execution check skipped" >&2
fi
echo "ARM64_BUILD_PASS output=$OUTPUT_DIR commit=$LLAMA_COMMIT"
