# g++ C-simulation and HLS C-simulation for the Selector -> LayerNorm-Q8_0
# adapter and the independent QKV bias-add kernel.
#
#   make test        regenerate fixtures, build and run the g++ testbench
#   make asan_run    build and run the same testbench under ASan + UBSan
#   make csim        HLS C-sim with top selector_ln_q8_adapter
#   make csim_bias   HLS C-sim with top qkv_bias_add
#
# Vitis HLS is used for C-simulation only. No csynth/cosim/export/Vivado/board.

VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls
HLS_INCLUDE ?= /home/xilinx/Vitis_HLS/2023.2/include
PYTHON ?= /home/zhaoxiangyu/conda-envs/robobrain25/bin/python
CXX ?= g++

ROOT := $(abspath $(CURDIR)/../..)
ARTIFACT_DIR ?= $(ROOT)/artifacts/token_selector/2026-09-28-selector-qkv-adapter-csim
BUILD_DIR ?= /tmp/robobrain_selector_qkv_adapter_build

INCLUDES := -Iinclude -isystem $(HLS_INCLUDE)
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas
# -fno-sanitize=shift: the Vitis HLS 2023.2 ap_int/ap_fixed headers contain
# left-shift-of-negative-value idioms (ap_private.h clearUnusedBits) that UBSan
# flags. Every shift in this kernel operates on unsigned values with bounded
# shift counts, so no kernel-side check is lost. All other UBSan checks and ASan
# remain fully enabled.
SANFLAGS ?= -std=c++14 -O1 -g -fsanitize=address,undefined -fno-sanitize=shift \
	-fno-omit-frame-pointer -Wall -Wextra -Wno-unknown-pragmas

SRC := src/selector_qkv_adapter.cpp
TB := tb/tb_selector_qkv_adapter.cpp

FIXTURE_ENV := SQA_ADAPTER_FIXTURE=$(ARTIFACT_DIR)/real_block1_adapter_fixture.bin \
	SQA_BIAS_FIXTURE=$(ARTIFACT_DIR)/real_block1_bias_fixture.bin

.PHONY: all tb asan fixtures test asan_run csim csim_bias clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

tb: | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(TB) -o $(BUILD_DIR)/tb_selector_qkv_adapter

asan: | $(BUILD_DIR)
	$(CXX) $(SANFLAGS) $(INCLUDES) $(SRC) $(TB) -o $(BUILD_DIR)/tb_selector_qkv_adapter_asan

fixtures:
	$(PYTHON) tools/generate_fixtures.py --out-dir $(ARTIFACT_DIR)

test: tb fixtures
	$(FIXTURE_ENV) $(BUILD_DIR)/tb_selector_qkv_adapter

asan_run: asan fixtures
	$(FIXTURE_ENV) $(BUILD_DIR)/tb_selector_qkv_adapter_asan

csim:
	VITIS_HLS_BIN=$(VITIS_HLS) SQA_ARTIFACT_DIR=$(ARTIFACT_DIR) \
		bash scripts/run_stage.sh csim_adapter

csim_bias:
	VITIS_HLS_BIN=$(VITIS_HLS) SQA_ARTIFACT_DIR=$(ARTIFACT_DIR) \
		bash scripts/run_stage.sh csim_bias

clean:
	rm -rf $(BUILD_DIR)
