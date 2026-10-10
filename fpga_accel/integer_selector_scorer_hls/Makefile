# Standalone bit-exact integer Selector scorer: HLS kernel + testbenches.
#
#   make fixtures     regenerate the compact C++ fixtures from the frozen golden
#   make python_test  run the focused Python fixture/contract tests
#   make test         regenerate fixtures, build and run the g++ testbench
#   make asan_run     build and run the same testbench under ASan + UBSan
#   make csim         Vitis HLS C-simulation (csim_design only)
#   make csynth       Vitis HLS C-synthesis  (csynth_design only, reports archived)
#   make clean
#
# Vitis HLS is used for C-simulation and C-synthesis only.  No C/RTL cosim, no
# IP export, no Vivado, no bitstream, no board access.

VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls
PYTHON ?= /home/zhaoxiangyu/conda-envs/robobrain25/bin/python
CXX ?= g++

ROOT := $(abspath $(CURDIR)/../..)
ARTIFACT_DIR ?= $(ROOT)/artifacts/token_selector/2026-10-03-integer-selector-scorer-hls
FIXTURE_DIR ?= $(ARTIFACT_DIR)/fixtures
BUILD_DIR ?= /tmp/robobrain_integer_selector_scorer_build

INCLUDES := -Iinclude
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas
SANFLAGS ?= -std=c++14 -O1 -g -fsanitize=address,undefined \
	-fno-omit-frame-pointer -Wall -Wextra -Wno-unknown-pragmas

SRC := src/integer_selector_scorer.cpp
TB := tb/tb_integer_selector_scorer.cpp

.PHONY: all tb asan fixtures python_test test asan_run csim csynth clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

tb: | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(TB) -o $(BUILD_DIR)/tb_integer_selector_scorer

asan: | $(BUILD_DIR)
	$(CXX) $(SANFLAGS) $(INCLUDES) $(SRC) $(TB) -o $(BUILD_DIR)/tb_integer_selector_scorer_asan

fixtures:
	$(PYTHON) tools/generate_fixtures.py --out-dir $(FIXTURE_DIR)

python_test: fixtures
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test: tb fixtures
	$(BUILD_DIR)/tb_integer_selector_scorer $(FIXTURE_DIR)

asan_run: asan fixtures
	$(BUILD_DIR)/tb_integer_selector_scorer_asan $(FIXTURE_DIR)

csim:
	VITIS_HLS_BIN=$(VITIS_HLS) ARTIFACT_DIR=$(ARTIFACT_DIR) \
		bash scripts/run_stage.sh csim

csynth:
	VITIS_HLS_BIN=$(VITIS_HLS) ARTIFACT_DIR=$(ARTIFACT_DIR) \
		bash scripts/run_stage.sh csynth

clean:
	rm -rf $(BUILD_DIR)
