# g++ C-simulation for the fixed-point hardware-regular token selector.
#
#   make test      regenerate fixtures, build and run the C testbench (g++)
#   make asan      run the same testbench under ASan+UBSan
#   make clean
#
# Vitis HLS is intentionally NOT invoked here (no csynth/cosim).

PYTHON ?= /home/zhaoxiangyu/conda-envs/robobrain25/bin/python
CXX ?= g++

ROOT := $(abspath $(CURDIR)/../..)
FIXTURE_DIR ?= $(ROOT)/artifacts/token_selector/2026-09-27-fixedpoint-selector-fixtures
BUILD_DIR ?= $(CURDIR)/build

INCLUDES := -Iinclude
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas
SANFLAGS ?= -std=c++17 -O1 -g -fsanitize=address,undefined \
	-fno-omit-frame-pointer -Wall -Wextra -Wno-unknown-pragmas

SRC := src/token_selector_hls.cpp
TB := tb/tb_token_selector_hls.cpp

.PHONY: all fixtures tb run test asan asan_run clean

all: test

fixtures:
	$(PYTHON) tools/generate_fixtures.py --out-dir $(FIXTURE_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

tb: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(TB) -o $(BUILD_DIR)/tb_token_selector_hls

asan: $(BUILD_DIR)
	$(CXX) $(SANFLAGS) $(INCLUDES) $(SRC) $(TB) -o $(BUILD_DIR)/tb_token_selector_hls_asan

run: fixtures tb
	$(BUILD_DIR)/tb_token_selector_hls $(FIXTURE_DIR)

test: run

asan_run: fixtures asan
	$(BUILD_DIR)/tb_token_selector_hls_asan $(FIXTURE_DIR)

clean:
	rm -rf $(BUILD_DIR)
