# Compact AXI interface wrapper around the accepted integer Selector pipeline.
#
#   make fixtures     regenerate the compact fixtures from the frozen goldens
#   make python_test  focused Python layout/fixture/interface tests
#   make tb           build the reference .so + the g++ testbench
#   make test         fixtures + build + run the g++ testbench
#   make asan_run     same testbench under ASan + UBSan
#   make csim         Vitis HLS C-simulation (csim_design only)
#   make csynth       Vitis HLS C-synthesis  (csynth_design only, reports archived)
#   make report       assemble the JSON/Markdown reports + harness result
#   make clean
#
# Vitis HLS is used for C-simulation and C-synthesis only.  No C/RTL cosim, no
# IP export, no Vivado, no bitstream, no board access.  Large HLS projects live
# in /tmp; only small reports/logs are copied into the artifact directory.

VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls
PYTHON ?= /home/zhaoxiangyu/conda-envs/robobrain25/bin/python
CXX ?= g++

ROOT := $(abspath $(CURDIR)/../..)
ARTIFACT_DIR ?= $(ROOT)/artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls
FIXTURE_DIR ?= $(ARTIFACT_DIR)/fixtures
BUILD_DIR ?= /tmp/robobrain_integer_selector_compact_build

INCLUDES := -Iinclude -I../integer_selector_scorer_hls/include \
	-I../token_selector_hls/include -I../integer_selector_pipeline_hls/include
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas
SANFLAGS ?= -std=c++14 -O1 -g -fsanitize=address,undefined \
	-fno-omit-frame-pointer -Wall -Wextra -Wno-unknown-pragmas

SRC := src/integer_selector_pipeline_compact.cpp
TB := tb/tb_integer_selector_pipeline_compact.cpp
REF_SRC := ../integer_selector_pipeline_hls/src/integer_selector_pipeline.cpp
REF_SO := $(BUILD_DIR)/libinteger_selector_pipeline_reference.so
TB_BIN := $(BUILD_DIR)/tb_integer_selector_pipeline_compact
TB_ASAN := $(BUILD_DIR)/tb_integer_selector_pipeline_compact_asan
LAYOUT_DUMP := $(BUILD_DIR)/compact_layout_dump

.PHONY: all tb asan ref_so layout_dump fixtures python_test test asan_run csim csynth report clean

all: test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

ref_so: | $(BUILD_DIR)
	$(CXX) -std=c++14 -O2 -fPIC -shared -Wall -Wextra -Wno-unknown-pragmas \
		$(INCLUDES) $(REF_SRC) -o $(REF_SO)

layout_dump: | $(BUILD_DIR)
	$(CXX) -std=c++14 -O1 -Wall -Wextra -Wno-unknown-pragmas \
		$(INCLUDES) tools/compact_layout_dump.cpp -o $(LAYOUT_DUMP)

tb: ref_so | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) $(TB) -ldl -o $(TB_BIN)

asan: ref_so | $(BUILD_DIR)
	$(CXX) $(SANFLAGS) $(INCLUDES) $(SRC) $(TB) -ldl -o $(TB_ASAN)

fixtures:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B tools/generate_fixtures.py --out-dir $(FIXTURE_DIR)

python_test: fixtures
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v

test: tb fixtures
	ISPC_REF_SO=$(REF_SO) $(TB_BIN) $(FIXTURE_DIR)

asan_run: asan fixtures
	ISPC_REF_SO=$(REF_SO) $(TB_ASAN) $(FIXTURE_DIR)

csim:
	VITIS_HLS_BIN=$(VITIS_HLS) ARTIFACT_DIR=$(ARTIFACT_DIR) BUILD_DIR=$(BUILD_DIR) \
		bash scripts/run_stage.sh csim

csynth:
	VITIS_HLS_BIN=$(VITIS_HLS) ARTIFACT_DIR=$(ARTIFACT_DIR) BUILD_DIR=$(BUILD_DIR) \
		bash scripts/run_stage.sh csynth

report:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B tools/build_report.py

clean:
	rm -rf $(BUILD_DIR)
