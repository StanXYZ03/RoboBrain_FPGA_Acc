CC ?= cc
ARM64_CC ?= /home/xilinx/Vitis/2023.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-gcc
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror
ROOT := ../..
RUNTIME := ../q8q8_runtime
UPSTREAM_ROOT := $(ROOT)/profiling_src/llama.cpp/ggml
UPSTREAM := $(UPSTREAM_ROOT)/src
CPPFLAGS := -Iinclude -I$(RUNTIME)/include
COMMON := src/robobrain_ggml_q8q8_bridge.c \
	$(RUNTIME)/src/q8q8_dispatch.c $(RUNTIME)/src/q8q8_arena.c \
	$(RUNTIME)/src/q8q8_runtime.c
BUILD ?= build
ARTIFACTS ?= artifacts

.PHONY: all test sanitize upstream-layout upstream-quantizer upstream-adapter arm64 prepare-benchmark weight-path-benchmark output-path-benchmark clean

all: test sanitize upstream-layout upstream-quantizer upstream-adapter arm64

$(BUILD) $(ARTIFACTS):
	mkdir -p $@

$(BUILD)/test_bridge: tests/test_bridge.c $(COMMON) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -lm -pthread -o $@

test: $(BUILD)/test_bridge | $(ARTIFACTS)
	$(BUILD)/test_bridge > $(ARTIFACTS)/bridge_unit_test.log
	cat $(ARTIFACTS)/bridge_unit_test.log

sanitize: | $(BUILD) $(ARTIFACTS)
	$(CC) $(CPPFLAGS) -O1 -g -std=c11 -Wall -Wextra -Werror \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		tests/test_bridge.c $(COMMON) -lm -pthread -o $(BUILD)/test_bridge_sanitize
	ASAN_OPTIONS=detect_leaks=0 $(BUILD)/test_bridge_sanitize \
		> $(ARTIFACTS)/bridge_sanitizer_test.log
	cat $(ARTIFACTS)/bridge_sanitizer_test.log

upstream-layout: | $(BUILD) $(ARTIFACTS)
	$(CC) $(CPPFLAGS) -I$(UPSTREAM) $(CFLAGS) tests/test_upstream_layout.c \
		-o $(BUILD)/test_upstream_layout
	$(BUILD)/test_upstream_layout > $(ARTIFACTS)/upstream_layout_test.log
	cat $(ARTIFACTS)/upstream_layout_test.log

upstream-quantizer: | $(BUILD) $(ARTIFACTS)
	$(CC) $(CPPFLAGS) -I$(UPSTREAM) -I$(UPSTREAM_ROOT)/include $(CFLAGS) \
		-Wno-error=unused-function -ffunction-sections -fdata-sections \
		tests/test_upstream_quantizer.c \
		src/robobrain_ggml_q8q8_bridge.c $(RUNTIME)/src/q8q8_dispatch.c \
		$(RUNTIME)/src/q8q8_arena.c $(RUNTIME)/src/q8q8_runtime.c \
		$(UPSTREAM)/ggml-quants.c -Wl,--gc-sections -lm -pthread \
		-o $(BUILD)/test_upstream_quantizer
	$(BUILD)/test_upstream_quantizer > $(ARTIFACTS)/upstream_quantizer_test.log
	cat $(ARTIFACTS)/upstream_quantizer_test.log

upstream-adapter: | $(BUILD) $(ARTIFACTS)
	$(CC) $(CPPFLAGS) -I$(UPSTREAM_ROOT)/include $(CFLAGS) \
		tests/test_tensor_adapter.c src/robobrain_ggml_tensor_adapter.c \
		src/robobrain_ggml_q8q8_bridge.c $(RUNTIME)/src/q8q8_dispatch.c \
		$(RUNTIME)/src/q8q8_arena.c $(RUNTIME)/src/q8q8_runtime.c -lm -pthread \
		-o $(BUILD)/test_tensor_adapter
	$(BUILD)/test_tensor_adapter > $(ARTIFACTS)/tensor_adapter_test.log
	cat $(ARTIFACTS)/tensor_adapter_test.log

arm64: | $(ARTIFACTS)
	$(ARM64_CC) $(CPPFLAGS) $(CFLAGS) tests/test_bridge.c $(COMMON) -lm -pthread \
		-o $(ARTIFACTS)/test_bridge_arm64
	$(ARM64_CC) $(CPPFLAGS) -I$(UPSTREAM_ROOT)/include $(CFLAGS) \
		tests/test_tensor_adapter.c src/robobrain_ggml_tensor_adapter.c \
		src/robobrain_ggml_q8q8_bridge.c $(RUNTIME)/src/q8q8_dispatch.c \
		$(RUNTIME)/src/q8q8_arena.c $(RUNTIME)/src/q8q8_runtime.c -lm -pthread \
		-o $(ARTIFACTS)/test_tensor_adapter_arm64
	file $(ARTIFACTS)/test_bridge_arm64 | tee $(ARTIFACTS)/arm64_build.log
	file $(ARTIFACTS)/test_tensor_adapter_arm64 | tee -a $(ARTIFACTS)/arm64_build.log

$(BUILD)/prepare_benchmark: src/prepare_benchmark.c $(COMMON) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -lm -pthread -o $@

prepare-benchmark: $(BUILD)/prepare_benchmark | $(ARTIFACTS)
	$(BUILD)/prepare_benchmark > $(ARTIFACTS)/prepare_benchmark_host.log
	grep -q 'PREP_BENCH_PASS cases=6 measurements=24 numeric_cases=6' \
		$(ARTIFACTS)/prepare_benchmark_host.log
	$(ARM64_CC) $(CPPFLAGS) $(CFLAGS) src/prepare_benchmark.c $(COMMON) -lm -pthread \
		-static -o $(ARTIFACTS)/prepare_benchmark_arm64
	file $(ARTIFACTS)/prepare_benchmark_arm64 | \
		tee $(ARTIFACTS)/prepare_benchmark_arm64_build.log

$(BUILD)/weight_path_benchmark: src/weight_path_benchmark.c $(COMMON) \
		$(RUNTIME)/src/q8q8_uio.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -lm -pthread -o $@

weight-path-benchmark: $(BUILD)/weight_path_benchmark | $(ARTIFACTS)
	$(BUILD)/weight_path_benchmark --cache-only \
		> $(ARTIFACTS)/weight_path_benchmark_host.log
	grep -q 'WEIGHT_PATH_PASS cases=3 destinations=1 byte_equal=yes' \
		$(ARTIFACTS)/weight_path_benchmark_host.log
	$(ARM64_CC) $(CPPFLAGS) $(CFLAGS) src/weight_path_benchmark.c $(COMMON) \
		$(RUNTIME)/src/q8q8_uio.c -lm -pthread -static \
		-o $(ARTIFACTS)/weight_path_benchmark_arm64
	file $(ARTIFACTS)/weight_path_benchmark_arm64 | \
		tee $(ARTIFACTS)/weight_path_benchmark_arm64_build.log

output-path-benchmark: | $(ARTIFACTS)
	$(ARM64_CC) $(CPPFLAGS) $(CFLAGS) src/output_path_benchmark.c $(COMMON) \
		$(RUNTIME)/src/q8q8_uio.c -lm -pthread -static \
		-o $(ARTIFACTS)/output_path_benchmark_arm64
	file $(ARTIFACTS)/output_path_benchmark_arm64 | \
		tee $(ARTIFACTS)/output_path_benchmark_arm64_build.log

clean:
	$(RM) -r $(BUILD) $(ARTIFACTS)
