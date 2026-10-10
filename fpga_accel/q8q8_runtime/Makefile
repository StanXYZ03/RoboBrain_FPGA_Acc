CC ?= cc
ARM64_CC ?= /home/xilinx/Vitis/2023.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-gcc
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror
CPPFLAGS := -Iinclude
LDLIBS := -lm
BUILD ?= build
ARTIFACTS ?= artifacts
COMMON := src/q8q8_runtime.c src/q8q8_smoke.c src/q8q8_arena.c
APM := src/q8q8_apm.c
DISPATCH := src/q8q8_dispatch.c

.PHONY: all test sanitize arm64 dt-test register-map-test dispatch-test dispatch-sanitize clean

all: test sanitize arm64 dt-test register-map-test dispatch-test dispatch-sanitize

$(BUILD)/test_q8q8_dispatch: tests/test_q8q8_dispatch.c src/q8q8_dispatch.c \
		src/q8q8_arena.c src/q8q8_runtime.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/test_q8q8_runtime: tests/test_q8q8_runtime.c $(COMMON) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/q8q8-selftest: src/q8q8_selftest.c src/q8q8_uio.c $(COMMON) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/q8q8-benchmark: src/q8q8_benchmark.c src/q8q8_uio.c $(COMMON) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(APM) $(DISPATCH) $(LDLIBS) -o $@

$(BUILD)/test_q8q8_apm: tests/test_q8q8_apm.c $(APM) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@

test: $(BUILD)/test_q8q8_runtime $(BUILD)/test_q8q8_apm $(BUILD)/q8q8-selftest $(BUILD)/q8q8-benchmark | $(ARTIFACTS)
	$(BUILD)/test_q8q8_runtime > $(ARTIFACTS)/runtime_unit_test.log
	cat $(ARTIFACTS)/runtime_unit_test.log
	$(BUILD)/test_q8q8_apm > $(ARTIFACTS)/apm_unit_test.log
	cat $(ARTIFACTS)/apm_unit_test.log
	$(BUILD)/q8q8-benchmark --self-test > $(ARTIFACTS)/benchmark_unit_test.log
	cat $(ARTIFACTS)/benchmark_unit_test.log

sanitize: | $(BUILD) $(ARTIFACTS)
	$(CC) $(CPPFLAGS) -O1 -g -std=c11 -Wall -Wextra -Werror \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		tests/test_q8q8_runtime.c $(COMMON) $(LDLIBS) \
		-o $(BUILD)/test_q8q8_runtime_sanitize
	ASAN_OPTIONS=detect_leaks=0 $(BUILD)/test_q8q8_runtime_sanitize \
		> $(ARTIFACTS)/runtime_sanitizer_test.log
	cat $(ARTIFACTS)/runtime_sanitizer_test.log
	$(CC) $(CPPFLAGS) -O1 -g -std=c11 -Wall -Wextra -Werror \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		src/q8q8_benchmark.c src/q8q8_uio.c $(COMMON) $(APM) $(DISPATCH) $(LDLIBS) \
		-o $(BUILD)/q8q8_benchmark_sanitize
	ASAN_OPTIONS=detect_leaks=0 $(BUILD)/q8q8_benchmark_sanitize --self-test \
		> $(ARTIFACTS)/benchmark_sanitizer_test.log
	cat $(ARTIFACTS)/benchmark_sanitizer_test.log

arm64: | $(ARTIFACTS)
	$(ARM64_CC) $(CPPFLAGS) $(CFLAGS) \
		src/q8q8_selftest.c src/q8q8_uio.c $(COMMON) $(LDLIBS) \
		-o $(ARTIFACTS)/q8q8-selftest-arm64
	$(ARM64_CC) $(CPPFLAGS) $(CFLAGS) \
		src/q8q8_benchmark.c src/q8q8_uio.c $(COMMON) $(APM) $(DISPATCH) $(LDLIBS) \
		-o $(ARTIFACTS)/q8q8-benchmark-arm64
	file $(ARTIFACTS)/q8q8-selftest-arm64
	file $(ARTIFACTS)/q8q8-benchmark-arm64

dt-test:
	sh scripts/test_dt_overlay.sh > $(ARTIFACTS)/dt_overlay_test.log
	cat $(ARTIFACTS)/dt_overlay_test.log

register-map-test: | $(ARTIFACTS)
	sh scripts/test_register_map.sh > $(ARTIFACTS)/register_map_test.log
	cat $(ARTIFACTS)/register_map_test.log

dispatch-test: $(BUILD)/test_q8q8_dispatch | $(ARTIFACTS)
	$(BUILD)/test_q8q8_dispatch > $(ARTIFACTS)/dispatch_unit_test.log
	cat $(ARTIFACTS)/dispatch_unit_test.log

dispatch-sanitize: | $(BUILD) $(ARTIFACTS)
	$(CC) $(CPPFLAGS) -O1 -g -std=c11 -Wall -Wextra -Werror \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		tests/test_q8q8_dispatch.c src/q8q8_dispatch.c \
		src/q8q8_arena.c src/q8q8_runtime.c $(LDLIBS) \
		-o $(BUILD)/test_q8q8_dispatch_sanitize
	ASAN_OPTIONS=detect_leaks=0 $(BUILD)/test_q8q8_dispatch_sanitize \
		> $(ARTIFACTS)/dispatch_sanitizer_test.log
	cat $(ARTIFACTS)/dispatch_sanitizer_test.log

$(BUILD) $(ARTIFACTS):
	mkdir -p $@

clean:
	$(RM) -r $(BUILD) $(ARTIFACTS)
