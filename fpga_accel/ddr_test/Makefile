CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
BUILD_DIR := build
TB := $(BUILD_DIR)/tb_ddr_test

.PHONY: all sim clean

all: sim

$(TB): src/ddr_test.c tb/tb_ddr_test.c include/ddr_test.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -Iinclude src/ddr_test.c tb/tb_ddr_test.c -o $(TB)

sim: $(TB)
	$(TB)

clean:
	rm -f $(TB)

