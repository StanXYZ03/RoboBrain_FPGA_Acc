CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror

BUILD := build
TB := $(BUILD)/tb_eth_test_protocol

.PHONY: sim clean

sim: $(TB)
	$(TB)

$(TB): src/eth_test_protocol.c tb/tb_eth_test_protocol.c include/eth_test_protocol.h
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -Iinclude src/eth_test_protocol.c tb/tb_eth_test_protocol.c -o $(TB)

clean:
	rm -f $(TB)
