CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror -pedantic
CPPFLAGS += -Iinclude

.PHONY: all test clean
all: build/test_token_sparse_descriptor build/export_fixture

build/test_token_sparse_descriptor: src/token_sparse_descriptor.c tests/test_token_sparse_descriptor.c include/token_sparse_descriptor.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/token_sparse_descriptor.c tests/test_token_sparse_descriptor.c -o $@

test: build/test_token_sparse_descriptor
	./build/test_token_sparse_descriptor

build/export_fixture: src/token_sparse_descriptor.c tools/export_fixture.c include/token_sparse_descriptor.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/token_sparse_descriptor.c tools/export_fixture.c -o $@

clean:
	rm -rf build
