#ifndef ROBOBRAIN_DDR_TEST_H
#define ROBOBRAIN_DDR_TEST_H

#include <stddef.h>
#include <stdint.h>

#define DDR_TEST_MAGIC 0x44445234u

enum ddr_test_phase {
    DDR_TEST_NOT_STARTED = 0,
    DDR_TEST_WRITING = 1,
    DDR_TEST_VERIFYING = 2,
    DDR_TEST_ALIAS_CHECK = 3,
    DDR_TEST_PASS = 0x600Du,
    DDR_TEST_FAIL = 0xDEADu,
};

struct ddr_test_status {
    uint32_t magic;
    uint32_t phase;
    uint32_t checks;
    uint32_t failures;
    uint64_t last_address;
    uint32_t expected;
    uint32_t actual;
};

typedef void (*ddr_write32_fn)(void *context, uint64_t address, uint32_t value);
typedef uint32_t (*ddr_read32_fn)(void *context, uint64_t address);

int ddr_test_sparse(
    const uint64_t *addresses,
    size_t address_count,
    ddr_write32_fn write32,
    ddr_read32_fn read32,
    void *context,
    volatile struct ddr_test_status *status);

#endif

