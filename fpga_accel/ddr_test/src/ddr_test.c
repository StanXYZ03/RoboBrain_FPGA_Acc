#include "ddr_test.h"

static uint32_t pattern_for(size_t index, uint64_t address)
{
    uint32_t folded = (uint32_t)address ^ (uint32_t)(address >> 32);
    return 0xA5C30000u ^ folded ^ (uint32_t)(index * 0x1F123BB5u);
}

int ddr_test_sparse(
    const uint64_t *addresses,
    size_t address_count,
    ddr_write32_fn write32,
    ddr_read32_fn read32,
    void *context,
    volatile struct ddr_test_status *status)
{
    size_t index;

    status->magic = DDR_TEST_MAGIC;
    status->phase = DDR_TEST_WRITING;
    status->checks = 0;
    status->failures = 0;
    status->last_address = 0;
    status->expected = 0;
    status->actual = 0;

    for (index = 0; index < address_count; ++index) {
        write32(context, addresses[index], pattern_for(index, addresses[index]));
    }

    status->phase = DDR_TEST_VERIFYING;
    for (index = 0; index < address_count; ++index) {
        uint32_t expected = pattern_for(index, addresses[index]);
        uint32_t actual = read32(context, addresses[index]);
        status->checks++;
        if (actual != expected) {
            status->failures++;
            status->last_address = addresses[index];
            status->expected = expected;
            status->actual = actual;
        }
    }

    /* Re-read after all writes: aliased addresses overwrite earlier patterns. */
    status->phase = DDR_TEST_ALIAS_CHECK;
    for (index = 0; index < address_count; ++index) {
        uint32_t expected = pattern_for(index, addresses[index]);
        uint32_t actual = read32(context, addresses[index]);
        status->checks++;
        if (actual != expected) {
            status->failures++;
            status->last_address = addresses[index];
            status->expected = expected;
            status->actual = actual;
        }
    }

    status->phase = status->failures == 0 ? DDR_TEST_PASS : DDR_TEST_FAIL;
    return status->failures == 0 ? 0 : -1;
}

