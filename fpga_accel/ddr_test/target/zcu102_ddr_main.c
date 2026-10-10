#include "ddr_test.h"

#include "xil_cache.h"
#include "xil_types.h"

#define DDR_STATUS_ADDRESS UINT64_C(0x00000000FFFEF000)

static void target_write32(void *context, uint64_t address, uint32_t value)
{
    (void)context;
    *(volatile uint32_t *)(uintptr_t)address = value;
    __asm__ volatile("dsb sy" ::: "memory");
}

static uint32_t target_read32(void *context, uint64_t address)
{
    uint32_t value;
    (void)context;
    __asm__ volatile("dsb sy" ::: "memory");
    value = *(volatile uint32_t *)(uintptr_t)address;
    __asm__ volatile("dsb sy" ::: "memory");
    return value;
}

int main(void)
{
    static const uint64_t addresses[] = {
        UINT64_C(0x0000000010000000),
        UINT64_C(0x0000000050000000),
        UINT64_C(0x0000000800000000),
        UINT64_C(0x0000000840000000),
    };
    volatile struct ddr_test_status *status =
        (volatile struct ddr_test_status *)(uintptr_t)DDR_STATUS_ADDRESS;

    /* Prevent cache hits from hiding physical DDR address aliasing. */
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    (void)ddr_test_sparse(addresses, 4, target_write32, target_read32, 0, status);
    __asm__ volatile("dsb sy" ::: "memory");
    for (;;) {
        __asm__ volatile("wfe");
    }
}
