#include "ddr_test.h"

#include "xil_cache.h"
#include "xil_mmu.h"
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
        UINT64_C(0x0000000010000000), /* 0.25 GiB DRAM offset */
        UINT64_C(0x0000000050000000), /* 1.25 GiB */
        UINT64_C(0x0000000800000000), /* 2 GiB */
        UINT64_C(0x0000000840000000), /* 3 GiB */
        UINT64_C(0x0000000880000000), /* 4 GiB */
        UINT64_C(0x0000000980000000), /* 8 GiB */
        UINT64_C(0x0000000A80000000), /* 12 GiB */
        UINT64_C(0x0000000B40000000), /* 15 GiB */
    };
    volatile struct ddr_test_status *status =
        (volatile struct ddr_test_status *)(uintptr_t)DDR_STATUS_ADDRESS;
    size_t index;

    Xil_DCacheDisable();
    Xil_ICacheDisable();
    for (index = 0; index < sizeof(addresses) / sizeof(addresses[0]); ++index) {
        Xil_SetTlbAttributes((UINTPTR)addresses[index], NORM_NONCACHE);
    }
    (void)ddr_test_sparse(addresses, sizeof(addresses) / sizeof(addresses[0]),
                          target_write32, target_read32, 0, status);
    __asm__ volatile("dsb sy" ::: "memory");
    for (;;) {
        __asm__ volatile("wfe");
    }
}
