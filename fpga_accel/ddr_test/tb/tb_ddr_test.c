#include "ddr_test.h"

#include <stdio.h>
#include <string.h>

#define SLOT_COUNT 16

struct slot {
    uint64_t address;
    uint32_t value;
    int valid;
};

struct memory_model {
    struct slot slots[SLOT_COUNT];
    int alias_high_pair;
    int corrupt_read;
};

static uint64_t map_address(struct memory_model *model, uint64_t address)
{
    if (model->alias_high_pair && address == UINT64_C(0x0000000880000000)) {
        return UINT64_C(0x0000000800000000);
    }
    return address;
}

static struct slot *find_slot(struct memory_model *model, uint64_t address)
{
    size_t index;
    struct slot *free_slot = NULL;
    address = map_address(model, address);
    for (index = 0; index < SLOT_COUNT; ++index) {
        if (model->slots[index].valid && model->slots[index].address == address) {
            return &model->slots[index];
        }
        if (!model->slots[index].valid && free_slot == NULL) {
            free_slot = &model->slots[index];
        }
    }
    return free_slot;
}

static void model_write32(void *context, uint64_t address, uint32_t value)
{
    struct memory_model *model = context;
    struct slot *slot = find_slot(model, address);
    slot->valid = 1;
    slot->address = map_address(model, address);
    slot->value = value;
}

static uint32_t model_read32(void *context, uint64_t address)
{
    struct memory_model *model = context;
    struct slot *slot = find_slot(model, address);
    uint32_t value = slot != NULL && slot->valid ? slot->value : 0;
    if (model->corrupt_read && address == UINT64_C(0x0000000050000000)) {
        value ^= 1u;
    }
    return value;
}

static int run_case(const char *name, int alias, int corrupt, int expect_pass)
{
    static const uint64_t addresses[] = {
        UINT64_C(0x0000000010000000),
        UINT64_C(0x0000000050000000),
        UINT64_C(0x0000000800000000),
        UINT64_C(0x0000000840000000),
        UINT64_C(0x0000000880000000),
        UINT64_C(0x0000000980000000),
        UINT64_C(0x0000000A80000000),
        UINT64_C(0x0000000B40000000),
    };
    struct memory_model model;
    struct ddr_test_status status;
    int result;

    memset(&model, 0, sizeof(model));
    memset(&status, 0, sizeof(status));
    model.alias_high_pair = alias;
    model.corrupt_read = corrupt;
    result = ddr_test_sparse(addresses, sizeof(addresses) / sizeof(addresses[0]),
                             model_write32, model_read32, &model, &status);

    printf("TB_CASE name=%s result=%s checks=%u failures=%u\n",
           name, result == 0 ? "PASS" : "FAIL", status.checks, status.failures);
    if (status.magic != DDR_TEST_MAGIC ||
        (result == 0) != expect_pass ||
        (status.phase == DDR_TEST_PASS) != expect_pass) {
        fprintf(stderr, "TB_ASSERT_FAIL name=%s\n", name);
        return 1;
    }
    return 0;
}

int main(void)
{
    int failures = 0;
    failures += run_case("healthy_memory", 0, 0, 1);
    failures += run_case("address_alias", 1, 0, 0);
    failures += run_case("corrupt_read", 0, 1, 0);
    printf("TB_RESULT=%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
