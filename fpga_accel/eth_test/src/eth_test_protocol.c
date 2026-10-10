#include "eth_test_protocol.h"

#define ETH_TEST_VERSION 1U
#define ETH_TEST_COMMAND_ECHO 1U
#define ETH_TEST_FNV_OFFSET 2166136261U
#define ETH_TEST_FNV_PRIME 16777619U

static uint32_t load_le32(const uint8_t *data)
{
    return ((uint32_t)data[0]) |
           ((uint32_t)data[1] << 8U) |
           ((uint32_t)data[2] << 16U) |
           ((uint32_t)data[3] << 24U);
}

static void store_le32(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)value;
    data[1] = (uint8_t)(value >> 8U);
    data[2] = (uint8_t)(value >> 16U);
    data[3] = (uint8_t)(value >> 24U);
}

uint32_t eth_test_checksum(const uint8_t *data, size_t length)
{
    uint32_t hash = ETH_TEST_FNV_OFFSET;
    size_t index;

    if (data == NULL) {
        return 0U;
    }

    for (index = 0U; index < length; ++index) {
        hash ^= data[index];
        hash *= ETH_TEST_FNV_PRIME;
    }
    return hash;
}

int eth_test_build_request(uint8_t *message, size_t capacity, uint32_t sequence)
{
    uint32_t checksum;

    if (message == NULL) {
        return ETH_TEST_BAD_ARGUMENT;
    }
    if (capacity < ETH_TEST_MESSAGE_SIZE) {
        return ETH_TEST_BAD_LENGTH;
    }

    message[0] = 'R';
    message[1] = 'B';
    message[2] = 'E';
    message[3] = 'T';
    message[4] = ETH_TEST_VERSION;
    message[5] = ETH_TEST_COMMAND_ECHO;
    message[6] = 0U;
    message[7] = 0U;
    store_le32(&message[8], sequence);
    checksum = eth_test_checksum(message, 12U);
    store_le32(&message[12], checksum);

    return ETH_TEST_OK;
}

int eth_test_validate_request(const uint8_t *message, size_t length,
                              uint32_t *sequence)
{
    uint32_t expected;
    uint32_t actual;

    if (message == NULL || sequence == NULL) {
        return ETH_TEST_BAD_ARGUMENT;
    }
    if (length != ETH_TEST_MESSAGE_SIZE) {
        return ETH_TEST_BAD_LENGTH;
    }
    if (message[0] != 'R' || message[1] != 'B' ||
        message[2] != 'E' || message[3] != 'T') {
        return ETH_TEST_BAD_MAGIC;
    }
    if (message[4] != ETH_TEST_VERSION) {
        return ETH_TEST_BAD_VERSION;
    }
    if (message[5] != ETH_TEST_COMMAND_ECHO) {
        return ETH_TEST_BAD_COMMAND;
    }

    expected = eth_test_checksum(message, 12U);
    actual = load_le32(&message[12]);
    if (actual != expected) {
        return ETH_TEST_BAD_CHECKSUM;
    }

    *sequence = load_le32(&message[8]);
    return ETH_TEST_OK;
}
