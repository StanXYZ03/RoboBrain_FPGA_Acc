#ifndef ETH_TEST_PROTOCOL_H
#define ETH_TEST_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define ETH_TEST_MESSAGE_SIZE 16U

enum eth_test_result {
    ETH_TEST_OK = 0,
    ETH_TEST_BAD_ARGUMENT = -1,
    ETH_TEST_BAD_LENGTH = -2,
    ETH_TEST_BAD_MAGIC = -3,
    ETH_TEST_BAD_VERSION = -4,
    ETH_TEST_BAD_COMMAND = -5,
    ETH_TEST_BAD_CHECKSUM = -6
};

uint32_t eth_test_checksum(const uint8_t *data, size_t length);
int eth_test_build_request(uint8_t *message, size_t capacity, uint32_t sequence);
int eth_test_validate_request(const uint8_t *message, size_t length,
                              uint32_t *sequence);

#endif
