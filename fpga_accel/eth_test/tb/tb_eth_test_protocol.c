#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "eth_test_protocol.h"

static unsigned failures;
static unsigned checks;

static void expect(const char *name, int condition)
{
    ++checks;
    if (!condition) {
        ++failures;
        printf("TB_CHECK name=%s result=FAIL\n", name);
    }
}

int main(void)
{
    uint8_t message[ETH_TEST_MESSAGE_SIZE];
    uint8_t reply[ETH_TEST_MESSAGE_SIZE];
    uint32_t sequence = 0U;
    int status;

    status = eth_test_build_request(message, sizeof(message), 0x1234ABCDU);
    expect("build_valid", status == ETH_TEST_OK);
    status = eth_test_validate_request(message, sizeof(message), &sequence);
    expect("validate_valid", status == ETH_TEST_OK);
    expect("sequence_round_trip", sequence == 0x1234ABCDU);

    memcpy(reply, message, sizeof(reply));
    expect("echo_exact", memcmp(reply, message, sizeof(reply)) == 0);

    message[0] ^= 1U;
    status = eth_test_validate_request(message, sizeof(message), &sequence);
    expect("detect_bad_magic", status == ETH_TEST_BAD_MAGIC);
    message[0] ^= 1U;

    message[12] ^= 0x80U;
    status = eth_test_validate_request(message, sizeof(message), &sequence);
    expect("detect_bad_checksum", status == ETH_TEST_BAD_CHECKSUM);
    message[12] ^= 0x80U;

    status = eth_test_validate_request(message, sizeof(message) - 1U, &sequence);
    expect("detect_short_message", status == ETH_TEST_BAD_LENGTH);
    status = eth_test_build_request(message, sizeof(message) - 1U, 1U);
    expect("detect_small_buffer", status == ETH_TEST_BAD_LENGTH);
    status = eth_test_validate_request(NULL, sizeof(message), &sequence);
    expect("detect_null", status == ETH_TEST_BAD_ARGUMENT);

    printf("TB_ETH_RESULT=%s checks=%u failures=%u\n",
           failures == 0U ? "PASS" : "FAIL", checks, failures);
    return failures == 0U ? 0 : 1;
}
