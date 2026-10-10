#include "token_sparse_descriptor.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line=%d: %s\n", __LINE__, #x); return 1; } } while (0)

static void put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static int encode_sparse(uint8_t *buffer, size_t capacity, size_t *written) {
    const struct tsd_entry entries[] = {
        {1u, {0u, 0u, 1u, 0u}},
        {4u, {0u, 1u, 1u, 0u}},
        {7u, {0u, 2u, 1u, 0u}},
    };
    const struct tsd_descriptor descriptor = {
        TSD_FLAG_HAS_POSITIONS | TSD_FLAG_SORTED_UNIQUE,
        8u, 3u, 4u, entries
    };
    return tsd_encode(&descriptor, buffer, capacity, written);
}

int main(void) {
    uint8_t encoded[256];
    uint8_t damaged[256];
    size_t written = 0u;
    struct tsd_view view;
    struct tsd_entry entry;
    struct tsd_entry identity_entries[4];
    struct tsd_descriptor identity;
    struct tsd_entry invalid_entries[2] = {{2u, {0}}, {2u, {0}}};
    unsigned i;

    CHECK(tsd_crc32((const uint8_t *)"123456789", 9u) == 0xcbf43926u);
    puts("DESCRIPTOR_CASE name=crc_known_vector result=PASS");

    CHECK(encode_sparse(encoded, sizeof(encoded), &written) == TSD_OK);
    CHECK(written == TSD_HEADER_BYTES + 3u * 20u);
    CHECK(tsd_decode(encoded, written, &view) == TSD_OK);
    CHECK(view.full_count == 8u && view.kept_count == 3u && view.position_dims == 4u);
    CHECK(tsd_view_entry(&view, 1u, &entry) == TSD_OK);
    CHECK(entry.original_index == 4u && entry.position[1] == 1u);
    puts("DESCRIPTOR_CASE name=sparse_roundtrip result=PASS");

    for (i = 0; i < 4u; ++i) {
        identity_entries[i].original_index = i;
        memset(identity_entries[i].position, 0, sizeof(identity_entries[i].position));
    }
    identity.flags = TSD_FLAG_IDENTITY | TSD_FLAG_SORTED_UNIQUE;
    identity.full_count = 4u;
    identity.kept_count = 4u;
    identity.position_dims = 0u;
    identity.entries = identity_entries;
    CHECK(tsd_encode(&identity, encoded, sizeof(encoded), &written) == TSD_OK);
    CHECK(tsd_decode(encoded, written, &view) == TSD_OK);
    puts("DESCRIPTOR_CASE name=identity result=PASS");

    memcpy(damaged, encoded, written);
    damaged[0] ^= 1u;
    CHECK(tsd_decode(damaged, written, &view) == TSD_ERR_FORMAT);
    puts("DESCRIPTOR_CASE name=bad_magic result=PASS");

    memcpy(damaged, encoded, written);
    damaged[TSD_HEADER_BYTES] ^= 1u;
    CHECK(tsd_decode(damaged, written, &view) == TSD_ERR_CRC);
    puts("DESCRIPTOR_CASE name=bad_payload_crc result=PASS");

    memcpy(damaged, encoded, written);
    damaged[12] ^= 1u;
    CHECK(tsd_decode(damaged, written, &view) == TSD_ERR_CRC);
    puts("DESCRIPTOR_CASE name=bad_header_crc result=PASS");

    memcpy(damaged, encoded, written);
    damaged[36] = 1u;
    put_u32(damaged + 32u, 0u);
    put_u32(damaged + 32u, tsd_crc32(damaged, TSD_HEADER_BYTES));
    CHECK(tsd_decode(damaged, written, &view) == TSD_ERR_FORMAT);
    puts("DESCRIPTOR_CASE name=nonzero_reserved result=PASS");

    CHECK(tsd_decode(encoded, written - 1u, &view) == TSD_ERR_FORMAT);
    puts("DESCRIPTOR_CASE name=truncated result=PASS");

    identity.flags = TSD_FLAG_SORTED_UNIQUE;
    identity.full_count = 4u;
    identity.kept_count = 2u;
    identity.entries = invalid_entries;
    CHECK(tsd_encode(&identity, encoded, sizeof(encoded), &written) == TSD_ERR_ARGUMENT);
    invalid_entries[1].original_index = 5u;
    CHECK(tsd_encode(&identity, encoded, sizeof(encoded), &written) == TSD_ERR_ARGUMENT);
    puts("DESCRIPTOR_CASE name=invalid_indices result=PASS rejected=2");

    CHECK(encode_sparse(encoded, TSD_HEADER_BYTES, &written) == TSD_ERR_CAPACITY);
    puts("DESCRIPTOR_CASE name=capacity result=PASS");
    puts("TOKEN_SPARSE_DESCRIPTOR_TESTS_PASS cases=10");
    return 0;
}
