#ifndef TOKEN_SPARSE_DESCRIPTOR_H
#define TOKEN_SPARSE_DESCRIPTOR_H

#include <stddef.h>
#include <stdint.h>

#define TSD_VERSION 1u
#define TSD_HEADER_BYTES 64u
#define TSD_MAX_POSITION_DIMS 4u
#define TSD_FLAG_HAS_POSITIONS 0x1u
#define TSD_FLAG_IDENTITY 0x2u
#define TSD_FLAG_SORTED_UNIQUE 0x4u

enum tsd_status {
    TSD_OK = 0,
    TSD_ERR_ARGUMENT = -1,
    TSD_ERR_CAPACITY = -2,
    TSD_ERR_FORMAT = -3,
    TSD_ERR_CRC = -4
};

struct tsd_entry {
    uint32_t original_index;
    uint32_t position[TSD_MAX_POSITION_DIMS];
};

struct tsd_descriptor {
    uint32_t flags;
    uint32_t full_count;
    uint32_t kept_count;
    uint16_t position_dims;
    const struct tsd_entry *entries;
};

struct tsd_view {
    uint32_t flags;
    uint32_t full_count;
    uint32_t kept_count;
    uint16_t position_dims;
    uint16_t entry_bytes;
    const uint8_t *payload;
};

size_t tsd_serialized_size(const struct tsd_descriptor *descriptor);
int tsd_encode(const struct tsd_descriptor *descriptor, uint8_t *output,
               size_t capacity, size_t *written);
int tsd_decode(const uint8_t *data, size_t size, struct tsd_view *view);
int tsd_view_entry(const struct tsd_view *view, uint32_t ordinal,
                   struct tsd_entry *entry);
uint32_t tsd_crc32(const uint8_t *data, size_t size);

#endif
