#include "token_sparse_descriptor.h"

#include <string.h>

#define TSD_MAGIC 0x44535452u
#define TSD_HEADER_CRC_OFFSET 32u
#define TSD_ALLOWED_FLAGS (TSD_FLAG_HAS_POSITIONS | TSD_FLAG_IDENTITY | TSD_FLAG_SORTED_UNIQUE)

static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u16(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

uint32_t tsd_crc32(const uint8_t *data, size_t size) {
    uint32_t crc = 0xffffffffu;
    size_t i;
    if (data == NULL && size != 0u) {
        return 0u;
    }
    for (i = 0; i < size; ++i) {
        uint32_t value = crc ^ data[i];
        unsigned bit;
        for (bit = 0; bit < 8u; ++bit) {
            value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
        }
        crc = value;
    }
    return crc ^ 0xffffffffu;
}

static int validate_descriptor(const struct tsd_descriptor *d) {
    uint32_t i;
    if (d == NULL || d->entries == NULL || d->full_count == 0u ||
        d->kept_count == 0u || d->kept_count > d->full_count ||
        d->position_dims > TSD_MAX_POSITION_DIMS ||
        (d->flags & ~TSD_ALLOWED_FLAGS) != 0u) {
        return TSD_ERR_ARGUMENT;
    }
    if (((d->flags & TSD_FLAG_HAS_POSITIONS) != 0u) != (d->position_dims != 0u) ||
        (d->flags & TSD_FLAG_SORTED_UNIQUE) == 0u) {
        return TSD_ERR_ARGUMENT;
    }
    for (i = 0; i < d->kept_count; ++i) {
        if (d->entries[i].original_index >= d->full_count ||
            (i != 0u && d->entries[i - 1u].original_index >= d->entries[i].original_index)) {
            return TSD_ERR_ARGUMENT;
        }
        if ((d->flags & TSD_FLAG_IDENTITY) != 0u && d->entries[i].original_index != i) {
            return TSD_ERR_ARGUMENT;
        }
    }
    if ((d->flags & TSD_FLAG_IDENTITY) != 0u && d->kept_count != d->full_count) {
        return TSD_ERR_ARGUMENT;
    }
    return TSD_OK;
}

size_t tsd_serialized_size(const struct tsd_descriptor *d) {
    size_t entry_bytes;
    if (validate_descriptor(d) != TSD_OK) {
        return 0u;
    }
    entry_bytes = 4u * (1u + d->position_dims);
    if (d->kept_count > (SIZE_MAX - TSD_HEADER_BYTES) / entry_bytes ||
        d->kept_count > UINT32_MAX / entry_bytes) {
        return 0u;
    }
    return TSD_HEADER_BYTES + (size_t)d->kept_count * entry_bytes;
}

int tsd_encode(const struct tsd_descriptor *d, uint8_t *output,
               size_t capacity, size_t *written) {
    size_t total = tsd_serialized_size(d);
    uint16_t entry_bytes;
    uint32_t i;
    if (total == 0u || output == NULL || written == NULL) {
        return TSD_ERR_ARGUMENT;
    }
    if (capacity < total) {
        return TSD_ERR_CAPACITY;
    }
    entry_bytes = (uint16_t)(4u * (1u + d->position_dims));
    memset(output, 0, total);
    put_u32(output + 0u, TSD_MAGIC);
    put_u16(output + 4u, TSD_VERSION);
    put_u16(output + 6u, TSD_HEADER_BYTES);
    put_u32(output + 8u, d->flags);
    put_u32(output + 12u, d->full_count);
    put_u32(output + 16u, d->kept_count);
    put_u16(output + 20u, entry_bytes);
    put_u16(output + 22u, d->position_dims);
    put_u32(output + 24u, (uint32_t)(total - TSD_HEADER_BYTES));
    for (i = 0; i < d->kept_count; ++i) {
        uint8_t *entry = output + TSD_HEADER_BYTES + (size_t)i * entry_bytes;
        uint16_t axis;
        put_u32(entry, d->entries[i].original_index);
        for (axis = 0; axis < d->position_dims; ++axis) {
            put_u32(entry + 4u * (axis + 1u), d->entries[i].position[axis]);
        }
    }
    put_u32(output + 28u, tsd_crc32(output + TSD_HEADER_BYTES, total - TSD_HEADER_BYTES));
    put_u32(output + TSD_HEADER_CRC_OFFSET, 0u);
    put_u32(output + TSD_HEADER_CRC_OFFSET, tsd_crc32(output, TSD_HEADER_BYTES));
    *written = total;
    return TSD_OK;
}

int tsd_decode(const uint8_t *data, size_t size, struct tsd_view *view) {
    uint8_t header[TSD_HEADER_BYTES];
    uint32_t expected_header_crc;
    uint32_t payload_bytes;
    uint32_t i;
    if (data == NULL || view == NULL || size < TSD_HEADER_BYTES) {
        return TSD_ERR_ARGUMENT;
    }
    if (get_u32(data) != TSD_MAGIC || get_u16(data + 4u) != TSD_VERSION ||
        get_u16(data + 6u) != TSD_HEADER_BYTES) {
        return TSD_ERR_FORMAT;
    }
    memcpy(header, data, TSD_HEADER_BYTES);
    expected_header_crc = get_u32(header + TSD_HEADER_CRC_OFFSET);
    put_u32(header + TSD_HEADER_CRC_OFFSET, 0u);
    if (tsd_crc32(header, TSD_HEADER_BYTES) != expected_header_crc) {
        return TSD_ERR_CRC;
    }
    for (i = 36u; i < TSD_HEADER_BYTES; ++i) {
        if (data[i] != 0u) {
            return TSD_ERR_FORMAT;
        }
    }
    view->flags = get_u32(data + 8u);
    view->full_count = get_u32(data + 12u);
    view->kept_count = get_u32(data + 16u);
    view->entry_bytes = get_u16(data + 20u);
    view->position_dims = get_u16(data + 22u);
    payload_bytes = get_u32(data + 24u);
    view->payload = data + TSD_HEADER_BYTES;
    if ((view->flags & ~TSD_ALLOWED_FLAGS) != 0u ||
        (view->flags & TSD_FLAG_SORTED_UNIQUE) == 0u ||
        view->full_count == 0u || view->kept_count == 0u ||
        view->kept_count > view->full_count ||
        view->position_dims > TSD_MAX_POSITION_DIMS ||
        (((view->flags & TSD_FLAG_HAS_POSITIONS) != 0u) != (view->position_dims != 0u)) ||
        view->entry_bytes != 4u * (1u + view->position_dims) ||
        (uint64_t)payload_bytes != (uint64_t)view->kept_count * view->entry_bytes ||
        size != TSD_HEADER_BYTES + payload_bytes) {
        return TSD_ERR_FORMAT;
    }
    if (tsd_crc32(view->payload, payload_bytes) != get_u32(data + 28u)) {
        return TSD_ERR_CRC;
    }
    for (i = 0; i < view->kept_count; ++i) {
        uint32_t index = get_u32(view->payload + (size_t)i * view->entry_bytes);
        if (index >= view->full_count ||
            (i != 0u && get_u32(view->payload + (size_t)(i - 1u) * view->entry_bytes) >= index) ||
            ((view->flags & TSD_FLAG_IDENTITY) != 0u && index != i)) {
            return TSD_ERR_FORMAT;
        }
    }
    if ((view->flags & TSD_FLAG_IDENTITY) != 0u && view->kept_count != view->full_count) {
        return TSD_ERR_FORMAT;
    }
    return TSD_OK;
}

int tsd_view_entry(const struct tsd_view *view, uint32_t ordinal,
                   struct tsd_entry *entry) {
    const uint8_t *source;
    uint16_t axis;
    if (view == NULL || entry == NULL || view->payload == NULL || ordinal >= view->kept_count) {
        return TSD_ERR_ARGUMENT;
    }
    memset(entry, 0, sizeof(*entry));
    source = view->payload + (size_t)ordinal * view->entry_bytes;
    entry->original_index = get_u32(source);
    for (axis = 0; axis < view->position_dims; ++axis) {
        entry->position[axis] = get_u32(source + 4u * (axis + 1u));
    }
    return TSD_OK;
}
