#include "token_sparse_descriptor.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const struct tsd_entry entries[] = {
        {1u, {0u, 0u, 1u, 0u}},
        {4u, {0u, 1u, 1u, 0u}},
        {7u, {0u, 2u, 1u, 0u}},
    };
    const struct tsd_descriptor descriptor = {
        TSD_FLAG_HAS_POSITIONS | TSD_FLAG_SORTED_UNIQUE,
        8u, 3u, 4u, entries
    };
    uint8_t bytes[256];
    size_t written = 0u;
    FILE *stream;
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT\n", argv[0]);
        return 2;
    }
    if (tsd_encode(&descriptor, bytes, sizeof(bytes), &written) != TSD_OK) {
        return 3;
    }
    stream = fopen(argv[1], "wb");
    if (stream == NULL) {
        return 4;
    }
    if (fwrite(bytes, 1u, written, stream) != written || fclose(stream) != 0) {
        return 5;
    }
    printf("TSD_FIXTURE_WRITE_PASS bytes=%zu\n", written);
    return 0;
}
