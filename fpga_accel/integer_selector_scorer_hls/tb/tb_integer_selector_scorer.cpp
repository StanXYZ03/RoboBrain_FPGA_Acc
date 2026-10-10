/*
 * tb_integer_selector_scorer.cpp - host / HLS C-simulation testbench.
 *
 * It re-reads the compact fixtures produced by tools/generate_fixtures.py and
 * proves the kernel is byte-exact against the frozen Python golden:
 *
 *   * real grounding/pointing fixtures (correct + wrong prompt) - every
 *     score_q byte, semantic_q16, l2_int, semantic_norm_q15 and l2_norm_q15;
 *   * deterministic synthetic zero / all-equal / G=1 / mixed-sign cases;
 *   * a streaming G=400 stress case (group means regenerated in-kernel by the
 *     documented xorshift64* generator, golden score_q/hash from Python);
 *   * direct primitive vectors for the exact floor sqrt, the exact signed
 *     round-half-away division (positive and negative .5 ties) and min-max;
 *   * the illegal group counts 0 / 4097 / -1 and the null-pointer guard.
 *
 * The production top `integer_selector_scorer` is always called and its
 * score_q bytes are compared to the golden.  The host-only debug twin is used
 * only to check the per-group intermediates.
 *
 * Usage: tb_integer_selector_scorer <fixture_dir>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "integer_selector_scorer.h"

namespace {

int g_passed = 0;
int g_failed = 0;
int g_fixtures = 0;

void check(bool ok, const char* what) {
    if (ok) {
        g_passed++;
    } else {
        g_failed++;
        printf("  FAIL: %s\n", what);
    }
}

bool read_file(const char* path, std::vector<uint8_t>* data) {
    FILE* handle = fopen(path, "rb");
    if (handle == NULL) {
        printf("cannot open %s\n", path);
        return false;
    }
    if (fseek(handle, 0, SEEK_END) != 0) {
        fclose(handle);
        return false;
    }
    long size = ftell(handle);
    if (size < 0) {
        fclose(handle);
        return false;
    }
    if (fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        return false;
    }
    data->resize((size_t)size);
    if (size > 0) {
        size_t got = fread(data->data(), 1, (size_t)size, handle);
        if (got != (size_t)size) {
            fclose(handle);
            return false;
        }
    }
    fclose(handle);
    return true;
}

uint32_t rd_u32(const uint8_t* p) {
    uint32_t value = 0;
    memcpy(&value, p, 4);
    return value;
}

uint64_t rd_u64(const uint8_t* p) {
    uint64_t value = 0;
    memcpy(&value, p, 8);
    return value;
}

int64_t rd_i64(const uint8_t* p) {
    int64_t value = 0;
    memcpy(&value, p, 8);
    return value;
}

uint16_t rd_u16(const uint8_t* p) {
    uint16_t value = 0;
    memcpy(&value, p, 2);
    return value;
}

int32_t rd_i32(const uint8_t* p) {
    int32_t value = 0;
    memcpy(&value, p, 4);
    return value;
}

uint64_t fnv1a64(const uint8_t* data, size_t count) {
    uint64_t value = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < count; i++) {
        value ^= (uint64_t)data[i];
        value *= 0x100000001B3ULL;
    }
    return value;
}

/* Documented deterministic generator (must match generate_fixtures.py). */
void xorshift_fill(uint64_t seed, uint8_t* out, size_t count) {
    uint64_t state = seed;
    for (size_t i = 0; i < count; i++) {
        state ^= (state << 13);
        state ^= (state >> 7);
        state ^= (state << 17);
        uint64_t product = state * 0x2545F4914F6CDD1DULL;
        out[i] = (uint8_t)((product >> 56) & 0xFF);
    }
}

struct FixtureHeader {
    uint32_t version;
    uint32_t kind;
    uint32_t group_count;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t role_count;
    uint32_t generator;
    uint64_t generator_seed;
    uint64_t score_fnv1a64;
    uint64_t semantic_fnv1a64;
    uint64_t l2_fnv1a64;
    uint64_t group_mean_fnv1a64;
    uint64_t payload_bytes;
};

bool parse_header(const std::vector<uint8_t>& data, const char magic[4],
                  FixtureHeader* header) {
    if (data.size() < 128) {
        return false;
    }
    if (memcmp(data.data(), magic, 4) != 0) {
        return false;
    }
    header->version = rd_u32(data.data() + 4);
    header->kind = rd_u32(data.data() + 8);
    header->group_count = rd_u32(data.data() + 12);
    header->grid_h = rd_u32(data.data() + 16);
    header->grid_w = rd_u32(data.data() + 20);
    header->role_count = rd_u32(data.data() + 24);
    header->generator = rd_u32(data.data() + 28);
    header->generator_seed = rd_u64(data.data() + 32);
    header->score_fnv1a64 = rd_u64(data.data() + 40);
    header->semantic_fnv1a64 = rd_u64(data.data() + 48);
    header->l2_fnv1a64 = rd_u64(data.data() + 56);
    header->group_mean_fnv1a64 = rd_u64(data.data() + 64);
    header->payload_bytes = rd_u64(data.data() + 72);
    return header->version == 1 && header->payload_bytes == data.size() - 128;
}

/* Run one scorer fixture (kinds 0/1/2). */
void run_scorer_fixture(const char* dir, const char* name) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    std::vector<uint8_t> data;
    if (!read_file(path, &data)) {
        check(false, "fixture read");
        return;
    }
    FixtureHeader header;
    if (!parse_header(data, "ISF1", &header)) {
        printf("  FAIL: header %s\n", name);
        g_failed++;
        return;
    }
    const int g_count = (int)header.group_count;
    const uint8_t* payload = data.data() + 128;
    const size_t group_bytes = (size_t)g_count * IS_EMBED_DIM;
    const size_t role_stride = 128 + (size_t)12 * g_count;

    /* Rebuild group_mean_q bytes (stored for kinds 0/1, generated for kind 2). */
    std::vector<uint8_t> group_raw(group_bytes);
    size_t roles_base = 0;
    if (header.generator == 0) {
        memcpy(group_raw.data(), payload, group_bytes);
        roles_base = group_bytes;
    } else {
        xorshift_fill(header.generator_seed, group_raw.data(), group_bytes);
    }
    check(fnv1a64(group_raw.data(), group_bytes) == header.group_mean_fnv1a64,
          "group_mean generator/hash");

    /* Pack the group means and the fixed Wv once. */
    std::vector<is_word_t> group_words(group_bytes / IS_PACKED_LANES);
    is_pack_int8_to_words((const int8_t*)group_raw.data(), (int)group_bytes,
                          group_words.data(), (int)group_words.size());

    std::vector<uint8_t> wv_bytes;
    snprintf(path, sizeof(path), "%s/wv_q.bin", dir);
    if (!read_file(path, &wv_bytes) ||
        wv_bytes.size() != (size_t)IS_WV_ROWS * IS_EMBED_DIM) {
        check(false, "wv_q.bin");
        return;
    }
    std::vector<is_word_t> wv_words(IS_WV_WORDS);
    is_pack_int8_to_words((const int8_t*)wv_bytes.data(),
                          (int)wv_bytes.size(), wv_words.data(), IS_WV_WORDS);

    std::vector<uint16_t> score(g_count);
    std::vector<uint16_t> top_score(g_count);
    std::vector<int32_t> semantic(g_count);
    std::vector<uint16_t> l2(g_count);
    std::vector<uint16_t> sem_norm(g_count);
    std::vector<uint16_t> l2_norm(g_count);

    for (uint32_t role = 0; role < header.role_count; role++) {
        const uint8_t* role_base = payload + roles_base + role * role_stride;
        std::vector<is_word_t> query_words(IS_QUERY_WORDS);
        is_pack_int8_to_words((const int8_t*)role_base, IS_QUERY_DIM,
                              query_words.data(), IS_QUERY_WORDS);
        const uint8_t* golden_score = role_base + 128;
        const uint8_t* golden_semantic = golden_score + (size_t)2 * g_count;
        const uint8_t* golden_l2 = golden_semantic + (size_t)4 * g_count;
        const uint8_t* golden_sem_norm = golden_l2 + (size_t)2 * g_count;
        const uint8_t* golden_l2_norm = golden_sem_norm + (size_t)2 * g_count;

        int status = integer_selector_scorer_debug(
            group_words.data(), wv_words.data(), query_words.data(), score.data(),
            semantic.data(), l2.data(), sem_norm.data(), l2_norm.data(), g_count);
        check(status == IS_STATUS_OK, "debug status OK");

        int status_top = integer_selector_scorer(
            group_words.data(), wv_words.data(), query_words.data(),
            top_score.data(), g_count);
        check(status_top == IS_STATUS_OK, "top status OK");

        bool score_ok = true;
        bool top_ok = true;
        bool semantic_ok = true;
        bool l2_ok = true;
        bool sem_norm_ok = true;
        bool l2_norm_ok = true;
        for (int i = 0; i < g_count; i++) {
            score_ok = score_ok &&
                       score[i] == rd_u16(golden_score + (size_t)2 * i);
            top_ok = top_ok &&
                     top_score[i] == rd_u16(golden_score + (size_t)2 * i);
            semantic_ok = semantic_ok &&
                          semantic[i] == rd_i32(golden_semantic + (size_t)4 * i);
            l2_ok = l2_ok && l2[i] == rd_u16(golden_l2 + (size_t)2 * i);
            sem_norm_ok =
                sem_norm_ok &&
                sem_norm[i] == rd_u16(golden_sem_norm + (size_t)2 * i);
            l2_norm_ok =
                l2_norm_ok && l2_norm[i] == rd_u16(golden_l2_norm + (size_t)2 * i);
        }
        check(score_ok, "debug score_q bytes");
        check(top_ok, "top score_q bytes");
        check(semantic_ok, "semantic_q16 bytes");
        check(l2_ok, "l2_int bytes");
        check(sem_norm_ok, "semantic_norm_q15 bytes");
        check(l2_norm_ok, "l2_norm_q15 bytes");

        /* Independent FNV-1a over the score bytes (kind 2 generator check). */
        std::vector<uint8_t> score_bytes((size_t)g_count * 2);
        for (int i = 0; i < g_count; i++) {
            score_bytes[(size_t)2 * i] = (uint8_t)(score[i] & 0xFF);
            score_bytes[(size_t)2 * i + 1] = (uint8_t)((score[i] >> 8) & 0xFF);
        }
        uint64_t expected_hash =
            fnv1a64(golden_score, (size_t)2 * g_count);
        check(fnv1a64(score_bytes.data(), score_bytes.size()) == expected_hash,
              "score_q FNV-1a");
    }
    if (name[0] == 'f') {
        printf("  %-26s G=%-4d roles=%u score_fnv=0x%016llx\n", name, g_count,
               header.role_count, (unsigned long long)header.score_fnv1a64);
    }
    g_fixtures++;
}

void run_primitives(const char* dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/primitives.bin", dir);
    std::vector<uint8_t> data;
    if (!read_file(path, &data)) {
        check(false, "primitives read");
        return;
    }
    if (data.size() < 64 || memcmp(data.data(), "ISP1", 4) != 0) {
        check(false, "primitives header");
        return;
    }
    uint32_t n_sqrt = rd_u32(data.data() + 8);
    uint32_t n_div = rd_u32(data.data() + 12);
    uint32_t n_minmax = rd_u32(data.data() + 16);
    const uint8_t* cursor = data.data() + 64;

    bool sqrt_ok = true;
    for (uint32_t i = 0; i < n_sqrt; i++) {
        uint64_t input = rd_u64(cursor);
        uint64_t expected = rd_u64(cursor + 8);
        cursor += 16;
        sqrt_ok = sqrt_ok && is_test_floor_sqrt_u64(input) == expected;
    }
    check(sqrt_ok, "floor_sqrt vectors");

    bool div_ok = true;
    bool div_tie_ok = true;
    for (uint32_t i = 0; i < n_div; i++) {
        int64_t numerator = rd_i64(cursor);
        int64_t denominator = rd_i64(cursor + 8);
        int64_t expected = rd_i64(cursor + 16);
        cursor += 24;
        int64_t got = is_test_round_half_away_div(numerator, denominator);
        div_ok = div_ok && got == expected;
        /* 2*|num| == den is the exact .5 tie; round-half-away gives +/-1. */
        if (2 * (numerator < 0 ? -numerator : numerator) == denominator) {
            int64_t expected_tie = numerator > 0 ? 1 : -1;
            div_tie_ok = div_tie_ok && got == expected_tie;
        }
    }
    check(div_ok, "round_half_away vectors");
    check(div_tie_ok, "round_half_away exact ties");

    bool minmax_ok = true;
    for (uint32_t i = 0; i < n_minmax; i++) {
        int64_t low = rd_i64(cursor);
        int64_t high = rd_i64(cursor + 8);
        uint32_t count = rd_u32(cursor + 16);
        cursor += 20;
        const uint8_t* values = cursor;
        cursor += (size_t)count * 8;
        const uint8_t* expected = cursor;
        cursor += (size_t)count * 2;
        for (uint32_t k = 0; k < count; k++) {
            minmax_ok = minmax_ok &&
                        is_test_minmax_q15(rd_i64(values + (size_t)8 * k), low,
                                           high) == rd_u16(expected + (size_t)2 * k);
        }
    }
    check(minmax_ok, "minmax vectors");
    printf("  primitives.bin             n_sqrt=%u n_div=%u n_minmax=%u\n", n_sqrt,
           n_div, n_minmax);
}

void run_guards(const char* dir) {
    (void)dir;
    std::vector<uint8_t> group_raw((size_t)8 * IS_EMBED_DIM);
    std::vector<is_word_t> group_words((size_t)8 * IS_GM_WORDS_PER_GROUP);
    std::vector<uint8_t> wv_raw((size_t)IS_WV_ROWS * IS_EMBED_DIM, 0);
    std::vector<is_word_t> wv_words(IS_WV_WORDS);
    std::vector<is_word_t> query_words(IS_QUERY_WORDS);
    std::vector<uint16_t> score(IS_MAX_GROUPS);
    is_pack_int8_to_words((const int8_t*)group_raw.data(),
                          (int)group_raw.size(), group_words.data(),
                          (int)group_words.size());
    is_pack_int8_to_words((const int8_t*)wv_raw.data(), (int)wv_raw.size(),
                          wv_words.data(), IS_WV_WORDS);
    is_pack_int8_to_words((const int8_t*)group_raw.data(), IS_QUERY_DIM,
                          query_words.data(), IS_QUERY_WORDS);

    check(!is_group_count_legal(0), "group_count 0 illegal");
    check(!is_group_count_legal(4097), "group_count 4097 illegal");
    check(!is_group_count_legal(-1), "group_count -1 illegal");
    check(is_group_count_legal(1) && is_group_count_legal(4096),
          "group_count 1/4096 legal");

    check(integer_selector_scorer(group_words.data(), wv_words.data(),
                                  query_words.data(), score.data(), 0) ==
              IS_STATUS_ERR_GROUP_COUNT,
          "top rejects G=0");
    check(integer_selector_scorer(group_words.data(), wv_words.data(),
                                  query_words.data(), score.data(), 4097) ==
              IS_STATUS_ERR_GROUP_COUNT,
          "top rejects G=4097");
    check(integer_selector_scorer(group_words.data(), wv_words.data(),
                                  query_words.data(), score.data(), -1) ==
              IS_STATUS_ERR_GROUP_COUNT,
          "top rejects G=-1");
    check(integer_selector_scorer(0, wv_words.data(), query_words.data(),
                                  score.data(), 8) == IS_STATUS_ERR_NULL,
          "top rejects null group_mean_q");
    check(integer_selector_scorer(group_words.data(), 0, query_words.data(),
                                  score.data(), 8) == IS_STATUS_ERR_NULL,
          "top rejects null wv_q");
    check(integer_selector_scorer(group_words.data(), wv_words.data(), 0,
                                  score.data(), 8) == IS_STATUS_ERR_NULL,
          "top rejects null query_q");
    check(integer_selector_scorer(group_words.data(), wv_words.data(),
                                  query_words.data(), 0, 8) == IS_STATUS_ERR_NULL,
          "top rejects null score_q");
    printf("  guards                     illegal G / nulls rejected\n");
}

} /* namespace */

int main(int argc, char** argv) {
    const char* dir = (argc >= 2) ? argv[1] : getenv("IS_FIXTURE_DIR");
    if (dir == 0) {
        printf("usage: %s <fixture_dir>   (or set IS_FIXTURE_DIR)\n", argv[0]);
        return 2;
    }

    const char* fixtures[] = {
        "full_grounding.bin",     "full_pointing.bin",
        "synthetic_zero.bin",     "synthetic_all_equal.bin",
        "synthetic_g1.bin",       "synthetic_mixed_sign.bin",
        "stress_g400.bin",
    };
    const int fixture_count = (int)(sizeof(fixtures) / sizeof(fixtures[0]));

    printf("IS_TESTBEGIN %s\n", dir);
    for (int i = 0; i < fixture_count; i++) {
        run_scorer_fixture(dir, fixtures[i]);
    }
    run_primitives(dir);
    run_guards(dir);
    printf("IS_SUMMARY fixtures=%d passed=%d failed=%d\n", g_fixtures, g_passed,
           g_failed);
    if (g_failed != 0) {
        printf("IS_RESULT FAIL\n");
        return 1;
    }
    printf("IS_RESULT PASS\n");
    return 0;
}
