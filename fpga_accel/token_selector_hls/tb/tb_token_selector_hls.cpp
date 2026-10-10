/*
 * tb_token_selector_hls.cpp - g++ C-simulation testbench for the fixed-point
 * token selector.
 *
 * It reads the frozen fixture directory produced by
 * `tools/generate_fixtures.py`, runs `token_selector_top` for every case and
 * compares status/counts/scale and every output byte against the Python golden.
 *
 * Usage: tb_token_selector_hls <fixture_dir>
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "token_selector_hls.h"

namespace {

int failures = 0;
int passes = 0;

bool read_file(const std::string& path, std::vector<uint8_t>* data) {
  FILE* handle = fopen(path.c_str(), "rb");
  if (handle == NULL) {
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
  data->resize(static_cast<size_t>(size));
  if (size > 0) {
    size_t got = fread(data->data(), 1, static_cast<size_t>(size), handle);
    if (got != static_cast<size_t>(size)) {
      fclose(handle);
      return false;
    }
  }
  fclose(handle);
  return true;
}

bool take_bytes(const std::vector<uint8_t>& data, size_t* offset, void* out,
                size_t size) {
  if (*offset + size > data.size()) {
    return false;
  }
  memcpy(out, data.data() + *offset, size);
  *offset += size;
  return true;
}

bool take_i32(const std::vector<uint8_t>& data, size_t* offset, int* out) {
  int32_t value = 0;
  if (!take_bytes(data, offset, &value, sizeof(value))) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool take_f32(const std::vector<uint8_t>& data, size_t* offset, float* out) {
  return take_bytes(data, offset, out, sizeof(float));
}

void report(const std::string& name, bool ok, const std::string& detail) {
  if (ok) {
    passes += 1;
    printf("[PASS] %s\n", name.c_str());
  } else {
    failures += 1;
    printf("[FAIL] %s : %s\n", name.c_str(), detail.c_str());
  }
}

/* Deterministic xorshift64 so the random differential is reproducible. */
uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

uint64_t rng_next() {
  uint64_t x = rng_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  rng_state = x;
  return x;
}

int rng_range(int lo, int hi) {
  return lo + static_cast<int>(rng_next() % static_cast<uint64_t>(hi - lo + 1));
}

/*
 * Byte-for-byte differential of the new multi-pass packed kernel against the
 * frozen v3 bounded-insertion int8 kernel (the `#ifndef __SYNTHESIS__`
 * reference).  Only the selection structures differ, so any divergence shows up
 * in status/counts/tokens/metadata.
 */
bool compare_old_vs_new(const std::vector<int8_t>& tokens_q,
                        const std::vector<uint16_t>& score_q, int group_count,
                        int grid_h, int grid_w, int budget, int local_cap,
                        float scale_in, std::string* detail) {
  int max_output = (group_count > budget) ? group_count : budget;

  std::vector<int8_t> out_tokens(static_cast<size_t>(max_output) * TS_GROUP_ELEMENTS);
  std::vector<uint16_t> out_gi(max_output, 0);
  std::vector<uint8_t> out_st(max_output, 0);
  std::vector<uint8_t> out_ri(max_output, 0);
  std::vector<uint8_t> out_v(max_output, 0);
  float scale_out = 0.0f;
  int oc = -1, rc = -1, cc = -1;
  int status = token_selector_top(
      tokens_q.data(), score_q.data(), out_tokens.data(), out_gi.data(),
      out_st.data(), out_ri.data(), out_v.data(), group_count, grid_h, grid_w,
      budget, local_cap, scale_in, &scale_out, &oc, &rc, &cc);

  std::vector<ts_word_t> tokens_words(static_cast<size_t>(group_count) * TS_GROUP_WORDS);
  ts_pack_int8_to_words(tokens_q.data(), static_cast<int>(tokens_q.size()),
                        tokens_words.data(),
                        static_cast<int>(tokens_words.size()));
  std::vector<ts_word_t> out_tokens_p(static_cast<size_t>(max_output) * TS_GROUP_WORDS);
  std::vector<uint16_t> out_gi_p(max_output, 0);
  std::vector<uint8_t> out_st_p(max_output, 0);
  std::vector<uint8_t> out_ri_p(max_output, 0);
  std::vector<uint8_t> out_v_p(max_output, 0);
  float scale_out_p = 0.0f;
  int oc_p = -1, rc_p = -1, cc_p = -1;
  int status_p = token_selector_top_packed(
      tokens_words.data(), score_q.data(), out_tokens_p.data(), out_gi_p.data(),
      out_st_p.data(), out_ri_p.data(), out_v_p.data(), group_count, grid_h,
      grid_w, budget, local_cap, scale_in, &scale_out_p, &oc_p, &rc_p, &cc_p);

  if (status_p != status) {
    *detail = "v3-vs-v4 status mismatch";
    return false;
  }
  if (oc_p != oc || rc_p != rc || cc_p != cc) {
    *detail = "v3-vs-v4 count mismatch";
    return false;
  }
  if (memcmp(&scale_out_p, &scale_out, sizeof(float)) != 0) {
    *detail = "v3-vs-v4 scale mismatch";
    return false;
  }
  if (status != TS_STATUS_OK) {
    return true;
  }
  size_t token_bytes = static_cast<size_t>(oc) * TS_GROUP_ELEMENTS;
  std::vector<int8_t> unpacked(token_bytes);
  ts_unpack_words_to_int8(out_tokens_p.data(), oc * TS_GROUP_WORDS,
                          unpacked.data(), static_cast<int>(token_bytes));
  if (memcmp(unpacked.data(), out_tokens.data(), token_bytes) != 0) {
    *detail = "v3-vs-v4 tokens mismatch";
    return false;
  }
  if (memcmp(out_gi_p.data(), out_gi.data(), static_cast<size_t>(oc) * sizeof(uint16_t)) != 0 ||
      memcmp(out_st_p.data(), out_st.data(), static_cast<size_t>(oc)) != 0 ||
      memcmp(out_ri_p.data(), out_ri.data(), static_cast<size_t>(oc)) != 0 ||
      memcmp(out_v_p.data(), out_v.data(), static_cast<size_t>(oc)) != 0) {
    *detail = "v3-vs-v4 metadata mismatch";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <fixture_dir>\n", argv[0]);
    return 2;
  }
  std::string fixture_dir = argv[1];
  std::string manifest_path = fixture_dir + "/manifest.txt";
  FILE* manifest = fopen(manifest_path.c_str(), "r");
  if (manifest == NULL) {
    fprintf(stderr, "cannot open %s\n", manifest_path.c_str());
    return 2;
  }

  char line[512];
  while (fgets(line, sizeof(line), manifest) != NULL) {
    char name[256];
    int group_count = 0;
    int grid_h = 0;
    int grid_w = 0;
    int budget = 0;
    int local_cap = 0;
    if (sscanf(line, "%255s %d %d %d %d %d", name, &group_count, &grid_h,
               &grid_w, &budget, &local_cap) != 6) {
      continue;
    }

    std::vector<uint8_t> input;
    std::string input_path = fixture_dir + "/" + name + ".in.bin";
    if (!read_file(input_path, &input) || input.size() < 4 ||
        memcmp(input.data(), "TSIN", 4) != 0) {
      report(name, false, "cannot read input fixture");
      continue;
    }
    size_t in_offset = 4;
    int in_g = 0;
    int in_gh = 0;
    int in_gw = 0;
    int in_budget = 0;
    int in_cap = 0;
    float in_scale = 0.0f;
    if (!take_i32(input, &in_offset, &in_g) ||
        !take_i32(input, &in_offset, &in_gh) ||
        !take_i32(input, &in_offset, &in_gw) ||
        !take_i32(input, &in_offset, &in_budget) ||
        !take_i32(input, &in_offset, &in_cap) ||
        !take_f32(input, &in_offset, &in_scale)) {
      report(name, false, "malformed input header");
      continue;
    }
    if (in_g != group_count || in_gh != grid_h || in_gw != grid_w ||
        in_budget != budget || in_cap != local_cap) {
      report(name, false, "input header mismatches manifest");
      continue;
    }
    std::vector<int8_t> tokens_q(static_cast<size_t>(group_count) * TS_GROUP_ELEMENTS);
    std::vector<uint16_t> score_q(static_cast<size_t>(group_count));
    if (!take_bytes(input, &in_offset, tokens_q.data(), tokens_q.size() * sizeof(int8_t)) ||
        !take_bytes(input, &in_offset, score_q.data(), score_q.size() * sizeof(uint16_t))) {
      report(name, false, "malformed input payload");
      continue;
    }

    /* Same byte stream, reinterpreted as 128-bit little-endian words. */
    std::vector<ts_word_t> tokens_words(
        static_cast<size_t>(group_count) * TS_GROUP_WORDS);
    ts_pack_int8_to_words(tokens_q.data(), static_cast<int>(tokens_q.size()),
                          tokens_words.data(),
                          static_cast<int>(tokens_words.size()));

    std::vector<uint8_t> expected;
    std::string expected_path = fixture_dir + "/" + name + ".exp.bin";
    if (!read_file(expected_path, &expected) || expected.size() < 4 ||
        memcmp(expected.data(), "TSEX", 4) != 0) {
      report(name, false, "cannot read expected fixture");
      continue;
    }
    size_t exp_offset = 4;
    int exp_status = 0;
    int exp_out = 0;
    int exp_real = 0;
    int exp_ctx = 0;
    float exp_scale = 0.0f;
    if (!take_i32(expected, &exp_offset, &exp_status) ||
        !take_i32(expected, &exp_offset, &exp_out) ||
        !take_i32(expected, &exp_offset, &exp_real) ||
        !take_i32(expected, &exp_offset, &exp_ctx) ||
        !take_f32(expected, &exp_offset, &exp_scale)) {
      report(name, false, "malformed expected header");
      continue;
    }

    int max_output = (group_count > budget) ? group_count : budget;
    std::vector<int8_t> out_tokens(static_cast<size_t>(max_output) * TS_GROUP_ELEMENTS);
    std::vector<uint16_t> out_group_index(static_cast<size_t>(max_output), 0);
    std::vector<uint8_t> out_source_type(static_cast<size_t>(max_output), 0);
    std::vector<uint8_t> out_region_id(static_cast<size_t>(max_output), 0);
    std::vector<uint8_t> out_valid(static_cast<size_t>(max_output), 0);
    float scale_out = 0.0f;
    int output_count = -1;
    int real_count = -1;
    int context_count = -1;
    int status = token_selector_top(
        tokens_q.data(), score_q.data(), out_tokens.data(), out_group_index.data(),
        out_source_type.data(), out_region_id.data(), out_valid.data(), group_count,
        grid_h, grid_w, budget, local_cap, in_scale, &scale_out, &output_count,
        &real_count, &context_count);

    bool ok = true;
    std::string detail;
    if (status != exp_status) {
      ok = false;
      detail = "status mismatch";
    } else if (output_count != exp_out || real_count != exp_real ||
               context_count != exp_ctx) {
      ok = false;
      detail = "count mismatch";
    } else if (memcmp(&scale_out, &exp_scale, sizeof(float)) != 0) {
      ok = false;
      detail = "scale_out mismatch";
    }

    if (ok && status == TS_STATUS_OK) {
      size_t token_bytes = static_cast<size_t>(exp_out) * TS_GROUP_ELEMENTS;
      size_t index_bytes = static_cast<size_t>(exp_out) * sizeof(uint16_t);
      size_t meta_bytes = static_cast<size_t>(exp_out);
      size_t needed = exp_offset + token_bytes + index_bytes + 3 * meta_bytes;
      if (expected.size() < needed) {
        ok = false;
        detail = "expected payload too short";
      } else {
        const uint8_t* base = expected.data() + exp_offset;
        if (memcmp(out_tokens.data(), base, token_bytes) != 0) {
          ok = false;
          detail = "tokens_out mismatch";
        } else if (memcmp(out_group_index.data(), base + token_bytes, index_bytes) != 0) {
          ok = false;
          detail = "group_index mismatch";
        } else if (memcmp(out_source_type.data(), base + token_bytes + index_bytes, meta_bytes) != 0) {
          ok = false;
          detail = "source_type mismatch";
        } else if (memcmp(out_region_id.data(), base + token_bytes + index_bytes + meta_bytes, meta_bytes) != 0) {
          ok = false;
          detail = "region_id mismatch";
        } else if (memcmp(out_valid.data(), base + token_bytes + index_bytes + 2 * meta_bytes, meta_bytes) != 0) {
          ok = false;
          detail = "valid mismatch";
        } else {
          for (int i = 0; i < exp_out; i++) {
            if (out_valid[i] != 1) {
              ok = false;
              detail = "valid contains false";
              break;
            }
          }
          if (ok && exp_out > 1) {
            for (int i = 1; i < exp_out; i++) {
              if (out_group_index[i] <= out_group_index[i - 1]) {
                ok = false;
                detail = "group_index not strictly ascending/unique";
                break;
              }
            }
          }
        }
      }
    }
    /* The 128-bit packed top must produce the identical result. */
    std::vector<ts_word_t> out_tokens_words(
        static_cast<size_t>(max_output) * TS_GROUP_WORDS);
    std::vector<uint16_t> out_group_index_p(static_cast<size_t>(max_output), 0);
    std::vector<uint8_t> out_source_type_p(static_cast<size_t>(max_output), 0);
    std::vector<uint8_t> out_region_id_p(static_cast<size_t>(max_output), 0);
    std::vector<uint8_t> out_valid_p(static_cast<size_t>(max_output), 0);
    float scale_out_p = 0.0f;
    int output_count_p = -1;
    int real_count_p = -1;
    int context_count_p = -1;
    int status_p = token_selector_top_packed(
        tokens_words.data(), score_q.data(), out_tokens_words.data(),
        out_group_index_p.data(), out_source_type_p.data(),
        out_region_id_p.data(), out_valid_p.data(), group_count, grid_h, grid_w,
        budget, local_cap, in_scale, &scale_out_p, &output_count_p, &real_count_p,
        &context_count_p);

    if (status_p != status) {
      ok = false;
      detail = "packed status mismatch";
    } else if (output_count_p != output_count || real_count_p != real_count ||
               context_count_p != context_count) {
      ok = false;
      detail = "packed count mismatch";
    } else if (memcmp(&scale_out_p, &scale_out, sizeof(float)) != 0) {
      ok = false;
      detail = "packed scale_out mismatch";
    } else if (status == TS_STATUS_OK) {
      size_t token_bytes = static_cast<size_t>(exp_out) * TS_GROUP_ELEMENTS;
      std::vector<int8_t> unpacked(token_bytes);
      ts_unpack_words_to_int8(out_tokens_words.data(), exp_out * TS_GROUP_WORDS,
                              unpacked.data(), static_cast<int>(token_bytes));
      const uint8_t* base = expected.data() + exp_offset;
      if (memcmp(unpacked.data(), out_tokens.data(), token_bytes) != 0) {
        ok = false;
        detail = "packed tokens differ from int8 top";
      } else if (memcmp(unpacked.data(), base, token_bytes) != 0) {
        ok = false;
        detail = "packed tokens differ from golden";
      } else if (memcmp(out_group_index_p.data(), out_group_index.data(),
                        static_cast<size_t>(exp_out) * sizeof(uint16_t)) != 0 ||
                 memcmp(out_source_type_p.data(), out_source_type.data(),
                        static_cast<size_t>(exp_out)) != 0 ||
                 memcmp(out_region_id_p.data(), out_region_id.data(),
                        static_cast<size_t>(exp_out)) != 0 ||
                 memcmp(out_valid_p.data(), out_valid.data(),
                        static_cast<size_t>(exp_out)) != 0) {
        ok = false;
        detail = "packed metadata differs from int8 top";
      }
    }
    report(name, ok, detail);
  }
  fclose(manifest);

  /*
   * The 220 deterministic random differential cases retained from v4 against
   * the frozen v3 bounded-insertion int8 kernel (new multi-pass packed vs old
   * insertion).  Scores are drawn from a quantised range (ceiling
   * 32767/score_mod, rounded to a multiple of score_mod); score_mod == 1 is the
   * unquantised contract range, while larger values raise the tie frequency.
   * This is NOT an explicit all-tie mode -- the explicit tie patterns live in
   * the 48 deterministic boundary differentials below.
   */
  {
    static const int shapes[][2] = {
        {1, 48},  {2, 24},  {3, 16},  {4, 12},  {6, 8},   {8, 8},
        {9, 9},   {10, 12}, {12, 12}, {16, 16}, {1, 64},  {2, 32},
        {4, 16},  {5, 20},  {8, 16},  {16, 8},  {3, 21},  {7, 9},
        {13, 13}, {1, 96},  {2, 48},  {4, 24},  {12, 8},  {20, 5},
        {1, 128}, {2, 64},  {4, 32},  {8, 4},   {16, 2},  {32, 4},
    };
    const int shape_count = static_cast<int>(sizeof(shapes) / sizeof(shapes[0]));
    static const int budgets[6] = {16, 32, 48, 64, 80, 96};
    const int case_count = 220;
    for (int c = 0; c < case_count; c++) {
      const int* shape = shapes[rng_next() % static_cast<uint64_t>(shape_count)];
      int grid_h = shape[0];
      int grid_w = shape[1];
      int group_count = grid_h * grid_w;
      int budget = budgets[rng_next() % 6];
      int local_cap = rng_range(1, TS_MAX_LOCAL_CANDIDATE_CAP);
      int score_mod = rng_range(1, 6);
      float scale_in = static_cast<float>(rng_range(1, 1000)) / 37.0f;

      std::vector<int8_t> tokens(static_cast<size_t>(group_count) * TS_GROUP_ELEMENTS);
      for (size_t i = 0; i < tokens.size(); i++) {
        tokens[i] = static_cast<int8_t>(rng_next() & 0xFF);
      }
      std::vector<uint16_t> scores(group_count);
      int score_hi = 32767 / score_mod;
      for (int i = 0; i < group_count; i++) {
        scores[i] = static_cast<uint16_t>((rng_next() % static_cast<uint64_t>(score_hi + 1)) * score_mod);
      }

      std::string detail;
      bool ok = compare_old_vs_new(tokens, scores, group_count, grid_h, grid_w,
                                   budget, local_cap, scale_in, &detail);
      char name[64];
      snprintf(name, sizeof(name), "randdiff_v3_%03d", c);
      report(name, ok, detail);
    }
  }

  /*
   * >= 40 deterministic boundary differentials: the packed v4 multi-pass
   * kernel byte-for-byte against the frozen v3 bounded-insertion int8 kernel.
   *
   * Coverage: valid large grids 32x128, 64x64, 128x32, 31x127, 127x31 plus the
   * typical 16x16, 24x24, 32x32; every boundary budget K=16/32/48/64/80/96;
   * local_cap 1 and 16; and four explicit score patterns -- all-equal,
   * alternating 0/65535, repeated tiers, random saturated values.
   *
   * Memory: the input and both packed/int8 outputs for one case are scoped to
   * the loop body and destroyed before the next case, so peak usage stays at a
   * single case even for G=4096.  The 0/65535 alternating mode leaves at most
   * half of each region's weights non-zero, so the proven centroid bound
   * (total <= 512*32767) still holds and ASan/UBSan stays overflow-free.
   */
  {
    static const int boundary_shapes[][2] = {
        {32, 128}, {64, 64}, {128, 32}, {31, 127}, {127, 31},
        {16, 16},  {24, 24}, {32, 32},
    };
    const int boundary_shape_count =
        static_cast<int>(sizeof(boundary_shapes) / sizeof(boundary_shapes[0]));
    static const int boundary_budgets[6] = {16, 32, 48, 64, 80, 96};
    int boundary_index = 0;
    for (int s = 0; s < boundary_shape_count; s++) {
      for (int b = 0; b < 6; b++) {
        int grid_h = boundary_shapes[s][0];
        int grid_w = boundary_shapes[s][1];
        int group_count = grid_h * grid_w;
        int budget = boundary_budgets[b];
        int local_cap = ((budget / 16) % 2 == 1) ? 1 : 16;
        int mode = boundary_index % 4;
        float scale_in = 0.5f + 0.25f * static_cast<float>(boundary_index % 5);
        {
          std::vector<int8_t> tokens(
              static_cast<size_t>(group_count) * TS_GROUP_ELEMENTS);
          for (size_t i = 0; i < tokens.size(); i++) {
            tokens[i] = static_cast<int8_t>(rng_next() & 0xFF);
          }
          std::vector<uint16_t> scores(group_count);
          switch (mode) {
            case 0: /* all equal */
              for (int i = 0; i < group_count; i++) {
                scores[i] = 32767;
              }
              break;
            case 1: /* alternating 0 / 65535 (half-zero => proven bound holds) */
              for (int i = 0; i < group_count; i++) {
                scores[i] = (i & 1) ? static_cast<uint16_t>(65535)
                                    : static_cast<uint16_t>(0);
              }
              break;
            case 2: /* many repeated tiers: {0,10922,21844,32766} */
              for (int i = 0; i < group_count; i++) {
                scores[i] = static_cast<uint16_t>((i % 4) * 10922);
              }
              break;
            default: /* random saturated values (50% at the Q0.15 max) */
              for (int i = 0; i < group_count; i++) {
                uint64_t h = static_cast<uint64_t>(i + 1) *
                             0x9E3779B97F4A7C15ULL;
                h ^= h >> 29;
                h *= 0xBF58476D1CE4E5B9ULL;
                h ^= h >> 32;
                scores[i] = (h % 100 < 50) ? static_cast<uint16_t>(32767)
                                           : static_cast<uint16_t>((h >> 7) % 32768);
              }
              break;
          }
          std::string detail;
          bool ok = compare_old_vs_new(tokens, scores, group_count, grid_h,
                                       grid_w, budget, local_cap, scale_in,
                                       &detail);
          char name[96];
          snprintf(name, sizeof(name), "bounddiff_v4_%02d_%dx%d_K%d_cap%d_m%d",
                   boundary_index, grid_h, grid_w, budget, local_cap, mode);
          report(name, ok, detail);
        }
        boundary_index++;
      }
    }
  }

  /* Layout parity. */
  FILE* layout_file = fopen((fixture_dir + "/layout.txt").c_str(), "r");
  if (layout_file != NULL) {
    while (fgets(line, sizeof(line), layout_file) != NULL) {
      int budget = 0;
      int total = 0;
      int offsets[TS_NUM_SEGMENTS];
      int sizes[TS_NUM_SEGMENTS];
      int read = sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %d", &budget,
                        &total, &offsets[0], &offsets[1], &offsets[2],
                        &offsets[3], &offsets[4], &sizes[0], &sizes[1],
                        &sizes[2], &sizes[3], &sizes[4]);
      if (read != 12) {
        continue;
      }
      struct TokenSelectorLayout layout;
      int status = token_selector_layout(budget, &layout);
      bool ok = status == TS_STATUS_OK && layout.total_bytes == total;
      for (int i = 0; ok && i < TS_NUM_SEGMENTS; i++) {
        ok = layout.offsets[i] == offsets[i] && layout.sizes[i] == sizes[i];
      }
      char layout_name[64];
      snprintf(layout_name, sizeof(layout_name), "layout_K%d", budget);
      report(layout_name, ok, "layout mismatch");
    }
    fclose(layout_file);
  }

  /* Group-index guard helpers. */
  report("group_limit_1", token_selector_check_group_limit(1) == 1, "expected 1");
  report("group_limit_4096", token_selector_check_group_limit(4096) == 1, "expected 1");
  report("group_limit_65535", token_selector_check_group_limit(65535) == 1, "expected 1");
  report("group_limit_0", token_selector_check_group_limit(0) == 0, "expected 0");
  report("group_limit_65536", token_selector_check_group_limit(65536) == 0, "expected 0");

  /* MAX_GRID_DIM = 128 boundary helpers. */
  report("grid_1x128", token_selector_check_grid_dims(1, 128) == 1, "expected 1");
  report("grid_128x1", token_selector_check_grid_dims(128, 1) == 1, "expected 1");
  report("grid_128x32", token_selector_check_grid_dims(128, 32) == 1, "expected 1");
  report("grid_128x128", token_selector_check_grid_dims(128, 128) == 1, "expected 1");
  report("grid_1x129", token_selector_check_grid_dims(1, 129) == 0, "expected 0");
  report("grid_129x1", token_selector_check_grid_dims(129, 1) == 0, "expected 0");
  report("grid_0x5", token_selector_check_grid_dims(0, 5) == 0, "expected 0");
  report("grid_5x0", token_selector_check_grid_dims(5, 0) == 0, "expected 0");
  report("budget_47", token_selector_check_budget(47) == 0, "expected 0");
  report("budget_80", token_selector_check_budget(80) == 1, "expected 1");
  report("budget_96", token_selector_check_budget(96) == 1, "expected 1");
  report("budget_112", token_selector_check_budget(112) == 0, "expected 0");
  report("budget_128", token_selector_check_budget(128) == 0, "expected 0");

  /* ---- 128-bit pack/unpack contract: arbitrary byte offset / lane ---- */
  {
    bool ok = true;
    std::string detail;
    for (int shift = 0; shift < TS_PACKED_LANES && ok; shift++) {
      std::vector<ts_word_t> words(4, (ts_word_t)0);
      std::vector<int8_t> expected(4 * TS_PACKED_LANES);
      for (int index = 0; index < 4 * TS_PACKED_LANES; index++) {
        expected[index] = (int8_t)((index * 7 + shift * 13 + 5) & 0xFF);
        int w = index / TS_PACKED_LANES;
        int lane = (index + shift) % TS_PACKED_LANES;
        words[w] = ts_word_set_lane(words[w], lane, expected[index]);
      }
      for (int index = 0; index < 4 * TS_PACKED_LANES && ok; index++) {
        int w = index / TS_PACKED_LANES;
        int lane = (index + shift) % TS_PACKED_LANES;
        if (ts_word_lane(words[w], lane) != (int)expected[index]) {
          ok = false;
          detail = "lane readback mismatch";
        }
      }
      std::vector<int8_t> packed(4 * TS_PACKED_LANES);
      ts_unpack_words_to_int8(words.data(), 4, packed.data(),
                              static_cast<int>(packed.size()));
      std::vector<ts_word_t> again(4, (ts_word_t)0);
      ts_pack_int8_to_words(packed.data(), static_cast<int>(packed.size()),
                            again.data(), 4);
      if (ok && memcmp(again.data(), words.data(), 4 * sizeof(ts_word_t)) != 0) {
        ok = false;
        detail = "word array not stable under pack/unpack";
      }
    }
    report("pack_lane_offsets", ok, detail);
  }
  {
    /* Non-zero word offset and a non-multiple-of-16 tail. */
    std::vector<int8_t> bytes(70);
    for (int i = 0; i < 70; i++) {
      bytes[i] = (int8_t)((i * 31 + 7) & 0xFF);
    }
    std::vector<ts_word_t> words(9, (ts_word_t)0);
    ts_pack_int8_to_words(bytes.data(), 70, words.data() + 3, 5);
    bool ok = true;
    std::string detail;
    for (int w = 0; w < 3; w++) {
      if (words[w] != (ts_word_t)0) {
        ok = false;
        detail = "leading word not preserved";
      }
    }
    std::vector<int8_t> out(70);
    ts_unpack_words_to_int8(words.data() + 3, 5, out.data(), 70);
    if (ok && memcmp(out.data(), bytes.data(), 70) != 0) {
      ok = false;
      detail = "byte offset round-trip mismatch";
    }
    if (ok) {
      for (int index = 70; index < 5 * TS_PACKED_LANES; index++) {
        int w = 3 + index / TS_PACKED_LANES;
        int lane = index % TS_PACKED_LANES;
        if (ts_word_lane(words[w], lane) != 0) {
          ok = false;
          detail = "tail not zero padded";
          break;
        }
      }
    }
    report("pack_byte_offset_tail", ok, detail);
  }

  printf("TB token_selector_hls: passed=%d failed=%d\n", passes, failures);
  return failures == 0 ? 0 : 1;
}
