/*
 * tb_integer_selector_pipeline.cpp - host / HLS C-simulation testbench.
 *
 * Proves, byte for byte, that the fused top
 *
 *   group_mean_q + wv_q + query_q -> (on-chip score_q) -> selected tokens
 *
 * reproduces:
 *
 *   1. the frozen Python scorer golden ``integer_chain`` for the on-chip
 *      intermediate ``score_q`` (via the host-only debug twin);
 *   2. the frozen Python selector golden ``select_fixedpoint`` for the final
 *      5-segment output (tokens + group_index + source_type + region_id +
 *      valid, plus status/counts/scale);
 *   3. the exact sequential composition of the two accepted standalone HLS
 *      kernels: ``integer_selector_scorer`` then
 *      ``token_selector_core_packed`` (byte-for-byte);
 *   4. the frozen int8 v3 selector kernel (extra differential);
 *   5. determinism: the same input twice gives identical bytes;
 *   6. the rejection paths: null pointers, illegal G / grid / budget /
 *      local_cap.
 *
 * Usage: tb_integer_selector_pipeline <fixture_dir>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "integer_selector_pipeline.h"

namespace {

int g_passed = 0;
int g_failed = 0;
int g_cases = 0;

void check(bool ok, const std::string& what) {
  if (ok) {
    g_passed++;
  } else {
    g_failed++;
    printf("  FAIL: %s\n", what.c_str());
  }
}

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
int32_t rd_i32(const uint8_t* p) {
  int32_t value = 0;
  memcpy(&value, p, 4);
  return value;
}
uint16_t rd_u16(const uint8_t* p) {
  uint16_t value = 0;
  memcpy(&value, p, 2);
  return value;
}
float rd_f32(const uint8_t* p) {
  float value = 0.0f;
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

/* ---------------------------------------------------------------- */
/* Scene                                                             */
/* ---------------------------------------------------------------- */
struct Scene {
  int group_count;
  int grid_h;
  int grid_w;
  std::vector<int8_t> tokens_raw;
  std::vector<int8_t> group_mean_raw;
  std::vector<int8_t> query_raw;
};

bool load_scene(const std::string& path, Scene* scene) {
  std::vector<uint8_t> data;
  if (!read_file(path, &data) || data.size() < 64 ||
      memcmp(data.data(), "ISPSCN01", 8) != 0) {
    return false;
  }
  scene->group_count = (int)rd_u32(data.data() + 8);
  scene->grid_h = (int)rd_u32(data.data() + 12);
  scene->grid_w = (int)rd_u32(data.data() + 16);
  uint64_t tokens_fnv = rd_u64(data.data() + 24);
  uint64_t gm_fnv = rd_u64(data.data() + 32);
  uint64_t query_fnv = rd_u64(data.data() + 40);
  uint64_t payload = rd_u64(data.data() + 48);
  if (payload != data.size() - 64) {
    return false;
  }
  const uint8_t* p = data.data() + 64;
  size_t token_bytes = (size_t)scene->group_count * 4096;
  size_t gm_bytes = (size_t)scene->group_count * 1024;
  scene->tokens_raw.assign((const int8_t*)p, (const int8_t*)p + token_bytes);
  p += token_bytes;
  scene->group_mean_raw.assign((const int8_t*)p, (const int8_t*)p + gm_bytes);
  p += gm_bytes;
  scene->query_raw.assign((const int8_t*)p, (const int8_t*)p + 128);
  if (fnv1a64((const uint8_t*)scene->tokens_raw.data(), token_bytes) != tokens_fnv ||
      fnv1a64((const uint8_t*)scene->group_mean_raw.data(), gm_bytes) != gm_fnv ||
      fnv1a64((const uint8_t*)scene->query_raw.data(), 128) != query_fnv) {
    return false;
  }
  return true;
}

/* ---------------------------------------------------------------- */
/* Expected                                                          */
/* ---------------------------------------------------------------- */
struct Expected {
  int status;
  int output_count;
  int real_count;
  int context_count;
  float scale_out;
  uint64_t score_fnv;
  uint64_t tokens_fnv;
  uint64_t group_index_fnv;
  uint64_t metadata_fnv;
  std::vector<uint16_t> score_q;
  std::vector<int8_t> tokens_out;
  std::vector<uint16_t> group_index;
  std::vector<uint8_t> source_type;
  std::vector<uint8_t> region_id;
  std::vector<uint8_t> valid;
};

bool load_expected(const std::string& path, int group_count, Expected* exp) {
  std::vector<uint8_t> data;
  if (!read_file(path, &data) || data.size() < 80 ||
      memcmp(data.data(), "ISPEX001", 8) != 0) {
    return false;
  }
  exp->status = rd_i32(data.data() + 8);
  exp->output_count = rd_i32(data.data() + 12);
  exp->real_count = rd_i32(data.data() + 16);
  exp->context_count = rd_i32(data.data() + 20);
  exp->scale_out = rd_f32(data.data() + 24);
  exp->score_fnv = rd_u64(data.data() + 32);
  exp->tokens_fnv = rd_u64(data.data() + 40);
  exp->group_index_fnv = rd_u64(data.data() + 48);
  exp->metadata_fnv = rd_u64(data.data() + 56);
  uint64_t payload = rd_u64(data.data() + 64);
  if (payload != data.size() - 80) {
    return false;
  }
  const uint8_t* p = data.data() + 80;
  size_t score_bytes = (size_t)group_count * 2;
  exp->score_q.resize((size_t)group_count);
  for (int i = 0; i < group_count; i++) {
    exp->score_q[(size_t)i] = rd_u16(p + (size_t)2 * i);
  }
  if (fnv1a64(p, score_bytes) != exp->score_fnv) {
    return false;
  }
  p += score_bytes;
  if (exp->status == 0) {
    size_t token_bytes = (size_t)exp->output_count * 4096;
    exp->tokens_out.assign((const int8_t*)p, (const int8_t*)p + token_bytes);
    p += token_bytes;
    exp->group_index.resize((size_t)exp->output_count);
    for (int i = 0; i < exp->output_count; i++) {
      exp->group_index[(size_t)i] = rd_u16(p + (size_t)2 * i);
    }
    exp->group_index_fnv = fnv1a64(p, (size_t)exp->output_count * 2);
    p += (size_t)exp->output_count * 2;
    exp->source_type.assign((const uint8_t*)p, (const uint8_t*)p + exp->output_count);
    p += exp->output_count;
    exp->region_id.assign((const uint8_t*)p, (const uint8_t*)p + exp->output_count);
    p += exp->output_count;
    exp->valid.assign((const uint8_t*)p, (const uint8_t*)p + exp->output_count);
    p += exp->output_count;
    std::vector<uint8_t> metadata;
    metadata.insert(metadata.end(), exp->source_type.begin(), exp->source_type.end());
    metadata.insert(metadata.end(), exp->region_id.begin(), exp->region_id.end());
    metadata.insert(metadata.end(), exp->valid.begin(), exp->valid.end());
    exp->metadata_fnv = fnv1a64(metadata.data(), metadata.size());
  }
  return true;
}

/* ---------------------------------------------------------------- */
/* One case                                                          */
/* ---------------------------------------------------------------- */
bool run_case(const std::string& fixture_dir, const std::string& case_name,
              const std::string& scene_file, int budget, int local_cap,
              float scale, const std::vector<is_word_t>& wv_words) {
  Scene scene;
  if (!load_scene(fixture_dir + "/" + scene_file, &scene)) {
    check(false, case_name + ": scene load");
    return false;
  }
  if (scene.group_count < 1 || scene.group_count > IS_MAX_GROUPS) {
    check(false, case_name + ": illegal scene G");
    return false;
  }
  Expected exp;
  if (!load_expected(fixture_dir + "/" + case_name + ".exp.bin",
                     scene.group_count, &exp)) {
    check(false, case_name + ": expected load");
    return false;
  }

  /* Packed inputs. */
  std::vector<ts_word_t> tokens_words((size_t)scene.group_count * TS_GROUP_WORDS);
  ts_pack_int8_to_words(scene.tokens_raw.data(), (int)scene.tokens_raw.size(),
                        tokens_words.data(), (int)tokens_words.size());
  std::vector<is_word_t> gm_words((size_t)scene.group_count * IS_GM_WORDS_PER_GROUP);
  is_pack_int8_to_words(scene.group_mean_raw.data(), (int)scene.group_mean_raw.size(),
                        gm_words.data(), (int)gm_words.size());
  std::vector<is_word_t> query_words(IS_QUERY_WORDS);
  is_pack_int8_to_words(scene.query_raw.data(), IS_QUERY_DIM, query_words.data(),
                        IS_QUERY_WORDS);

  int max_output = (scene.group_count > budget) ? scene.group_count : budget;
  size_t out_words = (size_t)max_output * TS_GROUP_WORDS;

  /* ---- debug twin: on-chip score_q + selection ---- */
  std::vector<ts_word_t> dbg_tokens(out_words);
  std::vector<uint16_t> dbg_gi((size_t)max_output, 0);
  std::vector<uint8_t> dbg_st((size_t)max_output, 0);
  std::vector<uint8_t> dbg_ri((size_t)max_output, 0);
  std::vector<uint8_t> dbg_v((size_t)max_output, 0);
  std::vector<uint16_t> dbg_score((size_t)scene.group_count, 0);
  std::vector<int32_t> dbg_sem((size_t)scene.group_count, 0);
  std::vector<uint16_t> dbg_l2((size_t)scene.group_count, 0);
  std::vector<uint16_t> dbg_semnorm((size_t)scene.group_count, 0);
  std::vector<uint16_t> dbg_l2norm((size_t)scene.group_count, 0);
  float dbg_scale = 0.0f;
  int dbg_oc = -1, dbg_rc = -1, dbg_cc = -1;
  int dbg_status = integer_selector_pipeline_debug(
      tokens_words.data(), gm_words.data(), wv_words.data(), query_words.data(),
      dbg_tokens.data(), dbg_gi.data(), dbg_st.data(), dbg_ri.data(), dbg_v.data(),
      dbg_score.data(), dbg_sem.data(), dbg_l2.data(), dbg_semnorm.data(),
      dbg_l2norm.data(), scene.group_count, scene.grid_h, scene.grid_w, budget,
      local_cap, scale, &dbg_scale, &dbg_oc, &dbg_rc, &dbg_cc);

  /* ---- top ---- */
  std::vector<ts_word_t> top_tokens(out_words);
  std::vector<uint16_t> top_gi((size_t)max_output, 0);
  std::vector<uint8_t> top_st((size_t)max_output, 0);
  std::vector<uint8_t> top_ri((size_t)max_output, 0);
  std::vector<uint8_t> top_v((size_t)max_output, 0);
  float top_scale = 0.0f;
  int top_oc = -1, top_rc = -1, top_cc = -1;
  int top_status = integer_selector_pipeline(
      tokens_words.data(), gm_words.data(), wv_words.data(), query_words.data(),
      top_tokens.data(), top_gi.data(), top_st.data(), top_ri.data(), top_v.data(),
      scene.group_count, scene.grid_h, scene.grid_w, budget, local_cap, scale,
      &top_scale, &top_oc, &top_rc, &top_cc);

  /* ---- standalone sequential composition: scorer then packed selector ---- */
  std::vector<uint16_t> ref_score((size_t)scene.group_count, 0);
  int ref_score_status = integer_selector_scorer(
      gm_words.data(), wv_words.data(), query_words.data(), ref_score.data(),
      scene.group_count);
  std::vector<ts_word_t> ref_tokens(out_words);
  std::vector<uint16_t> ref_gi((size_t)max_output, 0);
  std::vector<uint8_t> ref_st((size_t)max_output, 0);
  std::vector<uint8_t> ref_ri((size_t)max_output, 0);
  std::vector<uint8_t> ref_v((size_t)max_output, 0);
  struct TokenSelectorResult ref_result;
  float ref_scale = 0.0f;
  int ref_status = token_selector_core_packed(
      tokens_words.data(), ref_score.data(), scene.group_count, scene.grid_h,
      scene.grid_w, budget, local_cap, scale, &ref_scale, ref_tokens.data(),
      ref_gi.data(), ref_st.data(), ref_ri.data(), ref_v.data(), &ref_result);

  /* ---- int8 v3 selector differential over the same score ---- */
  std::vector<int8_t> v3_tokens((size_t)max_output * TS_GROUP_ELEMENTS);
  std::vector<uint16_t> v3_gi((size_t)max_output, 0);
  std::vector<uint8_t> v3_st((size_t)max_output, 0);
  std::vector<uint8_t> v3_ri((size_t)max_output, 0);
  std::vector<uint8_t> v3_v((size_t)max_output, 0);
  struct TokenSelectorResult v3_result;
  float v3_scale = 0.0f;
  int v3_status = token_selector_core(
      scene.tokens_raw.data(), ref_score.data(), scene.group_count, scene.grid_h,
      scene.grid_w, budget, local_cap, scale, &v3_scale, v3_tokens.data(),
      v3_gi.data(), v3_st.data(), v3_ri.data(), v3_v.data(), &v3_result);

  /* ---- checks ---- */
  char detail[256];

  snprintf(detail, sizeof(detail), "%s: debug status", case_name.c_str());
  check(dbg_status == exp.status, detail);
  snprintf(detail, sizeof(detail), "%s: top status", case_name.c_str());
  check(top_status == exp.status, detail);
  snprintf(detail, sizeof(detail), "%s: scorer status", case_name.c_str());
  check(ref_score_status == IS_STATUS_OK, detail);

  /* Intermediate score_q byte-exact vs the frozen Python scorer golden. */
  bool score_ok = true;
  for (int i = 0; i < scene.group_count; i++) {
    score_ok = score_ok && dbg_score[(size_t)i] == exp.score_q[(size_t)i];
  }
  snprintf(detail, sizeof(detail), "%s: on-chip score_q vs golden", case_name.c_str());
  check(score_ok, detail);

  /* The standalone scorer top must reproduce the same score bytes. */
  bool ref_score_ok = true;
  for (int i = 0; i < scene.group_count; i++) {
    ref_score_ok = ref_score_ok && ref_score[(size_t)i] == exp.score_q[(size_t)i];
  }
  snprintf(detail, sizeof(detail), "%s: standalone scorer score_q vs golden",
           case_name.c_str());
  check(ref_score_ok, detail);

  /* The debug twin and the production top must agree exactly. */
  bool debug_vs_top = (dbg_status == top_status) && (dbg_oc == top_oc) &&
                      (dbg_rc == top_rc) && (dbg_cc == top_cc) &&
                      (memcmp(&dbg_scale, &top_scale, sizeof(float)) == 0);
  if (debug_vs_top && exp.status == TS_STATUS_OK) {
    size_t token_bytes = (size_t)exp.output_count * 4096;
    std::vector<int8_t> a(token_bytes), b(token_bytes);
    ts_unpack_words_to_int8(dbg_tokens.data(), exp.output_count * TS_GROUP_WORDS,
                            a.data(), (int)token_bytes);
    ts_unpack_words_to_int8(top_tokens.data(), exp.output_count * TS_GROUP_WORDS,
                            b.data(), (int)token_bytes);
    debug_vs_top = memcmp(a.data(), b.data(), token_bytes) == 0 &&
                   memcmp(dbg_gi.data(), top_gi.data(),
                          (size_t)exp.output_count * 2) == 0 &&
                   memcmp(dbg_st.data(), top_st.data(), (size_t)exp.output_count) == 0 &&
                   memcmp(dbg_ri.data(), top_ri.data(), (size_t)exp.output_count) == 0 &&
                   memcmp(dbg_v.data(), top_v.data(), (size_t)exp.output_count) == 0;
  }
  snprintf(detail, sizeof(detail), "%s: debug twin == top", case_name.c_str());
  check(debug_vs_top, detail);

  if (exp.status == TS_STATUS_OK) {
    size_t token_bytes = (size_t)exp.output_count * 4096;
    bool counts_ok = dbg_oc == exp.output_count && dbg_rc == exp.real_count &&
                     dbg_cc == exp.context_count &&
                     ref_result.output_count == exp.output_count &&
                     ref_result.real_count == exp.real_count &&
                     ref_result.context_count == exp.context_count;
    snprintf(detail, sizeof(detail), "%s: counts", case_name.c_str());
    check(counts_ok, detail);
    snprintf(detail, sizeof(detail), "%s: scale_out", case_name.c_str());
    check(memcmp(&dbg_scale, &exp.scale_out, sizeof(float)) == 0, detail);

    std::vector<int8_t> pipe_tokens(token_bytes);
    ts_unpack_words_to_int8(dbg_tokens.data(), exp.output_count * TS_GROUP_WORDS,
                            pipe_tokens.data(), (int)token_bytes);
    snprintf(detail, sizeof(detail), "%s: tokens_out vs golden", case_name.c_str());
    check(memcmp(pipe_tokens.data(), exp.tokens_out.data(), token_bytes) == 0, detail);
    snprintf(detail, sizeof(detail), "%s: group_index vs golden", case_name.c_str());
    check(memcmp(dbg_gi.data(), exp.group_index.data(),
                 (size_t)exp.output_count * 2) == 0, detail);
    snprintf(detail, sizeof(detail), "%s: source_type vs golden", case_name.c_str());
    check(memcmp(dbg_st.data(), exp.source_type.data(), (size_t)exp.output_count) == 0,
          detail);
    snprintf(detail, sizeof(detail), "%s: region_id vs golden", case_name.c_str());
    check(memcmp(dbg_ri.data(), exp.region_id.data(), (size_t)exp.output_count) == 0,
          detail);
    snprintf(detail, sizeof(detail), "%s: valid vs golden", case_name.c_str());
    check(memcmp(dbg_v.data(), exp.valid.data(), (size_t)exp.output_count) == 0, detail);

    /* Pipeline == standalone scorer + packed selector, byte for byte. */
    bool compose_ok = (ref_status == exp.status) &&
                      (ref_result.output_count == dbg_oc) &&
                      (ref_result.real_count == dbg_rc) &&
                      (ref_result.context_count == dbg_cc) &&
                      (memcmp(&ref_scale, &dbg_scale, sizeof(float)) == 0);
    if (compose_ok) {
      std::vector<int8_t> ref_bytes(token_bytes), pipe_bytes(token_bytes);
      ts_unpack_words_to_int8(ref_tokens.data(), exp.output_count * TS_GROUP_WORDS,
                              ref_bytes.data(), (int)token_bytes);
      ts_unpack_words_to_int8(dbg_tokens.data(), exp.output_count * TS_GROUP_WORDS,
                              pipe_bytes.data(), (int)token_bytes);
      compose_ok = memcmp(ref_bytes.data(), pipe_bytes.data(), token_bytes) == 0 &&
                   memcmp(ref_gi.data(), dbg_gi.data(),
                          (size_t)exp.output_count * 2) == 0 &&
                   memcmp(ref_st.data(), dbg_st.data(), (size_t)exp.output_count) == 0 &&
                   memcmp(ref_ri.data(), dbg_ri.data(), (size_t)exp.output_count) == 0 &&
                   memcmp(ref_v.data(), dbg_v.data(), (size_t)exp.output_count) == 0;
    }
    snprintf(detail, sizeof(detail), "%s: pipeline == scorer+selector composition",
             case_name.c_str());
    check(compose_ok, detail);

    /* Packed v4 == int8 v3 over the same score. */
    bool v3_ok = (v3_status == ref_status) &&
                 (v3_result.output_count == ref_result.output_count) &&
                 (v3_result.real_count == ref_result.real_count) &&
                 (v3_result.context_count == ref_result.context_count) &&
                 (memcmp(&v3_scale, &ref_scale, sizeof(float)) == 0);
    if (v3_ok) {
      std::vector<int8_t> ref_bytes(token_bytes), v3_bytes(token_bytes);
      ts_unpack_words_to_int8(ref_tokens.data(), exp.output_count * TS_GROUP_WORDS,
                              ref_bytes.data(), (int)token_bytes);
      memcpy(v3_bytes.data(), v3_tokens.data(), token_bytes);
      v3_ok = memcmp(ref_bytes.data(), v3_bytes.data(), token_bytes) == 0 &&
              memcmp(ref_gi.data(), v3_gi.data(), (size_t)exp.output_count * 2) == 0 &&
              memcmp(ref_st.data(), v3_st.data(), (size_t)exp.output_count) == 0 &&
              memcmp(ref_ri.data(), v3_ri.data(), (size_t)exp.output_count) == 0 &&
              memcmp(ref_v.data(), v3_v.data(), (size_t)exp.output_count) == 0;
    }
    snprintf(detail, sizeof(detail), "%s: packed v4 == int8 v3", case_name.c_str());
    check(v3_ok, detail);
  }

  /* Determinism: a second run of the production top is byte-identical. */
  std::vector<ts_word_t> top2_tokens(out_words);
  std::vector<uint16_t> top2_gi((size_t)max_output, 0);
  std::vector<uint8_t> top2_st((size_t)max_output, 0);
  std::vector<uint8_t> top2_ri((size_t)max_output, 0);
  std::vector<uint8_t> top2_v((size_t)max_output, 0);
  float top2_scale = 0.0f;
  int top2_oc = -1, top2_rc = -1, top2_cc = -1;
  int top2_status = integer_selector_pipeline(
      tokens_words.data(), gm_words.data(), wv_words.data(), query_words.data(),
      top2_tokens.data(), top2_gi.data(), top2_st.data(), top2_ri.data(), top2_v.data(),
      scene.group_count, scene.grid_h, scene.grid_w, budget, local_cap, scale,
      &top2_scale, &top2_oc, &top2_rc, &top2_cc);
  bool deterministic =
      top2_status == top_status && top2_oc == top_oc && top2_rc == top_rc &&
      top2_cc == top_cc && memcmp(&top2_scale, &top_scale, sizeof(float)) == 0;
  if (deterministic && exp.status == TS_STATUS_OK) {
    deterministic = memcmp(top2_tokens.data(), top_tokens.data(), out_words * sizeof(ts_word_t)) == 0 &&
                    memcmp(top2_gi.data(), top_gi.data(), (size_t)exp.output_count * 2) == 0 &&
                    memcmp(top2_st.data(), top_st.data(), (size_t)exp.output_count) == 0 &&
                    memcmp(top2_ri.data(), top_ri.data(), (size_t)exp.output_count) == 0 &&
                    memcmp(top2_v.data(), top_v.data(), (size_t)exp.output_count) == 0;
  }
  snprintf(detail, sizeof(detail), "%s: deterministic repeat", case_name.c_str());
  check(deterministic, detail);

  g_cases++;
  printf("  %-34s G=%-4d grid=%dx%d K=%-3d status=%d out=%d ctx=%d\n",
         case_name.c_str(), scene.group_count, scene.grid_h, scene.grid_w, budget,
         exp.status, exp.output_count, exp.context_count);
  return true;
}

/* ---------------------------------------------------------------- */
/* Guard / interface tests                                           */
/* ---------------------------------------------------------------- */
void run_guards(const std::vector<is_word_t>& wv_words) {
  const int g = 16;
  std::vector<ts_word_t> tokens((size_t)g * TS_GROUP_WORDS, (ts_word_t)0);
  std::vector<is_word_t> gm((size_t)g * IS_GM_WORDS_PER_GROUP, (is_word_t)0);
  std::vector<is_word_t> query(IS_QUERY_WORDS, (is_word_t)0);
  std::vector<ts_word_t> out_tokens((size_t)96 * TS_GROUP_WORDS);
  std::vector<uint16_t> gi(96, 0);
  std::vector<uint8_t> st(96, 0), ri(96, 0), v(96, 0);
  float scale_out = 0.0f;
  int oc = 0, rc = 0, cc = 0;

  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), g, 4, 4,
            48, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_OK,
        "guard: valid baseline OK");

  check(integer_selector_pipeline(
            0, gm.data(), wv_words.data(), query.data(), out_tokens.data(),
            gi.data(), st.data(), ri.data(), v.data(), g, 4, 4, 48, 16, 1.0f,
            &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_NULL,
        "guard: null tokens");
  check(integer_selector_pipeline(
            tokens.data(), 0, wv_words.data(), query.data(), out_tokens.data(),
            gi.data(), st.data(), ri.data(), v.data(), g, 4, 4, 48, 16, 1.0f,
            &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_NULL,
        "guard: null group_mean");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), 0, query.data(), out_tokens.data(),
            gi.data(), st.data(), ri.data(), v.data(), g, 4, 4, 48, 16, 1.0f,
            &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_NULL,
        "guard: null wv");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), 0, out_tokens.data(),
            gi.data(), st.data(), ri.data(), v.data(), g, 4, 4, 48, 16, 1.0f,
            &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_NULL,
        "guard: null query");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(), 0,
            gi.data(), st.data(), ri.data(), v.data(), g, 4, 4, 48, 16, 1.0f,
            &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_NULL,
        "guard: null out_tokens");

  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), 0, 4, 4,
            48, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_GROUP_COUNT,
        "guard: G=0");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), 4097, 4, 4,
            48, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_GROUP_COUNT,
        "guard: G=4097");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), -1, 4, 4,
            48, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_GROUP_COUNT,
        "guard: G=-1");

  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), g, 5, 5,
            48, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_GRID,
        "guard: grid product mismatch");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), g, 0, 4,
            48, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_GRID,
        "guard: grid dim 0");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), g, 4, 4,
            47, 16, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_BUDGET,
        "guard: K=47");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), g, 4, 4,
            48, 0, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_LOCAL_CAP,
        "guard: local_cap=0");
  check(integer_selector_pipeline(
            tokens.data(), gm.data(), wv_words.data(), query.data(),
            out_tokens.data(), gi.data(), st.data(), ri.data(), v.data(), g, 4, 4,
            48, 17, 1.0f, &scale_out, &oc, &rc, &cc) == ISP_STATUS_ERR_LOCAL_CAP,
        "guard: local_cap=17");

  printf("  guards                     G/grid/budget/local_cap/null rejected\n");
}

} /* namespace */

int main(int argc, char** argv) {
  const char* dir = (argc >= 2) ? argv[1] : getenv("ISP_FIXTURE_DIR");
  if (dir == 0) {
    printf("usage: %s <fixture_dir>   (or set ISP_FIXTURE_DIR)\n", argv[0]);
    return 2;
  }
  std::string fixture_dir = dir;

  /* Wv is shared by every case. */
  std::vector<uint8_t> wv_bytes;
  if (!read_file(fixture_dir + "/wv_q.bin", &wv_bytes) ||
      wv_bytes.size() != (size_t)IS_WV_ROWS * IS_EMBED_DIM) {
    printf("cannot read wv_q.bin\n");
    return 2;
  }
  std::vector<is_word_t> wv_words(IS_WV_WORDS);
  is_pack_int8_to_words((const int8_t*)wv_bytes.data(), (int)wv_bytes.size(),
                        wv_words.data(), IS_WV_WORDS);

  /* Compile-time contract: the on-chip score buffer is large enough. */
  check(IS_MAX_GROUPS >= TS_MAX_GROUPS, "score buffer covers G");

  std::vector<uint8_t> manifest;
  if (!read_file(fixture_dir + "/manifest.txt", &manifest)) {
    printf("cannot read manifest.txt\n");
    return 2;
  }
  std::string text((const char*)manifest.data(), manifest.size());
  size_t pos = 0;
  printf("ISP_TESTBEGIN %s\n", fixture_dir.c_str());
  while (pos < text.size()) {
    size_t end = text.find('\n', pos);
    if (end == std::string::npos) {
      end = text.size();
    }
    std::string line = text.substr(pos, end - pos);
    pos = end + 1;
    if (line.empty()) {
      continue;
    }
    char name[256], scene_file[256];
    int budget = 0, local_cap = 0;
    float scale = 0.0f;
    if (sscanf(line.c_str(), "%255s %255s %d %d %f", name, scene_file, &budget,
               &local_cap, &scale) != 5) {
      continue;
    }
    run_case(fixture_dir, name, scene_file, budget, local_cap, scale, wv_words);
  }

  run_guards(wv_words);

  printf("ISP_SUMMARY cases=%d passed=%d failed=%d\n", g_cases, g_passed, g_failed);
  if (g_failed != 0) {
    printf("ISP_RESULT FAIL\n");
    return 1;
  }
  printf("ISP_RESULT PASS\n");
  return 0;
}
