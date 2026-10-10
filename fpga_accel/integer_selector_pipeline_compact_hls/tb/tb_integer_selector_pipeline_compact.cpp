/*
 * tb_integer_selector_pipeline_compact.cpp - host / HLS C-simulation testbench.
 *
 * For every accepted fixture case it proves, byte for byte, that the compact
 * top reproduces:
 *
 *   1. the frozen canonical output arena (Python selector golden placed at the
 *      frozen fpga_transfer_layout offsets; every gap/unused byte zero) and the
 *      accepted status / counts / scale;
 *   2. the outputs of the accepted ``integer_selector_pipeline`` itself
 *      (loaded from a shared object built from the accepted, unmodified source)
 *      placed into the same canonical arena;
 *   3. the exact sequential composition of the two accepted exported kernels
 *      ``integer_selector_scorer`` + ``token_selector_core_packed``;
 *   4. determinism: a second compact run is byte-identical;
 *   5. the rejection paths (null pointers, illegal G / grid / budget /
 *      local_cap) plus canary checks that nothing is written outside the
 *      runtime-budget arena.
 *
 * Usage: tb_integer_selector_pipeline_compact <fixture_dir> [reference.so]
 *        (or set ISPC_FIXTURE_DIR / ISPC_REF_SO).
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "integer_selector_pipeline_compact.h"

namespace {

int g_passed = 0;
int g_failed = 0;
int g_cases = 0;
bool g_ref_so_ok = false;
long g_ref_checks = 0;

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
  if (handle == NULL) return false;
  if (fseek(handle, 0, SEEK_END) != 0) { fclose(handle); return false; }
  long size = ftell(handle);
  if (size < 0) { fclose(handle); return false; }
  if (fseek(handle, 0, SEEK_SET) != 0) { fclose(handle); return false; }
  data->resize((size_t)size);
  if (size > 0) {
    size_t got = fread(data->data(), 1, (size_t)size, handle);
    if (got != (size_t)size) { fclose(handle); return false; }
  }
  fclose(handle);
  return true;
}

uint64_t rd_u64(const uint8_t* p) { uint64_t v = 0; memcpy(&v, p, 8); return v; }
int32_t rd_i32(const uint8_t* p) { int32_t v = 0; memcpy(&v, p, 4); return v; }
float rd_f32(const uint8_t* p) { float v = 0.0f; memcpy(&v, p, 4); return v; }

uint64_t fnv1a64(const uint8_t* data, size_t count) {
  uint64_t value = 0xCBF29CE484222325ULL;
  for (size_t i = 0; i < count; i++) {
    value ^= (uint64_t)data[i];
    value *= 0x100000001B3ULL;
  }
  return value;
}

/* Accepted-pipeline function pointer (loaded from the reference .so). */
typedef int (*ref_top_fn)(
    const ts_word_t*, const is_word_t*, const is_word_t*, const is_word_t*,
    ts_word_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, int, int, int, int,
    int, float, float*, int*, int*, int*);

const uint64_t CANARY = 0xDEADBEEFCAFEF00DULL;

/* ---------------------------------------------------------------- */
/* Input arena                                                       */
/* ---------------------------------------------------------------- */
bool load_arena_words(const std::string& path, int group_count,
                      std::vector<ts_word_t>* words, size_t* byte_count) {
  std::vector<uint8_t> data;
  if (!read_file(path, &data)) return false;
  size_t expected = (size_t)group_count * 5120 + 128;
  if (data.size() != expected || (data.size() % TS_WORD_BYTES) != 0) return false;
  words->resize(data.size() / TS_WORD_BYTES);
  ts_pack_int8_to_words((const int8_t*)data.data(), (int)data.size(),
                        words->data(), (int)words->size());
  *byte_count = data.size();
  return true;
}

/* ---------------------------------------------------------------- */
/* Expected                                                          */
/* ---------------------------------------------------------------- */
struct Expected {
  int status = 0;
  int output_count = 0;
  int real_count = 0;
  int context_count = 0;
  float scale_out = 0.0f;
  int budget = 0;
  int total_bytes = 0;
  std::vector<uint8_t> arena;
};

bool load_expected(const std::string& path, Expected* exp) {
  std::vector<uint8_t> data;
  if (!read_file(path, &data) || data.size() < 52 ||
      memcmp(data.data(), "ISPCMP01", 8) != 0) {
    return false;
  }
  exp->status = rd_i32(data.data() + 8);
  exp->output_count = rd_i32(data.data() + 12);
  exp->real_count = rd_i32(data.data() + 16);
  exp->context_count = rd_i32(data.data() + 20);
  exp->scale_out = rd_f32(data.data() + 24);
  exp->budget = rd_i32(data.data() + 28);
  exp->total_bytes = rd_i32(data.data() + 32);
  uint64_t arena_fnv = rd_u64(data.data() + 36);
  uint64_t payload = rd_u64(data.data() + 44);
  if (payload != (uint64_t)exp->total_bytes ||
      data.size() != 52u + (size_t)exp->total_bytes) {
    return false;
  }
  exp->arena.assign(data.begin() + 52, data.end());
  if (fnv1a64(exp->arena.data(), exp->arena.size()) != arena_fnv) return false;
  return true;
}

/* Build a canonical fixed-budget arena from raw accepted outputs. */
void build_canonical(const struct IspcLayout& layout, const uint8_t* tokens,
                     const uint16_t* group_index, const uint8_t* source_type,
                     const uint8_t* region_id, const uint8_t* valid,
                     int output_count, std::vector<uint8_t>* arena) {
  arena->assign((size_t)layout.total_bytes, 0);
  if (output_count <= 0) return;
  memcpy(arena->data(), tokens, (size_t)output_count * 4096);
  memcpy(arena->data() + layout.offsets[1], group_index,
         (size_t)output_count * 2);
  memcpy(arena->data() + layout.offsets[2], source_type, (size_t)output_count);
  memcpy(arena->data() + layout.offsets[3], region_id, (size_t)output_count);
  memcpy(arena->data() + layout.offsets[4], valid, (size_t)output_count);
}

/* ---------------------------------------------------------------- */
/* One case                                                          */
/* ---------------------------------------------------------------- */
bool run_case(const std::string& fixture_dir, const std::string& case_name,
              const std::string& arena_file, int budget, int local_cap,
              float scale, int grid_h, int grid_w,
              const std::vector<is_word_t>& wv_words, ref_top_fn ref_top) {
  std::vector<ts_word_t> arena_words;
  size_t arena_bytes = 0;
  int group_count = 0;

  /* group count is implied by the arena byte count: bytes = G*5120 + 128. */
  {
    std::vector<uint8_t> raw;
    if (!read_file(fixture_dir + "/" + arena_file, &raw) ||
        raw.size() < 128 || ((raw.size() - 128) % 5120) != 0) {
      check(false, case_name + ": arena load/size");
      return false;
    }
    group_count = (int)((raw.size() - 128) / 5120);
  }
  if (!load_arena_words(fixture_dir + "/" + arena_file, group_count,
                        &arena_words, &arena_bytes)) {
    check(false, case_name + ": arena pack");
    return false;
  }

  Expected exp;
  if (!load_expected(fixture_dir + "/" + case_name + ".exp.bin", &exp)) {
    check(false, case_name + ": expected load");
    return false;
  }
  if (exp.budget != budget) {
    check(false, case_name + ": expected budget");
    return false;
  }

  const ts_word_t* tokens_q = arena_words.data();
  const is_word_t* group_mean_q = (const is_word_t*)(
      arena_words.data() + (size_t)group_count * ISPC_TOKENS_WORDS_PER_GROUP);
  const is_word_t* query_q = (const is_word_t*)(
      arena_words.data() +
      (size_t)group_count *
          (ISPC_TOKENS_WORDS_PER_GROUP + ISPC_GM_WORDS_PER_GROUP));

  struct IspcLayout layout;
  int layout_status = ispc_fpga_layout(budget, &layout);
  check(layout_status == ISPC_STATUS_OK && layout.total_bytes == exp.total_bytes,
        case_name + ": layout");

  int total_words = layout.total_bytes / TS_WORD_BYTES;

  /* ---- compact top with canaries ---- */
  std::vector<ts_word_t> out(total_words + 2, (ts_word_t)CANARY);
  float scale_out = -123.0f;
  int oc = -1, rc = -1, cc = -1;
  int status = integer_selector_pipeline_compact(
      tokens_q, wv_words.data(), out.data() + 1, group_count, grid_h, grid_w,
      budget, local_cap, scale, &scale_out, &oc, &rc, &cc);

  /* canaries */
  check(out[0] == (ts_word_t)CANARY && out[total_words + 1] == (ts_word_t)CANARY,
        case_name + ": canaries intact");

  /* ---- expected arena byte-exact ---- */
  const uint8_t* got = (const uint8_t*)(out.data() + 1);
  check(memcmp(got, exp.arena.data(), exp.arena.size()) == 0,
        case_name + ": arena byte-exact vs frozen golden");
  check(status == exp.status, case_name + ": status");
  check(oc == exp.output_count && rc == exp.real_count && cc == exp.context_count,
        case_name + ": counts");
  check(memcmp(&scale_out, &exp.scale_out, sizeof(float)) == 0,
        case_name + ": scale_out");

  /* ---- accepted integer_selector_pipeline via the reference .so ---- */
  if (ref_top != 0 && exp.status == TS_STATUS_OK) {
    int max_output = (group_count > budget) ? group_count : budget;
    std::vector<ts_word_t> r_tokens((size_t)max_output * TS_GROUP_WORDS);
    std::vector<uint16_t> r_gi((size_t)max_output, 0);
    std::vector<uint8_t> r_st((size_t)max_output, 0);
    std::vector<uint8_t> r_ri((size_t)max_output, 0);
    std::vector<uint8_t> r_v((size_t)max_output, 0);
    float r_scale = 0.0f;
    int r_oc = -1, r_rc = -1, r_cc = -1;
    int r_status = ref_top(tokens_q, group_mean_q, wv_words.data(), query_q,
                           r_tokens.data(), r_gi.data(), r_st.data(),
                           r_ri.data(), r_v.data(), group_count, grid_h, grid_w, budget,
                           local_cap, scale, &r_scale, &r_oc, &r_rc, &r_cc);
    std::vector<uint8_t> ref_arena;
    build_canonical(layout, (const uint8_t*)r_tokens.data(), r_gi.data(),
                    r_st.data(), r_ri.data(), r_v.data(), r_oc, &ref_arena);
    bool accepted_ok = (r_status == exp.status) && (r_oc == exp.output_count) &&
                       (r_rc == exp.real_count) && (r_cc == exp.context_count) &&
                       (memcmp(&r_scale, &exp.scale_out, sizeof(float)) == 0) &&
                       (memcmp(ref_arena.data(), exp.arena.data(),
                               exp.arena.size()) == 0) &&
                       (memcmp(got, ref_arena.data(), ref_arena.size()) == 0);
    check(accepted_ok, case_name + ": compact == accepted integer_selector_pipeline");
    g_ref_checks += 1;
  }

  /* ---- in-process accepted datapath composition ---- */
  if (exp.status == TS_STATUS_OK) {
    std::vector<uint16_t> score_ref((size_t)group_count, 0);
    int score_status = integer_selector_scorer(
        group_mean_q, wv_words.data(), query_q, score_ref.data(), group_count);
    check(score_status == IS_STATUS_OK, case_name + ": composition scorer status");
    int max_output = (group_count > budget) ? group_count : budget;
    std::vector<ts_word_t> c_tokens((size_t)max_output * TS_GROUP_WORDS);
    std::vector<uint16_t> c_gi((size_t)max_output, 0);
    std::vector<uint8_t> c_st((size_t)max_output, 0);
    std::vector<uint8_t> c_ri((size_t)max_output, 0);
    std::vector<uint8_t> c_v((size_t)max_output, 0);
    struct TokenSelectorResult c_res;
    float c_scale = 0.0f;
    int c_status = token_selector_core_packed(
        tokens_q, score_ref.data(), group_count, grid_h, grid_w, budget, local_cap, scale,
        &c_scale, c_tokens.data(), c_gi.data(), c_st.data(), c_ri.data(),
        c_v.data(), &c_res);
    std::vector<uint8_t> comp_arena;
    build_canonical(layout, (const uint8_t*)c_tokens.data(), c_gi.data(),
                    c_st.data(), c_ri.data(), c_v.data(), c_res.output_count,
                    &comp_arena);
    bool comp_ok = (c_status == exp.status) &&
                   (c_res.output_count == exp.output_count) &&
                   (c_res.real_count == exp.real_count) &&
                   (c_res.context_count == exp.context_count) &&
                   (memcmp(&c_scale, &exp.scale_out, sizeof(float)) == 0) &&
                   (memcmp(comp_arena.data(), exp.arena.data(),
                           exp.arena.size()) == 0);
    check(comp_ok, case_name + ": compact == scorer+packed-selector composition");
  }

  /* ---- determinism ---- */
  std::vector<ts_word_t> out2(total_words + 2, (ts_word_t)CANARY);
  float scale_out2 = 0.0f;
  int oc2 = -1, rc2 = -1, cc2 = -1;
  int status2 = integer_selector_pipeline_compact(
      tokens_q, wv_words.data(), out2.data() + 1, group_count, grid_h, grid_w, budget,
      local_cap, scale, &scale_out2, &oc2, &rc2, &cc2);
  bool deterministic =
      status2 == status && oc2 == oc && rc2 == rc && cc2 == cc &&
      memcmp(&scale_out2, &scale_out, sizeof(float)) == 0 &&
      memcmp(out2.data(), out.data(), (total_words + 2) * sizeof(ts_word_t)) == 0;
  check(deterministic, case_name + ": deterministic repeat");

  g_cases++;
  printf("  %-34s G=%-4d K=%-3d status=%d out=%d ctx=%d\n", case_name.c_str(),
         group_count, budget, exp.status, exp.output_count, exp.context_count);
  return true;
}

/* ---------------------------------------------------------------- */
/* Guards + canaries                                                 */
/* ---------------------------------------------------------------- */
void run_guards(const std::vector<is_word_t>& wv_words) {
  const int g = 16;
  struct IspcLayout layout;
  ispc_fpga_layout(48, &layout);
  int total_words = layout.total_bytes / TS_WORD_BYTES;

  std::vector<ts_word_t> arena((size_t)g * (ISPC_TOKENS_WORDS_PER_GROUP +
                                             ISPC_GM_WORDS_PER_GROUP) +
                               ISPC_QUERY_WORDS, (ts_word_t)0);
  std::vector<ts_word_t> out((size_t)total_words + 2, (ts_word_t)CANARY);
  float scale_out = 0.0f;
  int oc = 0, rc = 0, cc = 0;

  /* valid baseline populates the arena */
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, g, 4, 4, 48, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_OK,
        "guard: valid baseline OK");

  check(integer_selector_pipeline_compact(
            0, wv_words.data(), out.data() + 1, g, 4, 4, 48, 16, 1.0f,
            &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_NULL,
        "guard: null input_arena");
  check(integer_selector_pipeline_compact(
            arena.data(), 0, out.data() + 1, g, 4, 4, 48, 16, 1.0f, &scale_out,
            &oc, &rc, &cc) == ISPC_STATUS_ERR_NULL,
        "guard: null wv_q");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), 0, g, 4, 4, 48, 16, 1.0f, &scale_out,
            &oc, &rc, &cc) == ISPC_STATUS_ERR_NULL,
        "guard: null output_arena");

  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, 0, 4, 4, 48, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_GROUP_COUNT,
        "guard: G=0");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, 4097, 4, 4, 48, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_GROUP_COUNT,
        "guard: G=4097");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, -1, 4, 4, 48, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_GROUP_COUNT,
        "guard: G=-1");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, g, 5, 5, 48, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_GRID,
        "guard: grid product mismatch");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, g, 0, 4, 48, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_GRID,
        "guard: grid dim 0");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, g, 4, 4, 47, 16,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_BUDGET,
        "guard: K=47");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, g, 4, 4, 48, 0,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_LOCAL_CAP,
        "guard: local_cap=0");
  check(integer_selector_pipeline_compact(
            arena.data(), wv_words.data(), out.data() + 1, g, 4, 4, 48, 17,
            1.0f, &scale_out, &oc, &rc, &cc) == ISPC_STATUS_ERR_LOCAL_CAP,
        "guard: local_cap=17");

  /* Canary: an invalid-budget call must not touch the arena body or canaries. */
  {
    std::vector<ts_word_t> poisoned((size_t)total_words + 2, (ts_word_t)CANARY);
    for (int i = 1; i <= total_words; i++) {
      poisoned[(size_t)i] = (ts_word_t)0x0123456789ABCDEFULL;
    }
    int bad = integer_selector_pipeline_compact(
        arena.data(), wv_words.data(), poisoned.data() + 1, g, 4, 4, 47, 16,
        1.0f, &scale_out, &oc, &rc, &cc);
    bool untouched =
        bad == ISPC_STATUS_ERR_BUDGET && poisoned[0] == (ts_word_t)CANARY &&
        poisoned[total_words + 1] == (ts_word_t)CANARY;
    for (int i = 1; i <= total_words && untouched; i++) {
      untouched = poisoned[(size_t)i] == (ts_word_t)0x0123456789ABCDEFULL;
    }
    check(untouched, "guard: invalid budget does not write the arena");
  }

  /* Canary: invalid grid returns before the arena is touched?  The compact top
   * zeroes the arena before the selector grid check, so only the bytes outside
   * [0, total_bytes) are required to stay intact. */
  {
    std::vector<ts_word_t> poisoned((size_t)total_words + 2, (ts_word_t)CANARY);
    int bad = integer_selector_pipeline_compact(
        arena.data(), wv_words.data(), poisoned.data() + 1, g, 5, 5, 48, 16,
        1.0f, &scale_out, &oc, &rc, &cc);
    check(bad == ISPC_STATUS_ERR_GRID && poisoned[0] == (ts_word_t)CANARY &&
              poisoned[total_words + 1] == (ts_word_t)CANARY,
          "guard: invalid grid keeps outside-arena canaries");
  }

  printf("  guards                     G/grid/budget/local_cap/null refused\n");
}

} /* namespace */

int main(int argc, char** argv) {
  const char* dir = (argc >= 2) ? argv[1] : getenv("ISPC_FIXTURE_DIR");
  const char* ref_so = (argc >= 3) ? argv[2] : getenv("ISPC_REF_SO");
  if (dir == 0) {
    printf("usage: %s <fixture_dir> [reference.so]\n", argv[0]);
    return 2;
  }
  std::string fixture_dir = dir;

  std::vector<uint8_t> wv_bytes;
  if (!read_file(fixture_dir + "/wv_q.bin", &wv_bytes) ||
      wv_bytes.size() != (size_t)IS_WV_ROWS * IS_EMBED_DIM) {
    printf("cannot read wv_q.bin\n");
    return 2;
  }
  std::vector<is_word_t> wv_words(IS_WV_WORDS);
  is_pack_int8_to_words((const int8_t*)wv_bytes.data(), (int)wv_bytes.size(),
                        wv_words.data(), IS_WV_WORDS);

  ref_top_fn ref_top = 0;
  void* ref_handle = 0;
  if (ref_so != 0) {
    ref_handle = dlopen(ref_so, RTLD_NOW | RTLD_LOCAL);
    if (ref_handle == 0) {
      printf("warning: dlopen(%s) failed: %s\n", ref_so, dlerror());
    } else {
      void* sym = dlsym(ref_handle, "integer_selector_pipeline");
      if (sym == 0) {
        printf("warning: integer_selector_pipeline not found in %s\n", ref_so);
      } else {
        ref_top = reinterpret_cast<ref_top_fn>(sym);
        g_ref_so_ok = true;
      }
    }
  }

  std::vector<uint8_t> manifest;
  if (!read_file(fixture_dir + "/manifest.txt", &manifest)) {
    printf("cannot read manifest.txt\n");
    return 2;
  }
  std::string text((const char*)manifest.data(), manifest.size());
  size_t pos = 0;
  printf("ISPC_TESTBEGIN %s\n", fixture_dir.c_str());
  while (pos < text.size()) {
    size_t end = text.find('\n', pos);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(pos, end - pos);
    pos = end + 1;
    if (line.empty()) continue;
    char name[256], arena_file[256];
    int budget = 0, local_cap = 0, grid_h = 1, grid_w = 1;
    float scale = 0.0f;
    if (sscanf(line.c_str(), "%255s %255s %d %d %f %d %d", name, arena_file,
               &budget, &local_cap, &scale, &grid_h, &grid_w) != 7) {
      continue;
    }
    run_case(fixture_dir, name, arena_file, budget, local_cap, scale, grid_h,
             grid_w, wv_words, ref_top);
  }

  run_guards(wv_words);

  if (ref_handle != 0) dlclose(ref_handle);

  printf("ISPC_REF_SO %s\n", g_ref_so_ok ? "loaded" : "not_loaded");
  printf("ISP_SUMMARY cases=%d passed=%d failed=%d\n", g_cases, g_passed,
         g_failed);
  if (g_failed != 0 || !g_ref_so_ok) {
    printf("ISPC_RESULT FAIL\n");
    return 1;
  }
  printf("ISPC_RESULT PASS\n");
  return 0;
}
