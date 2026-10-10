/*
 * tb_integer_selector_pipeline_compact_cosim.cpp
 *
 * Dedicated, self-contained testbench for the bounded C/RTL co-simulation of
 * the accepted compact integer Selector pipeline
 * (``integer_selector_pipeline_compact``).
 *
 * Design constraints (all deliberate):
 *   * NO ``dlopen``: the frozen expected status / counts / scale_out and the
 *     full canonical output arena are read straight from the accepted
 *     fixture files and compared byte for byte.
 *   * Plain C strings / ``snprintf`` + ``std::vector`` buffers only, so the
 *     Vitis HLS ``xilinx-tb-process`` instrumentation can rewrite the single
 *     top-function call into the RTL co-simulation stub.
 *   * Every m_axi host buffer spans the *declared* port depth
 *     (``ISPC_INPUT_ARENA_MAX_WORDS`` / ``ISPC_WV_WORDS`` /
 *     ``ISPC_OUTPUT_ARENA_MAX_WORDS``): the co-simulation wrapper dumps the
 *     whole declared depth, so a shorter buffer is not safe.
 *   * Canaries live one past the declared depth (never dumped / never touched
 *     by the wrapper) plus a zero-tail check on the runtime arena; the input
 *     arena used region is snapshotted and compared after the call.
 *
 * Exactly five representative accepted cases are replayed:
 *   1. synth_g80_K48_cap16             (production non-identity G=80 K=48)
 *   2. real_grounding_correct_K64_cap16(real correct prompt G=260 K=64)
 *   3. synth_g400_K96_cap16            (production stress G=400 K=96)
 *   4. synth_g80_K96_cap16             (identity bypass G=80 K=96)
 *   5. zero_g128_K48_cap16             (deterministic all-zero G=128 K=48)
 *
 * Usage: tb_integer_selector_pipeline_compact_cosim <fixture_dir> <case_list>
 *        (or set ISPC_COSIM_FIXTURE_DIR / ISPC_COSIM_CASES)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "integer_selector_pipeline_compact.h"

namespace {

const uint64_t CANARY64 = 0xDEADBEEFCAFEF00DULL;
const int EXPECTED_CASE_COUNT = 5;

int g_cases = 0;
int g_checks = 0;
int g_passed = 0;
int g_failed = 0;
int g_scalar_mismatches = 0;
int g_arena_mismatches = 0;

bool read_file(const char* path, std::vector<uint8_t>* data) {
  FILE* handle = fopen(path, "rb");
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

int32_t rd_i32(const uint8_t* p) {
  int32_t v = 0;
  memcpy(&v, p, 4);
  return v;
}

float rd_f32(const uint8_t* p) {
  float v = 0.0f;
  memcpy(&v, p, 4);
  return v;
}

uint64_t rd_u64(const uint8_t* p) {
  uint64_t v = 0;
  memcpy(&v, p, 8);
  return v;
}

uint64_t fnv1a64(const uint8_t* data, size_t count) {
  uint64_t value = 0xCBF29CE484222325ULL;
  for (size_t i = 0; i < count; i++) {
    value ^= static_cast<uint64_t>(data[i]);
    value *= 0x100000001B3ULL;
  }
  return value;
}

struct Expected {
  int status;
  int output_count;
  int real_count;
  int context_count;
  float scale_out;
  int budget;
  int total_bytes;
  std::vector<uint8_t> arena;
};

bool load_expected(const char* path, Expected* exp) {
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
  if (payload != static_cast<uint64_t>(exp->total_bytes) ||
      data.size() != 52u + static_cast<size_t>(exp->total_bytes)) {
    return false;
  }
  exp->arena.assign(data.begin() + 52, data.end());
  if (fnv1a64(exp->arena.data(), exp->arena.size()) != arena_fnv) {
    return false;
  }
  return true;
}

void record(const char* case_name, const char* what, bool ok, const char* detail,
            long* case_checks, long* case_mismatch) {
  *case_checks += 1;
  g_checks += 1;
  if (ok) {
    g_passed += 1;
  } else {
    g_failed += 1;
    *case_mismatch += 1;
    if (strcmp(what, "arena") == 0) {
      g_arena_mismatches += 1;
    } else if (strcmp(what, "tail") != 0 && strcmp(what, "canary") != 0 &&
               strcmp(what, "input") != 0) {
      g_scalar_mismatches += 1;
    }
    printf("[COSIM-FAIL] %s : %s (%s)\n", case_name, what, detail);
  }
}

bool run_case(const char* fixture_dir, const char* case_name,
              const char* arena_file, int budget, int local_cap, float scale_in,
              int grid_h, int grid_w, const is_word_t* wv_words) {
  char path[4096];
  long case_checks = 0;
  long case_mismatch = 0;

  /* Input arena byte length implies the group count: bytes = G*5120 + 128. */
  snprintf(path, sizeof(path), "%s/%s", fixture_dir, arena_file);
  std::vector<uint8_t> raw;
  if (!read_file(path, &raw) || raw.size() < 128 ||
      ((raw.size() - 128) % 5120) != 0) {
    record(case_name, "arena_load", false, "input arena missing/mis-sized",
           &case_checks, &case_mismatch);
    printf("[COSIM-CASE] name=%s G=-1 grid=%dx%d K=%d checks=%ld mismatches=%ld\n",
           case_name, grid_h, grid_w, budget, case_checks, case_mismatch);
    return false;
  }
  int group_count = static_cast<int>((raw.size() - 128) / 5120);
  size_t used_words = raw.size() / TS_WORD_BYTES;

  snprintf(path, sizeof(path), "%s/%s.exp.bin", fixture_dir, case_name);
  Expected exp;
  if (!load_expected(path, &exp)) {
    record(case_name, "expected_load", false, "expected file missing/malformed",
           &case_checks, &case_mismatch);
    printf("[COSIM-CASE] name=%s G=%d grid=%dx%d K=%d checks=%ld mismatches=%ld\n",
           case_name, group_count, grid_h, grid_w, budget, case_checks,
           case_mismatch);
    return false;
  }
  record(case_name, "expected_budget", exp.budget == budget,
         "expected budget does not match case list", &case_checks, &case_mismatch);
  record(case_name, "expected_group_count", group_count > 0,
         "group count not positive", &case_checks, &case_mismatch);

  struct IspcLayout layout;
  int layout_status = ispc_fpga_layout(budget, &layout);
  record(case_name, "layout",
         layout_status == ISPC_STATUS_OK &&
             layout.total_bytes == exp.total_bytes,
         "ISPcLayout total does not match the frozen expected arena",
         &case_checks, &case_mismatch);
  int total_words = layout.total_bytes / TS_WORD_BYTES;

  /* ---- input arena: declared port depth + canary word(s) past it ---- */
  std::vector<ts_word_t> in_words(
      static_cast<size_t>(ISPC_INPUT_ARENA_MAX_WORDS) + 2, (ts_word_t)0);
  ts_pack_int8_to_words(reinterpret_cast<const int8_t*>(raw.data()),
                        static_cast<int>(raw.size()), in_words.data(),
                        static_cast<int>(used_words));
  in_words[ISPC_INPUT_ARENA_MAX_WORDS] = (ts_word_t)CANARY64;
  in_words[ISPC_INPUT_ARENA_MAX_WORDS + 1] = (ts_word_t)CANARY64;
  std::vector<ts_word_t> in_snapshot(in_words.begin(),
                                     in_words.begin() + used_words);

  /* ---- output arena: declared port depth + canary word(s) past it ---- */
  std::vector<ts_word_t> out_words(
      static_cast<size_t>(ISPC_OUTPUT_ARENA_MAX_WORDS) + 2, (ts_word_t)0);
  out_words[ISPC_OUTPUT_ARENA_MAX_WORDS] = (ts_word_t)CANARY64;
  out_words[ISPC_OUTPUT_ARENA_MAX_WORDS + 1] = (ts_word_t)CANARY64;

  float scale_out = -12345.0f;
  int output_count = -1;
  int real_count = -1;
  int context_count = -1;
  int status = integer_selector_pipeline_compact(
      in_words.data(), wv_words, out_words.data(), group_count, grid_h, grid_w,
      budget, local_cap, scale_in, &scale_out, &output_count, &real_count,
      &context_count);

  /* ---- scalar checks ---- */
  record(case_name, "status", status == exp.status, "return status mismatch",
         &case_checks, &case_mismatch);
  record(case_name, "counts",
         output_count == exp.output_count && real_count == exp.real_count &&
             context_count == exp.context_count,
         "output/real/context count mismatch", &case_checks, &case_mismatch);
  record(case_name, "scale", memcmp(&scale_out, &exp.scale_out, sizeof(float)) == 0,
         "scale_out mismatch", &case_checks, &case_mismatch);

  /* ---- full fixed-arena byte comparison (tokens, metadata, gaps, tail) ---- */
  bool arena_ok = memcmp(reinterpret_cast<const uint8_t*>(out_words.data()),
                         exp.arena.data(), exp.arena.size()) == 0;
  record(case_name, "arena", arena_ok,
         "fixed output arena byte mismatch vs frozen expected", &case_checks,
         &case_mismatch);

  /* ---- zero tail: nothing written between runtime arena and port depth ---- */
  bool tail_ok = true;
  for (int w = total_words; w < ISPC_OUTPUT_ARENA_MAX_WORDS && tail_ok; w++) {
    if (out_words[static_cast<size_t>(w)] != (ts_word_t)0) {
      tail_ok = false;
    }
  }
  record(case_name, "tail", tail_ok,
         "non-zero write past the runtime budget arena", &case_checks,
         &case_mismatch);

  /* ---- canaries one past the declared depth ---- */
  bool canary_ok = out_words[ISPC_OUTPUT_ARENA_MAX_WORDS] == (ts_word_t)CANARY64 &&
                   out_words[ISPC_OUTPUT_ARENA_MAX_WORDS + 1] ==
                       (ts_word_t)CANARY64 &&
                   in_words[ISPC_INPUT_ARENA_MAX_WORDS] == (ts_word_t)CANARY64 &&
                   in_words[ISPC_INPUT_ARENA_MAX_WORDS + 1] == (ts_word_t)CANARY64;
  record(case_name, "canary", canary_ok,
         "canary past the declared m_axi port depth was modified",
         &case_checks, &case_mismatch);

  /* ---- input arena is read-only: used region unchanged ---- */
  bool input_ok =
      memcmp(in_words.data(), in_snapshot.data(),
             used_words * sizeof(ts_word_t)) == 0;
  record(case_name, "input", input_ok,
         "input arena changed during the call", &case_checks, &case_mismatch);

  g_cases += 1;
  printf(
      "[COSIM-CASE] name=%s G=%d grid=%dx%d K=%d cap=%d scale_in=%.6f "
      "status=%d output_count=%d real_count=%d context_count=%d scale_out=%.6f "
      "arena_bytes=%d checks=%ld mismatches=%ld\n",
      case_name, group_count, grid_h, grid_w, budget, local_cap, scale_in,
      status, output_count, real_count, context_count, scale_out,
      exp.total_bytes, case_checks, case_mismatch);
  return case_mismatch == 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* fixture_dir = (argc >= 2) ? argv[1] : getenv("ISPC_COSIM_FIXTURE_DIR");
  const char* case_list = (argc >= 3) ? argv[2] : getenv("ISPC_COSIM_CASES");

  /* The Vitis co-simulation launcher may deliver ``-argv "dir list"`` as a
   * single token; split on the first space when argv[2] is absent. */
  char split_buf[2048];
  if (fixture_dir != NULL && case_list == NULL) {
    const char* space = strchr(fixture_dir, ' ');
    if (space != NULL) {
      size_t n = static_cast<size_t>(space - fixture_dir);
      if (n < sizeof(split_buf)) {
        memcpy(split_buf, fixture_dir, n);
        split_buf[n] = '\0';
        fixture_dir = split_buf;
        case_list = space + 1;
      }
    }
  }
  if (fixture_dir == NULL || case_list == NULL) {
    fprintf(stderr,
            "usage: %s <fixture_dir> <case_list> "
            "(or set ISPC_COSIM_FIXTURE_DIR / ISPC_COSIM_CASES)\n",
            argv[0]);
    return 2;
  }

  printf("[COSIM-TB] fixture_dir=%s\n", fixture_dir);
  printf("[COSIM-TB] case_list=%s\n", case_list);

  /* ---- projection weights: declared depth + canary word(s) past it ---- */
  char path[4096];
  snprintf(path, sizeof(path), "%s/wv_q.bin", fixture_dir);
  std::vector<uint8_t> wv_bytes;
  if (!read_file(path, &wv_bytes) ||
      wv_bytes.size() !=
          static_cast<size_t>(IS_WV_ROWS) * static_cast<size_t>(IS_EMBED_DIM)) {
    fprintf(stderr, "cannot read wv_q.bin from %s\n", fixture_dir);
    return 2;
  }
  std::vector<is_word_t> wv_words(static_cast<size_t>(ISPC_WV_WORDS) + 2,
                                  (is_word_t)0);
  is_pack_int8_to_words(reinterpret_cast<const int8_t*>(wv_bytes.data()),
                        static_cast<int>(wv_bytes.size()), wv_words.data(),
                        ISPC_WV_WORDS);
  wv_words[ISPC_WV_WORDS] = (is_word_t)CANARY64;
  wv_words[ISPC_WV_WORDS + 1] = (is_word_t)CANARY64;

  FILE* cases = fopen(case_list, "r");
  if (cases == NULL) {
    fprintf(stderr, "cannot open case list %s\n", case_list);
    return 2;
  }
  printf("ISPC_COSIM_TESTBEGIN %s\n", case_list);

  char line[512];
  while (fgets(line, sizeof(line), cases) != NULL) {
    char name[256];
    char arena_file[256];
    int budget = 0;
    int local_cap = 0;
    int grid_h = 0;
    int grid_w = 0;
    float scale_in = 0.0f;
    if (sscanf(line, "%255s %255s %d %d %f %d %d", name, arena_file, &budget,
               &local_cap, &scale_in, &grid_h, &grid_w) != 7) {
      continue;
    }
    run_case(fixture_dir, name, arena_file, budget, local_cap, scale_in, grid_h,
             grid_w, wv_words.data());
  }
  fclose(cases);

  bool wv_ok = wv_words[ISPC_WV_WORDS] == (is_word_t)CANARY64 &&
               wv_words[ISPC_WV_WORDS + 1] == (is_word_t)CANARY64;
  g_checks += 1;
  if (wv_ok) {
    g_passed += 1;
  } else {
    g_failed += 1;
    g_scalar_mismatches += 1;
    printf("[COSIM-FAIL] wv_q : canary past the declared depth was modified\n");
  }

  printf("[COSIM-SUMMARY] cases=%d expected_cases=%d checks=%d passed=%d failed=%d "
         "scalar_mismatches=%d arena_mismatches=%d\n",
         g_cases, EXPECTED_CASE_COUNT, g_checks, g_passed, g_failed,
         g_scalar_mismatches, g_arena_mismatches);
  bool pass = (g_failed == 0) && (g_cases == EXPECTED_CASE_COUNT) &&
              (g_scalar_mismatches == 0) && (g_arena_mismatches == 0);
  printf("[COSIM-RESULT] %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
