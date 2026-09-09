// Issued primitive oracle, adapted from research caa2364 and wide-read controls.
// No source/registry/FP-adapter claim: this test establishes the leaf environment.
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

struct Memref {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
static_assert(std::is_standard_layout_v<Memref> && sizeof(Memref) == 56);
static_assert(offsetof(Memref, offset) == 16 &&
              offsetof(Memref, sizes) == 24 && offsetof(Memref, strides) == 40);
extern "C" void _mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1(Memref *, Memref *, Memref *);

static std::uint64_t cases = 0, checks = 0, failures = 0;
static bool corrupt = false;
static bool require_strict = false;
static std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}
static bool equal(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b);
}
static void check(bool value, const char *why) {
  ++checks;
  if (!value && ++failures < 12) std::fprintf(stderr, "FAIL: %s\n", why);
}
static Memref view(float *pointer, int rows, int columns) {
  return {rows && columns ? pointer : nullptr,
          rows && columns ? pointer : nullptr, 0, {rows, columns}, {columns, 1}};
}
static float strict_step(float a, float b, float accumulator) {
  volatile float product = a * b;
  volatile float result = accumulator + product;
  return result;
}
static std::vector<float> execute(float *a, float *b, int m, int n, int k) {
  ++cases;
  const auto count_a = static_cast<std::size_t>(m) * k;
  const auto count_b = static_cast<std::size_t>(k) * n;
  std::vector<float> before_a(count_a), before_b(count_b);
  if (count_a) std::memcpy(before_a.data(), a, count_a * sizeof(float));
  if (count_b) std::memcpy(before_b.data(), b, count_b * sizeof(float));
  std::vector<float> output(static_cast<std::size_t>(m) * n + 2, -9876.0f);
  auto av = view(a, m, k), bv = view(b, k, n);
  auto cv = view(output.data() + 1, m, n);
  const auto av_before = av, bv_before = bv, cv_before = cv;
  _mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1(&av, &bv, &cv);
  if (corrupt && m && n) output[1] = 12345.0f;
  check(bits(output.front()) == bits(-9876.0f), "leading output canary");
  check(bits(output.back()) == bits(-9876.0f), "trailing output canary");
  check(!count_a || !std::memcmp(a, before_a.data(), count_a * sizeof(float)),
        "lhs immutable, including NaN payloads");
  check(!count_b || !std::memcmp(b, before_b.data(), count_b * sizeof(float)),
        "rhs immutable, including NaN payloads");
  check(!std::memcmp(&av, &av_before, sizeof av) &&
            !std::memcmp(&bv, &bv_before, sizeof bv) &&
            !std::memcmp(&cv, &cv_before, sizeof cv), "descriptor identity unchanged");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float expected = 0.0f;
      double ordinary = 0.0;
      const bool fused = !require_strict && i < (m / 4) * 4 && j < (n / 8) * 8;
      for (int t = 0; t < k; ++t) {
        const float x = a[static_cast<std::size_t>(i) * k + t];
        const float y = b[static_cast<std::size_t>(t) * n + j];
        expected = fused ? std::fma(x, y, expected) : strict_step(x, y, expected);
        ordinary += static_cast<double>(x) * y;
      }
      check(equal(output[1 + static_cast<std::size_t>(i) * n + j], expected),
            "chosen legal realization: full-tile FMA chain, separate tail chain");
      (void)ordinary; // General IEEE results are not validated with a tolerance.
    }
  }
  return {output.begin() + 1, output.end() - 1};
}
static std::vector<float> execute(std::vector<float> &a, std::vector<float> &b,
                                 int m, int n, int k) {
  return execute(a.data(), b.data(), m, n, k);
}

int main(int argc, char **argv) {
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
    std::puts("SKIP: issued primitive requires hardware/OS AVX2 and FMA");
    return 77;
  }
  std::fenv_t saved;
  if (fegetenv(&saved) || fesetenv(FE_DFL_ENV)) return 2;
  if (argc == 2 && std::string_view(argv[1]) == "--asan-invalid-capacity") {
    // Preserve the original A-capacity control: its generated broadcast reads
    // one f32, so this establishes a scalar 4-byte read, not a wide B load.
    std::vector<float> a(1, 1), b(16, 1), c(32, 0);
    auto av = view(a.data(), 4, 2), bv = view(b.data(), 2, 8), cv = view(c.data(), 4, 8);
    _mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1(&av, &bv, &cv);
    return 99;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--asan-invalid-b-capacity") {
    // A and private C are truthful. The full 4x8 tile reads eight contiguous
    // f32 elements from B, whose actual allocation has only one element.
    std::vector<float> a(4, 1), b(1, 1), c(32, 0);
    auto av = view(a.data(), 4, 1), bv = view(b.data(), 1, 8), cv = view(c.data(), 4, 8);
    _mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1(&av, &bv, &cv);
    return 99;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--corrupt-output") corrupt = true;
  else if (argc == 2 && std::string_view(argv[1]) == "--require-strict") require_strict = true;
  else if (argc != 1) return 2;

  std::uint32_t state = 0x129403;
  auto next = [&] { state ^= state << 13; state ^= state >> 17;
                   state ^= state << 5; return state; };
  const float special[] = {0, -0.0f, 1, -1, 0.25f, -0.25f,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  for (int trial = 0; trial < 700; ++trial) {
    const int m = static_cast<int>(next() % 12), n = static_cast<int>(next() % 23),
              k = static_cast<int>(next() % 35);
    std::vector<float> a(static_cast<std::size_t>(m) * k),
                       b(static_cast<std::size_t>(k) * n);
    for (auto &v : a) v = trial % 3 == 0 ? special[next() % 13]
        : static_cast<float>(static_cast<int>(next() % 201) - 100) / 13;
    for (auto &v : b) v = trial % 3 == 0 ? special[next() % 13]
        : static_cast<float>(static_cast<int>(next() % 201) - 100) / 17;
    execute(a, b, m, n, k);
  }
  for (int m : {0, 1, 3, 4, 5, 7, 8, 9})
    for (int n : {0, 1, 7, 8, 9, 15, 16, 17})
      for (int k : {0, 1, 2, 3, 17, 31, 32, 33, 63, 64, 65}) {
        std::vector<float> a(static_cast<std::size_t>(m) * k, 1.5f),
                           b(static_cast<std::size_t>(k) * n, -0.25f);
        const auto result = execute(a, b, m, n, k);
        for (float cell : result)
          check(equal(cell, static_cast<float>(-0.375 * k) + 0.0f),
                "independent exactly representable double oracle");
      }
  // Rectangular dependent GEMMs with each complete f32 result boundary retained.
  std::vector<float> a(5 * 3), b(3 * 9), lhs(7 * 5), rhs(9 * 7);
  for (auto *values : {&a, &b, &lhs, &rhs})
    for (auto &v : *values) v = static_cast<float>(static_cast<int>(next() % 17) - 8) / 4;
  auto c = execute(a, b, 5, 9, 3);
  execute(lhs, c, 7, 9, 5);
  execute(c, rhs, 5, 7, 9);
  // Input/input aliasing stays legal, distinct from forbidden output/input aliasing.
  std::vector<float> overlap(128);
  for (auto &v : overlap) v = static_cast<float>(static_cast<int>(next() % 21) - 10) / 8;
  execute(overlap.data(), overlap.data(), 8, 8, 8);
  execute(overlap.data(), overlap.data() + 1, 5, 9, 7);
  execute(overlap.data() + 1, overlap.data(), 5, 9, 7);
  // A full-tile cell must show actual FMA; tails deliberately remain unfused.
  std::vector<float> fma_a(5 * 2), fma_b(2 * 9);
  for (int i = 0; i < 5; ++i) { fma_a[i * 2] = -1; fma_a[i * 2 + 1] = 0x1.000002p0f; }
  for (int j = 0; j < 9; ++j) { fma_b[j] = 1; fma_b[9 + j] = 0x1.fffffep-1f; }
  auto result = execute(fma_a, fma_b, 5, 9, 2);
  check(bits(result[0]) == bits(std::fma(fma_a[1], fma_b[9], -1.0f)) && bits(result[0]) != 0,
        "real full-tile FMA discriminator");
  check(bits(result[8]) == 0 && bits(result[4 * 9]) == 0,
        "full/tail distinct realizations are both source-authorized");
  // Nonfinite last real term is never replaced by fabricated K-padding terms.
  for (int k : {1, 3, 5, 31, 33, 65}) {
    std::vector<float> x(5 * k, 1), y(k * 9, 0);
    for (int j = 0; j < 9; ++j) y[(k - 1) * 9 + j] = std::numeric_limits<float>::infinity();
    auto actual = execute(x, y, 5, 9, k);
    for (float cell : actual) check(cell == std::numeric_limits<float>::infinity(),
                                   "no fictitious zero-times-infinity padding");
  }
  check(fesetenv(&saved) == 0, "test caller restores original FP environment");
  std::printf("issued reassociate register: %llu cases; %llu checks; %llu failures\n",
      static_cast<unsigned long long>(cases), static_cast<unsigned long long>(checks),
      static_cast<unsigned long long>(failures));
  return failures != 0;
}
