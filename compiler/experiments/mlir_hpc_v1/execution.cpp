// Research-only machine-code oracle; no frontend or candidate-issuance claim.
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
extern "C" void _mlir_ciface_research_gemm(Memref *, Memref *, Memref *);

static Memref view(float *p, int rows, int columns) {
  return {p, p, 0, {rows, columns}, {columns, 1}};
}
static std::uint32_t bits(float v) { return std::bit_cast<std::uint32_t>(v); }
static bool equal(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b);
}
static std::uint64_t checks = 0, failures = 0;
static void check(bool value, const char *message) {
  ++checks;
  if (!value && ++failures < 10)
    std::fprintf(stderr, "FAIL: %s\n", message);
}

static std::vector<float> execute(std::vector<float> &a, std::vector<float> &b,
                                 int m, int n, int k) {
  auto before_a = a, before_b = b;
  std::vector<float> output(static_cast<std::size_t>(m) * n + 2, -9876.0f);
  auto av = view(a.empty() ? nullptr : a.data(), m, k);
  auto bv = view(b.empty() ? nullptr : b.data(), k, n);
  auto cv = view(m && n ? output.data() + 1 : nullptr, m, n);
  _mlir_ciface_research_gemm(&av, &bv, &cv);
  check(output.front() == -9876.0f, "leading output sentinel");
  check(output.back() == -9876.0f, "trailing output sentinel");
  check(a.empty() || !std::memcmp(a.data(), before_a.data(), a.size() * sizeof(float)),
        "lhs immutable including NaN payload bits");
  check(b.empty() || !std::memcmp(b.data(), before_b.data(), b.size() * sizeof(float)),
        "rhs immutable including NaN payload bits");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      volatile float sum = 0.0f;
      for (int t = 0; t < k; ++t) {
        volatile float product = a[static_cast<std::size_t>(i) * k + t] *
                                 b[static_cast<std::size_t>(t) * n + j];
        sum = sum + product;
      }
      check(equal(output[1 + static_cast<std::size_t>(i) * n + j], sum),
            "increasing K / separate f32 product and sum oracle");
    }
  }
  return {output.begin() + 1, output.end() - 1};
}

int main(int argc, char **argv) {
  std::fenv_t saved;
  if (fegetenv(&saved) || fesetenv(FE_DFL_ENV)) return 2;
  if (argc == 2 && std::string_view(argv[1]) == "--asan-invalid-capacity") {
    // The leaf is deliberately guard-free. The owning adapter MUST validate
    // capacity first. This control verifies instrumentation of generated loads.
    std::vector<float> a(1, 1), b(16, 1), c(1, 0);
    auto av = view(a.data(), 1, 16), bv = view(b.data(), 16, 1),
         cv = view(c.data(), 1, 1);
    _mlir_ciface_research_gemm(&av, &bv, &cv);
    return 99; // Reaching here is not an acceptable ASan negative result.
  }
  if (argc != 1) return 2;
  std::uint32_t state = 0x129403;
  auto next = [&] { state ^= state << 13; state ^= state >> 17;
                   state ^= state << 5; return state; };
  const float special[] = {
      0, -0.0f, 1, -1, 0.25f, -0.25f,
      std::numeric_limits<float>::denorm_min(),
      -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  for (int trial = 0; trial < 700; ++trial) {
    const int m = static_cast<int>(next() % 12),
              n = static_cast<int>(next() % 23),
              k = static_cast<int>(next() % 35);
    std::vector<float> a(static_cast<std::size_t>(m) * k),
                       b(static_cast<std::size_t>(k) * n);
    for (auto &v : a)
      v = trial % 3 == 0 ? special[next() % 13]
                         : static_cast<float>(static_cast<int>(next() % 201) - 100) / 13;
    for (auto &v : b)
      v = trial % 3 == 0 ? special[next() % 13]
                         : static_cast<float>(static_cast<int>(next() % 201) - 100) / 17;
    execute(a, b, m, n, k);
  }
  // Fixed matrix operands are deliberately rectangular/noncommuting.
  std::vector<float> a{1, 2, 3, 4, 5, 6}, b{2, 1, 0, 3, 4, 5};
  auto c = execute(a, b, 2, 2, 3);
  std::vector<float> left{1, 2, 3, 4, 5, 6}, right{1, 2, 3, 4, 5, 6};
  execute(left, c, 3, 2, 2); // rhs-carried C
  execute(c, right, 2, 3, 2); // lhs-carried C
  execute(c, c, 2, 2, 2); // same immutable storage for A/B is legal
  std::vector<float> fma_a{-1.0f, 0x1.000002p0f};
  std::vector<float> fma_b{1.0f, 0x1.fffffep-1f};
  const auto strict = execute(fma_a, fma_b, 1, 1, 2);
  check(bits(strict[0]) == 0, "FMA discriminator must produce positive zero");
  check(bits(std::fma(fma_a[1], fma_b[1], -1.0f)) != 0,
        "FMA counteroracle actually distinguishes the source contract");
  // Full tails exercise allocated exact sizes under sanitizer, not padding.
  for (int m : {0, 1, 3, 4, 5, 7, 8, 9})
    for (int n : {0, 1, 7, 8, 9, 15, 16, 17})
      for (int k : {0, 1, 2, 3, 17}) {
        std::vector<float> x(static_cast<std::size_t>(m) * k, 1.5f),
                           y(static_cast<std::size_t>(k) * n, -0.25f);
        execute(x, y, m, n, k);
      }
  check(fesetenv(&saved) == 0, "oracle restores caller FP environment");
  std::printf("research tiled/vector execution: %llu checks; %llu failures\n",
              static_cast<unsigned long long>(checks),
              static_cast<unsigned long long>(failures));
  return failures != 0;
}
