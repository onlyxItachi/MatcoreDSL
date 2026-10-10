// Independent raw-issued-leaf oracle. No source admission, adapter failure
// frontier, automatic dispatch, performance or GPU execution claim follows.
#include <algorithm>
#include <array>
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
#include "strict_isa_test_support.h"

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#endif

#ifndef MDSLC_PATTERN_TILE_M
#define MDSLC_PATTERN_TILE_M 3
#endif
#ifndef MDSLC_PATTERN_TILE_N
#define MDSLC_PATTERN_TILE_N 5
#endif

struct MemRef {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
static_assert(std::is_standard_layout_v<MemRef> && sizeof(MemRef) == 56 &&
              alignof(MemRef) == 8 && offsetof(MemRef, aligned) == 8 &&
              offsetof(MemRef, sizes) == 24 && offsetof(MemRef, strides) == 40);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_v1(MemRef *, MemRef *, MemRef *);

namespace {
constexpr int tileM = MDSLC_PATTERN_TILE_M, tileN = MDSLC_PATTERN_TILE_N;
static_assert(tileM >= 1 && tileM <= 64 && tileN >= 1 && tileN <= 64);
constexpr std::size_t guard = 16;
constexpr float canary = -0x1.937ap12F;
std::uint64_t checks = 0, failures = 0, cases = 0;
bool corruptOutput = false;
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
bool same(float actual, float expected) {
  return (std::isnan(actual) && std::isnan(expected)) || bits(actual) == bits(expected);
}
void check(bool condition, const char *label, int m, int n, int k) {
  ++checks;
  if (!condition && failures++ < 16)
    std::fprintf(stderr, "FAIL output pattern %s M=%d N=%d K=%d tile=%dx%d\n",
                 label, m, n, k, tileM, tileN);
}
MemRef descriptor(float *data, int rows, int columns) {
  return {data, data, 0, {rows, columns}, {columns, 1}};
}
std::vector<float> oracle(const float *a, const float *b, int m, int n, int k) {
  std::vector<float> result(static_cast<std::size_t>(m * n));
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      volatile float accumulator = 0.0F;
      for (int q = 0; q < k; ++q) {
        volatile float term = a[i * k + q] * b[q * n + j];
        accumulator = accumulator + term;
      }
      result[static_cast<std::size_t>(i * n + j)] = accumulator;
    }
  return result;
}
void run(int m, int n, int k, const std::vector<float> &a,
         const std::vector<float> &b, const char *label) {
  ++cases;
  auto lhs = a, rhs = b;
  const auto expected = oracle(lhs.data(), rhs.data(), m, n, k);
  const auto count = static_cast<std::size_t>(m * n);
  std::vector<float> output(count + 2 * guard, canary);
  auto ad = descriptor(lhs.empty() ? nullptr : lhs.data(), m, k);
  auto bd = descriptor(rhs.empty() ? nullptr : rhs.data(), k, n);
  auto cd = descriptor(output.data() + guard, m, n);
  const auto originalA = ad, originalB = bd, originalC = cd;
  _mlir_ciface___matcore_strict_gemm_f32_v1(&ad, &bd, &cd);
  if (corruptOutput && count) output[guard] = 0x1.abcdep20F;
  for (std::size_t i = 0; i < count; ++i)
    check(same(output[guard + i], expected[i]), label, m, n, k);
  for (std::size_t i = 0; i < lhs.size(); ++i)
    check(bits(lhs[i]) == bits(a[i]), "lhs unchanged", m, n, k);
  for (std::size_t i = 0; i < rhs.size(); ++i)
    check(bits(rhs[i]) == bits(b[i]), "rhs unchanged", m, n, k);
  for (std::size_t i = 0; i < guard; ++i) {
    check(bits(output[i]) == bits(canary), "leading output canary", m, n, k);
    check(bits(output[guard + count + i]) == bits(canary), "trailing output canary", m, n, k);
  }
  check(std::memcmp(&ad, &originalA, sizeof(ad)) == 0 &&
            std::memcmp(&bd, &originalB, sizeof(bd)) == 0 &&
            std::memcmp(&cd, &originalC, sizeof(cd)) == 0,
        "descriptor identity unchanged", m, n, k);
}
void ordinary(int m, int n, int k) {
  std::vector<float> a(static_cast<std::size_t>(m * k)),
      b(static_cast<std::size_t>(k * n));
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < k; ++q)
      a[static_cast<std::size_t>(i * k + q)] = float((i * 7 + q * 11) % 29 - 14) * 0.125F;
  for (int q = 0; q < k; ++q)
    for (int j = 0; j < n; ++j)
      b[static_cast<std::size_t>(q * n + j)] = float((q * 13 + j * 5) % 31 - 15) * 0.0625F;
  run(m, n, k, a, b, "lane-distinct rectangular/tail result");
}
void alias(int m, int n, int k, std::size_t aOffset, std::size_t bOffset) {
  ++cases;
  const auto aCount = static_cast<std::size_t>(m * k), bCount = static_cast<std::size_t>(k * n);
  const auto count = std::max(aOffset + aCount, bOffset + bCount);
  std::vector<float> shared(count + 2 * guard, canary);
  const float samples[]{0, -0.0F, 0x1.000002p0F, -0x1.fffffep-1F,
      0x1p-149F, 0x1p-126F, 1.0F, -1.0F, 0.5F,
      std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  for (std::size_t i = 0; i < count; ++i) shared[guard + i] = samples[(i * 7 + 3) % 11];
  const auto original = shared;
  auto *a = shared.data() + guard + aOffset, *b = shared.data() + guard + bOffset;
  const auto expected = oracle(a, b, m, n, k);
  std::vector<float> output(static_cast<std::size_t>(m * n) + 2 * guard, canary);
  auto ad = descriptor(a, m, k), bd = descriptor(b, k, n);
  auto cd = descriptor(output.data() + guard, m, n);
  _mlir_ciface___matcore_strict_gemm_f32_v1(&ad, &bd, &cd);
  if (corruptOutput && m && n) output[guard] = 0x1.abcdep20F;
  for (std::size_t i = 0; i < expected.size(); ++i)
    check(same(output[guard + i], expected[i]), "overlapping input result", m, n, k);
  for (std::size_t i = 0; i < shared.size(); ++i)
    check(bits(shared[i]) == bits(original[i]), "overlapping inputs/guards unchanged", m, n, k);
  for (std::size_t i = 0; i < guard; ++i) {
    check(bits(output[i]) == bits(canary), "alias output leading canary", m, n, k);
    check(bits(output[guard + expected.size() + i]) == bits(canary), "alias output trailing canary", m, n, k);
  }
}
void numerical() {
  const int m = tileM + 1, n = std::max(129, 2 * tileN + 1);
  // This discriminates full increasing K from split/re-zeroed partial sums.
  constexpr int k = 65;
  std::vector<float> terms(k);
  terms[0] = 0x1p25F; terms[31] = 1; terms[32] = -0x1p25F;
  terms[33] = 1; terms[64] = 2;
  std::vector<float> lhs(static_cast<std::size_t>(m * k));
  for (int i = 0; i < m; ++i) std::copy(terms.begin(), terms.end(), lhs.begin() + i * k);
  const std::vector<float> ones(static_cast<std::size_t>(k * n), 1.0F);
  const auto ordered = oracle(terms.data(), std::vector<float>(k, 1).data(), 1, 1, k)[0];
  volatile float split = 0.0F, reset = 0.0F;
  for (int base = 0; base < k; base += 32) {
    volatile float partial = 0.0F;
    for (int q = base; q < k && q < base + 32; ++q) partial = partial + terms[q];
    split = split + partial; reset = partial;
  }
  check(bits(ordered) == bits(3) && bits(split) == bits(2) && bits(reset) == bits(2),
        "ordered-K counteroracles are discriminating", 1, 1, k);
  run(m, n, k, lhs, ones, "ordered-K across K32/K64 and output tails");

  for (const int reduction : {2, 3, 33, 65}) {
    std::vector<float> a(static_cast<std::size_t>(m * reduction)),
        b(static_cast<std::size_t>(reduction * n), 1.0F);
    for (int i = 0; i < m; ++i) {
      a[static_cast<std::size_t>(i * reduction + reduction - 2)] = -1;
      a[static_cast<std::size_t>(i * reduction + reduction - 1)] = 0x1.000002p0F;
    }
    for (int j = 0; j < n; ++j)
      b[static_cast<std::size_t>((reduction - 1) * n + j)] = 0x1.fffffcp-1F;
    volatile float separatelyRounded = 0x1.000002p0F * 0x1.fffffcp-1F;
    separatelyRounded = -1.0F + separatelyRounded;
    check(bits(separatelyRounded) == 0 && std::fma(0x1.000002p0F, 0x1.fffffcp-1F, -1.0F) != separatelyRounded,
          "FMA counteroracle is discriminating", 1, 1, reduction);
    run(m, n, reduction, a, b, "nonfused f32 across K/output tails");
  }
  const float values[]{std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
      -0.0F, 0x1p-149F, 0x1p-126F, -0x1p-126F, 0.5F, 1.0F};
  const float factors[]{0, 1, -1, 0.5F, -0.5F, 2, 1, -0.0F, -1};
  std::vector<float> a(static_cast<std::size_t>(m * k)), b(static_cast<std::size_t>(k * n), 1);
  for (int i = 0; i < m; ++i) a[static_cast<std::size_t>(i * k + k - 1)] = values[i % 9];
  for (int j = 0; j < n; ++j) b[static_cast<std::size_t>((k - 1) * n + j)] = factors[j % 9];
  run(m, n, k, a, b, "nonfinite/subnormal/signed-zero final tails");
  for (const float value : values)
    for (const float factor : factors)
      run(1, 1, 1, {value}, {factor}, "strict K1 class and zero sign");
}
} // namespace

int main(int argc, char **argv) {
  if (!strictIsaTestAvailable()) return 77;
  if (argc == 2 && std::string_view(argv[1]) == "--corrupt-output") corruptOutput = true;
  else if (argc == 2 && std::string_view(argv[1]) == "--oob") {
    // Deliberate private-leaf capacity violation. ASan must diagnose B's read
    // inside the issued kernel, not merely an out-of-bounds harness operation.
    auto *a = new float[2]{1, 1}, *b = new float[1]{1}, *c = new float[1]{0};
    auto ad = descriptor(a, 1, 2), bd = descriptor(b, 2, 1), cd = descriptor(c, 1, 1);
    _mlir_ciface___matcore_strict_gemm_f32_v1(&ad, &bd, &cd);
    delete[] a; delete[] b; delete[] c;
    return 0;
  } else if (argc != 1) return 2;
  std::fenv_t caller;
  if (std::fegetenv(&caller) || std::fesetenv(FE_DFL_ENV) || std::fegetround() != FE_TONEAREST)
    return 2;
  // M/N boundary matrix is parameter-relative, not a power-of-two assumption.
  for (const int m : {std::max(1, tileM - 1), tileM, tileM + 1, 2 * tileM + 1})
    for (const int n : {std::max(1, tileN - 1), tileN, tileN + 1, 2 * tileN + 1})
      ordinary(m, n, 3);
  for (const int n : {7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129})
    for (const int k : {1, 3, 31, 32, 33, 63, 64, 65})
      ordinary(tileM + 1, n, k);
  for (const auto shape : {std::array<int, 3>{0, tileN + 1, 3},
                            {tileM + 1, 0, 3}, {0, 0, 0},
                            {tileM + 1, tileN + 1, 0}, {1, 1, 0}})
    ordinary(shape[0], shape[1], shape[2]);
  numerical();
  for (const auto offsets : {std::array<std::size_t, 2>{0, 0}, {0, 1}, {1, 0}}) {
    alias(3, 3, 3, offsets[0], offsets[1]);
    alias(tileM + 1, tileN + 1, 7, offsets[0], offsets[1]);
    alias(tileM + 1, 129, 33, offsets[0], offsets[1]);
  }
  // Reproducible bit patterns include values that ordinary small-integer data
  // misses; only NaN payload identity is excluded from result comparisons.
  std::uint32_t state = 0x69ab37d1U;
  const auto next = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
  const std::uint32_t special[]{0, 0x80000000U, 0x00000001U, 0x007fffffU,
      0x00800000U, 0x3f800001U, 0xbf7ffffeU, 0x7f7fffffU,
      0x7f800000U, 0xff800000U, 0x7fc01234U, 0xffc04321U};
  for (int trial = 0; trial < 48; ++trial) {
    const int m = 1 + int(next() % unsigned(tileM + 2));
    const int n = 1 + int(next() % unsigned(std::max(34, tileN + 2)));
    const int k = 1 + int(next() % 67);
    std::vector<float> a(static_cast<std::size_t>(m * k)), b(static_cast<std::size_t>(k * n));
    const auto value = [&]() {
      const auto random = next();
      return std::bit_cast<float>(trial % 2 == 0 ? special[random % 12]
          : ((random & 0x807fffffU) | ((118U + next() % 19) << 23)));
    };
    for (auto &v : a) v = value();
    for (auto &v : b) v = value();
    run(m, n, k, a, b, "independent reproducible bit-pattern result");
  }
  if (std::fesetenv(&caller)) return 2;
  std::printf("output pattern raw leaf tile=%dx%d: %llu cases, %llu checks, %llu failures\n",
      tileM, tileN, static_cast<unsigned long long>(cases),
      static_cast<unsigned long long>(checks), static_cast<unsigned long long>(failures));
  return failures ? 1 : 0;
}
