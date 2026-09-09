// Independent direct-leaf oracle. A/B may share all or part of their storage;
// C is always a separate live allocation, as the private candidate requires.
// This checks supported input overlap, not absence of LLVM attributes: noalias
// on read-only arguments alone need not prohibit aliasing of unmodified memory.
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#endif

struct MemRef {
  float *allocated;
  float *aligned;
  std::int64_t offset;
  std::int64_t sizes[2];
  std::int64_t strides[2];
};
static_assert(sizeof(MemRef) == 56 && offsetof(MemRef, sizes) == 24 &&
              offsetof(MemRef, strides) == 40);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

extern "C" void _mlir_ciface___matcore_strict_gemm_f32_v1(
    MemRef *, MemRef *, MemRef *);

namespace {
constexpr std::size_t guard = 16;
constexpr float canary = -0x1.35ap12F;
std::uint64_t checks = 0, failures = 0, cases = 0;

void check(bool condition, const char *label, std::int64_t m,
           std::int64_t n, std::int64_t k) {
  ++checks;
  if (condition) return;
  if (failures++ < 12)
    std::fprintf(stderr, "FAIL input-alias %s M=%lld N=%lld K=%lld\n", label,
                 static_cast<long long>(m), static_cast<long long>(n),
                 static_cast<long long>(k));
}
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
bool sameNumber(float actual, float expected) {
  return (std::isnan(actual) && std::isnan(expected)) ||
         bits(actual) == bits(expected);
}
MemRef descriptor(float *data, std::int64_t rows, std::int64_t columns) {
  return {data, data, 0, {rows, columns}, {columns, 1}};
}

// Test data are prepared before either overlapping input is read. No fictitious
// independent A/B initialization is assumed after one view overwrites another.
float input(std::size_t index, unsigned pattern) {
  if (pattern == 0)
    return static_cast<float>(static_cast<int>((index * 13 + 5) % 31) - 15) *
           0.125F;
  const float values[] = {
      0.0F, -0.0F, 1.0F, -1.0F, 0x1.000002p0F, 0x1.fffffep-1F,
      std::numeric_limits<float>::denorm_min(), -0x1p-126F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(), 0.5F};
  return values[(index * 5 + 1) % (sizeof(values) / sizeof(values[0]))];
}

void run(std::int64_t m, std::int64_t n, std::int64_t k,
         std::size_t aOffset, std::size_t bOffset, unsigned pattern,
         bool corrupt) {
  ++cases;
  const auto aCount = static_cast<std::size_t>(m * k);
  const auto bCount = static_cast<std::size_t>(k * n);
  const auto cCount = static_cast<std::size_t>(m * n);
  const auto backingCount = std::max(aOffset + aCount, bOffset + bCount);
  std::vector<float> backing(backingCount + 2 * guard, canary);
  for (std::size_t i = 0; i < backingCount; ++i)
    backing[guard + i] = input(i, pattern);
  const auto original = backing;
  float *a = aCount ? backing.data() + guard + aOffset : nullptr;
  float *b = bCount ? backing.data() + guard + bOffset : nullptr;
  std::vector<float> output(cCount + 2 * guard, canary);
  float *c = cCount ? output.data() + guard : nullptr;
  std::vector<float> expected(cCount);
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j) {
      // Volatile intermediates make this scalar increasing-K oracle independent
      // of vectorization, contraction and the generated loop organization.
      volatile float sum = 0.0F;
      double exact = 0.0;
      for (std::int64_t r = 0; r < k; ++r) {
        volatile float product = a[i * k + r] * b[r * n + j];
        sum = sum + product;
        exact += static_cast<double>(a[i * k + r]) *
                 static_cast<double>(b[r * n + j]);
      }
      expected[static_cast<std::size_t>(i * n + j)] = sum;
      if (pattern == 0)
        check(static_cast<double>(sum) == exact,
              "ordinary f32 oracle agrees exactly with double", m, n, k);
    }
  auto ad = descriptor(a, m, k), bd = descriptor(b, k, n);
  auto cd = descriptor(c, m, n);
  const auto originalA = ad, originalB = bd, originalC = cd;
  _mlir_ciface___matcore_strict_gemm_f32_v1(&ad, &bd, &cd);
  if (corrupt && cCount) c[0] = 1234567.0F;
  for (std::size_t i = 0; i < cCount; ++i)
    check(sameNumber(c[i], expected[i]), "strict result", m, n, k);
  for (std::size_t i = 0; i < backing.size(); ++i)
    check(bits(backing[i]) == bits(original[i]), "input/backing unchanged", m, n, k);
  for (std::size_t i = 0; i < guard; ++i) {
    check(bits(output[i]) == bits(canary), "output leading canary", m, n, k);
    check(bits(output[guard + cCount + i]) == bits(canary),
          "output trailing canary", m, n, k);
  }
  check(std::memcmp(&ad, &originalA, sizeof(ad)) == 0 &&
            std::memcmp(&bd, &originalB, sizeof(bd)) == 0 &&
            std::memcmp(&cd, &originalC, sizeof(cd)) == 0,
        "descriptor objects unchanged", m, n, k);
}
} // namespace

int main(int argc, char **argv) {
  const bool corrupt = argc == 2 && std::strcmp(argv[1], "--corrupt-output") == 0;
  if (argc != 1 && !corrupt) return 2;
  std::fenv_t caller;
  if (std::fegetenv(&caller) != 0 || std::fesetenv(FE_DFL_ENV) != 0 ||
      std::fegetround() != FE_TONEAREST)
    return 2;
  // Same-storage square inputs include the actual same-Value geometry.
  // Rectangles with unequal offsets overlap in both address directions.
  constexpr std::int64_t dimensions[][3] = {
      {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {5, 5, 5},
      {3, 7, 5}, {5, 65, 33}, {7, 67, 65}, {9, 129, 31},
      {0, 7, 5}, {5, 0, 7}, {3, 7, 0}, {0, 0, 0}};
  for (const auto &shape : dimensions)
    for (unsigned pattern : {0U, 1U})
      for (const auto &offsets : {std::pair<std::size_t, std::size_t>{0, 0},
                                 {0, 1}, {1, 0}})
        run(shape[0], shape[1], shape[2], offsets.first, offsets.second,
            pattern, corrupt);
  if (std::fesetenv(&caller) != 0) return 2;
  std::printf("input alias execution: %llu cases, %llu checks, %llu failures\n",
              static_cast<unsigned long long>(cases),
              static_cast<unsigned long long>(checks),
              static_cast<unsigned long long>(failures));
  return failures != 0;
}
