#ifndef MDSLC_TEST_SPIRV_STRICT_ORACLE_H
#define MDSLC_TEST_SPIRV_STRICT_ORACLE_H

// Test-only software binary32 arithmetic: no host float evaluation or FP state.
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strict_oracle {
using Word = std::uint32_t;
inline llvm::APFloat value(Word bits) {
  return llvm::APFloat(llvm::APFloat::IEEEsingle(), llvm::APInt(32, bits));
}
inline Word bits(const llvm::APFloat &v) {
  return static_cast<Word>(v.bitcastToAPInt().getZExtValue());
}
inline Word integer(std::int64_t v) {
  auto result = value(0);
  result.convertFromAPInt(llvm::APInt(64, static_cast<std::uint64_t>(v)), true,
                         llvm::APFloat::rmNearestTiesToEven);
  return bits(result);
}
inline bool nan(Word v) { return (v & 0x7fffffffU) > 0x7f800000U; }
inline bool equal(Word expected, Word actual) {
  return nan(expected) ? nan(actual) : expected == actual;
}
inline Word dot(const Word *a, const Word *b, std::size_t size,
                bool reverse = false) {
  auto sum = value(0);
  for (std::size_t p = 0; p < size; ++p) {
    const auto k = reverse ? size - p - 1 : p;
    auto product = value(a[k]);
    product.multiply(value(b[k]), llvm::APFloat::rmNearestTiesToEven);
    sum.add(product, llvm::APFloat::rmNearestTiesToEven);
  }
  return bits(sum);
}
struct Spec {
  const char *name;
  std::array<Word, 3> a, b;
  Word expected;
  bool negative_zero_padding = false;
};
// Independently calculated with rational/integer binary32 RNE arithmetic.
// NaNs compare by class only; every other result is bit-exact, including zeros.
inline constexpr Spec specs[] = {
    {"subnormal-input", {1, 0, 0}, {0x3f800000, 0, 0}, 1},
    {"negative-subnormal", {0x80000001, 0, 0}, {0x3f800000, 0, 0}, 0x80000001},
    {"subnormal-product", {0x00800000, 0, 0}, {0x3f000000, 0, 0}, 0x00400000},
    {"inexact-underflow-rne", {0x00800000, 0, 0}, {0x3f000003, 0, 0}, 0x00400002},
    {"negative-underflow-rne", {0x80800000, 0, 0}, {0x3f000003, 0, 0}, 0x80400002},
    {"signed-zero-seed", {0x80000000, 0x80000000, 0x80000000},
     {0x3f800000, 0x3f800000, 0x3f800000}, 0, true},
    {"non-fma", {0xbf800000, 0x3f800001, 0}, {0x3f800000, 0x3f7ffffe, 0}, 0},
    {"increasing-k", {0x4b800000, 0x3f800000, 0xcb800000},
     {0x3f800000, 0x3f800000, 0x3f800000}, 0},
    {"overflow-order", {0x7f7fffff, 0x7f7fffff, 0xff7fffff},
     {0x3f800000, 0x3f800000, 0x3f800000}, 0x7f800000},
    {"rte-not-rtz", {0x3f800000, 0x33c00000, 0}, {0x3f800000, 0x3f800000, 0}, 0x3f800001},
    {"tie-even-low", {0x3f800000, 0x33800000, 0}, {0x3f800000, 0x3f800000, 0}, 0x3f800000},
    {"tie-even-high", {0x3f800001, 0x33800000, 0}, {0x3f800000, 0x3f800000, 0}, 0x3f800002},
    {"positive-infinity", {0x7f800000, 0, 0}, {0x3f800000, 0, 0}, 0x7f800000},
    {"negative-infinity", {0xff800000, 0, 0}, {0x3f800000, 0, 0}, 0xff800000},
    {"infinity-times-zero", {0x7f800000, 0, 0}, {0, 0, 0}, 0x7fc00000},
    {"opposed-infinities", {0x7f800000, 0xff800000, 0}, {0x3f800000, 0x3f800000, 0}, 0x7fc00000},
    {"quiet-nan", {0x7fc00001, 0, 0}, {0x3f800000, 0, 0}, 0x7fc00000},
};
inline unsigned selfCheck() {
  unsigned checks = 0;
  for (const auto &s : specs) {
    if (!equal(s.expected, dot(s.a.data(), s.b.data(), 3)))
      throw std::runtime_error(std::string("APFloat known constant failed: ") + s.name);
    ++checks;
  }
  auto fused = value(0x3f800001);
  fused.fusedMultiplyAdd(value(0x3f7ffffe), value(0xbf800000),
                         llvm::APFloat::rmNearestTiesToEven);
  if (bits(fused) != 0xa8800000 ||
      dot(specs[7].a.data(), specs[7].b.data(), 3, true) != 0x3f800000)
    throw std::runtime_error("APFloat FMA/reverse-order negative control failed");
  return checks + 2;
}
struct Case {
  std::string name;
  std::vector<Word> a, b, expected;
};
// Caller first enforces the test work envelope. Independently check each integer
// prefix stays exactly representable, rather than inferring that from a work cap.
inline std::vector<Case> freeze(std::size_t m, std::size_t k, std::size_t n) {
  std::vector<Case> cases;
  auto append = [&](std::string name) -> Case & {
    cases.push_back({std::move(name), std::vector<Word>(m * k),
                     std::vector<Word>(k * n), std::vector<Word>(m * n)});
    return cases.back();
  };
  auto &base = append("nonuniform-rectangular-integers");
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t p = 0; p < k; ++p)
      base.a[i * k + p] = integer(static_cast<std::int64_t>((17 * i + 5 * p) % 19) - 9);
  for (std::size_t p = 0; p < k; ++p)
    for (std::size_t j = 0; j < n; ++j)
      base.b[p * n + j] = integer(static_cast<std::int64_t>((7 * p + 11 * j) % 23) - 11);
  auto &last = append("last-k-only-row-column");
  for (std::size_t i = 0; i < m; ++i) last.a[i * k + k - 1] = integer(i % 7 + 1);
  for (std::size_t j = 0; j < n; ++j) last.b[(k - 1) * n + j] = integer(j % 11 + 1);
  if (k >= 3)
    for (const auto &s : specs) {
      auto &c = append(s.name);
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p)
          c.a[i * k + p] = p < 3 ? s.a[p] : (s.negative_zero_padding ? 0x80000000 : 0);
      for (std::size_t p = 0; p < k; ++p)
        for (std::size_t j = 0; j < n; ++j)
          c.b[p * n + j] = p < 3 ? s.b[p] : (s.negative_zero_padding ? 0x3f800000 : 0);
    }
  std::vector<Word> column(k);
  for (std::size_t c = 0; c < cases.size(); ++c)
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t p = 0; p < k; ++p) column[p] = cases[c].b[p * n + j];
        const Word expected = dot(cases[c].a.data() + i * k, column.data(), k);
        if (c >= 2 && !equal(specs[c - 2].expected, expected))
          throw std::runtime_error("padded oracle differs from analytic constant");
        if (c < 2) {
          std::int64_t exact = 0;
          if (c == 1) exact = (i % 7 + 1) * (j % 11 + 1);
          else for (std::size_t p = 0; p < k; ++p) {
            exact += (static_cast<std::int64_t>((17 * i + 5 * p) % 19) - 9) *
                     (static_cast<std::int64_t>((7 * p + 11 * j) % 23) - 11);
            if (exact <= -(1LL << 24) || exact >= (1LL << 24))
              throw std::runtime_error("integer oracle prefix not exactly representable");
          }
          if (expected != integer(exact)) throw std::runtime_error("integer analytic oracle disagrees");
        }
        cases[c].expected[i * n + j] = expected;
      }
  return cases;
}
} // namespace strict_oracle
#endif
