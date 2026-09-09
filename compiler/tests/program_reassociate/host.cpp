#include "api.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#pragma STDC FP_CONTRACT OFF
namespace {
unsigned checks = 0, failures = 0;
volatile unsigned host_trace = 0;
void check(bool good, const char *what) {
  ++checks;
  if (!good) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}
bool same(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const bool fused = std::strcmp(argv[1], "generated-reassociate") == 0;
  if (!fused && std::strcmp(argv[1], "generated-strict") != 0) return 2;
  if (fused && (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))) {
    std::cout << "SKIP program_reassociate: AVX2/FMA unavailable\n";
    return 77;
  }
  std::array<float, 8> a;
  std::array<float, 16> b;
  std::array<float, 34> first;
  std::array<float, 24> right;
  std::array<float, 14> last;
  first.fill(-777);
  last.fill(-888);
  for (unsigned row = 0; row < 4; ++row) {
    a[2*row] = -1;
    a[2*row+1] = 0x1.000002p0F;
  }
  for (unsigned col = 0; col < 8; ++col) {
    b[col] = 1;
    b[8+col] = 0x1.fffffep-1F;
  }
  const auto savedA = a;
  const auto savedB = b;
  const float fusedOracle = std::fma(a[1], b[8], -1.0F);
  volatile float roundedProduct = a[1] * b[8];
  volatile float strictOracle = -1.0F + roundedProduct;
  check(std::bit_cast<std::uint32_t>(fusedOracle) == 0x337ffffeU &&
        same(strictOracle, 0.0F) && !same(fusedOracle, strictOracle),
        "full 4x8 oracle distinguishes actual FMA from strict arithmetic");
  const float expectedFirst = fused ? fusedOracle : strictOracle;
  md::Observation old;
  {
    auto result = permissive_first(
        {a.data(), 4, 2, 8, md::Access::read_write},
        {b.data(), 2, 8, 16, md::Access::read_write},
        {first.data()+1, 4, 8, 32, md::Access::read_write});
    check(result.ok() && result.error() == md::Error::ok,
          "first source region succeeds before later strict refusal");
    check(result.publication_count() == 1 && result.observation_count() == 1 &&
          result.completed_frontier() == 6 && result.completed_effect_frontier() == 5 &&
          result.failed_frontier() == 0, "first exact effect and completion frontier");
    old = result.observation(0);
    check(old.valid() && old.rows() == 4 && old.columns() == 8,
          "first source region creates an owning observation");
    for (unsigned i = 0; i < 32; ++i) {
      check(same(first[1+i], expectedFirst), "actual first full-tile arithmetic");
      check(old.valid() && same(old.data()[i], expectedFirst), "first observation exact math");
    }
  }
  host_trace = 1;
  // Ordinary host mutation is between independent Result-returning calls. It
  // cannot be moved into either pure region or treated as cross-region fusion.
  for (unsigned i = 0; i < 32; ++i) first[1+i] = float(int((i*3+1)%7)-3);
  for (unsigned i = 0; i < 24; ++i) right[i] = float(int((i*5+2)%9)-4);
  const auto savedRight = right;
  const auto atBoundary = first;
  std::array<float, 12> expectedLast{};
  for (unsigned i = 0; i < 4; ++i)
    for (unsigned j = 0; j < 3; ++j) {
      double value = 0;
      for (unsigned k = 0; k < 8; ++k)
        value += double(atBoundary[1+8*i+k]) * double(right[3*k+j]);
      expectedLast[3*i+j] = static_cast<float>(value);
    }
  host_trace = host_trace * 10 + 2;
  check(host_trace == 12, "ordinary host boundary precedes second region call");
  for (unsigned i = 0; i < 32; ++i)
    check(old.valid() && same(old.data()[i], expectedFirst),
          "first observation survives first Result and host mutation");
  {
    auto result = strict_second(
        {first.data()+1, 4, 8, 32, md::Access::read_write},
        {right.data(), 8, 3, 24, md::Access::read_write},
        {last.data()+1, 4, 3, 12, md::Access::read_write});
    check(host_trace == 12, "region preserves ordinary host trace");
    if (fused) {
      check(!result && result.error() == md::Error::candidate_incompatible,
            "later strict GEMM refuses forced reassociate without fallback");
      check(result.failed_frontier() == 3 && result.completed_frontier() == 2 &&
            result.completed_effect_frontier() == 0 && result.publication_count() == 0 &&
            result.observation_count() == 0, "second region retains exact pre-GEMM prefix");
      check(result.failure_location().line == 6 && result.failure_location().column == 13 &&
            result.failure_location().file &&
            std::strstr(result.failure_location().file, "second.mdsl"),
            "later failure names physical second source GEMM");
      for (unsigned i = 0; i < 12; ++i)
        check(same(last[1+i], -888.0F), "strict refusal never publishes second output");
    } else {
      check(result.ok() && result.error() == md::Error::ok,
            "generated strict accepts both source numerical profiles");
      check(result.failed_frontier() == 0 && result.completed_frontier() == 6 &&
            result.completed_effect_frontier() == 5 && result.publication_count() == 1 &&
            result.observation_count() == 1, "strict control second region completes all effects");
      const auto newer = result.observation(0);
      check(newer.valid() && newer.rows() == 4 && newer.columns() == 3,
            "second strict observation has rectangular dimensions");
      for (unsigned i = 0; i < 12; ++i) {
        check(same(last[1+i], expectedLast[i]), "second GEMM reads ordinary host mutation");
        check(newer.valid() && same(newer.data()[i], expectedLast[i]),
              "second strict owning observation exact math");
      }
    }
  }
  host_trace = host_trace * 10 + 3;
  check(host_trace == 123, "ordinary host resumes after second Result retirement");
  check(first == atBoundary && a == savedA && b == savedB && right == savedRight,
        "second success or refusal preserves all source resources");
  check(same(first.front(), -777) && same(first.back(), -777) &&
        same(last.front(), -888) && same(last.back(), -888), "all destination canaries survive");
  first.fill(999);
  last.fill(999);
  for (unsigned i = 0; i < 32; ++i)
    check(old.valid() && same(old.data()[i], expectedFirst),
          "first owning observation survives both Result lifetimes and final mutations");
  std::cout << "Program reassociate composition: " << argv[1] << "; " << checks
            << " checks; " << failures << " failures; host trace " << host_trace << '\n';
  return failures ? 1 : 0;
}
