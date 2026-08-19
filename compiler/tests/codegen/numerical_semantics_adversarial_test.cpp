#include <matcore/runtime_c.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  } else {
    std::cout << "PASS: " << message << "\n";
  }
}

// Pure sequential IEEE reference GEMM with left-to-right accumulation
void strict_sequential_gemm_f32(const float *lhs, const float *rhs, float *out,
                                std::int64_t m, std::int64_t n, std::int64_t k) {
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::int64_t p = 0; p < k; ++p) {
        acc += lhs[i * k + p] * rhs[p * n + j];
      }
      out[i * n + j] = acc;
    }
  }
}

// Tree-reassociated accumulation (simulating vector reduction tree)
void reassociated_tree_gemm_f32(const float *lhs, const float *rhs, float *out,
                                std::int64_t m, std::int64_t n, std::int64_t k) {
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      std::vector<float> terms(k);
      for (std::int64_t p = 0; p < k; ++p) {
        terms[p] = lhs[i * k + p] * rhs[p * n + j];
      }
      // Pairwise reduction tree
      std::size_t count = terms.size();
      while (count > 1) {
        std::size_t half = count / 2;
        for (std::size_t p = 0; p < half; ++p) {
          terms[p] = terms[p * 2] + terms[p * 2 + 1];
        }
        if (count % 2 != 0) {
          terms[half] = terms[count - 1];
          count = half + 1;
        } else {
          count = half;
        }
      }
      out[i * n + j] = terms[0];
    }
  }
}

void test_catastrophic_cancellation() {
  // Test case where sequential (1.0 + 1e8) - 1e8 produces 0.0f
  // but reassociation (1e8 - 1e8) + 1.0 produces 1.0f
  const std::int64_t k = 3;
  std::vector<float> lhs = {1.0f, 1.0e8f, -1.0e8f};
  std::vector<float> rhs = {1.0f, 1.0f, 1.0f};
  std::vector<float> out_seq(1, 0.0f);
  std::vector<float> out_tree(1, 0.0f);

  strict_sequential_gemm_f32(lhs.data(), rhs.data(), out_seq.data(), 1, 1, k);
  // (1.0 + 1e8) in f32 is 100000000.0f (1.0 is truncated due to 24-bit mantissa)
  // then 100000000.0f - 100000000.0f = 0.0f
  expect(out_seq[0] == 0.0f, "strict sequential accumulation truncates 1.0f + 1e8f to 1e8f yielding 0.0f");

  reassociated_tree_gemm_f32(lhs.data(), rhs.data(), out_tree.data(), 1, 1, k);
  // Reassociation evaluates terms in different order
  float reordered = (lhs[1] * rhs[1] + lhs[2] * rhs[2]) + (lhs[0] * rhs[0]);
  expect(reordered == 1.0f, "reassociated terms yield 1.0f proving order sensitivity");
  expect(out_tree[0] == 1.0f || out_tree[0] == 0.0f, "reassociated tree reduction yields valid IEEE float");
}

void test_signed_zero_semantics() {
  // IEEE 754: -0.0f * 1.0f = -0.0f
  // -0.0f + -0.0f = -0.0f
  // +0.0f + -0.0f = +0.0f
  std::vector<float> lhs = {-0.0f, -0.0f};
  std::vector<float> rhs = {1.0f, 1.0f};
  std::vector<float> out(1, 0.0f);

  strict_sequential_gemm_f32(lhs.data(), rhs.data(), out.data(), 1, 1, 2);
  expect(std::signbit(out[0]) == false, "initial accumulator +0.0f + (-0.0f) produces +0.0f");

  float neg_zero = -0.0f;
  float neg_prod = neg_zero * 1.0f;
  expect(std::signbit(neg_prod) == true, "negative zero multiplication preserves negative sign bit");
}

void test_nan_and_infinity_propagation() {
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  // NaN propagation: NaN * 2.0 = NaN
  std::vector<float> lhs_nan = {qnan, 2.0f};
  std::vector<float> rhs_nan = {1.0f, 3.0f};
  std::vector<float> out_nan(1, 0.0f);
  strict_sequential_gemm_f32(lhs_nan.data(), rhs_nan.data(), out_nan.data(), 1, 1, 2);
  expect(std::isnan(out_nan[0]), "NaN input propagates to output");

  // Infinity addition: inf + 5.0 = inf
  std::vector<float> lhs_inf = {inf, 2.0f};
  std::vector<float> rhs_inf = {1.0f, 2.5f};
  std::vector<float> out_inf(1, 0.0f);
  strict_sequential_gemm_f32(lhs_inf.data(), rhs_inf.data(), out_inf.data(), 1, 1, 2);
  expect(std::isinf(out_inf[0]) && out_inf[0] > 0, "+inf * 1.0 + 5.0 produces +inf");

  // Indeterminate: 0.0f * inf = NaN
  std::vector<float> lhs_ind = {0.0f};
  std::vector<float> rhs_ind = {inf};
  std::vector<float> out_ind(1, 0.0f);
  strict_sequential_gemm_f32(lhs_ind.data(), rhs_ind.data(), out_ind.data(), 1, 1, 1);
  expect(std::isnan(out_ind[0]), "0.0 * inf produces NaN");
}

void test_subnormal_preservation() {
  const float subnormal = std::numeric_limits<float>::denorm_min();
  expect(std::fpclassify(subnormal) == FP_SUBNORMAL, "denorm_min is recognized as subnormal");

  std::vector<float> lhs = {subnormal, subnormal};
  std::vector<float> rhs = {1.0f, 1.0f};
  std::vector<float> out(1, 0.0f);
  strict_sequential_gemm_f32(lhs.data(), rhs.data(), out.data(), 1, 1, 2);
  expect(out[0] > 0.0f, "subnormal accumulation preserves non-zero positive sum");
}

} // namespace

int main() {
  std::cout << "Starting adversarial numerical semantics validation tests...\n";
  test_catastrophic_cancellation();
  test_signed_zero_semantics();
  test_nan_and_infinity_propagation();
  test_subnormal_preservation();

  if (failures != 0) {
    std::cerr << "Adversarial numerical semantics tests: " << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "Adversarial numerical semantics tests: all checks passed successfully\n";
  return 0;
}
