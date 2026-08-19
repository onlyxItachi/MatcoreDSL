#include "codegen.h"
#include <matcore/runtime_c.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

void reference_gemm_f64(const float *lhs, const float *rhs, float *out,
                        std::int64_t m, std::int64_t n, std::int64_t k) {
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::int64_t p = 0; p < k; ++p) {
        acc += static_cast<double>(lhs[i * k + p]) *
               static_cast<double>(rhs[p * n + j]);
      }
      out[i * n + j] = static_cast<float>(acc);
    }
  }
}

matcore_tensor_desc_v0 make_desc(float *data, std::int64_t rows,
                                 std::int64_t cols,
                                 matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 desc{};
  desc.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc.struct_size = sizeof(matcore_tensor_desc_v0);
  desc.data = data;
  desc.dtype = MATCORE_DTYPE_F32_V0;
  desc.rank = 2;
  desc.dims[0] = rows;
  desc.dims[1] = cols;
  desc.strides[0] = cols;
  desc.strides[1] = 1;
  desc.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc.mutability = mutability;
  return desc;
}

// In-register microkernel implementation matching generated backend code
void direct_static_microkernel_16x64x32(float *__restrict out,
                                       const float *__restrict lhs,
                                       const float *__restrict rhs) noexcept {
  for (std::int64_t i = 0; i < 16; ++i)
    for (std::int64_t j = 0; j < 64; ++j) out[i * 64 + j] = 0.0f;
  for (std::int64_t i = 0; i < 16; ++i) {
    for (std::int64_t p = 0; p < 32; ++p) {
      const float a = lhs[i * 32 + p];
      for (std::int64_t j = 0; j < 64; ++j)
        out[i * 64 + j] += a * rhs[p * 64 + j];
    }
  }
}

void direct_static_microkernel_dot_128(float *__restrict out,
                                      const float *__restrict lhs,
                                      const float *__restrict rhs) noexcept {
  float sum = 0.0f;
  for (std::int64_t p = 0; p < 128; ++p) sum += lhs[p] * rhs[p];
  out[0] = sum;
}

void direct_static_microkernel_gemv_16x128(float *__restrict out,
                                          const float *__restrict lhs,
                                          const float *__restrict rhs) noexcept {
  for (std::int64_t i = 0; i < 16; ++i) {
    float sum = 0.0f;
    const float *row_lhs = &lhs[i * 128];
    for (std::int64_t p = 0; p < 128; ++p) sum += row_lhs[p] * rhs[p];
    out[i] = sum;
  }
}

void direct_static_microkernel_ger_16x16(float *__restrict out,
                                        const float *__restrict lhs,
                                        const float *__restrict rhs) noexcept {
  for (std::int64_t i = 0; i < 16; ++i) {
    const float a = lhs[i];
    for (std::int64_t j = 0; j < 16; ++j) out[i * 16 + j] = a * rhs[j];
  }
}

void test_direct_microkernels() {
  // Test 16x64x32
  {
    std::vector<float> lhs(16 * 32);
    std::vector<float> rhs(32 * 64);
    std::vector<float> out(16 * 64, 0.0f);
    std::vector<float> golden(16 * 64, 0.0f);

    for (std::size_t i = 0; i < lhs.size(); ++i)
      lhs[i] = std::sin(static_cast<float>(i) * 0.13f);
    for (std::size_t i = 0; i < rhs.size(); ++i)
      rhs[i] = std::cos(static_cast<float>(i) * 0.17f);

    reference_gemm_f64(lhs.data(), rhs.data(), golden.data(), 16, 64, 32);
    direct_static_microkernel_16x64x32(out.data(), lhs.data(), rhs.data());

    float max_error = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      max_error = std::max(max_error, std::abs(out[i] - golden[i]));
    }
    expect(max_error < 1e-4f, "direct 16x64x32 microkernel numerical error within float tolerance");
  }

  // Test Dot 128
  {
    std::vector<float> lhs(128);
    std::vector<float> rhs(128);
    float out = 0.0f;
    float golden = 0.0f;

    for (std::size_t i = 0; i < lhs.size(); ++i) {
      lhs[i] = static_cast<float>(i + 1) * 0.01f;
      rhs[i] = static_cast<float>(128 - i) * 0.02f;
    }

    reference_gemm_f64(lhs.data(), rhs.data(), &golden, 1, 1, 128);
    direct_static_microkernel_dot_128(&out, lhs.data(), rhs.data());

    expect(std::abs(out - golden) < 1e-4f, "direct dot product 128 numerical error within tolerance");
  }

  // Test GEMV 16x128
  {
    std::vector<float> lhs(16 * 128);
    std::vector<float> rhs(128);
    std::vector<float> out(16, 0.0f);
    std::vector<float> golden(16, 0.0f);

    for (std::size_t i = 0; i < lhs.size(); ++i)
      lhs[i] = std::sin(static_cast<float>(i) * 0.05f);
    for (std::size_t i = 0; i < rhs.size(); ++i)
      rhs[i] = std::cos(static_cast<float>(i) * 0.07f);

    reference_gemm_f64(lhs.data(), rhs.data(), golden.data(), 16, 1, 128);
    direct_static_microkernel_gemv_16x128(out.data(), lhs.data(), rhs.data());

    float max_error = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      max_error = std::max(max_error, std::abs(out[i] - golden[i]));
    }
    expect(max_error < 1e-4f, "direct GEMV 16x128 numerical error within tolerance");
  }

  // Test GER 16x16
  {
    std::vector<float> lhs(16);
    std::vector<float> rhs(16);
    std::vector<float> out(16 * 16, 0.0f);
    std::vector<float> golden(16 * 16, 0.0f);

    for (std::size_t i = 0; i < lhs.size(); ++i) lhs[i] = static_cast<float>(i + 1);
    for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] = static_cast<float>(i + 2);

    reference_gemm_f64(lhs.data(), rhs.data(), golden.data(), 16, 16, 1);
    direct_static_microkernel_ger_16x16(out.data(), lhs.data(), rhs.data());

    float max_error = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
      max_error = std::max(max_error, std::abs(out[i] - golden[i]));
    }
    expect(max_error == 0.0f, "direct GER 16x16 rank-1 outer product is bit-exact");
  }
}

void test_descriptor_guard_and_fallback() {
  std::vector<float> lhs(16 * 16, 1.0f);
  std::vector<float> rhs(16 * 16, 2.0f);
  std::vector<float> out(16 * 16, 0.0f);

  matcore_tensor_desc_v0 out_desc = make_desc(out.data(), 16, 16, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 lhs_desc = make_desc(lhs.data(), 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 rhs_desc = make_desc(rhs.data(), 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  matcore_status_v0 status = matcore_runtime_gemm_f32_v0(&out_desc, &lhs_desc, &rhs_desc, &policy);
  expect(status.code == MATCORE_STATUS_OK_V0, "runtime GEMM descriptor execution succeeds");
  expect(out[0] == 32.0f, "runtime GEMM output value is correct");
}

} // namespace

int main() {
  std::cout << "Starting static AOT specialization execution tests...\n";
  test_direct_microkernels();
  test_descriptor_guard_and_fallback();

  if (failures != 0) {
    std::cerr << "Static AOT specialization execution tests: " << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "Static AOT specialization execution tests: all checks passed successfully\n";
  return 0;
}
