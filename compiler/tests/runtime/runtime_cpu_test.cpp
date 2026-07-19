#include "matcore/runtime_c.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

matcore_tensor_desc_v0 desc(void *data, std::int64_t rows,
                            std::int64_t cols,
                            matcore_mutability_v0 mutability =
                                MATCORE_MUTABILITY_READ_ONLY_V0) {
  matcore_tensor_desc_v0 value{};
  value.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  value.struct_size = sizeof(value);
  value.data = data;
  value.dtype = MATCORE_DTYPE_F32_V0;
  value.rank = 2;
  value.dims[0] = rows;
  value.dims[1] = cols;
  value.strides[0] = cols;
  value.strides[1] = 1;
  value.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  value.mutability = mutability;
  return value;
}

matcore_policy_v0 policy() {
  matcore_policy_v0 value{};
  value.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  value.struct_size = sizeof(value);
  value.target = MATCORE_TARGET_CPU_V0;
  value.fallback = MATCORE_FALLBACK_ERROR_V0;
  return value;
}

void run_shape(std::int64_t m, std::int64_t k, std::int64_t n) {
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  std::vector<float> c(static_cast<std::size_t>(m * n), -99.0F);
  std::vector<double> oracle(static_cast<std::size_t>(m * n), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 2.0F;
  for (std::int64_t p = 0; p < k; ++p)
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < m; ++i)
        oracle[static_cast<std::size_t>(i * n + j)] +=
            static_cast<double>(a[static_cast<std::size_t>(i * k + p)]) *
            static_cast<double>(b[static_cast<std::size_t>(p * n + j)]);

  auto out = desc(c.data(), m, n, MATCORE_MUTABILITY_READ_WRITE_V0);
  auto lhs = desc(a.data(), m, k);
  auto rhs = desc(b.data(), k, n);
  const auto p = policy();
  const auto result = matcore_runtime_gemm_f32_v0(&out, &lhs, &rhs, &p);
  expect(result.code == MATCORE_STATUS_OK_V0, "valid GEMM returns success");
  for (std::size_t i = 0; i < c.size(); ++i)
    expect(std::fabs(static_cast<double>(c[i]) - oracle[i]) < 1.0e-5,
           "GEMM matches independent oracle");
}

void expect_error(matcore_status_code_v0 wanted,
                  const matcore_tensor_desc_v0 &out,
                  const matcore_tensor_desc_v0 &lhs,
                  const matcore_tensor_desc_v0 &rhs,
                  const matcore_policy_v0 &p, std::string_view name) {
  const auto result = matcore_runtime_gemm_f32_v0(&out, &lhs, &rhs, &p);
  expect(result.code == wanted, name);
  expect(result.message != nullptr, "errors carry a static diagnostic");
}

void negative_cases() {
  std::vector<float> a(6), b(6), c(4, 17.0F);
  auto out = desc(c.data(), 2, 2, MATCORE_MUTABILITY_READ_WRITE_V0);
  auto lhs = desc(a.data(), 2, 3);
  auto rhs = desc(b.data(), 3, 2);
  auto p = policy();

  expect(matcore_runtime_gemm_f32_v0(nullptr, &lhs, &rhs, &p).code ==
             MATCORE_STATUS_INVALID_ARGUMENT_V0,
         "null descriptor is rejected");
  auto changed = out;
  changed.abi_version = 1;
  expect_error(MATCORE_STATUS_ABI_MISMATCH_V0, changed, lhs, rhs, p,
               "ABI mismatch is rejected");
  changed = out;
  changed.reserved[0] = 1;
  expect_error(MATCORE_STATUS_INVALID_ARGUMENT_V0, changed, lhs, rhs, p,
               "nonzero tensor reserved fields are rejected");
  auto bad_policy = p;
  bad_policy.struct_size = 0;
  expect_error(MATCORE_STATUS_ABI_MISMATCH_V0, out, lhs, rhs, bad_policy,
               "policy ABI mismatch is rejected");
  bad_policy = p;
  bad_policy.reserved[0] = 1;
  expect_error(MATCORE_STATUS_INVALID_ARGUMENT_V0, out, lhs, rhs, bad_policy,
               "nonzero policy reserved fields are rejected");
  changed = lhs;
  changed.data = nullptr;
  expect_error(MATCORE_STATUS_INVALID_ARGUMENT_V0, out, changed, rhs, p,
               "null tensor data is rejected");
  changed = lhs;
  changed.dtype = MATCORE_DTYPE_F64_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_DTYPE_V0, out, changed, rhs, p,
               "unsupported dtype is rejected");
  changed = lhs;
  changed.rank = 1;
  expect_error(MATCORE_STATUS_UNSUPPORTED_RANK_V0, out, changed, rhs, p,
               "wrong rank is rejected");
  changed = lhs;
  changed.dims[0] = 0;
  expect_error(MATCORE_STATUS_INVALID_SHAPE_V0, out, changed, rhs, p,
               "nonpositive shape is rejected");
  changed = lhs;
  changed.strides[0] += 1;
  expect_error(MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0, out, changed, rhs, p,
               "noncontiguous layout is rejected");
  changed = out;
  changed.mutability = MATCORE_MUTABILITY_READ_ONLY_V0;
  expect_error(MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0, changed, lhs, rhs, p,
               "const output is rejected");
  changed = lhs;
  changed.mutability = MATCORE_MUTABILITY_INVALID_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_MUTABILITY_V0, out, changed, rhs, p,
               "invalid mutability is rejected");
  changed = lhs;
  changed.memory_space = MATCORE_MEMORY_SPACE_INVALID_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0, out, changed, rhs, p,
               "unknown memory space is rejected");
  changed = rhs;
  changed.memory_space = MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0;
  expect_error(MATCORE_STATUS_MIXED_MEMORY_SPACES_V0, out, lhs, changed, p,
               "mixed residency is rejected");
  auto device_out = out;
  auto device_lhs = lhs;
  auto device_rhs = rhs;
  device_out.memory_space = device_lhs.memory_space = device_rhs.memory_space =
      MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0, device_out,
               device_lhs, device_rhs, p, "device tensors are rejected");
  bad_policy = p;
  bad_policy.target = MATCORE_TARGET_CUDA_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_TARGET_V0, out, lhs, rhs, bad_policy,
               "non-CPU target is rejected");
  bad_policy = p;
  bad_policy.fallback = MATCORE_FALLBACK_ALLOW_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_FALLBACK_V0, out, lhs, rhs,
               bad_policy, "fallback other than error is rejected");
  changed = rhs;
  changed.dims[0] = 2;
  changed.strides[0] = 2;
  expect_error(MATCORE_STATUS_SHAPE_MISMATCH_V0, out, lhs, changed, p,
               "shape mismatch is rejected");
  auto alias_out = desc(a.data(), 2, 2, MATCORE_MUTABILITY_READ_WRITE_V0);
  expect_error(MATCORE_STATUS_ALIAS_VIOLATION_V0, alias_out, lhs, rhs, p,
               "output/input alias is rejected");
  changed = lhs;
  changed.data = static_cast<void *>(reinterpret_cast<char *>(a.data()) + 1);
  expect_error(MATCORE_STATUS_INVALID_ALIGNMENT_V0, out, changed, rhs, p,
               "misaligned f32 storage is rejected");
  auto huge_out = out;
  auto huge_lhs = lhs;
  huge_lhs.dims[0] = std::numeric_limits<std::int64_t>::max();
  huge_out.dims[0] = huge_lhs.dims[0];
  expect_error(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0, huge_out, huge_lhs, rhs,
               p, "overflowing byte range is rejected");
  expect(c == std::vector<float>(4, 17.0F),
         "failing calls do not modify output");
}

}  // namespace

int main() {
  run_shape(1, 1, 1);
  run_shape(2, 3, 2);
  run_shape(3, 2, 4);
  negative_cases();
  if (failures != 0) return 1;
  std::cout << "runtime CPU GEMM v0: all tests passed\n";
  return 0;
}
