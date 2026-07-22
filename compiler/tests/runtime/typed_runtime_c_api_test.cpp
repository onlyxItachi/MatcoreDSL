#include "matcore/runtime_c.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

matcore_tensor_desc_v0 matrix(void *data, matcore_dtype_v0 dtype,
                              std::int64_t rows, std::int64_t columns,
                              matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.data = data;
  result.dtype = dtype;
  result.rank = 2;
  result.dims[0] = rows;
  result.dims[1] = columns;
  result.strides[0] = columns;
  result.strides[1] = 1;
  result.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  result.mutability = mutability;
  return result;
}

matcore_policy_v0 cpu_policy() {
  matcore_policy_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.target = MATCORE_TARGET_CPU_V0;
  result.fallback = MATCORE_FALLBACK_ERROR_V0;
  return result;
}

void bf16_public_contract() {
  // Exact BF16 encodings for [[1,2,3],[-1,0,2]] and [[1,2],[3,4],[5,6]].
  std::array<matcore_bf16_v1, 6> lhs{
      UINT16_C(0x3f80), UINT16_C(0x4000), UINT16_C(0x4040),
      UINT16_C(0xbf80), UINT16_C(0x0000), UINT16_C(0x4000)};
  std::array<matcore_bf16_v1, 6> rhs{
      UINT16_C(0x3f80), UINT16_C(0x4000), UINT16_C(0x4040),
      UINT16_C(0x4080), UINT16_C(0x40a0), UINT16_C(0x40c0)};
  std::array<float, 4> out{-71.0F, -71.0F, -71.0F, -71.0F};
  auto lhs_desc = matrix(lhs.data(), MATCORE_DTYPE_BF16_V0, 2, 3,
                         MATCORE_MUTABILITY_READ_ONLY_V0);
  auto rhs_desc = matrix(rhs.data(), MATCORE_DTYPE_BF16_V0, 3, 2,
                         MATCORE_MUTABILITY_READ_ONLY_V0);
  auto out_desc = matrix(out.data(), MATCORE_DTYPE_F32_V0, 2, 2,
                         MATCORE_MUTABILITY_READ_WRITE_V0);
  const auto policy = cpu_policy();
  const matcore_status_v0 result =
      matcore_runtime_gemm_bf16_f32_reference_v1(
          &out_desc, &lhs_desc, &rhs_desc, &policy);
  expect(result.code == MATCORE_STATUS_OK_V0,
         "public BF16/F32 reference GEMM succeeds");
  constexpr std::array<float, 4> expected{22.0F, 28.0F, 9.0F, 10.0F};
  expect(std::equal(out.begin(), out.end(), expected.begin()),
         "public BF16/F32 result is exact for integral inputs");

  const auto unchanged = out;
  out_desc.dtype = MATCORE_DTYPE_BF16_V0;
  const matcore_status_v0 wrong_output =
      matcore_runtime_gemm_bf16_f32_reference_v1(
          &out_desc, &lhs_desc, &rhs_desc, &policy);
  expect(wrong_output.code == MATCORE_STATUS_UNSUPPORTED_DTYPE_V0 &&
             out == unchanged,
         "wrong BF16 output dtype fails before mutation");
}

void i8_public_contract() {
  std::array<std::int8_t, 6> lhs{1, 2, 3, -1, 0, 2};
  std::array<std::int8_t, 6> rhs{1, 2, 3, 4, 5, 6};
  std::array<std::int32_t, 4> out{-17, -17, -17, -17};
  auto lhs_desc = matrix(lhs.data(), MATCORE_DTYPE_I8_V0, 2, 3,
                         MATCORE_MUTABILITY_READ_ONLY_V0);
  auto rhs_desc = matrix(rhs.data(), MATCORE_DTYPE_I8_V0, 3, 2,
                         MATCORE_MUTABILITY_READ_ONLY_V0);
  auto out_desc = matrix(out.data(), MATCORE_DTYPE_I32_V0, 2, 2,
                         MATCORE_MUTABILITY_READ_WRITE_V0);
  const auto policy = cpu_policy();
  const matcore_status_v0 result =
      matcore_runtime_gemm_i8_i32_reference_v1(
          &out_desc, &lhs_desc, &rhs_desc, &policy);
  expect(result.code == MATCORE_STATUS_OK_V0,
         "public I8/I32 reference GEMM succeeds");
  constexpr std::array<std::int32_t, 4> expected{22, 28, 9, 10};
  expect(out == expected, "public I8/I32 result is exact");

  const auto unchanged = out;
  out_desc.mutability = MATCORE_MUTABILITY_READ_ONLY_V0;
  const matcore_status_v0 immutable =
      matcore_runtime_gemm_i8_i32_reference_v1(
          &out_desc, &lhs_desc, &rhs_desc, &policy);
  expect(immutable.code == MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0 &&
             out == unchanged,
         "immutable I32 output fails before mutation");

  out_desc.mutability = MATCORE_MUTABILITY_READ_WRITE_V0;
  out_desc.data = lhs.data();
  const matcore_status_v0 alias =
      matcore_runtime_gemm_i8_i32_reference_v1(
          &out_desc, &lhs_desc, &rhs_desc, &policy);
  expect(alias.code == MATCORE_STATUS_ALIAS_VIOLATION_V0,
         "typed C ABI rejects overlapping output storage");
}

}  // namespace

int main() {
  bf16_public_contract();
  i8_public_contract();
  if (failures != 0) return 1;
  std::cout << "typed runtime C ABI: all tests passed\n";
  return 0;
}
