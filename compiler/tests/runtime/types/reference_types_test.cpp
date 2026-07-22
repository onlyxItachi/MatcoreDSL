#include "cpu_numeric_reference.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace {

namespace runtime = matcore::mdslc::runtime;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

float float_from_bits(std::uint32_t bits) {
  return std::bit_cast<float>(bits);
}

void conversion_contract() {
  expect(runtime::bfloat16_from_float_v1(1.0F).bits == UINT16_C(0x3f80),
         "BF16 conversion preserves one exactly");
  expect(runtime::bfloat16_from_float_v1(-2.5F).bits == UINT16_C(0xc020),
         "BF16 conversion preserves an exact negative value");
  expect(runtime::bfloat16_from_float_v1(float_from_bits(UINT32_C(0x3f808000)))
             .bits == UINT16_C(0x3f80),
         "BF16 halfway value rounds to the even lower encoding");
  expect(runtime::bfloat16_from_float_v1(float_from_bits(UINT32_C(0x3f818000)))
             .bits == UINT16_C(0x3f82),
         "BF16 halfway value rounds away from an odd lower encoding");
  expect(runtime::bfloat16_from_float_v1(float_from_bits(UINT32_C(0x3f808001)))
             .bits == UINT16_C(0x3f81),
         "BF16 value above halfway rounds upward");
  expect(runtime::bfloat16_from_float_v1(-0.0F).bits == UINT16_C(0x8000),
         "BF16 preserves signed zero");
  expect(runtime::bfloat16_from_float_v1(
             std::numeric_limits<float>::infinity())
             .bits == UINT16_C(0x7f80),
         "BF16 preserves infinity");
  expect(runtime::bfloat16_from_float_v1(float_from_bits(UINT32_C(0x7f812345)))
             .bits == UINT16_C(0x7fc0),
         "BF16 canonicalizes positive NaN to a quiet payload");
  expect(runtime::bfloat16_from_float_v1(float_from_bits(UINT32_C(0xff812345)))
             .bits == UINT16_C(0xffc0),
         "BF16 canonical NaN preserves its sign");
  expect(std::bit_cast<std::uint32_t>(runtime::bfloat16_to_float_v1(
             runtime::BFloat16V1{UINT16_C(0x4049)})) == UINT32_C(0x40490000),
         "BF16 widening appends sixteen zero fraction bits exactly");
}

void bf16_reference_contract() {
  constexpr std::size_t m = 7;
  constexpr std::size_t k = 19;
  constexpr std::size_t n = 11;
  const runtime::CpuTypedGemmShapeV1 shape{
      runtime::kCpuNumericReferenceVersionV1,
      static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
      static_cast<std::int64_t>(k)};
  std::mt19937 generator(0x42463136U);
  std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
  std::vector<runtime::BFloat16V1> lhs(m * k);
  std::vector<runtime::BFloat16V1> rhs(k * n);
  for (auto &value : lhs)
    value = runtime::bfloat16_from_float_v1(distribution(generator));
  for (auto &value : rhs)
    value = runtime::bfloat16_from_float_v1(distribution(generator));
  std::vector<float> out(m * n, -99.0F);

  expect(runtime::cpu_reference_gemm_bf16_f32_v1(
             shape, lhs.data(), rhs.data(), out.data()) ==
             runtime::CpuNumericReferenceStatusV1::success,
         "BF16/F32 reference GEMM succeeds");
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      double expected = 0.0;
      for (std::size_t p = 0; p < k; ++p) {
        expected += static_cast<double>(runtime::bfloat16_to_float_v1(
                        lhs[row * k + p])) *
                    static_cast<double>(runtime::bfloat16_to_float_v1(
                        rhs[p * n + column]));
      }
      const double scale = std::max(1.0, std::fabs(expected));
      expect(std::fabs(static_cast<double>(out[row * n + column]) - expected) <=
                 2.0e-6 * scale,
             "BF16/F32 result matches an independent double oracle");
    }
  }
}

std::int32_t modular_i8_oracle(const std::int8_t *lhs,
                               const std::int8_t *rhs, std::size_t row,
                               std::size_t column, std::size_t k,
                               std::size_t n) {
  std::uint64_t accumulator = 0;
  for (std::size_t p = 0; p < k; ++p) {
    const std::int64_t product =
        static_cast<std::int64_t>(lhs[row * k + p]) *
        static_cast<std::int64_t>(rhs[p * n + column]);
    accumulator = (accumulator + static_cast<std::uint64_t>(product)) &
                  UINT64_C(0xffffffff);
  }
  return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(accumulator));
}

void i8_reference_contract() {
  constexpr std::size_t m = 9;
  constexpr std::size_t k = 37;
  constexpr std::size_t n = 13;
  const runtime::CpuTypedGemmShapeV1 shape{
      runtime::kCpuNumericReferenceVersionV1,
      static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
      static_cast<std::int64_t>(k)};
  std::mt19937 generator(0x49384933U);
  std::uniform_int_distribution<int> distribution(-128, 127);
  std::vector<std::int8_t> lhs(m * k);
  std::vector<std::int8_t> rhs(k * n);
  for (auto &value : lhs) value = static_cast<std::int8_t>(distribution(generator));
  for (auto &value : rhs) value = static_cast<std::int8_t>(distribution(generator));
  std::vector<std::int32_t> out(m * n, -99);

  expect(runtime::cpu_reference_gemm_i8_i32_v1(
             shape, lhs.data(), rhs.data(), out.data()) ==
             runtime::CpuNumericReferenceStatusV1::success,
         "I8/I32 reference GEMM succeeds");
  for (std::size_t row = 0; row < m; ++row)
    for (std::size_t column = 0; column < n; ++column)
      expect(out[row * n + column] ==
                 modular_i8_oracle(lhs.data(), rhs.data(), row, column, k, n),
             "I8/I32 result matches the independent modular oracle");

  constexpr std::size_t overflow_k = 300000;
  const runtime::CpuTypedGemmShapeV1 overflow_shape{
      runtime::kCpuNumericReferenceVersionV1, 1, 1,
      static_cast<std::int64_t>(overflow_k)};
  std::vector<std::int8_t> overflow_lhs(overflow_k, INT8_C(-128));
  std::vector<std::int8_t> overflow_rhs(overflow_k, INT8_C(-128));
  std::int32_t overflow_out = 0;
  expect(runtime::cpu_reference_gemm_i8_i32_v1(
             overflow_shape, overflow_lhs.data(), overflow_rhs.data(),
             &overflow_out) == runtime::CpuNumericReferenceStatusV1::success,
         "I8 accumulation exceeding INT32_MAX remains defined");
  expect(overflow_out == modular_i8_oracle(overflow_lhs.data(),
                                            overflow_rhs.data(), 0, 0,
                                            overflow_k, 1),
         "I8 accumulation wraps exactly modulo 2^32");
}

void rejection_contract() {
  runtime::CpuTypedGemmShapeV1 shape{runtime::kCpuNumericReferenceVersionV1,
                                     2, 2, 2};
  std::vector<runtime::BFloat16V1> bf16_lhs(4);
  std::vector<runtime::BFloat16V1> bf16_rhs(4);
  std::vector<float> f32_out(4, 17.0F);
  const auto unchanged_f32 = f32_out;

  auto invalid = shape;
  invalid.version = 99;
  expect(runtime::cpu_reference_gemm_bf16_f32_v1(
             invalid, bf16_lhs.data(), bf16_rhs.data(), f32_out.data()) ==
             runtime::CpuNumericReferenceStatusV1::invalid_problem &&
             f32_out == unchanged_f32,
         "unknown typed-reference version fails without output mutation");
  invalid = shape;
  invalid.m = 0;
  expect(runtime::cpu_reference_gemm_bf16_f32_v1(
             invalid, bf16_lhs.data(), bf16_rhs.data(), f32_out.data()) ==
             runtime::CpuNumericReferenceStatusV1::invalid_problem &&
             f32_out == unchanged_f32,
         "zero typed dimension fails without output mutation");
  expect(runtime::cpu_reference_gemm_bf16_f32_v1(
             shape, nullptr, bf16_rhs.data(), f32_out.data()) ==
             runtime::CpuNumericReferenceStatusV1::null_pointer &&
             f32_out == unchanged_f32,
         "null BF16 input fails without output mutation");
  expect(runtime::cpu_reference_gemm_bf16_f32_v1(
             shape, bf16_lhs.data(), bf16_rhs.data(),
             reinterpret_cast<float *>(bf16_lhs.data())) ==
             runtime::CpuNumericReferenceStatusV1::alias_violation,
         "BF16 output/input overlap is rejected");

  std::vector<std::int8_t> i8_lhs(4);
  std::vector<std::int8_t> i8_rhs(4);
  std::vector<std::int32_t> i32_out(4, 29);
  const auto unchanged_i32 = i32_out;
  expect(runtime::cpu_reference_gemm_i8_i32_v1(
             shape, i8_lhs.data(), i8_rhs.data(), nullptr) ==
             runtime::CpuNumericReferenceStatusV1::null_pointer &&
             i32_out == unchanged_i32,
         "null I32 output is rejected");

  invalid = shape;
  invalid.m = std::numeric_limits<std::int64_t>::max();
  invalid.k = std::numeric_limits<std::int64_t>::max();
  expect(runtime::cpu_reference_gemm_i8_i32_v1(
             invalid, i8_lhs.data(), i8_rhs.data(), i32_out.data()) ==
             runtime::CpuNumericReferenceStatusV1::arithmetic_overflow &&
             i32_out == unchanged_i32,
         "typed GEMM byte-count overflow fails before output mutation");
}

}  // namespace

int main() {
  conversion_contract();
  bf16_reference_contract();
  i8_reference_contract();
  rejection_contract();
  if (failures != 0) return 1;
  std::cout << "BF16/F32 and I8/I32 reference semantics: all tests passed\n";
  return 0;
}
