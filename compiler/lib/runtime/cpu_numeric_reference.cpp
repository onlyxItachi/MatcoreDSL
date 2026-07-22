#include "cpu_numeric_reference.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace matcore::mdslc::runtime {
namespace {

struct ByteSpan {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t *result) noexcept {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool pointer_has_alignment(const void *pointer,
                           std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0 &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

bool make_span(const void *pointer, std::size_t elements,
               std::size_t element_bytes, ByteSpan *span) noexcept {
  std::size_t bytes = 0;
  if (pointer == nullptr || span == nullptr ||
      !checked_multiply(elements, element_bytes, &bytes)) {
    return false;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
  span->begin = address;
  span->end = address + bytes;
  return true;
}

bool overlaps(const ByteSpan &lhs, const ByteSpan &rhs) noexcept {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

CpuNumericReferenceStatusV1 dimensions(const CpuTypedGemmShapeV1 &shape,
                                       std::size_t *m, std::size_t *n,
                                       std::size_t *k) noexcept {
  if (shape.version != kCpuNumericReferenceVersionV1 || shape.m <= 0 ||
      shape.n <= 0 || shape.k <= 0 || m == nullptr || n == nullptr ||
      k == nullptr) {
    return CpuNumericReferenceStatusV1::invalid_problem;
  }
  const auto maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  if (static_cast<std::uint64_t>(shape.m) > maximum ||
      static_cast<std::uint64_t>(shape.n) > maximum ||
      static_cast<std::uint64_t>(shape.k) > maximum) {
    return CpuNumericReferenceStatusV1::arithmetic_overflow;
  }
  *m = static_cast<std::size_t>(shape.m);
  *n = static_cast<std::size_t>(shape.n);
  *k = static_cast<std::size_t>(shape.k);
  return CpuNumericReferenceStatusV1::success;
}

template <typename Lhs, typename Rhs, typename Out>
CpuNumericReferenceStatusV1 validate_buffers(
    const Lhs *lhs, const Rhs *rhs, Out *out, std::size_t m, std::size_t n,
    std::size_t k) noexcept {
  if (lhs == nullptr || rhs == nullptr || out == nullptr) {
    return CpuNumericReferenceStatusV1::null_pointer;
  }
  if (!pointer_has_alignment(lhs, alignof(Lhs)) ||
      !pointer_has_alignment(rhs, alignof(Rhs)) ||
      !pointer_has_alignment(out, alignof(Out))) {
    return CpuNumericReferenceStatusV1::invalid_pointer_alignment;
  }

  std::size_t lhs_elements = 0;
  std::size_t rhs_elements = 0;
  std::size_t out_elements = 0;
  ByteSpan lhs_span;
  ByteSpan rhs_span;
  ByteSpan out_span;
  if (!checked_multiply(m, k, &lhs_elements) ||
      !checked_multiply(k, n, &rhs_elements) ||
      !checked_multiply(m, n, &out_elements) ||
      !make_span(lhs, lhs_elements, sizeof(Lhs), &lhs_span) ||
      !make_span(rhs, rhs_elements, sizeof(Rhs), &rhs_span) ||
      !make_span(out, out_elements, sizeof(Out), &out_span)) {
    return CpuNumericReferenceStatusV1::arithmetic_overflow;
  }
  if (overlaps(out_span, lhs_span) || overlaps(out_span, rhs_span)) {
    return CpuNumericReferenceStatusV1::alias_violation;
  }
  return CpuNumericReferenceStatusV1::success;
}

template <typename Storage, typename Decode>
CpuNumericReferenceStatusV1 execute_bf16_reference(
    const CpuTypedGemmShapeV1 &shape, const Storage *lhs,
    const Storage *rhs, float *out, Decode decode) noexcept {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  auto status = dimensions(shape, &m, &n, &k);
  if (status != CpuNumericReferenceStatusV1::success) return status;
  status = validate_buffers(lhs, rhs, out, m, n, k);
  if (status != CpuNumericReferenceStatusV1::success) return status;

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      float accumulator = 0.0F;
      for (std::size_t p = 0; p < k; ++p) {
        accumulator = std::fma(decode(lhs[row * k + p]),
                               decode(rhs[p * n + column]), accumulator);
      }
      out[row * n + column] = accumulator;
    }
  }
  return CpuNumericReferenceStatusV1::success;
}

}  // namespace

BFloat16V1 bfloat16_from_float_v1(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t magnitude = bits & UINT32_C(0x7fffffff);
  if (magnitude > UINT32_C(0x7f800000)) {
    return {static_cast<std::uint16_t>((bits >> 16U & UINT32_C(0x8000)) |
                                      UINT32_C(0x7fc0))};
  }
  const std::uint32_t least_significant_retained_bit = (bits >> 16U) & 1U;
  const std::uint32_t rounded =
      bits + UINT32_C(0x7fff) + least_significant_retained_bit;
  return {static_cast<std::uint16_t>(rounded >> 16U)};
}

float bfloat16_to_float_v1(BFloat16V1 value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value.bits) << 16U);
}

std::string_view cpu_numeric_reference_status_message_v1(
    CpuNumericReferenceStatusV1 status) noexcept {
  switch (status) {
    case CpuNumericReferenceStatusV1::success:
      return "success";
    case CpuNumericReferenceStatusV1::invalid_problem:
      return "typed GEMM shape is invalid";
    case CpuNumericReferenceStatusV1::null_pointer:
      return "required typed GEMM pointer is null";
    case CpuNumericReferenceStatusV1::invalid_pointer_alignment:
      return "typed GEMM pointer violates natural alignment";
    case CpuNumericReferenceStatusV1::alias_violation:
      return "typed GEMM output overlaps an input";
    case CpuNumericReferenceStatusV1::arithmetic_overflow:
      return "typed GEMM size arithmetic overflowed";
  }
  return "unknown typed GEMM reference status";
}

CpuNumericReferenceStatusV1 cpu_reference_gemm_bf16_f32_v1(
    const CpuTypedGemmShapeV1 &shape, const BFloat16V1 *lhs,
    const BFloat16V1 *rhs, float *out) noexcept {
  return execute_bf16_reference(
      shape, lhs, rhs, out,
      [](BFloat16V1 value) { return bfloat16_to_float_v1(value); });
}

CpuNumericReferenceStatusV1 cpu_reference_gemm_bf16_storage_f32_v1(
    const CpuTypedGemmShapeV1 &shape, const std::uint16_t *lhs,
    const std::uint16_t *rhs, float *out) noexcept {
  return execute_bf16_reference(shape, lhs, rhs, out, [](std::uint16_t value) {
    return bfloat16_to_float_v1(BFloat16V1{value});
  });
}

CpuNumericReferenceStatusV1 cpu_reference_gemm_i8_i32_v1(
    const CpuTypedGemmShapeV1 &shape, const std::int8_t *lhs,
    const std::int8_t *rhs, std::int32_t *out) noexcept {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  auto status = dimensions(shape, &m, &n, &k);
  if (status != CpuNumericReferenceStatusV1::success) return status;
  status = validate_buffers(lhs, rhs, out, m, n, k);
  if (status != CpuNumericReferenceStatusV1::success) return status;

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      std::uint32_t accumulator = 0;
      for (std::size_t p = 0; p < k; ++p) {
        const std::int32_t product =
            static_cast<std::int32_t>(lhs[row * k + p]) *
            static_cast<std::int32_t>(rhs[p * n + column]);
        accumulator += static_cast<std::uint32_t>(product);
      }
      out[row * n + column] = std::bit_cast<std::int32_t>(accumulator);
    }
  }
  return CpuNumericReferenceStatusV1::success;
}

}  // namespace matcore::mdslc::runtime
