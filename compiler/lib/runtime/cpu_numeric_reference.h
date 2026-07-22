#ifndef MATCORE_MDSLC_RUNTIME_CPU_NUMERIC_REFERENCE_H
#define MATCORE_MDSLC_RUNTIME_CPU_NUMERIC_REFERENCE_H

#include <cstdint>
#include <string_view>

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuNumericReferenceVersionV1 = 1;

// Sixteen-bit storage with the IEEE binary32 sign/exponent and high seven
// fraction bits. Float conversion is round-to-nearest, ties-to-even. NaN input
// is converted to one quiet NaN payload (sign preserved), making serialization
// deterministic rather than preserving host-specific signaling payloads.
struct BFloat16V1 {
  std::uint16_t bits = 0;
};
static_assert(sizeof(BFloat16V1) == sizeof(std::uint16_t));

BFloat16V1 bfloat16_from_float_v1(float value) noexcept;
float bfloat16_to_float_v1(BFloat16V1 value) noexcept;

struct CpuTypedGemmShapeV1 {
  std::uint32_t version = kCpuNumericReferenceVersionV1;
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
};

enum class CpuNumericReferenceStatusV1 : std::uint32_t {
  success = 0,
  invalid_problem = 1,
  null_pointer = 2,
  invalid_pointer_alignment = 3,
  alias_violation = 4,
  arithmetic_overflow = 5,
};

std::string_view cpu_numeric_reference_status_message_v1(
    CpuNumericReferenceStatusV1 status) noexcept;

// Row-major BF16 x BF16 -> F32. Each operand is exactly widened to binary32;
// accumulation is one correctly rounded binary32 fused multiply-add in
// increasing K order. Output is overwritten and no allocation is performed.
CpuNumericReferenceStatusV1 cpu_reference_gemm_bf16_f32_v1(
    const CpuTypedGemmShapeV1 &shape, const BFloat16V1 *lhs,
    const BFloat16V1 *rhs, float *out) noexcept;

// Storage-form entry point for the public C ABI's uint16_t BF16 payload.
// It has identical arithmetic and validation semantics without type-punning.
CpuNumericReferenceStatusV1 cpu_reference_gemm_bf16_storage_f32_v1(
    const CpuTypedGemmShapeV1 &shape, const std::uint16_t *lhs,
    const std::uint16_t *rhs, float *out) noexcept;

// Row-major I8 x I8 -> I32. Each signed product is exact. Accumulation and the
// final output are defined modulo 2^32, avoiding signed-overflow undefined
// behavior and matching non-saturating integer dot-product semantics.
CpuNumericReferenceStatusV1 cpu_reference_gemm_i8_i32_v1(
    const CpuTypedGemmShapeV1 &shape, const std::int8_t *lhs,
    const std::int8_t *rhs, std::int32_t *out) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
