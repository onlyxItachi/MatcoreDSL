#ifndef MATCORE_MDSLC_PLATFORM_FP_ENVIRONMENT_V1_H
#define MATCORE_MDSLC_PLATFORM_FP_ENVIRONMENT_V1_H

#include <cstdint>
#include <string_view>

namespace matcore::mdslc::platform {

inline constexpr std::uint32_t kFpEnvironmentVersionV1 = 1;

enum class FpEnvironmentBackendV1 : std::uint8_t {
  unavailable = 0,
  linux_x86_64 = 1,
  windows_x86_64 = 2,
};

enum class FpRoundingModeV1 : std::uint8_t {
  unknown = 0,
  nearest_even = 1,
  downward = 2,
  upward = 3,
  toward_zero = 4,
};

enum class X87PrecisionModeV1 : std::uint8_t {
  unknown = 0,
  single_24 = 1,
  reserved = 2,
  double_53 = 3,
  extended_64 = 4,
};

// Read-only description of the current thread's floating-point environment.
// Exception status flags are deliberately absent from the legality decision:
// only rounding, exception masks, and subnormal handling constrain the
// explicit-gemm-f32-v1 semantic profile.
struct FpEnvironmentReportV1 {
  std::uint32_t version = kFpEnvironmentVersionV1;
  FpEnvironmentBackendV1 backend =
      FpEnvironmentBackendV1::unavailable;
  bool discovery_complete = false;
  bool mxcsr_known = false;
  bool control_word_known = false;
  std::uint32_t raw_mxcsr = 0;
  std::uint32_t raw_control_word = 0;
  FpRoundingModeV1 mxcsr_rounding = FpRoundingModeV1::unknown;
  FpRoundingModeV1 control_word_rounding = FpRoundingModeV1::unknown;
  X87PrecisionModeV1 x87_precision = X87PrecisionModeV1::unknown;
  bool mxcsr_exceptions_masked = false;
  bool control_word_exceptions_masked = false;
  bool control_word_denormals_preserved = false;
  bool flush_to_zero = false;
  bool denormals_are_zero = false;
  bool explicit_gemm_f32_v1_compatible = false;
};

// Pure decoder used by physical and synthetic tests. The status bits in
// MXCSR[5:0] never affect compatibility.
constexpr FpEnvironmentReportV1 decode_linux_x86_fp_environment_v1(
    std::uint32_t mxcsr, std::uint16_t x87_control_word) noexcept {
  FpEnvironmentReportV1 result;
  result.backend = FpEnvironmentBackendV1::linux_x86_64;
  result.discovery_complete = true;
  result.mxcsr_known = true;
  result.control_word_known = true;
  result.raw_mxcsr = mxcsr;
  result.raw_control_word = x87_control_word;

  switch ((mxcsr >> 13U) & 0x3U) {
    case 0:
      result.mxcsr_rounding = FpRoundingModeV1::nearest_even;
      break;
    case 1:
      result.mxcsr_rounding = FpRoundingModeV1::downward;
      break;
    case 2:
      result.mxcsr_rounding = FpRoundingModeV1::upward;
      break;
    case 3:
      result.mxcsr_rounding = FpRoundingModeV1::toward_zero;
      break;
  }
  switch ((x87_control_word >> 10U) & 0x3U) {
    case 0:
      result.control_word_rounding = FpRoundingModeV1::nearest_even;
      break;
    case 1:
      result.control_word_rounding = FpRoundingModeV1::downward;
      break;
    case 2:
      result.control_word_rounding = FpRoundingModeV1::upward;
      break;
    case 3:
      result.control_word_rounding = FpRoundingModeV1::toward_zero;
      break;
  }
  switch ((x87_control_word >> 8U) & 0x3U) {
    case 0:
      result.x87_precision = X87PrecisionModeV1::single_24;
      break;
    case 1:
      result.x87_precision = X87PrecisionModeV1::reserved;
      break;
    case 2:
      result.x87_precision = X87PrecisionModeV1::double_53;
      break;
    case 3:
      result.x87_precision = X87PrecisionModeV1::extended_64;
      break;
  }

  result.mxcsr_exceptions_masked = (mxcsr & 0x1F80U) == 0x1F80U;
  result.control_word_exceptions_masked =
      (x87_control_word & 0x003FU) == 0x003FU;
  result.control_word_denormals_preserved = true;
  result.flush_to_zero = (mxcsr & (1U << 15U)) != 0;
  result.denormals_are_zero = (mxcsr & (1U << 6U)) != 0;
  result.explicit_gemm_f32_v1_compatible =
      result.mxcsr_rounding == FpRoundingModeV1::nearest_even &&
      result.control_word_rounding == FpRoundingModeV1::nearest_even &&
      result.mxcsr_exceptions_masked &&
      result.control_word_exceptions_masked &&
      result.control_word_denormals_preserved && !result.flush_to_zero &&
      !result.denormals_are_zero;
  return result;
}

FpEnvironmentReportV1 inspect_current_fp_environment_v1() noexcept;

// Compare only control state. MXCSR exception status flags are excluded from
// the equality decision, just as they are from execution legality.
bool fp_environment_control_state_equal_v1(
    const FpEnvironmentReportV1 &lhs,
    const FpEnvironmentReportV1 &rhs) noexcept;

// Restore a previously authenticated control snapshot after an opaque
// provider call. This is preservation of caller state, not normalization to a
// compiler-chosen environment. Unknown backends fail closed.
bool restore_fp_environment_control_state_v1(
    const FpEnvironmentReportV1 &snapshot) noexcept;

const char *fp_environment_rejection_reason_v1(
    const FpEnvironmentReportV1 &report) noexcept;

std::string_view to_string(FpEnvironmentBackendV1 backend) noexcept;
std::string_view to_string(FpRoundingModeV1 mode) noexcept;

}  // namespace matcore::mdslc::platform

#endif
