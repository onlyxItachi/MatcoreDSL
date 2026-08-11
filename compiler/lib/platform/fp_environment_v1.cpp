#include "fp_environment_v1.h"

#if (defined(__x86_64__) || defined(_M_X64))
#include <xmmintrin.h>
#endif

#if defined(_WIN32) && defined(_M_X64)
#include <float.h>
#endif

namespace matcore::mdslc::platform {
namespace {

#if defined(_WIN32) && defined(_M_X64)
FpRoundingModeV1 decode_windows_rounding(unsigned int control) noexcept {
  switch (control & _MCW_RC) {
    case _RC_NEAR:
      return FpRoundingModeV1::nearest_even;
    case _RC_DOWN:
      return FpRoundingModeV1::downward;
    case _RC_UP:
      return FpRoundingModeV1::upward;
    case _RC_CHOP:
      return FpRoundingModeV1::toward_zero;
    default:
      return FpRoundingModeV1::unknown;
  }
}
#endif

}  // namespace

FpEnvironmentReportV1 inspect_current_fp_environment_v1() noexcept {
#if defined(__linux__) && defined(__x86_64__)
  const std::uint32_t mxcsr = _mm_getcsr();
  std::uint16_t x87_control_word = 0;
  __asm__ volatile("fnstcw %0" : "=m"(x87_control_word));
  return decode_linux_x86_fp_environment_v1(mxcsr, x87_control_word);
#elif defined(_WIN32) && defined(_M_X64)
  FpEnvironmentReportV1 result;
  result.backend = FpEnvironmentBackendV1::windows_x86_64;
  const std::uint32_t mxcsr = _mm_getcsr();
  unsigned int control = 0;
  if (_controlfp_s(&control, 0, 0) != 0) return result;

  // _controlfp_s is the documented Windows x64 control-state interface;
  // _mm_getcsr supplies the exact FTZ/DAZ and SIMD rounding/mask bits.
  const auto simd = decode_linux_x86_fp_environment_v1(mxcsr, 0x003FU);
  result.discovery_complete = true;
  result.mxcsr_known = true;
  result.control_word_known = true;
  result.raw_mxcsr = mxcsr;
  result.raw_control_word = control;
  result.mxcsr_rounding = simd.mxcsr_rounding;
  result.control_word_rounding = decode_windows_rounding(control);
  result.mxcsr_exceptions_masked = simd.mxcsr_exceptions_masked;
  result.control_word_exceptions_masked =
      (control & _MCW_EM) == _MCW_EM;
  result.control_word_denormals_preserved =
      (control & _MCW_DN) == _DN_SAVE;
  result.flush_to_zero = simd.flush_to_zero;
  result.denormals_are_zero = simd.denormals_are_zero;
  result.explicit_gemm_f32_v1_compatible =
      result.mxcsr_rounding == FpRoundingModeV1::nearest_even &&
      result.control_word_rounding == FpRoundingModeV1::nearest_even &&
      result.mxcsr_exceptions_masked &&
      result.control_word_exceptions_masked &&
      result.control_word_denormals_preserved && !result.flush_to_zero &&
      !result.denormals_are_zero;
  return result;
#else
  return {};
#endif
}

bool fp_environment_control_state_equal_v1(
    const FpEnvironmentReportV1 &lhs,
    const FpEnvironmentReportV1 &rhs) noexcept {
  if (lhs.version != kFpEnvironmentVersionV1 ||
      rhs.version != kFpEnvironmentVersionV1 || lhs.backend != rhs.backend ||
      !lhs.discovery_complete || !rhs.discovery_complete ||
      !lhs.mxcsr_known || !rhs.mxcsr_known || !lhs.control_word_known ||
      !rhs.control_word_known) {
    return false;
  }
  constexpr std::uint32_t kMxcsrStatusMask = 0x3FU;
  if ((lhs.raw_mxcsr & ~kMxcsrStatusMask) !=
      (rhs.raw_mxcsr & ~kMxcsrStatusMask)) {
    return false;
  }
#if defined(_WIN32) && defined(_M_X64)
  constexpr std::uint32_t kWindowsRelevantControl =
      _MCW_RC | _MCW_EM | _MCW_DN;
  return (lhs.raw_control_word & kWindowsRelevantControl) ==
         (rhs.raw_control_word & kWindowsRelevantControl);
#else
  return lhs.raw_control_word == rhs.raw_control_word;
#endif
}

bool restore_fp_environment_control_state_v1(
    const FpEnvironmentReportV1 &snapshot) noexcept {
#if defined(__linux__) && defined(__x86_64__)
  if (snapshot.version != kFpEnvironmentVersionV1 ||
      snapshot.backend != FpEnvironmentBackendV1::linux_x86_64 ||
      !snapshot.discovery_complete || !snapshot.mxcsr_known ||
      !snapshot.control_word_known) {
    return false;
  }
  _mm_setcsr(snapshot.raw_mxcsr);
  const auto control = static_cast<std::uint16_t>(snapshot.raw_control_word);
  __asm__ volatile("fnclex\n\tfldcw %0" : : "m"(control));
  return fp_environment_control_state_equal_v1(
      snapshot, inspect_current_fp_environment_v1());
#elif defined(_WIN32) && defined(_M_X64)
  if (snapshot.version != kFpEnvironmentVersionV1 ||
      snapshot.backend != FpEnvironmentBackendV1::windows_x86_64 ||
      !snapshot.discovery_complete || !snapshot.mxcsr_known ||
      !snapshot.control_word_known) {
    return false;
  }
  unsigned int restored = 0;
  constexpr unsigned int kRelevantControl = _MCW_RC | _MCW_EM | _MCW_DN;
  if (_controlfp_s(&restored, snapshot.raw_control_word,
                   kRelevantControl) != 0) {
    return false;
  }
  // _controlfp_s may synchronize overlapping SSE control fields. Restore the
  // exact authenticated MXCSR control state after the documented CRT call.
  _mm_setcsr(snapshot.raw_mxcsr);
  return fp_environment_control_state_equal_v1(
      snapshot, inspect_current_fp_environment_v1());
#else
  (void)snapshot;
  return false;
#endif
}

const char *fp_environment_rejection_reason_v1(
    const FpEnvironmentReportV1 &report) noexcept {
  if (report.version != kFpEnvironmentVersionV1)
    return "floating-point environment report version is unsupported";
  if (!report.discovery_complete || !report.mxcsr_known ||
      !report.control_word_known)
    return "floating-point environment is not completely discoverable";
  if (report.mxcsr_rounding != FpRoundingModeV1::nearest_even ||
      report.control_word_rounding != FpRoundingModeV1::nearest_even)
    return "floating-point environment must use round-to-nearest-even";
  if (!report.mxcsr_exceptions_masked ||
      !report.control_word_exceptions_masked)
    return "floating-point exceptions must be masked";
  if (!report.control_word_denormals_preserved)
    return "floating-point control state must preserve denormal operands and results";
  if (report.flush_to_zero)
    return "flush-to-zero is incompatible with gradual subnormal semantics";
  if (report.denormals_are_zero)
    return "denormals-are-zero is incompatible with gradual subnormal semantics";
  if (!report.explicit_gemm_f32_v1_compatible)
    return "floating-point environment is incompatible with explicit-gemm-f32-v1";
  return "ok";
}

std::string_view to_string(FpEnvironmentBackendV1 backend) noexcept {
  switch (backend) {
    case FpEnvironmentBackendV1::unavailable:
      return "unavailable";
    case FpEnvironmentBackendV1::linux_x86_64:
      return "linux-x86_64";
    case FpEnvironmentBackendV1::windows_x86_64:
      return "windows-x86_64";
  }
  return "invalid";
}

std::string_view to_string(FpRoundingModeV1 mode) noexcept {
  switch (mode) {
    case FpRoundingModeV1::unknown:
      return "unknown";
    case FpRoundingModeV1::nearest_even:
      return "nearest-even";
    case FpRoundingModeV1::downward:
      return "downward";
    case FpRoundingModeV1::upward:
      return "upward";
    case FpRoundingModeV1::toward_zero:
      return "toward-zero";
  }
  return "invalid";
}

}  // namespace matcore::mdslc::platform
