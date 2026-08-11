#include "fp_environment_v1.h"

#include <cstdint>
#include <iostream>
#include <string_view>

#if (defined(__linux__) && defined(__x86_64__)) || \
    (defined(_WIN32) && defined(_M_X64))
#include <xmmintrin.h>
#endif

#if defined(_WIN32) && defined(_M_X64)
#include <float.h>
#endif

namespace platform = matcore::mdslc::platform;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

#if defined(__linux__) && defined(__x86_64__)
class ScopedX86FpEnvironment {
 public:
  ScopedX86FpEnvironment() noexcept : mxcsr_(_mm_getcsr()) {
    __asm__ volatile("fnstcw %0" : "=m"(x87_control_word_));
  }
  ScopedX86FpEnvironment(const ScopedX86FpEnvironment &) = delete;
  ScopedX86FpEnvironment &operator=(const ScopedX86FpEnvironment &) = delete;
  ~ScopedX86FpEnvironment() {
    _mm_setcsr(mxcsr_);
    __asm__ volatile("fldcw %0" : : "m"(x87_control_word_));
  }

  void set_mxcsr(std::uint32_t value) noexcept { _mm_setcsr(value); }
  void set_x87_control_word(std::uint16_t value) noexcept {
    __asm__ volatile("fnclex\n\tfldcw %0" : : "m"(value));
  }
  std::uint32_t mxcsr() const noexcept { return mxcsr_; }
  std::uint16_t x87_control_word() const noexcept {
    return x87_control_word_;
  }

 private:
  std::uint32_t mxcsr_ = 0;
  std::uint16_t x87_control_word_ = 0;
};
#elif defined(_WIN32) && defined(_M_X64)
class ScopedWindowsFpEnvironment {
 public:
  explicit ScopedWindowsFpEnvironment(
      const platform::FpEnvironmentReportV1 &snapshot) noexcept
      : snapshot_(snapshot) {}
  ScopedWindowsFpEnvironment(const ScopedWindowsFpEnvironment &) = delete;
  ScopedWindowsFpEnvironment &operator=(const ScopedWindowsFpEnvironment &) =
      delete;
  ~ScopedWindowsFpEnvironment() {
    if (snapshot_.explicit_gemm_f32_v1_compatible)
      (void)platform::restore_fp_environment_control_state_v1(snapshot_);
  }

  void set_mxcsr(std::uint32_t value) noexcept { _mm_setcsr(value); }
  bool set_control(unsigned int value, unsigned int mask) noexcept {
    unsigned int ignored = 0;
    return _controlfp_s(&ignored, value, mask) == 0;
  }

 private:
  platform::FpEnvironmentReportV1 snapshot_;
};
#endif

void pure_decoder_contract() {
  constexpr std::uint32_t default_mxcsr = 0x1F80U;
  constexpr std::uint16_t default_x87 = 0x037FU;
  constexpr auto compatible =
      platform::decode_linux_x86_fp_environment_v1(default_mxcsr,
                                                    default_x87);
  static_assert(compatible.explicit_gemm_f32_v1_compatible);
  expect(compatible.discovery_complete && compatible.mxcsr_known &&
             compatible.control_word_known,
         "pure x86 decoder reports complete control state");
  expect(compatible.mxcsr_rounding ==
             platform::FpRoundingModeV1::nearest_even &&
             compatible.control_word_rounding ==
                 platform::FpRoundingModeV1::nearest_even &&
             compatible.x87_precision ==
                 platform::X87PrecisionModeV1::extended_64,
         "pure decoder authenticates default rounding and x87 precision");

  const auto status_flags = platform::decode_linux_x86_fp_environment_v1(
      default_mxcsr | 0x3FU, default_x87);
  expect(status_flags.explicit_gemm_f32_v1_compatible,
         "MXCSR exception status flags are ignored for legality");

  const auto double_precision = platform::decode_linux_x86_fp_environment_v1(
      default_mxcsr,
      static_cast<std::uint16_t>((default_x87 & ~(3U << 8U)) | (2U << 8U)));
  expect(double_precision.x87_precision ==
             platform::X87PrecisionModeV1::double_53 &&
             double_precision.explicit_gemm_f32_v1_compatible,
         "x87 precision is diagnostic-only for the consumed F32 backend contract");

  expect(!platform::decode_linux_x86_fp_environment_v1(
              default_mxcsr | (1U << 13U), default_x87)
              .explicit_gemm_f32_v1_compatible,
         "non-RNE MXCSR fails closed");
  expect(!platform::decode_linux_x86_fp_environment_v1(
              default_mxcsr, static_cast<std::uint16_t>(default_x87 |
                                                        (1U << 10U)))
              .explicit_gemm_f32_v1_compatible,
         "non-RNE x87 control fails closed");
  expect(!platform::decode_linux_x86_fp_environment_v1(
              default_mxcsr & ~(1U << 7U), default_x87)
              .explicit_gemm_f32_v1_compatible,
         "unmasked MXCSR exception fails closed");
  expect(!platform::decode_linux_x86_fp_environment_v1(
              default_mxcsr,
              static_cast<std::uint16_t>(default_x87 & ~(1U << 0U)))
              .explicit_gemm_f32_v1_compatible,
         "unmasked x87 exception fails closed");
  expect(!platform::decode_linux_x86_fp_environment_v1(
              default_mxcsr | (1U << 15U), default_x87)
              .explicit_gemm_f32_v1_compatible &&
             !platform::decode_linux_x86_fp_environment_v1(
                  default_mxcsr | (1U << 6U), default_x87)
                  .explicit_gemm_f32_v1_compatible,
         "FTZ and DAZ each fail closed");
}

void physical_environment_contract() {
  const auto initial = platform::inspect_current_fp_environment_v1();
#if defined(__linux__) && defined(__x86_64__)
  expect(initial.backend == platform::FpEnvironmentBackendV1::linux_x86_64 &&
             initial.discovery_complete,
         "physical Linux x86-64 environment is exactly discoverable");
  expect(initial.explicit_gemm_f32_v1_compatible,
         "test process starts with the supported floating-point environment");
  if (!initial.explicit_gemm_f32_v1_compatible) return;

  {
    ScopedX86FpEnvironment scope;
    scope.set_mxcsr(scope.mxcsr() | (1U << 15U));
    const auto violated = platform::inspect_current_fp_environment_v1();
    expect(!violated.explicit_gemm_f32_v1_compatible &&
               std::string_view(platform::fp_environment_rejection_reason_v1(
                   violated))
                       .find("flush-to-zero") != std::string_view::npos,
           "physical FTZ violation is detected with an actionable reason");
    expect(platform::restore_fp_environment_control_state_v1(initial) &&
               platform::fp_environment_control_state_equal_v1(
                   initial, platform::inspect_current_fp_environment_v1()),
           "opaque-provider restoration returns to the exact control state");
  }
  {
    ScopedX86FpEnvironment scope;
    scope.set_mxcsr(scope.mxcsr() | (1U << 6U));
    expect(!platform::inspect_current_fp_environment_v1()
                .explicit_gemm_f32_v1_compatible,
           "physical DAZ violation is detected");
  }
  bool all_mxcsr_rounding_modes_rejected = true;
  bool all_x87_rounding_modes_rejected = true;
  for (std::uint32_t mode = 1; mode != 4; ++mode) {
    {
      ScopedX86FpEnvironment scope;
      scope.set_mxcsr((scope.mxcsr() & ~(3U << 13U)) | (mode << 13U));
      all_mxcsr_rounding_modes_rejected &=
          !platform::inspect_current_fp_environment_v1()
               .explicit_gemm_f32_v1_compatible;
    }
    {
      ScopedX86FpEnvironment scope;
      const auto changed = static_cast<std::uint16_t>(
          (scope.x87_control_word() & ~(3U << 10U)) | (mode << 10U));
      scope.set_x87_control_word(changed);
      all_x87_rounding_modes_rejected &=
          !platform::inspect_current_fp_environment_v1()
               .explicit_gemm_f32_v1_compatible;
    }
  }
  expect(all_mxcsr_rounding_modes_rejected,
         "every non-RNE physical MXCSR rounding mode is rejected");
  expect(all_x87_rounding_modes_rejected,
         "every non-RNE physical x87 rounding mode is rejected");

  bool all_mxcsr_masks_required = true;
  bool all_x87_masks_required = true;
  for (std::uint32_t exception = 0; exception != 6; ++exception) {
    {
      ScopedX86FpEnvironment scope;
      // Clear every pending status flag before unmasking this family.
      scope.set_mxcsr((scope.mxcsr() & ~0x3FU) &
                      ~(1U << (7U + exception)));
      all_mxcsr_masks_required &=
          !platform::inspect_current_fp_environment_v1()
               .explicit_gemm_f32_v1_compatible;
    }
    {
      ScopedX86FpEnvironment scope;
      // set_x87_control_word issues fnclex before installing the mask.
      const auto unmasked = static_cast<std::uint16_t>(
          scope.x87_control_word() & ~(1U << exception));
      scope.set_x87_control_word(unmasked);
      all_x87_masks_required &=
          !platform::inspect_current_fp_environment_v1()
               .explicit_gemm_f32_v1_compatible;
    }
  }
  expect(all_mxcsr_masks_required,
         "every physical MXCSR exception-mask family is required");
  expect(all_x87_masks_required,
         "every physical x87 exception-mask family is required");
  {
    ScopedX86FpEnvironment scope;
    scope.set_mxcsr((scope.mxcsr() & ~0x3FU) | 0x3FU);
    expect(platform::inspect_current_fp_environment_v1()
               .explicit_gemm_f32_v1_compatible,
           "physical MXCSR status flags remain legality-neutral");
  }
  expect(platform::inspect_current_fp_environment_v1()
             .explicit_gemm_f32_v1_compatible,
         "physical RAII violation restores the original environment");
#elif defined(_WIN32) && defined(_M_X64)
  expect(initial.backend == platform::FpEnvironmentBackendV1::windows_x86_64 &&
             initial.discovery_complete && initial.mxcsr_known &&
             initial.control_word_known,
         "physical Windows x86-64 environment is exactly discoverable");
  expect(initial.explicit_gemm_f32_v1_compatible,
         "Windows test process starts with the supported floating-point environment");
  if (!initial.explicit_gemm_f32_v1_compatible) return;

  {
    ScopedWindowsFpEnvironment scope(initial);
    scope.set_mxcsr(initial.raw_mxcsr | (1U << 15U));
    const auto violated = platform::inspect_current_fp_environment_v1();
    expect(!violated.explicit_gemm_f32_v1_compatible &&
               std::string_view(platform::fp_environment_rejection_reason_v1(
                   violated))
                       .find("flush-to-zero") != std::string_view::npos,
           "physical Windows FTZ violation is detected with an actionable reason");
    expect(platform::restore_fp_environment_control_state_v1(initial) &&
               platform::fp_environment_control_state_equal_v1(
                   initial, platform::inspect_current_fp_environment_v1()),
           "Windows restoration returns to the exact control state");
  }
  {
    ScopedWindowsFpEnvironment scope(initial);
    scope.set_mxcsr(initial.raw_mxcsr | (1U << 6U));
    expect(!platform::inspect_current_fp_environment_v1()
                .explicit_gemm_f32_v1_compatible,
           "physical Windows DAZ violation is detected");
  }
  {
    ScopedWindowsFpEnvironment scope(initial);
    expect(scope.set_control(_RC_DOWN, _MCW_RC) &&
               !platform::inspect_current_fp_environment_v1()
                    .explicit_gemm_f32_v1_compatible,
           "physical Windows non-RNE control state is rejected");
  }
  {
    ScopedWindowsFpEnvironment scope(initial);
    (void)_clearfp();
    scope.set_mxcsr(initial.raw_mxcsr & ~0x3FU);
    expect(scope.set_control(initial.raw_control_word & ~_EM_INVALID,
                             _MCW_EM) &&
               !platform::inspect_current_fp_environment_v1()
                    .explicit_gemm_f32_v1_compatible,
           "physical Windows unmasked exception state is rejected");
  }
  {
    ScopedWindowsFpEnvironment scope(initial);
    expect(scope.set_control(_DN_FLUSH, _MCW_DN) &&
               !platform::inspect_current_fp_environment_v1()
                    .explicit_gemm_f32_v1_compatible,
           "physical Windows denormal-flush control state is rejected");
  }
  {
    ScopedWindowsFpEnvironment scope(initial);
    scope.set_mxcsr((initial.raw_mxcsr & ~0x3FU) | 0x3FU);
    expect(platform::inspect_current_fp_environment_v1()
               .explicit_gemm_f32_v1_compatible,
           "physical Windows MXCSR status flags remain legality-neutral");
  }
  expect(platform::fp_environment_control_state_equal_v1(
             initial, platform::inspect_current_fp_environment_v1()),
         "physical Windows RAII violations restore the original environment");
#else
  expect(!initial.discovery_complete &&
             !initial.explicit_gemm_f32_v1_compatible,
         "unsupported physical architectures fail closed");
#endif
}

}  // namespace

int main() {
  pure_decoder_contract();
  physical_environment_contract();
  if (failures != 0) return 1;
  std::cout << "floating-point environment v1 tests PASS\n";
  return 0;
}
