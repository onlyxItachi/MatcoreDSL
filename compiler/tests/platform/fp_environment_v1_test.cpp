#include "fp_environment_v1.h"

#include <cstdint>
#include <iostream>
#include <string_view>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
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
  std::uint32_t mxcsr() const noexcept { return mxcsr_; }

 private:
  std::uint32_t mxcsr_ = 0;
  std::uint16_t x87_control_word_ = 0;
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
  }
  expect(platform::inspect_current_fp_environment_v1()
             .explicit_gemm_f32_v1_compatible,
         "physical RAII violation restores the original environment");
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
