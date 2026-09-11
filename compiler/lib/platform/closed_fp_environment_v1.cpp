#include "closed_fp_environment_v1.h"

#include <exception>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(__FAST_MATH__)
#error "closed FP environment scope requires precise floating-point compilation"
#endif
#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#endif

namespace matcore::mdslc::platform {
namespace {
#if defined(__linux__) && defined(__aarch64__)
std::uint64_t readFpcr() noexcept {
  std::uint64_t value = 0;
  __asm__ volatile("mrs %0, fpcr" : "=r"(value) : : "memory");
  return value;
}
std::uint64_t readFpsr() noexcept {
  std::uint64_t value = 0;
  __asm__ volatile("mrs %0, fpsr" : "=r"(value) : : "memory");
  return value;
}
void writeArmEnvironment(std::uint64_t fpcr, std::uint64_t fpsr) noexcept {
  // Restore status without raising pending exceptions. ISB also orders control
  // changes before subsequent floating-point instructions on this thread.
  __asm__ volatile("msr fpcr, %0\n\tmsr fpsr, %1\n\tisb"
                   : : "r"(fpcr), "r"(fpsr) : "memory");
}
#endif
}

ClosedFpEnvironmentV1::ClosedFpEnvironmentV1() noexcept {
#if defined(__linux__) && defined(__x86_64__)
  if (std::fegetenv(&saved_) != 0) return;
  saved_mxcsr_ = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(saved_control_));
  __asm__ volatile("fnstsw %0" : "=am"(saved_status_));
  captured_ = true;
  if (std::fesetenv(FE_DFL_ENV) != 0) { restore(); return; }
  expected_mxcsr_ = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(expected_control_));
  valid_ = (expected_mxcsr_ & 0xFFC0U) == 0x1F80U &&
           (expected_control_ & 0x0C3FU) == 0x003FU;
  if (!valid_) restore();
#elif defined(__linux__) && defined(__aarch64__)
  saved_fpcr_ = readFpcr();
  saved_fpsr_ = readFpsr();
  captured_ = true;
  // Use the complete architectural FPCR/FPSR, not libc's potentially narrower
  // mask of recognized exception/control bits. This baseline uses neither SME
  // streaming state nor non-f32 alternative arithmetic formats.
  writeArmEnvironment(0, 0);
  valid_ = closedAarch64FpControlCompatibleV1(readFpcr()) && readFpsr() == 0;
  if (!valid_) restore();
#endif
}

ClosedFpEnvironmentV1::~ClosedFpEnvironmentV1() { restore(); }

bool ClosedFpEnvironmentV1::controlsUnchanged() const noexcept {
  if (!valid_) return false;
#if defined(__linux__) && defined(__x86_64__)
  std::uint16_t control = 0;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  return (_mm_getcsr() & ~0x3FU) == (expected_mxcsr_ & ~0x3FU) &&
         control == expected_control_;
#elif defined(__linux__) && defined(__aarch64__)
  return closedAarch64FpControlCompatibleV1(readFpcr());
#else
  return false;
#endif
}

void ClosedFpEnvironmentV1::restore() noexcept {
  if (!captured_) return;
#if defined(__linux__) && defined(__x86_64__)
  if (std::fesetenv(&saved_) != 0) std::terminate();
  std::uint16_t control = 0, status = 0;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  __asm__ volatile("fnstsw %0" : "=am"(status));
  if (_mm_getcsr() != saved_mxcsr_ || control != saved_control_ ||
      status != saved_status_)
    std::terminate();
#elif defined(__linux__) && defined(__aarch64__)
  writeArmEnvironment(saved_fpcr_, saved_fpsr_);
  if (readFpcr() != saved_fpcr_ || readFpsr() != saved_fpsr_) std::terminate();
#endif
  captured_ = false;
  valid_ = false;
}

} // namespace matcore::mdslc::platform
