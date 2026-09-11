#pragma once

// Test-only raw snapshots: do not rely on libc fenv_t covering every ARM bit.
#include <cstdint>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace closed_fp_fixture {
struct Snapshot {
#if defined(__x86_64__)
  unsigned mxcsr;
  std::uint16_t control, status;
#elif defined(__aarch64__)
  std::uint64_t control, status;
#else
#error "Closed FP fixture requires a supported Linux CPU"
#endif
  bool operator==(const Snapshot &) const = default;
};
inline Snapshot snapshot() {
  Snapshot result{};
#if defined(__x86_64__)
  result.mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(result.control));
  __asm__ volatile("fnstsw %0" : "=am"(result.status));
#else
  __asm__ volatile("mrs %0, fpcr" : "=r"(result.control));
  __asm__ volatile("mrs %0, fpsr" : "=r"(result.status));
#endif
  return result;
}
inline void enableFlush() {
#if defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040U);
#else
  auto value = snapshot().control | (std::uint64_t{1} << 24);
  __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(value) : "memory");
#endif
}
} // namespace closed_fp_fixture
