#include "closed_fp_environment_v1.h"

#include <atomic>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>
#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#endif

namespace p = matcore::mdslc::platform;
static_assert(!std::is_copy_constructible_v<p::ClosedFpEnvironmentV1>);
static_assert(!std::is_move_constructible_v<p::ClosedFpEnvironmentV1>);
static_assert(p::closedAarch64FpControlCompatibleV1(0));
static_assert(!p::closedAarch64FpControlCompatibleV1(UINT64_MAX));

namespace {
struct Snapshot {
  std::uint64_t control = 0, status = 0, other_control = 0, other_status = 0;
  bool operator==(const Snapshot &) const = default;
};
Snapshot snapshot() {
  Snapshot result;
#if defined(__linux__) && defined(__x86_64__)
  const auto mxcsr = _mm_getcsr();
  std::uint16_t control = 0, status = 0;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  __asm__ volatile("fnstsw %0" : "=am"(status));
  result = {mxcsr & ~0x3FU, mxcsr & 0x3FU, control, status};
#elif defined(__linux__) && defined(__aarch64__)
  __asm__ volatile("mrs %0, fpcr" : "=r"(result.control) : : "memory");
  __asm__ volatile("mrs %0, fpsr" : "=r"(result.status) : : "memory");
#endif
  return result;
}

void hostileEnvironment() {
  (void)std::fesetround(FE_DOWNWARD);
#if defined(__linux__) && defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040U | 0x21U); // FTZ, DAZ, sticky flags.
  (void)std::feraiseexcept(FE_INVALID | FE_INEXACT);
#elif defined(__linux__) && defined(__aarch64__)
  // DN, FZ, RMode=downward; sticky IOC, IXC and SIMD saturation QC.
  const std::uint64_t fpcr = (UINT64_C(1) << 25) | (UINT64_C(1) << 24) |
                             (UINT64_C(2) << 22);
  const std::uint64_t fpsr = (UINT64_C(1) << 27) | 0x11;
  __asm__ volatile("msr fpcr, %0\n\tmsr fpsr, %1\n\tisb"
                   : : "r"(fpcr), "r"(fpsr) : "memory");
#endif
}

bool arithmetic() {
  volatile float tiny = std::numeric_limits<float>::denorm_min();
  volatile float one = 1.0F;
  volatile float product = tiny * one;
  volatile float subnormal = 0.0F + product;
  volatile float x = 0x1.000002p0F, y = 0x1.fffffcp-1F;
  volatile float rounded = x * y;
  volatile float separate = -1.0F + rounded;
  volatile float infinity = std::numeric_limits<float>::infinity();
  volatile float zero = 0;
  const float invalid = infinity * zero;
  return std::bit_cast<std::uint32_t>(float(subnormal)) == 1 &&
         float(separate) == 0.0F && std::fma(float(x), float(y), -1.0F) != 0.0F &&
         std::isnan(invalid);
}

bool threadCase() {
  const auto entry = snapshot();
  bool ok = true;
  {
    p::ClosedFpEnvironmentV1 outer;
    ok &= outer.valid() && outer.controlsUnchanged();
    hostileEnvironment();
    const auto hostile = snapshot();
    {
      p::ClosedFpEnvironmentV1 inner;
      ok &= inner.valid() && inner.controlsUnchanged();
      ok &= std::fegetround() == FE_TONEAREST && arithmetic();
      // Sticky status changes are legal during computation.
      ok &= inner.controlsUnchanged();
      (void)std::fesetround(FE_UPWARD);
      ok &= !inner.controlsUnchanged();
      inner.restore();
      ok &= !inner.valid() && snapshot() == hostile;
      inner.restore(); // Idempotence must not overwrite later caller state.
      ok &= snapshot() == hostile;
    }
    ok &= snapshot() == hostile;
  }
  ok &= snapshot() == entry;
  return ok;
}
}

int main() {
  unsigned checks = 0, failures = 0;
  const auto check = [&](bool ok) { ++checks; failures += !ok; };
  for (unsigned bit = 0; bit < 64; ++bit)
    check(!p::closedAarch64FpControlCompatibleV1(UINT64_C(1) << bit));
  if (!p::closedFpPlatformSupportedV1()) {
    p::ClosedFpEnvironmentV1 scope;
    check(!scope.valid() && !scope.controlsUnchanged());
    std::printf("closed FP unsupported-platform checks: %u, failures: %u\n", checks, failures);
    return failures != 0;
  }
  check(threadCase());
  std::atomic<unsigned> thread_failures{0};
  std::vector<std::thread> workers;
  for (unsigned i = 0; i < 4; ++i)
    workers.emplace_back([&] {
      for (unsigned repeat = 0; repeat < 32; ++repeat)
        if (!threadCase()) ++thread_failures;
    });
  for (auto &worker : workers) worker.join();
  check(thread_failures == 0);
  std::printf("closed FP scope: %u checks plus 128 thread cases, %u failures\n", checks, failures);
  return failures != 0;
}
