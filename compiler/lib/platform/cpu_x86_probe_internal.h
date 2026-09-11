#pragma once

#include <cstdint>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#endif

namespace matcore::mdslc::platform::x86_probe_internal {
// Shared low-level mechanics, not an execution-availability certificate.
struct CpuidResult {
  std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};
inline bool read_cpuid(std::uint32_t leaf, std::uint32_t subleaf,
                       CpuidResult *output) noexcept {
  if (output == nullptr) return false;
#if defined(_MSC_VER) && defined(_M_X64)
  int maximum[4]{};
  __cpuidex(maximum, 0, 0);
  if (leaf > static_cast<std::uint32_t>(maximum[0])) return false;
  int registers[4]{};
  __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
  *output = {static_cast<std::uint32_t>(registers[0]),
             static_cast<std::uint32_t>(registers[1]),
             static_cast<std::uint32_t>(registers[2]),
             static_cast<std::uint32_t>(registers[3])};
  return true;
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  const unsigned int maximum = __get_cpuid_max(0, nullptr);
  if (leaf > maximum) return false;
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  *output = {eax, ebx, ecx, edx};
  return true;
#else
  (void)leaf;
  (void)subleaf;
  return false;
#endif
}
// The caller MUST establish hardware XSAVE and OSXSAVE before this read.
inline bool read_xcr0(std::uint64_t *output) noexcept {
  if (output == nullptr) return false;
#if defined(_MSC_VER) && defined(_M_X64)
  *output = _xgetbv(0);
  return true;
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  std::uint32_t eax = 0, edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  *output = (static_cast<std::uint64_t>(edx) << 32U) | eax;
  return true;
#else
  return false;
#endif
}
} // namespace matcore::mdslc::platform::x86_probe_internal
