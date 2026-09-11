#pragma once

#include <cstdint>

namespace matcore::mdslc::platform {
// Separate private vocabulary. The public/versioned CpuFeatureV2 has no AVX bit
// and is intentionally not extended or reinterpreted by this closed path.
enum class StrictCpuIsaV1 { Baseline, AVX, AVX2, AVX512F };
constexpr const char *strictCpuIsaNameV1(StrictCpuIsaV1 isa) noexcept {
  switch (isa) {
  case StrictCpuIsaV1::Baseline: return "baseline";
  case StrictCpuIsaV1::AVX: return "avx";
  case StrictCpuIsaV1::AVX2: return "avx2";
  case StrictCpuIsaV1::AVX512F: return "avx512f";
  }
  return nullptr;
}
struct ClosedCpuIsaFactsV1 {
  std::uint32_t version = 1;
  bool native_linux_x86 = false;
  bool leaf1_known = false, leaf7_known = false, xcr0_known = false;
  std::uint32_t leaf1_ecx = 0, leaf1_edx = 0, leaf7_ebx = 0;
  std::uint64_t xcr0 = 0;
};
// LLVM 21.1.8 +avx implies the SSE4.2 predecessor chain. +avx512f
// additionally implies AVX2, FMA and F16C. These are machine availability
// requirements, NEVER permission for fused or f16 mathematical operations.
inline constexpr std::uint32_t kClosedAvxEcxV1 =
    (1U << 0) | (1U << 9) | (1U << 19) | (1U << 20) |
    (1U << 26) | (1U << 27) | (1U << 28);
inline constexpr std::uint32_t kClosedX86EdxV1 =
    (1U << 0) | (1U << 8) | (1U << 15) | (1U << 23) |
    (1U << 24) | (1U << 25) | (1U << 26);
inline constexpr std::uint32_t kClosedAvx512EcxV1 = (1U << 12) | (1U << 29);
constexpr bool closedCpuIsaSupportedV1(StrictCpuIsaV1 isa,
                                      const ClosedCpuIsaFactsV1 &facts) noexcept {
  if (facts.version != 1 || !facts.native_linux_x86 || !facts.leaf1_known ||
      !facts.xcr0_known || (facts.leaf1_ecx & kClosedAvxEcxV1) != kClosedAvxEcxV1 ||
      (facts.leaf1_edx & kClosedX86EdxV1) != kClosedX86EdxV1 ||
      (facts.xcr0 & 0x6U) != 0x6U)
    return false;
  switch (isa) {
  case StrictCpuIsaV1::AVX: return true;
  case StrictCpuIsaV1::AVX2:
    return facts.leaf7_known && (facts.leaf7_ebx & (1U << 5)) != 0;
  case StrictCpuIsaV1::AVX512F:
    return facts.leaf7_known &&
           (facts.leaf7_ebx & ((1U << 5) | (1U << 16))) == ((1U << 5) | (1U << 16)) &&
           (facts.leaf1_ecx & kClosedAvx512EcxV1) == kClosedAvx512EcxV1 &&
           (facts.xcr0 & 0xe6U) == 0xe6U;
  case StrictCpuIsaV1::Baseline: return false; // not an ISA-specific gate
  }
  return false;
}
ClosedCpuIsaFactsV1 discoverClosedCpuIsaFactsV1() noexcept;
} // namespace matcore::mdslc::platform
