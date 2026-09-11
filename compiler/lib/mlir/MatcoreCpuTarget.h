#ifndef MATCORE_MDSLC_MLIR_CPU_TARGET_H
#define MATCORE_MDSLC_MLIR_CPU_TARGET_H

#include "../platform/closed_cpu_isa_v1.h"

namespace matcore::mdslc::cpu_candidate {

// Compiler-private baseline target realization, not source semantics or runtime
// capability proof. No arbitrary triple/features or imported IR are admitted.
enum class CpuTargetV1 { LinuxX86_64, LinuxAArch64 };
using StrictCpuIsaV1 = platform::StrictCpuIsaV1;

// Closed LLVM 21.1.8 realization facts, never arithmetic permissions. In
// particular AVX512F implies FMA/F16C availability but strict IR still contains
// only unflagged, separate f32 multiply/add. No caller-supplied feature string.
constexpr const char *strictCpuFeaturesV1(StrictCpuIsaV1 isa) noexcept {
  switch (isa) {
  case StrictCpuIsaV1::Baseline: return "";
  case StrictCpuIsaV1::AVX: return "+avx,-avx2,-fma,-f16c,-avx512f";
  case StrictCpuIsaV1::AVX2: return "+avx,+avx2,-fma,-f16c,-avx512f";
  case StrictCpuIsaV1::AVX512F:
    return "+avx,+avx2,+fma,+f16c,+avx512f,+evex512,-avx512bw,-avx512dq,-avx512vl";
  }
  return nullptr;
}
constexpr const char *strictCpuVectorWidthV1(StrictCpuIsaV1 isa) noexcept {
  return isa == StrictCpuIsaV1::AVX512F ? "512" : "256";
}
constexpr const char *strictCpuLeafSymbolV1(StrictCpuIsaV1 isa) noexcept {
  switch (isa) {
  case StrictCpuIsaV1::Baseline: return "__matcore_strict_gemm_f32_v1";
  case StrictCpuIsaV1::AVX: return "__matcore_strict_gemm_f32_avx_v1";
  case StrictCpuIsaV1::AVX2: return "__matcore_strict_gemm_f32_avx2_v1";
  case StrictCpuIsaV1::AVX512F: return "__matcore_strict_gemm_f32_avx512f_v1";
  }
  return nullptr;
}

constexpr const char *cpuTargetTripleV1(CpuTargetV1 target) noexcept {
  switch (target) {
  case CpuTargetV1::LinuxX86_64: return "x86_64-pc-linux-gnu";
  case CpuTargetV1::LinuxAArch64: return "aarch64-unknown-linux-gnu";
  }
  return nullptr;
}

} // namespace matcore::mdslc::cpu_candidate
#endif
