#ifndef MATCORE_MDSLC_MLIR_CPU_TARGET_H
#define MATCORE_MDSLC_MLIR_CPU_TARGET_H

namespace matcore::mdslc::cpu_candidate {

// Compiler-private baseline target realization, not source semantics or runtime
// capability proof. No arbitrary triple/features or imported IR are admitted.
enum class CpuTargetV1 { LinuxX86_64, LinuxAArch64 };

constexpr const char *cpuTargetTripleV1(CpuTargetV1 target) noexcept {
  switch (target) {
  case CpuTargetV1::LinuxX86_64: return "x86_64-pc-linux-gnu";
  case CpuTargetV1::LinuxAArch64: return "aarch64-unknown-linux-gnu";
  }
  return nullptr;
}

} // namespace matcore::mdslc::cpu_candidate
#endif
