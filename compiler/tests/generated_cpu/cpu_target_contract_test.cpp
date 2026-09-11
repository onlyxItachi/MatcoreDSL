#include "MatcoreCpuGemmCandidate.h"
#include <cstdio>
#include <string>

namespace cc = matcore::mdslc::cpu_candidate;
static_assert(cc::cpuTargetTripleV1(static_cast<cc::CpuTargetV1>(99)) == nullptr);

int main() {
  unsigned checks = 0, failures = 0;
  const auto check = [&](bool ok) { ++checks; failures += !ok; };
  mlir::MLIRContext context;
  const auto implicit = cc::issueStrictGemmArtifactV1(context, false);
  const auto x86 = cc::issueStrictGemmArtifactV1(
      context, false, cc::StrictGemmScheduleV1::ScalarMNK,
      cc::CpuTargetV1::LinuxX86_64);
  check(implicit && x86 && implicit.llvm_ir == x86.llvm_ir &&
        implicit.manifest == x86.manifest);
  for (const auto schedule : {cc::StrictGemmScheduleV1::ScalarMNK,
                             cc::StrictGemmScheduleV1::RowContiguousMKN}) {
    const auto native = cc::issueStrictGemmArtifactV1(
        context, false, schedule, cc::CpuTargetV1::LinuxX86_64);
    const auto arm = cc::issueStrictGemmArtifactV1(
        context, false, schedule, cc::CpuTargetV1::LinuxAArch64);
    check(native && arm);
    check(native.semantic_ir == arm.semantic_ir &&
          native.structured_ir == arm.structured_ir &&
          native.bufferized_ir == arm.bufferized_ir &&
          native.scheduled_ir == arm.scheduled_ir &&
          native.transform_ir == arm.transform_ir);
    check(arm.llvm_ir.find("target triple = \"aarch64-unknown-linux-gnu\"") != std::string::npos);
    check(arm.manifest.find("\ntarget=aarch64-unknown-linux-gnu\n") != std::string::npos);
    check(arm.llvm_ir.find("llvm.fma") == std::string::npos &&
          arm.llvm_ir.find("fmul float") != std::string::npos &&
          arm.llvm_ir.find("fadd float") != std::string::npos);
    const auto sanitized = cc::issueStrictGemmArtifactV1(
        context, true, schedule, cc::CpuTargetV1::LinuxAArch64);
    check(sanitized && sanitized.llvm_ir.find("sanitize_address") != std::string::npos);
  }
  const auto invalid = cc::issueStrictGemmArtifactV1(
      context, false, cc::StrictGemmScheduleV1::ScalarMNK,
      static_cast<cc::CpuTargetV1>(99));
  check(!invalid && invalid.semantic_ir.empty() && invalid.llvm_ir.empty() &&
        invalid.error.find("unknown strict GEMM CPU target") != std::string::npos);
  std::printf("closed CPU target contract: %u checks, %u failures\n", checks, failures);
  return failures != 0;
}
