#include "MatcoreCpuGemmCandidate.h"
#include "MatcoreCpuReassociateGemmCandidate.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

int main(int argc, char **argv) {
  namespace candidate = matcore::mdslc::cpu_candidate;
  bool asan = false;
  auto schedule = candidate::StrictGemmScheduleV1::ScalarMNK;
  bool valid = argc >= 3 && argc <= 7 && std::string(argv[1]) == "--output";
  bool has_schedule = false;
  bool has_target = false;
  bool has_isa = false;
  auto isa = candidate::StrictCpuIsaV1::Baseline;
  auto target = candidate::CpuTargetV1::LinuxX86_64;
  bool reassociate = false;
  for (int i = 3; i < argc; ++i) {
    const std::string option(argv[i]);
    if (option == "--asan" && !asan)
      asan = true;
    else if (option == "--schedule=row-contiguous" && !has_schedule) {
      has_schedule = true;
      schedule = candidate::StrictGemmScheduleV1::RowContiguousMKN;
    } else if (option == "--candidate=reassociate-register" && !reassociate) {
      reassociate = true;
    } else if (option == "--target=linux-x86_64" && !has_target) {
      has_target = true;
      target = candidate::CpuTargetV1::LinuxX86_64;
    } else if (option == "--target=linux-aarch64" && !has_target) {
      has_target = true;
      target = candidate::CpuTargetV1::LinuxAArch64;
    } else if (option.starts_with("--isa=") && !has_isa) {
      has_isa = true;
      if (option == "--isa=baseline") isa = candidate::StrictCpuIsaV1::Baseline;
      else if (option == "--isa=avx") isa = candidate::StrictCpuIsaV1::AVX;
      else if (option == "--isa=avx2") isa = candidate::StrictCpuIsaV1::AVX2;
      else if (option == "--isa=avx512f") isa = candidate::StrictCpuIsaV1::AVX512F;
      else valid = false;
    } else
      valid = false;
  }
  if (reassociate && has_schedule) valid = false;
  if (reassociate && target != candidate::CpuTargetV1::LinuxX86_64) valid = false;
  if (reassociate && has_isa) valid = false;
  if (isa != candidate::StrictCpuIsaV1::Baseline &&
      (!has_schedule || target != candidate::CpuTargetV1::LinuxX86_64)) valid = false;
  if (!valid) {
    llvm::errs()
        << "private built-in candidate generator: --output FILE [--asan] "
           "[--schedule=row-contiguous | --candidate=reassociate-register] "
           "[--target=linux-x86_64 | --target=linux-aarch64]\n"
           "[--isa=baseline | --isa=avx | --isa=avx2 | --isa=avx512f]\n"
           "Nonbaseline ISA requires x86 row-contiguous strict, not reassociate.\n"
           "AArch64 is a strict-only target; target selection is not runtime legality.\n"
           "No source/MLIR input is accepted. This does not admit a program.\n";
    return 2;
  }
  mlir::MLIRContext context;
  auto artifact = reassociate ? candidate::issueReassociateGemmArtifactV1(context, asan)
                             : candidate::issueStrictGemmArtifactV1(context, asan, schedule, target, isa);
  if (!artifact) {
    llvm::errs() << artifact.error << '\n';
    return 1;
  }
  auto write = [&](const std::string &path, const std::string &contents) {
    std::error_code error;
    llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
    if (error) {
      llvm::errs() << error.message() << '\n';
      return false;
    }
    stream << contents;
    stream.close();
    return !stream.has_error();
  };
  const std::string path(argv[2]);
  return !(write(path, artifact.llvm_ir) &&
           write(path + ".manifest", artifact.manifest) &&
           write(path + ".semantic.mlir", artifact.semantic_ir) &&
           write(path + ".structured.mlir", artifact.structured_ir) &&
           write(path + ".bufferized.mlir", artifact.bufferized_ir) &&
           write(path + ".scheduled.mlir", artifact.scheduled_ir) &&
           write(path + ".transform.mlir", artifact.transform_ir));
}
