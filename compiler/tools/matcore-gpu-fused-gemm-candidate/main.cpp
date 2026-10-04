#include "MatcoreGpuFusedGemmCandidate.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
int main(int argc,char **argv) {
  namespace candidate=matcore::mdslc::gpu_candidate;
  if (argc!=5 || std::string(argv[1])!="--output-prefix" ||
      std::string(argv[3])!="--target" ||
      (std::string(argv[4])!="nvvm-sm89" && std::string(argv[4])!="rocdl-gfx1150")) {
    llvm::errs()<<"private built-in strict GPU pair issuer: --output-prefix PATH --target "
      "nvvm-sm89|rocdl-gfx1150\nNo source/MLIR/LLVM/Transform input or execution authority is accepted.\n";
    return 2;
  }
  mlir::MLIRContext context; context.disableMultithreading();
  auto artifact=candidate::issueStrictGpuFusedGemmArtifactV1(context,
      std::string(argv[4])=="nvvm-sm89" ? candidate::TargetV1::NvvmSm89 : candidate::TargetV1::RocdlGfx1150);
  if (!artifact) { llvm::errs()<<artifact.error<<'\n'; return 1; }
  auto write=[&](const char *suffix,const std::string &text) {
    std::error_code error; llvm::raw_fd_ostream stream(std::string(argv[2])+suffix,error,llvm::sys::fs::OF_None);
    if (error) { llvm::errs()<<error.message()<<'\n'; return false; }
    stream<<text; stream.close(); return !stream.has_error();
  };
  return !(write(".pair.ll",artifact.device_llvm_ir) && write(".semantic.mlir",artifact.semantic_ir) &&
           write(".structured.mlir",artifact.structured_ir) && write(".scheduled.mlir",artifact.scheduled_ir) &&
           write(".bufferized.mlir",artifact.bufferized_ir) && write(".outlined.mlir",artifact.outlined_ir) &&
           write(".manifest",artifact.manifest));
}
