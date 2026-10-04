#include "MatcoreCpuFusedGemmCandidate.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

int main(int argc, char **argv) {
  const bool asan = argc == 4 && std::string(argv[3]) == "--asan";
  if ((argc != 3 && !asan) || std::string(argv[1]) != "--output" || !argv[2][0]) {
    llvm::errs() << "private built-in pair issuer: --output FILE [--asan]\n"
        "Exact 21.1.8 Linux x64 baseline, C=A*B; E=C*D, row panel 4.\n"
        "No source/MLIR/LLVM/Transform input is accepted; no source authority.\n";
    return 2;
  }
  mlir::MLIRContext context;
  auto artifact = matcore::mdslc::cpu_candidate::issueStrictFusedGemmArtifactV1(context, asan);
  if (!artifact) { llvm::errs() << artifact.error << '\n'; return 1; }
  auto write = [&](const std::string &path, const std::string &contents) {
    std::error_code error;
    llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_None);
    if (error) { llvm::errs() << error.message() << '\n'; return false; }
    out << contents; out.close(); return !out.has_error();
  };
  const std::string path(argv[2]);
  return !(write(path, artifact.llvm_ir) && write(path + ".manifest", artifact.manifest) &&
      write(path + ".semantic.mlir", artifact.semantic_ir) && write(path + ".structured.mlir", artifact.structured_ir) &&
      write(path + ".scheduled.mlir", artifact.scheduled_ir) && write(path + ".bufferized.mlir", artifact.bufferized_ir) &&
      write(path + ".transform.mlir", artifact.transform_ir));
}
