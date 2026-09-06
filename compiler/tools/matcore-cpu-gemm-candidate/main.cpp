#include "MatcoreCpuGemmCandidate.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4 || std::string(argv[1]) != "--output" ||
      (argc == 4 && std::string(argv[3]) != "--asan")) {
    llvm::errs()
        << "private built-in candidate generator: --output FILE [--asan]\n"
           "No source/MLIR input is accepted. This does not admit a program.\n";
    return 2;
  }
  mlir::MLIRContext context;
  auto artifact = matcore::mdslc::cpu_candidate::issueStrictGemmArtifactV1(
      context, argc == 4);
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
           write(path + ".bufferized.mlir", artifact.bufferized_ir));
}
