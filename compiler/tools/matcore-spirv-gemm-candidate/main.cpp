// Private inspection tool. No source candidate, artifact import or dispatch.
#include "MatcoreSpirvGemmCandidate.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <charconv>
#include <filesystem>
#include <string>
#include <string_view>

int main(int argc, char **argv) {
  namespace candidate = matcore::mdslc::spirv_candidate;
  if (argc != 5 || std::string_view(argv[1]) != "--shape" ||
      std::string_view(argv[3]) != "--output-directory") {
    llvm::errs() << "private SPIR-V inspection issuer: --shape M,K,N "
                    "--output-directory NEW_DIRECTORY\n"
                    "No executable IR input, source/runtime authority or dispatch.\n";
    return 2;
  }
  candidate::StaticGemmShapeV1 shape{0, 0, 0};
  auto badShape = []() {
    llvm::errs() << "shape must contain exactly three signed decimal integers M,K,N\n";
    return 2;
  };
  std::string_view text(argv[2]);
  std::int64_t *dimensions[] = {&shape.m, &shape.k, &shape.n};
  for (unsigned i = 0; i != 3; ++i) {
    auto comma = text.find(',');
    if ((i != 2) != (comma != std::string_view::npos)) return badShape();
    auto component = text.substr(0, comma);
    auto parsed = std::from_chars(component.data(), component.data() + component.size(), *dimensions[i]);
    if (parsed.ec != std::errc{} || parsed.ptr != component.data() + component.size()) return badShape();
    if (i != 2) text.remove_prefix(comma + 1);
  }
  mlir::MLIRContext context;
  auto artifact = candidate::issueStrictGemmArtifactV1(context, shape);
  std::string error;
  if (!artifact || !candidate::verifyStrictGemmArtifactV1(context, artifact, error)) {
    llvm::errs() << (artifact ? error : artifact.error) << '\n';
    return 1;
  }
  const std::filesystem::path directory(argv[4]);
  std::error_code ec;
  // Existing directories (including symlinks) are never reusable imports/cache.
  if (!std::filesystem::create_directory(directory, ec)) {
    llvm::errs() << "output directory must be new: " << ec.message() << '\n';
    return 1;
  }
  auto write = [&](const char *name, llvm::StringRef bytes) {
    std::error_code status;
    llvm::raw_fd_ostream output((directory / name).string(), status,
                               llvm::sys::fs::CD_CreateNew, llvm::sys::fs::FA_Write,
                               llvm::sys::fs::OF_None);
    if (status) { llvm::errs() << status.message() << '\n'; return false; }
    output << bytes;
    output.close();
    return !output.has_error();
  };
  auto binary = [&](const char *name, const std::vector<std::uint32_t> &words) {
    // Stable little-endian inspection artifact, independent of host byte order.
    std::string bytes;
    bytes.reserve(words.size() * 4);
    for (auto word : words)
      for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.push_back(static_cast<char>(word >> shift));
    return write(name, bytes);
  };
  if (!binary("fill.spv", artifact.fill_words) || !binary("gemm.spv", artifact.gemm_words) ||
      !write("semantic.mlir", artifact.semantic_ir) ||
      !write("structured.mlir", artifact.structured_ir) ||
      !write("bufferized.mlir", artifact.bufferized_ir) ||
      !write("outlined.mlir", artifact.outlined_ir) ||
      !write("fill.mlir", artifact.fill_spirv_ir) ||
      !write("gemm.mlir", artifact.gemm_spirv_ir) ||
      !write("manifest.txt", artifact.manifest)) return 1;
  llvm::outs() << "ISSUED_INSPECTION_ONLY " << shape.m << ',' << shape.k << ',' << shape.n
               << "; no source/runtime authority or target qualification\n";
  return 0;
}
