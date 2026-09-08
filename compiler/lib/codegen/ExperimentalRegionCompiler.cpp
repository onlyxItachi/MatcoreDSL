#include "ExperimentalRegionCompiler.h"
#include "AuthenticatedHostThunk.h"
#include "FrozenHostCodegen.h"
#include "../frontend/ClosedRegionAdmissionInternal.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <algorithm>

namespace matcore::mdslc::codegen {
namespace {
#include "ExperimentalRegionRuntimeHeader.inc"
namespace fs = std::filesystem;
namespace host = frontend::closed_region_host;

bool ownedHeader(const fs::path &path, const char *expected, std::string &error) {
  auto contents = llvm::MemoryBuffer::getFile(path.string());
  if (!contents || (*contents)->getBuffer() != expected) {
    error = "compiler installation header differs from its compiled contract: " + path.string();
    return false;
  }
  return true;
}
} // namespace

bool ExperimentalLLVMCompilation::inputsUnchanged(std::string &error) const {
  if (!host || !helper) {
    error = "compiled region is missing its immutable compiler-input closure";
    return false;
  }
  return host->unchanged(error) && helper->unchanged(error);
}

static ExperimentalLLVMCompilationResult compileRegion(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const ExperimentalCompilerInputs &inputs, ClosedCpuPolicy policy,
    const std::function<void()> &after_staging) {
  ExperimentalLLVMCompilationResult result;
  if (std::count_if(inputs.symbol_artifacts.begin(), inputs.symbol_artifacts.end(),
          [](const auto &artifact) { return artifact.owner == SymbolArtifactOwner::PrivateCandidates; }) != 1) {
    result.error = "region execution compilation requires its isolated private candidate DSO";
    return result;
  }
  auto emitted = emitExperimentalRegion(evidence, policy);
  if (!emitted) { result.error = emitted.error; return result; }
  auto original = frontend::detail::ClosedRegionCompilationAccess::host(evidence);
  if (!original || !original->unchanged(result.error)) return result;
  std::error_code ec;
  if (!inputs.staging_directory.is_absolute() ||
      !fs::is_directory(inputs.staging_directory, ec) || ec ||
      !inputs.public_include_directory.is_absolute() ||
      !inputs.private_runtime_header.is_absolute() ||
      inputs.private_runtime_header.filename() != "closed_host_v1.h") {
    result.error = "region compilation requires owned absolute installation and staging paths";
    return result;
  }
  if (!ownedHeader(inputs.public_include_directory / "matcore/region.h",
                   frontend::detail::experimentalRegionHeaderSource(), result.error) ||
      !ownedHeader(inputs.public_include_directory / "matcore/detail/region_storage.h",
                   frontend::detail::experimentalRegionStorageHeaderSource(), result.error) ||
      !ownedHeader(inputs.private_runtime_header, expectedPrivateRuntimeHeader, result.error))
    return result;

  ExperimentalLLVMCompilation compilation;
  compilation.emission = std::move(*emitted.emission);
  compilation.host = std::move(original);
  std::string host_ir;
  if (!detail::compileFrozenHostToLLVM(*compilation.host, host_ir, result.error)) return result;

  const auto helper_path = inputs.staging_directory / "region-helper.cpp";
  if (fs::exists(helper_path, ec) || ec) {
    result.error = "compiler-owned helper output already exists";
    return result;
  }
  {
    std::ofstream output(helper_path, std::ios::binary);
    output << compilation.emission.helper_cpp;
    output.close();
    if (!output) { result.error = "cannot stage compiler-owned region implementation"; return result; }
  }
  if (after_staging) after_staging();

  // Separate translation unit: never copy the host's macros/include paths or
  // other compilation options into private runtime orchestration. The adapter
  // owns its private ABI and the final link enforces its revision dependency.
  frontend::Options helper_options;
  helper_options.input_path = helper_path.string();
  helper_options.clang_path = inputs.clang_path;
  helper_options.clang_resource_directory = inputs.clang_resource_directory;
  helper_options.compiler_arguments = {"-I/__mdsl_private__"};
  if (inputs.address_undefined_sanitizers)
    helper_options.compiler_arguments.push_back("-fsanitize=address,undefined");
  auto capture = host::prepareHostInputs(helper_options,
      inputs.staging_directory.string(),
      {{"/__mdsl_private__/fixture.h", "#pragma once\n"},
       {"/__mdsl_private__/closed_host_v1.h", expectedPrivateRuntimeHeader},
       {"/__mdsl_private__/matcore/region.h", frontend::detail::experimentalRegionHeaderSource()},
       {"/__mdsl_private__/matcore/detail/region_storage.h", frontend::detail::experimentalRegionStorageHeaderSource()}},
      result.error);
  if (!capture) return result;
  if (capture->sourceSnapshot() != compilation.emission.helper_cpp) {
    result.error = "staged region implementation differs from compiler-issued bytes";
    return result;
  }
  clang::FileSystemOptions file_options;
  file_options.WorkingDir = inputs.staging_directory.string();
  auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(
      file_options, capture->fileSystem());
  clang::tooling::ToolInvocation syntax(capture->arguments(),
      std::make_unique<clang::SyntaxOnlyAction>(), files.get());
  if (!syntax.run()) { result.error = "compiler-owned region implementation failed Clang Sema"; return result; }
  compilation.helper = capture->freeze(result.error);
  if (!compilation.helper) return result;
  std::string helper_ir;
  if (!detail::compileFrozenHostToLLVM(*compilation.helper, helper_ir, result.error)) return result;

  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto original_module = llvm::parseAssemblyString(host_ir, diagnostic, context);
  auto implementation = llvm::parseAssemblyString(helper_ir, diagnostic, context);
  if (!original_module || !implementation) {
    llvm::raw_string_ostream output(result.error);
    diagnostic.print("mdslc-region", output);
    return result;
  }
  ArtifactSymbolOwnershipReport ownership;
  if (!verifyHostArtifactSymbolOwnership(*original_module, inputs.symbol_artifacts,
                                        ownership, result.error)) return result;
  auto linked = linkAuthenticatedHostThunk(*original_module, *implementation,
      {compilation.emission.host_symbol, compilation.emission.helper_symbol,
       compilation.emission.retired_value_helpers});
  if (!linked) { result.error = linked.error; return result; }
  if (!compilation.inputsUnchanged(result.error)) return result;
  {
    llvm::raw_string_ostream output(compilation.llvm_ir);
    linked.module->print(output, nullptr);
  }
  result.compilation = std::move(compilation);
  return result;
}

ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVM(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const ExperimentalCompilerInputs &inputs, ClosedCpuPolicy policy) {
  return compileRegion(evidence, inputs, policy, {});
}

ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVMForTesting(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const ExperimentalCompilerInputs &inputs, ClosedCpuPolicy policy,
    const std::function<void()> &after_staging) {
  return compileRegion(evidence, inputs, policy, after_staging);
}
} // namespace matcore::mdslc::codegen
