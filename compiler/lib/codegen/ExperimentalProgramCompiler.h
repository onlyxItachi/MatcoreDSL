#pragma once
#include "ExperimentalRegionCompiler.h"

namespace matcore::mdslc::codegen {
struct ExperimentalProgramSource {
  frontend::Options options;
  // Empty means an ordinary host TU, not an imported mathematical body.
  std::string region;
};
struct ExperimentalProgramCompilation {
  std::string llvm_ir;
  std::vector<std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot>> inputs;
  std::vector<ExperimentalLLVMCompilation> regions;
  bool inputsUnchanged(std::string &error) const;
};
struct ExperimentalProgramCompilationResult {
  std::optional<ExperimentalProgramCompilation> compilation;
  std::string error;
  explicit operator bool() const { return compilation.has_value(); }
};
// A single invocation owns 2..8 original source TUs, selected region seals,
// ordinary host interfaces and private artifacts. No objects/LLVM/serialized
// module argument is accepted. No cross-region mathematical transform occurs.
ExperimentalProgramCompilationResult compileExperimentalProgramToLLVM(
    const std::vector<ExperimentalProgramSource> &, const std::string &working_directory,
    const ExperimentalCompilerInputs &, ClosedCpuPolicy = ClosedCpuPolicy::Automatic);
// Compiler test-only deterministic mutation seam, not source or CLI authority.
ExperimentalProgramCompilationResult compileExperimentalProgramToLLVMForTesting(
    const std::vector<ExperimentalProgramSource> &, const std::string &working_directory,
    const ExperimentalCompilerInputs &, ClosedCpuPolicy,
    const std::function<void()> &after_interface_capture);
} // namespace matcore::mdslc::codegen
