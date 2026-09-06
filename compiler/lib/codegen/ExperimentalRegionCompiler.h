#pragma once

#include "ExperimentalRegionEmitter.h"
#include "../frontend/ClosedRegionHostInputs.h"
#include <filesystem>
#include <functional>

namespace matcore::mdslc::codegen {

// Compiler installation/build inputs, never source or serialized authority.
// The driver creates a private staging directory before capturing host inputs.
struct ExperimentalCompilerInputs {
  std::string clang_path;
  std::string clang_resource_directory;
  std::filesystem::path public_include_directory;
  std::filesystem::path private_runtime_header;
  std::filesystem::path staging_directory;
  bool address_undefined_sanitizers = false;
};

struct ExperimentalLLVMCompilation {
  ExperimentalRegionEmission emission;
  std::string llvm_ir;
  std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot> host;
  std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot> helper;
  bool inputsUnchanged(std::string &error) const;
};
struct ExperimentalLLVMCompilationResult {
  std::optional<ExperimentalLLVMCompilation> compilation;
  std::string error;
  explicit operator bool() const { return compilation.has_value(); }
};

// Derived compiler output, not an importable LLVM authorization interface.
// A successful result has not yet been linked to a runtime or published as an
// executable. The driver must check inputs again before final publication.
ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVM(
    const frontend::AuthenticatedClosedRegionEvidence &,
    const ExperimentalCompilerInputs &,
    ClosedCpuPolicy policy = ClosedCpuPolicy::Automatic);

// Noninstalled deterministic staging test seam. No runtime/source option or
// driver argument exposes this callback; all issued-byte checks still apply.
ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVMForTesting(
    const frontend::AuthenticatedClosedRegionEvidence &,
    const ExperimentalCompilerInputs &, ClosedCpuPolicy,
    const std::function<void()> &after_staging);

} // namespace matcore::mdslc::codegen
