#pragma once

#include "ExperimentalRegionEmitter.h"
#include "ArtifactSymbolOwnership.h"
#include "../frontend/ClosedRegionHostInputs.h"
#include <filesystem>
#include <functional>

namespace matcore::mdslc::driver { class SourcePatternInstance; }
namespace matcore::mdslc::codegen {

struct ExperimentalCompilerInputs;
struct ExperimentalLLVMCompilation;
struct ExperimentalLLVMCompilationResult;
// Private immutable per-source artifact binding. There is no public/serialized
// constructor: only the source driver's closed issuer can pair a genuine
// authenticated source seal with its freshly issued private candidate bytes.
class SourcePatternCompilationBinding {
public:
  bool matches(const frontend::AuthenticatedClosedRegionEvidence &,
      const ExperimentalCompilerInputs &, ClosedCpuPolicy,
      ClosedHostOptimization, std::string &error) const;
private:
  SourcePatternCompilationBinding(const frontend::AuthenticatedClosedRegionEvidence &,
      std::string candidate_sha256, ClosedCpuPolicy, ClosedHostOptimization,
      bool requires_worker_threads, bool address_undefined_sanitizers);
  frontend::AuthenticatedClosedRegionEvidence evidence_;
  std::string candidate_sha256_;
  ClosedCpuPolicy policy_;
  ClosedHostOptimization optimization_;
  bool requires_worker_threads_;
  bool address_undefined_sanitizers_;
  friend class matcore::mdslc::driver::SourcePatternInstance;
};

// Special-case source-pattern publication bookkeeping, not an IR certificate
// or importer. Only a successful authenticated region compilation can retain
// this immutable pairing of issuance, produced LLVM bytes and frozen inputs.
class SourcePatternCompiledBinding {
public:
  bool matches(const ExperimentalLLVMCompilation &,
      const std::shared_ptr<const SourcePatternCompilationBinding> &,
      std::string &error) const;
private:
  SourcePatternCompiledBinding(std::shared_ptr<const SourcePatternCompilationBinding>,
      const ExperimentalLLVMCompilation &);
  std::shared_ptr<const SourcePatternCompilationBinding> source_;
  std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot> host_, helper_;
  std::string llvm_sha256_;
  friend ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVM(
      const frontend::AuthenticatedClosedRegionEvidence &, const ExperimentalCompilerInputs &,
      ClosedCpuPolicy, ClosedHostOptimization);
  friend ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVMForTesting(
      const frontend::AuthenticatedClosedRegionEvidence &, const ExperimentalCompilerInputs &,
      ClosedCpuPolicy, const std::function<void()> &, ClosedHostOptimization);
};

// Compiler installation/build inputs, never source or serialized authority.
// The driver creates a private staging directory before capturing host inputs.
struct ExperimentalCompilerInputs {
  std::string clang_path;
  std::string clang_resource_directory;
  std::filesystem::path public_include_directory;
  std::filesystem::path private_runtime_header;
  std::filesystem::path staging_directory;
  bool address_undefined_sanitizers = false;
  // Borrowed immutable bytes of exact pinned link inputs, owned by the driver
  // until compilation returns. Exactly one canonical runtime is mandatory.
  std::vector<TrustedSymbolArtifact> symbol_artifacts;
  std::shared_ptr<const SourcePatternCompilationBinding> source_pattern_binding = {};
};

struct ExperimentalLLVMCompilation {
  ExperimentalRegionEmission emission;
  std::string llvm_ir;
  std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot> host;
  std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot> helper;
  std::shared_ptr<const SourcePatternCompiledBinding> source_pattern_compilation = {};
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
    ClosedCpuPolicy policy = ClosedCpuPolicy::Automatic,
    ClosedHostOptimization optimization = ClosedHostOptimization::None);

// Noninstalled deterministic staging test seam. No runtime/source option or
// driver argument exposes this callback; all issued-byte checks still apply.
ExperimentalLLVMCompilationResult compileExperimentalRegionToLLVMForTesting(
    const frontend::AuthenticatedClosedRegionEvidence &,
    const ExperimentalCompilerInputs &, ClosedCpuPolicy,
    const std::function<void()> &after_staging,
    ClosedHostOptimization optimization = ClosedHostOptimization::None);

} // namespace matcore::mdslc::codegen
