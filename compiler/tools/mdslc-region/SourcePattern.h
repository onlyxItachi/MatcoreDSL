#pragma once
// Private source-compilation ownership. No source/runtime callback, submitted
// IR, artifact importer, cache reader, or installed consumer API is provided.
#include "DriverSupport.h"
#include <cstdint>
#include <memory>

namespace matcore::mdslc::driver {
struct OutputTileRequest {
  std::int64_t m = 0, n = 0;
};
OutputTileRequest parseOutputTileRequest(const std::string &);
void validateSourcePatternInvocation(codegen::ClosedCpuPolicy,
    codegen::ClosedHostOptimization, bool compile_only);

class SourcePatternBundle {
public:
  explicit SourcePatternBundle(const fs::path &output);
  ~SourcePatternBundle();
  SourcePatternBundle(const SourcePatternBundle &) = delete;
  SourcePatternBundle &operator=(const SourcePatternBundle &) = delete;
  const fs::path &path() const { return path_; }
  void unchangedOwnership() const;
  void retain() noexcept { retained_ = true; }
private:
  fs::path path_;
  support::FileSnapshotV1 ownership_;
  bool retained_ = false;
};

// Noninstalled deterministic test-only phase hooks. Each runs AFTER immutable
// capture and is followed by the same mandatory recheck as normal issuance.
struct SourcePatternTestHooks {
  std::function<void(const fs::path &)> after_issue;
  std::function<void(const fs::path &)> after_object;
  std::function<void(const fs::path &)> after_dso;
};

class SourcePatternInstance {
public:
  static SourcePatternInstance issue(const Installation &, SourcePatternBundle &,
      const frontend::AuthenticatedClosedRegionEvidence &, const std::string &region,
      codegen::ClosedCpuPolicy, codegen::ClosedHostOptimization, OutputTileRequest);
  static SourcePatternInstance issueForTesting(const Installation &, SourcePatternBundle &,
      const frontend::AuthenticatedClosedRegionEvidence &, const std::string &region,
      codegen::ClosedCpuPolicy, codegen::ClosedHostOptimization, OutputTileRequest,
      const SourcePatternTestHooks &);
  codegen::ExperimentalCompilerInputs compilerInputs(const fs::path &,
      const frontend::AuthenticatedClosedRegionEvidence &) const;
  const fs::path &candidatePath() const { return candidates_.path; }
  void unchanged() const;
  void verifyCompilation(const codegen::ExperimentalLLVMCompilation &) const;
  void verifyExecutable(const Artifact &) const;
private:
  SourcePatternInstance(const Installation &, SourcePatternBundle &,
      const frontend::AuthenticatedClosedRegionEvidence &, const std::string &region,
      codegen::ClosedCpuPolicy, codegen::ClosedHostOptimization, OutputTileRequest,
      const SourcePatternTestHooks &);
  const Installation *installation_;
  const SourcePatternBundle *bundle_;
  std::shared_ptr<const frontend::closed_region_host::HostInputSnapshot> host_;
  Artifact kit_, exports_, candidates_;
  std::vector<Artifact> issued_;
  std::string soname_;
  std::shared_ptr<const codegen::SourcePatternCompilationBinding> compilation_binding_;
};
} // namespace matcore::mdslc::driver
