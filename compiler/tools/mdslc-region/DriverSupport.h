#pragma once
// Private executable support, not an installed LLVM import/authority interface.
#include "ExperimentalRegionCompiler.h"
#include "ClosedRegionAdmission.h"
#include "platform_support.h"
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace matcore::mdslc::driver {
namespace fs = std::filesystem;
namespace support = matcore::mdslc::support;
namespace frontend = matcore::mdslc::frontend;
namespace codegen = matcore::mdslc::codegen;
[[noreturn]] void reject(const std::string &);
struct Artifact {
  fs::path path;
  support::FileSnapshotV1 snapshot;
  std::string sha, bytes;
  static Artifact capture(const fs::path &, const char *expected = nullptr);
  void unchanged() const;
};
struct Layout {
  fs::path include, header, candidates, runtime;
  bool build_tree = false;
};
Layout layout();
struct Staging {
  fs::path path;
  explicit Staging(const fs::path &parent);
  ~Staging();
  Staging(const Staging &) = delete;
  Staging &operator=(const Staging &) = delete;
};
void write(const fs::path &, const std::string &);
void validateNewOutput(const fs::path &);
codegen::ClosedCpuPolicy parseCandidatePolicy(const std::string &);
const char *candidatePolicyUsage();
struct Installation {
  // Declaration order is the original driver's exact capture order.
  Layout installed;
  Artifact clang, linker, candidates, runtime;
  std::optional<Artifact> provider;
  Artifact public_header, storage_header, private_header;
  Installation();
  std::vector<codegen::TrustedSymbolArtifact> symbolArtifacts() const;
  void unchanged() const;
  frontend::Options options(const fs::path &, const std::vector<std::string> &) const;
  codegen::ExperimentalCompilerInputs compilerInputs(const fs::path &) const;
};
// Only compiler-derived output reaches this private helper. The mandatory
// callback checks the entire frozen source/helper closure before native
// compilation and again before atomic no-clobber publication.
void compileAndPublish(const Installation &, const Staging &, const fs::path &,
    bool compile_only, const std::string &derived_ir,
    const std::function<void()> &inputs_unchanged);
} // namespace matcore::mdslc::driver
