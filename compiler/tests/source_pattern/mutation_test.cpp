// Noninstalled deterministic attack controls. Each hook runs after capture,
// so a valid foreign leaf/object/DSO cannot borrow this source request's seal.
#include "SourcePattern.h"
#include "../../lib/support/platform_support.h"
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dr = matcore::mdslc::driver;
namespace fe = matcore::mdslc::frontend;
namespace cg = matcore::mdslc::codegen;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
namespace {
unsigned checks = 0, failures = 0;
void check(bool okay, const std::string &label) {
  ++checks;
  if (!okay) { ++failures; std::cerr << "FAIL source pattern mutation: " << label << '\n'; }
}
std::string read(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("missing independent fixture artifact");
  return {std::istreambuf_iterator<char>(input), {}};
}
constexpr auto helper = R"cpp(#pragma once
#include <matcore/region.h>
namespace helper {
inline matcore::mdsl::Value product(matcore::mdsl::Value a,
                                   matcore::mdsl::Value b) noexcept {
  return matcore::mdsl::gemm(a,b,matcore::mdsl::Numerics::strict_f32);
}
}
)cpp";
constexpr auto source = R"cpp(#include "math.h"
namespace md = matcore::mdsl;
MATCORE_REGION md::Result pipeline(md::Storage A,md::Storage C) noexcept {
  auto a=md::read(A,1,1);
  auto c=helper::product(a,a);
  md::publish(c,C);
  md::observe(C);
  return md::complete();
}
int main(){float a=3,c=-1;auto r=pipeline({&a,1,1,1},{&c,1,1,1,md::Access::read_write});return !r||c!=9;}
)cpp";
struct Fixture {
  std::optional<support::TempDirectoryV1> inputs;
  std::optional<support::TempDirectoryV1> outputs;
  fs::path main, header, output;
  explicit Fixture(const std::string &label) {
    std::string error;
    inputs = support::create_temp_directory_v1("mdslc-source-pattern-input-" + label, error);
    outputs = support::create_temp_directory_v1("mdslc-source-pattern-output-" + label, error);
    if (!inputs || !outputs) throw std::runtime_error(error);
    main = inputs->path() / "source.mdsl";
    header = inputs->path() / "math.h";
    output = outputs->path() / "issued-executable";
    dr::write(main, source); dr::write(header, helper);
  }
  fe::ClosedRegionAdmissionResult admit(const dr::Installation &installation,
                                       const std::string &region = "pipeline") {
    return fe::admitExperimentalRegionHost(installation.options(main, {}), inputs->path().string(),
        {installation.public_header.path.string(), installation.storage_header.path.string()}, region);
  }
};
struct Donor { std::string llvm, object, dso, manifest; };
Donor donor(const dr::Installation &installation, dr::OutputTileRequest tiles) {
  Fixture fixture("donor");
  dr::SourcePatternBundle bundle(fixture.output);
  auto admitted = fixture.admit(installation);
  if (!admitted) throw std::runtime_error(admitted.error);
  auto issued = dr::SourcePatternInstance::issue(installation, bundle, *admitted.evidence,
      "pipeline", cg::ClosedCpuPolicy::GeneratedStrict, cg::ClosedHostOptimization::None, tiles);
  issued.unchanged();
  return {read(bundle.path() / "gemm.ll"), read(bundle.path() / "gemm.o"),
          read(issued.candidatePath()), read(bundle.path() / "source-pattern.manifest")};
}
template<class MakeHook>
void reject(const dr::Installation &installation, const char *label, MakeHook make_hook) {
  Fixture fixture(label);
  fs::path bundle_path;
  bool invoked = false, rejected = false;
  std::string error;
  {
    dr::SourcePatternBundle bundle(fixture.output);
    bundle_path = bundle.path();
    auto admitted = fixture.admit(installation);
    check(bool(admitted), std::string(label) + " authentic source control: " + admitted.error);
    if (!admitted) return;
    dr::SourcePatternTestHooks hooks;
    make_hook(hooks, fixture, invoked);
    try {
      auto instance = dr::SourcePatternInstance::issueForTesting(installation, bundle,
          *admitted.evidence, "pipeline", cg::ClosedCpuPolicy::GeneratedStrict,
          cg::ClosedHostOptimization::None, {3, 5}, hooks);
      instance.unchanged();
    } catch (const std::exception &failure) { rejected = true; error = failure.what(); }
    check(invoked && rejected && !error.empty(), std::string(label) + " reached captured phase and failed closed: " + error);
    check(!fs::exists(fixture.output), std::string(label) + " did not publish executable");
  }
  check(!fs::exists(bundle_path), std::string(label) + " removed only its failed owned bundle");
}
}
int main() {
  try {
    const auto installation = dr::Installation::pinnedBuildTreeForTesting();
    const auto foreign = donor(installation, {5, 3});
    const auto stale = donor(installation, {3, 5});
    check(foreign.llvm != stale.llvm && foreign.object != stale.object && foreign.dso != stale.dso,
          "independent valid source donors differ in tile realization");
    {
      Fixture bound("consumer-binding"), other("other-source-binding");
      dr::SourcePatternBundle bundle(bound.output);
      auto a = bound.admit(installation), b = other.admit(installation);
      if (!a || !b) throw std::runtime_error(a.error + b.error);
      auto instance = dr::SourcePatternInstance::issue(installation, bundle, *a.evidence,
          "pipeline", cg::ClosedCpuPolicy::GeneratedStrict, cg::ClosedHostOptimization::None, {3, 5});
      const auto stage = [&](const char *name) {
        auto path = bound.outputs->path() / name;
        fs::create_directory(path);
        return path;
      };
      auto inputs = instance.compilerInputs(stage("positive"), *a.evidence);
      const auto positive = cg::compileExperimentalRegionToLLVM(*a.evidence, inputs,
          cg::ClosedCpuPolicy::GeneratedStrict, cg::ClosedHostOptimization::None);
      check(bool(positive), "exact source-bound consumer compiles: " + positive.error);
      bool factory_rejected = false;
      try { (void)instance.compilerInputs(stage("wrong-source-factory"), *b.evidence); }
      catch (const std::exception &error) {
        factory_rejected = std::string(error.what()).find("source-pattern") != std::string::npos;
      }
      check(factory_rejected, "private input factory refuses another authenticated physical source");
      const auto reject_consumer = [&](const fe::AuthenticatedClosedRegionEvidence &evidence,
          cg::ExperimentalCompilerInputs changed, cg::ClosedCpuPolicy policy,
          cg::ClosedHostOptimization optimization, const char *label) {
        changed.staging_directory = stage(label);
        const auto result = cg::compileExperimentalRegionToLLVM(evidence, changed, policy, optimization);
        check(!result && result.error.find("source-pattern") != std::string::npos &&
                  !fs::exists(changed.staging_directory / "region-helper.cpp"),
              std::string("compileRegion independently refuses ") + label + ": " + result.error);
      };
      reject_consumer(*b.evidence, inputs, cg::ClosedCpuPolicy::GeneratedStrict,
          cg::ClosedHostOptimization::None, "other-source-direct-consumer");
      reject_consumer(*a.evidence, inputs, cg::ClosedCpuPolicy::NativeStrict,
          cg::ClosedHostOptimization::None, "candidate-policy-change");
      reject_consumer(*a.evidence, inputs, cg::ClosedCpuPolicy::GeneratedStrict,
          cg::ClosedHostOptimization::PublicationReadForwarding, "optimization-change");
      auto profile = inputs;
      profile.address_undefined_sanitizers = !profile.address_undefined_sanitizers;
      reject_consumer(*a.evidence, profile, cg::ClosedCpuPolicy::GeneratedStrict,
          cg::ClosedHostOptimization::None, "sanitizer-profile-change");
      auto bytes = inputs, workers = inputs;
      for (auto &artifact : bytes.symbol_artifacts)
        if (artifact.owner == cg::SymbolArtifactOwner::PrivateCandidates)
          artifact.bytes = llvm::MemoryBufferRef(foreign.dso, "valid foreign candidate DSO");
      for (auto &artifact : workers.symbol_artifacts)
        if (artifact.owner == cg::SymbolArtifactOwner::PrivateCandidates)
          artifact.requires_worker_threads = !artifact.requires_worker_threads;
      reject_consumer(*a.evidence, bytes, cg::ClosedCpuPolicy::GeneratedStrict,
          cg::ClosedHostOptimization::None, "valid-candidate-bytes-change");
      reject_consumer(*a.evidence, workers, cg::ClosedCpuPolicy::GeneratedStrict,
          cg::ClosedHostOptimization::None, "worker-contract-change");
    }
    {
      // The two entries share the exact physical source/header closure, so a
      // source-file digest alone cannot distinguish their execution authority.
      Fixture fixture("same-source-different-entry");
      std::string two_entries = source;
      const auto insertion = two_entries.find("int main()");
      if (insertion == std::string::npos) throw std::runtime_error("missing ordinary fixture main");
      two_entries.insert(insertion, R"cpp(MATCORE_REGION md::Result alternate(md::Storage A,md::Storage C) noexcept {
  auto a=md::read(A,1,1);
  auto c=helper::product(a,a);
  md::publish(c,C);
  md::observe(C);
  return md::complete();
}
)cpp");
      dr::write(fixture.main, two_entries);
      dr::SourcePatternBundle bundle(fixture.output);
      auto a = fixture.admit(installation), b = fixture.admit(installation, "alternate");
      if (!a || !b) throw std::runtime_error(a.error + b.error);
      auto instance = dr::SourcePatternInstance::issue(installation, bundle, *a.evidence,
          "pipeline", cg::ClosedCpuPolicy::GeneratedStrict, cg::ClosedHostOptimization::None, {3, 5});
      const auto staging = fixture.outputs->path() / "entry-consumer";
      fs::create_directory(staging);
      auto inputs = instance.compilerInputs(staging, *a.evidence);
      const auto changed = cg::compileExperimentalRegionToLLVM(*b.evidence, inputs,
          cg::ClosedCpuPolicy::GeneratedStrict, cg::ClosedHostOptimization::None);
      check(!changed && changed.error.find("source-pattern") != std::string::npos &&
                !fs::exists(staging / "region-helper.cpp"),
            "compileRegion refuses same physical source but different selected entry");
      bool refused = false;
      try { (void)instance.compilerInputs(staging, *b.evidence); }
      catch (const std::exception &error) {
        refused = std::string(error.what()).find("source-pattern") != std::string::npos;
      }
      check(refused, "private input factory refuses a different entry from the same source");
    }
    for (const auto phase : {0, 1, 2}) {
      const auto source_label = "main-after-phase-" + std::to_string(phase);
      reject(installation, source_label.c_str(), [&](auto &hooks, auto &fixture, bool &invoked) {
        auto change = [&](const fs::path &) { invoked = true; dr::write(fixture.main, std::string(source) + "\n// stale closure\n"); };
        if (phase == 0) hooks.after_issue = change;
        else if (phase == 1) hooks.after_object = change;
        else hooks.after_dso = change;
      });
      const auto header_label = "header-after-phase-" + std::to_string(phase);
      reject(installation, header_label.c_str(), [&](auto &hooks, auto &fixture, bool &invoked) {
        auto change = [&](const fs::path &) { invoked = true; dr::write(fixture.header, std::string(helper) + "\n// stale physical inclusion\n"); };
        if (phase == 0) hooks.after_issue = change;
        else if (phase == 1) hooks.after_object = change;
        else hooks.after_dso = change;
      });
    }
    reject(installation, "valid-other-tiles-llvm", [&](auto &hooks, auto &, bool &invoked) {
      hooks.after_issue = [&](const fs::path &path) { invoked = true; dr::write(path / "gemm.ll", foreign.llvm); };
    });
    reject(installation, "valid-other-tiles-object", [&](auto &hooks, auto &, bool &invoked) {
      hooks.after_object = [&](const fs::path &path) { invoked = true; dr::write(path / "gemm.o", foreign.object); };
    });
    const auto replace_dso = [&](const Donor &replacement, const char *label) {
      reject(installation, label, [&](auto &hooks, auto &, bool &invoked) {
        hooks.after_dso = [&](const fs::path &path) {
          invoked = true;
          bool found = false;
          for (const auto &entry : fs::directory_iterator(path))
            if (entry.path().extension() == ".so") {
              found = true; dr::write(entry.path(), replacement.dso);
            }
          if (!found) throw std::runtime_error("DSO hook did not reach an issued candidate");
        };
      });
    };
    replace_dso(foreign, "valid-other-tiles-dso");
    replace_dso(stale, "valid-stale-source-dso");
    reject(installation, "forged-issued-MLIR", [&](auto &hooks, auto &, bool &invoked) {
      hooks.after_issue = [&](const fs::path &path) {
        invoked = true;
        dr::write(path / "gemm.ll.semantic.mlir", read(path / "gemm.ll.semantic.mlir") + "\n// submitted authority\n");
      };
    });
    reject(installation, "stale-manifest-prepopulation", [&](auto &hooks, auto &, bool &invoked) {
      hooks.after_dso = [&](const fs::path &path) { invoked = true; dr::write(path / "source-pattern.manifest", stale.manifest); };
    });
    // A racing symlink must neither obtain authority nor become a recursive
    // cleanup target. All directories here are exact task-created temp roots.
    Fixture raced("directory-race");
    const auto foreign_directory = raced.outputs->path() / "foreign";
    fs::create_directory(foreign_directory);
    dr::write(foreign_directory / "sentinel", "caller bytes must survive\n");
    fs::path original_bundle;
    bool hook_ran = false, race_rejected = false;
    {
      dr::SourcePatternBundle bundle(raced.output);
      original_bundle = bundle.path();
      auto admitted = raced.admit(installation);
      if (!admitted) throw std::runtime_error(admitted.error);
      dr::SourcePatternTestHooks hooks;
      hooks.after_issue = [&](const fs::path &path) {
        hook_ran = true;
        fs::rename(path, path.string() + ".moved");
        fs::create_directory_symlink(foreign_directory, path);
      };
      try {
        auto instance = dr::SourcePatternInstance::issueForTesting(installation, bundle, *admitted.evidence,
            "pipeline", cg::ClosedCpuPolicy::GeneratedStrict, cg::ClosedHostOptimization::None, {3, 5}, hooks);
        instance.unchanged();
      } catch (const std::exception &) { race_rejected = true; }
    }
    check(hook_ran && race_rejected && !fs::exists(raced.output), "bundle path race did not publish an executable");
    check(read(foreign_directory / "sentinel") == "caller bytes must survive\n" &&
              fs::exists(original_bundle.string() + ".moved"),
          "path substitution did not delete foreign or moved owned bytes");
    std::cout << "Source pattern phase controls: " << checks << " checks; " << failures << " failures\n";
    return failures ? 1 : 0;
  } catch (const std::exception &failure) {
    std::cerr << "independent source-pattern setup failed: " << failure.what() << '\n';
    return 2;
  }
}
