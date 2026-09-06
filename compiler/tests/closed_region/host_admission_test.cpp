#include "ClosedRegionAdmissionInternal.h"
#include "MatcoreCpuRuntimeLowering.h"
#include "host_adversarial_sources.h"
#include "../../lib/support/platform_support.h"
#include "mlir/IR/Builders.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fe = matcore::mdslc::frontend;
namespace cr = matcore::mdslc::closed_region;
namespace hi = fe::closed_region_host;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
namespace {
unsigned checks = 0, failures = 0;
void check(bool value, const std::string &label) {
  ++checks;
  if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
void write(const fs::path &path, const std::string &bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (!stream) throw std::runtime_error("could not write owned fixture: " + path.string());
}
struct Fixture {
  support::TempDirectoryV1 temporary;
  fe::Options options;
  Fixture(const std::string &clang, const std::string &resource,
          const std::string &source) {
    std::string error;
    auto directory = support::create_temp_directory_v1("mdslc-host-context-test", error);
    if (!directory) throw std::runtime_error(error);
    temporary = std::move(*directory);
    options.input_path = (path() / "source.mdsl").string();
    options.clang_path = clang;
    options.clang_resource_directory = resource;
    options.compiler_arguments = {"-std=c++20", "-I" + path().string()};
    write(options.input_path, source);
  }
  const fs::path &path() const { return temporary.path(); }
  fe::ClosedRegionAdmissionResult admit() const {
    return fe::admitClosedRegionHost(options, path().string());
  }
};
const std::string simple =
    "using namespace mdsl_probe;\n"
    "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n"
    "void region(Storage A,Storage B,Storage C,Shape m,Shape k,Shape n) {\n"
    "auto a=read(A,m,k); auto b=read(B,k,n);\n"
    "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C);\n}\n";
bool paired(const fe::AuthenticatedClosedRegionEvidence &evidence,
            mlir::ModuleOp module) {
  std::string error;
  const bool okay = fe::verifyClosedRegionMatchesEvidence(evidence, module, error);
  if (!okay) std::cerr << "pairing diagnostic: " << error << '\n';
  return okay;
}
struct EnvironmentValue {
  std::string name;
  std::optional<std::string> old;
  explicit EnvironmentValue(const char *key) : name(key) {
    if (const char *value = std::getenv(key)) old = value;
  }
  ~EnvironmentValue() {
    if (old) setenv(name.c_str(), old->c_str(), 1);
    else unsetenv(name.c_str());
  }
};
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  const std::string clang = argv[1], resource = argv[2];
  try {
    for (const auto &item : matcore::mdslc::test::closedRegionHostSources()) {
      Fixture fixture(clang, resource, item.source);
      for (const auto &[name, bytes] : item.headers) write(fixture.path() / name, bytes);
      auto result = fixture.admit();
      check(result.syntax_valid == item.expect_syntax_valid,
            item.name + " Clang/Sema phase: " + result.error);
      check(static_cast<bool>(result) == item.expect_admission,
            item.name + ": " + item.obligation + " " + result.error);
      if (!result) { check(!result.error.empty(), item.name + " actionable rejection"); continue; }
      check(result.evidence->hasHostContext() && !result.evidence->hostContextIdentity().empty(),
            item.name + " owns immutable host context");
      mlir::MLIRContext context;
      auto module = cr::buildModule(result.evidence->program(), context);
      check(static_cast<bool>(module), item.name + " same frontend-neutral model: " + module.error);
      if (!module) continue;
      check(paired(*result.evidence, *module.module), item.name + " source/context pairing");
      std::string error;
      std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1> dispatch(1);
      check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
                *module.module, dispatch, error) && dispatch.empty(),
            item.name + " actual CPU execution consumer still rejects");
    }

    // The same source body has the same semantic graph; only the additional
    // compilation-context provenance differs. Normalization is a test oracle,
    // never an authority-issuing operation.
    Fixture control(clang, resource, simple);
    auto host = control.admit();
    auto hermetic = fe::admitClosedRegionSource(simple, control.options.input_path);
    check(host && hermetic, "host and hermetic controls admit: " + host.error);
    if (host && hermetic) {
      mlir::MLIRContext context;
      auto left = cr::buildModule(host.evidence->program(), context);
      auto right = cr::buildModule(hermetic.evidence->program(), context);
      check(left && right, "both controls build");
      if (left && right) {
        (*left.module)->removeAttr("mdsl_admission.compiler_identity");
        (*right.module)->removeAttr("mdsl_admission.compiler_identity");
        check(cr::printModule(*left.module) == cr::printModule(*right.module),
              "host context does not change the approved value/effect grammar");
      }
    }

    // Context identity binds inputs even when they leave the admitted graph
    // unchanged. In particular, do not accidentally drop std::string fields
    // through an unqualified helper call resolved by argument-dependent lookup.
    auto flagOptions = control.options;
    flagOptions.compiler_arguments.push_back("-DHOST_DIAGNOSTIC_ONLY=1");
    auto flagOne = fe::admitClosedRegionHost(flagOptions, control.path().string());
    flagOptions.compiler_arguments.back() = "-DHOST_DIAGNOSTIC_ONLY=2";
    auto flagTwo = fe::admitClosedRegionHost(flagOptions, control.path().string());
    check(flagOne && flagTwo && flagOne.evidence->hostContextIdentity() !=
              flagTwo.evidence->hostContextIdentity(),
          "unused compiler definitions remain bound context inputs");
    fs::create_directory(control.path() / "other-cwd");
    auto sameCwd = control.admit();
    auto otherCwd = fe::admitClosedRegionHost(control.options,
                                             (control.path() / "other-cwd").string());
    check(sameCwd && otherCwd && sameCwd.evidence->hostContextIdentity() !=
              otherCwd.evidence->hostContextIdentity(),
          "working directory remains bound with an identical absolute source");

    for (const std::string &body : {
             std::string("auto a=read(A,2,3); auto b=read(B,3,4); "
                         "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); "
                         "observe(C); auto d=read(D,5,2); "
                         "auto e=gemm(d,c,Numerics::strict_f32); publish(e,E);"),
             std::string("auto a=read(A,m,k); auto b=read(B,k,n); "
                         "auto c=product(a,b); publish(c,C); "
                         "if (m<n) { auto late=read(C,m,n); "
                         "auto d=read(D,n,k); auto e=product(late,d); publish(e,E); } "
                         "else { auto d=read(D,k,m); auto e=product(d,c); publish(e,E); } "
                         "observe(E);")}) {
      Fixture math(clang, resource,
          "#include <iostream>\nusing namespace mdsl_probe;\n"
          "void host_only() { std::cout << 42; }\n"
          "Value product(Value a,Value b) { return gemm(a,b,Numerics::reassociate_f32); }\n"
          "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n"
          "void region(Storage A,Storage B,Storage C,Storage D,Storage E,"
          "Shape m,Shape k,Shape n) { " + body + " }\n");
      auto admitted = math.admit();
      check(static_cast<bool>(admitted), "real host rectangular/RHS or symbolic/branch helper: " + admitted.error);
      if (admitted) {
        mlir::MLIRContext context;
        auto module = cr::buildModule(admitted.evidence->program(), context);
        check(module && paired(*admitted.evidence, *module.module),
              "useful host mathematical specimen passes source/context pairing");
      }
    }

    Fixture search(clang, resource, "#include <choice.h>\n" + simple);
    write(search.path() / "first/choice.h", "#pragma once\n");
    write(search.path() / "second/choice.h", "#pragma once\n");
    search.options.compiler_arguments.push_back("-I" + (search.path() / "first").string());
    search.options.compiler_arguments.push_back("-I" + (search.path() / "second").string());
    auto searchOne = search.admit();
    std::swap(search.options.compiler_arguments[2], search.options.compiler_arguments[3]);
    auto searchTwo = search.admit();
    check(searchOne && searchTwo && searchOne.evidence->hostContextIdentity() !=
              searchTwo.evidence->hostContextIdentity(),
          "include search order binds even when resolved header bytes match");

    if (host) {
      mlir::MLIRContext context;
      auto module = cr::buildModule(host.evidence->program(), context);
      check(static_cast<bool>(module), "environment replay control builds");
      for (const char *key : {"CPATH", "SOURCE_DATE_EPOCH"}) {
        EnvironmentValue guard(key);
        setenv(key, key == std::string("CPATH") ? control.path().c_str() : "0", 1);
        std::string diagnostic;
        check(module && !fe::verifyClosedRegionMatchesEvidence(
                  *host.evidence, *module.module, diagnostic) && !diagnostic.empty(),
              std::string("sealed replay rejects newly poisoned environment: ") + key);
      }
    }

    // Captured header bytes are immutable replay inputs, not a request to read
    // whatever happens to be at the path when pairing later occurs.
    Fixture headers(clang, resource, "#include \"alias.h\"\n" + simple);
    write(headers.path() / "alias.h", "#pragma once\nusing HostExtent=mdsl_probe::Shape;\n");
    auto first = headers.admit();
    check(static_cast<bool>(first), "capture local header before mutation: " + first.error);
    if (first) {
      mlir::MLIRContext context;
      auto module = cr::buildModule(first.evidence->program(), context);
      write(headers.path() / "alias.h", "#pragma once\nusing HostExtent=mdsl_probe::Shape;\n// changed bytes\n");
      check(module && paired(*first.evidence, *module.module), "replay uses old sealed header, not changed physical bytes");
      auto second = headers.admit();
      check(static_cast<bool>(second), "fresh changed-header capture admits: " + second.error);
      if (second && module) {
        check(first.evidence->hostContextIdentity() != second.evidence->hostContextIdentity(),
              "header-only change alters context despite identical main source and mathematics");
        std::string error;
        check(!fe::verifyClosedRegionMatchesEvidence(*second.evidence, *module.module, error),
              "new context cannot authenticate a previous context's specimen");
      }
    }

    Fixture absent(clang, resource,
        "#if __has_include(\"optional.h\")\n#error unexpected optional header\n#endif\n" + simple);
    auto negativeLookup = absent.admit();
    check(static_cast<bool>(negativeLookup), "capture absent optional lookup: " + negativeLookup.error);
    if (negativeLookup) {
      mlir::MLIRContext context;
      auto module = cr::buildModule(negativeLookup.evidence->program(), context);
      write(absent.path() / "optional.h", "new physical header\n");
      check(module && paired(*negativeLookup.evidence, *module.module), "negative lookup stays absent during immutable replay");
      check(!absent.admit(), "fresh physical capture sees the newly present optional header");
    }

    Fixture presence(clang, resource,
        "#if !__has_include(\"presence.h\")\n#error required presence missing\n#endif\n" + simple);
    write(presence.path() / "presence.h", "intentionally not included and not valid C++\n");
    auto presentLookup = presence.admit();
    check(static_cast<bool>(presentLookup), "positive-but-unopened lookup captured: " + presentLookup.error);
    if (presentLookup) {
      mlir::MLIRContext context;
      auto module = cr::buildModule(presentLookup.evidence->program(), context);
      fs::remove(presence.path() / "presence.h");
      check(module && paired(*presentLookup.evidence, *module.module), "positive-but-unopened status survives immutable replay");
      check(!presence.admit(), "fresh capture observes removed presence-only dependency");
    }

    // Mutations occur after the actual first Clang parse but before issuance;
    // the hook is private test harness code, never an admitted host callback.
    Fixture race(clang, resource, "#include \"race.h\"\n" + simple);
    write(race.path() / "race.h", "#pragma once\n");
    auto changed = fe::detail::admitClosedRegionHostForTesting(
        race.options, race.path().string(), "region", [&] {
          write(race.path() / "race.h", "#pragma once\n// mid-capture mutation\n");
        });
    check(!changed && !changed.error.empty(), "mid-capture header byte mutation rejects before issuing evidence");
    auto sourceChanged = fe::detail::admitClosedRegionHostForTesting(
        race.options, race.path().string(), "region", [&] { write(race.options.input_path, simple + "// changed\n"); });
    check(!sourceChanged && !sourceChanged.error.empty(), "mid-capture main-source mutation rejects");

    Fixture identity(clang, resource, "#include \"identity.h\"\n" + simple);
    write(identity.path() / "identity.h", "#pragma once\n");
    write(identity.path() / "replacement.h", "#pragma once\n");
    auto replaced = fe::detail::admitClosedRegionHostForTesting(
        identity.options, identity.path().string(), "region", [&] {
          fs::rename(identity.path() / "replacement.h", identity.path() / "identity.h");
        });
    check(!replaced && !replaced.error.empty(), "identical bytes on a replaced physical identity reject");

    Fixture links(clang, resource, "#include \"link.h\"\n" + simple);
    write(links.path() / "one.h", "#pragma once\n");
    write(links.path() / "two.h", "#pragma once\n");
    fs::create_symlink("one.h", links.path() / "link.h");
    fs::create_symlink("two.h", links.path() / "replacement-link.h");
    auto retargeted = fe::detail::admitClosedRegionHostForTesting(
        links.options, links.path().string(), "region", [&] {
          fs::rename(links.path() / "replacement-link.h", links.path() / "link.h");
        });
    check(!retargeted && !retargeted.error.empty(), "same-byte dependency symlink retarget rejects");

    Fixture negativeRace(clang, resource,
        "#if __has_include(\"new.h\")\n#error appeared\n#endif\n" + simple);
    auto appeared = fe::detail::admitClosedRegionHostForTesting(
        negativeRace.options, negativeRace.path().string(), "region", [&] { write(negativeRace.path() / "new.h", "\n"); });
    check(!appeared && !appeared.error.empty(), "negative lookup changing before seal rejects");

    // A frozen transcript must distinguish known ENOENT from an unrecorded
    // query; neither may silently fall through to the physical filesystem.
    std::string error;
    auto inputs = hi::prepareHostInputs(control.options, control.path().string(),
        {{fe::detail::closedRegionOwnedHeaderPath(), fe::detail::closedRegionOwnedHeaderSource()}}, error);
    check(static_cast<bool>(inputs), "prepare independent input-recorder control: " + error);
    if (inputs) {
      auto missing = inputs->fileSystem()->status((control.path() / "known-missing.h").string());
      check(!missing, "record real negative lookup");
      auto snapshot = inputs->freeze(error);
      check(static_cast<bool>(snapshot), "freeze independent input transcript: " + error);
      if (snapshot) {
        auto replay = snapshot->replay();
        check(!replay.filesystem->status((control.path() / "known-missing.h").string()) && replay.ok(error),
              "known negative lookup is a recorded result");
        check(!replay.filesystem->status((control.path() / "never-recorded.h").string()) && !replay.ok(error),
              "unrecorded replay query is an authority error, not fabricated absence");
      }
    }

    for (const std::vector<std::string> &flags : {
             std::vector<std::string>{"-ivfsoverlay", "/not/a/trusted/overlay"},
             {"-Xclang", "-load", "/not/a/trusted/plugin"},
             {"-include-pch", "/not/a/trusted/pch"}, {"-fmodules"},
             {"-ffast-math"}, {"--target=nvptx64-nvidia-cuda"}, {"-Dnoexcept="}}) {
      auto options = control.options;
      options.compiler_arguments.insert(options.compiler_arguments.end(), flags.begin(), flags.end());
      auto result = fe::admitClosedRegionHost(options, control.path().string());
      check(!result && !result.error.empty(), "untrusted compiler/header control rejects: " + flags.front());
    }
    {
      EnvironmentValue guard("CPATH");
      setenv("CPATH", control.path().c_str(), 1);
      check(!control.admit(), "inherited include-path injection rejects");
    }
    auto wrongCompiler = control.options;
    wrongCompiler.clang_path = "/not/the/configured/compiler";
    check(!fe::admitClosedRegionHost(wrongCompiler, control.path().string()), "unconfigured compiler identity rejects without executing it");
    auto wrongResource = control.options;
    wrongResource.clang_resource_directory = control.path().string();
    check(!fe::admitClosedRegionHost(wrongResource, control.path().string()), "unconfigured builtin-header identity rejects");
  } catch (const std::exception &error) {
    check(false, std::string("fixture harness exception: ") + error.what());
  }
  std::cout << checks << " host-context checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
