#include "FrozenHostCodegen.h"
#include "../../lib/support/platform_support.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fe = matcore::mdslc::frontend;
namespace host = fe::closed_region_host;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
namespace {
unsigned checks = 0, failures = 0;
void check(bool value, const std::string &reason) {
  ++checks;
  if (!value) { ++failures; std::cerr << "FAIL: " << reason << '\n'; }
}
void write(const fs::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << text;
  output.close();
  if (!output) throw std::runtime_error("fixture write failed");
}
support::ProcessResultV1 run(const std::vector<std::string> &args,
                             const fs::path &cwd) {
  support::ProcessRequestV1 request;
  request.argv = args;
  request.working_directory = cwd;
  request.environment = support::compiler_environment_sanitization_v1();
  return support::run_process_v1(request);
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  std::string error;
  auto directory = support::create_temp_directory_v1("mdslc-frozen-codegen", error);
  if (!directory) throw std::runtime_error(error);
  auto artifacts = support::create_temp_directory_v1("mdslc-frozen-artifacts", error);
  if (!artifacts) throw std::runtime_error(error);
  const auto source = directory->path() / "source.cpp";
  const auto dependency = directory->path() / "host.h";
  // The helper consumer tests compiler mechanics only: no mathematical source
  // is admitted and no seal/region authority is constructed by this fixture.
  write(dependency, "#pragma once\nconstexpr int host_value = 47;\n");
  const std::string text =
      "#include <cstdio>\n#include <source_location>\n#include \"host.h\"\n"
      "void region() {} constexpr unsigned column = __builtin_COLUMN();\n"
      "int effects = 0; struct Host { Host(){++effects;} ~Host(){++effects;} };\n"
      "int main(){ { Host host; } std::printf(\"%u %u %d %d\\n\", column, "
      "std::source_location::current().line(), host_value, effects); }\n";
  write(source, text);
  fe::Options options;
  options.input_path = source.string();
  options.clang_path = argv[1];
  options.clang_resource_directory = argv[2];
  options.compiler_arguments = {"-I" + directory->path().string()};
  auto capture = host::prepareHostInputs(options, directory->path().string(),
      {{"/__mdsl_private__/fixture.h", "#pragma once\n"}}, error);
  check(bool(capture), "recording inputs: " + error);
  if (!capture) return 1;
  clang::FileSystemOptions file_options;
  file_options.WorkingDir = directory->path().string();
  auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(
      file_options, capture->fileSystem());
  clang::tooling::ToolInvocation syntax(capture->arguments(),
      std::make_unique<clang::SyntaxOnlyAction>(), files.get());
  check(syntax.run(), "initial host Sema captures exact dependencies");
  auto snapshot = capture->freeze(error);
  check(bool(snapshot), "freeze initial compiler inputs: " + error);
  if (!snapshot) return 1;
  std::string ir;
  check(matcore::mdslc::codegen::detail::compileFrozenHostToLLVM(*snapshot, ir, error),
        "Clang code generation on immutable original host: " + error);
  if (ir.empty()) return 1;
  check(ir.find("define") != std::string::npos && ir.find("@main") != std::string::npos,
        "real host LLVM function definitions were produced");
  // A first prototype wrote into the captured source search directory and
  // correctly invalidated its recorded directory metadata. Stage outside that
  // closure: do not weaken input authentication to accommodate compiler output.
  const auto generated = artifacts->path() / "host.ll";
  const auto compiled = artifacts->path() / "host";
  const auto ordinary = artifacts->path() / "ordinary";
  write(generated, ir);
  auto object = run({argv[1], "-x", "ir", generated.string(), "-o", compiled.string()}, directory->path());
  check(object.launched && object.exit_code == 0, "host LLVM links: " + object.stderr_text);
  auto direct = run({argv[1], "-x", "c++", "-std=c++20", source.string(), "-o", ordinary.string()}, directory->path());
  check(direct.launched && direct.exit_code == 0, "ordinary Clang control links: " + direct.stderr_text);
  auto actual = run({compiled.string()}, directory->path());
  auto expected = run({ordinary.string()}, directory->path());
  check(actual.exit_code == 0 && expected.exit_code == 0 &&
        actual.stdout_text == expected.stdout_text && actual.stdout_text == "46 6 47 2\n",
        "original columns, source_location, header semantics and host RAII match");
  check(snapshot->unchanged(error),
        "compiler-owned artifacts do not invalidate unchanged host inputs: " + error);
  // Later dependency mutation must not be silently accepted even though frozen
  // code generation could still reconstruct the formerly admitted source.
  write(dependency, "#pragma once\nconstexpr int host_value = 48;\n");
  check(!matcore::mdslc::codegen::detail::compileFrozenHostToLLVM(*snapshot, ir, error) && ir.empty(),
        "changed dependency prevents any LLVM result publication");
  std::cout << checks << " frozen host checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
