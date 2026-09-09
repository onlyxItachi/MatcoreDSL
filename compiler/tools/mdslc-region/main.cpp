#include "region_driver_paths.h"
#include "DriverSupport.h"
#include <iostream>

namespace {
using namespace matcore::mdslc::driver;
struct Invocation {
  fs::path source, output;
  std::string region;
  codegen::ClosedCpuPolicy policy = codegen::ClosedCpuPolicy::Automatic;
  std::vector<std::string> host_options;
  bool compile_only = false;
};
Invocation parse(int argc, char **argv) {
  Invocation args;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value = [&] {
      if (++i >= argc || std::string(argv[i]).empty()) reject("missing value for " + argument);
      return std::string(argv[i]);
    };
    if (argument == "-o") { if (!args.output.empty()) reject("duplicate -o"); args.output = value(); }
    else if (argument == "--region") { if (!args.region.empty()) reject("duplicate --region"); args.region = value(); }
    else if (argument == "-c") args.compile_only = true;
    else if (argument == "--candidate") {
      args.policy = parseCandidatePolicy(value());
    } else if (argument == "--") {
      for (++i; i < argc; ++i) {
        const std::string option = argv[i];
        if (option.starts_with("-fsanitize")) reject("sanitizer profile belongs to this compiler installation");
        args.host_options.push_back(option);
      }
    } else if (!argument.starts_with('-') && args.source.empty()) args.source = argument;
    else reject("unknown or unsupported argument: " + argument);
  }
  if (args.source.empty() || args.output.empty() || args.region.empty())
    reject(std::string("usage: mdslc-region source.mdsl --region qualified_name [-c] [--candidate ") + candidatePolicyUsage() + "] -o NEW_OUTPUT [-- bounded C++ include/macro options]");
  return args;
}

int run(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "MDSLC experimental regions: native Linux x86-64, Clang/MLIR 21.1.8, "
              << (REGION_SANITIZED ? "ASan+UBSan" : "uninstrumented") << "; API/ABI not frozen\n";
    return 0;
  }
  auto args = parse(argc, argv);
  const auto cwd = fs::current_path();
  // Preserve OS traversal: symlink/../file is not equivalent to lexical ../
  // cancellation. Admission already records each traversed path identity.
  args.source = fs::absolute(args.source);
  args.output = fs::absolute(args.output);
  validateNewOutput(args.output);
  Staging staging(args.output.parent_path()); // Before any host directory capture.
  write(staging.path / "host.ll", ""); write(staging.path / "result", "");
  const Installation installation;
  auto options = installation.options(args.source, args.host_options);
  auto admitted = frontend::admitExperimentalRegionHost(options, cwd.string(),
      {installation.public_header.path.string(), installation.storage_header.path.string()}, args.region);
  if (!admitted) reject(admitted.error);
  auto compilation = codegen::compileExperimentalRegionToLLVM(*admitted.evidence,
      installation.compilerInputs(staging.path / "helper"), args.policy);
  if (!compilation) reject(compilation.error);
  auto unchanged = [&] {
    std::string error;
    if (!compilation.compilation->inputsUnchanged(error)) reject(error);
  };
  compileAndPublish(installation, staging, args.output, args.compile_only,
      compilation.compilation->llvm_ir, unchanged);
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try { return run(argc, argv); }
  catch (const std::exception &error) { std::cerr << "mdslc-region: " << error.what() << '\n'; return 1; }
}
