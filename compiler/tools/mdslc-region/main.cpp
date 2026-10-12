#include "region_driver_paths.h"
#include "DriverSupport.h"
#include "SourcePattern.h"
#include "program_main.h"
#include <iostream>

namespace {
using namespace matcore::mdslc::driver;
struct Invocation {
  fs::path source, output;
  std::string region;
  codegen::ClosedCpuPolicy policy = codegen::ClosedCpuPolicy::Automatic;
  codegen::ClosedHostOptimization optimization = codegen::ClosedHostOptimization::None;
  std::vector<std::string> host_options;
  bool compile_only = false;
  bool optimization_seen = false;
  unsigned candidate_count = 0;
  std::optional<OutputTileRequest> output_tiles;
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
      ++args.candidate_count;
      args.policy = parseCandidatePolicy(value());
    } else if (argument == "--gemm-output-tiles") {
      if (args.output_tiles) reject("duplicate --gemm-output-tiles");
      args.output_tiles = parseOutputTileRequest(value());
    } else if (argument == "--optimization") {
      if (args.optimization_seen) reject("duplicate --optimization");
      args.optimization_seen = true;
      args.optimization = parseOptimization(value());
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
    reject(std::string("usage: mdslc-region source.mdsl --region qualified_name [-c] [--candidate ") + candidatePolicyUsage() + "] [--optimization " + optimizationUsage() + "] [--gemm-output-tiles M,N] -o NEW_OUTPUT [-- bounded C++ include/macro options]");
  if (args.output_tiles) {
    if (args.candidate_count != 1) reject("--gemm-output-tiles requires exactly one --candidate generated-strict");
    validateSourcePatternInvocation(args.policy, args.optimization, args.compile_only);
  }
  return args;
}

int run(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
#if defined(__aarch64__)
    std::cout << "MDSLC experimental regions: native Linux AArch64 strict-only candidates, Clang/MLIR 21.1.8, "
#else
    std::cout << "MDSLC experimental regions: native Linux x86-64, Clang/MLIR 21.1.8, "
#endif
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
  std::optional<SourcePatternBundle> bundle;
  if (args.output_tiles) bundle.emplace(args.output); // Before host directory capture.
  Staging staging(args.output.parent_path()); // Before any host directory capture.
  write(staging.path / "host.ll", ""); write(staging.path / "result", "");
  const Installation installation;
  auto options = installation.options(args.source, args.host_options);
  auto admitted = frontend::admitExperimentalRegionHost(options, cwd.string(),
      {installation.public_header.path.string(), installation.storage_header.path.string()}, args.region);
  if (!admitted) reject(admitted.error);
  std::optional<SourcePatternInstance> source_pattern;
  if (args.output_tiles)
    source_pattern.emplace(SourcePatternInstance::issue(installation, *bundle,
        *admitted.evidence, args.region, args.policy, args.optimization, *args.output_tiles));
  auto compilation = codegen::compileExperimentalRegionToLLVM(*admitted.evidence,
      source_pattern ? source_pattern->compilerInputs(staging.path / "helper", *admitted.evidence)
                     : installation.compilerInputs(staging.path / "helper"),
      args.policy, args.optimization);
  if (!compilation) reject(compilation.error);
  auto unchanged = [&] {
    std::string error;
    if (!compilation.compilation->inputsUnchanged(error)) reject(error);
  };
  if (source_pattern)
    compileSourcePatternAndPublish(installation, staging, args.output,
        *compilation.compilation, *source_pattern);
  else
    compileAndPublish(installation, staging, args.output, args.compile_only,
        compilation.compilation->llvm_ir, unchanged);
  if (bundle) bundle->retain();
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--program") return runProgram(argc, argv);
    return run(argc, argv);
  }
  catch (const std::exception &error) { std::cerr << "mdslc-region: " << error.what() << '\n'; return 1; }
}
