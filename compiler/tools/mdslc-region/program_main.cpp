#include "program_main.h"
#include "DriverSupport.h"
#include "ExperimentalProgramCompiler.h"
#include <iostream>

namespace matcore::mdslc::driver {
namespace {
struct ProgramInvocation {
  struct Source { fs::path path; std::string region; };
  std::vector<Source> sources;
  fs::path output;
  codegen::ClosedCpuPolicy policy = codegen::ClosedCpuPolicy::Automatic;
  std::vector<std::string> host_options;
  bool compile_only = false;
};
ProgramInvocation parseProgram(int argc, char **argv) {
  ProgramInvocation result;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    auto next = [&] {
      if (++i >= argc || std::string(argv[i]).empty()) reject("missing value for " + argument);
      return std::string(argv[i]);
    };
    if (argument == "--host") result.sources.push_back({fs::absolute(next()), ""});
    else if (argument == "--region") {
      auto source = fs::absolute(next());
      result.sources.push_back({source, next()});
    } else if (argument == "-o") {
      if (!result.output.empty()) reject("duplicate -o");
      result.output = fs::absolute(next());
    } else if (argument == "-c") result.compile_only = true;
    else if (argument == "--candidate") result.policy = parseCandidatePolicy(next());
    else if (argument == "--") {
      for (++i; i < argc; ++i) {
        const std::string option = argv[i];
        if (option.starts_with("-fsanitize")) reject("sanitizer profile belongs to this compiler installation");
        result.host_options.push_back(option);
      }
    } else reject("unknown or unsupported program argument: " + argument);
  }
  if (result.sources.size() < 2 || result.sources.size() > 8 || result.output.empty())
    reject(std::string("usage: mdslc-region --program --host SOURCE --region SOURCE NAME "
      "[--host SOURCE | --region SOURCE NAME ...] [-c] [--candidate ") +
      candidatePolicyUsage() + "] -o NEW_OUTPUT [-- bounded C++ include/macro options]; 2..8 TUs");
  return result;
}
} // namespace

int runProgram(int argc, char **argv) {
  auto args = parseProgram(argc, argv);
  validateNewOutput(args.output);
  Staging staging(args.output.parent_path());
  write(staging.path / "host.ll", ""); write(staging.path / "result", "");
  const Installation installation;
  std::vector<codegen::ExperimentalProgramSource> sources;
  for (const auto &source : args.sources)
    sources.push_back({installation.options(source.path, args.host_options), source.region});
  auto compilation = codegen::compileExperimentalProgramToLLVM(sources, fs::current_path().string(),
      installation.compilerInputs(staging.path / "helper"), args.policy);
  if (!compilation) reject(compilation.error);
  auto unchanged = [&] {
    std::string error;
    if (!compilation.compilation->inputsUnchanged(error)) reject(error);
  };
  compileAndPublish(installation, staging, args.output, args.compile_only,
      compilation.compilation->llvm_ir, unchanged);
  std::cout << "PUBLISHED authenticated multi-source program\n";
  return 0;
}
} // namespace matcore::mdslc::driver
