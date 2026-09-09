#include "ExperimentalProgramCompiler.h"
#include "platform_support.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace cg = matcore::mdslc::codegen;
namespace fe = matcore::mdslc::frontend;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
namespace {
void write(const fs::path &path, const std::string &bytes) {
  std::ofstream file(path, std::ios::binary);
  file << bytes; file.close();
  if (!file) throw std::runtime_error("test fixture write failed");
}
std::string read(const fs::path &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) throw std::runtime_error("test fixture read failed");
  return (*buffer)->getBuffer().str();
}
struct CaptureOutput {
  std::ostringstream text;
  std::streambuf *original = std::cout.rdbuf(text.rdbuf());
  ~CaptureOutput() { std::cout.rdbuf(original); }
};
}
int main(int argc, char **argv) {
  if (argc != 6 && argc != 7) return 2;
  const bool sanitized = argc == 7 && std::string(argv[6]) == "--asan";
  const fs::path compiler = fs::absolute(argv[3]);
  auto candidate = llvm::MemoryBuffer::getFile(argv[4]);
  auto runtime = llvm::MemoryBuffer::getFile(argv[5]);
  if (!candidate || !runtime) return 2;
  unsigned checks = 0;
  for (const std::string mode : {"host", "region", "header", "negative_lookup", "missing_candidate", "duplicate_candidate", "no_prelude"}) {
    std::string error;
    auto source_dir = support::create_temp_directory_v1("mdslc-program-mutation", error);
    auto stage = support::create_temp_directory_v1("mdslc-program-stage", error);
    if (!source_dir || !stage) throw std::runtime_error(error);
    const auto root = source_dir->path();
    for (const auto &file : fs::directory_iterator(compiler / "tests/program/fixtures"))
      if (file.is_regular_file())
        fs::copy_file(file.path(), root / file.path().filename());
    const auto main_path = root / "main.cpp";
    write(main_path, "#if __has_include(\"optional_program_header.h\")\n"
      "#include \"optional_program_header.h\"\n#endif\n" + read(main_path));
    std::vector<cg::ExperimentalProgramSource> sources;
    for (const auto &[name, selected] : std::vector<std::pair<std::string, std::string>>{
        {"main.cpp", ""}, {"first.mdsl", "first"}, {"second.mdsl", "second"}}) {
      fe::Options options;
      options.input_path = (root / name).string();
      options.clang_path = argv[1]; options.clang_resource_directory = argv[2];
      options.compiler_arguments = {"-I" + (compiler / "include").string()};
      if (sanitized) options.compiler_arguments.push_back("-fsanitize=address,undefined");
      sources.push_back({options, selected});
    }
    cg::ExperimentalCompilerInputs inputs{argv[1], argv[2], compiler / "include",
      compiler / "lib/runtime/closed_host_v1.h", stage->path(), sanitized,
      {{cg::SymbolArtifactOwner::MatcoreRuntime, (*runtime)->getMemBufferRef()},
       {cg::SymbolArtifactOwner::PrivateCandidates, (*candidate)->getMemBufferRef()}}};
    if (mode == "missing_candidate") inputs.symbol_artifacts.pop_back();
    if (mode == "duplicate_candidate") inputs.symbol_artifacts.push_back(inputs.symbol_artifacts.back());
    bool hook = false;
    cg::ExperimentalProgramCompilationResult result;
    std::string output;
    {
      CaptureOutput capture;
      result = cg::compileExperimentalProgramToLLVMForTesting(sources, root.string(), inputs,
        cg::ClosedCpuPolicy::GeneratedStrict, [&] {
          hook = true;
          if (mode == "no_prelude") return;
          if (mode == "negative_lookup") write(root / "optional_program_header.h", "#pragma once\n");
          else {
            const auto target = root / (mode == "host" ? "main.cpp" : mode == "region" ? "first.mdsl" : "api.h");
            write(target, read(target) + "\n// changed after complete interface capture\n");
          }
        });
      output = capture.text.str();
    }
    if (mode == "no_prelude") {
      if (!result || !hook || result.compilation->inputs.size() != sources.size() + 1) {
        std::cerr << "FAIL no-prelude compilation: " << result.error << '\n'; return 1;
      }
      for (std::size_t i = 0; i < sources.size(); ++i) {
        // The first snapshot belongs to the compiler-issued interface witness.
        const auto &snapshot = *result.compilation->inputs[i + 1];
        const auto &args = snapshot.arguments();
        if (snapshot.sourceSnapshot() != read(sources[i].options.input_path) ||
            args.back() != sources[i].options.input_path ||
            std::find(args.begin(), args.end(), "-include") != args.end()) {
          std::cerr << "FAIL original program source capture was polluted\n"; return 1;
        }
      }
      ++checks;
      continue;
    }
    const bool artifact_case = mode == "missing_candidate" || mode == "duplicate_candidate";
    const bool expected = artifact_case
      ? !hook && result.error.find("isolated private candidate DSO") != std::string::npos
      : hook && result.error.find("captured") != std::string::npos &&
                result.error.find("changed") != std::string::npos;
    if (result || !expected || output.find("VALIDATED program") != std::string::npos ||
        output.find("CROSS_TU_LINK") != std::string::npos) {
      std::cerr << "FAIL " << mode << ": " << result.error << "\n" << output;
      return 1;
    }
    ++checks;
  }
  std::cout << "PASS program input mutation/artifact gates " << checks << " checks\n";
  return checks == 7 ? 0 : 1;
}
