#include "../../lib/frontend/frontend.h"
#include "../../lib/codegen/codegen.h"
#include "../../lib/ir/matcore_ir_v1.h"
#if MDSLC_HAS_MATCORE_MLIR
#include "../../lib/mlir/MatcoreCpuRuntimeLowering.h"
#include "../../lib/mlir/MatcoreStructuredGemmHandoff.h"
#include "../../lib/mlir/MatcoreV1Bridge.h"
#endif
#include "platform_support.h"
#include "mdslc_config.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef MDSLC_INSTALL_BINDIR
#define MDSLC_INSTALL_BINDIR "bin"
#endif
#ifndef MDSLC_INSTALL_INCLUDEDIR
#define MDSLC_INSTALL_INCLUDEDIR "include"
#endif

namespace {

namespace support = matcore::mdslc::support;

enum class SemanticPipeline {
  CaptureV0,
  MatcoreMlirCpuV1,
};

struct CommandLine {
  matcore::mdslc::frontend::Options frontend;
  std::string frontend_name = "native";
  std::string tool_include_directory;
  std::string ir_output;
  std::string rewrite_output;
  std::string sites_output;
  std::string stubs_output;
  std::string backend_output;
  std::string semantic_ir_output;
  std::string structured_ir_output;
  std::string verify_ir;
  std::string compiler_arguments_file;
  std::string recovered_gemm_report;
  std::uint32_t ir_version = matcore::mdslc::ir::kMatcoreIrVersion;
  SemanticPipeline semantic_pipeline = SemanticPipeline::CaptureV0;
  bool semantic_pipeline_was_explicit = false;
  bool compiler_was_explicit = false;
  bool ir_version_was_explicit = false;
};

bool isWindowsHost() {
  return support::process_launch_backend_v1() ==
         support::ProcessLaunchBackendV1::windows_create_process_w;
}

std::optional<std::filesystem::path> pathFromUtf8(std::string_view value,
                                                  std::string_view role) {
  std::string error;
  std::optional<std::filesystem::path> path =
      support::path_from_utf8_v1(value, error);
  if (!path) {
    std::cerr << "matcore-extract: invalid UTF-8 " << role << " path: "
              << error << '\n';
  }
  return path;
}

std::optional<std::string> pathToUtf8(const std::filesystem::path &path,
                                      std::string_view role) {
  std::string error;
  std::optional<std::string> encoded =
      support::path_to_utf8_v1(path, error);
  if (!encoded) {
    std::cerr << "matcore-extract: cannot represent " << role
              << " path as UTF-8: " << error << '\n';
  }
  return encoded;
}

std::filesystem::path normalizedPath(const std::filesystem::path &path) {
  std::string error;
  const std::filesystem::path normalized =
      support::normalize_path_v1(path, true, error);
  return error.empty() ? normalized : path.lexically_normal();
}

bool pathsReferToSameLocation(const std::filesystem::path &left,
                              const std::filesystem::path &right) {
  std::string error;
  const bool same =
      support::paths_refer_to_same_location_v1(left, right, error);
  if (!error.empty()) {
    std::cerr << "matcore-extract: cannot authenticate prospective path "
                 "identity: "
              << error << '\n';
    return true;
  }
  return same;
}

std::optional<std::filesystem::path>
installedPrefixForExecutable(const std::filesystem::path &executable) {
  const std::filesystem::path bindir(MDSLC_INSTALL_BINDIR);
  if (bindir.empty() || bindir.is_absolute()) {
    return std::nullopt;
  }
  std::vector<std::filesystem::path> components;
  for (const std::filesystem::path &component : bindir) {
    if (component != "." && !component.empty()) {
      components.push_back(component);
    }
  }
  std::filesystem::path cursor = executable.parent_path();
  for (auto iterator = components.rbegin(); iterator != components.rend();
       ++iterator) {
    if (cursor.filename() != *iterator) {
      return std::nullopt;
    }
    cursor = cursor.parent_path();
  }
  return cursor;
}

std::optional<std::filesystem::path>
discoverToolIncludeDirectory(const std::filesystem::path &executable) {
  std::vector<std::filesystem::path> candidates;
  if (const auto prefix = installedPrefixForExecutable(executable)) {
    candidates.push_back(*prefix / MDSLC_INSTALL_INCLUDEDIR);
  }
  // Standalone build-tree layout remains bin/ plus include/ even when a
  // non-default install bindir was configured.
  candidates.push_back(executable.parent_path().parent_path() / "include");
  for (const std::filesystem::path &candidate : candidates) {
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate / "matcore" / "mdsl.h",
                                         error)) {
      return normalizedPath(candidate);
    }
  }
  return std::nullopt;
}

bool validateOutputPaths(const CommandLine &command) {
  const std::vector<std::pair<std::string_view, std::string_view>> outputs = {
      {"--ir-out", command.ir_output},
      {"--rewrite-out", command.rewrite_output},
      {"--sites-out", command.sites_output},
      {"--stubs-out", command.stubs_output},
      {"--backend-out", command.backend_output},
      {"--semantic-ir-out", command.semantic_ir_output},
      {"--structured-ir-out", command.structured_ir_output},
      {"--inspect-recovered-gemm", command.recovered_gemm_report},
  };
  std::vector<std::pair<std::string_view, std::filesystem::path>> validated;
  std::optional<std::string_view> standard_output_owner;
  for (const auto &[option, encoded_path] : outputs) {
    if (encoded_path.empty()) {
      continue;
    }
    if (encoded_path == "-") {
      if (standard_output_owner) {
        std::cerr << "matcore-extract: " << option << " and "
                  << *standard_output_owner
                  << " cannot both write to standard output\n";
        return false;
      }
      standard_output_owner = option;
      continue;
    }
    const std::optional<std::filesystem::path> path =
        pathFromUtf8(encoded_path, "output");
    const std::optional<std::filesystem::path> input =
        pathFromUtf8(command.frontend.input_path, "input");
    if (!path || !input) return false;
    std::string output_error;
    if (!support::prospective_output_path_supported_v1(*path, output_error)) {
      std::cerr << "matcore-extract: " << option
                << " is not a supported output path: " << output_error
                << '\n';
      return false;
    }
    if (pathsReferToSameLocation(*input, *path)) {
      std::cerr << "matcore-extract: " << option
                << " must not overwrite or alias the input .mdsl file\n";
      return false;
    }
    for (const auto &[prior_option, prior_path] : validated) {
      if (pathsReferToSameLocation(prior_path, *path)) {
        std::cerr << "matcore-extract: " << option << " and " << prior_option
                  << " must refer to distinct output files\n";
        return false;
      }
    }
    validated.emplace_back(option, *path);
  }
  return true;
}

void usage(std::ostream &output) {
  output
      << "usage: matcore-extract --input FILE.mdsl --ir-out FILE.json [options] "
         "-- [clang-driver-placeholder] COMPILE_ARGS\n"
      << "\n"
      << "Frontend selection:\n"
      << "  --frontend=native             supported Clang LibTooling frontend "
         "(default)\n"
      << "  --frontend=ast-json-bootstrap compatibility/differential frontend\n"
      << "\n"
      << "Frontend options:\n"
      << "  --clang PATH          exact coherent Clang "
      << MDSLC_TOOLCHAIN_VERSION << " executable\n"
      << "  --ast-byte-limit N    maximum captured AST JSON bytes\n"
      << "  --verbose             print the exact Clang command\n"
      << "  --rewrite-out FILE    rewritten host C++ (requires all outputs)\n"
      << "  --sites-out FILE      generated C++ site declarations\n"
      << "  --stubs-out FILE      generated C++ descriptor stubs\n"
      << "  --backend-out FILE    generated C ABI backend forwarding entries\n"
      << "  --semantic-pipeline=capture-v0|matcore-mlir\n"
         "                         choose the executed backend producer; "
         "matcore-mlir is explicit\n"
      << "  --semantic-ir-out FILE\n"
         "                         write verified Matcore MLIR inspection "
         "text\n"
      << "  --structured-ir-out FILE\n"
         "                         write verified analysis-only structured "
         "GEMM MLIR\n"
      << "  --ir-version N        emit Matcore IR 0 (default) or typed IR 1\n"
      << "  --verify-ir FILE      verify serialized Matcore IR v0/v1 and exit\n"
      << "  --frontend-info       describe the built frontend modes\n"
      << "  --inspect-recovered-gemm FILE\n"
         "                         native-only ordinary-C++ GEMM inspection; "
         "never rewrites\n"
      << "  --compiler-arguments-file FILE\n"
         "                         bounded v1 argv transport used by mdslc++\n"
      << "\n"
      << "A bare clang++ token after -- is a command-shape placeholder; the "
         "configured\n"
      << "Clang " << MDSLC_TOOLCHAIN_VERSION
      << " executable remains in use. Pass --clang PATH to override it "
         "explicitly.\n";
}

bool takeValue(int argc, char **argv, int &index, std::string &destination,
               std::string_view option) {
  if (index + 1 == argc) {
    std::cerr << "matcore-extract: " << option << " requires a value\n";
    return false;
  }
  destination = argv[++index];
  return true;
}

bool setIrVersion(CommandLine &command, std::string_view value) {
  if (value == "0") {
    command.ir_version = matcore::mdslc::ir::kMatcoreIrVersion;
  } else if (value == "1") {
    command.ir_version = matcore::mdslc::ir::v1::kMatcoreIrVersion;
  } else {
    std::cerr << "matcore-extract: unsupported --ir-version value: " << value
              << " (expected 0 or 1)\n";
    return false;
  }
  command.ir_version_was_explicit = true;
  return true;
}

bool setSemanticPipeline(CommandLine &command, std::string_view value) {
  if (command.semantic_pipeline_was_explicit) {
    std::cerr << "matcore-extract: --semantic-pipeline may be specified only "
                 "once\n";
    return false;
  }
  if (value == "capture-v0") {
    command.semantic_pipeline = SemanticPipeline::CaptureV0;
  } else if (value == "matcore-mlir") {
    command.semantic_pipeline = SemanticPipeline::MatcoreMlirCpuV1;
  } else {
    std::cerr << "matcore-extract: unsupported --semantic-pipeline value: "
              << value << " (expected capture-v0 or matcore-mlir)\n";
    return false;
  }
  command.semantic_pipeline_was_explicit = true;
  return true;
}

struct ConfiguredCompiler {
  std::filesystem::path invocation_path;
  std::filesystem::path resource_directory;
};

std::optional<ConfiguredCompiler>
discoverConfiguredCompiler(std::string_view requested_compiler,
                           bool require_prefix_coherence) {
  std::string error;
  const std::string_view requested = requested_compiler.empty()
                                         ? (isWindowsHost()
                                                ? std::string_view("clang-cl.exe")
                                                : std::string_view(
                                                      MDSLC_DEFAULT_CLANGXX))
                                         : requested_compiler;
  const std::optional<std::filesystem::path> discovered =
      support::find_executable_v1(requested, error);
  if (!discovered) {
    std::cerr << "matcore-extract: cannot locate the configured Clang driver: "
              << error << '\n';
    return std::nullopt;
  }
  std::filesystem::path compiler = *discovered;
  const std::optional<std::string> compiler_utf8 =
      pathToUtf8(compiler, "compiler");
  if (!compiler_utf8) return std::nullopt;
  support::ProcessRequestV1 request;
  request.argv = {*compiler_utf8};
  if (isWindowsHost()) request.argv.emplace_back("--no-default-config");
  request.argv.emplace_back("--version");
  request.environment = support::compiler_environment_sanitization_v1();
  const support::ProcessResultV1 result = support::run_process_v1(request);
  if (!result.launched || !result.error.empty() || result.exit_code != 0 ||
      (result.stdout_text + result.stderr_text)
              .find("clang version " MDSLC_TOOLCHAIN_VERSION) ==
          std::string::npos) {
    std::cerr << "matcore-extract: compiler must be the coherent Clang "
              << MDSLC_TOOLCHAIN_VERSION << " driver: "
              << compiler << '\n';
    return std::nullopt;
  }
  support::ProcessRequestV1 resource_request;
  resource_request.argv = {*compiler_utf8};
  if (isWindowsHost()) {
    resource_request.argv.emplace_back("--no-default-config");
  }
  resource_request.argv.emplace_back("-print-resource-dir");
  resource_request.environment =
      support::compiler_environment_sanitization_v1();
  const support::ProcessResultV1 resource_result =
      support::run_process_v1(resource_request);
  if (!resource_result.launched || !resource_result.error.empty() ||
      resource_result.exit_code != 0) {
    std::cerr << "matcore-extract: cannot query the Clang "
              << MDSLC_TOOLCHAIN_VERSION << " resource directory\n";
    return std::nullopt;
  }
  std::string resource_text = resource_result.stdout_text;
  while (!resource_text.empty() &&
         (resource_text.back() == '\r' || resource_text.back() == '\n' ||
          resource_text.back() == ' ' || resource_text.back() == '\t')) {
    resource_text.pop_back();
  }
  const std::optional<std::filesystem::path> resource_input =
      pathFromUtf8(resource_text, "Clang resource");
  if (!resource_input) return std::nullopt;
  const std::filesystem::path resource_directory =
      support::normalize_path_v1(*resource_input, true, error);
  if (!error.empty() || !std::filesystem::is_regular_file(
                            resource_directory / "include" / "stddef.h")) {
    std::cerr << "matcore-extract: coherent Clang resource headers are "
                 "unavailable at "
              << resource_directory << '\n';
    return std::nullopt;
  }
  const std::filesystem::path compiler_prefix = normalizedPath(
      compiler.parent_path().parent_path());
  std::error_code relative_error;
  const std::filesystem::path relative_resource = std::filesystem::relative(
      resource_directory, compiler_prefix, relative_error);
  if (require_prefix_coherence &&
      (relative_error || relative_resource.empty() ||
       relative_resource.is_absolute() || *relative_resource.begin() == "..")) {
    std::cerr << "matcore-extract: Clang resource directory is outside the "
                 "selected "
              << MDSLC_TOOLCHAIN_VERSION << " toolchain prefix\n";
    return std::nullopt;
  }
  return ConfiguredCompiler{compiler, resource_directory};
}

bool selectCompilerArgument(CommandLine &command) {
  if (command.frontend.compiler_arguments.empty() ||
      command.frontend.compiler_arguments.front().starts_with("-") ||
      (isWindowsHost() &&
       command.frontend.compiler_arguments.front().starts_with("/"))) {
    return true;
  }
  const std::optional<std::filesystem::path> candidate = pathFromUtf8(
      command.frontend.compiler_arguments.front(), "compiler");
  if (!candidate) return false;
  const std::optional<std::string> filename_utf8 =
      pathToUtf8(candidate->filename(), "compiler filename");
  if (!filename_utf8) return false;
  const std::string &filename = *filename_utf8;
  if (filename.starts_with("clang")) {
    if (command.compiler_was_explicit) {
      std::cerr << "matcore-extract: specify the compiler either with "
                   "--clang or after --, not both\n";
      return false;
    }
    const bool canonical_placeholder =
        (!isWindowsHost() && filename == "clang++") ||
        (isWindowsHost() &&
         (filename == "clang-cl" || filename == "clang-cl.exe"));
    if (candidate->has_parent_path() || !canonical_placeholder) {
      command.frontend.clang_path =
          command.frontend.compiler_arguments.front();
      command.compiler_was_explicit = true;
    }
    command.frontend.compiler_arguments.erase(
        command.frontend.compiler_arguments.begin());
    return true;
  }
  if (filename.starts_with("g++") || filename.starts_with("c++")) {
    std::cerr << "matcore-extract: MDSLC extraction requires Clang; "
                 "use --clang with a compatible clang++ executable\n";
    return false;
  }
  return true;
}

std::optional<CommandLine> parseCommandLine(int argc, char **argv) {
  CommandLine command;
  bool after_separator = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (after_separator) {
      command.frontend.compiler_arguments.push_back(argument);
      continue;
    }
    if (argument == "--") {
      after_separator = true;
    } else if (argument == "--input") {
      if (!takeValue(argc, argv, index, command.frontend.input_path,
                     "--input")) {
        return std::nullopt;
      }
    } else if (argument == "--ir-out") {
      if (!takeValue(argc, argv, index, command.ir_output, "--ir-out")) {
        return std::nullopt;
      }
    } else if (argument == "--rewrite-out") {
      if (!takeValue(argc, argv, index, command.rewrite_output,
                     "--rewrite-out")) {
        return std::nullopt;
      }
    } else if (argument == "--sites-out") {
      if (!takeValue(argc, argv, index, command.sites_output, "--sites-out")) {
        return std::nullopt;
      }
    } else if (argument == "--stubs-out") {
      if (!takeValue(argc, argv, index, command.stubs_output, "--stubs-out")) {
        return std::nullopt;
      }
    } else if (argument == "--backend-out") {
      if (!takeValue(argc, argv, index, command.backend_output,
                     "--backend-out")) {
        return std::nullopt;
      }
    } else if (argument == "--semantic-ir-out") {
      if (!takeValue(argc, argv, index, command.semantic_ir_output,
                     "--semantic-ir-out")) {
        return std::nullopt;
      }
    } else if (argument == "--structured-ir-out") {
      if (!takeValue(argc, argv, index, command.structured_ir_output,
                     "--structured-ir-out")) {
        return std::nullopt;
      }
    } else if (argument == "--semantic-pipeline") {
      std::string value;
      if (!takeValue(argc, argv, index, value, "--semantic-pipeline") ||
          !setSemanticPipeline(command, value)) {
        return std::nullopt;
      }
    } else if (argument.starts_with("--semantic-pipeline=")) {
      if (!setSemanticPipeline(
              command,
              argument.substr(std::string("--semantic-pipeline=").size()))) {
        return std::nullopt;
      }
    } else if (argument == "--verify-ir") {
      if (!takeValue(argc, argv, index, command.verify_ir, "--verify-ir")) {
        return std::nullopt;
      }
    } else if (argument == "--compiler-arguments-file") {
      if (!takeValue(argc, argv, index, command.compiler_arguments_file,
                     "--compiler-arguments-file")) {
        return std::nullopt;
      }
    } else if (argument == "--inspect-recovered-gemm") {
      if (!takeValue(argc, argv, index, command.recovered_gemm_report,
                     "--inspect-recovered-gemm")) {
        return std::nullopt;
      }
    } else if (argument == "--ir-version") {
      std::string value;
      if (!takeValue(argc, argv, index, value, "--ir-version") ||
          !setIrVersion(command, value)) {
        return std::nullopt;
      }
    } else if (argument.starts_with("--ir-version=")) {
      if (!setIrVersion(
              command,
              argument.substr(std::string("--ir-version=").size()))) {
        return std::nullopt;
      }
    } else if (argument == "--clang") {
      if (!takeValue(argc, argv, index, command.frontend.clang_path,
                     "--clang")) {
        return std::nullopt;
      }
      command.compiler_was_explicit = true;
    } else if (argument.starts_with("--clang=")) {
      command.frontend.clang_path =
          argument.substr(std::string("--clang=").size());
      if (command.frontend.clang_path.empty()) {
        std::cerr << "matcore-extract: --clang requires a non-empty path\n";
        return std::nullopt;
      }
      command.compiler_was_explicit = true;
    } else if (argument == "--frontend") {
      if (!takeValue(argc, argv, index, command.frontend_name,
                     "--frontend")) {
        return std::nullopt;
      }
    } else if (argument.starts_with("--frontend=")) {
      command.frontend_name = argument.substr(std::string("--frontend=").size());
    } else if (argument == "--ast-byte-limit") {
      std::string value;
      if (!takeValue(argc, argv, index, value, "--ast-byte-limit")) {
        return std::nullopt;
      }
      try {
        command.frontend.maximum_ast_bytes = std::stoull(value);
      } catch (...) {
        std::cerr << "matcore-extract: invalid --ast-byte-limit value\n";
        return std::nullopt;
      }
    } else if (argument == "--verbose") {
      command.frontend.verbose = true;
    } else if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      std::exit(0);
    } else if (argument == "--frontend-info") {
      std::cout << "default: native\n";
#if MDSLC_HAS_NATIVE_FRONTEND
      std::cout << "native [built]: clang-libtooling-v1; in-process Clang "
                << MDSLC_TOOLCHAIN_VERSION
                << " PPCallbacks, parse/Sema, ASTMatcher, canonical declaration "
                   "and AnnotateAttr authentication, SourceManager ranges\n";
#else
      std::cout << "native [not built]\n";
#endif
#if MDSLC_HAS_BOOTSTRAP_FRONTEND
      if (isWindowsHost()) {
        std::cout << "ast-json-bootstrap [unavailable on Windows]\n";
      } else {
        std::cout << "ast-json-bootstrap [built, compatibility-only]: "
                     "clang-ast-json-bootstrap-v0\n";
      }
#else
      std::cout << "ast-json-bootstrap [not built]\n";
#endif
#if MDSLC_HAS_MATCORE_MLIR
      std::cout << "semantic-pipeline matcore-mlir [built]: verified Matcore "
                   "IR v1 bridge, analysis-only structured GEMM handoff, and "
                   "CPU runtime-dispatch lowering\n";
#else
      std::cout << "semantic-pipeline matcore-mlir [not built]\n";
#endif
      std::exit(0);
    } else {
      std::cerr << "matcore-extract: unknown tool option: " << argument
                << "\n";
      return std::nullopt;
    }
  }

  if (!command.verify_ir.empty()) {
    if (!command.frontend.input_path.empty() || !command.ir_output.empty() ||
        !command.rewrite_output.empty() || !command.sites_output.empty() ||
        !command.stubs_output.empty() || !command.backend_output.empty() ||
        !command.semantic_ir_output.empty() ||
        !command.structured_ir_output.empty() ||
        !command.recovered_gemm_report.empty() ||
        !command.compiler_arguments_file.empty() ||
        !command.frontend.compiler_arguments.empty() ||
        command.ir_version_was_explicit ||
        command.semantic_pipeline_was_explicit) {
      std::cerr << "matcore-extract: --verify-ir cannot be combined with "
                   "extraction or generation options\n";
      return std::nullopt;
    }
    return command;
  }
  if (command.frontend_name != "native" &&
      command.frontend_name != "ast-json-bootstrap") {
    std::cerr << "matcore-extract: unsupported --frontend value: "
              << command.frontend_name << '\n';
    return std::nullopt;
  }
  if (!command.recovered_gemm_report.empty() &&
      command.frontend_name != "native") {
    std::cerr << "matcore-extract: --inspect-recovered-gemm is available only "
                 "with --frontend=native\n";
    return std::nullopt;
  }
  if (command.semantic_pipeline == SemanticPipeline::MatcoreMlirCpuV1) {
    if (command.frontend_name != "native") {
      std::cerr << "matcore-extract: the Matcore MLIR semantic pipeline "
                   "requires the authenticated native frontend\n";
      return std::nullopt;
    }
    if (!command.ir_version_was_explicit ||
        command.ir_version != matcore::mdslc::ir::v1::kMatcoreIrVersion) {
      std::cerr << "matcore-extract: --semantic-pipeline=matcore-mlir requires "
                   "--ir-version=1\n";
      return std::nullopt;
    }
    if (!command.recovered_gemm_report.empty()) {
      std::cerr << "matcore-extract: recovered GEMM inspection is analysis-only "
                   "and cannot enter the executable Matcore MLIR pipeline\n";
      return std::nullopt;
    }
#if !MDSLC_HAS_MATCORE_MLIR
    std::cerr << "matcore-extract: Matcore MLIR support was not built; "
                 "reconfigure with MDSLC_ENABLE_MATCORE_MLIR=ON and a coherent "
                 "MLIR "
              << MDSLC_TOOLCHAIN_VERSION << " package\n";
    return std::nullopt;
#endif
  } else {
    if (!command.semantic_ir_output.empty()) {
      std::cerr << "matcore-extract: --semantic-ir-out requires "
                   "--semantic-pipeline=matcore-mlir\n";
      return std::nullopt;
    }
    if (!command.structured_ir_output.empty()) {
      std::cerr << "matcore-extract: --structured-ir-out requires "
                   "--semantic-pipeline=matcore-mlir\n";
      return std::nullopt;
    }
  }
  if (!command.compiler_arguments_file.empty()) {
    if (!command.frontend.compiler_arguments.empty()) {
      std::cerr << "matcore-extract: --compiler-arguments-file cannot be "
                   "combined with arguments after --\n";
      return std::nullopt;
    }
    const std::optional<std::filesystem::path> argument_path = pathFromUtf8(
        command.compiler_arguments_file, "compiler arguments");
    if (!argument_path) return std::nullopt;
    std::string argument_error;
    const std::optional<std::vector<std::string>> arguments =
        support::read_argument_file_v1(*argument_path, argument_error);
    if (!arguments) {
      std::cerr << "matcore-extract: cannot read compiler arguments file: "
                << argument_error << '\n';
      return std::nullopt;
    }
    command.frontend.compiler_arguments = *arguments;
  }
  if (isWindowsHost() && command.frontend_name == "ast-json-bootstrap") {
    std::cerr << "matcore-extract: the AST-JSON bootstrap frontend is not "
                 "available on Windows; use native LibTooling\n";
    return std::nullopt;
  }
  if (command.frontend.input_path.empty() || command.ir_output.empty()) {
    std::cerr << "matcore-extract: --input and --ir-out are required\n";
    return std::nullopt;
  }
  if (!selectCompilerArgument(command)) return std::nullopt;
  // Verification and --frontend-info deliberately do not require an external
  // Clang installation. Discover the compiler/resource/header tuple only for
  // a real extraction.
  const std::optional<ConfiguredCompiler> configured_compiler =
      discoverConfiguredCompiler(command.compiler_was_explicit
                                     ? command.frontend.clang_path
                                     : std::string_view{},
                                 command.frontend_name == "native");
  if (!configured_compiler) return std::nullopt;
  const std::optional<std::string> configured_compiler_utf8 =
      pathToUtf8(configured_compiler->invocation_path, "compiler");
  if (!configured_compiler_utf8) return std::nullopt;
  if (!command.compiler_was_explicit) {
    command.frontend.clang_path = *configured_compiler_utf8;
  }
  const std::optional<std::string> configured_resource_utf8 = pathToUtf8(
      configured_compiler->resource_directory, "Clang resource");
  if (!configured_resource_utf8) return std::nullopt;
  command.frontend.clang_resource_directory = *configured_resource_utf8;

  std::string executable_error;
  const std::optional<std::filesystem::path> executable =
      support::current_executable_path_v1(executable_error);
  if (!executable) {
    std::cerr << "matcore-extract: cannot locate the running extractor: "
              << executable_error << '\n';
    return std::nullopt;
  }
  if (const auto include_directory = discoverToolIncludeDirectory(*executable)) {
    const std::optional<std::string> encoded =
        pathToUtf8(*include_directory, "tool include");
    if (!encoded) return std::nullopt;
    command.tool_include_directory = *encoded;
  }
  const std::optional<std::filesystem::path> tool_include_directory =
      pathFromUtf8(command.tool_include_directory, "tool include");
  if (!tool_include_directory) return std::nullopt;
  const std::filesystem::path tool_public_header =
      *tool_include_directory / "matcore" / "mdsl.h";
  std::error_code header_error;
  if (command.tool_include_directory.empty() ||
      !std::filesystem::is_regular_file(tool_public_header, header_error)) {
    std::cerr << "matcore-extract: unable to locate the tool-owned "
                 "<matcore/mdsl.h> beside the executable\n";
    return std::nullopt;
  }
  const std::optional<std::string> tool_public_header_utf8 =
      pathToUtf8(tool_public_header, "trusted public header");
  if (!tool_public_header_utf8) return std::nullopt;
  command.frontend.trusted_public_headers.push_back(*tool_public_header_utf8);
  const unsigned generated_output_count =
      static_cast<unsigned>(!command.rewrite_output.empty()) +
      static_cast<unsigned>(!command.sites_output.empty()) +
      static_cast<unsigned>(!command.stubs_output.empty()) +
      static_cast<unsigned>(!command.backend_output.empty());
  if (!command.recovered_gemm_report.empty() && generated_output_count != 0) {
    std::cerr << "matcore-extract: recovered GEMM inspection never authorizes "
                 "host rewrite or generated execution artifacts\n";
    return std::nullopt;
  }
  command.frontend.inspect_recovered_cpp_gemm =
      !command.recovered_gemm_report.empty();
  if (generated_output_count != 0 && generated_output_count != 4) {
    std::cerr << "matcore-extract: --rewrite-out, --sites-out, --stubs-out, "
                 "and --backend-out must be supplied together\n";
    return std::nullopt;
  }
  if (!validateOutputPaths(command)) {
    return std::nullopt;
  }
  if (generated_output_count == 4) {
    std::vector<const std::string *> generated_paths{
        &command.ir_output, &command.rewrite_output, &command.sites_output,
        &command.stubs_output, &command.backend_output};
    if (!command.semantic_ir_output.empty())
      generated_paths.push_back(&command.semantic_ir_output);
    if (!command.structured_ir_output.empty())
      generated_paths.push_back(&command.structured_ir_output);
    for (const std::string *path : generated_paths) {
      if (*path == "-") {
        std::cerr << "matcore-extract: generated artifact mode requires file "
                     "paths, not standard output\n";
        return std::nullopt;
      }
    }
  }
  const std::optional<std::filesystem::path> selected_compiler =
      pathFromUtf8(command.frontend.clang_path, "compiler");
  if (!selected_compiler) return std::nullopt;
  if (command.frontend_name == "native" &&
      !pathsReferToSameLocation(*selected_compiler,
                                configured_compiler->invocation_path)) {
    std::cerr << "matcore-extract: native frontend is linked to the configured "
                 "Clang "
              << MDSLC_TOOLCHAIN_VERSION
              << " tuple and cannot honor a different --clang executable: "
              << command.frontend.clang_path << '\n';
    return std::nullopt;
  }
  if (isWindowsHost()) {
    // The configured Windows compiler tuple is clang-cl/MSVC.  Direct
    // extractor callers pass clang-cl spellings such as /TP and /I, but must
    // not need to supply a mode-changing control argument themselves.  Record
    // the already-authenticated driver flavor internally before LibTooling
    // constructs its FixedCompilationDatabase.
    if (std::find(command.frontend.compiler_arguments.begin(),
                  command.frontend.compiler_arguments.end(),
                  "--driver-mode=cl") ==
        command.frontend.compiler_arguments.end()) {
      command.frontend.compiler_arguments.insert(
          command.frontend.compiler_arguments.begin(), "--driver-mode=cl");
    }
    command.frontend.compiler_arguments.insert(
        command.frontend.compiler_arguments.begin(),
        {"/I", command.tool_include_directory});
  } else {
    command.frontend.compiler_arguments.insert(
        command.frontend.compiler_arguments.begin(),
        "-I" + command.tool_include_directory);
  }
  return command;
}

void printDiagnostic(const matcore::mdslc::frontend::Diagnostic &diagnostic) {
  std::cerr << (diagnostic.file.empty() ? "<unknown>" : diagnostic.file);
  if (diagnostic.line != 0) {
    std::cerr << ':' << diagnostic.line;
    if (diagnostic.column != 0) {
      std::cerr << ':' << diagnostic.column;
    }
  }
  std::cerr << ": error: " << diagnostic.message << '\n';
}

bool writeAtomically(const std::string &path, std::string_view contents) {
  if (path == "-") {
    std::cout << contents;
    return static_cast<bool>(std::cout);
  }
  const std::optional<std::filesystem::path> destination =
      pathFromUtf8(path, "output");
  if (!destination) return false;
  static std::atomic<std::uint64_t> sequence{0};
  std::random_device entropy;
  const std::uint64_t nonce =
      (static_cast<std::uint64_t>(entropy()) << 32U) ^ entropy() ^
      static_cast<std::uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count()) ^
      sequence.fetch_add(1, std::memory_order_relaxed);
  std::filesystem::path temporary = *destination;
  temporary += ".tmp." + std::to_string(nonce);
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output || !output.write(contents.data(), contents.size())) {
      std::cerr << "matcore-extract: failed to write temporary IR file "
                << temporary << '\n';
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      return false;
    }
  }
  std::string publish_error;
  if (!support::replace_file_atomically_v1(temporary, *destination,
                                           publish_error)) {
    std::cerr << "matcore-extract: failed to publish IR file " << path << ": "
              << publish_error << '\n';
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return false;
  }
  return true;
}

std::optional<std::string> readFile(const std::string &path) {
  const std::optional<std::filesystem::path> native_path =
      pathFromUtf8(path, "input");
  if (!native_path) return std::nullopt;
  std::ifstream input(*native_path, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

int ExtractorMain(int argc, char **argv) {
  std::string environment_error;
  const std::optional<std::string> poisoned_environment =
      support::poisoned_compiler_environment_v1(environment_error);
  if (!environment_error.empty()) {
    std::cerr << "matcore-extract: " << environment_error << '\n';
    return 2;
  }
  if (poisoned_environment) {
    std::cerr << "matcore-extract: inherited compiler control variable "
              << *poisoned_environment
              << " is set; unset it before invoking the in-process frontend"
              << '\n';
    return 2;
  }
  const std::optional<CommandLine> command = parseCommandLine(argc, argv);
  if (!command) {
    usage(std::cerr);
    return 2;
  }

  if (!command->verify_ir.empty()) {
    const std::optional<std::string> encoded = readFile(command->verify_ir);
    if (!encoded) {
      std::cerr << command->verify_ir << ": error: unable to read Matcore IR\n";
      return 1;
    }
    std::string error;
    std::uint32_t version = 0;
    if (!matcore::mdslc::ir::v1::probeJsonVersion(*encoded, version, error)) {
      std::cerr << command->verify_ir << ": error: " << error << '\n';
      return 1;
    }
    if (version == matcore::mdslc::ir::kMatcoreIrVersion) {
      matcore::mdslc::ir::Module module;
      if (!matcore::mdslc::ir::parseAndVerifyJson(*encoded, module, error)) {
        std::cerr << command->verify_ir << ": error: " << error << '\n';
        return 1;
      }
      std::cout << "verified Matcore IR v0: " << module.operations.size()
                << " operation(s)\n";
      return 0;
    }
    if (version == matcore::mdslc::ir::v1::kMatcoreIrVersion) {
      matcore::mdslc::ir::v1::Module module;
      if (!matcore::mdslc::ir::v1::parseAndVerifyJson(*encoded, module,
                                                       error)) {
        std::cerr << command->verify_ir << ": error: " << error << '\n';
        return 1;
      }
      std::cout << "verified Matcore IR v1: " << module.operations.size()
                << " operation(s)\n";
      return 0;
    }
    std::cerr << command->verify_ir
              << ": error: unsupported Matcore IR version " << version
              << '\n';
    return 1;
  }

  std::unique_ptr<matcore::mdslc::frontend::Frontend> frontend;
  if (command->frontend_name == "native") {
#if MDSLC_HAS_NATIVE_FRONTEND
    frontend = matcore::mdslc::frontend::createClangLibToolingFrontend();
#else
    std::cerr << "matcore-extract: native frontend is the default but was not "
                 "built; rebuild with MDSLC_ENABLE_NATIVE_FRONTEND=ON or "
                 "explicitly request the compatibility frontend\n";
    return 1;
#endif
  } else {
#if MDSLC_HAS_BOOTSTRAP_FRONTEND
    frontend =
        matcore::mdslc::frontend::createClangAstJsonBootstrapFrontend();
#else
    std::cerr << "matcore-extract: AST-JSON bootstrap frontend was not built\n";
    return 1;
#endif
  }
  matcore::mdslc::frontend::Result result;
  if (!frontend->extract(command->frontend, result)) {
    for (const auto &diagnostic : result.diagnostics) {
      printDiagnostic(diagnostic);
    }
    return 1;
  }

  std::string verification_error;
  if (!matcore::mdslc::ir::verify(result.module, verification_error)) {
    std::cerr << command->frontend.input_path
              << ": error: Matcore IR verification failed: "
              << verification_error << '\n';
    return 1;
  }
  matcore::mdslc::ir::v1::Module typed_module;
  if (!matcore::mdslc::ir::v1::fromV0(result.module, typed_module,
                                       verification_error) ||
      !matcore::mdslc::ir::v1::verify(typed_module, verification_error)) {
    std::cerr << command->frontend.input_path
              << ": error: Matcore IR v0-to-v1 boundary failed: "
              << verification_error << '\n';
    return 1;
  }
  matcore::mdslc::ir::Module projected_module;
  if (!matcore::mdslc::ir::v1::projectToV0(
          typed_module, projected_module, verification_error)) {
    std::cerr << command->frontend.input_path
              << ": error: Matcore IR v1-to-v0 lowering projection failed: "
              << verification_error << '\n';
    return 1;
  }

  std::string semantic_ir;
  std::string structured_ir;
  std::vector<matcore::mdslc::codegen::RuntimeDispatchBackendEntryV1>
      semantic_backend_entries;
  if (command->semantic_pipeline == SemanticPipeline::MatcoreMlirCpuV1) {
#if MDSLC_HAS_MATCORE_MLIR
    mlir::MLIRContext semantic_context;
    semantic_context.allowUnregisteredDialects(false);
    auto semantic =
        matcore::mdslc::mlir_bridge::bridgeV1ToMatcoreMlir(
            typed_module, semantic_context,
            matcore::mdslc::mlir_bridge::explicitGemmF32V1BridgeContext());
    if (!semantic) {
      std::cerr << command->frontend.input_path
                << ": error: Matcore MLIR semantic bridge failed: "
                << semantic.error << '\n';
      return 1;
    }
    if (!command->structured_ir_output.empty()) {
      auto structured =
          matcore::mdslc::mlir_bridge::deriveStructuredGemmHandoffV1(
              *semantic.module);
      if (!structured) {
        std::cerr << command->frontend.input_path
                  << ": error: structured GEMM handoff failed: "
                  << structured.error << '\n';
        return 1;
      }
      structured_ir =
          matcore::mdslc::mlir_bridge::serializeDeterministicMlir(
              *structured.module);
    }
    std::vector<
        matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1>
        dispatch_records;
    if (!matcore::mdslc::mlir_lowering::
            lowerExplicitGemmToCpuRuntimeDispatchV1(
                *semantic.module, dispatch_records, verification_error)) {
      std::cerr << command->frontend.input_path
                << ": error: Matcore MLIR CPU lowering failed: "
                << verification_error << '\n';
      return 1;
    }
    semantic_ir =
        matcore::mdslc::mlir_bridge::serializeDeterministicMlir(
            *semantic.module);
    semantic_backend_entries.reserve(dispatch_records.size());
    for (const auto &record : dispatch_records) {
      if (record.runtime_symbol !=
          matcore::mdslc::mlir_lowering::kCpuRuntimeDispatchSymbolV1) {
        std::cerr << command->frontend.input_path
                  << ": error: Matcore MLIR CPU lowering selected an "
                     "unsupported runtime boundary\n";
        return 1;
      }
      semantic_backend_entries.push_back({record.site_id});
    }
#else
    std::cerr << "matcore-extract: internal error: unavailable Matcore MLIR "
                 "pipeline passed command validation\n";
    return 1;
#endif
  }
  const std::string json =
      command->ir_version == matcore::mdslc::ir::v1::kMatcoreIrVersion
          ? matcore::mdslc::ir::v1::serializeDeterministicJson(typed_module)
          : matcore::mdslc::ir::serializeDeterministicJson(projected_module);
  if (command->rewrite_output.empty()) {
    if (!command->recovered_gemm_report.empty() &&
        !writeAtomically(command->recovered_gemm_report,
                         matcore::mdslc::frontend::
                             serializeRecoveredGemmInspection(result))) {
      return 1;
    }
    if (!command->semantic_ir_output.empty() &&
        !writeAtomically(command->semantic_ir_output, semantic_ir)) {
      return 1;
    }
    if (!command->structured_ir_output.empty() &&
        !writeAtomically(command->structured_ir_output, structured_ir)) {
      return 1;
    }
    return writeAtomically(command->ir_output, json) ? 0 : 1;
  }

  const std::optional<std::filesystem::path> sites_path =
      pathFromUtf8(command->sites_output, "sites output");
  if (!sites_path) return 1;
  const std::optional<std::string> sites_include =
      pathToUtf8(sites_path->filename(), "sites include");
  if (!sites_include) return 1;
  matcore::mdslc::codegen::Artifacts artifacts;
  std::string generation_error;
  if (!matcore::mdslc::codegen::generate(
          projected_module, result.source_snapshot, *sites_include, artifacts,
          generation_error)) {
    std::cerr << command->frontend.input_path
              << ": error: generated artifact validation failed: "
              << generation_error << '\n';
    return 1;
  }
  if (command->semantic_pipeline == SemanticPipeline::MatcoreMlirCpuV1 &&
      !matcore::mdslc::codegen::generateRuntimeDispatchBackendV1(
          semantic_backend_entries,
          matcore::mdslc::codegen::
              RuntimeDispatchBackendProducerV1::MatcoreMlirCpuV1,
          artifacts.backend_source, generation_error)) {
    std::cerr << command->frontend.input_path
              << ": error: semantic backend generation failed: "
              << generation_error << '\n';
    return 1;
  }

  // All contents are generated and verified before any destination is
  // published. Each individual file is published through a temporary rename.
  std::vector<std::pair<std::string, std::string_view>> outputs = {
      {command->ir_output, json},
      {command->rewrite_output, artifacts.rewritten_host},
      {command->sites_output, artifacts.sites_header},
      {command->stubs_output, artifacts.stubs_source},
      {command->backend_output, artifacts.backend_source},
  };
  if (!command->semantic_ir_output.empty()) {
    outputs.emplace_back(command->semantic_ir_output, semantic_ir);
  }
  if (!command->structured_ir_output.empty()) {
    outputs.emplace_back(command->structured_ir_output, structured_ir);
  }
  for (const auto &[path, contents] : outputs) {
    if (!writeAtomically(path, contents)) {
      return 1;
    }
  }
  return 0;
}

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t **argv) {
  std::string error;
  const std::optional<std::vector<std::string>> utf8_arguments =
      support::wide_arguments_to_utf8_v1(argc, argv, error);
  if (!utf8_arguments) {
    std::cerr << "matcore-extract: cannot decode Windows process arguments: "
              << error << '\n';
    return 2;
  }
  std::vector<char *> narrow_arguments;
  narrow_arguments.reserve(utf8_arguments->size());
  for (const std::string &argument : *utf8_arguments) {
    narrow_arguments.push_back(const_cast<char *>(argument.c_str()));
  }
  return ExtractorMain(static_cast<int>(narrow_arguments.size()),
                       narrow_arguments.data());
}
#else
int main(int argc, char **argv) { return ExtractorMain(argc, argv); }
#endif
