#include "../../lib/frontend/frontend.h"
#include "../../lib/codegen/codegen.h"
#include "mdslc_config.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct CommandLine {
  matcore::mdslc::frontend::Options frontend;
  std::string tool_include_directory;
  std::string ir_output;
  std::string rewrite_output;
  std::string sites_output;
  std::string stubs_output;
  std::string backend_output;
  std::string verify_ir;
  bool compiler_was_explicit = false;
};

std::filesystem::path normalizedPath(const std::filesystem::path &path) {
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    absolute = path;
  }
  error.clear();
  const std::filesystem::path normalized =
      std::filesystem::weakly_canonical(absolute, error);
  return error ? absolute.lexically_normal() : normalized;
}

bool pathsReferToSameLocation(const std::filesystem::path &left,
                              const std::filesystem::path &right) {
  std::error_code error;
  if (std::filesystem::equivalent(left, right, error) && !error) {
    return true;
  }
  return normalizedPath(left) == normalizedPath(right);
}

bool validateOutputPaths(const CommandLine &command) {
  const std::vector<std::pair<std::string_view, std::string_view>> outputs = {
      {"--ir-out", command.ir_output},
      {"--rewrite-out", command.rewrite_output},
      {"--sites-out", command.sites_output},
      {"--stubs-out", command.stubs_output},
      {"--backend-out", command.backend_output},
  };
  std::vector<std::pair<std::string_view, std::filesystem::path>> validated;
  for (const auto &[option, encoded_path] : outputs) {
    if (encoded_path.empty() || encoded_path == "-") {
      continue;
    }
    const std::filesystem::path path(encoded_path);
    if (pathsReferToSameLocation(command.frontend.input_path, path)) {
      std::cerr << "matcore-extract: " << option
                << " must not overwrite or alias the input .mdsl file\n";
      return false;
    }
    for (const auto &[prior_option, prior_path] : validated) {
      if (pathsReferToSameLocation(prior_path, path)) {
        std::cerr << "matcore-extract: " << option << " and " << prior_option
                  << " must refer to distinct output files\n";
        return false;
      }
    }
    validated.emplace_back(option, path);
  }
  return true;
}

void usage(std::ostream &output) {
  output
      << "usage: matcore-extract --input FILE.mdsl --ir-out FILE.json [options] "
         "-- [clang++-placeholder] COMPILE_ARGS\n"
      << "\n"
      << "Bootstrap frontend options:\n"
      << "  --clang PATH          Clang executable (default: "
         "/usr/bin/clang++-21)\n"
      << "  --ast-byte-limit N    maximum captured AST JSON bytes\n"
      << "  --verbose             print the exact Clang command\n"
      << "  --rewrite-out FILE    rewritten host C++ (requires all outputs)\n"
      << "  --sites-out FILE      generated C++ site declarations\n"
      << "  --stubs-out FILE      generated C++ descriptor stubs\n"
      << "  --backend-out FILE    generated C ABI backend forwarding entries\n"
      << "  --verify-ir FILE      verify serialized Matcore IR v0 and exit\n"
      << "  --frontend-info       describe this non-LibTooling fallback\n"
      << "\n"
      << "A bare clang++ token after -- is a command-shape placeholder; the "
         "configured\n"
      << "Clang 21 executable remains in use. Pass --clang PATH to override it "
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

std::optional<CommandLine> parseCommandLine(int argc, char **argv) {
  CommandLine command;
  std::error_code executable_error;
  const std::filesystem::path executable =
      std::filesystem::canonical("/proc/self/exe", executable_error);
  if (!executable_error) {
    command.tool_include_directory =
        (executable.parent_path().parent_path() / "include").string();
  }
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
    } else if (argument == "--verify-ir") {
      if (!takeValue(argc, argv, index, command.verify_ir, "--verify-ir")) {
        return std::nullopt;
      }
    } else if (argument == "--clang") {
      if (!takeValue(argc, argv, index, command.frontend.clang_path,
                     "--clang")) {
        return std::nullopt;
      }
      command.compiler_was_explicit = true;
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
      std::cout
          << "clang-ast-json-bootstrap-v0: executes Clang parsing/Sema, then "
             "structurally consumes one bounded JSON AST. This is not the "
             "planned LibTooling frontend. Clang JSON exposes AnnotateAttr "
             "presence but not its payload.\n";
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
        !command.frontend.compiler_arguments.empty()) {
      std::cerr << "matcore-extract: --verify-ir cannot be combined with "
                   "extraction or generation options\n";
      return std::nullopt;
    }
    return command;
  }
  if (command.frontend.input_path.empty() || command.ir_output.empty()) {
    std::cerr << "matcore-extract: --input and --ir-out are required\n";
    return std::nullopt;
  }
  const std::filesystem::path tool_public_header =
      std::filesystem::path(command.tool_include_directory) / "matcore" /
      "mdsl.h";
  std::error_code header_error;
  if (command.tool_include_directory.empty() ||
      !std::filesystem::is_regular_file(tool_public_header, header_error)) {
    std::cerr << "matcore-extract: unable to locate the tool-owned "
                 "<matcore/mdsl.h> beside the executable\n";
    return std::nullopt;
  }
  command.frontend.trusted_public_headers.push_back(tool_public_header.string());
  const unsigned generated_output_count =
      static_cast<unsigned>(!command.rewrite_output.empty()) +
      static_cast<unsigned>(!command.sites_output.empty()) +
      static_cast<unsigned>(!command.stubs_output.empty()) +
      static_cast<unsigned>(!command.backend_output.empty());
  if (generated_output_count != 0 && generated_output_count != 4) {
    std::cerr << "matcore-extract: --rewrite-out, --sites-out, --stubs-out, "
                 "and --backend-out must be supplied together\n";
    return std::nullopt;
  }
  if (!validateOutputPaths(command)) {
    return std::nullopt;
  }
  if (generated_output_count == 4) {
    for (const std::string *path :
         {&command.ir_output, &command.rewrite_output, &command.sites_output,
          &command.stubs_output, &command.backend_output}) {
      if (*path == "-") {
        std::cerr << "matcore-extract: generated artifact mode requires file "
                     "paths, not standard output\n";
        return std::nullopt;
      }
    }
  }
  if (!command.frontend.compiler_arguments.empty() &&
      !command.frontend.compiler_arguments.front().starts_with("-")) {
    const std::filesystem::path candidate(
        command.frontend.compiler_arguments.front());
    const std::string filename = candidate.filename().string();
    if (filename.starts_with("clang")) {
      if (command.compiler_was_explicit) {
        std::cerr << "matcore-extract: specify the compiler either with "
                     "--clang or after --, not both\n";
        return std::nullopt;
      }
      if (candidate.has_parent_path() || filename != "clang++") {
        command.frontend.clang_path =
            command.frontend.compiler_arguments.front();
      }
      command.frontend.compiler_arguments.erase(
          command.frontend.compiler_arguments.begin());
    } else if (filename.starts_with("g++") || filename.starts_with("c++")) {
      std::cerr << "matcore-extract: the bootstrap frontend requires Clang; "
                   "use --clang with a compatible clang++ executable\n";
      return std::nullopt;
    }
  }
  command.frontend.compiler_arguments.insert(
      command.frontend.compiler_arguments.begin(),
      "-I" + command.tool_include_directory);
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
  const std::string temporary =
      path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
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
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::cerr << "matcore-extract: failed to publish IR file " << path << ": "
              << error.message() << '\n';
    std::filesystem::remove(temporary, error);
    return false;
  }
  return true;
}

std::optional<std::string> readFile(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

} // namespace

int main(int argc, char **argv) {
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
    matcore::mdslc::ir::Module module;
    std::string error;
    if (!matcore::mdslc::ir::parseAndVerifyJson(*encoded, module, error)) {
      std::cerr << command->verify_ir << ": error: " << error << '\n';
      return 1;
    }
    std::cout << "verified Matcore IR v0: " << module.operations.size()
              << " operation(s)\n";
    return 0;
  }

  std::unique_ptr<matcore::mdslc::frontend::Frontend> frontend =
      matcore::mdslc::frontend::createClangAstJsonBootstrapFrontend();
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
  const std::string json =
      matcore::mdslc::ir::serializeDeterministicJson(result.module);
  if (command->rewrite_output.empty()) {
    return writeAtomically(command->ir_output, json) ? 0 : 1;
  }

  const std::string sites_include =
      std::filesystem::path(command->sites_output).filename().string();
  matcore::mdslc::codegen::Artifacts artifacts;
  std::string generation_error;
  if (!matcore::mdslc::codegen::generate(result.module, result.source_snapshot,
                                         sites_include, artifacts,
                                         generation_error)) {
    std::cerr << command->frontend.input_path
              << ": error: generated artifact validation failed: "
              << generation_error << '\n';
    return 1;
  }

  // All contents are generated and verified before any destination is
  // published. Each individual file is published through a temporary rename.
  const std::vector<std::pair<std::string, std::string_view>> outputs = {
      {command->ir_output, json},
      {command->rewrite_output, artifacts.rewritten_host},
      {command->sites_output, artifacts.sites_header},
      {command->stubs_output, artifacts.stubs_source},
      {command->backend_output, artifacts.backend_source},
  };
  for (const auto &[path, contents] : outputs) {
    if (!writeAtomically(path, contents)) {
      return 1;
    }
  }
  return 0;
}
