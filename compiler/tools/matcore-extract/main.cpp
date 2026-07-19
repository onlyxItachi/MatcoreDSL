#include "../../lib/frontend/frontend.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CommandLine {
  matcore::mdslc::frontend::Options frontend;
  std::string ir_output;
  bool compiler_was_explicit = false;
};

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

  if (command.frontend.input_path.empty() || command.ir_output.empty()) {
    std::cerr << "matcore-extract: --input and --ir-out are required\n";
    return std::nullopt;
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

} // namespace

int main(int argc, char **argv) {
  const std::optional<CommandLine> command = parseCommandLine(argc, argv);
  if (!command) {
    usage(std::cerr);
    return 2;
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
  return writeAtomically(command->ir_output, json) ? 0 : 1;
}
