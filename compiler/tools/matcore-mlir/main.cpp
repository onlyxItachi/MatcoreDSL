#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/IR/MLIRContext.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::string numerical_profile;
  std::string execution_intent;
  std::string emit_stage = "semantic";
  bool output_to_stdout = true;
};

void printUsage(std::ostream &stream) {
  stream << "usage: matcore-mlir --input <capture.v1.json> "
            "--numerical-profile explicit-gemm-f32-v1 "
            "--execution-intent generic "
            "[--emit-stage semantic|structured-gemm-v1] "
            "[--output <inspection.mlir>]\n";
}

bool parseOptions(int argc, char **argv, Options &options,
                  std::string &error) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      printUsage(std::cout);
      return false;
    }
    const auto takeValue = [&](std::string_view name,
                               std::string &value) -> bool {
      if (index + 1 >= argc) {
        error = std::string(name) + " requires one value";
        return false;
      }
      value = argv[++index];
      return true;
    };
    if (argument == "--input") {
      std::string value;
      if (!takeValue(argument, value))
        return false;
      options.input = value;
    } else if (argument == "--output") {
      std::string value;
      if (!takeValue(argument, value))
        return false;
      options.output = value;
      options.output_to_stdout = value == "-";
    } else if (argument == "--numerical-profile") {
      if (!takeValue(argument, options.numerical_profile))
        return false;
    } else if (argument == "--execution-intent") {
      if (!takeValue(argument, options.execution_intent))
        return false;
    } else if (argument == "--emit-stage") {
      if (!takeValue(argument, options.emit_stage))
        return false;
    } else {
      error = "unknown argument: " + std::string(argument);
      return false;
    }
  }
  if (options.input.empty()) {
    error = "--input is required";
    return false;
  }
  if (options.numerical_profile.empty()) {
    error = "--numerical-profile is required; numerical permissions are never inferred";
    return false;
  }
  if (options.execution_intent.empty()) {
    error = "--execution-intent is required; compilation intent is never inferred";
    return false;
  }
  if (options.numerical_profile !=
      matcore::mdslc::mlir_bridge::kExplicitGemmF32Profile) {
    error = "unsupported numerical profile: " + options.numerical_profile;
    return false;
  }
  if (options.execution_intent != "generic") {
    error = "unsupported execution intent for the v1 bridge: " +
            options.execution_intent;
    return false;
  }
  if (options.emit_stage != "semantic" &&
      options.emit_stage != "structured-gemm-v1") {
    error = "unsupported inspection stage: " + options.emit_stage;
    return false;
  }
  if (!options.output_to_stdout) {
    std::error_code input_error;
    std::error_code output_error;
    const auto canonical_input =
        std::filesystem::weakly_canonical(options.input, input_error);
    const auto canonical_output =
        std::filesystem::weakly_canonical(options.output, output_error);
    if (!input_error && !output_error && canonical_input == canonical_output) {
      error = "input and output paths must differ";
      return false;
    }
  }
  return true;
}

bool readFile(const std::filesystem::path &path, std::string &contents,
              std::string &error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    error = "cannot open Matcore IR v1 input: " + path.string();
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    error = "failed while reading Matcore IR v1 input: " + path.string();
    return false;
  }
  contents = buffer.str();
  return true;
}

bool writeFile(const std::filesystem::path &path, std::string_view contents,
               std::string &error) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    error = "cannot open Matcore MLIR output: " + path.string();
    return false;
  }
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream) {
    error = "failed while writing Matcore MLIR output: " + path.string();
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  std::string error;
  if (!parseOptions(argc, argv, options, error)) {
    if (!error.empty()) {
      std::cerr << "matcore-mlir: error: " << error << '\n';
      printUsage(std::cerr);
      return 2;
    }
    return 0;
  }

  std::string json;
  if (!readFile(options.input, json, error)) {
    std::cerr << "matcore-mlir: error: " << error << '\n';
    return 1;
  }
  matcore::mdslc::ir::v1::Module capture;
  if (!matcore::mdslc::ir::v1::parseAndVerifyJson(json, capture, error)) {
    std::cerr << "matcore-mlir: error: verified Matcore IR v1 input required: "
              << error << '\n';
    return 1;
  }

  mlir::MLIRContext context;
  context.allowUnregisteredDialects(false);
  auto bridge_context =
      matcore::mdslc::mlir_bridge::explicitGemmF32V1BridgeContext();
  auto result = matcore::mdslc::mlir_bridge::bridgeV1ToMatcoreMlir(
      capture, context, bridge_context);
  if (!result) {
    std::cerr << "matcore-mlir: error: " << result.error << '\n';
    return 1;
  }
  mlir::ModuleOp output_module = *result.module;
  matcore::mdslc::mlir_bridge::StructuredGemmHandoffResultV1 structured;
  if (options.emit_stage == "structured-gemm-v1") {
    structured =
        matcore::mdslc::mlir_bridge::deriveStructuredGemmHandoffV1(
            *result.module);
    if (!structured) {
      std::cerr << "matcore-mlir: error: " << structured.error << '\n';
      return 1;
    }
    output_module = *structured.module;
  }
  const std::string output =
      matcore::mdslc::mlir_bridge::serializeDeterministicMlir(output_module);
  if (options.output_to_stdout) {
    std::cout << output;
    return std::cout ? 0 : 1;
  }
  if (!writeFile(options.output, output, error)) {
    std::cerr << "matcore-mlir: error: " << error << '\n';
    return 1;
  }
  return 0;
}
