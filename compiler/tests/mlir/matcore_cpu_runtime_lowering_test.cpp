#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;
namespace lowering = matcore::mdslc::mlir_lowering;
namespace v1 = matcore::mdslc::ir::v1;

int checks = 0;
int failures = 0;

void expect(bool condition, std::string_view message) {
  ++checks;
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

std::string readFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  expect(stream.good() || stream.eof(), "fixture is readable");
  return contents.str();
}

v1::Module readCapture() {
  const std::string json =
      readFile(std::string(MDSLC_IR_TEST_SOURCE_DIR) +
               "/gemm_capture.v1.golden.json");
  v1::Module module;
  std::string error;
  expect(v1::parseAndVerifyJson(json, module, error),
         "typed capture parses and verifies");
  return module;
}

} // namespace

int main() {
  const v1::Module capture = readCapture();
  expect(capture.operations.size() == 1,
         "lowering fixture contains exactly one GEMM");
  if (capture.operations.size() != 1) return 1;

  mlir::MLIRContext context;
  auto bridged = bridge::bridgeV1ToMatcoreMlir(
      capture, context, bridge::explicitGemmF32V1BridgeContext());
  expect(static_cast<bool>(bridged), "typed capture bridges before lowering");
  if (!bridged) return 1;

  std::vector<lowering::CpuRuntimeDispatchRecordV1> records;
  std::string error;
  expect(lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
             *bridged.module, records, error),
         "verified explicit GEMM lowers to runtime dispatch");
  expect(error.empty() && records.size() == 1,
         "successful lowering is transactional and singular");
  if (records.size() == 1) {
    const auto &record = records.front();
    const auto &operation = capture.operations.front();
    expect(record.version == lowering::kCpuRuntimeDispatchRecordVersionV1 &&
               record.site_id == operation.site_id &&
               record.semantic_symbol ==
                   "__matcore_semantic_" + operation.site_id,
           "record preserves semantic site identity");
    expect(record.operation == "gemm" && record.dtype == "f32" &&
               record.accumulation_dtype == "f32" &&
               record.target == "cpu" && record.fallback == "error",
           "record preserves supported mathematical and policy envelope");
    expect(record.numerical_profile == "explicit-gemm-f32-v1" &&
               record.execution_intent == "generic" &&
               record.runtime_symbol ==
                   lowering::kCpuRuntimeDispatchSymbolV1,
           "record names the reviewed numerical, intent, and runtime boundary");
    expect(record.required_guards ==
               std::vector<std::string>{
                   "descriptor_v0", "alias_no_overlap",
                   "required_alignment",
                   "explicit_gemm_f32_v1_fp_environment"},
           "record retains every dynamic guard family");
    expect(record.source_file == operation.source.file &&
               record.source_line == operation.source.line &&
               record.source_column == operation.source.column &&
               record.source_range_begin == operation.call_range.begin &&
               record.source_range_end == operation.call_range.end,
           "record preserves exact source provenance and call range");
  }

  std::vector<lowering::CpuRuntimeDispatchRecordV1> second;
  expect(lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
             *bridged.module, second, error) &&
             !records.empty() && !second.empty() &&
             second.size() == records.size() &&
             second.front().site_id == records.front().site_id &&
             second.front().required_guards == records.front().required_guards,
         "repeated lowering is deterministic");

  (*bridged.module)->setAttr("mdsl.execution_intent",
                             mlir::StringAttr::get(&context, "training"));
  records.push_back({});
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  expect(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
             *bridged.module, records, error),
         "unsupported execution intent fails closed");
  expect(records.empty() && !error.empty(),
         "failed lowering publishes no dispatch records");

  mlir::MLIRContext composition_context;
  bridge::registerMatcoreSemanticDialects(composition_context);
  const std::string composition =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) +
               "/gemm_sin_all.semantic.golden.mlir");
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(
      composition, &composition_context);
  expect(static_cast<bool>(parsed), "composition fixture parses");
  if (parsed) {
    records.push_back({});
    expect(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
               *parsed, records, error),
           "unsupported map/domain composition cannot be silently dropped");
    expect(records.empty(),
           "composition rejection publishes no partial dispatch records");
  }

  if (failures != 0) {
    std::cerr << "CPU runtime-dispatch lowering: " << failures << " of "
              << checks << " checks failed\n";
    return 1;
  }
  std::cout << "CPU runtime-dispatch lowering: " << checks
            << " checks, 0 failures\n";
  return 0;
}
