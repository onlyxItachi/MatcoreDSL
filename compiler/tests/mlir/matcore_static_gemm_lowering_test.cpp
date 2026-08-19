#include "MatcoreStaticGemmLowering.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include <cmath>
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

v1::Module makeStaticModule(std::uint64_t m, std::uint64_t n, std::uint64_t k) {
  v1::Module mod = readCapture();
  mod.operations[0].operands[0].type.shape = {
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = m, .symbol = ""},
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = k, .symbol = ""}
  };
  mod.operations[0].operands[0].type.strides = {
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = k, .symbol = ""},
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = 1, .symbol = ""}
  };
  mod.operations[0].operands[1].type.shape = {
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = k, .symbol = ""},
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = n, .symbol = ""}
  };
  mod.operations[0].operands[1].type.strides = {
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = n, .symbol = ""},
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = 1, .symbol = ""}
  };
  mod.operations[0].output.type.shape = {
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = m, .symbol = ""},
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = n, .symbol = ""}
  };
  mod.operations[0].output.type.strides = {
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = n, .symbol = ""},
      v1::ScalarExpr{.kind = v1::ScalarExpr::Kind::Static, .value = 1, .symbol = ""}
  };
  return mod;
}

} // namespace

int main() {
  mlir::MLIRContext context;

  // Test 1: Static General GEMM (16x16x16)
  {
    const v1::Module capture = makeStaticModule(16, 16, 16);
    auto bridged = bridge::bridgeV1ToMatcoreMlir(
        capture, context, bridge::explicitGemmF32V1BridgeContext());
    if (!bridged) {
      std::cerr << "Bridge error: " << bridged.error << '\n';
    }
    expect(static_cast<bool>(bridged), "bridged static 16x16x16 module successfully");

    std::vector<lowering::StaticGemmSpecializationRecordV1> records;
    std::string error;
    bool ok = lowering::lowerExplicitGemmToStaticSpecializationV1(
        bridged.module.get(), records, error);
    expect(ok, "lowered static 16x16x16 to specialization");
    expect(records.size() == 1, "exactly one record generated");
    if (records.size() == 1) {
      expect(records[0].m == 16, "record m is 16");
      expect(records[0].n == 16, "record n is 16");
      expect(records[0].k == 16, "record k is 16");
      expect(records[0].function_symbol == "matcore_mlir_static_gemm_f32_16x16x16",
             "symbol name matches 16x16x16");
      expect(records[0].declaration_c.find("matcore_mlir_static_gemm_f32_16x16x16") != std::string::npos,
             "declaration contains symbol");
      expect(records[0].definition_c.find("float* MATCORE_RESTRICT c") != std::string::npos,
             "definition contains restrict C");
    }
  }

  // Test 2: DOT reduction special case (1x1x32)
  {
    const v1::Module capture = makeStaticModule(1, 1, 32);
    auto bridged = bridge::bridgeV1ToMatcoreMlir(
        capture, context, bridge::explicitGemmF32V1BridgeContext());
    expect(static_cast<bool>(bridged), "bridged static 1x1x32 module successfully");

    std::vector<lowering::StaticGemmSpecializationRecordV1> records;
    std::string error;
    bool ok = lowering::lowerExplicitGemmToStaticSpecializationV1(
        bridged.module.get(), records, error);
    expect(ok, "lowered static 1x1x32 to specialization");
    if (records.size() == 1) {
      expect(records[0].m == 1 && records[0].n == 1 && records[0].k == 32, "1x1x32 dimensions");
      expect(records[0].function_symbol == "matcore_mlir_static_gemm_f32_1x1x32", "1x1x32 symbol");
      expect(records[0].definition_c.find("c[0] = acc;") != std::string::npos, "dot assignment to c[0]");
    }
  }

  // Test 3: GEMV special case (32x1x32)
  {
    const v1::Module capture = makeStaticModule(32, 1, 32);
    auto bridged = bridge::bridgeV1ToMatcoreMlir(
        capture, context, bridge::explicitGemmF32V1BridgeContext());
    expect(static_cast<bool>(bridged), "bridged static 32x1x32 module successfully");

    std::vector<lowering::StaticGemmSpecializationRecordV1> records;
    std::string error;
    bool ok = lowering::lowerExplicitGemmToStaticSpecializationV1(
        bridged.module.get(), records, error);
    expect(ok, "lowered static 32x1x32 to specialization");
    if (records.size() == 1) {
      expect(records[0].m == 32 && records[0].n == 1 && records[0].k == 32, "32x1x32 dimensions");
      expect(records[0].function_symbol == "matcore_mlir_static_gemm_f32_32x1x32", "32x1x32 symbol");
      expect(records[0].definition_c.find("c[i] = acc;") != std::string::npos, "gemv assignment to c[i]");
    }
  }

  // Test 4: GEVM special case (1x32x32)
  {
    const v1::Module capture = makeStaticModule(1, 32, 32);
    auto bridged = bridge::bridgeV1ToMatcoreMlir(
        capture, context, bridge::explicitGemmF32V1BridgeContext());
    expect(static_cast<bool>(bridged), "bridged static 1x32x32 module successfully");

    std::vector<lowering::StaticGemmSpecializationRecordV1> records;
    std::string error;
    bool ok = lowering::lowerExplicitGemmToStaticSpecializationV1(
        bridged.module.get(), records, error);
    expect(ok, "lowered static 1x32x32 to specialization");
    if (records.size() == 1) {
      expect(records[0].m == 1 && records[0].n == 32 && records[0].k == 32, "1x32x32 dimensions");
      expect(records[0].function_symbol == "matcore_mlir_static_gemm_f32_1x32x32", "1x32x32 symbol");
    }
  }

  // Test 5: Reject dynamic shapes
  {
    const v1::Module capture = readCapture(); // Original golden has dynamic symbols (M, N, K)
    auto bridged = bridge::bridgeV1ToMatcoreMlir(
        capture, context, bridge::explicitGemmF32V1BridgeContext());
    expect(static_cast<bool>(bridged), "bridged dynamic module successfully");

    std::vector<lowering::StaticGemmSpecializationRecordV1> records;
    std::string error;
    bool ok = lowering::lowerExplicitGemmToStaticSpecializationV1(
        bridged.module.get(), records, error);
    expect(!ok, "rejects dynamic shape module for static specialization");
    expect(error.find("static ranked tensor shapes") != std::string::npos, "appropriate error message");
  }

  std::cout << "MLIR static GEMM specialization lowering: " << checks
            << " checks, " << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
