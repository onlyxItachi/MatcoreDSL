#include "MatcoreStaticGemmLowering.h"

#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace matcore::mdslc::mlir_lowering {
namespace {

bool readUnsigned(mlir::DictionaryAttr dictionary, llvm::StringRef name,
                  std::uint64_t &value, std::string &error) {
  const auto attribute = dictionary.getAs<mlir::IntegerAttr>(name);
  if (!attribute || attribute.getInt() < 0) {
    error = "MLIR static GEMM lowering requires nonnegative provenance." +
            name.str();
    return false;
  }
  const llvm::APInt encoded = attribute.getValue();
  if (encoded.getActiveBits() > 64) {
    error = "MLIR static GEMM provenance." + name.str() + " exceeds uint64";
    return false;
  }
  value = encoded.getZExtValue();
  return true;
}

bool readString(mlir::DictionaryAttr dictionary, llvm::StringRef name,
                std::string &value, std::string &error) {
  const auto attribute = dictionary.getAs<mlir::StringAttr>(name);
  if (!attribute || attribute.getValue().empty()) {
    error = "MLIR static GEMM lowering requires provenance." + name.str();
    return false;
  }
  value = attribute.getValue().str();
  return true;
}

std::string generateKernelDeclaration(const std::string &symbol) {
  return "void " + symbol + "(float* MATCORE_RESTRICT c, const float* MATCORE_RESTRICT a, const float* MATCORE_RESTRICT b);";
}

std::string generateKernelDefinition(const std::string &symbol, std::int64_t m,
                                     std::int64_t n, std::int64_t k) {
  std::ostringstream os;
  os << "void " << symbol << "(float* MATCORE_RESTRICT c, const float* MATCORE_RESTRICT a, const float* MATCORE_RESTRICT b) {\n";

  // Dot product special case (1x1xK)
  if (m == 1 && n == 1) {
    os << "  float acc = 0.0f;\n";
    if (k % 4 == 0 && k > 0) {
      os << "  for (int64_t p = 0; p < " << k << "; p += 4) {\n";
      os << "    acc += a[p + 0] * b[p + 0];\n";
      os << "    acc += a[p + 1] * b[p + 1];\n";
      os << "    acc += a[p + 2] * b[p + 2];\n";
      os << "    acc += a[p + 3] * b[p + 3];\n";
      os << "  }\n";
    } else {
      os << "  for (int64_t p = 0; p < " << k << "; ++p) {\n";
      os << "    acc += a[p] * b[p];\n";
      os << "  }\n";
    }
    os << "  c[0] = acc;\n";
    os << "}\n";
    return os.str();
  }

  // Vector-Matrix special case (1xNxK)
  if (m == 1) {
    os << "  for (int64_t j = 0; j < " << n << "; ++j) {\n";
    os << "    c[j] = 0.0f;\n";
    os << "  }\n";
    os << "  for (int64_t p = 0; p < " << k << "; ++p) {\n";
    os << "    const float a_val = a[p];\n";
    os << "    const float* b_row = b + p * " << n << ";\n";
    os << "    for (int64_t j = 0; j < " << n << "; ++j) {\n";
    os << "      c[j] += a_val * b_row[j];\n";
    os << "    }\n";
    os << "  }\n";
    os << "}\n";
    return os.str();
  }

  // Matrix-Vector special case (Mx1xK)
  if (n == 1) {
    os << "  for (int64_t i = 0; i < " << m << "; ++i) {\n";
    os << "    const float* a_row = a + i * " << k << ";\n";
    os << "    float acc = 0.0f;\n";
    os << "    for (int64_t p = 0; p < " << k << "; ++p) {\n";
    os << "      acc += a_row[p] * b[p];\n";
    os << "    }\n";
    os << "    c[i] = acc;\n";
    os << "  }\n";
    os << "}\n";
    return os.str();
  }

  // General 2D GEMM
  os << "  for (int64_t i = 0; i < " << m << "; ++i) {\n";
  os << "    float* c_row = c + i * " << n << ";\n";
  os << "    for (int64_t j = 0; j < " << n << "; ++j) {\n";
  os << "      c_row[j] = 0.0f;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  for (int64_t i = 0; i < " << m << "; ++i) {\n";
  os << "    const float* a_row = a + i * " << k << ";\n";
  os << "    float* c_row = c + i * " << n << ";\n";
  os << "    for (int64_t p = 0; p < " << k << "; ++p) {\n";
  os << "      const float a_val = a_row[p];\n";
  os << "      const float* b_row = b + p * " << n << ";\n";
  os << "      for (int64_t j = 0; j < " << n << "; ++j) {\n";
  os << "        c_row[j] += a_val * b_row[j];\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";
  os << "}\n";
  return os.str();
}

} // namespace

bool lowerExplicitGemmToStaticSpecializationV1(
    mlir::ModuleOp module,
    std::vector<StaticGemmSpecializationRecordV1> &records,
    std::string &error) {
  records.clear();
  error.clear();

  if (module && module->hasAttr("mdsl.analysis_only")) {
    error = "MLIR static GEMM lowering rejects mdsl.analysis_only modules";
    return false;
  }
  if (!mlir_bridge::verifyMatcoreV1BridgeModule(module, error)) {
    error = "MLIR static GEMM lowering requires the verified explicit "
            "Matcore IR v1 bridge envelope: " +
            error;
    return false;
  }

  const auto producer =
      module->getAttrOfType<mlir::StringAttr>("mdsl.producer");
  if (!producer || producer.getValue() != "clang-libtooling-v1") {
    error = "MLIR static GEMM lowering requires executable authorization "
            "from mdsl.producer=clang-libtooling-v1; bootstrap capture is "
            "inspection-only";
    return false;
  }

  const auto execution_intent =
      module->getAttrOfType<mlir::StringAttr>("mdsl.execution_intent");
  const auto numerical_profile =
      module->getAttrOfType<mlir::StringAttr>("mdsl.numerical_profile");
  if (!execution_intent || !numerical_profile) {
    error = "MLIR static GEMM lowering requires explicit module context";
    return false;
  }

  std::vector<StaticGemmSpecializationRecordV1> pending;
  for (mlir::Operation &operation : module.getBody()->getOperations()) {
    auto function = mlir::cast<mlir::func::FuncOp>(operation);
    auto gemm = mlir::cast<mlir_dialect::GemmOp>(
        function.getBody().front().front());

    auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(gemm.getLhs().getType());
    auto rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(gemm.getRhs().getType());
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(gemm.getOutput().getType());

    if (!lhs_type || !rhs_type || !out_type ||
        !lhs_type.hasStaticShape() || !rhs_type.hasStaticShape() || !out_type.hasStaticShape()) {
      error = "MLIR static GEMM lowering requires static ranked tensor shapes on all operands";
      return false;
    }

    const std::int64_t m = out_type.getDimSize(0);
    const std::int64_t n = out_type.getDimSize(1);
    const std::int64_t k = lhs_type.getDimSize(1);

    if (m <= 0 || n <= 0 || k <= 0) {
      error = "MLIR static GEMM lowering requires positive static dimensions";
      return false;
    }

    const auto provenance = gemm.getProvenance();
    const auto call_range =
        provenance.getAs<mlir::DictionaryAttr>("call_range");
    if (!call_range) {
      error = "verified GEMM lacks call range provenance";
      return false;
    }

    StaticGemmSpecializationRecordV1 record;
    record.site_id = gemm.getSiteId().str();
    record.m = m;
    record.n = n;
    record.k = k;
    record.dtype = "f32";
    record.accumulation_dtype = "f32";
    record.function_symbol = "matcore_mlir_static_gemm_f32_" + std::to_string(m) +
                             "x" + std::to_string(n) + "x" + std::to_string(k);
    record.declaration_c = generateKernelDeclaration(record.function_symbol);
    record.definition_c = generateKernelDefinition(record.function_symbol, m, n, k);

    if (!readString(provenance, "file", record.source_file, error) ||
        !readUnsigned(provenance, "line", record.source_line, error) ||
        !readUnsigned(provenance, "column", record.source_column, error) ||
        !readUnsigned(call_range, "begin", record.source_range_begin, error) ||
        !readUnsigned(call_range, "end", record.source_range_end, error)) {
      return false;
    }

    if (record.source_range_end <= record.source_range_begin) {
      error = "MLIR static GEMM lowering requires a nonempty call range";
      return false;
    }

    pending.push_back(std::move(record));
  }

  records = std::move(pending);
  return true;
}

} // namespace matcore::mdslc::mlir_lowering
