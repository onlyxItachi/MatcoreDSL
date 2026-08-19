#include "MatcoreCpuRuntimeLowering.h"

#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <iterator>
#include <utility>

namespace matcore::mdslc::mlir_lowering {
namespace {

bool readUnsigned(mlir::DictionaryAttr dictionary, llvm::StringRef name,
                  std::uint64_t &value, std::string &error) {
  const auto attribute = dictionary.getAs<mlir::IntegerAttr>(name);
  if (!attribute || attribute.getInt() < 0) {
    error = "CPU runtime-dispatch lowering requires nonnegative provenance." +
            name.str();
    return false;
  }
  const llvm::APInt encoded = attribute.getValue();
  if (encoded.getActiveBits() > 64) {
    error = "CPU runtime-dispatch provenance." + name.str() +
            " exceeds uint64";
    return false;
  }
  value = encoded.getZExtValue();
  return true;
}

bool readString(mlir::DictionaryAttr dictionary, llvm::StringRef name,
                std::string &value, std::string &error) {
  const auto attribute = dictionary.getAs<mlir::StringAttr>(name);
  if (!attribute || attribute.getValue().empty()) {
    error = "CPU runtime-dispatch lowering requires provenance." + name.str();
    return false;
  }
  value = attribute.getValue().str();
  return true;
}

} // namespace

bool lowerExplicitGemmToCpuRuntimeDispatchV1(
    mlir::ModuleOp module, std::vector<CpuRuntimeDispatchRecordV1> &records,
    std::string &error) {
  records.clear();
  error.clear();
  if (module && module->hasAttr("mdsl.analysis_only")) {
    error = "CPU runtime-dispatch lowering rejects mdsl.analysis_only "
            "modules; analysis evidence is never executable permission";
    return false;
  }
  if (!mlir_bridge::verifyMatcoreV1BridgeModule(module, error)) {
    error = "CPU runtime-dispatch lowering requires the verified explicit "
            "Matcore IR v1 bridge envelope: " +
            error;
    return false;
  }
  const auto producer =
      module->getAttrOfType<mlir::StringAttr>("mdsl.producer");
  if (!producer || producer.getValue() != "clang-libtooling-v1") {
    error = "CPU runtime-dispatch lowering requires executable authorization "
            "from mdsl.producer=clang-libtooling-v1; bootstrap capture is "
            "inspection-only";
    return false;
  }

  const auto execution_intent =
      module->getAttrOfType<mlir::StringAttr>("mdsl.execution_intent");
  const auto numerical_profile =
      module->getAttrOfType<mlir::StringAttr>("mdsl.numerical_profile");
  if (!execution_intent || !numerical_profile) {
    error = "CPU runtime-dispatch lowering requires explicit module context";
    return false;
  }

  std::vector<CpuRuntimeDispatchRecordV1> pending;
  pending.reserve(std::distance(module.getBody()->begin(),
                                module.getBody()->end()));
  for (mlir::Operation &operation : module.getBody()->getOperations()) {
    auto function = mlir::cast<mlir::func::FuncOp>(operation);
    auto gemm = mlir::cast<mlir_dialect::GemmOp>(
        function.getBody().front().front());
    const auto policy = gemm.getPolicy();
    const auto target = policy.getAs<mlir::StringAttr>("target");
    const auto fallback = policy.getAs<mlir::StringAttr>("fallback");
    const auto provenance = gemm.getProvenance();
    const auto call_range =
        provenance.getAs<mlir::DictionaryAttr>("call_range");
    if (!target || !fallback || !call_range) {
      error = "verified GEMM unexpectedly lacks CPU policy or call range";
      return false;
    }

    CpuRuntimeDispatchRecordV1 record;
    record.site_id = gemm.getSiteId().str();
    record.semantic_symbol = function.getName().str();
    record.operation = "gemm";
    record.dtype = "f32";
    record.accumulation_dtype = "f32";
    record.target = target.getValue().str();
    record.fallback = fallback.getValue().str();
    record.numerical_profile = numerical_profile.getValue().str();
    record.execution_intent = execution_intent.getValue().str();
    record.runtime_symbol = kCpuRuntimeDispatchSymbolV1;
    auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(gemm.getLhs().getType());
    auto rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(gemm.getRhs().getType());
    auto out_type = mlir::dyn_cast<mlir::RankedTensorType>(gemm.getOutput().getType());
    if (lhs_type && rhs_type && out_type &&
        lhs_type.hasStaticShape() && rhs_type.hasStaticShape() && out_type.hasStaticShape()) {
      record.static_m = out_type.getDimSize(0);
      record.static_n = out_type.getDimSize(1);
      record.static_k = lhs_type.getDimSize(1);
    }
    record.required_guards = {
        "descriptor_v0", "alias_no_overlap", "required_alignment",
        "explicit_gemm_f32_v1_fp_environment"};
    if (!readString(provenance, "file", record.source_file, error) ||
        !readUnsigned(provenance, "line", record.source_line, error) ||
        !readUnsigned(provenance, "column", record.source_column, error) ||
        !readUnsigned(call_range, "begin", record.source_range_begin, error) ||
        !readUnsigned(call_range, "end", record.source_range_end, error)) {
      return false;
    }
    if (record.source_range_end <= record.source_range_begin) {
      error = "CPU runtime-dispatch lowering requires a nonempty call range";
      return false;
    }
    pending.push_back(std::move(record));
  }
  records = std::move(pending);
  return true;
}

} // namespace matcore::mdslc::mlir_lowering
