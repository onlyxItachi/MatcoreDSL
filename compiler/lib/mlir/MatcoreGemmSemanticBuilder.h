#ifndef MATCORE_MDSLC_MLIR_MATCORE_GEMM_SEMANTIC_BUILDER_H
#define MATCORE_MDSLC_MLIR_MATCORE_GEMM_SEMANTIC_BUILDER_H

#include "matcore_ir_v1.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"

#include <cstddef>
#include <string>
#include <vector>

namespace matcore::mdslc::mlir_bridge {

// Internal, non-serialized construction contract shared by authenticated
// explicit capture and authenticated recovered-source analysis. It contains
// semantic tensor/effect data, but deliberately has no canonical-callee field
// and cannot itself grant execution permission.
struct GemmSemanticSiteV1 {
  std::string site_id;
  std::size_t ordinal = 0;
  ir::v1::TensorValue lhs;
  ir::v1::TensorValue rhs;
  ir::v1::TensorValue output;
  ir::v1::DType accumulation_dtype = ir::v1::DType::F32;
  std::vector<ir::v1::SemanticRequirement> requirements;
  std::vector<ir::v1::AliasRequirement> alias_requirements;
  ir::v1::Effects effects;
  std::string target;
  std::string fallback;
  mlir::DictionaryAttr origin;
  mlir::DictionaryAttr numerical;
  mlir::DictionaryAttr provenance;
  mlir::LocationAttr source_location;
};

// Appends one destination-tied mdsl.gemm semantic root. Dialect verification
// remains authoritative for the closed origin/numerical/provenance
// cross-product. This function performs no source authentication and no
// planning or lowering.
bool appendGemmSemanticSiteV1(mlir::ModuleOp module, mlir::OpBuilder &builder,
                              const GemmSemanticSiteV1 &site,
                              std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_MATCORE_GEMM_SEMANTIC_BUILDER_H
