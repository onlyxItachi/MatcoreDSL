#ifndef MATCORE_MDSLC_MLIR_STRUCTURED_GEMM_HANDOFF_H
#define MATCORE_MDSLC_MLIR_STRUCTURED_GEMM_HANDOFF_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <string>

namespace matcore::mdslc::mlir_bridge {

struct StructuredHandoffCertificateProfileV1;

inline constexpr std::uint32_t kStructuredGemmHandoffVersionV1 = 1;
inline constexpr char kStructuredGemmHandoffSchemaV1[] =
    "matcore-structured-gemm-handoff-v1";
inline constexpr char kStructuredGemmInspectionAuthorityV1[] =
    "inspection_only";

struct StructuredGemmHandoffResultV1 {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;

  explicit operator bool() const { return static_cast<bool>(module); }
};

// Registers the exact upstream dialects used by the structured inspection
// handoff. This does not register a lowering pipeline.
void registerStructuredGemmHandoffDialectsV1(mlir::MLIRContext &context);

// Internal profile shared by certified transformations derived from the
// structured GEMM handoff. This is not a public compiler/runtime contract.
const StructuredHandoffCertificateProfileV1 &
structuredGemmHandoffCertificateProfileV1();

// Derives a new structured-only module from the exact explicit Matcore IR v1
// semantic bridge envelope. The input is never mutated. Every source
// mdsl.gemm is replaced in the result by:
//
//   arith.constant +0.0 -> linalg.fill(output) ->
//   linalg.matmul(lhs, rhs, filled-output) -> func.return
//
// The complete mdsl.gemm attribute dictionary is retained as a versioned,
// verifier-checked function contract. The result is inspection-only and is
// not executable authorization.
StructuredGemmHandoffResultV1
deriveStructuredGemmHandoffV1(mlir::ModuleOp semantic_module);

// Verifies the self-contained structured envelope, including its retained
// semantic contract, exact overwrite dataflow, logical matmul maps, scalar
// region, numerical flags, provenance/location agreement, and inspection-only
// authority. Normal MLIR verification alone is insufficient.
bool verifyStructuredGemmHandoffV1(mlir::ModuleOp structured_module,
                                   std::string &error);

// Additionally proves that a structured module is the exact projection of a
// particular verified semantic module. Both inputs are read-only.
bool verifyStructuredGemmHandoffMatchesV1(mlir::ModuleOp semantic_module,
                                          mlir::ModuleOp structured_module,
                                          std::string &error);

// Revalidates the exact retained mdsl.gemm contract against its original
// tensor function type for a certified downstream carrier. Source pairing is
// handled by MatcoreStructuredHandoffCertificate; this helper owns only GEMM
// semantic-contract interpretation.
bool verifyRetainedStructuredGemmContractV1(
    mlir::ModuleOp carrier_module, mlir::func::FuncOp carrier_function,
    mlir::FunctionType source_structured_function_type, std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_STRUCTURED_GEMM_HANDOFF_H
