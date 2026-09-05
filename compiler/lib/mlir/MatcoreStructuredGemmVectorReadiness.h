#ifndef MATCORE_MDSLC_MLIR_STRUCTURED_GEMM_VECTOR_READINESS_H
#define MATCORE_MDSLC_MLIR_STRUCTURED_GEMM_VECTOR_READINESS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <string>

namespace matcore::mdslc::mlir_bridge {

inline constexpr std::uint32_t kStructuredGemmVectorReadinessVersionV1 = 1;
inline constexpr char kStructuredGemmVectorReadinessSchemaV1[] =
    "matcore-structured-gemm-vector-readiness-v1";

struct StructuredGemmVectorReadinessResultV1 {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;

  explicit operator bool() const { return static_cast<bool>(module); }
};

// Registers the exact upstream dialects and the Linalg Transform dialect
// extension used by this inspection seam. This does not register or authorize
// an executable lowering pipeline.
void registerStructuredGemmVectorReadinessDialectsV1(
    mlir::MLIRContext &context);

// Returns the deterministic, target-independent Transform dialect schedule
// used by this checkpoint. It selects isolated func.func payloads and applies
// upstream transform.structured.vectorize_children_and_apply_patterns without
// tile sizes, vector widths, scalable-vector choices, or target information.
std::string structuredGemmVectorReadinessTransformV1();

// Applies the schedule above to a clone of an exact verified structured GEMM
// handoff. Only fully static positive rank-2 f32 GEMM is admitted: upstream
// regular vectorization cannot select a target-independent vector shape for a
// dynamic problem. The result is inspection-only and carries no execution
// authority.
StructuredGemmVectorReadinessResultV1
deriveStructuredGemmVectorReadinessV1(mlir::ModuleOp structured_module);

// Verifies the self-consistent retained structured-source identity, the exact
// semantic/consumption ledger, and the resulting whole-static-problem
// vector.contract dataflow. In particular, the contract accumulator must be
// positive zero and the sole transfer write must target the original output
// argument; reading initial C is forbidden. Standalone verification does not
// prove which external structured module supplied the retained identity.
bool verifyStructuredGemmVectorReadinessV1(mlir::ModuleOp vector_module,
                                           std::string &error);

// Additionally proves that the vector module is paired with the particular
// exact verified structured handoff supplied by the caller, using the shared
// derived-structured certificate and ordered site-set fingerprints. Both
// inputs are read-only. This is substitution detection and source pairing,
// not authority for imported MLIR.
bool verifyStructuredGemmVectorReadinessMatchesV1(
    mlir::ModuleOp structured_module, mlir::ModuleOp vector_module,
    std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_STRUCTURED_GEMM_VECTOR_READINESS_H
