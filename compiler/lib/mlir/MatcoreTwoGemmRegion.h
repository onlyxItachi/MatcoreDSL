#ifndef MATCORE_MDSLC_MLIR_TWO_GEMM_REGION_H
#define MATCORE_MDSLC_MLIR_TWO_GEMM_REGION_H

#include "frontend.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace matcore::mdslc::mlir_bridge {

struct TwoGemmRegionResultV1 {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;
  explicit operator bool() const { return static_cast<bool>(module); }
};

void registerTwoGemmRegionDialectsV1(mlir::MLIRContext &context);

// Requires immutable native admission evidence. The resulting region is an
// inspection representation, never generated-code or source-rewrite authority.
TwoGemmRegionResultV1 deriveAuthenticatedTwoGemmRegionsV1(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence,
    mlir::MLIRContext &context);

// Self-consistency only. Editable metadata cannot authenticate source bytes.
bool verifyTwoGemmRegionModuleV1(mlir::ModuleOp module, std::string &error);

// Recomputes source contracts and descriptor relationships from sealed native
// evidence. Computation may use named or generalized upstream Linalg and need
// not retain an exact operation count, incidental order, or source location.
bool verifyTwoGemmRegionMatchesNativeEvidenceV1(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence,
    mlir::ModuleOp module, std::string &error);

} // namespace matcore::mdslc::mlir_bridge
#endif
