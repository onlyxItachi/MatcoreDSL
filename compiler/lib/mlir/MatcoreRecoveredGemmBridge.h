#ifndef MATCORE_MDSLC_MLIR_MATCORE_RECOVERED_GEMM_BRIDGE_H
#define MATCORE_MDSLC_MLIR_MATCORE_RECOVERED_GEMM_BRIDGE_H

#include "frontend.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstddef>
#include <string>

namespace matcore::mdslc::mlir_bridge {

inline constexpr char kRecoveredGemmCaptureSchemaV1[] =
    "recovered-cpp-gemm-inspection-v1";
inline constexpr char kRecoveredGemmNumericalProfileV1[] =
    "recovered-cpp-gemm-f32-source-proven-v1";

struct RecoveredGemmBridgeResultV1 {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;

  explicit operator bool() const { return static_cast<bool>(module); }
};

// Authenticates one diagnostic-only candidate against the exact source bytes
// and extraction options that produced it, then builds one recovered mdsl.gemm
// analysis site. The result never authorizes rewrite or execution.
RecoveredGemmBridgeResultV1 bridgeRecoveredGemmToMatcoreMlirV1(
    const frontend::Result &source, const frontend::Options &options,
    std::size_t candidate_index, mlir::MLIRContext &context);

// Verifies the closed analysis-only module envelope. Source authentication is
// performed by bridgeRecoveredGemmToMatcoreMlirV1 before construction; this
// structural verifier is not a replacement for that trusted producer step.
bool verifyRecoveredGemmAnalysisModuleV1(mlir::ModuleOp module,
                                         std::string &error);

struct MathematicalGemmFingerprintV1 {
  std::string canonical_contract;
  std::string sha256;
};

// Produces a normalized mathematical WHAT fingerprint for exactly one
// authenticated explicit-v1 or recovered-analysis module. Origin,
// provenance, source-expression spelling, site/symbol identity, policy, and
// numerical profile/derivation labels are excluded. Expanded numerical
// semantics and every tensor/effect/alias contract remain.
bool fingerprintMathematicalGemmV1(
    mlir::ModuleOp module, MathematicalGemmFingerprintV1 &fingerprint,
    std::string &error);

bool equivalentMathematicalGemmV1(mlir::ModuleOp left, mlir::ModuleOp right,
                                  bool &equivalent, std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_MATCORE_RECOVERED_GEMM_BRIDGE_H
