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
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence,
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
// structurally verified explicit-v1 or recovered-analysis module. This API is
// diagnostic-only and intentionally does not authenticate source provenance.
// Origin, provenance, source-expression spelling, site/symbol identity,
// policy, and numerical profile/derivation labels are excluded. Expanded
// numerical semantics and every tensor/effect/alias contract remain.
bool fingerprintStructuralMathematicalGemmV1(
    mlir::ModuleOp module, MathematicalGemmFingerprintV1 &fingerprint,
    std::string &error);

bool equivalentStructuralMathematicalGemmV1(
    mlir::ModuleOp left, mlir::ModuleOp right, bool &equivalent,
    std::string &error);

struct AuthenticatedExplicitRecoveredEquivalenceV1 {
  bool equivalent = false;
  MathematicalGemmFingerprintV1 explicit_fingerprint;
  MathematicalGemmFingerprintV1 recovered_fingerprint;
  std::string error;

  explicit operator bool() const { return error.empty(); }
};

// Authenticated equivalence accepts only immutable native-frontend evidence.
// It internally routes the explicit site through v0 -> verified v1 -> MLIR and
// the recovered site through its source-proven analysis bridge, then compares
// their structural mathematical fingerprints. It returns no MLIR wrapper that
// a caller could reuse as execution permission.
AuthenticatedExplicitRecoveredEquivalenceV1
compareAuthenticatedExplicitAndRecoveredGemmV1(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &explicit_evidence,
    std::size_t explicit_operation_index,
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &recovered_evidence,
    std::size_t recovered_candidate_index);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_MATCORE_RECOVERED_GEMM_BRIDGE_H
