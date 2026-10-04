#ifndef MATCORE_MDSLC_MLIR_CPU_FUSED_GEMM_CANDIDATE_H
#define MATCORE_MDSLC_MLIR_CPU_FUSED_GEMM_CANDIDATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace llvm { class Module; }
namespace matcore::mdslc::cpu_candidate {

inline constexpr char kStrictFusedGemmSymbolV1[] = "__matcore_strict_fused_gemm_f32_v1";
inline constexpr char kStrictFusedGemmCInterfaceV1[] =
    "_mlir_ciface___matcore_strict_fused_gemm_f32_v1";

// Compiler-private arithmetic primitive only: C=A[M,K]*B[K,N], E=C*D[N,P].
// Five canonical dense dynamic rank-2 f32 descriptors: A, B, D, E, workspace.
// Caller must validate both original GEMM shape/profile/frontier obligations,
// all signed-index and byte-count products INCLUDING full logical M*N, live
// capacities/objects, host access, offset=0, strides={columns,1}, race freedom,
// nearest-even/gradual-underflow/masked FP controls and actual leaf availability.
// E[M,P] and workspace[min(4,M),N] are writable private storage, disjoint from
// each other and all inputs. A/B/D may alias. E MUST be nonempty: caller retires
// both original guard frontiers before skipping M=0/P=0; no zero-output loop is
// authorized. Validated E byte extent also bounds row IV+4 without overflow.
// Empty input/workspace data may be null; descriptor objects must exist. K=0
// with N>0/nonempty E must still evaluate the consumer (0*Inf produces NaN).
// The leaf allocates/frees no tensor storage, performs no
// publication, guards no runtime precondition and restores no caller FP state.
// O2 may lower positive-zero fills to trusted conforming memset; do not infer
// machine-code call freedom from the preoptimization LLVM whitelist.
// Trusted memset must have only the bounded private fill write effects, no
// recoverable failure/arbitrary host effects, and preserve FP control state.
struct StrictFusedGemmStagesV1 {
  mlir::OwningOpRef<mlir::ModuleOp> semantic;
  mlir::OwningOpRef<mlir::ModuleOp> structured;
  mlir::OwningOpRef<mlir::ModuleOp> scheduled;
  mlir::OwningOpRef<mlir::ModuleOp> bufferized;
  std::string error;
  explicit operator bool() const { return bool(bufferized); }
};

StrictFusedGemmStagesV1 buildStrictFusedGemmStagesV1(mlir::MLIRContext &context);
// Exact built-in stage self-consistency, not source/execution authority. No
// transformation of the original whole-region paired witness is authorized.
bool verifyStrictFusedGemmStructuredV1(mlir::ModuleOp module, std::string &error);
bool verifyStrictFusedGemmScheduledV1(mlir::ModuleOp module, std::string &error);
bool verifyStrictFusedGemmBufferizedV1(mlir::ModuleOp module, std::string &error);
// Issuer-owned preoptimization LLVM only. This verifier is not a general LLVM
// importer/body theorem or an artifact-derived runtime execution certificate.
bool verifyStrictFusedGemmLLVMV1(const llvm::Module &module,
                               bool address_sanitizer, std::string &error);

struct StrictFusedGemmArtifactV1 {
  std::string semantic_ir, structured_ir, scheduled_ir, bufferized_ir;
  std::string transform_ir, llvm_ir, manifest, error;
  explicit operator bool() const { return !llvm_ir.empty(); }
};
// Closed built-in issuer, exact 21.1.8 and Linux x64 baseline only. It accepts
// no source, supplied MLIR/LLVM or caller Transform program. Runtime/source
// matching, guard retirement and dispatch integration are separate boundaries.
StrictFusedGemmArtifactV1 issueStrictFusedGemmArtifactV1(
    mlir::MLIRContext &context, bool address_sanitizer);

} // namespace matcore::mdslc::cpu_candidate
#endif
