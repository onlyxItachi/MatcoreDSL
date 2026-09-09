#ifndef MATCORE_MDSLC_MLIR_CPU_REASSOCIATE_GEMM_CANDIDATE_H
#define MATCORE_MDSLC_MLIR_CPU_REASSOCIATE_GEMM_CANDIDATE_H

#include "MatcoreCpuGemmCandidate.h"

namespace matcore::mdslc::cpu_candidate {
inline constexpr char kReassociateGemmSymbolV1[] =
    "__matcore_reassociate_gemm_f32_avx2_v1";
inline constexpr char kReassociateGemmCInterfaceV1[] =
    "_mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1";

// A distinct numerical primitive, NOT a StrictGemmScheduleV1. Per-operation
// reassociate_f32 permits the fused full-tile and separately rounded scalar-tail
// realizations. No blanket fast math or cross-GEMM reassociation is authorized.
GemmStagesV1 buildReassociateGemmStagesV1(mlir::MLIRContext &context);
bool verifyReassociateGemmStructuredV1(mlir::ModuleOp module, std::string &error);
bool verifyReassociateGemmBufferizedV1(mlir::ModuleOp module, std::string &error);

// Fixed, pinned, serial upstream Transform derivation. Verification combines
// narrow independently checked numerical/storage invariants with exact replay
// identity of the trusted upstream pipeline. It does not independently prove
// upstream transformations or authorize arbitrary supplied IR for execution.
mlir::OwningOpRef<mlir::ModuleOp>
deriveReassociateGemmRegisterV1(mlir::ModuleOp bufferized, std::string &error);
bool verifyReassociateGemmRegisterV1(mlir::ModuleOp module, std::string &error);
bool verifyReassociateGemmLLVMV1(llvm::Module &module, std::string &error);

// Same private storage/shape/FP caller contract as the strict leaf. Additionally
// requires source reassociate_f32 permission and runtime AVX2/FMA + OS XMM/YMM
// legality before invocation. This issuer creates no source/runtime authority.
// Isolated object target: baseline x86-64 plus AVX2/FMA, NOT x86-64-v3.
GemmArtifactV1 issueReassociateGemmArtifactV1(mlir::MLIRContext &context,
                                           bool address_sanitizer);
} // namespace matcore::mdslc::cpu_candidate
#endif
