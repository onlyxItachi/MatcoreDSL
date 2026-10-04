#ifndef MATCORE_MDSLC_GPU_FUSED_GEMM_CANDIDATE_H
#define MATCORE_MDSLC_GPU_FUSED_GEMM_CANDIDATE_H
#include "MatcoreGpuGemmCandidate.h"
namespace llvm { class Module; }

namespace matcore::mdslc::gpu_candidate {
inline constexpr char kFusedGemmKernelV1[] = "__matcore_strict_fused_gemm_f32_v1_kernel";

// Private build-issued strict arithmetic primitive C=A[M,K]*B[K,N];E=C*D[N,P].
// ONE kernel, grid/block exactly {1,1,1}; deliberately serial row-panel4 proof.
// Expanded ABI: A,B,D,E,workspace; each ptr,ptr,i64,i64,i64,i64,i64 (35 fields).
// All five descriptors have offset0 and strides{columns,1}; live allocations,
// signed-index/byte bounds including full logical M*N, original GEMM shape,
// profile/target limits and combined M*N*(K+P)<=2^18 are caller obligations.
// M,N,P MUST be positive. K may0 with null empty A/B, but consumer still executes
// 0*Inf/NaN. Caller retires required original guards before Eempty/N0 bypass;
// no zero-output outer loop or source bypass is authorized by this leaf.
// Inputs MAY alias. E[M,P] and workspace[min(4,M),N] are private, writable,
// disjoint from each other and all inputs. No tensor allocation/free/copy inside
// the kernel. Actual image gate separately verifies imports, target and strict
// arithmetic/FP controls; LLVM self-check alone is not a machine-code proof.
// This issuer grants no source/runtime/frontier/error/completion/residency or
// performance authority. Checked transfers, completion, quarantine and original
// source derivation are separate obligations of an explicit combined adapter.
struct GpuFusedGemmArtifactV1 {
  std::string semantic_ir, structured_ir, scheduled_ir, bufferized_ir;
  std::string outlined_ir, device_llvm_ir, manifest, error;
  explicit operator bool() const { return !device_llvm_ir.empty(); }
};
GpuFusedGemmArtifactV1 issueStrictGpuFusedGemmArtifactV1(
    mlir::MLIRContext &context, TargetV1 target);

// Exact pinned issuer self-consistency, NEVER arbitrary IR acceptance authority.
// These diagnostic predicates do not create an artifact or source-derived plan.
bool verifyStrictGpuFusedGemmOutlinedV1(mlir::ModuleOp module, std::string &error);
bool verifyStrictGpuFusedGemmLLVMV1(const llvm::Module &module, TargetV1 target,
                                 std::string &error);
} // namespace matcore::mdslc::gpu_candidate
#endif
