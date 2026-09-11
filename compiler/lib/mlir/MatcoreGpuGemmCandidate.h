#ifndef MATCORE_MDSLC_GPU_GEMM_CANDIDATE_H
#define MATCORE_MDSLC_GPU_GEMM_CANDIDATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <string>

namespace matcore::mdslc::gpu_candidate {

enum class TargetV1 { NvvmSm89, RocdlGfx1150 };
inline constexpr char kGemmKernelV1[] = "__matcore_strict_gemm_f32_v1_kernel";

struct GpuGemmArtifactV1 {
  std::string semantic_ir, structured_ir, bufferized_ir, outlined_ir;
  std::string fill_llvm_ir, gemm_llvm_ir, manifest, error;
  explicit operator bool() const {
    return error.empty() && !fill_llvm_ir.empty() && !gemm_llvm_ir.empty();
  }
};

// Inspection only: accepts exactly the reviewed scalar GPU schedule. This is
// not source authentication, an importer, or permission to execute supplied IR.
bool verifyStrictGpuGemmOutlinedV1(mlir::ModuleOp, std::string &error);

// Closed issuer: starts from the existing compiler-owned strict mathematical
// primitive, never caller-supplied IR. Emits two device modules: fill first,
// GEMM second, each with kGemmKernelV1; launch ABI is respectively 7 and 21
// expanded dynamic rank-2 memref fields. No target facts enter semantic IR.
// Machine compilation must retain -fp-contract=off and IEEE denormals. NVVM
// cubin assembly additionally requires --fmad=false. The runtime must guard
// shapes/grid bounds, host/device transfers, live storage, input/output
// isolation, completed execution, and artifact/device identity.
GpuGemmArtifactV1 issueStrictGpuGemmArtifactV1(mlir::MLIRContext &, TargetV1);

} // namespace matcore::mdslc::gpu_candidate
#endif
