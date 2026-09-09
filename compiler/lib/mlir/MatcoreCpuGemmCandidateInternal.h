#ifndef MATCORE_MDSLC_MLIR_CPU_GEMM_CANDIDATE_INTERNAL_H
#define MATCORE_MDSLC_MLIR_CPU_GEMM_CANDIDATE_INTERNAL_H

#include <string>
namespace llvm { class Function; }
namespace matcore::mdslc::cpu_candidate::detail {
// Only after each closed issuer verifies its own exact module/body/intrinsics.
// Shared private descriptor mapping check; not module or execution authority.
bool preserveGemmOutputData(llvm::Function &leaf, llvm::Function &wrapper,
                            std::string &error);
}
#endif
