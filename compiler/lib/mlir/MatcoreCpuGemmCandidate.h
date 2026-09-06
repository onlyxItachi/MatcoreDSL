#ifndef MATCORE_MDSLC_MLIR_CPU_GEMM_CANDIDATE_H
#define MATCORE_MDSLC_MLIR_CPU_GEMM_CANDIDATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace matcore::mdslc::cpu_candidate {

inline constexpr char kStrictGemmSymbolV1[] = "__matcore_strict_gemm_f32_v1";
inline constexpr char kStrictGemmCInterfaceV1[] =
    "_mlir_ciface___matcore_strict_gemm_f32_v1";
inline constexpr char kCpuTargetV1[] = "x86_64-pc-linux-gnu";

// Private compiler-owned primitive, not source admission and not a public ABI.
// The leaf accepts three canonical dynamic rank-2 memref descriptors. Callers
// must check M/K/N relations, nonnegative signed-index extents, representable
// byte counts, offset=0 and strides={columns,1}, host f32 access, live
// capacity, race freedom, compatible FP controls, and a destination isolated
// from both inputs. Zero M/N is empty; zero K overwrites every output element
// with positive zero. The descriptor objects themselves must exist; empty
// operand data may be null. The leaf allocates no tensor storage, performs no
// publication, and preserves neither host FP status nor failure ordering
// itself: the region adapter owns these obligations. No imported/serialized
// MLIR can be submitted to its issuer.
struct StrictGemmStagesV1 {
  mlir::OwningOpRef<mlir::ModuleOp> semantic;
  mlir::OwningOpRef<mlir::ModuleOp> structured;
  mlir::OwningOpRef<mlir::ModuleOp> bufferized;
  std::string error;
  explicit operator bool() const { return bool(bufferized); }
};

StrictGemmStagesV1 buildStrictGemmStagesV1(mlir::MLIRContext &context);
// Self-consistency checks only; none grants source/execution authority. Linalg
// reduction iterator identity does not itself encode increasing-K order. Only
// the issuer's fixed scalar loop pipeline is defended here; arbitrary tiling,
// vectorization or reassociation needs a separate numerical-order proof.
bool verifyStrictGemmStructuredV1(mlir::ModuleOp module, std::string &error);
bool verifyStrictGemmBufferizedV1(mlir::ModuleOp module, std::string &error);

struct StrictGemmArtifactV1 {
  std::string llvm_ir;
  std::string semantic_ir;
  std::string structured_ir;
  std::string bufferized_ir;
  std::string manifest;
  std::string error;
  explicit operator bool() const { return !llvm_ir.empty(); }
};

// Closed issuer: only the built-in verified strict primitive, exact 21.1.8
// pipeline and baseline Linux x86-64 target. Address instrumentation is carried
// as LLVM function attributes, not presumed from the host link command.
StrictGemmArtifactV1 issueStrictGemmArtifactV1(mlir::MLIRContext &context,
                                               bool address_sanitizer);

} // namespace matcore::mdslc::cpu_candidate
#endif
