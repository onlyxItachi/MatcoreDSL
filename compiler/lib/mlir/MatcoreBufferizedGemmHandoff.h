#ifndef MATCORE_MDSLC_MLIR_BUFFERIZED_GEMM_HANDOFF_H
#define MATCORE_MDSLC_MLIR_BUFFERIZED_GEMM_HANDOFF_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <string>

namespace matcore::mdslc::mlir_bridge {

inline constexpr std::uint32_t kBufferizedGemmHandoffVersionV1 = 1;
inline constexpr char kBufferizedGemmHandoffSchemaV1[] =
    "matcore-bufferized-gemm-handoff-v1";

struct BufferizedGemmHandoffResultV1 {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::uint64_t buffer_allocations = 0;
  std::uint64_t buffer_deallocations = 0;
  std::uint64_t tensor_in_place = 0;
  std::uint64_t tensor_out_of_place = 0;
  std::string error;

  explicit operator bool() const { return static_cast<bool>(module); }
};

// Registers only the upstream dialects and BufferizableOpInterface models used
// by this inspection boundary. This does not register an executable lowering.
void registerBufferizedGemmHandoffDialectsV1(mlir::MLIRContext &context);

// Clones an exact, verified structured-GEMM-v1 module and runs upstream MLIR
// One-Shot Module Bufferize with function boundaries and identity layout maps.
// The source module is never mutated. The result remains inspection-only.
// Derivation fails unless every function has all of these postconditions:
//
//   linalg.fill outs(original output argument)
//   linalg.matmul outs(the same output argument)
//   func.return original output argument
//
// and contains no allocation, copy, cast, tensor/buffer bridge, or additional
// operation. Alias/alignment requirements remain retained preconditions; this
// interface does not prove them for concrete runtime buffers.
BufferizedGemmHandoffResultV1
deriveBufferizedGemmHandoffV1(mlir::ModuleOp structured_module);

// Verifies the self-contained bufferized envelope, reusable retained-source
// fingerprint certificate, authoritative retained GEMM contract,
// identity-layout memref types, exact destination dataflow, zero-before-read
// overwrite rule, and inspection-only authority. Standalone verification is
// internal self-consistency; it is not proof of a particular external source.
bool verifyBufferizedGemmHandoffV1(mlir::ModuleOp bufferized_module,
                                  std::string &error);

// Additionally compares that certificate with a particular verified
// structured-GEMM-v1 module, including its ordered complete site set. Both
// inputs are read-only.
bool verifyBufferizedGemmHandoffMatchesStructuredV1(
    mlir::ModuleOp structured_module, mlir::ModuleOp bufferized_module,
    std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_BUFFERIZED_GEMM_HANDOFF_H
