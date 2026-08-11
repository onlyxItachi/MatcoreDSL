#ifndef MATCORE_MDSLC_MLIR_MATCORE_DIALECT_H
#define MATCORE_MDSLC_MLIR_MATCORE_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

#include "MatcoreDialect.h.inc"

namespace matcore::mdslc::mlir_dialect {

inline constexpr llvm::StringLiteral kCompositionSchemaV1 =
    "matcore-mlir-composition-v1";

// Caller-authenticated source bytes used to authorize source-backed semantic
// provenance. All string views must remain valid for the duration of the
// verification call. The verifier recomputes source_digest from source_bytes
// and requires source_byte_length to equal the exact byte count; no filesystem
// access occurs in the dialect verifier.
struct AuthenticatedSourceSnapshotV1 {
  llvm::StringRef source_identity;
  llvm::StringRef source_digest;
  llvm::StringRef source_bytes;
  std::uint64_t source_byte_length = 0;
};

// Verifies the versioned multi-operation semantic envelope. This is separate
// from, and does not loosen, the exact Matcore IR v1 capture bridge envelope.
// The context-free entry point fails closed if any operation claims
// source_authenticated provenance.
bool verifyCompositionV1Module(mlir::ModuleOp module, std::string &error);

// Verifies the same envelope while authenticating every source-backed range
// and line/column against caller-trusted source bytes.
bool verifyCompositionV1Module(
    mlir::ModuleOp module,
    const AuthenticatedSourceSnapshotV1 &authenticated_source,
    std::string &error);

} // namespace matcore::mdslc::mlir_dialect

#endif // MATCORE_MDSLC_MLIR_MATCORE_DIALECT_H
