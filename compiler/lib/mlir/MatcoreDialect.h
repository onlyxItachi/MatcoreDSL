#ifndef MATCORE_MDSLC_MLIR_MATCORE_DIALECT_H
#define MATCORE_MDSLC_MLIR_MATCORE_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"

#include <string>

#include "MatcoreDialect.h.inc"

namespace matcore::mdslc::mlir_dialect {

inline constexpr llvm::StringLiteral kCompositionSchemaV1 =
    "matcore-mlir-composition-v1";

// Verifies the versioned multi-operation semantic envelope. This is separate
// from, and does not loosen, the exact Matcore IR v1 capture bridge envelope.
bool verifyCompositionV1Module(mlir::ModuleOp module, std::string &error);

} // namespace matcore::mdslc::mlir_dialect

#endif // MATCORE_MDSLC_MLIR_MATCORE_DIALECT_H
