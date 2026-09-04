#ifndef MATCORE_MDSLC_MLIR_STRUCTURED_HANDOFF_CERTIFICATE_H
#define MATCORE_MDSLC_MLIR_STRUCTURED_HANDOFF_CERTIFICATE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace matcore::mdslc::mlir_bridge {

// Operation-neutral profile for the proof-carrying part of an internal
// semantic-to-structured projection. Body legality and semantic-contract
// interpretation remain operation-specific.
struct StructuredHandoffCertificateProfileV1 {
  llvm::StringRef schema;
  llvm::StringRef producer;
  llvm::StringRef authority;
  llvm::StringRef source_operation;
  llvm::StringRef destination_rule;
  llvm::StringRef structured_function_prefix;
  llvm::StringRef semantic_function_prefix;
  std::uint32_t version = 1;
};

struct VerifiedStructuredHandoffSiteV1 {
  mlir::func::FuncOp function;
  mlir::StringAttr site_id;
  mlir::StringAttr source_semantic_symbol;
  mlir::DictionaryAttr semantic_contract;
};

using SemanticContractSelectorV1 = llvm::function_ref<mlir::DictionaryAttr(
    mlir::func::FuncOp source_function, std::string &error)>;

// Exact source envelope accepted by the current authenticated semantic bridge.
// This is reusable by operation-specific structured projections but does not
// itself grant a semantic operation admission decision.
bool verifyStructuredHandoffSourceEnvelopeV1(mlir::ModuleOp source,
                                              std::string &error);

mlir::DictionaryAttr selectExactSemanticContractV1(
    mlir::Builder &builder, mlir::Operation *operation,
    llvm::ArrayRef<llvm::StringLiteral> names, llvm::StringRef context,
    std::string &error);

// Copies the source identity into a new analysis-only module and installs the
// profile-specific certificate envelope. The caller must supply an empty
// target module and later verify its operation-specific bodies.
bool initializeStructuredHandoffCertificateV1(
    mlir::ModuleOp source, mlir::ModuleOp target, mlir::Builder &builder,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error);

// Attaches the common source/site/opaque-contract binding to a structured
// function. The contract is intentionally opaque here: an operation-specific
// verifier must interpret and re-verify it.
bool attachStructuredHandoffSiteCertificateV1(
    mlir::func::FuncOp structured_function,
    mlir::func::FuncOp source_function, mlir::DictionaryAttr semantic_contract,
    llvm::StringRef site_id, mlir::Builder &builder,
    const StructuredHandoffCertificateProfileV1 &profile, std::string &error);

bool verifyStructuredHandoffCertificateEnvelopeV1(
    mlir::ModuleOp structured_module,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error);

bool verifyStructuredHandoffSiteCertificateV1(
    mlir::func::FuncOp function, std::size_t expected_ordinal,
    const StructuredHandoffCertificateProfileV1 &profile,
    VerifiedStructuredHandoffSiteV1 &verified, std::string &error);

// Compares all common module and site bindings with one authoritative semantic
// module. The selector must return the exact operation-specific source
// contract after the caller has authenticated that operation.
bool verifyStructuredHandoffCertificateMatchesSourceV1(
    mlir::ModuleOp semantic_module, mlir::ModuleOp structured_module,
    const StructuredHandoffCertificateProfileV1 &profile,
    SemanticContractSelectorV1 contract_selector, std::string &error);

// Low-level, internal diagnostic primitives for deterministic semantic
// identity over source-module identity, capture order, source operation, site
// identity, symbol, type, function location, ordered entry-block argument
// locations, and exact semantic contract. Each helper verifies generic MLIR and
// rejects a function that is not a direct member of the supplied module. They
// do not authenticate operation-specific semantics; authority-bearing callers
// must first use the operation-specific paired verifier above. These detect
// accidental substitution and are neither a source-byte signature nor an
// execution capability.
std::string computeSourceSemanticFingerprintV1(
    mlir::ModuleOp semantic_module, mlir::func::FuncOp source_function,
    mlir::DictionaryAttr semantic_contract, llvm::StringRef source_operation,
    std::string &error);
std::string computeStructuredSemanticFingerprintV1(
    mlir::ModuleOp structured_module,
    mlir::func::FuncOp structured_function, llvm::StringRef source_operation,
    std::string &error);

} // namespace matcore::mdslc::mlir_bridge

#endif // MATCORE_MDSLC_MLIR_STRUCTURED_HANDOFF_CERTIFICATE_H
