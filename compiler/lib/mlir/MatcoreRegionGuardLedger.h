#ifndef MATCORE_MDSLC_MLIR_REGION_GUARD_LEDGER_H
#define MATCORE_MDSLC_MLIR_REGION_GUARD_LEDGER_H

#include "matcore_ir_v1.h"
#include "mlir/IR/Builders.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace matcore::mdslc::mlir_bridge {

// Closed internal vocabulary for this inspection milestone. In particular,
// there is no executed/discharged state or constructor of runtime authority.
enum class RegionGuardEvidenceV1 {
  RepresentationOnly,
  RuntimeValidationRequired,
  CallerPreconditionUnproven,
  DispatchExecutionObligationRetained,
};
enum class RegionGuardFrontierV1 {
  // Ownership/obligation scopes, NOT an execution schedule. For example, the
  // current runtime checks fenv after provider/resource discovery. Neither this
  // enum order nor ledger row order changes original first-failure precedence.
  SourceContract,
  CallValidationBeforeCompute,
  CallerThroughCompletion,
  DispatchBeforeExecution,
  ExecutionAndReturn,
};
enum class RegionGuardPredicateV1 {
  SourceTensorContract,
  SourceLayoutContract,
  SourceHostDesignation,
  SourceAccessDesignation,
  SourceNumericalContract,
  SourcePolicyIntent,
  SourceSynchronousIntent,
  DataNonnull,
  // Both signed source dimensions must be positive. No captured static shape
  // or executed descriptor validation is implied by these predicates.
  PositiveDimensions,
  // lhs.columns == rhs.rows; output.rows == lhs.rows;
  // output.columns == rhs.columns, respectively.
  ContractionDimensionEqual,
  OutputRowsEqual,
  OutputColumnsEqual,
  PointerAlignmentRequired,
  // Element/byte products and address-range end must be representable in the
  // selected runtime address domain. This is NOT allocated backing capacity.
  ByteRangeRepresentable,
  OutputInputNoOverlap,
  FloatingEnvironmentCompatible,
  DescriptorObjectValid,
  BackingHostAccessible,
  BackingCapacitySufficient,
  BackingLifetimeValid,
  BackingAccessPermitted,
  NoConflictingConcurrentAccess,
  SelectedImplementationEligible,
  SelectedImplementationResourcesAvailable,
  SelectedImplementationNumericalConformant,
  // Retained execution/return obligations, never pre-compute validation facts.
  ProviderStateAndSynchronousCompletion,
  MayWriteBeforeFailureNoRollback,
};

struct RegionGuardEntryV1 {
  RegionGuardPredicateV1 predicate;
  RegionGuardEvidenceV1 evidence;
  RegionGuardFrontierV1 frontier;
  std::vector<ir::v1::ValueId> subjects;
  // Present only for PointerAlignmentRequired; this is a required minimum,
  // never a detected property of a concrete pointer.
  std::uint64_t alignment_bytes = 0;
};
struct RegionGuardLedgerV1 {
  std::string site_id;
  unsigned stage = 0;
  // Canonical order: output, lhs, rhs. Different IDs may alias physical bytes.
  std::array<std::string, 3> bindings;
  std::vector<RegionGuardEntryV1> entries;
};

// Decoding checks the closed vocabulary and exact field shapes, not source
// authenticity. The enclosing region must still pair with native evidence.
bool decodeRegionGuardLedgerV1(mlir::DictionaryAttr encoded,
                               RegionGuardLedgerV1 &ledger,
                               std::string &error);

// Builds obligations for a declared per-call source contract. No runtime or
// provider query occurs. Source pairing remains the authority for that contract.
mlir::DictionaryAttr buildRegionGuardLedgerV1(
    mlir::Builder &builder, mlir::DictionaryAttr source_contract,
    mlir::ArrayAttr source_bindings, unsigned stage, std::string &error);

// Requires the exact regenerated bounded catalog. Its row order is merely
// canonical serialization, never permission to reorder within-call checks.
bool verifyRegionGuardLedgerV1(mlir::DictionaryAttr encoded,
                               mlir::DictionaryAttr source_contract,
                               mlir::ArrayAttr source_bindings, unsigned stage,
                               std::string &error);

} // namespace matcore::mdslc::mlir_bridge
#endif
