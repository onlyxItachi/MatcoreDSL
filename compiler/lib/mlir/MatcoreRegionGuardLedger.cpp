#include "MatcoreRegionGuardLedger.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {
using P = RegionGuardPredicateV1;
using E = RegionGuardEvidenceV1;
using F = RegionGuardFrontierV1;
using V = ir::v1::ValueId;

constexpr std::pair<P, llvm::StringLiteral> kPredicates[] = {
    {P::SourceTensorContract, "source_tensor_contract"},
    {P::SourceLayoutContract, "source_layout_contract"},
    {P::SourceHostDesignation, "source_host_designation"},
    {P::SourceAccessDesignation, "source_access_designation"},
    {P::SourceNumericalContract, "source_numerical_contract"},
    {P::SourcePolicyIntent, "source_policy_intent"},
    {P::SourceSynchronousIntent, "source_synchronous_intent"},
    {P::DataNonnull, "data_nonnull"},
    {P::PositiveDimensions, "positive_dimensions"},
    {P::ContractionDimensionEqual, "contraction_dimension_equal"},
    {P::OutputRowsEqual, "output_rows_equal"},
    {P::OutputColumnsEqual, "output_columns_equal"},
    {P::PointerAlignmentRequired, "pointer_alignment_required"},
    {P::ByteRangeRepresentable, "byte_range_representable"},
    {P::OutputInputNoOverlap, "output_input_no_overlap"},
    {P::FloatingEnvironmentCompatible, "floating_environment_compatible"},
    {P::DescriptorObjectValid, "descriptor_object_valid"},
    {P::BackingHostAccessible, "backing_host_accessible"},
    {P::BackingCapacitySufficient, "backing_capacity_sufficient"},
    {P::BackingLifetimeValid, "backing_lifetime_valid"},
    {P::BackingAccessPermitted, "backing_access_permitted"},
    {P::NoConflictingConcurrentAccess, "no_conflicting_concurrent_access"},
    {P::SelectedImplementationEligible, "selected_implementation_eligible"},
    {P::SelectedImplementationResourcesAvailable, "selected_implementation_resources_available"},
    {P::SelectedImplementationNumericalConformant, "selected_implementation_numerical_conformant"},
    {P::ProviderStateAndSynchronousCompletion, "provider_state_and_synchronous_completion"},
    {P::MayWriteBeforeFailureNoRollback, "may_write_before_failure_no_rollback"},
};
constexpr std::pair<E, llvm::StringLiteral> kEvidence[] = {
    {E::RepresentationOnly, "representation_only"},
    {E::RuntimeValidationRequired, "runtime_validation_required"},
    {E::CallerPreconditionUnproven, "caller_precondition_unproven"},
    {E::DispatchExecutionObligationRetained, "dispatch_execution_obligation_retained"},
};
constexpr std::pair<F, llvm::StringLiteral> kFrontiers[] = {
    {F::SourceContract, "source_contract"},
    {F::CallValidationBeforeCompute, "call_validation_before_compute"},
    {F::CallerThroughCompletion, "caller_through_completion"},
    {F::DispatchBeforeExecution, "dispatch_before_execution"},
    {F::ExecutionAndReturn, "execution_and_return"},
};
constexpr std::pair<V, llvm::StringLiteral> kSubjects[] = {
    {V::Output, "output"}, {V::Lhs, "lhs"}, {V::Rhs, "rhs"}};

template <typename T, std::size_t N>
llvm::StringRef name(T value, const std::pair<T, llvm::StringLiteral> (&table)[N]) {
  for (const auto &[key, spelling] : table)
    if (key == value)
      return spelling;
  llvm_unreachable("invalid internal guard ledger enum");
}
template <typename T, std::size_t N>
bool parse(mlir::Attribute value, T &result,
           const std::pair<T, llvm::StringLiteral> (&table)[N]) {
  auto text = mlir::dyn_cast_or_null<mlir::StringAttr>(value);
  if (!text)
    return false;
  for (const auto &[key, spelling] : table)
    if (text.getValue() == spelling) {
      result = key;
      return true;
    }
  return false;
}
bool fail(std::string &error, llvm::StringRef message) {
  error = message.str();
  return false;
}
bool fields(mlir::DictionaryAttr value,
            llvm::ArrayRef<llvm::StringRef> expected) {
  return value && value.size() == expected.size() &&
         llvm::all_of(expected, [&](llvm::StringRef key) { return value.get(key); });
}
bool integer(mlir::Attribute value, unsigned bits, std::int64_t &result) {
  auto number = mlir::dyn_cast_or_null<mlir::IntegerAttr>(value);
  if (!number || !number.getType().isSignlessInteger(bits))
    return false;
  result = number.getInt();
  return true;
}

mlir::DictionaryAttr encode(mlir::Builder &b, const RegionGuardLedgerV1 &ledger) {
  llvm::SmallVector<mlir::Attribute> entries;
  for (const auto &entry : ledger.entries) {
    llvm::SmallVector<mlir::Attribute> subjects;
    for (auto subject : entry.subjects)
      subjects.push_back(b.getStringAttr(name(subject, kSubjects)));
    llvm::SmallVector<mlir::NamedAttribute> fields{
        b.getNamedAttr("predicate", b.getStringAttr(name(entry.predicate, kPredicates))),
        b.getNamedAttr("evidence", b.getStringAttr(name(entry.evidence, kEvidence))),
        b.getNamedAttr("frontier", b.getStringAttr(name(entry.frontier, kFrontiers))),
        b.getNamedAttr("subjects", b.getArrayAttr(subjects))};
    if (entry.predicate == P::PointerAlignmentRequired)
      fields.push_back(b.getNamedAttr("alignment_bytes", b.getI64IntegerAttr(
          static_cast<std::int64_t>(entry.alignment_bytes))));
    entries.push_back(b.getDictionaryAttr(fields));
  }
  return b.getDictionaryAttr({
      b.getNamedAttr("version", b.getI32IntegerAttr(1)),
      b.getNamedAttr("site_id", b.getStringAttr(ledger.site_id)),
      b.getNamedAttr("stage", b.getI64IntegerAttr(ledger.stage)),
      b.getNamedAttr("bindings", b.getDictionaryAttr({
          b.getNamedAttr("output", b.getStringAttr(ledger.bindings[0])),
          b.getNamedAttr("lhs", b.getStringAttr(ledger.bindings[1])),
          b.getNamedAttr("rhs", b.getStringAttr(ledger.bindings[2]))})),
      b.getNamedAttr("entries", b.getArrayAttr(entries))});
}
} // namespace

bool decodeRegionGuardLedgerV1(mlir::DictionaryAttr encoded,
                               RegionGuardLedgerV1 &ledger, std::string &error) {
  error.clear();
  ledger = {};
  std::int64_t version = 0, stage = 0;
  if (!fields(encoded, {"version", "site_id", "stage", "bindings", "entries"}) ||
      !integer(encoded.get("version"), 32, version) || version != 1 ||
      !integer(encoded.get("stage"), 64, stage) || stage < 0 || stage > 1)
    return fail(error, "guard ledger has unknown fields, version or call stage");
  auto site = encoded.getAs<mlir::StringAttr>("site_id");
  auto bindings = encoded.getAs<mlir::DictionaryAttr>("bindings");
  auto entries = encoded.getAs<mlir::ArrayAttr>("entries");
  if (!site || site.getValue().empty() || !entries || entries.empty() ||
      !fields(bindings, {"output", "lhs", "rhs"}))
    return fail(error, "guard ledger lacks source site, descriptor bindings or entries");
  ledger.site_id = site.getValue().str();
  ledger.stage = static_cast<unsigned>(stage);
  for (unsigned role = 0; role != 3; ++role) {
    auto descriptor = bindings.getAs<mlir::StringAttr>(kSubjects[role].second);
    if (!descriptor || descriptor.getValue().empty())
      return fail(error, "guard ledger descriptor binding must be nonempty");
    ledger.bindings[role] = descriptor.getValue().str();
  }
  for (auto attribute : entries) {
    auto row = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    RegionGuardEntryV1 entry{};
    if (!row || !parse(row.get("predicate"), entry.predicate, kPredicates) ||
        !parse(row.get("evidence"), entry.evidence, kEvidence) ||
        !parse(row.get("frontier"), entry.frontier, kFrontiers))
      return fail(error, "guard ledger has an unknown predicate, evidence class or frontier");
    bool alignment = entry.predicate == P::PointerAlignmentRequired;
    if (!(alignment ? fields(row, {"predicate", "evidence", "frontier", "subjects", "alignment_bytes"})
                    : fields(row, {"predicate", "evidence", "frontier", "subjects"})))
      return fail(error, "guard ledger predicate has unknown or missing fields");
    auto subjects = row.getAs<mlir::ArrayAttr>("subjects");
    if (!subjects || subjects.size() > 3)
      return fail(error, "guard ledger subjects must be bounded operand roles");
    for (auto attribute : subjects) {
      V role{};
      if (!parse(attribute, role, kSubjects) || llvm::is_contained(entry.subjects, role))
        return fail(error, "guard ledger subjects contain unknown or duplicate roles");
      entry.subjects.push_back(role);
    }
    if (alignment) {
      std::int64_t bytes = 0;
      if (!integer(row.get("alignment_bytes"), 64, bytes) || bytes <= 0 ||
          (bytes & (bytes - 1)) != 0)
        return fail(error, "guard ledger alignment must be a positive power-of-two requirement");
      entry.alignment_bytes = static_cast<std::uint64_t>(bytes);
    }
    ledger.entries.push_back(std::move(entry));
  }
  return true;
}

mlir::DictionaryAttr buildRegionGuardLedgerV1(
    mlir::Builder &b, mlir::DictionaryAttr contract,
    mlir::ArrayAttr bindings, unsigned stage, std::string &error) {
  error.clear();
  if (!contract || !bindings || bindings.size() != 3 || stage > 1) {
    fail(error, "guard ledger requires a per-call semantic contract and three bindings");
    return {};
  }
  auto site = contract.getAs<mlir::StringAttr>("site_id");
  if (!site || site.getValue().empty()) {
    fail(error, "guard ledger requires a semantic source site");
    return {};
  }
  RegionGuardLedgerV1 ledger;
  ledger.site_id = site.getValue().str();
  ledger.stage = stage;
  std::array<std::uint64_t, 3> alignments{};
  constexpr llvm::StringLiteral tensor_keys[] = {
      "output_semantics", "lhs_semantics", "rhs_semantics"};
  for (unsigned role = 0; role != 3; ++role) {
    auto binding = mlir::dyn_cast<mlir::DictionaryAttr>(bindings[role]);
    auto descriptor = binding ? binding.getAs<mlir::StringAttr>("descriptor") : mlir::StringAttr{};
    auto semantics = contract.getAs<mlir::DictionaryAttr>(tensor_keys[role]);
    std::int64_t snapshot = 0, alignment = 0;
    if (!descriptor || descriptor.getValue().empty() ||
        !integer(binding.get("snapshot_stage"), 64, snapshot) || snapshot != stage ||
        !semantics || !integer(semantics.get("alignment_bytes"), 64, alignment) ||
        alignment <= 0 || (alignment & (alignment - 1)) != 0) {
      fail(error, "guard ledger lost a descriptor snapshot or source-required alignment");
      return {};
    }
    ledger.bindings[role] = descriptor.getValue().str();
    alignments[role] = static_cast<std::uint64_t>(alignment);
  }
  const auto append = [&](P predicate, E evidence, F frontier,
                          std::initializer_list<V> subjects,
                          std::uint64_t alignment = 0) {
    ledger.entries.push_back({predicate, evidence, frontier, subjects, alignment});
  };
  // These are facts about the authenticated representation/designation only.
  // Actual addresses, sizes, accessibility and runtime state remain unproved.
  for (auto predicate : {P::SourceTensorContract, P::SourceLayoutContract,
                         P::SourceHostDesignation, P::SourceAccessDesignation})
    append(predicate, E::RepresentationOnly, F::SourceContract,
           {V::Output, V::Lhs, V::Rhs});
  for (auto predicate : {P::SourceNumericalContract, P::SourcePolicyIntent,
                         P::SourceSynchronousIntent})
    append(predicate, E::RepresentationOnly, F::SourceContract, {});
  for (unsigned role = 0; role != 3; ++role) {
    V subject = kSubjects[role].first;
    for (auto predicate : {P::DataNonnull, P::PositiveDimensions,
                           P::ByteRangeRepresentable})
      append(predicate, E::RuntimeValidationRequired, F::CallValidationBeforeCompute, {subject});
    append(P::PointerAlignmentRequired, E::RuntimeValidationRequired,
           F::CallValidationBeforeCompute, {subject}, alignments[role]);
  }
  // Exact equations: lhs.columns == rhs.rows; out.rows == lhs.rows;
  // out.columns == rhs.columns. They are not static shape facts.
  append(P::ContractionDimensionEqual, E::RuntimeValidationRequired,
         F::CallValidationBeforeCompute, {V::Lhs, V::Rhs});
  append(P::OutputRowsEqual, E::RuntimeValidationRequired,
         F::CallValidationBeforeCompute, {V::Output, V::Lhs});
  append(P::OutputColumnsEqual, E::RuntimeValidationRequired,
         F::CallValidationBeforeCompute, {V::Output, V::Rhs});
  append(P::OutputInputNoOverlap, E::RuntimeValidationRequired,
         F::CallValidationBeforeCompute, {V::Output, V::Lhs});
  append(P::OutputInputNoOverlap, E::RuntimeValidationRequired,
         F::CallValidationBeforeCompute, {V::Output, V::Rhs});
  append(P::FloatingEnvironmentCompatible, E::RuntimeValidationRequired,
         F::CallValidationBeforeCompute, {});
  for (auto predicate : {P::DescriptorObjectValid, P::BackingHostAccessible,
                         P::BackingCapacitySufficient, P::BackingLifetimeValid,
                         P::BackingAccessPermitted, P::NoConflictingConcurrentAccess})
    append(predicate, E::CallerPreconditionUnproven, F::CallerThroughCompletion,
           {V::Output, V::Lhs, V::Rhs});
  for (auto predicate : {P::SelectedImplementationEligible,
                         P::SelectedImplementationResourcesAvailable,
                         P::SelectedImplementationNumericalConformant})
    append(predicate, E::DispatchExecutionObligationRetained,
           F::DispatchBeforeExecution, {});
  for (auto predicate : {P::ProviderStateAndSynchronousCompletion,
                         P::MayWriteBeforeFailureNoRollback})
    append(predicate, E::DispatchExecutionObligationRetained,
           F::ExecutionAndReturn, {});
  return encode(b, ledger);
}

bool verifyRegionGuardLedgerV1(mlir::DictionaryAttr encoded,
                               mlir::DictionaryAttr contract,
                               mlir::ArrayAttr bindings, unsigned stage,
                               std::string &error) {
  RegionGuardLedgerV1 decoded;
  if (!decodeRegionGuardLedgerV1(encoded, decoded, error))
    return false;
  mlir::Builder b(encoded.getContext());
  auto expected = buildRegionGuardLedgerV1(b, contract, bindings, stage, error);
  if (!expected)
    return false;
  if (encoded != expected)
    return fail(error, "guard ledger differs from required source-scoped predicates, evidence or frontiers");
  return true;
}
} // namespace matcore::mdslc::mlir_bridge
