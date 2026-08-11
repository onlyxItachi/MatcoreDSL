#include "MatcoreRecoveredGemmBridge.h"

#include "MatcoreGemmSemanticBuilder.h"
#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <string_view>
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {

constexpr llvm::StringLiteral kBridgeSchema = "matcore-mlir-semantic-v1";
constexpr llvm::StringLiteral kRecoveredPattern =
    "canonical-row-major-f32-gemm-v1";
constexpr llvm::StringLiteral kRecoveredSemanticContract =
    "f32_row_major_overwrite_m_k__k_n__m_n";

constexpr llvm::StringLiteral kProofRoles[] = {
    "outer_loop",       "outer_init",       "outer_condition",
    "outer_increment",  "outer_bound_m",    "middle_init",
    "middle_condition", "middle_increment", "middle_bound_n",
    "inner_init",       "inner_condition",  "inner_increment",
    "inner_bound_k",    "accumulator_update", "output_store",
    "lhs_base",         "rhs_base",         "output_base",
};

constexpr llvm::StringLiteral kRequiredGuards[] = {
    "positive_m_n_k",
    "overflow_safe_shapes_elements_and_bytes",
    "nonnull_observable_pointers",
    "natural_f32_alignment",
    "output_disjoint_from_lhs_and_rhs",
    "compatible_runtime_floating_point_environment",
    "legal_cpu_implementation_available",
};

bool fitsSigned64(std::uint64_t value) {
  return value <= static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max());
}

std::string sha256Identity(llvm::StringRef bytes) {
  const std::array<std::uint8_t, 32> digest =
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes));
  return "sha256:" + llvm::toHex(llvm::ArrayRef(digest), true);
}

bool sourceLineColumnAtOffset(llvm::StringRef bytes, std::uint64_t offset,
                              std::uint64_t &line, std::uint64_t &column) {
  if (offset >= bytes.size())
    return false;
  line = 1;
  column = 1;
  for (std::uint64_t index = 0; index < offset; ++index) {
    const char byte = bytes[static_cast<std::size_t>(index)];
    if (byte == '\r') {
      if (index + 1 < offset &&
          bytes[static_cast<std::size_t>(index + 1)] == '\n')
        ++index;
      ++line;
      column = 1;
      continue;
    }
    if (byte == '\n') {
      ++line;
      column = 1;
      continue;
    }
    ++column;
  }
  return true;
}

std::string canonicalInputPath(const std::string &input) {
  std::error_code error;
  std::filesystem::path path = std::filesystem::absolute(input, error);
  if (error)
    path = input;
  error.clear();
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(path, error);
  return (error ? path.lexically_normal() : canonical).generic_string();
}

bool exactRelaxedFpProof(const frontend::RecoveredFpProof &proof) {
  return proof.allow_reassociation && proof.contract_across_statement &&
         proof.honor_nans && proof.honor_infinities &&
         !proof.preserve_signed_zero && !proof.allow_reciprocal &&
         !proof.allow_approximate_functions && !proof.fenv_access &&
         !proof.fast_math_profile && proof.evaluation_method == "source" &&
         proof.rounding_mode == "tonearest" &&
         proof.exception_mode == "ignore" &&
         proof.denormal_mode == "ieee,ieee" &&
         proof.fp32_denormal_mode == "ieee,ieee" &&
         proof.optimization_level > 0;
}

bool authenticateRecoveredCandidate(
    const frontend::Result &source, const frontend::Options &options,
    std::size_t candidate_index, const frontend::RecoveredGemmCandidate *&out,
    std::string &error) {
  out = nullptr;
  if (!options.inspect_recovered_cpp_gemm) {
    error = "recovered MLIR analysis requires explicit native recovery inspection";
    return false;
  }
  if (source.module.producer != "clang-libtooling-v1") {
    error = "recovered MLIR analysis requires an authenticated native LibTooling result";
    return false;
  }
  if (candidate_index >= source.recovered_gemm_candidates.size()) {
    error = "recovered GEMM candidate index is out of range";
    return false;
  }
  const frontend::RecoveredGemmCandidate &candidate =
      source.recovered_gemm_candidates[candidate_index];
  if (candidate.state !=
          frontend::RecoveredGemmState::recognized_guard_required ||
      !candidate.rejection_reasons.empty()) {
    error = "only a zero-rejection recognized_guard_required candidate may enter recovered MLIR analysis";
    return false;
  }
  if (candidate.pattern != kRecoveredPattern ||
      candidate.semantic_contract != kRecoveredSemanticContract) {
    error = "recovered GEMM pattern or semantic contract is not the closed v1 contract";
    return false;
  }
  if (candidate.source_file.empty() ||
      !std::string_view(candidate.source_file).ends_with(".mdsl") ||
      candidate.source_file != source.module.source_file ||
      source.module.translation_unit != source.module.source_file) {
    error = "recovered GEMM source file does not match the native frontend result";
    return false;
  }
  const std::string expected_source_identity =
      frontend::stableSourceIdentity(canonicalInputPath(options.input_path));
  if (candidate.source_identity != expected_source_identity) {
    error = "recovered GEMM source identity does not match the extraction input";
    return false;
  }
  const std::string compilation_identity =
      frontend::stableCompilationIdentity(options);
  if (compilation_identity.empty() ||
      candidate.compilation_identity != compilation_identity) {
    error = "recovered GEMM compilation identity does not match the extraction options";
    return false;
  }
  if (source.source_snapshot.empty()) {
    error = "recovered GEMM analysis requires the exact parsed source snapshot";
    return false;
  }
  const std::string source_digest = sha256Identity(source.source_snapshot);
  if (candidate.source_snapshot_sha256 != source_digest) {
    error = "recovered GEMM source digest does not authenticate the parsed snapshot";
    return false;
  }
  if (!fitsSigned64(candidate.offset) ||
      !fitsSigned64(candidate.outer_loop_range.begin) ||
      !fitsSigned64(candidate.outer_loop_range.end) ||
      candidate.outer_loop_range.begin != candidate.offset ||
      candidate.outer_loop_range.end <= candidate.outer_loop_range.begin ||
      candidate.outer_loop_range.end > source.source_snapshot.size()) {
    error = "recovered GEMM outer-loop range is not an exact bounded source range";
    return false;
  }
  std::uint64_t expected_line = 0;
  std::uint64_t expected_column = 0;
  if (!sourceLineColumnAtOffset(source.source_snapshot, candidate.offset,
                                expected_line, expected_column) ||
      candidate.line != expected_line || candidate.column != expected_column) {
    error = "recovered GEMM line/column does not match its source byte offset";
    return false;
  }
  const std::string expected_site = frontend::makeStableSiteId(
      candidate.source_identity, compilation_identity, source.source_snapshot,
      candidate.offset, "recovered.cpp.gemm.v1");
  if (candidate.site_id != expected_site) {
    error = "recovered GEMM site ID does not authenticate source, compilation, and offset";
    return false;
  }
  if (candidate.proof_ranges.size() != std::size(kProofRoles)) {
    error = "recovered GEMM proof range set is incomplete";
    return false;
  }
  for (std::size_t index = 0; index < std::size(kProofRoles); ++index) {
    const frontend::RecoveredNamedRange &proof = candidate.proof_ranges[index];
    if (proof.role != kProofRoles[index] || !fitsSigned64(proof.range.begin) ||
        !fitsSigned64(proof.range.end) ||
        proof.range.begin < candidate.outer_loop_range.begin ||
        proof.range.end <= proof.range.begin ||
        proof.range.end > candidate.outer_loop_range.end) {
      error = "recovered GEMM proof ranges are not the exact ordered v1 proof set";
      return false;
    }
  }
  if (candidate.proof_ranges.front().range.begin !=
          candidate.outer_loop_range.begin ||
      candidate.proof_ranges.front().range.end !=
          candidate.outer_loop_range.end) {
    error = "recovered GEMM outer-loop proof does not equal its source range";
    return false;
  }
  if (!std::equal(candidate.required_runtime_guards.begin(),
                  candidate.required_runtime_guards.end(),
                  std::begin(kRequiredGuards), std::end(kRequiredGuards)) ||
      candidate.required_runtime_guards.size() != std::size(kRequiredGuards)) {
    error = "recovered GEMM runtime guards are not the exact ordered v1 set";
    return false;
  }
  if (!exactRelaxedFpProof(candidate.fp_proof)) {
    error = "recovered GEMM effective floating-point proof is not the exact relaxed v1 contract";
    return false;
  }
  if (candidate.function_name.empty() || candidate.output_parameter.empty() ||
      candidate.lhs_parameter.empty() || candidate.rhs_parameter.empty() ||
      candidate.m_parameter.empty() || candidate.n_parameter.empty() ||
      candidate.k_parameter.empty()) {
    error = "recovered GEMM semantic bindings are incomplete";
    return false;
  }
  out = &candidate;
  return true;
}

ir::v1::TensorValue recoveredTensor(
    ir::v1::ValueId id, std::string expression, ir::v1::Mutability mutability,
    const std::string &major, const std::string &minor) {
  ir::v1::TensorValue value;
  value.id = id;
  value.source_expression = std::move(expression);
  value.mutability = mutability;
  value.type.element_dtype = ir::v1::DType::F32;
  value.type.rank = 2;
  value.type.shape = {ir::v1::ScalarExpr::dynamic(major),
                      ir::v1::ScalarExpr::dynamic(minor)};
  value.type.strides = {ir::v1::ScalarExpr::dynamic(minor),
                        ir::v1::ScalarExpr::staticValue(1)};
  value.type.layout = ir::v1::Layout::RowMajorContiguous;
  value.type.memory_space = ir::v1::MemorySpace::Host;
  value.type.required_alignment_bytes = 4;
  return value;
}

mlir::DictionaryAttr recoveredNumerical(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("accumulation_dtype", builder.getStringAttr("f32")),
       builder.getNamedAttr("approximate_math", builder.getBoolAttr(false)),
       builder.getNamedAttr("contraction", builder.getStringAttr("allowed")),
       builder.getNamedAttr("derivation",
                            builder.getStringAttr("effective_cpp_semantics")),
       builder.getNamedAttr(
           "exception_status",
           builder.getStringAttr("incoming_not_preserved_postcall_unspecified")),
       builder.getNamedAttr(
           "infinity", builder.getStringAttr("ieee_no_no_infs_assumption")),
       builder.getNamedAttr("inplace", builder.getBoolAttr(false)),
       builder.getNamedAttr(
           "nan", builder.getStringAttr(
                      "preserve_classification_payload_order_unspecified")),
       builder.getNamedAttr(
           "profile", builder.getStringAttr(kRecoveredGemmNumericalProfileV1)),
       builder.getNamedAttr("reassociation",
                            builder.getStringAttr("within_k_reduction")),
       builder.getNamedAttr(
           "reduction_order",
           builder.getStringAttr("implementation_defined_within_k")),
       builder.getNamedAttr("rounding",
                            builder.getStringAttr("nearest_ties_even")),
       builder.getNamedAttr("signed_zero", builder.getStringAttr("relaxed")),
       builder.getNamedAttr(
           "subnormals",
           builder.getStringAttr("ieee_gradual_ftz_daz_forbidden")),
       builder.getNamedAttr("trapping_exceptions",
                            builder.getStringAttr("unsupported"))});
}

mlir::DictionaryAttr sourceRangeAttribute(
    mlir::Builder &builder, const frontend::RecoveredSourceRange &range) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "begin", builder.getI64IntegerAttr(
                        static_cast<std::int64_t>(range.begin))),
       builder.getNamedAttr(
           "end", builder.getI64IntegerAttr(
                      static_cast<std::int64_t>(range.end)))});
}

mlir::DictionaryAttr recoveredProvenance(
    mlir::Builder &builder,
    const frontend::RecoveredGemmCandidate &candidate) {
  llvm::SmallVector<mlir::Attribute> proof_ranges;
  proof_ranges.reserve(candidate.proof_ranges.size());
  for (const frontend::RecoveredNamedRange &proof : candidate.proof_ranges) {
    proof_ranges.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr(
             "begin", builder.getI64IntegerAttr(
                          static_cast<std::int64_t>(proof.range.begin))),
         builder.getNamedAttr(
             "end", builder.getI64IntegerAttr(
                        static_cast<std::int64_t>(proof.range.end))),
         builder.getNamedAttr("kind", builder.getStringAttr(proof.role))}));
  }
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("column",
                            builder.getI64IntegerAttr(candidate.column)),
       builder.getNamedAttr(
           "compilation_identity",
           builder.getStringAttr(candidate.compilation_identity)),
       builder.getNamedAttr("file", builder.getStringAttr(candidate.source_file)),
       builder.getNamedAttr("kind",
                            builder.getStringAttr("recovered_cpp_loop")),
       builder.getNamedAttr("line", builder.getI64IntegerAttr(candidate.line)),
       builder.getNamedAttr(
           "offset", builder.getI64IntegerAttr(
                         static_cast<std::int64_t>(candidate.offset))),
       builder.getNamedAttr("proof_ranges", builder.getArrayAttr(proof_ranges)),
       builder.getNamedAttr(
           "source_range",
           sourceRangeAttribute(builder, candidate.outer_loop_range)),
       builder.getNamedAttr(
           "source_snapshot",
           builder.getStringAttr(candidate.source_snapshot_sha256)),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

bool exactModuleAttributes(mlir::ModuleOp module,
                           llvm::ArrayRef<llvm::StringRef> expected) {
  if (module->getAttrs().size() != expected.size())
    return false;
  llvm::StringSet<> names;
  for (llvm::StringRef name : expected)
    names.insert(name);
  return llvm::all_of(module->getAttrs(), [&](mlir::NamedAttribute attribute) {
    return names.contains(attribute.getName().strref());
  });
}

bool moduleString(mlir::ModuleOp module, llvm::StringRef name,
                  llvm::StringRef expected) {
  const auto value = module->getAttrOfType<mlir::StringAttr>(name);
  return value && value.getValue() == expected;
}

mlir_dialect::GemmOp singleGemm(mlir::ModuleOp module, std::string &error) {
  mlir_dialect::GemmOp gemm;
  std::size_t count = 0;
  module.walk([&](mlir_dialect::GemmOp operation) {
    gemm = operation;
    ++count;
  });
  if (count != 1) {
    error = "mathematical GEMM fingerprint requires exactly one semantic GEMM";
    return {};
  }
  return gemm;
}

std::string printedAttribute(mlir::Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  stream.flush();
  return text;
}

std::string normalizedScalar(mlir::DictionaryAttr scalar,
                             std::map<std::string, std::string> &symbols) {
  const auto kind = scalar.getAs<mlir::StringAttr>("kind");
  if (kind.getValue() == "static") {
    return "s" +
           std::to_string(scalar.getAs<mlir::IntegerAttr>("value").getInt());
  }
  const std::string original =
      scalar.getAs<mlir::StringAttr>("symbol").getValue().str();
  const auto [iterator, inserted] = symbols.try_emplace(original);
  if (inserted)
    iterator->second = "d" + std::to_string(symbols.size() - 1);
  return iterator->second;
}

void appendTensorFingerprint(std::string &output, llvm::StringRef label,
                             mlir::DictionaryAttr semantics,
                             mlir::RankedTensorType type,
                             std::map<std::string, std::string> &symbols) {
  output += label.str() + "{type=";
  {
    std::string type_text;
    llvm::raw_string_ostream stream(type_text);
    type.print(stream);
    stream.flush();
    output += type_text;
  }
  output += ";shape=";
  for (mlir::Attribute encoded : semantics.getAs<mlir::ArrayAttr>("shape")) {
    output += normalizedScalar(mlir::cast<mlir::DictionaryAttr>(encoded), symbols);
    output += ',';
  }
  output += ";strides=";
  for (mlir::Attribute encoded : semantics.getAs<mlir::ArrayAttr>("strides")) {
    output += normalizedScalar(mlir::cast<mlir::DictionaryAttr>(encoded), symbols);
    output += ',';
  }
  constexpr llvm::StringRef fields[] = {
      "alignment_bytes", "alignment_contract", "layout", "memory_space",
      "mutability", "role"};
  for (llvm::StringRef field : fields) {
    output += ';';
    output += field.str();
    output += '=';
    output += printedAttribute(semantics.get(field));
  }
  output += "};";
}

bool acceptedFingerprintEnvelope(mlir::ModuleOp module, std::string &error) {
  const auto schema =
      module ? module->getAttrOfType<mlir::StringAttr>("mdsl.capture_schema")
             : mlir::StringAttr{};
  if (!schema) {
    error = "semantic GEMM fingerprint requires a versioned module envelope";
    return false;
  }
  if (schema.getValue() == "matcore-ir-v1")
    return verifyMatcoreV1BridgeModule(module, error);
  if (schema.getValue() == kRecoveredGemmCaptureSchemaV1)
    return verifyRecoveredGemmAnalysisModuleV1(module, error);
  error = "semantic GEMM fingerprint rejects unknown capture envelopes";
  return false;
}

} // namespace

RecoveredGemmBridgeResultV1 bridgeRecoveredGemmToMatcoreMlirV1(
    const frontend::Result &source, const frontend::Options &options,
    std::size_t candidate_index, mlir::MLIRContext &context) {
  RecoveredGemmBridgeResultV1 result;
  const frontend::RecoveredGemmCandidate *candidate = nullptr;
  if (!authenticateRecoveredCandidate(source, options, candidate_index,
                                      candidate, result.error))
    return result;

  registerMatcoreSemanticDialects(context);
  mlir::OpBuilder builder(&context);
  result.module = mlir::ModuleOp::create(builder.getUnknownLoc());
  (*result.module)->setAttr("mdsl.analysis_only", builder.getBoolAttr(true));
  (*result.module)->setAttr("mdsl.bridge_schema",
                            builder.getStringAttr(kBridgeSchema));
  (*result.module)->setAttr(
      "mdsl.capture_schema",
      builder.getStringAttr(kRecoveredGemmCaptureSchemaV1));
  (*result.module)->setAttr("mdsl.capture_version",
                            builder.getI32IntegerAttr(1));
  (*result.module)->setAttr(
      "mdsl.compilation_identity",
      builder.getStringAttr(candidate->compilation_identity));
  (*result.module)->setAttr("mdsl.execution_intent",
                            builder.getStringAttr("generic"));
  (*result.module)->setAttr(
      "mdsl.numerical_profile",
      builder.getStringAttr(kRecoveredGemmNumericalProfileV1));
  (*result.module)->setAttr("mdsl.producer",
                            builder.getStringAttr("clang-libtooling-v1"));
  llvm::SmallVector<mlir::Attribute> required_guards;
  required_guards.reserve(candidate->required_runtime_guards.size());
  for (const std::string &guard : candidate->required_runtime_guards)
    required_guards.push_back(builder.getStringAttr(guard));
  (*result.module)->setAttr("mdsl.required_runtime_guards",
                            builder.getArrayAttr(required_guards));
  (*result.module)->setAttr(
      "mdsl.recovered_function",
      builder.getStringAttr(candidate->function_name));
  (*result.module)->setAttr(
      "mdsl.rewrite_policy",
      builder.getStringAttr("preserve_original_cpp"));
  (*result.module)->setAttr("mdsl.semantic_version",
                            builder.getI32IntegerAttr(1));
  (*result.module)->setAttr("mdsl.source_file",
                            builder.getStringAttr(candidate->source_file));
  (*result.module)->setAttr(
      "mdsl.source_identity",
      builder.getStringAttr(candidate->source_identity));
  (*result.module)->setAttr(
      "mdsl.translation_unit",
      builder.getStringAttr(source.module.translation_unit));

  GemmSemanticSiteV1 site;
  site.site_id = candidate->site_id;
  site.lhs = recoveredTensor(ir::v1::ValueId::Lhs, candidate->lhs_parameter,
                             ir::v1::Mutability::ReadOnly,
                             candidate->m_parameter, candidate->k_parameter);
  site.rhs = recoveredTensor(ir::v1::ValueId::Rhs, candidate->rhs_parameter,
                             ir::v1::Mutability::ReadOnly,
                             candidate->k_parameter, candidate->n_parameter);
  site.output = recoveredTensor(
      ir::v1::ValueId::Output, candidate->output_parameter,
      ir::v1::Mutability::WriteOnly, candidate->m_parameter,
      candidate->n_parameter);
  site.requirements = {ir::v1::SemanticRequirement::Rank2Gemm,
                       ir::v1::SemanticRequirement::F32Arithmetic,
                       ir::v1::SemanticRequirement::HostAddressable,
                       ir::v1::SemanticRequirement::SynchronousExecution};
  site.alias_requirements = {
      {.relation = ir::v1::AliasRelation::NoAlias,
       .first = ir::v1::ValueId::Output,
       .second = ir::v1::ValueId::Lhs},
      {.relation = ir::v1::AliasRelation::NoAlias,
       .first = ir::v1::ValueId::Output,
       .second = ir::v1::ValueId::Rhs},
  };
  site.effects.reads = {ir::v1::ValueId::Lhs, ir::v1::ValueId::Rhs};
  site.effects.writes = {ir::v1::ValueId::Output};
  site.effects.synchronization = ir::v1::Synchronization::Synchronous;
  site.target = "generic";
  site.fallback = "preserve_original_cpp";
  site.origin = builder.getDictionaryAttr(
      {builder.getNamedAttr("kind",
                            builder.getStringAttr("recovered_cpp_loop")),
       builder.getNamedAttr("pattern", builder.getStringAttr(kRecoveredPattern)),
       builder.getNamedAttr(
           "permission",
           builder.getStringAttr("source_proven_guard_required")),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
  site.numerical = recoveredNumerical(builder);
  site.provenance = recoveredProvenance(builder, *candidate);
  site.source_location = mlir::FileLineColLoc::get(
      &context, candidate->source_file, candidate->line, candidate->column);
  if (!appendGemmSemanticSiteV1(*result.module, builder, site, result.error) ||
      !verifyRecoveredGemmAnalysisModuleV1(*result.module, result.error)) {
    result.module = nullptr;
  }
  return result;
}

bool verifyRecoveredGemmAnalysisModuleV1(mlir::ModuleOp module,
                                         std::string &error) {
  error.clear();
  if (!module) {
    error = "recovered GEMM analysis module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(module))) {
    error = "recovered GEMM analysis module failed MLIR/dialect verification";
    return false;
  }
  constexpr llvm::StringRef fields[] = {
      "mdsl.analysis_only",      "mdsl.bridge_schema",
      "mdsl.capture_schema",     "mdsl.capture_version",
      "mdsl.compilation_identity", "mdsl.execution_intent",
      "mdsl.numerical_profile",  "mdsl.producer",
      "mdsl.required_runtime_guards", "mdsl.recovered_function",
      "mdsl.rewrite_policy",     "mdsl.semantic_version",
      "mdsl.source_file",        "mdsl.source_identity",
      "mdsl.translation_unit",
  };
  if (!exactModuleAttributes(module, fields) ||
      !moduleString(module, "mdsl.bridge_schema", kBridgeSchema) ||
      !moduleString(module, "mdsl.capture_schema",
                    kRecoveredGemmCaptureSchemaV1) ||
      !moduleString(module, "mdsl.execution_intent", "generic") ||
      !moduleString(module, "mdsl.numerical_profile",
                    kRecoveredGemmNumericalProfileV1) ||
      !moduleString(module, "mdsl.producer", "clang-libtooling-v1") ||
      !moduleString(module, "mdsl.rewrite_policy", "preserve_original_cpp")) {
    error = "recovered GEMM analysis module metadata is incomplete or invalid";
    return false;
  }
  const auto analysis_only =
      module->getAttrOfType<mlir::BoolAttr>("mdsl.analysis_only");
  const auto capture_version =
      module->getAttrOfType<mlir::IntegerAttr>("mdsl.capture_version");
  const auto semantic_version =
      module->getAttrOfType<mlir::IntegerAttr>("mdsl.semantic_version");
  const auto compilation =
      module->getAttrOfType<mlir::StringAttr>("mdsl.compilation_identity");
  const auto source_file =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_file");
  const auto source_identity =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_identity");
  const auto translation_unit =
      module->getAttrOfType<mlir::StringAttr>("mdsl.translation_unit");
  const auto recovered_function =
      module->getAttrOfType<mlir::StringAttr>("mdsl.recovered_function");
  const auto required_guards =
      module->getAttrOfType<mlir::ArrayAttr>("mdsl.required_runtime_guards");
  if (!analysis_only || !analysis_only.getValue() || !capture_version ||
      !capture_version.getType().isSignlessInteger(32) ||
      capture_version.getInt() != 1 || !semantic_version ||
      !semantic_version.getType().isSignlessInteger(32) ||
      semantic_version.getInt() != 1 || !compilation ||
      compilation.getValue().empty() || !source_file ||
      !source_file.getValue().ends_with(".mdsl") || !source_identity ||
      source_identity.getValue().empty() || !recovered_function ||
      recovered_function.getValue().empty() || !required_guards ||
      required_guards.size() != std::size(kRequiredGuards) ||
      !translation_unit ||
      translation_unit.getValue() != source_file.getValue()) {
    error = "recovered GEMM analysis version, source, or permission metadata is invalid";
    return false;
  }
  for (std::size_t index = 0; index < std::size(kRequiredGuards); ++index) {
    const auto guard =
        mlir::dyn_cast<mlir::StringAttr>(required_guards[index]);
    if (!guard || guard.getValue() != kRequiredGuards[index]) {
      error = "recovered GEMM analysis module lost its exact ordered runtime guards";
      return false;
    }
  }
  if (!llvm::hasSingleElement(module.getBody()->getOperations())) {
    error = "recovered GEMM analysis module requires exactly one semantic site";
    return false;
  }
  auto function =
      mlir::dyn_cast<mlir::func::FuncOp>(module.getBody()->front());
  if (!function || !function.isPublic() ||
      !llvm::hasSingleElement(function.getBody()) ||
      function.getBody().front().getNumArguments() != 3 ||
      std::distance(function.getBody().front().begin(),
                    function.getBody().front().end()) != 2) {
    error = "recovered GEMM analysis site has an invalid function envelope";
    return false;
  }
  const auto site = function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  const auto ordinal =
      function->getAttrOfType<mlir::IntegerAttr>("mdsl.capture_ordinal");
  mlir::Block &block = function.getBody().front();
  auto gemm = mlir::dyn_cast<mlir_dialect::GemmOp>(block.front());
  auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
  if (!site || !ordinal || !ordinal.getType().isSignlessInteger(64) ||
      ordinal.getInt() != 0 ||
      function.getName() !=
          (llvm::Twine("__matcore_semantic_") + site.getValue()).str() ||
      !gemm || !return_op || gemm.getSiteId() != site.getValue() ||
      gemm.getLhs() != block.getArgument(0) ||
      gemm.getRhs() != block.getArgument(1) ||
      gemm.getOutput() != block.getArgument(2) ||
      return_op.getNumOperands() != 1 ||
      return_op.getOperand(0) != gemm.getResult()) {
    error = "recovered GEMM analysis site is not one destination-tied GEMM";
    return false;
  }
  const auto origin_kind = gemm.getOrigin().getAs<mlir::StringAttr>("kind");
  const auto permission =
      gemm.getOrigin().getAs<mlir::StringAttr>("permission");
  const auto profile =
      gemm.getNumerical().getAs<mlir::StringAttr>("profile");
  const auto provenance_kind =
      gemm.getProvenance().getAs<mlir::StringAttr>("kind");
  const auto provenance_file =
      gemm.getProvenance().getAs<mlir::StringAttr>("file");
  const auto provenance_compilation =
      gemm.getProvenance().getAs<mlir::StringAttr>("compilation_identity");
  if (!origin_kind || origin_kind.getValue() != "recovered_cpp_loop" ||
      !permission ||
      permission.getValue() != "source_proven_guard_required" || !profile ||
      profile.getValue() != kRecoveredGemmNumericalProfileV1 ||
      !provenance_kind ||
      provenance_kind.getValue() != "recovered_cpp_loop" ||
      !provenance_file || provenance_file != source_file ||
      !provenance_compilation || provenance_compilation != compilation ||
      gemm.getOrigin().get("canonical_callee")) {
    error = "recovered GEMM analysis site lost its source-proven, non-executable origin";
    return false;
  }
  return true;
}

bool fingerprintMathematicalGemmV1(
    mlir::ModuleOp module, MathematicalGemmFingerprintV1 &fingerprint,
    std::string &error) {
  fingerprint = {};
  error.clear();
  if (!acceptedFingerprintEnvelope(module, error))
    return false;
  mlir_dialect::GemmOp gemm = singleGemm(module, error);
  if (!gemm)
    return false;

  std::map<std::string, std::string> symbols;
  std::string canonical = "matcore-mathematical-gemm-v1;";
  appendTensorFingerprint(
      canonical, "lhs", gemm.getLhsSemantics(),
      mlir::cast<mlir::RankedTensorType>(gemm.getLhs().getType()), symbols);
  appendTensorFingerprint(
      canonical, "rhs", gemm.getRhsSemantics(),
      mlir::cast<mlir::RankedTensorType>(gemm.getRhs().getType()), symbols);
  appendTensorFingerprint(
      canonical, "output", gemm.getOutputSemantics(),
      mlir::cast<mlir::RankedTensorType>(gemm.getOutput().getType()), symbols);
  canonical += "destination_result=tied_overwrite;accumulation=";
  canonical += printedAttribute(gemm.getAccumulationTypeAttr());
  canonical += ";requirements=" +
               printedAttribute(gemm.getSemanticRequirements());
  canonical += ";aliasing=" + printedAttribute(gemm.getAliasing());
  canonical += ";effects=" + printedAttribute(gemm.getEffects());
  canonical += ";synchronization=" + gemm.getSynchronization().str();

  llvm::SmallVector<mlir::NamedAttribute> numerical_fields;
  for (mlir::NamedAttribute field : gemm.getNumerical()) {
    if (field.getName().strref() != "profile" &&
        field.getName().strref() != "derivation")
      numerical_fields.push_back(field);
  }
  mlir::Builder builder(module.getContext());
  canonical += ";numerical=" +
               printedAttribute(builder.getDictionaryAttr(numerical_fields));
  fingerprint.canonical_contract = std::move(canonical);
  fingerprint.sha256 = sha256Identity(fingerprint.canonical_contract);
  return true;
}

bool equivalentMathematicalGemmV1(mlir::ModuleOp left, mlir::ModuleOp right,
                                  bool &equivalent, std::string &error) {
  equivalent = false;
  MathematicalGemmFingerprintV1 left_fingerprint;
  MathematicalGemmFingerprintV1 right_fingerprint;
  if (!fingerprintMathematicalGemmV1(left, left_fingerprint, error)) {
    error = "left semantic GEMM is not fingerprintable: " + error;
    return false;
  }
  if (!fingerprintMathematicalGemmV1(right, right_fingerprint, error)) {
    error = "right semantic GEMM is not fingerprintable: " + error;
    return false;
  }
  equivalent = left_fingerprint.canonical_contract ==
                   right_fingerprint.canonical_contract &&
               left_fingerprint.sha256 == right_fingerprint.sha256;
  return true;
}

} // namespace matcore::mdslc::mlir_bridge
