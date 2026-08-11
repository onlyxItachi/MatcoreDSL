#include "MatcoreOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>

namespace matcore::mdslc::mlir_dialect {
namespace {

bool isCanonicalSiteId(llvm::StringRef value) {
  if (value.size() != 35 || !value.starts_with("mc_"))
    return false;
  return llvm::all_of(value.drop_front(3), [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

bool isCanonicalSymbol(llvm::StringRef value) {
  if (value.empty())
    return false;
  const auto first = static_cast<unsigned char>(value.front());
  if (std::isalpha(first) == 0 && first != '_')
    return false;
  return llvm::all_of(value.drop_front(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) != 0 || byte == '_';
  });
}

bool isSha256Identity(llvm::StringRef value) {
  if (!value.consume_front("sha256:") || value.size() != 64)
    return false;
  return llvm::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

mlir::LogicalResult requireExactDictionary(
    GemmOp operation, mlir::DictionaryAttr dictionary,
    llvm::ArrayRef<llvm::StringRef> names, llvm::StringRef context) {
  if (!dictionary || dictionary.size() != names.size())
    return operation.emitOpError()
           << context << " must contain exactly " << names.size()
           << " fields";
  llvm::StringSet<> expected;
  for (llvm::StringRef name : names)
    expected.insert(name);
  for (mlir::NamedAttribute attribute : dictionary) {
    if (!expected.contains(attribute.getName().strref()))
      return operation.emitOpError()
             << context << " contains unexpected field '"
             << attribute.getName().strref() << "'";
  }
  return mlir::success();
}

mlir::LogicalResult requireString(GemmOp operation,
                                  mlir::DictionaryAttr dictionary,
                                  llvm::StringRef name,
                                  llvm::StringRef expected,
                                  llvm::StringRef context) {
  const auto value = dictionary.getAs<mlir::StringAttr>(name);
  if (!value || value.getValue() != expected)
    return operation.emitOpError()
           << context << "." << name << " must be '" << expected << "'";
  return mlir::success();
}

mlir::LogicalResult requireStringOneOf(
    GemmOp operation, mlir::DictionaryAttr dictionary, llvm::StringRef name,
    llvm::ArrayRef<llvm::StringRef> supported, llvm::StringRef context) {
  const auto value = dictionary.getAs<mlir::StringAttr>(name);
  if (!value || !llvm::is_contained(supported, value.getValue()))
    return operation.emitOpError()
           << context << "." << name
           << " is not in the closed supported vocabulary";
  return mlir::success();
}

mlir::LogicalResult requireBoolean(GemmOp operation,
                                   mlir::DictionaryAttr dictionary,
                                   llvm::StringRef name, bool expected,
                                   llvm::StringRef context) {
  const auto value = dictionary.getAs<mlir::BoolAttr>(name);
  if (!value || value.getValue() != expected)
    return operation.emitOpError()
           << context << "." << name << " must be "
           << (expected ? "true" : "false");
  return mlir::success();
}

mlir::LogicalResult requireInteger(GemmOp operation,
                                   mlir::DictionaryAttr dictionary,
                                   llvm::StringRef name, std::int64_t expected,
                                   llvm::StringRef context) {
  const auto value = dictionary.getAs<mlir::IntegerAttr>(name);
  if (!value || !value.getType().isSignlessInteger(32) ||
      value.getInt() != expected)
    return operation.emitOpError()
           << context << "." << name << " must be " << expected;
  return mlir::success();
}

mlir::LogicalResult verifyScalarExpression(GemmOp operation,
                                           mlir::DictionaryAttr scalar,
                                           int64_t type_dimension,
                                           llvm::StringRef context) {
  const auto kind = scalar.getAs<mlir::StringAttr>("kind");
  if (!kind)
    return operation.emitOpError() << context << " requires a kind";
  if (kind.getValue() == "static") {
    if (mlir::failed(requireExactDictionary(operation, scalar,
                                            {"kind", "value"}, context)))
      return mlir::failure();
    const auto value = scalar.getAs<mlir::IntegerAttr>("value");
    if (!value || !value.getType().isSignlessInteger(64) ||
        value.getInt() <= 0 || mlir::ShapedType::isDynamic(type_dimension) ||
        value.getInt() != type_dimension)
      return operation.emitOpError()
             << context << " static value must equal its positive tensor dimension";
    return mlir::success();
  }
  if (kind.getValue() == "dynamic") {
    if (mlir::failed(requireExactDictionary(operation, scalar,
                                            {"kind", "symbol"}, context)))
      return mlir::failure();
    const auto symbol = scalar.getAs<mlir::StringAttr>("symbol");
    if (!symbol || !isCanonicalSymbol(symbol.getValue()) ||
        !mlir::ShapedType::isDynamic(type_dimension))
      return operation.emitOpError()
             << context
             << " dynamic symbol must be canonical and map to a dynamic tensor dimension";
    return mlir::success();
  }
  return operation.emitOpError()
         << context << " has unsupported scalar-expression kind '"
         << kind.getValue() << "'";
}

mlir::LogicalResult verifyStrideExpression(GemmOp operation,
                                           mlir::DictionaryAttr scalar,
                                           llvm::StringRef context) {
  const auto kind = scalar.getAs<mlir::StringAttr>("kind");
  if (!kind)
    return operation.emitOpError() << context << " requires a kind";
  if (kind.getValue() == "static") {
    if (mlir::failed(requireExactDictionary(operation, scalar,
                                            {"kind", "value"}, context)))
      return mlir::failure();
    const auto value = scalar.getAs<mlir::IntegerAttr>("value");
    if (!value || !value.getType().isSignlessInteger(64) ||
        value.getInt() <= 0)
      return operation.emitOpError()
             << context << " static stride must be positive";
    return mlir::success();
  }
  if (kind.getValue() == "dynamic") {
    if (mlir::failed(requireExactDictionary(operation, scalar,
                                            {"kind", "symbol"}, context)))
      return mlir::failure();
    const auto symbol = scalar.getAs<mlir::StringAttr>("symbol");
    if (!symbol || !isCanonicalSymbol(symbol.getValue()))
      return operation.emitOpError()
             << context << " dynamic stride symbol must be canonical";
    return mlir::success();
  }
  return operation.emitOpError()
         << context << " has unsupported stride-expression kind '"
         << kind.getValue() << "'";
}

mlir::LogicalResult verifyTensorSemantics(
    GemmOp operation, mlir::DictionaryAttr semantics,
    mlir::RankedTensorType type, llvm::StringRef role,
    llvm::StringRef mutability) {
  if (mlir::failed(requireExactDictionary(
          operation, semantics,
          {"alignment_bytes", "alignment_contract", "layout", "memory_space",
           "mutability", "role", "shape", "source_expression", "strides"},
          role)))
    return mlir::failure();
  if (mlir::failed(requireString(operation, semantics, "role", role, role)) ||
      mlir::failed(requireString(operation, semantics, "mutability",
                                 mutability, role)) ||
      mlir::failed(requireString(operation, semantics, "memory_space", "host",
                                 role)) ||
      mlir::failed(requireString(operation, semantics, "alignment_contract",
                                 "required_precondition", role)))
    return mlir::failure();

  const auto expression = semantics.getAs<mlir::StringAttr>("source_expression");
  if (!expression || expression.getValue().empty())
    return operation.emitOpError()
           << role << ".source_expression must not be empty";
  if (type.getRank() != 2 || !type.getElementType().isF32())
    return operation.emitOpError()
           << role << " must be a rank-two f32 tensor";

  const auto shape = semantics.getAs<mlir::ArrayAttr>("shape");
  const auto strides = semantics.getAs<mlir::ArrayAttr>("strides");
  if (!shape || !strides || shape.size() != 2 || strides.size() != 2)
    return operation.emitOpError()
           << role << " shape and strides must each contain two expressions";
  for (unsigned index = 0; index < 2; ++index) {
    const auto shape_expression =
        mlir::dyn_cast<mlir::DictionaryAttr>(shape[index]);
    const auto stride_expression =
        mlir::dyn_cast<mlir::DictionaryAttr>(strides[index]);
    if (!shape_expression || !stride_expression)
      return operation.emitOpError()
             << role << " shape/stride entries must be dictionaries";
    if (mlir::failed(verifyScalarExpression(
            operation, shape_expression, type.getDimSize(index),
            (role + ".shape[" + llvm::Twine(index) + "]").str())) ||
        mlir::failed(verifyStrideExpression(
            operation, stride_expression,
            (role + ".strides[" + llvm::Twine(index) + "]").str())))
      return mlir::failure();
  }

  const auto layout = semantics.getAs<mlir::StringAttr>("layout");
  if (!layout || layout.getValue() != "row_major_contiguous")
    return operation.emitOpError()
           << role << ".layout must be row_major_contiguous";
  if (strides[0] != shape[1])
    return operation.emitOpError()
           << role << " row-major leading stride must equal shape[1]";
  const auto unit_stride = mlir::dyn_cast<mlir::DictionaryAttr>(strides[1]);
  const auto unit_kind = unit_stride.getAs<mlir::StringAttr>("kind");
  const auto unit_value = unit_stride.getAs<mlir::IntegerAttr>("value");
  if (!unit_kind || unit_kind.getValue() != "static" || !unit_value ||
      !unit_value.getType().isSignlessInteger(64) ||
      unit_value.getInt() != 1)
    return operation.emitOpError()
           << role << " row-major minor stride must be static one";

  const auto alignment = semantics.getAs<mlir::IntegerAttr>("alignment_bytes");
  if (!alignment || !alignment.getType().isSignlessInteger(64) ||
      alignment.getInt() < 4 ||
      (alignment.getInt() & (alignment.getInt() - 1)) != 0)
    return operation.emitOpError()
           << role << ".alignment_bytes must be a power of two at least four";
  return mlir::success();
}

mlir::LogicalResult verifyStringArray(GemmOp operation, mlir::ArrayAttr values,
                                      llvm::ArrayRef<llvm::StringRef> expected,
                                      llvm::StringRef context) {
  if (!values || values.size() != expected.size())
    return operation.emitOpError()
           << context << " has the wrong number of entries";
  for (auto [value, name] : llvm::zip(values, expected)) {
    const auto string = mlir::dyn_cast<mlir::StringAttr>(value);
    if (!string || string.getValue() != name)
      return operation.emitOpError()
             << context << " must use the canonical ordered entries";
  }
  return mlir::success();
}

mlir::LogicalResult verifyAliasing(GemmOp operation) {
  const auto aliasing = operation.getAliasing();
  if (aliasing.size() != 2)
    return operation.emitOpError()
           << "aliasing must contain exactly output/lhs and output/rhs required no-alias preconditions";
  constexpr llvm::StringLiteral seconds[] = {"lhs", "rhs"};
  for (unsigned index = 0; index < 2; ++index) {
    const auto relation =
        mlir::dyn_cast<mlir::DictionaryAttr>(aliasing[index]);
    if (mlir::failed(requireExactDictionary(
            operation, relation,
            {"contract", "first", "relation", "second"},
            "aliasing relation")) ||
        mlir::failed(requireString(operation, relation, "contract",
                                   "required_precondition",
                                   "aliasing relation")) ||
        mlir::failed(requireString(operation, relation, "first", "output",
                                   "aliasing relation")) ||
        mlir::failed(requireString(operation, relation, "relation", "no_alias",
                                   "aliasing relation")) ||
        mlir::failed(requireString(operation, relation, "second", seconds[index],
                                   "aliasing relation")))
      return mlir::failure();
  }
  return mlir::success();
}

enum class SemanticOriginKind {
  ExplicitCall,
  RecoveredSourceProvenGuardRequired,
  RecoveredRecognizedRewriteRejected,
};

bool isRecoveredOrigin(SemanticOriginKind origin_kind) {
  return origin_kind != SemanticOriginKind::ExplicitCall;
}

mlir::LogicalResult verifyOrigin(GemmOp operation,
                                 SemanticOriginKind &origin_kind) {
  const auto origin = operation.getOrigin();
  const auto kind = origin.getAs<mlir::StringAttr>("kind");
  if (!kind)
    return operation.emitOpError() << "origin.kind is required";
  if (kind.getValue() == "explicit_call") {
    origin_kind = SemanticOriginKind::ExplicitCall;
    if (mlir::failed(requireExactDictionary(
            operation, origin, {"canonical_callee", "kind", "version"},
            "origin")) ||
        mlir::failed(requireInteger(operation, origin, "version", 1,
                                    "origin")) ||
        mlir::failed(requireString(operation, origin, "canonical_callee",
                                   "matcore::mdsl::gemm", "origin")))
      return mlir::failure();
    return mlir::success();
  }
  if (kind.getValue() == "recovered_cpp_loop") {
    if (mlir::failed(requireExactDictionary(
            operation, origin, {"kind", "pattern", "permission", "version"},
            "origin")) ||
        mlir::failed(requireInteger(operation, origin, "version", 1,
                                    "origin")) ||
        mlir::failed(requireString(
            operation, origin, "pattern",
            "canonical-row-major-f32-gemm-v1", "origin")))
      return mlir::failure();
    const auto permission = origin.getAs<mlir::StringAttr>("permission");
    if (!permission)
      return operation.emitOpError()
             << "origin.permission is required for recovered_cpp_loop";
    if (permission.getValue() == "source_proven_guard_required") {
      origin_kind = SemanticOriginKind::RecoveredSourceProvenGuardRequired;
      return mlir::success();
    }
    if (permission.getValue() == "recognized_rewrite_rejected") {
      origin_kind = SemanticOriginKind::RecoveredRecognizedRewriteRejected;
      return mlir::success();
    }
    return operation.emitOpError()
           << "origin.permission has unsupported value '"
           << permission.getValue() << "'";
  }
  return operation.emitOpError()
         << "origin.kind has unsupported value '" << kind.getValue() << "'";
}

mlir::LogicalResult verifyNumerical(GemmOp operation,
                                    SemanticOriginKind origin_kind) {
  const auto numerical = operation.getNumerical();
  if (mlir::failed(requireExactDictionary(
          operation, numerical,
          {"accumulation_dtype", "approximate_math", "contraction",
           "derivation", "exception_status", "infinity", "inplace", "nan",
           "profile", "reassociation", "reduction_order", "rounding",
           "signed_zero", "subnormals", "trapping_exceptions"},
          "numerical")))
    return mlir::failure();
  if (mlir::failed(requireStringOneOf(operation, numerical,
                                      "accumulation_dtype", {"f32"},
                                      "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "profile",
          {"explicit-gemm-f32-v1",
           "recovered-cpp-gemm-f32-source-proven-v1",
           "recovered-cpp-gemm-f32-strict-v1"},
          "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "derivation",
          {"explicit_edsl_contract", "effective_cpp_semantics"},
          "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "reassociation",
          {"within_k_reduction", "forbidden"}, "numerical")) ||
      mlir::failed(requireStringOneOf(operation, numerical, "contraction",
                                      {"allowed", "within_statement"},
                                      "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "reduction_order",
          {"implementation_defined_within_k", "increasing_k"},
          "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "nan",
          {"preserve_classification_payload_order_unspecified", "strict"},
          "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "infinity",
          {"ieee_no_no_infs_assumption"}, "numerical")) ||
      mlir::failed(requireStringOneOf(operation, numerical, "signed_zero",
                                      {"relaxed", "preserve"},
                                      "numerical")) ||
      mlir::failed(requireStringOneOf(operation, numerical, "rounding",
                                      {"nearest_ties_even"}, "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "exception_status",
          {"incoming_not_preserved_postcall_unspecified"}, "numerical")) ||
      mlir::failed(requireStringOneOf(
          operation, numerical, "subnormals",
          {"ieee_gradual_ftz_daz_forbidden"}, "numerical")) ||
      mlir::failed(requireStringOneOf(operation, numerical,
                                      "trapping_exceptions", {"unsupported"},
                                      "numerical")) ||
      mlir::failed(requireBoolean(operation, numerical, "approximate_math",
                                  false, "numerical")) ||
      mlir::failed(requireBoolean(operation, numerical, "inplace", false,
                                  "numerical")))
    return mlir::failure();

  llvm::StringRef profile;
  llvm::StringRef derivation;
  llvm::StringRef reassociation;
  llvm::StringRef contraction;
  llvm::StringRef reduction_order;
  llvm::StringRef nan;
  llvm::StringRef signed_zero;
  switch (origin_kind) {
  case SemanticOriginKind::ExplicitCall:
    profile = "explicit-gemm-f32-v1";
    derivation = "explicit_edsl_contract";
    reassociation = "within_k_reduction";
    contraction = "allowed";
    reduction_order = "implementation_defined_within_k";
    nan = "preserve_classification_payload_order_unspecified";
    signed_zero = "relaxed";
    break;
  case SemanticOriginKind::RecoveredSourceProvenGuardRequired:
    profile = "recovered-cpp-gemm-f32-source-proven-v1";
    derivation = "effective_cpp_semantics";
    reassociation = "within_k_reduction";
    contraction = "allowed";
    reduction_order = "implementation_defined_within_k";
    nan = "preserve_classification_payload_order_unspecified";
    signed_zero = "relaxed";
    break;
  case SemanticOriginKind::RecoveredRecognizedRewriteRejected:
    profile = "recovered-cpp-gemm-f32-strict-v1";
    derivation = "effective_cpp_semantics";
    reassociation = "forbidden";
    contraction = "within_statement";
    reduction_order = "increasing_k";
    nan = "strict";
    signed_zero = "preserve";
    break;
  }
  return mlir::failure(
      mlir::failed(requireString(operation, numerical, "profile", profile,
                                 "numerical")) ||
      mlir::failed(requireString(operation, numerical, "derivation",
                                 derivation, "numerical")) ||
      mlir::failed(requireString(operation, numerical, "reassociation",
                                 reassociation, "numerical")) ||
      mlir::failed(requireString(operation, numerical, "contraction",
                                 contraction, "numerical")) ||
      mlir::failed(requireString(operation, numerical, "reduction_order",
                                 reduction_order, "numerical")) ||
      mlir::failed(requireString(operation, numerical, "nan", nan,
                                 "numerical")) ||
      mlir::failed(requireString(operation, numerical, "signed_zero",
                                 signed_zero, "numerical")));
}

mlir::LogicalResult verifyPolicy(GemmOp operation,
                                 SemanticOriginKind origin_kind) {
  const auto policy = operation.getPolicy();
  if (mlir::failed(requireExactDictionary(operation, policy,
                                          {"fallback", "target"}, "policy")))
    return mlir::failure();
  if (!isRecoveredOrigin(origin_kind))
    return mlir::failure(
        mlir::failed(requireString(operation, policy, "target", "cpu",
                                   "policy")) ||
        mlir::failed(requireString(operation, policy, "fallback", "error",
                                   "policy")));
  return mlir::failure(
      mlir::failed(requireString(operation, policy, "target", "generic",
                                 "policy")) ||
      mlir::failed(requireString(operation, policy, "fallback",
                                 "preserve_original_cpp", "policy")));
}

mlir::LogicalResult verifyRange(GemmOp operation, mlir::DictionaryAttr range,
                                llvm::StringRef context, std::int64_t &begin,
                                std::int64_t &end) {
  if (mlir::failed(requireExactDictionary(operation, range, {"begin", "end"},
                                          context)))
    return mlir::failure();
  const auto begin_attr = range.getAs<mlir::IntegerAttr>("begin");
  const auto end_attr = range.getAs<mlir::IntegerAttr>("end");
  if (!begin_attr || !end_attr ||
      !begin_attr.getType().isSignlessInteger(64) ||
      !end_attr.getType().isSignlessInteger(64) || begin_attr.getInt() < 0 ||
      end_attr.getInt() <= begin_attr.getInt())
    return operation.emitOpError() << context << " must be a nonempty range";
  begin = begin_attr.getInt();
  end = end_attr.getInt();
  return mlir::success();
}

mlir::LogicalResult verifySourceLocation(GemmOp operation,
                                         mlir::DictionaryAttr provenance) {
  const auto file = provenance.getAs<mlir::StringAttr>("file");
  const auto line = provenance.getAs<mlir::IntegerAttr>("line");
  const auto column = provenance.getAs<mlir::IntegerAttr>("column");
  const auto offset = provenance.getAs<mlir::IntegerAttr>("offset");
  constexpr std::uint64_t max_unsigned =
      std::numeric_limits<unsigned>::max();
  if (!file || file.getValue().empty() || !line ||
      !line.getType().isSignlessInteger(64) || line.getInt() <= 0 ||
      static_cast<std::uint64_t>(line.getInt()) > max_unsigned || !column ||
      !column.getType().isSignlessInteger(64) || column.getInt() <= 0 ||
      static_cast<std::uint64_t>(column.getInt()) > max_unsigned || !offset ||
      !offset.getType().isSignlessInteger(64) || offset.getInt() < 0)
    return operation.emitOpError() << "provenance source fields are incomplete";
  const auto location = mlir::dyn_cast<mlir::FileLineColLoc>(operation.getLoc());
  if (!location || location.getFilename() != file.getValue() ||
      location.getLine() != static_cast<unsigned>(line.getInt()) ||
      location.getColumn() != static_cast<unsigned>(column.getInt()))
    return operation.emitOpError()
           << "MLIR location must match the preserved source file/line/column";
  return mlir::success();
}

mlir::LogicalResult verifyExplicitProvenance(
    GemmOp operation, mlir::DictionaryAttr provenance) {
  if (mlir::failed(requireExactDictionary(
          operation, provenance,
          {"argument_ranges", "call_range", "column", "file", "kind", "line",
           "offset", "version"},
          "provenance")) ||
      mlir::failed(requireString(operation, provenance, "kind", "explicit_call",
                                 "provenance")) ||
      mlir::failed(requireInteger(operation, provenance, "version", 1,
                                  "provenance")) ||
      mlir::failed(verifySourceLocation(operation, provenance)))
    return mlir::failure();
  const auto offset = provenance.getAs<mlir::IntegerAttr>("offset");
  const auto call_range = provenance.getAs<mlir::DictionaryAttr>("call_range");
  const auto argument_ranges =
      provenance.getAs<mlir::ArrayAttr>("argument_ranges");
  if (!call_range || !argument_ranges ||
      (argument_ranges.size() != 3 && argument_ranges.size() != 4))
    return operation.emitOpError()
           << "explicit provenance requires three or four argument ranges";
  std::int64_t call_begin = 0;
  std::int64_t call_end = 0;
  if (mlir::failed(verifyRange(operation, call_range, "provenance.call_range",
                               call_begin, call_end)))
    return mlir::failure();
  if (call_begin != offset.getInt())
    return operation.emitOpError()
           << "provenance call range must begin at source offset";
  std::int64_t previous_end = call_begin;
  for (mlir::Attribute encoded_range : argument_ranges) {
    const auto range = mlir::dyn_cast<mlir::DictionaryAttr>(encoded_range);
    std::int64_t begin = 0;
    std::int64_t end = 0;
    if (!range || mlir::failed(verifyRange(operation, range,
                                            "provenance.argument_range", begin,
                                            end)))
      return mlir::failure();
    if (begin < previous_end || end > call_end)
      return operation.emitOpError()
             << "provenance argument ranges must be ordered and inside the call";
    previous_end = end;
  }
  return mlir::success();
}

mlir::LogicalResult verifyRecoveredProvenance(
    GemmOp operation, mlir::DictionaryAttr provenance) {
  if (mlir::failed(requireExactDictionary(
          operation, provenance,
          {"column", "compilation_identity", "file", "kind", "line", "offset",
           "proof_ranges", "source_range", "source_snapshot", "version"},
          "provenance")) ||
      mlir::failed(requireString(operation, provenance, "kind",
                                 "recovered_cpp_loop", "provenance")) ||
      mlir::failed(requireInteger(operation, provenance, "version", 1,
                                  "provenance")) ||
      mlir::failed(verifySourceLocation(operation, provenance)))
    return mlir::failure();
  const auto identity =
      provenance.getAs<mlir::StringAttr>("compilation_identity");
  const auto snapshot = provenance.getAs<mlir::StringAttr>("source_snapshot");
  const auto offset = provenance.getAs<mlir::IntegerAttr>("offset");
  const auto source_range =
      provenance.getAs<mlir::DictionaryAttr>("source_range");
  const auto proof_ranges = provenance.getAs<mlir::ArrayAttr>("proof_ranges");
  if (!identity || identity.getValue().empty() || !snapshot ||
      !isSha256Identity(snapshot.getValue()) || !source_range || !proof_ranges ||
      proof_ranges.size() < 3)
    return operation.emitOpError()
           << "recovered provenance requires authenticated source and proof ranges";
  std::int64_t source_begin = 0;
  std::int64_t source_end = 0;
  if (mlir::failed(verifyRange(operation, source_range,
                               "provenance.source_range", source_begin,
                               source_end)))
    return mlir::failure();
  if (source_begin != offset.getInt())
    return operation.emitOpError()
           << "recovered source range must begin at source offset";

  llvm::StringSet<> proof_kinds;
  for (mlir::Attribute encoded_range : proof_ranges) {
    const auto range = mlir::dyn_cast<mlir::DictionaryAttr>(encoded_range);
    if (mlir::failed(requireExactDictionary(
            operation, range, {"begin", "end", "kind"},
            "provenance.proof_range")))
      return mlir::failure();
    const auto kind = range.getAs<mlir::StringAttr>("kind");
    const auto begin = range.getAs<mlir::IntegerAttr>("begin");
    const auto end = range.getAs<mlir::IntegerAttr>("end");
    if (!kind || !isCanonicalSymbol(kind.getValue()) ||
        !proof_kinds.insert(kind.getValue()).second || !begin || !end ||
        !begin.getType().isSignlessInteger(64) ||
        !end.getType().isSignlessInteger(64) ||
        begin.getInt() < source_begin || end.getInt() <= begin.getInt() ||
        end.getInt() > source_end)
      return operation.emitOpError()
             << "recovered proof ranges must be unique, nonempty, and inside the source range";
    if (kind.getValue() == "outer_loop" &&
        (begin.getInt() != source_begin || end.getInt() != source_end))
      return operation.emitOpError()
             << "outer_loop proof range must equal the recovered source range";
  }
  if (!proof_kinds.contains("outer_loop") ||
      !proof_kinds.contains("accumulator_update") ||
      !proof_kinds.contains("output_store"))
    return operation.emitOpError()
           << "recovered provenance requires outer-loop, accumulator-update, and output-store proofs";
  return mlir::success();
}

mlir::LogicalResult verifyProvenance(GemmOp operation,
                                     SemanticOriginKind origin_kind) {
  if (!isRecoveredOrigin(origin_kind))
    return verifyExplicitProvenance(operation, operation.getProvenance());
  return verifyRecoveredProvenance(operation, operation.getProvenance());
}

template <typename OpTy>
mlir::LogicalResult requireExactDictionaryFor(
    OpTy operation, mlir::DictionaryAttr dictionary,
    llvm::ArrayRef<llvm::StringRef> names, llvm::StringRef context) {
  if (!dictionary || dictionary.size() != names.size())
    return operation.emitOpError()
           << context << " must contain exactly " << names.size()
           << " fields";
  llvm::StringSet<> expected;
  for (llvm::StringRef name : names)
    expected.insert(name);
  for (mlir::NamedAttribute attribute : dictionary) {
    if (!expected.contains(attribute.getName().strref()))
      return operation.emitOpError()
             << context << " contains unexpected field '"
             << attribute.getName().strref() << "'";
  }
  return mlir::success();
}

template <typename OpTy>
mlir::LogicalResult requireStringFor(OpTy operation,
                                     mlir::DictionaryAttr dictionary,
                                     llvm::StringRef name,
                                     llvm::StringRef expected,
                                     llvm::StringRef context) {
  const auto value = dictionary.template getAs<mlir::StringAttr>(name);
  if (!value || value.getValue() != expected)
    return operation.emitOpError()
           << context << "." << name << " must be '" << expected << "'";
  return mlir::success();
}

template <typename OpTy>
mlir::LogicalResult requireBooleanFor(OpTy operation,
                                      mlir::DictionaryAttr dictionary,
                                      llvm::StringRef name, bool expected,
                                      llvm::StringRef context) {
  const auto value = dictionary.template getAs<mlir::BoolAttr>(name);
  if (!value || value.getValue() != expected)
    return operation.emitOpError()
           << context << "." << name << " must be "
           << (expected ? "true" : "false");
  return mlir::success();
}

mlir::LogicalResult verifyMapScalarExpressions(
    MapOp operation, mlir::ArrayAttr values, mlir::RankedTensorType type,
    llvm::StringRef context, bool shape) {
  if (!values || values.size() != static_cast<std::size_t>(type.getRank()))
    return operation.emitOpError()
           << context << " must have one entry per tensor dimension";
  for (auto [index, encoded] : llvm::enumerate(values)) {
    const auto expression = mlir::dyn_cast<mlir::DictionaryAttr>(encoded);
    if (!expression)
      return operation.emitOpError()
             << context << " entries must be dictionaries";
    const auto kind = expression.getAs<mlir::StringAttr>("kind");
    if (!kind)
      return operation.emitOpError() << context << " entry requires kind";
    if (kind.getValue() == "static") {
      if (mlir::failed(requireExactDictionaryFor(
              operation, expression, {"kind", "value"}, context)))
        return mlir::failure();
      const auto value = expression.getAs<mlir::IntegerAttr>("value");
      if (!value || !value.getType().isSignlessInteger(64) ||
          value.getInt() <= 0)
        return operation.emitOpError()
               << context << " static values must be positive signless i64";
      if (shape &&
          (mlir::ShapedType::isDynamic(type.getDimSize(index)) ||
           value.getInt() != type.getDimSize(index)))
        return operation.emitOpError()
               << context << " static values must match tensor dimensions";
      continue;
    }
    if (kind.getValue() == "dynamic") {
      if (mlir::failed(requireExactDictionaryFor(
              operation, expression, {"kind", "symbol"}, context)))
        return mlir::failure();
      const auto symbol = expression.getAs<mlir::StringAttr>("symbol");
      if (!symbol || !isCanonicalSymbol(symbol.getValue()) ||
          (shape && !mlir::ShapedType::isDynamic(type.getDimSize(index))))
        return operation.emitOpError()
               << context
               << " dynamic symbols must be canonical and map to dynamic dimensions";
      continue;
    }
    return operation.emitOpError()
           << context << " contains unsupported scalar-expression kind '"
           << kind.getValue() << "'";
  }
  return mlir::success();
}

mlir::LogicalResult verifyMapTensorContract(MapOp operation,
                                            mlir::RankedTensorType type) {
  const auto contract = operation.getTensorContract();
  if (mlir::failed(requireExactDictionaryFor(
          operation, contract,
          {"aliasing", "alignment_bytes", "alignment_contract", "dtype",
           "input_mutability", "layout", "memory_space", "rank",
           "result_mutability", "shape", "strides"},
          "tensor_contract")) ||
      mlir::failed(requireStringFor(operation, contract, "aliasing",
                                    "functional_result_no_inplace",
                                    "tensor_contract")) ||
      mlir::failed(requireStringFor(
          operation, contract, "alignment_contract",
          "required_precondition_and_result_contract", "tensor_contract")) ||
      mlir::failed(requireStringFor(operation, contract, "dtype", "f32",
                                    "tensor_contract")) ||
      mlir::failed(requireStringFor(operation, contract, "input_mutability",
                                    "read", "tensor_contract")) ||
      mlir::failed(requireStringFor(operation, contract, "layout",
                                    "row_major_contiguous",
                                    "tensor_contract")) ||
      mlir::failed(requireStringFor(operation, contract, "memory_space",
                                    "host", "tensor_contract")) ||
      mlir::failed(requireStringFor(operation, contract, "result_mutability",
                                    "functional_write_once",
                                    "tensor_contract")))
    return mlir::failure();

  const auto rank = contract.getAs<mlir::IntegerAttr>("rank");
  const auto alignment =
      contract.getAs<mlir::IntegerAttr>("alignment_bytes");
  if (!rank || !rank.getType().isSignlessInteger(64) || rank.getInt() <= 0 ||
      rank.getInt() != type.getRank())
    return operation.emitOpError()
           << "tensor_contract.rank must match the positive tensor rank";
  if (!alignment || !alignment.getType().isSignlessInteger(64) ||
      alignment.getInt() < 4 ||
      (alignment.getInt() & (alignment.getInt() - 1)) != 0)
    return operation.emitOpError()
           << "tensor_contract.alignment_bytes must be a power of two at least four";

  const auto shape = contract.getAs<mlir::ArrayAttr>("shape");
  const auto strides = contract.getAs<mlir::ArrayAttr>("strides");
  if (mlir::failed(verifyMapScalarExpressions(operation, shape, type,
                                               "tensor_contract.shape",
                                               /*shape=*/true)) ||
      mlir::failed(verifyMapScalarExpressions(operation, strides, type,
                                               "tensor_contract.strides",
                                               /*shape=*/false)))
    return mlir::failure();

  if (type.getRank() != 2 || !type.getElementType().isF32())
    return operation.emitOpError()
           << "version-1 map input and result must be rank-two f32 tensors";
  const auto unit_stride =
      mlir::dyn_cast<mlir::DictionaryAttr>(strides[type.getRank() - 1]);
  const auto unit_kind = unit_stride.getAs<mlir::StringAttr>("kind");
  const auto unit_value = unit_stride.getAs<mlir::IntegerAttr>("value");
  if (!unit_kind || unit_kind.getValue() != "static" || !unit_value ||
      !unit_value.getType().isSignlessInteger(64) || unit_value.getInt() != 1)
    return operation.emitOpError()
           << "row-major tensor contract requires static unit minor stride";
  for (int64_t index = 0; index + 1 < type.getRank(); ++index) {
    const auto stride = mlir::dyn_cast<mlir::DictionaryAttr>(strides[index]);
    const auto dimension =
        mlir::dyn_cast<mlir::DictionaryAttr>(shape[index + 1]);
    if (stride != dimension)
      return operation.emitOpError()
             << "version-1 row-major strides must equal the following dimension";
  }
  return mlir::success();
}

mlir::LogicalResult verifyStaticCoordinate(MapOp operation,
                                           mlir::Attribute encoded,
                                           int64_t upper_bound,
                                           llvm::StringRef context,
                                           bool allow_upper_bound) {
  const auto value = mlir::dyn_cast<mlir::IntegerAttr>(encoded);
  if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0)
    return operation.emitOpError()
           << context << " must contain nonnegative signless i64 values";
  const bool in_bounds = allow_upper_bound ? value.getInt() <= upper_bound
                                           : value.getInt() < upper_bound;
  if (!in_bounds)
    return operation.emitOpError() << context << " is out of bounds";
  return mlir::success();
}

mlir::LogicalResult verifyDomain(MapOp operation,
                                 mlir::RankedTensorType type) {
  const auto domain = operation.getDomain();
  const auto kind = domain.getAs<mlir::StringAttr>("kind");
  const auto version = domain.getAs<mlir::IntegerAttr>("version");
  if (!kind || !version || !version.getType().isSignlessInteger(32) ||
      version.getInt() != 1)
    return operation.emitOpError()
           << "domain requires a closed kind and version 1 : i32";

  const bool has_mask = static_cast<bool>(operation.getMask());
  if (kind.getValue() == "all") {
    if (mlir::failed(requireExactDictionaryFor(operation, domain,
                                                {"kind", "version"},
                                                "domain")))
      return mlir::failure();
    if (has_mask || operation.getOutsideDomain() != "not_applicable")
      return operation.emitOpError()
             << "domain(all) forbids a mask and requires outside_domain=not_applicable";
    return mlir::success();
  }

  if (operation.getOutsideDomain() != "preserve_input")
    return operation.emitOpError()
           << "partial domains require outside_domain=preserve_input";

  if (kind.getValue() == "mask") {
    if (mlir::failed(requireExactDictionaryFor(operation, domain,
                                                {"kind", "shape_equality",
                                                 "version"},
                                                "domain")))
      return mlir::failure();
    if (mlir::failed(requireStringFor(
            operation, domain, "shape_equality", "required_precondition",
            "domain")))
      return mlir::failure();
    if (!has_mask)
      return operation.emitOpError()
             << "domain(mask) requires one ranked i1 mask operand";
    const auto mask_type =
        mlir::dyn_cast<mlir::RankedTensorType>(operation.getMask().getType());
    if (!mask_type || !mask_type.getElementType().isInteger(1) ||
        mask_type.getRank() != type.getRank())
      return operation.emitOpError()
             << "domain(mask) requires a rank-compatible i1 tensor";
    for (int64_t index = 0; index < type.getRank(); ++index) {
      const int64_t input_dimension = type.getDimSize(index);
      const int64_t mask_dimension = mask_type.getDimSize(index);
      if (!mlir::ShapedType::isDynamic(input_dimension) &&
          !mlir::ShapedType::isDynamic(mask_dimension) &&
          input_dimension != mask_dimension)
        return operation.emitOpError()
               << "domain(mask) has a statically incompatible shape";
    }
    return mlir::success();
  }

  if (has_mask)
    return operation.emitOpError()
           << "only domain(mask) may carry a mask operand";
  if (!type.hasStaticShape())
    return operation.emitOpError()
           << "version-1 slice/indices domains require a static tensor shape";

  if (kind.getValue() == "slice") {
    if (mlir::failed(requireExactDictionaryFor(
            operation, domain, {"begin", "end", "kind", "step", "version"},
            "domain")))
      return mlir::failure();
    const auto begin = domain.getAs<mlir::ArrayAttr>("begin");
    const auto end = domain.getAs<mlir::ArrayAttr>("end");
    const auto step = domain.getAs<mlir::ArrayAttr>("step");
    const auto rank = static_cast<std::size_t>(type.getRank());
    if (!begin || !end || !step || begin.size() != rank ||
        end.size() != rank || step.size() != rank)
      return operation.emitOpError()
             << "domain(slice) begin/end/step must match tensor rank";
    bool partial = false;
    for (int64_t index = 0; index < type.getRank(); ++index) {
      if (mlir::failed(verifyStaticCoordinate(
              operation, begin[index], type.getDimSize(index),
              "domain(slice).begin", /*allow_upper_bound=*/false)) ||
          mlir::failed(verifyStaticCoordinate(
              operation, end[index], type.getDimSize(index),
              "domain(slice).end", /*allow_upper_bound=*/true)))
        return mlir::failure();
      const auto first = mlir::cast<mlir::IntegerAttr>(begin[index]).getInt();
      const auto last = mlir::cast<mlir::IntegerAttr>(end[index]).getInt();
      const auto increment = mlir::dyn_cast<mlir::IntegerAttr>(step[index]);
      if (last <= first || !increment ||
          !increment.getType().isSignlessInteger(64) ||
          increment.getInt() <= 0)
        return operation.emitOpError()
               << "domain(slice) requires nonempty bounds and positive i64 steps";
      partial = partial || first != 0 || last != type.getDimSize(index) ||
                increment.getInt() != 1;
    }
    if (!partial)
      return operation.emitOpError()
             << "a full static slice must use the canonical domain(all) form";
    return mlir::success();
  }

  if (kind.getValue() == "indices") {
    if (mlir::failed(requireExactDictionaryFor(
            operation, domain, {"coordinates", "kind", "version"},
            "domain")))
      return mlir::failure();
    const auto coordinates = domain.getAs<mlir::ArrayAttr>("coordinates");
    if (!coordinates || coordinates.empty())
      return operation.emitOpError()
             << "domain(indices) requires at least one coordinate";
    llvm::StringSet<> seen;
    for (mlir::Attribute encoded_coordinate : coordinates) {
      const auto coordinate =
          mlir::dyn_cast<mlir::ArrayAttr>(encoded_coordinate);
      if (!coordinate ||
          coordinate.size() != static_cast<std::size_t>(type.getRank()))
        return operation.emitOpError()
               << "domain(indices) coordinates must match tensor rank";
      std::string key;
      for (int64_t index = 0; index < type.getRank(); ++index) {
        if (mlir::failed(verifyStaticCoordinate(
                operation, coordinate[index], type.getDimSize(index),
                "domain(indices).coordinate", /*allow_upper_bound=*/false)))
          return mlir::failure();
        key += std::to_string(
            mlir::cast<mlir::IntegerAttr>(coordinate[index]).getInt());
        key += ':';
      }
      if (!seen.insert(key).second)
        return operation.emitOpError()
               << "domain(indices) coordinates must be unique";
    }
    return mlir::success();
  }

  return operation.emitOpError()
         << "domain.kind has unsupported value '" << kind.getValue() << "'";
}

mlir::LogicalResult verifyMapEffects(MapOp operation) {
  const auto effects = operation.getEffects();
  if (mlir::failed(requireExactDictionaryFor(
          operation, effects, {"read_write", "reads", "writes"},
          "effects")))
    return mlir::failure();
  const auto reads = effects.getAs<mlir::ArrayAttr>("reads");
  const auto writes = effects.getAs<mlir::ArrayAttr>("writes");
  const auto read_write = effects.getAs<mlir::ArrayAttr>("read_write");
  const bool has_mask = static_cast<bool>(operation.getMask());
  const std::size_t expected_reads = has_mask ? 2 : 1;
  if (!reads || reads.size() != expected_reads ||
      !mlir::isa<mlir::StringAttr>(reads[0]) ||
      mlir::cast<mlir::StringAttr>(reads[0]).getValue() != "input" ||
      (has_mask &&
       (!mlir::isa<mlir::StringAttr>(reads[1]) ||
        mlir::cast<mlir::StringAttr>(reads[1]).getValue() != "mask")) ||
      !writes || !writes.empty() || !read_write || !read_write.empty())
    return operation.emitOpError()
           << "effects must encode input/mask reads and no writes/read_write effects";
  return mlir::success();
}

mlir::LogicalResult verifyMapNumerical(MapOp operation) {
  const auto numerical = operation.getNumerical();
  if (mlir::failed(requireExactDictionaryFor(
          operation, numerical,
          {"approximate_math", "domain_application", "inplace", "profile",
           "result_identity"},
          "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "profile",
                                    "map-f32-v1", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "domain_application",
                                    "active_elements_only", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "result_identity",
                                    "new_functional_value", "numerical")) ||
      mlir::failed(requireBooleanFor(operation, numerical, "approximate_math",
                                     false, "numerical")) ||
      mlir::failed(requireBooleanFor(operation, numerical, "inplace", false,
                                     "numerical")))
    return mlir::failure();
  return mlir::success();
}

template <typename OpTy>
mlir::LogicalResult verifySemanticProvenance(OpTy operation,
                                             llvm::StringRef kind) {
  const auto provenance = operation.getProvenance();
  if (mlir::failed(requireExactDictionaryFor(
          operation, provenance,
          {"column", "file", "kind", "line", "source_anchor", "version"},
          "provenance")) ||
      mlir::failed(requireStringFor(operation, provenance, "kind", kind,
                                    "provenance")))
    return mlir::failure();
  const auto version = provenance.template getAs<mlir::IntegerAttr>("version");
  const auto file = provenance.template getAs<mlir::StringAttr>("file");
  const auto line = provenance.template getAs<mlir::IntegerAttr>("line");
  const auto column = provenance.template getAs<mlir::IntegerAttr>("column");
  const auto anchor =
      provenance.template getAs<mlir::StringAttr>("source_anchor");
  constexpr std::uint64_t max_unsigned =
      std::numeric_limits<unsigned>::max();
  if (!version || !version.getType().isSignlessInteger(32) ||
      version.getInt() != 1 || !file || file.getValue().empty() || !line ||
      !line.getType().isSignlessInteger(64) || line.getInt() <= 0 ||
      static_cast<std::uint64_t>(line.getInt()) > max_unsigned || !column ||
      !column.getType().isSignlessInteger(64) || column.getInt() <= 0 ||
      static_cast<std::uint64_t>(column.getInt()) > max_unsigned || !anchor ||
      anchor.getValue().empty())
    return operation.emitOpError()
           << "provenance requires a versioned source anchor and valid file/line/column";
  const auto location =
      mlir::dyn_cast<mlir::FileLineColLoc>(operation.getLoc());
  if (!location || location.getFilename() != file.getValue() ||
      location.getLine() != static_cast<unsigned>(line.getInt()) ||
      location.getColumn() != static_cast<unsigned>(column.getInt()))
    return operation.emitOpError()
           << "location must match exact semantic provenance";
  return mlir::success();
}

mlir::LogicalResult verifySinNumerical(SinOp operation) {
  const auto numerical = operation.getNumerical();
  if (mlir::failed(requireExactDictionaryFor(
          operation, numerical,
          {"accuracy", "approximate_math", "exception_status", "infinity",
           "nan", "profile", "rounding", "signed_zero", "subnormals",
           "trapping_exceptions"},
          "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "profile",
                                    "sin-f32-ieee-v1", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "accuracy",
                                    "correctly_rounded_f32", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "nan",
                                    "quiet_nan_payload_unspecified",
                                    "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "infinity",
                                    "quiet_nan", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "signed_zero",
                                    "preserve", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "rounding",
                                    "nearest_ties_even", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical, "exception_status",
                                    "postcall_unspecified", "numerical")) ||
      mlir::failed(requireStringFor(
          operation, numerical, "subnormals",
          "ieee_gradual_ftz_daz_forbidden", "numerical")) ||
      mlir::failed(requireStringFor(operation, numerical,
                                    "trapping_exceptions", "unsupported",
                                    "numerical")) ||
      mlir::failed(requireBooleanFor(operation, numerical, "approximate_math",
                                     false, "numerical")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult verifyMapInputContractPropagation(MapOp operation) {
  const auto contract = operation.getTensorContract();
  const auto anchor =
      operation.getProvenance().getAs<mlir::StringAttr>("source_anchor");
  if (auto producer = operation.getInput().getDefiningOp<GemmOp>()) {
    const auto semantics = producer.getOutputSemantics();
    constexpr llvm::StringLiteral shared_fields[] = {
        "alignment_bytes", "layout", "memory_space", "shape", "strides"};
    for (llvm::StringRef field : shared_fields) {
      if (contract.get(field) != semantics.get(field))
        return operation.emitOpError()
               << "tensor_contract." << field
               << " must be propagated exactly from the GEMM result";
    }
    if (!anchor || anchor.getValue() != producer.getSiteId())
      return operation.emitOpError()
             << "provenance.source_anchor must authenticate the producing GEMM site";
    return mlir::success();
  }
  if (auto producer = operation.getInput().getDefiningOp<MapOp>()) {
    if (contract != producer.getTensorContract())
      return operation.emitOpError()
             << "tensor contract must be preserved exactly across map composition";
    const auto producer_anchor =
        producer.getProvenance().getAs<mlir::StringAttr>("source_anchor");
    if (!anchor || !producer_anchor || anchor != producer_anchor)
      return operation.emitOpError()
             << "map composition must preserve the authenticated source anchor";
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult GemmOp::verify() {
  if (!isCanonicalSiteId(getSiteId()))
    return emitOpError() << "site_id must be canonical mc_<32 lowercase hex>";
  SemanticOriginKind origin_kind = SemanticOriginKind::ExplicitCall;
  if (mlir::failed(verifyOrigin(*this, origin_kind)))
    return mlir::failure();
  if (!getAccumulationType().isF32())
    return emitOpError() << "accumulation_type must be f32";

  const auto lhs_type = mlir::dyn_cast<mlir::RankedTensorType>(getLhs().getType());
  const auto rhs_type = mlir::dyn_cast<mlir::RankedTensorType>(getRhs().getType());
  const auto output_type =
      mlir::dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  const auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!lhs_type || !rhs_type || !output_type || !result_type)
    return emitOpError() << "all values must have ranked tensor types";
  if (output_type != result_type)
    return emitOpError()
           << "destination and result types must be identical";
  if (mlir::failed(verifyTensorSemantics(*this, getLhsSemantics(), lhs_type,
                                         "lhs", "read")) ||
      mlir::failed(verifyTensorSemantics(*this, getRhsSemantics(), rhs_type,
                                         "rhs", "read")) ||
      mlir::failed(verifyTensorSemantics(*this, getOutputSemantics(), output_type,
                                         "output", "write")))
    return mlir::failure();

  const auto lhs_shape = getLhsSemantics().getAs<mlir::ArrayAttr>("shape");
  const auto rhs_shape = getRhsSemantics().getAs<mlir::ArrayAttr>("shape");
  const auto output_shape =
      getOutputSemantics().getAs<mlir::ArrayAttr>("shape");
  if (lhs_shape[0] != output_shape[0] || lhs_shape[1] != rhs_shape[0] ||
      rhs_shape[1] != output_shape[1])
    return emitOpError()
           << "requires exact operation-local symbolic M/K/N relationships";

  if (mlir::failed(verifyStringArray(
          *this, getSemanticRequirements(),
          {"rank2_gemm", "f32_arithmetic", "host_addressable",
           "synchronous_execution"},
          "semantic_requirements")) ||
      mlir::failed(verifyAliasing(*this)))
    return mlir::failure();

  const auto effects = getEffects();
  if (mlir::failed(requireExactDictionary(
          *this, effects, {"read_write", "reads", "writes"}, "effects")) ||
      mlir::failed(verifyStringArray(*this,
                                     effects.getAs<mlir::ArrayAttr>("reads"),
                                     {"lhs", "rhs"}, "effects.reads")) ||
      mlir::failed(verifyStringArray(*this,
                                     effects.getAs<mlir::ArrayAttr>("writes"),
                                     {"output"}, "effects.writes")) ||
      mlir::failed(verifyStringArray(
          *this, effects.getAs<mlir::ArrayAttr>("read_write"), {},
          "effects.read_write")))
    return mlir::failure();
  if (getSynchronization() != "synchronous")
    return emitOpError() << "synchronization must be synchronous";

  if (mlir::failed(verifyPolicy(*this, origin_kind)) ||
      mlir::failed(verifyNumerical(*this, origin_kind)) ||
      mlir::failed(verifyProvenance(*this, origin_kind)))
    return mlir::failure();
  return mlir::success();
}

void GemmOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(0));
  effects.emplace_back(mlir::MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(1));
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getOperation()->getOpOperand(2));
}

mlir::LogicalResult MapOp::verify() {
  const auto input_type =
      mlir::dyn_cast<mlir::RankedTensorType>(getInput().getType());
  const auto result_type =
      mlir::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!input_type || !result_type || input_type != result_type)
    return emitOpError()
           << "input and functional result must have identical ranked tensor types";
  if (mlir::failed(verifyMapTensorContract(*this, input_type)) ||
      mlir::failed(verifyDomain(*this, input_type)) ||
      mlir::failed(verifyMapEffects(*this)) ||
      mlir::failed(verifyMapNumerical(*this)) ||
      mlir::failed(
          verifySemanticProvenance(*this, "semantic_composition")) ||
      mlir::failed(verifyMapInputContractPropagation(*this)))
    return mlir::failure();

  if (!getBody().hasOneBlock())
    return emitOpError() << "body must contain exactly one block";
  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != 1 || !block.getArgument(0).getType().isF32())
    return emitOpError() << "body must accept exactly one scalar f32 argument";
  if (block.empty())
    return emitOpError() << "body must end in mdsl.yield";

  mlir::Value current = block.getArgument(0);
  const auto map_anchor =
      getProvenance().getAs<mlir::StringAttr>("source_anchor");
  bool saw_scalar_operation = false;
  for (mlir::Operation &nested : block.without_terminator()) {
    auto sin = mlir::dyn_cast<SinOp>(&nested);
    if (!sin)
      return emitOpError()
             << "body permits only pure mdsl.sin operations before mdsl.yield";
    if (sin.getInput() != current)
      return emitOpError()
             << "body must form one linear scalar SSA chain without dead computations";
    const auto sin_anchor =
        sin.getProvenance().getAs<mlir::StringAttr>("source_anchor");
    if (!map_anchor || !sin_anchor || map_anchor != sin_anchor)
      return emitOpError()
             << "body scalar provenance must preserve the map source anchor";
    current = sin.getResult();
    saw_scalar_operation = true;
  }
  auto yield = mlir::dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getValue() != current || !saw_scalar_operation)
    return emitOpError()
           << "body must yield the result of a nonempty linear mdsl.sin chain";
  return mlir::success();
}

mlir::LogicalResult SinOp::verify() {
  if (!getInput().getType().isF32() || !getResult().getType().isF32())
    return emitOpError() << "input and result must be scalar f32";
  if (mlir::failed(verifySinNumerical(*this)) ||
      mlir::failed(verifySemanticProvenance(*this, "source_expression")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult YieldOp::verify() {
  auto map = mlir::dyn_cast<MapOp>((*this)->getParentOp());
  if (!map || (*this)->getBlock() != &map.getBody().front())
    return emitOpError() << "must terminate the sole mdsl.map body block";
  if (!getValue().getType().isF32())
    return emitOpError() << "value must be scalar f32";
  return mlir::success();
}

} // namespace matcore::mdslc::mlir_dialect

#define GET_OP_CLASSES
#include "MatcoreOps.cpp.inc"
