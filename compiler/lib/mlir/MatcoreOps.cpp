#include "MatcoreOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cctype>
#include <cstdint>

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
    if (!value || value.getInt() <= 0 || mlir::ShapedType::isDynamic(type_dimension) ||
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
    if (!value || value.getInt() <= 0)
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
      unit_value.getInt() != 1)
    return operation.emitOpError()
           << role << " row-major minor stride must be static one";

  const auto alignment = semantics.getAs<mlir::IntegerAttr>("alignment_bytes");
  if (!alignment || alignment.getInt() < 4 ||
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

mlir::LogicalResult verifyNumerical(GemmOp operation) {
  const auto numerical = operation.getNumerical();
  if (mlir::failed(requireExactDictionary(
          operation, numerical,
          {"accumulation_dtype", "approximate_math", "contraction",
           "exception_status", "inplace", "nan", "profile", "reassociation",
           "reduction_order", "rounding", "signed_zero", "subnormals",
           "trapping_exceptions"},
          "numerical")))
    return mlir::failure();
  return mlir::failure(
      mlir::failed(requireString(operation, numerical, "profile",
                                 "explicit-gemm-f32-v1", "numerical")) ||
      mlir::failed(requireString(operation, numerical, "accumulation_dtype",
                                 "f32", "numerical")) ||
      mlir::failed(requireString(operation, numerical, "reassociation",
                                 "within_k_reduction", "numerical")) ||
      mlir::failed(requireString(operation, numerical, "contraction", "allowed",
                                 "numerical")) ||
      mlir::failed(requireString(operation, numerical, "reduction_order",
                                 "implementation_defined_within_k",
                                 "numerical")) ||
      mlir::failed(requireString(
          operation, numerical, "nan",
          "preserve_classification_payload_order_unspecified", "numerical")) ||
      mlir::failed(requireString(operation, numerical, "signed_zero", "relaxed",
                                 "numerical")) ||
      mlir::failed(requireString(operation, numerical, "rounding",
                                 "nearest_ties_even", "numerical")) ||
      mlir::failed(requireString(
          operation, numerical, "exception_status",
          "incoming_not_preserved_postcall_unspecified", "numerical")) ||
      mlir::failed(requireString(
          operation, numerical, "subnormals",
          "ieee_gradual_ftz_daz_forbidden", "numerical")) ||
      mlir::failed(requireString(operation, numerical, "trapping_exceptions",
                                 "unsupported", "numerical")) ||
      mlir::failed(requireBoolean(operation, numerical, "approximate_math",
                                  false, "numerical")) ||
      mlir::failed(requireBoolean(operation, numerical, "inplace", false,
                                  "numerical")));
}

} // namespace

mlir::LogicalResult GemmOp::verify() {
  if (!isCanonicalSiteId(getSiteId()))
    return emitOpError() << "site_id must be canonical mc_<32 lowercase hex>";
  const auto origin = getOrigin();
  if (mlir::failed(requireExactDictionary(
          *this, origin, {"canonical_callee", "kind"}, "origin")) ||
      mlir::failed(requireString(*this, origin, "kind", "explicit_call",
                                 "origin")) ||
      mlir::failed(requireString(*this, origin, "canonical_callee",
                                 "matcore::mdsl::gemm", "origin")))
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

  const auto policy = getPolicy();
  if (mlir::failed(requireExactDictionary(*this, policy,
                                          {"fallback", "target"}, "policy")) ||
      mlir::failed(requireString(*this, policy, "target", "cpu", "policy")) ||
      mlir::failed(requireString(*this, policy, "fallback", "error", "policy")) ||
      mlir::failed(verifyNumerical(*this)))
    return mlir::failure();

  const auto provenance = getProvenance();
  if (mlir::failed(requireExactDictionary(
          *this, provenance,
          {"argument_ranges", "call_range", "column", "file", "line",
           "offset"},
          "provenance")))
    return mlir::failure();
  const auto file = provenance.getAs<mlir::StringAttr>("file");
  const auto line = provenance.getAs<mlir::IntegerAttr>("line");
  const auto column = provenance.getAs<mlir::IntegerAttr>("column");
  const auto offset = provenance.getAs<mlir::IntegerAttr>("offset");
  const auto call_range = provenance.getAs<mlir::DictionaryAttr>("call_range");
  const auto argument_ranges =
      provenance.getAs<mlir::ArrayAttr>("argument_ranges");
  if (!file || file.getValue().empty() || !line || line.getInt() <= 0 ||
      !column || column.getInt() <= 0 || !offset || offset.getInt() < 0 ||
      !call_range || !argument_ranges ||
      (argument_ranges.size() != 3 && argument_ranges.size() != 4))
    return emitOpError() << "provenance fields are incomplete";
  if (mlir::failed(requireExactDictionary(*this, call_range, {"begin", "end"},
                                          "provenance.call_range")))
    return mlir::failure();
  const auto begin = call_range.getAs<mlir::IntegerAttr>("begin");
  const auto end = call_range.getAs<mlir::IntegerAttr>("end");
  if (!begin || !end || begin.getInt() != offset.getInt() ||
      end.getInt() <= begin.getInt())
    return emitOpError()
           << "provenance call range must be nonempty and begin at offset";
  int64_t previous_end = begin.getInt();
  for (mlir::Attribute encoded_range : argument_ranges) {
    const auto range = mlir::dyn_cast<mlir::DictionaryAttr>(encoded_range);
    if (mlir::failed(requireExactDictionary(*this, range, {"begin", "end"},
                                            "provenance.argument_range")))
      return mlir::failure();
    const auto argument_begin = range.getAs<mlir::IntegerAttr>("begin");
    const auto argument_end = range.getAs<mlir::IntegerAttr>("end");
    if (!argument_begin || !argument_end ||
        argument_begin.getInt() < previous_end ||
        argument_end.getInt() <= argument_begin.getInt() ||
        argument_end.getInt() > end.getInt())
      return emitOpError()
             << "provenance argument ranges must be ordered and inside the call";
    previous_end = argument_end.getInt();
  }
  const auto location = mlir::dyn_cast<mlir::FileLineColLoc>(getLoc());
  if (!location || location.getFilename() != file.getValue() ||
      location.getLine() != static_cast<unsigned>(line.getInt()) ||
      location.getColumn() != static_cast<unsigned>(column.getInt()))
    return emitOpError()
           << "MLIR location must match the preserved source file/line/column";
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

} // namespace matcore::mdslc::mlir_dialect

#define GET_OP_CLASSES
#include "MatcoreOps.cpp.inc"
