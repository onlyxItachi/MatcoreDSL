#include "MatcoreGemmSemanticBuilder.h"

#include "MatcoreOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>

namespace matcore::mdslc::mlir_bridge {
namespace {

bool fitsSigned64(std::uint64_t value) {
  return value <= static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max());
}

mlir::DictionaryAttr scalarAttribute(mlir::Builder &builder,
                                     const ir::v1::ScalarExpr &value,
                                     std::string &error) {
  if (value.kind == ir::v1::ScalarExpr::Kind::Static) {
    if (!fitsSigned64(value.value)) {
      error = "semantic GEMM scalar exceeds MLIR signed dimension range";
      return {};
    }
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("kind", builder.getStringAttr("static")),
         builder.getNamedAttr(
             "value", builder.getI64IntegerAttr(
                          static_cast<std::int64_t>(value.value)))});
  }
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("kind", builder.getStringAttr("dynamic")),
       builder.getNamedAttr("symbol", builder.getStringAttr(value.symbol))});
}

mlir::ArrayAttr scalarArrayAttribute(
    mlir::Builder &builder, const std::vector<ir::v1::ScalarExpr> &values,
    std::string &error) {
  llvm::SmallVector<mlir::Attribute> encoded;
  encoded.reserve(values.size());
  for (const ir::v1::ScalarExpr &value : values) {
    mlir::DictionaryAttr attribute = scalarAttribute(builder, value, error);
    if (!attribute)
      return {};
    encoded.push_back(attribute);
  }
  return builder.getArrayAttr(encoded);
}

mlir::Type elementType(mlir::Builder &builder, ir::v1::DType dtype) {
  switch (dtype) {
  case ir::v1::DType::F16:
    return builder.getF16Type();
  case ir::v1::DType::BF16:
    return builder.getBF16Type();
  case ir::v1::DType::F32:
    return builder.getF32Type();
  case ir::v1::DType::F64:
    return builder.getF64Type();
  case ir::v1::DType::I8:
    return builder.getI8Type();
  case ir::v1::DType::I32:
    return builder.getI32Type();
  }
  return {};
}

mlir::RankedTensorType tensorType(mlir::Builder &builder,
                                  const ir::v1::TensorType &source,
                                  std::string &error) {
  llvm::SmallVector<std::int64_t> dimensions;
  dimensions.reserve(source.shape.size());
  for (const ir::v1::ScalarExpr &dimension : source.shape) {
    if (dimension.kind == ir::v1::ScalarExpr::Kind::Dynamic) {
      dimensions.push_back(mlir::ShapedType::kDynamic);
      continue;
    }
    if (!fitsSigned64(dimension.value)) {
      error = "semantic GEMM tensor dimension exceeds MLIR signed range";
      return {};
    }
    dimensions.push_back(static_cast<std::int64_t>(dimension.value));
  }
  mlir::Type element = elementType(builder, source.element_dtype);
  if (!element) {
    error = "semantic GEMM tensor has no MLIR element type mapping";
    return {};
  }
  return mlir::RankedTensorType::get(dimensions, element);
}

mlir::DictionaryAttr tensorSemantics(mlir::Builder &builder,
                                     const ir::v1::TensorValue &value,
                                     std::string &error) {
  mlir::ArrayAttr shape = scalarArrayAttribute(builder, value.type.shape, error);
  if (!shape)
    return {};
  mlir::ArrayAttr strides =
      scalarArrayAttribute(builder, value.type.strides, error);
  if (!strides)
    return {};
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "alignment_bytes",
           builder.getI64IntegerAttr(value.type.required_alignment_bytes)),
       builder.getNamedAttr("alignment_contract",
                            builder.getStringAttr("required_precondition")),
       builder.getNamedAttr(
           "layout", builder.getStringAttr(ir::v1::toString(value.type.layout))),
       builder.getNamedAttr(
           "memory_space",
           builder.getStringAttr(ir::v1::toString(value.type.memory_space))),
       builder.getNamedAttr(
           "mutability",
           builder.getStringAttr(ir::v1::toString(value.mutability))),
       builder.getNamedAttr("role",
                            builder.getStringAttr(ir::v1::toString(value.id))),
       builder.getNamedAttr("shape", shape),
       builder.getNamedAttr("source_expression",
                            builder.getStringAttr(value.source_expression)),
       builder.getNamedAttr("strides", strides)});
}

mlir::ArrayAttr stringArray(mlir::Builder &builder,
                            llvm::ArrayRef<llvm::StringRef> values) {
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(values.size());
  for (llvm::StringRef value : values)
    attributes.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attributes);
}

mlir::ArrayAttr requirementAttributes(
    mlir::Builder &builder,
    const std::vector<ir::v1::SemanticRequirement> &requirements) {
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(requirements.size());
  for (ir::v1::SemanticRequirement requirement : requirements)
    attributes.push_back(builder.getStringAttr(ir::v1::toString(requirement)));
  return builder.getArrayAttr(attributes);
}

mlir::ArrayAttr aliasAttributes(
    mlir::Builder &builder,
    const std::vector<ir::v1::AliasRequirement> &requirements) {
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(requirements.size());
  for (const ir::v1::AliasRequirement &requirement : requirements) {
    attributes.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("contract",
                              builder.getStringAttr("required_precondition")),
         builder.getNamedAttr(
             "first", builder.getStringAttr(ir::v1::toString(requirement.first))),
         builder.getNamedAttr(
             "relation",
             builder.getStringAttr(ir::v1::toString(requirement.relation))),
         builder.getNamedAttr(
             "second",
             builder.getStringAttr(ir::v1::toString(requirement.second)))}));
  }
  return builder.getArrayAttr(attributes);
}

mlir::DictionaryAttr effectsAttribute(mlir::Builder &builder,
                                      const ir::v1::Effects &effects) {
  llvm::SmallVector<llvm::StringRef> reads;
  llvm::SmallVector<llvm::StringRef> writes;
  for (ir::v1::ValueId value : effects.reads)
    reads.push_back(ir::v1::toString(value));
  for (ir::v1::ValueId value : effects.writes)
    writes.push_back(ir::v1::toString(value));
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("read_write", builder.getArrayAttr({})),
       builder.getNamedAttr("reads", stringArray(builder, reads)),
       builder.getNamedAttr("writes", stringArray(builder, writes))});
}

} // namespace

bool appendGemmSemanticSiteV1(mlir::ModuleOp module, mlir::OpBuilder &builder,
                              const GemmSemanticSiteV1 &site,
                              std::string &error) {
  if (!module || !site.origin || !site.numerical || !site.provenance ||
      !site.source_location) {
    error = "semantic GEMM site construction context is incomplete";
    return false;
  }
  mlir::RankedTensorType lhs_type = tensorType(builder, site.lhs.type, error);
  mlir::RankedTensorType rhs_type = tensorType(builder, site.rhs.type, error);
  mlir::RankedTensorType output_type =
      tensorType(builder, site.output.type, error);
  if (!lhs_type || !rhs_type || !output_type)
    return false;

  mlir::DictionaryAttr lhs_semantics =
      tensorSemantics(builder, site.lhs, error);
  mlir::DictionaryAttr rhs_semantics =
      tensorSemantics(builder, site.rhs, error);
  mlir::DictionaryAttr output_semantics =
      tensorSemantics(builder, site.output, error);
  mlir::Type accumulation_type = elementType(builder, site.accumulation_dtype);
  if (!lhs_semantics || !rhs_semantics || !output_semantics ||
      !accumulation_type)
    return false;

  const std::string function_name = "__matcore_semantic_" + site.site_id;
  const auto function_type = builder.getFunctionType(
      {lhs_type, rhs_type, output_type}, {output_type});
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(site.source_location,
                                              function_name, function_type);
  function->setAttr(
      "mdsl.capture_ordinal",
      builder.getI64IntegerAttr(static_cast<std::int64_t>(site.ordinal)));
  function->setAttr("mdsl.site_id", builder.getStringAttr(site.site_id));
  module.push_back(function);

  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::OperationState state(site.source_location,
                             mlir_dialect::GemmOp::getOperationName());
  state.addOperands({entry->getArgument(0), entry->getArgument(1),
                     entry->getArgument(2)});
  state.addTypes(output_type);
  state.addAttributes(
      {builder.getNamedAttr("site_id", builder.getStringAttr(site.site_id)),
       builder.getNamedAttr("origin", site.origin),
       builder.getNamedAttr("accumulation_type",
                            mlir::TypeAttr::get(accumulation_type)),
       builder.getNamedAttr("lhs_semantics", lhs_semantics),
       builder.getNamedAttr("rhs_semantics", rhs_semantics),
       builder.getNamedAttr("output_semantics", output_semantics),
       builder.getNamedAttr("semantic_requirements",
                            requirementAttributes(builder, site.requirements)),
       builder.getNamedAttr("aliasing",
                            aliasAttributes(builder, site.alias_requirements)),
       builder.getNamedAttr("effects",
                            effectsAttribute(builder, site.effects)),
       builder.getNamedAttr(
           "synchronization",
           builder.getStringAttr(ir::v1::toString(
               site.effects.synchronization))),
       builder.getNamedAttr(
           "policy",
           builder.getDictionaryAttr(
               {builder.getNamedAttr("fallback",
                                     builder.getStringAttr(site.fallback)),
                builder.getNamedAttr("target",
                                     builder.getStringAttr(site.target))})),
       builder.getNamedAttr("numerical", site.numerical),
       builder.getNamedAttr("provenance", site.provenance)});
  mlir::Operation *raw_operation = builder.create(state);
  auto gemm = mlir::cast<mlir_dialect::GemmOp>(raw_operation);
  builder.create<mlir::func::ReturnOp>(site.source_location, gemm.getResult());
  return true;
}

} // namespace matcore::mdslc::mlir_bridge
