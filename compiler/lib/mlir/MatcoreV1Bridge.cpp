#include "MatcoreV1Bridge.h"

#include "MatcoreDialect.h"
#include "MatcoreOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {

constexpr llvm::StringLiteral kCaptureSchema = "matcore-ir-v1";
constexpr llvm::StringLiteral kBridgeSchema = "matcore-mlir-semantic-v1";

bool exactExplicitGemmF32Context(const BridgeContext &context) {
  const NumericalSemantics &numerical = context.numerical;
  return context.numerical_profile == kExplicitGemmF32Profile &&
         numerical.accumulation_dtype == ir::v1::DType::F32 &&
         numerical.reassociation ==
             ReassociationSemantics::WithinReduction &&
         numerical.contraction == ContractionSemantics::Allowed &&
         numerical.reduction_order ==
             ReductionOrderSemantics::ImplementationDefinedWithinK &&
         numerical.nan ==
             NaNSemantics::PreserveClassificationPayloadOrderUnspecified &&
         numerical.signed_zero == SignedZeroSemantics::Relaxed &&
         numerical.rounding == RoundingSemantics::NearestTiesEven &&
         numerical.trapping_exceptions ==
             TrappingExceptionSemantics::Unsupported &&
         numerical.exception_status ==
             ExceptionStatusSemantics::IncomingNotPreservedPostCallUnspecified &&
         numerical.subnormals ==
             SubnormalSemantics::IeeeGradualFtzDazForbidden &&
         numerical.approximate_math == Permission::Forbidden &&
         numerical.inplace == Permission::Forbidden;
}

bool fitsSigned64(std::uint64_t value) {
  return value <= static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max());
}

mlir::DictionaryAttr scalarAttribute(mlir::Builder &builder,
                                     const ir::v1::ScalarExpr &value,
                                     std::string &error) {
  if (value.kind == ir::v1::ScalarExpr::Kind::Static) {
    if (!fitsSigned64(value.value)) {
      error = "Matcore IR v1 scalar exceeds MLIR signed dimension range";
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
      error = "Matcore IR v1 tensor dimension exceeds MLIR signed range";
      return {};
    }
    dimensions.push_back(static_cast<std::int64_t>(dimension.value));
  }
  mlir::Type element = elementType(builder, source.element_dtype);
  if (!element) {
    error = "Matcore IR v1 tensor has no MLIR element type mapping";
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
       builder.getNamedAttr("layout",
                            builder.getStringAttr(ir::v1::toString(value.type.layout))),
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

mlir::DictionaryAttr rangeAttribute(mlir::Builder &builder,
                                    const ir::SourceRange &range,
                                    std::string &error) {
  if (!fitsSigned64(range.begin) || !fitsSigned64(range.end)) {
    error = "Matcore IR v1 source range exceeds MLIR signed integer range";
    return {};
  }
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "begin", builder.getI64IntegerAttr(
                        static_cast<std::int64_t>(range.begin))),
       builder.getNamedAttr(
           "end", builder.getI64IntegerAttr(
                      static_cast<std::int64_t>(range.end)))});
}

mlir::DictionaryAttr provenanceAttribute(mlir::Builder &builder,
                                         const ir::v1::Operation &operation,
                                         std::string &error) {
  if (!fitsSigned64(operation.source.offset)) {
    error = "Matcore IR v1 source offset exceeds MLIR signed integer range";
    return {};
  }
  mlir::DictionaryAttr call_range =
      rangeAttribute(builder, operation.call_range, error);
  if (!call_range)
    return {};
  llvm::SmallVector<mlir::Attribute> argument_ranges;
  argument_ranges.reserve(operation.argument_ranges.size());
  for (const ir::SourceRange &range : operation.argument_ranges) {
    mlir::DictionaryAttr encoded = rangeAttribute(builder, range, error);
    if (!encoded)
      return {};
    argument_ranges.push_back(encoded);
  }
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("argument_ranges",
                            builder.getArrayAttr(argument_ranges)),
       builder.getNamedAttr("call_range", call_range),
       builder.getNamedAttr("column",
                            builder.getI64IntegerAttr(operation.source.column)),
       builder.getNamedAttr("file",
                            builder.getStringAttr(operation.source.file)),
       builder.getNamedAttr("line",
                            builder.getI64IntegerAttr(operation.source.line)),
       builder.getNamedAttr(
           "offset", builder.getI64IntegerAttr(
                         static_cast<std::int64_t>(operation.source.offset)))});
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

mlir::DictionaryAttr numericalAttribute(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("accumulation_dtype", builder.getStringAttr("f32")),
       builder.getNamedAttr("approximate_math", builder.getBoolAttr(false)),
       builder.getNamedAttr("contraction", builder.getStringAttr("allowed")),
       builder.getNamedAttr(
           "exception_status",
           builder.getStringAttr("incoming_not_preserved_postcall_unspecified")),
       builder.getNamedAttr("inplace", builder.getBoolAttr(false)),
       builder.getNamedAttr(
           "nan",
           builder.getStringAttr(
               "preserve_classification_payload_order_unspecified")),
       builder.getNamedAttr("profile",
                            builder.getStringAttr(kExplicitGemmF32Profile)),
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

mlir::LogicalResult addOperationFunction(
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const ir::v1::Operation &operation, std::size_t ordinal,
    std::string &error) {
  if (operation.operands.size() != 2) {
    error = "verified Matcore IR v1 GEMM must have two operands";
    return mlir::failure();
  }
  mlir::RankedTensorType lhs_type =
      tensorType(builder, operation.operands[0].type, error);
  mlir::RankedTensorType rhs_type =
      tensorType(builder, operation.operands[1].type, error);
  mlir::RankedTensorType output_type =
      tensorType(builder, operation.output.type, error);
  if (!lhs_type || !rhs_type || !output_type)
    return mlir::failure();

  mlir::DictionaryAttr lhs_semantics =
      tensorSemantics(builder, operation.operands[0], error);
  mlir::DictionaryAttr rhs_semantics =
      tensorSemantics(builder, operation.operands[1], error);
  mlir::DictionaryAttr output_semantics =
      tensorSemantics(builder, operation.output, error);
  mlir::DictionaryAttr provenance =
      provenanceAttribute(builder, operation, error);
  if (!lhs_semantics || !rhs_semantics || !output_semantics || !provenance)
    return mlir::failure();

  const auto source_location = mlir::FileLineColLoc::get(
      builder.getContext(), operation.source.file, operation.source.line,
      operation.source.column);
  const std::string function_name =
      "__matcore_semantic_" + operation.site_id;
  const auto function_type = builder.getFunctionType(
      {lhs_type, rhs_type, output_type}, {output_type});
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(source_location, function_name,
                                              function_type);
  function.setPrivate();
  function->setAttr("mdsl.capture_ordinal",
                    builder.getI64IntegerAttr(static_cast<std::int64_t>(ordinal)));
  function->setAttr("mdsl.site_id", builder.getStringAttr(operation.site_id));
  module.push_back(function);

  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::OperationState state(source_location,
                             mlir_dialect::GemmOp::getOperationName());
  state.addOperands({entry->getArgument(0), entry->getArgument(1),
                     entry->getArgument(2)});
  state.addTypes(output_type);
  state.addAttributes(
      {builder.getNamedAttr("site_id", builder.getStringAttr(operation.site_id)),
       builder.getNamedAttr(
           "origin",
           builder.getDictionaryAttr(
               {builder.getNamedAttr(
                    "canonical_callee",
                    builder.getStringAttr(operation.canonical_callee)),
                builder.getNamedAttr("kind",
                                     builder.getStringAttr("explicit_call"))})),
       builder.getNamedAttr(
           "accumulation_type",
           mlir::TypeAttr::get(elementType(builder, operation.accumulation_dtype))),
       builder.getNamedAttr("lhs_semantics", lhs_semantics),
       builder.getNamedAttr("rhs_semantics", rhs_semantics),
       builder.getNamedAttr("output_semantics", output_semantics),
       builder.getNamedAttr("semantic_requirements",
                            requirementAttributes(builder,
                                                  operation.requirements)),
       builder.getNamedAttr("aliasing",
                            aliasAttributes(builder,
                                            operation.alias_requirements)),
       builder.getNamedAttr("effects",
                            effectsAttribute(builder, operation.effects)),
       builder.getNamedAttr(
           "synchronization",
           builder.getStringAttr(ir::v1::toString(
               operation.effects.synchronization))),
       builder.getNamedAttr(
           "policy",
           builder.getDictionaryAttr(
               {builder.getNamedAttr(
                    "fallback",
                    builder.getStringAttr(
                        ir::v1::toString(operation.policy.fallback))),
                builder.getNamedAttr(
                    "target",
                    builder.getStringAttr(
                        ir::v1::toString(operation.policy.target)))})),
       builder.getNamedAttr("numerical", numericalAttribute(builder)),
       builder.getNamedAttr("provenance", provenance)});
  mlir::Operation *raw_operation = builder.create(state);
  auto gemm = mlir::cast<mlir_dialect::GemmOp>(raw_operation);
  builder.create<mlir::func::ReturnOp>(source_location, gemm.getResult());
  return mlir::success();
}

bool requireModuleString(mlir::ModuleOp module, llvm::StringRef name,
                         llvm::StringRef expected, std::string &error) {
  const auto value = module->getAttrOfType<mlir::StringAttr>(name);
  if (!value || value.getValue() != expected) {
    error = "Matcore semantic module field '" + name.str() +
            "' is missing or invalid";
    return false;
  }
  return true;
}

} // namespace

BridgeContext explicitGemmF32V1BridgeContext() {
  BridgeContext context;
  context.numerical_profile = kExplicitGemmF32Profile;
  context.numerical.accumulation_dtype = ir::v1::DType::F32;
  context.numerical.reassociation = ReassociationSemantics::WithinReduction;
  context.numerical.contraction = ContractionSemantics::Allowed;
  context.numerical.reduction_order =
      ReductionOrderSemantics::ImplementationDefinedWithinK;
  context.numerical.nan =
      NaNSemantics::PreserveClassificationPayloadOrderUnspecified;
  context.numerical.signed_zero = SignedZeroSemantics::Relaxed;
  context.numerical.rounding = RoundingSemantics::NearestTiesEven;
  context.numerical.trapping_exceptions =
      TrappingExceptionSemantics::Unsupported;
  context.numerical.exception_status =
      ExceptionStatusSemantics::IncomingNotPreservedPostCallUnspecified;
  context.numerical.subnormals =
      SubnormalSemantics::IeeeGradualFtzDazForbidden;
  context.numerical.approximate_math = Permission::Forbidden;
  context.numerical.inplace = Permission::Forbidden;
  return context;
}

void registerMatcoreSemanticDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir_dialect::MatcoreDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
}

BridgeResult bridgeV1ToMatcoreMlir(const ir::v1::Module &source,
                                   mlir::MLIRContext &context,
                                   const BridgeContext &bridge_context) {
  BridgeResult result;
  if (!exactExplicitGemmF32Context(bridge_context)) {
    result.error =
        "bridge requires the complete explicit-gemm-f32-v1 numerical profile";
    return result;
  }
  if (!ir::v1::verify(source, result.error)) {
    result.error = "Matcore IR v1 verification failed before MLIR bridge: " +
                   result.error;
    return result;
  }
  for (const ir::v1::Operation &operation : source.operations) {
    if (operation.output.type.element_dtype != ir::v1::DType::F32 ||
        operation.operands[0].type.element_dtype != ir::v1::DType::F32 ||
        operation.operands[1].type.element_dtype != ir::v1::DType::F32 ||
        operation.accumulation_dtype != ir::v1::DType::F32) {
      result.error = "explicit-gemm-f32-v1 cannot authorize a non-F32 capture";
      return result;
    }
  }

  registerMatcoreSemanticDialects(context);
  mlir::OpBuilder builder(&context);
  result.module = mlir::ModuleOp::create(builder.getUnknownLoc());
  (*result.module)->setAttr("mdsl.bridge_schema",
                            builder.getStringAttr(kBridgeSchema));
  (*result.module)->setAttr("mdsl.capture_schema",
                            builder.getStringAttr(kCaptureSchema));
  (*result.module)->setAttr(
      "mdsl.capture_version",
      builder.getI32IntegerAttr(ir::v1::kMatcoreIrVersion));
  (*result.module)->setAttr("mdsl.numerical_profile",
                            builder.getStringAttr(kExplicitGemmF32Profile));
  (*result.module)->setAttr("mdsl.producer",
                            builder.getStringAttr(source.producer));
  (*result.module)->setAttr(
      "mdsl.semantic_version",
      builder.getI32IntegerAttr(kMatcoreSemanticModuleVersion));
  (*result.module)->setAttr("mdsl.source_file",
                            builder.getStringAttr(source.source_file));
  (*result.module)->setAttr("mdsl.translation_unit",
                            builder.getStringAttr(source.translation_unit));

  for (std::size_t index = 0; index < source.operations.size(); ++index) {
    if (mlir::failed(addOperationFunction(*result.module, builder,
                                          source.operations[index], index,
                                          result.error))) {
      result.module = nullptr;
      return result;
    }
  }
  if (!verifyMatcoreSemanticModule(*result.module, result.error)) {
    result.module = nullptr;
    return result;
  }
  return result;
}

bool verifyMatcoreSemanticModule(mlir::ModuleOp module, std::string &error) {
  error.clear();
  if (!module) {
    error = "Matcore semantic module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(module))) {
    error = "Matcore semantic module failed MLIR/dialect verification";
    return false;
  }
  if (!requireModuleString(module, "mdsl.bridge_schema", kBridgeSchema, error) ||
      !requireModuleString(module, "mdsl.capture_schema", kCaptureSchema,
                           error) ||
      !requireModuleString(module, "mdsl.numerical_profile",
                           kExplicitGemmF32Profile, error))
    return false;
  const auto capture_version =
      module->getAttrOfType<mlir::IntegerAttr>("mdsl.capture_version");
  const auto semantic_version =
      module->getAttrOfType<mlir::IntegerAttr>("mdsl.semantic_version");
  const auto producer = module->getAttrOfType<mlir::StringAttr>("mdsl.producer");
  const auto source_file =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_file");
  const auto translation_unit =
      module->getAttrOfType<mlir::StringAttr>("mdsl.translation_unit");
  if (!capture_version ||
      capture_version.getInt() != ir::v1::kMatcoreIrVersion ||
      !semantic_version ||
      semantic_version.getInt() != kMatcoreSemanticModuleVersion || !producer ||
      (producer.getValue() != "clang-libtooling-v1" &&
       producer.getValue() != "clang-ast-json-bootstrap-v0") || !source_file ||
      !source_file.getValue().ends_with(".mdsl") || !translation_unit ||
      translation_unit.getValue().empty()) {
    error = "Matcore semantic module capture metadata is incomplete or invalid";
    return false;
  }

  llvm::StringSet<> sites;
  std::size_t ordinal = 0;
  for (mlir::Operation &operation : module.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function) {
      error = "Matcore semantic module may contain only per-site func.func ops";
      return false;
    }
    const auto site = function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
    const auto captured_ordinal =
        function->getAttrOfType<mlir::IntegerAttr>("mdsl.capture_ordinal");
    if (!site || !sites.insert(site.getValue()).second || !captured_ordinal ||
        captured_ordinal.getInt() != static_cast<std::int64_t>(ordinal) ||
        function.getName() !=
            (llvm::Twine("__matcore_semantic_") + site.getValue()).str() ||
        !function.isPrivate()) {
      error = "Matcore semantic functions require unique ordered site identity";
      return false;
    }
    if (!llvm::hasSingleElement(function.getBody()) ||
        function.getBody().front().getNumArguments() != 3 ||
        std::distance(function.getBody().front().begin(),
                      function.getBody().front().end()) != 2) {
      error = "each Matcore semantic site must have one gemm and one return";
      return false;
    }
    mlir::Block &block = function.getBody().front();
    auto gemm = mlir::dyn_cast<mlir_dialect::GemmOp>(block.front());
    auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
    if (!gemm || !return_op || gemm.getSiteId() != site.getValue() ||
        gemm.getLhs() != block.getArgument(0) ||
        gemm.getRhs() != block.getArgument(1) ||
        gemm.getOutput() != block.getArgument(2) ||
        return_op.getNumOperands() != 1 ||
        return_op.getOperand(0) != gemm.getResult()) {
      error = "Matcore semantic site must return the destination-tied GEMM SSA result";
      return false;
    }
    ++ordinal;
  }
  return true;
}

std::string serializeDeterministicMlir(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
  module.print(stream, flags);
  stream.flush();
  while (!text.empty() && text.back() == '\n')
    text.pop_back();
  text.push_back('\n');
  return text;
}

} // namespace matcore::mdslc::mlir_bridge
