#include "MatcoreV1Bridge.h"

#include "MatcoreDialect.h"
#include "MatcoreGemmSemanticBuilder.h"
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
         numerical.infinity == InfinitySemantics::IeeeNoNoInfsAssumption &&
         numerical.signed_zero == SignedZeroSemantics::Relaxed &&
         numerical.rounding == RoundingSemantics::NearestTiesEven &&
         numerical.trapping_exceptions ==
             TrappingExceptionSemantics::Unsupported &&
         numerical.exception_status ==
             ExceptionStatusSemantics::IncomingNotPreservedPostCallUnspecified &&
         numerical.subnormals ==
             SubnormalSemantics::IeeeGradualFtzDazForbidden &&
         numerical.approximate_math == Permission::Forbidden &&
         numerical.inplace == Permission::Forbidden &&
         context.execution_intent == ExecutionIntent::Generic;
}

bool fitsSigned64(std::uint64_t value) {
  return value <= static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max());
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
       builder.getNamedAttr("kind", builder.getStringAttr("explicit_call")),
       builder.getNamedAttr("line",
                            builder.getI64IntegerAttr(operation.source.line)),
       builder.getNamedAttr(
           "offset", builder.getI64IntegerAttr(
                         static_cast<std::int64_t>(operation.source.offset))),
       builder.getNamedAttr("version", builder.getI32IntegerAttr(1))});
}

mlir::DictionaryAttr numericalAttribute(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("accumulation_dtype", builder.getStringAttr("f32")),
       builder.getNamedAttr("approximate_math", builder.getBoolAttr(false)),
       builder.getNamedAttr("contraction", builder.getStringAttr("allowed")),
       builder.getNamedAttr("derivation",
                            builder.getStringAttr("explicit_edsl_contract")),
       builder.getNamedAttr(
           "exception_status",
           builder.getStringAttr("incoming_not_preserved_postcall_unspecified")),
       builder.getNamedAttr(
           "infinity",
           builder.getStringAttr("ieee_no_no_infs_assumption")),
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
  mlir::DictionaryAttr provenance =
      provenanceAttribute(builder, operation, error);
  if (!provenance)
    return mlir::failure();

  const auto source_location = mlir::FileLineColLoc::get(
      builder.getContext(), operation.source.file, operation.source.line,
      operation.source.column);
  GemmSemanticSiteV1 site{
      .site_id = operation.site_id,
      .ordinal = ordinal,
      .lhs = operation.operands[0],
      .rhs = operation.operands[1],
      .output = operation.output,
      .accumulation_dtype = operation.accumulation_dtype,
      .requirements = operation.requirements,
      .alias_requirements = operation.alias_requirements,
      .effects = operation.effects,
      .target = std::string(ir::v1::toString(operation.policy.target)),
      .fallback = std::string(ir::v1::toString(operation.policy.fallback)),
      .origin = builder.getDictionaryAttr(
          {builder.getNamedAttr(
               "canonical_callee",
               builder.getStringAttr(operation.canonical_callee)),
           builder.getNamedAttr("kind",
                                builder.getStringAttr("explicit_call")),
           builder.getNamedAttr("version", builder.getI32IntegerAttr(1))}),
      .numerical = numericalAttribute(builder),
      .provenance = provenance,
      .source_location = source_location,
  };
  return mlir::success(
      appendGemmSemanticSiteV1(module, builder, site, error));
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
  context.numerical.infinity = InfinitySemantics::IeeeNoNoInfsAssumption;
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
  context.execution_intent = ExecutionIntent::Generic;
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
  (*result.module)->setAttr("mdsl.execution_intent",
                            builder.getStringAttr("generic"));
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
  if (!verifyMatcoreV1BridgeModule(*result.module, result.error)) {
    result.module = nullptr;
    return result;
  }
  return result;
}

bool verifyMatcoreV1BridgeModule(mlir::ModuleOp module, std::string &error) {
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
      !requireModuleString(module, "mdsl.execution_intent", "generic", error) ||
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
      !capture_version.getType().isSignlessInteger(32) ||
      capture_version.getInt() != ir::v1::kMatcoreIrVersion ||
      !semantic_version ||
      !semantic_version.getType().isSignlessInteger(32) ||
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
        !captured_ordinal.getType().isSignlessInteger(64) ||
        captured_ordinal.getInt() != static_cast<std::int64_t>(ordinal) ||
        function.getName() !=
            (llvm::Twine("__matcore_semantic_") + site.getValue()).str() ||
        !function.isPublic()) {
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
    for (mlir::BlockArgument argument : block.getArguments()) {
      if (argument.getLoc() != function.getLoc()) {
        error = "Matcore semantic function arguments must retain the exact "
                "authenticated source location";
        return false;
      }
    }
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
    const auto origin_kind =
        gemm.getOrigin().getAs<mlir::StringAttr>("kind");
    const auto provenance_kind =
        gemm.getProvenance().getAs<mlir::StringAttr>("kind");
    const auto provenance_file =
        gemm.getProvenance().getAs<mlir::StringAttr>("file");
    const auto numerical_profile =
        gemm.getNumerical().getAs<mlir::StringAttr>("profile");
    if (!origin_kind || origin_kind.getValue() != "explicit_call" ||
        !provenance_kind || provenance_kind.getValue() != "explicit_call" ||
        !provenance_file || provenance_file.getValue() != source_file.getValue() ||
        !numerical_profile ||
        numerical_profile.getValue() != kExplicitGemmF32Profile) {
      error = "Matcore IR v1 bridge sites require authenticated explicit-call provenance from mdsl.source_file";
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
