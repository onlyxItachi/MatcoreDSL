#include "MatcoreStructuredGemmVectorReadiness.h"

#include "MatcoreContractionModel.h"
#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreStructuredHandoffCertificate.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {

constexpr llvm::StringLiteral kVectorArtifactProducer =
    "matcore-structured-gemm-vector-readiness-v1";
constexpr llvm::StringLiteral kTransformOperation =
    "transform.structured.vectorize_children_and_apply_patterns";
constexpr llvm::StringLiteral kVectorizationScope =
    "whole_static_problem_inspection";
constexpr llvm::StringLiteral kDestinationEncoding =
    "zero_accumulator_full_transfer_write_original_output";
constexpr llvm::StringLiteral kOperationLocations =
    "source_on_zero_reads_return_unknown_on_generated_glue";

constexpr llvm::StringLiteral kTransformSchedule = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %transformed = transform.structured.vectorize_children_and_apply_patterns %functions : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
)mlir";

constexpr llvm::StringLiteral kVectorModuleFields[] = {
    "mdsl.analysis_only",
    "mdsl.capture_schema",
    "mdsl.capture_version",
    "mdsl.execution_authority",
    "mdsl.execution_intent",
    "mdsl.numerical_profile",
    "mdsl.producer",
    "mdsl.source_bridge_schema",
    "mdsl.source_file",
    "mdsl.source_producer",
    "mdsl.source_semantic_version",
    "mdsl.source_structured_producer",
    "mdsl.source_structured_site_count",
    "mdsl.source_structured_fingerprint",
    "mdsl.structured_handoff_schema",
    "mdsl.structured_handoff_version",
    "mdsl.translation_unit",
    "mdsl.vector_readiness_schema",
    "mdsl.vector_readiness_version",
};

constexpr llvm::StringLiteral kVectorFunctionFields[] = {
    "mdsl.capture_ordinal",
    "mdsl.semantic_contract",
    "mdsl.site_id",
    "mdsl.source_semantic_symbol",
    "mdsl.source_structured_fingerprint",
    "mdsl.source_structured_function_type",
    "mdsl.structured_handoff",
    "mdsl.vector_readiness",
};

constexpr llvm::StringLiteral kReadinessFields[] = {
    "authority",
    "destination_encoding",
    "numerical_permissions",
    "operation_locations",
    "semantic_contract",
    "source_operations",
    "transform",
    "unconsumed_requirements",
    "vectorization_scope",
    "version",
};

constexpr llvm::StringLiteral kUnconsumedRequirements[] = {
    "alias_preconditions",
    "alignment_preconditions",
    "effects",
    "layout_and_strides",
    "memory_space",
    "numerical_permissions",
    "provenance",
    "target_policy",
};

bool requireExactNames(mlir::DictionaryAttr dictionary,
                       llvm::ArrayRef<llvm::StringLiteral> expected,
                       llvm::StringRef context, std::string &error) {
  if (!dictionary || dictionary.size() != expected.size()) {
    error = context.str() + " must contain exactly " +
            std::to_string(expected.size()) + " fields (found " +
            std::to_string(dictionary ? dictionary.size() : 0) + ")";
    if (dictionary) {
      error += ":";
      for (mlir::NamedAttribute attribute : dictionary)
        error += " " + attribute.getName().strref().str();
    }
    return false;
  }
  llvm::StringSet<> names;
  for (llvm::StringRef name : expected)
    names.insert(name);
  for (mlir::NamedAttribute attribute : dictionary) {
    if (!names.contains(attribute.getName().strref())) {
      error = context.str() + " contains unexpected field '" +
              attribute.getName().strref().str() + "'";
      return false;
    }
  }
  for (llvm::StringRef name : expected) {
    if (!dictionary.get(name)) {
      error = context.str() + " is missing field '" + name.str() + "'";
      return false;
    }
  }
  return true;
}

bool isString(mlir::Attribute attribute, llvm::StringRef expected) {
  const auto value = mlir::dyn_cast_or_null<mlir::StringAttr>(attribute);
  return value && value.getValue() == expected;
}

bool isI32(mlir::Attribute attribute, std::int64_t expected) {
  const auto value = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute);
  return value && value.getType().isSignlessInteger(32) &&
         value.getInt() == expected;
}

bool isTrue(mlir::Attribute attribute) {
  const auto value = mlir::dyn_cast_or_null<mlir::BoolAttr>(attribute);
  return value && value.getValue();
}

bool isExactStringArray(mlir::Attribute attribute,
                        llvm::ArrayRef<llvm::StringLiteral> expected) {
  const auto values = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attribute);
  if (!values || values.size() != expected.size())
    return false;
  for (auto [value, required] : llvm::zip_equal(values, expected)) {
    if (!isString(value, required))
      return false;
  }
  return true;
}

bool hasExactModuleEnvelope(mlir::ModuleOp module, std::string &error) {
  if (!requireExactNames(module->getAttrDictionary(), kVectorModuleFields,
                         "vector-readiness module", error))
    return false;
  if (!isTrue(module->getAttr("mdsl.analysis_only")) ||
      !isString(module->getAttr("mdsl.execution_authority"),
                kStructuredGemmInspectionAuthorityV1) ||
      !isString(module->getAttr("mdsl.producer"), kVectorArtifactProducer) ||
      !isString(module->getAttr("mdsl.source_structured_producer"),
                kStructuredGemmHandoffSchemaV1) ||
      !isString(module->getAttr("mdsl.structured_handoff_schema"),
                kStructuredGemmHandoffSchemaV1) ||
      !isI32(module->getAttr("mdsl.structured_handoff_version"),
             kStructuredGemmHandoffVersionV1) ||
      !isString(module->getAttr("mdsl.vector_readiness_schema"),
                kStructuredGemmVectorReadinessSchemaV1) ||
      !isI32(module->getAttr("mdsl.vector_readiness_version"),
             kStructuredGemmVectorReadinessVersionV1)) {
    error = "vector-readiness module schema or inspection authority is invalid";
    return false;
  }
  return true;
}

mlir::DictionaryAttr
readinessAttribute(mlir::Builder &builder,
                   mlir::DictionaryAttr numerical_permissions) {
  llvm::SmallVector<mlir::Attribute> source_operations = {
      builder.getStringAttr("linalg.fill"),
      builder.getStringAttr("linalg.matmul"),
  };
  llvm::SmallVector<mlir::Attribute> unconsumed;
  for (llvm::StringRef name : kUnconsumedRequirements)
    unconsumed.push_back(builder.getStringAttr(name));
  return builder.getDictionaryAttr({
      builder.getNamedAttr(
          "authority",
          builder.getStringAttr(kStructuredGemmInspectionAuthorityV1)),
      builder.getNamedAttr("destination_encoding",
                           builder.getStringAttr(kDestinationEncoding)),
      builder.getNamedAttr("numerical_permissions", numerical_permissions),
      builder.getNamedAttr("operation_locations",
                           builder.getStringAttr(kOperationLocations)),
      builder.getNamedAttr("semantic_contract",
                           builder.getStringAttr("retained_exactly")),
      builder.getNamedAttr("source_operations",
                           builder.getArrayAttr(source_operations)),
      builder.getNamedAttr("transform",
                           builder.getStringAttr(kTransformOperation)),
      builder.getNamedAttr("unconsumed_requirements",
                           builder.getArrayAttr(unconsumed)),
      builder.getNamedAttr("vectorization_scope",
                           builder.getStringAttr(kVectorizationScope)),
      builder.getNamedAttr(
          "version",
          builder.getI32IntegerAttr(kStructuredGemmVectorReadinessVersionV1)),
  });
}

bool verifyReadinessAttribute(mlir::DictionaryAttr readiness,
                              mlir::DictionaryAttr semantic_contract,
                              std::string &error) {
  constexpr llvm::StringLiteral source_operations[] = {"linalg.fill",
                                                       "linalg.matmul"};
  if (!requireExactNames(readiness, kReadinessFields, "vector-readiness ledger",
                         error))
    return false;
  const auto numerical_permissions =
      semantic_contract
          ? semantic_contract.getAs<mlir::DictionaryAttr>("numerical")
          : mlir::DictionaryAttr{};
  if (!isString(readiness.get("authority"),
                kStructuredGemmInspectionAuthorityV1) ||
      !isString(readiness.get("destination_encoding"), kDestinationEncoding) ||
      !numerical_permissions ||
      readiness.getAs<mlir::DictionaryAttr>("numerical_permissions") !=
          numerical_permissions ||
      !isString(readiness.get("operation_locations"), kOperationLocations) ||
      !isString(readiness.get("semantic_contract"), "retained_exactly") ||
      !isExactStringArray(readiness.get("source_operations"),
                          source_operations) ||
      !isString(readiness.get("transform"), kTransformOperation) ||
      !isExactStringArray(readiness.get("unconsumed_requirements"),
                          kUnconsumedRequirements) ||
      !isString(readiness.get("vectorization_scope"), kVectorizationScope) ||
      !isI32(readiness.get("version"),
             kStructuredGemmVectorReadinessVersionV1)) {
    error = "vector-readiness ledger changed its exact transformation or "
            "semantic-consumption contract";
    return false;
  }
  return true;
}

bool getStaticGemmShape(mlir::func::FuncOp function,
                        std::array<std::int64_t, 3> &mnk, std::string &error) {
  if (function.getNumArguments() != 3 || function.getNumResults() != 1) {
    error = "vector readiness requires the three-argument, one-result GEMM "
            "function contract";
    return false;
  }
  const auto lhs =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getArgument(0).getType());
  const auto rhs =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getArgument(1).getType());
  const auto output =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getArgument(2).getType());
  const auto result =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getResultTypes().front());
  if (!lhs || !rhs || !output || !result || lhs.getRank() != 2 ||
      rhs.getRank() != 2 || output.getRank() != 2 || result != output ||
      !lhs.getElementType().isF32() || !rhs.getElementType().isF32() ||
      !output.getElementType().isF32()) {
    error = "vector readiness requires rank-2 f32 tensor GEMM types";
    return false;
  }
  if (!lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      !output.hasStaticShape()) {
    error = "target-independent regular vectorization cannot prove a vector "
            "shape for dynamic GEMM; tile/bound evidence is required first";
    return false;
  }
  const std::int64_t m = lhs.getShape()[0];
  const std::int64_t k = lhs.getShape()[1];
  const std::int64_t n = rhs.getShape()[1];
  if (m <= 0 || n <= 0 || k <= 0 || rhs.getShape()[0] != k ||
      output.getShape()[0] != m || output.getShape()[1] != n) {
    error = "static GEMM vector readiness requires positive consistent M/K/N";
    return false;
  }
  mnk = {m, n, k};
  return true;
}

bool isExactIndexZero(mlir::arith::ConstantOp constant) {
  const auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return constant.getType().isIndex() && value && value.getInt() == 0;
}

bool isExactPositiveZeroVector(mlir::arith::ConstantOp constant,
                               mlir::VectorType expected_type) {
  if (constant.getType() != expected_type)
    return false;
  const auto elements =
      mlir::dyn_cast<mlir::DenseFPElementsAttr>(constant.getValue());
  if (!elements || !elements.isSplat())
    return false;
  const llvm::APFloat value = elements.getSplatValue<llvm::APFloat>();
  return value.isZero() && !value.isNegative();
}

bool allTrue(mlir::ArrayAttr values, std::size_t expected_size) {
  if (!values || values.size() != expected_size)
    return false;
  return llvm::all_of(values, [](mlir::Attribute value) {
    const auto boolean = mlir::dyn_cast<mlir::BoolAttr>(value);
    return boolean && boolean.getValue();
  });
}

bool isZeroIndexRange(mlir::ValueRange indices, mlir::Value expected_zero) {
  return indices.size() == 2 && llvm::all_of(indices, [&](mlir::Value value) {
           return value == expected_zero;
         });
}

bool hasNoMask(mlir::Value value) { return !value; }

bool verifyVectorFunction(mlir::func::FuncOp function,
                          std::size_t expected_ordinal,
                          mlir::FunctionType source_structured_function_type,
                          std::string &error) {
  if (!requireExactNames(function->getDiscardableAttrDictionary(),
                         kVectorFunctionFields, "vector-readiness function",
                         error))
    return false;
  const auto ordinal =
      function->getAttrOfType<mlir::IntegerAttr>("mdsl.capture_ordinal");
  const auto readiness =
      function->getAttrOfType<mlir::DictionaryAttr>("mdsl.vector_readiness");
  const auto semantic_contract =
      function->getAttrOfType<mlir::DictionaryAttr>("mdsl.semantic_contract");
  if (!function.isPublic() || !ordinal ||
      !ordinal.getType().isSignlessInteger(64) ||
      ordinal.getInt() != static_cast<std::int64_t>(expected_ordinal) ||
      !verifyReadinessAttribute(readiness, semantic_contract, error)) {
    if (error.empty())
      error = "vector-readiness function identity is incomplete or unordered";
    return false;
  }
  if (function.getArgAttrsAttr() || function.getResAttrsAttr()) {
    error = "vector-readiness function may not invent argument/result facts";
    return false;
  }
  if (function.getNoInline() || function.getSymVisibilityAttr()) {
    error = "vector-readiness function may not inherit policy attributes";
    return false;
  }
  if (!source_structured_function_type ||
      function.getFunctionType() != source_structured_function_type) {
    error = "vector-readiness tensor function type must remain identical to "
            "its certified structured source";
    return false;
  }

  std::array<std::int64_t, 3> mnk{};
  if (!getStaticGemmShape(function, mnk, error))
    return false;
  const auto [m, n, k] = mnk;
  if (!llvm::hasSingleElement(function.getBody()) ||
      std::distance(function.getBody().front().begin(),
                    function.getBody().front().end()) != 8) {
    error = "vector-readiness function must contain the exact MLIR 21.1.8 "
            "whole-static vectorization result";
    return false;
  }

  mlir::Block &block = function.getBody().front();
  if (llvm::any_of(block.getArguments(), [&](mlir::BlockArgument argument) {
        return argument.getLoc() != function.getLoc();
      })) {
    error = "vector-readiness function arguments must retain the exact "
            "authenticated function location";
    return false;
  }
  auto operation = block.begin();
  auto poison = mlir::dyn_cast<mlir::ub::PoisonOp>(*operation++);
  auto zero_index = mlir::dyn_cast<mlir::arith::ConstantOp>(*operation++);
  auto zero_vector = mlir::dyn_cast<mlir::arith::ConstantOp>(*operation++);
  auto lhs_read = mlir::dyn_cast<mlir::vector::TransferReadOp>(*operation++);
  auto rhs_read = mlir::dyn_cast<mlir::vector::TransferReadOp>(*operation++);
  auto contract = mlir::dyn_cast<mlir::vector::ContractionOp>(*operation++);
  auto output_write =
      mlir::dyn_cast<mlir::vector::TransferWriteOp>(*operation++);
  auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(*operation++);
  if (!poison || !zero_index || !zero_vector || !lhs_read || !rhs_read ||
      !contract || !output_write || !return_op) {
    error = "vector-readiness function contains a noncanonical vector "
            "operation sequence";
    return false;
  }

  const auto lhs_vector = mlir::VectorType::get(
      {m, k}, mlir::Float32Type::get(function.getContext()));
  const auto rhs_vector = mlir::VectorType::get(
      {k, n}, mlir::Float32Type::get(function.getContext()));
  const auto output_vector = mlir::VectorType::get(
      {m, n}, mlir::Float32Type::get(function.getContext()));
  if (!poison.getType().isF32() || !isExactIndexZero(zero_index) ||
      !isExactPositiveZeroVector(zero_vector, output_vector)) {
    error = "vector-readiness transfer padding/index or contraction zero "
            "accumulator is invalid";
    return false;
  }

  const mlir::AffineMap identity =
      mlir::AffineMap::getMultiDimIdentityMap(2, function.getContext());
  const auto verify_read = [&](mlir::vector::TransferReadOp read,
                               mlir::Value source, mlir::VectorType type,
                               llvm::StringRef role) {
    if (read.getBase() != source || read.getVectorType() != type ||
        !isZeroIndexRange(read.getIndices(), zero_index.getResult()) ||
        read.getPadding() != poison.getResult() || !hasNoMask(read.getMask()) ||
        read.getPermutationMap() != identity ||
        !allTrue(read.getInBounds(), 2)) {
      error = "vector-readiness " + role.str() +
              " transfer does not encode the full logical tensor";
      return false;
    }
    return true;
  };
  if (!verify_read(lhs_read, block.getArgument(0), lhs_vector, "lhs") ||
      !verify_read(rhs_read, block.getArgument(1), rhs_vector, "rhs"))
    return false;

  ContractionTopologyResultV1 topology = buildCanonicalContractionTopologyV1(
      *function.getContext(), StandardLinearAlgebraOperationV1::Gemm);
  std::string topology_error;
  if (!topology || !verifyCanonicalContractionTopologyV1(topology.topology,
                                                         topology_error)) {
    error = "vector readiness could not establish canonical GEMM topology: " +
            (topology.error.empty() ? topology_error : topology.error);
    return false;
  }
  llvm::SmallVector<mlir::vector::IteratorType> expected_iterators;
  for (mlir::utils::IteratorType iterator : topology.topology.iterator_types) {
    switch (iterator) {
    case mlir::utils::IteratorType::parallel:
      expected_iterators.push_back(mlir::vector::IteratorType::parallel);
      break;
    case mlir::utils::IteratorType::reduction:
      expected_iterators.push_back(mlir::vector::IteratorType::reduction);
      break;
    default:
      error = "canonical GEMM topology contains an unsupported Vector "
              "iterator role";
      return false;
    }
  }
  if (contract.getLhs() != lhs_read.getVector() ||
      contract.getRhs() != rhs_read.getVector() ||
      contract.getAcc() != zero_vector.getResult() ||
      contract.getResult().getType() != output_vector ||
      contract.getKind() != mlir::vector::CombiningKind::ADD ||
      contract.getIndexingMapsArray() != topology.topology.indexing_maps ||
      contract.getIteratorTypesArray() != expected_iterators) {
    error = "vector.contract does not exactly encode logical GEMM indexing, "
            "f32 accumulation, and positive-zero overwrite";
    return false;
  }

  if (output_write.getValueToStore() != contract.getResult() ||
      output_write.getBase() != block.getArgument(2) ||
      !isZeroIndexRange(output_write.getIndices(), zero_index.getResult()) ||
      !hasNoMask(output_write.getMask()) ||
      output_write.getPermutationMap() != identity ||
      !allTrue(output_write.getInBounds(), 2) ||
      output_write.getNumResults() != 1 ||
      output_write.getResult().getType() != block.getArgument(2).getType() ||
      return_op.getNumOperands() != 1 ||
      return_op.getOperand(0) != output_write.getResult()) {
    error = "vector-readiness result must be one full transfer write to the "
            "original output and the returned tensor";
    return false;
  }

  // MLIR 21.1.8's upstream Linalg vectorizer retains the payload source
  // location on the data-bearing zero, reads, and return, but gives generated
  // structural glue an unknown location. Preserve and verify that observed
  // limitation instead of fabricating source locations downstream.
  if (!mlir::isa<mlir::UnknownLoc>(poison.getLoc()) ||
      !mlir::isa<mlir::UnknownLoc>(zero_index.getLoc()) ||
      zero_vector.getLoc() != function.getLoc() ||
      lhs_read.getLoc() != function.getLoc() ||
      rhs_read.getLoc() != function.getLoc() ||
      !mlir::isa<mlir::UnknownLoc>(contract.getLoc()) ||
      !mlir::isa<mlir::UnknownLoc>(output_write.getLoc()) ||
      return_op.getLoc() != function.getLoc()) {
    error = "vector-readiness operation locations do not match the observed "
            "MLIR 21.1.8 provenance boundary";
    return false;
  }

  for (mlir::Operation &nested : block) {
    if (!nested.getDiscardableAttrDictionary().empty()) {
      error = "vector-readiness operations may not carry unreviewed hints";
      return false;
    }
  }
  return true;
}

bool applyUpstreamTransform(mlir::ModuleOp payload, std::string &error) {
  mlir::MLIRContext *context = payload.getContext();
  mlir::ParserConfig parser_config(context, /*verifyAfterParse=*/true);
  auto transform_module = mlir::parseSourceString<mlir::ModuleOp>(
      kTransformSchedule, parser_config);
  if (!transform_module) {
    error = "failed to parse the pinned MLIR 21.1.8 Transform schedule";
    return false;
  }
  auto transform_root =
      transform_module->lookupSymbol<mlir::transform::NamedSequenceOp>(
          mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  if (!transform_root) {
    error = "Transform schedule is missing __transform_main";
    return false;
  }
  const mlir::transform::TransformOptions options =
      mlir::transform::TransformOptions().enableExpensiveChecks(true);
  if (mlir::failed(mlir::transform::applyTransformNamedSequence(
          payload.getOperation(), transform_root.getOperation(),
          *transform_module, options))) {
    error = "upstream MLIR 21.1.8 Transform vectorization failed";
    return false;
  }
  return true;
}

} // namespace

void registerStructuredGemmVectorReadinessDialectsV1(
    mlir::MLIRContext &context) {
  registerStructuredGemmHandoffDialectsV1(context);
  mlir::DialectRegistry registry;
  mlir::linalg::registerTransformDialectExtension(registry);
  context.appendDialectRegistry(registry);
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::transform::TransformDialect>();
  context.getOrLoadDialect<mlir::ub::UBDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
}

std::string structuredGemmVectorReadinessTransformV1() {
  return kTransformSchedule.str();
}

StructuredGemmVectorReadinessResultV1
deriveStructuredGemmVectorReadinessV1(mlir::ModuleOp structured_module) {
  StructuredGemmVectorReadinessResultV1 result;
  if (!verifyStructuredGemmHandoffV1(structured_module, result.error)) {
    result.error = "vector readiness requires the exact verified structured "
                   "GEMM handoff: " +
                   result.error;
    return result;
  }
  if (structured_module.getBody()->empty()) {
    result.error = "vector readiness requires at least one structured GEMM";
    return result;
  }
  for (mlir::func::FuncOp function :
       structured_module.getOps<mlir::func::FuncOp>()) {
    std::array<std::int64_t, 3> mnk{};
    if (!getStaticGemmShape(function, mnk, result.error))
      return result;
  }

  mlir::MLIRContext &context = *structured_module.getContext();
  registerStructuredGemmVectorReadinessDialectsV1(context);
  result.module = mlir::cast<mlir::ModuleOp>(structured_module->clone());
  if (!applyUpstreamTransform(*result.module, result.error)) {
    result.module = nullptr;
    return result;
  }

  mlir::Builder builder(&context);
  if (!attachDerivedStructuredHandoffSourceIdentityV1(
          structured_module, *result.module,
          structuredGemmHandoffCertificateProfileV1(), builder, result.error)) {
    result.module = nullptr;
    return result;
  }
  (*result.module)
      ->setAttr("mdsl.producer",
                builder.getStringAttr(kVectorArtifactProducer));
  (*result.module)
      ->setAttr("mdsl.vector_readiness_schema",
                builder.getStringAttr(kStructuredGemmVectorReadinessSchemaV1));
  (*result.module)
      ->setAttr(
          "mdsl.vector_readiness_version",
          builder.getI32IntegerAttr(kStructuredGemmVectorReadinessVersionV1));
  for (mlir::func::FuncOp function :
       result.module->getOps<mlir::func::FuncOp>()) {
    const auto semantic_contract =
        function->getAttrOfType<mlir::DictionaryAttr>("mdsl.semantic_contract");
    const auto numerical_permissions =
        semantic_contract.getAs<mlir::DictionaryAttr>("numerical");
    function->setAttr("mdsl.vector_readiness",
                      readinessAttribute(builder, numerical_permissions));
  }

  if (!verifyStructuredGemmVectorReadinessV1(*result.module, result.error) ||
      !verifyStructuredGemmVectorReadinessMatchesV1(
          structured_module, *result.module, result.error)) {
    result.module = nullptr;
    return result;
  }
  return result;
}

bool verifyStructuredGemmVectorReadinessV1(mlir::ModuleOp module,
                                           std::string &error) {
  error.clear();
  if (!module) {
    error = "vector-readiness module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(module))) {
    error = "vector-readiness module failed upstream MLIR verification";
    return false;
  }
  if (!verifyDerivedStructuredHandoffSourceEnvelopeV1(
          module, structuredGemmHandoffCertificateProfileV1(), error) ||
      !hasExactModuleEnvelope(module, error) || module.getBody()->empty())
    return false;

  std::size_t ordinal = 0;
  for (mlir::Operation &operation : module.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function) {
      error = "vector-readiness module may contain only func.func sites";
      return false;
    }
    VerifiedDerivedStructuredHandoffSiteV1 source_identity;
    if (!verifyDerivedStructuredHandoffSourceIdentityV1(
            module, function, ordinal,
            structuredGemmHandoffCertificateProfileV1(), source_identity,
            error) ||
        !verifyRetainedStructuredGemmContractV1(
            module, function, source_identity.source_structured_function_type,
            error) ||
        !verifyVectorFunction(function, ordinal,
                              source_identity.source_structured_function_type,
                              error))
      return false;
    ++ordinal;
  }
  return true;
}

bool verifyStructuredGemmVectorReadinessMatchesV1(
    mlir::ModuleOp structured_module, mlir::ModuleOp vector_module,
    std::string &error) {
  error.clear();
  if (!verifyStructuredGemmHandoffV1(structured_module, error)) {
    error = "vector/source comparison requires a verified structured module: " +
            error;
    return false;
  }
  if (!verifyStructuredGemmVectorReadinessV1(vector_module, error))
    return false;
  return verifyDerivedStructuredHandoffMatchesSourceV1(
      structured_module, vector_module,
      structuredGemmHandoffCertificateProfileV1(), error);
}

} // namespace matcore::mdslc::mlir_bridge
