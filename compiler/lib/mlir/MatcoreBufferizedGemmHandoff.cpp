#include "MatcoreBufferizedGemmHandoff.h"

#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreStructuredHandoffCertificate.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {

constexpr llvm::StringLiteral kBufferizedArtifactProducer =
    "matcore-bufferized-gemm-handoff-v1";
constexpr llvm::StringLiteral kOverwriteRole =
    "destination_overwrite_zero_fill";
constexpr llvm::StringLiteral kContractionRole = "gemm_contraction";

constexpr llvm::StringLiteral kBufferizedModuleFields[] = {
    "mdsl.analysis_only",
    "mdsl.bufferization_handoff_schema",
    "mdsl.bufferization_handoff_version",
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
    "mdsl.source_structured_fingerprint",
    "mdsl.source_structured_producer",
    "mdsl.source_structured_site_count",
    "mdsl.structured_handoff_schema",
    "mdsl.structured_handoff_version",
    "mdsl.translation_unit",
};

constexpr llvm::StringLiteral kBufferizedFunctionFields[] = {
    "mdsl.bufferization_handoff",
    "mdsl.capture_ordinal",
    "mdsl.semantic_contract",
    "mdsl.site_id",
    "mdsl.source_semantic_symbol",
    "mdsl.source_structured_fingerprint",
    "mdsl.source_structured_function_type",
    "mdsl.structured_handoff",
};

bool requireExactNames(mlir::DictionaryAttr dictionary,
                       llvm::ArrayRef<llvm::StringLiteral> expected,
                       llvm::StringRef context, std::string &error) {
  if (!dictionary || dictionary.size() != expected.size()) {
    error = context.str() + " must contain exactly " +
            std::to_string(expected.size()) + " fields";
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
  return true;
}

bool requireString(mlir::Operation *operation, llvm::StringRef name,
                   llvm::StringRef expected, llvm::StringRef context,
                   std::string &error) {
  const auto value = operation->getAttrOfType<mlir::StringAttr>(name);
  if (!value || value.getValue() != expected) {
    error = context.str() + " field '" + name.str() + "' must be '" +
            expected.str() + "'";
    return false;
  }
  return true;
}

bool requireI32(mlir::Operation *operation, llvm::StringRef name,
                std::int64_t expected, llvm::StringRef context,
                std::string &error) {
  const auto value = operation->getAttrOfType<mlir::IntegerAttr>(name);
  if (!value || !value.getType().isSignlessInteger(32) ||
      value.getInt() != expected) {
    error = context.str() + " field '" + name.str() +
            "' must be the exact signless i32 version";
    return false;
  }
  return true;
}

mlir::DictionaryAttr bufferizationHandoffAttribute(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "algorithm",
           builder.getStringAttr("mlir-one-shot-module-bufferize")),
       builder.getNamedAttr("alias_preconditions",
                            builder.getStringAttr("retained_unproven")),
       builder.getNamedAttr("allocation",
                            builder.getStringAttr("none_in_verified_function")),
       builder.getNamedAttr("alignment_preconditions",
                            builder.getStringAttr("retained_unproven")),
       builder.getNamedAttr("authority",
                            builder.getStringAttr(
                                kStructuredGemmInspectionAuthorityV1)),
       builder.getNamedAttr("copies",
                            builder.getStringAttr("none_in_verified_function")),
       builder.getNamedAttr(
           "dtype_rank_static_extents",
           builder.getStringAttr("encoded_memref_type")),
       builder.getNamedAttr(
           "destination",
           builder.getStringAttr("returned_original_output_argument_2")),
       builder.getNamedAttr(
           "dynamic_shape_relations",
           builder.getStringAttr("retained_unproven")),
       builder.getNamedAttr(
           "effects",
           builder.getStringAttr("retained_structurally_checked")),
       builder.getNamedAttr(
           "function_boundary_layout",
           builder.getStringAttr("identity_row_major")),
       builder.getNamedAttr(
           "initial_output_value",
           builder.getStringAttr(
               "fully_overwritten_before_contraction_read")),
       builder.getNamedAttr("memory_space",
                            builder.getStringAttr("retained_unproven")),
       builder.getNamedAttr(
           "numerical",
           builder.getStringAttr(
               "retained_unconsumed_scalar_region_checked")),
       builder.getNamedAttr(
           "provenance",
           builder.getStringAttr(
               "retained_fingerprint_self_checked_pairing_required")),
       builder.getNamedAttr(
           "runtime_preconditions",
           builder.getStringAttr("retained_unproven_execution_forbidden")),
       builder.getNamedAttr(
           "semantic_contract",
           builder.getStringAttr("retained_verifier_checked")),
       builder.getNamedAttr("source_handoff",
                            builder.getStringAttr(
                                kStructuredGemmHandoffSchemaV1)),
       builder.getNamedAttr(
           "version",
           builder.getI32IntegerAttr(kBufferizedGemmHandoffVersionV1))});
}

bool verifyBufferizationHandoffAttribute(mlir::DictionaryAttr handoff,
                                         std::string &error) {
  constexpr llvm::StringLiteral fields[] = {
      "algorithm",
      "alias_preconditions",
      "allocation",
      "alignment_preconditions",
      "authority",
      "copies",
      "destination",
      "dtype_rank_static_extents",
      "dynamic_shape_relations",
      "effects",
      "function_boundary_layout",
      "initial_output_value",
      "memory_space",
      "numerical",
      "provenance",
      "runtime_preconditions",
      "semantic_contract",
      "source_handoff",
      "version",
  };
  if (!requireExactNames(handoff, fields, "bufferization handoff ledger",
                         error))
    return false;
  const auto require = [&](llvm::StringRef name, llvm::StringRef expected) {
    const auto value = handoff.getAs<mlir::StringAttr>(name);
    if (!value || value.getValue() != expected) {
      error = "bufferization handoff ledger field '" + name.str() +
              "' is invalid";
      return false;
    }
    return true;
  };
  const auto version = handoff.getAs<mlir::IntegerAttr>("version");
  return require("algorithm", "mlir-one-shot-module-bufferize") &&
         require("alias_preconditions", "retained_unproven") &&
         require("allocation", "none_in_verified_function") &&
         require("alignment_preconditions", "retained_unproven") &&
         require("authority", kStructuredGemmInspectionAuthorityV1) &&
         require("copies", "none_in_verified_function") &&
         require("destination", "returned_original_output_argument_2") &&
         require("dtype_rank_static_extents", "encoded_memref_type") &&
         require("dynamic_shape_relations", "retained_unproven") &&
         require("effects", "retained_structurally_checked") &&
         require("function_boundary_layout", "identity_row_major") &&
         require("initial_output_value",
                 "fully_overwritten_before_contraction_read") &&
         require("memory_space", "retained_unproven") &&
         require("numerical",
                 "retained_unconsumed_scalar_region_checked") &&
         require("provenance",
                 "retained_fingerprint_self_checked_pairing_required") &&
         require("runtime_preconditions",
                 "retained_unproven_execution_forbidden") &&
         require("semantic_contract", "retained_verifier_checked") &&
         require("source_handoff", kStructuredGemmHandoffSchemaV1) &&
         version && version.getType().isSignlessInteger(32) &&
         version.getInt() == kBufferizedGemmHandoffVersionV1;
}

bool verifyBufferizedModuleFields(mlir::ModuleOp module,
                                  std::string &error) {
  if (!requireExactNames(module->getAttrDictionary(), kBufferizedModuleFields,
                         "bufferized handoff module", error))
    return false;
  const auto analysis_only =
      module->getAttrOfType<mlir::BoolAttr>("mdsl.analysis_only");
  if (!analysis_only || !analysis_only.getValue()) {
    error = "bufferized handoff module must remain analysis-only";
    return false;
  }
  return requireString(module, "mdsl.producer", kBufferizedArtifactProducer,
                       "bufferized handoff module", error) &&
         requireString(module, "mdsl.source_structured_producer",
                       kStructuredGemmHandoffSchemaV1,
                       "bufferized handoff module", error) &&
         requireString(module, "mdsl.bufferization_handoff_schema",
                       kBufferizedGemmHandoffSchemaV1,
                       "bufferized handoff module", error) &&
         requireI32(module, "mdsl.bufferization_handoff_version",
                    kBufferizedGemmHandoffVersionV1,
                    "bufferized handoff module", error) &&
         requireString(module, "mdsl.execution_authority",
                       kStructuredGemmInspectionAuthorityV1,
                       "bufferized handoff module", error) &&
         requireString(module, "mdsl.structured_handoff_schema",
                       kStructuredGemmHandoffSchemaV1,
                       "bufferized handoff module", error) &&
         requireI32(module, "mdsl.structured_handoff_version",
                    kStructuredGemmHandoffVersionV1,
                    "bufferized handoff module", error);
}

bool hasExactMarker(mlir::Operation *operation, llvm::StringRef site,
                    llvm::StringRef role, std::string &error) {
  constexpr llvm::StringLiteral fields[] = {"mdsl.site_id",
                                             "mdsl.structured_role"};
  if (!requireExactNames(operation->getDiscardableAttrDictionary(), fields,
                         "bufferized Linalg operation marker", error))
    return false;
  const auto encoded_site =
      operation->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  const auto encoded_role =
      operation->getAttrOfType<mlir::StringAttr>("mdsl.structured_role");
  if (!encoded_site || encoded_site.getValue() != site || !encoded_role ||
      encoded_role.getValue() != role) {
    error = "bufferized Linalg operation lost its exact site/role marker";
    return false;
  }
  return true;
}

bool verifyDefaultMatmulMaps(mlir::linalg::MatmulOp matmul,
                             std::string &error) {
  if (matmul.hasUserDefinedMaps()) {
    error = "bufferized GEMM forbids user-defined transpose or broadcast maps";
    return false;
  }
  if (matmul.getCast() != mlir::linalg::TypeFn::cast_signed) {
    error = "bufferized GEMM requires the canonical matmul scalar cast";
    return false;
  }
  const mlir::ArrayAttr actual = matmul.getIndexingMaps();
  const llvm::SmallVector<mlir::AffineMap> expected =
      mlir::linalg::MatmulOp::getDefaultIndexingMaps(matmul.getContext());
  if (!actual || actual.size() != expected.size()) {
    error = "bufferized GEMM requires canonical matmul indexing maps";
    return false;
  }
  for (auto [attribute, map] : llvm::zip(actual, expected)) {
    const auto encoded = mlir::dyn_cast<mlir::AffineMapAttr>(attribute);
    if (!encoded || encoded.getValue() != map) {
      error = "bufferized GEMM matmul indexing maps are not canonical";
      return false;
    }
  }
  const auto iterators = matmul.getIteratorTypesArray();
  if (iterators.size() != 3 ||
      iterators[0] != mlir::utils::IteratorType::parallel ||
      iterators[1] != mlir::utils::IteratorType::parallel ||
      iterators[2] != mlir::utils::IteratorType::reduction) {
    error = "bufferized GEMM requires parallel M/N and reduction K";
    return false;
  }
  return true;
}

bool verifyMatmulScalarRegion(mlir::linalg::MatmulOp matmul,
                              std::string &error) {
  if (!llvm::hasSingleElement(matmul.getRegion())) {
    error = "bufferized GEMM matmul must have one canonical scalar block";
    return false;
  }
  mlir::Block &block = matmul.getRegion().front();
  if (block.getNumArguments() != 3 ||
      std::distance(block.begin(), block.end()) != 3) {
    error = "bufferized GEMM matmul scalar block must contain multiply, add, "
            "and yield";
    return false;
  }
  auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(block.front());
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(*std::next(block.begin()));
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(block.back());
  if (!multiply || !add || !yield ||
      multiply.getLhs() != block.getArgument(0) ||
      multiply.getRhs() != block.getArgument(1) ||
      add.getLhs() != block.getArgument(2) ||
      add.getRhs() != multiply.getResult() || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != add.getResult() ||
      multiply.getFastmath() != mlir::arith::FastMathFlags::none ||
      add.getFastmath() != mlir::arith::FastMathFlags::none) {
    error = "bufferized GEMM matmul scalar region or fast-math contract is "
            "not canonical";
    return false;
  }
  for (mlir::Operation &operation : block) {
    if (!operation.getDiscardableAttrDictionary().empty()) {
      error = "bufferized GEMM scalar operations may not carry discardable "
              "semantic hints";
      return false;
    }
  }
  return true;
}

bool verifyFillScalarRegion(mlir::linalg::FillOp fill, std::string &error) {
  if (!llvm::hasSingleElement(fill.getRegion())) {
    error = "bufferized GEMM fill must have one canonical scalar block";
    return false;
  }
  mlir::Block &block = fill.getRegion().front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isF32() ||
      !block.getArgument(1).getType().isF32() ||
      !llvm::hasSingleElement(block)) {
    error = "bufferized GEMM fill scalar block must contain only its f32 "
            "yield";
    return false;
  }
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(block.front());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != block.getArgument(0) ||
      !yield->getDiscardableAttrDictionary().empty()) {
    error = "bufferized GEMM fill must yield zero, never the old destination";
    return false;
  }
  return true;
}

bool isExactIdentityF32MemRef(mlir::Type type) {
  const auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 2 || !memref.getElementType().isF32())
    return false;
  const auto expected =
      mlir::MemRefType::get(memref.getShape(), memref.getElementType());
  return memref == expected;
}

bool memrefMatchesTensor(mlir::Type buffer_type, mlir::Type tensor_type) {
  const auto memref = mlir::dyn_cast<mlir::MemRefType>(buffer_type);
  const auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(tensor_type);
  return memref && tensor && memref.getShape() == tensor.getShape() &&
         memref.getElementType().isF32() && tensor.getElementType().isF32() &&
         isExactIdentityF32MemRef(buffer_type);
}

bool verifyBufferizedFunction(mlir::ModuleOp module,
                              mlir::func::FuncOp function,
                              std::size_t expected_ordinal,
                              std::string &error) {
  if (!requireExactNames(function->getDiscardableAttrDictionary(),
                         kBufferizedFunctionFields,
                         "bufferized GEMM function", error))
    return false;
  VerifiedDerivedStructuredHandoffSiteV1 source_identity;
  if (!verifyDerivedStructuredHandoffSourceIdentityV1(
          module, function, expected_ordinal,
          structuredGemmHandoffCertificateProfileV1(), source_identity,
          error))
    return false;
  if (!verifyRetainedStructuredGemmContractV1(
          module, function, source_identity.source_structured_function_type,
          error))
    return false;
  const auto site = source_identity.structured_identity.site_id;
  const auto handoff = function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.bufferization_handoff");
  if (!site || !handoff || !function.isPublic() ||
      function.getNumArguments() != 3 || function.getNumResults() != 1 ||
      !verifyBufferizationHandoffAttribute(handoff, error)) {
    if (error.empty())
      error = "bufferized GEMM function identity is incomplete";
    return false;
  }
  for (mlir::Type type : function.getArgumentTypes()) {
    if (!isExactIdentityF32MemRef(type)) {
      error = "bufferized GEMM arguments require rank-2 f32 identity-layout "
              "memrefs";
      return false;
    }
  }
  if (!isExactIdentityF32MemRef(function.getResultTypes().front()) ||
      function.getResultTypes().front() != function.getArgument(2).getType()) {
    error = "bufferized GEMM result type must exactly equal the output "
            "argument memref type";
    return false;
  }
  for (auto [buffer_type, tensor_type] :
       llvm::zip(function.getArgumentTypes(),
                 source_identity.source_structured_function_type.getInputs())) {
    if (!memrefMatchesTensor(buffer_type, tensor_type)) {
      error = "bufferized GEMM argument shape/type/layout differs from its "
              "retained source-structured type";
      return false;
    }
  }
  if (!memrefMatchesTensor(
          function.getResultTypes().front(),
          source_identity.source_structured_function_type.getResult(0))) {
    error = "bufferized GEMM result shape/type/layout differs from its "
            "retained source-structured type";
    return false;
  }

  if (!llvm::hasSingleElement(function.getBody()) ||
      std::distance(function.getBody().front().begin(),
                    function.getBody().front().end()) != 4) {
    error = "bufferized GEMM function must contain exactly zero, fill, "
            "matmul, and return; allocations and copies are forbidden";
    return false;
  }
  mlir::Block &block = function.getBody().front();
  for (mlir::BlockArgument argument : block.getArguments()) {
    if (argument.getLoc() != function.getLoc()) {
      error = "bufferized GEMM block arguments must retain the authenticated "
              "function location";
      return false;
    }
  }
  auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(block.front());
  auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(*std::next(block.begin()));
  auto matmul =
      mlir::dyn_cast<mlir::linalg::MatmulOp>(*std::next(block.begin(), 2));
  auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
  if (!constant || !fill || !matmul || !return_op) {
    error = "bufferized GEMM function has a noncanonical operation sequence";
    return false;
  }
  if (!constant->getDiscardableAttrDictionary().empty() ||
      !return_op->getDiscardableAttrDictionary().empty()) {
    error = "bufferized GEMM constant/return may not carry discardable hints";
    return false;
  }
  const auto zero = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
  if (!constant.getType().isF32() || !zero || !zero.getValue().isZero() ||
      zero.getValue().isNegative()) {
    error = "bufferized GEMM overwrite requires exact positive f32 zero";
    return false;
  }
  if (fill.getInputs().size() != 1 || fill.getOutputs().size() != 1 ||
      !fill.getResultTensors().empty() ||
      fill.getInputs().front() != constant.getResult() ||
      fill.getOutputs().front() != block.getArgument(2) ||
      !hasExactMarker(fill, site.getValue(), kOverwriteRole, error)) {
    if (error.empty())
      error = "bufferized GEMM fill must overwrite original output argument 2";
    return false;
  }
  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      !matmul.getResultTensors().empty() ||
      matmul.getInputs()[0] != block.getArgument(0) ||
      matmul.getInputs()[1] != block.getArgument(1) ||
      matmul.getOutputs().front() != block.getArgument(2) ||
      !hasExactMarker(matmul, site.getValue(), kContractionRole, error)) {
    if (error.empty())
      error = "bufferized GEMM matmul must use lhs/rhs and the original "
              "output buffer";
    return false;
  }
  if (return_op.getNumOperands() != 1 ||
      return_op.getOperand(0) != block.getArgument(2)) {
    error = "bufferized GEMM must return original output argument 2";
    return false;
  }
  if (constant.getLoc() != function.getLoc() ||
      fill.getLoc() != function.getLoc() ||
      matmul.getLoc() != function.getLoc() ||
      return_op.getLoc() != function.getLoc()) {
    error = "bufferized GEMM operations must retain source location";
    return false;
  }
  return verifyFillScalarRegion(fill, error) &&
         verifyDefaultMatmulMaps(matmul, error) &&
         verifyMatmulScalarRegion(matmul, error);
}

bool attachBufferizationCertificate(mlir::ModuleOp structured_source,
                                    mlir::ModuleOp module,
                                    std::string &error) {
  mlir::Builder builder(module.getContext());
  if (!attachDerivedStructuredHandoffSourceIdentityV1(
          structured_source, module,
          structuredGemmHandoffCertificateProfileV1(), builder, error))
    return false;
  module->setAttr("mdsl.producer",
                  builder.getStringAttr(kBufferizedArtifactProducer));
  module->setAttr("mdsl.bufferization_handoff_schema",
                  builder.getStringAttr(kBufferizedGemmHandoffSchemaV1));
  module->setAttr(
      "mdsl.bufferization_handoff_version",
      builder.getI32IntegerAttr(kBufferizedGemmHandoffVersionV1));
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    function->setAttr("mdsl.bufferization_handoff",
                      bufferizationHandoffAttribute(builder));
  return true;
}

std::uint64_t nonnegativeStatistic(std::int64_t value) {
  return value < 0 ? 0 : static_cast<std::uint64_t>(value);
}

} // namespace

void registerBufferizedGemmHandoffDialectsV1(mlir::MLIRContext &context) {
  registerStructuredGemmHandoffDialectsV1(context);
  mlir::DialectRegistry registry;
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
}

BufferizedGemmHandoffResultV1
deriveBufferizedGemmHandoffV1(mlir::ModuleOp structured_module) {
  BufferizedGemmHandoffResultV1 result;
  if (!verifyStructuredGemmHandoffV1(structured_module, result.error)) {
    result.error = "bufferized GEMM handoff requires an exact verified "
                   "structured-GEMM-v1 source: " +
                   result.error;
    return result;
  }

  registerBufferizedGemmHandoffDialectsV1(*structured_module.getContext());
  result.module = structured_module.clone();
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = true;
  options.copyBeforeWrite = false;
  options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(
          *result.module, options, state, &statistics))) {
    result.error = "MLIR One-Shot Module Bufferize rejected the verified "
                   "structured GEMM handoff";
    result.module = nullptr;
    return result;
  }
  result.buffer_allocations =
      nonnegativeStatistic(statistics.numBufferAlloc);
  result.buffer_deallocations =
      nonnegativeStatistic(statistics.numBufferDealloc);
  result.tensor_in_place =
      nonnegativeStatistic(statistics.numTensorInPlace);
  result.tensor_out_of_place =
      nonnegativeStatistic(statistics.numTensorOutOfPlace);

  if (statistics.numBufferAlloc != 0 || statistics.numBufferDealloc != 0 ||
      statistics.numTensorOutOfPlace != 0) {
    result.error = "MLIR One-Shot Module Bufferize did not satisfy the exact "
                   "zero-allocation, zero-deallocation, in-place certificate "
                   "envelope";
    result.module = nullptr;
    return result;
  }

  if (!attachBufferizationCertificate(structured_module, *result.module,
                                      result.error)) {
    result.module = nullptr;
    return result;
  }
  if (!verifyBufferizedGemmHandoffV1(*result.module, result.error) ||
      !verifyBufferizedGemmHandoffMatchesStructuredV1(
          structured_module, *result.module, result.error)) {
    result.module = nullptr;
    return result;
  }
  return result;
}

bool verifyBufferizedGemmHandoffV1(mlir::ModuleOp module,
                                  std::string &error) {
  error.clear();
  if (!module) {
    error = "bufferized GEMM handoff module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(module))) {
    error = "bufferized GEMM handoff failed upstream MLIR verification";
    return false;
  }
  if (!verifyBufferizedModuleFields(module, error) ||
      module.getBody()->empty()) {
    if (error.empty())
      error = "bufferized GEMM handoff must contain at least one function";
    return false;
  }
  if (!verifyDerivedStructuredHandoffSourceEnvelopeV1(
          module, structuredGemmHandoffCertificateProfileV1(), error))
    return false;
  std::size_t ordinal = 0;
  for (mlir::Operation &operation : module.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function) {
      error = "bufferized GEMM module may contain only func.func sites";
      return false;
    }
    if (!verifyBufferizedFunction(module, function, ordinal, error))
      return false;
    ++ordinal;
  }
  return true;
}

bool verifyBufferizedGemmHandoffMatchesStructuredV1(
    mlir::ModuleOp structured_module, mlir::ModuleOp bufferized_module,
    std::string &error) {
  error.clear();
  if (!verifyStructuredGemmHandoffV1(structured_module, error)) {
    error = "bufferized/structured comparison requires a verified structured "
            "source: " +
            error;
    return false;
  }
  if (!verifyBufferizedGemmHandoffV1(bufferized_module, error))
    return false;
  return verifyDerivedStructuredHandoffMatchesSourceV1(
      structured_module, bufferized_module,
      structuredGemmHandoffCertificateProfileV1(), error);
}

} // namespace matcore::mdslc::mlir_bridge
