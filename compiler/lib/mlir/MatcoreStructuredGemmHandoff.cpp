#include "MatcoreStructuredGemmHandoff.h"

#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {

constexpr llvm::StringLiteral kSemanticBridgeSchema =
    "matcore-mlir-semantic-v1";
constexpr llvm::StringLiteral kStructuredFunctionPrefix =
    "__matcore_structured_";
constexpr llvm::StringLiteral kSemanticFunctionPrefix =
    "__matcore_semantic_";
constexpr llvm::StringLiteral kStructuredArtifactProducer =
    "matcore-structured-gemm-handoff-v1";
constexpr llvm::StringLiteral kOverwriteRole =
    "destination_overwrite_zero_fill";
constexpr llvm::StringLiteral kContractionRole = "gemm_contraction";

constexpr llvm::StringLiteral kSemanticModuleFields[] = {
    "mdsl.bridge_schema",      "mdsl.capture_schema",
    "mdsl.capture_version",    "mdsl.execution_intent",
    "mdsl.numerical_profile",  "mdsl.producer",
    "mdsl.semantic_version",   "mdsl.source_file",
    "mdsl.translation_unit",
};

constexpr llvm::StringLiteral kStructuredModuleFields[] = {
    "mdsl.analysis_only",
    "mdsl.capture_schema",
    "mdsl.capture_version",
    "mdsl.execution_authority",
    "mdsl.execution_intent",
    "mdsl.numerical_profile",
    "mdsl.producer",
    "mdsl.source_producer",
    "mdsl.source_bridge_schema",
    "mdsl.source_file",
    "mdsl.source_semantic_version",
    "mdsl.structured_handoff_schema",
    "mdsl.structured_handoff_version",
    "mdsl.translation_unit",
};

constexpr llvm::StringLiteral kGemmContractFields[] = {
    "accumulation_type",    "aliasing",      "effects",
    "lhs_semantics",        "numerical",     "origin",
    "output_semantics",     "policy",        "provenance",
    "rhs_semantics",        "semantic_requirements",
    "site_id",              "synchronization",
};

constexpr llvm::StringLiteral kStructuredFunctionFields[] = {
    "mdsl.capture_ordinal",       "mdsl.semantic_contract",
    "mdsl.site_id",               "mdsl.source_semantic_symbol",
    "mdsl.structured_handoff",
};

constexpr llvm::StringLiteral kSemanticFunctionFields[] = {
    "mdsl.capture_ordinal", "mdsl.site_id"};

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

mlir::DictionaryAttr selectedAttributes(
    mlir::Builder &builder, mlir::Operation *operation,
    llvm::ArrayRef<llvm::StringLiteral> names, llvm::StringRef context,
    std::string &error) {
  llvm::SmallVector<mlir::NamedAttribute> selected;
  selected.reserve(names.size());
  for (llvm::StringRef name : names) {
    mlir::Attribute value = operation->getAttr(name);
    if (!value) {
      error = context.str() + " is missing field '" + name.str() + "'";
      return {};
    }
    selected.push_back(builder.getNamedAttr(name, value));
  }
  mlir::DictionaryAttr result = builder.getDictionaryAttr(selected);
  if (!requireExactNames(operation->getAttrDictionary(), names, context,
                         error))
    return {};
  return result;
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

template <typename MlirValue>
std::string textualIdentity(MlirValue value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  stream.flush();
  return text;
}

bool verifySourceModuleFields(mlir::ModuleOp module, std::string &error) {
  if (!requireExactNames(module->getAttrDictionary(), kSemanticModuleFields,
                         "semantic source module", error))
    return false;
  if (!requireString(module, "mdsl.bridge_schema", kSemanticBridgeSchema,
                     "semantic source module", error))
    return false;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (!requireExactNames(function->getDiscardableAttrDictionary(),
                           kSemanticFunctionFields,
                           "semantic source function", error))
      return false;
    if (function.getArgAttrsAttr() || function.getResAttrsAttr()) {
      error = "semantic source functions may not carry argument or result "
              "optimizer attributes";
      return false;
    }
    if (function.getNoInline() || function.getSymVisibilityAttr()) {
      error = "semantic source functions may not carry no_inline or an "
              "explicit visibility property";
      return false;
    }
  }
  return true;
}

bool verifyStructuredModuleFields(mlir::ModuleOp module, std::string &error) {
  if (!requireExactNames(module->getAttrDictionary(), kStructuredModuleFields,
                         "structured handoff module", error))
    return false;
  const auto analysis_only =
      module->getAttrOfType<mlir::BoolAttr>("mdsl.analysis_only");
  if (!analysis_only || !analysis_only.getValue()) {
    error = "structured handoff module must be explicitly analysis-only";
    return false;
  }
  const auto producer =
      module->getAttrOfType<mlir::StringAttr>("mdsl.producer");
  const auto source_producer =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_producer");
  const auto source_file =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_file");
  const auto translation_unit =
      module->getAttrOfType<mlir::StringAttr>("mdsl.translation_unit");
  if (!producer || producer.getValue() != kStructuredArtifactProducer ||
      !source_producer ||
      (source_producer.getValue() != "clang-libtooling-v1" &&
       source_producer.getValue() != "clang-ast-json-bootstrap-v0") ||
      !source_file ||
      !source_file.getValue().ends_with(".mdsl") || !translation_unit ||
      translation_unit.getValue().empty()) {
    error = "structured handoff module capture identity is invalid";
    return false;
  }
  return requireString(module, "mdsl.capture_schema", "matcore-ir-v1",
                       "structured handoff module", error) &&
         requireI32(module, "mdsl.capture_version",
                    ir::v1::kMatcoreIrVersion,
                    "structured handoff module", error) &&
         requireString(module, "mdsl.execution_intent", "generic",
                       "structured handoff module", error) &&
         requireString(module, "mdsl.numerical_profile",
                       kExplicitGemmF32Profile,
                       "structured handoff module", error) &&
         requireString(module, "mdsl.execution_authority",
                       kStructuredGemmInspectionAuthorityV1,
                       "structured handoff module", error) &&
         requireString(module, "mdsl.source_bridge_schema",
                       kSemanticBridgeSchema, "structured handoff module",
                       error) &&
         requireString(module, "mdsl.structured_handoff_schema",
                       kStructuredGemmHandoffSchemaV1,
                       "structured handoff module", error) &&
         requireI32(module, "mdsl.structured_handoff_version",
                    kStructuredGemmHandoffVersionV1,
                    "structured handoff module", error) &&
         requireI32(module, "mdsl.source_semantic_version",
                    kMatcoreSemanticModuleVersion,
                    "structured handoff module", error);
}

mlir::DictionaryAttr handoffAttribute(mlir::Builder &builder) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "authority",
           builder.getStringAttr(kStructuredGemmInspectionAuthorityV1)),
       builder.getNamedAttr(
           "destination",
           builder.getStringAttr("original_output_full_zero_fill")),
       builder.getNamedAttr("source_operation",
                            builder.getStringAttr("mdsl.gemm")),
       builder.getNamedAttr(
           "version",
           builder.getI32IntegerAttr(kStructuredGemmHandoffVersionV1))});
}

bool verifyHandoffAttribute(mlir::DictionaryAttr handoff,
                            std::string &error) {
  constexpr llvm::StringLiteral fields[] = {
      "authority", "destination", "source_operation", "version"};
  if (!requireExactNames(handoff, fields, "structured function handoff",
                         error))
    return false;
  const auto authority = handoff.getAs<mlir::StringAttr>("authority");
  const auto destination = handoff.getAs<mlir::StringAttr>("destination");
  const auto source_operation =
      handoff.getAs<mlir::StringAttr>("source_operation");
  const auto version = handoff.getAs<mlir::IntegerAttr>("version");
  if (!authority ||
      authority.getValue() != kStructuredGemmInspectionAuthorityV1 ||
      !destination ||
      destination.getValue() != "original_output_full_zero_fill" ||
      !source_operation || source_operation.getValue() != "mdsl.gemm" ||
      !version || !version.getType().isSignlessInteger(32) ||
      version.getInt() != kStructuredGemmHandoffVersionV1) {
    error = "structured function handoff contract is incomplete or invalid";
    return false;
  }
  return true;
}

bool verifyGemmContractStorageTypes(mlir::DictionaryAttr contract,
                                    std::string &error) {
  constexpr llvm::StringLiteral dictionary_fields[] = {
      "effects",          "lhs_semantics", "numerical",  "origin",
      "output_semantics", "policy",        "provenance", "rhs_semantics",
  };
  for (llvm::StringRef name : dictionary_fields) {
    if (!mlir::isa<mlir::DictionaryAttr>(contract.get(name))) {
      error = "structured GEMM semantic contract field '" + name.str() +
              "' must be a dictionary";
      return false;
    }
  }
  if (!mlir::isa<mlir::TypeAttr>(contract.get("accumulation_type")) ||
      !mlir::isa<mlir::ArrayAttr>(contract.get("aliasing")) ||
      !mlir::isa<mlir::ArrayAttr>(contract.get("semantic_requirements")) ||
      !mlir::isa<mlir::StringAttr>(contract.get("site_id")) ||
      !mlir::isa<mlir::StringAttr>(contract.get("synchronization"))) {
    error = "structured GEMM semantic contract has an invalid outer field "
            "type";
    return false;
  }
  return true;
}

bool hasExactMarker(mlir::Operation *operation, llvm::StringRef site,
                    llvm::StringRef role, std::string &error) {
  constexpr llvm::StringLiteral fields[] = {"mdsl.site_id",
                                             "mdsl.structured_role"};
  if (!requireExactNames(operation->getDiscardableAttrDictionary(), fields,
                         "structured Linalg operation marker", error))
    return false;
  const auto encoded_site =
      operation->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  const auto encoded_role =
      operation->getAttrOfType<mlir::StringAttr>("mdsl.structured_role");
  if (!encoded_site || encoded_site.getValue() != site || !encoded_role ||
      encoded_role.getValue() != role) {
    error = "structured Linalg operation lost its exact site/role marker";
    return false;
  }
  return true;
}

bool verifyDefaultMatmulMaps(mlir::linalg::MatmulOp matmul,
                             std::string &error) {
  if (matmul.hasUserDefinedMaps()) {
    error = "structured GEMM forbids user-defined transpose or broadcast maps";
    return false;
  }
  if (matmul.getCast() != mlir::linalg::TypeFn::cast_signed) {
    error = "structured GEMM requires the canonical matmul scalar cast";
    return false;
  }
  const mlir::ArrayAttr actual = matmul.getIndexingMaps();
  const llvm::SmallVector<mlir::AffineMap> expected =
      mlir::linalg::MatmulOp::getDefaultIndexingMaps(matmul.getContext());
  if (!actual || actual.size() != expected.size()) {
    error = "structured GEMM requires the canonical matmul indexing maps";
    return false;
  }
  for (auto [attribute, map] : llvm::zip(actual, expected)) {
    const auto encoded = mlir::dyn_cast<mlir::AffineMapAttr>(attribute);
    if (!encoded || encoded.getValue() != map) {
      error = "structured GEMM matmul indexing maps are not canonical "
              "(m,k),(k,n),(m,n)";
      return false;
    }
  }
  const auto iterators = matmul.getIteratorTypesArray();
  if (iterators.size() != 3 ||
      iterators[0] != mlir::utils::IteratorType::parallel ||
      iterators[1] != mlir::utils::IteratorType::parallel ||
      iterators[2] != mlir::utils::IteratorType::reduction) {
    error = "structured GEMM matmul requires parallel M/N and reduction K";
    return false;
  }
  return true;
}

bool verifyMatmulScalarRegion(mlir::linalg::MatmulOp matmul,
                              std::string &error) {
  if (!llvm::hasSingleElement(matmul.getRegion())) {
    error = "structured GEMM matmul must have one canonical scalar block";
    return false;
  }
  mlir::Block &block = matmul.getRegion().front();
  if (block.getNumArguments() != 3 ||
      std::distance(block.begin(), block.end()) != 3) {
    error = "structured GEMM matmul scalar block must contain multiply, add, "
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
    error = "structured GEMM matmul scalar region or fast-math contract is "
            "not canonical";
    return false;
  }
  return true;
}

bool verifyFillScalarRegion(mlir::linalg::FillOp fill, std::string &error) {
  if (!llvm::hasSingleElement(fill.getRegion())) {
    error = "structured GEMM fill must have one canonical scalar block";
    return false;
  }
  mlir::Block &block = fill.getRegion().front();
  if (block.getNumArguments() != 2 ||
      !block.getArgument(0).getType().isF32() ||
      !block.getArgument(1).getType().isF32() ||
      !llvm::hasSingleElement(block)) {
    error = "structured GEMM fill scalar block must contain only its f32 "
            "yield";
    return false;
  }
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(block.front());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != block.getArgument(0) ||
      !yield->getDiscardableAttrDictionary().empty()) {
    error = "structured GEMM fill must yield the zero input, never the old "
            "destination element";
    return false;
  }
  return true;
}

bool verifyStructuredFunction(mlir::func::FuncOp function,
                              std::size_t expected_ordinal,
                              llvm::StringRef module_source_file,
                              llvm::StringRef module_numerical_profile,
                              std::string &error) {
  if (!requireExactNames(function->getDiscardableAttrDictionary(),
                         kStructuredFunctionFields,
                         "structured GEMM function", error))
    return false;
  if (function.getArgAttrsAttr() || function.getResAttrsAttr() ||
      function.getNoInline() || function.getSymVisibilityAttr()) {
    error = "structured GEMM functions may not carry argument/result "
            "optimizer attributes, no_inline, or explicit visibility";
    return false;
  }

  const auto site = function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  const auto ordinal =
      function->getAttrOfType<mlir::IntegerAttr>("mdsl.capture_ordinal");
  const auto source_symbol = function->getAttrOfType<mlir::StringAttr>(
      "mdsl.source_semantic_symbol");
  const auto contract = function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.semantic_contract");
  const auto handoff = function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.structured_handoff");
  if (!site || !ordinal || !ordinal.getType().isSignlessInteger(64) ||
      ordinal.getInt() != static_cast<std::int64_t>(expected_ordinal) ||
      !source_symbol || !contract || !handoff || !function.isPublic() ||
      function.getNumArguments() != 3 || function.getNumResults() != 1) {
    error = "structured GEMM function identity is incomplete or unordered";
    return false;
  }
  const std::string expected_name =
      (llvm::Twine(kStructuredFunctionPrefix) + site.getValue()).str();
  const std::string expected_source =
      (llvm::Twine(kSemanticFunctionPrefix) + site.getValue()).str();
  if (function.getName() != expected_name ||
      source_symbol.getValue() != expected_source) {
    error = "structured GEMM function/source symbols must match the exact "
            "semantic site identity";
    return false;
  }
  if (!verifyHandoffAttribute(handoff, error) ||
      !requireExactNames(contract, kGemmContractFields,
                         "structured GEMM semantic contract", error) ||
      !verifyGemmContractStorageTypes(contract, error))
    return false;

  const auto contract_site = contract.getAs<mlir::StringAttr>("site_id");
  const auto numerical =
      contract.getAs<mlir::DictionaryAttr>("numerical");
  const auto provenance =
      contract.getAs<mlir::DictionaryAttr>("provenance");
  const auto numerical_profile =
      numerical ? numerical.getAs<mlir::StringAttr>("profile")
                : mlir::StringAttr{};
  const auto provenance_file =
      provenance ? provenance.getAs<mlir::StringAttr>("file")
                 : mlir::StringAttr{};
  if (!contract_site || contract_site != site || !numerical_profile ||
      numerical_profile.getValue() != module_numerical_profile ||
      !provenance_file || provenance_file.getValue() != module_source_file) {
    error = "structured GEMM contract does not match module/site provenance";
    return false;
  }

  if (!llvm::hasSingleElement(function.getBody()) ||
      std::distance(function.getBody().front().begin(),
                    function.getBody().front().end()) != 4) {
    error = "structured GEMM function must contain exactly zero, fill, matmul, "
            "and return";
    return false;
  }
  mlir::Block &block = function.getBody().front();
  auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(block.front());
  auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(*std::next(block.begin()));
  auto matmul =
      mlir::dyn_cast<mlir::linalg::MatmulOp>(*std::next(block.begin(), 2));
  auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
  if (!constant || !fill || !matmul || !return_op) {
    error = "structured GEMM function has a noncanonical operation sequence";
    return false;
  }
  if (!constant->getDiscardableAttrDictionary().empty() ||
      !return_op->getDiscardableAttrDictionary().empty()) {
    error = "structured GEMM constant/return may not carry discardable "
            "semantic hints";
    return false;
  }

  const auto zero = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
  if (!constant.getType().isF32() || !zero || !zero.getValue().isZero() ||
      zero.getValue().isNegative()) {
    error = "structured GEMM overwrite requires an exact positive f32 zero";
    return false;
  }
  if (fill.getInputs().size() != 1 || fill.getOutputs().size() != 1 ||
      fill.getResultTensors().size() != 1 ||
      fill.getInputs().front() != constant.getResult() ||
      fill.getOutputs().front() != block.getArgument(2) ||
      !hasExactMarker(fill, site.getValue(), kOverwriteRole, error)) {
    if (error.empty())
      error = "structured GEMM fill must overwrite the original output";
    return false;
  }
  if (matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      matmul.getResultTensors().size() != 1 ||
      matmul.getInputs()[0] != block.getArgument(0) ||
      matmul.getInputs()[1] != block.getArgument(1) ||
      matmul.getOutputs().front() != fill.getResultTensors().front() ||
      !hasExactMarker(matmul, site.getValue(), kContractionRole, error)) {
    if (error.empty())
      error = "structured GEMM matmul must consume lhs/rhs and the filled "
              "destination";
    return false;
  }
  if (return_op.getNumOperands() != 1 ||
      return_op.getOperand(0) != matmul.getResultTensors().front() ||
      function.getResultTypes().front() != return_op.getOperand(0).getType()) {
    error = "structured GEMM function must return the matmul result";
    return false;
  }
  if (constant.getLoc() != function.getLoc() ||
      fill.getLoc() != function.getLoc() ||
      matmul.getLoc() != function.getLoc() ||
      return_op.getLoc() != function.getLoc()) {
    error = "structured GEMM operations must retain the authenticated source "
            "location";
    return false;
  }
  for (mlir::Operation &scalar_operation : matmul.getRegion().front()) {
    if (!scalar_operation.getDiscardableAttrDictionary().empty()) {
      error = "structured GEMM scalar operations may not carry discardable "
              "semantic hints";
      return false;
    }
  }
  return verifyFillScalarRegion(fill, error) &&
         verifyDefaultMatmulMaps(matmul, error) &&
         verifyMatmulScalarRegion(matmul, error);
}

bool copySourceModuleMetadata(mlir::ModuleOp source, mlir::ModuleOp target,
                              mlir::Builder &builder, std::string &error) {
  if (!verifySourceModuleFields(source, error))
    return false;
  const auto copy = [&](llvm::StringRef source_name,
                        llvm::StringRef target_name) {
    target->setAttr(target_name, source->getAttr(source_name));
  };
  copy("mdsl.capture_schema", "mdsl.capture_schema");
  copy("mdsl.capture_version", "mdsl.capture_version");
  copy("mdsl.execution_intent", "mdsl.execution_intent");
  copy("mdsl.numerical_profile", "mdsl.numerical_profile");
  copy("mdsl.producer", "mdsl.source_producer");
  copy("mdsl.source_file", "mdsl.source_file");
  copy("mdsl.translation_unit", "mdsl.translation_unit");
  copy("mdsl.bridge_schema", "mdsl.source_bridge_schema");
  copy("mdsl.semantic_version", "mdsl.source_semantic_version");
  target->setAttr("mdsl.analysis_only", builder.getBoolAttr(true));
  target->setAttr("mdsl.producer",
                  builder.getStringAttr(kStructuredArtifactProducer));
  target->setAttr(
      "mdsl.execution_authority",
      builder.getStringAttr(kStructuredGemmInspectionAuthorityV1));
  target->setAttr("mdsl.structured_handoff_schema",
                  builder.getStringAttr(kStructuredGemmHandoffSchemaV1));
  target->setAttr(
      "mdsl.structured_handoff_version",
      builder.getI32IntegerAttr(kStructuredGemmHandoffVersionV1));
  return true;
}

bool appendStructuredFunction(mlir::ModuleOp target, mlir::OpBuilder &builder,
                              mlir::func::FuncOp source_function,
                              mlir_dialect::GemmOp source_gemm,
                              std::string &error) {
  mlir::DictionaryAttr contract =
      selectedAttributes(builder, source_gemm.getOperation(),
                         kGemmContractFields, "source mdsl.gemm", error);
  if (!contract)
    return false;
  const auto site = source_gemm.getSiteId();
  const std::string function_name =
      (llvm::Twine(kStructuredFunctionPrefix) + site).str();
  auto function = mlir::func::FuncOp::create(
      source_function.getLoc(), function_name,
      source_function.getFunctionType());
  function->setAttr("mdsl.capture_ordinal",
                    source_function->getAttr("mdsl.capture_ordinal"));
  function->setAttr("mdsl.site_id", builder.getStringAttr(site));
  function->setAttr("mdsl.source_semantic_symbol",
                    builder.getStringAttr(source_function.getName()));
  function->setAttr("mdsl.semantic_contract", contract);
  function->setAttr("mdsl.structured_handoff", handoffAttribute(builder));
  target.push_back(function);

  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  const auto location = source_gemm.getLoc();
  auto zero = mlir::arith::ConstantOp::create(
      builder, location, builder.getF32FloatAttr(0.0));
  const llvm::SmallVector<mlir::NamedAttribute> fill_attributes = {
      builder.getNamedAttr("mdsl.site_id", builder.getStringAttr(site)),
      builder.getNamedAttr("mdsl.structured_role",
                           builder.getStringAttr(kOverwriteRole)),
  };
  auto fill = mlir::linalg::FillOp::create(
      builder, location, mlir::TypeRange{entry->getArgument(2).getType()},
      mlir::ValueRange{zero.getResult()},
      mlir::ValueRange{entry->getArgument(2)}, fill_attributes);
  const llvm::SmallVector<mlir::NamedAttribute> matmul_attributes = {
      builder.getNamedAttr("mdsl.site_id", builder.getStringAttr(site)),
      builder.getNamedAttr("mdsl.structured_role",
                           builder.getStringAttr(kContractionRole)),
  };
  auto matmul = mlir::linalg::MatmulOp::create(
      builder, location, mlir::TypeRange{entry->getArgument(2).getType()},
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1)},
      mlir::ValueRange{fill.getResultTensors().front()}, matmul_attributes);
  mlir::func::ReturnOp::create(builder, location,
                               matmul.getResultTensors().front());
  return true;
}

mlir::OwningOpRef<mlir::ModuleOp>
reconstructSemanticWitness(mlir::ModuleOp structured_module,
                           std::string &error) {
  mlir::MLIRContext *context = structured_module.getContext();
  mlir::OpBuilder builder(context);
  auto semantic = mlir::ModuleOp::create(structured_module.getLoc());
  constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> module_map[] = {
      {"mdsl.capture_schema", "mdsl.capture_schema"},
      {"mdsl.capture_version", "mdsl.capture_version"},
      {"mdsl.execution_intent", "mdsl.execution_intent"},
      {"mdsl.numerical_profile", "mdsl.numerical_profile"},
      {"mdsl.source_producer", "mdsl.producer"},
      {"mdsl.source_bridge_schema", "mdsl.bridge_schema"},
      {"mdsl.source_file", "mdsl.source_file"},
      {"mdsl.source_semantic_version", "mdsl.semantic_version"},
      {"mdsl.translation_unit", "mdsl.translation_unit"},
  };
  for (auto [structured_name, semantic_name] : module_map)
    semantic->setAttr(semantic_name,
                      structured_module->getAttr(structured_name));

  for (mlir::func::FuncOp structured_function :
       structured_module.getOps<mlir::func::FuncOp>()) {
    const auto source_symbol =
        structured_function->getAttrOfType<mlir::StringAttr>(
            "mdsl.source_semantic_symbol");
    const auto site =
        structured_function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
    const auto contract =
        structured_function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.semantic_contract");
    if (!source_symbol || !site || !contract) {
      error = "cannot reconstruct semantic witness from incomplete structured "
              "function metadata";
      return {};
    }
    auto semantic_function = mlir::func::FuncOp::create(
        structured_function.getLoc(), source_symbol.getValue(),
        structured_function.getFunctionType());
    semantic_function->setAttr(
        "mdsl.capture_ordinal",
        structured_function->getAttr("mdsl.capture_ordinal"));
    semantic_function->setAttr("mdsl.site_id", site);
    semantic.push_back(semantic_function);

    mlir::Block *entry = semantic_function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    mlir::OperationState state(structured_function.getLoc(),
                               mlir_dialect::GemmOp::getOperationName());
    state.addOperands(entry->getArguments());
    state.addTypes(semantic_function.getResultTypes());
    state.addAttributes(contract.getValue());
    auto gemm = mlir::cast<mlir_dialect::GemmOp>(builder.create(state));
    mlir::func::ReturnOp::create(builder, structured_function.getLoc(),
                                 gemm.getResult());
  }
  return semantic;
}

} // namespace

void registerStructuredGemmHandoffDialectsV1(mlir::MLIRContext &context) {
  registerMatcoreSemanticDialects(context);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
}

StructuredGemmHandoffResultV1
deriveStructuredGemmHandoffV1(mlir::ModuleOp semantic_module) {
  StructuredGemmHandoffResultV1 result;
  if (!verifyMatcoreV1BridgeModule(semantic_module, result.error)) {
    result.error = "structured GEMM handoff requires the exact verified "
                   "explicit Matcore IR v1 bridge envelope: " +
                   result.error;
    return result;
  }
  if (!verifySourceModuleFields(semantic_module, result.error))
    return result;
  if (semantic_module.getBody()->empty()) {
    result.error =
        "structured GEMM handoff requires at least one explicit mdsl.gemm";
    return result;
  }

  mlir::MLIRContext &context = *semantic_module.getContext();
  registerStructuredGemmHandoffDialectsV1(context);
  mlir::OpBuilder builder(&context);
  result.module = mlir::ModuleOp::create(semantic_module.getLoc());
  if (!copySourceModuleMetadata(semantic_module, *result.module, builder,
                                result.error)) {
    result.module = nullptr;
    return result;
  }

  for (mlir::Operation &operation : semantic_module.getBody()->getOperations()) {
    auto source_function = mlir::cast<mlir::func::FuncOp>(operation);
    auto source_gemm = mlir::cast<mlir_dialect::GemmOp>(
        source_function.getBody().front().front());
    if (!appendStructuredFunction(*result.module, builder, source_function,
                                  source_gemm, result.error)) {
      result.module = nullptr;
      return result;
    }
  }

  if (!verifyStructuredGemmHandoffV1(*result.module, result.error) ||
      !verifyStructuredGemmHandoffMatchesV1(
          semantic_module, *result.module, result.error)) {
    result.module = nullptr;
    return result;
  }
  return result;
}

bool verifyStructuredGemmHandoffV1(mlir::ModuleOp module,
                                   std::string &error) {
  error.clear();
  if (!module) {
    error = "structured GEMM handoff module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(module))) {
    error = "structured GEMM handoff failed upstream MLIR verification";
    return false;
  }
  if (!verifyStructuredModuleFields(module, error))
    return false;
  const auto source_file =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_file");
  const auto numerical_profile =
      module->getAttrOfType<mlir::StringAttr>("mdsl.numerical_profile");
  if (!source_file || source_file.getValue().empty() || !numerical_profile ||
      numerical_profile.getValue() != kExplicitGemmF32Profile ||
      module.getBody()->empty()) {
    error = "structured GEMM module capture/numerical identity is invalid";
    return false;
  }

  std::size_t ordinal = 0;
  for (mlir::Operation &operation : module.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function) {
      error = "structured GEMM module may contain only func.func sites";
      return false;
    }
    if (!verifyStructuredFunction(function, ordinal, source_file.getValue(),
                                  numerical_profile.getValue(), error))
      return false;
    ++ordinal;
  }
  mlir::OwningOpRef<mlir::ModuleOp> semantic_witness =
      reconstructSemanticWitness(module, error);
  if (!semantic_witness)
    return false;
  mlir::ScopedDiagnosticHandler silence(
      module.getContext(),
      [](mlir::Diagnostic &) { return mlir::success(); });
  std::string witness_error;
  if (!verifyMatcoreV1BridgeModule(*semantic_witness, witness_error)) {
    error = "structured GEMM retained contract does not reconstruct the exact "
            "verified explicit bridge envelope: " +
            witness_error;
    return false;
  }
  return true;
}

bool verifyStructuredGemmHandoffMatchesV1(mlir::ModuleOp semantic_module,
                                          mlir::ModuleOp structured_module,
                                          std::string &error) {
  error.clear();
  if (!verifyMatcoreV1BridgeModule(semantic_module, error)) {
    error = "structured/source comparison requires a verified semantic "
            "module: " +
            error;
    return false;
  }
  if (!verifySourceModuleFields(semantic_module, error) ||
      !verifyStructuredGemmHandoffV1(structured_module, error))
    return false;

  constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> module_map[] = {
      {"mdsl.capture_schema", "mdsl.capture_schema"},
      {"mdsl.capture_version", "mdsl.capture_version"},
      {"mdsl.execution_intent", "mdsl.execution_intent"},
      {"mdsl.numerical_profile", "mdsl.numerical_profile"},
      {"mdsl.producer", "mdsl.source_producer"},
      {"mdsl.bridge_schema", "mdsl.source_bridge_schema"},
      {"mdsl.source_file", "mdsl.source_file"},
      {"mdsl.semantic_version", "mdsl.source_semantic_version"},
      {"mdsl.translation_unit", "mdsl.translation_unit"},
  };
  for (auto [source_name, structured_name] : module_map) {
    if (textualIdentity(semantic_module->getAttr(source_name)) !=
        textualIdentity(structured_module->getAttr(structured_name))) {
      error = "structured handoff module metadata differs from semantic "
              "source field '" +
              source_name.str() + "'";
      return false;
    }
  }

  auto semantic_iterator = semantic_module.getBody()->begin();
  auto structured_iterator = structured_module.getBody()->begin();
  for (; semantic_iterator != semantic_module.getBody()->end() &&
         structured_iterator != structured_module.getBody()->end();
       ++semantic_iterator, ++structured_iterator) {
    auto source_function = mlir::cast<mlir::func::FuncOp>(*semantic_iterator);
    auto source_gemm = mlir::cast<mlir_dialect::GemmOp>(
        source_function.getBody().front().front());
    auto structured_function =
        mlir::cast<mlir::func::FuncOp>(*structured_iterator);
    mlir::Builder builder(semantic_module.getContext());
    std::string selection_error;
    mlir::DictionaryAttr source_contract = selectedAttributes(
        builder, source_gemm.getOperation(), kGemmContractFields,
        "source mdsl.gemm", selection_error);
    if (!source_contract) {
      error = selection_error;
      return false;
    }
    const auto structured_contract =
        structured_function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.semantic_contract");
    const auto source_symbol =
        structured_function->getAttrOfType<mlir::StringAttr>(
            "mdsl.source_semantic_symbol");
    if (textualIdentity(structured_function.getFunctionType()) !=
            textualIdentity(source_function.getFunctionType()) ||
        textualIdentity(structured_function.getLoc()) !=
            textualIdentity(source_function.getLoc()) ||
        textualIdentity(structured_contract) !=
            textualIdentity(source_contract) ||
        !source_symbol ||
        source_symbol.getValue() != source_function.getName()) {
      error = "structured GEMM function is not an exact projection of its "
              "semantic source";
      return false;
    }
  }
  if (semantic_iterator != semantic_module.getBody()->end() ||
      structured_iterator != structured_module.getBody()->end()) {
    error = "structured GEMM handoff changed the number of semantic sites";
    return false;
  }
  return true;
}

} // namespace matcore::mdslc::mlir_bridge
