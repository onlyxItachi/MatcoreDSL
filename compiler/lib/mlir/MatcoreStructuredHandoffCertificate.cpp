#include "MatcoreStructuredHandoffCertificate.h"

#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <string>
#include <utility>

namespace matcore::mdslc::mlir_bridge {
namespace {

constexpr llvm::StringLiteral kSemanticBridgeSchema =
    "matcore-mlir-semantic-v1";

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

constexpr llvm::StringLiteral kStructuredFunctionFields[] = {
    "mdsl.capture_ordinal",       "mdsl.semantic_contract",
    "mdsl.site_id",               "mdsl.source_semantic_symbol",
    "mdsl.structured_handoff",
};

constexpr llvm::StringLiteral kSemanticFunctionFields[] = {
    "mdsl.capture_ordinal", "mdsl.site_id"};

bool verifyProfile(const StructuredHandoffCertificateProfileV1 &profile,
                   std::string &error) {
  if (profile.schema.empty() || profile.producer.empty() ||
      profile.authority != "inspection_only" ||
      profile.source_operation.empty() || profile.destination_rule.empty() ||
      profile.structured_function_prefix.empty() ||
      profile.semantic_function_prefix.empty() || profile.version != 1) {
    error = "structured handoff certificate profile is incomplete, "
            "unversioned, or requests non-inspection authority";
    return false;
  }
  return true;
}

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

template <typename MlirValue>
std::string textualIdentity(MlirValue value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  stream.flush();
  return text;
}

mlir::DictionaryAttr handoffAttribute(
    mlir::Builder &builder,
    const StructuredHandoffCertificateProfileV1 &profile) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("authority",
                            builder.getStringAttr(profile.authority)),
       builder.getNamedAttr("destination",
                            builder.getStringAttr(profile.destination_rule)),
       builder.getNamedAttr("source_operation",
                            builder.getStringAttr(profile.source_operation)),
       builder.getNamedAttr("version",
                            builder.getI32IntegerAttr(profile.version))});
}

bool verifyHandoffAttribute(
    mlir::DictionaryAttr handoff,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error) {
  constexpr llvm::StringLiteral fields[] = {
      "authority", "destination", "source_operation", "version"};
  if (!requireExactNames(handoff, fields, "structured function handoff", error))
    return false;
  const auto authority = handoff.getAs<mlir::StringAttr>("authority");
  const auto destination = handoff.getAs<mlir::StringAttr>("destination");
  const auto source_operation =
      handoff.getAs<mlir::StringAttr>("source_operation");
  const auto version = handoff.getAs<mlir::IntegerAttr>("version");
  if (!authority || authority.getValue() != profile.authority ||
      !destination || destination.getValue() != profile.destination_rule ||
      !source_operation ||
      source_operation.getValue() != profile.source_operation || !version ||
      !version.getType().isSignlessInteger(32) ||
      version.getInt() != profile.version) {
    error = "structured function handoff contract is incomplete or invalid";
    return false;
  }
  return true;
}

bool verifyStructuredModuleFields(
    mlir::ModuleOp module,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error) {
  if (!requireExactNames(module->getAttrDictionary(), kStructuredModuleFields,
                         "structured handoff module", error))
    return false;
  const auto analysis_only =
      module->getAttrOfType<mlir::BoolAttr>("mdsl.analysis_only");
  const auto source_producer =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_producer");
  const auto source_file =
      module->getAttrOfType<mlir::StringAttr>("mdsl.source_file");
  const auto translation_unit =
      module->getAttrOfType<mlir::StringAttr>("mdsl.translation_unit");
  const auto numerical_profile =
      module->getAttrOfType<mlir::StringAttr>("mdsl.numerical_profile");
  if (!analysis_only || !analysis_only.getValue() || !source_producer ||
      (source_producer.getValue() != "clang-libtooling-v1" &&
       source_producer.getValue() != "clang-ast-json-bootstrap-v0") ||
      !source_file || !source_file.getValue().ends_with(".mdsl") ||
      !translation_unit || translation_unit.getValue().empty() ||
      !numerical_profile || numerical_profile.getValue().empty()) {
    error = "structured handoff module capture identity is invalid";
    return false;
  }
  return requireString(module, "mdsl.capture_schema", "matcore-ir-v1",
                       "structured handoff module", error) &&
         requireI32(module, "mdsl.capture_version", ir::v1::kMatcoreIrVersion,
                    "structured handoff module", error) &&
         requireString(module, "mdsl.execution_intent", "generic",
                       "structured handoff module", error) &&
         requireString(module, "mdsl.producer", profile.producer,
                       "structured handoff module", error) &&
         requireString(module, "mdsl.execution_authority", profile.authority,
                       "structured handoff module", error) &&
         requireString(module, "mdsl.source_bridge_schema",
                       kSemanticBridgeSchema, "structured handoff module",
                       error) &&
         requireString(module, "mdsl.structured_handoff_schema", profile.schema,
                       "structured handoff module", error) &&
         requireI32(module, "mdsl.structured_handoff_version", profile.version,
                    "structured handoff module", error) &&
         requireI32(module, "mdsl.source_semantic_version",
                    kMatcoreSemanticModuleVersion,
                    "structured handoff module", error);
}

void appendFingerprintField(std::string &bytes, llvm::StringRef name,
                            llvm::StringRef value) {
  bytes += std::to_string(name.size());
  bytes.push_back(':');
  bytes.append(name.data(), name.size());
  bytes.push_back('=');
  bytes += std::to_string(value.size());
  bytes.push_back(':');
  bytes.append(value.data(), value.size());
  bytes.push_back('\n');
}

std::string hashFingerprintBytes(llvm::StringRef bytes) {
  const std::array<std::uint8_t, 32> digest =
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes));
  return "sha256:" + llvm::toHex(llvm::ArrayRef(digest), true);
}

std::string computeSemanticFingerprint(
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> module_fields,
    llvm::StringRef source_operation, llvm::StringRef source_symbol,
    mlir::StringAttr site, mlir::func::FuncOp carrier_function,
    mlir::DictionaryAttr contract, std::string &error) {
  if (source_operation.empty() || source_symbol.empty() || !site ||
      !carrier_function || !contract) {
    error = "semantic fingerprint input is incomplete";
    return {};
  }
  if (!llvm::hasSingleElement(carrier_function.getBody())) {
    error = "semantic fingerprint requires one entry block";
    return {};
  }
  mlir::Block &entry = carrier_function.getBody().front();
  if (entry.getNumArguments() != carrier_function.getNumArguments()) {
    error = "semantic fingerprint entry arguments do not match the function "
            "type";
    return {};
  }
  std::string bytes = "matcore-structured-semantic-fingerprint-v1\n";
  for (const auto &[name, value] : module_fields) {
    if (!value) {
      error = "semantic fingerprint is missing module field '" + name.str() +
              "'";
      return {};
    }
    appendFingerprintField(bytes, name, textualIdentity(value));
  }
  appendFingerprintField(bytes, "source_operation", source_operation);
  appendFingerprintField(bytes, "source_symbol", source_symbol);
  appendFingerprintField(bytes, "site_id", site.getValue());
  appendFingerprintField(bytes, "function_type",
                         textualIdentity(carrier_function.getFunctionType()));
  appendFingerprintField(bytes, "location",
                         textualIdentity(carrier_function.getLoc()));
  appendFingerprintField(bytes, "entry_argument_count",
                         std::to_string(entry.getNumArguments()));
  for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
    const std::string field =
        (llvm::Twine("entry_argument_location[") + llvm::Twine(index) + "]")
            .str();
    appendFingerprintField(bytes, field, textualIdentity(argument.getLoc()));
  }
  appendFingerprintField(bytes, "semantic_contract", textualIdentity(contract));
  return hashFingerprintBytes(bytes);
}

bool verifyExactEntryArgumentLocations(mlir::func::FuncOp source,
                                       mlir::func::FuncOp structured,
                                       std::string &error) {
  if (!llvm::hasSingleElement(source.getBody()) ||
      !llvm::hasSingleElement(structured.getBody())) {
    error = "paired semantic functions require one entry block";
    return false;
  }
  mlir::Block &source_entry = source.getBody().front();
  mlir::Block &structured_entry = structured.getBody().front();
  if (source_entry.getNumArguments() != structured_entry.getNumArguments()) {
    error = "structured entry-argument count differs from its semantic source";
    return false;
  }
  for (auto [source_argument, structured_argument] :
       llvm::zip_equal(source_entry.getArguments(),
                       structured_entry.getArguments())) {
    if (textualIdentity(source_argument.getLoc()) !=
        textualIdentity(structured_argument.getLoc())) {
      error = "structured entry-argument location differs from its semantic "
              "source";
      return false;
    }
  }
  return true;
}

} // namespace

bool verifyStructuredHandoffSourceEnvelopeV1(mlir::ModuleOp source,
                                              std::string &error) {
  error.clear();
  if (!source) {
    error = "semantic source module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(source))) {
    error = "semantic source envelope failed upstream MLIR verification";
    return false;
  }
  if (!requireExactNames(source->getAttrDictionary(), kSemanticModuleFields,
                         "semantic source module", error) ||
      !requireString(source, "mdsl.bridge_schema", kSemanticBridgeSchema,
                     "semantic source module", error))
    return false;
  for (mlir::Operation &operation : source.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function) {
      error = "semantic source module may contain only func.func sites";
      return false;
    }
    if (!requireExactNames(function->getDiscardableAttrDictionary(),
                           kSemanticFunctionFields,
                           "semantic source function", error))
      return false;
    if (function.getArgAttrsAttr() || function.getResAttrsAttr() ||
        function.getNoInline() || function.getSymVisibilityAttr()) {
      error = "semantic source functions may not carry argument/result "
              "optimizer attributes, no_inline, or explicit visibility";
      return false;
    }
  }
  return true;
}

mlir::DictionaryAttr selectExactSemanticContractV1(
    mlir::Builder &builder, mlir::Operation *operation,
    llvm::ArrayRef<llvm::StringLiteral> names, llvm::StringRef context,
    std::string &error) {
  error.clear();
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
  if (!requireExactNames(operation->getAttrDictionary(), names, context, error))
    return {};
  return builder.getDictionaryAttr(selected);
}

bool initializeStructuredHandoffCertificateV1(
    mlir::ModuleOp source, mlir::ModuleOp target, mlir::Builder &builder,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error) {
  error.clear();
  if (!verifyProfile(profile, error))
    return false;
  if (!target || !target.getBody()->empty()) {
    error = "structured handoff certificate requires an empty target module";
    return false;
  }
  if (!verifyStructuredHandoffSourceEnvelopeV1(source, error))
    return false;
  constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> module_map[] = {
      {"mdsl.capture_schema", "mdsl.capture_schema"},
      {"mdsl.capture_version", "mdsl.capture_version"},
      {"mdsl.execution_intent", "mdsl.execution_intent"},
      {"mdsl.numerical_profile", "mdsl.numerical_profile"},
      {"mdsl.producer", "mdsl.source_producer"},
      {"mdsl.source_file", "mdsl.source_file"},
      {"mdsl.translation_unit", "mdsl.translation_unit"},
      {"mdsl.bridge_schema", "mdsl.source_bridge_schema"},
      {"mdsl.semantic_version", "mdsl.source_semantic_version"},
  };
  for (auto [source_name, target_name] : module_map)
    target->setAttr(target_name, source->getAttr(source_name));
  target->setAttr("mdsl.analysis_only", builder.getBoolAttr(true));
  target->setAttr("mdsl.producer", builder.getStringAttr(profile.producer));
  target->setAttr("mdsl.execution_authority",
                  builder.getStringAttr(profile.authority));
  target->setAttr("mdsl.structured_handoff_schema",
                  builder.getStringAttr(profile.schema));
  target->setAttr("mdsl.structured_handoff_version",
                  builder.getI32IntegerAttr(profile.version));
  return true;
}

bool attachStructuredHandoffSiteCertificateV1(
    mlir::func::FuncOp structured_function,
    mlir::func::FuncOp source_function, mlir::DictionaryAttr semantic_contract,
    llvm::StringRef site_id, mlir::Builder &builder,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error) {
  error.clear();
  if (!verifyProfile(profile, error) || !structured_function ||
      !source_function || !semantic_contract || site_id.empty()) {
    if (error.empty())
      error = "structured handoff site certificate input is incomplete";
    return false;
  }
  if (!source_function->getParentOfType<mlir::ModuleOp>() ||
      structured_function.getContext() != builder.getContext() ||
      source_function.getContext() != builder.getContext()) {
    error = "structured handoff source must be attached and both functions "
            "must use the builder's MLIR context";
    return false;
  }
  const auto source_site =
      source_function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  if (!source_site || source_site.getValue() != site_id ||
      !llvm::hasSingleElement(source_function.getBody()) ||
      source_function.getBody().front().empty() ||
      source_function.getBody().front().front().getName().getStringRef() !=
          profile.source_operation) {
    error = "structured handoff site does not match the declared source "
            "operation/site identity";
    return false;
  }
  structured_function->setAttr(
      "mdsl.capture_ordinal",
      source_function->getAttr("mdsl.capture_ordinal"));
  structured_function->setAttr("mdsl.site_id", builder.getStringAttr(site_id));
  structured_function->setAttr("mdsl.source_semantic_symbol",
                               builder.getStringAttr(source_function.getName()));
  structured_function->setAttr("mdsl.semantic_contract", semantic_contract);
  structured_function->setAttr("mdsl.structured_handoff",
                               handoffAttribute(builder, profile));
  return true;
}

bool verifyStructuredHandoffCertificateEnvelopeV1(
    mlir::ModuleOp structured_module,
    const StructuredHandoffCertificateProfileV1 &profile,
    std::string &error) {
  error.clear();
  if (!verifyProfile(profile, error))
    return false;
  if (!structured_module) {
    error = "structured handoff module is null";
    return false;
  }
  if (mlir::failed(mlir::verify(structured_module))) {
    error = "structured handoff certificate failed upstream MLIR verification";
    return false;
  }
  if (!verifyStructuredModuleFields(structured_module, profile, error) ||
      structured_module.getBody()->empty()) {
    if (error.empty())
      error = "structured handoff module requires at least one site";
    return false;
  }
  std::size_t ordinal = 0;
  for (mlir::Operation &operation :
       structured_module.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function) {
      error = "structured handoff module may contain only func.func sites";
      return false;
    }
    VerifiedStructuredHandoffSiteV1 verified;
    if (!verifyStructuredHandoffSiteCertificateV1(function, ordinal, profile,
                                                   verified, error))
      return false;
    ++ordinal;
  }
  return true;
}

bool verifyStructuredHandoffSiteCertificateV1(
    mlir::func::FuncOp function, std::size_t expected_ordinal,
    const StructuredHandoffCertificateProfileV1 &profile,
    VerifiedStructuredHandoffSiteV1 &verified, std::string &error) {
  error.clear();
  verified = {};
  if (!verifyProfile(profile, error))
    return false;
  if (!function) {
    error = "structured handoff function is null";
    return false;
  }
  if (!function->getParentOfType<mlir::ModuleOp>() ||
      mlir::failed(mlir::verify(function))) {
    error = "structured handoff function must be attached and pass upstream "
            "MLIR verification";
    return false;
  }
  if (!requireExactNames(function->getDiscardableAttrDictionary(),
                         kStructuredFunctionFields,
                         "structured handoff function", error))
    return false;
  if (function.getArgAttrsAttr() || function.getResAttrsAttr() ||
      function.getNoInline() || function.getSymVisibilityAttr()) {
    error = "structured handoff functions may not carry argument/result "
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
  if (!site || site.getValue().empty() || !ordinal ||
      !ordinal.getType().isSignlessInteger(64) ||
      ordinal.getInt() != static_cast<std::int64_t>(expected_ordinal) ||
      !source_symbol || !contract || contract.empty() || !handoff ||
      !function.isPublic()) {
    error = "structured handoff function identity is incomplete or unordered";
    return false;
  }
  const std::string expected_name =
      (llvm::Twine(profile.structured_function_prefix) + site.getValue()).str();
  const std::string expected_source =
      (llvm::Twine(profile.semantic_function_prefix) + site.getValue()).str();
  if (function.getName() != expected_name ||
      source_symbol.getValue() != expected_source) {
    error = "structured function/source symbols must match the exact semantic "
            "site identity";
    return false;
  }
  if (!verifyHandoffAttribute(handoff, profile, error))
    return false;
  verified.function = function;
  verified.site_id = site;
  verified.source_semantic_symbol = source_symbol;
  verified.semantic_contract = contract;
  return true;
}

bool verifyStructuredHandoffCertificateMatchesSourceV1(
    mlir::ModuleOp semantic_module, mlir::ModuleOp structured_module,
    const StructuredHandoffCertificateProfileV1 &profile,
    SemanticContractSelectorV1 contract_selector, std::string &error) {
  error.clear();
  if (!verifyStructuredHandoffSourceEnvelopeV1(semantic_module, error) ||
      !verifyStructuredHandoffCertificateEnvelopeV1(structured_module, profile,
                                                    error))
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

  auto source_iterator = semantic_module.getBody()->begin();
  auto structured_iterator = structured_module.getBody()->begin();
  for (std::size_t ordinal = 0;
       source_iterator != semantic_module.getBody()->end() &&
       structured_iterator != structured_module.getBody()->end();
       ++source_iterator, ++structured_iterator, ++ordinal) {
    auto source_function = mlir::cast<mlir::func::FuncOp>(*source_iterator);
    auto structured_function =
        mlir::cast<mlir::func::FuncOp>(*structured_iterator);
    if (source_function->getParentOp() != semantic_module.getOperation() ||
        structured_function->getParentOp() !=
            structured_module.getOperation()) {
      error = "paired structured handoff functions must be direct members of "
              "their supplied modules";
      return false;
    }
    mlir::DictionaryAttr source_contract =
        contract_selector(source_function, error);
    if (!source_contract)
      return false;
    VerifiedStructuredHandoffSiteV1 verified;
    if (!verifyStructuredHandoffSiteCertificateV1(
            structured_function, ordinal, profile, verified, error))
      return false;
    const auto source_site =
        source_function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
    if (!source_site ||
        verified.site_id.getValue() != source_site.getValue() ||
        verified.source_semantic_symbol.getValue() != source_function.getName() ||
        textualIdentity(structured_function.getFunctionType()) !=
            textualIdentity(source_function.getFunctionType()) ||
        textualIdentity(structured_function.getLoc()) !=
            textualIdentity(source_function.getLoc()) ||
        textualIdentity(verified.semantic_contract) !=
            textualIdentity(source_contract)) {
      error = "structured function is not an exact projection of its semantic "
              "source";
      return false;
    }
    if (!verifyExactEntryArgumentLocations(source_function,
                                           structured_function, error))
      return false;
    std::string source_fingerprint = computeSourceSemanticFingerprintV1(
        semantic_module, source_function, source_contract,
        profile.source_operation, error);
    std::string structured_fingerprint = computeStructuredSemanticFingerprintV1(
        structured_module, structured_function, profile.source_operation,
        error);
    if (source_fingerprint.empty() || structured_fingerprint.empty() ||
        source_fingerprint != structured_fingerprint) {
      if (error.empty())
        error = "structured semantic fingerprint differs from its source";
      return false;
    }
  }
  if (source_iterator != semantic_module.getBody()->end() ||
      structured_iterator != structured_module.getBody()->end()) {
    error = "structured handoff changed the number of semantic sites";
    return false;
  }
  return true;
}

std::string computeSourceSemanticFingerprintV1(
    mlir::ModuleOp semantic_module, mlir::func::FuncOp source_function,
    mlir::DictionaryAttr semantic_contract, llvm::StringRef source_operation,
    std::string &error) {
  error.clear();
  if (!semantic_module || !source_function) {
    error = "source semantic fingerprint input is null";
    return {};
  }
  if (source_function->getParentOp() != semantic_module.getOperation()) {
    error = "source semantic fingerprint function is not a direct member of "
            "the supplied module";
    return {};
  }
  if (!verifyStructuredHandoffSourceEnvelopeV1(semantic_module, error))
    return {};
  constexpr llvm::StringLiteral names[] = {
      "mdsl.capture_schema",     "mdsl.capture_version",
      "mdsl.execution_intent",   "mdsl.numerical_profile",
      "mdsl.producer",           "mdsl.bridge_schema",
      "mdsl.source_file",        "mdsl.semantic_version",
      "mdsl.translation_unit",
  };
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>, 10> fields;
  for (llvm::StringRef name : names)
    fields.emplace_back(name, semantic_module->getAttr(name));
  fields.emplace_back("mdsl.capture_ordinal",
                      source_function->getAttr("mdsl.capture_ordinal"));
  const auto site =
      source_function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  if (!llvm::hasSingleElement(source_function.getBody()) ||
      source_function.getBody().front().empty() ||
      source_function.getBody().front().front().getName().getStringRef() !=
          source_operation) {
    error = "semantic fingerprint source operation identity does not match";
    return {};
  }
  if (source_function.getBody().front().front().getAttrDictionary() !=
      semantic_contract) {
    error = "semantic fingerprint contract is not the exact source operation "
            "contract";
    return {};
  }
  return computeSemanticFingerprint(
      fields, source_operation, source_function.getName(), site,
      source_function, semantic_contract, error);
}

std::string computeStructuredSemanticFingerprintV1(
    mlir::ModuleOp structured_module,
    mlir::func::FuncOp structured_function, llvm::StringRef source_operation,
    std::string &error) {
  error.clear();
  if (!structured_module || !structured_function) {
    error = "structured semantic fingerprint input is null";
    return {};
  }
  if (structured_function->getParentOp() !=
      structured_module.getOperation()) {
    error = "structured semantic fingerprint function is not a direct member "
            "of the supplied module";
    return {};
  }
  if (mlir::failed(mlir::verify(structured_module))) {
    error = "structured semantic fingerprint module failed upstream MLIR "
            "verification";
    return {};
  }
  if (!requireExactNames(structured_function->getDiscardableAttrDictionary(),
                         kStructuredFunctionFields,
                         "structured semantic fingerprint function", error))
    return {};
  constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> names[] = {
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
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Attribute>, 10> fields;
  for (auto [canonical_name, structured_name] : names)
    fields.emplace_back(canonical_name,
                        structured_module->getAttr(structured_name));
  fields.emplace_back("mdsl.capture_ordinal",
                      structured_function->getAttr("mdsl.capture_ordinal"));
  const auto source_symbol = structured_function->getAttrOfType<mlir::StringAttr>(
      "mdsl.source_semantic_symbol");
  const auto site =
      structured_function->getAttrOfType<mlir::StringAttr>("mdsl.site_id");
  const auto contract = structured_function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.semantic_contract");
  const auto handoff = structured_function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.structured_handoff");
  const auto encoded_source_operation =
      handoff ? handoff.getAs<mlir::StringAttr>("source_operation")
              : mlir::StringAttr{};
  if (!encoded_source_operation ||
      encoded_source_operation.getValue() != source_operation) {
    error = "structured semantic fingerprint source operation does not match "
            "its certificate";
    return {};
  }
  return computeSemanticFingerprint(
      fields, source_operation,
      source_symbol ? source_symbol.getValue() : llvm::StringRef{}, site,
      structured_function, contract, error);
}

} // namespace matcore::mdslc::mlir_bridge
