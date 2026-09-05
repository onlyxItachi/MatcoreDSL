#include "MatcoreStructuredHandoffCertificate.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;

int checks = 0;
int failures = 0;

void check(bool condition, std::string_view message) {
  ++checks;
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

void checkContains(std::string_view value, std::string_view expected,
                   std::string_view message) {
  check(value.find(expected) != std::string_view::npos, message);
}

bridge::StructuredHandoffCertificateProfileV1 testProfile() {
  return {
      "matcore-test-structured-handoff-v1",
      "matcore-test-structured-producer-v1",
      "inspection_only",
      "test.contract",
      "test_destination",
      "__matcore_test_structured_",
      "__matcore_test_semantic_",
      1,
  };
}

struct SiteSpec {
  std::string id;
  std::string contract_identity;
  std::int64_t extent;
  unsigned line;
};

mlir::DictionaryAttr handoffAttribute(
    mlir::Builder &builder,
    const bridge::StructuredHandoffCertificateProfileV1 &profile) {
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

mlir::OwningOpRef<mlir::ModuleOp>
makeStructuredModule(mlir::MLIRContext &context,
                     llvm::ArrayRef<SiteSpec> sites,
                     llvm::StringRef source_file = "certificate-source.mdsl") {
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::Builder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  const auto profile = testProfile();
  (*module)->setAttr("mdsl.analysis_only", builder.getBoolAttr(true));
  (*module)->setAttr("mdsl.capture_schema",
                     builder.getStringAttr("matcore-ir-v1"));
  (*module)->setAttr("mdsl.capture_version", builder.getI32IntegerAttr(1));
  (*module)->setAttr("mdsl.execution_authority",
                     builder.getStringAttr(profile.authority));
  (*module)->setAttr("mdsl.execution_intent",
                     builder.getStringAttr("generic"));
  (*module)->setAttr("mdsl.numerical_profile",
                     builder.getStringAttr("test-numerical-v1"));
  (*module)->setAttr("mdsl.producer",
                     builder.getStringAttr(profile.producer));
  (*module)->setAttr("mdsl.source_producer",
                     builder.getStringAttr("clang-libtooling-v1"));
  (*module)->setAttr("mdsl.source_bridge_schema",
                     builder.getStringAttr("matcore-mlir-semantic-v1"));
  (*module)->setAttr("mdsl.source_file", builder.getStringAttr(source_file));
  (*module)->setAttr("mdsl.source_semantic_version",
                     builder.getI32IntegerAttr(1));
  (*module)->setAttr("mdsl.structured_handoff_schema",
                     builder.getStringAttr(profile.schema));
  (*module)->setAttr("mdsl.structured_handoff_version",
                     builder.getI32IntegerAttr(profile.version));
  (*module)->setAttr("mdsl.translation_unit",
                     builder.getStringAttr(source_file));

  for (auto [ordinal, site] : llvm::enumerate(sites)) {
    const auto tensor =
        mlir::RankedTensorType::get({site.extent}, builder.getF32Type());
    const llvm::SmallVector<mlir::Type, 1> types = {tensor};
    const auto function_type = builder.getFunctionType(types, types);
    const auto function_location = mlir::FileLineColLoc::get(
        &context, source_file, site.line, 3);
    const std::string name =
        (profile.structured_function_prefix + site.id).str();
    auto function =
        mlir::func::FuncOp::create(function_location, name, function_type);
    function->setAttr("mdsl.capture_ordinal",
                      builder.getI64IntegerAttr(ordinal));
    function->setAttr(
        "mdsl.semantic_contract",
        builder.getDictionaryAttr({builder.getNamedAttr(
            "operation_identity",
            builder.getStringAttr(site.contract_identity))}));
    function->setAttr("mdsl.site_id", builder.getStringAttr(site.id));
    function->setAttr(
        "mdsl.source_semantic_symbol",
        builder.getStringAttr(
            (profile.semantic_function_prefix + site.id).str()));
    function->setAttr("mdsl.structured_handoff",
                      handoffAttribute(builder, profile));
    mlir::Block *entry = function.addEntryBlock();
    entry->getArgument(0).setLoc(mlir::FileLineColLoc::get(
        &context, source_file, site.line, 11));
    mlir::OpBuilder body_builder = mlir::OpBuilder::atBlockEnd(entry);
    body_builder.create<mlir::func::ReturnOp>(function_location,
                                               entry->getArgument(0));
    module->push_back(function);
  }
  return module;
}

llvm::SmallVector<mlir::func::FuncOp> functions(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::func::FuncOp> result;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    result.push_back(function);
  return result;
}

mlir::DictionaryAttr withField(mlir::Builder &builder,
                               mlir::DictionaryAttr dictionary,
                               llvm::StringRef field,
                               mlir::Attribute replacement) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  bool replaced = false;
  for (mlir::NamedAttribute attribute : dictionary) {
    if (attribute.getName().strref() == field) {
      attributes.push_back(builder.getNamedAttr(field, replacement));
      replaced = true;
    } else {
      attributes.push_back(attribute);
    }
  }
  check(replaced, "mutated certificate dictionary field must exist");
  return builder.getDictionaryAttr(attributes);
}

mlir::OwningOpRef<mlir::ModuleOp>
cloneAndAttach(mlir::ModuleOp source, std::string_view context_message) {
  mlir::OwningOpRef<mlir::ModuleOp> derived = source.clone();
  mlir::Builder builder(source.getContext());
  std::string error;
  check(bridge::attachDerivedStructuredHandoffSourceIdentityV1(
            source, *derived, testProfile(), builder, error),
        std::string(context_message) + " must attach derived identity");
  if (!error.empty())
    std::cerr << "attachment diagnostic: " << error << '\n';
  return derived;
}

using Mutation = std::function<void(mlir::ModuleOp, mlir::Builder &)>;

void expectStandaloneAndPairedRejected(llvm::ArrayRef<SiteSpec> sites,
                                       const Mutation &mutate,
                                       std::string_view message) {
  mlir::MLIRContext context;
  auto source = makeStructuredModule(context, sites);
  std::string error;
  check(bridge::verifyStructuredHandoffCertificateEnvelopeV1(
            *source, testProfile(), error),
        "mutation source certificate must verify before derivation");
  auto derived = cloneAndAttach(*source, "mutation fixture");
  mlir::Builder builder(&context);
  mutate(*derived, builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*derived)),
        std::string(message) + " must remain valid generic MLIR");
  error.clear();
  check(!bridge::verifyDerivedStructuredHandoffSourceEnvelopeV1(
             *derived, testProfile(), error),
        message);
  check(!error.empty(), "standalone rejection must provide a diagnostic");
  error.clear();
  check(!bridge::verifyDerivedStructuredHandoffMatchesSourceV1(
             *source, *derived, testProfile(), error),
        std::string(message) + " (paired verifier)");
  check(!error.empty(), "paired rejection must provide a diagnostic");
}

void testPositiveSingleAndMultiSite() {
  mlir::MLIRContext context;
  const std::vector<SiteSpec> single = {{"alpha", "contract-alpha", 2, 7}};
  auto source = makeStructuredModule(context, single);
  std::string error;
  check(bridge::verifyStructuredHandoffCertificateEnvelopeV1(
            *source, testProfile(), error),
        "operation-neutral single-site source certificate must verify");
  auto derived = cloneAndAttach(*source, "single-site fixture");
  check(!(*source)->hasAttr("mdsl.source_structured_fingerprint"),
        "derived attachment must not mutate its structured source");
  check(bridge::verifyDerivedStructuredHandoffSourceEnvelopeV1(
            *derived, testProfile(), error),
        "single-site derived certificate must verify standalone");
  check(bridge::verifyDerivedStructuredHandoffMatchesSourceV1(
            *source, *derived, testProfile(), error),
        "single-site derived certificate must pair with its exact source");

  auto single_functions = functions(*derived);
  check(single_functions.size() == 1,
        "single-site derived certificate must retain one site");
  if (single_functions.size() == 1) {
    bridge::VerifiedDerivedStructuredHandoffSiteV1 verified;
    check(bridge::verifyDerivedStructuredHandoffSourceIdentityV1(
              *derived, single_functions.front(), 0, testProfile(), verified,
              error),
          "single-site identity API must return a verified carrier");
    check(verified.source_structured_fingerprint &&
              verified.source_structured_fingerprint.getValue().starts_with(
                  "sha256:") &&
              verified.source_structured_fingerprint.getValue().size() == 71,
          "single-site identity must retain a canonical SHA-256 digest");
    check(verified.source_structured_function_type ==
              single_functions.front().getFunctionType(),
          "single-site identity must retain the original function type");
  }

  const std::vector<SiteSpec> multiple = {
      {"alpha", "contract-alpha", 2, 7},
      {"beta", "contract-beta", 5, 19},
  };
  auto multi_source = makeStructuredModule(context, multiple);
  auto multi_derived = cloneAndAttach(*multi_source, "multi-site fixture");
  check(bridge::verifyDerivedStructuredHandoffSourceEnvelopeV1(
            *multi_derived, testProfile(), error),
        "multi-site derived certificate must verify standalone");
  check(bridge::verifyDerivedStructuredHandoffMatchesSourceV1(
            *multi_source, *multi_derived, testProfile(), error),
        "multi-site derived certificate must pair in exact source order");
  const auto count = (*multi_derived)->getAttrOfType<mlir::IntegerAttr>(
      "mdsl.source_structured_site_count");
  const auto aggregate = (*multi_derived)->getAttrOfType<mlir::StringAttr>(
      "mdsl.source_structured_fingerprint");
  check(count && count.getInt() == 2 && aggregate &&
            aggregate.getValue().starts_with("sha256:") &&
            aggregate.getValue().size() == 71,
        "multi-site envelope must bind its ordered site count and digest");

  // This fixture deliberately contains only func.return. Passing here proves
  // that the generic layer binds identity but does not authenticate an
  // operation body. Every consumer must add its operation-specific verifier.
  const std::size_t operation_count = functions(*multi_derived)
                                          .front()
                                          .getBody()
                                          .front()
                                          .getOperations()
                                          .size();
  check(operation_count == 1,
        "generic certificate positive must not imply operation semantics");
}

void testSiteSetAndIdentityMutations() {
  const std::vector<SiteSpec> sites = {
      {"alpha", "contract-alpha", 2, 7},
      {"beta", "contract-beta", 5, 19},
  };
  expectStandaloneAndPairedRejected(
      sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto site_functions = functions(module);
        site_functions[1]->moveBefore(site_functions[0]);
      },
      "derived certificate must reject reordered sites");

  expectStandaloneAndPairedRejected(
      sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto site_functions = functions(module);
        site_functions[1].erase();
      },
      "derived certificate must reject a dropped site");

  expectStandaloneAndPairedRejected(
      sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto site_functions = functions(module);
        auto duplicate = mlir::cast<mlir::func::FuncOp>(
            site_functions[1]->clone());
        duplicate.setName("__matcore_test_structured_duplicate_control");
        module.push_back(duplicate);
      },
      "derived certificate must reject a duplicated site");

  expectStandaloneAndPairedRejected(
      sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto site_functions = functions(module);
        mlir::Attribute first = site_functions[0]->getAttr(
            "mdsl.source_structured_fingerprint");
        mlir::Attribute second = site_functions[1]->getAttr(
            "mdsl.source_structured_fingerprint");
        site_functions[0]->setAttr("mdsl.source_structured_fingerprint",
                                   second);
        site_functions[1]->setAttr("mdsl.source_structured_fingerprint",
                                   first);
      },
      "derived certificate must reject cross-site fingerprint substitution");

  expectStandaloneAndPairedRejected(
      {sites.front()},
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = functions(module).front();
        const auto replacement_tensor =
            mlir::RankedTensorType::get({17}, builder.getF32Type());
        const llvm::SmallVector<mlir::Type, 1> types = {replacement_tensor};
        function->setAttr(
            "mdsl.source_structured_function_type",
            mlir::TypeAttr::get(builder.getFunctionType(types, types)));
      },
      "derived certificate must reject a retained source-type mutation");

  expectStandaloneAndPairedRejected(
      {sites.front()},
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        functions(module).front()->setAttr(
            "mdsl.source_structured_fingerprint",
            builder.getStringAttr(
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"));
      },
      "derived certificate must reject a forged site fingerprint");

  expectStandaloneAndPairedRejected(
      sites,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr(
            "mdsl.source_structured_fingerprint",
            builder.getStringAttr(
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"));
      },
      "derived certificate must reject a forged aggregate fingerprint");

  expectStandaloneAndPairedRejected(
      sites,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.source_structured_site_count",
                        builder.getI64IntegerAttr(1));
      },
      "derived certificate must reject a forged aggregate site count");
}

void testAuthorityProvenanceAndLocationMutations() {
  const std::vector<SiteSpec> single = {{"alpha", "contract-alpha", 2, 7}};
  expectStandaloneAndPairedRejected(
      single,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.execution_authority",
                        builder.getStringAttr("generated_execution"));
      },
      "derived certificate must reject module authority escalation");

  expectStandaloneAndPairedRejected(
      single,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = functions(module).front();
        auto handoff = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.structured_handoff");
        function->setAttr(
            "mdsl.structured_handoff",
            withField(builder, handoff, "authority",
                      builder.getStringAttr("generated_execution")));
      },
      "derived certificate must reject site authority escalation");

  expectStandaloneAndPairedRejected(
      single,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.translation_unit",
                        builder.getStringAttr("alternate-source.mdsl"));
      },
      "derived certificate must reject module provenance drift");

  expectStandaloneAndPairedRejected(
      single,
      [](mlir::ModuleOp module, mlir::Builder &) {
        functions(module).front()->setLoc(mlir::FileLineColLoc::get(
            module.getContext(), "certificate-source.mdsl", 99, 3));
      },
      "derived certificate must reject function-location drift");

  expectStandaloneAndPairedRejected(
      single,
      [](mlir::ModuleOp module, mlir::Builder &) {
        functions(module).front().getArgument(0).setLoc(
            mlir::FileLineColLoc::get(module.getContext(),
                                       "certificate-source.mdsl", 7, 99));
      },
      "derived certificate must reject block-argument-location drift");

  mlir::MLIRContext context;
  auto source = makeStructuredModule(context, single);
  mlir::OwningOpRef<mlir::ModuleOp> target = (*source).clone();
  auto unsafe_profile = testProfile();
  unsafe_profile.authority = "generated_execution";
  mlir::Builder builder(&context);
  std::string error;
  check(!bridge::attachDerivedStructuredHandoffSourceIdentityV1(
             *source, *target, unsafe_profile, builder, error),
        "derived attachment must reject a non-inspection profile");
  checkContains(error, "non-inspection authority",
                "unsafe-profile rejection must explain the authority limit");
}

void testExactSourcePairing() {
  mlir::MLIRContext context;
  const std::vector<SiteSpec> original_spec = {
      {"alpha", "contract-original", 2, 7}};
  auto original = makeStructuredModule(context, original_spec);
  auto derived = cloneAndAttach(*original, "exact-pairing fixture");
  std::string error;
  check(bridge::verifyDerivedStructuredHandoffSourceEnvelopeV1(
            *derived, testProfile(), error),
        "exact-pairing fixture must be self-consistent standalone");

  const std::vector<SiteSpec> alternate_contract_spec = {
      {"alpha", "contract-alternate", 2, 7}};
  auto alternate_contract =
      makeStructuredModule(context, alternate_contract_spec);
  check(bridge::verifyStructuredHandoffCertificateEnvelopeV1(
            *alternate_contract, testProfile(), error),
        "same-shaped alternate contract source must be structurally valid");
  check(!bridge::verifyDerivedStructuredHandoffMatchesSourceV1(
             *alternate_contract, *derived, testProfile(), error),
        "paired verification must reject a same-shaped alternate contract");
  check(!error.empty(),
        "alternate-contract rejection must provide a diagnostic");
  check(bridge::verifyDerivedStructuredHandoffSourceEnvelopeV1(
            *derived, testProfile(), error),
        "failed alternate pairing must not invalidate standalone identity");

  auto alternate_provenance = makeStructuredModule(
      context, original_spec, "same-shape-alternate.mdsl");
  check(bridge::verifyStructuredHandoffCertificateEnvelopeV1(
            *alternate_provenance, testProfile(), error),
        "same-shaped alternate provenance source must be structurally valid");
  check(!bridge::verifyDerivedStructuredHandoffMatchesSourceV1(
             *alternate_provenance, *derived, testProfile(), error),
        "paired verification must reject a same-shaped alternate source");
  checkContains(error, "source field",
                "alternate-source rejection must identify module pairing");
}

} // namespace

int main() {
  testPositiveSingleAndMultiSite();
  testSiteSetAndIdentityMutations();
  testAuthorityProvenanceAndLocationMutations();
  testExactSourcePairing();
  if (failures != 0) {
    std::cerr << "Derived structured certificate tests: " << failures
              << " of " << checks << " checks failed\n";
    return 1;
  }
  std::cout << "Derived structured certificate tests: " << checks
            << " checks, 0 failures\n";
  return 0;
}
