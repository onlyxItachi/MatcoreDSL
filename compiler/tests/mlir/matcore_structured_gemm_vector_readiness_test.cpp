#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreStructuredGemmVectorReadiness.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;
namespace lowering = matcore::mdslc::mlir_lowering;
namespace v1 = matcore::mdslc::ir::v1;

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

std::string readFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  check(stream.good() || stream.eof(), "fixture must be readable");
  return contents.str();
}

v1::Module readCapture() {
  const std::string json = readFile(std::string(MDSLC_IR_TEST_SOURCE_DIR) +
                                    "/gemm_capture.v1.golden.json");
  v1::Module module;
  std::string error;
  check(v1::parseAndVerifyJson(json, module, error),
        "reviewed Matcore IR v1 capture must parse and verify");
  return module;
}

void makeStatic(v1::TensorValue &value, std::uint64_t rows,
                std::uint64_t columns) {
  value.type.shape = {v1::ScalarExpr::staticValue(rows),
                      v1::ScalarExpr::staticValue(columns)};
  value.type.strides = {v1::ScalarExpr::staticValue(columns),
                        v1::ScalarExpr::staticValue(1)};
}

void makeStaticGemm(v1::Module &capture, std::uint64_t m, std::uint64_t k,
                    std::uint64_t n) {
  check(capture.operations.size() == 1,
        "static vector fixture must contain exactly one GEMM");
  if (capture.operations.size() != 1)
    return;
  makeStatic(capture.operations[0].operands[0], m, k);
  makeStatic(capture.operations[0].operands[1], k, n);
  makeStatic(capture.operations[0].output, m, n);
}

void makeStaticGemmOperation(v1::Operation &operation, std::uint64_t m,
                             std::uint64_t k, std::uint64_t n) {
  makeStatic(operation.operands[0], m, k);
  makeStatic(operation.operands[1], k, n);
  makeStatic(operation.output, m, n);
}

v1::Module makeTwoSiteStaticCapture(v1::Module capture) {
  makeStaticGemm(capture, 2, 3, 4);
  v1::Operation second = capture.operations.front();
  second.site_id = "mc_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  constexpr std::uint64_t offset_delta = 1000;
  second.source.offset += offset_delta;
  second.source.line += 10;
  second.call_range.begin += offset_delta;
  second.call_range.end += offset_delta;
  for (auto &range : second.argument_ranges) {
    range.begin += offset_delta;
    range.end += offset_delta;
  }
  second.operands[0].source_expression = "A_second";
  second.operands[1].source_expression = "B_second";
  second.output.source_expression = "C_second";
  makeStaticGemmOperation(second, 3, 2, 5);
  capture.operations.push_back(std::move(second));
  return capture;
}

struct PipelineFixture {
  mlir::MLIRContext context;
  bridge::BridgeResult semantic;
  bridge::StructuredGemmHandoffResultV1 structured;

  explicit PipelineFixture(const v1::Module &capture) {
    semantic = bridge::bridgeV1ToMatcoreMlir(
        capture, context, bridge::explicitGemmF32V1BridgeContext());
    if (semantic)
      structured = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  }
};

template <typename OpTy> OpTy findOne(mlir::ModuleOp module) {
  OpTy result;
  module.walk([&](OpTy operation) {
    if (!result)
      result = operation;
  });
  return result;
}

mlir::arith::ConstantOp vectorConstant(mlir::ModuleOp module) {
  mlir::arith::ConstantOp result;
  module.walk([&](mlir::arith::ConstantOp operation) {
    if (!result && mlir::isa<mlir::VectorType>(operation.getType()))
      result = operation;
  });
  return result;
}

mlir::func::FuncOp function(mlir::ModuleOp module) {
  return findOne<mlir::func::FuncOp>(module);
}

std::vector<mlir::func::FuncOp> functions(mlir::ModuleOp module) {
  std::vector<mlir::func::FuncOp> result;
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>())
    result.push_back(candidate);
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
  check(replaced, "mutated vector-certificate field must exist");
  return builder.getDictionaryAttr(attributes);
}

mlir::DictionaryAttr withExtraField(mlir::Builder &builder,
                                    mlir::DictionaryAttr dictionary,
                                    llvm::StringRef field,
                                    mlir::Attribute value) {
  llvm::SmallVector<mlir::NamedAttribute> attributes(dictionary.begin(),
                                                     dictionary.end());
  attributes.push_back(builder.getNamedAttr(field, value));
  return builder.getDictionaryAttr(attributes);
}

using VectorMutation = std::function<void(mlir::ModuleOp, mlir::Builder &)>;

void expectVectorRejected(const v1::Module &capture,
                          const VectorMutation &mutate,
                          std::string_view message) {
  PipelineFixture fixture(capture);
  check(static_cast<bool>(fixture.semantic),
        "vector mutation fixture must bridge before damage");
  check(static_cast<bool>(fixture.structured),
        "vector mutation fixture must structure before damage");
  if (!fixture.semantic || !fixture.structured)
    return;
  auto vector =
      bridge::deriveStructuredGemmVectorReadinessV1(*fixture.structured.module);
  check(static_cast<bool>(vector),
        "vector mutation fixture must derive before damage");
  if (!vector)
    return;

  mlir::Builder builder(&fixture.context);
  mutate(*vector.module, builder);
  mlir::ScopedDiagnosticHandler silence(
      &fixture.context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*vector.module)),
        std::string(message) + " (mutation must remain generic MLIR-valid)");
  std::string error;
  check(!bridge::verifyStructuredGemmVectorReadinessV1(*vector.module, error),
        message);
  check(!error.empty(), "vector rejection must provide a diagnostic");
}

void testStaticVectorContract(v1::Module capture) {
  makeStaticGemm(capture, 2, 3, 4);
  std::string error;
  check(v1::verify(capture, error),
        "constructed non-square 2x3 by 3x4 vector specimen must verify");
  PipelineFixture fixture(capture);
  check(static_cast<bool>(fixture.semantic),
        "constructed static vector specimen must bridge to semantic MLIR");
  check(static_cast<bool>(fixture.structured),
        "constructed static specimen must derive certified structured MLIR");
  if (!fixture.semantic || !fixture.structured)
    return;

  const std::string structured_before =
      bridge::serializeDeterministicMlir(*fixture.structured.module);
  auto vector =
      bridge::deriveStructuredGemmVectorReadinessV1(*fixture.structured.module);
  check(
      static_cast<bool>(vector),
      "verifier-approved constructed static GEMM must derive vector readiness");
  if (!vector) {
    std::cerr << vector.error << '\n';
    return;
  }
  check(bridge::serializeDeterministicMlir(*fixture.structured.module) ==
            structured_before,
        "vector derivation must not mutate the structured source");
  check(bridge::verifyStructuredGemmVectorReadinessV1(*vector.module, error),
        "static vector-readiness module must verify standalone");
  check(bridge::verifyStructuredGemmVectorReadinessMatchesV1(
            *fixture.structured.module, *vector.module, error),
        "vector readiness must pair with its exact structured source");

  auto contract = findOne<mlir::vector::ContractionOp>(*vector.module);
  auto lhs_read = findOne<mlir::vector::TransferReadOp>(*vector.module);
  std::vector<mlir::vector::TransferReadOp> reads;
  vector.module->walk(
      [&](mlir::vector::TransferReadOp read) { reads.push_back(read); });
  auto write = findOne<mlir::vector::TransferWriteOp>(*vector.module);
  auto vector_function = function(*vector.module);
  check(contract && lhs_read && reads.size() == 2 && write && vector_function,
        "vector result must contain two input transfers, one contract, and "
        "one output transfer");
  if (!contract || reads.size() != 2 || !write || !vector_function)
    return;
  check(reads[0].getBase() == vector_function.getArgument(0) &&
            reads[1].getBase() == vector_function.getArgument(1),
        "only original lhs/rhs may be read");
  check(write.getBase() == vector_function.getArgument(2),
        "contract result must write the original output tensor");
  check(contract.getAcc() == vectorConstant(*vector.module).getResult(),
        "vector.contract accumulator must be the explicit zero vector");
  check(!findOne<mlir::linalg::FillOp>(*vector.module) &&
            !findOne<mlir::linalg::MatmulOp>(*vector.module),
        "upstream transform must consume the certified fill/matmul pair");

  const auto source_contract =
      function(*fixture.structured.module)
          ->getAttrOfType<mlir::DictionaryAttr>("mdsl.semantic_contract");
  const auto retained_contract =
      vector_function->getAttrOfType<mlir::DictionaryAttr>(
          "mdsl.semantic_contract");
  check(source_contract == retained_contract,
        "vector stage must retain the exact semantic dictionary in-context");
  const auto readiness = vector_function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.vector_readiness");
  check(readiness &&
            readiness.getAs<mlir::StringAttr>("semantic_contract").getValue() ==
                "retained_exactly",
        "vector stage must declare exact contract retention");
  const auto unconsumed =
      readiness.getAs<mlir::ArrayAttr>("unconsumed_requirements");
  check(unconsumed && unconsumed.size() == 8,
        "vector stage must enumerate requirements it did not consume");
  check(readiness.getAs<mlir::DictionaryAttr>("numerical_permissions") ==
            retained_contract.getAs<mlir::DictionaryAttr>("numerical"),
        "vector ledger must retain the exact unconsumed numerical permissions");

  const std::string schedule =
      bridge::structuredGemmVectorReadinessTransformV1();
  checkContains(schedule,
                "transform.structured.vectorize_children_and_apply_patterns",
                "checkpoint must expose the exact upstream Transform op");
  check(schedule.find("tile") == std::string::npos &&
            schedule.find("vector_sizes") == std::string::npos &&
            schedule.find("target") == std::string::npos,
        "inspection schedule must encode no tile, width, or target choice");

  const std::string text = bridge::serializeDeterministicMlir(*vector.module);
  const std::string golden =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) +
               "/gemm_capture.vector-readiness.golden.mlir");
  check(text == golden,
        "pinned MLIR 21.1.8 vector-readiness artifact must match golden");
  checkContains(text, "vector.contract",
                "deterministic output must expose vector contraction");
  checkContains(text, "whole_static_problem_inspection",
                "deterministic output must bound whole-problem vectorization");
  checkContains(text, "alias_preconditions",
                "alias preconditions must remain explicitly unconsumed");
  checkContains(text, "alignment_preconditions",
                "alignment preconditions must remain explicitly unconsumed");
  checkContains(text, "layout_and_strides",
                "layout/stride contract must remain explicitly unconsumed");
  checkContains(text, "numerical_permissions",
                "numerical permissions must remain explicitly unconsumed");
  checkContains(text, "source_on_zero_reads_return_unknown_on_generated_glue",
                "artifact must disclose the upstream location-provenance gap");

  auto second =
      bridge::deriveStructuredGemmVectorReadinessV1(*fixture.structured.module);
  check(second && bridge::serializeDeterministicMlir(*second.module) == text,
        "repeated upstream Transform application must be byte deterministic");

  mlir::MLIRContext parse_context;
  bridge::registerStructuredGemmVectorReadinessDialectsV1(parse_context);
  mlir::ParserConfig parser_config(&parse_context,
                                   /*verifyAfterParse=*/true);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, parser_config);
  check(static_cast<bool>(parsed),
        "serialized vector-readiness artifact must parse and MLIR-verify");
  if (parsed) {
    check(bridge::serializeDeterministicMlir(*parsed) == text,
          "vector artifact parse/print must be byte stable");
    check(bridge::verifyStructuredGemmVectorReadinessV1(*parsed, error),
          "reparsed vector artifact must verify standalone");
    check(bridge::verifyStructuredGemmVectorReadinessMatchesV1(
              *fixture.structured.module, *parsed, error),
          "cross-context vector artifact must pair with structured source");
  }

  std::vector<lowering::CpuRuntimeDispatchRecordV1> records(1);
  check(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(*vector.module,
                                                           records, error),
        "inspection-only vector artifact must not enter CPU runtime lowering");
  check(records.empty(),
        "runtime rejection must publish no partial dispatch record");
}

void testDynamicUpstreamNoOpControl(const v1::Module &capture) {
  PipelineFixture fixture(capture);
  check(static_cast<bool>(fixture.semantic),
        "reviewed dynamic capture must bridge for the upstream control");
  check(static_cast<bool>(fixture.structured),
        "reviewed dynamic capture must reach the certified structured seam");
  if (!fixture.semantic || !fixture.structured)
    return;

  bridge::registerStructuredGemmVectorReadinessDialectsV1(fixture.context);
  mlir::OwningOpRef<mlir::ModuleOp> payload =
      mlir::cast<mlir::ModuleOp>(fixture.structured.module->clone());
  const std::string before = bridge::serializeDeterministicMlir(*payload);
  mlir::ParserConfig parser_config(&fixture.context,
                                   /*verifyAfterParse=*/true);
  auto transform_module = mlir::parseSourceString<mlir::ModuleOp>(
      bridge::structuredGemmVectorReadinessTransformV1(), parser_config);
  check(static_cast<bool>(transform_module),
        "pinned upstream Transform control schedule must parse");
  if (!transform_module)
    return;
  auto transform_root =
      transform_module->lookupSymbol<mlir::transform::NamedSequenceOp>(
          mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  check(static_cast<bool>(transform_root),
        "pinned upstream Transform control must expose __transform_main");
  if (!transform_root)
    return;

  const auto options =
      mlir::transform::TransformOptions().enableExpensiveChecks(true);
  check(mlir::succeeded(mlir::transform::applyTransformNamedSequence(
            payload->getOperation(), transform_root.getOperation(),
            *transform_module, options)),
        "MLIR 21.1.8 dynamic Transform control must report success");
  check(mlir::succeeded(mlir::verify(*payload)),
        "dynamic Transform control output must remain valid MLIR");
  check(findOne<mlir::linalg::FillOp>(*payload) &&
            findOne<mlir::linalg::MatmulOp>(*payload) &&
            !findOne<mlir::vector::ContractionOp>(*payload) &&
            !findOne<mlir::vector::TransferReadOp>(*payload) &&
            !findOne<mlir::vector::TransferWriteOp>(*payload),
        "successful dynamic Transform control must leave Linalg in place and "
        "produce no Vector operations");
  check(bridge::serializeDeterministicMlir(*payload) == before,
        "selected MLIR 21.1.8 Transform must be an exact no-op on the "
        "reviewed dynamic structured capture");
}

void testDynamicNeedsPriorSchedule(const v1::Module &capture) {
  PipelineFixture fixture(capture);
  check(static_cast<bool>(fixture.semantic),
        "dynamic vector fixture must bridge");
  check(static_cast<bool>(fixture.structured),
        "dynamic vector fixture must structure");
  if (!fixture.semantic || !fixture.structured)
    return;
  const std::string before =
      bridge::serializeDeterministicMlir(*fixture.structured.module);
  auto rejected =
      bridge::deriveStructuredGemmVectorReadinessV1(*fixture.structured.module);
  check(!rejected,
        "dynamic GEMM must not forge target-independent vector readiness");
  check(!rejected.module,
        "dynamic rejection must publish no partial vector artifact");
  checkContains(rejected.error, "dynamic GEMM",
                "dynamic rejection must name the missing vector-shape proof");
  check(bridge::serializeDeterministicMlir(*fixture.structured.module) ==
            before,
        "failed dynamic derivation must leave its source unchanged");
}

void testStaticDegenerateGemmGeometries(const v1::Module &source) {
  struct Shape {
    std::uint64_t m;
    std::uint64_t k;
    std::uint64_t n;
  };
  constexpr std::array<Shape, 5> shapes = {
      Shape{1, 3, 1}, Shape{1, 3, 4}, Shape{2, 3, 1}, Shape{2, 1, 4},
      Shape{1, 1, 1}};
  for (const Shape shape : shapes) {
    v1::Module capture = source;
    makeStaticGemm(capture, shape.m, shape.k, shape.n);
    PipelineFixture fixture(capture);
    check(static_cast<bool>(fixture.structured),
          "rank-2 degenerate GEMM must reach the structured seam");
    if (!fixture.structured)
      continue;
    auto vector = bridge::deriveStructuredGemmVectorReadinessV1(
        *fixture.structured.module);
    check(static_cast<bool>(vector),
          "positive static rank-2 degenerate GEMM must vectorize");
    if (!vector)
      continue;
    std::string error;
    check(bridge::verifyStructuredGemmVectorReadinessMatchesV1(
              *fixture.structured.module, *vector.module, error),
          "degenerate vector witness must remain paired to its source");
    auto contract = findOne<mlir::vector::ContractionOp>(*vector.module);
    const auto expected =
        mlir::VectorType::get({static_cast<std::int64_t>(shape.m),
                               static_cast<std::int64_t>(shape.n)},
                              mlir::Float32Type::get(&fixture.context));
    check(contract && contract.getResult().getType() == expected,
          "degenerate GEMM must retain its exact rank-2 output geometry");
  }
}

void testMultiSiteIdentityAndOrdering(v1::Module source) {
  source = makeTwoSiteStaticCapture(std::move(source));
  std::string error;
  check(v1::verify(source, error),
        "two-site static verifier specimen must satisfy Matcore IR v1");
  PipelineFixture fixture(source);
  check(static_cast<bool>(fixture.semantic),
        "two-site specimen must bridge to semantic MLIR");
  check(static_cast<bool>(fixture.structured),
        "two-site specimen must derive the certified structured handoff");
  if (!fixture.structured)
    return;
  auto vector =
      bridge::deriveStructuredGemmVectorReadinessV1(*fixture.structured.module);
  check(static_cast<bool>(vector),
        "two-site structured source must derive one vector proof per site");
  if (!vector)
    return;
  check(functions(*vector.module).size() == 2,
        "two-site vector artifact must retain both ordered sites");
  std::size_t contracts = 0;
  vector.module->walk([&](mlir::vector::ContractionOp) { ++contracts; });
  check(contracts == 2,
        "two-site vector artifact must contain two exact contractions");
  check(bridge::verifyStructuredGemmVectorReadinessMatchesV1(
            *fixture.structured.module, *vector.module, error),
        "two-site vector artifact must pair with its complete ordered source");

  mlir::OwningOpRef<mlir::ModuleOp> reordered =
      mlir::cast<mlir::ModuleOp>(vector.module->clone());
  auto reordered_functions = functions(*reordered);
  check(reordered_functions.size() == 2,
        "reordering control must find both vector sites");
  if (reordered_functions.size() == 2) {
    reordered_functions[1]->moveBefore(reordered_functions[0]);
    mlir::ScopedDiagnosticHandler silence(
        &fixture.context, [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::succeeded(mlir::verify(*reordered)),
          "reordered vector sites must remain generic MLIR-valid");
    check(!bridge::verifyStructuredGemmVectorReadinessV1(*reordered, error),
          "ordered site-set certificate must reject site reordering");
  }

  mlir::OwningOpRef<mlir::ModuleOp> dropped =
      mlir::cast<mlir::ModuleOp>(vector.module->clone());
  auto dropped_functions = functions(*dropped);
  if (dropped_functions.size() == 2) {
    dropped_functions.back().erase();
    check(mlir::succeeded(mlir::verify(*dropped)),
          "dropped-site vector artifact must remain generic MLIR-valid");
    check(!bridge::verifyStructuredGemmVectorReadinessV1(*dropped, error),
          "site-set count/fingerprint must reject a dropped source site");
  }
}

void testStructuredSourceSubstitutionRejected(v1::Module source) {
  v1::Module original = source;
  v1::Module substitute = std::move(source);
  makeStaticGemm(original, 2, 3, 4);
  makeStaticGemm(substitute, 2, 3, 5);
  PipelineFixture original_fixture(original);
  PipelineFixture substitute_fixture(substitute);
  check(static_cast<bool>(original_fixture.structured) &&
            static_cast<bool>(substitute_fixture.structured),
        "shape-substitution controls must both derive certified sources");
  if (!original_fixture.structured || !substitute_fixture.structured)
    return;
  auto vector = bridge::deriveStructuredGemmVectorReadinessV1(
      *original_fixture.structured.module);
  check(static_cast<bool>(vector),
        "source-substitution control must derive its original vector artifact");
  if (!vector)
    return;
  std::string error;
  check(bridge::verifyStructuredGemmVectorReadinessV1(*vector.module, error),
        "source-substitution control must remain standalone-valid");
  check(!bridge::verifyStructuredGemmVectorReadinessMatchesV1(
            *substitute_fixture.structured.module, *vector.module, error),
        "paired certificate must reject a different valid structured shape");
  check(!error.empty(),
        "structured-source substitution rejection must diagnose failure");
}

void testMalformedStructuredSourceRejected(v1::Module capture) {
  makeStaticGemm(capture, 2, 3, 4);
  PipelineFixture fixture(capture);
  check(static_cast<bool>(fixture.structured),
        "malformed-source fixture must structure before damage");
  if (!fixture.structured)
    return;
  mlir::Builder builder(&fixture.context);
  auto fill = findOne<mlir::linalg::FillOp>(*fixture.structured.module);
  fill->setAttr("mdsl.structured_role", builder.getStringAttr("accumulate"));
  mlir::ScopedDiagnosticHandler silence(
      &fixture.context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*fixture.structured.module)),
        "malformed source must remain generic MLIR-valid");
  auto rejected =
      bridge::deriveStructuredGemmVectorReadinessV1(*fixture.structured.module);
  check(!rejected,
        "generic Linalg resembling GEMM must not acquire vector permission");
  checkContains(rejected.error, "exact verified structured GEMM",
                "source rejection must name the certified boundary");
}

void testAdversarialVectorMutations(v1::Module capture) {
  makeStaticGemm(capture, 2, 3, 4);
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.execution_authority",
                        builder.getStringAttr("generated_execution"));
      },
      "vector verifier must reject execution-authority escalation");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.vector_readiness_version",
                        builder.getI32IntegerAttr(2));
      },
      "vector verifier must reject an unsupported schema version");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        const auto readiness =
            vector_function->getAttrOfType<mlir::DictionaryAttr>(
                "mdsl.vector_readiness");
        vector_function->setAttr("mdsl.vector_readiness",
                                 withField(builder, readiness,
                                           "unconsumed_requirements",
                                           builder.getArrayAttr({})));
      },
      "vector verifier must reject a forged semantic-consumption ledger");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        const auto readiness =
            vector_function->getAttrOfType<mlir::DictionaryAttr>(
                "mdsl.vector_readiness");
        const auto requirements =
            readiness.getAs<mlir::ArrayAttr>("unconsumed_requirements");
        llvm::SmallVector<mlir::Attribute> retained;
        for (mlir::Attribute requirement : requirements) {
          const auto name = mlir::cast<mlir::StringAttr>(requirement);
          if (name.getValue() != "numerical_permissions")
            retained.push_back(requirement);
        }
        vector_function->setAttr("mdsl.vector_readiness",
                                 withField(builder, readiness,
                                           "unconsumed_requirements",
                                           builder.getArrayAttr(retained)));
      },
      "vector verifier must reject removal of unconsumed numerical "
      "permissions");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        const auto readiness =
            vector_function->getAttrOfType<mlir::DictionaryAttr>(
                "mdsl.vector_readiness");
        const auto requirements =
            readiness.getAs<mlir::ArrayAttr>("unconsumed_requirements");
        llvm::SmallVector<mlir::Attribute> forged(requirements.begin(),
                                                  requirements.end());
        for (mlir::Attribute &requirement : forged) {
          const auto name = mlir::cast<mlir::StringAttr>(requirement);
          if (name.getValue() == "numerical_permissions")
            requirement =
                builder.getStringAttr("numerical_permissions_consumed");
        }
        vector_function->setAttr("mdsl.vector_readiness",
                                 withField(builder, readiness,
                                           "unconsumed_requirements",
                                           builder.getArrayAttr(forged)));
      },
      "vector verifier must reject a forged numerical-consumption claim");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        const auto readiness =
            vector_function->getAttrOfType<mlir::DictionaryAttr>(
                "mdsl.vector_readiness");
        vector_function->setAttr(
            "mdsl.vector_readiness",
            withField(builder, readiness, "authority",
                      builder.getStringAttr("generated_execution")));
      },
      "vector verifier must reject function-ledger authority escalation");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        const auto readiness =
            vector_function->getAttrOfType<mlir::DictionaryAttr>(
                "mdsl.vector_readiness");
        vector_function->setAttr(
            "mdsl.vector_readiness",
            withField(builder, readiness, "operation_locations",
                      builder.getStringAttr("all_source_locations")));
      },
      "vector verifier must reject forged location-provenance retention");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.vector_width", builder.getI64IntegerAttr(8));
      },
      "vector verifier must reject module-level vector scheduling policy");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        const auto contract =
            vector_function->getAttrOfType<mlir::DictionaryAttr>(
                "mdsl.semantic_contract");
        vector_function->setAttr(
            "mdsl.semantic_contract",
            withExtraField(
                builder, contract, "tile_sizes",
                builder.getArrayAttr({builder.getI64IntegerAttr(4),
                                      builder.getI64IntegerAttr(4)})));
      },
      "vector verifier must reject scheduling metadata in semantic contract");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        vectorConstant(module).setValueAttr(mlir::DenseElementsAttr::get(
            mlir::cast<mlir::ShapedType>(vectorConstant(module).getType()),
            builder.getF32FloatAttr(1.0)));
      },
      "vector verifier must reject a nonzero contraction accumulator");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto write = findOne<mlir::vector::TransferWriteOp>(module);
        write->setOperand(0, vectorConstant(module).getResult());
      },
      "vector verifier must reject bypassing vector.contract at output");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto vector_function = function(module);
        auto return_op = findOne<mlir::func::ReturnOp>(module);
        return_op->setOperand(0, vector_function.getArgument(2));
      },
      "vector verifier must reject bypassing the transfer-write result");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        findOne<mlir::vector::ContractionOp>(module)->setAttr(
            "mdsl.unreviewed_hint", builder.getBoolAttr(true));
      },
      "vector verifier must reject unreviewed downstream hints");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto vector_function = function(module);
        findOne<mlir::vector::ContractionOp>(module)->setLoc(
            vector_function.getLoc());
      },
      "vector verifier must reject fabricated source location on generated "
      "contraction glue");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        vectorConstant(module)->setLoc(
            mlir::UnknownLoc::get(module.getContext()));
      },
      "vector verifier must reject loss of the retained zero source location");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto read = findOne<mlir::vector::TransferReadOp>(module);
        read.setInBoundsAttr(builder.getArrayAttr(
            {builder.getBoolAttr(false), builder.getBoolAttr(true)}));
      },
      "vector verifier must reject a partial/masked whole-problem transfer");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        auto contract = vector_function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.semantic_contract");
        auto aliasing = contract.getAs<mlir::ArrayAttr>("aliasing");
        auto first = mlir::cast<mlir::DictionaryAttr>(aliasing[0]);
        first = withField(builder, first, "contract",
                          builder.getStringAttr("proven_fact"));
        llvm::SmallVector<mlir::Attribute> relations(aliasing.begin(),
                                                     aliasing.end());
        relations[0] = first;
        vector_function->setAttr("mdsl.semantic_contract",
                                 withField(builder, contract, "aliasing",
                                           builder.getArrayAttr(relations)));
      },
      "vector verifier must reject upgrading alias preconditions to facts");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        auto contract = vector_function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.semantic_contract");
        auto numerical = contract.getAs<mlir::DictionaryAttr>("numerical");
        numerical = withField(builder, numerical, "reassociation",
                              builder.getStringAttr("forbidden"));
        vector_function->setAttr(
            "mdsl.semantic_contract",
            withField(builder, contract, "numerical", numerical));
      },
      "vector verifier must reject mutation of retained numerical "
      "permissions");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        auto readiness = vector_function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.vector_readiness");
        auto numerical =
            readiness.getAs<mlir::DictionaryAttr>("numerical_permissions");
        numerical = withField(builder, numerical, "approximate_math",
                              builder.getBoolAttr(true));
        vector_function->setAttr(
            "mdsl.vector_readiness",
            withField(builder, readiness, "numerical_permissions", numerical));
      },
      "vector verifier must reject a forged retained numerical-permission "
      "ledger");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr(
            "mdsl.source_structured_fingerprint",
            builder.getStringAttr("sha256:"
                                  "00000000000000000000000000000000000000000000"
                                  "00000000000000000000"));
      },
      "vector verifier must reject a forged aggregate structured-source "
      "fingerprint");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        function(module)->setAttr(
            "mdsl.source_structured_fingerprint",
            builder.getStringAttr("sha256:"
                                  "00000000000000000000000000000000000000000000"
                                  "00000000000000000000"));
      },
      "vector verifier must reject a forged per-site structured-source "
      "fingerprint");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto vector_function = function(module);
        auto element = builder.getF32Type();
        auto lhs = mlir::RankedTensorType::get({2, 3}, element);
        auto rhs = mlir::RankedTensorType::get({3, 5}, element);
        auto output = mlir::RankedTensorType::get({2, 5}, element);
        vector_function->setAttr(
            "mdsl.source_structured_function_type",
            mlir::TypeAttr::get(builder.getFunctionType(
                mlir::TypeRange{lhs, rhs, output}, mlir::TypeRange{output})));
      },
      "vector verifier must reject a forged retained structured source shape");
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto vector_function = function(module);
        vector_function->setLoc(mlir::UnknownLoc::get(module.getContext()));
      },
      "vector verifier must reject loss of certified function location");

  for (unsigned argument = 0; argument != 3; ++argument) {
    expectVectorRejected(
        capture,
        [argument](mlir::ModuleOp module, mlir::Builder &builder) {
          function(module).getBody().front().getArgument(argument).setLoc(
              mlir::FileLineColLoc::get(builder.getContext(),
                                        "forged-argument-location.mdsl", 1,
                                        argument + 1));
        },
        "vector verifier must reject a changed function block-argument "
        "location");
  }

  v1::Module square = readCapture();
  makeStaticGemm(square, 2, 2, 2);
  expectVectorRejected(
      square,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto contract = findOne<mlir::vector::ContractionOp>(module);
        const llvm::SmallVector<mlir::AffineMap> original_maps =
            contract.getIndexingMapsArray();
        llvm::SmallVector<mlir::AffineMap> maps(original_maps.begin(),
                                                original_maps.end());
        std::swap(maps[0], maps[1]);
        contract.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
      },
      "vector verifier must reject a type-compatible forged contraction map");
}

void testInitialOutputReadRejected(v1::Module capture) {
  makeStaticGemm(capture, 2, 2, 2);
  expectVectorRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto vector_function = function(module);
        auto read = findOne<mlir::vector::TransferReadOp>(module);
        read->setOperand(0, vector_function.getArgument(2));
      },
      "vector verifier must reject reading initial C as a contraction input");
}

} // namespace

int main() {
  const v1::Module capture = readCapture();
  testStaticVectorContract(capture);
  testDynamicUpstreamNoOpControl(capture);
  testDynamicNeedsPriorSchedule(capture);
  testStaticDegenerateGemmGeometries(capture);
  testMultiSiteIdentityAndOrdering(capture);
  testStructuredSourceSubstitutionRejected(capture);
  testMalformedStructuredSourceRejected(capture);
  testAdversarialVectorMutations(capture);
  testInitialOutputReadRejected(capture);

  if (failures != 0) {
    std::cerr << "Structured GEMM vector-readiness adversarial tests: "
              << failures << " of " << checks << " checks failed\n";
    return 1;
  }
  std::cout << "Structured GEMM vector-readiness adversarial tests: " << checks
            << " checks, 0 failures\n";
  return 0;
}
