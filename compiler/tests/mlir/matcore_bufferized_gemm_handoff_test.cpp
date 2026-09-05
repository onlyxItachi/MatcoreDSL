#include "MatcoreBufferizedGemmHandoff.h"
#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;
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
  const std::string json =
      readFile(std::string(MDSLC_IR_TEST_SOURCE_DIR) +
               "/gemm_capture.v1.golden.json");
  v1::Module module;
  std::string error;
  check(v1::parseAndVerifyJson(json, module, error),
        "reviewed Matcore IR v1 capture must parse and verify");
  return module;
}

bridge::BridgeResult buildSemantic(const v1::Module &capture,
                                   mlir::MLIRContext &context) {
  return bridge::bridgeV1ToMatcoreMlir(
      capture, context, bridge::explicitGemmF32V1BridgeContext());
}

bridge::StructuredGemmHandoffResultV1
buildStructured(const v1::Module &capture, mlir::MLIRContext &context) {
  auto semantic = buildSemantic(capture, context);
  check(static_cast<bool>(semantic),
        "bufferization fixture must bridge to semantic MLIR");
  if (!semantic)
    return {};
  return bridge::deriveStructuredGemmHandoffV1(*semantic.module);
}

void makeStatic(v1::TensorValue &value, std::uint64_t rows,
                std::uint64_t columns) {
  value.type.shape = {v1::ScalarExpr::staticValue(rows),
                      v1::ScalarExpr::staticValue(columns)};
  value.type.strides = {v1::ScalarExpr::staticValue(columns),
                        v1::ScalarExpr::staticValue(1)};
}

v1::Module makeTwoSiteCapture(v1::Module capture) {
  check(capture.operations.size() == 1,
        "two-site fixture must begin with one reviewed GEMM");
  if (capture.operations.size() != 1)
    return capture;
  v1::Operation independent = capture.operations.front();
  independent.site_id = "mc_11111111111111111111111111111111";
  const std::uint64_t shift = independent.call_range.end + 10 -
                              independent.call_range.begin;
  independent.source.offset += shift;
  independent.source.line += 10;
  independent.call_range.begin += shift;
  independent.call_range.end += shift;
  for (auto &range : independent.argument_ranges) {
    range.begin += shift;
    range.end += shift;
  }
  independent.output.source_expression = "Z";
  independent.operands[0].source_expression = "X";
  independent.operands[1].source_expression = "Y";
  capture.operations.push_back(std::move(independent));
  std::string error;
  check(v1::verify(capture, error),
        "independent two-site GEMM capture must verify");
  return capture;
}

template <typename OpTy>
OpTy findOne(mlir::ModuleOp module) {
  OpTy result;
  module.walk([&](OpTy operation) {
    if (!result)
      result = operation;
  });
  return result;
}

mlir::func::FuncOp oneFunction(mlir::ModuleOp module) {
  return findOne<mlir::func::FuncOp>(module);
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
  check(replaced, "mutated certificate field must exist");
  return builder.getDictionaryAttr(attributes);
}

void replaceContractField(mlir::func::FuncOp function, mlir::Builder &builder,
                          llvm::StringRef field,
                          mlir::Attribute replacement) {
  auto contract = function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.semantic_contract");
  function->setAttr("mdsl.semantic_contract",
                    withField(builder, contract, field, replacement));
}

using BufferizedMutation =
    std::function<void(mlir::ModuleOp, mlir::Builder &)>;

void expectBufferizedRejected(const v1::Module &capture,
                              const BufferizedMutation &mutate,
                              std::string_view message) {
  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "mutation fixture must derive structured handoff before damage");
  if (!structured)
    return;
  auto bufferized =
      bridge::deriveBufferizedGemmHandoffV1(*structured.module);
  check(static_cast<bool>(bufferized),
        "mutation fixture must bufferize before damage");
  if (!bufferized)
    return;

  mlir::Builder builder(&context);
  mutate(*bufferized.module, builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*bufferized.module)),
        std::string(message) + " (mutation must remain valid generic MLIR)");
  std::string error;
  check(!bridge::verifyBufferizedGemmHandoffV1(*bufferized.module, error),
        message);
  check(!error.empty(), "bufferized rejection must provide a diagnostic");
  error.clear();
  check(!bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
             *structured.module, *bufferized.module, error),
        std::string(message) + " (paired verifier)");
  check(!error.empty(),
        "paired bufferized rejection must provide a diagnostic");
}

void testDynamicBufferization(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "dynamic explicit GEMM must derive structured handoff");
  if (!structured)
    return;
  const std::string source_before =
      bridge::serializeDeterministicMlir(*structured.module);
  auto bufferized =
      bridge::deriveBufferizedGemmHandoffV1(*structured.module);
  check(static_cast<bool>(bufferized),
        "dynamic structured GEMM must derive certified bufferization");
  if (!bufferized) {
    std::cerr << bufferized.error << '\n';
    return;
  }
  check(bridge::serializeDeterministicMlir(*structured.module) == source_before,
        "bufferization derivation must not mutate structured source");
  check(bufferized.buffer_allocations == 0,
        "certified dynamic path must report zero buffer allocations");
  check(bufferized.buffer_deallocations == 0,
        "certified dynamic path must report zero buffer deallocations");
  check(bufferized.tensor_out_of_place == 0,
        "certified dynamic path must report no out-of-place tensor decision");

  std::string error;
  check(bridge::verifyBufferizedGemmHandoffV1(*bufferized.module, error),
        "dynamic bufferized handoff must verify standalone");
  check(bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
            *structured.module, *bufferized.module, error),
        "dynamic bufferized handoff must match structured source");

  auto function = oneFunction(*bufferized.module);
  auto fill = findOne<mlir::linalg::FillOp>(*bufferized.module);
  auto matmul = findOne<mlir::linalg::MatmulOp>(*bufferized.module);
  auto return_op = findOne<mlir::func::ReturnOp>(*bufferized.module);
  check(function && fill && matmul && return_op,
        "bufferized handoff must contain func/fill/matmul/return");
  if (!function || !fill || !matmul || !return_op)
    return;
  check(!function.getArgAttrsAttr() && !function.getResAttrsAttr(),
        "bufferization must not synthesize noalias/alignment optimizer facts");
  check(fill.getOutputs().front() == function.getArgument(2),
        "full zero fill must target original output argument 2");
  check(matmul.getOutputs().front() == function.getArgument(2),
        "contraction must target the same original output buffer");
  check(return_op.getOperand(0) == function.getArgument(2),
        "bufferized result must be exact SSA identity of output argument 2");

  bool forbidden_operation = false;
  (*bufferized.module).walk([&](mlir::Operation *operation) {
    const llvm::StringRef name = operation->getName().getStringRef();
    forbidden_operation |=
        name == "memref.alloc" || name == "memref.alloca" ||
        name == "memref.copy" || name == "bufferization.clone" ||
        name == "bufferization.to_buffer" ||
        name == "bufferization.to_tensor";
  });
  check(!forbidden_operation,
        "certified function-boundary path must contain no allocation/copy "
        "or tensor-buffer bridge");

  const auto ledger = function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.bufferization_handoff");
  check(ledger &&
            ledger.getAs<mlir::StringAttr>("alias_preconditions").getValue() ==
                "retained_unproven" &&
            ledger.getAs<mlir::StringAttr>("alignment_preconditions")
                    .getValue() == "retained_unproven" &&
            ledger.getAs<mlir::StringAttr>("dynamic_shape_relations")
                    .getValue() == "retained_unproven" &&
            ledger.getAs<mlir::StringAttr>("memory_space").getValue() ==
                "retained_unproven" &&
            ledger.getAs<mlir::StringAttr>("numerical").getValue() ==
                "retained_unconsumed_scalar_region_checked" &&
            ledger.getAs<mlir::StringAttr>("provenance").getValue() ==
                "retained_fingerprint_self_checked_pairing_required" &&
            ledger.getAs<mlir::StringAttr>("runtime_preconditions")
                    .getValue() ==
                "retained_unproven_execution_forbidden",
        "certificate must distinguish standalone fingerprint consistency from "
        "source pairing and keep numerical/runtime preconditions unconsumed");
  const auto source_type = function->getAttrOfType<mlir::TypeAttr>(
      "mdsl.source_structured_function_type");
  const auto source_fingerprint = function->getAttrOfType<mlir::StringAttr>(
      "mdsl.source_structured_fingerprint");
  check(source_type && mlir::isa<mlir::FunctionType>(source_type.getValue()) &&
            source_fingerprint && source_fingerprint.getValue().size() == 71 &&
            source_fingerprint.getValue().starts_with("sha256:"),
        "derived site must retain its exact structured source type and digest");
  const auto semantic_contract = function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.semantic_contract");
  const auto aliasing = semantic_contract.getAs<mlir::ArrayAttr>("aliasing");
  const auto lhs =
      semantic_contract.getAs<mlir::DictionaryAttr>("lhs_semantics");
  check(aliasing && aliasing.size() == 2 && lhs &&
            lhs.getAs<mlir::StringAttr>("alignment_contract").getValue() ==
                "required_precondition",
        "semantic contract must retain noalias and alignment requirements");
}

void testStaticNonSquareBufferization(v1::Module capture) {
  check(capture.operations.size() == 1,
        "static bufferization fixture must contain one operation");
  if (capture.operations.size() != 1)
    return;
  makeStatic(capture.operations[0].operands[0], 2, 3);
  makeStatic(capture.operations[0].operands[1], 3, 4);
  makeStatic(capture.operations[0].output, 2, 4);
  std::string error;
  check(v1::verify(capture, error),
        "static non-square capture must verify before bufferization");
  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "static non-square GEMM must derive structured handoff");
  if (!structured)
    return;
  auto bufferized =
      bridge::deriveBufferizedGemmHandoffV1(*structured.module);
  check(static_cast<bool>(bufferized),
        "static non-square GEMM must derive certified bufferization");
  if (!bufferized)
    return;
  check(bufferized.buffer_allocations == 0 &&
            bufferized.tensor_out_of_place == 0,
        "static certified path must remain allocation-free and in-place");
  auto function = oneFunction(*bufferized.module);
  check(function, "static bufferized module must contain a function");
  if (!function)
    return;
  const auto lhs =
      mlir::dyn_cast<mlir::MemRefType>(function.getArgument(0).getType());
  const auto rhs =
      mlir::dyn_cast<mlir::MemRefType>(function.getArgument(1).getType());
  const auto output =
      mlir::dyn_cast<mlir::MemRefType>(function.getArgument(2).getType());
  check(lhs && lhs == mlir::MemRefType::get({2, 3},
                                            mlir::Float32Type::get(&context)),
        "static lhs must be identity-layout memref<2x3xf32>");
  check(rhs && rhs == mlir::MemRefType::get({3, 4},
                                            mlir::Float32Type::get(&context)),
        "static rhs must be identity-layout memref<3x4xf32>");
  check(output && output == mlir::MemRefType::get(
                                {2, 4}, mlir::Float32Type::get(&context)),
        "static output must be identity-layout memref<2x4xf32>");
  check(function.getResultTypes().front() == function.getArgument(2).getType(),
        "static result type must equal original output argument type");
  check(bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
            *structured.module, *bufferized.module, error),
        "static bufferized handoff must match structured source");
}

void testMixedStaticDynamicMultiSiteBufferization(v1::Module capture) {
  capture = makeTwoSiteCapture(std::move(capture));
  if (capture.operations.size() != 2)
    return;
  makeStatic(capture.operations[0].operands[0], 2, 3);
  makeStatic(capture.operations[0].operands[1], 3, 4);
  makeStatic(capture.operations[0].output, 2, 4);
  std::string error;
  check(v1::verify(capture, error),
        "mixed static/dynamic two-site capture must verify");

  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "mixed static/dynamic sites must derive one structured module");
  if (!structured)
    return;
  auto bufferized =
      bridge::deriveBufferizedGemmHandoffV1(*structured.module);
  check(static_cast<bool>(bufferized),
        "mixed static/dynamic sites must derive one bufferized module");
  if (!bufferized) {
    std::cerr << bufferized.error << '\n';
    return;
  }
  check(bufferized.buffer_allocations == 0 &&
            bufferized.buffer_deallocations == 0 &&
            bufferized.tensor_out_of_place == 0,
        "multi-site certified path must remain allocation/copy/out-of-place "
        "free");
  check(bridge::verifyBufferizedGemmHandoffV1(*bufferized.module, error),
        "multi-site bufferized handoff must verify standalone");
  check(bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
            *structured.module, *bufferized.module, error),
        "multi-site bufferized handoff must pair with its exact source");

  auto sites = functions(*bufferized.module);
  check(sites.size() == 2,
        "multi-site bufferized handoff must retain both ordered sites");
  if (sites.size() != 2)
    return;
  const auto first_lhs =
      mlir::dyn_cast<mlir::MemRefType>(sites[0].getArgument(0).getType());
  const auto second_lhs =
      mlir::dyn_cast<mlir::MemRefType>(sites[1].getArgument(0).getType());
  check(first_lhs && first_lhs.getShape().size() == 2 &&
            first_lhs.getShape()[0] == 2 && first_lhs.getShape()[1] == 3,
        "first multi-site function must retain static 2x3 lhs");
  check(second_lhs && second_lhs.isDynamicDim(0) &&
            second_lhs.isDynamicDim(1),
        "second multi-site function must retain independent dynamic lhs");
  const auto count = (*bufferized.module)
                         ->getAttrOfType<mlir::IntegerAttr>(
                             "mdsl.source_structured_site_count");
  const auto aggregate = (*bufferized.module)
                             ->getAttrOfType<mlir::StringAttr>(
                                 "mdsl.source_structured_fingerprint");
  check(count && count.getInt() == 2 && aggregate &&
            aggregate.getValue().size() == 71 &&
            aggregate.getValue().starts_with("sha256:"),
        "multi-site certificate must bind ordered source count and digest");
  for (mlir::func::FuncOp function : sites) {
    check(function.getBody().front().back().getOperand(0) ==
              function.getArgument(2),
          "each certified site must return its own original output argument");
  }
}

void testCrossContextRoundTrip(const v1::Module &capture) {
  mlir::MLIRContext source_context;
  auto structured = buildStructured(capture, source_context);
  check(static_cast<bool>(structured),
        "round-trip fixture must derive structured handoff");
  if (!structured)
    return;
  auto bufferized =
      bridge::deriveBufferizedGemmHandoffV1(*structured.module);
  check(static_cast<bool>(bufferized),
        "round-trip fixture must derive bufferized handoff");
  if (!bufferized)
    return;
  const std::string text =
      bridge::serializeDeterministicMlir(*bufferized.module);

  mlir::MLIRContext parse_context;
  bridge::registerBufferizedGemmHandoffDialectsV1(parse_context);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, &parse_context);
  check(static_cast<bool>(parsed),
        "serialized bufferized handoff must parse in a fresh context");
  if (!parsed)
    return;
  std::string error;
  check(bridge::verifyBufferizedGemmHandoffV1(*parsed, error),
        "reparsed bufferized handoff must verify standalone");
  check(bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
            *structured.module, *parsed, error),
        "cross-context bufferized handoff must match structured source");
  check(bridge::serializeDeterministicMlir(*parsed) == text,
        "bufferized handoff serialization must be deterministic");
}

void testSourceMismatch(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto dynamic = buildStructured(capture, context);
  check(static_cast<bool>(dynamic),
        "source mismatch fixture must derive dynamic structured source");
  if (!dynamic)
    return;
  auto bufferized = bridge::deriveBufferizedGemmHandoffV1(*dynamic.module);
  check(static_cast<bool>(bufferized),
        "source mismatch fixture must derive bufferized handoff");
  if (!bufferized)
    return;

  v1::Module static_capture = capture;
  makeStatic(static_capture.operations[0].operands[0], 2, 3);
  makeStatic(static_capture.operations[0].operands[1], 3, 4);
  makeStatic(static_capture.operations[0].output, 2, 4);
  auto static_structured = buildStructured(static_capture, context);
  check(static_cast<bool>(static_structured),
        "source mismatch fixture must derive static structured source");
  if (!static_structured)
    return;
  std::string error;
  check(!bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
             *static_structured.module, *bufferized.module, error),
        "paired verifier must reject a different structured source");
  check(!error.empty(), "source mismatch must provide a diagnostic");
}

void testSameShapedAlternateSource(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto original = buildStructured(capture, context);
  check(static_cast<bool>(original),
        "same-shape source fixture must derive original structured source");
  if (!original)
    return;
  auto bufferized = bridge::deriveBufferizedGemmHandoffV1(*original.module);
  check(static_cast<bool>(bufferized),
        "same-shape source fixture must derive bufferized handoff");
  if (!bufferized)
    return;

  v1::Module alternate_capture = capture;
  alternate_capture.translation_unit =
      "compiler/tests/frontend/same_shape_alternate.mdsl";
  std::string error;
  check(v1::verify(alternate_capture, error),
        "same-shape alternate translation-unit identity must remain valid v1");
  auto alternate = buildStructured(alternate_capture, context);
  check(static_cast<bool>(alternate),
        "same-shape alternate structured source must derive");
  if (!alternate)
    return;
  check(bridge::verifyBufferizedGemmHandoffV1(*bufferized.module, error),
        "bufferized standalone self-consistency must remain valid before "
        "alternate pairing");
  check(!bridge::verifyBufferizedGemmHandoffMatchesStructuredV1(
             *alternate.module, *bufferized.module, error),
        "paired verifier must reject a same-shaped alternate source identity");
  checkContains(error, "translation_unit",
                "same-shaped alternate source rejection must name module "
                "provenance");
}

void testMultiSiteIdentityMutations(const v1::Module &capture) {
  const v1::Module two_sites = makeTwoSiteCapture(capture);
  if (two_sites.operations.size() != 2)
    return;
  expectBufferizedRejected(
      two_sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto sites = functions(module);
        sites[1]->moveBefore(sites[0]);
      },
      "derived source identity must reject reordered sites");

  expectBufferizedRejected(
      two_sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto sites = functions(module);
        sites[1].erase();
      },
      "derived source identity must reject a dropped site");

  expectBufferizedRejected(
      two_sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto sites = functions(module);
        auto duplicate =
            mlir::cast<mlir::func::FuncOp>(sites[1]->clone());
        duplicate.setName("__matcore_structured_duplicate_control");
        module.push_back(duplicate);
      },
      "derived source identity must reject a duplicated site");

  expectBufferizedRejected(
      two_sites,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto sites = functions(module);
        mlir::Attribute first =
            sites[0]->getAttr("mdsl.source_structured_fingerprint");
        mlir::Attribute second =
            sites[1]->getAttr("mdsl.source_structured_fingerprint");
        sites[0]->setAttr("mdsl.source_structured_fingerprint", second);
        sites[1]->setAttr("mdsl.source_structured_fingerprint", first);
      },
      "derived source identity must reject cross-site fingerprint "
      "substitution");

  expectBufferizedRejected(
      two_sites,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr(
            "mdsl.source_structured_fingerprint",
            builder.getStringAttr(
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"));
      },
      "derived source identity must reject a forged ordered site-set digest");
}

void testAdversarialMutations(const v1::Module &capture) {
  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.execution_authority",
                        builder.getStringAttr("generated_execution"));
      },
      "bufferized verifier must reject execution authority escalation");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr(
            "mdsl.translation_unit",
            builder.getStringAttr(
                "compiler/tests/frontend/forged-buffer-source.mdsl"));
      },
      "bufferized verifier must reject module-provenance drift not reflected "
      "by its source fingerprint");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        auto ledger = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.bufferization_handoff");
        function->setAttr(
            "mdsl.bufferization_handoff",
            withField(builder, ledger, "alias_preconditions",
                      builder.getStringAttr("proven_noalias")));
      },
      "bufferized verifier must reject forged noalias proof");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        auto ledger = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.bufferization_handoff");
        function->setAttr(
            "mdsl.bufferization_handoff",
            withField(builder, ledger, "alignment_preconditions",
                      builder.getStringAttr("proven_alignment")));
      },
      "bufferized verifier must reject forged alignment proof");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        auto ledger = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.bufferization_handoff");
        function->setAttr(
            "mdsl.bufferization_handoff",
            withField(builder, ledger, "runtime_preconditions",
                      builder.getStringAttr("materialized_runtime_guards")));
      },
      "bufferized verifier must reject forged runtime-precondition "
      "consumption");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        auto source_type_attr = function->getAttrOfType<mlir::TypeAttr>(
            "mdsl.source_structured_function_type");
        auto source_type =
            mlir::cast<mlir::FunctionType>(source_type_attr.getValue());
        llvm::SmallVector<mlir::Type> inputs(source_type.getInputs());
        inputs[0] = mlir::RankedTensorType::get(
            {7, mlir::ShapedType::kDynamic}, builder.getF32Type());
        function->setAttr(
            "mdsl.source_structured_function_type",
            mlir::TypeAttr::get(
                builder.getFunctionType(inputs, source_type.getResults())));
      },
      "bufferized verifier must reject forged dynamic-shape source identity");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        auto contract = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.semantic_contract");
        function->setAttr(
            "mdsl.semantic_contract",
            withField(builder, contract, "aliasing", builder.getArrayAttr({})));
      },
      "bufferized verifier must reject removed alias preconditions");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        auto contract = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.semantic_contract");
        auto numerical =
            contract.getAs<mlir::DictionaryAttr>("numerical");
        numerical =
            withField(builder, numerical, "reassociation",
                      builder.getStringAttr("forbidden"));
        replaceContractField(function, builder, "numerical", numerical);
      },
      "bufferized verifier must reject altered numerical permission");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto function = oneFunction(module);
        auto return_op = findOne<mlir::func::ReturnOp>(module);
        return_op->setOperand(0, function.getArgument(0));
      },
      "bufferized verifier must reject a result not identical to output arg2");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto function = oneFunction(module);
        auto fill = findOne<mlir::linalg::FillOp>(module);
        fill->setOperand(1, function.getArgument(0));
      },
      "bufferized verifier must reject zero fill of the wrong buffer");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto fill = findOne<mlir::linalg::FillOp>(module);
        fill.erase();
      },
      "bufferized verifier must reject direct read of initial C without fill");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto function = oneFunction(module);
        auto matmul = findOne<mlir::linalg::MatmulOp>(module);
        matmul->setOperand(2, function.getArgument(0));
      },
      "bufferized verifier must reject contraction into the wrong buffer");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto function = oneFunction(module);
        auto matmul = findOne<mlir::linalg::MatmulOp>(module);
        mlir::Value lhs = function.getArgument(0);
        matmul->setOperand(0, function.getArgument(1));
        matmul->setOperand(1, lhs);
      },
      "bufferized verifier must reject swapped contraction inputs");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        llvm::SmallVector<mlir::Type> types;
        for (mlir::BlockArgument argument : function.getArguments()) {
          const auto old_type =
              mlir::cast<mlir::MemRefType>(argument.getType());
          auto new_type = mlir::MemRefType::get(
              old_type.getShape(), old_type.getElementType(),
              old_type.getLayout(), builder.getI64IntegerAttr(1));
          argument.setType(new_type);
          types.push_back(new_type);
        }
        function.setType(builder.getFunctionType(types, types[2]));
      },
      "bufferized verifier must reject a forged non-default memory space");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = oneFunction(module);
        llvm::SmallVector<mlir::Type> types;
        const std::int64_t dynamic = mlir::ShapedType::kDynamic;
        auto layout = mlir::StridedLayoutAttr::get(
            module.getContext(), dynamic, {dynamic, dynamic});
        for (mlir::BlockArgument argument : function.getArguments()) {
          const auto old_type =
              mlir::cast<mlir::MemRefType>(argument.getType());
          auto new_type = mlir::MemRefType::get(
              old_type.getShape(), old_type.getElementType(), layout);
          argument.setType(new_type);
          types.push_back(new_type);
        }
        function.setType(builder.getFunctionType(types, types[2]));
      },
      "bufferized verifier must reject loss of identity row-major layout");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        oneFunction(module)->setLoc(mlir::FileLineColLoc::get(
            builder.getContext(), "forged-function.mdsl", 77, 3));
      },
      "bufferized verifier must reject function-location drift");

  expectBufferizedRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        oneFunction(module).getArgument(0).setLoc(mlir::FileLineColLoc::get(
            builder.getContext(), "forged-argument.mdsl", 88, 4));
      },
      "bufferized verifier must reject block-argument-location drift");
}

void testUpstreamNoFunctionBoundaryFalsifier(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "no-function-boundary fixture must derive structured handoff");
  if (!structured)
    return;
  bridge::registerBufferizedGemmHandoffDialectsV1(context);
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = false;
  options.copyBeforeWrite = false;
  options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  check(mlir::succeeded(mlir::bufferization::runOneShotModuleBufferize(
            *structured.module, options, state, &statistics)),
        "upstream no-function-boundary control must bufferize");
  bool saw_allocation = false;
  bool saw_to_buffer = false;
  bool saw_to_tensor = false;
  (*structured.module).walk([&](mlir::Operation *operation) {
    saw_allocation |= mlir::isa<mlir::memref::AllocOp>(operation);
    saw_to_buffer |=
        operation->getName().getStringRef() == "bufferization.to_buffer";
    saw_to_tensor |=
        operation->getName().getStringRef() == "bufferization.to_tensor";
  });
  check(saw_allocation && saw_to_buffer && saw_to_tensor &&
            statistics.numBufferAlloc > 0,
        "disabling function-boundary bufferization must expose the allocation "
        "and tensor/buffer bridges hidden by tensor boundaries");
  auto function = oneFunction(*structured.module);
  check(function && mlir::isa<mlir::RankedTensorType>(
                        function.getArgument(2).getType()),
        "no-function-boundary control must leave the original C boundary as "
        "a tensor rather than proving a caller memref identity");
  std::string error;
  check(!bridge::verifyBufferizedGemmHandoffV1(*structured.module, error),
        "no-function-boundary control must not satisfy the certified buffer "
        "envelope");
}

void testUpstreamConflictFalsifier(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "conflict fixture must derive structured handoff");
  if (!structured)
    return;
  auto function = oneFunction(*structured.module);
  auto return_op = findOne<mlir::func::ReturnOp>(*structured.module);
  mlir::OpBuilder builder(return_op);
  auto zero = mlir::arith::ConstantOp::create(builder, function.getLoc(),
                                              builder.getIndexAttr(0));
  mlir::tensor::ExtractOp::create(
      builder, function.getLoc(), function.getArgument(2),
      mlir::ValueRange{zero.getResult(), zero.getResult()});

  std::string error;
  check(mlir::succeeded(mlir::verify(*structured.module)),
        "conflicting post-write destination read must remain valid generic "
        "tensor MLIR");
  check(!bridge::verifyStructuredGemmHandoffV1(*structured.module, error),
        "certified structured verifier must reject extra destination read");
  auto rejected = bridge::deriveBufferizedGemmHandoffV1(*structured.module);
  check(!rejected,
        "certified bufferization must reject conflicting source before pass");

  bridge::registerBufferizedGemmHandoffDialectsV1(context);
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = true;
  options.copyBeforeWrite = false;
  options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  check(mlir::succeeded(mlir::bufferization::runOneShotModuleBufferize(
            *structured.module, options, state, &statistics)),
        "upstream MLIR must bufferize the deliberate conflicting-use control");
  bool saw_allocation = false;
  (*structured.module).walk([&](mlir::memref::AllocOp) {
    saw_allocation = true;
  });
  check(saw_allocation && statistics.numBufferAlloc > 0 &&
            statistics.numTensorOutOfPlace > 0,
        "post-write read conflict must force upstream allocation/out-of-place "
        "bufferization");
  auto buffer_function = oneFunction(*structured.module);
  auto buffer_return = findOne<mlir::func::ReturnOp>(*structured.module);
  check(buffer_return.getOperand(0) != buffer_function.getArgument(2),
        "conflicting-use control result must not claim original C identity");
}

void testUpstreamCopyBeforeWriteFalsifier(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto structured = buildStructured(capture, context);
  check(static_cast<bool>(structured),
        "copy-before-write fixture must derive structured handoff");
  if (!structured)
    return;
  bridge::registerBufferizedGemmHandoffDialectsV1(context);
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = true;
  options.copyBeforeWrite = true;
  options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  check(mlir::succeeded(mlir::bufferization::runOneShotModuleBufferize(
            *structured.module, options, state, &statistics)),
        "upstream copy-before-write control must bufferize");
  bool saw_allocation = false;
  bool saw_copy = false;
  (*structured.module).walk([&](mlir::Operation *operation) {
    saw_allocation |= mlir::isa<mlir::memref::AllocOp>(operation);
    saw_copy |= mlir::isa<mlir::memref::CopyOp>(operation);
  });
  check(saw_allocation && saw_copy && statistics.numBufferAlloc > 0,
        "copy-before-write control must expose allocation and copy rather "
        "than satisfying certified no-copy postconditions");
  auto function = oneFunction(*structured.module);
  auto return_op = findOne<mlir::func::ReturnOp>(*structured.module);
  check(return_op.getOperand(0) != function.getArgument(2),
        "copy-before-write result must not claim original C identity");
}

} // namespace

int main() {
  v1::Module capture = readCapture();
  check(capture.operations.size() == 1,
        "bufferization suite requires one reviewed GEMM capture");
  if (capture.operations.size() == 1) {
    testDynamicBufferization(capture);
    testStaticNonSquareBufferization(capture);
    testMixedStaticDynamicMultiSiteBufferization(capture);
    testCrossContextRoundTrip(capture);
    testSourceMismatch(capture);
    testSameShapedAlternateSource(capture);
    testMultiSiteIdentityMutations(capture);
    testAdversarialMutations(capture);
    testUpstreamNoFunctionBoundaryFalsifier(capture);
    testUpstreamConflictFalsifier(capture);
    testUpstreamCopyBeforeWriteFalsifier(capture);
  }
  if (failures != 0) {
    std::cerr << "FAIL: " << failures << " of " << checks
              << " bufferization checks failed\n";
    return 1;
  }
  std::cout << "PASS: " << checks
            << " certified bufferization checks, 0 failures\n";
  return 0;
}
