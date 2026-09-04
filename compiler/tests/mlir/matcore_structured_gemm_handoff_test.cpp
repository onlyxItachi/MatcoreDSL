#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreContractionModel.h"
#include "MatcoreOps.h"
#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreStructuredHandoffCertificate.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;
namespace dialect = matcore::mdslc::mlir_dialect;
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

void makeStatic(v1::TensorValue &value, std::uint64_t rows,
                std::uint64_t columns) {
  value.type.shape = {v1::ScalarExpr::staticValue(rows),
                      v1::ScalarExpr::staticValue(columns)};
  value.type.strides = {v1::ScalarExpr::staticValue(columns),
                        v1::ScalarExpr::staticValue(1)};
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
  check(replaced, "mutated structured-contract field must exist");
  return builder.getDictionaryAttr(attributes);
}

mlir::DictionaryAttr sourceRange(mlir::Builder &builder, std::int64_t begin,
                                 std::int64_t end) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("begin", builder.getI64IntegerAttr(begin)),
       builder.getNamedAttr("end", builder.getI64IntegerAttr(end))});
}

mlir::DictionaryAttr proofRange(mlir::Builder &builder, llvm::StringRef kind,
                                std::int64_t begin, std::int64_t end) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("begin", builder.getI64IntegerAttr(begin)),
       builder.getNamedAttr("end", builder.getI64IntegerAttr(end)),
       builder.getNamedAttr("kind", builder.getStringAttr(kind))});
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

mlir::func::FuncOp structuredFunction(mlir::ModuleOp module) {
  return findOne<mlir::func::FuncOp>(module);
}

mlir::DictionaryAttr semanticContract(mlir::func::FuncOp function) {
  return function->getAttrOfType<mlir::DictionaryAttr>(
      "mdsl.semantic_contract");
}

std::vector<float>
evaluateStaticStructuredFunction(mlir::func::FuncOp function,
                                 llvm::ArrayRef<float> lhs,
                                 llvm::ArrayRef<float> rhs,
                                 llvm::ArrayRef<float> destination) {
  llvm::DenseMap<mlir::Value, std::vector<float>> tensors;
  llvm::DenseMap<mlir::Value, float> scalars;
  tensors[function.getArgument(0)] = std::vector<float>(lhs.begin(), lhs.end());
  tensors[function.getArgument(1)] = std::vector<float>(rhs.begin(), rhs.end());
  tensors[function.getArgument(2)] =
      std::vector<float>(destination.begin(), destination.end());

  for (mlir::Operation &operation : function.getBody().front()) {
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation)) {
      const auto value = mlir::cast<mlir::FloatAttr>(constant.getValue());
      scalars[constant.getResult()] =
          static_cast<float>(value.getValueAsDouble());
      continue;
    }
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(operation)) {
      const auto destination_value = tensors.find(fill.getOutputs().front());
      const auto scalar_value = scalars.find(fill.getInputs().front());
      check(destination_value != tensors.end() && scalar_value != scalars.end(),
            "test evaluator must resolve fill operands through actual SSA");
      if (destination_value == tensors.end() || scalar_value == scalars.end())
        return {};
      tensors[fill.getResultTensors().front()] = std::vector<float>(
          destination_value->second.size(), scalar_value->second);
      continue;
    }
    if (auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(operation)) {
      const auto lhs_value = tensors.find(matmul.getInputs()[0]);
      const auto rhs_value = tensors.find(matmul.getInputs()[1]);
      const auto output_value = tensors.find(matmul.getOutputs().front());
      check(lhs_value != tensors.end() && rhs_value != tensors.end() &&
                output_value != tensors.end(),
            "test evaluator must resolve matmul operands through actual SSA");
      if (lhs_value == tensors.end() || rhs_value == tensors.end() ||
          output_value == tensors.end())
        return {};
      const auto lhs_type =
          mlir::cast<mlir::RankedTensorType>(matmul.getInputs()[0].getType());
      const auto rhs_type =
          mlir::cast<mlir::RankedTensorType>(matmul.getInputs()[1].getType());
      const std::int64_t m = lhs_type.getShape()[0];
      const std::int64_t k = lhs_type.getShape()[1];
      const std::int64_t n = rhs_type.getShape()[1];
      std::vector<float> output = output_value->second;
      for (std::int64_t row = 0; row < m; ++row) {
        for (std::int64_t column = 0; column < n; ++column) {
          float accumulator = output[static_cast<std::size_t>(row * n + column)];
          for (std::int64_t reduction = 0; reduction < k; ++reduction) {
            accumulator +=
                lhs_value->second[static_cast<std::size_t>(row * k + reduction)] *
                rhs_value->second[static_cast<std::size_t>(reduction * n + column)];
          }
          output[static_cast<std::size_t>(row * n + column)] = accumulator;
        }
      }
      tensors[matmul.getResultTensors().front()] = std::move(output);
      continue;
    }
    if (auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(operation)) {
      const auto result = tensors.find(return_op.getOperand(0));
      check(result != tensors.end(),
            "test evaluator must return the value selected by actual SSA");
      return result == tensors.end() ? std::vector<float>{} : result->second;
    }
  }
  check(false, "test evaluator must encounter a structured return");
  return {};
}

void replaceContract(mlir::func::FuncOp function, mlir::Builder &builder,
                     llvm::StringRef field, mlir::Attribute replacement) {
  function->setAttr(
      "mdsl.semantic_contract",
      withField(builder, semanticContract(function), field, replacement));
}

// Convert a valid explicit operation into the already supported, core-valid
// recovered-C++ analysis form. This lets the handoff test the recognition /
// permission firewall rather than merely feeding it malformed attributes.
void makeRecoveredCppLoopContract(dialect::GemmOp operation,
                                  mlir::Builder &builder) {
  const auto old_provenance = operation.getProvenance();
  const auto old_range =
      old_provenance.getAs<mlir::DictionaryAttr>("call_range");
  const std::int64_t begin =
      old_range.getAs<mlir::IntegerAttr>("begin").getInt();
  const std::int64_t end =
      old_range.getAs<mlir::IntegerAttr>("end").getInt();

  operation->setAttr(
      "origin",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("kind",
                                builder.getStringAttr("recovered_cpp_loop")),
           builder.getNamedAttr(
               "pattern",
               builder.getStringAttr("canonical-row-major-f32-gemm-v1")),
           builder.getNamedAttr(
               "permission",
               builder.getStringAttr("source_proven_guard_required")),
           builder.getNamedAttr("version", builder.getI32IntegerAttr(1))}));

  auto numerical = operation.getNumerical();
  numerical = withField(
      builder, numerical, "profile",
      builder.getStringAttr("recovered-cpp-gemm-f32-source-proven-v1"));
  numerical = withField(builder, numerical, "derivation",
                        builder.getStringAttr("effective_cpp_semantics"));
  operation->setAttr("numerical", numerical);
  operation->setAttr(
      "policy",
      builder.getDictionaryAttr(
          {builder.getNamedAttr(
               "fallback", builder.getStringAttr("preserve_original_cpp")),
           builder.getNamedAttr("target",
                                builder.getStringAttr("generic"))}));

  const auto file = old_provenance.getAs<mlir::StringAttr>("file");
  const auto line = old_provenance.getAs<mlir::IntegerAttr>("line");
  const auto column = old_provenance.getAs<mlir::IntegerAttr>("column");
  const auto offset = old_provenance.getAs<mlir::IntegerAttr>("offset");
  operation->setAttr(
      "provenance",
      builder.getDictionaryAttr(
          {builder.getNamedAttr(
               "column", builder.getI64IntegerAttr(column.getInt())),
           builder.getNamedAttr(
               "compilation_identity",
               builder.getStringAttr("handoff-adversarial-test")),
           builder.getNamedAttr("file", file),
           builder.getNamedAttr(
               "kind", builder.getStringAttr("recovered_cpp_loop")),
           builder.getNamedAttr("line",
                                builder.getI64IntegerAttr(line.getInt())),
           builder.getNamedAttr("offset",
                                builder.getI64IntegerAttr(offset.getInt())),
           builder.getNamedAttr(
               "proof_ranges",
               builder.getArrayAttr(
                   {proofRange(builder, "outer_loop", begin, end),
                    proofRange(builder, "accumulator_update", begin + 1,
                               begin + 2),
                    proofRange(builder, "output_store", begin + 2,
                               begin + 3)})),
           builder.getNamedAttr("source_range",
                                sourceRange(builder, begin, end)),
           builder.getNamedAttr(
               "source_snapshot",
               builder.getStringAttr(
                   "sha256:0000000000000000000000000000000000000000000000000000000000000000")),
           builder.getNamedAttr("version", builder.getI32IntegerAttr(1))}));
}

using StructuredMutation =
    std::function<void(mlir::ModuleOp, mlir::Builder &)>;

void expectStructuredRejected(const v1::Module &capture,
                              const StructuredMutation &mutate,
                              std::string_view message) {
  mlir::MLIRContext context;
  auto semantic = buildSemantic(capture, context);
  check(static_cast<bool>(semantic),
        "structured mutation fixture must bridge before damage");
  if (!semantic)
    return;
  auto structured = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  check(static_cast<bool>(structured),
        "structured mutation fixture must derive before damage");
  if (!structured)
    return;

  mlir::Builder builder(&context);
  mutate(*structured.module, builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(mlir::succeeded(mlir::verify(*structured.module)),
        std::string(message) + " (mutation must remain valid generic MLIR)");
  std::string error;
  check(!bridge::verifyStructuredGemmHandoffV1(*structured.module, error),
        message);
  check(!error.empty(),
        "structured handoff rejection must provide a diagnostic");
}

void testDynamicHandoff(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto semantic = buildSemantic(capture, context);
  check(static_cast<bool>(semantic),
        "dynamic v1 capture must bridge before structured projection");
  if (!semantic)
    return;

  const std::string semantic_before =
      bridge::serializeDeterministicMlir(*semantic.module);
  auto structured = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  check(static_cast<bool>(structured),
        "verified dynamic explicit GEMM must derive a structured handoff");
  if (!structured) {
    std::cerr << structured.error << '\n';
    return;
  }
  check(bridge::serializeDeterministicMlir(*semantic.module) == semantic_before,
        "successful derivation must not mutate its semantic source");

  std::string error;
  check(bridge::verifyStructuredGemmHandoffV1(*structured.module, error),
        "dynamic structured handoff must verify standalone");
  check(bridge::verifyStructuredGemmHandoffMatchesV1(
            *semantic.module, *structured.module, error),
        "dynamic structured handoff must exactly match its semantic source");
  const auto producer = (*structured.module)
                            ->getAttrOfType<mlir::StringAttr>("mdsl.producer");
  const auto source_producer =
      (*structured.module)
          ->getAttrOfType<mlir::StringAttr>("mdsl.source_producer");
  check(producer &&
            producer.getValue() == "matcore-structured-gemm-handoff-v1",
        "structured artifact must identify its actual producer");
  check(source_producer && source_producer.getValue() == "clang-libtooling-v1",
        "structured artifact must retain the declared source producer");

  auto function = structuredFunction(*structured.module);
  auto constant = findOne<mlir::arith::ConstantOp>(*structured.module);
  auto fill = findOne<mlir::linalg::FillOp>(*structured.module);
  auto matmul = findOne<mlir::linalg::MatmulOp>(*structured.module);
  auto return_op = findOne<mlir::func::ReturnOp>(*structured.module);
  check(function && constant && fill && matmul && return_op,
        "handoff must contain func, positive zero, fill, matmul, and return");
  if (!function || !constant || !fill || !matmul || !return_op)
    return;
  mlir::Block &block = function.getBody().front();
  const auto zero = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
  check(zero && zero.getValue().isZero() && !zero.getValue().isNegative(),
        "overwrite seed must be exact positive floating-point zero");
  check(fill.getInputs().front() == constant.getResult() &&
            fill.getOutputs().front() == block.getArgument(2),
        "fill must overwrite the original destination argument");
  check(matmul.getInputs()[0] == block.getArgument(0) &&
            matmul.getInputs()[1] == block.getArgument(1) &&
            matmul.getOutputs().front() == fill.getResultTensors().front(),
        "matmul must consume lhs/rhs and only the zero-filled destination");
  check(return_op.getOperand(0) == matmul.getResultTensors().front(),
        "structured function must return the matmul result");
  check(!matmul.hasUserDefinedMaps(),
        "structured handoff must use default canonical matmul maps");
  auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(
      matmul.getRegion().front().front());
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(
      *std::next(matmul.getRegion().front().begin()));
  check(multiply && add &&
            multiply.getFastmath() == mlir::arith::FastMathFlags::none &&
            add.getFastmath() == mlir::arith::FastMathFlags::none,
        "structured scalar contraction must introduce no fast-math flags");

  const std::string text =
      bridge::serializeDeterministicMlir(*structured.module);
  checkContains(text, "mdsl.analysis_only = true",
                "structured text must carry the analysis-only firewall");
  checkContains(text, "linalg.fill",
                "structured text must expose destination overwrite");
  checkContains(text, "linalg.matmul",
                "structured text must expose the contraction");
  checkContains(text, "alignment_contract = \"required_precondition\"",
                "structured text must retain alignment as a precondition");
  checkContains(text, "contract = \"required_precondition\"",
                "structured text must retain no-alias as a precondition");
  checkContains(text, "derivation = \"explicit_edsl_contract\"",
                "structured text must retain numerical derivation");

  auto second = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  check(second &&
            bridge::serializeDeterministicMlir(*second.module) == text,
        "repeated derivation must be byte deterministic");

  mlir::MLIRContext parse_context;
  bridge::registerStructuredGemmHandoffDialectsV1(parse_context);
  mlir::ParserConfig parser_config(&parse_context,
                                   /*verifyAfterParse=*/true);
  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(text, parser_config);
  check(static_cast<bool>(parsed),
        "printed structured handoff must parse as verified MLIR");
  if (parsed) {
    check(bridge::serializeDeterministicMlir(*parsed) == text,
          "structured parse/print round trip must be byte stable");
    check(bridge::verifyStructuredGemmHandoffV1(*parsed, error),
          "reparsed structured handoff must verify standalone");
    check(bridge::verifyStructuredGemmHandoffMatchesV1(
              *semantic.module, *parsed, error),
          "independent-context reparsed handoff must match its semantic "
          "source");
  }

  mlir::ParserConfig source_parser_config(&context,
                                          /*verifyAfterParse=*/true);
  auto source_context_parse = mlir::parseSourceString<mlir::ModuleOp>(
      text, source_parser_config);
  check(source_context_parse &&
            bridge::verifyStructuredGemmHandoffMatchesV1(
                *semantic.module, *source_context_parse, error),
        "same-context reparsed handoff must still match its semantic source");

  std::vector<lowering::CpuRuntimeDispatchRecordV1> records(1);
  check(!lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
             *structured.module, records, error),
        "analysis-only structured handoff must not enter CPU runtime lowering");
  check(records.empty(),
        "runtime-lowering rejection must publish no partial dispatch record");
  checkContains(error, "analysis_only",
                "runtime-lowering rejection must name the authority firewall");
}

void testStaticNonSquareHandoff(v1::Module capture) {
  check(capture.operations.size() == 1,
        "static handoff fixture must contain exactly one operation");
  if (capture.operations.size() != 1)
    return;
  makeStatic(capture.operations[0].operands[0], 2, 3);
  makeStatic(capture.operations[0].operands[1], 3, 4);
  makeStatic(capture.operations[0].output, 2, 4);
  std::string error;
  check(v1::verify(capture, error),
        "non-square 2x3 times 3x4 capture must verify");

  mlir::MLIRContext context;
  auto semantic = buildSemantic(capture, context);
  check(static_cast<bool>(semantic),
        "non-square static capture must bridge");
  if (!semantic)
    return;
  auto structured = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  check(static_cast<bool>(structured),
        "non-square static GEMM must derive a structured handoff");
  if (!structured)
    return;
  check(bridge::verifyStructuredGemmHandoffMatchesV1(
            *semantic.module, *structured.module, error),
        "non-square structured handoff must match its source");

  auto function = structuredFunction(*structured.module);
  check(static_cast<bool>(function),
        "non-square structured handoff must contain a function");
  if (!function)
    return;
  const auto lhs =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getArgument(0).getType());
  const auto rhs =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getArgument(1).getType());
  const auto output =
      mlir::dyn_cast<mlir::RankedTensorType>(function.getArgument(2).getType());
  check(lhs && lhs.getShape()[0] == 2 && lhs.getShape()[1] == 3,
        "static lhs must remain tensor<2x3xf32>");
  check(rhs && rhs.getShape()[0] == 3 && rhs.getShape()[1] == 4,
        "static rhs must remain tensor<3x4xf32>");
  check(output && output.getShape()[0] == 2 && output.getShape()[1] == 4,
        "static destination must remain tensor<2x4xf32>");

  const std::vector<float> zero_lhs(6, 0.0F);
  const std::vector<float> positive_rhs = {1.0F, 2.0F, 3.0F, 4.0F,
                                           5.0F, 6.0F, 7.0F, 8.0F,
                                           9.0F, 10.0F, 11.0F, 12.0F};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> hostile_destination =
      {91.0F, nan, -13.0F, nan, 5.0F, -7.0F, nan, 23.0F};
  const std::vector<float> overwritten = evaluateStaticStructuredFunction(
      function, zero_lhs, positive_rhs, hostile_destination);
  check(overwritten.size() == 8 &&
            llvm::all_of(overwritten, [](float value) {
              return value == 0.0F && !std::signbit(value);
            }),
        "zero lhs must produce positive zero despite nonzero/NaN initial C");

  const std::vector<float> sentinel_lhs = {1.0F, 2.0F, 3.0F,
                                           4.0F, 5.0F, 6.0F};
  const std::vector<float> sentinel_rhs = {
      7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F,
      13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F};
  const std::vector<float> expected = {74.0F,  80.0F,  86.0F,  92.0F,
                                       173.0F, 188.0F, 203.0F, 218.0F};
  check(evaluateStaticStructuredFunction(function, sentinel_lhs, sentinel_rhs,
                                         hostile_destination) == expected,
        "2x3 by 3x4 sentinel GEMM must preserve logical M/K/N indexing");
}

void testUnitExtentIdentityAndZeroExtentAdmission(v1::Module capture) {
  check(capture.operations.size() == 1,
        "extent-boundary fixture must contain exactly one GEMM");
  if (capture.operations.size() != 1)
    return;
  const v1::Module dynamic_capture = capture;

  auto check_unit_geometry = [&](std::uint64_t m, std::uint64_t n,
                                 std::uint64_t k,
                                 std::string_view description) {
    v1::Module unit_capture = dynamic_capture;
    makeStatic(unit_capture.operations[0].operands[0], m, k);
    makeStatic(unit_capture.operations[0].operands[1], k, n);
    makeStatic(unit_capture.operations[0].output, m, n);
    std::string error;
    const bool valid = v1::verify(unit_capture, error);
    check(valid,
          std::string(description) +
              " must remain a valid rank-two GEMM capture");
    if (!valid)
      return;

    mlir::MLIRContext context;
    auto semantic = buildSemantic(unit_capture, context);
    check(static_cast<bool>(semantic),
          std::string(description) + " must bridge as mdsl.gemm");
    if (!semantic)
      return;
    auto structured =
        bridge::deriveStructuredGemmHandoffV1(*semantic.module);
    check(static_cast<bool>(structured),
          std::string(description) + " must derive through the GEMM handoff");
    if (!structured)
      return;
    auto function = structuredFunction(*structured.module);
    auto handoff = function->getAttrOfType<mlir::DictionaryAttr>(
        "mdsl.structured_handoff");
    auto source_operation =
        handoff ? handoff.getAs<mlir::StringAttr>("source_operation")
                : mlir::StringAttr{};
    check(source_operation && source_operation.getValue() == "mdsl.gemm",
          std::string(description) +
              " must not relabel GEMM as GEMV or DOT");
    const auto lhs = mlir::dyn_cast<mlir::RankedTensorType>(
        function.getArgument(0).getType());
    const auto rhs = mlir::dyn_cast<mlir::RankedTensorType>(
        function.getArgument(1).getType());
    const auto output = mlir::dyn_cast<mlir::RankedTensorType>(
        function.getArgument(2).getType());
    check(lhs && rhs && output && lhs.getRank() == 2 && rhs.getRank() == 2 &&
              output.getRank() == 2,
          std::string(description) +
              " must preserve rank-two GEMM geometry");

    auto topology = bridge::buildCanonicalContractionTopologyV1(
        context, bridge::StandardLinearAlgebraOperationV1::Gemm);
    check(topology && topology.topology.operand_ranks ==
                          llvm::ArrayRef<unsigned>({2, 2, 2}),
          std::string(description) +
              " must retain extent-neutral GEMM topology");
  };

  check_unit_geometry(/*m=*/1, /*n=*/4, /*k=*/3, "unit-M GEMM");
  check_unit_geometry(/*m=*/2, /*n=*/1, /*k=*/3, "unit-N GEMM");
  check_unit_geometry(/*m=*/2, /*n=*/4, /*k=*/1, "unit-K GEMM");

  auto check_zero_rejected = [&](std::uint64_t m, std::uint64_t n,
                                 std::uint64_t k,
                                 std::string_view description) {
    v1::Module zero_capture = dynamic_capture;
    makeStatic(zero_capture.operations[0].operands[0], m, k);
    makeStatic(zero_capture.operations[0].operands[1], k, n);
    makeStatic(zero_capture.operations[0].output, m, n);
    std::string error;
    check(!v1::verify(zero_capture, error),
          std::string(description) +
              " must be rejected by authoritative Matcore IR v1");
    checkContains(error, "must be positive",
                  std::string(description) +
                      " rejection must preserve the positive-extent rule");
  };

  check_zero_rejected(/*m=*/0, /*n=*/4, /*k=*/3, "zero-M GEMM");
  check_zero_rejected(/*m=*/2, /*n=*/0, /*k=*/3, "zero-N GEMM");
  check_zero_rejected(/*m=*/2, /*n=*/4, /*k=*/0, "zero-K GEMM");

  std::string error;
  check(v1::verify(dynamic_capture, error),
        "symbolic dynamic GEMM must retain its positive-runtime extent "
        "precondition");
  mlir::MLIRContext dynamic_context;
  auto dynamic_semantic = buildSemantic(dynamic_capture, dynamic_context);
  check(static_cast<bool>(dynamic_semantic),
        "symbolic positive-runtime extents must bridge to dynamic tensor "
        "types without admitting concrete zero");
  if (dynamic_semantic) {
    auto function = findOne<mlir::func::FuncOp>(*dynamic_semantic.module);
    const auto lhs = mlir::dyn_cast<mlir::RankedTensorType>(
        function.getArgument(0).getType());
    check(lhs && lhs.isDynamicDim(0) && lhs.isDynamicDim(1),
          "dynamic MLIR dimensions record unknown extents, not zero-extent "
          "execution authority");
  }
}

void testReusableCertificateFingerprint(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto semantic = buildSemantic(capture, context);
  check(static_cast<bool>(semantic),
        "certificate fingerprint fixture must bridge");
  if (!semantic)
    return;
  auto structured = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  check(static_cast<bool>(structured),
        "certificate fingerprint fixture must derive");
  if (!structured)
    return;
  auto source_function = findOne<mlir::func::FuncOp>(*semantic.module);
  auto source_gemm = findOne<dialect::GemmOp>(*semantic.module);
  auto structured_function = structuredFunction(*structured.module);
  check(source_function && source_gemm && structured_function,
        "certificate fingerprint fixture must contain paired sites");
  if (!source_function || !source_gemm || !structured_function)
    return;
  std::string error;
  const std::string source_fingerprint =
      bridge::computeSourceSemanticFingerprintV1(
          *semantic.module, source_function, source_gemm->getAttrDictionary(),
          "mdsl.gemm", error);
  const std::string structured_fingerprint =
      bridge::computeStructuredSemanticFingerprintV1(
          *structured.module, structured_function, "mdsl.gemm", error);
  check(source_fingerprint.size() == 71 &&
            source_fingerprint.starts_with("sha256:") &&
            source_fingerprint == structured_fingerprint,
        "generic certificate fingerprint must bind the exact source and "
        "structured semantic identities");

  const std::string structured_text =
      bridge::serializeDeterministicMlir(*structured.module);
  mlir::MLIRContext parse_context;
  bridge::registerStructuredGemmHandoffDialectsV1(parse_context);
  mlir::ParserConfig parser_config(&parse_context,
                                   /*verifyAfterParse=*/true);
  auto reparsed = mlir::parseSourceString<mlir::ModuleOp>(structured_text,
                                                          parser_config);
  check(static_cast<bool>(reparsed),
        "fingerprint cross-context fixture must reparse");
  if (reparsed) {
    const std::string cross_context_fingerprint =
        bridge::computeStructuredSemanticFingerprintV1(
            *reparsed, structuredFunction(*reparsed), "mdsl.gemm", error);
    check(cross_context_fingerprint == source_fingerprint,
          "versioned semantic fingerprint must be canonical across MLIR "
          "contexts");
  }

  mlir::Builder builder(&context);
  auto changed_contract = semanticContract(structured_function);
  auto changed_policy = changed_contract.getAs<mlir::DictionaryAttr>("policy");
  changed_policy = withField(builder, changed_policy, "target",
                             builder.getStringAttr("generic"));
  replaceContract(structured_function, builder, "policy", changed_policy);
  const std::string changed_fingerprint =
      bridge::computeStructuredSemanticFingerprintV1(
          *structured.module, structured_function, "mdsl.gemm", error);
  check(!changed_fingerprint.empty() &&
            changed_fingerprint != source_fingerprint,
        "semantic fingerprint must change when an opaque operation contract "
        "changes");
}

void testGenericCertificateVerificationHardening(const v1::Module &capture) {
  mlir::MLIRContext first_context;
  auto first_source = buildSemantic(capture, first_context);
  check(static_cast<bool>(first_source),
        "generic certificate hardening source must bridge");
  if (!first_source)
    return;
  auto first_structured =
      bridge::deriveStructuredGemmHandoffV1(*first_source.module);
  check(static_cast<bool>(first_structured),
        "generic certificate hardening structured fixture must derive");
  if (!first_structured)
    return;

  auto second_source = buildSemantic(capture, first_context);
  check(static_cast<bool>(second_source),
        "same-context mixed-module certificate source must bridge "
        "independently");
  if (!second_source)
    return;
  auto second_structured =
      bridge::deriveStructuredGemmHandoffV1(*second_source.module);
  check(static_cast<bool>(second_structured),
        "same-context mixed-module structured fixture must derive "
        "independently");
  if (!second_structured)
    return;

  auto first_source_function = findOne<mlir::func::FuncOp>(*first_source.module);
  auto first_source_gemm = findOne<dialect::GemmOp>(*first_source.module);
  auto first_structured_function =
      structuredFunction(*first_structured.module);
  auto second_source_function =
      findOne<mlir::func::FuncOp>(*second_source.module);
  auto second_source_gemm = findOne<dialect::GemmOp>(*second_source.module);
  auto second_structured_function =
      structuredFunction(*second_structured.module);
  check(first_source_function && first_source_gemm &&
            first_structured_function && second_source_function &&
            second_source_gemm && second_structured_function,
        "mixed-module certificate fixtures must contain all paired handles");
  if (!first_source_function || !first_source_gemm ||
      !first_structured_function || !second_source_function ||
      !second_source_gemm || !second_structured_function)
    return;

  std::string error;
  check(bridge::computeSourceSemanticFingerprintV1(
            *first_source.module, second_source_function,
            second_source_gemm->getAttrDictionary(), "mdsl.gemm", error)
            .empty(),
        "source fingerprint must reject an identical-looking same-context "
        "function from another module");
  checkContains(error, "direct member",
                "mixed source/module rejection must identify membership");

  check(bridge::computeStructuredSemanticFingerprintV1(
            *first_structured.module, second_structured_function,
            "mdsl.gemm", error)
            .empty(),
        "structured fingerprint must reject an identical-looking same-context "
        "function from another module");
  checkContains(error, "direct member",
                "mixed structured/module rejection must identify membership");

  check(bridge::computeSourceSemanticFingerprintV1(
            *first_source.module, first_structured_function,
            semanticContract(first_structured_function), "mdsl.gemm", error)
            .empty(),
        "source fingerprint must reject a structured-carrier handle even "
        "when its site identity is paired");
  checkContains(error, "direct member",
                "hybrid source/structured rejection must identify membership");

  check(bridge::computeStructuredSemanticFingerprintV1(
            *first_structured.module, first_source_function, "mdsl.gemm",
            error)
            .empty(),
        "structured fingerprint must reject a semantic-carrier handle even "
        "when its site identity is paired");
  checkContains(error, "direct member",
                "hybrid structured/source rejection must identify membership");

  mlir::MLIRContext malformed_context;
  auto malformed_source = buildSemantic(capture, malformed_context);
  check(static_cast<bool>(malformed_source),
        "core-malformed source fixture must bridge before damage");
  if (!malformed_source)
    return;
  auto return_op = findOne<mlir::func::ReturnOp>(*malformed_source.module);
  check(static_cast<bool>(return_op),
        "core-malformed source fixture must contain a return");
  if (!return_op)
    return;
  return_op.erase();
  mlir::ScopedDiagnosticHandler silence(
      &malformed_context, [](mlir::Diagnostic &) { return mlir::success(); });
  check(!bridge::verifyStructuredHandoffSourceEnvelopeV1(
             *malformed_source.module, error),
        "generic source envelope must reject core-invalid MLIR before "
        "certificate inspection");
  checkContains(error, "upstream MLIR verification",
                "core-invalid source rejection must name upstream verification");
}

void testStandaloneVsSourceMatch(const v1::Module &capture) {
  mlir::MLIRContext context;
  auto semantic = buildSemantic(capture, context);
  check(static_cast<bool>(semantic),
        "source-match fixture must bridge before projection");
  if (!semantic)
    return;
  auto structured = bridge::deriveStructuredGemmHandoffV1(*semantic.module);
  check(static_cast<bool>(structured),
        "source-match fixture must derive before metadata mutation");
  if (!structured)
    return;

  mlir::Builder builder(&context);
  (*structured.module)
      ->setAttr("mdsl.translation_unit",
                builder.getStringAttr("different-authenticated-unit.mdsl"));
  std::string error;
  check(bridge::verifyStructuredGemmHandoffV1(*structured.module, error),
        "standalone verifier may accept internally valid independent identity");
  check(!bridge::verifyStructuredGemmHandoffMatchesV1(
             *semantic.module, *structured.module, error),
        "source-match verifier must reject changed translation-unit identity");
  checkContains(error, "translation_unit",
                "source mismatch diagnostic must identify the changed field");
}

void testContractAndDataflowMutations(const v1::Module &capture) {
  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.analysis_only", builder.getBoolAttr(false));
      },
      "structured verifier must reject removal of the analysis-only firewall");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        module->setAttr("mdsl.structured_handoff_version",
                        builder.getI32IntegerAttr(2));
      },
      "structured verifier must reject an unsupported handoff version");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        function.setArgAttr(0, "mdsl.unreviewed_arg_fact",
                            builder.getUnitAttr());
      },
      "structured verifier must reject argument-level optimizer facts");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        structuredFunction(module).setNoInline(true);
      },
      "structured verifier must reject an inherited no_inline policy");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        structuredFunction(module).setSymVisibilityAttr(
            builder.getStringAttr("public"));
      },
      "structured verifier must reject explicit symbol visibility");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        structuredFunction(module)->setAttr(
            "mdsl.source_semantic_symbol",
            builder.getStringAttr("__matcore_semantic_mc_00000000000000000000000000000000"));
      },
      "structured verifier must reject a mismatched source semantic symbol");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto handoff = function->getAttrOfType<mlir::DictionaryAttr>(
            "mdsl.structured_handoff");
        function->setAttr(
            "mdsl.structured_handoff",
            withField(builder, handoff, "destination",
                      builder.getStringAttr("accumulate_initial_output")));
      },
      "structured verifier must reject an altered destination contract");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        replaceContract(structuredFunction(module), builder,
                        "accumulation_type",
                        builder.getStringAttr("not-a-type"));
      },
      "structured verifier must reject malformed contract storage types");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto contract = semanticContract(function);
        auto numerical =
            contract.getAs<mlir::DictionaryAttr>("numerical");
        numerical = withField(builder, numerical, "reassociation",
                              builder.getStringAttr("forbidden"));
        replaceContract(function, builder, "numerical", numerical);
      },
      "structured verifier must reject altered numerical permissions");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto contract = semanticContract(function);
        auto provenance =
            contract.getAs<mlir::DictionaryAttr>("provenance");
        provenance = withField(builder, provenance, "file",
                               builder.getStringAttr("other.mdsl"));
        replaceContract(function, builder, "provenance", provenance);
      },
      "structured verifier must reject forged source provenance");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto contract = semanticContract(function);
        auto aliasing = contract.getAs<mlir::ArrayAttr>("aliasing");
        auto first = mlir::cast<mlir::DictionaryAttr>(aliasing[0]);
        first = withField(builder, first, "contract",
                          builder.getStringAttr("proven_fact"));
        llvm::SmallVector<mlir::Attribute> relations(aliasing.begin(),
                                                     aliasing.end());
        relations[0] = first;
        replaceContract(function, builder, "aliasing",
                        builder.getArrayAttr(relations));
      },
      "structured verifier must not upgrade no-alias preconditions to facts");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto contract = semanticContract(function);
        auto lhs = contract.getAs<mlir::DictionaryAttr>("lhs_semantics");
        lhs = withField(builder, lhs, "alignment_contract",
                        builder.getStringAttr("proven_fact"));
        replaceContract(function, builder, "lhs_semantics", lhs);
      },
      "structured verifier must not upgrade alignment preconditions to facts");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto contract = semanticContract(function);
        auto lhs = contract.getAs<mlir::DictionaryAttr>("lhs_semantics");
        lhs = withField(builder, lhs, "layout",
                        builder.getStringAttr("column_major"));
        replaceContract(function, builder, "lhs_semantics", lhs);
      },
      "structured verifier must reject a layout claim inconsistent with types");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto function = structuredFunction(module);
        auto contract = semanticContract(function);
        auto effects = contract.getAs<mlir::DictionaryAttr>("effects");
        effects = withField(builder, effects, "writes",
                            builder.getArrayAttr({}));
        replaceContract(function, builder, "effects", effects);
      },
      "structured verifier must reject loss of the observable output write");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto constant = findOne<mlir::arith::ConstantOp>(module);
        constant.setValueAttr(builder.getF32FloatAttr(1.0));
      },
      "structured verifier must reject a nonzero overwrite seed");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        findOne<mlir::arith::ConstantOp>(module).setValueAttr(
            builder.getF32FloatAttr(-0.0));
      },
      "structured verifier must reject negative-zero overwrite seed");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto fill = findOne<mlir::linalg::FillOp>(module);
        mlir::Block &block = fill.getRegion().front();
        auto yield = mlir::cast<mlir::linalg::YieldOp>(block.front());
        yield->setOperand(0, block.getArgument(1));
      },
      "structured verifier must reject a fill that yields old C");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto function = structuredFunction(module);
        auto matmul = findOne<mlir::linalg::MatmulOp>(module);
        matmul->setOperand(2, function.getArgument(2));
      },
      "structured verifier must reject beta-one/direct-destination matmul");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto function = structuredFunction(module);
        auto matmul = findOne<mlir::linalg::MatmulOp>(module);
        matmul->setOperand(0, function.getArgument(1));
        matmul->setOperand(1, function.getArgument(0));
      },
      "structured verifier must reject swapped lhs/rhs dataflow");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto fill = findOne<mlir::linalg::FillOp>(module);
        auto return_op = findOne<mlir::func::ReturnOp>(module);
        return_op->setOperand(0, fill.getResultTensors().front());
      },
      "structured verifier must reject bypassing the matmul result at return");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto fill = findOne<mlir::linalg::FillOp>(module);
        fill->setAttr("mdsl.unreviewed_hint", builder.getBoolAttr(true));
      },
      "structured verifier must reject unexpected discardable attributes");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        findOne<mlir::linalg::FillOp>(module)->setAttr(
            "mdsl.structured_role", builder.getStringAttr("accumulate"));
      },
      "structured verifier must reject an altered fill role");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        findOne<mlir::linalg::FillOp>(module)->setLoc(builder.getUnknownLoc());
      },
      "structured verifier must reject loss of the source operation location");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &builder) {
        auto matmul = findOne<mlir::linalg::MatmulOp>(module);
        mlir::MLIRContext *context = module.getContext();
        const mlir::AffineExpr m = mlir::getAffineDimExpr(0, context);
        const mlir::AffineExpr n = mlir::getAffineDimExpr(1, context);
        const mlir::AffineExpr k = mlir::getAffineDimExpr(2, context);
        matmul.setIndexingMapsAttr(builder.getAffineMapArrayAttr(
            {mlir::AffineMap::get(3, 0, {k, m}, context),
             mlir::AffineMap::get(3, 0, {k, n}, context),
             mlir::AffineMap::get(3, 0, {m, n}, context)}));
      },
      "structured verifier must reject noncanonical matmul indexing maps");

  expectStructuredRejected(
      capture,
      [](mlir::ModuleOp module, mlir::Builder &) {
        auto matmul = findOne<mlir::linalg::MatmulOp>(module);
        auto multiply = mlir::cast<mlir::arith::MulFOp>(
            matmul.getRegion().front().front());
        multiply.setFastmath(mlir::arith::FastMathFlags::fast);
      },
      "structured verifier must reject introduced fast-math permissions");
}

void testUnsupportedSourceFirewalls(const v1::Module &capture) {
  mlir::MLIRContext recovered_context;
  auto recovered = buildSemantic(capture, recovered_context);
  check(static_cast<bool>(recovered),
        "recovered firewall fixture must bridge before origin conversion");
  if (recovered) {
    auto gemm = findOne<dialect::GemmOp>(*recovered.module);
    mlir::Builder builder(&recovered_context);
    makeRecoveredCppLoopContract(gemm, builder);
    check(mlir::succeeded(mlir::verify(*recovered.module)),
          "recovered firewall fixture must remain core-valid semantic MLIR");
    const std::string before =
        bridge::serializeDeterministicMlir(*recovered.module);
    mlir::ScopedDiagnosticHandler silence(
        &recovered_context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    auto rejected = bridge::deriveStructuredGemmHandoffV1(*recovered.module);
    check(!rejected,
          "core-valid recovered GEMM must not acquire structured permission");
    checkContains(rejected.error, "exact verified explicit",
                  "recovered rejection must name the strict source envelope");
    check(bridge::serializeDeterministicMlir(*recovered.module) == before,
          "failed recovered derivation must not mutate its source");
  }

  mlir::MLIRContext map_context;
  bridge::registerMatcoreSemanticDialects(map_context);
  const std::string composition =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) +
               "/gemm_sin_all.semantic.golden.mlir");
  mlir::ParserConfig parser_config(&map_context, /*verifyAfterParse=*/true);
  auto map_module =
      mlir::parseSourceString<mlir::ModuleOp>(composition, parser_config);
  check(static_cast<bool>(map_module),
        "map/GEMM composition fixture must parse and core-verify");
  if (map_module) {
    const std::string before = bridge::serializeDeterministicMlir(*map_module);
    mlir::ScopedDiagnosticHandler silence(
        &map_context, [](mlir::Diagnostic &) { return mlir::success(); });
    auto rejected = bridge::deriveStructuredGemmHandoffV1(*map_module);
    check(!rejected,
          "map/GEMM composition must not enter the explicit-only handoff");
    check(bridge::serializeDeterministicMlir(*map_module) == before,
          "failed composition derivation must not mutate its source");
  }

  mlir::MLIRContext malformed_context;
  auto malformed = buildSemantic(capture, malformed_context);
  check(static_cast<bool>(malformed),
        "malformed-source fixture must bridge before damage");
  if (malformed) {
    mlir::Builder builder(&malformed_context);
    (*malformed.module)
        ->setAttr("mdsl.execution_intent",
                  builder.getStringAttr("training"));
    const std::string before =
        bridge::serializeDeterministicMlir(*malformed.module);
    mlir::ScopedDiagnosticHandler silence(
        &malformed_context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    auto rejected = bridge::deriveStructuredGemmHandoffV1(*malformed.module);
    check(!rejected, "malformed source envelope must fail closed");
    check(!rejected.module,
          "failed source validation must publish no partial structured module");
    check(bridge::serializeDeterministicMlir(*malformed.module) == before,
          "failed malformed derivation must leave its source transactional");
  }

  mlir::MLIRContext extended_source_context;
  auto extended_source = buildSemantic(capture, extended_source_context);
  check(static_cast<bool>(extended_source),
        "extended-source fixture must bridge before metadata injection");
  if (extended_source) {
    auto function = findOne<mlir::func::FuncOp>(*extended_source.module);
    mlir::Builder builder(&extended_source_context);
    function->setAttr("mdsl.unreviewed_source_hint",
                      builder.getBoolAttr(true));
    mlir::ScopedDiagnosticHandler silence(
        &extended_source_context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::succeeded(mlir::verify(*extended_source.module)),
          "extra source metadata fixture must remain generic MLIR-valid");
    auto rejected =
        bridge::deriveStructuredGemmHandoffV1(*extended_source.module);
    check(!rejected,
          "handoff must not silently drop unversioned source metadata");
  }

  mlir::MLIRContext attributed_source_context;
  auto attributed_source = buildSemantic(capture, attributed_source_context);
  check(static_cast<bool>(attributed_source),
        "attributed-source fixture must bridge before fact injection");
  if (attributed_source) {
    auto function = findOne<mlir::func::FuncOp>(*attributed_source.module);
    mlir::Builder builder(&attributed_source_context);
    function.setArgAttr(0, "mdsl.unreviewed_arg_fact",
                        builder.getUnitAttr());
    mlir::ScopedDiagnosticHandler silence(
        &attributed_source_context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::succeeded(mlir::verify(*attributed_source.module)),
          "source argument fact fixture must remain generic MLIR-valid");
    auto rejected =
        bridge::deriveStructuredGemmHandoffV1(*attributed_source.module);
    check(!rejected,
          "handoff must not silently drop source argument optimizer facts");
  }

  mlir::MLIRContext no_inline_source_context;
  auto no_inline_source = buildSemantic(capture, no_inline_source_context);
  check(static_cast<bool>(no_inline_source),
        "no-inline source fixture must bridge before policy injection");
  if (no_inline_source) {
    auto function = findOne<mlir::func::FuncOp>(*no_inline_source.module);
    function.setNoInline(true);
    mlir::ScopedDiagnosticHandler silence(
        &no_inline_source_context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::succeeded(mlir::verify(*no_inline_source.module)),
          "source no_inline fixture must remain generic MLIR-valid");
    auto rejected =
        bridge::deriveStructuredGemmHandoffV1(*no_inline_source.module);
    check(!rejected,
          "handoff must not silently drop source no_inline policy");
  }

  mlir::MLIRContext visibility_source_context;
  auto visibility_source = buildSemantic(capture, visibility_source_context);
  check(static_cast<bool>(visibility_source),
        "visibility source fixture must bridge before policy injection");
  if (visibility_source) {
    auto function = findOne<mlir::func::FuncOp>(*visibility_source.module);
    mlir::Builder builder(&visibility_source_context);
    function.setSymVisibilityAttr(builder.getStringAttr("public"));
    mlir::ScopedDiagnosticHandler silence(
        &visibility_source_context,
        [](mlir::Diagnostic &) { return mlir::success(); });
    check(mlir::succeeded(mlir::verify(*visibility_source.module)),
          "explicit source visibility fixture must remain generic MLIR-valid");
    auto rejected =
        bridge::deriveStructuredGemmHandoffV1(*visibility_source.module);
    check(!rejected,
          "handoff must not silently drop explicit source visibility");
  }
}

} // namespace

int main() {
  const v1::Module capture = readCapture();
  check(capture.operations.size() == 1,
        "structured handoff fixture must contain exactly one GEMM");
  if (capture.operations.size() != 1)
    return 1;

  testDynamicHandoff(capture);
  testStaticNonSquareHandoff(capture);
  testUnitExtentIdentityAndZeroExtentAdmission(capture);
  testReusableCertificateFingerprint(capture);
  testGenericCertificateVerificationHardening(capture);
  testStandaloneVsSourceMatch(capture);
  testContractAndDataflowMutations(capture);
  testUnsupportedSourceFirewalls(capture);

  if (failures != 0) {
    std::cerr << "Structured GEMM handoff adversarial tests: " << failures
              << " of " << checks << " checks failed\n";
    return 1;
  }
  std::cout << "Structured GEMM handoff adversarial tests: " << checks
            << " checks, 0 failures\n";
  return 0;
}
