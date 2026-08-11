#include "MatcoreOps.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"

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
namespace v1 = matcore::mdslc::ir::v1;

int failures = 0;
int checks = 0;

void check(bool condition, std::string_view message) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
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
        "reviewed v1 capture must parse and verify");
  return module;
}

bridge::BridgeResult build(const v1::Module &capture,
                           mlir::MLIRContext &context,
                           const bridge::BridgeContext &bridge_context) {
  return bridge::bridgeV1ToMatcoreMlir(capture, context, bridge_context);
}

dialect::GemmOp findGemm(mlir::ModuleOp module) {
  dialect::GemmOp result;
  module.walk([&](dialect::GemmOp operation) { result = operation; });
  return result;
}

void expectContextRejected(
    const v1::Module &capture, bridge::BridgeContext context,
    const std::function<void(bridge::BridgeContext &)> &mutate,
    std::string_view message) {
  mutate(context);
  mlir::MLIRContext mlir_context;
  auto result = build(capture, mlir_context, context);
  check(!result, message);
  checkContains(result.error, "complete explicit-gemm-f32-v1",
                "invalid bridge context diagnostic must name the profile");
}

mlir::DictionaryAttr withoutField(mlir::Builder &builder,
                                  mlir::DictionaryAttr dictionary,
                                  llvm::StringRef field) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (mlir::NamedAttribute attribute : dictionary) {
    if (attribute.getName().strref() != field)
      attributes.push_back(attribute);
  }
  return builder.getDictionaryAttr(attributes);
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
  check(replaced, "mutated semantic field must exist");
  return builder.getDictionaryAttr(attributes);
}

using OperationMutation =
    std::function<void(dialect::GemmOp, mlir::Builder &)>;

void expectOperationRejected(const v1::Module &capture,
                             const bridge::BridgeContext &profile,
                             const OperationMutation &mutate,
                             std::string_view message) {
  mlir::MLIRContext context;
  auto result = build(capture, context, profile);
  check(static_cast<bool>(result), "mutation fixture must bridge before damage");
  if (!result)
    return;
  dialect::GemmOp gemm = findGemm(*result.module);
  mlir::Builder builder(&context);
  mutate(gemm, builder);
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  std::string error;
  check(!bridge::verifyMatcoreSemanticModule(*result.module, error), message);
}

void makeStatic(v1::TensorValue &value, std::uint64_t rows,
                std::uint64_t columns) {
  value.type.shape = {v1::ScalarExpr::staticValue(rows),
                      v1::ScalarExpr::staticValue(columns)};
  value.type.strides = {v1::ScalarExpr::staticValue(columns),
                        v1::ScalarExpr::staticValue(1)};
}

} // namespace

int main() {
  const v1::Module capture = readCapture();
  const bridge::BridgeContext explicit_profile =
      bridge::explicitGemmF32V1BridgeContext();

  check(explicit_profile.numerical_profile == bridge::kExplicitGemmF32Profile,
        "profile constructor must name explicit-gemm-f32-v1");
  check(explicit_profile.numerical.accumulation_dtype == v1::DType::F32 &&
            explicit_profile.numerical.reassociation ==
                bridge::ReassociationSemantics::WithinReduction &&
            explicit_profile.numerical.contraction ==
                bridge::ContractionSemantics::Allowed &&
            explicit_profile.numerical.reduction_order ==
                bridge::ReductionOrderSemantics::ImplementationDefinedWithinK &&
            explicit_profile.numerical.nan ==
                bridge::NaNSemantics::
                    PreserveClassificationPayloadOrderUnspecified &&
            explicit_profile.numerical.signed_zero ==
                bridge::SignedZeroSemantics::Relaxed &&
            explicit_profile.numerical.rounding ==
                bridge::RoundingSemantics::NearestTiesEven &&
            explicit_profile.numerical.trapping_exceptions ==
                bridge::TrappingExceptionSemantics::Unsupported &&
            explicit_profile.numerical.exception_status ==
                bridge::ExceptionStatusSemantics::
                    IncomingNotPreservedPostCallUnspecified &&
            explicit_profile.numerical.subnormals ==
                bridge::SubnormalSemantics::IeeeGradualFtzDazForbidden &&
            explicit_profile.numerical.approximate_math ==
                bridge::Permission::Forbidden &&
            explicit_profile.numerical.inplace ==
                bridge::Permission::Forbidden,
        "profile must explicitly encode every numerical and FP-environment field");

  mlir::MLIRContext context;
  auto result = build(capture, context, explicit_profile);
  check(static_cast<bool>(result), "verified v1 must bridge to Matcore MLIR");
  if (!result) {
    std::cerr << result.error << '\n';
    return 1;
  }
  std::string verify_error;
  check(bridge::verifyMatcoreSemanticModule(*result.module, verify_error),
        "constructed Matcore semantic module must verify");

  const std::string text = bridge::serializeDeterministicMlir(*result.module);
  const std::string golden =
      readFile(std::string(MDSLC_MLIR_TEST_SOURCE_DIR) +
               "/gemm_capture.semantic.golden.mlir");
  check(text == golden, "semantic MLIR must match reviewed golden bytes");
  check(!text.empty() && text.back() == '\n' &&
            (text.size() == 1 || text[text.size() - 2] != '\n'),
        "semantic text must have exactly one trailing LF");
  checkContains(text, "mdsl.gemm", "text must contain registered GEMM op");
  checkContains(text, "origin = {canonical_callee = \"matcore::mdsl::gemm\", "
                      "kind = \"explicit_call\"}",
                "explicit capture origin must remain authenticated and extensible");
  checkContains(text, "required_precondition",
                "alias and alignment must remain required preconditions");
  checkContains(text, "incoming_not_preserved_postcall_unspecified",
                "exception-status contract must be inspectable");
  checkContains(text, "ieee_gradual_ftz_daz_forbidden",
                "subnormal contract must be inspectable");

  mlir::MLIRContext second_context;
  auto second = build(capture, second_context, explicit_profile);
  check(second && bridge::serializeDeterministicMlir(*second.module) == text,
        "two bridges of identical v1 input must be byte deterministic");

  mlir::MLIRContext parse_context;
  bridge::registerMatcoreSemanticDialects(parse_context);
  mlir::ParserConfig parser_config(&parse_context, /*verifyAfterParse=*/true);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, parser_config);
  check(static_cast<bool>(parsed), "printed semantic MLIR must parse and verify");
  check(parsed && bridge::verifyMatcoreSemanticModule(*parsed, verify_error),
        "reparsed module must satisfy bridge-level invariants");
  check(parsed && bridge::serializeDeterministicMlir(*parsed) == text,
        "parse/print round trip must be byte stable");

  dialect::GemmOp gemm = findGemm(*result.module);
  check(static_cast<bool>(gemm), "module must contain one typed GemmOp");
  check(mlir::isa<mlir::DestinationStyleOpInterface>(gemm.getOperation()),
        "GEMM must implement DestinationStyleOpInterface");
  check(mlir::isa<mlir::MemoryEffectOpInterface>(gemm.getOperation()),
        "GEMM must implement MemoryEffectOpInterface");
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  gemm.getEffects(effects);
  check(effects.size() == 3 &&
            mlir::isa<mlir::MemoryEffects::Read>(effects[0].getEffect()) &&
            effects[0].getValue() == gemm.getLhs() &&
            mlir::isa<mlir::MemoryEffects::Read>(effects[1].getEffect()) &&
            effects[1].getValue() == gemm.getRhs() &&
            mlir::isa<mlir::MemoryEffects::Write>(effects[2].getEffect()) &&
            effects[2].getValue() == gemm.getOutput(),
        "effect interface must report lhs/rhs reads and observable destination write");
  check(!mlir::isMemoryEffectFree(gemm.getOperation()) &&
            !mlir::wouldOpBeTriviallyDead(gemm.getOperation()),
        "observable destination write must prevent pure/dead classification");
  auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(gemm.getOperation());
  check(dps.getNumDpsInputs() == 2 && dps.getNumDpsInits() == 1 &&
            dps.getTiedOpResult(dps.getDpsInitOperand(0)) == gemm.getResult(),
        "GEMM result must be tied to its explicit destination, not an allocation");

  expectContextRejected(capture, bridge::BridgeContext{}, [](auto &) {},
                        "anonymous context must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) { value.numerical_profile = "other"; },
                        "unknown profile must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.accumulation_dtype.reset();
                        },
                        "missing accumulation dtype must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.reassociation =
                              bridge::ReassociationSemantics::Forbidden;
                        },
                        "contradictory reassociation must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.contraction =
                              bridge::ContractionSemantics::Forbidden;
                        },
                        "contradictory contraction must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.reduction_order =
                              bridge::ReductionOrderSemantics::IncreasingK;
                        },
                        "contradictory reduction order must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.nan = bridge::NaNSemantics::Strict;
                        },
                        "contradictory NaN policy must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.signed_zero =
                              bridge::SignedZeroSemantics::Preserve;
                        },
                        "contradictory signed-zero policy must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.rounding =
                              bridge::RoundingSemantics::Invalid;
                        },
                        "missing rounding contract must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.trapping_exceptions =
                              bridge::TrappingExceptionSemantics::Invalid;
                        },
                        "missing trapping contract must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.exception_status =
                              bridge::ExceptionStatusSemantics::Invalid;
                        },
                        "missing exception-status contract must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.subnormals =
                              bridge::SubnormalSemantics::Invalid;
                        },
                        "missing subnormal contract must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.approximate_math =
                              bridge::Permission::Allowed;
                        },
                        "approximate math permission must be rejected");
  expectContextRejected(capture, explicit_profile,
                        [](auto &value) {
                          value.numerical.inplace = bridge::Permission::Allowed;
                        },
                        "in-place permission must be rejected");

  v1::Module bf16 = capture;
  auto &bf16_operation = bf16.operations[0];
  bf16_operation.operands[0].type.element_dtype = v1::DType::BF16;
  bf16_operation.operands[0].type.required_alignment_bytes = 2;
  bf16_operation.operands[1].type.element_dtype = v1::DType::BF16;
  bf16_operation.operands[1].type.required_alignment_bytes = 2;
  check(v1::verify(bf16, verify_error),
        "BF16/F32 capture must be valid before profile rejection");
  mlir::MLIRContext bf16_context;
  auto bf16_result = build(bf16, bf16_context, explicit_profile);
  check(!bf16_result, "F32 profile must not authorize BF16 capture");
  checkContains(bf16_result.error, "non-F32",
                "dtype/profile rejection must be actionable");

  v1::Module too_wide = capture;
  const std::uint64_t beyond_signed =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1;
  too_wide.operations[0].output.type.shape[0] =
      v1::ScalarExpr::staticValue(beyond_signed);
  too_wide.operations[0].operands[0].type.shape[0] =
      v1::ScalarExpr::staticValue(beyond_signed);
  check(v1::verify(too_wide, verify_error),
        "v1 uint64 dimension remains valid before MLIR range rejection");
  mlir::MLIRContext wide_context;
  auto wide_result = build(too_wide, wide_context, explicit_profile);
  check(!wide_result, "unrepresentable MLIR dimension must fail closed");
  checkContains(wide_result.error, "signed range",
                "range rejection must explain conversion boundary");

  v1::Module static_capture = capture;
  makeStatic(static_capture.operations[0].output, 2, 4);
  makeStatic(static_capture.operations[0].operands[0], 2, 3);
  makeStatic(static_capture.operations[0].operands[1], 3, 4);
  check(v1::verify(static_capture, verify_error),
        "static capture must verify before bridge");
  mlir::MLIRContext static_context;
  auto static_result = build(static_capture, static_context, explicit_profile);
  check(static_result &&
            bridge::serializeDeterministicMlir(*static_result.module)
                    .find("tensor<2x3xf32>") != std::string::npos,
        "static dimensions must map into ranked tensor types");

  v1::Module two_sites = capture;
  v1::Operation second_operation = capture.operations[0];
  second_operation.site_id = "mc_11111111111111111111111111111111";
  const std::uint64_t shift = second_operation.call_range.end + 10 -
                              second_operation.call_range.begin;
  second_operation.source.offset += shift;
  second_operation.source.line += 1;
  second_operation.call_range.begin += shift;
  second_operation.call_range.end += shift;
  for (auto &range : second_operation.argument_ranges) {
    range.begin += shift;
    range.end += shift;
  }
  second_operation.output.source_expression = "Z";
  second_operation.operands[0].source_expression = "X";
  second_operation.operands[1].source_expression = "Y";
  two_sites.operations.push_back(second_operation);
  check(v1::verify(two_sites, verify_error),
        "two sites may reuse operation-scoped dynamic symbols");
  mlir::MLIRContext two_context;
  auto two_result = build(two_sites, two_context, explicit_profile);
  check(static_cast<bool>(two_result),
        "two independent sites must bridge independently");
  std::size_t function_count = 0;
  if (two_result) {
    for (auto function : (*two_result.module).getOps<mlir::func::FuncOp>()) {
      (void)function;
      ++function_count;
    }
  }
  check(function_count == 2,
        "bridge must emit one independent function per operation/site");

  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "origin",
            withField(builder, operation.getOrigin(), "kind",
                      builder.getStringAttr("recovered_cpp_loop")));
      },
      "v1 bridge verifier must reject an unauthenticated recovered origin");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "lhs_semantics",
            withField(builder, operation.getLhsSemantics(), "shape",
                      operation.getRhsSemantics().get("shape")));
      },
      "dialect verifier must reject tensor shape/type disagreement");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        auto strides = operation.getLhsSemantics().getAs<mlir::ArrayAttr>(
            "strides");
        operation->setAttr(
            "lhs_semantics",
            withField(builder, operation.getLhsSemantics(), "strides",
                      builder.getArrayAttr({strides[1], strides[0]})));
      },
      "dialect verifier must reject non-row-major strides");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "lhs_semantics",
            withField(builder, operation.getLhsSemantics(), "layout",
                      builder.getStringAttr("strided")));
      },
      "dialect verifier must reject unsupported layouts");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "lhs_semantics",
            withField(builder, operation.getLhsSemantics(), "alignment_bytes",
                      builder.getI64IntegerAttr(3)));
      },
      "dialect verifier must reject invalid alignment preconditions");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "output_semantics",
            withField(builder, operation.getOutputSemantics(), "mutability",
                      builder.getStringAttr("read")));
      },
      "dialect verifier must reject incorrect destination mutability");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        auto relations = operation.getAliasing();
        auto relation = mlir::cast<mlir::DictionaryAttr>(relations[0]);
        llvm::SmallVector<mlir::Attribute> replaced(relations.begin(),
                                                    relations.end());
        replaced[0] = withField(builder, relation, "contract",
                                builder.getStringAttr("proven_fact"));
        operation->setAttr("aliasing", builder.getArrayAttr(replaced));
      },
      "dialect verifier must reject alias contracts presented as proven facts");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "effects",
            withField(builder, operation.getEffects(), "writes",
                      builder.getArrayAttr({})));
      },
      "dialect verifier must reject missing observable destination write");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr("synchronization",
                           builder.getStringAttr("asynchronous"));
      },
      "dialect verifier must reject unsupported synchronization");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "policy", withField(builder, operation.getPolicy(), "target",
                                builder.getStringAttr("cuda")));
      },
      "dialect verifier must reject unsupported target policy");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "provenance",
            withField(builder, operation.getProvenance(), "file",
                      builder.getStringAttr("different.mdsl")));
      },
      "dialect verifier must reject provenance/location disagreement");
  expectOperationRejected(
      capture, explicit_profile,
      [](dialect::GemmOp operation, mlir::Builder &builder) {
        operation->setAttr(
            "numerical",
            withoutField(builder, operation.getNumerical(), "rounding"));
      },
      "dialect verifier must reject missing numerical fields");

  mlir::MLIRContext return_context;
  auto bad_return = build(capture, return_context, explicit_profile);
  check(static_cast<bool>(bad_return), "fresh module for return mutation must build");
  if (bad_return) {
    auto function = *(*bad_return.module).getOps<mlir::func::FuncOp>().begin();
    auto return_op =
        mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
    return_op.setOperand(0, function.getBody().front().getArgument(2));
    check(!bridge::verifyMatcoreSemanticModule(*bad_return.module, verify_error),
          "bridge verifier must reject a result not returned from GEMM");
  }

  std::cout << "Matcore MLIR core: " << checks << " checks, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
