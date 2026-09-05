#include "MatcoreContractionModel.h"
#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreStructuredGemmHandoff.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace bridge = matcore::mdslc::mlir_bridge;

using Operation = bridge::StandardLinearAlgebraOperationV1;
using Orientation = bridge::MatrixOrientationV1;
using TopologyClass = bridge::BilinearTopologyClassV1;

int checks = 0;
int failures = 0;

void check(bool condition, std::string_view message) {
  ++checks;
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

std::string mapText(mlir::AffineMap map) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  map.print(stream);
  stream.flush();
  return text;
}

mlir::DictionaryAttr withField(mlir::Builder &builder,
                               mlir::DictionaryAttr dictionary,
                               llvm::StringRef field,
                               mlir::Attribute replacement) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (mlir::NamedAttribute attribute : dictionary) {
    attributes.push_back(attribute.getName().strref() == field
                             ? builder.getNamedAttr(field, replacement)
                             : attribute);
  }
  return builder.getDictionaryAttr(attributes);
}

void checkTopology(
    mlir::MLIRContext &context, Operation operation,
    Orientation lhs_orientation, Orientation rhs_orientation,
    TopologyClass expected_class, llvm::ArrayRef<std::string> expected_loops,
    llvm::ArrayRef<mlir::utils::IteratorType> expected_iterators,
    llvm::ArrayRef<unsigned> expected_ranks,
    llvm::ArrayRef<std::string> expected_maps,
    llvm::ArrayRef<std::string> expected_reductions,
    std::string_view description) {
  auto result = bridge::buildCanonicalContractionTopologyV1(
      context, operation, lhs_orientation, rhs_orientation);
  check(static_cast<bool>(result),
        std::string(description) + " must build a canonical topology");
  if (!result)
    return;
  std::string error;
  check(bridge::verifyCanonicalContractionTopologyV1(result.topology, error),
        std::string(description) + " canonical topology must verify");
  check(result.topology.topology_class == expected_class,
        std::string(description) + " must retain its topology class");
  check(result.topology.loop_dimensions == expected_loops,
        std::string(description) + " must retain exact logical loop names");
  check(result.topology.iterator_types == expected_iterators,
        std::string(description) + " must retain parallel/reduction roles");
  check(result.topology.operand_ranks == expected_ranks,
        std::string(description) + " must retain lhs/rhs/output ranks");
  check(result.topology.reduction_dimensions == expected_reductions,
        std::string(description) + " must retain exact reduction dimensions");
  llvm::SmallVector<std::string, 3> actual_maps;
  for (mlir::AffineMap map : result.topology.indexing_maps)
    actual_maps.push_back(mapText(map));
  check(actual_maps == expected_maps,
        std::string(description) + " must retain exact affine indexing maps");

  mlir::Builder builder(&context);
  mlir::DictionaryAttr encoded =
      bridge::encodeContractionTopologyV1(builder, result.topology, error);
  check(static_cast<bool>(encoded),
        std::string(description) + " canonical topology must encode");
  if (!encoded)
    return;
  auto decoded = bridge::decodeContractionTopologyV1(encoded, context);
  check(static_cast<bool>(decoded),
        std::string(description) + " exact encoding must round-trip");
  if (decoded) {
    check(bridge::encodeContractionTopologyV1(builder, decoded.topology,
                                              error) == encoded,
          std::string(description) + " encoding must be deterministic");
    check(bridge::verifyStructuredIndexingAgainstContractionTopologyV1(
              decoded.topology, result.topology.indexing_maps,
              result.topology.iterator_types, result.topology.operand_ranks,
              error),
          std::string(description) +
              " upstream indexing carrier must match mechanically");
  }
}

void testStandardFamily() {
  mlir::MLIRContext context;
  const auto p = mlir::utils::IteratorType::parallel;
  const auto r = mlir::utils::IteratorType::reduction;
  const std::string gemm_nn_maps[] = {
      "(d0, d1, d2) -> (d0, d2)", "(d0, d1, d2) -> (d2, d1)",
      "(d0, d1, d2) -> (d0, d1)"};
  const std::string gemm_tn_maps[] = {
      "(d0, d1, d2) -> (d2, d0)", "(d0, d1, d2) -> (d2, d1)",
      "(d0, d1, d2) -> (d0, d1)"};
  const std::string gemm_nt_maps[] = {
      "(d0, d1, d2) -> (d0, d2)", "(d0, d1, d2) -> (d1, d2)",
      "(d0, d1, d2) -> (d0, d1)"};
  const std::string gemm_tt_maps[] = {
      "(d0, d1, d2) -> (d2, d0)", "(d0, d1, d2) -> (d1, d2)",
      "(d0, d1, d2) -> (d0, d1)"};
  const std::string gemm_loops[] = {"m", "n", "k"};
  const mlir::utils::IteratorType gemm_iterators[] = {p, p, r};
  const unsigned gemm_ranks[] = {2, 2, 2};
  const std::string reduction_k[] = {"k"};
  checkTopology(context, Operation::Gemm, Orientation::Normal,
                Orientation::Normal, TopologyClass::ReductionContraction,
                gemm_loops, gemm_iterators, gemm_ranks, gemm_nn_maps,
                reduction_k, "GEMM NN");
  checkTopology(context, Operation::Gemm, Orientation::Transpose,
                Orientation::Normal, TopologyClass::ReductionContraction,
                gemm_loops, gemm_iterators, gemm_ranks, gemm_tn_maps,
                reduction_k, "GEMM TN");
  checkTopology(context, Operation::Gemm, Orientation::Normal,
                Orientation::Transpose, TopologyClass::ReductionContraction,
                gemm_loops, gemm_iterators, gemm_ranks, gemm_nt_maps,
                reduction_k, "GEMM NT");
  checkTopology(context, Operation::Gemm, Orientation::Transpose,
                Orientation::Transpose, TopologyClass::ReductionContraction,
                gemm_loops, gemm_iterators, gemm_ranks, gemm_tt_maps,
                reduction_k, "GEMM TT");

  const std::string gemv_loops[] = {"m", "k"};
  const mlir::utils::IteratorType gemv_iterators[] = {p, r};
  const unsigned gemv_ranks[] = {2, 1, 1};
  const std::string gemv_n_maps[] = {"(d0, d1) -> (d0, d1)",
                                     "(d0, d1) -> (d1)",
                                     "(d0, d1) -> (d0)"};
  const std::string gemv_t_maps[] = {"(d0, d1) -> (d1, d0)",
                                     "(d0, d1) -> (d1)",
                                     "(d0, d1) -> (d0)"};
  checkTopology(context, Operation::Gemv, Orientation::Normal,
                Orientation::Normal, TopologyClass::ReductionContraction,
                gemv_loops, gemv_iterators, gemv_ranks, gemv_n_maps,
                reduction_k, "GEMV N");
  checkTopology(context, Operation::Gemv, Orientation::Transpose,
                Orientation::Normal, TopologyClass::ReductionContraction,
                gemv_loops, gemv_iterators, gemv_ranks, gemv_t_maps,
                reduction_k, "GEMV T");

  const std::string dot_loops[] = {"k"};
  const mlir::utils::IteratorType dot_iterators[] = {r};
  const unsigned dot_ranks[] = {1, 1, 0};
  const std::string dot_maps[] = {"(d0) -> (d0)", "(d0) -> (d0)",
                                  "(d0) -> ()"};
  checkTopology(context, Operation::Dot, Orientation::Normal,
                Orientation::Normal, TopologyClass::ReductionContraction,
                dot_loops, dot_iterators, dot_ranks, dot_maps, reduction_k,
                "DOT");

  const std::string ger_loops[] = {"m", "n"};
  const mlir::utils::IteratorType ger_iterators[] = {p, p};
  const unsigned ger_ranks[] = {1, 1, 2};
  const std::string ger_maps[] = {"(d0, d1) -> (d0)",
                                  "(d0, d1) -> (d1)",
                                  "(d0, d1) -> (d0, d1)"};
  checkTopology(context, Operation::Ger, Orientation::Normal,
                Orientation::Normal, TopologyClass::OuterProductUpdate,
                ger_loops, ger_iterators, ger_ranks, ger_maps,
                llvm::ArrayRef<std::string>{}, "GER");

  const std::string batch_loops[] = {"b", "m", "n", "k"};
  const mlir::utils::IteratorType batch_iterators[] = {p, p, p, r};
  const unsigned batch_ranks[] = {3, 3, 3};
  const std::string batch_nn_maps[] = {
      "(d0, d1, d2, d3) -> (d0, d1, d3)",
      "(d0, d1, d2, d3) -> (d0, d3, d2)",
      "(d0, d1, d2, d3) -> (d0, d1, d2)"};
  checkTopology(context, Operation::BatchedGemm, Orientation::Normal,
                Orientation::Normal, TopologyClass::ReductionContraction,
                batch_loops, batch_iterators, batch_ranks, batch_nn_maps,
                reduction_k, "batched GEMM NN");
  const std::string batch_tn_maps[] = {
      "(d0, d1, d2, d3) -> (d0, d3, d1)",
      "(d0, d1, d2, d3) -> (d0, d3, d2)",
      "(d0, d1, d2, d3) -> (d0, d1, d2)"};
  checkTopology(context, Operation::BatchedGemm, Orientation::Transpose,
                Orientation::Normal, TopologyClass::ReductionContraction,
                batch_loops, batch_iterators, batch_ranks, batch_tn_maps,
                reduction_k, "batched GEMM TN");
  const std::string batch_nt_maps[] = {
      "(d0, d1, d2, d3) -> (d0, d1, d3)",
      "(d0, d1, d2, d3) -> (d0, d2, d3)",
      "(d0, d1, d2, d3) -> (d0, d1, d2)"};
  checkTopology(context, Operation::BatchedGemm, Orientation::Normal,
                Orientation::Transpose, TopologyClass::ReductionContraction,
                batch_loops, batch_iterators, batch_ranks, batch_nt_maps,
                reduction_k, "batched GEMM NT");
  const std::string batch_tt_maps[] = {
      "(d0, d1, d2, d3) -> (d0, d3, d1)",
      "(d0, d1, d2, d3) -> (d0, d2, d3)",
      "(d0, d1, d2, d3) -> (d0, d1, d2)"};
  checkTopology(context, Operation::BatchedGemm, Orientation::Transpose,
                Orientation::Transpose, TopologyClass::ReductionContraction,
                batch_loops, batch_iterators, batch_ranks, batch_tt_maps,
                reduction_k, "batched GEMM TT");
}

void testIdentityAndUnsupportedCollapses() {
  mlir::MLIRContext context;
  auto gemm = bridge::buildCanonicalContractionTopologyV1(
      context, Operation::Gemm);
  auto gemv = bridge::buildCanonicalContractionTopologyV1(
      context, Operation::Gemv);
  auto dot = bridge::buildCanonicalContractionTopologyV1(context,
                                                         Operation::Dot);
  auto ger = bridge::buildCanonicalContractionTopologyV1(context,
                                                         Operation::Ger);
  check(gemm && gemv && dot && ger,
        "identity-boundary fixtures must all construct");
  if (!gemm || !gemv || !dot || !ger)
    return;
  check(gemm.topology.operand_ranks != gemv.topology.operand_ranks &&
            gemm.topology.operand_ranks != dot.topology.operand_ranks,
        "rank-two unit-extent GEMM must not acquire GEMV or DOT identity");
  check(gemm.topology.operation == Operation::Gemm,
        "operation identity must be explicit rather than inferred from extents");
  check(ger.topology.topology_class == TopologyClass::OuterProductUpdate &&
            ger.topology.reduction_dimensions.empty() &&
            llvm::none_of(ger.topology.iterator_types,
                          [](mlir::utils::IteratorType iterator) {
                            return iterator ==
                                   mlir::utils::IteratorType::reduction;
                          }),
        "GER must remain a no-reduction outer-product update");
  check(ger.topology.indexing_maps != gemm.topology.indexing_maps,
        "GER must not be encoded as GEMM with an invented singleton K");

  check(!bridge::buildCanonicalContractionTopologyV1(
             context, Operation::Dot, Orientation::Transpose),
        "DOT must reject invented transpose axes");
  check(!bridge::buildCanonicalContractionTopologyV1(
             context, Operation::Ger, Orientation::Normal,
             Orientation::Transpose),
        "GER must reject invented vector transpose axes");
  check(!bridge::buildCanonicalContractionTopologyV1(
             context, Operation::Gemv, Orientation::Normal,
             Orientation::Transpose),
        "GEMV must reject a matrix orientation on its vector operand");
}

void testAdversarialTopologyMutations() {
  mlir::MLIRContext context;
  auto canonical = bridge::buildCanonicalContractionTopologyV1(
      context, Operation::Gemm);
  check(static_cast<bool>(canonical),
        "adversarial GEMM topology fixture must construct");
  if (!canonical)
    return;
  std::string error;

  auto changed_map = canonical.topology;
  std::swap(changed_map.indexing_maps[0], changed_map.indexing_maps[1]);
  check(!bridge::verifyCanonicalContractionTopologyV1(changed_map, error),
        "topology verifier must reject swapped GEMM operands/maps");

  auto changed_iterator = canonical.topology;
  changed_iterator.iterator_types[2] = mlir::utils::IteratorType::parallel;
  check(!bridge::verifyCanonicalContractionTopologyV1(changed_iterator, error),
        "topology verifier must reject loss of GEMM reduction K");

  auto changed_rank = canonical.topology;
  changed_rank.operand_ranks[2] = 1;
  check(!bridge::verifyCanonicalContractionTopologyV1(changed_rank, error),
        "topology verifier must reject rank-based GEMM-to-GEMV collapse");

  auto changed_class = canonical.topology;
  changed_class.topology_class = TopologyClass::OuterProductUpdate;
  check(!bridge::verifyCanonicalContractionTopologyV1(changed_class, error),
        "topology verifier must reject a forged operation class");

  mlir::Builder builder(&context);
  mlir::DictionaryAttr encoded =
      bridge::encodeContractionTopologyV1(builder, canonical.topology, error);
  check(static_cast<bool>(encoded),
        "canonical adversarial fixture must encode before mutation");
  if (!encoded)
    return;

  check(!bridge::encodeContractionTopologyV1(builder, changed_map, error),
        "encoder must fail closed on a noncanonical topology");
  mlir::MLIRContext other_context;
  mlir::Builder other_builder(&other_context);
  check(!bridge::encodeContractionTopologyV1(other_builder,
                                             canonical.topology, error),
        "encoder must reject topology maps from another MLIR context");
  auto forged_class = withField(
      builder, encoded, "classification",
      builder.getStringAttr("outer_product_update"));
  check(!bridge::decodeContractionTopologyV1(forged_class, context),
        "decoder must reject a forged GEMM classification");

  auto forged_version = withField(builder, encoded, "version",
                                  builder.getI32IntegerAttr(2));
  check(!bridge::decodeContractionTopologyV1(forged_version, context),
        "decoder must reject an unrecognized topology version");

  auto wrong_maps = canonical.topology.indexing_maps;
  std::swap(wrong_maps[0], wrong_maps[1]);
  check(!bridge::verifyStructuredIndexingAgainstContractionTopologyV1(
             canonical.topology, wrong_maps, canonical.topology.iterator_types,
             canonical.topology.operand_ranks, error),
        "structured carrier check must reject swapped affine maps");
}

void verifyGenericCarrier(llvm::StringRef text, Operation operation,
                          Orientation lhs_orientation,
                          Orientation rhs_orientation,
                          std::string_view description) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  mlir::ParserConfig parser_config(&context, /*verifyAfterParse=*/true);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, parser_config);
  check(static_cast<bool>(module),
        std::string(description) + " upstream linalg.generic must verify");
  if (!module)
    return;
  mlir::linalg::GenericOp generic;
  module->walk([&](mlir::linalg::GenericOp operation) { generic = operation; });
  check(static_cast<bool>(generic),
        std::string(description) + " must contain linalg.generic");
  if (!generic)
    return;
  auto topology = bridge::buildCanonicalContractionTopologyV1(
      context, operation, lhs_orientation, rhs_orientation);
  check(static_cast<bool>(topology),
        std::string(description) + " model must construct");
  if (!topology)
    return;
  llvm::SmallVector<unsigned, 3> ranks;
  for (mlir::Value value :
       llvm::concat<mlir::Value>(generic.getInputs(), generic.getOutputs())) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    check(static_cast<bool>(type),
          std::string(description) + " operands must be ranked tensors");
    if (!type)
      return;
    ranks.push_back(type.getRank());
  }
  std::string error;
  check(bridge::verifyStructuredIndexingAgainstContractionTopologyV1(
            topology.topology, generic.getIndexingMapsArray(),
            generic.getIteratorTypesArray(), ranks, error),
        std::string(description) +
            " must map exactly onto upstream Linalg topology");
  if (operation == Operation::Ger) {
    bool has_fill = false;
    module->walk([&](mlir::linalg::FillOp) { has_fill = true; });
    mlir::Block &block = generic.getRegion().front();
    check(std::distance(block.begin(), block.end()) == 3,
          "GER model fixture must contain exactly multiply, add, and yield");
    auto iterator = block.begin();
    auto multiply = iterator == block.end()
                        ? mlir::arith::MulFOp{}
                        : mlir::dyn_cast<mlir::arith::MulFOp>(*iterator++);
    auto add = iterator == block.end()
                   ? mlir::arith::AddFOp{}
                   : mlir::dyn_cast<mlir::arith::AddFOp>(*iterator++);
    auto yield = iterator == block.end()
                     ? mlir::linalg::YieldOp{}
                     : mlir::dyn_cast<mlir::linalg::YieldOp>(*iterator++);
    check(multiply && multiply.getLhs() == block.getArgument(0) &&
              multiply.getRhs() == block.getArgument(1) &&
              multiply.getFastmath() == mlir::arith::FastMathFlags::none,
          "GER model fixture must compute exact no-fast-math mul(x, y)");
    check(add && multiply && add.getLhs() == block.getArgument(2) &&
              add.getRhs() == multiply.getResult() &&
              add.getFastmath() == mlir::arith::FastMathFlags::none,
          "GER model fixture must compute exact no-fast-math add(old C, mul)");
    check(yield && add && yield->getNumOperands() == 1 &&
              yield->getOperand(0) == add.getResult(),
          "GER model fixture must yield exactly the accumulating add result");
    check(generic->getParentOfType<mlir::func::FuncOp>() && !has_fill,
          "GER model fixture must not inherit GEMM overwrite initialization");
  }

  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  auto rejected = bridge::deriveStructuredGemmHandoffV1(*module);
  check(!rejected,
        std::string(description) +
            " model-only Linalg must not enter authenticated source handoff");
  std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1>
      records(1);
  std::string lowering_error;
  check(!matcore::mdslc::mlir_lowering::
            lowerExplicitGemmToCpuRuntimeDispatchV1(*module, records,
                                                    lowering_error) &&
            records.empty(),
        std::string(description) +
            " model-only Linalg must not gain CPU execution authority");
}

void verifyNamedCarrier(llvm::StringRef text, llvm::StringRef expected_name,
                        Operation operation, Orientation lhs_orientation,
                        Orientation rhs_orientation,
                        bool reorder_vecmat_operands,
                        std::string_view description) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  mlir::ParserConfig parser_config(&context, /*verifyAfterParse=*/true);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, parser_config);
  check(static_cast<bool>(module),
        std::string(description) + " upstream named Linalg op must verify");
  if (!module)
    return;

  mlir::linalg::LinalgOp carrier;
  module->walk([&](mlir::linalg::LinalgOp candidate) {
    if (candidate->getName().getStringRef() == expected_name)
      carrier = candidate;
  });
  check(static_cast<bool>(carrier),
        std::string(description) + " must contain " + expected_name.str());
  if (!carrier)
    return;

  llvm::SmallVector<mlir::AffineMap, 3> maps =
      carrier.getIndexingMapsArray();
  llvm::SmallVector<unsigned, 3> ranks;
  auto append_rank = [&](mlir::Value value) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    check(static_cast<bool>(type),
          std::string(description) + " operands must be ranked tensors");
    if (type)
      ranks.push_back(type.getRank());
  };
  for (mlir::Value input : carrier.getDpsInputs())
    append_rank(input);
  for (mlir::Value output : carrier.getDpsInits())
    append_rank(output);

  if (reorder_vecmat_operands) {
    check(expected_name == "linalg.vecmat" && maps.size() == 3 &&
              ranks == llvm::ArrayRef<unsigned>({1, 2, 1}),
          "GEMV-T adapter must recognize upstream vecmat's explicit "
          "vector,matrix,output operand order");
    if (maps.size() != 3 || ranks.size() != 3)
      return;
    maps = {maps[1], maps[0], maps[2]};
    ranks = {ranks[1], ranks[0], ranks[2]};
  }

  auto topology = bridge::buildCanonicalContractionTopologyV1(
      context, operation, lhs_orientation, rhs_orientation);
  check(static_cast<bool>(topology),
        std::string(description) + " model must construct");
  if (!topology)
    return;
  std::string error;
  check(bridge::verifyStructuredIndexingAgainstContractionTopologyV1(
            topology.topology, maps, carrier.getIteratorTypesArray(), ranks,
            error),
        std::string(description) +
            " named op must carry the exact canonical topology");

  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  auto rejected = bridge::deriveStructuredGemmHandoffV1(*module);
  check(!rejected,
        std::string(description) +
            " named model carrier must not enter authenticated source "
            "handoff");
  std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1>
      records(1);
  std::string lowering_error;
  check(!matcore::mdslc::mlir_lowering::
            lowerExplicitGemmToCpuRuntimeDispatchV1(*module, records,
                                                    lowering_error) &&
            records.empty(),
        std::string(description) +
            " named model carrier must not gain CPU execution authority");
}

void testUpstreamGenericCarriers() {
  verifyGenericCarrier(
      R"mlir(module {
  func.func @gemv(%a: tensor<2x3xf32>, %x: tensor<3xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = linalg.generic {indexing_maps = [affine_map<(m, k) -> (m, k)>, affine_map<(m, k) -> (k)>, affine_map<(m, k) -> (m)>], iterator_types = ["parallel", "reduction"]} ins(%a, %x : tensor<2x3xf32>, tensor<3xf32>) outs(%y : tensor<2xf32>) {
    ^bb0(%av: f32, %xv: f32, %yv: f32):
      %p = arith.mulf %av, %xv : f32
      %s = arith.addf %yv, %p : f32
      linalg.yield %s : f32
    } -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir",
      Operation::Gemv, Orientation::Normal, Orientation::Normal, "GEMV");

  verifyGenericCarrier(
      R"mlir(module {
  func.func @dot(%x: tensor<3xf32>, %y: tensor<3xf32>, %z: tensor<f32>) -> tensor<f32> {
    %0 = linalg.generic {indexing_maps = [affine_map<(k) -> (k)>, affine_map<(k) -> (k)>, affine_map<(k) -> ()>], iterator_types = ["reduction"]} ins(%x, %y : tensor<3xf32>, tensor<3xf32>) outs(%z : tensor<f32>) {
    ^bb0(%xv: f32, %yv: f32, %zv: f32):
      %p = arith.mulf %xv, %yv : f32
      %s = arith.addf %zv, %p : f32
      linalg.yield %s : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
  }
})mlir",
      Operation::Dot, Orientation::Normal, Orientation::Normal, "DOT");

  verifyGenericCarrier(
      R"mlir(module {
  func.func @ger(%x: tensor<2xf32>, %y: tensor<4xf32>, %a: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = linalg.generic {indexing_maps = [affine_map<(m, n) -> (m)>, affine_map<(m, n) -> (n)>, affine_map<(m, n) -> (m, n)>], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<2xf32>, tensor<4xf32>) outs(%a : tensor<2x4xf32>) {
    ^bb0(%xv: f32, %yv: f32, %av: f32):
      %p = arith.mulf %xv, %yv : f32
      %s = arith.addf %av, %p : f32
      linalg.yield %s : f32
    } -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
})mlir",
      Operation::Ger, Orientation::Normal, Orientation::Normal, "GER");

  verifyGenericCarrier(
      R"mlir(module {
  func.func @batch_gemm(%a: tensor<5x2x3xf32>, %b: tensor<5x3x4xf32>, %c: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
    %0 = linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<5x2x3xf32>, tensor<5x3x4xf32>) outs(%c : tensor<5x2x4xf32>) {
    ^bb0(%av: f32, %bv: f32, %cv: f32):
      %p = arith.mulf %av, %bv : f32
      %s = arith.addf %cv, %p : f32
      linalg.yield %s : f32
    } -> tensor<5x2x4xf32>
    return %0 : tensor<5x2x4xf32>
  }
})mlir",
      Operation::BatchedGemm, Orientation::Normal, Orientation::Normal,
      "batched GEMM");
}

void testUpstreamNamedCarriers() {
  verifyNamedCarrier(
      R"mlir(module {
  func.func @gemv_n(%a: tensor<2x3xf32>, %x: tensor<3xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = linalg.matvec ins(%a, %x : tensor<2x3xf32>, tensor<3xf32>) outs(%y : tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir",
      "linalg.matvec", Operation::Gemv, Orientation::Normal,
      Orientation::Normal, /*reorder_vecmat_operands=*/false,
      "GEMV-N named carrier");

  verifyNamedCarrier(
      R"mlir(module {
  func.func @gemv_t(%x: tensor<3xf32>, %a: tensor<3x2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %0 = linalg.vecmat ins(%x, %a : tensor<3xf32>, tensor<3x2xf32>) outs(%y : tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir",
      "linalg.vecmat", Operation::Gemv, Orientation::Transpose,
      Orientation::Normal, /*reorder_vecmat_operands=*/true,
      "GEMV-T named carrier");

  verifyNamedCarrier(
      R"mlir(module {
  func.func @dot(%x: tensor<3xf32>, %y: tensor<3xf32>, %z: tensor<f32>) -> tensor<f32> {
    %0 = linalg.dot ins(%x, %y : tensor<3xf32>, tensor<3xf32>) outs(%z : tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
})mlir",
      "linalg.dot", Operation::Dot, Orientation::Normal, Orientation::Normal,
      /*reorder_vecmat_operands=*/false, "DOT named carrier");

#if LLVM_VERSION_MAJOR >= 22
  verifyNamedCarrier(
      R"mlir(module {
  func.func @gemm_tn(%a: tensor<3x2xf32>, %b: tensor<3x4xf32>, %c: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (k, m)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<3x2xf32>, tensor<3x4xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
})mlir",
      "linalg.matmul", Operation::Gemm, Orientation::Transpose,
      Orientation::Normal, /*reorder_vecmat_operands=*/false,
      "GEMM-TN indexed matmul carrier");

  verifyNamedCarrier(
      R"mlir(module {
  func.func @gemm_nt(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>, %c: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<2x3xf32>, tensor<4x3xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
})mlir",
      "linalg.matmul", Operation::Gemm, Orientation::Normal,
      Orientation::Transpose, /*reorder_vecmat_operands=*/false,
      "GEMM-NT indexed matmul carrier");
#else
  verifyNamedCarrier(
      R"mlir(module {
  func.func @gemm_tn(%a: tensor<3x2xf32>, %b: tensor<3x4xf32>, %c: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = linalg.matmul_transpose_a ins(%a, %b : tensor<3x2xf32>, tensor<3x4xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
})mlir",
      "linalg.matmul_transpose_a", Operation::Gemm, Orientation::Transpose,
      Orientation::Normal, /*reorder_vecmat_operands=*/false,
      "GEMM-TN named carrier");

  verifyNamedCarrier(
      R"mlir(module {
  func.func @gemm_nt(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>, %c: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = linalg.matmul_transpose_b ins(%a, %b : tensor<2x3xf32>, tensor<4x3xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
})mlir",
      "linalg.matmul_transpose_b", Operation::Gemm, Orientation::Normal,
      Orientation::Transpose, /*reorder_vecmat_operands=*/false,
      "GEMM-NT named carrier");
#endif

  verifyNamedCarrier(
      R"mlir(module {
  func.func @batch_nn(%a: tensor<5x2x3xf32>, %b: tensor<5x3x4xf32>, %c: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
    %0 = linalg.batch_matmul ins(%a, %b : tensor<5x2x3xf32>, tensor<5x3x4xf32>) outs(%c : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
    return %0 : tensor<5x2x4xf32>
  }
})mlir",
      "linalg.batch_matmul", Operation::BatchedGemm, Orientation::Normal,
      Orientation::Normal, /*reorder_vecmat_operands=*/false,
      "batched GEMM-NN named carrier");

#if LLVM_VERSION_MAJOR >= 22
  verifyNamedCarrier(
      R"mlir(module {
  func.func @batch_tn(%a: tensor<5x3x2xf32>, %b: tensor<5x3x4xf32>, %c: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
    %0 = linalg.batch_matmul indexing_maps = [affine_map<(b, m, n, k) -> (b, k, m)>, affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>] ins(%a, %b : tensor<5x3x2xf32>, tensor<5x3x4xf32>) outs(%c : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
    return %0 : tensor<5x2x4xf32>
  }
})mlir",
      "linalg.batch_matmul", Operation::BatchedGemm, Orientation::Transpose,
      Orientation::Normal,
      /*reorder_vecmat_operands=*/false,
      "batched GEMM-TN indexed matmul carrier");

  verifyNamedCarrier(
      R"mlir(module {
  func.func @batch_nt(%a: tensor<5x2x3xf32>, %b: tensor<5x4x3xf32>, %c: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
    %0 = linalg.batch_matmul indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (b, n, k)>, affine_map<(b, m, n, k) -> (b, m, n)>] ins(%a, %b : tensor<5x2x3xf32>, tensor<5x4x3xf32>) outs(%c : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
    return %0 : tensor<5x2x4xf32>
  }
})mlir",
      "linalg.batch_matmul", Operation::BatchedGemm, Orientation::Normal,
      Orientation::Transpose,
      /*reorder_vecmat_operands=*/false,
      "batched GEMM-NT indexed matmul carrier");
#else
  verifyNamedCarrier(
      R"mlir(module {
  func.func @batch_tn(%a: tensor<5x3x2xf32>, %b: tensor<5x3x4xf32>, %c: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
    %0 = linalg.batch_matmul_transpose_a ins(%a, %b : tensor<5x3x2xf32>, tensor<5x3x4xf32>) outs(%c : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
    return %0 : tensor<5x2x4xf32>
  }
})mlir",
      "linalg.batch_matmul_transpose_a", Operation::BatchedGemm,
      Orientation::Transpose, Orientation::Normal,
      /*reorder_vecmat_operands=*/false, "batched GEMM-TN named carrier");

  verifyNamedCarrier(
      R"mlir(module {
  func.func @batch_nt(%a: tensor<5x2x3xf32>, %b: tensor<5x4x3xf32>, %c: tensor<5x2x4xf32>) -> tensor<5x2x4xf32> {
    %0 = linalg.batch_matmul_transpose_b ins(%a, %b : tensor<5x2x3xf32>, tensor<5x4x3xf32>) outs(%c : tensor<5x2x4xf32>) -> tensor<5x2x4xf32>
    return %0 : tensor<5x2x4xf32>
  }
})mlir",
      "linalg.batch_matmul_transpose_b", Operation::BatchedGemm,
      Orientation::Normal, Orientation::Transpose,
      /*reorder_vecmat_operands=*/false, "batched GEMM-NT named carrier");
#endif
}

} // namespace

int main() {
  testStandardFamily();
  testIdentityAndUnsupportedCollapses();
  testAdversarialTopologyMutations();
  testUpstreamGenericCarriers();
  testUpstreamNamedCarriers();
  if (failures != 0) {
    std::cerr << "Contraction topology adversarial tests: " << failures
              << " of " << checks << " checks failed\n";
    return 1;
  }
  std::cout << "Contraction topology adversarial tests: " << checks
            << " checks, 0 failures\n";
  return 0;
}
