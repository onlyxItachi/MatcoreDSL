#include "MatcoreContractionModel.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cstddef>
#include <initializer_list>
#include <string>

namespace matcore::mdslc::mlir_bridge {
namespace {

using Operation = StandardLinearAlgebraOperationV1;
using Orientation = MatrixOrientationV1;
using TopologyClass = BilinearTopologyClassV1;

constexpr llvm::StringLiteral kTopologyFields[] = {
    "classification",       "indexing_maps",   "iterator_types",
    "lhs_orientation",      "loop_dimensions", "operand_ranks",
    "operation",            "reduction_dimensions",
    "rhs_orientation",      "schema",          "version",
};

bool requireExactNames(mlir::DictionaryAttr dictionary,
                       llvm::ArrayRef<llvm::StringLiteral> expected,
                       std::string &error) {
  if (!dictionary || dictionary.size() != expected.size()) {
    error = "contraction topology must contain exactly " +
            std::to_string(expected.size()) + " fields";
    return false;
  }
  llvm::StringSet<> names;
  for (llvm::StringRef name : expected)
    names.insert(name);
  for (mlir::NamedAttribute attribute : dictionary) {
    if (!names.contains(attribute.getName().strref())) {
      error = "contraction topology contains unexpected field '" +
              attribute.getName().strref().str() + "'";
      return false;
    }
  }
  return true;
}

bool parseOperation(llvm::StringRef value, Operation &operation) {
  if (value == "gemm")
    operation = Operation::Gemm;
  else if (value == "gemv")
    operation = Operation::Gemv;
  else if (value == "dot")
    operation = Operation::Dot;
  else if (value == "ger")
    operation = Operation::Ger;
  else if (value == "batched_gemm")
    operation = Operation::BatchedGemm;
  else
    return false;
  return true;
}

bool parseOrientation(llvm::StringRef value, Orientation &orientation) {
  if (value == "normal")
    orientation = Orientation::Normal;
  else if (value == "transpose")
    orientation = Orientation::Transpose;
  else
    return false;
  return true;
}

bool parseTopologyClass(llvm::StringRef value, TopologyClass &topology_class) {
  if (value == "reduction_contraction")
    topology_class = TopologyClass::ReductionContraction;
  else if (value == "outer_product_update")
    topology_class = TopologyClass::OuterProductUpdate;
  else
    return false;
  return true;
}

bool sameTopology(const ContractionTopologyV1 &left,
                  const ContractionTopologyV1 &right, std::string &error) {
  if (left.operation != right.operation ||
      left.lhs_orientation != right.lhs_orientation ||
      left.rhs_orientation != right.rhs_orientation ||
      left.topology_class != right.topology_class ||
      left.loop_dimensions != right.loop_dimensions ||
      left.iterator_types != right.iterator_types ||
      left.indexing_maps != right.indexing_maps ||
      left.operand_ranks != right.operand_ranks ||
      left.reduction_dimensions != right.reduction_dimensions) {
    error = "contraction topology differs from the canonical model for its "
            "declared standard operation";
    return false;
  }
  return true;
}

mlir::AffineMap map(mlir::MLIRContext &context, unsigned dimensions,
                    std::initializer_list<unsigned> results) {
  llvm::SmallVector<mlir::AffineExpr, 3> expressions;
  expressions.reserve(results.size());
  for (unsigned result : results)
    expressions.push_back(mlir::getAffineDimExpr(result, &context));
  return mlir::AffineMap::get(dimensions, 0, expressions, &context);
}

} // namespace

llvm::StringRef
standardLinearAlgebraOperationNameV1(StandardLinearAlgebraOperationV1 value) {
  switch (value) {
  case Operation::Gemm:
    return "gemm";
  case Operation::Gemv:
    return "gemv";
  case Operation::Dot:
    return "dot";
  case Operation::Ger:
    return "ger";
  case Operation::BatchedGemm:
    return "batched_gemm";
  }
  return "invalid";
}

llvm::StringRef matrixOrientationNameV1(MatrixOrientationV1 value) {
  switch (value) {
  case Orientation::Normal:
    return "normal";
  case Orientation::Transpose:
    return "transpose";
  }
  return "invalid";
}

llvm::StringRef bilinearTopologyClassNameV1(BilinearTopologyClassV1 value) {
  switch (value) {
  case TopologyClass::ReductionContraction:
    return "reduction_contraction";
  case TopologyClass::OuterProductUpdate:
    return "outer_product_update";
  }
  return "invalid";
}

ContractionTopologyResultV1 buildCanonicalContractionTopologyV1(
    mlir::MLIRContext &context, StandardLinearAlgebraOperationV1 operation,
    MatrixOrientationV1 lhs_orientation,
    MatrixOrientationV1 rhs_orientation) {
  ContractionTopologyResultV1 result;
  result.topology.operation = operation;
  result.topology.lhs_orientation = lhs_orientation;
  result.topology.rhs_orientation = rhs_orientation;
  if ((lhs_orientation != Orientation::Normal &&
       lhs_orientation != Orientation::Transpose) ||
      (rhs_orientation != Orientation::Normal &&
       rhs_orientation != Orientation::Transpose)) {
    result.error = "standard operation has an invalid matrix orientation";
    return result;
  }

  const auto parallel = mlir::utils::IteratorType::parallel;
  const auto reduction = mlir::utils::IteratorType::reduction;
  switch (operation) {
  case Operation::Gemm: {
    result.topology.topology_class = TopologyClass::ReductionContraction;
    result.topology.loop_dimensions = {"m", "n", "k"};
    result.topology.iterator_types = {parallel, parallel, reduction};
    result.topology.indexing_maps = {
        lhs_orientation == Orientation::Normal ? map(context, 3, {0, 2})
                                               : map(context, 3, {2, 0}),
        rhs_orientation == Orientation::Normal ? map(context, 3, {2, 1})
                                               : map(context, 3, {1, 2}),
        map(context, 3, {0, 1}),
    };
    result.topology.operand_ranks = {2, 2, 2};
    result.topology.reduction_dimensions = {"k"};
    break;
  }
  case Operation::Gemv: {
    if (rhs_orientation != Orientation::Normal) {
      result.error = "GEMV vector orientation is not a matrix transpose axis";
      return result;
    }
    result.topology.topology_class = TopologyClass::ReductionContraction;
    result.topology.loop_dimensions = {"m", "k"};
    result.topology.iterator_types = {parallel, reduction};
    result.topology.indexing_maps = {
        lhs_orientation == Orientation::Normal ? map(context, 2, {0, 1})
                                               : map(context, 2, {1, 0}),
        map(context, 2, {1}), map(context, 2, {0})};
    result.topology.operand_ranks = {2, 1, 1};
    result.topology.reduction_dimensions = {"k"};
    break;
  }
  case Operation::Dot: {
    if (lhs_orientation != Orientation::Normal ||
        rhs_orientation != Orientation::Normal) {
      result.error = "DOT has no matrix orientation axis";
      return result;
    }
    result.topology.topology_class = TopologyClass::ReductionContraction;
    result.topology.loop_dimensions = {"k"};
    result.topology.iterator_types = {reduction};
    result.topology.indexing_maps = {map(context, 1, {0}),
                                     map(context, 1, {0}),
                                     map(context, 1, {})};
    result.topology.operand_ranks = {1, 1, 0};
    result.topology.reduction_dimensions = {"k"};
    break;
  }
  case Operation::Ger: {
    if (lhs_orientation != Orientation::Normal ||
        rhs_orientation != Orientation::Normal) {
      result.error = "GER vector operands have no matrix orientation axis";
      return result;
    }
    result.topology.topology_class = TopologyClass::OuterProductUpdate;
    result.topology.loop_dimensions = {"m", "n"};
    result.topology.iterator_types = {parallel, parallel};
    result.topology.indexing_maps = {map(context, 2, {0}),
                                     map(context, 2, {1}),
                                     map(context, 2, {0, 1})};
    result.topology.operand_ranks = {1, 1, 2};
    break;
  }
  case Operation::BatchedGemm: {
    result.topology.topology_class = TopologyClass::ReductionContraction;
    result.topology.loop_dimensions = {"b", "m", "n", "k"};
    result.topology.iterator_types = {parallel, parallel, parallel, reduction};
    result.topology.indexing_maps = {
        lhs_orientation == Orientation::Normal ? map(context, 4, {0, 1, 3})
                                               : map(context, 4, {0, 3, 1}),
        rhs_orientation == Orientation::Normal ? map(context, 4, {0, 3, 2})
                                               : map(context, 4, {0, 2, 3}),
        map(context, 4, {0, 1, 2}),
    };
    result.topology.operand_ranks = {3, 3, 3};
    result.topology.reduction_dimensions = {"k"};
    break;
  }
  default:
    result.error = "unsupported standard linear-algebra operation identity";
    return result;
  }
  result.valid = true;
  return result;
}

bool verifyCanonicalContractionTopologyV1(
    const ContractionTopologyV1 &topology, std::string &error) {
  error.clear();
  if (topology.indexing_maps.size() != 3 ||
      topology.operand_ranks.size() != 3) {
    error = "contraction topology requires lhs, rhs, and destination maps/ranks";
    return false;
  }
  mlir::MLIRContext *context = topology.indexing_maps.front().getContext();
  if (!context || llvm::any_of(topology.indexing_maps, [&](mlir::AffineMap map) {
        return map.getContext() != context;
      })) {
    error = "contraction topology maps must share one MLIR context";
    return false;
  }
  ContractionTopologyResultV1 canonical = buildCanonicalContractionTopologyV1(
      *context, topology.operation, topology.lhs_orientation,
      topology.rhs_orientation);
  if (!canonical) {
    error = canonical.error;
    return false;
  }
  return sameTopology(topology, canonical.topology, error);
}

mlir::DictionaryAttr encodeContractionTopologyV1(
    mlir::Builder &builder, const ContractionTopologyV1 &topology) {
  llvm::SmallVector<mlir::Attribute> loops;
  llvm::SmallVector<mlir::Attribute> iterators;
  llvm::SmallVector<mlir::Attribute> maps;
  llvm::SmallVector<mlir::Attribute> ranks;
  llvm::SmallVector<mlir::Attribute> reductions;
  for (const std::string &dimension : topology.loop_dimensions)
    loops.push_back(builder.getStringAttr(dimension));
  for (mlir::utils::IteratorType iterator : topology.iterator_types)
    iterators.push_back(builder.getStringAttr(
        mlir::utils::stringifyIteratorType(iterator)));
  for (mlir::AffineMap indexing_map : topology.indexing_maps)
    maps.push_back(mlir::AffineMapAttr::get(indexing_map));
  for (unsigned rank : topology.operand_ranks)
    ranks.push_back(builder.getI64IntegerAttr(rank));
  for (const std::string &dimension : topology.reduction_dimensions)
    reductions.push_back(builder.getStringAttr(dimension));
  return builder.getDictionaryAttr({
      builder.getNamedAttr("classification",
                           builder.getStringAttr(bilinearTopologyClassNameV1(
                               topology.topology_class))),
      builder.getNamedAttr("indexing_maps", builder.getArrayAttr(maps)),
      builder.getNamedAttr("iterator_types", builder.getArrayAttr(iterators)),
      builder.getNamedAttr("lhs_orientation",
                           builder.getStringAttr(matrixOrientationNameV1(
                               topology.lhs_orientation))),
      builder.getNamedAttr("loop_dimensions", builder.getArrayAttr(loops)),
      builder.getNamedAttr("operand_ranks", builder.getArrayAttr(ranks)),
      builder.getNamedAttr("operation",
                           builder.getStringAttr(
                               standardLinearAlgebraOperationNameV1(
                                   topology.operation))),
      builder.getNamedAttr("reduction_dimensions",
                           builder.getArrayAttr(reductions)),
      builder.getNamedAttr("rhs_orientation",
                           builder.getStringAttr(matrixOrientationNameV1(
                               topology.rhs_orientation))),
      builder.getNamedAttr("schema",
                           builder.getStringAttr(kContractionTopologySchemaV1)),
      builder.getNamedAttr("version", builder.getI32IntegerAttr(
                                          kContractionTopologyVersionV1)),
  });
}

ContractionTopologyResultV1 decodeContractionTopologyV1(
    mlir::DictionaryAttr attribute, mlir::MLIRContext &context) {
  ContractionTopologyResultV1 result;
  if (!requireExactNames(attribute, kTopologyFields, result.error))
    return result;
  const auto schema = attribute.getAs<mlir::StringAttr>("schema");
  const auto version = attribute.getAs<mlir::IntegerAttr>("version");
  const auto operation = attribute.getAs<mlir::StringAttr>("operation");
  const auto lhs_orientation =
      attribute.getAs<mlir::StringAttr>("lhs_orientation");
  const auto rhs_orientation =
      attribute.getAs<mlir::StringAttr>("rhs_orientation");
  const auto classification =
      attribute.getAs<mlir::StringAttr>("classification");
  if (!schema || schema.getValue() != kContractionTopologySchemaV1 ||
      !version || !version.getType().isSignlessInteger(32) ||
      version.getInt() != kContractionTopologyVersionV1 || !operation ||
      !lhs_orientation || !rhs_orientation || !classification ||
      !parseOperation(operation.getValue(), result.topology.operation) ||
      !parseOrientation(lhs_orientation.getValue(),
                        result.topology.lhs_orientation) ||
      !parseOrientation(rhs_orientation.getValue(),
                        result.topology.rhs_orientation) ||
      !parseTopologyClass(classification.getValue(),
                          result.topology.topology_class)) {
    result.error = "contraction topology identity is invalid";
    return result;
  }
  ContractionTopologyResultV1 canonical = buildCanonicalContractionTopologyV1(
      context, result.topology.operation, result.topology.lhs_orientation,
      result.topology.rhs_orientation);
  if (!canonical) {
    result.error = canonical.error;
    return result;
  }
  mlir::Builder builder(&context);
  if (attribute != encodeContractionTopologyV1(builder, canonical.topology)) {
    result.error = "encoded contraction topology differs from its canonical "
                   "standard-operation model";
    return result;
  }
  return canonical;
}

bool verifyStructuredIndexingAgainstContractionTopologyV1(
    const ContractionTopologyV1 &topology,
    llvm::ArrayRef<mlir::AffineMap> indexing_maps,
    llvm::ArrayRef<mlir::utils::IteratorType> iterator_types,
    llvm::ArrayRef<unsigned> operand_ranks, std::string &error) {
  if (!verifyCanonicalContractionTopologyV1(topology, error))
    return false;
  if (indexing_maps != llvm::ArrayRef(topology.indexing_maps) ||
      iterator_types != llvm::ArrayRef(topology.iterator_types) ||
      operand_ranks != llvm::ArrayRef(topology.operand_ranks)) {
    error = "structured operation indexing/iterator/rank topology does not "
            "match the canonical standard-operation model";
    return false;
  }
  return true;
}

} // namespace matcore::mdslc::mlir_bridge
