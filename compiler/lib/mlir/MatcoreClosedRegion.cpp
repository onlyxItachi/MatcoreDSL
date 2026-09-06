#include "MatcoreClosedRegion.h"
#include "MatcoreDialect.h"
#include "MatcoreRegionBoundaryOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>

namespace matcore::mdslc::closed_region {
namespace {
namespace mdsl = mlir_dialect;
using Shape = std::array<std::int64_t, 2>;
constexpr auto dynamic = mlir::ShapedType::kDynamic;
constexpr auto maximum = std::numeric_limits<std::int64_t>::max();

bool fail(std::string &error, const std::string &message) {
  error = "closed-region admission: " + message;
  return false;
}
bool digest(const std::string &value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
bool siteValid(const SourceSite &site) {
  return site.length && site.line && site.column && site.offset <= maximum &&
         site.length <= static_cast<std::uint64_t>(maximum) - site.offset &&
         site.line <= maximum && site.column <= maximum;
}
bool emptyDimension(const Dimension &dim) {
  return dim.kind == Dimension::Kind::Literal && !dim.literal && !dim.reference;
}
bool profileValid(NumericalProfile profile) {
  return profile == NumericalProfile::StrictF32 ||
         profile == NumericalProfile::ReassociateF32;
}
bool comparisonValid(Comparison comparison) {
  return comparison >= Comparison::Less && comparison <= Comparison::GreaterEqual;
}
bool dimension(const Dimension &dim, const std::set<Id> &parameters,
               const std::map<Id, Shape> &values, std::int64_t &extent,
               std::string &error) {
  switch (dim.kind) {
  case Dimension::Kind::Literal:
    if (dim.reference || dim.literal > static_cast<std::uint64_t>(maximum))
      return fail(error, "literal extent is not a nonnegative signed-index value");
    extent = static_cast<std::int64_t>(dim.literal);
    return true;
  case Dimension::Kind::ShapeParameter:
    if (dim.literal || !parameters.count(dim.reference))
      return fail(error, "dimension does not reference a declared shape parameter");
    extent = dynamic;
    return true;
  case Dimension::Kind::ValueRows:
  case Dimension::Kind::ValueColumns: {
    auto found = values.find(dim.reference);
    if (dim.literal || found == values.end())
      return fail(error, "dimension references a value outside its dominating scope");
    extent = found->second[dim.kind == Dimension::Kind::ValueRows ? 0 : 1];
    return true;
  }
  }
  return fail(error, "unknown dimension kind");
}
bool extentProduct(Shape shape, std::string &error) {
  if (shape[0] != dynamic && shape[1] != dynamic && shape[0] &&
      shape[1] > maximum / shape[0])
    return fail(error, "static extent product overflows the semantic index domain");
  return true;
}
bool verifyBody(const std::vector<Operation> &body,
                const std::set<Id> &resources, const std::set<Id> &parameters,
                std::map<Id, Shape> values, std::set<Id> &allValues,
                unsigned depth, std::string &error) {
  if (depth > 16)
    return fail(error, "shape-control nesting exceeds the bounded admission depth");
  for (const auto &op : body) {
    if (!siteValid(op.site) ||
        !std::all_of(op.helper_calls.begin(), op.helper_calls.end(), siteValid))
      return fail(error, "invalid source or helper-call site");
    if (!profileValid(op.numerical_profile) || !comparisonValid(op.comparison))
      return fail(error, "unknown numerical profile or shape comparison");
    const bool hasControl = !op.then_body.empty() || !op.else_body.empty() ||
                            !emptyDimension(op.condition_lhs) ||
                            !emptyDimension(op.condition_rhs) ||
                            op.comparison != Comparison::Equal;
    const bool hasShape = !emptyDimension(op.rows) || !emptyDimension(op.columns);
    auto newValue = [&](Shape shape) {
      if (!op.result || op.result > maximum || !allValues.insert(op.result).second)
        return fail(error, "value identity is zero, duplicate or unrepresentable");
      values.emplace(op.result, shape);
      return extentProduct(shape, error);
    };
    auto knownValue = [&](Id id) { return values.find(id) != values.end(); };
    switch (op.kind) {
    case Operation::Kind::Read: {
      if (!resources.count(op.resource) || op.lhs || op.rhs || hasControl ||
          op.numerical_profile != NumericalProfile::StrictF32)
        return fail(error, "read has an invalid resource or unrelated fields");
      Shape shape;
      if (!dimension(op.rows, parameters, values, shape[0], error) ||
          !dimension(op.columns, parameters, values, shape[1], error) ||
          !newValue(shape))
        return false;
      break;
    }
    case Operation::Kind::Gemm: {
      if (!knownValue(op.lhs) || !knownValue(op.rhs) || op.resource ||
          hasShape || hasControl)
        return fail(error, "GEMM needs two dominating immutable values only");
      const auto lhs = values.at(op.lhs), rhs = values.at(op.rhs);
      if (lhs[1] != dynamic && rhs[0] != dynamic && lhs[1] != rhs[0])
        return fail(error, "statically incompatible GEMM contraction dimensions");
      if (!newValue({lhs[0], rhs[1]}))
        return false;
      break;
    }
    case Operation::Kind::Publish:
      if (!resources.count(op.resource) || !knownValue(op.lhs) || op.result ||
          op.rhs || hasShape || hasControl ||
          op.numerical_profile != NumericalProfile::StrictF32)
        return fail(error, "publication needs a resource and a dominating value only");
      break;
    case Operation::Kind::Observe:
      if (!resources.count(op.resource) || op.result || op.lhs || op.rhs ||
          hasShape || hasControl || op.numerical_profile != NumericalProfile::StrictF32)
        return fail(error, "observation needs a declared resource only");
      break;
    case Operation::Kind::ShapeIf: {
      std::int64_t ignored;
      if (op.result || op.resource || op.lhs || op.rhs || hasShape ||
          op.numerical_profile != NumericalProfile::StrictF32)
        return fail(error, "shape-if cannot export branch-local values or resources");
      if (!dimension(op.condition_lhs, parameters, values, ignored, error) ||
          !dimension(op.condition_rhs, parameters, values, ignored, error) ||
          !verifyBody(op.then_body, resources, parameters, values, allValues,
                      depth + 1, error) ||
          !verifyBody(op.else_body, resources, parameters, values, allValues,
                      depth + 1, error))
        return false;
      break;
    }
    default:
      return fail(error, "unknown operation kind");
    }
  }
  return true;
}

// Registered private inspection operations. They intentionally provide no
// lowering, execution interface or speculation permission. Generic MLIR
// verification is followed by the closed whole-module verifier below.
template <class Concrete>
class AdmissionOp : public mlir::Op<Concrete, mlir::OpTrait::VariadicOperands,
                                    mlir::OpTrait::VariadicResults,
                                    mlir::OpTrait::VariadicRegions> {
public:
  using mlir::Op<Concrete, mlir::OpTrait::VariadicOperands,
                 mlir::OpTrait::VariadicResults,
                 mlir::OpTrait::VariadicRegions>::Op;
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
};
#define ADMISSION_OP(CLASS, NAME)                                                \
  class CLASS : public AdmissionOp<CLASS> {                                    \
  public:                                                                     \
    using AdmissionOp::AdmissionOp;                                            \
    static llvm::StringRef getOperationName() { return "mdsl_admission." NAME; } \
  }
ADMISSION_OP(BeginOp, "begin");
ADMISSION_OP(ReadOp, "read");
ADMISSION_OP(CheckOp, "check_gemm");
ADMISSION_OP(GemmOp, "gemm");
ADMISSION_OP(PublishOp, "publish");
ADMISSION_OP(ObserveOp, "observe");
ADMISSION_OP(ShapeIfOp, "shape_if");
ADMISSION_OP(DimOp, "dim");
#undef ADMISSION_OP
class YieldOp : public mlir::Op<YieldOp, mlir::OpTrait::OneOperand,
                                mlir::OpTrait::ZeroResults,
                                mlir::OpTrait::ZeroRegions,
                                mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mdsl_admission.yield"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
};
class AdmissionDialect : public mlir::Dialect {
public:
  explicit AdmissionDialect(mlir::MLIRContext *context)
      : Dialect("mdsl_admission", context, mlir::TypeID::get<AdmissionDialect>()) {
    addOperations<BeginOp, ReadOp, CheckOp, GemmOp, PublishOp, ObserveOp,
                  ShapeIfOp, DimOp, YieldOp>();
  }
  static llvm::StringRef getDialectNamespace() { return "mdsl_admission"; }
};

mlir::DictionaryAttr sourceAttr(mlir::Builder &b, SourceSite site) {
  return b.getDictionaryAttr({b.getNamedAttr("offset", b.getI64IntegerAttr(site.offset)),
                             b.getNamedAttr("length", b.getI64IntegerAttr(site.length)),
                             b.getNamedAttr("line", b.getI64IntegerAttr(site.line)),
                             b.getNamedAttr("column", b.getI64IntegerAttr(site.column))});
}
mlir::ArrayAttr strings(mlir::Builder &b, llvm::ArrayRef<llvm::StringRef> names) {
  llvm::SmallVector<mlir::Attribute> result;
  for (auto name : names)
    result.push_back(b.getStringAttr(name));
  return b.getArrayAttr(result);
}
mlir::DictionaryAttr obligations(mlir::Builder &b, llvm::StringRef kind) {
  if (kind == "read")
    return b.getDictionaryAttr({
        b.getNamedAttr("runtime_required", strings(b, {"extent_index_representable", "extent_product_representable", "dense_row_major_f32_view_compatible", "read_access"})),
        b.getNamedAttr("caller_unproven", strings(b, {"initialized_f32_objects", "capacity", "lifetime", "race_freedom"}))});
  if (kind == "check_gemm")
    return b.getDictionaryAttr({
        b.getNamedAttr("runtime_required", strings(b, {"contraction_shape_equal", "result_extent_product_representable", "rounding_nearest_ties_even", "gradual_underflow"})),
        b.getNamedAttr("dispatch_retained", strings(b, {"implementation_legality", "fp_status_traps_preservation", "completion", "failure_attribution"}))});
  if (kind == "publish")
    return b.getDictionaryAttr({
        b.getNamedAttr("runtime_required", strings(b, {"destination_shape_layout", "dense_row_major_f32_view_compatible", "write_access"})),
        b.getNamedAttr("caller_unproven", strings(b, {"capacity", "lifetime", "race_freedom"})),
        b.getNamedAttr("realization_required", strings(b, {"preserve_live_immutable_values", "alias_safe_publication", "completion"}))});
  return b.getDictionaryAttr({
      b.getNamedAttr("runtime_required", strings(b, {"read_access", "observation_ready"})),
      b.getNamedAttr("caller_unproven", strings(b, {"capacity", "lifetime", "race_freedom"})),
      b.getNamedAttr("dispatch_retained", strings(b, {"observation_completion", "failure_attribution"}))});
}
void addSource(mlir::Operation *op, mlir::Builder &b, const Operation &source,
               const std::string &path) {
  op->setAttr("source", sourceAttr(b, source.site));
  llvm::SmallVector<mlir::Attribute> calls;
  for (const auto &site : source.helper_calls)
    calls.push_back(sourceAttr(b, site));
  op->setAttr("helper_calls", b.getArrayAttr(calls));
  op->setAttr("frontier", b.getStringAttr(path));
}
void addNumerics(mlir::Operation *op, mlir::Builder &b, NumericalProfile profile) {
  const bool strict = profile == NumericalProfile::StrictF32;
  op->setAttr("numerical_profile", b.getStringAttr(strict ? "strict_f32" : "reassociate_f32"));
  op->setAttr("accumulator", b.getStringAttr("f32"));
  op->setAttr("reduction_order", b.getStringAttr(strict ? "increasing_k" : "reassociation_permitted"));
  op->setAttr("multiply_add_contraction", b.getStringAttr(strict ? "forbidden" : "permitted"));
  op->setAttr("operation_boundary_rounding", b.getStringAttr("f32_nearest_ties_even"));
  op->setAttr("cross_operation_reassociation", b.getStringAttr("forbidden"));
  op->setAttr("special_values", b.getStringAttr("ieee_f32_no_finite_assumption"));
  op->setAttr("signed_zero", b.getStringAttr("not_ignored"));
  op->setAttr("nan_payload", b.getStringAttr("unspecified"));
  op->setAttr("fp_status_traps", b.getStringAttr("adaptation_unresolved_execution_forbidden"));
}
mlir::Operation *create(mlir::OpBuilder &b, llvm::StringRef name,
                       mlir::ValueRange operands, mlir::TypeRange results,
                       unsigned regions = 0) {
  mlir::OperationState state(b.getUnknownLoc(), name);
  state.addOperands(operands);
  state.addTypes(results);
  for (unsigned i = 0; i != regions; ++i)
    state.addRegion();
  return b.create(state);
}
struct BuildState {
  std::map<Id, mlir::Value> resources, parameters, values;
  mlir::Value order;
  std::string epoch = "entry";
};
mlir::Value buildDimension(mlir::OpBuilder &b, const Dimension &dim,
                            const BuildState &state) {
  if (dim.kind == Dimension::Kind::Literal)
    return b.create<mlir::arith::ConstantIntOp>(b.getUnknownLoc(), dim.literal, 64);
  if (dim.kind == Dimension::Kind::ShapeParameter)
    return state.parameters.at(dim.reference);
  auto *op = create(b, DimOp::getOperationName(), state.values.at(dim.reference), b.getI64Type());
  op->setAttr("axis", b.getI64IntegerAttr(dim.kind == Dimension::Kind::ValueRows ? 0 : 1));
  return op->getResult(0);
}
std::int64_t dimensionExtent(const Dimension &dim, const BuildState &state) {
  if (dim.kind == Dimension::Kind::Literal)
    return static_cast<std::int64_t>(dim.literal);
  if (dim.kind == Dimension::Kind::ShapeParameter)
    return dynamic;
  return mlir::cast<mlir::RankedTensorType>(state.values.at(dim.reference).getType())
      .getDimSize(dim.kind == Dimension::Kind::ValueRows ? 0 : 1);
}
void buildBody(mlir::OpBuilder &b, const std::vector<Operation> &body,
               BuildState &state, const std::string &prefix) {
  const auto orderType = mdsl::RegionOrderType::get(b.getContext());
  for (std::size_t i = 0; i < body.size(); ++i) {
    const auto &source = body[i];
    const std::string path = prefix + "." + std::to_string(i);
    mlir::Operation *op = nullptr;
    switch (source.kind) {
    case Operation::Kind::Read: {
      auto rows = buildDimension(b, source.rows, state);
      auto columns = buildDimension(b, source.columns, state);
      auto type = mlir::RankedTensorType::get(
          {dimensionExtent(source.rows, state), dimensionExtent(source.columns, state)}, b.getF32Type());
      op = create(b, ReadOp::getOperationName(), {state.order, state.resources.at(source.resource), rows, columns}, {type, orderType});
      op->setAttr("value_id", b.getI64IntegerAttr(source.result));
      op->setAttr("read_semantics", b.getStringAttr("immutable_contents_at_frontier"));
      op->setAttr("failure", b.getStringAttr("ordered_checked_read_failure_no_write"));
      state.values[source.result] = op->getResult(0);
      state.order = op->getResult(1);
      break;
    }
    case Operation::Kind::Gemm: {
      auto lhs = state.values.at(source.lhs), rhs = state.values.at(source.rhs);
      auto *check = create(b, CheckOp::getOperationName(), {state.order, lhs, rhs}, orderType);
      addSource(check, b, source, path + ".check");
      addNumerics(check, b, source.numerical_profile);
      check->setAttr("obligations", obligations(b, "check_gemm"));
      check->setAttr("evidence", b.getStringAttr("required_not_discharged"));
      check->setAttr("failure", b.getStringAttr("ordered_checked_failure_no_write"));
      check->setAttr("resource_epoch", b.getStringAttr(state.epoch));
      auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
      auto rhsType = mlir::cast<mlir::RankedTensorType>(rhs.getType());
      auto type = mlir::RankedTensorType::get({lhsType.getDimSize(0), rhsType.getDimSize(1)}, b.getF32Type());
      state.order = check->getResult(0);
      op = create(b, GemmOp::getOperationName(), {state.order, lhs, rhs}, type);
      addNumerics(op, b, source.numerical_profile);
      op->setAttr("value_id", b.getI64IntegerAttr(source.result));
      op->setAttr("semantics", b.getStringAttr("immutable_mathematical_result"));
      state.values[source.result] = op->getResult(0);
      break;
    }
    case Operation::Kind::Publish:
      op = create(b, PublishOp::getOperationName(), {state.order, state.resources.at(source.resource), state.values.at(source.lhs)}, orderType);
      op->setAttr("failure", b.getStringAttr("may_partially_mutate_no_rollback"));
      op->setAttr("input_epoch", b.getStringAttr(state.epoch));
      state.epoch = path;
      op->setAttr("output_epoch", b.getStringAttr(state.epoch));
      state.order = op->getResult(0);
      break;
    case Operation::Kind::Observe:
      op = create(b, ObserveOp::getOperationName(), {state.order, state.resources.at(source.resource)}, orderType);
      op->setAttr("semantics", b.getStringAttr("guaranteed_current_resource_observation"));
      op->setAttr("failure", b.getStringAttr("ordered_observation_failure_no_rollback"));
      state.order = op->getResult(0);
      break;
    case Operation::Kind::ShapeIf: {
      auto lhs = buildDimension(b, source.condition_lhs, state);
      auto rhs = buildDimension(b, source.condition_rhs, state);
      const std::array predicates{mlir::arith::CmpIPredicate::ult, mlir::arith::CmpIPredicate::ule,
                                 mlir::arith::CmpIPredicate::eq, mlir::arith::CmpIPredicate::ne,
                                 mlir::arith::CmpIPredicate::ugt, mlir::arith::CmpIPredicate::uge};
      auto condition = mlir::arith::CmpIOp::create(b, b.getUnknownLoc(), predicates[static_cast<unsigned>(source.comparison)], lhs, rhs);
      op = create(b, ShapeIfOp::getOperationName(), {state.order, condition}, orderType, 2);
      op->setAttr("input_epoch", b.getStringAttr(state.epoch));
      for (unsigned branch = 0; branch != 2; ++branch) {
        auto *block = new mlir::Block;
        op->getRegion(branch).push_back(block);
        block->addArgument(orderType, b.getUnknownLoc());
        mlir::OpBuilder nested(b.getContext());
        nested.setInsertionPointToStart(block);
        auto branchState = state;
        branchState.order = block->getArgument(0);
        buildBody(nested, branch ? source.else_body : source.then_body, branchState,
                  path + (branch ? ".else" : ".then"));
        create(nested, YieldOp::getOperationName(), branchState.order, {});
        op->setAttr(branch ? "else_epoch" : "then_epoch", b.getStringAttr(branchState.epoch));
      }
      state.epoch = path + ".join";
      op->setAttr("output_epoch", b.getStringAttr(state.epoch));
      state.order = op->getResult(0);
      break;
    }
    }
    addSource(op, b, source, path);
    op->setAttr("resource_epoch", b.getStringAttr(state.epoch));
    if (source.kind == Operation::Kind::Read || source.kind == Operation::Kind::Publish ||
        source.kind == Operation::Kind::Observe) {
      op->setAttr("resource_id", b.getI64IntegerAttr(source.resource));
      auto kind = source.kind == Operation::Kind::Read ? "read" :
                  source.kind == Operation::Kind::Publish ? "publish" : "observe";
      op->setAttr("obligations", obligations(b, kind));
      op->setAttr("evidence", b.getStringAttr("required_not_discharged"));
      if (source.kind != Operation::Kind::Observe)
        op->setAttr("storage_view", b.getStringAttr("requested_dense_row_major_f32"));
    }
  }
}

bool textIs(mlir::Operation *op, llvm::StringRef name, llvm::StringRef expected) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(name);
  return attr && attr.getValue() == expected;
}
bool i64(mlir::Attribute attr, std::int64_t &value) {
  auto integer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr);
  if (!integer || !integer.getType().isSignlessInteger(64))
    return false;
  value = integer.getInt();
  return true;
}
bool sourceValid(mlir::Attribute attr) {
  auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  std::int64_t offset, length, line, column;
  return dict && dict.size() == 4 && i64(dict.get("offset"), offset) &&
         i64(dict.get("length"), length) && i64(dict.get("line"), line) &&
         i64(dict.get("column"), column) && offset >= 0 && length > 0 &&
         line > 0 && column > 0 && length <= maximum - offset;
}
bool tensor(mlir::Type type) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return ranked && ranked.getRank() == 2 && ranked.getElementType().isF32() && !ranked.getEncoding();
}
bool numericsValid(mlir::Operation *op) {
  const bool strict = textIs(op, "numerical_profile", "strict_f32");
  return (strict || textIs(op, "numerical_profile", "reassociate_f32")) &&
         textIs(op, "accumulator", "f32") &&
         textIs(op, "reduction_order", strict ? "increasing_k" : "reassociation_permitted") &&
         textIs(op, "multiply_add_contraction", strict ? "forbidden" : "permitted") &&
         textIs(op, "operation_boundary_rounding", "f32_nearest_ties_even") &&
         textIs(op, "cross_operation_reassociation", "forbidden") &&
         textIs(op, "special_values", "ieee_f32_no_finite_assumption") &&
         textIs(op, "signed_zero", "not_ignored") && textIs(op, "nan_payload", "unspecified") &&
         textIs(op, "fp_status_traps", "adaptation_unresolved_execution_forbidden");
}
bool attrsAre(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> allowed,
              std::string &error) {
  for (auto attr : op->getAttrs())
    if (!llvm::is_contained(allowed, attr.getName().strref()))
      return fail(error, "unknown attribute on " + op->getName().getStringRef().str());
  return true;
}
struct VerifyState {
  mlir::Value order;
  std::string epoch = "entry";
  std::map<Id, mlir::Value> resources;
  llvm::DenseSet<mlir::Value> values;
};
std::int64_t knownExtent(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  auto *op = value.getDefiningOp();
  std::int64_t axis;
  if (op && op->getName().getStringRef() == DimOp::getOperationName() &&
      op->getNumOperands() == 1 && tensor(op->getOperand(0).getType()) &&
      i64(op->getAttr("axis"), axis) && axis >= 0 && axis < 2)
    return mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType()).getDimSize(axis);
  return dynamic;
}
bool verifyBlock(mlir::Block &block, VerifyState &state, std::set<Id> &ids,
                  std::set<std::string> &frontiers, bool branch, unsigned depth,
                  std::string &error) {
  if (depth > 16)
    return fail(error, "shape-control nesting exceeds the bounded admission depth");
  mlir::Builder b(block.getParent()->getContext());
  auto orderType = mdsl::RegionOrderType::get(b.getContext());
  auto resourceType = mdsl::RegionDescriptorType::get(b.getContext());
  bool ended = false;
  mlir::Operation *pendingCheck = nullptr;
  for (auto &operation : block) {
    auto *op = &operation;
    auto name = op->getName().getStringRef();
    if (ended)
      return fail(error, "operation follows a terminator");
    if (name == "func.return" || name == YieldOp::getOperationName()) {
      if (pendingCheck || op->getNumOperands() != 1 || op->getOperand(0) != state.order ||
          op->getNumResults() || op->getNumRegions() || !op->getAttrs().empty() ||
          (branch != (name == YieldOp::getOperationName())))
        return fail(error, "effect completion token is dropped or branch terminator is invalid");
      ended = true;
      continue;
    }
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantIntOp>(op)) {
      if (pendingCheck || !constant.getType().isSignlessInteger(64) || constant.value() < 0 || !attrsAre(op, {"value"}, error))
        return fail(error, "invalid or misplaced nonnegative extent literal");
      continue;
    }
    if (auto comparison = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      if (pendingCheck || !comparison.getLhs().getType().isSignlessInteger(64) ||
          !comparison.getRhs().getType().isSignlessInteger(64) ||
          !attrsAre(op, {"predicate"}, error))
        return fail(error, "invalid shape comparison");
      const auto predicate = comparison.getPredicate();
      if (predicate != mlir::arith::CmpIPredicate::eq && predicate != mlir::arith::CmpIPredicate::ne &&
          predicate != mlir::arith::CmpIPredicate::ult && predicate != mlir::arith::CmpIPredicate::ule &&
          predicate != mlir::arith::CmpIPredicate::ugt && predicate != mlir::arith::CmpIPredicate::uge)
        return fail(error, "shape comparison is not an admitted unsigned predicate");
      continue;
    }
    if (name == DimOp::getOperationName()) {
      std::int64_t axis;
      if (pendingCheck || op->getNumOperands() != 1 || !state.values.count(op->getOperand(0)) ||
          op->getNumResults() != 1 || !op->getResult(0).getType().isSignlessInteger(64) ||
          op->getNumRegions() || !i64(op->getAttr("axis"), axis) || axis < 0 || axis > 1 ||
          !attrsAre(op, {"axis"}, error))
        return fail(error, "invalid value-dimension query");
      continue;
    }
    if (name == BeginOp::getOperationName()) {
      if (branch || state.order || op->getNumOperands() || op->getNumResults() != 1 ||
          op->getResult(0).getType() != orderType || op->getNumRegions() || !op->getAttrs().empty())
        return fail(error, "invalid entry sequencing token");
      state.order = op->getResult(0);
      continue;
    }
    const bool read = name == ReadOp::getOperationName(), check = name == CheckOp::getOperationName(),
               gemm = name == GemmOp::getOperationName(), publish = name == PublishOp::getOperationName(),
               observe = name == ObserveOp::getOperationName(), shapeIf = name == ShapeIfOp::getOperationName();
    if ((!read && !check && !gemm && !publish && !observe && !shapeIf) ||
        op->getNumOperands() == 0 || op->getOperand(0) != state.order)
      return fail(error, "unknown operation or broken ordered effect chain");
    if (pendingCheck && !gemm)
      return fail(error, "checked GEMM frontier is separated from its computation");
    auto calls = op->getAttrOfType<mlir::ArrayAttr>("helper_calls");
    auto frontier = op->getAttrOfType<mlir::StringAttr>("frontier");
    if (!sourceValid(op->getAttr("source")) || !calls || !frontier || frontier.getValue().empty() ||
        !std::all_of(calls.begin(), calls.end(), sourceValid))
      return fail(error, "missing source/frontier identity");
    if (!frontiers.insert(frontier.getValue().str()).second)
      return fail(error, "duplicate source-effect frontier identity");
    if (!shapeIf && op->getNumRegions())
      return fail(error, "unexpected nested region");
    if (!publish && !shapeIf && !textIs(op, "resource_epoch", state.epoch))
      return fail(error, "stale or forged resource epoch");
    llvm::SmallVector<llvm::StringRef> allowed{"source", "helper_calls", "frontier", "resource_epoch"};
    if (check || gemm) {
      for (auto attr : {"numerical_profile", "accumulator", "reduction_order", "multiply_add_contraction",
                        "operation_boundary_rounding", "cross_operation_reassociation", "special_values",
                        "signed_zero", "nan_payload", "fp_status_traps"})
        allowed.push_back(attr);
      if (!numericsValid(op) || op->getNumOperands() != 3 ||
          !state.values.count(op->getOperand(1)) || !state.values.count(op->getOperand(2)))
        return fail(error, "GEMM lost operand values or numerical contract");
    }
    if (read || publish || observe) {
      allowed.push_back("resource_id");
      std::int64_t resource;
      if (!i64(op->getAttr("resource_id"), resource) || resource <= 0 ||
          !state.resources.count(resource) || op->getNumOperands() < 2 ||
          op->getOperand(1) != state.resources.at(resource) || op->getOperand(1).getType() != resourceType)
        return fail(error, "resource identity does not bind its imported capability");
      if (read || publish) {
        allowed.push_back("storage_view");
        if (!textIs(op, "storage_view", "requested_dense_row_major_f32"))
          return fail(error, "resource effect lacks its required dense row-major f32 view contract");
      }
    }
    if (read || check || publish || observe) {
      allowed.append({"obligations", "evidence"});
      const auto kind = read ? "read" : check ? "check_gemm" : publish ? "publish" : "observe";
      if (!textIs(op, "evidence", "required_not_discharged") || op->getAttr("obligations") != obligations(b, kind))
        return fail(error, "obligations were dropped, changed or asserted discharged");
    }
    if (read || gemm) {
      allowed.push_back("value_id");
      std::int64_t id;
      if (!i64(op->getAttr("value_id"), id) || id <= 0 || !ids.insert(id).second ||
          op->getNumResults() != (read ? 2u : 1u) || !tensor(op->getResult(0).getType()))
        return fail(error, "immutable value identity or tensor type is invalid");
      state.values.insert(op->getResult(0));
    }
    if (read) {
      allowed.append({"read_semantics", "failure"});
      if (op->getNumOperands() != 4 || !op->getOperand(2).getType().isSignlessInteger(64) ||
          !op->getOperand(3).getType().isSignlessInteger(64) || op->getResult(1).getType() != orderType ||
          !textIs(op, "read_semantics", "immutable_contents_at_frontier") ||
          !textIs(op, "failure", "ordered_checked_read_failure_no_write"))
        return fail(error, "read is not an immutable value at the current frontier");
      auto type = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      for (unsigned axis = 0; axis != 2; ++axis) {
        const auto extent = knownExtent(op->getOperand(axis + 2));
        if (extent != dynamic && extent != type.getDimSize(axis))
          return fail(error, "read tensor shape contradicts its explicit extent");
        if (extent == dynamic && !type.isDynamicDim(axis))
          return fail(error, "read invents an unproved static extent");
      }
      if (!extentProduct({type.getDimSize(0), type.getDimSize(1)}, error))
        return false;
      state.order = op->getResult(1);
    } else if (check) {
      allowed.push_back("failure");
      if (op->getNumResults() != 1 || op->getResult(0).getType() != orderType ||
          !textIs(op, "failure", "ordered_checked_failure_no_write"))
        return fail(error, "checked failure contract is invalid");
      pendingCheck = op;
      state.order = op->getResult(0);
    } else if (gemm) {
      allowed.push_back("semantics");
      if (!pendingCheck || pendingCheck->getOperand(1) != op->getOperand(1) ||
          pendingCheck->getOperand(2) != op->getOperand(2) ||
          pendingCheck->getAttr("numerical_profile") != op->getAttr("numerical_profile") ||
          !textIs(op, "semantics", "immutable_mathematical_result"))
        return fail(error, "GEMM bypasses its exact checked operand/profile contract");
      auto lhs = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
      auto rhs = mlir::cast<mlir::RankedTensorType>(op->getOperand(2).getType());
      auto result = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
      if (result.getDimSize(0) != lhs.getDimSize(0) || result.getDimSize(1) != rhs.getDimSize(1) ||
          (!lhs.isDynamicDim(1) && !rhs.isDynamicDim(0) && lhs.getDimSize(1) != rhs.getDimSize(0)))
        return fail(error, "GEMM result geometry or contraction is invalid");
      if (!extentProduct({result.getDimSize(0), result.getDimSize(1)}, error))
        return false;
      pendingCheck = nullptr;
    } else if (publish || observe) {
      allowed.push_back("failure");
      if (op->getNumResults() != 1 || op->getResult(0).getType() != orderType ||
          op->getNumOperands() != (publish ? 3u : 2u))
        return fail(error, "resource effect has invalid operands/results");
      if (publish) {
        allowed.append({"input_epoch", "output_epoch"});
        if (!state.values.count(op->getOperand(2)) || !textIs(op, "input_epoch", state.epoch) ||
            !textIs(op, "output_epoch", frontier.getValue()) ||
            !textIs(op, "resource_epoch", frontier.getValue()) ||
            !textIs(op, "failure", "may_partially_mutate_no_rollback"))
          return fail(error, "publication lost value, version or partial-failure meaning");
        state.epoch = frontier.getValue().str();
      } else {
        allowed.push_back("semantics");
        if (!textIs(op, "semantics", "guaranteed_current_resource_observation") ||
            !textIs(op, "failure", "ordered_observation_failure_no_rollback"))
          return fail(error, "observation was weakened or lost its failure frontier");
      }
      state.order = op->getResult(0);
    } else if (shapeIf) {
      allowed.append({"input_epoch", "then_epoch", "else_epoch", "output_epoch"});
      if (op->getNumOperands() != 2 || !op->getOperand(1).getType().isInteger(1) ||
          op->getNumResults() != 1 || op->getResult(0).getType() != orderType ||
          op->getNumRegions() != 2 || !textIs(op, "input_epoch", state.epoch))
        return fail(error, "shape-if lost its condition or incoming resource state");
      for (unsigned i = 0; i != 2; ++i) {
        auto &region = op->getRegion(i);
        if (!region.hasOneBlock() || region.front().getNumArguments() != 1 ||
            region.front().getArgument(0).getType() != orderType)
          return fail(error, "shape-if branch lacks its effect argument");
        auto branchState = state;
        branchState.order = region.front().getArgument(0);
        if (!verifyBlock(region.front(), branchState, ids, frontiers, true, depth + 1, error) ||
            !textIs(op, i ? "else_epoch" : "then_epoch", branchState.epoch))
          return fail(error, "shape-if branch resource state is not joined");
      }
      state.epoch = frontier.getValue().str() + ".join";
      if (!textIs(op, "output_epoch", state.epoch) || !textIs(op, "resource_epoch", state.epoch))
        return fail(error, "shape-if joined resource state is invalid");
      if (!frontiers.insert(state.epoch).second)
        return fail(error, "shape-if join reuses a prior source-effect frontier");
      state.order = op->getResult(0);
    }
    if (!attrsAre(op, allowed, error))
      return false;
  }
  return ended || fail(error, "region drops its completion effect token");
}
} // namespace

bool verifyProgram(const Program &program, std::string &error) {
  error.clear();
  if (program.source_identity.empty() || !digest(program.source_sha256) ||
      !digest(program.header_sha256) || program.compiler_identity.empty() || program.regions.empty())
    return fail(error, "missing source/header/compiler identity or region");
  std::set<std::string> names;
  for (const auto &region : program.regions) {
    if (region.name.empty() || !names.insert(region.name).second || !siteValid(region.site))
      return fail(error, "duplicate region name or invalid region source site");
    std::set<Id> resources, parameters, allValues, parameterIndices;
    for (const auto &resource : region.resources)
      if (!resource.id || resource.id > maximum || resource.name.empty() ||
          !resources.insert(resource.id).second || resource.parameter_index > maximum ||
          !parameterIndices.insert(resource.parameter_index).second)
        return fail(error, "duplicate or invalid resource parameter");
    for (const auto &parameter : region.shape_parameters)
      if (!parameter.id || parameter.id > maximum || parameter.name.empty() ||
          !parameters.insert(parameter.id).second || parameter.parameter_index > maximum ||
          !parameterIndices.insert(parameter.parameter_index).second)
        return fail(error, "duplicate or invalid shape parameter");
    if (!verifyBody(region.body, resources, parameters, {}, allValues, 0, error))
      return false;
  }
  return true;
}
void registerDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mdsl::MatcoreDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<AdmissionDialect>();
}
Result buildModule(const Program &program, mlir::MLIRContext &context) {
  Result result;
  if (!verifyProgram(program, result.error))
    return result;
  registerDialects(context);
  mlir::OpBuilder b(&context);
  result.module = mlir::ModuleOp::create(b.getUnknownLoc());
  auto module = *result.module;
  module->setAttr("mdsl_admission.schema", b.getStringAttr("private_closed_region_v1"));
  module->setAttr("mdsl_admission.authority", b.getStringAttr("inspection_only_no_execution"));
  module->setAttr("mdsl_admission.aliasing", b.getStringAttr("all_resources_may_alias"));
  module->setAttr("mdsl_admission.shape_scalar", b.getStringAttr("unsigned_64_no_target_index_narrowing"));
  module->setAttr("mdsl_admission.source_identity", b.getStringAttr(program.source_identity));
  module->setAttr("mdsl_admission.source_sha256", b.getStringAttr(program.source_sha256));
  module->setAttr("mdsl_admission.header_sha256", b.getStringAttr(program.header_sha256));
  module->setAttr("mdsl_admission.compiler_identity", b.getStringAttr(program.compiler_identity));
  for (std::size_t i = 0; i < program.regions.size(); ++i) {
    const auto &region = program.regions[i];
    llvm::SmallVector<mlir::Type> inputs(region.resources.size(), mdsl::RegionDescriptorType::get(&context));
    inputs.append(region.shape_parameters.size(), b.getI64Type());
    auto orderType = mdsl::RegionOrderType::get(&context);
    auto function = mlir::func::FuncOp::create(b.getUnknownLoc(), region.name, b.getFunctionType(inputs, orderType));
    function.setPrivate();
    function->setAttr("mdsl_admission.source", sourceAttr(b, region.site));
    function->setAttr("mdsl_admission.aliasing", b.getStringAttr("all_resources_may_alias"));
    module.push_back(function);
    auto *block = function.addEntryBlock();
    b.setInsertionPointToStart(block);
    BuildState state;
    unsigned argument = 0;
    for (const auto &resource : region.resources) {
      state.resources[resource.id] = block->getArgument(argument);
      function.setArgAttr(argument, "mdsl_admission.resource_id", b.getI64IntegerAttr(resource.id));
      function.setArgAttr(argument, "mdsl_admission.source_parameter", b.getI64IntegerAttr(resource.parameter_index));
      function.setArgAttr(argument++, "mdsl_admission.name", b.getStringAttr(resource.name));
    }
    for (const auto &parameter : region.shape_parameters) {
      state.parameters[parameter.id] = block->getArgument(argument);
      function.setArgAttr(argument, "mdsl_admission.shape_id", b.getI64IntegerAttr(parameter.id));
      function.setArgAttr(argument, "mdsl_admission.source_parameter", b.getI64IntegerAttr(parameter.parameter_index));
      function.setArgAttr(argument++, "mdsl_admission.name", b.getStringAttr(parameter.name));
    }
    state.order = create(b, BeginOp::getOperationName(), {}, orderType)->getResult(0);
    buildBody(b, region.body, state, "region" + std::to_string(i));
    mlir::func::ReturnOp::create(b, b.getUnknownLoc(), state.order);
  }
  if (!verifyModule(module, result.error))
    result.module = nullptr;
  return result;
}
bool verifyModule(mlir::ModuleOp module, std::string &error) {
  error.clear();
  if (!module || mlir::failed(mlir::verify(module)))
    return fail(error, "invalid registered MLIR");
  if (!textIs(module, "mdsl_admission.schema", "private_closed_region_v1") ||
      !textIs(module, "mdsl_admission.authority", "inspection_only_no_execution") ||
      !textIs(module, "mdsl_admission.aliasing", "all_resources_may_alias") ||
      !textIs(module, "mdsl_admission.shape_scalar", "unsigned_64_no_target_index_narrowing") ||
      !attrsAre(module, {"mdsl_admission.schema", "mdsl_admission.authority", "mdsl_admission.aliasing",
                        "mdsl_admission.shape_scalar",
                        "mdsl_admission.source_identity", "mdsl_admission.source_sha256", "mdsl_admission.header_sha256",
                        "mdsl_admission.compiler_identity"}, error))
    return fail(error, "module changes private inspection authority or alias contract");
  for (auto name : {"mdsl_admission.source_identity", "mdsl_admission.compiler_identity"}) {
    auto value = module->getAttrOfType<mlir::StringAttr>(name);
    if (!value || value.getValue().empty())
      return fail(error, "module lacks source/compiler identity");
  }
  for (auto name : {"mdsl_admission.source_sha256", "mdsl_admission.header_sha256"}) {
    auto value = module->getAttrOfType<mlir::StringAttr>(name);
    if (!value || !digest(value.str()))
      return fail(error, "invalid source/header digest");
  }
  if (module.getBody()->empty())
    return fail(error, "empty admission module");
  for (auto &operation : *module.getBody()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function || function.isExternal() || !function.isPrivate() ||
        !function.getBody().hasOneBlock() || function.getNumResults() != 1 ||
        function.getResultTypes()[0] != mdsl::RegionOrderType::get(module.getContext()) ||
        !sourceValid(function->getAttr("mdsl_admission.source")) ||
        !textIs(function, "mdsl_admission.aliasing", "all_resources_may_alias") ||
        !attrsAre(function, {"sym_name", "sym_visibility", "function_type", "arg_attrs",
                            "mdsl_admission.source", "mdsl_admission.aliasing"}, error))
      return fail(error, "invalid private region function");
    VerifyState state;
    std::set<Id> ids, shapeIds, parameterIds;
    for (unsigned i = 0; i < function.getNumArguments(); ++i) {
      auto attrs = function.getArgAttrDict(i);
      std::int64_t id, parameter;
      auto name = attrs ? attrs.getAs<mlir::StringAttr>("mdsl_admission.name") : mlir::StringAttr{};
      if (!attrs || attrs.size() != 3 || !name || name.getValue().empty() ||
          !i64(attrs.get("mdsl_admission.source_parameter"), parameter) || parameter < 0 ||
          !parameterIds.insert(parameter).second)
        return fail(error, "invalid parameter source binding");
      auto argument = function.getArgument(i);
      if (argument.getType() == mdsl::RegionDescriptorType::get(module.getContext())) {
        if (!i64(attrs.get("mdsl_admission.resource_id"), id) || id <= 0 ||
            !state.resources.emplace(id, argument).second)
          return fail(error, "invalid resource argument identity");
      } else if (argument.getType().isSignlessInteger(64)) {
        if (!i64(attrs.get("mdsl_admission.shape_id"), id) || id <= 0 || !shapeIds.insert(id).second)
          return fail(error, "invalid shape argument identity");
      } else {
        return fail(error, "unsupported argument type");
      }
    }
    std::set<std::string> frontiers{"entry"};
    if (!verifyBlock(function.front(), state, ids, frontiers, false, 0, error))
      return false;
  }
  return true;
}
std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, mlir::OpPrintingFlags().useLocalScope());
  return text;
}
bool verifyModuleMatchesProgram(const Program &program, mlir::ModuleOp module,
                                std::string &error) {
  if (!verifyModule(module, error))
    return false;
  auto expected = buildModule(program, *module.getContext());
  if (!expected)
    return fail(error, "paired admission record is invalid: " + expected.error);
  if (printModule(*expected.module) != printModule(module))
    return fail(error, "untransformed semantic module does not match admitted source evidence");
  return true;
}
} // namespace matcore::mdslc::closed_region
