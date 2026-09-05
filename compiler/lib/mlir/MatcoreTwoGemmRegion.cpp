#include "MatcoreTwoGemmRegion.h"
#include "MatcoreContractionModel.h"
#include "MatcoreOps.h"
#include "MatcoreRegionBoundaryOps.h"
#include "MatcoreRegionGuardLedger.h"
#include "MatcoreStructuredGemmHandoff.h"
#include "MatcoreV1Bridge.h"
#include "matcore_ir_v1.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <iterator>
#include <string>
#include <vector>

namespace matcore::mdslc::mlir_bridge {
namespace {
namespace d = mlir_dialect;
constexpr llvm::StringLiteral kFailure =
    "may_write_output_before_failure_no_rollback";
std::string printed(mlir::Attribute value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  if (value)
    value.print(stream);
  return result;
}
bool same(mlir::Attribute a, mlir::Attribute b) {
  return a && b && printed(a) == printed(b);
}
template <typename T>
T build(mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange operands,
        mlir::TypeRange results, llvm::ArrayRef<mlir::NamedAttribute> attrs = {}) {
  mlir::OperationState state(loc, T::getOperationName());
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return mlir::cast<T>(b.create(state));
}

struct SourceRegion {
  mlir::DictionaryAttr contract;
  std::array<mlir::DictionaryAttr, 2> semantics;
  std::array<mlir::FunctionType, 2> types;
  std::array<std::array<unsigned, 3>, 2> bindings;
  std::array<mlir::DictionaryAttr, 2> ledgers;
  mlir::Location location;
  std::string symbol;
  unsigned descriptor_count;
};

bool collectSource(const frontend::AuthenticatedNativeFrontendEvidenceV1 &seal,
                   mlir::MLIRContext &context,
                   std::vector<SourceRegion> &regions, std::string &identity,
                   std::string &error) {
  if (!seal.valid()) {
    error = "two-GEMM regions require valid sealed native evidence";
    return false;
  }
  const auto &options = detail::AuthenticatedNativeFrontendEvidenceAccessV1::options(seal);
  const auto &source = detail::AuthenticatedNativeFrontendEvidenceAccessV1::result(seal);
  if (!options.inspect_two_gemm_regions ||
      source.module.producer != "clang-libtooling-v1" ||
      source.region_capture_identity.empty() || !source.diagnostics.empty()) {
    error = "two-GEMM source must be successfully admitted in native region inspection mode";
    return false;
  }
  identity = source.region_capture_identity;
  ir::v1::Module v1;
  if (!ir::v1::fromV0(source.module, v1, error))
    return false;
  auto semantic = bridgeV1ToMatcoreMlir(v1, context,
                                       explicitGemmF32V1BridgeContext());
  if (!semantic) {
    error = semantic.error;
    return false;
  }
  llvm::SmallVector<mlir::func::FuncOp> functions(
      semantic.module->getOps<mlir::func::FuncOp>());
  mlir::Builder b(&context);
  for (const auto &candidate : source.two_gemm_regions) {
    if (!candidate.admitted)
      continue;
    if (candidate.sites.size() != 2 || !candidate.rejection_reasons.empty() ||
        candidate.region_id.empty() || candidate.source_snapshot_sha256.empty() ||
        candidate.sites[0].bindings[0].descriptor_id !=
            candidate.sites[1].bindings[1].descriptor_id ||
        candidate.sites[0].bindings[0].descriptor_id ==
            candidate.sites[1].bindings[0].descriptor_id) {
      error = "sealed two-GEMM region lacks the admitted producer/consumer relationship";
      return false;
    }
    SourceRegion region{{}, {}, {}, {}, {}, b.getUnknownLoc(),
                        "__matcore_region_" + candidate.region_id, 0};
    llvm::StringMap<unsigned> descriptors;
    llvm::SmallVector<mlir::Attribute> descriptor_ids;
    llvm::SmallVector<mlir::Attribute> sites;
    for (unsigned stage = 0; stage != 2; ++stage) {
      const auto &site = candidate.sites[stage];
      if (site.capture_ordinal >= functions.size()) {
        error = "sealed region refers to a missing capture site";
        return false;
      }
      auto function = functions[site.capture_ordinal];
      auto gemm = mlir::cast<d::GemmOp>(function.getBody().front().front());
      if (gemm.getSiteId() != site.site_id) {
        error = "sealed descriptor site differs from the verified capture";
        return false;
      }
      region.semantics[stage] = gemm->getAttrDictionary();
      region.types[stage] = function.getFunctionType();
      if (!stage)
        region.location = gemm.getLoc();
      llvm::SmallVector<mlir::Attribute> bindings;
      for (unsigned role = 0; role != 3; ++role) {
        const auto &binding = site.bindings[role];
        if (binding.descriptor_id.empty() || binding.declaration_id.empty() ||
            binding.snapshot_stage != stage) {
          error = "sealed descriptor bindings require per-call snapshot timing";
          return false;
        }
        auto inserted = descriptors.try_emplace(binding.descriptor_id,
                                                 descriptors.size());
        if (inserted.second)
          descriptor_ids.push_back(b.getStringAttr(binding.descriptor_id));
        region.bindings[stage][role] = inserted.first->second;
        bindings.push_back(b.getDictionaryAttr({
            b.getNamedAttr("descriptor", b.getStringAttr(binding.descriptor_id)),
            b.getNamedAttr("declaration", b.getStringAttr(binding.declaration_id)),
            b.getNamedAttr("source_expression", b.getStringAttr(binding.source_expression)),
            b.getNamedAttr("snapshot_stage", b.getI64IntegerAttr(stage)),
            b.getNamedAttr("argument", b.getI64IntegerAttr(inserted.first->second))}));
      }
      region.ledgers[stage] = buildRegionGuardLedgerV1(
          b, region.semantics[stage], b.getArrayAttr(bindings), stage, error);
      if (!region.ledgers[stage])
        return false;
      sites.push_back(b.getDictionaryAttr({
          b.getNamedAttr("site_id", b.getStringAttr(site.site_id)),
          b.getNamedAttr("source_contract", region.semantics[stage]),
          b.getNamedAttr("bindings", b.getArrayAttr(bindings)),
          b.getNamedAttr("tensor_type", mlir::TypeAttr::get(region.types[stage]))}));
    }
    region.descriptor_count = descriptors.size();
    region.contract = b.getDictionaryAttr({
        b.getNamedAttr("region_id", b.getStringAttr(candidate.region_id)),
        b.getNamedAttr("function_identity", b.getStringAttr(candidate.function_identity)),
        b.getNamedAttr("source_snapshot", b.getStringAttr(candidate.source_snapshot_sha256)),
        b.getNamedAttr("begin", b.getI64IntegerAttr(candidate.source_range.begin)),
        b.getNamedAttr("end", b.getI64IntegerAttr(candidate.source_range.end)),
        b.getNamedAttr("descriptors", b.getArrayAttr(descriptor_ids)),
        b.getNamedAttr("sites", b.getArrayAttr(sites))});
    regions.push_back(std::move(region));
  }
  if (regions.empty()) {
    error = "no admitted adjacent two-GEMM region in sealed native evidence";
    for (const auto &candidate : source.two_gemm_regions)
      for (const auto &reason : candidate.rejection_reasons)
        error += "; " + reason;
    return false;
  }
  return true;
}

mlir::Value dim(mlir::OpBuilder &b, mlir::Location loc, mlir::Value tensor,
                unsigned index) {
  return mlir::tensor::DimOp::create(b, loc, tensor, index).getResult();
}
void appendRegion(mlir::ModuleOp module, const SourceRegion &source) {
  mlir::OpBuilder b(module.getContext());
  const auto descriptor_type = d::RegionDescriptorType::get(module.getContext());
  const auto order_type = d::RegionOrderType::get(module.getContext());
  llvm::SmallVector<mlir::Type> arguments(source.descriptor_count, descriptor_type);
  auto function = mlir::func::FuncOp::create(
      source.location, source.symbol, b.getFunctionType(arguments, {}));
  function->setAttr("mdsl.region", source.contract);
  module.push_back(function);
  auto *entry = function.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto current = build<d::RegionBeginOp>(b, source.location, {}, {order_type}).getOrder();
  mlir::Value previous_value;
  for (unsigned stage = 0; stage != 2; ++stage) {
    auto output = entry->getArgument(source.bindings[stage][0]);
    auto lhs_desc = entry->getArgument(source.bindings[stage][1]);
    auto rhs_desc = entry->getArgument(source.bindings[stage][2]);
    auto guard = build<d::RegionGuardOp>(
        b, source.location, {current, lhs_desc, rhs_desc, output}, {order_type},
        {b.getNamedAttr("stage", b.getI64IntegerAttr(stage)),
         b.getNamedAttr("semantic_contract", source.semantics[stage]),
         b.getNamedAttr("guard_ledger", source.ledgers[stage])});
    const auto read = [&](mlir::Value descriptor, unsigned index,
                           llvm::StringRef role) -> mlir::Value {
      return build<d::RegionReadOp>(
          b, source.location, {descriptor, guard.getChecked()},
          {source.types[stage].getInput(index)},
          {b.getNamedAttr("stage", b.getI64IntegerAttr(stage)),
           b.getNamedAttr("role", b.getStringAttr(role))}).getValue();
    };
    mlir::Value lhs = stage ? previous_value : read(lhs_desc, 0, "lhs");
    mlir::Value rhs = read(rhs_desc, 1, "rhs");
    auto output_type = mlir::cast<mlir::RankedTensorType>(source.types[stage].getInput(2));
    llvm::SmallVector<mlir::Value> sizes;
    if (output_type.isDynamicDim(0))
      sizes.push_back(dim(b, source.location, lhs, 0));
    if (output_type.isDynamicDim(1))
      sizes.push_back(dim(b, source.location, rhs, 1));
    auto empty = mlir::tensor::EmptyOp::create(
        b, source.location, output_type.getShape(), output_type.getElementType(), sizes);
    auto zero = mlir::arith::ConstantOp::create(b, source.location, b.getF32FloatAttr(0.0));
    auto fill = mlir::linalg::FillOp::create(
        b, source.location, mlir::TypeRange{output_type},
        mlir::ValueRange{zero.getResult()}, mlir::ValueRange{empty.getResult()});
    auto matmul = mlir::linalg::MatmulOp::create(
        b, source.location, mlir::TypeRange{output_type}, mlir::ValueRange{lhs, rhs},
        mlir::ValueRange{fill.getResultTensors().front()});
    auto commit = build<d::RegionCommitOp>(
        b, source.location, {matmul.getResultTensors().front(), output, guard.getChecked()},
        {output_type, order_type},
        {b.getNamedAttr("stage", b.getI64IntegerAttr(stage)),
         b.getNamedAttr("failure_behavior", b.getStringAttr(kFailure))});
    previous_value = commit.getCommitted();
    current = commit.getOrder();
  }
  build<d::RegionEndOp>(b, source.location, {current}, {});
  mlir::func::ReturnOp::create(b, source.location);
}

bool fail(std::string &error, llvm::StringRef message) {
  error = message.str();
  return false;
}
mlir::Value stripCast(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::tensor::CastOp>()) {
    if (cast.getSource().getType() != cast.getDest().getType())
      break;
    value = cast.getSource();
  }
  return value;
}
bool dimension(mlir::Value size, mlir::Value value, unsigned index) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  if (auto constant = size.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return !type.isDynamicDim(index) && constant.value() == type.getDimSize(index);
  auto op = size.getDefiningOp<mlir::tensor::DimOp>();
  return op && stripCast(op.getSource()) == stripCast(value) &&
         op.getConstantIndex() == index;
}
bool safeIncidental(mlir::Operation &operation) {
  if (mlir::isa<mlir::arith::ConstantOp>(operation))
    return true;
  if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(operation))
    return cast.getSource().getType() == cast.getDest().getType();
  if (auto query = mlir::dyn_cast<mlir::tensor::DimOp>(operation)) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(query.getSource().getType());
    auto index = query.getConstantIndex();
    return type && index && *index >= 0 && *index < type.getRank();
  }
  return false;
}
bool canonicalScalar(mlir::linalg::LinalgOp operation, bool contraction) {
  auto &block = operation->getRegion(0).front();
  for (auto &nested : block.without_terminator()) {
    bool supported = mlir::isa<mlir::arith::ConstantOp>(nested);
    if (auto add = mlir::dyn_cast<mlir::arith::AddFOp>(nested))
      supported = add.getFastmath() == mlir::arith::FastMathFlags::none;
    if (auto multiply = mlir::dyn_cast<mlir::arith::MulFOp>(nested))
      supported = multiply.getFastmath() == mlir::arith::FastMathFlags::none;
    if (!supported || !mlir::isMemoryEffectFree(&nested) ||
        !mlir::isSpeculatable(&nested) || nested.getNumRegions())
      return false;
  }
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  if (!contraction)
    return block.getNumArguments() == 2 && yield.getOperand(0) == block.getArgument(0);
  if (block.getNumArguments() != 3)
    return false;
  auto add = yield.getOperand(0).getDefiningOp<mlir::arith::AddFOp>();
  if (!add || add.getFastmath() != mlir::arith::FastMathFlags::none)
    return false;
  auto product = add.getRhs().getDefiningOp<mlir::arith::MulFOp>();
  return product && product.getFastmath() == mlir::arith::FastMathFlags::none &&
         add.getLhs() == block.getArgument(2) &&
         product.getLhs() == block.getArgument(0) &&
         product.getRhs() == block.getArgument(1);
}
bool computation(d::RegionCommitOp commit, mlir::Value lhs, mlir::Value rhs,
                 llvm::SmallPtrSetImpl<mlir::Operation *> &validated,
                 std::string &error) {
  auto matmul = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
      stripCast(commit.getValue()).getDefiningOp());
  if (!matmul || matmul.getNumDpsInputs() != 2 ||
      matmul.getNumDpsInits() != 1 || matmul->getNumResults() != 1 ||
      stripCast(matmul.getDpsInputOperand(0)->get()) != stripCast(lhs) ||
      stripCast(matmul.getDpsInputOperand(1)->get()) != stripCast(rhs))
    return fail(error, "region contraction must consume the current guarded input versions");
  auto topology = buildCanonicalContractionTopologyV1(*commit.getContext(),
                                                     StandardLinearAlgebraOperationV1::Gemm);
  if (!topology || matmul.getIndexingMapsArray() != topology.topology.indexing_maps ||
      matmul.getIteratorTypesArray() != topology.topology.iterator_types ||
      !canonicalScalar(matmul, true))
    return fail(error, "region contraction indexing or numerical computation changed");
  auto fill = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
      stripCast(matmul.getDpsInitOperand(0)->get()).getDefiningOp());
  if (!fill || fill.getNumDpsInputs() != 1 || fill.getNumDpsInits() != 1 ||
      fill->getNumResults() != 1 || !canonicalScalar(fill, false))
    return fail(error, "region overwrite must initialize every output element");
  validated.insert(matmul.getOperation());
  validated.insert(fill.getOperation());
  auto zero = fill.getDpsInputOperand(0)->get().getDefiningOp<mlir::arith::ConstantOp>();
  auto constant = zero ? mlir::dyn_cast<mlir::FloatAttr>(zero.getValue()) : mlir::FloatAttr{};
  auto *context = commit.getContext();
  llvm::SmallVector<mlir::AffineMap> fill_maps{
      mlir::AffineMap::get(2, 0, context),
      mlir::AffineMap::getMultiDimIdentityMap(2, context)};
  if (!constant || !constant.getType().isF32() || !constant.getValue().isZero() ||
      constant.getValue().isNegative() || fill.getIndexingMapsArray() != fill_maps ||
      fill.getIteratorTypesArray() != llvm::SmallVector<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::parallel, mlir::utils::IteratorType::parallel})
    return fail(error, "region initialization must be a full positive-zero fill");
  auto empty = stripCast(fill.getDpsInitOperand(0)->get()).getDefiningOp<mlir::tensor::EmptyOp>();
  if (!empty || empty.getType() != commit.getValue().getType())
    return fail(error, "overwrite initialization may not read original destination data");
  validated.insert(empty.getOperation());
  unsigned dynamic = 0;
  for (unsigned axis = 0; axis != 2; ++axis) {
    if (!empty.getType().isDynamicDim(axis))
      continue;
    if (!dimension(empty.getDynamicSizes()[dynamic++], axis ? rhs : lhs, axis))
      return fail(error, "region output extent does not follow current input shapes");
  }
  return true;
}

bool verifyFunction(mlir::func::FuncOp function, std::string &error) {
  if (!function.isPublic() || !function.getBody().hasOneBlock() ||
      function.getNumResults() || function.getArgAttrsAttr() || function.getResAttrsAttr())
    return fail(error, "region requires one public block with descriptor inputs and observable void result");
  auto contract = function->getAttrOfType<mlir::DictionaryAttr>("mdsl.region");
  auto sites = contract ? contract.getAs<mlir::ArrayAttr>("sites") : mlir::ArrayAttr{};
  auto descriptors = contract ? contract.getAs<mlir::ArrayAttr>("descriptors") : mlir::ArrayAttr{};
  if (!sites || sites.size() != 2 || !descriptors ||
      function.getNumArguments() != descriptors.size())
    return fail(error, "region source contract requires two sites and all descriptor bindings");
  for (auto argument : function.getArguments())
    if (!mlir::isa<d::RegionDescriptorType>(argument.getType()))
      return fail(error, "region inputs must be source descriptor references");
  std::array<d::RegionGuardOp, 2> guard;
  std::array<d::RegionCommitOp, 2> commit;
  std::array<std::array<d::RegionReadOp, 2>, 2> reads;
  d::RegionBeginOp begin;
  d::RegionEndOp end;
  unsigned boundary = 0;
  for (mlir::Operation &op : function.getBody().front().without_terminator()) {
    if (auto value = mlir::dyn_cast<d::RegionBeginOp>(op)) {
      if (begin || boundary++)
        return fail(error, "region begin must precede both failure frontiers");
      begin = value;
    } else if (auto value = mlir::dyn_cast<d::RegionGuardOp>(op)) {
      auto stage = value.getStage();
      if (!begin || guard[stage] || boundary != 1 + 2 * stage)
        return fail(error, "guard ordering changed relative to observable commits");
      guard[stage] = value;
      ++boundary;
    } else if (auto value = mlir::dyn_cast<d::RegionCommitOp>(op)) {
      auto stage = value.getStage();
      if (commit[stage] || boundary != 2 + 2 * stage)
        return fail(error, "commit must precede the next call's failure frontier");
      commit[stage] = value;
      ++boundary;
    } else if (auto value = mlir::dyn_cast<d::RegionReadOp>(op)) {
      auto stage = value.getStage();
      unsigned role = value.getRole() == "lhs" ? 0 : 1;
      if (!guard[stage] || boundary != 2 + 2 * stage || reads[stage][role] ||
          (stage == 1 && role == 0))
        return fail(error, "input snapshots must follow their guard and previous commit");
      reads[stage][role] = value;
    } else if (auto value = mlir::dyn_cast<d::RegionEndOp>(op)) {
      if (end || boundary != 5)
        return fail(error, "region end must follow both commits");
      end = value;
      ++boundary;
    } else if (!mlir::isMemoryEffectFree(&op) ||
               (!mlir::isa<mlir::linalg::LinalgOp, mlir::tensor::EmptyOp>(&op) &&
                (!safeIncidental(op) || !mlir::isSpeculatable(&op) ||
                 op.getNumRegions()))) {
      return fail(error, "region contains an unmodeled effect or unsafe speculative computation");
    }
  }
  if (!begin || !end || !guard[0] || !guard[1] || !commit[0] || !commit[1] ||
      !reads[0][0] || !reads[0][1] || !reads[1][1] || boundary != 6)
    return fail(error, "region lost a required validation, input snapshot or observable commit");
  auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(function.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() || guard[0].getOrder() != begin.getOrder() ||
      guard[1].getOrder() != commit[0].getOrder() || end.getOrder() != commit[1].getOrder())
    return fail(error, "region successful-continuation chain is disconnected");
  llvm::SmallPtrSet<mlir::Operation *, 4> validated_computations;
  for (unsigned stage = 0; stage != 2; ++stage) {
    auto site = mlir::dyn_cast<mlir::DictionaryAttr>(sites[stage]);
    auto bindings = site ? site.getAs<mlir::ArrayAttr>("bindings") : mlir::ArrayAttr{};
    auto encoded_type = site ? site.getAs<mlir::TypeAttr>("tensor_type") : mlir::TypeAttr{};
    auto type = encoded_type ? mlir::dyn_cast<mlir::FunctionType>(encoded_type.getValue()) : mlir::FunctionType{};
    if (!bindings || bindings.size() != 3 || !type || type.getNumInputs() != 3 ||
        type.getNumResults() != 1 || !same(guard[stage].getSemanticContract(), site.get("source_contract")) ||
        commit[stage].getChecked() != guard[stage].getChecked())
      return fail(error, "region guard lost a source contract or conditional runtime obligation");
    if (!verifyRegionGuardLedgerV1(guard[stage].getGuardLedger(),
                                  guard[stage].getSemanticContract(), bindings,
                                  stage, error))
      return false;
    std::array<mlir::Value, 3> arguments;
    for (unsigned role = 0; role != 3; ++role) {
      auto binding = mlir::dyn_cast<mlir::DictionaryAttr>(bindings[role]);
      auto index = binding ? binding.getAs<mlir::IntegerAttr>("argument") : mlir::IntegerAttr{};
      auto descriptor = binding ? binding.getAs<mlir::StringAttr>("descriptor") : mlir::StringAttr{};
      auto snapshot = binding ? binding.getAs<mlir::IntegerAttr>("snapshot_stage") : mlir::IntegerAttr{};
      if (!index || !index.getType().isSignlessInteger(64) || index.getInt() < 0 ||
          index.getInt() >= function.getNumArguments() ||
          !descriptor || descriptor != descriptors[index.getInt()] ||
          !snapshot || snapshot.getInt() != stage)
        return fail(error, "region descriptor binding or snapshot stage changed");
      arguments[role] = function.getArgument(index.getInt());
    }
    if (guard[stage].getOutput() != arguments[0] || guard[stage].getLhs() != arguments[1] ||
        guard[stage].getRhs() != arguments[2] || commit[stage].getDescriptor() != arguments[0] ||
        commit[stage].getCommitted().getType() != type.getResult(0))
      return fail(error, "guard and commit do not refer to their original descriptor bindings");
    for (unsigned role = 0; role != 2; ++role) {
      if (stage && !role)
        continue;
      auto read = reads[stage][role];
      if (read.getDescriptor() != arguments[role + 1] ||
          read.getChecked() != guard[stage].getChecked() ||
          read.getValue().getType() != type.getInput(role))
        return fail(error, "region input snapshot has stale guard or descriptor");
    }
    auto lhs = stage ? commit[0].getCommitted() : reads[0][0].getValue();
    auto rhs = reads[stage][1].getValue();
    if (stage && (arguments[1] != commit[0].getDescriptor() ||
                  arguments[0] == commit[0].getDescriptor()))
      return fail(error, "second GEMM must consume the first committed descriptor version");
    if (!computation(commit[stage], lhs, rhs, validated_computations, error))
      return false;
  }
  for (auto &operation : function.getBody().front())
    if (mlir::isa<mlir::linalg::LinalgOp, mlir::tensor::EmptyOp>(&operation) &&
        !validated_computations.contains(&operation))
      return fail(error, "region contains an unvalidated structured computation");
  return true;
}
} // namespace

void registerTwoGemmRegionDialectsV1(mlir::MLIRContext &context) {
  registerStructuredGemmHandoffDialectsV1(context);
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
}

TwoGemmRegionResultV1 deriveAuthenticatedTwoGemmRegionsV1(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence,
    mlir::MLIRContext &context) {
  registerTwoGemmRegionDialectsV1(context);
  TwoGemmRegionResultV1 result;
  std::vector<SourceRegion> regions;
  std::string identity;
  if (!collectSource(evidence, context, regions, identity, result.error))
    return result;
  mlir::Builder b(&context);
  result.module = mlir::ModuleOp::create(b.getUnknownLoc());
  (*result.module)->setAttr("mdsl.analysis_only", b.getBoolAttr(true));
  (*result.module)->setAttr("mdsl.execution_authority", b.getStringAttr("inspection_only"));
  (*result.module)->setAttr("mdsl.two_gemm_region_version", b.getI32IntegerAttr(1));
  (*result.module)->setAttr("mdsl.capture_identity", b.getStringAttr(identity));
  (*result.module)->setAttr("mdsl.failure_obligations", b.getStringAttr(kFailure));
  (*result.module)->setAttr("mdsl.alias_facts", b.getStringAttr("distinct_descriptors_may_alias"));
  (*result.module)->setAttr("mdsl.guard_status", b.getStringAttr("conditional_success_no_runtime_facts_discharged"));
  for (const auto &region : regions)
    appendRegion(*result.module, region);
  if (!verifyTwoGemmRegionMatchesNativeEvidenceV1(evidence, *result.module, result.error))
    result.module = nullptr;
  return result;
}

bool verifyTwoGemmRegionModuleV1(mlir::ModuleOp module, std::string &error) {
  error.clear();
  if (!module || mlir::failed(mlir::verify(module)))
    return fail(error, "two-GEMM region failed upstream/dialect verification");
  mlir::Builder b(module.getContext());
  if (module->getAttr("mdsl.analysis_only") != b.getBoolAttr(true) ||
      module->getAttr("mdsl.execution_authority") != b.getStringAttr("inspection_only") ||
      module->getAttr("mdsl.two_gemm_region_version") != b.getI32IntegerAttr(1) ||
      module->getAttr("mdsl.failure_obligations") != b.getStringAttr(kFailure) ||
      module->getAttr("mdsl.alias_facts") != b.getStringAttr("distinct_descriptors_may_alias") ||
      module->getAttr("mdsl.guard_status") != b.getStringAttr("conditional_success_no_runtime_facts_discharged") ||
      !module->getAttrOfType<mlir::StringAttr>("mdsl.capture_identity") ||
      module.getBody()->empty())
    return fail(error, "region authority, alias or conditional failure boundary changed");
  for (auto &operation : module.getBody()->getOperations()) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(operation);
    if (!function || !verifyFunction(function, error))
      return false;
  }
  return true;
}

bool verifyTwoGemmRegionMatchesNativeEvidenceV1(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence,
    mlir::ModuleOp module, std::string &error) {
  if (!verifyTwoGemmRegionModuleV1(module, error))
    return false;
  std::vector<SourceRegion> sources;
  std::string identity;
  if (!collectSource(evidence, *module.getContext(), sources, identity, error))
    return false;
  if (module->getAttrOfType<mlir::StringAttr>("mdsl.capture_identity").getValue() != identity ||
      std::distance(module.getBody()->begin(), module.getBody()->end()) !=
          static_cast<std::ptrdiff_t>(sources.size()))
    return fail(error, "region artifact does not match the sealed native compilation");
  llvm::StringSet<> found;
  for (auto function : module.getOps<mlir::func::FuncOp>()) {
    auto source = llvm::find_if(sources, [&](const SourceRegion &candidate) {
      return function.getName() == candidate.symbol;
    });
    if (source == sources.end() || !found.insert(function.getName()).second ||
        !same(function->getAttr("mdsl.region"), source->contract))
      return fail(error, "region descriptor/source contract differs from sealed native evidence");
  }
  return true;
}
} // namespace matcore::mdslc::mlir_bridge
