#include "MatcoreCpuFusedGemmCandidate.h"
#include "MatcoreBufferizedGemmHandoff.h"
#include "MatcoreClosedRegion.h"
#include "MatcoreContractionModel.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

namespace matcore::mdslc::cpu_candidate {
namespace {
constexpr llvm::StringLiteral contract =
    "matcore.builtin.strict-fused-gemm-f32.v1:A[M,K],B[K,N],D[N,P]->E[M,P];"
    "C=A*B;E=C*D;single-private-lhs-use;no-effects;increasing-k-and-n;"
    "positive-zero;separate-f32-mul-add;intermediate-f32-round;"
    "nearest-even;gradual-underflow;nan-payload-unspecified;no-fast-math;"
    "no-cross-op-reassociation;caller-private-E-and-row-panel;no-publication";
constexpr llvm::StringLiteral transformRecipe = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %consumer = transform.structured.match attributes{matcore.pair.consumer} in %root : (!transform.any_op) -> !transform.any_op
    %producer = transform.structured.match attributes{matcore.pair.producer} in %root : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match attributes{matcore.pair.fill} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %consumer tile_sizes [4, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %updated = transform.structured.fuse_into_containing_op %producer into %loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %updated_fill = transform.structured.fuse_into_containing_op %fill into %updated : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
)mlir";

// Comparison references ONLY, never scheduled payloads or execution inputs.
// Whole-operation equivalence checks all types, attributes, ordered operations,
// SSA bindings, loop structure, maps and nested scalar bodies, ignoring locations
// alone. Actual payload loops arise solely through upstream Transform/Linalg.
constexpr llvm::StringLiteral scheduledReference = R"mlir(
#map = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
module {
  func.func @__matcore_strict_fused_gemm_f32_v1(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %d: tensor<?x?xf32>, %e: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %four = arith.constant 4 : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %z = arith.constant 0.0 : f32
    %m = tensor.dim %a, %zero : tensor<?x?xf32>
    %n = tensor.dim %b, %one : tensor<?x?xf32>
    %init = linalg.fill ins(%z : f32) outs(%e : tensor<?x?xf32>) -> tensor<?x?xf32>
    %p = tensor.dim %d, %one : tensor<?x?xf32>
    %result = scf.for %i = %zero to %m step %four iter_args(%out = %init) -> (tensor<?x?xf32>) {
      %rows = affine.min #map(%i)[%m]
      %k = tensor.dim %a, %one : tensor<?x?xf32>
      %ar = tensor.extract_slice %a[%i, 0] [%rows, %k] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %br = tensor.extract_slice %b[0, 0] [%k, %n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %panel = tensor.empty(%rows, %n) : tensor<?x?xf32>
      %ci = linalg.fill {matcore.pair.fill} ins(%z : f32) outs(%panel : tensor<?x?xf32>) -> tensor<?x?xf32>
      %c = linalg.matmul {matcore.pair.producer} ins(%ar, %br : tensor<?x?xf32>, tensor<?x?xf32>) outs(%ci : tensor<?x?xf32>) -> tensor<?x?xf32>
      %dr = tensor.extract_slice %d[0, 0] [%n, %p] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %er = tensor.extract_slice %out[%i, 0] [%rows, %p] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %product = linalg.matmul {matcore.pair.consumer} ins(%c, %dr : tensor<?x?xf32>, tensor<?x?xf32>) outs(%er : tensor<?x?xf32>) -> tensor<?x?xf32>
      %inserted = tensor.insert_slice %product into %out[%i, 0] [%rows, %p] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %inserted : tensor<?x?xf32>
    }
    return %result : tensor<?x?xf32>
  }
}
)mlir";
constexpr llvm::StringLiteral bufferReference = R"mlir(
#map = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
module {
  func.func @__matcore_strict_fused_gemm_f32_v1(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %d: memref<?x?xf32>, %e: memref<?x?xf32>, %workspace: memref<?x?xf32>) {
    %four = arith.constant 4 : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %z = arith.constant 0.0 : f32
    %m = memref.dim %a, %zero : memref<?x?xf32>
    %n = memref.dim %b, %one : memref<?x?xf32>
    linalg.fill ins(%z : f32) outs(%e : memref<?x?xf32>)
    %p = memref.dim %d, %one : memref<?x?xf32>
    scf.for %i = %zero to %m step %four {
      %rows = affine.min #map(%i)[%m]
      %k = memref.dim %a, %one : memref<?x?xf32>
      %ar = memref.subview %a[%i, 0] [%rows, %k] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %br = memref.subview %b[0, 0] [%k, %n] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
      %panel = memref.reinterpret_cast %workspace to offset: [0], sizes: [%rows, %n], strides: [%n, 1] : memref<?x?xf32> to memref<?x?xf32>
      linalg.fill {matcore.pair.fill} ins(%z : f32) outs(%panel : memref<?x?xf32>)
      linalg.matmul {matcore.pair.producer} ins(%ar, %br : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1]>>) outs(%panel : memref<?x?xf32>)
      %dr = memref.subview %d[0, 0] [%n, %p] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
      %er = memref.subview %e[%i, 0] [%rows, %p] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      linalg.matmul {matcore.pair.consumer} ins(%panel, %dr : memref<?x?xf32>, memref<?x?xf32, strided<[?, 1]>>) outs(%er : memref<?x?xf32, strided<[?, 1], offset: ?>>)
    }
    return
  }
}
)mlir";

bool fail(std::string &error, llvm::StringRef message) {
  error = "strict fused CPU GEMM candidate: " + message.str();
  return false;
}
std::string digest(llvm::StringRef value) {
  auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(value));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}
std::string print(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream, mlir::OpPrintingFlags().useLocalScope());
  return result;
}
closed_region::Program primitive() {
  namespace cr = closed_region;
  cr::Program p;
  p.source_identity = "matcore-builtin:strict-fused-gemm-f32-v1";
  p.source_sha256 = digest(contract);
  p.source_files = {{1, p.source_identity, p.source_sha256, contract.size()}};
  p.header_sha256 = digest("no-source-header:compiler-owned-primitive");
  p.compiler_identity = "matcore-cpu-fused-candidate:LLVM-" LLVM_VERSION_STRING;
  cr::Region r;
  r.name = "strict_fused_gemm_primitive";
  r.site = {0, 1, 1, 1, 1};
  r.resources = {{1, "A", 0}, {2, "B", 1}, {3, "D", 2}};
  r.shape_parameters = {{1, "M", 3}, {2, "K", 4}, {3, "N", 5}, {4, "P", 6}};
  auto shape = [](cr::Id id) { return cr::Dimension{cr::Dimension::Kind::ShapeParameter, 0, id}; };
  auto read = [&](cr::Id id, cr::Id rows, cr::Id cols) {
    cr::Operation op;
    op.site = {id - 1, 1, id, 1, 1}; op.result = id; op.resource = id;
    op.rows = shape(rows); op.columns = shape(cols); return op;
  };
  cr::Operation first, second;
  first.kind = second.kind = cr::Operation::Kind::Gemm;
  first.site = {3, 1, 4, 1, 1}; first.result = 4; first.lhs = 1; first.rhs = 2;
  second.site = {4, 1, 5, 1, 1}; second.result = 5; second.lhs = 4; second.rhs = 3;
  r.body = {read(1, 1, 2), read(2, 2, 3), read(3, 3, 4), first, second};
  p.regions = {r}; return p;
}
void registerDialects(mlir::MLIRContext &context) {
  mlir_bridge::registerBufferizedGemmHandoffDialectsV1(context);
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::scf::SCFDialect>();
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::transform::TransformDialect>();
}
mlir::OwningOpRef<mlir::ModuleOp> structured(mlir::MLIRContext &context) {
  mlir::OpBuilder b(&context); auto loc = b.getUnknownLoc();
  auto type = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, b.getF32Type());
  mlir::OwningOpRef<mlir::ModuleOp> m = mlir::ModuleOp::create(loc);
  auto fn = mlir::func::FuncOp::create(loc, kStrictFusedGemmSymbolV1,
      b.getFunctionType({type, type, type, type}, {type}));
  m->push_back(fn); auto *block = fn.addEntryBlock(); b.setInsertionPointToStart(block);
  auto zero = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto z = mlir::arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0));
  auto rows = mlir::tensor::DimOp::create(b, loc, block->getArgument(0), zero);
  auto cols = mlir::tensor::DimOp::create(b, loc, block->getArgument(1), one);
  auto empty = mlir::tensor::EmptyOp::create(b, loc, llvm::ArrayRef<std::int64_t>{mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, b.getF32Type(), mlir::ValueRange{rows, cols});
  auto init = mlir::linalg::FillOp::create(b, loc, mlir::ValueRange{z}, mlir::ValueRange{empty});
  init->setAttr("matcore.pair.fill", b.getUnitAttr());
  auto c = mlir::linalg::MatmulOp::create(b, loc, mlir::TypeRange{type},
      mlir::ValueRange{block->getArgument(0), block->getArgument(1)}, init.getResults());
  c->setAttr("matcore.pair.producer", b.getUnitAttr());
  auto ei = mlir::linalg::FillOp::create(b, loc, mlir::ValueRange{z}, mlir::ValueRange{block->getArgument(3)});
  auto e = mlir::linalg::MatmulOp::create(b, loc, mlir::TypeRange{type},
      mlir::ValueRange{c.getResult(0), block->getArgument(2)}, ei.getResults());
  e->setAttr("matcore.pair.consumer", b.getUnitAttr());
  mlir::func::ReturnOp::create(b, loc, e.getResults()); return m;
}
bool verifyArithmetic(mlir::ModuleOp module, std::string &error) {
  unsigned products = 0, fills = 0; bool bad = false;
  auto topology = mlir_bridge::buildCanonicalContractionTopologyV1(*module.getContext(), mlir_bridge::StandardLinearAlgebraOperationV1::Gemm);
  module.walk([&](mlir::linalg::MatmulOp op) {
    ++products; llvm::SmallVector<mlir::AffineMap> maps;
    for (auto a : op.getIndexingMaps()) maps.push_back(mlir::cast<mlir::AffineMapAttr>(a).getValue());
    bad |= op.hasUserDefinedMaps() || op.getCast() != mlir::linalg::TypeFn::cast_signed ||
        !mlir_bridge::verifyStructuredIndexingAgainstContractionTopologyV1(topology.topology, maps, op.getIteratorTypesArray(), {2, 2, 2}, error);
    auto &block = op.getRegion().front();
    if (block.getOperations().size() != 3) { bad = true; return; }
    auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(block.front());
    auto add = mlir::dyn_cast<mlir::arith::AddFOp>(*std::next(block.begin()));
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(block.back());
    bad |= !mul || !add || !yield || mul.getLhs() != block.getArgument(0) ||
        mul.getRhs() != block.getArgument(1) || add.getLhs() != block.getArgument(2) ||
        add.getRhs() != mul.getResult() || yield.getOperand(0) != add.getResult() ||
        mul.getFastmath() != mlir::arith::FastMathFlags::none || add.getFastmath() != mlir::arith::FastMathFlags::none;
  });
  module.walk([&](mlir::linalg::FillOp op) {
    ++fills; auto constant = op.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
    auto attr = constant ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue()) : mlir::FloatAttr{};
    bad |= !attr || !attr.getType().isF32() || !attr.getValue().isZero() || attr.getValue().isNegative();
  });
  return products == 2 && fills == 2 && !bad ? true : fail(error, "strict two-matmul arithmetic/topology/positive-zero footprint changed");
}
bool equivalent(mlir::ModuleOp module, mlir::ModuleOp reference, std::string &error) {
  if (!module || !reference || mlir::failed(mlir::verify(module)) ||
      !mlir::OperationEquivalence::isEquivalentTo(module, reference, mlir::OperationEquivalence::IgnoreLocations))
    return fail(error, "stage differs from exact compiler-owned operation graph");
  return verifyArithmetic(module, error);
}
mlir::Value originalTensorArgument(mlir::Value value) {
  if (auto op = value.getDefiningOp<mlir::bufferization::ToBufferOp>()) value = op.getTensor();
  if (auto op = value.getDefiningOp<mlir::bufferization::ToTensorOp>()) value = op.getBuffer();
  return value;
}
bool mapWorkspace(mlir::ModuleOp module, std::string &error) {
  auto fn = *module.getOps<mlir::func::FuncOp>().begin();
  auto type = mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, mlir::Float32Type::get(module.getContext()));
  if (mlir::failed(fn.insertArgument(4, type, {}, fn.getLoc()))) return fail(error, "cannot add private workspace descriptor");
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false; options.bufferizeFunctionBoundaries = true;
  options.copyBeforeWrite = false;
  options.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  unsigned requests = 0;
  options.allocationFn = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::MemRefType requested,
      mlir::ValueRange sizes, unsigned) -> mlir::FailureOr<mlir::Value> {
    if (++requests != 1 || requested != type || sizes.size() != 2) return mlir::failure();
    auto rows = sizes[0].getDefiningOp<mlir::affine::AffineMinOp>();
    auto cols = sizes[1].getDefiningOp<mlir::memref::DimOp>();
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(b.getInsertionBlock()->getParentOp());
    auto bound = loop ? loop.getUpperBound().getDefiningOp<mlir::memref::DimOp>() : mlir::memref::DimOp{};
    auto lower = loop ? loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>() : mlir::arith::ConstantIndexOp{};
    auto step = loop ? loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>() : mlir::arith::ConstantIndexOp{};
    auto expected = mlir::AffineMap::get(1, 1, {-mlir::getAffineDimExpr(0, module.getContext()) + mlir::getAffineSymbolExpr(0, module.getContext()), mlir::getAffineConstantExpr(4, module.getContext())}, module.getContext());
    if (!rows || !cols || !loop || loop->getParentOp() != fn || rows->getParentOp() != loop ||
        !bound || originalTensorArgument(bound.getSource()) != fn.getArgument(0) || bound.getConstantIndex() != 0 ||
        !lower || lower.value() != 0 || !step || step.value() != 4 || rows.getAffineMap() != expected ||
        rows.getOperand(0) != loop.getInductionVar() || rows.getOperand(1) != loop.getUpperBound() ||
        originalTensorArgument(cols.getSource()) != fn.getArgument(1) || cols.getConstantIndex() != 1) return mlir::failure();
    llvm::SmallVector<mlir::OpFoldResult> shape{sizes[0], sizes[1]}, strides{sizes[1], b.getIndexAttr(1)};
    return mlir::memref::ReinterpretCastOp::create(b, loc, type, fn.getArgument(4), b.getIndexAttr(0), shape, strides).getResult();
  };
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(module, options, state, &statistics)) || requests != 1 || statistics.numBufferDealloc)
    return fail(error, "One-Shot did not map exactly the certified caller-owned row panel");
  mlir::PassManager cleanup(module.getContext());
  cleanup.addPass(mlir::createCanonicalizerPass()); cleanup.addPass(mlir::createCSEPass()); cleanup.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(cleanup.run(module)) || mlir::failed(mlir::bufferization::dropEquivalentBufferResults(module)))
    return fail(error, "workspace/output equivalence cleanup failed");
  return verifyStrictFusedGemmBufferizedV1(module, error);
}

// Resolve only the pinned inserted-descriptor/extracted-pointer chain. No
// arbitrary pointer provenance, PHI or alias certificate is inferred here.
const llvm::Value *field(const llvm::Value *value, llvm::ArrayRef<unsigned> path, unsigned budget) {
  if (!budget) return nullptr;
  if (path.empty()) return value;
  if (auto *insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
    auto indices = insert->getIndices();
    if (path.size() >= indices.size() && path.take_front(indices.size()) == indices)
      return field(insert->getInsertedValueOperand(), path.drop_front(indices.size()), budget - 1);
    return field(insert->getAggregateOperand(), path, budget - 1);
  }
  if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(value)) {
    llvm::SmallVector<unsigned> combined(extract->getIndices()); combined.append(path.begin(), path.end());
    return field(extract->getAggregateOperand(), combined, budget - 1);
  }
  return nullptr;
}
const llvm::Argument *pointerArgument(const llvm::Value *value, unsigned budget = 128) {
  if (!value || !budget) return nullptr;
  if (auto *arg = llvm::dyn_cast<llvm::Argument>(value)) return arg;
  if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(value)) return pointerArgument(gep->getPointerOperand(), budget - 1);
  if (auto *cast = llvm::dyn_cast<llvm::BitCastInst>(value)) return pointerArgument(cast->getOperand(0), budget - 1);
  if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(value))
    return pointerArgument(field(extract->getAggregateOperand(), extract->getIndices(), budget - 1), budget - 1);
  return nullptr;
}
} // namespace

bool verifyStrictFusedGemmStructuredV1(mlir::ModuleOp module, std::string &error) {
  error.clear(); if (!module) return fail(error, "missing structured pair");
  auto expected = structured(*module.getContext()); return equivalent(module, *expected, error);
}
bool verifyStrictFusedGemmScheduledV1(mlir::ModuleOp module, std::string &error) {
  error.clear(); if (!module) return fail(error, "missing scheduled pair");
  auto expected = mlir::parseSourceString<mlir::ModuleOp>(scheduledReference, module.getContext());
  return equivalent(module, expected ? *expected : mlir::ModuleOp{}, error);
}
bool verifyStrictFusedGemmBufferizedV1(mlir::ModuleOp module, std::string &error) {
  error.clear(); if (!module) return fail(error, "missing bufferized pair");
  auto expected = mlir::parseSourceString<mlir::ModuleOp>(bufferReference, module.getContext());
  return equivalent(module, expected ? *expected : mlir::ModuleOp{}, error);
}
StrictFusedGemmStagesV1 buildStrictFusedGemmStagesV1(mlir::MLIRContext &context) {
  StrictFusedGemmStagesV1 result;
  if (llvm::StringRef(LLVM_VERSION_STRING) != "21.1.8") { fail(result.error, "requires coherent LLVM/MLIR 21.1.8"); return result; }
  registerDialects(context);
  auto semantic = closed_region::buildModule(primitive(), context);
  if (!semantic || !closed_region::verifyModuleMatchesProgram(primitive(), *semantic.module, result.error)) {
    if (result.error.empty()) result.error = semantic.error; return result;
  }
  result.semantic = std::move(semantic.module);
  result.structured = structured(context);
  if (!verifyStrictFusedGemmStructuredV1(*result.structured, result.error)) return result;
  auto transform = mlir::parseSourceString<mlir::ModuleOp>(transformRecipe, &context);
  if (!transform) { fail(result.error, "cannot parse built-in Transform recipe"); return result; }
  auto entry = transform->lookupSymbol<mlir::transform::NamedSequenceOp>(mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  result.scheduled = result.structured->clone();
  if (!entry || mlir::failed(mlir::transform::applyTransformNamedSequence(result.scheduled->getOperation(), entry.getOperation(), *transform,
      mlir::transform::TransformOptions().enableExpensiveChecks(true))) || !verifyStrictFusedGemmScheduledV1(*result.scheduled, result.error)) {
    if (result.error.empty()) fail(result.error, "pinned row-panel Transform failed"); result.scheduled = nullptr; return result;
  }
  result.bufferized = result.scheduled->clone();
  if (!mapWorkspace(*result.bufferized, result.error)) result.bufferized = nullptr;
  return result;
}

bool verifyStrictFusedGemmLLVMV1(const llvm::Module &module, bool address_sanitizer, std::string &error) {
  error.clear();
  auto *leaf = module.getFunction(kStrictFusedGemmSymbolV1), *wrapper = module.getFunction(kStrictFusedGemmCInterfaceV1);
  if (llvm::verifyModule(module) || !module.global_empty() || !module.alias_empty() ||
      module.getTargetTriple().str() != "x86_64-pc-linux-gnu" || !leaf || !wrapper ||
      leaf->isDeclaration() || wrapper->isDeclaration() || leaf->isVarArg() || wrapper->isVarArg() ||
      leaf->getLinkage() != llvm::GlobalValue::ExternalLinkage || wrapper->getLinkage() != llvm::GlobalValue::ExternalLinkage ||
      !leaf->getReturnType()->isVoidTy() || !wrapper->getReturnType()->isVoidTy() ||
      leaf->getCallingConv() != llvm::CallingConv::C || wrapper->getCallingConv() != llvm::CallingConv::C ||
      leaf->arg_size() != 35 || wrapper->arg_size() != 5)
    return fail(error, "private LLVM module/target/five-descriptor ABI changed");
  for (const auto &arg : leaf->args()) {
    unsigned i = arg.getArgNo(); auto *type = arg.getType();
    if ((i % 7 < 2 ? !type->isPointerTy() || type->getPointerAddressSpace() != 0 : !type->isIntegerTy(64)) || leaf->getAttributes().getParamAttrs(i).hasAttributes())
      return fail(error, "leaf descriptor expansion/attributes changed");
  }
  for (const auto &arg : wrapper->args())
    if (!arg.getType()->isPointerTy() || arg.getType()->getPointerAddressSpace() != 0 || wrapper->getAttributes().getParamAttrs(arg.getArgNo()).hasAttributes())
      return fail(error, "wrapper descriptor pointers changed");
  unsigned definitions = 0, declarations = 0, multiplies = 0, adds = 0, floatsLoaded = 0, floatsStored = 0, wrapperCalls = 0, minCalls = 0;
  for (const auto &fn : module) {
    if (fn.isDeclaration()) {
      if (fn.getIntrinsicID() != llvm::Intrinsic::smin || fn.getName() != "llvm.smin.i64") return fail(error, "unknown LLVM declaration");
      ++declarations; continue;
    }
    if (&fn != leaf && &fn != wrapper) return fail(error, "unexpected executable function");
    ++definitions;
    if (fn.hasFnAttribute(llvm::Attribute::SanitizeAddress) != address_sanitizer ||
        fn.getFnAttribute("target-cpu").getValueAsString() != "x86-64" || fn.hasFnAttribute("target-features"))
      return fail(error, "LLVM baseline/instrumentation attributes changed");
    for (const auto &block : fn) for (const auto &inst : block) {
      if (auto *fp = llvm::dyn_cast<llvm::FPMathOperator>(&inst); fp && fp->getFastMathFlags().any()) return fail(error, "LLVM fast-math permission");
      if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
        if (load->isVolatile() || load->isAtomic()) return fail(error, "volatile/atomic load");
        if (load->getType()->isFloatTy()) {
          ++floatsLoaded; auto *base = pointerArgument(load->getPointerOperand());
          if (&fn != leaf || !base || base->getParent() != leaf ||
              (base->getArgNo() != 1 && base->getArgNo() != 8 && base->getArgNo() != 15 && base->getArgNo() != 22 && base->getArgNo() != 29)) return fail(error, "float load escaped declared data pointers");
        }
      }
      if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
        if (store->isVolatile() || store->isAtomic()) return fail(error, "volatile/atomic store");
        if (store->getValueOperand()->getType()->isFloatTy()) {
          ++floatsStored; auto *base = pointerArgument(store->getPointerOperand());
          if (&fn != leaf || !base || base->getParent() != leaf || (base->getArgNo() != 22 && base->getArgNo() != 29)) return fail(error, "float store escaped private output/workspace");
          if (auto *seed = llvm::dyn_cast<llvm::ConstantFP>(store->getValueOperand())) {
            if (!seed->getValueAPF().isZero() || seed->getValueAPF().isNegative()) return fail(error, "LLVM fill seed is not positive zero");
          } else {
            auto *sum = llvm::dyn_cast<llvm::Instruction>(store->getValueOperand());
            auto *acc = sum && sum->getOpcode() == llvm::Instruction::FAdd ? llvm::dyn_cast<llvm::LoadInst>(sum->getOperand(0)) : nullptr;
            if (!acc || pointerArgument(acc->getPointerOperand()) != base) return fail(error, "LLVM result store changed accumulator destination");
          }
        } else {
          auto *allocation = llvm::dyn_cast<llvm::AllocaInst>(store->getPointerOperand());
          if (&fn != leaf || !allocation || !store->getValueOperand()->getType()->isArrayTy()) return fail(error, "nonmetadata store");
        }
      }
      if (auto *allocation = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
        auto *type = llvm::dyn_cast<llvm::ArrayType>(allocation->getAllocatedType());
        auto *count = llvm::dyn_cast<llvm::ConstantInt>(allocation->getArraySize());
        if (&fn != leaf || !type || type->getNumElements() != 2 || !type->getElementType()->isIntegerTy(64) || !count || !count->isOne()) return fail(error, "unexpected stack/tensor allocation");
      }
      if (inst.getOpcode() == llvm::Instruction::FMul) {
        ++multiplies; auto *lhs = llvm::dyn_cast<llvm::LoadInst>(inst.getOperand(0)); auto *rhs = llvm::dyn_cast<llvm::LoadInst>(inst.getOperand(1));
        auto *a = lhs ? pointerArgument(lhs->getPointerOperand()) : nullptr; auto *b = rhs ? pointerArgument(rhs->getPointerOperand()) : nullptr;
        if (!a || !b || (multiplies == 1 ? a->getArgNo() != 1 || b->getArgNo() != 8 : a->getArgNo() != 29 || b->getArgNo() != 15)) return fail(error, "LLVM ordered product inputs changed");
      }
      if (inst.getOpcode() == llvm::Instruction::FAdd) {
        ++adds; auto *acc = llvm::dyn_cast<llvm::LoadInst>(inst.getOperand(0)); auto *base = acc ? pointerArgument(acc->getPointerOperand()) : nullptr;
        auto *mul = llvm::dyn_cast<llvm::Instruction>(inst.getOperand(1));
        if (!base || base->getArgNo() != (adds == 1 ? 29U : 22U) || !mul || mul->getOpcode() != llvm::Instruction::FMul) return fail(error, "LLVM reduction accumulator changed");
      }
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
        auto *callee = call->getCalledFunction();
        if (&fn == wrapper && callee == leaf && call->arg_size() == 35) {
          ++wrapperCalls;
          for (unsigned i = 0; i < 35; ++i) {
            auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(call->getArgOperand(i));
            auto *load = extract ? llvm::dyn_cast<llvm::LoadInst>(extract->getAggregateOperand()) : nullptr;
            llvm::SmallVector<unsigned> indices;
            unsigned f = i % 7; if (f < 3) indices.push_back(f); else { indices.push_back(f < 5 ? 3 : 4); indices.push_back((f - 3) % 2); }
            if (!extract || !load || load->getPointerOperand() != wrapper->getArg(i / 7) || extract->getIndices() != llvm::ArrayRef<unsigned>(indices)) return fail(error, "wrapper descriptor-to-leaf bindings changed");
          }
        } else if (&fn == leaf && callee && callee->getIntrinsicID() == llvm::Intrinsic::smin && call->arg_size() == 2 && llvm::isa<llvm::ConstantInt>(call->getArgOperand(1)) && llvm::cast<llvm::ConstantInt>(call->getArgOperand(1))->equalsInt(4)) ++minCalls;
        else return fail(error, "allocation/copy/provider/unknown LLVM call");
      }
    }
  }
  if (definitions != 2 || declarations != 1 || multiplies != 2 || adds != 2 || floatsLoaded != 6 ||
      floatsStored != 4 || wrapperCalls != 1 || minCalls != 1)
    return fail(error, "exact LLVM arithmetic/storage/call footprint changed");
  // This exact coherent-21.1.8 recipe has a pinned complete preoptimization
  // graph. The independent checks above explain its numerical/ABI boundaries;
  // this fingerprint additionally closes integer IV/bounds, GEP expressions,
  // control flow, metadata and any unexamined opcode/attribute. Ignore only
  // the diagnostic ModuleID comment, which changes when the same issued IR is
  // reparsed. A match is self-consistency, NEVER source/execution authority.
  std::string graph;
  llvm::raw_string_ostream stream(graph);
  module.print(stream, nullptr); stream.flush();
  if (llvm::StringRef(graph).starts_with("; ModuleID = ")) graph.erase(0, graph.find('\n') + 1);
  const auto expected = address_sanitizer ?
      "f59028c30e1908a464095f8767369f63a4068fbea1454e502fab1b78a7fb432e" :
      "5e90b9e7c2724035aae97402455d0bc66ecca70b21f757a53e80102310cabcbe";
  return digest(graph) == expected ? true : fail(error, "pinned complete LLVM schedule graph changed");
}

StrictFusedGemmArtifactV1 issueStrictFusedGemmArtifactV1(mlir::MLIRContext &context, bool address_sanitizer) {
  StrictFusedGemmArtifactV1 result; auto stages = buildStrictFusedGemmStagesV1(context);
  if (!stages) { result.error = stages.error; return result; }
  result.semantic_ir = print(*stages.semantic); result.structured_ir = print(*stages.structured);
  result.scheduled_ir = print(*stages.scheduled); result.bufferized_ir = print(*stages.bufferized); result.transform_ir = transformRecipe.str();
  auto fn = *stages.bufferized->getOps<mlir::func::FuncOp>().begin(); mlir::OpBuilder b(&context);
  fn->setAttr("llvm.emit_c_interface", b.getUnitAttr()); stages.bufferized->getOperation()->setAttr("llvm.target_triple", b.getStringAttr("x86_64-pc-linux-gnu"));
  mlir::PassManager passes(&context);
  passes.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  passes.addPass(mlir::memref::createExpandStridedMetadataPass()); passes.addPass(mlir::createLowerAffinePass());
  passes.addPass(mlir::createSCFToControlFlowPass()); passes.addPass(mlir::createArithToLLVMConversionPass());
  passes.addPass(mlir::createFinalizeMemRefToLLVMConversionPass()); passes.addPass(mlir::createConvertFuncToLLVMPass());
  passes.addPass(mlir::createConvertControlFlowToLLVMPass()); passes.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(passes.run(*stages.bufferized))) { fail(result.error, "fixed upstream LLVM lowering failed"); return result; }
  mlir::registerBuiltinDialectTranslation(context); mlir::registerLLVMDialectTranslation(context); llvm::LLVMContext llvmContext;
  auto lowered = mlir::translateModuleToLLVMIR(*stages.bufferized, llvmContext);
  if (!lowered) { fail(result.error, "LLVM translation failed"); return result; }
  for (auto &function : *lowered) if (!function.isDeclaration()) {
    function.addFnAttr("target-cpu", "x86-64"); if (address_sanitizer) function.addFnAttr(llvm::Attribute::SanitizeAddress);
  }
  if (!verifyStrictFusedGemmLLVMV1(*lowered, address_sanitizer, result.error)) return result;
  llvm::raw_string_ostream output(result.llvm_ir); lowered->print(output, nullptr); output.flush();
  result.manifest = "schema=matcore-builtin-strict-fused-cpu-gemm-v1\nsource_authority=none_builtin_primitive_only\n"
      "toolchain=21.1.8\ntarget=x86_64-pc-linux-gnu\nisa=x86-64-baseline\nprofile=strict_f32_both\n"
      "chain=C=A*B;E=C*D\nreduction_order=increasing_K_then_increasing_N\nintermediate_round=f32\n"
      "schedule=serial-row-panel-4\nC_workspace_elements=min(4,M)*N\nlogical_C_extent_check=M*N_retained_caller_obligation\n"
      "caller_guards=not_discharged\nsource_frontiers=not_implemented_by_leaf\npublication=none\n"
      "leaf_precondition=E_nonempty_after_both_original_guard_frontiers\n"
      "tensor_allocations=0\ncopies=0\ninput_input_alias=permitted\n"
      "writable_storage=private_E_and_workspace_disjoint_from_each_other_and_inputs\n"
      "llvm_calls=smin_i64_and_ciface_to_leaf_only\nbackend_imports=trusted_conforming_memset_possible\n"
      "address_sanitizer=" + std::string(address_sanitizer ? "function_attributes" : "off") + "\n"
      "semantic_sha256=" + digest(result.semantic_ir) + "\nstructured_sha256=" + digest(result.structured_ir) +
      "\ntransform_sha256=" + digest(result.transform_ir) + "\nscheduled_sha256=" + digest(result.scheduled_ir) +
      "\nbufferized_sha256=" + digest(result.bufferized_ir) + "\nllvm_sha256=" + digest(result.llvm_ir) + "\n";
  return result;
}
} // namespace matcore::mdslc::cpu_candidate
