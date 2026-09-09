#include "MatcoreCpuReassociateGemmCandidate.h"
#include "MatcoreCpuGemmCandidateInternal.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore::mdslc::cpu_candidate {
namespace {
constexpr llvm::StringLiteral scheduleText = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %i, %j, %k = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    %j_full, %j_tail = transform.loop.peel %j : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %i_full, %i_tail = transform.loop.peel %i : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @vectorize(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %new = transform.structured.vectorize_children_and_apply_patterns %functions : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %new {
      transform.apply_patterns.vector.lower_contraction
    } : !transform.any_op
    transform.yield
  }
  transform.named_sequence @hoist(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %done = transform.structured.hoist_redundant_vector_transfers %functions : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
)mlir";

bool fail(std::string &error, llvm::StringRef message) {
  error = "reassociate CPU GEMM candidate: " + message.str();
  return false;
}
std::string print(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream, mlir::OpPrintingFlags().useLocalScope());
  return result;
}
std::string print(llvm::Module &module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream, nullptr);
  return result;
}
std::string digest(llvm::StringRef text) {
  const auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(text));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}
bool indexIs(mlir::Value value, int64_t expected) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>();
  return constant && constant.value() == expected;
}
bool dimensionIs(mlir::Value value, mlir::Value buffer, int64_t dimension) {
  auto dim = value.getDefiningOp<mlir::memref::DimOp>();
  return dim && dim.getSource() == buffer && indexIs(dim.getIndex(), dimension);
}
bool sameIndex(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs) return true;
  auto a = lhs.getDefiningOp<mlir::affine::AffineApplyOp>();
  auto b = rhs.getDefiningOp<mlir::affine::AffineApplyOp>();
  return a && b && a.getAffineMap() == b.getAffineMap() &&
         a.getOperands() == b.getOperands();
}
bool sameIndices(mlir::ValueRange lhs, mlir::ValueRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto [a, b] : llvm::zip(lhs, rhs)) if (!sameIndex(a, b)) return false;
  return true;
}
bool fullBound(mlir::Value value, mlir::Value buffer, int64_t dimension,
               int64_t tile) {
  auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>();
  if (!apply || apply.getNumOperands() != 1 ||
      !dimensionIs(apply.getOperand(0), buffer, dimension)) return false;
  auto symbol = mlir::getAffineSymbolExpr(0, value.getContext());
  auto expected = mlir::AffineMap::get(0, 1, (symbol.floorDiv(tile) * tile));
  return apply.getAffineMap() == expected;
}

// Independent narrow semantic envelope, NOT an arbitrary schedule verifier.
// Exact replay below additionally fixes every scalar-tail range/index/SSA use.
bool registerEnvelope(mlir::ModuleOp module, std::string &error) {
  if (!module || mlir::failed(mlir::verify(module)) || !module->getAttrs().empty() ||
      !llvm::hasSingleElement(module.getBody()->getOperations()))
    return fail(error, "scheduled module ownership changed");
  auto fn = mlir::dyn_cast<mlir::func::FuncOp>(module.getBody()->front());
  if (!fn || fn.getName() != kReassociateGemmSymbolV1 ||
      fn.getNumArguments() != 3 || fn.getNumResults() ||
      !llvm::hasSingleElement(fn.getBody()) || fn->getAttrs().size() != 3 ||
      !fn->getAttrOfType<mlir::UnitAttr>("llvm.emit_c_interface"))
    return fail(error, "scheduled function ABI/identity changed");
  for (auto argument : fn.getArguments()) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
    if (!type || type.getRank() != 2 || !type.getElementType().isF32() ||
        !type.isDynamicDim(0) || !type.isDynamicDim(1) ||
        !type.getLayout().isIdentity() || type.getMemorySpace())
      return fail(error, "scheduled data layout changed");
  }
  const auto a = fn.getArgument(0), b = fn.getArgument(1), c = fn.getArgument(2);
  unsigned outerProducts = 0, multiplies = 0, adds = 0, loads = 0, stores = 0;
  unsigned reads = 0, writes = 0, carried = 0;
  bool invalid = false;
  mlir::scf::ForOp reduction, zeroFill;
  mlir::vector::OuterProductOp outer;
  module.walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, mlir::func::ReturnOp,
          mlir::arith::ConstantOp, mlir::arith::MulFOp, mlir::arith::AddFOp,
          mlir::memref::DimOp, mlir::memref::LoadOp, mlir::memref::StoreOp,
          mlir::affine::AffineApplyOp, mlir::scf::ForOp, mlir::scf::YieldOp,
          mlir::vector::TransferReadOp, mlir::vector::TransferWriteOp,
          mlir::vector::TransposeOp, mlir::vector::ExtractOp,
          mlir::vector::OuterProductOp, mlir::ub::PoisonOp>(op)) invalid = true;
    if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(op)) {
      ++multiplies;
      auto lhs = mul.getLhs().getDefiningOp<mlir::memref::LoadOp>();
      auto rhs = mul.getRhs().getDefiningOp<mlir::memref::LoadOp>();
      invalid |= !mul.getType().isF32() || mul.getFastmath() != mlir::arith::FastMathFlags::none ||
                 !lhs || !rhs || lhs.getMemRef() != a || rhs.getMemRef() != b;
      if (lhs && rhs && lhs.getIndices().size() == 2 && rhs.getIndices().size() == 2) {
        auto k = mlir::dyn_cast<mlir::BlockArgument>(lhs.getIndices()[1]);
        auto loop = k ? mlir::dyn_cast<mlir::scf::ForOp>(k.getOwner()->getParentOp()) : mlir::scf::ForOp{};
        invalid |= !loop || rhs.getIndices()[0] != k || loop.getInductionVar() != k ||
                   !indexIs(loop.getLowerBound(), 0) || !indexIs(loop.getStep(), 1) ||
                   !dimensionIs(loop.getUpperBound(), a, 1);
      } else invalid = true;
    }
    if (auto add = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
      ++adds;
      auto old = add.getLhs().getDefiningOp<mlir::memref::LoadOp>();
      auto product = add.getRhs().getDefiningOp<mlir::arith::MulFOp>();
      invalid |= !add.getType().isF32() || add.getFastmath() != mlir::arith::FastMathFlags::none ||
                 !old || old.getMemRef() != c || !product;
    }
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      ++loads;
      invalid |= load.getMemRef() != a && load.getMemRef() != b && load.getMemRef() != c;
    }
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      ++stores;
      invalid |= store.getMemRef() != c;
      if (auto add = store.getValue().getDefiningOp<mlir::arith::AddFOp>()) {
        auto old = add.getLhs().getDefiningOp<mlir::memref::LoadOp>();
        invalid |= !old || !sameIndices(store.getIndices(), old.getIndices());
      } else {
        auto constant = store.getValue().getDefiningOp<mlir::arith::ConstantOp>();
        auto zero = constant ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue()) : mlir::FloatAttr{};
        auto n = store->getParentOfType<mlir::scf::ForOp>();
        auto m = n ? n->getParentOfType<mlir::scf::ForOp>() : mlir::scf::ForOp{};
        invalid |= !zero || !zero.getValue().isZero() || zero.getValue().isNegative() ||
                   !n || !m;
        if (n && m) {
          invalid |= bool(zeroFill) || m->getBlock() != &fn.front() ||
              !indexIs(m.getLowerBound(), 0) || !indexIs(n.getLowerBound(), 0) ||
              !indexIs(m.getStep(), 1) || !indexIs(n.getStep(), 1) ||
              !dimensionIs(m.getUpperBound(), c, 0) || !dimensionIs(n.getUpperBound(), c, 1) ||
              !sameIndices(store.getIndices(), mlir::ValueRange{m.getInductionVar(), n.getInductionVar()});
          zeroFill = m;
        }
      }
    }
    if (auto read = mlir::dyn_cast<mlir::vector::TransferReadOp>(op)) {
      ++reads;
      invalid |= bool(read.getMask()) || !read.getInBounds() ||
                 !llvm::all_of(read.getInBounds(), [](mlir::Attribute v) {
                   auto bit = mlir::dyn_cast<mlir::BoolAttr>(v); return bit && bit.getValue(); });
    }
    if (auto write = mlir::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
      ++writes;
      invalid |= write.getBase() != c || bool(write.getMask()) || !write.getInBounds() ||
                 !llvm::all_of(write.getInBounds(), [](mlir::Attribute v) {
                   auto bit = mlir::dyn_cast<mlir::BoolAttr>(v); return bit && bit.getValue(); });
    }
    if (auto opOuter = mlir::dyn_cast<mlir::vector::OuterProductOp>(op)) {
      ++outerProducts; outer = opOuter;
      invalid |= outer.getKind() != mlir::vector::CombiningKind::ADD;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op); loop && loop.getNumRegionIterArgs()) {
      ++carried; reduction = loop;
    }
  });
  if (invalid || outerProducts != 1 || multiplies != 3 || adds != 3 ||
      loads != 9 || stores != 4 || reads != 3 || writes != 1 || carried != 1)
    return fail(error, "scheduled memory/arithmetic/serial envelope changed");
  auto n = reduction->getParentOfType<mlir::scf::ForOp>();
  auto m = n ? n->getParentOfType<mlir::scf::ForOp>() : mlir::scf::ForOp{};
  if (!n || !m || m->getBlock() != &fn.front() ||
      !indexIs(m.getLowerBound(), 0) || !indexIs(m.getStep(), 4) ||
      !fullBound(m.getUpperBound(), a, 0, 4) ||
      !indexIs(n.getLowerBound(), 0) || !indexIs(n.getStep(), 8) ||
      !fullBound(n.getUpperBound(), b, 1, 8) ||
      !indexIs(reduction.getLowerBound(), 0) || !indexIs(reduction.getStep(), 1) ||
      !dimensionIs(reduction.getUpperBound(), a, 1) ||
      reduction.getNumRegionIterArgs() != 1 ||
      reduction.getResult(0).getType() != mlir::VectorType::get({4, 8}, mlir::Float32Type::get(module.getContext())) ||
      outer->getBlock() != reduction.getBody() || outer.getAcc() != reduction.getRegionIterArg(0))
    return fail(error, "full-tile guards or increasing-K register carry changed");
  auto inputC = reduction.getInitArgs()[0].getDefiningOp<mlir::vector::TransferReadOp>();
  auto outputC = mlir::dyn_cast<mlir::vector::TransferWriteOp>(reduction->getNextNode());
  auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(reduction.getBody()->back());
  if (!inputC || !outputC || !yield || inputC->getNextNode() != reduction ||
      inputC.getBase() != c || inputC->getBlock() != n.getBody() ||
      outputC.getVector() != reduction.getResult(0) ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != outer.getResult() ||
      !sameIndices(inputC.getIndices(), {m.getInductionVar(), n.getInductionVar()}) ||
      !sameIndices(outputC.getIndices(), inputC.getIndices()))
    return fail(error, "private C transfer/accumulation dataflow changed");
  auto aExtract = outer.getLhs().getDefiningOp<mlir::vector::ExtractOp>();
  auto bExtract = outer.getRhs().getDefiningOp<mlir::vector::ExtractOp>();
  auto transpose = aExtract ? aExtract.getVector().getDefiningOp<mlir::vector::TransposeOp>() : mlir::vector::TransposeOp{};
  auto aRead = transpose ? transpose.getVector().getDefiningOp<mlir::vector::TransferReadOp>() : mlir::vector::TransferReadOp{};
  auto bRead = bExtract ? bExtract.getVector().getDefiningOp<mlir::vector::TransferReadOp>() : mlir::vector::TransferReadOp{};
  if (!aRead || !bRead || aRead.getBase() != a || bRead.getBase() != b ||
      aRead->getBlock() != reduction.getBody() || bRead->getBlock() != reduction.getBody() ||
      !sameIndices(aRead.getIndices(), {m.getInductionVar(), reduction.getInductionVar()}) ||
      !sameIndices(bRead.getIndices(), {reduction.getInductionVar(), n.getInductionVar()}))
    return fail(error, "ordered vector A/B indexing or zero-K input-read boundary changed");
  const auto identity = mlir::AffineMap::getMultiDimIdentityMap(2, module.getContext());
  if (aRead.getVectorType().getShape() != llvm::ArrayRef<int64_t>({4, 1}) ||
      bRead.getVectorType().getShape() != llvm::ArrayRef<int64_t>({1, 8}) ||
      inputC.getPermutationMap() != identity || outputC.getPermutationMap() != identity ||
      aRead.getPermutationMap() != identity || bRead.getPermutationMap() != identity ||
      transpose.getPermutation() != llvm::ArrayRef<int64_t>({1, 0}) ||
      aExtract.getStaticPosition() != llvm::ArrayRef<int64_t>({0}) ||
      bExtract.getStaticPosition() != llvm::ArrayRef<int64_t>({0}))
    return fail(error, "full-tile vector shape/permutation or real K footprint changed");
  // There must be a complete dominating overwrite before the first tile. The
  // exact replay also fixes its top-level position, the peeled remainder ranges,
  // vector shapes/permutations/extractions and all otherwise allowed attributes.
  auto firstLoop = llvm::find_if(fn.front(), [](mlir::Operation &op) {
    return mlir::isa<mlir::scf::ForOp>(op);
  });
  if (!zeroFill || firstLoop == fn.front().end() || &*firstLoop != zeroFill.getOperation() ||
      !zeroFill->isBeforeInBlock(m))
    return fail(error, "destination initialization no longer dominates tiles");
  return true;
}

mlir::OwningOpRef<mlir::ModuleOp> runSchedule(mlir::ModuleOp bufferized,
                                           std::string &error) {
  auto *context = bufferized.getContext();
  mlir::DialectRegistry registry;
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);
  context->appendDialectRegistry(registry);
  context->loadDialect<mlir::affine::AffineDialect, mlir::scf::SCFDialect,
                       mlir::ub::UBDialect, mlir::vector::VectorDialect>();
  context->getOrLoadDialect<mlir::transform::TransformDialect>();
  auto transform = mlir::parseSourceString<mlir::ModuleOp>(scheduleText, context);
  if (!transform) { fail(error, "cannot parse pinned register schedule"); return {}; }
  mlir::OwningOpRef<mlir::ModuleOp> result = bufferized.clone();
  auto fn = *result->getOps<mlir::func::FuncOp>().begin();
  mlir::OpBuilder builder(context);
  mlir::cast<mlir::func::ReturnOp>(fn.front().back())->setOperands({});
  fn.setType(builder.getFunctionType(fn.getArgumentTypes(), {}));
  fn->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto apply = [&](llvm::StringRef name) {
    auto entry = transform->lookupSymbol<mlir::transform::NamedSequenceOp>(name);
    return entry && mlir::succeeded(mlir::transform::applyTransformNamedSequence(
        result->getOperation(), entry.getOperation(), *transform,
        mlir::transform::TransformOptions().enableExpensiveChecks(true)));
  };
  auto canonicalize = [&] {
    mlir::PassManager passes(context);
    passes.addPass(mlir::createCanonicalizerPass());
    return mlir::succeeded(passes.run(*result));
  };
  if (!apply("__transform_main") || !canonicalize() ||
      !apply("vectorize") || !canonicalize()) {
    fail(error, "upstream tile/peel/vectorize failed; isolated payload discarded"); return {};
  }
  mlir::PassManager loops(context);
  loops.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  loops.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  loops.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(loops.run(*result)) || !apply("hoist") ||
      !registerEnvelope(*result, error)) {
    if (error.empty()) fail(error, "upstream serial transfer hoist failed");
    return {};
  }
  return result;
}

std::unique_ptr<llvm::Module> lower(mlir::ModuleOp scheduled,
                                    llvm::LLVMContext &llvmContext,
                                    std::string &error) {
  auto *context = scheduled.getContext();
  mlir::OwningOpRef<mlir::ModuleOp> payload = scheduled.clone();
  payload->getOperation()->setAttr("llvm.target_triple", mlir::StringAttr::get(context, kCpuTargetV1));
  mlir::PassManager passes(context);
  passes.addPass(mlir::createConvertVectorToSCFPass());
  passes.addPass(mlir::memref::createExpandStridedMetadataPass());
  passes.addPass(mlir::createLowerAffinePass());
  passes.addPass(mlir::createSCFToControlFlowPass());
  mlir::ConvertVectorToLLVMPassOptions options;
  options.vectorContractLowering = mlir::vector::VectorContractLowering::OuterProduct;
  passes.addPass(mlir::createConvertVectorToLLVMPass(options));
  passes.addPass(mlir::createArithToLLVMConversionPass());
  passes.addPass(mlir::createUBToLLVMConversionPass());
  passes.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  passes.addPass(mlir::createConvertFuncToLLVMPass());
  passes.addPass(mlir::createConvertControlFlowToLLVMPass());
  passes.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(passes.run(*payload))) { fail(error, "upstream vector lowering failed"); return {}; }
  mlir::registerBuiltinDialectTranslation(*context);
  mlir::registerLLVMDialectTranslation(*context);
  auto result = mlir::translateModuleToLLVMIR(*payload, llvmContext);
  if (!result || llvm::verifyModule(*result)) { fail(error, "LLVM translation failed"); return {}; }
  return result;
}

bool finishLLVM(llvm::Module &module, bool asan, std::string &error) {
  auto *leaf = module.getFunction(kReassociateGemmSymbolV1);
  auto *wrapper = module.getFunction(kReassociateGemmCInterfaceV1);
  if (!leaf || !wrapper || !module.global_empty() || !module.alias_empty() ||
      !module.ifunc_empty() || !module.getModuleInlineAsm().empty() ||
      module.getTargetTriple().str() != kCpuTargetV1)
    return fail(error, "LLVM ownership/target boundary changed");
  unsigned definitions = 0, fmas = 0, multiplies = 0, adds = 0, wrapperCalls = 0;
  for (auto &fn : module) {
    if (fn.isDeclaration()) {
      if (fn.getIntrinsicID() != llvm::Intrinsic::fmuladd)
        return fail(error, "LLVM contains an external or unsupported intrinsic declaration");
      continue;
    }
    if (&fn != leaf && &fn != wrapper) return fail(error, "LLVM acquired another definition");
    ++definitions;
    for (auto &block : fn) for (auto &instruction : block) {
      if (auto *fp = llvm::dyn_cast<llvm::FPMathOperator>(&instruction);
          fp && fp->getFastMathFlags().any()) return fail(error, "LLVM acquired blanket fast math");
      multiplies += instruction.getOpcode() == llvm::Instruction::FMul;
      adds += instruction.getOpcode() == llvm::Instruction::FAdd;
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
        auto *callee = call->getCalledFunction();
        if (&fn == wrapper && callee == leaf) { ++wrapperCalls; continue; }
        if (&fn != leaf || !callee || callee->getIntrinsicID() != llvm::Intrinsic::fmuladd ||
            call->arg_size() != 3 || call->getType() != llvm::FixedVectorType::get(
                llvm::Type::getFloatTy(module.getContext()), 8))
          return fail(error, "LLVM leaf acquired allocation/provider/unknown call");
        ++fmas;
      }
    }
  }
  if (definitions != 2 || fmas != 4 || multiplies != 3 || adds != 3 || wrapperCalls != 1)
    return fail(error, "LLVM register/scalar-tail arithmetic footprint changed");
  if (!detail::preserveGemmOutputData(*leaf, *wrapper, error)) return false;
  for (auto *fn : {leaf, wrapper}) {
    fn->addFnAttr("target-cpu", "x86-64");
    fn->addFnAttr("target-features", "+avx2,+fma");
    if (asan) fn->addFnAttr(llvm::Attribute::SanitizeAddress);
  }
  return !llvm::verifyModule(module);
}
} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
deriveReassociateGemmRegisterV1(mlir::ModuleOp bufferized, std::string &error) {
  error.clear();
  if (!verifyReassociateGemmBufferizedV1(bufferized, error)) return {};
  return runSchedule(bufferized, error);
}
bool verifyReassociateGemmRegisterV1(mlir::ModuleOp module, std::string &error) {
  error.clear();
  if (!registerEnvelope(module, error)) return false;
  auto stages = buildReassociateGemmStagesV1(*module.getContext());
  if (!stages) { error = stages.error; return false; }
  auto expected = runSchedule(*stages.bufferized, error);
  if (!expected) return false;
  if (!mlir::OperationEquivalence::isEquivalentTo(module, *expected,
          mlir::OperationEquivalence::IgnoreLocations))
    return fail(error, "scheduled payload differs from exact trusted upstream replay");
  return true;
}
bool verifyReassociateGemmLLVMV1(llvm::Module &module, std::string &error) {
  error.clear();
  if (llvm::verifyModule(module)) return fail(error, "invalid LLVM module");
  auto *leaf = module.getFunction(kReassociateGemmSymbolV1);
  if (!leaf) return fail(error, "missing reassociate leaf");
  const bool asan = leaf->hasFnAttribute(llvm::Attribute::SanitizeAddress);
  mlir::MLIRContext context;
  auto stages = buildReassociateGemmStagesV1(context);
  if (!stages) { error = stages.error; return false; }
  auto scheduled = runSchedule(*stages.bufferized, error);
  if (!scheduled) return false;
  llvm::LLVMContext llvmContext;
  auto expected = lower(*scheduled, llvmContext, error);
  if (!expected || !finishLLVM(*expected, asan, error)) return false;
  // The parser may choose a diagnostic module identifier. Unlike source_file,
  // symbols, target, data layout and metadata, that comment is not ABI/semantics.
  expected->setModuleIdentifier(module.getModuleIdentifier());
  if (print(module) != print(*expected))
    return fail(error, "LLVM differs from exact compiler-owned lowering and ABI/target facts");
  return true;
}
GemmArtifactV1 issueReassociateGemmArtifactV1(mlir::MLIRContext &context, bool asan) {
  GemmArtifactV1 result;
  auto stages = buildReassociateGemmStagesV1(context);
  if (!stages) { result.error = stages.error; return result; }
  result.semantic_ir = print(*stages.semantic);
  result.structured_ir = print(*stages.structured);
  result.bufferized_ir = print(*stages.bufferized);
  auto scheduled = deriveReassociateGemmRegisterV1(*stages.bufferized, result.error);
  if (!scheduled || !verifyReassociateGemmRegisterV1(*scheduled, result.error)) return result;
  result.transform_ir = scheduleText.str();
  result.scheduled_ir = print(*scheduled);
  llvm::LLVMContext llvmContext;
  auto module = lower(*scheduled, llvmContext, result.error);
  if (!module || !finishLLVM(*module, asan, result.error) ||
      !verifyReassociateGemmLLVMV1(*module, result.error)) return result;
  result.llvm_ir = print(*module);
  result.manifest =
      "schema=matcore-builtin-reassociate-cpu-gemm-v1\n"
      "source_authority=none_builtin_primitive_only\n"
      "toolchain=21.1.8\ntarget=x86_64-pc-linux-gnu\n"
      "target_cpu=x86-64\ntarget_features=+avx2,+fma\n"
      "runtime_capability_guard=AVX2_FMA_and_OS_XMM_YMM_retained\n"
      "profile=reassociate_f32\nfull_tiles=increasing_k_fmuladd\n"
      "scalar_tails=increasing_k_separate_f32_mul_add\n"
      "cross_operation_reassociation=forbidden\nfast_math_flags=none\n"
      "shape=dynamic_nonnegative_M_N_K\ncaller_guards=retained_not_discharged\n"
      "tensor_allocations=0\ncopies=0\npublication=none\n"
      "output_storage=caller_private_disjoint_from_inputs\n"
      "output_alias_fact=llvm_leaf_aligned_c_only\ninput_input_alias=permitted\n"
      "schedule=serial_peeled_4x8x1_register_carry\n"
      "derivation=trusted_pinned_upstream_replay_with_narrow_independent_envelope\n"
      "hoist_preconditions=serial_full_tiles_after_fill_inputs_inside_K\n"
      "address_sanitizer=" + std::string(asan ? "function_attributes" : "off") +
      "\nsemantic_sha256=" + digest(result.semantic_ir) +
      "\nstructured_sha256=" + digest(result.structured_ir) +
      "\nbufferized_sha256=" + digest(result.bufferized_ir) +
      "\ntransform_sha256=" + digest(result.transform_ir) +
      "\nscheduled_sha256=" + digest(result.scheduled_ir) +
      "\nllvm_sha256=" + digest(result.llvm_ir) + "\n";
  return result;
}
} // namespace matcore::mdslc::cpu_candidate
