#include "MatcoreCpuGemmCandidate.h"
#include "MatcoreCpuGemmCandidateInternal.h"
#include "MatcoreCpuReassociateGemmCandidate.h"
#include "MatcoreBufferizedGemmHandoff.h"
#include "MatcoreClosedRegion.h"
#include "MatcoreContractionModel.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>

namespace matcore::mdslc::cpu_candidate {
namespace {
constexpr auto contract =
    "matcore.builtin.strict-gemm-f32.v1: A[M,K],B[K,N]->V[M,N];"
    "nonnegative-i64;increasing-k;positive-zero;separate-f32-mul-add;"
    "nearest-even;gradual-underflow;no-cross-op-reassociation;"
    "nan-payload-unspecified;no-finite-assumption;no-publication";
constexpr auto reassociateContract =
    "matcore.builtin.reassociate-gemm-f32.v1: A[M,K],B[K,N]->V[M,N];"
    "nonnegative-i64;positive-zero;per-operation-f32-reassociation-and-fma;"
    "nearest-even;gradual-underflow;no-cross-op-reassociation;"
    "signed-zero-not-ignored;nan-payload-unspecified;no-finite-assumption;"
    "no-publication";

// Change only the independent output traversal. For each (m,n), K remains
// scalar, increasing and complete. No vector contraction, reduction tiling,
// padding, arithmetic reassociation or new input/destination alias fact.
constexpr llvm::StringLiteral rowContiguousTransform = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    %scheduled = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
)mlir";

std::string digest(llvm::StringRef value) {
  const auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(value));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : bytes) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}
std::string print(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream, mlir::OpPrintingFlags().useLocalScope());
  return result;
}
bool fail(std::string &error, llvm::StringRef message) {
  error = "strict CPU GEMM candidate: " + message.str();
  return false;
}

closed_region::Program primitive(bool reassociate = false) {
  namespace cr = closed_region;
  cr::Program program;
  program.source_identity = reassociate ? "matcore-builtin:reassociate-gemm-f32-v1"
                                       : "matcore-builtin:strict-gemm-f32-v1";
  const llvm::StringRef selectedContract = reassociate ? reassociateContract : contract;
  program.source_sha256 = digest(selectedContract);
  program.source_files = {{1, program.source_identity, program.source_sha256,
                           selectedContract.size()}};
  // This is a compiler-owned primitive identity, explicitly not a C++ header.
  program.header_sha256 = digest("no-source-header:compiler-owned-primitive");
  program.compiler_identity = "matcore-cpu-candidate:LLVM-" LLVM_VERSION_STRING;
  cr::Region region;
  region.name = reassociate ? "reassociate_gemm_primitive" : "strict_gemm_primitive";
  region.site = {0, 1, 1, 1, 1};
  region.resources = {{1, "lhs", 0}, {2, "rhs", 1}};
  region.shape_parameters = {{1, "M", 2}, {2, "K", 3}, {3, "N", 4}};
  auto shape = [](cr::Id id) {
    return cr::Dimension{cr::Dimension::Kind::ShapeParameter, 0, id};
  };
  cr::Operation lhs, rhs, gemm;
  lhs.site = {0, 1, 1, 1, 1};
  lhs.result = 1;
  lhs.resource = 1;
  lhs.rows = shape(1);
  lhs.columns = shape(2);
  rhs.site = {1, 1, 1, 2, 1};
  rhs.result = 2;
  rhs.resource = 2;
  rhs.rows = shape(2);
  rhs.columns = shape(3);
  gemm.site = {2, 1, 1, 3, 1};
  gemm.kind = cr::Operation::Kind::Gemm;
  gemm.result = 3;
  gemm.lhs = 1;
  gemm.rhs = 2;
  if (reassociate) gemm.numerical_profile = cr::NumericalProfile::ReassociateF32;
  region.body = {lhs, rhs, gemm};
  program.regions = {region};
  return program;
}

mlir::OwningOpRef<mlir::ModuleOp>
structuredFromPrimitive(mlir::ModuleOp semantic, std::string &error,
                        bool reassociate = false) {
  if (!closed_region::verifyModuleMatchesProgram(primitive(reassociate), semantic, error))
    return {};
  mlir::Operation *gemm = nullptr;
  semantic.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "mdsl_admission.gemm")
      gemm = op;
  });
  if (!gemm || gemm->getNumOperands() != 3 || gemm->getNumResults() != 1 ||
      gemm->getAttrOfType<mlir::StringAttr>("numerical_profile").getValue() !=
          (reassociate ? "reassociate_f32" : "strict_f32")) {
    fail(error, "compiler-owned semantic primitive lost its GEMM profile");
    return {};
  }
  auto *context = semantic.getContext();
  mlir::OpBuilder builder(context);
  auto lhs = gemm->getOperand(1).getType(), rhs = gemm->getOperand(2).getType();
  auto output = gemm->getResult(0).getType();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  auto function = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), reassociate ? kReassociateGemmSymbolV1 : kStrictGemmSymbolV1,
      builder.getFunctionType({lhs, rhs, output}, {output}));
  module->push_back(function);
  auto *block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  auto zero = mlir::arith::ConstantOp::create(builder, function.getLoc(),
                                              builder.getF32FloatAttr(0.0));
  auto fill = mlir::linalg::FillOp::create(
      builder, function.getLoc(), mlir::ValueRange{zero},
      mlir::ValueRange{block->getArgument(2)});
  auto matmul = mlir::linalg::MatmulOp::create(
      builder, function.getLoc(), mlir::TypeRange{output},
      mlir::ValueRange{block->getArgument(0), block->getArgument(1)},
      fill.getResults());
  mlir::func::ReturnOp::create(builder, function.getLoc(), matmul.getResults());
  return module;
}

bool verifyStage(mlir::ModuleOp module, bool buffer, std::string &error,
                 bool rowContiguous = false, bool reassociate = false) {
  error.clear();
  if (!module || mlir::failed(mlir::verify(module)) ||
      !module->getAttrs().empty() ||
      !llvm::hasSingleElement(module.getBody()->getOperations()))
    return fail(error,
                "stage must contain exactly the compiler-owned function");
  bool discardable = false;
  module.walk([&](mlir::Operation *op) {
    discardable |= !op->getDiscardableAttrDictionary().empty();
  });
  if (discardable)
    return fail(
        error,
        "stage contains unsupported extra semantic or authority attributes");
  auto function = mlir::dyn_cast<mlir::func::FuncOp>(module.getBody()->front());
  if (!function || function.getName() !=
          (reassociate ? kReassociateGemmSymbolV1 : kStrictGemmSymbolV1) ||
      function.getNumArguments() != 3 || function.getNumResults() != 1 ||
      function->getAttrs().size() != 2 ||
      !llvm::hasSingleElement(function.getBody()))
    return fail(error,
                "function identity/type/attributes differ from primitive");
  for (auto type : function.getFunctionType().getInputs()) {
    auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
    if (!shaped || !shaped.hasRank() || shaped.getRank() != 2 ||
        !shaped.getElementType().isF32() || !shaped.isDynamicDim(0) ||
        !shaped.isDynamicDim(1))
      return fail(error, "candidate requires dynamic rank-2 f32 operands");
    if (buffer) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
      if (!memref || !memref.getLayout().isIdentity() ||
          memref.getMemorySpace())
        return fail(error,
                    "candidate requires default-space identity-layout memrefs");
    } else {
      auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
      if (!tensor || tensor.getEncoding())
        return fail(error, "structured candidate requires unencoded tensors");
    }
  }
  if (function.getResultTypes()[0] != function.getArgument(2).getType())
    return fail(error, "result lost destination identity type");
  auto &block = function.getBody().front();
  if (block.getOperations().size() != 4)
    return fail(error, "extra computation/allocation/copy in candidate");
  auto iterator = block.begin();
  auto zero = mlir::dyn_cast<mlir::arith::ConstantOp>(&*iterator++);
  auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(&*iterator++);
  auto *contraction = &*iterator++;
  auto matmul = mlir::dyn_cast<mlir::linalg::LinalgOp>(contraction);
  if (rowContiguous ? !mlir::isa<mlir::linalg::GenericOp>(contraction)
                    : !mlir::isa<mlir::linalg::MatmulOp>(contraction))
    return fail(error, "candidate has the wrong scheduled contraction kind");
  auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(&*iterator);
  auto value = zero ? mlir::dyn_cast<mlir::FloatAttr>(zero.getValue())
                    : mlir::FloatAttr{};
  if (!value || !value.getType().isF32() || !value.getValue().isZero() ||
      value.getValue().isNegative() || !fill || !matmul || !ret ||
      fill.getInputs().size() != 1 || fill.getOutputs().size() != 1 ||
      fill.getInputs()[0] != zero ||
      fill.getOutputs()[0] != block.getArgument(2) ||
      matmul.getDpsInputs().size() != 2 || matmul.getDpsInits().size() != 1 ||
      matmul.getDpsInputs()[0] != block.getArgument(0) ||
      matmul.getDpsInputs()[1] != block.getArgument(1))
    return fail(error,
                "candidate changed zero overwrite or ordered lhs/rhs dataflow");
  const auto destination = buffer ? block.getArgument(2) : fill.getResult(0);
  const mlir::Value returned = buffer ? mlir::Value(block.getArgument(2))
                                     : mlir::Value(matmul->getResult(0));
  if (matmul.getDpsInits()[0] != destination || ret.getNumOperands() != 1 ||
      ret.getOperand(0) != returned ||
      fill.getNumResults() != (buffer ? 0U : 1U) ||
      matmul->getNumResults() != (buffer ? 0U : 1U))
    return fail(error, "candidate lost exact original scratch destination");
  if (!buffer &&
      (fill.getResult(0).getType() != function.getResultTypes()[0] ||
       matmul->getResult(0).getType() != function.getResultTypes()[0]))
    return fail(error, "structured intermediate acquired a foreign tensor type/encoding");
  auto topology = mlir_bridge::buildCanonicalContractionTopologyV1(
      *module.getContext(),
      mlir_bridge::StandardLinearAlgebraOperationV1::Gemm);
  llvm::SmallVector<mlir::AffineMap> maps;
  for (auto map : matmul.getIndexingMaps())
    maps.push_back(mlir::cast<mlir::AffineMapAttr>(map).getValue());
  if (rowContiguous) {
    auto m = mlir::getAffineDimExpr(0, module.getContext());
    auto k = mlir::getAffineDimExpr(1, module.getContext());
    auto n = mlir::getAffineDimExpr(2, module.getContext());
    llvm::SmallVector<mlir::AffineMap> expected{
        mlir::AffineMap::get(3, 0, {m, k}, module.getContext()),
        mlir::AffineMap::get(3, 0, {k, n}, module.getContext()),
        mlir::AffineMap::get(3, 0, {m, n}, module.getContext())};
    const llvm::SmallVector<mlir::utils::IteratorType> iterators{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction,
        mlir::utils::IteratorType::parallel};
    if (!buffer || maps != expected ||
        matmul.getIteratorTypesArray() != iterators ||
        matmul->getAttrs().size() != 3)
      return fail(error, "row-contiguous schedule changed M/K/N indexing, "
                         "scalar reduction order or attributes");
  } else if (!topology ||
      !mlir_bridge::verifyStructuredIndexingAgainstContractionTopologyV1(
          topology.topology, maps, matmul.getIteratorTypesArray(), {2, 2, 2},
          error))
    return false;
  if (!rowContiguous) {
    auto named = mlir::cast<mlir::linalg::MatmulOp>(contraction);
    if (named.hasUserDefinedMaps() ||
        named.getCast() != mlir::linalg::TypeFn::cast_signed)
      return fail(error,
                  "candidate changes canonical matmul cast/indexing properties");
  }
  auto &scalar = matmul->getRegion(0).front();
  if (scalar.getOperations().size() != 3)
    return fail(error, "noncanonical scalar contraction");
  auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(scalar.front());
  auto add = mlir::dyn_cast<mlir::arith::AddFOp>(*std::next(scalar.begin()));
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(scalar.back());
  if (!mul || !add || !yield || mul.getLhs() != scalar.getArgument(0) ||
      mul.getRhs() != scalar.getArgument(1) ||
      add.getLhs() != scalar.getArgument(2) ||
      add.getRhs() != mul.getResult() ||
      yield.getOperand(0) != add.getResult() ||
      mul.getFastmath() != mlir::arith::FastMathFlags::none ||
      add.getFastmath() != mlir::arith::FastMathFlags::none)
    return fail(
        error, "strict separate multiply/add or reduction accumulator changed");
  auto &fillBlock = fill.getRegion().front();
  auto fillYield = mlir::dyn_cast<mlir::linalg::YieldOp>(fillBlock.front());
  if (fillBlock.getOperations().size() != 1 || !fillYield ||
      fillYield.getOperand(0) != fillBlock.getArgument(0))
    return fail(error, "fill no longer overwrites with supplied positive zero");
  return true;
}
} // namespace

bool preserveStrictGemmOutputStorageV1(llvm::Module &module, std::string &error) {
  error.clear();
  auto *leaf = module.getFunction(kStrictGemmSymbolV1);
  auto *wrapper = module.getFunction(kStrictGemmCInterfaceV1);
  if (module.size() != 2 || !leaf || !wrapper)
    return fail(error, "private output storage module/symbol boundary changed");
  return detail::preserveGemmOutputData(*leaf, *wrapper, error);
}

bool detail::preserveGemmOutputData(llvm::Function &leafRef,
                                   llvm::Function &wrapperRef,
                                   std::string &error) {
  auto *leaf = &leafRef;
  auto *wrapper = &wrapperRef;
  if (leaf->isDeclaration() ||
      wrapper->isDeclaration() || leaf->isVarArg() || wrapper->isVarArg() ||
      leaf->getCallingConv() != llvm::CallingConv::C ||
      wrapper->getCallingConv() != llvm::CallingConv::C ||
      !leaf->getReturnType()->isVoidTy() || !wrapper->getReturnType()->isVoidTy() ||
      leaf->arg_size() != 21 || wrapper->arg_size() != 3)
    return fail(error, "private output storage signature/ownership boundary changed");
  // Pinned rank-2 descriptor expansion: allocated, aligned, offset,
  // sizes[2], strides[2]. These checks guard the data-pointer binding, not
  // arbitrary LLVM pointer provenance or semantic correctness of an IR body.
  for (unsigned index = 0; index != 21; ++index) {
    auto *type = leaf->getArg(index)->getType();
    if ((index % 7 < 2 ? !type->isPointerTy() ||
                            type->getPointerAddressSpace() != 0
                      : !type->isIntegerTy(64)) ||
        leaf->getAttributes().getParamAttrs(index).hasAttributes())
      return fail(error, "private output storage descriptor mapping/attributes changed");
  }
  for (unsigned index = 0; index != 3; ++index) {
    auto *type = wrapper->getArg(index)->getType();
    if (!type->isPointerTy() || type->getPointerAddressSpace() != 0 ||
        wrapper->getAttributes().getParamAttrs(index).hasAttributes())
      return fail(error, "descriptor-pointer attributes cannot prove data isolation");
  }
  // This is already required by the private leaf's caller contract and
  // established by fresh output allocation while both input values stay alive.
  // Do not mark allocated-C, input data or wrapper descriptor pointers; do not
  // infer nonnull, alignment, dereferenceability or physical source noalias.
  leaf->addParamAttr(15, llvm::Attribute::NoAlias);
  return true;
}

bool verifyStrictGemmStructuredV1(mlir::ModuleOp module, std::string &error) {
  return verifyStage(module, false, error);
}
bool verifyStrictGemmBufferizedV1(mlir::ModuleOp module, std::string &error) {
  return verifyStage(module, true, error);
}

bool verifyStrictGemmRowContiguousV1(mlir::ModuleOp module, std::string &error) {
  return verifyStage(module, true, error, true);
}

mlir::OwningOpRef<mlir::ModuleOp>
deriveStrictGemmRowContiguousV1(mlir::ModuleOp bufferized, std::string &error) {
  if (!verifyStrictGemmBufferizedV1(bufferized, error))
    return {};
  auto *context = bufferized.getContext();
  mlir::DialectRegistry registry;
  mlir::linalg::registerTransformDialectExtension(registry);
  context->appendDialectRegistry(registry);
  context->getOrLoadDialect<mlir::transform::TransformDialect>();
  auto transform = mlir::parseSourceString<mlir::ModuleOp>(
      rowContiguousTransform, mlir::ParserConfig(context));
  if (!transform) {
    fail(error, "cannot parse pinned row-contiguous Transform schedule");
    return {};
  }
  auto entry = transform->lookupSymbol<mlir::transform::NamedSequenceOp>(
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  // Failed transforms may partially mutate payloads: never modify the verified
  // input or reuse a failed candidate. Only this isolated clone can survive.
  mlir::OwningOpRef<mlir::ModuleOp> scheduled = bufferized.clone();
  if (!entry || mlir::failed(mlir::transform::applyTransformNamedSequence(
                    scheduled->getOperation(), entry.getOperation(), *transform,
                    mlir::transform::TransformOptions().enableExpensiveChecks(true))) ||
      !verifyStrictGemmRowContiguousV1(*scheduled, error)) {
    if (error.empty())
      fail(error, "row-contiguous Transform derivation failed");
    return {};
  }
  return scheduled;
}

namespace {
GemmStagesV1 buildGemmStages(mlir::MLIRContext &context, bool reassociate) {
  GemmStagesV1 result;
  if (llvm::StringRef(LLVM_VERSION_STRING) != "21.1.8") {
    fail(result.error, "requires exact LLVM/MLIR 21.1.8");
    return result;
  }
  mlir_bridge::registerBufferizedGemmHandoffDialectsV1(context);
  auto witness = closed_region::buildModule(primitive(reassociate), context);
  if (!witness) {
    result.error = witness.error;
    return result;
  }
  result.semantic = std::move(witness.module);
  result.structured = structuredFromPrimitive(*result.semantic, result.error, reassociate);
  if (!result.structured ||
      !verifyStage(*result.structured, false, result.error, false, reassociate))
    return result;
  result.bufferized = result.structured->clone();
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = true;
  options.copyBeforeWrite = false;
  options.setFunctionBoundaryTypeConversion(
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(
          *result.bufferized, options, state, &statistics)) ||
      statistics.numBufferAlloc || statistics.numBufferDealloc ||
      statistics.numTensorOutOfPlace ||
      !verifyStage(*result.bufferized, true, result.error, false, reassociate)) {
    if (result.error.empty())
      fail(result.error, "One-Shot did not preserve isolated destination "
                         "without tensor allocation/copy");
    result.bufferized = nullptr;
  }
  return result;
}
} // namespace

StrictGemmStagesV1 buildStrictGemmStagesV1(mlir::MLIRContext &context) {
  return buildGemmStages(context, false);
}
GemmStagesV1 buildReassociateGemmStagesV1(mlir::MLIRContext &context) {
  return buildGemmStages(context, true);
}
bool verifyReassociateGemmStructuredV1(mlir::ModuleOp module, std::string &error) {
  return verifyStage(module, false, error, false, true);
}
bool verifyReassociateGemmBufferizedV1(mlir::ModuleOp module, std::string &error) {
  return verifyStage(module, true, error, false, true);
}

StrictGemmArtifactV1 issueStrictGemmArtifactV1(mlir::MLIRContext &context,
    bool address_sanitizer, StrictGemmScheduleV1 schedule, CpuTargetV1 target) {
  StrictGemmArtifactV1 result;
  const char *targetTriple = cpuTargetTripleV1(target);
  if (!targetTriple) {
    fail(result.error, "unknown strict GEMM CPU target");
    return result;
  }
  if (schedule != StrictGemmScheduleV1::ScalarMNK &&
      schedule != StrictGemmScheduleV1::RowContiguousMKN) {
    fail(result.error, "unknown strict GEMM schedule");
    return result;
  }
  auto stages = buildStrictGemmStagesV1(context);
  if (!stages) {
    result.error = stages.error;
    return result;
  }
  result.semantic_ir = print(*stages.semantic);
  result.structured_ir = print(*stages.structured);
  result.bufferized_ir = print(*stages.bufferized);
  if (schedule == StrictGemmScheduleV1::RowContiguousMKN) {
    stages.bufferized = deriveStrictGemmRowContiguousV1(
        *stages.bufferized, result.error);
    if (!stages.bufferized)
      return result;
    result.transform_ir = rowContiguousTransform.str();
  }
  result.scheduled_ir = print(*stages.bufferized);
  auto function = *stages.bufferized->getOps<mlir::func::FuncOp>().begin();
  // The result descriptor is provably exactly argument 2; drop only that
  // redundant return to keep the private C wrapper a void three-pointer ABI.
  mlir::OpBuilder builder(&context);
  auto ret =
      mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
  ret->setOperands({});
  function.setType(builder.getFunctionType(function.getArgumentTypes(), {}));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  stages.bufferized->getOperation()->setAttr(
      "llvm.target_triple", builder.getStringAttr(targetTriple));
  mlir::PassManager passes(&context);
  passes.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToLoopsPass());
  passes.addPass(mlir::createLowerAffinePass());
  passes.addPass(mlir::createSCFToControlFlowPass());
  passes.addPass(mlir::createArithToLLVMConversionPass());
  passes.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  passes.addPass(mlir::createConvertFuncToLLVMPass());
  passes.addPass(mlir::createConvertControlFlowToLLVMPass());
  passes.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(passes.run(*stages.bufferized))) {
    fail(result.error, "upstream scalar lowering failed");
    return result;
  }
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);
  llvm::LLVMContext llvmContext;
  auto lowered = mlir::translateModuleToLLVMIR(*stages.bufferized, llvmContext);
  if (!lowered || llvm::verifyModule(*lowered)) {
    fail(result.error, "LLVM translation/verification failed");
    return result;
  }
  unsigned multiplies = 0, adds = 0, definitions = 0;
  for (auto &fn : *lowered) {
    if (fn.isDeclaration() || (fn.getName() != kStrictGemmSymbolV1 &&
                               fn.getName() != kStrictGemmCInterfaceV1)) {
      fail(result.error, "unexpected declaration or executable symbol");
      return result;
    }
    ++definitions;
    if (address_sanitizer)
      fn.addFnAttr(llvm::Attribute::SanitizeAddress);
    for (auto &block : fn)
      for (auto &instruction : block) {
        if (auto *fp = llvm::dyn_cast<llvm::FPMathOperator>(&instruction);
            fp && fp->getFastMathFlags().any()) {
          fail(result.error, "LLVM arithmetic acquired fast-math permissions");
          return result;
        }
        multiplies += instruction.getOpcode() == llvm::Instruction::FMul;
        adds += instruction.getOpcode() == llvm::Instruction::FAdd;
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
          if (fn.getName() != kStrictGemmCInterfaceV1 ||
              !call->getCalledFunction() ||
              call->getCalledFunction()->getName() != kStrictGemmSymbolV1) {
            fail(result.error,
                 "generated leaf contains allocation/provider/unknown call");
            return result;
          }
        }
      }
  }
  if (definitions != 2 || multiplies != 1 || adds != 1) {
    fail(result.error, "LLVM strict scalar arithmetic footprint changed");
    return result;
  }
  if (!preserveStrictGemmOutputStorageV1(*lowered, result.error) ||
      llvm::verifyModule(*lowered)) {
    if (result.error.empty())
      fail(result.error, "LLVM output-data fact preservation failed verification");
    return result;
  }
  llvm::raw_string_ostream output(result.llvm_ir);
  lowered->print(output, nullptr);
  output.flush();
  result.manifest =
      "schema=matcore-builtin-strict-cpu-gemm-v1\nsource_authority=none_"
      "builtin_primitive_only\n"
      "toolchain=21.1.8\ntarget=" +
      std::string(targetTriple) +
      "\nprofile=strict_f32\n"
      "shape=dynamic_nonnegative_M_N_K\ncaller_guards=retained_not_discharged\n"
      "tensor_allocations=0\ncopies=0\npublication=none\n"
      "output_storage=caller_private_disjoint_from_inputs\n"
      "output_alias_fact=llvm_leaf_aligned_c_only\ninput_input_alias=permitted\n"
      "pipeline=one-shot-bufferize," +
      (schedule == StrictGemmScheduleV1::ScalarMNK ? std::string{} :
          "transform-generalize-interchange-mkn,") +
      "linalg-loops,affine-scf-cf-llvm,llvm-translation\n"
      "address_sanitizer=" +
      std::string(address_sanitizer ? "function_attributes" : "off") +
      "\nschedule=" +
      (schedule == StrictGemmScheduleV1::ScalarMNK ? "scalar-mnk" : "row-contiguous-mkn") +
      "\nsemantic_sha256=" + digest(result.semantic_ir) +
      "\nstructured_sha256=" + digest(result.structured_ir) +
      "\nbufferized_sha256=" + digest(result.bufferized_ir) +
      "\ntransform_sha256=" + digest(result.transform_ir) +
      "\nscheduled_sha256=" + digest(result.scheduled_ir) +
      "\nllvm_sha256=" + digest(result.llvm_ir) + "\n";
  return result;
}
} // namespace matcore::mdslc::cpu_candidate
