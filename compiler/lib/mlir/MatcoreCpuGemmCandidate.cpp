#include "MatcoreCpuGemmCandidate.h"
#include "MatcoreBufferizedGemmHandoff.h"
#include "MatcoreClosedRegion.h"
#include "MatcoreContractionModel.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
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

// Tile boundaries are HOW, not mathematical metadata. For each fixed (m,n),
// increasing outer K chunks and increasing inner K form the original ordered
// fold, starting from the existing C accumulator. No partial sum is zeroed.
constexpr llvm::StringLiteral cacheTiledTransform = R"mlir(
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %m, %n, %k = transform.structured.tile_using_for %matmul tile_sizes [4, 64, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %generic = transform.structured.generalize %tiled : (!transform.any_op) -> !transform.any_op
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

// This is compiler pass composition, not source admission or a proof issuer.
// Failed/partially applied transforms are discarded with their isolated clone.
mlir::OwningOpRef<mlir::ModuleOp>
applyPinnedTransform(mlir::ModuleOp bufferized, llvm::StringRef source,
                     std::string &error) {
  auto *context = bufferized.getContext();
  mlir::DialectRegistry registry;
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  context->appendDialectRegistry(registry);
  context->getOrLoadDialect<mlir::transform::TransformDialect>();
  auto transform = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(context));
  if (!transform) {
    fail(error, "cannot parse pinned Transform schedule");
    return {};
  }
  auto entry = transform->lookupSymbol<mlir::transform::NamedSequenceOp>(
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  mlir::OwningOpRef<mlir::ModuleOp> scheduled = bufferized.clone();
  if (!entry || mlir::failed(mlir::transform::applyTransformNamedSequence(
                    scheduled->getOperation(), entry.getOperation(), *transform,
                    mlir::transform::TransformOptions().enableExpensiveChecks(true))) ||
      mlir::failed(mlir::verify(*scheduled))) {
    fail(error, "pinned Transform derivation failed");
    return {};
  }
  return scheduled;
}

closed_region::Program primitive() {
  namespace cr = closed_region;
  cr::Program program;
  program.source_identity = "matcore-builtin:strict-gemm-f32-v1";
  program.source_sha256 = digest(contract);
  // This is a compiler-owned primitive identity, explicitly not a C++ header.
  program.header_sha256 = digest("no-source-header:compiler-owned-primitive");
  program.compiler_identity = "matcore-cpu-candidate:LLVM-" LLVM_VERSION_STRING;
  cr::Region region;
  region.name = "strict_gemm_primitive";
  region.site = {0, 1, 1, 1};
  region.resources = {{1, "lhs", 0}, {2, "rhs", 1}};
  region.shape_parameters = {{1, "M", 2}, {2, "K", 3}, {3, "N", 4}};
  auto shape = [](cr::Id id) {
    return cr::Dimension{cr::Dimension::Kind::ShapeParameter, 0, id};
  };
  cr::Operation lhs, rhs, gemm;
  lhs.site = {0, 1, 1, 1};
  lhs.result = 1;
  lhs.resource = 1;
  lhs.rows = shape(1);
  lhs.columns = shape(2);
  rhs.site = {1, 1, 1, 2};
  rhs.result = 2;
  rhs.resource = 2;
  rhs.rows = shape(2);
  rhs.columns = shape(3);
  gemm.site = {2, 1, 1, 3};
  gemm.kind = cr::Operation::Kind::Gemm;
  gemm.result = 3;
  gemm.lhs = 1;
  gemm.rhs = 2;
  region.body = {lhs, rhs, gemm};
  program.regions = {region};
  return program;
}

mlir::OwningOpRef<mlir::ModuleOp>
structuredFromPrimitive(mlir::ModuleOp semantic, std::string &error) {
  if (!closed_region::verifyModuleMatchesProgram(primitive(), semantic, error))
    return {};
  mlir::Operation *gemm = nullptr;
  semantic.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "mdsl_admission.gemm")
      gemm = op;
  });
  if (!gemm || gemm->getNumOperands() != 3 || gemm->getNumResults() != 1 ||
      gemm->getAttrOfType<mlir::StringAttr>("numerical_profile").getValue() !=
          "strict_f32") {
    fail(error, "compiler-owned semantic primitive lost strict GEMM");
    return {};
  }
  auto *context = semantic.getContext();
  mlir::OpBuilder builder(context);
  auto lhs = gemm->getOperand(1).getType(), rhs = gemm->getOperand(2).getType();
  auto output = gemm->getResult(0).getType();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  auto function = mlir::func::FuncOp::create(
      builder.getUnknownLoc(), kStrictGemmSymbolV1,
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
                 bool rowContiguous = false) {
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
  if (!function || function.getName() != kStrictGemmSymbolV1 ||
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
  auto scheduled = applyPinnedTransform(bufferized, rowContiguousTransform, error);
  if (!scheduled || !verifyStrictGemmRowContiguousV1(*scheduled, error))
    return {};
  return scheduled;
}

bool verifyStrictGemmCacheTiledV1(mlir::ModuleOp module, std::string &error) {
  error.clear();
  if (!module || mlir::failed(mlir::verify(module)))
    return fail(error, "invalid cache-tiled module");
  // Independent narrow envelope: no allocation, copy, call, parallel reduction,
  // extra arithmetic or hidden effect. Exact scalar/body/index relationships
  // are checked below by replay, using upstream's SSA-aware equivalence.
  unsigned fills = 0, generics = 0, loops = 0, views = 0, muls = 0, adds = 0;
  bool invalid = false;
  module.walk([&](mlir::Operation *op) {
    invalid |= !op->getDiscardableAttrDictionary().empty();
    invalid |= !mlir::isa<mlir::ModuleOp, mlir::func::FuncOp,
        mlir::func::ReturnOp, mlir::arith::ConstantOp, mlir::memref::DimOp,
        mlir::affine::AffineMinOp, mlir::affine::AffineApplyOp,
        mlir::scf::ForOp, mlir::scf::YieldOp, mlir::memref::SubViewOp,
        mlir::linalg::FillOp, mlir::linalg::GenericOp, mlir::linalg::YieldOp,
        mlir::arith::MulFOp, mlir::arith::AddFOp>(op);
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op)) {
      ++fills;
      invalid |= !mlir::isa<mlir::func::FuncOp>(fill->getParentOp());
    }
    generics += mlir::isa<mlir::linalg::GenericOp>(op);
    loops += mlir::isa<mlir::scf::ForOp>(op);
    views += mlir::isa<mlir::memref::SubViewOp>(op);
    if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(op)) {
      ++muls;
      invalid |= !mul.getType().isF32() ||
                 mul.getFastmath() != mlir::arith::FastMathFlags::none;
    }
    if (auto add = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
      ++adds;
      invalid |= !add.getType().isF32() ||
                 add.getFastmath() != mlir::arith::FastMathFlags::none;
    }
  });
  if (invalid || fills != 1 || generics != 1 || loops != 3 || views != 3 ||
      muls != 1 || adds != 1)
    return fail(error, "cache-tiled operation/effect/numerical envelope changed");

  // Trust the pinned upstream tiler under the exact original preconditions,
  // rather than implementing an overlapping affine-loop theorem checker.
  // Replay binds all shapes, bounds/steps, tail minima, subview sources/offsets,
  // iteration maps, scalar SSA, zeroing position and returned destination.
  // It is drift detection, not independent validation of upstream semantics.
  auto canonical = buildStrictGemmStagesV1(*module.getContext());
  if (!canonical) {
    error = canonical.error;
    return false;
  }
  auto expected = applyPinnedTransform(*canonical.bufferized,
                                       cacheTiledTransform, error);
  if (!expected || !mlir::OperationEquivalence::isEquivalentTo(
      module, *expected, mlir::OperationEquivalence::IgnoreLocations))
    return fail(error, "cache-tiled payload differs from pinned structural derivation");
  return true;
}

mlir::OwningOpRef<mlir::ModuleOp>
deriveStrictGemmCacheTiledV1(mlir::ModuleOp bufferized, std::string &error) {
  if (!verifyStrictGemmBufferizedV1(bufferized, error))
    return {};
  auto scheduled = applyPinnedTransform(bufferized, cacheTiledTransform, error);
  if (!scheduled || !verifyStrictGemmCacheTiledV1(*scheduled, error))
    return {};
  return scheduled;
}

StrictGemmStagesV1 buildStrictGemmStagesV1(mlir::MLIRContext &context) {
  StrictGemmStagesV1 result;
  if (llvm::StringRef(LLVM_VERSION_STRING) != "21.1.8") {
    fail(result.error, "requires exact LLVM/MLIR 21.1.8");
    return result;
  }
  mlir_bridge::registerBufferizedGemmHandoffDialectsV1(context);
  auto witness = closed_region::buildModule(primitive(), context);
  if (!witness) {
    result.error = witness.error;
    return result;
  }
  result.semantic = std::move(witness.module);
  result.structured = structuredFromPrimitive(*result.semantic, result.error);
  if (!result.structured ||
      !verifyStrictGemmStructuredV1(*result.structured, result.error))
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
      !verifyStrictGemmBufferizedV1(*result.bufferized, result.error)) {
    if (result.error.empty())
      fail(result.error, "One-Shot did not preserve isolated destination "
                         "without tensor allocation/copy");
    result.bufferized = nullptr;
  }
  return result;
}

StrictGemmArtifactV1 issueStrictGemmArtifactV1(mlir::MLIRContext &context,
    bool address_sanitizer, StrictGemmScheduleV1 schedule) {
  StrictGemmArtifactV1 result;
  if (schedule != StrictGemmScheduleV1::ScalarMNK &&
      schedule != StrictGemmScheduleV1::RowContiguousMKN &&
      schedule != StrictGemmScheduleV1::CacheTiledMKN) {
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
  } else if (schedule == StrictGemmScheduleV1::CacheTiledMKN) {
    stages.bufferized = deriveStrictGemmCacheTiledV1(
        *stages.bufferized, result.error);
    if (!stages.bufferized)
      return result;
    result.transform_ir = cacheTiledTransform.str();
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
      "llvm.target_triple", builder.getStringAttr(kCpuTargetV1));
  mlir::PassManager passes(&context);
  passes.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToLoopsPass());
  if (schedule == StrictGemmScheduleV1::CacheTiledMKN)
    passes.addPass(mlir::memref::createExpandStridedMetadataPass());
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
  unsigned multiplies = 0, adds = 0, definitions = 0, tailMinima = 0;
  auto isTailMinimum = [&](const llvm::Function &fn) {
    return schedule == StrictGemmScheduleV1::CacheTiledMKN &&
           fn.isDeclaration() && fn.getIntrinsicID() == llvm::Intrinsic::smin &&
           fn.getName() == "llvm.smin.i64" && fn.arg_size() == 2 &&
           !fn.isVarArg() && fn.getReturnType()->isIntegerTy(64) &&
           fn.getArg(0)->getType()->isIntegerTy(64) &&
           fn.getArg(1)->getType()->isIntegerTy(64);
  };
  for (auto &fn : *lowered) {
    // LowerAffine encodes each dynamic tile tail as the standard signed integer
    // minimum intrinsic. This is LLVM machinery, not a fallible external call.
    if (isTailMinimum(fn))
      continue;
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
          if (fn.getName() == kStrictGemmSymbolV1 &&
              call->getCalledFunction() &&
              isTailMinimum(*call->getCalledFunction())) {
            ++tailMinima;
            continue;
          }
          if (fn.getName() != kStrictGemmCInterfaceV1 ||
              !call->getCalledFunction() ||
              call->getCalledFunction()->getName() != kStrictGemmSymbolV1) {
            fail(result.error,
                 "generated leaf contains allocation/provider/unknown call: " +
                 (call->getCalledFunction() ? call->getCalledFunction()->getName().str()
                                            : std::string("indirect")));
            return result;
          }
        }
      }
  }
  if (definitions != 2 || multiplies != 1 || adds != 1 ||
      tailMinima != (schedule == StrictGemmScheduleV1::CacheTiledMKN ? 3U : 0U)) {
    fail(result.error, "LLVM strict scalar arithmetic footprint changed");
    return result;
  }
  llvm::raw_string_ostream output(result.llvm_ir);
  lowered->print(output, nullptr);
  output.flush();
  result.manifest =
      "schema=matcore-builtin-strict-cpu-gemm-v1\nsource_authority=none_"
      "builtin_primitive_only\n"
      "toolchain=21.1.8\ntarget=" +
      std::string(kCpuTargetV1) +
      "\nprofile=strict_f32\n"
      "shape=dynamic_nonnegative_M_N_K\ncaller_guards=retained_not_discharged\n"
      "tensor_allocations=0\ncopies=0\npublication=none\n"
      "pipeline=one-shot-bufferize," +
      (schedule == StrictGemmScheduleV1::ScalarMNK ? std::string{} :
       schedule == StrictGemmScheduleV1::RowContiguousMKN ?
          "transform-generalize-interchange-mkn," :
          "transform-tile-4x64x32-generalize-interchange-mkn,") +
      "linalg-loops," +
      (schedule == StrictGemmScheduleV1::CacheTiledMKN ?
          "expand-strided-metadata," : "") +
      "affine-scf-cf-llvm,llvm-translation\n"
      "address_sanitizer=" +
      std::string(address_sanitizer ? "function_attributes" : "off") +
      "\nschedule=" +
      (schedule == StrictGemmScheduleV1::ScalarMNK ? "scalar-mnk" :
       schedule == StrictGemmScheduleV1::RowContiguousMKN ? "row-contiguous-mkn" :
          "cache-tiled-4x64x32-mkn") +
      "\nsemantic_sha256=" + digest(result.semantic_ir) +
      "\nstructured_sha256=" + digest(result.structured_ir) +
      "\nbufferized_sha256=" + digest(result.bufferized_ir) +
      "\ntransform_sha256=" + digest(result.transform_ir) +
      "\nscheduled_sha256=" + digest(result.scheduled_ir) +
      "\nllvm_sha256=" + digest(result.llvm_ir) + "\n";
  return result;
}
} // namespace matcore::mdslc::cpu_candidate
