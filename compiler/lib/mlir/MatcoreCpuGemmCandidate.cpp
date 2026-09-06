#include "MatcoreCpuGemmCandidate.h"
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
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

bool verifyStage(mlir::ModuleOp module, bool buffer, std::string &error) {
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
  auto matmul = mlir::dyn_cast<mlir::linalg::MatmulOp>(&*iterator++);
  auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(&*iterator);
  auto value = zero ? mlir::dyn_cast<mlir::FloatAttr>(zero.getValue())
                    : mlir::FloatAttr{};
  if (!value || !value.getType().isF32() || !value.getValue().isZero() ||
      value.getValue().isNegative() || !fill || !matmul || !ret ||
      fill.getInputs().size() != 1 || fill.getOutputs().size() != 1 ||
      fill.getInputs()[0] != zero ||
      fill.getOutputs()[0] != block.getArgument(2) ||
      matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      matmul.getInputs()[0] != block.getArgument(0) ||
      matmul.getInputs()[1] != block.getArgument(1))
    return fail(error,
                "candidate changed zero overwrite or ordered lhs/rhs dataflow");
  const auto destination = buffer ? block.getArgument(2) : fill.getResult(0);
  const auto returned = buffer ? block.getArgument(2) : matmul.getResult(0);
  if (matmul.getOutputs()[0] != destination || ret.getNumOperands() != 1 ||
      ret.getOperand(0) != returned ||
      fill.getNumResults() != (buffer ? 0U : 1U) ||
      matmul.getNumResults() != (buffer ? 0U : 1U))
    return fail(error, "candidate lost exact original scratch destination");
  if (!buffer &&
      (fill.getResult(0).getType() != function.getResultTypes()[0] ||
       matmul.getResult(0).getType() != function.getResultTypes()[0]))
    return fail(error, "structured intermediate acquired a foreign tensor type/encoding");
  auto topology = mlir_bridge::buildCanonicalContractionTopologyV1(
      *module.getContext(),
      mlir_bridge::StandardLinearAlgebraOperationV1::Gemm);
  llvm::SmallVector<mlir::AffineMap> maps;
  for (auto map : matmul.getIndexingMaps())
    maps.push_back(mlir::cast<mlir::AffineMapAttr>(map).getValue());
  if (!topology ||
      !mlir_bridge::verifyStructuredIndexingAgainstContractionTopologyV1(
          topology.topology, maps, matmul.getIteratorTypesArray(), {2, 2, 2},
          error))
    return false;
  if (matmul.hasUserDefinedMaps() ||
      matmul.getCast() != mlir::linalg::TypeFn::cast_signed)
    return fail(error,
                "candidate changes canonical matmul cast/indexing properties");
  auto &scalar = matmul.getRegion().front();
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
                                               bool address_sanitizer) {
  StrictGemmArtifactV1 result;
  auto stages = buildStrictGemmStagesV1(context);
  if (!stages) {
    result.error = stages.error;
    return result;
  }
  result.semantic_ir = print(*stages.semantic);
  result.structured_ir = print(*stages.structured);
  result.bufferized_ir = print(*stages.bufferized);
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
      "pipeline=one-shot-bufferize,linalg-loops,affine-scf-cf-llvm,llvm-"
      "translation\n"
      "address_sanitizer=" +
      std::string(address_sanitizer ? "function_attributes" : "off") +
      "\nsemantic_sha256=" + digest(result.semantic_ir) +
      "\nstructured_sha256=" + digest(result.structured_ir) +
      "\nbufferized_sha256=" + digest(result.bufferized_ir) +
      "\nllvm_sha256=" + digest(result.llvm_ir) + "\n";
  return result;
}
} // namespace matcore::mdslc::cpu_candidate
