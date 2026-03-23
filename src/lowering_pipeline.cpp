#include "matcore/lowering_pipeline.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore lowering pipeline: " + message);
}

std::int64_t pickTilingFactor(std::int64_t dim, std::int64_t preferred) {
  if (preferred <= 0) {
    return 1;
  }
  if (dim <= 0) {
    return preferred;
  }

  std::int64_t tile = std::min(dim, preferred);
  while (tile > 1 && (dim % tile) != 0) {
    tile /= 2;
  }
  return std::max<std::int64_t>(tile, 1);
}

llvm::SmallVector<std::int64_t, 3> selectCpuMatmulTileSizes(
    mlir::linalg::MatmulOp matmul) {
  auto lhs_type = llvm::dyn_cast<mlir::ShapedType>(matmul.getInputs()[0].getType());
  auto rhs_type = llvm::dyn_cast<mlir::ShapedType>(matmul.getInputs()[1].getType());

  const bool low_precision = lhs_type && lhs_type.getElementType().isIntOrFloat() &&
                             (lhs_type.getElementType().isF16() ||
                              lhs_type.getElementType().isBF16());

  const std::int64_t preferred_m = low_precision ? 8 : 8;
  const std::int64_t preferred_n = low_precision ? 8 : 8;
  const std::int64_t preferred_k = low_precision ? 16 : 8;

  std::int64_t m = mlir::ShapedType::kDynamic;
  std::int64_t n = mlir::ShapedType::kDynamic;
  std::int64_t k = mlir::ShapedType::kDynamic;
  if (lhs_type && lhs_type.hasRank() && lhs_type.getRank() == 2) {
    m = lhs_type.getDimSize(0);
    k = lhs_type.getDimSize(1);
  }
  if (rhs_type && rhs_type.hasRank() && rhs_type.getRank() == 2) {
    n = rhs_type.getDimSize(1);
  }

  return {pickTilingFactor(m, preferred_m), pickTilingFactor(n, preferred_n),
          pickTilingFactor(k, preferred_k)};
}

struct TileAndVectorizeMatmulPass
    : public mlir::PassWrapper<TileAndVectorizeMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    std::vector<mlir::Operation *> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op.getOperation()); });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::Operation *matmul : matmuls) {
      auto typed_matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(matmul);
      if (!typed_matmul) {
        matmul->emitError("MatCore expected linalg.matmul while tiling");
        signalPassFailure();
        return;
      }

      mlir::linalg::LinalgTilingOptions tiling_options;
      tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
      tiling_options.setTileSizes(selectCpuMatmulTileSizes(typed_matmul));

      rewriter.setInsertionPoint(matmul);
      mlir::FailureOr<mlir::linalg::TiledLinalgOp> tiled =
          mlir::linalg::tileLinalgOp(rewriter, typed_matmul, tiling_options);
      if (mlir::failed(tiled)) {
        matmul->emitError("MatCore failed to tile linalg.matmul for x86 lowering");
        signalPassFailure();
        return;
      }

      mlir::Operation *vectorize_target = tiled->op.getOperation();
      if (mlir::failed(mlir::linalg::vectorizeOpPrecondition(vectorize_target))) {
        vectorize_target->emitError(
            "MatCore x86 lowering requires vectorizable tiled linalg.matmul");
        signalPassFailure();
        return;
      }
      if (mlir::failed(mlir::linalg::vectorize(rewriter, vectorize_target))) {
        vectorize_target->emitError("MatCore failed to vectorize tiled linalg.matmul");
        signalPassFailure();
        return;
      }
      if (typed_matmul.getOperation() != vectorize_target &&
          typed_matmul.getOperation()->getBlock() != nullptr) {
        rewriter.eraseOp(typed_matmul.getOperation());
      }
      if (vectorize_target->getBlock() != nullptr) {
        rewriter.eraseOp(vectorize_target);
      }
    }
  }
};

struct LowerResidualVectorOpsPass
    : public mlir::PassWrapper<LowerResidualVectorOpsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns, mlir::vector::VectorMultiReductionLowering::InnerReduction);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("MatCore failed to lower residual vector ops");
      signalPassFailure();
    }
  }
};

TensorDType decodeTensorDType(mlir::Type type) {
  if (type.isF32()) {
    return TensorDType::kFloat32;
  }
  if (type.isF16()) {
    return TensorDType::kFloat16;
  }
  if (type.isBF16()) {
    return TensorDType::kBFloat16;
  }
  if (type.isInteger(8)) {
    return TensorDType::kInt8;
  }
  if (type.isInteger(32)) {
    return TensorDType::kInt32;
  }
  if (type.isFloat8E4M3FN()) {
    return TensorDType::kFloat8E4M3FN;
  }
  fail("module carries unsupported dtype attribute");
}

LoweringRoute selectRoute(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
      return LoweringRoute::kCpuVector;
    case TargetKind::kNvidiaDGPU:
      return LoweringRoute::kNvidiaNvptx;
    case TargetKind::kAmdIGPU:
      return LoweringRoute::kAmdRocdl;
    case TargetKind::kAmdNPU:
      return LoweringRoute::kAmdNpuScaffold;
    case TargetKind::kARM:
      fail("ARM route exists but is not implemented in Phase 2");
    case TargetKind::kTPU:
      fail("TPU route exists but is not implemented in Phase 2");
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }
  fail("unsupported target route");
}

void addCommonLLVMLoweringPasses(mlir::PassManager &pm, bool enable_x86vector) {
  mlir::ConvertVectorToLLVMPassOptions vector_to_llvm_opts;
  vector_to_llvm_opts.x86Vector = enable_x86vector;

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass(vector_to_llvm_opts));
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void addLinalgToGpuLaunchPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
}

void addGpuCommonModulePasses(mlir::PassManager &pm, std::int64_t index_bitwidth) {
  mlir::ConvertIndexToLLVMPassOptions index_to_llvm_opts;
  index_to_llvm_opts.indexBitwidth = index_bitwidth;

  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass(index_to_llvm_opts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void addGpuHostPostPasses(mlir::PassManager &pm, const std::string &binary_target,
                          const std::string &toolkit_path = {}) {
  mlir::GpuToLLVMConversionPassOptions gpu_to_llvm_opts;
  gpu_to_llvm_opts.hostBarePtrCallConv = false;
  gpu_to_llvm_opts.kernelBarePtrCallConv = false;
  pm.addPass(mlir::createGpuToLLVMConversionPass(gpu_to_llvm_opts));

  mlir::GpuModuleToBinaryPassOptions binary_opts;
  binary_opts.compilationTarget = binary_target;
  binary_opts.toolkitPath = toolkit_path;
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary_opts));

  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool useCpuVectorPipeline(const MatmulLoweringSignature &signature) {
  if (signature.quantized_i8) {
    return false;
  }
  if (signature.lhs_dtype == TensorDType::kBFloat16 ||
      signature.lhs_dtype == TensorDType::kFloat8E4M3FN) {
    return false;
  }
  return signature.lhs_dtype == signature.out_dtype &&
         (signature.lhs_dtype == TensorDType::kFloat32 ||
          signature.lhs_dtype == TensorDType::kFloat16);
}

void configureCpuPassPipeline(mlir::PassManager &pm,
                              const MatmulLoweringSignature &signature) {
  if (useCpuVectorPipeline(signature)) {
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<TileAndVectorizeMatmulPass>());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::memref::createExpandOpsPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<LowerResidualVectorOpsPass>());
    mlir::VectorTransferToSCFOptions vector_to_scf_options;
    vector_to_scf_options.setTargetRank(1);
    pm.addPass(mlir::createConvertVectorToSCFPass(vector_to_scf_options));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/true);
    return;
  }

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/false);
}

void configureNvidiaPassPipeline(mlir::PassManager &pm, llvm::StringRef cubin_chip) {
  addLinalgToGpuLaunchPasses(pm);

  mlir::gpu::GPUToNVVMPipelineOptions nvvm_opts;
  nvvm_opts.indexBitWidth = 64;
  nvvm_opts.cubinTriple = "nvptx64-nvidia-cuda";
  nvvm_opts.cubinChip = cubin_chip.str();
  nvvm_opts.cubinFormat = "fatbin";
  nvvm_opts.optLevel = 2;
  nvvm_opts.kernelUseBarePtrCallConv = false;
  nvvm_opts.hostUseBarePtrCallConv = false;
  mlir::gpu::buildLowerToNVVMPassPipeline(pm, nvvm_opts);
}

void configureAmdPassPipeline(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);
  addGpuCommonModulePasses(pm, /*index_bitwidth=*/64);

  mlir::GpuROCDLAttachTargetOptions rocdl_target_opts;
  rocdl_target_opts.triple = "amdgcn-amd-amdhsa";
  rocdl_target_opts.chip = "gfx1150";
  pm.addPass(mlir::createGpuROCDLAttachTarget(rocdl_target_opts));

  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/false));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createLowerGpuOpsToROCDLOpsPass(
          rocdl_target_opts.chip, /*indexBitwidth=*/64,
          /*useBarePtrCallConv=*/false, mlir::gpu::amd::Runtime::HIP));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());
  // Ubuntu packages ROCm device bitcode under the LLVM 17 Clang resource dir.
  addGpuHostPostPasses(pm, "bin", "/usr/lib/llvm-17/lib/clang/17");
}

void validateFp8Support(const LoweringPlan &plan,
                        const MatmulLoweringSignature &signature) {
  if (signature.lhs_dtype != TensorDType::kFloat8E4M3FN &&
      signature.rhs_dtype != TensorDType::kFloat8E4M3FN) {
    return;
  }
  if (plan.route != LoweringRoute::kNvidiaNvptx) {
    fail("float8_e4m3fn matmul is currently limited to nvidia-dgpu");
  }
  fail("float8_e4m3fn matmul requires a dedicated native NVIDIA FP8 WGMMA "
       "lowering path, and MatCore does not implement that path yet");
}

}  // namespace

LoweringPlan selectLoweringPlan(TargetKind target) {
  LoweringPlan plan;
  plan.route = selectRoute(target);
  plan.route_name = routeName(plan.route);
  plan.route_description = routeDescription(plan.route);
  plan.executable = plan.route != LoweringRoute::kAmdNpuScaffold;
  return plan;
}

const char *routeName(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      return "x86-vector";
    case LoweringRoute::kNvidiaNvptx:
      return "nvidia-dgpu";
    case LoweringRoute::kAmdRocdl:
      return "amd-igpu";
    case LoweringRoute::kAmdNpuScaffold:
      return "amd-npu";
  }
  return "unknown";
}

std::string routeDescription(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      return "func+memref+linalg.matmul -> loops/vector -> llvm(x86vector)";
    case LoweringRoute::kNvidiaNvptx:
      return "linalg -> scf.parallel -> gpu.launch -> nvgpu/nvvm -> llvm";
    case LoweringRoute::kAmdRocdl:
      return "linalg -> scf.parallel -> gpu.launch -> rocdl -> llvm";
    case LoweringRoute::kAmdNpuScaffold:
      return "aie/xdna scaffold route (external toolchain required)";
  }
  return "unknown";
}

MatmulLoweringSignature decodeMatmulSignatureFromModule(mlir::ModuleOp module) {
  MatmulLoweringSignature signature;
  auto lhs_attr = module->getAttrOfType<mlir::TypeAttr>("matcore.lhs_dtype");
  auto rhs_attr = module->getAttrOfType<mlir::TypeAttr>("matcore.rhs_dtype");
  auto out_attr = module->getAttrOfType<mlir::TypeAttr>("matcore.out_dtype");
  if (lhs_attr && rhs_attr && out_attr) {
    signature.lhs_dtype = decodeTensorDType(lhs_attr.getValue());
    signature.rhs_dtype = decodeTensorDType(rhs_attr.getValue());
    signature.out_dtype = decodeTensorDType(out_attr.getValue());
    signature.quantized_i8 =
        signature.lhs_dtype == TensorDType::kInt8 &&
        signature.out_dtype == TensorDType::kInt32;
  }
  return signature;
}

void configureLoweringPipeline(mlir::PassManager &pm, const LoweringPlan &plan,
                               const MatmulLoweringSignature &signature,
                               llvm::StringRef nvidia_chip) {
  validateFp8Support(plan, signature);

  switch (plan.route) {
    case LoweringRoute::kCpuVector:
      configureCpuPassPipeline(pm, signature);
      return;
    case LoweringRoute::kNvidiaNvptx:
      configureNvidiaPassPipeline(pm, nvidia_chip);
      return;
    case LoweringRoute::kAmdRocdl:
      configureAmdPassPipeline(pm);
      return;
    case LoweringRoute::kAmdNpuScaffold:
      fail("amd-npu lowering remains unavailable without an external AIE/XDNA toolchain");
      return;
  }
}

void runLoweringPipeline(mlir::ModuleOp module, const LoweringPlan &plan,
                         const MatmulLoweringSignature &signature,
                         llvm::StringRef nvidia_chip) {
  mlir::PassManager pm(module.getContext());
  configureLoweringPipeline(pm, plan, signature, nvidia_chip);
  if (mlir::failed(pm.run(module))) {
    std::string module_ir;
    llvm::raw_string_ostream stream(module_ir);
    module.print(stream);
    stream.flush();
    fail("failed to run lowering pipeline for route " +
         std::string(routeName(plan.route)) + "\n" + module_ir);
  }
}

}  // namespace matcore
