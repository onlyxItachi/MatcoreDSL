#include "matcore/mlir_engine.h"

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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

enum class LoweringRoute {
  kCpuVector,
  kNvidiaNvptx,
  kAmdRocdl,
  kAmdNpuScaffold,
};

struct MatmulShape {
  std::int64_t m = 0;
  std::int64_t k = 0;
  std::int64_t n = 0;
};

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

  // Keep matmul tiles small enough that vectorization produces tractable
  // micro-kernels instead of a whole-tile mega-vector.
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
      // `linalg::vectorize` materializes the vector form but does not guarantee
      // that the source op is erased for bufferized named ops. Leaving the
      // tiled matmul alive makes the kernel execute the same math twice.
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

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore MLIR engine: " + message);
}

mlir::Type getElementType(TensorDType dtype, mlir::Builder &builder) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return builder.getF32Type();
    case TensorDType::kFloat16:
      return builder.getF16Type();
    case TensorDType::kBFloat16:
      return builder.getBF16Type();
  }
  fail("unsupported tensor dtype");
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

void validateRuntimeTensor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (!tensor.c_contiguous) {
    fail("tensor '" + tensor.symbol + "' must be C-contiguous");
  }
  if (tensor.shape.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for matmul");
  }
  if (tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must provide rank-2 strides");
  }
  switch (tensor.dtype) {
    case TensorDType::kFloat32:
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return;
  }
  fail("tensor '" + tensor.symbol + "' has unsupported dtype");
}

TensorDType dominantInputDType(const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 2) {
    fail("runtime must provide at least lhs and rhs tensors");
  }
  const TensorDType lhs = tensors[0].dtype;
  const TensorDType rhs = tensors[1].dtype;
  if (lhs != rhs) {
    fail("lhs/rhs dtype mismatch is not supported in Phase 2");
  }
  return lhs;
}

void validateKernel(const KernelIR &kernel, TargetKind target,
                    const std::vector<RuntimeTensorView> &tensors) {
  if (kernel.params.size() < 3) {
    fail("kernel must expose at least 3 params (lhs, rhs, out)");
  }
  if (tensors.size() < 3) {
    fail("runtime must provide at least 3 tensors (lhs, rhs, out)");
  }
  for (const RuntimeTensorView &tensor : tensors) {
    validateRuntimeTensor(tensor);
  }

  const RuntimeTensorView &lhs = tensors[0];
  const RuntimeTensorView &rhs = tensors[1];
  const RuntimeTensorView &out = tensors[2];
  const std::int64_t m = lhs.shape[0];
  const std::int64_t k = lhs.shape[1];
  const std::int64_t k_rhs = rhs.shape[0];
  const std::int64_t n = rhs.shape[1];
  if (k != k_rhs) {
    fail("matmul inner dimensions mismatch: lhs K != rhs K");
  }
  if (out.shape[0] != m || out.shape[1] != n) {
    fail("output shape mismatch for matmul result");
  }

  const TensorDType input_dtype = dominantInputDType(tensors);
  if (out.dtype != input_dtype) {
    fail("output dtype must match input dtype in Phase 2");
  }

  bool sawLoad = false;
  bool sawMatmul = false;
  bool sawStore = false;
  for (const KernelOp &op : kernel.ops) {
    if (std::holds_alternative<LoadOp>(op)) {
      sawLoad = true;
    } else if (std::holds_alternative<MatMulOp>(op)) {
      sawMatmul = true;
    } else if (std::holds_alternative<StoreOp>(op)) {
      sawStore = true;
    }
  }
  if (!(sawLoad && sawMatmul && sawStore)) {
    fail("kernel ops must include load -> matmul -> store pattern");
  }

  const TargetKind normalized = normalizeTarget(target);
  if (normalized == TargetKind::kARM || normalized == TargetKind::kTPU) {
    fail("target route is defined but not implemented in Phase 2");
  }
}

MatmulShape extractMatmulShape(const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 2) {
    fail("runtime must provide at least lhs and rhs tensors");
  }
  const RuntimeTensorView &lhs = tensors[0];
  const RuntimeTensorView &rhs = tensors[1];
  MatmulShape shape;
  shape.m = lhs.shape[0];
  shape.k = lhs.shape[1];
  shape.n = rhs.shape[1];
  if (shape.m <= 0 || shape.k <= 0 || shape.n <= 0) {
    fail("matmul dimensions must be positive");
  }
  return shape;
}

mlir::OwningOpRef<mlir::ModuleOp> buildMatmulModule(
    const KernelIR &kernel, TensorDType input_dtype, LoweringRoute route,
    const MatmulShape &shape,
    mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  const mlir::Type element_type = getElementType(input_dtype, builder);
  module->setAttr("matcore.input_dtype", mlir::TypeAttr::get(element_type));
  module->setAttr("matcore.route", builder.getStringAttr(routeName(route)));
  module->setAttr("matcore.route_description",
                  builder.getStringAttr(routeDescription(route)));

  const std::string entry_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const auto lhs_type = mlir::MemRefType::get({shape.m, shape.k}, element_type);
  const auto rhs_type = mlir::MemRefType::get({shape.k, shape.n}, element_type);
  const auto out_type = mlir::MemRefType::get({shape.m, shape.n}, element_type);

  auto func = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), entry_name,
      builder.getFunctionType({lhs_type, rhs_type, out_type}, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  auto zero = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getZeroAttr(element_type));
  builder.create<mlir::linalg::FillOp>(
      builder.getUnknownLoc(), mlir::ValueRange{zero},
      mlir::ValueRange{entry_block->getArgument(2)});
  builder.create<mlir::linalg::MatmulOp>(
      builder.getUnknownLoc(),
      mlir::ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      mlir::ValueRange{entry_block->getArgument(2)});
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return module;
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

void configureCpuPassPipeline(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<TileAndVectorizeMatmulPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<LowerResidualVectorOpsPass>());
  mlir::VectorTransferToSCFOptions vector_to_scf_options;
  vector_to_scf_options.setTargetRank(1);
  pm.addPass(mlir::createConvertVectorToSCFPass(vector_to_scf_options));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/true);
}

void configureNvidiaPassPipeline(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);

  mlir::gpu::GPUToNVVMPipelineOptions nvvm_opts;
  nvvm_opts.indexBitWidth = 64;
  nvvm_opts.cubinTriple = "nvptx64-nvidia-cuda";
  nvvm_opts.cubinChip = "sm_80";
  nvvm_opts.cubinFeatures = "+ptx80";
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
  // Ubuntu packages the ROCm device bitcode under the LLVM 17 Clang resource dir.
  addGpuHostPostPasses(pm, "bin", "/usr/lib/llvm-17/lib/clang/17");
}

void configurePassPipeline(mlir::PassManager &pm, LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      configureCpuPassPipeline(pm);
      return;
    case LoweringRoute::kNvidiaNvptx:
      configureNvidiaPassPipeline(pm);
      return;
    case LoweringRoute::kAmdRocdl:
      configureAmdPassPipeline(pm);
      return;
    case LoweringRoute::kAmdNpuScaffold:
      fail("amd-npu lowering remains unavailable without an external AIE/XDNA toolchain");
      return;
  }
}

void runLoweringPipeline(mlir::ModuleOp module, LoweringRoute route) {
  mlir::PassManager pm(module.getContext());
  configurePassPipeline(pm, route);
  if (mlir::failed(pm.run(module))) {
    std::string module_ir;
    llvm::raw_string_ostream stream(module_ir);
    module.print(stream);
    stream.flush();
    fail("failed to run lowering pipeline for route " + std::string(routeName(route)) +
         "\n" + module_ir);
  }
}

}  // namespace

LoweredModule MlirEngine::BuildAndLower(
    const KernelIR &kernel, TargetKind target,
    const std::vector<RuntimeTensorView> &tensors,
    mlir::MLIRContext &context) {
  context.loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                      mlir::func::FuncDialect, mlir::gpu::GPUDialect,
                      mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                      mlir::nvgpu::NVGPUDialect, mlir::NVVM::NVVMDialect,
                      mlir::ROCDL::ROCDLDialect, mlir::scf::SCFDialect,
                      mlir::vector::VectorDialect, mlir::x86vector::X86VectorDialect,
                      mlir::LLVM::LLVMDialect>();

  validateKernel(kernel, target, tensors);
  const TargetKind normalized_target = normalizeTarget(target);
  const LoweringRoute route = selectRoute(normalized_target);
  const TensorDType input_dtype = dominantInputDType(tensors);
  const MatmulShape shape = extractMatmulShape(tensors);

  auto module = buildMatmulModule(kernel, input_dtype, route, shape, context);
  runLoweringPipeline(*module, route);

  LoweredModule lowered;
  lowered.module = std::move(module);
  lowered.entry_point =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  lowered.target = normalized_target;
  lowered.route_description = routeDescription(route);
  lowered.executable = route != LoweringRoute::kAmdNpuScaffold;
  return lowered;
}

}  // namespace matcore
