#include "matcore/mlir_engine.h"

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
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

struct VectorizeMatmulPass
    : public mlir::PassWrapper<VectorizeMatmulPass,
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
      if (mlir::failed(mlir::linalg::vectorizeOpPrecondition(matmul))) {
        matmul->emitError("MatCore x86 lowering requires vectorizable linalg.matmul");
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(matmul);
      if (mlir::failed(mlir::linalg::vectorize(rewriter, matmul))) {
        matmul->emitError("MatCore failed to vectorize linalg.matmul");
        signalPassFailure();
        return;
      }
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

void configureCpuPassPipeline(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<VectorizeMatmulPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/true);
}

void configureNvidiaPassPipeline(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);

  mlir::GpuNVVMAttachTargetOptions nvvm_target_opts;
  nvvm_target_opts.triple = "nvptx64-nvidia-cuda";
  nvvm_target_opts.chip = "sm_80";
  nvvm_target_opts.features = "+ptx80";
  pm.addPass(mlir::createGpuNVVMAttachTarget(nvvm_target_opts));

  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/true));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertNVGPUToNVVMPass());
  mlir::GpuModuleToBinaryPassOptions binary_opts;
  binary_opts.toolkitPath = "/usr/local/cuda-13.2";
  binary_opts.compilationTarget = "fatbin";
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary_opts));
  pm.addPass(mlir::createGpuToLLVMConversionPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/false);
}

void configureAmdPassPipeline(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);

  mlir::GpuROCDLAttachTargetOptions rocdl_target_opts;
  rocdl_target_opts.triple = "amdgcn-amd-amdhsa";
  rocdl_target_opts.chip = "gfx900";
  pm.addPass(mlir::createGpuROCDLAttachTarget(rocdl_target_opts));

  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/false));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createLowerGpuOpsToROCDLOpsPass(rocdl_target_opts.chip));
  mlir::GpuModuleToBinaryPassOptions binary_opts;
  binary_opts.toolkitPath = "/usr";
  binary_opts.compilationTarget = "hsaco";
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary_opts));
  pm.addPass(mlir::createGpuToLLVMConversionPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/false);
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
    fail("failed to run lowering pipeline for route " + std::string(routeName(route)));
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
