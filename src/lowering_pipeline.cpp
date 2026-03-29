#include "matcore/lowering_pipeline.h"

#include "matcore/cpu_lowering.h"
#include "matcore/gpu_mapping.h"
#include "matcore/gpu_nvvm_lowering.h"
#include "matcore/gpu_tiling.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore lowering pipeline: " + message);
}

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

void addGpuHostPostPasses(mlir::PassManager &pm,
                          const std::string &binary_target,
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

NvidiaMappingConfig selectNvidiaMappingForModule(
    mlir::ModuleOp module, const MatmulLoweringSignature &signature) {
  mlir::Operation *first_matmul = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (first_matmul != nullptr) {
      return;
    }
    if (!llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::QuantizedMatmulOp>(op)) {
      return;
    }
    first_matmul = op;
  });
  if (first_matmul == nullptr) {
    fail("NVIDIA lowering expected a linalg matmul op before transform");
  }
  return SelectNvidiaMappingConfig(
      llvm::cast<mlir::linalg::LinalgOp>(first_matmul), signature);
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
      return "linalg.matmul -> tiled gpu mapping -> workgroup promotion -> mma sync -> gpu.launch -> nvvm -> llvm";
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
      ConfigureNvidiaGenericGpuStage(pm);
      ConfigureNvidiaNvvmStage(pm, nvidia_chip);
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
  auto run_stage = [&](llvm::StringRef stage_name, auto &&configure_stage) {
    mlir::PassManager pm(module.getContext());
    configure_stage(pm);
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler diag_handler(
        module.getContext(), [&](mlir::Diagnostic &diag) {
          llvm::raw_string_ostream stream(diagnostics);
          diag.print(stream);
          stream << '\n';
          stream.flush();
          return mlir::success();
        });
    if (mlir::failed(pm.run(module))) {
      fail("failed to run lowering pipeline for route " +
           std::string(routeName(plan.route)) + " at stage '" +
           stage_name.str() + "'" +
           (diagnostics.empty() ? std::string() : "\n" + diagnostics) + "\n" +
           DumpModuleIR(module));
    }
  };

  if (plan.route == LoweringRoute::kNvidiaNvptx) {
    run_stage("nvidia-tensor-bufferize", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
      pm.addPass(mlir::bufferization::createOneShotBufferizePass());
      pm.addPass(mlir::createBufferizationToMemRefPass());
      pm.addPass(
          mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
      pm.addPass(
          mlir::bufferization::createBufferDeallocationSimplificationPass());
      pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    });
    const NvidiaMappingConfig mapping =
        selectNvidiaMappingForModule(module, signature);
    try {
      ApplyNvidiaMmaTransformToModule(module, signature, mapping);
    } catch (const std::exception &exc) {
      fail("failed to run lowering pipeline for route " +
           std::string(routeName(plan.route)) +
           " at stage 'nvidia-apply-transform'\n" + exc.what() + "\n" +
           DumpModuleIR(module));
    }
    run_stage("nvidia-dynamic-macro-topology", [&](mlir::PassManager &pm) {
      AddNvidiaDynamicMacroGridMappingPasses(pm, mapping);
    });

    if (mapping.rewrite_to_mma_sync) {
      run_stage("nvidia-post-transform-canonicalize",
                [&](mlir::PassManager &pm) {
                  pm.addPass(mlir::createCanonicalizerPass());
                  pm.addPass(mlir::createCSEPass());
                });
      run_stage("nvidia-mma-preparation", [&](mlir::PassManager &pm) {
        AddNvidiaMmaPreparationPasses(pm);
      });
      try {
        ApplyNvidiaMmaRewriteToModule(module);
      } catch (const std::exception &exc) {
        fail("failed to run lowering pipeline for route " +
             std::string(routeName(plan.route)) +
             " at stage 'nvidia-rewrite-mma-sync'\n" + exc.what() + "\n" +
             DumpModuleIR(module));
      }
      VerifyNoResidualNvidiaMatmulOnModule(module);
      run_stage("nvidia-launch-config", [&](mlir::PassManager &pm) {
        AddNvidiaLaunchConfigurationPasses(pm);
      });
      run_stage("nvidia-loop-materialization", [&](mlir::PassManager &pm) {
        AddNvidiaLoopMaterializationPasses(pm);
      });
      run_stage("nvidia-vector-to-gpu", [&](mlir::PassManager &pm) {
        ConfigureNvidiaVectorToGpuStage(pm);
      });
    } else {
      try {
        ApplyNvidiaThreadMappingToModule(module, mapping);
      } catch (const std::exception &exc) {
        fail("failed to run lowering pipeline for route " +
             std::string(routeName(plan.route)) +
             " at stage 'nvidia-map-threads'\n" + exc.what() + "\n" +
             DumpModuleIR(module));
      }
      run_stage("nvidia-post-thread-map-canonicalize",
                [&](mlir::PassManager &pm) {
                  pm.addPass(mlir::createCanonicalizerPass());
                  pm.addPass(mlir::createCSEPass());
                });
      run_stage("nvidia-launch-config", [&](mlir::PassManager &pm) {
        AddNvidiaLaunchConfigurationPasses(pm);
      });
      run_stage("nvidia-loop-materialization", [&](mlir::PassManager &pm) {
        AddNvidiaLoopMaterializationPasses(pm);
      });
      run_stage("nvidia-vector-to-gpu", [&](mlir::PassManager &pm) {
        ConfigureNvidiaVectorToGpuStage(pm);
      });
    }
    run_stage("nvidia-nvvm", [&](mlir::PassManager &pm) {
      ConfigureNvidiaNvvmStage(pm, nvidia_chip);
    });
    return;
  }

  mlir::PassManager pm(module.getContext());
  configureLoweringPipeline(pm, plan, signature, nvidia_chip);
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diag_handler(
      module.getContext(), [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream stream(diagnostics);
        diag.print(stream);
        stream << '\n';
        stream.flush();
        return mlir::success();
      });
  if (mlir::failed(pm.run(module))) {
    fail("failed to run lowering pipeline for route " +
         std::string(routeName(plan.route)) +
         (diagnostics.empty() ? std::string() : "\n" + diagnostics) + "\n" +
         DumpModuleIR(module));
  }
}

}  // namespace matcore
