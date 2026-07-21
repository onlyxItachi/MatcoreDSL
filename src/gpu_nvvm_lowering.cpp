#include "matcore/gpu_nvvm_lowering.h"

#include <stdexcept>
#include <string>

#include "transform_builder.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore lowering pipeline: " + message);
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

void applyNamedSequenceToModule(mlir::ModuleOp module,
                                mlir::ModuleOp transform_module,
                                const char *phase_name) {
  auto entry_point = transform_module.lookupSymbol<mlir::transform::NamedSequenceOp>(
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  if (!entry_point) {
    fail(std::string("failed to build ") + phase_name + " entry point");
  }

  auto named_sequence_attr =
      mlir::transform::TransformDialect::kWithNamedSequenceAttrName;
  mlir::Attribute previous_attr = module->getAttr(named_sequence_attr);
  module->setAttr(named_sequence_attr, mlir::UnitAttr::get(module.getContext()));

  mlir::Operation *cloned_entry = nullptr;
  std::string cloned_entry_name;
  {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    cloned_entry = builder.clone(*entry_point);
    auto cloned_named_sequence =
        llvm::cast<mlir::transform::NamedSequenceOp>(cloned_entry);
    unsigned symbol_suffix = 0;
    do {
      cloned_entry_name = "__matcore_transform_entry_" +
                          std::to_string(symbol_suffix++);
    } while (module.lookupSymbol<mlir::transform::NamedSequenceOp>(
        cloned_entry_name));
    cloned_named_sequence.setSymName(cloned_entry_name);
  }
  auto cleanup = llvm::make_scope_exit([&] {
    if (cloned_entry && cloned_entry->getParentOp()) {
      cloned_entry->erase();
    }
    if (previous_attr) {
      module->setAttr(named_sequence_attr, previous_attr);
    } else {
      module->removeAttr(named_sequence_attr);
    }
  });

  mlir::PassManager pm(module.getContext());
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = cloned_entry_name;
  options.disableExpensiveChecks = false;
  pm.addPass(mlir::transform::createInterpreterPass(options));

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
    fail(std::string("failed to apply ") + phase_name +
         (diagnostics.empty() ? std::string() : ":\n" + diagnostics));
  }
}

}  // namespace

void RegisterNvidiaTransformDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::transform::TransformDialect>();
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::nvgpu::registerTransformDialectExtension(registry);
  mlir::transform::registerDebugExtension(registry);
}

std::string DumpModuleIR(mlir::ModuleOp module) {
  std::string module_ir;
  llvm::raw_string_ostream stream(module_ir);
  module.print(stream);
  stream.flush();
  return module_ir;
}

void ApplyNvidiaMmaTransformToModule(mlir::ModuleOp module,
                                     const MatmulLoweringSignature &signature,
                                     const NvidiaMappingConfig &config) {
  mlir::DialectRegistry registry;
  RegisterNvidiaTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  mlir::Operation *first_matmul = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (first_matmul == nullptr &&
        llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::QuantizedMatmulOp>(op)) {
      first_matmul = op;
    }
  });
  if (first_matmul == nullptr) {
    return;
  }

  auto transform_module = BuildNvidiaTransformMappingModule(
      module.getContext(), module.getLoc(), signature, config);
  applyNamedSequenceToModule(module, *transform_module,
                             "NVIDIA MMA transform sequence");
}

void ApplyNvidiaThreadMappingToModule(mlir::ModuleOp module,
                                      const NvidiaMappingConfig &config) {
  mlir::DialectRegistry registry;
  RegisterNvidiaTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  bool has_launch = false;
  module.walk([&](mlir::Operation *op) {
    if (!has_launch && llvm::isa<mlir::gpu::LaunchOp>(op)) {
      has_launch = true;
    }
  });
  if (!has_launch) {
    return;
  }

  auto transform_module = BuildNvidiaThreadMappingModule(
      module.getContext(), module.getLoc(), config);
  applyNamedSequenceToModule(module, *transform_module,
                             "NVIDIA thread mapping sequence");
}

void ApplyNvidiaMmaRewriteToModule(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  RegisterNvidiaTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  bool has_matmul = false;
  module.walk([&](mlir::Operation *op) {
    if (!has_matmul && llvm::isa<mlir::linalg::MatmulOp>(op)) {
      has_matmul = true;
    }
  });
  if (!has_matmul) {
    return;
  }

  auto transform_module =
      BuildNvidiaMmaRewriteModule(module.getContext(), module.getLoc());
  applyNamedSequenceToModule(module, *transform_module,
                             "NVIDIA MMA rewrite sequence");
}

void ApplyNvidiaAsyncPipelineToModule(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  RegisterNvidiaTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  bool has_launch = false;
  module.walk([&](mlir::Operation *op) {
    if (!has_launch && llvm::isa<mlir::gpu::LaunchOp>(op)) {
      has_launch = true;
    }
  });
  if (!has_launch) {
    return;
  }

  auto transform_module = mlir::parseSourceString<mlir::ModuleOp>(
      BuildNvidiaAsyncPipelineSequence(), module.getContext());
  if (!transform_module) {
    fail("failed to parse NVIDIA async pipeline transform module");
  }
  applyNamedSequenceToModule(module, *transform_module,
                             "NVIDIA async pipeline sequence");
}

void VerifyNoResidualNvidiaMatmulOnModule(mlir::ModuleOp module) {
  bool residual_matmul = false;
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::linalg::MatmulOp>(op)) {
      residual_matmul = true;
    }
  });
  if (residual_matmul) {
    fail("NVIDIA lowering left a residual linalg matmul");
  }
}

void ApplyNvidiaMultiWarpTransformToModule(
    mlir::ModuleOp module, const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config) {
  mlir::DialectRegistry registry;
  RegisterNvidiaTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  auto transform_module = BuildNvidiaMultiWarpTransformModule(
      module.getContext(), module.getLoc(), signature, config);
  applyNamedSequenceToModule(module, *transform_module,
                             "NVIDIA multi-warp transform sequence");
}

void ApplyNvidiaMultiWarpThreadMappingToModule(
    mlir::ModuleOp module, const NvidiaMappingConfig &config) {
  mlir::DialectRegistry registry;
  RegisterNvidiaTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  bool has_launch = false;
  module.walk([&](mlir::Operation *op) {
    if (!has_launch && llvm::isa<mlir::gpu::LaunchOp>(op)) {
      has_launch = true;
    }
  });
  if (!has_launch) {
    return;
  }

  auto transform_module = BuildNvidiaMultiWarpThreadMappingModule(
      module.getContext(), module.getLoc(), config);
  applyNamedSequenceToModule(module, *transform_module,
                             "NVIDIA multi-warp thread mapping sequence");
}

void ConfigureNvidiaGenericGpuStage(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);
  addGpuCommonModulePasses(pm, /*index_bitwidth=*/64);
  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/true));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::nvgpu::createOptimizeSharedMemoryPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void ConfigureNvidiaVectorToGpuStage(mlir::PassManager &pm) {
  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/true));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::nvgpu::createOptimizeSharedMemoryPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void ConfigureNvidiaNvvmStage(mlir::PassManager &pm, llvm::StringRef cubin_chip,
                              ObservabilityContext * /*obs*/) {
  // Custom NVVM pipeline that handles gpu.alloc/memcpy/dealloc from the
  // GpuDataStagingPass.  The stock buildLowerToNVVMPassPipeline runs
  // convert-func-to-llvm BEFORE gpu-to-llvm, which breaks the memref-typed
  // staging ops (unrealized_conversion_casts that can't be reconciled).
  // Our pipeline moves gpu-to-llvm BEFORE convert-func-to-llvm so the staging
  // ops are lowered while their operands are still memref-typed.

  // Phase 1: Kernel outlining — separate GPU code from host code.
  pm.addPass(mlir::createConvertNVGPUToNVVMPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());

  // Phase 2: Host-side scalar/vector lowering (memrefs stay as-is).
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertNVVMToLLVMPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createArithToLLVMConversionPass());

  mlir::ConvertIndexToLLVMPassOptions idx_opts;
  idx_opts.indexBitwidth = 64;
  pm.addPass(mlir::createConvertIndexToLLVMPass(idx_opts));

  // Phase 3: Attach NVVM target to gpu.module (needed by serialisation).
  mlir::GpuNVVMAttachTargetOptions target_opts;
  target_opts.triple = "nvptx64-nvidia-cuda";
  target_opts.chip = cubin_chip.str();
  target_opts.optLevel = 2;
  pm.addPass(mlir::createGpuNVVMAttachTarget(target_opts));

  // Phase 4: GPU-module internal lowering (NVVM + reconcile inside module).
  {
    auto &gpu_pm = pm.nest<mlir::gpu::GPUModuleOp>();
    gpu_pm.addPass(mlir::createStripDebugInfoPass());
    mlir::ConvertGpuOpsToNVVMOpsOptions nvvm_conv_opts;
    nvvm_conv_opts.indexBitwidth = 64;
    nvvm_conv_opts.useBarePtrCallConv = false;
    gpu_pm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvm_conv_opts));
    gpu_pm.addPass(mlir::createCanonicalizerPass());
    gpu_pm.addPass(mlir::createCSEPass());
    gpu_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  // Phase 5: Host-side GPU memory ops → LLVM runtime calls.
  // Converts gpu.alloc/memcpy/dealloc → mgpuMemAlloc/mgpuMemcpy/mgpuMemFree.
  // MUST run BEFORE convert-func-to-llvm so staging ops have memref-typed operands.
  // NOTE: gpu.launch_func is NOT converted here — it's handled later by
  // MLIR-to-LLVM-IR translation in the ExecutionEngine (together with gpu.binary).
  pm.addPass(mlir::createGpuToLLVMConversionPass());

  // Phase 5b: Immediately finalize remaining host ops → LLVM.
  // GpuToLLVMConversionPass (Phase 5) partially converts function types via its
  // LLVMTypeConverter.  If we defer memref/cf/func lowering to after binary
  // serialisation (Phase 6), the mixed type state creates unrealized_conversion_casts
  // that ReconcileUnrealizedCasts cannot resolve (the host-level linalg.fill loop
  // produces memref.alloc + cf.br that reference LLVM-typed values from Phase 5).
  // By running these passes immediately after Phase 5, all host ops are converted
  // within a consistent type-conversion context.
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Phase 6: Serialise gpu.module → gpu.binary (fatbin).
  // The gpu.binary + gpu.launch_func pair is translated to LLVM IR
  // by ExecutionEngine's GPUDialectLLVMIRTranslationInterface.
  mlir::GpuModuleToBinaryPassOptions bin_opts;
  bin_opts.compilationTarget = "fatbin";
  pm.addPass(mlir::createGpuModuleToBinaryPass(bin_opts));

  // Phase 7: Final cleanup after binary serialisation.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

}  // namespace matcore
