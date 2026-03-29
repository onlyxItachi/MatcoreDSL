#include "matcore/gpu_nvvm_lowering.h"

#include <stdexcept>
#include <string>

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
                                const std::string &transform_ir,
                                const char *phase_name) {
  auto transform_module =
      mlir::parseSourceString<mlir::ModuleOp>(transform_ir, module.getContext());
  if (!transform_module) {
    fail(std::string("failed to parse ") + phase_name + ":\n" + transform_ir);
  }

  auto entry_point = transform_module->lookupSymbol<mlir::transform::NamedSequenceOp>(
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  if (!entry_point) {
    fail(std::string("failed to build ") + phase_name + " entry point");
  }

  module->setAttr(mlir::transform::TransformDialect::kWithNamedSequenceAttrName,
                  mlir::UnitAttr::get(module.getContext()));
  {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    builder.clone(*entry_point);
  }

  mlir::PassManager pm(module.getContext());
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint =
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName.str();
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

  llvm::SmallVector<mlir::Operation *, 2> transform_symbols;
  module.walk([&](mlir::transform::NamedSequenceOp op) {
    transform_symbols.push_back(op.getOperation());
  });
  for (mlir::Operation *op : llvm::reverse(transform_symbols)) {
    op->erase();
  }
  module->removeAttr(mlir::transform::TransformDialect::kWithNamedSequenceAttrName);
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

  const std::string transform_ir =
      BuildNvidiaTransformMappingSequence(signature, config);
  applyNamedSequenceToModule(module, transform_ir, "NVIDIA MMA transform sequence");
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

  applyNamedSequenceToModule(module, BuildNvidiaThreadMappingSequence(config),
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

  applyNamedSequenceToModule(module, BuildNvidiaMmaRewriteSequence(),
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

  applyNamedSequenceToModule(module, BuildNvidiaAsyncPipelineSequence(),
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

void ConfigureNvidiaNvvmStage(mlir::PassManager &pm, llvm::StringRef cubin_chip) {
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

}  // namespace matcore
