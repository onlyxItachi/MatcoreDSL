#include "gpu_amd_lowering.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "gpu_data_staging.h"
#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace matcore {
namespace {

std::string normalizeChip(std::string chip) {
  if (chip.empty()) {
    return "gfx90a";
  }
  std::transform(chip.begin(), chip.end(), chip.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return chip;
}

}  // namespace

AmdGpuConfig detectAmdGpuConfig(const std::string &chip) {
  AmdGpuConfig config;
  config.chip = normalizeChip(chip);
  config.wavefront_size = 64;
  config.lds_size_kb = 64;

  if (config.chip == "gfx908" || config.chip == "gfx90a") {
    config.has_mfma = true;
    config.has_wmma = false;
    return config;
  }
  if (config.chip == "gfx942") {
    config.has_mfma = true;
    config.has_wmma = false;
    return config;
  }
  if (config.chip == "gfx1100" || config.chip == "gfx1101") {
    config.wavefront_size = 32;
    config.has_mfma = false;
    config.has_wmma = true;
    return config;
  }

  return config;
}

bool isEligibleForAmdGpu(const MatmulLoweringSignature &sig,
                         const AmdGpuConfig &config) {
  return amdIneligibilityReason(sig, config).empty();
}

std::string amdIneligibilityReason(const MatmulLoweringSignature &sig,
                                   const AmdGpuConfig &config) {
  if (sig.lhs_dtype != sig.rhs_dtype) {
    return "AMD lowering requires lhs/rhs operand dtypes to match";
  }
  if (!config.has_mfma && !config.has_wmma) {
    return "AMD chip '" + config.chip + "' does not expose MFMA or WMMA support";
  }

  switch (sig.lhs_dtype) {
    case TensorDType::kFloat32:
      if (sig.out_dtype != TensorDType::kFloat32) {
        return "float32 matmul on AMD requires float32 output";
      }
      return {};
    case TensorDType::kFloat16:
      if (sig.out_dtype != TensorDType::kFloat16 &&
          sig.out_dtype != TensorDType::kFloat32) {
        return "float16 matmul on AMD requires float16 or float32 output";
      }
      return {};
    case TensorDType::kBFloat16:
      if (sig.out_dtype != TensorDType::kBFloat16 &&
          sig.out_dtype != TensorDType::kFloat32) {
        return "bfloat16 matmul on AMD requires bfloat16 or float32 output";
      }
      return {};
    case TensorDType::kInt8:
      if (!config.has_mfma || sig.out_dtype != TensorDType::kInt32) {
        return "int8 matmul on AMD requires MFMA support with int32 output";
      }
      return {};
    case TensorDType::kFloat8E4M3FN:
      return "float8_e4m3fn is disabled on AMD targets: MatCore FP8 uses E4M3FN "
             "while MLIR 18 AMDGPU lowering currently expects FNUZ FP8 types";
    case TensorDType::kInt32:
      break;
  }
  return "unsupported AMD matmul dtype combination";
}

void configureAmdLoweringPipeline(mlir::PassManager &pm,
                                  const AmdGpuConfig &config) {
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());

  // Stage host memrefs to GPU device memory before kernel outlining.
  pm.addNestedPass<mlir::func::FuncOp>(CreateGpuDataStagingPass());

  pm.addPass(mlir::createGpuKernelOutliningPass());

  mlir::GpuROCDLAttachTargetOptions rocdl_target_opts;
  rocdl_target_opts.triple = "amdgcn-amd-amdhsa";
  rocdl_target_opts.chip = config.chip;
  rocdl_target_opts.wave64Flag = config.wavefront_size == 64;
  pm.addPass(mlir::createGpuROCDLAttachTarget(rocdl_target_opts));

  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertAMDGPUToROCDLPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createLowerGpuOpsToROCDLOpsPass(
      config.chip, /*indexBitwidth=*/64, /*useBarePtrCallConv=*/false,
      mlir::gpu::amd::Runtime::HIP));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());
}

}  // namespace matcore
