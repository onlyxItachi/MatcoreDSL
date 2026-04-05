#pragma once

#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

#include "matcore/gpu_mapping.h"
#include "matcore/lowering_pipeline.h"

namespace matcore {

void RegisterNvidiaTransformDialects(mlir::DialectRegistry &registry);

void ApplyNvidiaMmaTransformToModule(mlir::ModuleOp module,
                                     const MatmulLoweringSignature &signature,
                                     const NvidiaMappingConfig &config);
void ApplyNvidiaThreadMappingToModule(mlir::ModuleOp module,
                                      const NvidiaMappingConfig &config);
void ApplyNvidiaMmaRewriteToModule(mlir::ModuleOp module);

void VerifyNoResidualNvidiaMatmulOnModule(mlir::ModuleOp module);

void ConfigureNvidiaGenericGpuStage(mlir::PassManager &pm);
void ConfigureNvidiaVectorToGpuStage(mlir::PassManager &pm);
void ConfigureNvidiaNvvmStage(mlir::PassManager &pm, llvm::StringRef cubin_chip,
                              ObservabilityContext *obs = nullptr);

std::string DumpModuleIR(mlir::ModuleOp module);

}  // namespace matcore
