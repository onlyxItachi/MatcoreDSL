#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "matcore/lowering_pipeline.h"

namespace matcore {

// Geometry and transform metadata for NVIDIA GPU block/thread mapping.
struct NvidiaMappingConfig {
  std::int64_t block_tile_m = 128;
  std::int64_t block_tile_n = 128;
  // For the generic path these are per-thread tiles. For the MMA path they
  // represent the per-warp matmul tile handled by one warp inside the CTA.
  std::int64_t thread_tile_m = 16;
  std::int64_t thread_tile_n = 8;
  std::int64_t block_threads_x = 16;
  std::int64_t block_threads_y = 8;
  std::int64_t block_threads_z = 1;
  std::int64_t k_tile = 16;
  bool rewrite_to_mma_sync = false;
};

NvidiaMappingConfig SelectNvidiaMappingConfig(
    mlir::linalg::LinalgOp op, const MatmulLoweringSignature &signature);

bool UsesSingleWarpMmaSync(const NvidiaMappingConfig &config);

std::string BuildNvidiaTransformMappingSequence(
    const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config);

std::string BuildNvidiaThreadMappingSequence(
    const NvidiaMappingConfig &config);

std::string BuildNvidiaMmaRewriteSequence();
std::string BuildNvidiaAsyncPipelineSequence();

std::unique_ptr<mlir::Pass> CreateConfigureNvidiaLaunchPass();
void AddNvidiaLaunchConfigurationPasses(mlir::PassManager &pm);

std::unique_ptr<mlir::Pass> CreateNvidiaDynamicMacroGridMappingPass(
    const NvidiaMappingConfig &config);
void AddNvidiaDynamicMacroGridMappingPasses(mlir::PassManager &pm,
                                            const NvidiaMappingConfig &config);

}  // namespace matcore
