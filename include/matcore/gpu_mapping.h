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
  std::int64_t thread_tile_m = 16;
  std::int64_t thread_tile_n = 8;
  std::int64_t block_threads_x = 16;
  std::int64_t block_threads_y = 8;
  std::int64_t block_threads_z = 1;
  std::int64_t k_tile = 16;
  std::int64_t mma_micro_k = 0;  // Inner K-tile for mma.sync (0 = no inner tiling)
  bool rewrite_to_mma_sync = false;

  // Multi-warp configuration (V4)
  std::int64_t num_warps = 1;       // 1 = single-warp (V3), 4 = multi-warp (V4)
  std::int64_t warp_tile_m = 0;     // Per-warp M tile (0 = not multi-warp)
  std::int64_t warp_tile_n = 0;     // Per-warp N tile
  bool use_vectorize_path = false;  // true = vectorize + VectorToGPU
};

NvidiaMappingConfig SelectNvidiaMappingConfig(
    mlir::linalg::LinalgOp op, const MatmulLoweringSignature &signature);

bool UsesSingleWarpMmaSync(const NvidiaMappingConfig &config);
bool UsesMultiWarpMmaSync(const NvidiaMappingConfig &config);

std::string BuildNvidiaTransformMappingSequence(
    const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config);

std::string BuildNvidiaThreadMappingSequence(
    const NvidiaMappingConfig &config);

std::string BuildNvidiaMmaRewriteSequence();

std::unique_ptr<mlir::Pass> CreateConfigureNvidiaLaunchPass();
void AddNvidiaLaunchConfigurationPasses(mlir::PassManager &pm);

std::unique_ptr<mlir::Pass> CreateNvidiaDynamicMacroGridMappingPass(
    const NvidiaMappingConfig &config);
void AddNvidiaDynamicMacroGridMappingPasses(mlir::PassManager &pm,
                                            const NvidiaMappingConfig &config);

}  // namespace matcore
