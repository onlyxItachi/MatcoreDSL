#pragma once

#include <cstdint>
#include <memory>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "matcore/lowering_pipeline.h"

namespace matcore {

struct NvidiaTileConfig {
  std::int64_t block_tile_m = 128;
  std::int64_t block_tile_n = 128;
  std::int64_t thread_tile_m = 16;
  std::int64_t thread_tile_n = 8;
  std::int64_t block_threads_y = 8;
  std::int64_t block_threads_x = 16;
  std::int64_t k_tile = 16;
  std::int64_t mma_micro_k = 0;
  bool rewrite_to_mma_sync = true;

  // Multi-warp configuration (V4)
  std::int64_t num_warps = 1;
  std::int64_t warp_tile_m = 0;
  std::int64_t warp_tile_n = 0;
  bool use_vectorize_path = false;
};

bool IsLowPrecisionTensorType(TensorDType dtype);
bool IsTensorCoreMmaSyncType(const MatmulLoweringSignature &signature);

bool IsWorkgroupMemorySpace(mlir::Attribute memory_space);
bool IsWorkgroupMemRefType(mlir::MemRefType type);

NvidiaTileConfig SelectNvidiaTileConfig(mlir::linalg::LinalgOp op,
                                        const MatmulLoweringSignature &signature);

std::unique_ptr<mlir::Pass> CreatePromoteGpuWorkgroupAllocationsPass();
std::unique_ptr<mlir::Pass> CreateSpecializeNvidiaWorkgroupMatmulOperandsPass();
std::unique_ptr<mlir::Pass> CreateDynamicMatmulPaddingPass();

void AddNvidiaMmaPreparationPasses(mlir::PassManager &pm);
void AddNvidiaLoopMaterializationPasses(mlir::PassManager &pm);

}  // namespace matcore
