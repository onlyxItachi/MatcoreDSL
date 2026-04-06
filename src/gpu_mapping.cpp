#include "matcore/gpu_mapping.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include "transform_builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

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

std::int64_t ceilDiv(std::int64_t lhs, std::int64_t rhs) {
  if (rhs <= 0) {
    return 1;
  }
  if (lhs <= 0) {
    return 1;
  }
  return (lhs + rhs - 1) / rhs;
}

bool isTensorCoreMmaSyncType(const MatmulLoweringSignature &signature) {
  return signature.lhs_dtype == TensorDType::kFloat16 &&
         signature.rhs_dtype == TensorDType::kFloat16 &&
         signature.out_dtype == TensorDType::kFloat16;
}

std::optional<std::int64_t> matchConstantIndex(mlir::Value value) {
  llvm::APInt constant;
  if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant))) {
    return std::nullopt;
  }
  return constant.getSExtValue();
}

std::optional<mlir::Value> findRank2MemRefArgument(mlir::func::FuncOp func,
                                                    unsigned ordinal) {
  unsigned seen = 0;
  for (mlir::BlockArgument arg : func.getArguments()) {
    auto type = llvm::dyn_cast<mlir::MemRefType>(arg.getType());
    if (!type || !type.hasRank() || type.getRank() != 2) {
      continue;
    }
    if (seen == ordinal) {
      return arg;
    }
    ++seen;
  }
  return std::nullopt;
}

mlir::Value buildCeilDivIndex(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value lhs, mlir::Value rhs) {
  return builder.create<mlir::arith::CeilDivUIOp>(loc, lhs, rhs);
}

bool isBlockMappedForall(mlir::scf::ForallOp forall) {
  auto mapping = forall.getMapping();
  if (!mapping || mapping->empty()) {
    return false;
  }
  for (mlir::Attribute attr : *mapping) {
    if (!llvm::isa<mlir::gpu::GPUBlockMappingAttr>(attr)) {
      return false;
    }
  }
  return true;
}

mlir::Value blockIdForMapping(mlir::gpu::LaunchOp launch,
                              mlir::gpu::GPUBlockMappingAttr attr) {
  mlir::gpu::KernelDim3 ids = launch.getBlockIds();
  switch (attr.getRelativeIndex()) {
    case 0:
      return ids.x;
    case 1:
      return ids.y;
    case 2:
      return ids.z;
    default:
      return {};
  }
}

}  // namespace

NvidiaMappingConfig SelectNvidiaMappingConfig(
    mlir::linalg::LinalgOp op, const MatmulLoweringSignature &signature) {
  llvm::SmallVector<mlir::Value> inputs = op.getDpsInputs();
  auto lhs_type = inputs.size() > 0
                      ? llvm::dyn_cast<mlir::ShapedType>(inputs[0].getType())
                      : mlir::ShapedType();
  auto rhs_type = inputs.size() > 1
                      ? llvm::dyn_cast<mlir::ShapedType>(inputs[1].getType())
                      : mlir::ShapedType();

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

  NvidiaMappingConfig config;
  config.block_tile_m = pickTilingFactor(m, 128);
  config.block_tile_n = pickTilingFactor(n, 128);
  config.k_tile = pickTilingFactor(k, signature.quantized_i8 ? 32 : 16);

  const bool statically_compatible_mma =
      m != mlir::ShapedType::kDynamic && n != mlir::ShapedType::kDynamic &&
      k != mlir::ShapedType::kDynamic && m >= 16 && n >= 8 && k >= 16 &&
      (m % 16) == 0 && (n % 8) == 0 && (k % 16) == 0 &&
      isTensorCoreMmaSyncType(signature);
  // GpuDataStagingPass now handles host↔device memory transfers safely.
  config.rewrite_to_mma_sync = statically_compatible_mma;
  if (config.rewrite_to_mma_sync) {
    // Occupancy-aware adaptive tiling: choose the largest block tile
    // that keeps enough grid blocks for SM occupancy. With 1 warp per
    // block, we need many blocks to hide memory latency.
    constexpr int64_t kMinGridBlocks = 100;

    struct TileCandidate { int64_t m_pref, n_pref; };
    constexpr TileCandidate candidates[] = {{64, 64}, {32, 32}};

    config.block_tile_m = 16;
    config.block_tile_n = 8;
    for (const auto &c : candidates) {
      auto tm = pickTilingFactor(m, c.m_pref);
      auto tn = pickTilingFactor(n, c.n_pref);
      if (tm >= 16 && tn >= 8) {
        auto grid = (m / tm) * (n / tn);
        if (grid >= kMinGridBlocks) {
          config.block_tile_m = tm;
          config.block_tile_n = tn;
          break;
        }
      }
    }

    config.thread_tile_m = 16;
    config.thread_tile_n = 8;
    config.block_threads_y = 1;
    config.block_threads_x = 32;
    config.block_threads_z = 1;
    // Only use k_tile=32 when block tiles are large enough for sub-tiling.
    // For 16×8 blocks, inner K-tiling adds loop overhead with no reuse benefit.
    const bool needs_subtiling =
        config.block_tile_m > 16 || config.block_tile_n > 8;
    config.k_tile = needs_subtiling ? pickTilingFactor(k, 32) : 16;
    config.mma_micro_k = needs_subtiling ? 16 : 0;
    return config;
  }

  config.thread_tile_m = std::max<std::int64_t>(
      1, pickTilingFactor(config.block_tile_m, 16));
  config.thread_tile_n = std::max<std::int64_t>(
      1, pickTilingFactor(config.block_tile_n, 8));
  config.block_threads_y = std::max<std::int64_t>(
      1, ceilDiv(config.block_tile_m, config.thread_tile_m));
  config.block_threads_x = std::max<std::int64_t>(
      1, ceilDiv(config.block_tile_n, config.thread_tile_n));
  return config;
}

bool UsesSingleWarpMmaSync(const NvidiaMappingConfig &config) {
  return config.rewrite_to_mma_sync;
}

std::string BuildNvidiaTransformMappingSequence(
    const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config) {
  mlir::MLIRContext transform_ctx;
  auto transform_module = BuildNvidiaTransformMappingModule(
      &transform_ctx, mlir::UnknownLoc::get(&transform_ctx), signature, config);
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  transform_module->print(stream);
  stream.flush();
  return ir;
}

std::string BuildNvidiaThreadMappingSequence(
    const NvidiaMappingConfig &config) {
  mlir::MLIRContext transform_ctx;
  auto transform_module = BuildNvidiaThreadMappingModule(
      &transform_ctx, mlir::UnknownLoc::get(&transform_ctx), config);
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  transform_module->print(stream);
  stream.flush();
  return ir;
}

std::string BuildNvidiaMmaRewriteSequence() {
  mlir::MLIRContext transform_ctx;
  auto transform_module = BuildNvidiaMmaRewriteModule(
      &transform_ctx, mlir::UnknownLoc::get(&transform_ctx));
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  transform_module->print(stream);
  stream.flush();
  return ir;
}

namespace {

struct DynamicMacroGridMappingPass
    : public mlir::PassWrapper<DynamicMacroGridMappingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  explicit DynamicMacroGridMappingPass(const NvidiaMappingConfig &config)
      : block_tile_m(config.block_tile_m),
        block_tile_n(config.block_tile_n),
        block_threads_x(config.block_threads_x),
        block_threads_y(config.block_threads_y),
        block_threads_z(config.block_threads_z) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  mlir::LogicalResult lowerForallToLaunch(mlir::func::FuncOp func,
                                          mlir::scf::ForallOp forall) {
    if (!isBlockMappedForall(forall)) {
      return mlir::success();
    }
    if (!forall.getOutputs().empty()) {
      forall.emitError(
          "MatCore dynamic macro topology requires scf.forall without outputs");
      return mlir::failure();
    }

    auto lhs = findRank2MemRefArgument(func, 0);
    auto rhs = findRank2MemRefArgument(func, 1);
    if (!lhs.has_value() || !rhs.has_value()) {
      forall.emitError(
          "MatCore dynamic macro topology requires two rank-2 memref arguments");
      return mlir::failure();
    }

    mlir::OpBuilder bounds_builder(forall);
    mlir::Location loc = forall.getLoc();
    llvm::SmallVector<mlir::Value, 3> lower_bounds =
        forall.getLowerBound(bounds_builder);
    llvm::SmallVector<mlir::Value, 3> steps = forall.getStep(bounds_builder);
    llvm::SmallVector<mlir::Attribute, 3> mapping_attrs;
    mapping_attrs.assign(forall.getMapping()->begin(), forall.getMapping()->end());

    mlir::OpBuilder builder(forall);
    mlir::Value m = builder.create<mlir::memref::DimOp>(loc, *lhs, 0);
    mlir::Value n = builder.create<mlir::memref::DimOp>(loc, *rhs, 1);
    mlir::Value tile_m =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_tile_m);
    mlir::Value tile_n =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_tile_n);
    mlir::Value grid_y = buildCeilDivIndex(builder, loc, m, tile_m);
    mlir::Value grid_x = buildCeilDivIndex(builder, loc, n, tile_n);
    mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value block_x =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_threads_x);
    mlir::Value block_y =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_threads_y);
    mlir::Value block_z =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_threads_z);

    auto launch = builder.create<mlir::gpu::LaunchOp>(
        loc, grid_x, grid_y, one, block_x, block_y, block_z);

    mlir::Block &launch_body = launch.getBody().front();

    if (lower_bounds.size() != forall.getRank() || steps.size() != forall.getRank() ||
        mapping_attrs.size() != static_cast<std::size_t>(forall.getRank())) {
      forall.emitError(
          "MatCore dynamic macro topology found invalid scf.forall bounds");
      return mlir::failure();
    }

    mlir::IRMapping mapping;
    mlir::OpBuilder launch_builder = mlir::OpBuilder::atBlockEnd(&launch_body);
    for (int64_t dim = 0; dim < forall.getRank(); ++dim) {
      auto map_attr =
          llvm::dyn_cast<mlir::gpu::GPUBlockMappingAttr>(mapping_attrs[dim]);
      if (!map_attr) {
        forall.emitError("MatCore expected gpu.block mapping on top-level scf.forall");
        return mlir::failure();
      }
      mlir::Value block_id = blockIdForMapping(launch, map_attr);
      if (!block_id) {
        forall.emitError("MatCore encountered unsupported gpu.block mapping id");
        return mlir::failure();
      }
      mlir::Value scaled =
          launch_builder.create<mlir::arith::MulIOp>(loc, block_id, steps[dim]);
      mlir::Value iv =
          launch_builder.create<mlir::arith::AddIOp>(loc, lower_bounds[dim], scaled);
      mapping.map(forall.getInductionVar(dim), iv);
    }

    for (mlir::Operation &op :
         llvm::make_early_inc_range(forall.getBody()->without_terminator())) {
      launch_builder.clone(op, mapping);
    }
    launch_builder.create<mlir::gpu::TerminatorOp>(loc);
    forall.erase();
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::scf::ForallOp, 4> top_level_foralls;
    for (mlir::Operation &op : func.getBody().front().without_terminator()) {
      auto forall = llvm::dyn_cast<mlir::scf::ForallOp>(&op);
      if (!forall || !isBlockMappedForall(forall)) {
        continue;
      }
      top_level_foralls.push_back(forall);
    }

    for (mlir::scf::ForallOp forall : top_level_foralls) {
      if (mlir::failed(lowerForallToLaunch(func, forall))) {
        signalPassFailure();
        return;
      }
    }
  }

  std::int64_t block_tile_m;
  std::int64_t block_tile_n;
  std::int64_t block_threads_x;
  std::int64_t block_threads_y;
  std::int64_t block_threads_z;
};

struct ConfigureNvidiaLaunchPass
    : public mlir::PassWrapper<ConfigureNvidiaLaunchPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::OpBuilder builder(&getContext());
    func.walk([&](mlir::gpu::LaunchOp launch) {
      auto block_x = matchConstantIndex(launch.getBlockSizeX());
      auto block_y = matchConstantIndex(launch.getBlockSizeY());
      auto block_z = matchConstantIndex(launch.getBlockSizeZ());

      // Preserve any launch geometry already materialized by the mapping
      // passes. Only fill a 32x1x1 fallback for degenerate 1x1x1 launches.
      if (block_x.has_value() && block_y.has_value() && block_z.has_value() &&
          (*block_x != 1 || *block_y != 1 || *block_z != 1)) {
        return;
      }

      builder.setInsertionPoint(launch);
      auto c1 = builder.create<mlir::arith::ConstantIndexOp>(launch.getLoc(), 1);
      auto c32 =
          builder.create<mlir::arith::ConstantIndexOp>(launch.getLoc(), 32);
      launch.getBlockSizeXMutable().set(c32.getResult());
      launch.getBlockSizeYMutable().set(c1.getResult());
      launch.getBlockSizeZMutable().set(c1.getResult());
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateConfigureNvidiaLaunchPass() {
  return std::make_unique<ConfigureNvidiaLaunchPass>();
}

void AddNvidiaLaunchConfigurationPasses(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(CreateConfigureNvidiaLaunchPass());
}

std::unique_ptr<mlir::Pass> CreateNvidiaDynamicMacroGridMappingPass(
    const NvidiaMappingConfig &config) {
  return std::make_unique<DynamicMacroGridMappingPass>(config);
}

void AddNvidiaDynamicMacroGridMappingPasses(mlir::PassManager &pm,
                                            const NvidiaMappingConfig &config) {
  pm.addNestedPass<mlir::func::FuncOp>(
      CreateNvidiaDynamicMacroGridMappingPass(config));
}

}  // namespace matcore
