#include "matcore/gpu_mapping.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <string>

#include "transform_builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
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

std::int64_t pickMmaMacroKTile(std::int64_t k) {
  return std::max<std::int64_t>(16, pickTilingFactor(k, 64));
}

bool isTensorCoreMmaSyncType(const MatmulLoweringSignature &signature) {
  return signature.lhs_dtype == TensorDType::kFloat16 &&
         signature.rhs_dtype == TensorDType::kFloat16 &&
         signature.out_dtype == TensorDType::kFloat16;
}

bool isStaticallyCompatibleMmaSync(std::int64_t m, std::int64_t n,
                                   std::int64_t k) {
  if (m == mlir::ShapedType::kDynamic || n == mlir::ShapedType::kDynamic ||
      k == mlir::ShapedType::kDynamic) {
    return false;
  }
  if (m < 16 || n < 8 || k < 16) {
    return false;
  }
  return (m % 16) == 0 && (n % 8) == 0 && (k % 16) == 0;
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

bool isThreadMappedForall(mlir::scf::ForallOp forall) {
  auto mapping = forall.getMapping();
  if (!mapping || mapping->empty()) {
    return false;
  }
  for (mlir::Attribute attr : *mapping) {
    if (!llvm::isa<mlir::gpu::GPUThreadMappingAttr>(attr)) {
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

/// Compute split_k_factor for under-filled grids.
/// Partitions K across gridDim.z so more blocks can run in parallel.
/// Returns 1 (no split) if the grid already saturates SMs.
void computeSplitKFactor(NvidiaMappingConfig &config,
                         int64_t m, int64_t n, int64_t k,
                         int64_t sm_count) {
  int64_t grid_mn = ceilDiv(m, config.block_tile_m) *
                    ceilDiv(n, config.block_tile_n);
  int64_t k_tiles = k / config.k_tile;

  // Only split if grid doesn't fill SMs and K has enough tiles
  if (grid_mn >= sm_count || k_tiles < 4) {
    config.split_k_factor = 1;
    return;
  }

  // Target: 1 block per SM minimum (diminishing returns beyond that
  // because smem limits us to ~2 blocks/SM for current configs)
  int64_t desired = ceilDiv(sm_count, grid_mn);

  // Prefer ≥3 K-tiles per slice for good double-buffer overlap
  int64_t max_split_profitable = k_tiles / 3;
  int64_t split_k = std::min(desired, max_split_profitable);

  // Relax to ≥2 K-tiles per slice if ≥3 doesn't yield a useful split
  if (split_k < 2) {
    int64_t max_split_correct = k_tiles / 2;
    split_k = std::min(desired, max_split_correct);
  }

  // Ensure split_k divides k_tiles evenly (avoid remainder slices)
  while (split_k > 1 && (k_tiles % split_k) != 0)
    split_k--;

  config.split_k_factor = (split_k >= 2) ? split_k : 1;

  if (config.split_k_factor > 1) {
    int64_t total_blocks = grid_mn * config.split_k_factor;
    int64_t tiles_per_slice = k_tiles / config.split_k_factor;
    fprintf(stderr,
            "[SplitK] grid_mn=%lld, k_tiles=%lld → split_k=%lld "
            "(total_blocks=%lld, %lld K-tiles/slice)\n",
            (long long)grid_mn, (long long)k_tiles,
            (long long)config.split_k_factor,
            (long long)total_blocks, (long long)tiles_per_slice);
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
  config.rewrite_to_mma_sync = statically_compatible_mma;
  if (config.rewrite_to_mma_sync) {
    // --- Multi-warp path (V4): adaptive tile selection ---
    // Two MW configs available:
    //   128×128 / 16 warps (4×4): highest AI (64 FLOP/byte), 512 threads
    //    64×64  /  4 warps (2×2): lower AI (32 FLOP/byte), 128 threads
    // Use 128×128 when grid fills SMs (≥ kSmCount); otherwise fall back to
    // 64×64 which generates 4× more blocks for better SM utilization.
    constexpr int64_t kSmCount = 24;  // RTX 4060 Laptop (AD107, sm_89)

    // Try 128×128 / 16-warp first (highest arithmetic intensity)
    constexpr int64_t kMW128Tile = 128;
    const bool eligible_128 =
        m >= kMW128Tile && n >= kMW128Tile &&
        (m % kMW128Tile) == 0 && (n % kMW128Tile) == 0;
    if (eligible_128) {
      auto grid = (m / kMW128Tile) * (n / kMW128Tile);
      if (grid >= kSmCount) {
        config.block_tile_m = kMW128Tile;
        config.block_tile_n = kMW128Tile;
        config.num_warps = 16;               // 4×4 warp layout
        config.warp_tile_m = 32;             // Each warp: 32×32
        config.warp_tile_n = 32;
        config.k_tile = pickTilingFactor(k, 32);
        config.mma_micro_k = 16;
        config.use_vectorize_path = false;
        config.block_threads_x = 128;        // 128×4×1 = 512 threads
        config.block_threads_y = 4;
        config.block_threads_z = 1;
        config.thread_tile_m = 16;
        config.thread_tile_n = 8;
        config.sm_count = kSmCount;
        computeSplitKFactor(config, m, n, k, kSmCount);
        return config;
      }
    }

    // Fall back to 64×64 / 4-warp for better SM utilization on small grids
    constexpr int64_t kMW64Tile = 64;
    constexpr int64_t kMW64MinGrid = 4;
    const bool eligible_64 =
        m >= kMW64Tile && n >= kMW64Tile &&
        (m % kMW64Tile) == 0 && (n % kMW64Tile) == 0;
    if (eligible_64) {
      auto grid = (m / kMW64Tile) * (n / kMW64Tile);
      if (grid >= kMW64MinGrid) {
        config.block_tile_m = kMW64Tile;
        config.block_tile_n = kMW64Tile;
        config.num_warps = 4;                // 2×2 warp layout
        config.warp_tile_m = 32;             // Each warp: 32×32
        config.warp_tile_n = 32;
        config.k_tile = pickTilingFactor(k, 32);
        config.mma_micro_k = 16;
        config.use_vectorize_path = false;
        config.block_threads_x = 64;         // 64×2×1 = 128 threads
        config.block_threads_y = 2;
        config.block_threads_z = 1;
        config.thread_tile_m = 16;
        config.thread_tile_n = 8;
        config.sm_count = kSmCount;
        computeSplitKFactor(config, m, n, k, kSmCount);
        return config;
      }
    }

    // --- Single-warp fallback (V3): occupancy-aware adaptive tiling ---
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
  return config.rewrite_to_mma_sync && config.num_warps <= 1;
}

bool UsesMultiWarpMmaSync(const NvidiaMappingConfig &config) {
  return config.rewrite_to_mma_sync && config.num_warps > 1;
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

std::string BuildNvidiaAsyncPipelineSequence() {
  std::ostringstream ir;
  ir << "module attributes {transform.with_named_sequence} {\n";
  ir << "  transform.named_sequence @__transform_main"
        "(%root: !transform.any_op) {\n";
  ir << "    %launch = transform.structured.match ops{[\"gpu.launch\"]} in %root"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    %async = transform.nvgpu.create_async_groups %launch {bypass_l1}"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    %loops = transform.structured.match ops{[\"scf.for\"]}"
        " attributes {matcore.async_k_loop} in %async"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    %pipelined = transform.nvgpu.pipeline_shared_memory_copies"
        " failures(suppress) %loops {depth = 2, peel_epilogue}"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    transform.yield\n";
  ir << "  }\n";
  ir << "}\n";
  return ir.str();
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

    mlir::OpBuilder bounds_builder(forall);
    mlir::Location loc = forall.getLoc();
    llvm::SmallVector<mlir::Value, 3> lower_bounds =
        forall.getLowerBound(bounds_builder);
    llvm::SmallVector<mlir::Value, 3> upper_bounds =
        forall.getUpperBound(bounds_builder);
    llvm::SmallVector<mlir::Value, 3> steps = forall.getStep(bounds_builder);
    llvm::SmallVector<mlir::Attribute, 3> mapping_attrs;
    mapping_attrs.assign(forall.getMapping()->begin(), forall.getMapping()->end());

    if (lower_bounds.size() != forall.getRank() ||
        upper_bounds.size() != forall.getRank() ||
        steps.size() != forall.getRank() ||
        mapping_attrs.size() != static_cast<std::size_t>(forall.getRank())) {
      forall.emitError(
          "MatCore dynamic macro topology found invalid scf.forall bounds");
      return mlir::failure();
    }

    mlir::OpBuilder builder(forall);
    mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value grid_x = one;
    mlir::Value grid_y = one;
    mlir::Value grid_z = one;
    for (int64_t dim = 0; dim < forall.getRank(); ++dim) {
      auto map_attr =
          llvm::dyn_cast<mlir::gpu::GPUBlockMappingAttr>(mapping_attrs[dim]);
      if (!map_attr) {
        forall.emitError("MatCore expected gpu.block mapping on top-level scf.forall");
        return mlir::failure();
      }
      mlir::Value extent = builder.create<mlir::arith::SubIOp>(
          loc, upper_bounds[dim], lower_bounds[dim]);
      mlir::Value grid_dim = buildCeilDivIndex(builder, loc, extent, steps[dim]);
      switch (map_attr.getRelativeIndex()) {
        case 0:
          grid_x = grid_dim;
          break;
        case 1:
          grid_y = grid_dim;
          break;
        case 2:
          grid_z = grid_dim;
          break;
        default:
          forall.emitError(
              "MatCore encountered unsupported gpu.block mapping dimension");
          return mlir::failure();
      }
    }
    mlir::Value block_x =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_threads_x);
    mlir::Value block_y =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_threads_y);
    mlir::Value block_z =
        builder.create<mlir::arith::ConstantIndexOp>(loc, block_threads_z);

    auto launch = builder.create<mlir::gpu::LaunchOp>(
        loc, grid_x, grid_y, grid_z, block_x, block_y, block_z);

    mlir::Block &launch_body = launch.getBody().front();

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

  mlir::LogicalResult lowerThreadForallToPredicatedRegion(
      mlir::gpu::LaunchOp launch, mlir::scf::ForallOp forall) {
    if (!isThreadMappedForall(forall)) {
      return mlir::success();
    }
    if (!forall.getOutputs().empty()) {
      forall.emitError(
          "MatCore warp CTA lowering requires scf.forall without outputs");
      return mlir::failure();
    }

    mlir::OpBuilder bounds_builder(forall);
    mlir::Location loc = forall.getLoc();
    llvm::SmallVector<mlir::Value, 3> lower_bounds =
        forall.getLowerBound(bounds_builder);
    llvm::SmallVector<mlir::Value, 3> upper_bounds =
        forall.getUpperBound(bounds_builder);
    llvm::SmallVector<mlir::Value, 3> steps = forall.getStep(bounds_builder);
    llvm::SmallVector<mlir::Attribute, 3> mapping_attrs;
    mapping_attrs.assign(forall.getMapping()->begin(), forall.getMapping()->end());

    if (lower_bounds.size() != forall.getRank() ||
        upper_bounds.size() != forall.getRank() ||
        steps.size() != forall.getRank() ||
        mapping_attrs.size() != static_cast<std::size_t>(forall.getRank())) {
      forall.emitError("MatCore warp CTA lowering found invalid scf.forall bounds");
      return mlir::failure();
    }

    mlir::OpBuilder builder(forall);
    mlir::Value active;
    mlir::IRMapping mapping;
    for (int64_t dim = 0; dim < forall.getRank(); ++dim) {
      auto map_attr =
          llvm::dyn_cast<mlir::gpu::GPUThreadMappingAttr>(mapping_attrs[dim]);
      if (!map_attr) {
        forall.emitError("MatCore expected gpu.thread mapping on nested scf.forall");
        return mlir::failure();
      }
      mlir::Value thread_id = threadIdForMapping(launch, map_attr);
      if (!thread_id) {
        forall.emitError("MatCore encountered unsupported gpu.thread mapping id");
        return mlir::failure();
      }
      mlir::Value extent = builder.create<mlir::arith::SubIOp>(
          loc, upper_bounds[dim], lower_bounds[dim]);
      mlir::Value trip_count =
          buildCeilDivIndex(builder, loc, extent, steps[dim]);
      mlir::Value block_size = blockSizeForThreadMapping(launch, map_attr);
      auto lower_const = matchConstantIndex(lower_bounds[dim]);
      auto upper_const = matchConstantIndex(upper_bounds[dim]);
      auto step_const = matchConstantIndex(steps[dim]);
      std::optional<std::int64_t> trip_count_const;
      if (lower_const.has_value() && upper_const.has_value() &&
          step_const.has_value()) {
        trip_count_const =
            ceilDiv(*upper_const - *lower_const, *step_const);
      }
      auto block_size_const = block_size ? matchConstantIndex(block_size)
                                         : std::optional<std::int64_t>();
      const bool fully_covered =
          trip_count_const.has_value() && block_size_const.has_value() &&
          *trip_count_const == *block_size_const;
      if (!fully_covered) {
        mlir::Value in_range = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, thread_id, trip_count);
        active = active ? builder.create<mlir::arith::AndIOp>(loc, active, in_range)
                        : in_range;
      }
      mlir::Value scaled =
          builder.create<mlir::arith::MulIOp>(loc, thread_id, steps[dim]);
      mlir::Value iv =
          builder.create<mlir::arith::AddIOp>(loc, lower_bounds[dim], scaled);
      mapping.map(forall.getInductionVar(dim), iv);
    }

    if (active) {
      auto if_op = builder.create<mlir::scf::IfOp>(loc, active,
                                                   /*withElseRegion=*/false);
      mlir::OpBuilder then_builder =
          mlir::OpBuilder::atBlockTerminator(&if_op.getThenRegion().front());
      for (mlir::Operation &op :
           llvm::make_early_inc_range(forall.getBody()->without_terminator())) {
        then_builder.clone(op, mapping);
      }
      builder.setInsertionPointAfter(if_op);
    } else {
      for (mlir::Operation &op :
           llvm::make_early_inc_range(forall.getBody()->without_terminator())) {
        builder.clone(op, mapping);
      }
    }
    builder.create<mlir::gpu::BarrierOp>(loc);
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

    llvm::SmallVector<mlir::scf::ForallOp, 8> thread_foralls;
    func.walk([&](mlir::gpu::LaunchOp launch) {
      launch.walk([&](mlir::scf::ForallOp forall) {
        if (isThreadMappedForall(forall)) {
          thread_foralls.push_back(forall);
        }
      });
    });
    for (mlir::scf::ForallOp forall : thread_foralls) {
      auto launch = forall->getParentOfType<mlir::gpu::LaunchOp>();
      if (!launch) {
        forall.emitError("MatCore expected nested thread forall inside gpu.launch");
        signalPassFailure();
        return;
      }
      if (mlir::failed(lowerThreadForallToPredicatedRegion(launch, forall))) {
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
      launch->setAttr(
          kMatcoreStaticBlockSizeAttr,
          builder.getArrayAttr({
              builder.getI64IntegerAttr(32),
              builder.getI64IntegerAttr(1),
              builder.getI64IntegerAttr(1),
          }));
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
