#include "transform_builder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace matcore {
namespace {

void ensureBuilderDialects(mlir::MLIRContext *ctx) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::transform::TransformDialect, mlir::gpu::GPUDialect>();
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::nvgpu::registerTransformDialectExtension(registry);
  ctx->appendDialectRegistry(registry);
  ctx->loadDialect<mlir::transform::TransformDialect, mlir::gpu::GPUDialect>();
}

std::string nvidiaMatmulOpName(const MatmulLoweringSignature &signature) {
  return signature.quantized_i8 ? "linalg.quantized_matmul" : "linalg.matmul";
}

}  // namespace

TransformBuilder::TransformBuilder(mlir::MLIRContext *ctx, mlir::Location loc)
    : ctx_(ctx), loc_(loc), builder_(ctx) {
  ensureBuilderDialects(ctx_);
}

void TransformBuilder::matchOp(llvm::StringRef opName) {
  const std::string op_name = opName.str();
  steps_.emplace_back([op_name](mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value handle) -> mlir::Value {
    llvm::SmallVector<llvm::StringRef, 1> ops{llvm::StringRef(op_name)};
    auto matched = builder.create<mlir::transform::MatchOp>(loc, handle, ops);
    return matched.getResults();
  });
}

void TransformBuilder::addTileUsingForallStep(mlir::ArrayRef<int64_t> sizes,
                                              bool useNumThreads,
                                              bool useBlockMapping) {
  const std::vector<int64_t> captured_sizes(sizes.begin(), sizes.end());
  steps_.emplace_back([captured_sizes, useNumThreads, useBlockMapping](
                          mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value handle) -> mlir::Value {
    llvm::SmallVector<mlir::Attribute, 2> mapping;
    if (useBlockMapping) {
      mapping.push_back(mlir::gpu::GPUBlockMappingAttr::get(
          builder.getContext(), mlir::gpu::MappingId::DimY));
      mapping.push_back(mlir::gpu::GPUBlockMappingAttr::get(
          builder.getContext(), mlir::gpu::MappingId::DimX));
    } else {
      mapping.push_back(mlir::gpu::GPUThreadMappingAttr::get(
          builder.getContext(), mlir::gpu::MappingId::DimY));
      mapping.push_back(mlir::gpu::GPUThreadMappingAttr::get(
          builder.getContext(), mlir::gpu::MappingId::DimX));
    }

    auto mapping_attr = builder.getArrayAttr(mapping);
    auto tiled = useNumThreads
                     ? builder.create<mlir::transform::TileUsingForallOp>(
                           loc, handle, captured_sizes,
                           mlir::transform::NumThreadsSpec{}, mapping_attr)
                     : builder.create<mlir::transform::TileUsingForallOp>(
                           loc, handle, captured_sizes,
                           mlir::transform::TileSizesSpec{}, mapping_attr);
    return tiled.getTiledOp();
  });
}

void TransformBuilder::tileLinalg(mlir::ArrayRef<int64_t> tileSizes) {
  const std::vector<int64_t> captured_sizes(tileSizes.begin(), tileSizes.end());
  steps_.emplace_back([captured_sizes](mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value handle) -> mlir::Value {
    auto tiled = builder.create<mlir::transform::TileUsingForallOp>(
        loc, handle, captured_sizes, mlir::transform::TileSizesSpec{},
        mlir::ArrayAttr{});
    return tiled.getTiledOp();
  });
}

void TransformBuilder::tileLinalgToGpuBlocks(
    mlir::ArrayRef<int64_t> tileSizes) {
  addTileUsingForallStep(tileSizes, /*useNumThreads=*/false,
                         /*useBlockMapping=*/true);
}

void TransformBuilder::tileLinalgToGpuThreads(
    mlir::ArrayRef<int64_t> numThreads) {
  addTileUsingForallStep(numThreads, /*useNumThreads=*/true,
                         /*useBlockMapping=*/false);
}

void TransformBuilder::tileLinalgToGpuWarps(
    mlir::ArrayRef<int64_t> tileSizes) {
  const std::vector<int64_t> captured_sizes(tileSizes.begin(), tileSizes.end());
  steps_.emplace_back([captured_sizes](mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value handle) -> mlir::Value {
    llvm::SmallVector<mlir::Attribute, 2> mapping;
    mapping.push_back(mlir::gpu::GPUWarpMappingAttr::get(
        builder.getContext(), mlir::gpu::MappingId::DimY));
    mapping.push_back(mlir::gpu::GPUWarpMappingAttr::get(
        builder.getContext(), mlir::gpu::MappingId::DimX));

    auto mapping_attr = builder.getArrayAttr(mapping);
    auto tiled = builder.create<mlir::transform::TileUsingForallOp>(
        loc, handle, captured_sizes, mlir::transform::TileSizesSpec{},
        mapping_attr);
    return tiled.getTiledOp();
  });
}

void TransformBuilder::tileLinalgWithFor(mlir::ArrayRef<int64_t> tileSizes) {
  const std::vector<int64_t> captured_sizes(tileSizes.begin(), tileSizes.end());
  steps_.emplace_back([captured_sizes](mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::Value handle) -> mlir::Value {
    auto tiled =
        builder.create<mlir::transform::TileUsingForOp>(loc, handle, captured_sizes);
    return tiled.getTiledLinalgOp();
  });
}

void TransformBuilder::vectorize() {
  steps_.emplace_back([](mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value handle) -> mlir::Value {
    builder.create<mlir::transform::VectorizeOp>(
        loc, handle, mlir::ValueRange{}, mlir::UnitAttr{},
        llvm::ArrayRef<bool>{}, llvm::ArrayRef<int64_t>{});
    return handle;
  });
}

void TransformBuilder::mapToGpuThreads(mlir::ArrayRef<int64_t> blockDims) {
  const std::vector<int64_t> captured_dims(blockDims.begin(), blockDims.end());
  steps_.emplace_back([captured_dims](mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::Value handle) -> mlir::Value {
    auto mapped = builder.create<mlir::transform::MapNestedForallToThreads>(
        loc, mlir::transform::AnyOpType::get(builder.getContext()), handle,
        captured_dims, /*sync_after_distribute=*/true, /*warp_size=*/32);
    return mapped.getResult();
  });
}

void TransformBuilder::mapToGpuBlocks(mlir::ArrayRef<int64_t> gridDims) {
  const std::vector<int64_t> captured_dims(gridDims.begin(), gridDims.end());
  steps_.emplace_back([captured_dims](mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::Value handle) -> mlir::Value {
    auto mapped = builder.create<mlir::transform::MapForallToBlocks>(
        loc, mlir::transform::AnyOpType::get(builder.getContext()), handle,
        captured_dims, /*generate_gpu_launch=*/false);
    return mapped.getResult();
  });
}

void TransformBuilder::bufferize() {
  steps_.emplace_back([](mlir::OpBuilder &, mlir::Location,
                         mlir::Value) -> mlir::Value {
    llvm::report_fatal_error(
        "TransformBuilder::bufferize is not implemented in MatcoreDSL yet");
  });
}

void TransformBuilder::promoteTensorToSharedMemory() {
  steps_.emplace_back([](mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value handle) -> mlir::Value {
    auto operands_to_promote = builder.getArrayAttr(
        {builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(1)});
    auto use_full_tile_buffers = builder.getArrayAttr({});
    auto memory_space = mlir::gpu::AddressSpaceAttr::get(
        builder.getContext(), mlir::gpu::AddressSpace::Workgroup);
    auto promoted = builder.create<mlir::transform::PromoteOp>(
        loc, mlir::transform::AnyOpType::get(builder.getContext()), handle,
        operands_to_promote, use_full_tile_buffers,
        /*use_full_tiles_by_default=*/true,
        /*use_alloca=*/false,
        memory_space,
        /*mapping=*/mlir::ArrayAttr{},
        /*alignment=*/mlir::IntegerAttr{});
    return promoted.getTransformed();
  });
}

void TransformBuilder::rewriteMatmulAsMmaSync() {
  steps_.emplace_back([](mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value handle) -> mlir::Value {
    builder.create<mlir::transform::RewriteMatmulAsMmaSyncOp>(loc, handle);
    return handle;
  });
}

mlir::OwningOpRef<mlir::ModuleOp> TransformBuilder::build() {
  auto transform_module = mlir::ModuleOp::create(loc_);
  transform_module->setAttr(
      mlir::transform::TransformDialect::kWithNamedSequenceAttrName,
      mlir::UnitAttr::get(ctx_));

  builder_.setInsertionPointToStart(transform_module.getBody());
  builder_.create<mlir::transform::NamedSequenceOp>(
      loc_, mlir::transform::TransformDialect::kTransformEntryPointSymbolName,
      mlir::transform::AnyOpType::get(ctx_), mlir::TypeRange{},
      [&](mlir::OpBuilder &body_builder, mlir::Location body_loc,
          mlir::BlockArgument root) {
        mlir::Value current = root;
        for (const StepFn &step : steps_) {
          current = step(body_builder, body_loc, current);
        }
        body_builder.create<mlir::transform::YieldOp>(body_loc);
      });
  return transform_module;
}

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaTransformMappingModule(
    mlir::MLIRContext *ctx, mlir::Location loc,
    const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config) {
  TransformBuilder transform_builder(ctx, loc);
  transform_builder.matchOp("func.func");
  transform_builder.matchOp(nvidiaMatmulOpName(signature));
  transform_builder.tileLinalgToGpuBlocks(
      {config.block_tile_m, config.block_tile_n, 0});
  transform_builder.tileLinalgWithFor({0, 0, config.k_tile});
  transform_builder.promoteTensorToSharedMemory();

  if (config.rewrite_to_mma_sync) {
    if (config.block_tile_m > 16 || config.block_tile_n > 8 ||
        config.k_tile > 16) {
      transform_builder.tileLinalgWithFor({16, 8, 16});
    }
  } else {
    transform_builder.tileLinalgToGpuThreads(
        {config.block_threads_y, config.block_threads_x});
  }
  return transform_builder.build();
}

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaThreadMappingModule(
    mlir::MLIRContext *ctx, mlir::Location loc,
    const NvidiaMappingConfig &config) {
  TransformBuilder transform_builder(ctx, loc);
  transform_builder.matchOp("gpu.launch");
  transform_builder.mapToGpuThreads(
      {config.block_threads_x, config.block_threads_y, config.block_threads_z});
  return transform_builder.build();
}

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaMmaRewriteModule(
    mlir::MLIRContext *ctx, mlir::Location loc) {
  TransformBuilder transform_builder(ctx, loc);
  transform_builder.matchOp("gpu.launch");
  transform_builder.matchOp("linalg.matmul");
  transform_builder.rewriteMatmulAsMmaSync();
  return transform_builder.build();
}

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaMultiWarpTransformModule(
    mlir::MLIRContext *ctx, mlir::Location loc,
    const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config) {
  TransformBuilder transform_builder(ctx, loc);
  transform_builder.matchOp("func.func");
  transform_builder.matchOp(nvidiaMatmulOpName(signature));

  // Step 1: Block tiling — partition output across threadblocks
  transform_builder.tileLinalgToGpuBlocks(
      {config.block_tile_m, config.block_tile_n, 0});

  // Step 2: K reduction loop
  transform_builder.tileLinalgWithFor({0, 0, config.k_tile});

  // Step 3: Shared memory promotion for A and B tiles
  transform_builder.promoteTensorToSharedMemory();

  // Step 4: Warp tiling — split block work across warps (e.g. 2×2 = 4 warps)
  transform_builder.tileLinalgToGpuWarps(
      {config.warp_tile_m, config.warp_tile_n, 0});

  // Step 5: Sub-tile to MMA-compatible size (16×8×16 for mma.sync.m16n8k16)
  // k_tile>16 requires K sub-tiling to match MMA micro-K dimension.
  transform_builder.tileLinalgWithFor({16, 8, config.mma_micro_k});

  // Step 6: Vectorize the inner 16×8×K matmul → vector.contract
  if (config.use_vectorize_path) {
    transform_builder.vectorize();
  }

  return transform_builder.build();
}

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaMultiWarpThreadMappingModule(
    mlir::MLIRContext *ctx, mlir::Location loc,
    const NvidiaMappingConfig &config) {
  TransformBuilder transform_builder(ctx, loc);
  transform_builder.matchOp("gpu.launch");
  // Map warp foralls to thread indices.
  // block_dims must match the warp topology:
  //   16 warps (4×4): block_dims = {128, 4, 1} = 512 threads
  //    4 warps (2×2): block_dims = {64, 2, 1} = 128 threads
  transform_builder.mapToGpuThreads(
      {config.block_threads_x, config.block_threads_y,
       config.block_threads_z});
  return transform_builder.build();
}

}  // namespace matcore
