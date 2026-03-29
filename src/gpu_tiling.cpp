#include "matcore/gpu_tiling.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore GPU tiling: " + message);
}

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
  if (rhs <= 0 || lhs <= 0) {
    return 1;
  }
  return (lhs + rhs - 1) / rhs;
}

std::optional<std::int64_t> matchConstantIndex(mlir::Value value) {
  llvm::APInt constant;
  if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant))) {
    return std::nullopt;
  }
  return constant.getSExtValue();
}

std::optional<mlir::MemRefType> inferStaticViewType(mlir::memref::ViewOp view) {
  auto view_type = llvm::dyn_cast<mlir::MemRefType>(view.getType());
  if (!view_type || !view_type.hasRank()) {
    return std::nullopt;
  }
  if (view_type.hasStaticShape()) {
    return view_type;
  }

  auto byte_shift = matchConstantIndex(view.getByteShift());
  if (!byte_shift.has_value() || *byte_shift != 0) {
    return std::nullopt;
  }

  llvm::SmallVector<std::int64_t, 4> shape;
  shape.reserve(view_type.getRank());
  for (mlir::Value size : view.getSizes()) {
    auto constant_size = matchConstantIndex(size);
    if (!constant_size.has_value() || *constant_size < 0) {
      return std::nullopt;
    }
    shape.push_back(*constant_size);
  }

  return mlir::MemRefType::get(shape, view_type.getElementType(),
                               mlir::AffineMap(), view_type.getMemorySpace());
}

std::optional<std::int64_t> getConstantIndexFromOpFoldResult(
    mlir::OpFoldResult ofr) {
  if (auto attr = llvm::dyn_cast<mlir::Attribute>(ofr)) {
    if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      return int_attr.getInt();
    }
  }
  if (auto value = llvm::dyn_cast<mlir::Value>(ofr)) {
    return matchConstantIndex(value);
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<std::int64_t, 4>> inferStaticShape(mlir::Value value) {
  if (auto type = llvm::dyn_cast<mlir::MemRefType>(value.getType())) {
    if (type.hasStaticShape()) {
      return llvm::SmallVector<std::int64_t, 4>(type.getShape().begin(),
                                                type.getShape().end());
    }
  }

  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
    return inferStaticShape(cast.getSource());
  }
  if (auto view = value.getDefiningOp<mlir::memref::ViewOp>()) {
    auto static_view_type = inferStaticViewType(view);
    if (!static_view_type.has_value()) {
      return std::nullopt;
    }
    return llvm::SmallVector<std::int64_t, 4>((*static_view_type).getShape().begin(),
                                              (*static_view_type).getShape().end());
  }
  return std::nullopt;
}

bool isSameBufferOrFullSubview(mlir::Value maybe_subview, mlir::Value base) {
  if (maybe_subview == base) {
    return true;
  }
  auto subview = maybe_subview.getDefiningOp<mlir::memref::SubViewOp>();
  if (!subview) {
    return false;
  }

  if (subview.getSource() != base) {
    if (auto cast = subview.getSource().getDefiningOp<mlir::memref::CastOp>()) {
      if (cast.getSource() != base) {
        return false;
      }
    } else {
      return false;
    }
  }

  auto subview_type = llvm::dyn_cast<mlir::MemRefType>(subview.getType());
  if (!subview_type) {
    return false;
  }
  const unsigned rank = subview_type.getRank();
  auto base_shape = inferStaticShape(base);
  if (!base_shape.has_value() || static_cast<std::size_t>(rank) != base_shape->size()) {
    return false;
  }

  auto mixed_offsets = subview.getMixedOffsets();
  auto mixed_sizes = subview.getMixedSizes();
  auto mixed_strides = subview.getMixedStrides();
  for (unsigned i = 0; i < rank; ++i) {
    auto offset = getConstantIndexFromOpFoldResult(mixed_offsets[i]);
    auto size = getConstantIndexFromOpFoldResult(mixed_sizes[i]);
    auto stride = getConstantIndexFromOpFoldResult(mixed_strides[i]);
    if (!offset.has_value() || !size.has_value() || !stride.has_value()) {
      return false;
    }
    if (*offset != 0 || *stride != 1 || *size != (*base_shape)[i]) {
      return false;
    }
  }
  return true;
}

bool opMayReadBuffer(mlir::Operation *op, mlir::Value buffer) {
  if (auto fill = llvm::dyn_cast<mlir::linalg::FillOp>(op)) {
    for (mlir::Value out : fill.getOutputs()) {
      if (isSameBufferOrFullSubview(out, buffer)) {
        return false;
      }
    }
  }
  if (auto copy = llvm::dyn_cast<mlir::linalg::CopyOp>(op)) {
    for (mlir::Value out : copy.getOutputs()) {
      if (isSameBufferOrFullSubview(out, buffer)) {
        return false;
      }
    }
  }

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects) {
    return false;
  }

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> effect_instances;
  effects.getEffectsOnValue(buffer, effect_instances);
  for (const auto &effect : effect_instances) {
    if (llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
      return true;
    }
  }
  return false;
}

bool copyFullyOverwritesBuffer(mlir::linalg::CopyOp copy, mlir::Value buffer) {
  for (mlir::Value out : copy.getOutputs()) {
    if (isSameBufferOrFullSubview(out, buffer)) {
      return true;
    }
  }
  return false;
}

void collectValueDefiningOpsInBlock(mlir::Value value, mlir::Block *block,
                                    llvm::SmallPtrSetImpl<mlir::Operation *> &ops) {
  mlir::Operation *def = value.getDefiningOp();
  if (def == nullptr || def->getBlock() != block ||
      llvm::isa<mlir::gpu::ThreadIdOp>(def)) {
    return;
  }
  if (!ops.insert(def).second) {
    return;
  }
  for (mlir::Value operand : def->getOperands()) {
    collectValueDefiningOpsInBlock(operand, block, ops);
  }
}

void collectForwardUsersInBlock(mlir::Value value, mlir::Block *block,
                                llvm::SmallPtrSetImpl<mlir::Operation *> &ops) {
  for (mlir::Operation *user : value.getUsers()) {
    if (user->getBlock() != block) {
      continue;
    }
    if (!ops.insert(user).second) {
      continue;
    }
    for (mlir::Value result : user->getResults()) {
      collectForwardUsersInBlock(result, block, ops);
    }
  }
}

bool findUniqueAccumulatorBuffer(mlir::Value value, mlir::Block *block,
                                 llvm::SmallPtrSetImpl<mlir::Operation *> &visited,
                                 mlir::Value &buffer) {
  mlir::Operation *def = value.getDefiningOp();
  if (def == nullptr || def->getBlock() != block) {
    return true;
  }
  if (!visited.insert(def).second) {
    return true;
  }

  if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(def)) {
    if (!buffer) {
      buffer = load.getMemRef();
      return true;
    }
    return buffer == load.getMemRef();
  }

  for (mlir::Value operand : def->getOperands()) {
    if (!findUniqueAccumulatorBuffer(operand, block, visited, buffer)) {
      return false;
    }
  }
  return true;
}

struct MmaAccumulatorIndices {
  mlir::Value row0;
  mlir::Value row1;
  mlir::Value col0;
  mlir::Value col1;
};

MmaAccumulatorIndices buildMmaAccumulatorIndices(mlir::OpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::Value thread_id) {
  auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
  auto c4 = builder.create<mlir::arith::ConstantIndexOp>(loc, 4);
  auto c8 = builder.create<mlir::arith::ConstantIndexOp>(loc, 8);
  auto row0 = builder.create<mlir::arith::DivUIOp>(loc, thread_id, c4);
  auto row1 = builder.create<mlir::arith::AddIOp>(loc, row0, c8);
  auto doubled_thread = builder.create<mlir::arith::MulIOp>(loc, thread_id, c2);
  auto row0_stride = builder.create<mlir::arith::MulIOp>(loc, row0, c8);
  auto col0 = builder.create<mlir::arith::SubIOp>(loc, doubled_thread, row0_stride);
  auto col1 = builder.create<mlir::arith::AddIOp>(loc, col0, c1);
  return {
      row0.getResult(),
      row1.getResult(),
      col0.getResult(),
      col1.getResult(),
  };
}

mlir::Value buildAccumulatorFragmentFromWorkgroup(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value thread_id) {
  auto buffer_type = llvm::dyn_cast<mlir::MemRefType>(buffer.getType());
  auto element_type = buffer_type.getElementType();
  auto vector_type = mlir::VectorType::get({2, 2}, element_type);
  auto indices = buildMmaAccumulatorIndices(builder, loc, thread_id);
  auto load00 =
      builder.create<mlir::memref::LoadOp>(loc, buffer, mlir::ValueRange{indices.row0, indices.col0});
  auto load01 =
      builder.create<mlir::memref::LoadOp>(loc, buffer, mlir::ValueRange{indices.row0, indices.col1});
  auto load10 =
      builder.create<mlir::memref::LoadOp>(loc, buffer, mlir::ValueRange{indices.row1, indices.col0});
  auto load11 =
      builder.create<mlir::memref::LoadOp>(loc, buffer, mlir::ValueRange{indices.row1, indices.col1});
  auto initial = builder.create<mlir::vector::SplatOp>(loc, vector_type, load00.getResult());
  auto with00 =
      builder.create<mlir::vector::InsertOp>(loc, load00.getResult(), initial.getResult(),
                                             llvm::ArrayRef<int64_t>{0, 0});
  auto with01 =
      builder.create<mlir::vector::InsertOp>(loc, load01.getResult(), with00.getResult(),
                                             llvm::ArrayRef<int64_t>{0, 1});
  auto with10 =
      builder.create<mlir::vector::InsertOp>(loc, load10.getResult(), with01.getResult(),
                                             llvm::ArrayRef<int64_t>{1, 0});
  auto with11 =
      builder.create<mlir::vector::InsertOp>(loc, load11.getResult(), with10.getResult(),
                                             llvm::ArrayRef<int64_t>{1, 1});
  return with11.getResult();
}

void storeAccumulatorFragmentToWorkgroup(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value buffer,
                                         mlir::Value thread_id,
                                         mlir::Value accumulator) {
  auto indices = buildMmaAccumulatorIndices(builder, loc, thread_id);
  auto extract00 =
      builder.create<mlir::vector::ExtractOp>(loc, accumulator, llvm::ArrayRef<int64_t>{0, 0});
  auto extract01 =
      builder.create<mlir::vector::ExtractOp>(loc, accumulator, llvm::ArrayRef<int64_t>{0, 1});
  auto extract10 =
      builder.create<mlir::vector::ExtractOp>(loc, accumulator, llvm::ArrayRef<int64_t>{1, 0});
  auto extract11 =
      builder.create<mlir::vector::ExtractOp>(loc, accumulator, llvm::ArrayRef<int64_t>{1, 1});
  builder.create<mlir::memref::StoreOp>(loc, extract00.getResult(), buffer,
                                        mlir::ValueRange{indices.row0, indices.col0});
  builder.create<mlir::memref::StoreOp>(loc, extract01.getResult(), buffer,
                                        mlir::ValueRange{indices.row0, indices.col1});
  builder.create<mlir::memref::StoreOp>(loc, extract10.getResult(), buffer,
                                        mlir::ValueRange{indices.row1, indices.col0});
  builder.create<mlir::memref::StoreOp>(loc, extract11.getResult(), buffer,
                                        mlir::ValueRange{indices.row1, indices.col1});
}

mlir::BlockArgument addLaunchWorkgroupAttribution(mlir::gpu::LaunchOp launch,
                                                  mlir::Type type,
                                                  mlir::Location loc) {
  mlir::Block &body = launch.getBody().front();
  const unsigned index =
      launch.getNumConfigRegionAttributes() + launch.getNumWorkgroupAttributions();
  mlir::BlockArgument argument = body.insertArgument(index, type, loc);

  auto existing_count = launch->getAttrOfType<mlir::IntegerAttr>(
      mlir::gpu::LaunchOp::getNumWorkgroupAttributionsAttrName());
  const std::int64_t next_count = (existing_count ? existing_count.getInt() : 0) + 1;
  auto count_type = existing_count
                        ? existing_count.getType()
                        : mlir::IntegerType::get(launch.getContext(), 64);
  launch->setAttr(mlir::gpu::LaunchOp::getNumWorkgroupAttributionsAttrName(),
                  mlir::IntegerAttr::get(count_type, next_count));
  return argument;
}

struct PromoteGpuWorkgroupAllocationsPass
    : public mlir::PassWrapper<PromoteGpuWorkgroupAllocationsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::gpu::LaunchOp, 4> launches;
    func.walk([&](mlir::gpu::LaunchOp launch) { launches.push_back(launch); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::gpu::LaunchOp launch : launches) {
      llvm::SmallVector<mlir::memref::AllocOp, 4> workgroup_allocs;
      launch.walk([&](mlir::memref::AllocOp alloc) {
        if (alloc->getParentOfType<mlir::gpu::LaunchOp>() != launch) {
          return;
        }
        if (IsWorkgroupMemRefType(alloc.getType())) {
          workgroup_allocs.push_back(alloc);
        }
      });

      if (workgroup_allocs.empty()) {
        continue;
      }

      builder.setInsertionPointToStart(&launch.getBody().front());
      for (mlir::memref::AllocOp alloc : workgroup_allocs) {
        if (!alloc->use_empty() && alloc->hasOneUse() &&
            llvm::isa<mlir::memref::DeallocOp>(*alloc->user_begin())) {
          alloc.emitError("MatCore cannot legalize a dead workgroup allocation");
          signalPassFailure();
          return;
        }

        llvm::SmallVector<mlir::memref::DeallocOp, 2> deallocs;
        llvm::SmallVector<mlir::memref::ViewOp, 4> views;
        llvm::SmallVector<mlir::Operation *, 4> non_dealloc_non_view_users;
        for (mlir::Operation *user : alloc->getUsers()) {
          if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(user)) {
            deallocs.push_back(dealloc);
            continue;
          }
          if (auto candidate_view = llvm::dyn_cast<mlir::memref::ViewOp>(user)) {
            views.push_back(candidate_view);
            continue;
          }
          non_dealloc_non_view_users.push_back(user);
        }

        if (!views.empty()) {
          auto static_view_type = inferStaticViewType(views.front());
          if (!static_view_type.has_value()) {
            views.front().emitError(
                "MatCore requires statically sized promoted workgroup views");
            signalPassFailure();
            return;
          }

          for (mlir::memref::ViewOp view : llvm::drop_begin(views)) {
            auto candidate_type = inferStaticViewType(view);
            if (!candidate_type.has_value()) {
              view.emitError("MatCore requires statically sized promoted workgroup views");
              signalPassFailure();
              return;
            }
            if (*candidate_type != *static_view_type) {
              view.emitError(
                  "MatCore requires compatible static workgroup views for promotion");
              signalPassFailure();
              return;
            }
          }

          mlir::BlockArgument attribution =
              addLaunchWorkgroupAttribution(launch, *static_view_type, alloc.getLoc());

          for (mlir::memref::ViewOp view : views) {
            mlir::Value replacement = attribution;
            if (attribution.getType() != view.getType()) {
              replacement = builder.create<mlir::memref::CastOp>(
                  view.getLoc(), view.getType(), attribution);
            }
            view.getResult().replaceAllUsesWith(replacement);
            view.erase();
          }

          if (!non_dealloc_non_view_users.empty()) {
            if (!alloc.getType().hasStaticShape()) {
              alloc.emitError(
                  "MatCore requires static alloc type when replacing direct promoted uses");
              signalPassFailure();
              return;
            }

            mlir::Value replacement = attribution;
            if (attribution.getType() != alloc.getType()) {
              replacement = builder.create<mlir::memref::CastOp>(
                  alloc.getLoc(), alloc.getType(), attribution);
            }

            llvm::SmallVector<mlir::OpOperand *, 8> alloc_uses;
            for (mlir::OpOperand &use : alloc->getUses()) {
              if (!llvm::isa<mlir::memref::DeallocOp>(use.getOwner()) &&
                  !llvm::isa<mlir::memref::ViewOp>(use.getOwner())) {
                alloc_uses.push_back(&use);
              }
            }
            for (mlir::OpOperand *use : alloc_uses) {
              use->set(replacement);
            }
          }
        } else {
          if (!alloc.getType().hasStaticShape()) {
            alloc.emitError("MatCore requires statically shaped workgroup allocations");
            signalPassFailure();
            return;
          }
          mlir::BlockArgument attribution =
              addLaunchWorkgroupAttribution(launch, alloc.getType(), alloc.getLoc());
          llvm::SmallVector<mlir::OpOperand *, 8> non_dealloc_uses;
          for (mlir::OpOperand &use : alloc->getUses()) {
            if (!llvm::isa<mlir::memref::DeallocOp>(use.getOwner())) {
              non_dealloc_uses.push_back(&use);
            }
          }
          for (mlir::OpOperand *use : non_dealloc_uses) {
            use->set(attribution);
          }
        }

        for (mlir::memref::DeallocOp dealloc : deallocs) {
          dealloc.erase();
        }
        if (alloc->use_empty()) {
          alloc.erase();
        }
      }
    }
  }
};

struct SpecializeNvidiaWorkgroupMatmulOperandsPass
    : public mlir::PassWrapper<SpecializeNvidiaWorkgroupMatmulOperandsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::linalg::MatmulOp, 4> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::linalg::MatmulOp op : matmuls) {
      llvm::SmallVector<mlir::Value, 2> specialized_inputs;
      llvm::SmallVector<mlir::Value, 1> specialized_outputs;
      bool changed = false;
      for (mlir::Value input : op.getInputs()) {
        mlir::Value replacement = input;
        if (auto cast = input.getDefiningOp<mlir::memref::CastOp>()) {
          auto source_type = llvm::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
          auto result_type = llvm::dyn_cast<mlir::MemRefType>(cast.getType());
          if (source_type && result_type && source_type.hasStaticShape() &&
              !result_type.hasStaticShape() &&
              IsWorkgroupMemorySpace(source_type.getMemorySpace())) {
            replacement = cast.getSource();
            changed = true;
          }
        }
        specialized_inputs.push_back(replacement);
      }

      for (mlir::Value output : op.getOutputs()) {
        mlir::Value replacement = output;
        if (auto cast = output.getDefiningOp<mlir::memref::CastOp>()) {
          auto source_type = llvm::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
          auto result_type = llvm::dyn_cast<mlir::MemRefType>(cast.getType());
          if (source_type && result_type && source_type.hasStaticShape() &&
              !result_type.hasStaticShape() &&
              IsWorkgroupMemorySpace(source_type.getMemorySpace())) {
            replacement = cast.getSource();
            changed = true;
          }
        }
        specialized_outputs.push_back(replacement);
      }

      if (!changed) {
        continue;
      }

      builder.setInsertionPoint(op);
      auto replacement = builder.create<mlir::linalg::MatmulOp>(
          op.getLoc(), mlir::ValueRange(specialized_inputs),
          mlir::ValueRange(specialized_outputs));
      op->replaceAllUsesWith(replacement->getResults());
      op.erase();
    }
  }
};

struct PersistNvidiaMmaAccumulatorPass
    : public mlir::PassWrapper<PersistNvidiaMmaAccumulatorPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect, mlir::nvgpu::NVGPUDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::scf::ForOp, 8> loops;
    func.walk([&](mlir::scf::ForOp loop) { loops.push_back(loop); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::scf::ForOp loop : loops) {
      if (!loop->getBlock()) {
        continue;
      }
      mlir::Block *body = loop.getBody();
      if (body == nullptr) {
        continue;
      }

      mlir::gpu::ThreadIdOp thread_id_op;
      mlir::nvgpu::MmaSyncOp mma_sync_op;
      for (mlir::Operation &op : body->without_terminator()) {
        if (auto candidate = llvm::dyn_cast<mlir::gpu::ThreadIdOp>(&op)) {
          if (!thread_id_op &&
              candidate.getDimension() == mlir::gpu::Dimension::x) {
            thread_id_op = candidate;
          }
          continue;
        }
        if (auto candidate = llvm::dyn_cast<mlir::nvgpu::MmaSyncOp>(&op)) {
          if (mma_sync_op) {
            mma_sync_op = {};
            break;
          }
          mma_sync_op = candidate;
        }
      }

      if (!thread_id_op || !mma_sync_op || !loop.getInitArgs().empty()) {
        continue;
      }

      llvm::SmallPtrSet<mlir::Operation *, 16> visited;
      mlir::Value accumulator_buffer;
      if (!findUniqueAccumulatorBuffer(mma_sync_op.getOperand(2), body, visited,
                                       accumulator_buffer)) {
        continue;
      }
      auto accumulator_type =
          llvm::dyn_cast<mlir::MemRefType>(accumulator_buffer.getType());
      if (!accumulator_type || !IsWorkgroupMemRefType(accumulator_type) ||
          !accumulator_type.hasStaticShape() ||
          accumulator_type.getRank() != 2 ||
          accumulator_type.getShape()[0] != 16 ||
          accumulator_type.getShape()[1] != 8) {
        continue;
      }

      llvm::SmallPtrSet<mlir::Operation *, 32> accumulator_load_slice;
      collectValueDefiningOpsInBlock(mma_sync_op.getOperand(2), body,
                                     accumulator_load_slice);

      llvm::SmallPtrSet<mlir::Operation *, 16> accumulator_store_slice;
      collectForwardUsersInBlock(mma_sync_op.getResult(), body,
                                 accumulator_store_slice);
      bool supported_store_slice = llvm::all_of(
          accumulator_store_slice, [](mlir::Operation *op) {
            return llvm::isa<mlir::vector::ExtractOp, mlir::memref::StoreOp>(op);
          });
      if (!supported_store_slice) {
        continue;
      }

      builder.setInsertionPoint(loop);
      auto cloned_thread_id =
          builder.clone(*thread_id_op.getOperation())->getResult(0);
      mlir::Value initial_accumulator = buildAccumulatorFragmentFromWorkgroup(
          builder, loop.getLoc(), accumulator_buffer, cloned_thread_id);

      auto new_loop = builder.create<mlir::scf::ForOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep(), mlir::ValueRange{initial_accumulator});
      mlir::IRMapping mapping;
      mapping.map(loop.getInductionVar(), new_loop.getInductionVar());

      mlir::Block *new_body = new_loop.getBody();
      mlir::Value loop_carried_accumulator = new_loop.getRegionIterArgs().front();
      auto yield = llvm::cast<mlir::scf::YieldOp>(new_body->getTerminator());
      builder.setInsertionPoint(yield);
      for (mlir::Operation &op : body->without_terminator()) {
        if (&op == mma_sync_op.getOperation() ||
            accumulator_load_slice.contains(&op) ||
            accumulator_store_slice.contains(&op)) {
          continue;
        }
        builder.clone(op, mapping);
      }

      mlir::Operation *cloned_mma =
          builder.clone(*mma_sync_op.getOperation(), mapping);
      cloned_mma->setOperand(2, loop_carried_accumulator);
      yield->setOperands(cloned_mma->getResult(0));

      builder.setInsertionPointAfter(new_loop);
      storeAccumulatorFragmentToWorkgroup(builder, loop.getLoc(),
                                          accumulator_buffer, cloned_thread_id,
                                          new_loop.getResult(0));
      loop.erase();
    }
  }
};

struct ElideRedundantWorkgroupFillPass
    : public mlir::PassWrapper<ElideRedundantWorkgroupFillPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::linalg::FillOp, 16> fills;
    func.walk([&](mlir::linalg::FillOp fill) { fills.push_back(fill); });

    for (mlir::linalg::FillOp fill : fills) {
      if (!fill->getBlock()) {
        continue;
      }
      llvm::SmallVector<mlir::Value, 2> outputs(fill.getOutputs().begin(),
                                                fill.getOutputs().end());
      if (outputs.size() != 1) {
        continue;
      }
      mlir::Value buffer = outputs.front();
      auto buffer_type = llvm::dyn_cast<mlir::MemRefType>(buffer.getType());
      if (!buffer_type || !IsWorkgroupMemorySpace(buffer_type.getMemorySpace())) {
        continue;
      }

      mlir::Operation *cursor = fill->getNextNode();
      bool seen_overwrite = false;
      while (cursor) {
        if (auto copy = llvm::dyn_cast<mlir::linalg::CopyOp>(cursor)) {
          if (copyFullyOverwritesBuffer(copy, buffer)) {
            seen_overwrite = true;
            break;
          }
        }
        if (opMayReadBuffer(cursor, buffer)) {
          break;
        }
        cursor = cursor->getNextNode();
      }

      if (seen_overwrite) {
        fill.erase();
      }
    }
  }
};

}  // namespace

bool IsLowPrecisionTensorType(TensorDType dtype) {
  return dtype == TensorDType::kFloat16 || dtype == TensorDType::kBFloat16;
}

bool IsTensorCoreMmaSyncType(const MatmulLoweringSignature &signature) {
  return signature.lhs_dtype == TensorDType::kFloat16 &&
         signature.rhs_dtype == TensorDType::kFloat16 &&
         signature.out_dtype == TensorDType::kFloat16;
}

bool IsStaticallyCompatibleMmaSync(std::int64_t m, std::int64_t n,
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

bool IsWorkgroupMemorySpace(mlir::Attribute memory_space) {
  if (!memory_space) {
    return false;
  }
  if (auto gpu_space = llvm::dyn_cast<mlir::gpu::AddressSpaceAttr>(memory_space)) {
    return gpu_space.getValue() == mlir::gpu::AddressSpace::Workgroup;
  }
  if (auto int_space = llvm::dyn_cast<mlir::IntegerAttr>(memory_space)) {
    return int_space.getInt() ==
           static_cast<std::int64_t>(mlir::gpu::AddressSpace::Workgroup);
  }
  return false;
}

bool IsWorkgroupMemRefType(mlir::MemRefType type) {
  return type && IsWorkgroupMemorySpace(type.getMemorySpace());
}

NvidiaTileConfig SelectNvidiaTileConfig(
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

  NvidiaTileConfig config;
  config.block_tile_m = pickTilingFactor(m, 128);
  config.block_tile_n = pickTilingFactor(n, 128);
  config.k_tile = pickTilingFactor(k, signature.quantized_i8 ? 32 : 16);

  config.rewrite_to_mma_sync =
      IsTensorCoreMmaSyncType(signature) &&
      IsStaticallyCompatibleMmaSync(m, n, k);
  if (config.rewrite_to_mma_sync) {
    config.block_tile_m = 16;
    config.block_tile_n = 8;
    config.thread_tile_m = 16;
    config.thread_tile_n = 8;
    config.block_threads_y = 1;
    config.block_threads_x = 32;
    config.k_tile = 16;
    return config;
  }

  config.thread_tile_m =
      std::max<std::int64_t>(1, pickTilingFactor(config.block_tile_m, 16));
  config.thread_tile_n =
      std::max<std::int64_t>(1, pickTilingFactor(config.block_tile_n, 8));
  config.block_threads_y =
      std::max<std::int64_t>(1, ceilDiv(config.block_tile_m, config.thread_tile_m));
  config.block_threads_x =
      std::max<std::int64_t>(1, ceilDiv(config.block_tile_n, config.thread_tile_n));
  return config;
}

std::unique_ptr<mlir::Pass> CreatePromoteGpuWorkgroupAllocationsPass() {
  return std::make_unique<PromoteGpuWorkgroupAllocationsPass>();
}

std::unique_ptr<mlir::Pass> CreateSpecializeNvidiaWorkgroupMatmulOperandsPass() {
  return std::make_unique<SpecializeNvidiaWorkgroupMatmulOperandsPass>();
}

std::unique_ptr<mlir::Pass> CreatePersistNvidiaMmaAccumulatorPass() {
  return std::make_unique<PersistNvidiaMmaAccumulatorPass>();
}

void AddNvidiaMmaPreparationPasses(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ElideRedundantWorkgroupFillPass>());
  pm.addNestedPass<mlir::func::FuncOp>(CreatePromoteGpuWorkgroupAllocationsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      CreateSpecializeNvidiaWorkgroupMatmulOperandsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ElideRedundantWorkgroupFillPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void AddNvidiaAccumulatorLocalityPasses(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(CreatePersistNvidiaMmaAccumulatorPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void AddNvidiaLoopMaterializationPasses(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ElideRedundantWorkgroupFillPass>());
  pm.addNestedPass<mlir::func::FuncOp>(CreatePromoteGpuWorkgroupAllocationsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ElideRedundantWorkgroupFillPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

}  // namespace matcore
