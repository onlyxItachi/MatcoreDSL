#include "matcore/gpu_tiling.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Utils/MMAUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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

struct Rank2CopyLoopInfo {
  mlir::scf::ForOp outer;
  mlir::scf::ForOp inner;
  mlir::Value src;
  mlir::Value dst;
  bool src_workgroup = false;
  bool dst_workgroup = false;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
};

mlir::Operation *singlePayloadOp(mlir::Block &block) {
  mlir::Operation *payload = nullptr;
  for (mlir::Operation &op : block.without_terminator()) {
    if (payload != nullptr) {
      return nullptr;
    }
    payload = &op;
  }
  return payload;
}

std::optional<Rank2CopyLoopInfo> matchRank2CopyLoop(mlir::scf::ForOp outer) {
  auto lower = matchConstantIndex(outer.getLowerBound());
  auto step = matchConstantIndex(outer.getStep());
  auto rows = matchConstantIndex(outer.getUpperBound());
  if (!lower.has_value() || !step.has_value() || !rows.has_value() ||
      *lower != 0 || *step != 1 || *rows <= 0) {
    return std::nullopt;
  }

  auto inner = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
      singlePayloadOp(outer.getRegion().front()));
  if (!inner) {
    return std::nullopt;
  }

  auto inner_lower = matchConstantIndex(inner.getLowerBound());
  auto inner_step = matchConstantIndex(inner.getStep());
  auto cols = matchConstantIndex(inner.getUpperBound());
  if (!inner_lower.has_value() || !inner_step.has_value() ||
      !cols.has_value() || *inner_lower != 0 || *inner_step != 1 ||
      *cols <= 0) {
    return std::nullopt;
  }

  mlir::Operation *first = nullptr;
  mlir::Operation *second = nullptr;
  for (mlir::Operation &op : inner.getRegion().front().without_terminator()) {
    if (first == nullptr) {
      first = &op;
      continue;
    }
    if (second == nullptr) {
      second = &op;
      continue;
    }
    return std::nullopt;
  }
  auto load = llvm::dyn_cast_or_null<mlir::memref::LoadOp>(first);
  auto store = llvm::dyn_cast_or_null<mlir::memref::StoreOp>(second);
  if (!load || !store || store.getValue() != load.getResult()) {
    return std::nullopt;
  }

  if (load.getIndices().size() != 2 || store.getIndices().size() != 2 ||
      load.getIndices()[0] != outer.getInductionVar() ||
      load.getIndices()[1] != inner.getInductionVar() ||
      store.getIndices()[0] != outer.getInductionVar() ||
      store.getIndices()[1] != inner.getInductionVar()) {
    return std::nullopt;
  }

  auto src_type = llvm::dyn_cast<mlir::MemRefType>(load.getMemRef().getType());
  auto dst_type = llvm::dyn_cast<mlir::MemRefType>(store.getMemRef().getType());
  if (!src_type || !dst_type || !src_type.hasRank() || !dst_type.hasRank() ||
      src_type.getRank() != 2 || dst_type.getRank() != 2 ||
      src_type.getElementType() != dst_type.getElementType() ||
      !src_type.getElementType().isF16()) {
    return std::nullopt;
  }

  auto src_shape = inferStaticShape(load.getMemRef());
  auto dst_shape = inferStaticShape(store.getMemRef());
  if (!src_shape.has_value() || !dst_shape.has_value() || src_shape->size() != 2 ||
      dst_shape->size() != 2 || (*src_shape)[0] != *rows ||
      (*src_shape)[1] != *cols || (*dst_shape)[0] != *rows ||
      (*dst_shape)[1] != *cols) {
    return std::nullopt;
  }

  const bool src_workgroup = IsWorkgroupMemRefType(src_type);
  const bool dst_workgroup = IsWorkgroupMemRefType(dst_type);
  if (src_workgroup == dst_workgroup) {
    return std::nullopt;
  }
  if ((*cols % 8) != 0) {
    return std::nullopt;
  }

  Rank2CopyLoopInfo info;
  info.outer = outer;
  info.inner = inner;
  info.src = load.getMemRef();
  info.dst = store.getMemRef();
  info.src_workgroup = src_workgroup;
  info.dst_workgroup = dst_workgroup;
  info.rows = *rows;
  info.cols = *cols;
  return info;
}

bool isGlobalToWorkgroupCopyLoop(mlir::Operation *op) {
  auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(op);
  if (!loop) {
    return false;
  }
  auto info = matchRank2CopyLoop(loop);
  return info.has_value() && !info->src_workgroup && info->dst_workgroup;
}

void emitDistributedVectorCopy(mlir::OpBuilder &builder, mlir::Location loc,
                               const Rank2CopyLoopInfo &info) {
  auto element_type =
      llvm::cast<mlir::MemRefType>(info.src.getType()).getElementType();
  auto lane = builder.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto warp_y = builder.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::y);
  auto warp_z = builder.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::z);
  auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto c8 = builder.create<mlir::arith::ConstantIndexOp>(loc, 8);
  const std::int64_t segments = info.cols / 8;
  const std::int64_t total_vectors = info.rows * segments;
  auto c_segments =
      builder.create<mlir::arith::ConstantIndexOp>(loc, std::max<std::int64_t>(1, segments));
  auto c_total =
      builder.create<mlir::arith::ConstantIndexOp>(loc, total_vectors);
  auto vector_type = mlir::VectorType::get({8}, element_type);
  auto zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(element_type));
  auto permutation_map = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0,
      {mlir::getAffineDimExpr(1, builder.getContext())}, builder.getContext());
  auto in_bounds = builder.getArrayAttr({builder.getBoolAttr(true)});

  auto build_transfer = [&](mlir::OpBuilder &nested_builder) {
    mlir::Value row = lane;
    mlir::Value column = c0;
    if (segments > 1) {
      row = nested_builder.create<mlir::arith::DivUIOp>(loc, lane, c_segments);
      mlir::Value segment =
          nested_builder.create<mlir::arith::RemUIOp>(loc, lane, c_segments);
      column = nested_builder.create<mlir::arith::MulIOp>(loc, segment, c8);
    }
    auto vec = nested_builder.create<mlir::vector::TransferReadOp>(
        loc, vector_type, info.src, mlir::ValueRange{row, column},
        permutation_map, zero, mlir::Value(), in_bounds);
    nested_builder.create<mlir::vector::TransferWriteOp>(
        loc, mlir::Type(), vec, info.dst, mlir::ValueRange{row, column},
        permutation_map, mlir::Value(), in_bounds);
  };

  auto y_is_zero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, warp_y, c0);
  auto z_is_zero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, warp_z, c0);
  auto leader_warp = builder.create<mlir::arith::AndIOp>(loc, y_is_zero, z_is_zero);
  mlir::Value active = leader_warp;
  if (total_vectors < 32) {
    auto in_range = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, lane, c_total);
    active = builder.create<mlir::arith::AndIOp>(loc, leader_warp, in_range);
  }
  auto if_op = builder.create<mlir::scf::IfOp>(loc, active, /*withElseRegion=*/false);
  mlir::OpBuilder then_builder =
      mlir::OpBuilder::atBlockTerminator(&if_op.getThenRegion().front());
  build_transfer(then_builder);
}

struct VectorizeNvidiaSharedMemoryCopiesPass
    : public mlir::PassWrapper<VectorizeNvidiaSharedMemoryCopiesPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<Rank2CopyLoopInfo, 16> copy_loops;
    func.walk([&](mlir::scf::ForOp loop) {
      if (auto info = matchRank2CopyLoop(loop)) {
        copy_loops.push_back(*info);
      }
    });

    mlir::OpBuilder builder(&getContext());
    for (Rank2CopyLoopInfo &info : copy_loops) {
      if (!info.outer->getBlock()) {
        continue;
      }

      mlir::Operation *next_op = info.outer->getNextNode();
      const bool needs_pre_barrier = info.src_workgroup && !info.dst_workgroup;
      const bool needs_post_barrier =
          !info.src_workgroup && info.dst_workgroup &&
          !isGlobalToWorkgroupCopyLoop(next_op);

      builder.setInsertionPoint(info.outer);
      if (needs_pre_barrier) {
        builder.create<mlir::gpu::BarrierOp>(info.outer.getLoc());
      }
      emitDistributedVectorCopy(builder, info.outer.getLoc(), info);
      if (needs_post_barrier) {
        builder.create<mlir::gpu::BarrierOp>(info.outer.getLoc());
      }
      info.outer.erase();
    }
  }
};

mlir::memref::LoadOp extractBaseLoadFromFragment(mlir::Value value) {
  while (auto insert = value.getDefiningOp<mlir::vector::InsertOp>()) {
    value = insert.getDest();
  }
  if (auto splat = value.getDefiningOp<mlir::vector::SplatOp>()) {
    return splat.getInput().getDefiningOp<mlir::memref::LoadOp>();
  }
  return {};
}

bool IsF16LhsMmaFragmentType(mlir::VectorType type) {
  return type && type.getRank() == 2 && type.getShape()[0] == 4 &&
         type.getShape()[1] == 2 && type.getElementType().isF16();
}

bool IsF16RhsMmaFragmentType(mlir::VectorType type) {
  return type && type.getRank() == 2 && type.getShape()[0] == 2 &&
         type.getShape()[1] == 2 && type.getElementType().isF16();
}

struct RewriteNvidiaLdmatrixFragmentLoadsPass
    : public mlir::PassWrapper<RewriteNvidiaLdmatrixFragmentLoadsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect, mlir::nvgpu::NVGPUDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::nvgpu::MmaSyncOp, 16> mmas;
    func.walk([&](mlir::nvgpu::MmaSyncOp op) { mmas.push_back(op); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::nvgpu::MmaSyncOp mma : mmas) {
      if (!mma->getBlock()) {
        continue;
      }
      auto lhs_load = extractBaseLoadFromFragment(mma.getMatrixA());
      auto rhs_load = extractBaseLoadFromFragment(mma.getMatrixB());
      if (!lhs_load || !rhs_load || lhs_load.getIndices().size() != 2 ||
          rhs_load.getIndices().size() != 2) {
        continue;
      }

      auto lhs_memref = llvm::dyn_cast<mlir::MemRefType>(lhs_load.getMemRef().getType());
      auto rhs_memref = llvm::dyn_cast<mlir::MemRefType>(rhs_load.getMemRef().getType());
      auto lhs_type = llvm::dyn_cast<mlir::VectorType>(mma.getMatrixA().getType());
      auto rhs_type = llvm::dyn_cast<mlir::VectorType>(mma.getMatrixB().getType());
      if (!lhs_memref || !rhs_memref || !lhs_type || !rhs_type) {
        continue;
      }
      if (!IsWorkgroupMemRefType(lhs_memref) || !IsWorkgroupMemRefType(rhs_memref)) {
        continue;
      }

      if (!IsF16LhsMmaFragmentType(lhs_type) || !IsF16RhsMmaFragmentType(rhs_type)) {
        continue;
      }

      mlir::nvgpu::LdMatrixParams lhs_params{
          lhs_type,
          /*isAccum=*/false,
          /*numTiles=*/4,
          mlir::vector::IteratorType::parallel,
          mlir::NVVM::MMALayout::row};
      mlir::nvgpu::LdMatrixParams rhs_params{
          rhs_type,
          /*isAccum=*/false,
          /*numTiles=*/2,
          mlir::vector::IteratorType::parallel,
          mlir::NVVM::MMALayout::col};

      builder.setInsertionPoint(mma);
      auto lane = builder.create<mlir::gpu::ThreadIdOp>(mma.getLoc(), mlir::gpu::Dimension::x);
      auto c0 = builder.create<mlir::arith::ConstantIndexOp>(mma.getLoc(), 0);
      auto c8 = builder.create<mlir::arith::ConstantIndexOp>(mma.getLoc(), 8);
      auto c16 = builder.create<mlir::arith::ConstantIndexOp>(mma.getLoc(), 16);
      auto lhs_row = builder.create<mlir::arith::RemUIOp>(mma.getLoc(), lane, c16);
      auto lhs_col_block = builder.create<mlir::arith::DivUIOp>(mma.getLoc(), lane, c16);
      auto lhs_col =
          builder.create<mlir::arith::MulIOp>(mma.getLoc(), lhs_col_block, c8);
      auto rhs_row = builder.create<mlir::arith::RemUIOp>(mma.getLoc(), lane, c16);
      auto rhs_col = c0;
      auto lhs_ldmatrix = builder.create<mlir::nvgpu::LdMatrixOp>(
          mma.getLoc(), lhs_type, lhs_load.getMemRef(),
          mlir::ValueRange{lhs_row, lhs_col}, /*transpose=*/false,
          lhs_params.numTiles);
      auto rhs_ldmatrix = builder.create<mlir::nvgpu::LdMatrixOp>(
          mma.getLoc(), rhs_type, rhs_load.getMemRef(),
          mlir::ValueRange{rhs_row, rhs_col}, /*transpose=*/true,
          rhs_params.numTiles);
      mma->setOperand(0, lhs_ldmatrix.getResult());
      mma->setOperand(1, rhs_ldmatrix.getResult());
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
    config.block_tile_m = std::max<std::int64_t>(
        16, pickTilingFactor(m, 128));
    config.block_tile_n = std::max<std::int64_t>(
        8, pickTilingFactor(n, 128));
    config.thread_tile_m = std::max<std::int64_t>(
        16, pickTilingFactor(config.block_tile_m, 64));
    config.thread_tile_n = std::max<std::int64_t>(
        8, pickTilingFactor(config.block_tile_n, 64));
    config.block_threads_x = 32;
    config.block_threads_y = std::max<std::int64_t>(
        1, ceilDiv(config.block_tile_n, config.thread_tile_n));
    config.block_threads_z = std::max<std::int64_t>(
        1, ceilDiv(config.block_tile_m, config.thread_tile_m));
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

std::unique_ptr<mlir::Pass> CreateVectorizeNvidiaSharedMemoryCopiesPass() {
  return std::make_unique<VectorizeNvidiaSharedMemoryCopiesPass>();
}

std::unique_ptr<mlir::Pass> CreateRewriteNvidiaLdmatrixFragmentLoadsPass() {
  return std::make_unique<RewriteNvidiaLdmatrixFragmentLoadsPass>();
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

void AddNvidiaAsyncCopyPreparationPasses(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(CreateVectorizeNvidiaSharedMemoryCopiesPass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateRewriteNvidiaLdmatrixFragmentLoadsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

}  // namespace matcore
