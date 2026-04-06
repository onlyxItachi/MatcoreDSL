#include "matcore/gpu_tiling.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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

std::int64_t roundUpToMultiple(std::int64_t dim, std::int64_t tile) {
  if (dim <= 0 || tile <= 0) {
    return dim;
  }
  return ((dim + tile - 1) / tile) * tile;
}

mlir::OpFoldResult ceilDivMulIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::OpFoldResult dim,
                                   std::int64_t tile) {
  if (auto constant_dim = getConstantIndexFromOpFoldResult(dim)) {
    return builder.getIndexAttr(roundUpToMultiple(*constant_dim, tile));
  }

  mlir::Value dim_value = llvm::cast<mlir::Value>(dim);
  mlir::Value c_tile = builder.create<mlir::arith::ConstantIndexOp>(loc, tile);
  mlir::Value c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value dim_plus_tile_minus_one = builder.create<mlir::arith::AddIOp>(
      loc, dim_value, builder.create<mlir::arith::SubIOp>(loc, c_tile, c_one));
  mlir::Value quotient =
      builder.create<mlir::arith::DivUIOp>(loc, dim_plus_tile_minus_one, c_tile);
  return builder.create<mlir::arith::MulIOp>(loc, quotient, c_tile).getResult();
}

bool isStaticallyMmaAligned(mlir::MemRefType lhs_type, mlir::MemRefType rhs_type,
                            mlir::MemRefType out_type) {
  if (!lhs_type || !rhs_type || !out_type || !lhs_type.hasStaticShape() ||
      !rhs_type.hasStaticShape() || !out_type.hasStaticShape() ||
      lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
      out_type.getRank() != 2) {
    return false;
  }
  const std::int64_t m = lhs_type.getDimSize(0);
  const std::int64_t k = lhs_type.getDimSize(1);
  const std::int64_t n = rhs_type.getDimSize(1);
  const std::int64_t out_m = out_type.getDimSize(0);
  const std::int64_t out_n = out_type.getDimSize(1);
  if (out_m != m || out_n != n) {
    return false;
  }
  return m > 0 && n > 0 && k > 0 && (m % 16) == 0 && (n % 8) == 0 &&
         (k % 16) == 0;
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
        mlir::memref::ViewOp view;
        for (mlir::Operation *user : alloc->getUsers()) {
          if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(user)) {
            deallocs.push_back(dealloc);
            continue;
          }
          if (auto candidate_view = llvm::dyn_cast<mlir::memref::ViewOp>(user)) {
            if (view && candidate_view != view) {
              alloc.emitError("MatCore expected at most one workgroup memref.view");
              signalPassFailure();
              return;
            }
            view = candidate_view;
            continue;
          }
        }

        if (view) {
          auto static_view_type = inferStaticViewType(view);
          if (!static_view_type.has_value()) {
            view.emitError("MatCore requires statically sized promoted workgroup views");
            signalPassFailure();
            return;
          }

          mlir::BlockArgument attribution =
              addLaunchWorkgroupAttribution(launch, *static_view_type, view.getLoc());
          mlir::Value replacement = attribution;
          if (attribution.getType() != view.getType()) {
            replacement = builder.create<mlir::memref::CastOp>(view.getLoc(),
                                                                view.getType(),
                                                                attribution);
          }
          view.getResult().replaceAllUsesWith(replacement);
          view.erase();
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

      if (!changed) {
        continue;
      }

      builder.setInsertionPoint(op);
      auto replacement = builder.create<mlir::linalg::MatmulOp>(
          op.getLoc(), mlir::ValueRange(specialized_inputs), op.getOutputs());
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

struct DynamicMatmulPaddingPass
    : public mlir::PassWrapper<DynamicMatmulPaddingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::linalg::MatmulOp, 8> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::linalg::MatmulOp op : matmuls) {
      if (!op->getBlock()) {
        continue;
      }

      llvm::SmallVector<mlir::Value, 2> inputs(op.getInputs().begin(),
                                               op.getInputs().end());
      llvm::SmallVector<mlir::Value, 1> outputs(op.getOutputs().begin(),
                                                op.getOutputs().end());
      if (inputs.size() != 2 || outputs.size() != 1) {
        continue;
      }

      mlir::Value lhs = inputs[0];
      mlir::Value rhs = inputs[1];
      mlir::Value out = outputs[0];
      auto lhs_type = llvm::dyn_cast<mlir::MemRefType>(lhs.getType());
      auto rhs_type = llvm::dyn_cast<mlir::MemRefType>(rhs.getType());
      auto out_type = llvm::dyn_cast<mlir::MemRefType>(out.getType());
      if (!lhs_type || !rhs_type || !out_type || !lhs_type.hasRank() ||
          !rhs_type.hasRank() || !out_type.hasRank() || lhs_type.getRank() != 2 ||
          rhs_type.getRank() != 2 || out_type.getRank() != 2) {
        continue;
      }

      if (isStaticallyMmaAligned(lhs_type, rhs_type, out_type)) {
        continue;
      }

      builder.setInsertionPoint(op);
      const mlir::Location loc = op.getLoc();

      const mlir::OpFoldResult m_size = mlir::memref::getMixedSize(builder, loc, lhs, 0);
      const mlir::OpFoldResult k_size = mlir::memref::getMixedSize(builder, loc, lhs, 1);
      const mlir::OpFoldResult n_size = mlir::memref::getMixedSize(builder, loc, rhs, 1);
      const mlir::OpFoldResult out_m_size =
          mlir::memref::getMixedSize(builder, loc, out, 0);
      const mlir::OpFoldResult out_n_size =
          mlir::memref::getMixedSize(builder, loc, out, 1);

      const mlir::OpFoldResult mp_size = ceilDivMulIndex(builder, loc, m_size, 16);
      const mlir::OpFoldResult kp_size = ceilDivMulIndex(builder, loc, k_size, 16);
      const mlir::OpFoldResult np_size = ceilDivMulIndex(builder, loc, n_size, 8);

      llvm::SmallVector<mlir::OpFoldResult, 2> lhs_pad_sizes = {mp_size, kp_size};
      llvm::SmallVector<mlir::OpFoldResult, 2> rhs_pad_sizes = {kp_size, np_size};
      llvm::SmallVector<mlir::OpFoldResult, 2> out_pad_sizes = {mp_size, np_size};

      mlir::Value lhs_pad =
          builder
              .create<mlir::memref::AllocOp>(loc, lhs_pad_sizes,
                                             lhs_type.getElementType(),
                                             lhs_type.getMemorySpace())
              .getResult();
      mlir::Value rhs_pad =
          builder
              .create<mlir::memref::AllocOp>(loc, rhs_pad_sizes,
                                             rhs_type.getElementType(),
                                             rhs_type.getMemorySpace())
              .getResult();
      mlir::Value out_pad =
          builder
              .create<mlir::memref::AllocOp>(loc, out_pad_sizes,
                                             out_type.getElementType(),
                                             out_type.getMemorySpace())
              .getResult();

      auto lhs_zero = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getZeroAttr(lhs_type.getElementType()));
      auto rhs_zero = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getZeroAttr(rhs_type.getElementType()));
      auto out_zero = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getZeroAttr(out_type.getElementType()));
      builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{lhs_zero},
                                           mlir::ValueRange{lhs_pad});
      builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{rhs_zero},
                                           mlir::ValueRange{rhs_pad});
      builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{out_zero},
                                           mlir::ValueRange{out_pad});

      llvm::SmallVector<mlir::OpFoldResult, 2> offsets = {
          builder.getIndexAttr(0), builder.getIndexAttr(0)};
      llvm::SmallVector<mlir::OpFoldResult, 2> strides = {
          builder.getIndexAttr(1), builder.getIndexAttr(1)};
      llvm::SmallVector<mlir::OpFoldResult, 2> lhs_sizes = {m_size, k_size};
      llvm::SmallVector<mlir::OpFoldResult, 2> rhs_sizes = {k_size, n_size};
      llvm::SmallVector<mlir::OpFoldResult, 2> out_sizes = {out_m_size, out_n_size};

      mlir::Value lhs_pad_valid = builder
                                      .create<mlir::memref::SubViewOp>(
                                          loc, lhs_pad, offsets, lhs_sizes, strides)
                                      .getResult();
      mlir::Value rhs_pad_valid = builder
                                      .create<mlir::memref::SubViewOp>(
                                          loc, rhs_pad, offsets, rhs_sizes, strides)
                                      .getResult();
      mlir::Value out_pad_valid = builder
                                      .create<mlir::memref::SubViewOp>(
                                          loc, out_pad, offsets, out_sizes, strides)
                                      .getResult();

      builder.create<mlir::linalg::CopyOp>(loc, mlir::ValueRange{lhs},
                                           mlir::ValueRange{lhs_pad_valid});
      builder.create<mlir::linalg::CopyOp>(loc, mlir::ValueRange{rhs},
                                           mlir::ValueRange{rhs_pad_valid});
      builder.create<mlir::linalg::CopyOp>(loc, mlir::ValueRange{out},
                                           mlir::ValueRange{out_pad_valid});
      builder.create<mlir::linalg::MatmulOp>(
          loc, mlir::ValueRange{lhs_pad, rhs_pad}, mlir::ValueRange{out_pad});
      builder.create<mlir::linalg::CopyOp>(loc, mlir::ValueRange{out_pad_valid},
                                           mlir::ValueRange{out});

      op.erase();
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

  const bool statically_compatible_mma =
      m != mlir::ShapedType::kDynamic && n != mlir::ShapedType::kDynamic &&
      k != mlir::ShapedType::kDynamic && m >= 16 && n >= 8 && k >= 16 &&
      (m % 16) == 0 && (n % 8) == 0 && (k % 16) == 0 &&
      IsTensorCoreMmaSyncType(signature);
  // GpuDataStagingPass now handles host↔device memory transfers safely.
  config.rewrite_to_mma_sync = statically_compatible_mma;
  if (config.rewrite_to_mma_sync) {
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
    const bool needs_subtiling =
        config.block_tile_m > 16 || config.block_tile_n > 8;
    config.k_tile = needs_subtiling ? pickTilingFactor(k, 32) : 16;
    config.mma_micro_k = needs_subtiling ? 16 : 0;
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

std::unique_ptr<mlir::Pass> CreateDynamicMatmulPaddingPass() {
  return std::make_unique<DynamicMatmulPaddingPass>();
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

}  // namespace matcore
