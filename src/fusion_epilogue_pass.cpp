#include "fusion_emitter_internal.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace matcore::fusion_emit {

bool isSubviewChainRootedAt(mlir::Value view, mlir::Value root) {
  if (view == root) {
    return true;
  }
  if (auto metadata =
          view.getDefiningOp<mlir::memref::ExtractStridedMetadataOp>()) {
    return isSubviewChainRootedAt(metadata.getSource(), root);
  }
  if (auto reinterpret =
          view.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
    return isSubviewChainRootedAt(reinterpret.getSource(), root);
  }
  if (auto subview = view.getDefiningOp<mlir::memref::SubViewOp>()) {
    return isSubviewChainRootedAt(subview.getSource(), root);
  }
  return false;
}

mlir::Value cloneSubviewChainWithNewRoot(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value view,
                                         mlir::Value old_root,
                                         mlir::Value new_root) {
  if (view == old_root) {
    return new_root;
  }
  if (auto metadata =
          view.getDefiningOp<mlir::memref::ExtractStridedMetadataOp>()) {
    mlir::Value remapped_source = cloneSubviewChainWithNewRoot(
        builder, loc, metadata.getSource(), old_root, new_root);
    if (!remapped_source) {
      return nullptr;
    }
    auto remapped_metadata =
        builder.create<mlir::memref::ExtractStridedMetadataOp>(loc,
                                                               remapped_source);
    if (view == metadata.getBaseBuffer()) {
      return remapped_metadata.getBaseBuffer();
    }
    return nullptr;
  }
  if (auto reinterpret =
          view.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
    mlir::Value remapped_source = cloneSubviewChainWithNewRoot(
        builder, loc, reinterpret.getSource(), old_root, new_root);
    if (!remapped_source) {
      return nullptr;
    }
    return builder
        .create<mlir::memref::ReinterpretCastOp>(
            loc, reinterpret.getType(), remapped_source, reinterpret.getOffsets(),
            reinterpret.getSizes(), reinterpret.getStrides(),
            reinterpret.getStaticOffsets(), reinterpret.getStaticSizes(),
            reinterpret.getStaticStrides())
        .getResult();
  }
  auto subview = view.getDefiningOp<mlir::memref::SubViewOp>();
  if (!subview) {
    return nullptr;
  }
  mlir::Value remapped_source = cloneSubviewChainWithNewRoot(
      builder, loc, subview.getSource(), old_root, new_root);
  if (!remapped_source) {
    return nullptr;
  }
  return builder
      .create<mlir::memref::SubViewOp>(loc, remapped_source,
                                       subview.getMixedOffsets(),
                                       subview.getMixedSizes(),
                                       subview.getMixedStrides())
      .getResult();
}

bool hasAncestorWithAttr(mlir::Operation *op, llvm::StringRef attr_name) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->hasAttr(attr_name)) {
      return true;
    }
  }
  return false;
}

bool emitTileEpilogue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::func::FuncOp func, mlir::Value out_arg,
                      mlir::Value output_tile,
                      const llvm::SmallVector<FamilyAEpilogueSpec, 4>
                          &epilogue_specs) {
  auto tile_type = llvm::dyn_cast<mlir::MemRefType>(output_tile.getType());
  if (!tile_type || tile_type.getRank() != 2 || tile_type.getMemorySpace()) {
    return false;
  }

  llvm::SmallVector<mlir::Value, 4> boundary_tiles;
  boundary_tiles.reserve(epilogue_specs.size());
  for (const FamilyAEpilogueSpec &spec : epilogue_specs) {
    if (spec.boundary_arg_index < 0) {
      if (!isUnaryElementwiseKind(spec.kind)) {
        return false;
      }
      boundary_tiles.push_back(nullptr);
      continue;
    }
    if (spec.boundary_arg_index >= func.getNumArguments() - 1 ||
        spec.value_input_pos < 0 || spec.value_input_pos > 1 ||
        isUnaryElementwiseKind(spec.kind)) {
      return false;
    }
    mlir::Value boundary_tile = cloneSubviewChainWithNewRoot(
        builder, loc, output_tile, out_arg,
        func.getArgument(static_cast<unsigned>(spec.boundary_arg_index)));
    if (!boundary_tile) {
      return false;
    }
    boundary_tiles.push_back(boundary_tile);
  }

  mlir::Value rows = tile_type.isDynamicDim(0)
                         ? builder.create<mlir::memref::DimOp>(loc, output_tile, 0)
                               .getResult()
                         : builder.create<mlir::arith::ConstantIndexOp>(
                               loc, tile_type.getDimSize(0)).getResult();
  mlir::Value cols = tile_type.isDynamicDim(1)
                         ? builder.create<mlir::memref::DimOp>(loc, output_tile, 1)
                               .getResult()
                         : builder.create<mlir::arith::ConstantIndexOp>(
                               loc, tile_type.getDimSize(1)).getResult();
  mlir::Value zero =
      builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
  mlir::Value one =
      builder.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();

  auto row_loop = builder.create<mlir::scf::ForOp>(loc, zero, rows, one);
  {
    mlir::OpBuilder::InsertionGuard row_guard(builder);
    builder.setInsertionPointToStart(row_loop.getBody());
    mlir::Value row = row_loop.getInductionVar();
    auto col_loop = builder.create<mlir::scf::ForOp>(loc, zero, cols, one);
    {
      mlir::OpBuilder::InsertionGuard col_guard(builder);
      builder.setInsertionPointToStart(col_loop.getBody());
      mlir::Value col = col_loop.getInductionVar();
      llvm::SmallVector<mlir::Value, 2> indices = {row, col};
      mlir::Value result =
          builder.create<mlir::memref::LoadOp>(loc, output_tile, indices);

      for (std::size_t i = 0; i < epilogue_specs.size(); ++i) {
        const FamilyAEpilogueSpec &spec = epilogue_specs[i];
        if (spec.boundary_arg_index >= 0) {
          mlir::Value boundary = builder.create<mlir::memref::LoadOp>(
              loc, boundary_tiles[i], indices);
          if (spec.value_input_pos == 0) {
            result = emitBinaryElementwiseOnValues(builder, loc, spec.kind,
                                                   result, boundary);
          } else {
            result = emitBinaryElementwiseOnValues(builder, loc, spec.kind,
                                                   boundary, result);
          }
        } else {
          result = emitElementwiseOnValue(builder, loc, spec.kind, result,
                                          builder.getF32Type());
        }
      }

      builder.create<mlir::memref::StoreOp>(loc, result, output_tile, indices);
    }
  }

  return true;
}

mlir::Operation *findLaunchEpilogueInsertPoint(mlir::gpu::LaunchOp launch) {
  mlir::Block &body = launch.getBody().front();
  mlir::Operation *insert_before = body.getTerminator();
  for (mlir::Operation &op : body.without_terminator()) {
    if (llvm::isa<mlir::gpu::BarrierOp>(op)) {
      insert_before = &op;
      break;
    }
  }
  return insert_before;
}

unsigned countGpuLaunches(mlir::func::FuncOp func) {
  unsigned count = 0;
  func.walk([&](mlir::gpu::LaunchOp) { ++count; });
  return count;
}

bool inlineLastKIterationTileEpilogue(
    mlir::func::FuncOp func, mlir::Value out_arg,
    const llvm::SmallVector<FamilyAEpilogueSpec, 4> &epilogue_specs) {
  llvm::SmallVector<mlir::gpu::LaunchOp, 2> launches;
  func.walk([&](mlir::gpu::LaunchOp launch) { launches.push_back(launch); });
  if (launches.empty()) {
    return false;
  }

  bool rewrote_any = false;
  unsigned rewritten_tiles = 0;
  mlir::gpu::LaunchOp primary_launch = launches.front();
  llvm::SmallVector<mlir::scf::ForOp, 2> k_loops;
  primary_launch.getBody().walk([&](mlir::scf::ForOp loop) {
    if (loop->hasAttr("matcore.k_loop")) {
      k_loops.push_back(loop);
    }
  });

  for (mlir::scf::ForOp k_loop : k_loops) {
    llvm::SmallVector<mlir::Value, 4> output_tiles;
    k_loop.getBody()->walk([&](mlir::memref::StoreOp store) {
      mlir::Value output_tile = store.getMemRef();
      auto tile_type = llvm::dyn_cast<mlir::MemRefType>(output_tile.getType());
      if (!tile_type || tile_type.getRank() != 2 ||
          tile_type.getMemorySpace() ||
          !isSubviewChainRootedAt(output_tile, out_arg)) {
        return;
      }
      if (std::find(output_tiles.begin(), output_tiles.end(), output_tile) ==
          output_tiles.end()) {
        output_tiles.push_back(output_tile);
      }
    });
    if (output_tiles.empty()) {
      continue;
    }

    mlir::Operation *insert_before = k_loop.getBody()->getTerminator();
    for (mlir::Operation &op : k_loop.getBody()->without_terminator()) {
      if (llvm::isa<mlir::gpu::BarrierOp>(op)) {
        insert_before = &op;
        break;
      }
    }

    mlir::OpBuilder builder(insert_before);
    const mlir::Location loc = insert_before->getLoc();
    mlir::Value next_k = builder.create<mlir::arith::AddIOp>(
        loc, k_loop.getInductionVar(), k_loop.getStep());
    mlir::Value is_last_k = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::uge, next_k, k_loop.getUpperBound());
    auto if_last = builder.create<mlir::scf::IfOp>(loc, is_last_k,
                                                   /*withElse=*/false);

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&if_last.getThenRegion().front());
      for (mlir::Value output_tile : output_tiles) {
        if (emitTileEpilogue(builder, loc, func, out_arg, output_tile,
                             epilogue_specs)) {
          rewrote_any = true;
          ++rewritten_tiles;
        }
      }
    }
  }

  if (rewrote_any) {
    fprintf(stderr,
            "[FusionEpilogue] In-launch: rewrote %u output tile(s) on last "
            "K-iteration (no separate kernel)\n",
            rewritten_tiles);
  }
  return rewrote_any;
}

bool inlineSingleTileLaunchEpilogue(
    mlir::func::FuncOp func, mlir::Value out_arg,
    const llvm::SmallVector<FamilyAEpilogueSpec, 4> &epilogue_specs) {
  llvm::SmallVector<mlir::gpu::LaunchOp, 2> launches;
  func.walk([&](mlir::gpu::LaunchOp launch) { launches.push_back(launch); });
  if (launches.empty()) {
    return false;
  }

  mlir::gpu::LaunchOp primary_launch = launches.front();
  llvm::SmallVector<mlir::Value, 4> output_tiles;
  primary_launch.getBody().walk([&](mlir::memref::StoreOp store) {
    if (hasAncestorWithAttr(store.getOperation(), "matcore.k_loop")) {
      return;
    }
    mlir::Value output_tile = store.getMemRef();
    auto tile_type = llvm::dyn_cast<mlir::MemRefType>(output_tile.getType());
    if (!tile_type || tile_type.getRank() != 2 ||
        tile_type.getMemorySpace() ||
        !isSubviewChainRootedAt(output_tile, out_arg)) {
      return;
    }
    if (std::find(output_tiles.begin(), output_tiles.end(), output_tile) ==
        output_tiles.end()) {
      output_tiles.push_back(output_tile);
    }
  });
  if (output_tiles.empty()) {
    return false;
  }

  mlir::Operation *insert_before = findLaunchEpilogueInsertPoint(primary_launch);
  mlir::OpBuilder builder(insert_before);
  const mlir::Location loc = insert_before->getLoc();
  unsigned rewritten_tiles = 0;
  for (mlir::Value output_tile : output_tiles) {
    if (emitTileEpilogue(builder, loc, func, out_arg, output_tile,
                         epilogue_specs)) {
      ++rewritten_tiles;
    }
  }
  if (rewritten_tiles == 0) {
    return false;
  }

  fprintf(stderr,
          "[FusionEpilogue] In-launch: rewrote %u output tile(s) in "
          "single-tile launch (no separate kernel)\n",
          rewritten_tiles);
  return true;
}

llvm::SmallVector<FamilyAEpilogueSpec, 4>
decodeFamilyAEpilogueSpecs(mlir::ModuleOp module) {
  llvm::SmallVector<FamilyAEpilogueSpec, 4> specs;

  auto op_attr =
      module->getAttrOfType<mlir::ArrayAttr>("matcore.fusion_epilogue_ops");
  if (op_attr) {
    specs.reserve(op_attr.size());
    for (mlir::Attribute attr : op_attr) {
      auto dict = attr.dyn_cast<mlir::DictionaryAttr>();
      if (!dict) {
        throw std::runtime_error(
            "FusionEpiloguePass: matcore.fusion_epilogue_ops must contain dict attrs");
      }

      auto kind_attr = dict.getAs<mlir::IntegerAttr>("kind");
      if (!kind_attr) {
        throw std::runtime_error(
            "FusionEpiloguePass: epilogue op metadata is missing 'kind'");
      }

      FamilyAEpilogueSpec spec;
      spec.kind = static_cast<ElementwiseKind>(kind_attr.getInt());
      if (auto arg_attr = dict.getAs<mlir::IntegerAttr>("boundary_arg_index")) {
        spec.boundary_arg_index = static_cast<int>(arg_attr.getInt());
      }
      if (auto pos_attr = dict.getAs<mlir::IntegerAttr>("value_input_pos")) {
        spec.value_input_pos = static_cast<int>(pos_attr.getInt());
      }
      specs.push_back(spec);
    }
    return specs;
  }

  auto kind_attr =
      module->getAttrOfType<mlir::ArrayAttr>("matcore.fusion_epilogue_kinds");
  if (!kind_attr) {
    return specs;
  }

  specs.reserve(kind_attr.size());
  for (mlir::Attribute attr : kind_attr) {
    FamilyAEpilogueSpec spec;
    spec.kind = static_cast<ElementwiseKind>(
        attr.cast<mlir::IntegerAttr>().getInt());
    specs.push_back(spec);
  }
  return specs;
}

struct FusionEpiloguePass
    : public mlir::PassWrapper<FusionEpiloguePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionEpiloguePass)

  llvm::StringRef getArgument() const override {
    return "matcore-fusion-epilogue";
  }
  llvm::StringRef getDescription() const override {
    return "Insert fused unary/binary epilogue handling after MMA matmul";
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::SmallVector<FamilyAEpilogueSpec, 4> epilogue_specs;
    try {
      epilogue_specs = decodeFamilyAEpilogueSpecs(module);
    } catch (const std::runtime_error &) {
      signalPassFailure();
      return;
    }
    if (epilogue_specs.empty()) {
      return;  // No epilogue needed
    }

    // Find the entry function.
    mlir::func::FuncOp func;
    module.walk([&](mlir::func::FuncOp f) { func = f; });
    if (!func) {
      signalPassFailure();
      return;
    }
    mlir::Builder attr_builder(module.getContext());
    auto set_epilogue_strategy = [&](llvm::StringRef strategy) {
      module->setAttr("matcore.fusion_epilogue_strategy",
                      attr_builder.getStringAttr(strategy));
      module->setAttr(
          "matcore.fusion_launch_count",
          attr_builder.getI32IntegerAttr(static_cast<int>(countGpuLaunches(func))));
    };

    // === IN-REGISTER EPILOGUE FUSION ===
    // Try to find output stores tagged by AccumulatorHoistPass inside the
    // matmul gpu.launch and apply the epilogue chain directly on the stored
    // value (in registers, before the GMEM write). This eliminates the
    // separate epilogue kernel entirely.
    llvm::SmallVector<mlir::memref::StoreOp, 32> tagged_stores;
    func.walk([&](mlir::memref::StoreOp store) {
      if (store->hasAttr("matcore.epilogue_store"))
        tagged_stores.push_back(store);
    });

    if (!tagged_stores.empty()) {
      unsigned rewritten = 0;
      for (auto store : tagged_stores) {
        mlir::OpBuilder builder(store);
        auto loc = store.getLoc();
        mlir::Value val = store.getValueToStore();
        const mlir::ValueRange indices = store.getIndices();

        // Apply the full epilogue chain on the stored value.
        for (const FamilyAEpilogueSpec &spec : epilogue_specs) {
          if (spec.boundary_arg_index >= 0) {
            if (spec.boundary_arg_index >= func.getNumArguments() - 1 ||
                spec.value_input_pos < 0 || spec.value_input_pos > 1 ||
                isUnaryElementwiseKind(spec.kind)) {
              signalPassFailure();
              return;
            }
            mlir::Value boundary = builder.create<mlir::memref::LoadOp>(
                loc, func.getArgument(static_cast<unsigned>(spec.boundary_arg_index)),
                indices);
            if (spec.value_input_pos == 0) {
              val = emitBinaryElementwiseOnValues(builder, loc, spec.kind, val,
                                                  boundary);
            } else {
              val = emitBinaryElementwiseOnValues(builder, loc, spec.kind,
                                                  boundary, val);
            }
          } else {
            if (!isUnaryElementwiseKind(spec.kind)) {
              signalPassFailure();
              return;
            }
            val = emitElementwiseOnValue(builder, loc, spec.kind, val,
                                         builder.getF32Type());
          }
        }
        store.getValueMutable().assign(val);
        store->removeAttr("matcore.epilogue_store");
        ++rewritten;
      }
      fprintf(stderr,
              "[FusionEpilogue] In-register: rewrote %u tagged output "
              "stores with %lu epilogue ops (no separate kernel)\n",
              rewritten, (unsigned long)epilogue_specs.size());
      set_epilogue_strategy("in_register");
      return;  // Done — no separate kernel needed
    }

    // === INLINE LAST-K EPILOGUE FOR NON-TAGGED TILE-MATMUL PATH ===
    // F32/thread-mapped fusion keeps the accumulator update inside the K-loop,
    // so there are no tagged post-loop stores to rewrite in place. Instead,
    // apply the epilogue exactly once on the last K-iteration, over each
    // thread-local output tile, before falling back to a second launch.
    if (func.getNumArguments() > 0) {
      mlir::Value out_arg = func.getArgument(func.getNumArguments() - 1);
      if (inlineLastKIterationTileEpilogue(func, out_arg, epilogue_specs)) {
        set_epilogue_strategy("in_launch_last_k");
        return;
      }
      if (inlineSingleTileLaunchEpilogue(func, out_arg, epilogue_specs)) {
        set_epilogue_strategy("in_launch_single_tile");
        return;
      }
    }

    // === FALLBACK: SEPARATE EPILOGUE KERNEL ===
    // No tagged stores found (non-MMA path or single-warp without AccHoist).
    // Create a single gpu.launch that applies the full epilogue chain.
    fprintf(stderr,
            "[FusionEpilogue] Fallback: creating separate epilogue kernel "
            "(%lu ops)\n", (unsigned long)epilogue_specs.size());

    auto m_attr =
        module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_m");
    auto n_attr =
        module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_n");
    if (!m_attr || !n_attr) {
      signalPassFailure();
      return;
    }
    const int64_t M = m_attr.getInt();
    const int64_t N = n_attr.getInt();
    const int64_t total_elems = M * N;
    const int64_t block_size = 256;
    const int64_t grid_size = (total_elems + block_size - 1) / block_size;

    if (func.getNumArguments() == 0) {
      signalPassFailure();
      return;
    }
    auto out_arg = func.getArgument(func.getNumArguments() - 1);

    mlir::func::ReturnOp return_op;
    func.walk([&](mlir::func::ReturnOp r) { return_op = r; });
    if (!return_op) {
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(return_op);
    auto loc = builder.getUnknownLoc();

    // Single gpu.launch that chains ALL epilogue ops (load → chain → store).
    auto c_grid = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_size);
    auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_size);

    auto launch = builder.create<mlir::gpu::LaunchOp>(
        loc, c_grid, c_one, c_one, c_block, c_one, c_one);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&launch.getBody().front());

      auto tid = launch.getThreadIds().x;
      auto bid = launch.getBlockIds().x;
      auto c_bd = builder.create<mlir::arith::ConstantIndexOp>(loc, block_size);
      auto c_total = builder.create<mlir::arith::ConstantIndexOp>(loc, total_elems);
      auto c_N_val = builder.create<mlir::arith::ConstantIndexOp>(loc, N);

      auto idx = builder.create<mlir::arith::AddIOp>(
          loc, builder.create<mlir::arith::MulIOp>(loc, bid, c_bd), tid);

      auto in_bounds = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, idx, c_total);

      auto if_op = builder.create<mlir::scf::IfOp>(loc, in_bounds,
                                                     /*withElse=*/false);
      {
        mlir::OpBuilder::InsertionGuard guard2(builder);
        builder.setInsertionPointToStart(&if_op.getThenRegion().front());

        auto row = builder.create<mlir::arith::DivUIOp>(loc, idx, c_N_val);
        auto col = builder.create<mlir::arith::RemUIOp>(loc, idx, c_N_val);
        auto val = builder.create<mlir::memref::LoadOp>(
            loc, out_arg, mlir::ValueRange{row, col});

        // Apply ALL epilogue ops in one shot (single load → chain → store).
        mlir::Value result = val;
        for (const FamilyAEpilogueSpec &spec : epilogue_specs) {
          if (spec.boundary_arg_index >= 0) {
            if (spec.boundary_arg_index >= func.getNumArguments() - 1 ||
                spec.value_input_pos < 0 || spec.value_input_pos > 1 ||
                isUnaryElementwiseKind(spec.kind)) {
              signalPassFailure();
              return;
            }
            mlir::Value boundary = builder.create<mlir::memref::LoadOp>(
                loc, func.getArgument(static_cast<unsigned>(spec.boundary_arg_index)),
                mlir::ValueRange{row, col});
            if (spec.value_input_pos == 0) {
              result = emitBinaryElementwiseOnValues(builder, loc, spec.kind,
                                                     result, boundary);
            } else {
              result = emitBinaryElementwiseOnValues(builder, loc, spec.kind,
                                                     boundary, result);
            }
          } else {
            if (!isUnaryElementwiseKind(spec.kind)) {
              signalPassFailure();
              return;
            }
            result = emitElementwiseOnValue(builder, loc, spec.kind, result,
                                            builder.getF32Type());
          }
        }

        builder.create<mlir::memref::StoreOp>(
            loc, result, out_arg, mlir::ValueRange{row, col});
      }

      builder.setInsertionPointAfter(if_op);
      builder.create<mlir::gpu::TerminatorOp>(loc);
    }
    set_epilogue_strategy("fallback_launch");
  }
};

}  // namespace matcore::fusion_emit

namespace matcore {

std::unique_ptr<mlir::Pass> CreateFusionEpiloguePass() {
  return std::make_unique<fusion_emit::FusionEpiloguePass>();
}

}  // namespace matcore
