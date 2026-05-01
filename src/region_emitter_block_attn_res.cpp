#include "matcore/region_emitter.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace matcore {
namespace {

[[noreturn]] void failBlockAttnRes(const std::string &message) {
  throw std::runtime_error("RegionMlirEmitter: BlockAttnRes: " + message);
}

const TensorDesc &requireValue(const RegionIR &region, std::uint32_t id) {
  if (id >= region.values.size()) {
    failBlockAttnRes("value id out of range");
  }
  return region.values[id];
}

void requireFloat32(const TensorDesc &desc, const std::string &name) {
  if (desc.dtype != TensorDType::kFloat32) {
    failBlockAttnRes(name + " must be float32 in v1");
  }
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> RegionMlirEmitter::emitBlockAttnRes(
    const KernelIR &kernel, const RegionNode &node,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  if (normalizeTarget(target.kind) != TargetKind::kNvidiaDGPU) {
    failBlockAttnRes("v1 supports nvidia-dgpu targets only");
  }
  if (!kernel.region.has_value()) {
    failBlockAttnRes("missing region_v1 IR");
  }
  const RegionIR &region = *kernel.region;
  if (tensors.size() != region.input_values.size() + 1) {
    failBlockAttnRes("runtime tensor count must match inputs plus output");
  }

  const auto &attrs = std::get<BlockAttnResAttrs>(node.attrs);
  const TensorDesc &blocks_desc = requireValue(region, attrs.blocks);
  const TensorDesc &partial_desc = requireValue(region, attrs.partial);
  const TensorDesc &query_desc = requireValue(region, attrs.query);
  if (node.outputs.empty()) {
    failBlockAttnRes("missing output value");
  }
  const TensorDesc &out_desc = requireValue(region, node.outputs.front());

  requireFloat32(blocks_desc, "blocks");
  requireFloat32(partial_desc, "partial");
  requireFloat32(query_desc, "query");
  requireFloat32(out_desc, "output");
  if (blocks_desc.shape.size() != 4 || partial_desc.shape.size() != 3 ||
      query_desc.shape.size() != 1 || out_desc.shape.size() != 3) {
    failBlockAttnRes("expected blocks rank-4, partial/output rank-3, query rank-1");
  }

  const int64_t max_blocks = blocks_desc.shape[0];
  const int64_t B = blocks_desc.shape[1];
  const int64_t T = blocks_desc.shape[2];
  const int64_t D = blocks_desc.shape[3];
  if (max_blocks <= 0 || max_blocks > 32) {
    failBlockAttnRes("MAX_BLOCKS must be in [1, 32]");
  }
  if (partial_desc.shape[0] != B || partial_desc.shape[1] != T ||
      partial_desc.shape[2] != D || out_desc.shape[0] != B ||
      out_desc.shape[1] != T || out_desc.shape[2] != D ||
      query_desc.shape[0] != D) {
    failBlockAttnRes("shape mismatch for packed BlockAttnRes ABI");
  }
  if (attrs.block_count < 0 || attrs.block_count > max_blocks) {
    failBlockAttnRes("block_count must be in [0, MAX_BLOCKS]");
  }
  if (attrs.eps <= 0.0f) {
    failBlockAttnRes("eps must be positive");
  }

  auto find_input_arg_index = [&](std::uint32_t value_id) -> int {
    auto it = std::find(region.input_values.begin(), region.input_values.end(),
                        value_id);
    if (it == region.input_values.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(region.input_values.begin(), it));
  };
  const int blocks_arg_index = find_input_arg_index(attrs.blocks);
  const int partial_arg_index = find_input_arg_index(attrs.partial);
  const int query_arg_index = find_input_arg_index(attrs.query);
  if (blocks_arg_index < 0 || partial_arg_index < 0 || query_arg_index < 0) {
    failBlockAttnRes("blocks, partial, and query must be region inputs");
  }

  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                      mlir::gpu::GPUDialect, mlir::math::MathDialect>();

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  auto f32 = builder.getF32Type();
  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(region.input_values.size() + 1);
  for (std::uint32_t input_value_id : region.input_values) {
    const TensorDesc &input_desc = requireValue(region, input_value_id);
    arg_types.push_back(mlir::MemRefType::get(input_desc.shape, f32));
  }
  arg_types.push_back(mlir::MemRefType::get(out_desc.shape, f32));

  const std::string base_name =
      kernel.kernel_name.empty() ? "matcore_region" : kernel.kernel_name;
  const std::string entry_name = "fused_" + base_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name, builder.getFunctionType(arg_types, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setPublic();

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto blocks_arg = entry->getArgument(static_cast<unsigned>(blocks_arg_index));
  auto partial_arg = entry->getArgument(static_cast<unsigned>(partial_arg_index));
  auto query_arg = entry->getArgument(static_cast<unsigned>(query_arg_index));
  auto out_arg = entry->getArgument(static_cast<unsigned>(region.input_values.size()));

  const int block_dim = 128;
  const int64_t rows = B * T;
  const int64_t score_slots = max_blocks + 1;
  const int64_t active_sources =
      attrs.block_count + (attrs.has_partial ? 1 : 0);

  auto c_rows = builder.create<mlir::arith::ConstantIndexOp>(loc, rows);
  auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_dim);
  auto launch = builder.create<mlir::gpu::LaunchOp>(
      loc, c_rows, c_one, c_one, c_block, c_one, c_one);
  builder.setInsertionPointToStart(&launch.getBody().front());

  auto tid = launch.getThreadIds().x;
  auto row = launch.getBlockIds().x;
  auto c_zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto c_T = builder.create<mlir::arith::ConstantIndexOp>(loc, T);
  auto c_D = builder.create<mlir::arith::ConstantIndexOp>(loc, D);
  auto c_active_sources =
      builder.create<mlir::arith::ConstantIndexOp>(loc, active_sources);
  auto c_block_count =
      builder.create<mlir::arith::ConstantIndexOp>(loc, attrs.block_count);
  auto b_idx = builder.create<mlir::arith::DivUIOp>(loc, row, c_T);
  auto t_idx = builder.create<mlir::arith::RemUIOp>(loc, row, c_T);

  auto zero_f32 = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(0.0f));
  auto neg_inf = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
  auto inv_d = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(1.0f / static_cast<float>(D)));
  auto eps = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(attrs.eps));

  auto smem_as = mlir::gpu::AddressSpaceAttr::get(
      &context, mlir::gpu::AddressSpace::Workgroup);
  auto score_type = mlir::MemRefType::get({score_slots}, f32,
                                          mlir::MemRefLayoutAttrInterface(),
                                          smem_as);
  auto score_smem = launch.addWorkgroupAttribution(score_type, loc);
  auto inv_rms_type = mlir::MemRefType::get({score_slots}, f32,
                                            mlir::MemRefLayoutAttrInterface(),
                                            smem_as);
  auto inv_rms_smem = launch.addWorkgroupAttribution(inv_rms_type, loc);
  auto scratch_type = mlir::MemRefType::get({block_dim}, f32,
                                            mlir::MemRefLayoutAttrInterface(),
                                            smem_as);
  auto reduce_scratch = launch.addWorkgroupAttribution(scratch_type, loc);

  auto emitSourceLoadByConst = [&](int64_t source, mlir::Value col) -> mlir::Value {
    const bool partial_source = attrs.has_partial && source == attrs.block_count;
    if (partial_source) {
      return builder.create<mlir::memref::LoadOp>(
          loc, partial_arg, mlir::ValueRange{b_idx, t_idx, col});
    }
    auto c_source = builder.create<mlir::arith::ConstantIndexOp>(loc, source);
    return builder.create<mlir::memref::LoadOp>(
        loc, blocks_arg, mlir::ValueRange{c_source, b_idx, t_idx, col});
  };

  auto emitSourceLoadByIndex = [&](mlir::Value source,
                                   mlir::Value col) -> mlir::Value {
    if (!attrs.has_partial) {
      return builder.create<mlir::memref::LoadOp>(
          loc, blocks_arg, mlir::ValueRange{source, b_idx, t_idx, col});
    }
    auto is_partial = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, source, c_block_count);
    auto if_source = builder.create<mlir::scf::IfOp>(
        loc, mlir::TypeRange{f32}, is_partial, true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&if_source.getThenRegion().front());
      auto val = builder.create<mlir::memref::LoadOp>(
          loc, partial_arg, mlir::ValueRange{b_idx, t_idx, col});
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{val});
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&if_source.getElseRegion().front());
      auto val = builder.create<mlir::memref::LoadOp>(
          loc, blocks_arg, mlir::ValueRange{source, b_idx, t_idx, col});
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{val});
    }
    return if_source.getResult(0);
  };

  auto emitScratchSumReduction = [&]() {
    for (int64_t stride = block_dim / 2; stride >= 1; stride /= 2) {
      auto c_stride = builder.create<mlir::arith::ConstantIndexOp>(loc, stride);
      auto is_active_lane = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, tid, c_stride);
      auto if_reduce =
          builder.create<mlir::scf::IfOp>(loc, is_active_lane, false);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&if_reduce.getThenRegion().front());
        auto rhs_idx = builder.create<mlir::arith::AddIOp>(loc, tid, c_stride);
        auto lhs = builder.create<mlir::memref::LoadOp>(
            loc, reduce_scratch, mlir::ValueRange{tid});
        auto rhs = builder.create<mlir::memref::LoadOp>(
            loc, reduce_scratch, mlir::ValueRange{rhs_idx.getResult()});
        auto sum = builder.create<mlir::arith::AddFOp>(
            loc, lhs, rhs);
        builder.create<mlir::memref::StoreOp>(
            loc, sum.getResult(), reduce_scratch, mlir::ValueRange{tid});
      }
      builder.create<mlir::gpu::BarrierOp>(loc);
    }
  };

  auto emitSourceThreadSum = [&](int64_t source, bool dot_product) -> mlir::Value {
    auto loop = builder.create<mlir::scf::ForOp>(
        loc, tid, c_D, c_block, mlir::ValueRange{zero_f32.getResult()});
    {
      mlir::OpBuilder::InsertionGuard loop_guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      auto col = loop.getInductionVar();
      auto acc = loop.getRegionIterArgs()[0];
      auto val = emitSourceLoadByConst(source, col);
      mlir::Value term = val;
      if (dot_product) {
        auto c_source = builder.create<mlir::arith::ConstantIndexOp>(loc, source);
        auto inv_rms = builder.create<mlir::memref::LoadOp>(
            loc, inv_rms_smem, mlir::ValueRange{c_source});
        auto q = builder.create<mlir::memref::LoadOp>(
            loc, query_arg, mlir::ValueRange{col});
        auto normed = builder.create<mlir::arith::MulFOp>(
            loc, val, inv_rms.getResult());
        term = builder.create<mlir::arith::MulFOp>(
            loc, normed.getResult(), q).getResult();
      } else {
        term = builder.create<mlir::arith::MulFOp>(loc, val, val).getResult();
      }
      auto updated = builder.create<mlir::arith::AddFOp>(loc, acc, term);
      builder.create<mlir::scf::YieldOp>(
          loc, mlir::ValueRange{updated.getResult()});
    }
    return loop.getResult(0);
  };

  auto emitTidZero = [&](auto &&emit_body) {
    auto is_tid_zero = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, tid, c_zero);
    auto if_tid_zero =
        builder.create<mlir::scf::IfOp>(loc, is_tid_zero, false);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&if_tid_zero.getThenRegion().front());
    emit_body();
  };

  for (int64_t source = 0; source < active_sources; ++source) {
    auto c_source = builder.create<mlir::arith::ConstantIndexOp>(loc, source);

    auto partial_sumsq = emitSourceThreadSum(source, /*dot_product=*/false);
    builder.create<mlir::memref::StoreOp>(
        loc, partial_sumsq, reduce_scratch, mlir::ValueRange{tid});
    builder.create<mlir::gpu::BarrierOp>(loc);
    emitScratchSumReduction();

    emitTidZero([&]() {
      auto sumsq = builder.create<mlir::memref::LoadOp>(
          loc, reduce_scratch, mlir::ValueRange{c_zero});
      auto mean = builder.create<mlir::arith::MulFOp>(
          loc, sumsq, inv_d.getResult());
      auto mean_eps = builder.create<mlir::arith::AddFOp>(
          loc, mean.getResult(), eps.getResult());
      auto inv_rms = builder.create<mlir::math::RsqrtOp>(
          loc, mean_eps.getResult(), mlir::arith::FastMathFlags::fast);
      builder.create<mlir::memref::StoreOp>(
          loc, inv_rms.getResult(), inv_rms_smem, mlir::ValueRange{c_source});
    });
    builder.create<mlir::gpu::BarrierOp>(loc);

    auto partial_dot = emitSourceThreadSum(source, /*dot_product=*/true);
    builder.create<mlir::memref::StoreOp>(
        loc, partial_dot, reduce_scratch, mlir::ValueRange{tid});
    builder.create<mlir::gpu::BarrierOp>(loc);
    emitScratchSumReduction();

    emitTidZero([&]() {
      auto dot = builder.create<mlir::memref::LoadOp>(
          loc, reduce_scratch, mlir::ValueRange{c_zero});
      builder.create<mlir::memref::StoreOp>(
          loc, dot, score_smem, mlir::ValueRange{c_source});
    });
    builder.create<mlir::gpu::BarrierOp>(loc);
    }

  builder.create<mlir::gpu::BarrierOp>(loc);

  if (active_sources == 0) {
    auto zero_loop = builder.create<mlir::scf::ForOp>(loc, tid, c_D, c_block);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(zero_loop.getBody());
      auto col = zero_loop.getInductionVar();
      builder.create<mlir::memref::StoreOp>(
          loc, zero_f32.getResult(), out_arg, mlir::ValueRange{b_idx, t_idx, col});
    }
  } else {
    auto col_loop = builder.create<mlir::scf::ForOp>(loc, tid, c_D, c_block);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(col_loop.getBody());
      auto col = col_loop.getInductionVar();

      auto max_loop = builder.create<mlir::scf::ForOp>(
          loc, c_zero, c_active_sources, c_one,
          mlir::ValueRange{neg_inf.getResult()});
      {
        mlir::OpBuilder::InsertionGuard loop_guard(builder);
        builder.setInsertionPointToStart(max_loop.getBody());
        auto source = max_loop.getInductionVar();
        auto current = max_loop.getRegionIterArgs()[0];
        auto score = builder.create<mlir::memref::LoadOp>(
            loc, score_smem, mlir::ValueRange{source});
        auto updated = builder.create<mlir::arith::MaximumFOp>(
            loc, current, score);
        builder.create<mlir::scf::YieldOp>(
            loc, mlir::ValueRange{updated.getResult()});
      }

      auto denom_loop = builder.create<mlir::scf::ForOp>(
          loc, c_zero, c_active_sources, c_one,
          mlir::ValueRange{zero_f32.getResult()});
      {
        mlir::OpBuilder::InsertionGuard loop_guard(builder);
        builder.setInsertionPointToStart(denom_loop.getBody());
        auto source = denom_loop.getInductionVar();
        auto denom = denom_loop.getRegionIterArgs()[0];
        auto score = builder.create<mlir::memref::LoadOp>(
            loc, score_smem, mlir::ValueRange{source});
        auto shifted = builder.create<mlir::arith::SubFOp>(
            loc, score, max_loop.getResult(0));
        auto weight = builder.create<mlir::math::ExpOp>(loc, shifted);
        auto updated = builder.create<mlir::arith::AddFOp>(
            loc, denom, weight.getResult());
        builder.create<mlir::scf::YieldOp>(
            loc, mlir::ValueRange{updated.getResult()});
      }

      auto accum_loop = builder.create<mlir::scf::ForOp>(
          loc, c_zero, c_active_sources, c_one,
          mlir::ValueRange{zero_f32.getResult()});
      {
        mlir::OpBuilder::InsertionGuard loop_guard(builder);
        builder.setInsertionPointToStart(accum_loop.getBody());
        auto source = accum_loop.getInductionVar();
        auto acc = accum_loop.getRegionIterArgs()[0];
        auto score = builder.create<mlir::memref::LoadOp>(
            loc, score_smem, mlir::ValueRange{source});
        auto shifted = builder.create<mlir::arith::SubFOp>(
            loc, score, max_loop.getResult(0));
        auto weight_num = builder.create<mlir::math::ExpOp>(loc, shifted);
        auto weight = builder.create<mlir::arith::DivFOp>(
            loc, weight_num.getResult(), denom_loop.getResult(0));
        auto val = emitSourceLoadByIndex(source, col);
        auto prod = builder.create<mlir::arith::MulFOp>(
            loc, weight.getResult(), val);
        auto updated = builder.create<mlir::arith::AddFOp>(
            loc, acc, prod.getResult());
        builder.create<mlir::scf::YieldOp>(
            loc, mlir::ValueRange{updated.getResult()});
      }
      builder.create<mlir::memref::StoreOp>(
          loc, accum_loop.getResult(0), out_arg, mlir::ValueRange{b_idx, t_idx, col});
    }
  }

  builder.create<mlir::gpu::TerminatorOp>(loc);
  builder.setInsertionPointAfter(launch);
  builder.create<mlir::func::ReturnOp>(loc);

  module->setAttr("matcore.kernel_type",
                  mlir::StringAttr::get(&context, "region_block_attn_res"));
  module->setAttr("matcore.requested_target",
                  mlir::StringAttr::get(&context, target.canonical));
  module->setAttr("matcore.target_kind",
                  builder.getI32IntegerAttr(static_cast<int>(target.kind)));
  module->setAttr("matcore.region_strategy",
                  mlir::StringAttr::get(&context, "packed_block_attn_res_v1"));
  module->setAttr("matcore.fusion_launch_count",
                  builder.getI32IntegerAttr(1));
  module->setAttr("matcore.block_attn_res_max_blocks",
                  builder.getI32IntegerAttr(static_cast<int>(max_blocks)));
  module->setAttr("matcore.block_attn_res_block_count",
                  builder.getI32IntegerAttr(static_cast<int>(attrs.block_count)));

  return module;
}

}  // namespace matcore
