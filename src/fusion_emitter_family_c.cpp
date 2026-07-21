#include "fusion_emitter_internal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace matcore {
using namespace fusion_emit;

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::emitFamilyC(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                      mlir::gpu::GPUDialect, mlir::math::MathDialect>();

  if (!kernel.graph.has_value()) {
    throw std::runtime_error("FusionMlirEmitter: missing graph for fused kernel");
  }
  const auto &graph = *kernel.graph;
  if (!tensors.empty() && tensors.size() != graph.input_values.size() + 1) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C runtime tensor count does not match graph inputs");
  }

  auto find_input_arg_index = [&](std::uint32_t value_id) -> int {
    auto it = std::find(graph.input_values.begin(), graph.input_values.end(),
                        value_id);
    if (it == graph.input_values.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(graph.input_values.begin(), it));
  };

  std::uint32_t score_matmul_id = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t softmax_id = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t output_matmul_id = std::numeric_limits<std::uint32_t>::max();
  for (auto node_id : plan.node_ids) {
    const auto &node = graph.nodes.at(node_id);
    if (node.kind == OpKind::kMatMul) {
      if (score_matmul_id == std::numeric_limits<std::uint32_t>::max()) {
        score_matmul_id = node_id;
      } else {
        output_matmul_id = node_id;
      }
    } else if (node.kind == OpKind::kSoftmax) {
      softmax_id = node_id;
    }
  }

  if (score_matmul_id == std::numeric_limits<std::uint32_t>::max() ||
      softmax_id == std::numeric_limits<std::uint32_t>::max() ||
      output_matmul_id == std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C requires matmul -> softmax -> matmul");
  }

  const auto &score_node = graph.nodes.at(score_matmul_id);
  const auto &softmax_node = graph.nodes.at(softmax_id);
  const auto &output_node = graph.nodes.at(output_matmul_id);
  const auto &score_attrs = std::get<MatMulAttrs>(score_node.attrs);
  const auto &softmax_attrs = std::get<SoftmaxAttrs>(softmax_node.attrs);
  const auto &output_attrs = std::get<MatMulAttrs>(output_node.attrs);

  const auto &q_desc = graph.values.at(score_attrs.lhs.value_id);
  const auto &k_desc = graph.values.at(score_attrs.rhs.value_id);
  const auto &v_desc = graph.values.at(output_attrs.rhs.value_id);
  const std::uint32_t out_value_id = plan.output_value_ids.empty()
                                         ? output_node.outputs.front()
                                         : plan.output_value_ids.front();
  const auto &out_desc = graph.values.at(out_value_id);

  if (q_desc.shape.size() != 2 || k_desc.shape.size() != 2 ||
      v_desc.shape.size() != 2 || out_desc.shape.size() != 2) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C currently requires rank-2 tensors");
  }
  if (softmax_attrs.input != score_node.outputs.front() ||
      output_attrs.lhs.value_id != softmax_node.outputs.front()) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C must be score matmul -> softmax -> output matmul");
  }
  if (output_attrs.lhs.transpose_last2) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C does not support transposed softmax lhs");
  }

  if (softmax_attrs.causal) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C causal softmax is not implemented");
  }
  if (softmax_attrs.axis != -1 && softmax_attrs.axis != 1) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C only supports last-dimension softmax");
  }

  const int64_t M =
      score_attrs.lhs.transpose_last2 ? q_desc.shape.at(1) : q_desc.shape.at(0);
  const int64_t Dh =
      score_attrs.lhs.transpose_last2 ? q_desc.shape.at(0) : q_desc.shape.at(1);
  const int64_t k_inner =
      score_attrs.rhs.transpose_last2 ? k_desc.shape.at(1) : k_desc.shape.at(0);
  const int64_t N =
      score_attrs.rhs.transpose_last2 ? k_desc.shape.at(0) : k_desc.shape.at(1);
  if (Dh != k_inner) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C score matmul inner dimensions mismatch");
  }

  const int64_t softmax_m = graph.values.at(softmax_attrs.input).shape.at(0);
  const int64_t softmax_n = graph.values.at(softmax_attrs.input).shape.at(1);
  if (softmax_m != M || softmax_n != N) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C softmax shape mismatch");
  }

  const int64_t pm_m = output_attrs.lhs.transpose_last2
                           ? graph.values.at(output_attrs.lhs.value_id).shape.at(1)
                           : graph.values.at(output_attrs.lhs.value_id).shape.at(0);
  const int64_t pm_k = output_attrs.lhs.transpose_last2
                           ? graph.values.at(output_attrs.lhs.value_id).shape.at(0)
                           : graph.values.at(output_attrs.lhs.value_id).shape.at(1);
  const int64_t v_k =
      output_attrs.rhs.transpose_last2 ? v_desc.shape.at(1) : v_desc.shape.at(0);
  const int64_t D =
      output_attrs.rhs.transpose_last2 ? v_desc.shape.at(0) : v_desc.shape.at(1);
  if (pm_m != M || pm_k != N || v_k != N) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C output matmul dimensions mismatch");
  }
  if (out_desc.shape.at(0) != M || out_desc.shape.at(1) != D) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C output shape must match attention result");
  }

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  auto f32 = builder.getF32Type();
  auto q_elem = getElementType(q_desc.dtype, builder);
  auto k_elem = getElementType(k_desc.dtype, builder);
  auto v_elem = getElementType(v_desc.dtype, builder);
  auto out_elem = getElementType(out_desc.dtype, builder);

  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(graph.input_values.size() + 1);
  for (std::uint32_t input_value_id : graph.input_values) {
    const auto &input_desc = graph.values.at(input_value_id);
    if (input_desc.shape.size() != 2) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family C currently requires rank-2 graph inputs");
    }
    arg_types.push_back(mlir::MemRefType::get(
        {input_desc.shape.at(0), input_desc.shape.at(1)},
        getElementType(input_desc.dtype, builder)));
  }
  const auto out_type =
      mlir::MemRefType::get({out_desc.shape.at(0), out_desc.shape.at(1)}, out_elem);
  arg_types.push_back(out_type);

  const std::string base_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const std::string entry_name = "fused_" + base_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name, builder.getFunctionType(arg_types, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setPublic();

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  const int q_arg_index = find_input_arg_index(score_attrs.lhs.value_id);
  const int k_arg_index = find_input_arg_index(score_attrs.rhs.value_id);
  const int v_arg_index = find_input_arg_index(output_attrs.rhs.value_id);
  if (q_arg_index < 0 || k_arg_index < 0 || v_arg_index < 0) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family C Q/K/V operands must be graph inputs");
  }
  auto q_arg = entry->getArgument(static_cast<unsigned>(q_arg_index));
  auto k_arg = entry->getArgument(static_cast<unsigned>(k_arg_index));
  auto v_arg = entry->getArgument(static_cast<unsigned>(v_arg_index));
  auto out_arg =
      entry->getArgument(static_cast<unsigned>(graph.input_values.size()));

  const int Br = plan.tile.br > 0 ? plan.tile.br : 32;
  const int Bc = plan.tile.bc > 0 ? plan.tile.bc : 32;
  const int requested_dtile = std::max(plan.tile.d > 0 ? plan.tile.d : 64, 64);
  const int Dtile = std::max(
      1, std::min(requested_dtile,
                  static_cast<int>(std::min<int64_t>(
                      D, static_cast<int64_t>(std::numeric_limits<int>::max())))));
  const int block_dim =
      plan.tile.threads_per_block > 0 ? plan.tile.threads_per_block : 128;

  const int grid_m = static_cast<int>((M + Br - 1) / Br);
  const int grid_d = static_cast<int>((D + Dtile - 1) / Dtile);

  auto c_grid_m = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_m);
  auto c_grid_d = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_d);
  auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_dim);

  auto launch = builder.create<mlir::gpu::LaunchOp>(
      loc, c_grid_m, c_grid_d, c_one, c_block, c_one, c_one);
  builder.setInsertionPointToStart(&launch.getBody().front());

  auto tid = launch.getThreadIds().x;
  auto bid_m = launch.getBlockIds().x;
  auto bid_d = launch.getBlockIds().y;

  auto smem_as = mlir::gpu::AddressSpaceAttr::get(
      &context, mlir::gpu::AddressSpace::Workgroup);
  auto accum_type = mlir::MemRefType::get({Br, Dtile}, f32,
                                          mlir::MemRefLayoutAttrInterface(),
                                          smem_as);
  auto accum_smem = launch.addWorkgroupAttribution(accum_type, loc);
  auto score_type = mlir::MemRefType::get({Br, Bc}, f32,
                                          mlir::MemRefLayoutAttrInterface(),
                                          smem_as);
  auto score_smem = launch.addWorkgroupAttribution(score_type, loc);
  auto row_state_type = mlir::MemRefType::get({Br}, f32,
                                              mlir::MemRefLayoutAttrInterface(),
                                              smem_as);
  auto row_m_smem = launch.addWorkgroupAttribution(row_state_type, loc);
  auto row_l_smem = launch.addWorkgroupAttribution(row_state_type, loc);
  auto row_scale_smem = launch.addWorkgroupAttribution(row_state_type, loc);

  auto c_zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto c_Br = builder.create<mlir::arith::ConstantIndexOp>(loc, Br);
  auto c_Bc = builder.create<mlir::arith::ConstantIndexOp>(loc, Bc);
  auto c_Dtile = builder.create<mlir::arith::ConstantIndexOp>(loc, Dtile);
  auto c_M = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
  auto c_N = builder.create<mlir::arith::ConstantIndexOp>(loc, N);
  auto c_Dh = builder.create<mlir::arith::ConstantIndexOp>(loc, Dh);
  auto c_D = builder.create<mlir::arith::ConstantIndexOp>(loc, D);
  auto c_total_accum =
      builder.create<mlir::arith::ConstantIndexOp>(loc, Br * Dtile);
  auto c_total_score =
      builder.create<mlir::arith::ConstantIndexOp>(loc, Br * Bc);
  auto zero_f32 = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(0.0f));
  auto neg_inf = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  auto row_base = builder.create<mlir::arith::MulIOp>(loc, bid_m, c_Br);
  auto col_base = builder.create<mlir::arith::MulIOp>(loc, bid_d, c_Dtile);

  auto emitScoreDot = [&](mlir::Value global_row,
                          mlir::Value key_row) -> mlir::Value {
    auto dot_loop = builder.create<mlir::scf::ForOp>(
        loc, c_zero, c_Dh, c_one, mlir::ValueRange{zero_f32.getResult()});
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(dot_loop.getBody());
      auto kk = dot_loop.getInductionVar();
      auto acc = dot_loop.getRegionIterArgs()[0];

      mlir::Value q_row = global_row;
      mlir::Value q_col = kk;
      if (score_attrs.lhs.transpose_last2) {
        q_row = kk;
        q_col = global_row;
      }
      mlir::Value k_row = kk;
      mlir::Value k_col = key_row;
      if (score_attrs.rhs.transpose_last2) {
        k_row = key_row;
        k_col = kk;
      }

      auto q_val = builder.create<mlir::memref::LoadOp>(
          loc, q_arg, mlir::ValueRange{q_row, q_col});
      auto k_val = builder.create<mlir::memref::LoadOp>(
          loc, k_arg, mlir::ValueRange{k_row, k_col});
      auto q_f32 = castToF32(builder, loc, q_val, q_elem, f32);
      auto k_f32 = castToF32(builder, loc, k_val, k_elem, f32);
      auto prod = builder.create<mlir::arith::MulFOp>(loc, q_f32, k_f32);
      auto sum = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
      builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum.getResult()});
    }
    return dot_loop.getResult(0);
  };

  auto init_loop =
      builder.create<mlir::scf::ForOp>(loc, tid, c_total_accum, c_block);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(init_loop.getBody());
    auto elem_idx = init_loop.getInductionVar();
    auto local_row = builder.create<mlir::arith::DivUIOp>(loc, elem_idx, c_Dtile);
    auto local_col = builder.create<mlir::arith::RemUIOp>(loc, elem_idx, c_Dtile);
    builder.create<mlir::memref::StoreOp>(
        loc, zero_f32.getResult(), accum_smem,
        mlir::ValueRange{local_row, local_col});
  }

  auto row_state_init_loop =
      builder.create<mlir::scf::ForOp>(loc, tid, c_Br, c_block);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(row_state_init_loop.getBody());
    auto local_row = row_state_init_loop.getInductionVar();
    builder.create<mlir::memref::StoreOp>(
        loc, neg_inf.getResult(), row_m_smem, mlir::ValueRange{local_row});
    builder.create<mlir::memref::StoreOp>(
        loc, zero_f32.getResult(), row_l_smem, mlir::ValueRange{local_row});
    builder.create<mlir::memref::StoreOp>(
        loc, zero_f32.getResult(), row_scale_smem, mlir::ValueRange{local_row});
  }

  builder.create<mlir::gpu::BarrierOp>(loc);

  auto kchunk_loop =
      builder.create<mlir::scf::ForOp>(loc, c_zero, c_N, c_Bc);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(kchunk_loop.getBody());
    auto chunk_base = kchunk_loop.getInductionVar();

    auto score_loop =
        builder.create<mlir::scf::ForOp>(loc, tid, c_total_score, c_block);
    {
      mlir::OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(score_loop.getBody());
      auto score_idx = score_loop.getInductionVar();
      auto local_row = builder.create<mlir::arith::DivUIOp>(loc, score_idx, c_Bc);
      auto chunk_offset =
          builder.create<mlir::arith::RemUIOp>(loc, score_idx, c_Bc);
      auto global_row =
          builder.create<mlir::arith::AddIOp>(loc, row_base, local_row);
      auto key_row =
          builder.create<mlir::arith::AddIOp>(loc, chunk_base, chunk_offset);
      auto row_valid = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, global_row, c_M);
      auto key_valid = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, key_row, c_N);
      auto score_valid =
          builder.create<mlir::arith::AndIOp>(loc, row_valid, key_valid);
      auto if_score_valid =
          builder.create<mlir::scf::IfOp>(loc, score_valid, true);
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(&if_score_valid.getThenRegion().front());
        auto score = emitScoreDot(global_row, key_row);
        builder.create<mlir::memref::StoreOp>(
            loc, score, score_smem,
            mlir::ValueRange{local_row, chunk_offset});
      }
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(&if_score_valid.getElseRegion().front());
        builder.create<mlir::memref::StoreOp>(
            loc, neg_inf.getResult(), score_smem,
            mlir::ValueRange{local_row, chunk_offset});
      }
    }

    builder.create<mlir::gpu::BarrierOp>(loc);

    auto row_reduce_loop =
        builder.create<mlir::scf::ForOp>(loc, tid, c_Br, c_block);
    {
      mlir::OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(row_reduce_loop.getBody());
      auto local_row = row_reduce_loop.getInductionVar();
      auto global_row =
          builder.create<mlir::arith::AddIOp>(loc, row_base, local_row);
      auto row_valid = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, global_row, c_M);
      auto if_row_valid =
          builder.create<mlir::scf::IfOp>(loc, row_valid, false);
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(&if_row_valid.getThenRegion().front());

        auto chunk_max_loop =
            builder.create<mlir::scf::ForOp>(
                loc, c_zero, c_Bc, c_one,
                mlir::ValueRange{neg_inf.getResult()});
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(chunk_max_loop.getBody());
          auto chunk_offset = chunk_max_loop.getInductionVar();
          auto current_max = chunk_max_loop.getRegionIterArgs()[0];
          auto score = builder.create<mlir::memref::LoadOp>(
              loc, score_smem, mlir::ValueRange{local_row, chunk_offset});
          auto new_max =
              builder.create<mlir::arith::MaximumFOp>(loc, current_max, score);
          builder.create<mlir::scf::YieldOp>(
              loc, mlir::ValueRange{new_max.getResult()});
        }

        auto m_prev = builder.create<mlir::memref::LoadOp>(
            loc, row_m_smem, mlir::ValueRange{local_row});
        auto m_new = builder.create<mlir::arith::MaximumFOp>(
            loc, m_prev, chunk_max_loop.getResult(0));
        auto m_delta = builder.create<mlir::arith::SubFOp>(loc, m_prev, m_new);
        auto correction = builder.create<mlir::math::ExpOp>(loc, m_delta);
        auto l_prev = builder.create<mlir::memref::LoadOp>(
            loc, row_l_smem, mlir::ValueRange{local_row});
        auto l_scaled = builder.create<mlir::arith::MulFOp>(
            loc, correction, l_prev);

        auto weight_sum_loop =
            builder.create<mlir::scf::ForOp>(
                loc, c_zero, c_Bc, c_one,
                mlir::ValueRange{l_scaled.getResult()});
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(weight_sum_loop.getBody());
          auto chunk_offset = weight_sum_loop.getInductionVar();
          auto l_acc = weight_sum_loop.getRegionIterArgs()[0];
          auto score = builder.create<mlir::memref::LoadOp>(
              loc, score_smem, mlir::ValueRange{local_row, chunk_offset});
          auto shifted = builder.create<mlir::arith::SubFOp>(loc, score, m_new);
          auto weight = builder.create<mlir::math::ExpOp>(loc, shifted);
          builder.create<mlir::memref::StoreOp>(
              loc, weight.getResult(), score_smem,
              mlir::ValueRange{local_row, chunk_offset});
          auto l_new = builder.create<mlir::arith::AddFOp>(
              loc, l_acc, weight.getResult());
          builder.create<mlir::scf::YieldOp>(
              loc, mlir::ValueRange{l_new.getResult()});
        }

        builder.create<mlir::memref::StoreOp>(
            loc, m_new.getResult(), row_m_smem, mlir::ValueRange{local_row});
        builder.create<mlir::memref::StoreOp>(
            loc, weight_sum_loop.getResult(0), row_l_smem,
            mlir::ValueRange{local_row});
        builder.create<mlir::memref::StoreOp>(
            loc, correction.getResult(), row_scale_smem,
            mlir::ValueRange{local_row});
      }
    }

    builder.create<mlir::gpu::BarrierOp>(loc);

    auto accum_loop =
        builder.create<mlir::scf::ForOp>(loc, tid, c_total_accum, c_block);
    {
      mlir::OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(accum_loop.getBody());
      auto elem_idx = accum_loop.getInductionVar();
      auto local_row = builder.create<mlir::arith::DivUIOp>(loc, elem_idx, c_Dtile);
      auto local_col = builder.create<mlir::arith::RemUIOp>(loc, elem_idx, c_Dtile);
      auto global_row =
          builder.create<mlir::arith::AddIOp>(loc, row_base, local_row);
      auto global_col =
          builder.create<mlir::arith::AddIOp>(loc, col_base, local_col);
      auto row_valid = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, global_row, c_M);
      auto col_valid = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ult, global_col, c_D);
      auto elem_valid =
          builder.create<mlir::arith::AndIOp>(loc, row_valid, col_valid);
      auto if_elem_valid =
          builder.create<mlir::scf::IfOp>(loc, elem_valid, false);
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(&if_elem_valid.getThenRegion().front());
        auto prev = builder.create<mlir::memref::LoadOp>(
            loc, accum_smem, mlir::ValueRange{local_row, local_col});
        auto correction = builder.create<mlir::memref::LoadOp>(
            loc, row_scale_smem, mlir::ValueRange{local_row});
        auto scaled =
            builder.create<mlir::arith::MulFOp>(loc, correction, prev);
        auto d_accum_loop =
            builder.create<mlir::scf::ForOp>(
                loc, c_zero, c_Bc, c_one,
                mlir::ValueRange{scaled.getResult()});
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(d_accum_loop.getBody());
          auto chunk_offset = d_accum_loop.getInductionVar();
          auto acc = d_accum_loop.getRegionIterArgs()[0];
          auto key_row =
              builder.create<mlir::arith::AddIOp>(loc, chunk_base, chunk_offset);
          auto key_valid = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ult, key_row, c_N);
          auto if_key_valid =
              builder.create<mlir::scf::IfOp>(
                  loc, mlir::TypeRange{f32}, key_valid, true);
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(
                &if_key_valid.getThenRegion().front());
            mlir::Value v_row = key_row;
            mlir::Value v_col = global_col;
            if (output_attrs.rhs.transpose_last2) {
              v_row = global_col;
              v_col = key_row;
            }
            auto weight = builder.create<mlir::memref::LoadOp>(
                loc, score_smem, mlir::ValueRange{local_row, chunk_offset});
            auto v_val = builder.create<mlir::memref::LoadOp>(
                loc, v_arg, mlir::ValueRange{v_row, v_col});
            auto v_f32 = castToF32(builder, loc, v_val, v_elem, f32);
            auto prod =
                builder.create<mlir::arith::MulFOp>(loc, weight, v_f32);
            auto updated =
                builder.create<mlir::arith::AddFOp>(loc, acc, prod);
            builder.create<mlir::scf::YieldOp>(
                loc, mlir::ValueRange{updated.getResult()});
          }
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(
                &if_key_valid.getElseRegion().front());
            builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc});
          }
          builder.create<mlir::scf::YieldOp>(
              loc, mlir::ValueRange{if_key_valid.getResult(0)});
        }
        builder.create<mlir::memref::StoreOp>(
            loc, d_accum_loop.getResult(0), accum_smem,
            mlir::ValueRange{local_row, local_col});
      }
    }

    builder.create<mlir::gpu::BarrierOp>(loc);
  }

  auto store_loop =
      builder.create<mlir::scf::ForOp>(loc, tid, c_total_accum, c_block);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(store_loop.getBody());
    auto elem_idx = store_loop.getInductionVar();
    auto local_row = builder.create<mlir::arith::DivUIOp>(loc, elem_idx, c_Dtile);
    auto local_col = builder.create<mlir::arith::RemUIOp>(loc, elem_idx, c_Dtile);
    auto global_row =
        builder.create<mlir::arith::AddIOp>(loc, row_base, local_row);
    auto global_col =
        builder.create<mlir::arith::AddIOp>(loc, col_base, local_col);
    auto row_valid = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, global_row, c_M);
    auto col_valid = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, global_col, c_D);
    auto elem_valid =
        builder.create<mlir::arith::AndIOp>(loc, row_valid, col_valid);
    auto if_elem_valid =
        builder.create<mlir::scf::IfOp>(loc, elem_valid, false);
    {
      mlir::OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(&if_elem_valid.getThenRegion().front());
      auto acc = builder.create<mlir::memref::LoadOp>(
          loc, accum_smem, mlir::ValueRange{local_row, local_col});
      auto final_l = builder.create<mlir::memref::LoadOp>(
          loc, row_l_smem, mlir::ValueRange{local_row});
      auto normalized = builder.create<mlir::arith::DivFOp>(loc, acc, final_l);
      auto store_val = castFromF32(builder, loc, normalized, out_elem, f32);
      builder.create<mlir::memref::StoreOp>(
          loc, store_val, out_arg, mlir::ValueRange{global_row, global_col});
    }
  }

  builder.setInsertionPointAfter(store_loop);
  builder.create<mlir::gpu::TerminatorOp>(loc);
  builder.setInsertionPointAfter(launch);
  builder.create<mlir::func::ReturnOp>(loc);

  module->setAttr("matcore.kernel_type",
                  mlir::StringAttr::get(&context, "fused_family_c"));
  module->setAttr("matcore.requested_target",
                  mlir::StringAttr::get(&context, target.canonical));
  module->setAttr("matcore.target_kind",
                  builder.getI32IntegerAttr(static_cast<int>(target.kind)));
  module->setAttr("matcore.matmul_m", builder.getI32IntegerAttr(static_cast<int>(M)));
  module->setAttr("matcore.matmul_n", builder.getI32IntegerAttr(static_cast<int>(D)));
  module->setAttr("matcore.matmul_k",
                  builder.getI32IntegerAttr(static_cast<int>(Dh)));
  module->setAttr("matcore.max_regs",
                  builder.getI32IntegerAttr(plan.regs.total_regs));
  module->setAttr("matcore.fusion_pattern",
                  mlir::StringAttr::get(&context, "family_c"));
  module->setAttr("matcore.family_c_strategy",
                  mlir::StringAttr::get(
                      &context, "score_cached_block_coop_dtile64"));
  module->setAttr("matcore.family_c_dtile",
                  builder.getI32IntegerAttr(Dtile));
  module->setAttr("matcore.use_online_softmax",
                  builder.getBoolAttr(plan.use_online_softmax));
  module->setAttr("matcore.fusion_launch_count",
                  builder.getI32IntegerAttr(1));

  return module;
}

}  // namespace matcore
