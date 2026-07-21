#include "fusion_emitter_internal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace matcore {
using namespace fusion_emit;

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::emitFamilyB(
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
        "FusionMlirEmitter: Family B runtime tensor count does not match graph inputs");
  }

  auto find_input_arg_index = [&](std::uint32_t value_id) -> int {
    auto it = std::find(graph.input_values.begin(), graph.input_values.end(),
                        value_id);
    if (it == graph.input_values.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(graph.input_values.begin(), it));
  };

  std::uint32_t first_matmul_id = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t second_matmul_id = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> glue_node_ids;
  bool seen_first = false;
  for (auto node_id : plan.node_ids) {
    const auto &node = graph.nodes.at(node_id);
    if (node.kind == OpKind::kMatMul) {
      if (!seen_first) {
        first_matmul_id = node_id;
        seen_first = true;
      } else {
        second_matmul_id = node_id;
      }
      continue;
    }
    if (seen_first && second_matmul_id == std::numeric_limits<std::uint32_t>::max()) {
      glue_node_ids.push_back(node_id);
    }
  }

  if (first_matmul_id == std::numeric_limits<std::uint32_t>::max() ||
      second_matmul_id == std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Family B plan requires two matmul nodes");
  }

  const auto &mm0_node = graph.nodes.at(first_matmul_id);
  const auto &mm1_node = graph.nodes.at(second_matmul_id);
  const auto &mm0_attrs = std::get<MatMulAttrs>(mm0_node.attrs);
  const auto &mm1_attrs = std::get<MatMulAttrs>(mm1_node.attrs);

  const auto &a_desc = graph.values.at(mm0_attrs.lhs.value_id);
  const auto &b_desc = graph.values.at(mm0_attrs.rhs.value_id);
  const auto &mm1_lhs_desc = graph.values.at(mm1_attrs.lhs.value_id);
  const auto &w_desc = graph.values.at(mm1_attrs.rhs.value_id);

  if (a_desc.shape.size() != 2 || b_desc.shape.size() != 2 ||
      mm1_lhs_desc.shape.size() != 2 || w_desc.shape.size() != 2) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B currently requires rank-2 tensors");
  }

  const int64_t M =
      mm0_attrs.lhs.transpose_last2 ? a_desc.shape.at(1) : a_desc.shape.at(0);
  const int64_t K1 =
      mm0_attrs.lhs.transpose_last2 ? a_desc.shape.at(0) : a_desc.shape.at(1);
  const int64_t b_k =
      mm0_attrs.rhs.transpose_last2 ? b_desc.shape.at(1) : b_desc.shape.at(0);
  const int64_t N1 =
      mm0_attrs.rhs.transpose_last2 ? b_desc.shape.at(0) : b_desc.shape.at(1);
  if (K1 != b_k) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B first matmul inner dimensions mismatch");
  }

  const int64_t M2 = mm1_attrs.lhs.transpose_last2 ? mm1_lhs_desc.shape.at(1)
                                                   : mm1_lhs_desc.shape.at(0);
  const int64_t K2 = mm1_attrs.lhs.transpose_last2 ? mm1_lhs_desc.shape.at(0)
                                                   : mm1_lhs_desc.shape.at(1);
  const int64_t w_k =
      mm1_attrs.rhs.transpose_last2 ? w_desc.shape.at(1) : w_desc.shape.at(0);
  const int64_t N2 =
      mm1_attrs.rhs.transpose_last2 ? w_desc.shape.at(0) : w_desc.shape.at(1);
  if (M2 != M || K2 != N1 || K2 != w_k) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B second matmul dimensions mismatch");
  }
  if (mm1_attrs.lhs.transpose_last2) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B does not support transposed intermediate lhs yet");
  }

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  auto f32 = builder.getF32Type();
  auto a_elem = getElementType(a_desc.dtype, builder);
  auto b_elem = getElementType(b_desc.dtype, builder);
  auto w_elem = getElementType(w_desc.dtype, builder);
  const std::uint32_t out_value_id = plan.output_value_ids.empty()
                                         ? mm1_node.outputs.front()
                                         : plan.output_value_ids.front();
  const auto &out_desc = graph.values.at(out_value_id);
  auto out_elem = getElementType(out_desc.dtype, builder);
  if (out_desc.shape.size() != 2 || out_desc.shape.at(0) != M ||
      out_desc.shape.at(1) != N2) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B output shape must match second matmul result");
  }

  llvm::SmallVector<FamilyAEpilogueSpec, 4> glue_specs;
  std::uint32_t current_glue_value_id = mm0_node.outputs.front();
  for (auto glue_id : glue_node_ids) {
    const auto &glue_node = graph.nodes.at(glue_id);
    if (glue_node.kind != OpKind::kElementwise) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family B only supports elementwise glue nodes");
    }
    if (glue_node.outputs.size() != 1) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family B glue nodes must produce exactly one output");
    }
    const auto &ew_attrs = std::get<ElementwiseAttrs>(glue_node.attrs);
    FamilyAEpilogueSpec spec;
    spec.kind = ew_attrs.kind;
    if (ew_attrs.inputs.size() == 1) {
      if (ew_attrs.inputs.front() != current_glue_value_id) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B unary glue must consume the fused value");
      }
      if (!isUnaryElementwiseKind(spec.kind)) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B unary glue kind is invalid");
      }
    } else if (ew_attrs.inputs.size() == 2) {
      int current_input_pos = -1;
      std::uint32_t boundary_value_id = 0;
      for (std::size_t i = 0; i < ew_attrs.inputs.size(); ++i) {
        if (ew_attrs.inputs[i] == current_glue_value_id) {
          current_input_pos = static_cast<int>(i);
        } else {
          boundary_value_id = ew_attrs.inputs[i];
        }
      }
      if (current_input_pos < 0) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B binary glue must consume the fused value");
      }
      const int boundary_arg_index = find_input_arg_index(boundary_value_id);
      if (boundary_arg_index < 0) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B binary glue boundary operand must be a graph input");
      }
      const auto &boundary_desc = graph.values.at(boundary_value_id);
      if (boundary_desc.shape.size() != 2 || boundary_desc.shape.at(0) != M ||
          boundary_desc.shape.at(1) != N1) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B binary glue currently requires exact-shape rank-2 boundary tensors");
      }
      spec.boundary_arg_index = boundary_arg_index;
      spec.value_input_pos = current_input_pos;
    } else {
      throw std::runtime_error(
          "FusionMlirEmitter: Family B glue nodes must be unary or binary");
    }
    glue_specs.push_back(spec);
    current_glue_value_id = glue_node.outputs.front();
  }
  if (current_glue_value_id != mm1_attrs.lhs.value_id) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B glue chain does not feed the second matmul lhs");
  }

  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(graph.input_values.size() + 1);
  for (std::uint32_t input_value_id : graph.input_values) {
    const auto &input_desc = graph.values.at(input_value_id);
    if (input_desc.shape.size() != 2) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family B currently requires rank-2 graph inputs");
    }
    arg_types.push_back(mlir::MemRefType::get(
        {input_desc.shape.at(0), input_desc.shape.at(1)},
        getElementType(input_desc.dtype, builder)));
  }
  const auto out_type = mlir::MemRefType::get({M, N2}, out_elem);
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

  const int a_arg_index = find_input_arg_index(mm0_attrs.lhs.value_id);
  const int b_arg_index = find_input_arg_index(mm0_attrs.rhs.value_id);
  const int w_arg_index = find_input_arg_index(mm1_attrs.rhs.value_id);
  if (a_arg_index < 0 || b_arg_index < 0 || w_arg_index < 0) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family B matmul boundary operands must be graph inputs");
  }
  auto a_arg = entry->getArgument(static_cast<unsigned>(a_arg_index));
  auto b_arg = entry->getArgument(static_cast<unsigned>(b_arg_index));
  auto w_arg = entry->getArgument(static_cast<unsigned>(w_arg_index));
  auto out_arg =
      entry->getArgument(static_cast<unsigned>(graph.input_values.size()));

  // Family B backend is currently validated for a conservative 16x16 tile.
  const int Br = 16;
  const int block_dim = 256;
  const int grid_m = static_cast<int>((M + Br - 1) / Br);
  const int grid_n = static_cast<int>((N2 + Br - 1) / Br);

  auto c_grid_m = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_m);
  auto c_grid_n = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_n);
  auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_dim);

  auto launch = builder.create<mlir::gpu::LaunchOp>(
      loc, c_grid_m, c_grid_n, c_one, c_block, c_one, c_one);
  builder.setInsertionPointToStart(&launch.getBody().front());

  auto tid = launch.getThreadIds().x;
  auto bid_m = launch.getBlockIds().x;
  auto bid_n = launch.getBlockIds().y;

  auto c_Br = builder.create<mlir::arith::ConstantIndexOp>(loc, Br);
  auto c_M = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
  auto c_K1 = builder.create<mlir::arith::ConstantIndexOp>(loc, K1);
  auto c_N1 = builder.create<mlir::arith::ConstantIndexOp>(loc, N1);
  auto c_N2 = builder.create<mlir::arith::ConstantIndexOp>(loc, N2);
  auto c_zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto c_total_elems =
      builder.create<mlir::arith::ConstantIndexOp>(loc, Br * Br);

  auto row_base = builder.create<mlir::arith::MulIOp>(loc, bid_m, c_Br);
  auto col_base = builder.create<mlir::arith::MulIOp>(loc, bid_n, c_Br);

  auto elem_loop =
      builder.create<mlir::scf::ForOp>(loc, tid, c_total_elems, c_block);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(elem_loop.getBody());
    auto elem_idx = elem_loop.getInductionVar();
    auto local_row = builder.create<mlir::arith::DivUIOp>(loc, elem_idx, c_Br);
    auto local_col = builder.create<mlir::arith::RemUIOp>(loc, elem_idx, c_Br);
    auto global_row = builder.create<mlir::arith::AddIOp>(loc, row_base, local_row);
    auto global_col2 = builder.create<mlir::arith::AddIOp>(loc, col_base, local_col);

    auto row_valid = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, global_row, c_M);
    auto col_valid = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, global_col2, c_N2);
    auto valid = builder.create<mlir::arith::AndIOp>(loc, row_valid, col_valid);
    auto if_valid = builder.create<mlir::scf::IfOp>(loc, valid, false);
    {
      mlir::OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(&if_valid.getThenRegion().front());
      auto zero_f32 = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(0.0f));
      auto n1_loop = builder.create<mlir::scf::ForOp>(
          loc, c_zero, c_N1, c_one, mlir::ValueRange{zero_f32.getResult()});
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(n1_loop.getBody());
        auto n1_idx = n1_loop.getInductionVar();
        auto acc_outer = n1_loop.getRegionIterArgs()[0];
        auto k_loop = builder.create<mlir::scf::ForOp>(
            loc, c_zero, c_K1, c_one, mlir::ValueRange{zero_f32.getResult()});
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(k_loop.getBody());
          auto k = k_loop.getInductionVar();
          auto acc_inner = k_loop.getRegionIterArgs()[0];
          mlir::Value a_row = global_row;
          mlir::Value a_col = k;
          if (mm0_attrs.lhs.transpose_last2) {
            a_row = k;
            a_col = global_row;
          }
          mlir::Value b_row = k;
          mlir::Value b_col = n1_idx;
          if (mm0_attrs.rhs.transpose_last2) {
            b_row = n1_idx;
            b_col = k;
          }
          auto a_val = builder.create<mlir::memref::LoadOp>(
              loc, a_arg, mlir::ValueRange{a_row, a_col});
          auto b_val = builder.create<mlir::memref::LoadOp>(
              loc, b_arg, mlir::ValueRange{b_row, b_col});
          mlir::Value a_f32 = a_val;
          mlir::Value b_f32 = b_val;
          if (a_elem.isa<mlir::FloatType>() && a_elem != f32) {
            a_f32 = builder.create<mlir::arith::ExtFOp>(loc, f32, a_val);
          } else if (a_elem.isa<mlir::IntegerType>()) {
            a_f32 = builder.create<mlir::arith::SIToFPOp>(loc, f32, a_val);
          }
          if (b_elem.isa<mlir::FloatType>() && b_elem != f32) {
            b_f32 = builder.create<mlir::arith::ExtFOp>(loc, f32, b_val);
          } else if (b_elem.isa<mlir::IntegerType>()) {
            b_f32 = builder.create<mlir::arith::SIToFPOp>(loc, f32, b_val);
          }
          auto prod = builder.create<mlir::arith::MulFOp>(loc, a_f32, b_f32);
          auto sum_inner = builder.create<mlir::arith::AddFOp>(loc, acc_inner, prod);
          builder.create<mlir::scf::YieldOp>(loc,
                                             mlir::ValueRange{sum_inner.getResult()});
        }
        mlir::Value transformed = k_loop.getResult(0);
        for (const FamilyAEpilogueSpec &spec : glue_specs) {
          if (spec.boundary_arg_index >= 0) {
            mlir::Value boundary_arg =
                entry->getArgument(static_cast<unsigned>(spec.boundary_arg_index));
            auto boundary_type =
                mlir::cast<mlir::MemRefType>(boundary_arg.getType());
            mlir::Value boundary_val = builder.create<mlir::memref::LoadOp>(
                loc, boundary_arg, mlir::ValueRange{global_row, n1_idx});
            mlir::Value boundary_f32 = castToF32(
                builder, loc, boundary_val, boundary_type.getElementType(), f32);
            if (spec.value_input_pos == 0) {
              transformed = emitBinaryElementwiseOnValues(
                  builder, loc, spec.kind, transformed, boundary_f32);
            } else {
              transformed = emitBinaryElementwiseOnValues(
                  builder, loc, spec.kind, boundary_f32, transformed);
            }
          } else {
            transformed =
                emitElementwiseOnValue(builder, loc, spec.kind, transformed, f32);
          }
        }
        mlir::Value w_row = n1_idx;
        mlir::Value w_col = global_col2;
        if (mm1_attrs.rhs.transpose_last2) {
          w_row = global_col2;
          w_col = n1_idx;
        }
        auto w_val = builder.create<mlir::memref::LoadOp>(
            loc, w_arg, mlir::ValueRange{w_row, w_col});
        mlir::Value w_f32 = w_val;
        if (w_elem.isa<mlir::FloatType>() && w_elem != f32) {
          w_f32 = builder.create<mlir::arith::ExtFOp>(loc, f32, w_val);
        } else if (w_elem.isa<mlir::IntegerType>()) {
          w_f32 = builder.create<mlir::arith::SIToFPOp>(loc, f32, w_val);
        }
        auto prod_outer = builder.create<mlir::arith::MulFOp>(loc, transformed, w_f32);
        auto sum_outer =
            builder.create<mlir::arith::AddFOp>(loc, acc_outer, prod_outer);
        builder.create<mlir::scf::YieldOp>(loc,
                                           mlir::ValueRange{sum_outer.getResult()});
      }
      mlir::Value store_val = n1_loop.getResult(0);
      if (out_elem.isa<mlir::FloatType>() && out_elem != f32) {
        store_val = builder.create<mlir::arith::TruncFOp>(loc, out_elem, store_val);
      } else if (out_elem.isa<mlir::IntegerType>()) {
        store_val = builder.create<mlir::arith::FPToSIOp>(loc, out_elem, store_val);
      }
      builder.create<mlir::memref::StoreOp>(
          loc, store_val, out_arg, mlir::ValueRange{global_row, global_col2});
    }
  }

  builder.setInsertionPointAfter(elem_loop);
  builder.create<mlir::gpu::TerminatorOp>(loc);
  builder.setInsertionPointAfter(launch);
  builder.create<mlir::func::ReturnOp>(loc);

  module->setAttr("matcore.kernel_type",
                  mlir::StringAttr::get(&context, "fused_family_b"));
  module->setAttr("matcore.requested_target",
                  mlir::StringAttr::get(&context, target.canonical));
  module->setAttr("matcore.target_kind",
                  builder.getI32IntegerAttr(static_cast<int>(target.kind)));
  module->setAttr("matcore.matmul_m", builder.getI32IntegerAttr(static_cast<int>(M)));
  module->setAttr("matcore.matmul_n", builder.getI32IntegerAttr(static_cast<int>(N2)));
  module->setAttr("matcore.matmul_k",
                  builder.getI32IntegerAttr(static_cast<int>(K1)));
  module->setAttr("matcore.max_regs",
                  builder.getI32IntegerAttr(plan.regs.total_regs));
  module->setAttr("matcore.fusion_pattern",
                  mlir::StringAttr::get(&context, "family_b"));
  module->setAttr("matcore.fusion_launch_count",
                  builder.getI32IntegerAttr(1));

  return module;
}

}  // namespace matcore
