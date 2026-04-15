#include "matcore/fusion_emitter.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace matcore {
namespace {

mlir::Type getElementType(TensorDType dtype, mlir::OpBuilder &builder) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return builder.getF32Type();
    case TensorDType::kFloat16:
      return builder.getF16Type();
    case TensorDType::kBFloat16:
      return builder.getBF16Type();
    case TensorDType::kInt8:
      return builder.getI8Type();
    case TensorDType::kInt32:
      return builder.getI32Type();
    case TensorDType::kFloat8E4M3FN:
      return builder.getFloat8E4M3FNType();
  }
  throw std::runtime_error("FusionMlirEmitter: unsupported tensor dtype");
}

mlir::Value emitElementwiseOnValue(mlir::OpBuilder &builder, mlir::Location loc,
                                   ElementwiseKind kind, mlir::Value val,
                                   mlir::Type f32) {
  (void)f32;
  switch (kind) {
    case ElementwiseKind::kReLU: {
      auto zero = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(0.0f));
      return builder.create<mlir::arith::MaximumFOp>(loc, val, zero).getResult();
    }
    case ElementwiseKind::kExp:
      return builder.create<mlir::math::ExpOp>(loc, val).getResult();
    case ElementwiseKind::kLog:
      return builder.create<mlir::math::LogOp>(loc, val).getResult();
    case ElementwiseKind::kTanh:
      return builder.create<mlir::math::TanhOp>(loc, val).getResult();
    case ElementwiseKind::kSqrt:
      return builder.create<mlir::math::SqrtOp>(loc, val).getResult();
    case ElementwiseKind::kNeg:
      return builder.create<mlir::arith::NegFOp>(loc, val).getResult();
    case ElementwiseKind::kAbs:
      return builder.create<mlir::math::AbsFOp>(loc, val).getResult();
    case ElementwiseKind::kSigmoid: {
      auto one = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(1.0f));
      auto neg = builder.create<mlir::arith::NegFOp>(loc, val);
      auto exp_neg = builder.create<mlir::math::ExpOp>(loc, neg.getResult());
      auto denom = builder.create<mlir::arith::AddFOp>(
          loc, one.getResult(), exp_neg.getResult());
      return builder
          .create<mlir::arith::DivFOp>(loc, one.getResult(), denom.getResult())
          .getResult();
    }
    case ElementwiseKind::kGELU: {
      auto half = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(0.5f));
      auto one = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(1.0f));
      auto coeff = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(0.044715f));
      auto sqrt2pi = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getF32FloatAttr(0.7978845608f));
      auto x2 = builder.create<mlir::arith::MulFOp>(loc, val, val);
      auto x3 = builder.create<mlir::arith::MulFOp>(loc, x2.getResult(), val);
      auto cx3 = builder.create<mlir::arith::MulFOp>(loc, coeff.getResult(),
                                                     x3.getResult());
      auto inner = builder.create<mlir::arith::AddFOp>(loc, val, cx3.getResult());
      auto scaled = builder.create<mlir::arith::MulFOp>(loc, sqrt2pi.getResult(),
                                                        inner.getResult());
      auto tanh_val =
          builder.create<mlir::math::TanhOp>(loc, scaled.getResult());
      auto one_plus = builder.create<mlir::arith::AddFOp>(
          loc, one.getResult(), tanh_val.getResult());
      auto half_x =
          builder.create<mlir::arith::MulFOp>(loc, half.getResult(), val);
      return builder
          .create<mlir::arith::MulFOp>(loc, half_x.getResult(),
                                       one_plus.getResult())
          .getResult();
    }
    default:
      return val;
  }
}

mlir::Value castToF32(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type elem_type,
                      mlir::Type f32) {
  if (elem_type.isa<mlir::FloatType>() && elem_type != f32) {
    return builder.create<mlir::arith::ExtFOp>(loc, f32, value).getResult();
  }
  if (elem_type.isa<mlir::IntegerType>()) {
    return builder.create<mlir::arith::SIToFPOp>(loc, f32, value).getResult();
  }
  return value;
}

mlir::Value castFromF32(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, mlir::Type elem_type,
                        mlir::Type f32) {
  if (elem_type.isa<mlir::FloatType>() && elem_type != f32) {
    return builder.create<mlir::arith::TruncFOp>(loc, elem_type, value).getResult();
  }
  if (elem_type.isa<mlir::IntegerType>()) {
    return builder.create<mlir::arith::FPToSIOp>(loc, elem_type, value).getResult();
  }
  return value;
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::Emit(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  switch (plan.pattern) {
    case FusionPatternKind::kMatmulElementwise:
    case FusionPatternKind::kElementwiseMatmul:
      return emitFamilyA(kernel, plan, tensors, target, context);
    case FusionPatternKind::kMatmulElementwiseMatmul:
    case FusionPatternKind::kGenericTileChain:
      return emitFamilyB(kernel, plan, tensors, target, context);
    case FusionPatternKind::kMatmulSoftmaxMatmul:
      return emitFamilyC(kernel, plan, tensors, target, context);
    default:
      throw std::runtime_error("FusionMlirEmitter: unsupported pattern");
  }
}

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::emitFamilyA(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  (void)tensors;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                      mlir::gpu::GPUDialect, mlir::math::MathDialect,
                      mlir::linalg::LinalgDialect>();

  if (!kernel.graph.has_value()) {
    throw std::runtime_error("FusionMlirEmitter: missing graph for fused kernel");
  }
  const auto &graph = *kernel.graph;

  std::uint32_t matmul_node_id = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> epilogue_node_ids;
  for (auto node_id : plan.node_ids) {
    const auto &node = graph.nodes.at(node_id);
    if (node.kind == OpKind::kMatMul) {
      matmul_node_id = node_id;
    } else {
      epilogue_node_ids.push_back(node_id);
    }
  }

  if (matmul_node_id == std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Family A plan has no matmul node");
  }

  const auto &mm_node = graph.nodes.at(matmul_node_id);
  const auto &mm_attrs = std::get<MatMulAttrs>(mm_node.attrs);
  const auto &lhs_desc = graph.values.at(mm_attrs.lhs.value_id);
  const auto &rhs_desc = graph.values.at(mm_attrs.rhs.value_id);

  if (lhs_desc.shape.size() != 2 || rhs_desc.shape.size() != 2) {
    throw std::runtime_error("FusionMlirEmitter: Family A currently requires rank-2 tensors");
  }

  const int64_t M =
      mm_attrs.lhs.transpose_last2 ? lhs_desc.shape.at(1) : lhs_desc.shape.at(0);
  const int64_t K =
      mm_attrs.lhs.transpose_last2 ? lhs_desc.shape.at(0) : lhs_desc.shape.at(1);
  const int64_t rhs_k =
      mm_attrs.rhs.transpose_last2 ? rhs_desc.shape.at(1) : rhs_desc.shape.at(0);
  const int64_t N =
      mm_attrs.rhs.transpose_last2 ? rhs_desc.shape.at(0) : rhs_desc.shape.at(1);
  if (K != rhs_k) {
    throw std::runtime_error("FusionMlirEmitter: matmul inner dimensions mismatch");
  }

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  auto lhs_elem = getElementType(lhs_desc.dtype, builder);
  auto rhs_elem = getElementType(rhs_desc.dtype, builder);

  const std::uint32_t out_value_id =
      plan.output_value_ids.empty() ? mm_node.outputs.front() : plan.output_value_ids.front();
  const auto &out_desc = graph.values.at(out_value_id);
  auto out_elem = getElementType(out_desc.dtype, builder);

  const auto lhs_type = mlir::MemRefType::get({M, K}, lhs_elem);
  const auto rhs_type = mlir::MemRefType::get({K, N}, rhs_elem);
  const auto out_type = mlir::MemRefType::get({M, N}, out_elem);

  const std::string base_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const std::string entry_name = "fused_" + base_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name, builder.getFunctionType({lhs_type, rhs_type, out_type}, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setPublic();

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto lhs_arg = entry->getArgument(0);
  auto rhs_arg = entry->getArgument(1);
  auto out_arg = entry->getArgument(2);

  // Transposes not supported with MMA path (tests use non-transposed).
  if (mm_attrs.lhs.transpose_last2 || mm_attrs.rhs.transpose_last2) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family A MMA path does not support transposed inputs yet");
  }

  // Emit linalg.matmul on memrefs — the MMA transform pipeline in the
  // lowering stage will tile this and rewrite it to nvgpu.mma.sync.
  // Output is pre-zeroed by the runtime (C += A*B with C=0 → C = A*B).
  builder.create<mlir::linalg::MatmulOp>(
      loc,
      mlir::ValueRange{lhs_arg, rhs_arg},
      mlir::ValueRange{out_arg});
  builder.create<mlir::func::ReturnOp>(loc);

  // Store epilogue info as module attributes for FusionEpiloguePass.
  llvm::SmallVector<mlir::Attribute> epilogue_kinds;
  for (auto epi_node_id : epilogue_node_ids) {
    const auto &epi_node = graph.nodes.at(epi_node_id);
    if (epi_node.kind != OpKind::kElementwise) {
      continue;
    }
    const auto &ew_attrs = std::get<ElementwiseAttrs>(epi_node.attrs);
    if (ew_attrs.inputs.size() > 1) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family A binary epilogues are not yet implemented");
    }
    epilogue_kinds.push_back(
        builder.getI32IntegerAttr(static_cast<int>(ew_attrs.kind)));
  }
  if (!epilogue_kinds.empty()) {
    module->setAttr("matcore.fusion_epilogue_kinds",
                    builder.getArrayAttr(epilogue_kinds));
  }

  // Dtype attrs — required by decodeMatmulSignatureFromModule() to enable
  // MMA rewrite (rewrite_to_mma_sync). Without these, defaults to f32 and
  // tensor core path may not activate.
  module->setAttr("matcore.lhs_dtype", mlir::TypeAttr::get(lhs_elem));
  module->setAttr("matcore.rhs_dtype", mlir::TypeAttr::get(rhs_elem));
  module->setAttr("matcore.out_dtype", mlir::TypeAttr::get(out_elem));

  module->setAttr("matcore.kernel_type",
                  mlir::StringAttr::get(&context, "fused_family_a"));
  module->setAttr("matcore.requested_target",
                  mlir::StringAttr::get(&context, target.canonical));
  module->setAttr("matcore.target_kind",
                  builder.getI32IntegerAttr(static_cast<int>(target.kind)));
  module->setAttr("matcore.matmul_m", builder.getI32IntegerAttr(static_cast<int>(M)));
  module->setAttr("matcore.matmul_n", builder.getI32IntegerAttr(static_cast<int>(N)));
  module->setAttr("matcore.matmul_k", builder.getI32IntegerAttr(static_cast<int>(K)));
  module->setAttr("matcore.max_regs",
                  builder.getI32IntegerAttr(plan.regs.total_regs));
  module->setAttr("matcore.fusion_pattern",
                  mlir::StringAttr::get(&context, "family_a"));

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::emitFamilyB(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  (void)tensors;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                      mlir::gpu::GPUDialect, mlir::math::MathDialect>();

  if (!kernel.graph.has_value()) {
    throw std::runtime_error("FusionMlirEmitter: missing graph for fused kernel");
  }
  const auto &graph = *kernel.graph;

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

  const auto a_type = mlir::MemRefType::get({M, K1}, a_elem);
  const auto b_type = mlir::MemRefType::get({K1, N1}, b_elem);
  const auto w_type = mlir::MemRefType::get({N1, N2}, w_elem);
  const auto out_type = mlir::MemRefType::get({M, N2}, out_elem);

  const std::string base_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const std::string entry_name = "fused_" + base_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name,
      builder.getFunctionType({a_type, b_type, w_type, out_type}, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setPublic();

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto a_arg = entry->getArgument(0);
  auto b_arg = entry->getArgument(1);
  auto w_arg = entry->getArgument(2);
  auto out_arg = entry->getArgument(3);

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
        for (auto glue_id : glue_node_ids) {
          const auto &glue_node = graph.nodes.at(glue_id);
          if (glue_node.kind != OpKind::kElementwise) {
            throw std::runtime_error(
                "FusionMlirEmitter: Family B only supports elementwise glue nodes");
          }
          const auto &ew_attrs = std::get<ElementwiseAttrs>(glue_node.attrs);
          if (ew_attrs.inputs.size() > 1) {
            throw std::runtime_error(
                "FusionMlirEmitter: Family B binary elementwise glue is not implemented");
          }
          transformed =
              emitElementwiseOnValue(builder, loc, ew_attrs.kind, transformed, f32);
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

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::emitFamilyC(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  (void)tensors;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                      mlir::gpu::GPUDialect, mlir::math::MathDialect>();

  if (!kernel.graph.has_value()) {
    throw std::runtime_error("FusionMlirEmitter: missing graph for fused kernel");
  }
  const auto &graph = *kernel.graph;

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

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  auto f32 = builder.getF32Type();
  auto q_elem = getElementType(q_desc.dtype, builder);
  auto k_elem = getElementType(k_desc.dtype, builder);
  auto v_elem = getElementType(v_desc.dtype, builder);
  auto out_elem = getElementType(out_desc.dtype, builder);

  const auto q_type =
      mlir::MemRefType::get({q_desc.shape.at(0), q_desc.shape.at(1)}, q_elem);
  const auto k_type =
      mlir::MemRefType::get({k_desc.shape.at(0), k_desc.shape.at(1)}, k_elem);
  const auto v_type =
      mlir::MemRefType::get({v_desc.shape.at(0), v_desc.shape.at(1)}, v_elem);
  const auto out_type =
      mlir::MemRefType::get({out_desc.shape.at(0), out_desc.shape.at(1)}, out_elem);

  const std::string base_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const std::string entry_name = "fused_" + base_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name,
      builder.getFunctionType({q_type, k_type, v_type, out_type}, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setPublic();

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto q_arg = entry->getArgument(0);
  auto k_arg = entry->getArgument(1);
  auto v_arg = entry->getArgument(2);
  auto out_arg = entry->getArgument(3);

  const int Br = plan.tile.br > 0 ? plan.tile.br : 32;
  const int Bc = plan.tile.bc > 0 ? plan.tile.bc : 32;
  const int Dtile = plan.tile.d > 0 ? plan.tile.d : 16;
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

  builder.create<mlir::gpu::BarrierOp>(loc);

  auto row_loop = builder.create<mlir::scf::ForOp>(loc, tid, c_Br, c_block);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(row_loop.getBody());
    auto local_row = row_loop.getInductionVar();
    auto global_row = builder.create<mlir::arith::AddIOp>(loc, row_base, local_row);
    auto row_valid = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, global_row, c_M);
    auto if_row_valid = builder.create<mlir::scf::IfOp>(loc, row_valid, false);
    {
      mlir::OpBuilder::InsertionGuard guard2(builder);
      builder.setInsertionPointToStart(&if_row_valid.getThenRegion().front());

      auto kchunk_loop = builder.create<mlir::scf::ForOp>(
          loc, c_zero, c_N, c_Bc,
          mlir::ValueRange{neg_inf.getResult(), zero_f32.getResult()});
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(kchunk_loop.getBody());
        auto chunk_base = kchunk_loop.getInductionVar();
        auto m_prev = kchunk_loop.getRegionIterArgs()[0];
        auto l_prev = kchunk_loop.getRegionIterArgs()[1];

        auto chunk_max_loop = builder.create<mlir::scf::ForOp>(
            loc, c_zero, c_Bc, c_one, mlir::ValueRange{neg_inf.getResult()});
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(chunk_max_loop.getBody());
          auto chunk_offset = chunk_max_loop.getInductionVar();
          auto current_max = chunk_max_loop.getRegionIterArgs()[0];
          auto key_row =
              builder.create<mlir::arith::AddIOp>(loc, chunk_base, chunk_offset);
          auto key_valid = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ult, key_row, c_N);
          auto if_key_valid =
              builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, key_valid, true);
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(&if_key_valid.getThenRegion().front());
            auto score = emitScoreDot(global_row, key_row);
            auto new_max =
                builder.create<mlir::arith::MaximumFOp>(loc, current_max, score);
            builder.create<mlir::scf::YieldOp>(
                loc, mlir::ValueRange{new_max.getResult()});
          }
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(&if_key_valid.getElseRegion().front());
            builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{current_max});
          }
          builder.create<mlir::scf::YieldOp>(
              loc, mlir::ValueRange{if_key_valid.getResult(0)});
        }

        auto m_new = builder.create<mlir::arith::MaximumFOp>(
            loc, m_prev, chunk_max_loop.getResult(0));
        auto m_delta = builder.create<mlir::arith::SubFOp>(loc, m_prev, m_new);
        auto correction = builder.create<mlir::math::ExpOp>(loc, m_delta);
        auto l_scaled = builder.create<mlir::arith::MulFOp>(loc, correction, l_prev);

        auto rescale_loop =
            builder.create<mlir::scf::ForOp>(loc, c_zero, c_Dtile, c_one);
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(rescale_loop.getBody());
          auto local_col = rescale_loop.getInductionVar();
          auto global_col =
              builder.create<mlir::arith::AddIOp>(loc, col_base, local_col);
          auto col_valid = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ult, global_col, c_D);
          auto if_col_valid = builder.create<mlir::scf::IfOp>(loc, col_valid, false);
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(&if_col_valid.getThenRegion().front());
            auto prev = builder.create<mlir::memref::LoadOp>(
                loc, accum_smem, mlir::ValueRange{local_row, local_col});
            auto scaled =
                builder.create<mlir::arith::MulFOp>(loc, correction, prev);
            builder.create<mlir::memref::StoreOp>(
                loc, scaled.getResult(), accum_smem,
                mlir::ValueRange{local_row, local_col});
          }
        }

        auto chunk_accum_loop = builder.create<mlir::scf::ForOp>(
            loc, c_zero, c_Bc, c_one, mlir::ValueRange{l_scaled.getResult()});
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(chunk_accum_loop.getBody());
          auto chunk_offset = chunk_accum_loop.getInductionVar();
          auto l_acc = chunk_accum_loop.getRegionIterArgs()[0];
          auto key_row =
              builder.create<mlir::arith::AddIOp>(loc, chunk_base, chunk_offset);
          auto key_valid = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ult, key_row, c_N);
          auto if_key_valid =
              builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, key_valid, true);
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(&if_key_valid.getThenRegion().front());
            auto score = emitScoreDot(global_row, key_row);
            auto shifted = builder.create<mlir::arith::SubFOp>(loc, score, m_new);
            auto weight = builder.create<mlir::math::ExpOp>(loc, shifted);

            auto d_loop =
                builder.create<mlir::scf::ForOp>(loc, c_zero, c_Dtile, c_one);
            {
              mlir::OpBuilder::InsertionGuard guard6(builder);
              builder.setInsertionPointToStart(d_loop.getBody());
              auto local_col = d_loop.getInductionVar();
              auto global_col =
                  builder.create<mlir::arith::AddIOp>(loc, col_base, local_col);
              auto col_valid = builder.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::ult, global_col, c_D);
              auto if_col_valid =
                  builder.create<mlir::scf::IfOp>(loc, col_valid, false);
              {
                mlir::OpBuilder::InsertionGuard guard7(builder);
                builder.setInsertionPointToStart(&if_col_valid.getThenRegion().front());
                mlir::Value v_row = key_row;
                mlir::Value v_col = global_col;
                if (output_attrs.rhs.transpose_last2) {
                  v_row = global_col;
                  v_col = key_row;
                }
                auto prev = builder.create<mlir::memref::LoadOp>(
                    loc, accum_smem, mlir::ValueRange{local_row, local_col});
                auto v_val = builder.create<mlir::memref::LoadOp>(
                    loc, v_arg, mlir::ValueRange{v_row, v_col});
                auto v_f32 = castToF32(builder, loc, v_val, v_elem, f32);
                auto prod = builder.create<mlir::arith::MulFOp>(loc, weight, v_f32);
                auto updated =
                    builder.create<mlir::arith::AddFOp>(loc, prev, prod);
                builder.create<mlir::memref::StoreOp>(
                    loc, updated.getResult(), accum_smem,
                    mlir::ValueRange{local_row, local_col});
              }
            }

            auto l_new = builder.create<mlir::arith::AddFOp>(loc, l_acc, weight);
            builder.create<mlir::scf::YieldOp>(
                loc, mlir::ValueRange{l_new.getResult()});
          }
          {
            mlir::OpBuilder::InsertionGuard guard5(builder);
            builder.setInsertionPointToStart(&if_key_valid.getElseRegion().front());
            builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{l_acc});
          }
          builder.create<mlir::scf::YieldOp>(
              loc, mlir::ValueRange{if_key_valid.getResult(0)});
        }

        builder.create<mlir::scf::YieldOp>(
            loc,
            mlir::ValueRange{m_new.getResult(), chunk_accum_loop.getResult(0)});
      }

      auto final_l = kchunk_loop.getResult(1);
      auto store_loop =
          builder.create<mlir::scf::ForOp>(loc, c_zero, c_Dtile, c_one);
      {
        mlir::OpBuilder::InsertionGuard guard3(builder);
        builder.setInsertionPointToStart(store_loop.getBody());
        auto local_col = store_loop.getInductionVar();
        auto global_col =
            builder.create<mlir::arith::AddIOp>(loc, col_base, local_col);
        auto col_valid = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, global_col, c_D);
        auto if_col_valid = builder.create<mlir::scf::IfOp>(loc, col_valid, false);
        {
          mlir::OpBuilder::InsertionGuard guard4(builder);
          builder.setInsertionPointToStart(&if_col_valid.getThenRegion().front());
          auto acc = builder.create<mlir::memref::LoadOp>(
              loc, accum_smem, mlir::ValueRange{local_row, local_col});
          auto normalized = builder.create<mlir::arith::DivFOp>(loc, acc, final_l);
          auto store_val =
              castFromF32(builder, loc, normalized, out_elem, f32);
          builder.create<mlir::memref::StoreOp>(
              loc, store_val, out_arg, mlir::ValueRange{global_row, global_col});
        }
      }
    }
  }

  builder.setInsertionPointAfter(row_loop);
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
  module->setAttr("matcore.use_online_softmax",
                  builder.getBoolAttr(plan.use_online_softmax));

  return module;
}

// ============================================================================
// FusionEpiloguePass — Injects a second gpu.launch for elementwise epilogue
// after MMA matmul. Reads matcore.fusion_epilogue_kinds from module attrs.
// ============================================================================
namespace {

struct FusionEpiloguePass
    : public mlir::PassWrapper<FusionEpiloguePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionEpiloguePass)

  llvm::StringRef getArgument() const override {
    return "matcore-fusion-epilogue";
  }
  llvm::StringRef getDescription() const override {
    return "Insert gpu.launch for fusion epilogue (relu/gelu/exp) after MMA matmul";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto epilogue_attr =
        module->getAttrOfType<mlir::ArrayAttr>("matcore.fusion_epilogue_kinds");
    if (!epilogue_attr || epilogue_attr.empty()) {
      return;  // No epilogue needed
    }

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

    // Find the entry function
    mlir::func::FuncOp func;
    module.walk([&](mlir::func::FuncOp f) { func = f; });
    if (!func) {
      signalPassFailure();
      return;
    }

    // Output is the 3rd function argument (C memref)
    auto out_arg = func.getArgument(2);

    // Find the return op — insert epilogue launches before it
    mlir::func::ReturnOp return_op;
    func.walk([&](mlir::func::ReturnOp r) { return_op = r; });
    if (!return_op) {
      signalPassFailure();
      return;
    }

    mlir::OpBuilder builder(return_op);
    auto loc = builder.getUnknownLoc();

    for (auto kind_attr : epilogue_attr) {
      int kind = kind_attr.cast<mlir::IntegerAttr>().getInt();

      auto c_grid = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_size);
      auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_size);

      auto launch = builder.create<mlir::gpu::LaunchOp>(
          loc, c_grid, c_one, c_one, c_block, c_one, c_one);

      // Build the epilogue kernel body
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

        auto if_op = builder.create<mlir::scf::IfOp>(loc, in_bounds, /*withElse=*/false);
        {
          mlir::OpBuilder::InsertionGuard guard2(builder);
          builder.setInsertionPointToStart(&if_op.getThenRegion().front());

          auto row = builder.create<mlir::arith::DivUIOp>(loc, idx, c_N_val);
          auto col = builder.create<mlir::arith::RemUIOp>(loc, idx, c_N_val);
          auto val = builder.create<mlir::memref::LoadOp>(
              loc, out_arg, mlir::ValueRange{row, col});

          mlir::Value result = emitElementwiseOnValue(
              builder, loc, static_cast<ElementwiseKind>(kind),
              val, builder.getF32Type());

          builder.create<mlir::memref::StoreOp>(
              loc, result, out_arg, mlir::ValueRange{row, col});
        }

        builder.setInsertionPointAfter(if_op);
        builder.create<mlir::gpu::TerminatorOp>(loc);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateFusionEpiloguePass() {
  return std::make_unique<FusionEpiloguePass>();
}

}  // namespace matcore
