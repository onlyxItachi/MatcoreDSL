#include "fusion_emitter_internal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace matcore {
using namespace fusion_emit;

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::emitFamilyA(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
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
  if (!tensors.empty() && tensors.size() != graph.input_values.size() + 1) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family A runtime tensor count does not match graph inputs");
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

  const std::uint32_t out_value_id = plan.output_value_ids.empty()
                                         ? (epilogue_node_ids.empty()
                                                ? mm_node.outputs.front()
                                                : graph.nodes.at(epilogue_node_ids.back())
                                                      .outputs.front())
                                         : plan.output_value_ids.front();
  const auto &out_desc = graph.values.at(out_value_id);
  auto out_elem = getElementType(out_desc.dtype, builder);
  if (out_desc.shape.size() != 2 || out_desc.shape.at(0) != M ||
      out_desc.shape.at(1) != N) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family A output shape must match matmul result");
  }

  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.reserve(graph.input_values.size() + 1);
  for (std::uint32_t input_value_id : graph.input_values) {
    const auto &input_desc = graph.values.at(input_value_id);
    if (input_desc.shape.size() != 2) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family A currently requires rank-2 graph inputs");
    }
    arg_types.push_back(mlir::MemRefType::get(
        {input_desc.shape.at(0), input_desc.shape.at(1)},
        getElementType(input_desc.dtype, builder)));
  }
  const auto out_type = mlir::MemRefType::get({M, N}, out_elem);
  arg_types.push_back(out_type);

  auto find_input_arg_index = [&](std::uint32_t value_id) -> int {
    auto it = std::find(graph.input_values.begin(), graph.input_values.end(),
                        value_id);
    if (it == graph.input_values.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(graph.input_values.begin(), it));
  };

  const std::string base_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const std::string entry_name = "fused_" + base_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name, builder.getFunctionType(arg_types, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setPublic();

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  const int lhs_arg_index = find_input_arg_index(mm_attrs.lhs.value_id);
  const int rhs_arg_index = find_input_arg_index(mm_attrs.rhs.value_id);
  if (lhs_arg_index < 0 || rhs_arg_index < 0) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family A matmul operands must be graph inputs");
  }
  auto lhs_arg = entry->getArgument(static_cast<unsigned>(lhs_arg_index));
  auto rhs_arg = entry->getArgument(static_cast<unsigned>(rhs_arg_index));
  auto out_arg =
      entry->getArgument(static_cast<unsigned>(graph.input_values.size()));

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
  llvm::SmallVector<mlir::Attribute> epilogue_ops;
  std::uint32_t current_value_id = mm_node.outputs.front();
  for (auto epi_node_id : epilogue_node_ids) {
    const auto &epi_node = graph.nodes.at(epi_node_id);
    if (epi_node.kind != OpKind::kElementwise) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family A currently supports only elementwise epilogues");
    }
    const auto &ew_attrs = std::get<ElementwiseAttrs>(epi_node.attrs);
    if (epi_node.outputs.size() != 1) {
      throw std::runtime_error(
          "FusionMlirEmitter: Family A epilogue nodes must produce exactly one output");
    }
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        "kind", builder.getI32IntegerAttr(static_cast<int>(ew_attrs.kind))));

    if (ew_attrs.inputs.size() == 1) {
      if (ew_attrs.inputs.front() != current_value_id) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A unary epilogue must consume the fused value");
      }
    } else if (ew_attrs.inputs.size() == 2) {
      int current_input_pos = -1;
      std::uint32_t boundary_value_id = 0;
      for (std::size_t i = 0; i < ew_attrs.inputs.size(); ++i) {
        if (ew_attrs.inputs[i] == current_value_id) {
          current_input_pos = static_cast<int>(i);
        } else {
          boundary_value_id = ew_attrs.inputs[i];
        }
      }
      if (current_input_pos < 0) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A binary epilogue must consume the fused value");
      }
      const int boundary_arg_index = find_input_arg_index(boundary_value_id);
      if (boundary_arg_index < 0) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A binary epilogue boundary operand must be a graph input");
      }
      const auto &boundary_desc = graph.values.at(boundary_value_id);
      if (boundary_desc.shape.size() != 2 || boundary_desc.shape.at(0) != M ||
          boundary_desc.shape.at(1) != N) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A binary epilogues currently require exact-shape rank-2 boundary tensors");
      }
      if (boundary_desc.dtype != out_desc.dtype) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A binary epilogues currently require boundary tensors to match the output dtype");
      }
      attrs.push_back(builder.getNamedAttr(
          "boundary_arg_index",
          builder.getI32IntegerAttr(boundary_arg_index)));
      attrs.push_back(builder.getNamedAttr(
          "value_input_pos", builder.getI32IntegerAttr(current_input_pos)));
    } else {
      throw std::runtime_error(
          "FusionMlirEmitter: Family A epilogues must be unary or binary");
    }
    epilogue_ops.push_back(builder.getDictionaryAttr(attrs));
    current_value_id = epi_node.outputs.front();
  }
  if (current_value_id != out_value_id) {
    throw std::runtime_error(
        "FusionMlirEmitter: Family A epilogue chain does not match the selected output");
  }
  if (!epilogue_ops.empty()) {
    module->setAttr("matcore.fusion_epilogue_ops",
                    builder.getArrayAttr(epilogue_ops));
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

}  // namespace matcore
