#include "fusion_emitter_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace matcore::fusion_emit {

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

// Helper: create a float constant matching the element type of val.
static mlir::Value makeFloatConst(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Type elemType, double v) {
  return builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(elemType, v));
}

mlir::Value emitElementwiseOnValue(mlir::OpBuilder &builder, mlir::Location loc,
                                   ElementwiseKind kind, mlir::Value val,
                                   mlir::Type f32) {
  (void)f32;
  auto elemType = val.getType();
  switch (kind) {
    case ElementwiseKind::kAdd:
    case ElementwiseKind::kSub:
    case ElementwiseKind::kMul:
    case ElementwiseKind::kDiv:
    case ElementwiseKind::kMin:
    case ElementwiseKind::kMax:
      throw std::runtime_error(
          "FusionMlirEmitter::emitElementwiseOnValue: binary elementwise ops "
          "require two operands and must be handled by the caller");
    case ElementwiseKind::kReLU: {
      auto zero = makeFloatConst(builder, loc, elemType, 0.0);
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
    case ElementwiseKind::kSin:
      return builder.create<mlir::math::SinOp>(loc, val,
          mlir::arith::FastMathFlags::fast).getResult();
    case ElementwiseKind::kCos:
      return builder.create<mlir::math::CosOp>(loc, val,
          mlir::arith::FastMathFlags::fast).getResult();
    case ElementwiseKind::kRsqrt:
      return builder.create<mlir::math::RsqrtOp>(loc, val,
          mlir::arith::FastMathFlags::fast).getResult();
    case ElementwiseKind::kSigmoid: {
      auto one = makeFloatConst(builder, loc, elemType, 1.0);
      auto neg = builder.create<mlir::arith::NegFOp>(loc, val);
      auto exp_neg = builder.create<mlir::math::ExpOp>(loc, neg.getResult());
      auto denom = builder.create<mlir::arith::AddFOp>(
          loc, one, exp_neg.getResult());
      return builder
          .create<mlir::arith::DivFOp>(loc, one, denom.getResult())
          .getResult();
    }
    case ElementwiseKind::kGELU: {
      auto half    = makeFloatConst(builder, loc, elemType, 0.5);
      auto one     = makeFloatConst(builder, loc, elemType, 1.0);
      auto coeff   = makeFloatConst(builder, loc, elemType, 0.044715);
      auto sqrt2pi = makeFloatConst(builder, loc, elemType, 0.7978845608);
      auto x2 = builder.create<mlir::arith::MulFOp>(loc, val, val);
      auto x3 = builder.create<mlir::arith::MulFOp>(loc, x2.getResult(), val);
      auto cx3 = builder.create<mlir::arith::MulFOp>(loc, coeff,
                                                     x3.getResult());
      auto inner = builder.create<mlir::arith::AddFOp>(loc, val, cx3.getResult());
      auto scaled = builder.create<mlir::arith::MulFOp>(loc, sqrt2pi,
                                                        inner.getResult());
      auto tanh_val =
          builder.create<mlir::math::TanhOp>(loc, scaled.getResult());
      auto one_plus = builder.create<mlir::arith::AddFOp>(
          loc, one, tanh_val.getResult());
      auto half_x =
          builder.create<mlir::arith::MulFOp>(loc, half, val);
      return builder
          .create<mlir::arith::MulFOp>(loc, half_x.getResult(),
                                       one_plus.getResult())
          .getResult();
    }
    case ElementwiseKind::kSoftmax:
      throw std::runtime_error(
          "FusionMlirEmitter::emitElementwiseOnValue: softmax must not be "
          "dispatched through emitElementwiseOnValue");
  }
  throw std::runtime_error(
      "FusionMlirEmitter::emitElementwiseOnValue: unknown ElementwiseKind");
}

mlir::Value emitBinaryElementwiseOnValues(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          ElementwiseKind kind,
                                          mlir::Value lhs, mlir::Value rhs) {
  const mlir::Type lhs_type = lhs.getType();
  const mlir::Type rhs_type = rhs.getType();
  if (lhs_type != rhs_type) {
    throw std::runtime_error(
        "FusionMlirEmitter::emitBinaryElementwiseOnValues: operand type mismatch");
  }

  if (lhs_type.isa<mlir::FloatType>()) {
    switch (kind) {
      case ElementwiseKind::kAdd:
        return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kSub:
        return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kMul:
        return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kDiv:
        return builder.create<mlir::arith::DivFOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kMin:
        return builder.create<mlir::arith::MinimumFOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kMax:
        return builder.create<mlir::arith::MaximumFOp>(loc, lhs, rhs).getResult();
      default:
        break;
    }
  }

  if (lhs_type.isa<mlir::IntegerType>()) {
    switch (kind) {
      case ElementwiseKind::kAdd:
        return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kSub:
        return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kMul:
        return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kDiv:
        return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kMin:
        return builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs).getResult();
      case ElementwiseKind::kMax:
        return builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs).getResult();
      default:
        break;
    }
  }

  throw std::runtime_error(
      "FusionMlirEmitter::emitBinaryElementwiseOnValues: unsupported binary kind");
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

bool isUnaryElementwiseKind(ElementwiseKind kind) {
  switch (kind) {
    case ElementwiseKind::kAdd:
    case ElementwiseKind::kSub:
    case ElementwiseKind::kMul:
    case ElementwiseKind::kDiv:
    case ElementwiseKind::kMin:
    case ElementwiseKind::kMax:
      return false;
    case ElementwiseKind::kSoftmax:
      return false;
    default:
      return true;
  }
}

bool isGraphInputValue(const KernelGraphIR &graph, std::uint32_t value_id) {
  return std::find(graph.input_values.begin(), graph.input_values.end(),
                   value_id) != graph.input_values.end();
}

bool isFloatingDType(TensorDType dtype) {
  return dtype == TensorDType::kFloat32 || dtype == TensorDType::kFloat16 ||
         dtype == TensorDType::kBFloat16;
}

const KernelNode &requireNode(const KernelGraphIR &graph, std::uint32_t id) {
  return graph.nodes.at(id);
}

std::uint32_t requireSingleOutput(const KernelNode &node,
                                  const char *context) {
  if (node.outputs.size() != 1) {
    throw std::runtime_error(std::string(context) +
                             ": nodes must produce exactly one output");
  }
  return node.outputs.front();
}

void validateFusionPlanAgainstEmitter(const KernelIR &kernel,
                                      const FusedKernelPlan &plan,
                                      const RequestedTargetProfile &target) {
  if (!kernel.graph.has_value()) {
    throw std::runtime_error("FusionMlirEmitter: missing graph for fused kernel");
  }
  if (normalizeTarget(target.kind) != TargetKind::kNvidiaDGPU) {
    throw std::runtime_error(
        "FusionMlirEmitter: fused graph lowering currently supports only nvidia-dgpu");
  }
  if (plan.node_ids.empty()) {
    throw std::runtime_error("FusionMlirEmitter: empty fusion plan");
  }

  const auto &graph = *kernel.graph;
  int matmul_count = 0;
  int softmax_count = 0;
  for (std::uint32_t node_id : plan.node_ids) {
    const auto &node = requireNode(graph, node_id);
    if (node.kind == OpKind::kMatMul) {
      ++matmul_count;
    } else if (node.kind == OpKind::kSoftmax) {
      ++softmax_count;
    } else if (node.kind != OpKind::kElementwise) {
      throw std::runtime_error(
          "FusionMlirEmitter: fusion emitter currently supports only matmul, "
          "elementwise, and Family C softmax nodes");
    }
  }

  switch (plan.pattern) {
    case FusionPatternKind::kMatmulElementwise: {
      if (matmul_count != 1 || softmax_count != 0 ||
          requireNode(graph, plan.node_ids.front()).kind != OpKind::kMatMul) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A requires matmul followed by elementwise epilogues");
      }
      const auto &mm_node = requireNode(graph, plan.node_ids.front());
      const auto &mm_attrs = std::get<MatMulAttrs>(mm_node.attrs);
      if (!isGraphInputValue(graph, mm_attrs.lhs.value_id) ||
          !isGraphInputValue(graph, mm_attrs.rhs.value_id)) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family A matmul operands must be graph inputs");
      }
      break;
    }
    case FusionPatternKind::kMatmulElementwiseMatmul: {
      if (matmul_count != 2 || softmax_count != 0) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B requires matmul, elementwise glue, matmul");
      }
      std::uint32_t first_matmul_id = std::numeric_limits<std::uint32_t>::max();
      std::uint32_t second_matmul_id = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t node_id : plan.node_ids) {
        const auto &node = requireNode(graph, node_id);
        if (node.kind != OpKind::kMatMul) {
          continue;
        }
        if (first_matmul_id == std::numeric_limits<std::uint32_t>::max()) {
          first_matmul_id = node_id;
        } else {
          second_matmul_id = node_id;
        }
      }
      const auto &mm0 = requireNode(graph, first_matmul_id);
      const auto &mm1 = requireNode(graph, second_matmul_id);
      const auto &mm0_attrs = std::get<MatMulAttrs>(mm0.attrs);
      const auto &mm1_attrs = std::get<MatMulAttrs>(mm1.attrs);
      if (!isGraphInputValue(graph, mm0_attrs.lhs.value_id) ||
          !isGraphInputValue(graph, mm0_attrs.rhs.value_id) ||
          !isGraphInputValue(graph, mm1_attrs.rhs.value_id)) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B boundary matmul operands must be graph inputs");
      }
      if (mm1_attrs.lhs.transpose_last2) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B does not support transposed intermediate lhs");
      }
      std::uint32_t current = requireSingleOutput(mm0, "FusionMlirEmitter: Family B");
      for (std::uint32_t node_id : plan.node_ids) {
        if (node_id == first_matmul_id || node_id == second_matmul_id) {
          continue;
        }
        const auto &glue = requireNode(graph, node_id);
        const auto &ew_attrs = std::get<ElementwiseAttrs>(glue.attrs);
        if (ew_attrs.inputs.empty() || ew_attrs.inputs.size() > 2) {
          throw std::runtime_error(
              "FusionMlirEmitter: Family B glue nodes must be unary or binary");
        }
        if (std::find(ew_attrs.inputs.begin(), ew_attrs.inputs.end(), current) ==
            ew_attrs.inputs.end()) {
          throw std::runtime_error(
              "FusionMlirEmitter: Family B glue chain must consume the fused value");
        }
        current = requireSingleOutput(glue, "FusionMlirEmitter: Family B");
      }
      if (current != mm1_attrs.lhs.value_id) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family B glue chain must feed the second matmul lhs");
      }
      break;
    }
    case FusionPatternKind::kMatmulSoftmaxMatmul: {
      if (matmul_count != 2 || softmax_count != 1) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family C requires matmul -> softmax -> matmul");
      }
      std::uint32_t score_id = std::numeric_limits<std::uint32_t>::max();
      std::uint32_t softmax_id = std::numeric_limits<std::uint32_t>::max();
      std::uint32_t output_id = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t node_id : plan.node_ids) {
        const auto &node = requireNode(graph, node_id);
        if (node.kind == OpKind::kMatMul) {
          if (score_id == std::numeric_limits<std::uint32_t>::max()) {
            score_id = node_id;
          } else {
            output_id = node_id;
          }
        } else if (node.kind == OpKind::kSoftmax) {
          softmax_id = node_id;
        }
      }
      const auto &score_node = requireNode(graph, score_id);
      const auto &softmax_node = requireNode(graph, softmax_id);
      const auto &output_node = requireNode(graph, output_id);
      const auto &score_attrs = std::get<MatMulAttrs>(score_node.attrs);
      const auto &softmax_attrs = std::get<SoftmaxAttrs>(softmax_node.attrs);
      const auto &output_attrs = std::get<MatMulAttrs>(output_node.attrs);
      if (!isGraphInputValue(graph, score_attrs.lhs.value_id) ||
          !isGraphInputValue(graph, score_attrs.rhs.value_id) ||
          !isGraphInputValue(graph, output_attrs.rhs.value_id)) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family C Q/K/V operands must be graph inputs");
      }
      if (softmax_attrs.input != requireSingleOutput(score_node,
                                                     "FusionMlirEmitter: Family C") ||
          output_attrs.lhs.value_id != requireSingleOutput(
                                          softmax_node,
                                          "FusionMlirEmitter: Family C")) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family C must be score matmul -> softmax -> output matmul");
      }
      if (output_attrs.lhs.transpose_last2) {
        throw std::runtime_error(
            "FusionMlirEmitter: Family C does not support transposed softmax lhs");
      }
      for (std::uint32_t value_id : {score_attrs.lhs.value_id,
                                     score_attrs.rhs.value_id,
                                     output_attrs.rhs.value_id}) {
        if (!isFloatingDType(graph.values.at(value_id).dtype)) {
          throw std::runtime_error(
              "FusionMlirEmitter: Family C attention operands must be floating-point");
        }
      }
      break;
    }
    case FusionPatternKind::kElementwiseMatmul:
      throw std::runtime_error(
          "FusionMlirEmitter: elementwise-before-matmul fusion is not implemented");
    case FusionPatternKind::kGenericTileChain:
      throw std::runtime_error(
          "FusionMlirEmitter: generic tile-chain fusion is not implemented");
    default:
      throw std::runtime_error("FusionMlirEmitter: unsupported pattern");
  }
}

}  // namespace matcore::fusion_emit
