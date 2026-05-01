#include "matcore/mlir_engine.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"

#include "fp8_wgmma.h"
#include "matcore/fusion_analysis.h"
#include "matcore/fusion_emitter.h"
#include "matcore/lowering_pipeline.h"
#include "matcore/observability.h"
#include "matcore/region_emitter.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore MLIR engine: " + message);
}

struct MatmulShape {
  std::int64_t m = 0;
  std::int64_t k = 0;
  std::int64_t n = 0;
};

std::string opTypeName(const KernelOp &op);

std::string dtypeName(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return "float32";
    case TensorDType::kFloat16:
      return "float16";
    case TensorDType::kBFloat16:
      return "bfloat16";
    case TensorDType::kInt8:
      return "int8";
    case TensorDType::kInt32:
      return "int32";
    case TensorDType::kFloat8E4M3FN:
      return "float8_e4m3fn";
  }
  return "unknown";
}

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
  fail("unsupported tensor dtype");
}

void validateRuntimeTensor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (!tensor.c_contiguous) {
    fail("tensor '" + tensor.symbol + "' must be C-contiguous");
  }
  if (tensor.shape.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for matmul");
  }
  if (tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must provide rank-2 strides");
  }

  switch (tensor.dtype) {
    case TensorDType::kFloat32:
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
    case TensorDType::kInt8:
    case TensorDType::kInt32:
    case TensorDType::kFloat8E4M3FN:
      return;
  }
  fail("tensor '" + tensor.symbol + "' has unsupported dtype");
}

MatmulLoweringSignature inferMatmulSignature(
    const RequestedTargetProfile &target_profile,
    const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 3) {
    fail("runtime must provide lhs, rhs, and out tensors");
  }

  const TargetKind normalized_target = normalizeTarget(target_profile.kind);
  MatmulLoweringSignature signature;
  signature.lhs_dtype = tensors[0].dtype;
  signature.rhs_dtype = tensors[1].dtype;
  signature.out_dtype = tensors[2].dtype;
  signature.target_kind = normalized_target;
  signature.nvidia_sm_major = target_profile.nvidia_sm_major.value_or(0);
  signature.nvidia_sm_minor = target_profile.nvidia_sm_minor.value_or(0);
  signature.matmul_m = static_cast<int>(tensors[0].shape[0]);
  signature.matmul_k = static_cast<int>(tensors[0].shape[1]);
  signature.matmul_n = static_cast<int>(tensors[1].shape[1]);

  if (signature.lhs_dtype != signature.rhs_dtype) {
    fail("lhs/rhs dtype mismatch is not supported");
  }
  switch (signature.lhs_dtype) {
    case TensorDType::kFloat32:
      if (signature.out_dtype != TensorDType::kFloat32) {
        fail("float32 matmul requires float32 output");
      }
      return signature;
    case TensorDType::kFloat16:
      if (signature.out_dtype != TensorDType::kFloat16 &&
          signature.out_dtype != TensorDType::kFloat32) {
        fail("float16 matmul requires float16 or float32 output");
      }
      return signature;
    case TensorDType::kBFloat16:
      if (signature.out_dtype != TensorDType::kBFloat16 &&
          signature.out_dtype != TensorDType::kFloat32) {
        fail("bfloat16 matmul requires bfloat16 or float32 output");
      }
      return signature;
    case TensorDType::kInt8:
      if (signature.out_dtype != TensorDType::kInt32) {
        fail("int8 matmul currently requires an int32 output tensor");
      }
      signature.quantized_i8 = true;
      return signature;
    case TensorDType::kFloat8E4M3FN:
      if (signature.out_dtype != TensorDType::kFloat32) {
        fail("float8_e4m3fn matmul requires float32 output/accumulation for "
             "MLIR 18.1.3 FP8 WGMMA");
      }
      if (normalized_target == TargetKind::kAmdIGPU) {
        fail("float8_e4m3fn is disabled on AMD targets: MatCore FP8 uses E4M3FN "
             "while MLIR 18 AMDGPU lowering currently expects FNUZ FP8 types");
      }
      if (normalized_target != TargetKind::kNvidiaDGPU) {
        fail("float8_e4m3fn matmul is currently limited to nvidia-dgpu");
      }
      if (!target_profile.nvidia_sm_major.has_value() ||
          !target_profile.nvidia_sm_minor.has_value() ||
          *target_profile.nvidia_sm_major < 9) {
        fail("float8_e4m3fn matmul requires native NVIDIA FP8 tensor-core "
             "support (sm_90+ WGMMA); request nvidia-dgpu:sm_90 or newer");
      }
      if (!isEligibleForFp8Wgmma(signature)) {
        fail("float8_e4m3fn matmul is not eligible for NVIDIA FP8 WGMMA");
      }
      return signature;
    case TensorDType::kInt32:
      break;
  }

  fail("unsupported matmul dtype combination: lhs=" +
       dtypeName(signature.lhs_dtype) + ", rhs=" + dtypeName(signature.rhs_dtype) +
       ", out=" + dtypeName(signature.out_dtype));
}

void validateKernel(const KernelIR &kernel,
                    const RequestedTargetProfile &target_profile,
                    const std::vector<RuntimeTensorView> &tensors) {
  for (const KernelOp &op : kernel.ops) {
    if (std::holds_alternative<TransposeOp>(op) ||
        std::holds_alternative<CastOp>(op)) {
      fail("operation '" + opTypeName(op) +
           "' is not yet supported in MLIR lowering (Phase 3 pending)");
    }
  }

  if (kernel.params.size() < 3) {
    fail("kernel must expose at least 3 params (lhs, rhs, out)");
  }
  if (tensors.size() < 3) {
    fail("runtime must provide at least 3 tensors (lhs, rhs, out)");
  }
  for (const RuntimeTensorView &tensor : tensors) {
    validateRuntimeTensor(tensor);
  }

  const RuntimeTensorView &lhs = tensors[0];
  const RuntimeTensorView &rhs = tensors[1];
  const RuntimeTensorView &out = tensors[2];
  const std::int64_t m = lhs.shape[0];
  const std::int64_t k = lhs.shape[1];
  const std::int64_t k_rhs = rhs.shape[0];
  const std::int64_t n = rhs.shape[1];
  if (k != k_rhs) {
    fail("matmul inner dimensions mismatch: lhs K != rhs K");
  }
  if (out.shape[0] != m || out.shape[1] != n) {
    fail("output shape mismatch for matmul result");
  }

  (void)inferMatmulSignature(target_profile, tensors);

  bool saw_load = false;
  bool saw_matmul = false;
  bool saw_store = false;
  for (const KernelOp &op : kernel.ops) {
    if (std::holds_alternative<LoadOp>(op)) {
      saw_load = true;
    } else if (std::holds_alternative<MatMulOp>(op)) {
      saw_matmul = true;
    } else if (std::holds_alternative<StoreOp>(op)) {
      saw_store = true;
    }
  }
  if (!(saw_load && saw_matmul && saw_store)) {
    fail("kernel ops must include load -> matmul -> store pattern");
  }
}

static bool isElementwiseKernel(const KernelIR &kernel) {
  bool has_elementwise = false;
  for (const auto &op : kernel.ops) {
    if (std::holds_alternative<MatMulOp>(op)) return false;
    if (std::holds_alternative<ElementwiseOp>(op)) has_elementwise = true;
  }
  return has_elementwise;
}

static bool isUnaryOp(ElementwiseKind kind) {
  switch (kind) {
    case ElementwiseKind::kExp:
    case ElementwiseKind::kLog:
    case ElementwiseKind::kSqrt:
    case ElementwiseKind::kTanh:
    case ElementwiseKind::kSigmoid:
    case ElementwiseKind::kGELU:
    case ElementwiseKind::kReLU:
    case ElementwiseKind::kNeg:
    case ElementwiseKind::kAbs:
    case ElementwiseKind::kSoftmax:
    case ElementwiseKind::kSin:
    case ElementwiseKind::kCos:
    case ElementwiseKind::kRsqrt:
      return true;
    default:
      return false;
  }
}

void validateElementwiseKernel(const KernelIR &kernel,
                               const std::vector<RuntimeTensorView> &tensors) {
  const ElementwiseOp *elem_op = nullptr;
  for (const auto &op : kernel.ops) {
    if (auto *e = std::get_if<ElementwiseOp>(&op)) {
      elem_op = e;
      break;
    }
  }
  if (!elem_op) fail("elementwise kernel has no elementwise op");

  const bool unary = isUnaryOp(elem_op->kind);
  const std::size_t expected = unary ? 2 : 3;
  if (tensors.size() < expected) {
    fail("elementwise kernel requires " + std::to_string(expected) +
         " tensors, got " + std::to_string(tensors.size()));
  }

  for (std::size_t i = 0; i < expected; ++i) {
    const auto &tensor = tensors[i];
    if (tensor.data == nullptr) fail("tensor '" + tensor.symbol + "' has null data");
    if (!tensor.c_contiguous) fail("tensor '" + tensor.symbol + "' must be C-contiguous");
    if (tensor.shape.size() != 2) fail("tensor '" + tensor.symbol + "' must be rank-2");
  }

  const auto &in0 = tensors[0];
  const auto &out = tensors[expected - 1];
  if (in0.shape[0] != out.shape[0] || in0.shape[1] != out.shape[1]) {
    fail("input/output shape mismatch for elementwise op");
  }
  if (!unary) {
    const auto &in1 = tensors[1];
    if (in0.shape[0] != in1.shape[0] || in0.shape[1] != in1.shape[1]) {
      fail("binary elementwise inputs must have matching shapes");
    }
  }
}

mlir::Value emitElementwiseOp(mlir::OpBuilder &builder, mlir::Location loc,
                              ElementwiseKind kind, mlir::Value lhsVal,
                              mlir::Value rhsVal, mlir::Type f32) {
  using namespace mlir;
  switch (kind) {
    case ElementwiseKind::kExp:
      return builder.create<math::ExpOp>(loc, lhsVal);
    case ElementwiseKind::kLog:
      return builder.create<math::LogOp>(loc, lhsVal);
    case ElementwiseKind::kSqrt:
      return builder.create<math::SqrtOp>(loc, lhsVal);
    case ElementwiseKind::kTanh:
      return builder.create<math::TanhOp>(loc, lhsVal);
    case ElementwiseKind::kAbs:
      return builder.create<math::AbsFOp>(loc, lhsVal);
    case ElementwiseKind::kNeg:
      return builder.create<arith::NegFOp>(loc, lhsVal);
    case ElementwiseKind::kReLU: {
      auto zero =
          builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(f32, 0.0));
      return builder.create<arith::MaximumFOp>(loc, lhsVal, zero);
    }
    case ElementwiseKind::kSigmoid: {
      auto neg = builder.create<arith::NegFOp>(loc, lhsVal);
      auto exp_neg = builder.create<math::ExpOp>(loc, neg);
      auto one =
          builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(f32, 1.0));
      auto denom = builder.create<arith::AddFOp>(loc, one, exp_neg);
      return builder.create<arith::DivFOp>(loc, one, denom);
    }
    case ElementwiseKind::kGELU: {
      auto half =
          builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(f32, 0.5));
      auto coeff = builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(f32, 0.044715));
      auto sqrt2pi = builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(f32, 0.7978845608));
      auto one =
          builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(f32, 1.0));
      auto x2 = builder.create<arith::MulFOp>(loc, lhsVal, lhsVal);
      auto x3 = builder.create<arith::MulFOp>(loc, x2, lhsVal);
      auto cx3 = builder.create<arith::MulFOp>(loc, coeff, x3);
      auto inner = builder.create<arith::AddFOp>(loc, lhsVal, cx3);
      auto scaled = builder.create<arith::MulFOp>(loc, sqrt2pi, inner);
      auto tanh_val = builder.create<math::TanhOp>(loc, scaled);
      auto one_plus = builder.create<arith::AddFOp>(loc, one, tanh_val);
      auto half_x = builder.create<arith::MulFOp>(loc, half, lhsVal);
      return builder.create<arith::MulFOp>(loc, half_x, one_plus);
    }
    case ElementwiseKind::kAdd:
      return builder.create<arith::AddFOp>(loc, lhsVal, rhsVal);
    case ElementwiseKind::kSub:
      return builder.create<arith::SubFOp>(loc, lhsVal, rhsVal);
    case ElementwiseKind::kMul:
      return builder.create<arith::MulFOp>(loc, lhsVal, rhsVal);
    case ElementwiseKind::kDiv:
      return builder.create<arith::DivFOp>(loc, lhsVal, rhsVal);
    case ElementwiseKind::kMin:
      return builder.create<arith::MinimumFOp>(loc, lhsVal, rhsVal);
    case ElementwiseKind::kMax:
      return builder.create<arith::MaximumFOp>(loc, lhsVal, rhsVal);
    case ElementwiseKind::kSin:
      return builder.create<math::SinOp>(loc, lhsVal,
          arith::FastMathFlags::fast);
    case ElementwiseKind::kCos:
      return builder.create<math::CosOp>(loc, lhsVal,
          arith::FastMathFlags::fast);
    case ElementwiseKind::kRsqrt:
      return builder.create<math::RsqrtOp>(loc, lhsVal,
          arith::FastMathFlags::fast);
    case ElementwiseKind::kSoftmax:
      fail("softmax must not be dispatched through emitElementwiseOp");
  }
  fail("unknown elementwise kind");
}

mlir::OwningOpRef<mlir::ModuleOp> buildElementwiseModule(
    const KernelIR &kernel, const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target_profile, mlir::MLIRContext &context) {
  const ElementwiseOp *elem_op = nullptr;
  for (const auto &op : kernel.ops) {
    if (auto *e = std::get_if<ElementwiseOp>(&op)) {
      elem_op = e;
      break;
    }
  }
  if (!elem_op) {
    fail("elementwise kernel has no elementwise op");
  }

  const bool unary = isUnaryOp(elem_op->kind);
  const auto &input0 = tensors[0];
  const auto &output = tensors[unary ? 1 : 2];
  const std::int64_t M = input0.shape[0];
  const std::int64_t N = input0.shape[1];
  const std::int64_t total = M * N;
  constexpr std::int64_t block_size = 256;
  const std::int64_t grid_size = (total + block_size - 1) / block_size;

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&module->getRegion(0).front());
  auto loc = builder.getUnknownLoc();

  auto in_elem = getElementType(input0.dtype, builder);
  auto out_elem = getElementType(output.dtype, builder);
  auto in0_type = mlir::MemRefType::get({M, N}, in_elem);
  auto out_type = mlir::MemRefType::get({M, N}, out_elem);

  llvm::SmallVector<mlir::Type> arg_types;
  arg_types.push_back(in0_type);
  if (!unary) {
    auto in1_elem = getElementType(tensors[1].dtype, builder);
    arg_types.push_back(
        mlir::MemRefType::get({tensors[1].shape[0], tensors[1].shape[1]}, in1_elem));
  }
  arg_types.push_back(out_type);

  const std::string entry_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name, builder.getFunctionType(arg_types, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto c_total = builder.create<mlir::arith::ConstantIndexOp>(loc, total);
  auto c_grid = builder.create<mlir::arith::ConstantIndexOp>(loc, grid_size);
  auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_size);
  auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c_N = builder.create<mlir::arith::ConstantIndexOp>(loc, N);

  auto launch = builder.create<mlir::gpu::LaunchOp>(loc, c_grid, c_one, c_one, c_block,
                                                     c_one, c_one);
  builder.setInsertionPointToStart(&launch.getBody().front());

  auto tid = launch.getThreadIds().x;
  auto bid = launch.getBlockIds().x;
  auto block_offset = builder.create<mlir::arith::MulIOp>(loc, bid, c_block);
  auto global_id = builder.create<mlir::arith::AddIOp>(loc, block_offset, tid);
  auto stride = builder.create<mlir::arith::MulIOp>(loc, c_grid, c_block);
  auto loop = builder.create<mlir::scf::ForOp>(loc, global_id, c_total, stride);
  builder.setInsertionPointToStart(loop.getBody());
  auto idx = loop.getInductionVar();

  auto row = builder.create<mlir::arith::DivUIOp>(loc, idx, c_N);
  auto col = builder.create<mlir::arith::RemUIOp>(loc, idx, c_N);
  auto val = builder.create<mlir::memref::LoadOp>(loc, entry->getArgument(0),
                                                  mlir::ValueRange{row, col});

  auto f32 = builder.getF32Type();
  mlir::Value compute_lhs = val;
  if (in_elem != f32) {
    if (in_elem.isInteger(8) || in_elem.isInteger(32)) {
      compute_lhs = builder.create<mlir::arith::SIToFPOp>(loc, f32, val);
    } else {
      compute_lhs = builder.create<mlir::arith::ExtFOp>(loc, f32, val);
    }
  }

  mlir::Value compute_rhs = compute_lhs;
  if (!unary) {
    auto val2 = builder.create<mlir::memref::LoadOp>(loc, entry->getArgument(1),
                                                     mlir::ValueRange{row, col});
    compute_rhs = val2;
    auto in1_elem = getElementType(tensors[1].dtype, builder);
    if (in1_elem != f32) {
      if (in1_elem.isInteger(8) || in1_elem.isInteger(32)) {
        compute_rhs = builder.create<mlir::arith::SIToFPOp>(loc, f32, val2);
      } else {
        compute_rhs = builder.create<mlir::arith::ExtFOp>(loc, f32, val2);
      }
    }
  }

  mlir::Value result =
      emitElementwiseOp(builder, loc, elem_op->kind, compute_lhs, compute_rhs, f32);
  mlir::Value store_val = result;
  if (out_elem != f32) {
    if (out_elem.isInteger(8) || out_elem.isInteger(32)) {
      store_val = builder.create<mlir::arith::FPToSIOp>(loc, out_elem, result);
    } else {
      store_val = builder.create<mlir::arith::TruncFOp>(loc, out_elem, result);
    }
  }

  const int out_idx = unary ? 1 : 2;
  builder.create<mlir::memref::StoreOp>(loc, store_val, entry->getArgument(out_idx),
                                        mlir::ValueRange{row, col});
  // ForOp without iter_args auto-creates an empty scf.yield — do NOT add another

  builder.setInsertionPointAfter(loop);
  builder.create<mlir::gpu::TerminatorOp>(loc);
  builder.setInsertionPointAfter(launch);
  builder.create<mlir::func::ReturnOp>(loc);

  module->setAttr("matcore.kernel_type",
                  mlir::StringAttr::get(&context, "elementwise"));
  module->setAttr("matcore.requested_target",
                  mlir::StringAttr::get(&context, target_profile.canonical));
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> buildSoftmaxModule(
    const KernelIR &kernel, const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target_profile, mlir::MLIRContext &context) {
  const auto &input0 = tensors[0];
  const auto &output = tensors[1];
  const std::int64_t M = input0.shape[0];
  const std::int64_t N = input0.shape[1];
  if (N < 1) fail("softmax requires N >= 1 (row width must be positive)");
  if (input0.dtype == TensorDType::kInt8 || input0.dtype == TensorDType::kInt32)
    fail("softmax requires floating-point input (got integer dtype)");
  constexpr std::int64_t block_size = 256;

  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&module->getRegion(0).front());
  auto loc = builder.getUnknownLoc();

  auto in_elem = getElementType(input0.dtype, builder);
  auto out_elem = getElementType(output.dtype, builder);
  auto in_type = mlir::MemRefType::get({M, N}, in_elem);
  auto out_type = mlir::MemRefType::get({M, N}, out_elem);

  const std::string entry_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  auto func = builder.create<mlir::func::FuncOp>(
      loc, entry_name, builder.getFunctionType({in_type, out_type}, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto f32 = builder.getF32Type();
  auto c_M = builder.create<mlir::arith::ConstantIndexOp>(loc, M);
  auto c_one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto c_block = builder.create<mlir::arith::ConstantIndexOp>(loc, block_size);

  auto launch =
      builder.create<mlir::gpu::LaunchOp>(loc, c_M, c_one, c_one, c_block, c_one, c_one);
  builder.setInsertionPointToStart(&launch.getBody().front());

  auto row = launch.getBlockIds().x;
  auto tid = launch.getThreadIds().x;
  auto smem_as = mlir::gpu::AddressSpaceAttr::get(
      &context, mlir::gpu::AddressSpace::Workgroup);
  auto smem_type = mlir::MemRefType::get({block_size}, f32,
                                         mlir::MemRefLayoutAttrInterface(), smem_as);
  auto smem = launch.addWorkgroupAttribution(smem_type, loc);

  auto c_N = builder.create<mlir::arith::ConstantIndexOp>(loc, N);
  auto c_block_inner = builder.create<mlir::arith::ConstantIndexOp>(loc, block_size);
  auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto neg_inf = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getFloatAttr(f32, -std::numeric_limits<float>::infinity()));
  auto zero_f32 =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(f32, 0.0));

  auto max_loop = builder.create<mlir::scf::ForOp>(
      loc, tid, c_N, c_block_inner, mlir::ValueRange{neg_inf.getResult()});
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(max_loop.getBody());
    auto j = max_loop.getInductionVar();
    auto acc = max_loop.getRegionIterArgs()[0];
    auto val = builder.create<mlir::memref::LoadOp>(loc, entry->getArgument(0),
                                                    mlir::ValueRange{row, j});
    mlir::Value val_f32 = val;
    if (in_elem != f32) {
      val_f32 = builder.create<mlir::arith::ExtFOp>(loc, f32, val);
    }
    auto new_max = builder.create<mlir::arith::MaximumFOp>(loc, acc, val_f32);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_max.getResult()});
  }
  auto local_max = max_loop.getResult(0);
  builder.create<mlir::memref::StoreOp>(loc, local_max, smem, mlir::ValueRange{tid});
  builder.create<mlir::gpu::BarrierOp>(loc);

  for (std::int64_t s = block_size / 2; s >= 1; s >>= 1) {
    auto c_s = builder.create<mlir::arith::ConstantIndexOp>(loc, s);
    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                                   tid, c_s);
    auto if_op = builder.create<mlir::scf::IfOp>(loc, cmp, false);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&if_op.getThenRegion().front());
      auto v1 = builder.create<mlir::memref::LoadOp>(loc, smem, mlir::ValueRange{tid});
      auto offset = builder.create<mlir::arith::AddIOp>(loc, tid, c_s);
      auto v2 =
          builder.create<mlir::memref::LoadOp>(loc, smem, mlir::ValueRange{offset});
      auto max_v = builder.create<mlir::arith::MaximumFOp>(loc, v1, v2);
      builder.create<mlir::memref::StoreOp>(loc, max_v.getResult(), smem,
                                            mlir::ValueRange{tid});
    }
    builder.create<mlir::gpu::BarrierOp>(loc);
  }

  auto row_max = builder.create<mlir::memref::LoadOp>(loc, smem, mlir::ValueRange{c0});
  builder.create<mlir::gpu::BarrierOp>(loc);

  auto sum_loop = builder.create<mlir::scf::ForOp>(
      loc, tid, c_N, c_block_inner, mlir::ValueRange{zero_f32.getResult()});
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(sum_loop.getBody());
    auto j = sum_loop.getInductionVar();
    auto acc = sum_loop.getRegionIterArgs()[0];
    auto val = builder.create<mlir::memref::LoadOp>(loc, entry->getArgument(0),
                                                    mlir::ValueRange{row, j});
    mlir::Value val_f32 = val;
    if (in_elem != f32) {
      val_f32 = builder.create<mlir::arith::ExtFOp>(loc, f32, val);
    }
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, val_f32, row_max);
    auto exp_val = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto new_sum = builder.create<mlir::arith::AddFOp>(loc, acc, exp_val);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{new_sum.getResult()});
  }
  auto local_sum = sum_loop.getResult(0);
  builder.create<mlir::memref::StoreOp>(loc, local_sum, smem, mlir::ValueRange{tid});
  builder.create<mlir::gpu::BarrierOp>(loc);

  for (std::int64_t s = block_size / 2; s >= 1; s >>= 1) {
    auto c_s = builder.create<mlir::arith::ConstantIndexOp>(loc, s);
    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult,
                                                   tid, c_s);
    auto if_op = builder.create<mlir::scf::IfOp>(loc, cmp, false);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&if_op.getThenRegion().front());
      auto v1 = builder.create<mlir::memref::LoadOp>(loc, smem, mlir::ValueRange{tid});
      auto offset = builder.create<mlir::arith::AddIOp>(loc, tid, c_s);
      auto v2 =
          builder.create<mlir::memref::LoadOp>(loc, smem, mlir::ValueRange{offset});
      auto sum_v = builder.create<mlir::arith::AddFOp>(loc, v1, v2);
      builder.create<mlir::memref::StoreOp>(loc, sum_v.getResult(), smem,
                                            mlir::ValueRange{tid});
    }
    builder.create<mlir::gpu::BarrierOp>(loc);
  }

  auto row_sum = builder.create<mlir::memref::LoadOp>(loc, smem, mlir::ValueRange{c0});
  builder.create<mlir::gpu::BarrierOp>(loc);

  auto norm_loop = builder.create<mlir::scf::ForOp>(loc, tid, c_N, c_block_inner);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(norm_loop.getBody());
    auto j = norm_loop.getInductionVar();
    auto val = builder.create<mlir::memref::LoadOp>(loc, entry->getArgument(0),
                                                    mlir::ValueRange{row, j});
    mlir::Value val_f32 = val;
    if (in_elem != f32) {
      val_f32 = builder.create<mlir::arith::ExtFOp>(loc, f32, val);
    }
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, val_f32, row_max);
    auto exp_val = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto normed = builder.create<mlir::arith::DivFOp>(loc, exp_val, row_sum);
    mlir::Value store_val = normed;
    if (out_elem != f32) {
      store_val = builder.create<mlir::arith::TruncFOp>(loc, out_elem, normed);
    }
    builder.create<mlir::memref::StoreOp>(loc, store_val, entry->getArgument(1),
                                          mlir::ValueRange{row, j});
  }

  builder.create<mlir::gpu::TerminatorOp>(loc);
  builder.setInsertionPointAfter(launch);
  builder.create<mlir::func::ReturnOp>(loc);

  module->setAttr("matcore.kernel_type",
                  mlir::StringAttr::get(&context, "elementwise"));
  module->setAttr("matcore.requested_target",
                  mlir::StringAttr::get(&context, target_profile.canonical));
  return module;
}

MatmulShape extractMatmulShape(const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 2) {
    fail("runtime must provide at least lhs and rhs tensors");
  }

  MatmulShape shape;
  shape.m = tensors[0].shape[0];
  shape.k = tensors[0].shape[1];
  shape.n = tensors[1].shape[1];
  if (shape.m <= 0 || shape.k <= 0 || shape.n <= 0) {
    fail("matmul dimensions must be positive");
  }
  return shape;
}

std::string opTypeName(const KernelOp &op) {
  if (std::holds_alternative<LoadOp>(op)) {
    return "load";
  }
  if (std::holds_alternative<MatMulOp>(op)) {
    return "matmul";
  }
  if (std::holds_alternative<StoreOp>(op)) {
    return "store";
  }
  if (std::holds_alternative<AssignOp>(op)) {
    return "assign";
  }
  if (std::holds_alternative<TransposeOp>(op)) {
    return "transpose";
  }
  if (std::holds_alternative<ElementwiseOp>(op)) {
    return "elementwise";
  }
  if (std::holds_alternative<CastOp>(op)) {
    return "cast";
  }
  return "unknown";
}

std::string requestedNvidiaChip(const RequestedTargetProfile &target_profile) {
  if (normalizeTarget(target_profile.kind) != TargetKind::kNvidiaDGPU) {
    return "sm_80";
  }
  if (target_profile.nvidia_sm_major.has_value() &&
      target_profile.nvidia_sm_minor.has_value()) {
    return "sm_" + std::to_string(*target_profile.nvidia_sm_major) +
           std::to_string(*target_profile.nvidia_sm_minor);
  }
  return "sm_80";
}

std::string requestedAmdChip(const RequestedTargetProfile &target_profile) {
  if (normalizeTarget(target_profile.kind) != TargetKind::kAmdIGPU) {
    return "gfx90a";
  }
  if (!target_profile.amd_chip.empty()) {
    return target_profile.amd_chip;
  }
  const std::string &requested = target_profile.requested;
  const std::size_t split = requested.find(':');
  if (split == std::string::npos || split + 1 >= requested.size()) {
    return "gfx90a";
  }
  return requested.substr(split + 1);
}

std::int64_t roundUpToMultiple(std::int64_t dim, std::int64_t tile) {
  if (dim <= 0 || tile <= 0) {
    return dim;
  }
  return ((dim + tile - 1) / tile) * tile;
}

bool useTensorPadMatmul(const LoweringPlan &plan,
                        const MatmulLoweringSignature &signature) {
  return plan.route == LoweringRoute::kNvidiaNvptx &&
         signature.lhs_dtype == TensorDType::kFloat16 &&
         signature.rhs_dtype == TensorDType::kFloat16 &&
         signature.out_dtype == TensorDType::kFloat16 &&
         !signature.quantized_i8;
}

mlir::Value createZeroPaddedTensor(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value tensor,
                                   mlir::RankedTensorType result_type,
                                   std::int64_t high_pad0,
                                   std::int64_t high_pad1) {
  if (high_pad0 == 0 && high_pad1 == 0) {
    return tensor;
  }

  auto element_type = result_type.getElementType();
  auto zero = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getZeroAttr(element_type));
  llvm::SmallVector<mlir::OpFoldResult, 2> low = {builder.getIndexAttr(0),
                                                  builder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 2> high = {
      builder.getIndexAttr(high_pad0), builder.getIndexAttr(high_pad1)};
  return builder
      .create<mlir::tensor::PadOp>(loc, result_type, tensor, low, high,
                                   zero.getResult())
      .getResult();
}

mlir::OwningOpRef<mlir::ModuleOp> buildMatmulModule(
    const KernelIR &kernel, const MatmulLoweringSignature &signature,
    const LoweringPlan &plan, const std::vector<RuntimeTensorView> &tensors,
    const MatmulShape &shape, mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  const mlir::Type lhs_element_type = getElementType(signature.lhs_dtype, builder);
  const mlir::Type rhs_element_type = getElementType(signature.rhs_dtype, builder);
  const mlir::Type out_element_type = getElementType(signature.out_dtype, builder);
  module->setAttr("matcore.lhs_dtype", mlir::TypeAttr::get(lhs_element_type));
  module->setAttr("matcore.rhs_dtype", mlir::TypeAttr::get(rhs_element_type));
  module->setAttr("matcore.out_dtype", mlir::TypeAttr::get(out_element_type));
  module->setAttr("matcore.target_kind",
                  builder.getI32IntegerAttr(static_cast<int>(signature.target_kind)));
  module->setAttr("matcore.nvidia_sm_major",
                  builder.getI32IntegerAttr(signature.nvidia_sm_major));
  module->setAttr("matcore.nvidia_sm_minor",
                  builder.getI32IntegerAttr(signature.nvidia_sm_minor));
  module->setAttr("matcore.matmul_m", builder.getI32IntegerAttr(signature.matmul_m));
  module->setAttr("matcore.matmul_n", builder.getI32IntegerAttr(signature.matmul_n));
  module->setAttr("matcore.matmul_k", builder.getI32IntegerAttr(signature.matmul_k));
  module->setAttr("matcore.route", builder.getStringAttr(plan.route_name));
  module->setAttr("matcore.route_description",
                  builder.getStringAttr(plan.route_description));

  const std::string entry_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  const auto lhs_type = mlir::MemRefType::get({shape.m, shape.k}, lhs_element_type);
  const auto rhs_type = mlir::MemRefType::get({shape.k, shape.n}, rhs_element_type);
  const auto out_type = mlir::MemRefType::get({shape.m, shape.n}, out_element_type);

  auto func = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), entry_name,
      builder.getFunctionType({lhs_type, rhs_type, out_type}, {}));
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  const mlir::Location loc = builder.getUnknownLoc();
  auto zero = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getZeroAttr(out_element_type));

  // Check if any tensor is device-resident.
  const bool any_device_resident =
      std::any_of(tensors.begin(), tensors.end(),
                  [](const RuntimeTensorView &t) { return t.is_device_resident; });

  if (useTensorPadMatmul(plan, signature)) {
    const std::int64_t padded_m = roundUpToMultiple(shape.m, 16);
    const std::int64_t padded_k = roundUpToMultiple(shape.k, 16);
    const std::int64_t padded_n = roundUpToMultiple(shape.n, 8);
    const bool needs_actual_padding =
        padded_m != shape.m || padded_k != shape.k || padded_n != shape.n;

    // The padded path uses bufferization::ToTensorOp which reads func args
    // on the HOST side. Device-resident pointers would segfault here.
    // When dims are already aligned, skip the padded path and fall through
    // to the direct memref path which works with device pointers.
    if (any_device_resident && needs_actual_padding) {
      fail("device-resident tensors are not supported with tensor.pad lowering "
           "(shapes that require padding). Use dimensions that are multiples of "
           "16 (M,K) and 8 (N), or use host tensors.");
    }

    if (!any_device_resident) {

    auto lhs_tensor = builder.create<mlir::bufferization::ToTensorOp>(
        loc, entry_block->getArgument(0), /*restrict=*/true,
        /*writable=*/false);
    auto rhs_tensor = builder.create<mlir::bufferization::ToTensorOp>(
        loc, entry_block->getArgument(1), /*restrict=*/true,
        /*writable=*/false);

    auto lhs_tensor_type =
        mlir::RankedTensorType::get({shape.m, shape.k}, lhs_element_type);
    auto rhs_tensor_type =
        mlir::RankedTensorType::get({shape.k, shape.n}, rhs_element_type);
    auto padded_lhs_type =
        mlir::RankedTensorType::get({padded_m, padded_k}, lhs_element_type);
    auto padded_rhs_type =
        mlir::RankedTensorType::get({padded_k, padded_n}, rhs_element_type);
    auto padded_out_type =
        mlir::RankedTensorType::get({padded_m, padded_n}, out_element_type);
    auto out_tensor_type =
        mlir::RankedTensorType::get({shape.m, shape.n}, out_element_type);

    mlir::Value lhs_padded = createZeroPaddedTensor(
        builder, loc, lhs_tensor.getResult(), padded_lhs_type, padded_m - shape.m,
        padded_k - shape.k);
    mlir::Value rhs_padded = createZeroPaddedTensor(
        builder, loc, rhs_tensor.getResult(), padded_rhs_type, padded_k - shape.k,
        padded_n - shape.n);

    auto padded_empty = builder.create<mlir::tensor::EmptyOp>(
        loc, llvm::ArrayRef<std::int64_t>{padded_m, padded_n}, out_element_type);
    mlir::Value padded_zero =
        builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero},
                                             mlir::ValueRange{padded_empty})
            .getResult(0);
    mlir::Value padded_result =
        builder
            .create<mlir::linalg::MatmulOp>(
                loc, mlir::ValueRange{lhs_padded, rhs_padded},
                mlir::ValueRange{padded_zero})
            .getResult(0);

    llvm::SmallVector<mlir::OpFoldResult, 2> offsets = {
        builder.getIndexAttr(0), builder.getIndexAttr(0)};
    llvm::SmallVector<mlir::OpFoldResult, 2> sizes = {
        builder.getIndexAttr(shape.m), builder.getIndexAttr(shape.n)};
    llvm::SmallVector<mlir::OpFoldResult, 2> strides = {
        builder.getIndexAttr(1), builder.getIndexAttr(1)};
    mlir::Value sliced_result =
        builder
            .create<mlir::tensor::ExtractSliceOp>(
                loc, out_tensor_type, padded_result, offsets, sizes, strides)
            .getResult();
    builder.create<mlir::bufferization::MaterializeInDestinationOp>(
        loc, mlir::Type(), sliced_result, entry_block->getArgument(2),
        /*restrict=*/true, /*writable=*/true);
    builder.create<mlir::func::ReturnOp>(loc);
    return module;
    } // end if (!any_device_resident)
  }

  // Output zero-initialization is performed at RUNTIME (C++ memset) rather
  // than in the MLIR IR.  Generating linalg.fill(0, output_memref) at the
  // function level creates host-side memref.alloc/store/copy + cf.br loops
  // after ConvertLinalgToLoopsPass.  These ops poison the NVVM pipeline when
  // mixed with GpuDataStaging's gpu.alloc/memcpy/launch_func ops, because
  // GpuToLLVMConversionPass partially converts function types while the
  // memref/cf ops remain unconverted → irreconcilable casts.
  //
  // Instead, compileAndRun() / MatcorePlan::execute() zeroes the output
  // buffer BEFORE invoking the JIT function.  The linalg.matmul semantics
  // become C += A*B (accumulate), and since C is pre-zeroed, the result is
  // C = A*B.  Device-resident outputs are zeroed by the user via .zero_().
  //
  // NOTE: the padded path (above) uses tensor-level IR that goes through
  // bufferization and does NOT have this issue.

  if (signature.quantized_i8) {
    const std::int32_t lhs_zero_point =
        tensors[0].quantization.enabled ? tensors[0].quantization.zero_point
                                        : kernel.global_quantization.zero_point;
    const std::int32_t rhs_zero_point =
        tensors[1].quantization.enabled ? tensors[1].quantization.zero_point
                                        : kernel.global_quantization.zero_point;
    auto lhs_zero = builder.create<mlir::arith::ConstantIntOp>(
        loc, lhs_zero_point, 32);
    auto rhs_zero = builder.create<mlir::arith::ConstantIntOp>(
        loc, rhs_zero_point, 32);
    builder.create<mlir::linalg::QuantizedMatmulOp>(
        loc,
        mlir::ValueRange{entry_block->getArgument(0), entry_block->getArgument(1),
                         lhs_zero, rhs_zero},
        mlir::ValueRange{entry_block->getArgument(2)});
  } else {
    builder.create<mlir::linalg::MatmulOp>(
        loc,
        mlir::ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
        mlir::ValueRange{entry_block->getArgument(2)});
  }
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}

}  // namespace

LoweredModule MlirEngine::BuildAndLower(
    const KernelIR &kernel, const RequestedTargetProfile &target_profile,
    const std::vector<RuntimeTensorView> &tensors, mlir::MLIRContext &context,
    ObservabilityContext *obs, bool graph_mode) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::arith::ArithDialect,
                      mlir::bufferization::BufferizationDialect,
                      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                      mlir::gpu::GPUDialect, mlir::linalg::LinalgDialect,
                      mlir::math::MathDialect,
                      mlir::memref::MemRefDialect, mlir::nvgpu::NVGPUDialect,
                      mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                      mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                      mlir::vector::VectorDialect,
                      mlir::x86vector::X86VectorDialect, mlir::LLVM::LLVMDialect>();

  if (kernel.version == KernelIRVersion::kRegionV1 && kernel.region.has_value()) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        RegionMlirEmitter::Emit(kernel, tensors, target_profile, context);
    module->getOperation()->setAttr(
        "matcore.requested_target",
        mlir::StringAttr::get(&context, target_profile.canonical));
    if (graph_mode) {
      module->getOperation()->setAttr("matcore.graph_mode",
                                      mlir::UnitAttr::get(&context));
    }
    if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
      const std::string nvidia_chip = requestedNvidiaChip(target_profile);
      module->getOperation()->setAttr("matcore.nvidia_chip",
                                      mlir::StringAttr::get(&context, nvidia_chip));
    }

    {
      mlir::func::FuncOp func_op;
      module->walk([&](mlir::func::FuncOp f) { func_op = f; });
      if (func_op) {
        for (std::size_t i = 0; i < tensors.size() && i < func_op.getNumArguments();
             ++i) {
          if (tensors[i].is_device_resident) {
            func_op.setArgAttr(static_cast<unsigned>(i), "matcore.device_resident",
                               mlir::UnitAttr::get(&context));
          }
        }
      }
    }

    const LoweringPlan plan = selectLoweringPlan(target_profile.kind);
    const std::string nvidia_chip = requestedNvidiaChip(target_profile);
    const std::string amd_chip = requestedAmdChip(target_profile);
    matcore::runLoweringPipeline(*module, plan, MatmulLoweringSignature{},
                                 nvidia_chip, amd_chip, obs);

    const int actual_reg_count = [&]() {
      if (auto actual_attr =
              module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                  "matcore.actual_reg_count")) {
        return static_cast<int>(actual_attr.getInt());
      }
      return 0;
    }();
    const int fusion_launch_count = [&]() {
      if (auto launch_attr =
              module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                  "matcore.fusion_launch_count")) {
        return static_cast<int>(launch_attr.getInt());
      }
      return 0;
    }();

    LoweredModule lowered;
    lowered.module = std::move(module);
    const std::string base_name =
        kernel.kernel_name.empty() ? "matcore_region" : kernel.kernel_name;
    lowered.entry_point = "fused_" + base_name;
    lowered.target_profile = target_profile;
    lowered.execution_requirements = BuildExecutionRequirements(target_profile);
    lowered.route_description = "NVIDIA RegionV1 JIT kernel";
    lowered.executable = true;
    lowered.tensor_count = tensors.size();
    lowered.out_tensor_index = tensors.empty() ? 0 : tensors.size() - 1;
    lowered.needs_output_zeroing = false;
    lowered.actual_reg_count = actual_reg_count;
    lowered.reg_budget_exceeded = false;
    lowered.fusion_launch_count = fusion_launch_count;
    lowered.arguments.clear();
    lowered.arguments.reserve(tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      KernelArgumentDesc arg;
      arg.symbol = tensors[i].symbol;
      arg.dtype = tensors[i].dtype;
      arg.rank = static_cast<int>(tensors[i].shape.size());
      arg.is_output = (i == lowered.out_tensor_index);
      arg.is_input = !arg.is_output;
      lowered.arguments.push_back(std::move(arg));
    }
    if (!tensors.empty()) {
      lowered.output_tensor_indices = {lowered.out_tensor_index};
    }
    return lowered;
  }

  if (kernel.version == KernelIRVersion::kGraphV2 && kernel.graph.has_value()) {
    const FusionAnalysisResult analysis =
        FusionAnalyzer::Analyze(*kernel.graph, target_profile);
    if (analysis.accepted_plans.empty()) {
      fail("graph_v2 kernel produced no accepted fusion plan");
    }

    const FusedKernelPlan &fusion_plan = analysis.accepted_plans.front();
    mlir::OwningOpRef<mlir::ModuleOp> module =
        FusionMlirEmitter::Emit(kernel, fusion_plan, tensors, target_profile,
                                context);
    module->getOperation()->setAttr(
        "matcore.requested_target",
        mlir::StringAttr::get(&context, target_profile.canonical));
    if (graph_mode) {
      module->getOperation()->setAttr("matcore.graph_mode",
                                      mlir::UnitAttr::get(&context));
    }
    if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
      const std::string nvidia_chip = requestedNvidiaChip(target_profile);
      module->getOperation()->setAttr("matcore.nvidia_chip",
                                      mlir::StringAttr::get(&context, nvidia_chip));
    }

    {
      mlir::func::FuncOp func_op;
      module->walk([&](mlir::func::FuncOp f) { func_op = f; });
      if (func_op) {
        for (std::size_t i = 0; i < tensors.size() && i < func_op.getNumArguments();
             ++i) {
          if (tensors[i].is_device_resident) {
            func_op.setArgAttr(static_cast<unsigned>(i), "matcore.device_resident",
                               mlir::UnitAttr::get(&context));
          }
        }
      }
    }

    const LoweringPlan plan = selectLoweringPlan(target_profile.kind);
    const std::string nvidia_chip = requestedNvidiaChip(target_profile);
    const std::string amd_chip = requestedAmdChip(target_profile);
    matcore::runLoweringPipeline(*module, plan, MatmulLoweringSignature{},
                                 nvidia_chip, amd_chip, obs);

    const int actual_reg_count = [&]() {
      if (auto actual_attr =
              module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                  "matcore.actual_reg_count")) {
        return static_cast<int>(actual_attr.getInt());
      }
      if (auto requested_attr =
              module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                  "matcore.max_regs")) {
        return static_cast<int>(requested_attr.getInt());
      }
      return 0;
    }();
    const bool reg_budget_exceeded = [&]() {
      if (auto exceeded_attr =
              module->getOperation()->getAttrOfType<mlir::BoolAttr>(
                  "matcore.reg_budget_exceeded")) {
        return exceeded_attr.getValue();
      }
      return actual_reg_count > FusionAnalyzer::kRegHardCap;
    }();
    const int fusion_launch_count = [&]() {
      if (auto launch_attr =
              module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                  "matcore.fusion_launch_count")) {
        return static_cast<int>(launch_attr.getInt());
      }
      return 0;
    }();
    const std::string family_c_strategy = [&]() {
      if (auto strategy_attr =
              module->getOperation()->getAttrOfType<mlir::StringAttr>(
                  "matcore.family_c_strategy")) {
        return strategy_attr.getValue().str();
      }
      return std::string();
    }();
    const int family_c_dtile = [&]() {
      if (auto dtile_attr =
              module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                  "matcore.family_c_dtile")) {
        return static_cast<int>(dtile_attr.getInt());
      }
      return 0;
    }();

    LoweredModule lowered;
    lowered.module = std::move(module);
    const std::string base_name =
        kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
    lowered.entry_point = "fused_" + base_name;
    lowered.target_profile = target_profile;
    lowered.execution_requirements = BuildExecutionRequirements(target_profile);
    lowered.route_description = "NVIDIA fused GPU kernel";
    lowered.executable = true;
    lowered.tensor_count = tensors.size();
    lowered.fusion_launch_count = fusion_launch_count;
    lowered.family_c_strategy = family_c_strategy;
    lowered.family_c_dtile = family_c_dtile;
    // Family A/B use linalg.matmul which accumulates (C += A*B) — output must
    // be zeroed so the accumulation produces correct results.
    const bool has_matmul =
        (fusion_plan.pattern == FusionPatternKind::kMatmulElementwise ||
         fusion_plan.pattern == FusionPatternKind::kElementwiseMatmul ||
         fusion_plan.pattern == FusionPatternKind::kMatmulElementwiseMatmul);
    lowered.needs_output_zeroing = has_matmul;
    lowered.arguments.clear();
    lowered.arguments.reserve(tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const bool is_output = (i + 1 == tensors.size());
      lowered.arguments.push_back(
          {tensors[i].symbol.empty() ? ("arg" + std::to_string(i)) : tensors[i].symbol,
           tensors[i].dtype, 2, !is_output, is_output});
    }
    if (!tensors.empty()) {
      lowered.out_tensor_index = tensors.size() - 1;
      lowered.output_tensor_indices = {lowered.out_tensor_index};
    }
    lowered.actual_reg_count = actual_reg_count;
    lowered.reg_budget_exceeded = reg_budget_exceeded;
    return lowered;
  }

  if (isElementwiseKernel(kernel)) {
    validateElementwiseKernel(kernel, tensors);

    const ElementwiseOp *elem_op = nullptr;
    for (const auto &op : kernel.ops) {
      if (auto *ewOp = std::get_if<ElementwiseOp>(&op)) {
        elem_op = ewOp;
        break;
      }
    }
    if (!elem_op) {
      fail("elementwise kernel has no elementwise op");
    }
    const bool unary = isUnaryOp(elem_op->kind);

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (elem_op->kind == ElementwiseKind::kSoftmax) {
      module = buildSoftmaxModule(kernel, tensors, target_profile, context);
    } else {
      module = buildElementwiseModule(kernel, tensors, target_profile, context);
    }
    module->getOperation()->setAttr(
        "matcore.requested_target",
        mlir::StringAttr::get(&context, target_profile.canonical));
    if (graph_mode) {
      module->getOperation()->setAttr("matcore.graph_mode",
                                      mlir::UnitAttr::get(&context));
    }
    if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
      const std::string nvidia_chip = requestedNvidiaChip(target_profile);
      module->getOperation()->setAttr("matcore.nvidia_chip",
                                      mlir::StringAttr::get(&context, nvidia_chip));
    }

    {
      mlir::func::FuncOp func_op;
      module->walk([&](mlir::func::FuncOp f) { func_op = f; });
      if (func_op) {
        const std::size_t arg_count = unary ? 2 : 3;
        for (std::size_t i = 0;
             i < arg_count && i < tensors.size() && i < func_op.getNumArguments();
             ++i) {
          if (tensors[i].is_device_resident) {
            func_op.setArgAttr(static_cast<unsigned>(i), "matcore.device_resident",
                               mlir::UnitAttr::get(&context));
          }
        }
      }
    }

    const LoweringPlan plan = selectLoweringPlan(target_profile.kind);
    const std::string nvidia_chip = requestedNvidiaChip(target_profile);
    const std::string amd_chip = requestedAmdChip(target_profile);
    matcore::runLoweringPipeline(*module, plan, MatmulLoweringSignature{},
                                 nvidia_chip, amd_chip, obs);

    LoweredModule lowered;
    lowered.module = std::move(module);
    lowered.entry_point =
        kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
    lowered.target_profile = target_profile;
    lowered.execution_requirements = BuildExecutionRequirements(target_profile);
    lowered.route_description = "NVIDIA elementwise GPU kernel";
    lowered.executable = true;
    lowered.tensor_count = unary ? 2 : 3;
    lowered.needs_output_zeroing = false;
    if (elem_op->kind == ElementwiseKind::kSoftmax) {
      const TensorDType in_dtype = tensors[0].dtype;
      const TensorDType out_dtype = tensors[1].dtype;
      lowered.arguments = {
          {"x", in_dtype, 2, true, false},
          {"out", out_dtype, 2, false, true},
      };
      lowered.output_tensor_indices = {1};
    } else if (unary) {
      const TensorDType in_dtype = tensors[0].dtype;
      const TensorDType out_dtype = tensors[1].dtype;
      lowered.arguments = {
          {"x", in_dtype, 2, true, false},
          {"out", out_dtype, 2, false, true},
      };
      lowered.output_tensor_indices = {1};
    } else {
      const TensorDType in_dtype = tensors[0].dtype;
      const TensorDType out_dtype = tensors[2].dtype;
      lowered.arguments = {
          {"a", in_dtype, 2, true, false},
          {"b", in_dtype, 2, true, false},
          {"out", out_dtype, 2, false, true},
      };
      lowered.output_tensor_indices = {2};
    }
    return lowered;
  }

  validateKernel(kernel, target_profile, tensors);
  const LoweringPlan plan = selectLoweringPlan(target_profile.kind);
  const MatmulLoweringSignature signature =
      inferMatmulSignature(target_profile, tensors);
  const MatmulShape shape = extractMatmulShape(tensors);
  const std::string nvidia_chip = requestedNvidiaChip(target_profile);
  const std::string amd_chip = requestedAmdChip(target_profile);

  auto module =
      buildMatmulModule(kernel, signature, plan, tensors, shape, context);
  module->getOperation()->setAttr(
      "matcore.requested_target",
      mlir::StringAttr::get(&context, target_profile.canonical));
  module->getOperation()->setAttr(
      "matcore.requested_target_raw",
      mlir::StringAttr::get(&context, target_profile.requested));
  if (graph_mode) {
    module->getOperation()->setAttr("matcore.graph_mode",
                                    mlir::UnitAttr::get(&context));
  }
  if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
    module->getOperation()->setAttr("matcore.nvidia_chip",
                                    mlir::StringAttr::get(&context, nvidia_chip));
  } else if (normalizeTarget(target_profile.kind) == TargetKind::kAmdIGPU) {
    module->getOperation()->setAttr("matcore.amd_chip",
                                    mlir::StringAttr::get(&context, amd_chip));
  }
  if (obs) {
    obs->snapshot("mlir_construction", *module);
    obs->snapshotText("lowering_plan", plan.route_description);
  }

  // Tag function arguments that correspond to device-resident tensors
  // with "matcore.device_resident" so the staging pass can skip them.
  {
    mlir::func::FuncOp func_op;
    module->walk([&](mlir::func::FuncOp f) { func_op = f; });
    if (func_op) {
      for (std::size_t i = 0; i < tensors.size() && i < func_op.getNumArguments(); ++i) {
        if (tensors[i].is_device_resident) {
          func_op.setArgAttr(static_cast<unsigned>(i), "matcore.device_resident",
                             mlir::UnitAttr::get(&context));
        }
      }
    }
  }

  matcore::runLoweringPipeline(*module, plan, signature, nvidia_chip, amd_chip,
                               obs);

  LoweredModule lowered;
  lowered.module = std::move(module);
  lowered.entry_point =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  lowered.target_profile = target_profile;
  lowered.execution_requirements = BuildExecutionRequirements(target_profile);
  lowered.route_description = plan.route_description;
  lowered.executable = plan.executable;
  const TensorDType lhs_dtype = tensors[0].dtype;
  const TensorDType rhs_dtype = tensors[1].dtype;
  const TensorDType out_dtype = tensors[2].dtype;
  lowered.arguments = {
      {"lhs", lhs_dtype, 2, true, false},
      {"rhs", rhs_dtype, 2, true, false},
      {"out", out_dtype, 2, false, true},
  };
  lowered.output_tensor_indices = {2};
  return lowered;
}

}  // namespace matcore
