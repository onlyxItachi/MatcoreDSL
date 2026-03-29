#include "matcore/mlir_engine.h"

#include <cstdint>
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"

#include "matcore/lowering_pipeline.h"

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

  MatmulLoweringSignature signature;
  signature.lhs_dtype = tensors[0].dtype;
  signature.rhs_dtype = tensors[1].dtype;
  signature.out_dtype = tensors[2].dtype;

  if (signature.lhs_dtype != signature.rhs_dtype) {
    fail("lhs/rhs dtype mismatch is not supported");
  }

  const TargetKind normalized_target = normalizeTarget(target_profile.kind);
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
      if (normalized_target != TargetKind::kNvidiaDGPU) {
        fail("float8_e4m3fn matmul is currently limited to nvidia-dgpu");
      }
      if (!target_profile.nvidia_sm_major.has_value() ||
          !target_profile.nvidia_sm_minor.has_value() ||
          *target_profile.nvidia_sm_major < 9) {
        fail("float8_e4m3fn matmul requires native NVIDIA FP8 tensor-core "
             "support (sm_90+ WGMMA); request nvidia-dgpu:sm_90 or newer");
      }
      fail("float8_e4m3fn matmul requires a dedicated native NVIDIA FP8 WGMMA "
           "lowering path, and MatCore does not implement that path yet");
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

std::int64_t roundUpToMultiple(std::int64_t dim, std::int64_t tile) {
  if (dim <= 0 || tile <= 0) {
    return dim;
  }
  return ((dim + tile - 1) / tile) * tile;
}

bool useTensorPadMatmul(const LoweringPlan &plan,
                        const MatmulLoweringSignature &signature,
                        const MatmulShape &shape) {
  const bool needs_padding =
      (shape.m % 16) != 0 || (shape.k % 16) != 0 || (shape.n % 8) != 0;
  return plan.route == LoweringRoute::kNvidiaNvptx &&
         signature.lhs_dtype == TensorDType::kFloat16 &&
         signature.rhs_dtype == TensorDType::kFloat16 &&
         signature.out_dtype == TensorDType::kFloat16 &&
         !signature.quantized_i8 && needs_padding;
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
  if (useTensorPadMatmul(plan, signature, shape)) {
    const std::int64_t padded_m = roundUpToMultiple(shape.m, 16);
    const std::int64_t padded_k = roundUpToMultiple(shape.k, 16);
    const std::int64_t padded_n = roundUpToMultiple(shape.n, 8);

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
  }

  if (plan.route != LoweringRoute::kNvidiaNvptx) {
    builder.create<mlir::linalg::FillOp>(
        loc, mlir::ValueRange{zero}, mlir::ValueRange{entry_block->getArgument(2)});
  }
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
    const std::vector<RuntimeTensorView> &tensors, mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<mlir::arith::ArithDialect,
                      mlir::bufferization::BufferizationDialect,
                      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                      mlir::gpu::GPUDialect, mlir::linalg::LinalgDialect,
                      mlir::memref::MemRefDialect, mlir::nvgpu::NVGPUDialect,
                      mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                      mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                      mlir::vector::VectorDialect,
                      mlir::x86vector::X86VectorDialect, mlir::LLVM::LLVMDialect>();

  validateKernel(kernel, target_profile, tensors);
  const LoweringPlan plan = selectLoweringPlan(target_profile.kind);
  const MatmulLoweringSignature signature =
      inferMatmulSignature(target_profile, tensors);
  const MatmulShape shape = extractMatmulShape(tensors);
  const std::string nvidia_chip = requestedNvidiaChip(target_profile);

  auto module =
      buildMatmulModule(kernel, signature, plan, tensors, shape, context);
  module->getOperation()->setAttr(
      "matcore.requested_target",
      mlir::StringAttr::get(&context, target_profile.canonical));
  if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
    module->getOperation()->setAttr("matcore.nvidia_chip",
                                    mlir::StringAttr::get(&context, nvidia_chip));
  }
  matcore::runLoweringPipeline(*module, plan, signature, nvidia_chip);

  LoweredModule lowered;
  lowered.module = std::move(module);
  lowered.entry_point =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  lowered.target_profile = target_profile;
  lowered.execution_requirements = BuildExecutionRequirements(target_profile);
  lowered.route_description = plan.route_description;
  lowered.executable = plan.executable;
  return lowered;
}

}  // namespace matcore
