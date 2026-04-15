#include "executor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "jit_runner_internal.h"

namespace matcore {
namespace {

struct GenericStridedMemRef2D {
  void *basePtr = nullptr;
  void *data = nullptr;
  int64_t offset = 0;
  int64_t sizes[2] = {0, 0};
  int64_t strides[2] = {0, 0};
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore JIT runner: " + message);
}

template <typename ElementT>
::StridedMemRefType<ElementT, 2>
makeMemRef2DDescriptor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (tensor.shape.size() != 2 || tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for JIT invocation");
  }

  auto *typed = reinterpret_cast<ElementT *>(tensor.data);
  ::StridedMemRefType<ElementT, 2> descriptor;
  descriptor.basePtr = typed;
  descriptor.data = typed;
  descriptor.offset = 0;
  descriptor.sizes[0] = tensor.shape[0];
  descriptor.sizes[1] = tensor.shape[1];
  descriptor.strides[0] = tensor.strides[0];
  descriptor.strides[1] = tensor.strides[1];
  return descriptor;
}

template <typename LhsElementT, typename RhsElementT, typename OutElementT>
llvm::Error invokeWithTypedDescriptors(const CachedExecution &compiled,
                                       const std::string &entry_point,
                                       const RuntimeTensorView &lhs,
                                       const RuntimeTensorView &rhs,
                                       const RuntimeTensorView &out) {
  auto lhs_desc = makeMemRef2DDescriptor<LhsElementT>(lhs);
  auto rhs_desc = makeMemRef2DDescriptor<RhsElementT>(rhs);
  auto out_desc = makeMemRef2DDescriptor<OutElementT>(out);

  const std::string adapter_name = std::string("_mlir_ciface_") + entry_point;
  void *lhs_arg = &lhs_desc;
  void *rhs_arg = &rhs_desc;
  void *out_arg = &out_desc;
  std::vector<void *> packed_args = {&lhs_arg, &rhs_arg, &out_arg};
  if (compiled.backend == ExecutionBackend::kSharedLibrary) {
    if (compiled.ciface_entrypoint == nullptr) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cached shared library entrypoint missing");
    }
    auto fn = reinterpret_cast<void (*)(void *, void *, void *)>(
        compiled.ciface_entrypoint);
    fn(&lhs_desc, &rhs_desc, &out_desc);
    return llvm::Error::success();
  }

  if (compiled.engine == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "execution engine is not initialized");
  }

  llvm::Error packed_error = compiled.engine->invokePacked(adapter_name, packed_args);
  if (!packed_error) {
    return llvm::Error::success();
  }

  std::string packed_message = llvm::toString(std::move(packed_error));
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(), "packed invoke failed: %s",
      packed_message.c_str());
}

template <typename LhsElementT, typename RhsElementT>
llvm::Error invokeWithOutputType(const CachedExecution &compiled,
                                 const std::string &entry_point,
                                 const RuntimeTensorView &lhs,
                                 const RuntimeTensorView &rhs,
                                 const RuntimeTensorView &out) {
  switch (out.dtype) {
    case TensorDType::kFloat32:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, float>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::uint16_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kInt8:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::int8_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kInt32:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::int32_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kFloat8E4M3FN:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::uint8_t>(
          compiled, entry_point, lhs, rhs, out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported output dtype");
}

template <typename LhsElementT>
llvm::Error invokeWithRhsType(const CachedExecution &compiled,
                              const std::string &entry_point,
                              const RuntimeTensorView &lhs,
                              const RuntimeTensorView &rhs,
                              const RuntimeTensorView &out) {
  switch (rhs.dtype) {
    case TensorDType::kFloat32:
      return invokeWithOutputType<LhsElementT, float>(compiled, entry_point, lhs,
                                                      rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeWithOutputType<LhsElementT, std::uint16_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kInt8:
      return invokeWithOutputType<LhsElementT, std::int8_t>(compiled, entry_point,
                                                            lhs, rhs, out);
    case TensorDType::kInt32:
      return invokeWithOutputType<LhsElementT, std::int32_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kFloat8E4M3FN:
      return invokeWithOutputType<LhsElementT, std::uint8_t>(
          compiled, entry_point, lhs, rhs, out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported rhs dtype");
}

template <typename InElementT, typename OutElementT>
llvm::Error invokeUnaryTyped(const CachedExecution &compiled,
                             const std::string &entry_point,
                             const RuntimeTensorView &in,
                             const RuntimeTensorView &out) {
  auto in_desc = makeMemRef2DDescriptor<InElementT>(in);
  auto out_desc = makeMemRef2DDescriptor<OutElementT>(out);

  const std::string adapter_name = std::string("_mlir_ciface_") + entry_point;
  void *in_arg = &in_desc;
  void *out_arg = &out_desc;
  std::vector<void *> packed_args = {&in_arg, &out_arg};
  if (compiled.backend == ExecutionBackend::kSharedLibrary) {
    if (compiled.ciface_entrypoint == nullptr) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cached shared library entrypoint missing");
    }
    auto fn = reinterpret_cast<void (*)(void *, void *)>(compiled.ciface_entrypoint);
    fn(&in_desc, &out_desc);
    return llvm::Error::success();
  }

  if (compiled.engine == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "execution engine is not initialized");
  }

  llvm::Error packed_error = compiled.engine->invokePacked(adapter_name, packed_args);
  if (!packed_error) {
    return llvm::Error::success();
  }
  std::string packed_message = llvm::toString(std::move(packed_error));
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "packed invoke failed: %s",
                                 packed_message.c_str());
}

template <typename InElementT>
llvm::Error invokeUnaryWithOutputType(const CachedExecution &compiled,
                                      const std::string &entry_point,
                                      const RuntimeTensorView &in,
                                      const RuntimeTensorView &out) {
  switch (out.dtype) {
    case TensorDType::kFloat32:
      return invokeUnaryTyped<InElementT, float>(compiled, entry_point, in, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeUnaryTyped<InElementT, std::uint16_t>(compiled, entry_point, in,
                                                         out);
    case TensorDType::kInt8:
      return invokeUnaryTyped<InElementT, std::int8_t>(compiled, entry_point, in,
                                                       out);
    case TensorDType::kInt32:
      return invokeUnaryTyped<InElementT, std::int32_t>(compiled, entry_point, in,
                                                        out);
    case TensorDType::kFloat8E4M3FN:
      return invokeUnaryTyped<InElementT, std::uint8_t>(compiled, entry_point, in,
                                                        out);
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported output dtype");
}

llvm::Error invokeUnary(const CachedExecution &compiled,
                        const RuntimeTensorView &in,
                        const RuntimeTensorView &out) {
  switch (in.dtype) {
    case TensorDType::kFloat32:
      return invokeUnaryWithOutputType<float>(compiled, compiled.lowered.entry_point,
                                              in, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeUnaryWithOutputType<std::uint16_t>(
          compiled, compiled.lowered.entry_point, in, out);
    case TensorDType::kInt8:
      return invokeUnaryWithOutputType<std::int8_t>(compiled,
                                                    compiled.lowered.entry_point, in,
                                                    out);
    case TensorDType::kInt32:
      return invokeUnaryWithOutputType<std::int32_t>(
          compiled, compiled.lowered.entry_point, in, out);
    case TensorDType::kFloat8E4M3FN:
      return invokeUnaryWithOutputType<std::uint8_t>(
          compiled, compiled.lowered.entry_point, in, out);
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported input dtype");
}

GenericStridedMemRef2D
makeGenericMemRef2DDescriptor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (tensor.shape.size() != 2 || tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for JIT invocation");
  }
  GenericStridedMemRef2D descriptor;
  descriptor.basePtr = tensor.data;
  descriptor.data = tensor.data;
  descriptor.offset = 0;
  descriptor.sizes[0] = tensor.shape[0];
  descriptor.sizes[1] = tensor.shape[1];
  descriptor.strides[0] = tensor.strides[0];
  descriptor.strides[1] = tensor.strides[1];
  return descriptor;
}

llvm::Error invokeNTensorPacked(const CachedExecution &compiled,
                                const std::vector<RuntimeTensorView> &tensors,
                                std::size_t tensor_count) {
  if (tensor_count == 0 || tensors.size() < tensor_count) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "N-tensor invocation requires at least tensor_count runtime tensors");
  }

  std::vector<GenericStridedMemRef2D> descriptors(tensor_count);
  std::vector<void *> arg_ptrs(tensor_count);
  std::vector<void *> packed_args(tensor_count);
  for (std::size_t i = 0; i < tensor_count; ++i) {
    descriptors[i] = makeGenericMemRef2DDescriptor(tensors[i]);
    arg_ptrs[i] = &descriptors[i];
    packed_args[i] = &arg_ptrs[i];
  }

  if (compiled.backend == ExecutionBackend::kSharedLibrary) {
    if (compiled.ciface_entrypoint == nullptr) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cached shared library entrypoint missing");
    }
    switch (tensor_count) {
      case 4: {
        auto fn = reinterpret_cast<void (*)(void *, void *, void *, void *)>(
            compiled.ciface_entrypoint);
        fn(&descriptors[0], &descriptors[1], &descriptors[2], &descriptors[3]);
        return llvm::Error::success();
      }
      case 5: {
        auto fn = reinterpret_cast<void (*)(void *, void *, void *, void *, void *)>(
            compiled.ciface_entrypoint);
        fn(&descriptors[0], &descriptors[1], &descriptors[2], &descriptors[3],
           &descriptors[4]);
        return llvm::Error::success();
      }
      case 6: {
        auto fn = reinterpret_cast<void (*)(
            void *, void *, void *, void *, void *, void *)>(
            compiled.ciface_entrypoint);
        fn(&descriptors[0], &descriptors[1], &descriptors[2], &descriptors[3],
           &descriptors[4], &descriptors[5]);
        return llvm::Error::success();
      }
      default:
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "shared-library N-tensor invocation currently supports up to 6 tensors");
    }
  }

  if (compiled.engine == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "execution engine is not initialized");
  }

  const std::string adapter_name =
      std::string("_mlir_ciface_") + compiled.lowered.entry_point;
  llvm::Error packed_error = compiled.engine->invokePacked(adapter_name, packed_args);
  if (!packed_error) {
    return llvm::Error::success();
  }
  std::string packed_message = llvm::toString(std::move(packed_error));
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "packed invoke failed: %s",
                                 packed_message.c_str());
}

}  // namespace

llvm::Error invokeCompiledKernel(const CachedExecution &compiled,
                                 const std::vector<RuntimeTensorView> &tensors) {
  if (compiled.lowered.tensor_count == 2) {
    if (tensors.size() < 2) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unary invocation requires 2 tensors");
    }
    return invokeUnary(compiled, tensors[0], tensors[1]);
  }

  if (compiled.lowered.tensor_count >= 4) {
    return invokeNTensorPacked(compiled, tensors, compiled.lowered.tensor_count);
  }

  if (compiled.lowered.tensor_count != 3) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported lowered tensor_count for invocation");
  }

  if (tensors.size() < 3) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "binary/matmul invocation requires 3 tensors");
  }
  if (compiled.lowered.lhs_tensor_index >= tensors.size() ||
      compiled.lowered.rhs_tensor_index >= tensors.size() ||
      compiled.lowered.out_tensor_index >= tensors.size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "lowered tensor indices are out of range for runtime tensors");
  }

  const RuntimeTensorView &lhs = tensors[compiled.lowered.lhs_tensor_index];
  const RuntimeTensorView &rhs = tensors[compiled.lowered.rhs_tensor_index];
  const RuntimeTensorView &out = tensors[compiled.lowered.out_tensor_index];

  switch (lhs.dtype) {
    case TensorDType::kFloat32:
      return invokeWithRhsType<float>(compiled, compiled.lowered.entry_point, lhs,
                                      rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeWithRhsType<std::uint16_t>(compiled,
                                              compiled.lowered.entry_point, lhs,
                                              rhs, out);
    case TensorDType::kInt8:
      return invokeWithRhsType<std::int8_t>(compiled,
                                            compiled.lowered.entry_point, lhs, rhs,
                                            out);
    case TensorDType::kInt32:
      return invokeWithRhsType<std::int32_t>(compiled,
                                             compiled.lowered.entry_point, lhs, rhs,
                                             out);
    case TensorDType::kFloat8E4M3FN:
      return invokeWithRhsType<std::uint8_t>(compiled,
                                             compiled.lowered.entry_point, lhs, rhs,
                                             out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported runtime dtype");
}

}  // namespace matcore
