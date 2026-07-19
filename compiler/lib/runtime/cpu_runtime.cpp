#include "matcore/runtime_c.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

matcore_status_v0 status(matcore_status_code_v0 code,
                         const char *message) noexcept {
  return {MATCORE_RUNTIME_ABI_VERSION_V0,
          static_cast<uint32_t>(sizeof(matcore_status_v0)), code, 0, message,
          {0, 0}};
}

bool valid_header(uint32_t version, uint32_t size,
                  std::size_t expected) noexcept {
  return version == MATCORE_RUNTIME_ABI_VERSION_V0 && size == expected;
}

bool valid_memory_space(matcore_memory_space_v0 space) noexcept {
  return space == MATCORE_MEMORY_SPACE_HOST_V0 ||
         space == MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0 ||
         space == MATCORE_MEMORY_SPACE_ROCM_DEVICE_V0;
}

bool valid_mutability(matcore_mutability_v0 mutability) noexcept {
  return mutability == MATCORE_MUTABILITY_READ_ONLY_V0 ||
         mutability == MATCORE_MUTABILITY_READ_WRITE_V0;
}

template <std::size_t Size>
bool reserved_is_zero(const std::uint64_t (&reserved)[Size]) noexcept {
  for (const std::uint64_t value : reserved) {
    if (value != 0) {
      return false;
    }
  }
  return true;
}

matcore_status_v0 validate_tensor(const matcore_tensor_desc_v0 &tensor,
                                  const char *null_message) noexcept {
  if (!valid_header(tensor.abi_version, tensor.struct_size,
                    sizeof(matcore_tensor_desc_v0))) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "tensor descriptor ABI version or size mismatch");
  }
  if (!reserved_is_zero(tensor.reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "tensor descriptor reserved fields must be zero");
  }
  if (tensor.data == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0, null_message);
  }
  if (tensor.dtype != MATCORE_DTYPE_F32_V0) {
    return status(MATCORE_STATUS_UNSUPPORTED_DTYPE_V0,
                  "CPU GEMM v0 supports only f32 tensors");
  }
  if (tensor.rank != 2) {
    return status(MATCORE_STATUS_UNSUPPORTED_RANK_V0,
                  "CPU GEMM v0 requires rank-2 tensors");
  }
  if (tensor.dims[0] <= 0 || tensor.dims[1] <= 0) {
    return status(MATCORE_STATUS_INVALID_SHAPE_V0,
                  "CPU GEMM v0 dimensions must be positive");
  }
  if (tensor.strides[1] != 1 || tensor.strides[0] != tensor.dims[1]) {
    return status(MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0,
                  "CPU GEMM v0 requires row-major contiguous tensors");
  }
  if (!valid_memory_space(tensor.memory_space)) {
    return status(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0,
                  "tensor descriptor has an unknown memory space");
  }
  if (!valid_mutability(tensor.mutability)) {
    return status(MATCORE_STATUS_UNSUPPORTED_MUTABILITY_V0,
                  "tensor descriptor has invalid mutability");
  }
  if (reinterpret_cast<std::uintptr_t>(tensor.data) % alignof(float) != 0) {
    return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                  "f32 tensor data must satisfy float alignment");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

bool byte_count(const matcore_tensor_desc_v0 &tensor,
                std::uintptr_t *begin, std::uintptr_t *end) noexcept {
  const auto rows = static_cast<std::uint64_t>(tensor.dims[0]);
  const auto cols = static_cast<std::uint64_t>(tensor.dims[1]);
  constexpr auto max_u64 = std::numeric_limits<std::uint64_t>::max();
  if (rows > max_u64 / cols) {
    return false;
  }
  const std::uint64_t elements = rows * cols;
  if (elements > max_u64 / sizeof(float)) {
    return false;
  }
  const std::uint64_t bytes = elements * sizeof(float);
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(tensor.data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
    return false;
  }
  *begin = address;
  *end = address + static_cast<std::uintptr_t>(bytes);
  return true;
}

bool overlaps(std::uintptr_t lhs_begin, std::uintptr_t lhs_end,
              std::uintptr_t rhs_begin, std::uintptr_t rhs_end) noexcept {
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

}  // namespace

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_v0(const matcore_tensor_desc_v0 *out,
                            const matcore_tensor_desc_v0 *lhs,
                            const matcore_tensor_desc_v0 *rhs,
                            const matcore_policy_v0 *policy) noexcept {
  if (out == nullptr || lhs == nullptr || rhs == nullptr || policy == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "GEMM descriptors and policy must be non-null");
  }
  if (!valid_header(policy->abi_version, policy->struct_size,
                    sizeof(matcore_policy_v0))) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "policy ABI version or size mismatch");
  }
  if (!reserved_is_zero(policy->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "policy reserved fields must be zero");
  }
  if (policy->target != MATCORE_TARGET_CPU_V0) {
    return status(MATCORE_STATUS_UNSUPPORTED_TARGET_V0,
                  "CPU GEMM v0 requires target=cpu");
  }
  if (policy->fallback != MATCORE_FALLBACK_ERROR_V0) {
    return status(MATCORE_STATUS_UNSUPPORTED_FALLBACK_V0,
                  "CPU GEMM v0 requires fallback=error");
  }

  matcore_status_v0 result = validate_tensor(*out, "output data is null");
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_tensor(*lhs, "left input data is null");
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_tensor(*rhs, "right input data is null");
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  if (out->mutability != MATCORE_MUTABILITY_READ_WRITE_V0) {
    return status(MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0,
                  "GEMM output must be mutable");
  }
  if (out->memory_space != lhs->memory_space ||
      out->memory_space != rhs->memory_space) {
    return status(MATCORE_STATUS_MIXED_MEMORY_SPACES_V0,
                  "GEMM tensors must use one memory space");
  }
  if (out->memory_space != MATCORE_MEMORY_SPACE_HOST_V0) {
    return status(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0,
                  "CPU GEMM v0 accepts host tensors only");
  }

  const std::int64_t m = lhs->dims[0];
  const std::int64_t k = lhs->dims[1];
  const std::int64_t n = rhs->dims[1];
  if (rhs->dims[0] != k || out->dims[0] != m || out->dims[1] != n) {
    return status(MATCORE_STATUS_SHAPE_MISMATCH_V0,
                  "GEMM shapes must be lhs[M,K], rhs[K,N], out[M,N]");
  }

  std::uintptr_t out_begin = 0, out_end = 0, lhs_begin = 0, lhs_end = 0;
  std::uintptr_t rhs_begin = 0, rhs_end = 0;
  if (!byte_count(*out, &out_begin, &out_end) ||
      !byte_count(*lhs, &lhs_begin, &lhs_end) ||
      !byte_count(*rhs, &rhs_begin, &rhs_end)) {
    return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                  "tensor byte range overflows the address space");
  }
  if (overlaps(out_begin, out_end, lhs_begin, lhs_end) ||
      overlaps(out_begin, out_end, rhs_begin, rhs_end)) {
    return status(MATCORE_STATUS_ALIAS_VIOLATION_V0,
                  "GEMM output must not overlap either input");
  }

  const auto *a = static_cast<const float *>(lhs->data);
  const auto *b = static_cast<const float *>(rhs->data);
  auto *c = static_cast<float *>(out->data);
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float sum = 0.0F;
      for (std::int64_t p = 0; p < k; ++p) {
        sum += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = sum;
    }
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}
