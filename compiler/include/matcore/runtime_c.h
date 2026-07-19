#ifndef MATCORE_RUNTIME_C_H
#define MATCORE_RUNTIME_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(MATCORE_RUNTIME_BUILD)
#define MATCORE_RUNTIME_API __declspec(dllexport)
#else
#define MATCORE_RUNTIME_API __declspec(dllimport)
#endif
#else
#define MATCORE_RUNTIME_API __attribute__((visibility("default")))
#endif

#if defined(__cplusplus)
#define MATCORE_RUNTIME_NOEXCEPT noexcept
extern "C" {
#else
#define MATCORE_RUNTIME_NOEXCEPT
#endif

#define MATCORE_RUNTIME_ABI_VERSION_V0 UINT32_C(0)
#define MATCORE_RUNTIME_MAX_RANK_V0 UINT32_C(8)

typedef uint32_t matcore_dtype_v0;
enum {
  MATCORE_DTYPE_INVALID_V0 = 0,
  MATCORE_DTYPE_F32_V0 = 1,
  MATCORE_DTYPE_F64_V0 = 2
};

typedef uint32_t matcore_memory_space_v0;
enum {
  MATCORE_MEMORY_SPACE_INVALID_V0 = 0,
  MATCORE_MEMORY_SPACE_HOST_V0 = 1,
  MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0 = 2,
  MATCORE_MEMORY_SPACE_ROCM_DEVICE_V0 = 3
};

typedef uint32_t matcore_mutability_v0;
enum {
  MATCORE_MUTABILITY_INVALID_V0 = 0,
  MATCORE_MUTABILITY_READ_ONLY_V0 = 1,
  MATCORE_MUTABILITY_READ_WRITE_V0 = 2
};

typedef uint32_t matcore_target_v0;
enum {
  MATCORE_TARGET_INVALID_V0 = 0,
  MATCORE_TARGET_CPU_V0 = 1,
  MATCORE_TARGET_CUDA_V0 = 2,
  MATCORE_TARGET_AMD_GPU_V0 = 3
};

typedef uint32_t matcore_fallback_v0;
enum {
  MATCORE_FALLBACK_INVALID_V0 = 0,
  MATCORE_FALLBACK_ERROR_V0 = 1,
  MATCORE_FALLBACK_ALLOW_V0 = 2
};

typedef int32_t matcore_status_code_v0;
enum {
  MATCORE_STATUS_OK_V0 = 0,
  MATCORE_STATUS_INVALID_ARGUMENT_V0 = 1,
  MATCORE_STATUS_ABI_MISMATCH_V0 = 2,
  MATCORE_STATUS_UNSUPPORTED_DTYPE_V0 = 3,
  MATCORE_STATUS_UNSUPPORTED_RANK_V0 = 4,
  MATCORE_STATUS_INVALID_SHAPE_V0 = 5,
  MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0 = 6,
  MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0 = 7,
  MATCORE_STATUS_MIXED_MEMORY_SPACES_V0 = 8,
  MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0 = 9,
  MATCORE_STATUS_UNSUPPORTED_MUTABILITY_V0 = 10,
  MATCORE_STATUS_SHAPE_MISMATCH_V0 = 11,
  MATCORE_STATUS_ALIAS_VIOLATION_V0 = 12,
  MATCORE_STATUS_UNSUPPORTED_TARGET_V0 = 13,
  MATCORE_STATUS_UNSUPPORTED_FALLBACK_V0 = 14,
  MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0 = 15,
  MATCORE_STATUS_INVALID_ALIGNMENT_V0 = 16
};

/* Dimensions and strides are in elements. Reserved fields must be zeroed. */
typedef struct matcore_tensor_desc_v0 {
  uint32_t abi_version;
  uint32_t struct_size;
  void *data;
  matcore_dtype_v0 dtype;
  uint32_t rank;
  int64_t dims[MATCORE_RUNTIME_MAX_RANK_V0];
  int64_t strides[MATCORE_RUNTIME_MAX_RANK_V0];
  matcore_memory_space_v0 memory_space;
  matcore_mutability_v0 mutability;
  uint64_t reserved[4];
} matcore_tensor_desc_v0;

typedef struct matcore_policy_v0 {
  uint32_t abi_version;
  uint32_t struct_size;
  matcore_target_v0 target;
  matcore_fallback_v0 fallback;
  uint64_t reserved[4];
} matcore_policy_v0;

/* message is a borrowed, process-lifetime string and must not be freed. */
typedef struct matcore_status_v0 {
  uint32_t abi_version;
  uint32_t struct_size;
  matcore_status_code_v0 code;
  uint32_t reserved0;
  const char *message;
  uint64_t reserved[2];
} matcore_status_v0;

/*
 * Synchronously computes out = lhs * rhs. Bootstrap v0 accepts only positive
 * rank-2, row-major contiguous f32 host tensors, CPU target, and
 * fallback=error. It allocates and copies nothing. out must not overlap either
 * input. All failures are returned; no C++ exception crosses this boundary.
 */
MATCORE_RUNTIME_API matcore_status_v0 matcore_runtime_gemm_f32_v0(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) MATCORE_RUNTIME_NOEXCEPT;

#if defined(__cplusplus)
}
#endif

#undef MATCORE_RUNTIME_NOEXCEPT

#endif
