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
#define MATCORE_RUNTIME_PLAN_ABI_VERSION_V1 UINT32_C(1)
#define MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1 UINT32_C(3)
#define MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1 UINT32_C(1)
#define MATCORE_RUNTIME_PLAN_ABI_VERSION_V2 UINT32_C(2)
#define MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V2 UINT32_C(5)
#define MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2 UINT32_C(2)
#define MATCORE_RUNTIME_PLAN_ABI_VERSION_V3 UINT32_C(3)
#define MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V3 UINT32_C(8)
#define MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2 UINT32_C(2)
#define MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1 UINT32_C(1)
#define MATCORE_RUNTIME_TOPOLOGY_ABI_VERSION_V1 UINT32_C(1)

typedef uint32_t matcore_dtype_v0;
enum {
  MATCORE_DTYPE_INVALID_V0 = 0,
  MATCORE_DTYPE_F32_V0 = 1,
  MATCORE_DTYPE_F64_V0 = 2,
  MATCORE_DTYPE_BF16_V0 = 3,
  MATCORE_DTYPE_I8_V0 = 4,
  MATCORE_DTYPE_I32_V0 = 5
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
  MATCORE_STATUS_INVALID_ALIGNMENT_V0 = 16,
  MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0 = 17,
  MATCORE_STATUS_UNAVAILABLE_VARIANT_V0 = 18,
  MATCORE_STATUS_EXTERNAL_PROVIDER_FAILURE_V0 = 19,
  MATCORE_STATUS_INVALID_THREAD_COUNT_V0 = 20,
  MATCORE_STATUS_PREPACK_MISMATCH_V0 = 21,
  MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0 = 22,
  MATCORE_STATUS_EXECUTOR_FAILURE_V0 = 23,
  MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0 = 24,
  MATCORE_STATUS_ACCUMULATOR_OVERFLOW_V0 = 25
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

typedef uint32_t matcore_cpu_architecture_v1;
enum {
  MATCORE_CPU_ARCHITECTURE_UNKNOWN_V1 = 0,
  MATCORE_CPU_ARCHITECTURE_X86_64_V1 = 1,
  MATCORE_CPU_ARCHITECTURE_AARCH64_V1 = 2
};

typedef uint64_t matcore_cpu_feature_bits_v1;
enum {
  MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V1 = UINT64_C(1) << 0,
  MATCORE_CPU_FEATURE_AVX2_V1 = UINT64_C(1) << 1,
  MATCORE_CPU_FEATURE_FMA_V1 = UINT64_C(1) << 2
};

/*
 * Canonically ordered advanced-CPU feature vocabulary. A bit in the hardware
 * mask is not sufficient for execution: the matching OS, compiler,
 * implementation, and runtime-validation state is reported independently.
 */
typedef uint64_t matcore_cpu_feature_bits_v2;
enum {
  MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V2 = UINT64_C(1) << 0,
  MATCORE_CPU_FEATURE_AVX2_V2 = UINT64_C(1) << 1,
  MATCORE_CPU_FEATURE_FMA_V2 = UINT64_C(1) << 2,
  MATCORE_CPU_FEATURE_AVX512F_V2 = UINT64_C(1) << 3,
  MATCORE_CPU_FEATURE_AVX512DQ_V2 = UINT64_C(1) << 4,
  MATCORE_CPU_FEATURE_AVX512BW_V2 = UINT64_C(1) << 5,
  MATCORE_CPU_FEATURE_AVX512VL_V2 = UINT64_C(1) << 6,
  MATCORE_CPU_FEATURE_AVX512VNNI_V2 = UINT64_C(1) << 7,
  MATCORE_CPU_FEATURE_AVX512BF16_V2 = UINT64_C(1) << 8,
  MATCORE_CPU_FEATURE_AMX_TILE_V2 = UINT64_C(1) << 9,
  MATCORE_CPU_FEATURE_AMX_BF16_V2 = UINT64_C(1) << 10,
  MATCORE_CPU_FEATURE_AMX_INT8_V2 = UINT64_C(1) << 11
};

typedef struct matcore_cpu_capabilities_v2 {
  uint32_t abi_version;
  uint32_t struct_size;
  matcore_cpu_architecture_v1 architecture;
  uint32_t detection_complete;
  matcore_cpu_feature_bits_v2 hardware_known_features;
  matcore_cpu_feature_bits_v2 hardware_available_features;
  matcore_cpu_feature_bits_v2 os_known_features;
  matcore_cpu_feature_bits_v2 os_available_features;
  matcore_cpu_feature_bits_v2 compiler_known_features;
  matcore_cpu_feature_bits_v2 compiler_available_features;
  matcore_cpu_feature_bits_v2 implementation_known_features;
  matcore_cpu_feature_bits_v2 implementation_available_features;
  matcore_cpu_feature_bits_v2 runtime_validation_known_features;
  matcore_cpu_feature_bits_v2 runtime_validated_features;
  uint64_t os_xstate_mask;
  uint32_t usable_vector_bits;
  uint32_t os_xstate_mask_known;
  uint32_t amx_permission_known;
  uint32_t amx_permission_granted;
  uint32_t reserved0;
  uint64_t reserved[4];
} matcore_cpu_capabilities_v2;

typedef uint32_t matcore_cpu_affinity_policy_v1;
enum {
  MATCORE_CPU_AFFINITY_NONE_V1 = 0,
  MATCORE_CPU_AFFINITY_COMPACT_V1 = 1,
  MATCORE_CPU_AFFINITY_SCATTER_V1 = 2
};

typedef uint32_t matcore_cpu_numa_policy_v1;
enum {
  MATCORE_CPU_NUMA_SINGLE_NODE_V1 = 0,
  MATCORE_CPU_NUMA_LOCAL_FIRST_V1 = 1
};

typedef uint32_t matcore_cpu_plan_request_v1;
enum { MATCORE_CPU_PLAN_REQUEST_AUTOMATIC_V1 = 0 };

typedef uint32_t matcore_cpu_plan_status_v1;
enum {
  MATCORE_CPU_PLAN_STATUS_SELECTED_V1 = 1,
  MATCORE_CPU_PLAN_STATUS_NO_LEGAL_VARIANT_V1 = 2,
  MATCORE_CPU_PLAN_STATUS_FORCED_VARIANT_ILLEGAL_V1 = 3,
  MATCORE_CPU_PLAN_STATUS_INVALID_PROBLEM_V1 = 4,
  MATCORE_CPU_PLAN_STATUS_INVALID_CAPABILITIES_V1 = 5
};

/* stable_id and reason point to process-lifetime, read-only strings. */
typedef struct matcore_cpu_gemm_candidate_v1 {
  const char *stable_id;
  uint32_t legal;
  uint32_t deterministic_priority;
  uint64_t estimated_cost;
  const char *reason;
  uint64_t reserved[2];
} matcore_cpu_gemm_candidate_v1;

/*
 * Fixed-layout, machine-readable result for deterministic automatic CPU GEMM
 * planning. String pointers are process-lifetime and must not be freed.
 * Reserved fields are zero. estimated_cost is UINT64_MAX for illegal
 * candidates. Candidates appear in fixed registry order: reference, tiled,
 * compiler-vectorized.
 */
typedef struct matcore_cpu_gemm_plan_report_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t planner_version;
  matcore_cpu_plan_request_v1 request;
  matcore_cpu_plan_status_v1 plan_status;
  matcore_cpu_architecture_v1 architecture;
  uint32_t capability_detection_complete;
  uint32_t usable_vector_bits;
  matcore_cpu_feature_bits_v1 feature_bits;
  int64_t m;
  int64_t n;
  int64_t k;
  uint32_t minimum_alignment_bytes;
  uint32_t candidate_count;
  matcore_cpu_gemm_candidate_v1
      candidates[MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1];
  const char *selected_stable_id;
  const char *selection_reason;
  uint64_t reserved[4];
} matcore_cpu_gemm_plan_report_v1;

typedef uint32_t matcore_cpu_gemm_request_v2;
enum {
  MATCORE_CPU_GEMM_REQUEST_AUTOMATIC_V2 = 0,
  MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2 = 1,
  MATCORE_CPU_GEMM_REQUEST_FORCE_TILED_V2 = 2,
  MATCORE_CPU_GEMM_REQUEST_FORCE_COMPILER_VECTORIZED_V2 = 3,
  MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V2 = 4,
  MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V2 = 5
};

/*
 * Versioned execution controls for the additive workspace-aware path.
 * flags and reserved fields must be zero in v1. requested_threads is explicit;
 * candidates which are single-threaded report actual_threads=1.
 */
typedef struct matcore_cpu_gemm_execution_options_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  matcore_cpu_gemm_request_v2 request;
  uint32_t requested_threads;
  uint64_t flags;
  uint64_t reserved[4];
} matcore_cpu_gemm_execution_options_v1;

/* stable_id and reason point to process-lifetime, read-only strings. */
typedef struct matcore_cpu_gemm_candidate_v2 {
  const char *stable_id;
  uint32_t legal;
  uint32_t deterministic_priority;
  uint32_t actual_threads;
  uint32_t required_workspace_alignment;
  uint64_t estimated_cost;
  uint64_t required_workspace_bytes;
  const char *reason;
  uint64_t reserved[2];
} matcore_cpu_gemm_candidate_v2;

/*
 * Additive resource-aware plan report. Provider strings are borrowed,
 * process-lifetime strings. The five candidates use fixed registry order:
 * reference, tiled, compiler-vectorized, OpenBLAS, native-packed AVX2/FMA.
 */
typedef struct matcore_cpu_gemm_plan_report_v2 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t planner_version;
  matcore_cpu_gemm_request_v2 request;
  matcore_cpu_plan_status_v1 plan_status;
  matcore_cpu_architecture_v1 architecture;
  uint32_t capability_detection_complete;
  uint32_t usable_vector_bits;
  matcore_cpu_feature_bits_v1 feature_bits;
  int64_t m;
  int64_t n;
  int64_t k;
  uint32_t minimum_alignment_bytes;
  uint32_t requested_threads;
  uint32_t candidate_count;
  uint32_t selected_actual_threads;
  uint64_t selected_workspace_bytes;
  uint32_t selected_workspace_alignment;
  uint32_t reserved0;
  matcore_cpu_gemm_candidate_v2
      candidates[MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V2];
  const char *selected_stable_id;
  const char *selection_reason;
  const char *external_provider;
  const char *external_provider_version;
  const char *external_provider_config;
  uint64_t reserved[4];
} matcore_cpu_gemm_plan_report_v2;

typedef uint32_t matcore_cpu_gemm_request_v3;
enum {
  MATCORE_CPU_GEMM_REQUEST_AUTOMATIC_V3 = 0,
  MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V3 = 1,
  MATCORE_CPU_GEMM_REQUEST_FORCE_TILED_V3 = 2,
  MATCORE_CPU_GEMM_REQUEST_FORCE_COMPILER_VECTORIZED_V3 = 3,
  MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V3 = 4,
  MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V3 = 5,
  MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX512_FMA_V3 = 6,
  MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX2_FMA_V3 = 7,
  MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX512_FMA_V3 = 8
};

typedef struct matcore_cpu_gemm_execution_options_v2 {
  uint32_t abi_version;
  uint32_t struct_size;
  matcore_cpu_gemm_request_v3 request;
  uint32_t requested_threads;
  matcore_cpu_affinity_policy_v1 affinity_policy;
  matcore_cpu_numa_policy_v1 numa_policy;
  uint64_t flags;
  uint64_t reserved[4];
} matcore_cpu_gemm_execution_options_v2;

/* stable_id and reason point to process-lifetime, read-only strings. */
typedef struct matcore_cpu_gemm_candidate_v3 {
  const char *stable_id;
  uint32_t legal;
  uint32_t deterministic_priority;
  uint32_t actual_threads;
  uint32_t required_workspace_alignment;
  uint64_t estimated_cost;
  uint64_t required_workspace_bytes;
  matcore_cpu_feature_bits_v2 required_hardware_features;
  matcore_cpu_feature_bits_v2 required_os_features;
  matcore_cpu_feature_bits_v2 required_compiler_features;
  matcore_cpu_feature_bits_v2 required_implementation_features;
  uint32_t runtime_validated;
  uint32_t reserved0;
  const char *reason;
  uint64_t reserved[2];
} matcore_cpu_gemm_candidate_v3;

/*
 * Advanced F32 plan report. Candidates use fixed registry order: reference,
 * tiled, compiler-vectorized, OpenBLAS, packed AVX2, packed AVX-512,
 * parallel AVX2, parallel AVX-512. Topology fields are summaries; the runtime
 * retains the complete versioned mapping internally.
 */
typedef struct matcore_cpu_gemm_plan_report_v3 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t planner_version;
  matcore_cpu_gemm_request_v3 request;
  matcore_cpu_plan_status_v1 plan_status;
  matcore_cpu_architecture_v1 architecture;
  uint32_t capability_detection_complete;
  uint32_t usable_vector_bits;
  matcore_cpu_capabilities_v2 capabilities;
  int64_t m;
  int64_t n;
  int64_t k;
  uint32_t minimum_alignment_bytes;
  uint32_t requested_threads;
  uint32_t candidate_count;
  uint32_t selected_actual_threads;
  uint64_t selected_workspace_bytes;
  uint32_t selected_workspace_alignment;
  matcore_cpu_affinity_policy_v1 selected_affinity_policy;
  matcore_cpu_numa_policy_v1 selected_numa_policy;
  uint32_t topology_discovery_complete;
  uint32_t logical_cpu_count;
  uint32_t physical_core_count;
  uint32_t socket_count;
  uint32_t numa_node_count;
  matcore_cpu_gemm_candidate_v3
      candidates[MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V3];
  const char *selected_stable_id;
  const char *selection_reason;
  const char *external_provider;
  const char *external_provider_version;
  const char *external_provider_config;
  uint64_t reserved[4];
} matcore_cpu_gemm_plan_report_v3;

typedef struct matcore_gemm_workspace_requirements_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t workspace_bytes;
  uint32_t workspace_alignment;
  uint32_t actual_threads;
  const char *selected_stable_id;
  uint64_t reserved[4];
} matcore_gemm_workspace_requirements_v1;

typedef struct matcore_gemm_workspace_requirements_v2 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t workspace_bytes;
  uint32_t workspace_alignment;
  uint32_t actual_threads;
  uint64_t per_worker_workspace_bytes;
  uint64_t shared_workspace_bytes;
  const char *selected_stable_id;
  matcore_cpu_affinity_policy_v1 affinity_policy;
  matcore_cpu_numa_policy_v1 numa_policy;
  uint64_t reserved[4];
} matcore_gemm_workspace_requirements_v2;

typedef struct matcore_cpu_execution_context_options_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t requested_threads;
  matcore_cpu_affinity_policy_v1 affinity_policy;
  matcore_cpu_numa_policy_v1 numa_policy;
  uint32_t reserved0;
  uint64_t flags;
  uint64_t reserved[4];
} matcore_cpu_execution_context_options_v1;

typedef struct matcore_cpu_execution_context_report_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t requested_threads;
  uint32_t actual_worker_count;
  uint32_t persistent_worker_count;
  matcore_cpu_affinity_policy_v1 affinity_policy;
  matcore_cpu_numa_policy_v1 numa_policy;
  uint32_t topology_discovery_complete;
  uint32_t logical_cpu_count;
  uint32_t physical_core_count;
  uint32_t socket_count;
  uint32_t numa_node_count;
  uint64_t execution_generation;
  uint64_t reserved[4];
} matcore_cpu_execution_context_report_v1;

/* Opaque runtime-owned worker state. No C++ ABI type crosses this boundary. */
typedef struct matcore_cpu_execution_context_v1
    matcore_cpu_execution_context_v1;

/* Canonical storage representation for one bfloat16 value. */
typedef uint16_t matcore_bf16_v1;

/* Populates a zero-initialized, versioned fail-closed capability record. */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_query_cpu_capabilities_v2(
    matcore_cpu_capabilities_v2 *capabilities) MATCORE_RUNTIME_NOEXCEPT;

typedef struct matcore_gemm_prepacked_b_requirements_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t packed_b_bytes;
  uint32_t packed_b_alignment;
  uint32_t reserved0;
  uint64_t execution_workspace_bytes;
  uint32_t execution_workspace_alignment;
  uint32_t reserved1;
  uint64_t reserved[4];
} matcore_gemm_prepacked_b_requirements_v1;

/*
 * Borrowed descriptor for caller-owned native packed-B storage. Both source
 * and packed storage must remain alive and unmodified through execution. The
 * rhs descriptor supplied to execute_prepacked_b_v1 must retain the exact
 * source_data identity. Packed storage and this descriptor must not overlap
 * any GEMM tensor or each other. All fields are validated; callers must not
 * synthesize this descriptor.
 */
typedef struct matcore_packed_b_desc_v1 {
  uint32_t abi_version;
  uint32_t struct_size;
  const void *source_data;
  const void *packed_data;
  uint64_t storage_bytes;
  uint64_t packed_elements;
  int64_t k;
  int64_t n;
  uint32_t kc;
  uint32_t nc;
  uint32_t nr;
  uint32_t reserved0;
  uint64_t provenance;
  uint64_t reserved[4];
} matcore_packed_b_desc_v1;

/*
 * Validates the same descriptors and policy as matcore_runtime_gemm_f32_v0,
 * then returns the deterministic plan without executing or modifying output.
 * The caller must initialize report abi_version and struct_size; all other
 * report fields must be zero. No allocation, copy, or ambient diagnostic is
 * performed.
 */
MATCORE_RUNTIME_API matcore_status_v0 matcore_runtime_plan_gemm_f32_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    matcore_cpu_gemm_plan_report_v1 *report) MATCORE_RUNTIME_NOEXCEPT;

/*
 * Returns the deterministic resource-aware plan and exact caller-owned
 * workspace requirement without executing or modifying output data. options,
 * requirements, and report use strict version/size checks. Output fields in
 * requirements and report must be zero before the call.
 */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_workspace_size_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    matcore_gemm_workspace_requirements_v1 *requirements,
    matcore_cpu_gemm_plan_report_v2 *report) MATCORE_RUNTIME_NOEXCEPT;

/*
 * Executes exactly the selected resource-aware plan. The runtime allocates
 * nothing. Nonzero workspace must satisfy the queried size and alignment.
 * Every pre-execution validation failure leaves output data unchanged.
 */
MATCORE_RUNTIME_API matcore_status_v0 matcore_runtime_gemm_f32_execute_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    void *workspace,
    size_t workspace_bytes,
    matcore_cpu_gemm_plan_report_v2 *report) MATCORE_RUNTIME_NOEXCEPT;

/* Query caller-owned persistent B storage and transient A workspace sizes. */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_prepacked_b_size_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    matcore_gemm_prepacked_b_requirements_v1 *requirements)
    MATCORE_RUNTIME_NOEXCEPT;

/* Prepare persistent packed-B bytes without allocating or modifying output. */
MATCORE_RUNTIME_API matcore_status_v0 matcore_runtime_gemm_f32_prepack_b_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    void *packed_storage,
    size_t packed_storage_bytes,
    matcore_packed_b_desc_v1 *packed_b) MATCORE_RUNTIME_NOEXCEPT;

/*
 * Execute the forced native packed candidate with a validated packed-B view.
 * The rhs pointer must match the source identity captured by prepack_b_v1.
 */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_execute_prepacked_b_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    const matcore_packed_b_desc_v1 *packed_b,
    void *workspace,
    size_t workspace_bytes,
    matcore_cpu_gemm_plan_report_v2 *report) MATCORE_RUNTIME_NOEXCEPT;

/*
 * Creates a reusable CPU worker context. Worker-management storage is runtime
 * owned; GEMM packing/workspace storage remains caller owned. The output
 * handle must initially be null and is written only on success.
 */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_create_v1(
    const matcore_cpu_execution_context_options_v1 *options,
    matcore_cpu_execution_context_v1 **context,
    matcore_cpu_execution_context_report_v1 *report)
    MATCORE_RUNTIME_NOEXCEPT;

/* Stops and joins persistent workers. A null handle is rejected. */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_destroy_v1(
    matcore_cpu_execution_context_v1 *context) MATCORE_RUNTIME_NOEXCEPT;

/*
 * Plans context-backed F32 GEMM and reports total/shared/per-worker caller
 * workspace. Neither query nor execution performs hidden tensor copies.
 */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_context_workspace_size_v2(
    matcore_cpu_execution_context_v1 *context,
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v2 *options,
    matcore_gemm_workspace_requirements_v2 *requirements,
    matcore_cpu_gemm_plan_report_v3 *report) MATCORE_RUNTIME_NOEXCEPT;

MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_execute_context_v2(
    matcore_cpu_execution_context_v1 *context,
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v2 *options,
    void *workspace,
    size_t workspace_bytes,
    matcore_cpu_gemm_plan_report_v3 *report) MATCORE_RUNTIME_NOEXCEPT;

/*
 * Typed reference semantics used before any accelerated BF16 or INT8 variant:
 * BF16 inputs accumulate and output F32. I8 inputs accumulate and output I32;
 * mathematical accumulator overflow is rejected before output mutation.
 */
MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_bf16_f32_reference_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) MATCORE_RUNTIME_NOEXCEPT;

MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_i8_i32_reference_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) MATCORE_RUNTIME_NOEXCEPT;

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
