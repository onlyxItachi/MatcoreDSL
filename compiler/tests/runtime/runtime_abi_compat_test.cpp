#include "matcore/runtime_c.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(MATCORE_RUNTIME_ABI_VERSION_V0 == 0);
static_assert(MATCORE_RUNTIME_PLAN_ABI_VERSION_V1 == 1);
static_assert(MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1 == 3);
static_assert(MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2 == 2);
static_assert(MATCORE_RUNTIME_PLAN_ABI_VERSION_V3 == 3);
static_assert(MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V3 == 8);

static_assert(std::is_standard_layout_v<matcore_tensor_desc_v0>);
static_assert(std::is_standard_layout_v<matcore_policy_v0>);
static_assert(std::is_standard_layout_v<matcore_status_v0>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_candidate_v1>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_plan_report_v1>);
static_assert(
    std::is_standard_layout_v<matcore_cpu_gemm_execution_options_v1>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_candidate_v2>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_plan_report_v2>);
static_assert(
    std::is_standard_layout_v<matcore_gemm_workspace_requirements_v1>);
static_assert(
    std::is_standard_layout_v<matcore_gemm_prepacked_b_requirements_v1>);
static_assert(std::is_standard_layout_v<matcore_packed_b_desc_v1>);
static_assert(std::is_standard_layout_v<matcore_cpu_capabilities_v2>);
static_assert(
    std::is_standard_layout_v<matcore_cpu_gemm_execution_options_v2>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_candidate_v3>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_plan_report_v3>);
static_assert(
    std::is_standard_layout_v<matcore_gemm_workspace_requirements_v2>);
static_assert(
    std::is_standard_layout_v<matcore_cpu_execution_context_options_v1>);
static_assert(
    std::is_standard_layout_v<matcore_cpu_execution_context_report_v1>);

#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(matcore_tensor_desc_v0) == 192);
static_assert(sizeof(matcore_policy_v0) == 48);
static_assert(sizeof(matcore_status_v0) == 40);
static_assert(sizeof(matcore_cpu_gemm_candidate_v1) == 48);
static_assert(sizeof(matcore_cpu_gemm_plan_report_v1) == 264);
static_assert(sizeof(matcore_cpu_gemm_execution_options_v1) == 56);
static_assert(sizeof(matcore_cpu_gemm_candidate_v2) == 64);
static_assert(sizeof(matcore_cpu_gemm_plan_report_v2) == 488);
static_assert(sizeof(matcore_gemm_workspace_requirements_v1) == 64);
static_assert(sizeof(matcore_gemm_prepacked_b_requirements_v1) == 72);
static_assert(sizeof(matcore_packed_b_desc_v1) == 112);
static_assert(sizeof(matcore_cpu_capabilities_v2) == 160);
static_assert(sizeof(matcore_cpu_gemm_execution_options_v2) == 72);
static_assert(sizeof(matcore_cpu_gemm_candidate_v3) == 120);
static_assert(sizeof(matcore_cpu_gemm_plan_report_v3) == 1336);
static_assert(sizeof(matcore_gemm_workspace_requirements_v2) == 96);
static_assert(sizeof(matcore_cpu_execution_context_options_v1) == 64);
static_assert(sizeof(matcore_cpu_execution_context_report_v1) == 160);

static_assert(offsetof(matcore_tensor_desc_v0, data) == 8);
static_assert(offsetof(matcore_tensor_desc_v0, dims) == 24);
static_assert(offsetof(matcore_tensor_desc_v0, strides) == 88);
static_assert(offsetof(matcore_tensor_desc_v0, reserved) == 160);

static_assert(offsetof(matcore_status_v0, message) == 16);
static_assert(offsetof(matcore_cpu_gemm_candidate_v1, stable_id) == 0);
static_assert(offsetof(matcore_cpu_gemm_candidate_v1, estimated_cost) == 16);
static_assert(offsetof(matcore_cpu_gemm_candidate_v1, reason) == 24);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v1, feature_bits) == 32);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v1, candidates) == 72);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v1, selected_stable_id) ==
              216);
static_assert(offsetof(matcore_cpu_gemm_candidate_v2, estimated_cost) == 24);
static_assert(offsetof(matcore_cpu_gemm_candidate_v2, reason) == 40);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v2, candidates) == 96);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v2, selected_stable_id) ==
              416);
static_assert(
    offsetof(matcore_gemm_workspace_requirements_v1, selected_stable_id) ==
    24);
static_assert(
    offsetof(matcore_gemm_prepacked_b_requirements_v1,
             execution_workspace_bytes) == 24);
static_assert(offsetof(matcore_packed_b_desc_v1, provenance) == 72);
static_assert(offsetof(matcore_cpu_gemm_execution_options_v2, smt_policy) ==
              24);
static_assert(offsetof(matcore_cpu_gemm_candidate_v3,
                       shared_workspace_bytes) == 40);
static_assert(offsetof(matcore_cpu_gemm_candidate_v3,
                       per_worker_workspace_bytes) == 48);
static_assert(offsetof(matcore_cpu_gemm_candidate_v3,
                       required_hardware_features) == 56);
static_assert(offsetof(matcore_cpu_gemm_candidate_v3, reason) == 96);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v3,
                       worker_affinity_induced_by_smt_policy) == 288);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v3, creator_logical_cpu) ==
              296);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v3, candidates) == 304);
static_assert(offsetof(matcore_cpu_gemm_plan_report_v3, selected_stable_id) ==
              1264);
static_assert(offsetof(matcore_cpu_execution_context_report_v1,
                       available_logical_cpu_count) == 52);
static_assert(offsetof(matcore_cpu_execution_context_report_v1,
                       worker_affinity_status) == 68);
static_assert(offsetof(matcore_cpu_execution_context_report_v1,
                       worker_affinity_induced_by_smt_policy) == 100);
static_assert(offsetof(matcore_cpu_execution_context_report_v1,
                       creator_logical_cpu) == 108);
static_assert(offsetof(matcore_cpu_execution_context_report_v1,
                       execution_generation) == 120);
#endif

static_assert(MATCORE_STATUS_INVALID_ALIGNMENT_V0 == 16);
static_assert(MATCORE_STATUS_PREPACK_MISMATCH_V0 == 21);
static_assert(MATCORE_STATUS_ACCUMULATOR_OVERFLOW_V0 == 25);
static_assert(MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V1 == UINT64_C(1));
static_assert(MATCORE_CPU_FEATURE_AVX2_V1 == (UINT64_C(1) << 1));
static_assert(MATCORE_CPU_FEATURE_FMA_V1 == (UINT64_C(1) << 2));
static_assert(MATCORE_CPU_FEATURE_AVX512F_V2 == (UINT64_C(1) << 3));
static_assert(MATCORE_CPU_FEATURE_AMX_INT8_V2 == (UINT64_C(1) << 11));
static_assert(MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX512_FMA_V3 ==
              8);

int main() { return 0; }
