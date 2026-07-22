#include "matcore/runtime_c.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(MATCORE_RUNTIME_ABI_VERSION_V0 == 0);
static_assert(MATCORE_RUNTIME_PLAN_ABI_VERSION_V1 == 1);
static_assert(MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1 == 3);

static_assert(std::is_standard_layout_v<matcore_tensor_desc_v0>);
static_assert(std::is_standard_layout_v<matcore_policy_v0>);
static_assert(std::is_standard_layout_v<matcore_status_v0>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_candidate_v1>);
static_assert(std::is_standard_layout_v<matcore_cpu_gemm_plan_report_v1>);

#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(matcore_tensor_desc_v0) == 192);
static_assert(sizeof(matcore_policy_v0) == 48);
static_assert(sizeof(matcore_status_v0) == 40);
static_assert(sizeof(matcore_cpu_gemm_candidate_v1) == 48);
static_assert(sizeof(matcore_cpu_gemm_plan_report_v1) == 264);

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
#endif

static_assert(MATCORE_STATUS_INVALID_ALIGNMENT_V0 == 16);
static_assert(MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V1 == UINT64_C(1));
static_assert(MATCORE_CPU_FEATURE_AVX2_V1 == (UINT64_C(1) << 1));
static_assert(MATCORE_CPU_FEATURE_FMA_V1 == (UINT64_C(1) << 2));

int main() { return 0; }
