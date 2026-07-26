#ifndef MATCORE_MDSLC_RUNTIME_CPU_PACKED_AVX512_H
#define MATCORE_MDSLC_RUNTIME_CPU_PACKED_AVX512_H

#include "cpu_gemm_backend.h"

namespace matcore::mdslc::runtime {

// Build availability describes only whether the isolated function-targeted
// implementation was emitted. Runtime usability additionally requires the
// processor and operating system to expose usable AVX-512F state plus FMA.
bool cpu_packed_avx512_build_available_v1() noexcept;
bool cpu_packed_avx512_runtime_usable_v1() noexcept;

// AVX-512 deliberately reuses the packed GEMM v1 workspace and B-panel
// formats. A caller can therefore use the same queried layout and prepacked-B
// view with either the AVX2 or AVX-512 execution engine.
CpuPackedGemmStatusV1 cpu_packed_avx512_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceModeV1 mode,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept;

CpuPackedGemmStatusV1 cpu_packed_avx512_prepacked_b_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept;

CpuPackedGemmStatusV1 cpu_prepare_packed_b_avx512_v1(
    const planner::CpuGemmProblemV1 &problem, const float *rhs,
    void *packed_storage, std::size_t packed_storage_bytes,
    CpuPackedBViewV1 *view) noexcept;

CpuPackedGemmStatusV1 cpu_execute_packed_avx512_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes) noexcept;

CpuPackedGemmStatusV1 cpu_execute_packed_avx512_prepacked_b_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs, float *out,
    const CpuPackedBViewV1 &packed_b, void *workspace,
    std::size_t workspace_bytes) noexcept;

namespace detail {

// Internal prevalidated full-tile callback shared by the serial and parallel
// packed executors. It preserves the common packed-B v1 format but is not an
// installed or public ABI. Callers must prove a complete 4x32 output tile,
// nonzero K, valid packed panel extents, and usable AVX-512F/FMA state.
extern "C" void
matcore_internal_cpu_packed_avx512_4x32_full_microkernel_f32_m7(
    const float *packed_a, const float *packed_b0, const float *packed_b1,
    std::size_t k, float *output, std::size_t output_stride,
    bool accumulate) noexcept;

// Exact C-linkage symbol for machine-code inspection. Callers must use the
// checked execution entry points above; invoking this function on a machine
// without usable AVX-512F state is illegal.
extern "C" void matcore_cpu_packed_avx512_4x16_microkernel_f32_v1(
    const float *packed_a, const float *packed_b, std::size_t k,
    float *output, std::size_t output_stride, std::uint32_t rows,
    std::uint32_t columns, bool accumulate) noexcept;

}  // namespace detail
}  // namespace matcore::mdslc::runtime

#endif
