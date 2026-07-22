#ifndef MATCORE_MDSLC_RUNTIME_CPU_GEMM_BACKEND_H
#define MATCORE_MDSLC_RUNTIME_CPU_GEMM_BACKEND_H

#include "cpu_planner.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuPackedGemmBackendVersionV1 = 1;
inline constexpr std::size_t kCpuPackedGemmWorkspaceAlignmentV1 = 64;

// The 4x16 register tile uses eight YMM accumulators, two B vectors, and four
// broadcast A operands without exhausting the sixteen-register AVX2 file.
// KC=256 keeps one 4xKC A micro-panel (4 KiB) plus one KCx16 B micro-panel
// (16 KiB) close to L1 scale. MCxKC A (128 KiB) and KCxNC B (256 KiB) packed
// blocks total 384 KiB, a conservative L2-scale working set on the validation
// host. These are deterministic v1 parameters, not universal optima.
inline constexpr std::size_t kCpuPackedGemmMrV1 = 4;
inline constexpr std::size_t kCpuPackedGemmNrV1 = 16;
inline constexpr std::size_t kCpuPackedGemmMcV1 = 128;
inline constexpr std::size_t kCpuPackedGemmNcV1 = 256;
inline constexpr std::size_t kCpuPackedGemmKcV1 = 256;

enum class CpuPackedGemmStatusV1 : std::uint32_t {
  success = 0,
  invalid_problem = 1,
  null_pointer = 2,
  invalid_pointer_alignment = 3,
  alias_violation = 4,
  arithmetic_overflow = 5,
  isa_unavailable = 6,
  workspace_misaligned = 7,
  workspace_insufficient = 8,
  invalid_prepacked_b = 9,
};

enum class CpuPackedGemmWorkspaceModeV1 : std::uint32_t {
  transient_a_and_b = 0,
  transient_a_with_prepacked_b = 1,
};

struct CpuPackedGemmWorkspaceRequirementsV1 {
  std::uint32_t version = kCpuPackedGemmBackendVersionV1;
  CpuPackedGemmWorkspaceModeV1 mode =
      CpuPackedGemmWorkspaceModeV1::transient_a_and_b;
  std::size_t alignment_bytes = kCpuPackedGemmWorkspaceAlignmentV1;
  std::size_t total_bytes = 0;
  std::size_t packed_a_offset = 0;
  std::size_t packed_a_bytes = 0;
  std::size_t packed_b_offset = 0;
  std::size_t packed_b_bytes = 0;
};

// This is a borrowed view over caller-owned storage. The source pointer is
// retained only for the semantic no-alias check; execution reads packed_data.
// The caller must not mutate or release the source or packed storage until all
// executions using this view have completed.
struct CpuPackedBViewV1 {
  std::uint32_t version = 0;
  std::uint32_t struct_size = 0;
  const float *source_data = nullptr;
  const float *packed_data = nullptr;
  std::size_t storage_bytes = 0;
  std::size_t packed_elements = 0;
  std::int64_t k = 0;
  std::int64_t n = 0;
  std::uint32_t kc = 0;
  std::uint32_t nc = 0;
  std::uint32_t nr = 0;
  std::uint32_t reserved0 = 0;
  std::uint64_t provenance = 0;
};

std::string_view cpu_packed_gemm_status_message_v1(
    CpuPackedGemmStatusV1 status) noexcept;

bool cpu_packed_avx2_build_available_v1() noexcept;
bool cpu_packed_avx2_runtime_usable_v1() noexcept;

CpuPackedGemmStatusV1 cpu_packed_avx2_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceModeV1 mode,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept;

CpuPackedGemmStatusV1 cpu_packed_avx2_prepacked_b_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept;

CpuPackedGemmStatusV1 cpu_prepare_packed_b_avx2_v1(
    const planner::CpuGemmProblemV1 &problem, const float *rhs,
    void *packed_storage, std::size_t packed_storage_bytes,
    CpuPackedBViewV1 *view) noexcept;

CpuPackedGemmStatusV1 cpu_execute_packed_avx2_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes) noexcept;

CpuPackedGemmStatusV1 cpu_execute_packed_avx2_prepacked_b_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs, float *out,
    const CpuPackedBViewV1 &packed_b, void *workspace,
    std::size_t workspace_bytes) noexcept;

namespace detail {

// This symbol is deliberately isolated for exact machine-code inspection. It
// is an internal implementation detail despite its stable C linkage name.
extern "C" void matcore_cpu_packed_avx2_4x16_microkernel_f32_v1(
    const float *packed_a, const float *packed_b, std::size_t k,
    float *output, std::size_t output_stride, std::uint32_t rows,
    std::uint32_t columns, bool accumulate) noexcept;

}  // namespace detail
}  // namespace matcore::mdslc::runtime

#endif
