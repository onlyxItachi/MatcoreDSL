#include "cpu_gemm_backend.h"

#include <algorithm>
#include <limits>

namespace matcore::mdslc::runtime {
namespace {

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t *result) noexcept {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs,
                 std::size_t *result) noexcept {
  if (result == nullptr ||
      rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_round_up(std::size_t value, std::size_t multiple,
                      std::size_t *result) noexcept {
  if (multiple == 0 || result == nullptr) return false;
  const std::size_t remainder = value % multiple;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  return checked_add(value, multiple - remainder, result);
}

bool positive_dimension(std::int64_t value, std::size_t *converted) noexcept {
  if (value <= 0 || converted == nullptr) return false;
  const auto unsigned_value = static_cast<std::uint64_t>(value);
  if (unsigned_value > std::numeric_limits<std::size_t>::max()) return false;
  *converted = static_cast<std::size_t>(unsigned_value);
  return true;
}

CpuPackedGemmStatusV1 validate_problem(
    const planner::CpuGemmProblemV1 &problem, std::size_t *m, std::size_t *n,
    std::size_t *k) noexcept {
  if (problem.element_type != planner::CpuScalarTypeV1::f32 ||
      problem.accumulation_type != planner::CpuScalarTypeV1::f32 ||
      problem.layout != planner::CpuLayoutV1::row_major_contiguous ||
      problem.minimum_alignment_bytes < alignof(float) ||
      (problem.minimum_alignment_bytes &
       (problem.minimum_alignment_bytes - 1U)) != 0) {
    return CpuPackedGemmStatusV1::invalid_problem;
  }
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
    return CpuPackedGemmStatusV1::invalid_problem;
  }
  if (!positive_dimension(problem.m, m) ||
      !positive_dimension(problem.n, n) ||
      !positive_dimension(problem.k, k)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }

  // Reject dimensions whose canonical contiguous tensor byte counts cannot be
  // represented. This also makes every later pointer offset representable.
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!checked_multiply(*m, *k, &elements) ||
      !checked_multiply(elements, sizeof(float), &bytes) ||
      !checked_multiply(*k, *n, &elements) ||
      !checked_multiply(elements, sizeof(float), &bytes) ||
      !checked_multiply(*m, *n, &elements) ||
      !checked_multiply(elements, sizeof(float), &bytes)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 make_execution_requirements(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceModeV1 mode,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept {
  if (requirements == nullptr) return CpuPackedGemmStatusV1::null_pointer;
  if (mode != CpuPackedGemmWorkspaceModeV1::transient_a_and_b &&
      mode != CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b) {
    return CpuPackedGemmStatusV1::invalid_problem;
  }

  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  const auto status = validate_problem(problem, &m, &n, &k);
  if (status != CpuPackedGemmStatusV1::success) return status;

  const std::size_t block_m = std::min(m, kCpuPackedGemmMcV1);
  const std::size_t block_n = std::min(n, kCpuPackedGemmNcV1);
  const std::size_t block_k = std::min(k, kCpuPackedGemmKcV1);
  std::size_t padded_m = 0;
  std::size_t padded_n = 0;
  if (!checked_round_up(block_m, kCpuPackedGemmMrV1, &padded_m) ||
      !checked_round_up(block_n, kCpuPackedGemmNrV1, &padded_n)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }

  std::size_t packed_a_elements = 0;
  std::size_t packed_a_bytes = 0;
  if (!checked_multiply(padded_m, block_k, &packed_a_elements) ||
      !checked_multiply(packed_a_elements, sizeof(float), &packed_a_bytes)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }

  std::size_t packed_b_bytes = 0;
  if (mode == CpuPackedGemmWorkspaceModeV1::transient_a_and_b) {
    std::size_t packed_b_elements = 0;
    if (!checked_multiply(padded_n, block_k, &packed_b_elements) ||
        !checked_multiply(packed_b_elements, sizeof(float), &packed_b_bytes)) {
      return CpuPackedGemmStatusV1::arithmetic_overflow;
    }
  }

  std::size_t packed_b_offset = 0;
  std::size_t total_bytes = 0;
  if (mode == CpuPackedGemmWorkspaceModeV1::transient_a_and_b) {
    if (!checked_round_up(packed_a_bytes, kCpuPackedGemmWorkspaceAlignmentV1,
                          &packed_b_offset) ||
        !checked_add(packed_b_offset, packed_b_bytes, &total_bytes)) {
      return CpuPackedGemmStatusV1::arithmetic_overflow;
    }
  } else {
    total_bytes = packed_a_bytes;
  }
  if (!checked_round_up(total_bytes, kCpuPackedGemmWorkspaceAlignmentV1,
                        &total_bytes)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }

  CpuPackedGemmWorkspaceRequirementsV1 result;
  result.mode = mode;
  result.total_bytes = total_bytes;
  result.packed_a_bytes = packed_a_bytes;
  result.packed_b_offset = packed_b_offset;
  result.packed_b_bytes = packed_b_bytes;
  *requirements = result;
  return CpuPackedGemmStatusV1::success;
}

}  // namespace

std::string_view cpu_packed_gemm_status_message_v1(
    CpuPackedGemmStatusV1 status) noexcept {
  switch (status) {
    case CpuPackedGemmStatusV1::success:
      return "success";
    case CpuPackedGemmStatusV1::invalid_problem:
      return "invalid packed GEMM problem";
    case CpuPackedGemmStatusV1::null_pointer:
      return "required pointer is null";
    case CpuPackedGemmStatusV1::invalid_pointer_alignment:
      return "tensor pointer violates the declared alignment";
    case CpuPackedGemmStatusV1::alias_violation:
      return "packed GEMM buffers overlap illegally";
    case CpuPackedGemmStatusV1::arithmetic_overflow:
      return "packed GEMM size arithmetic overflowed";
    case CpuPackedGemmStatusV1::isa_unavailable:
      return "AVX2/FMA execution is unavailable";
    case CpuPackedGemmStatusV1::workspace_misaligned:
      return "packed GEMM workspace is not 64-byte aligned";
    case CpuPackedGemmStatusV1::workspace_insufficient:
      return "packed GEMM workspace is too small";
    case CpuPackedGemmStatusV1::invalid_prepacked_b:
      return "prepacked-B view is invalid or incompatible";
  }
  return "unknown packed GEMM status";
}

CpuPackedGemmStatusV1 cpu_packed_avx2_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceModeV1 mode,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept {
  return make_execution_requirements(problem, mode, requirements);
}

CpuPackedGemmStatusV1 cpu_packed_avx2_prepacked_b_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept {
  if (requirements == nullptr) return CpuPackedGemmStatusV1::null_pointer;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  const auto status = validate_problem(problem, &m, &n, &k);
  if (status != CpuPackedGemmStatusV1::success) return status;

  std::size_t padded_n = 0;
  std::size_t packed_elements = 0;
  std::size_t packed_bytes = 0;
  std::size_t total_bytes = 0;
  if (!checked_round_up(n, kCpuPackedGemmNrV1, &padded_n) ||
      !checked_multiply(padded_n, k, &packed_elements) ||
      !checked_multiply(packed_elements, sizeof(float), &packed_bytes) ||
      !checked_round_up(packed_bytes, kCpuPackedGemmWorkspaceAlignmentV1,
                        &total_bytes)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }

  CpuPackedGemmWorkspaceRequirementsV1 result;
  result.mode = CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b;
  result.total_bytes = total_bytes;
  result.packed_b_bytes = packed_bytes;
  *requirements = result;
  return CpuPackedGemmStatusV1::success;
}

}  // namespace matcore::mdslc::runtime
