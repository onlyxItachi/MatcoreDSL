#include "matcore/runtime_c.h"

#include "cpu_backend_registry.h"
#include "cpu_execution_context.h"
#include "cpu_gemm_backend.h"
#include "cpu_numeric_reference.h"
#include "cpu_openblas.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner_v3_resources.h"
#include "cpu_packed_avx512.h"
#include "cpu_planner.h"
#include "cpu_planner_v2.h"
#include "cpu_planner_v3.h"
#include "cpu_runtime_validation.h"
#include "cpu_capability_v2.h"
#include "cpu_topology_v1.h"
#include "thread_affinity_v1.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

static_assert(matcore::mdslc::planner::kCpuGemmCandidateCountV1 ==
                  MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1,
              "CPU planner registry and C plan report must stay in sync");
static_assert(matcore::mdslc::planner::kCpuGemmCandidateCountV2 ==
                  MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V2,
              "CPU planner v2 registry and C plan report must stay in sync");
static_assert(matcore::mdslc::planner::kCpuGemmCandidateCountV3 ==
                  MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V3,
              "CPU planner v3 registry and C plan report must stay in sync");

struct matcore_cpu_execution_context_v1 {
  std::uint64_t magic = 0;
  std::unique_ptr<matcore::mdslc::runtime::CpuExecutionContextV1> workers;
  matcore::mdslc::runtime::CpuRuntimeValidationEvidenceV1 validation_evidence;
  std::uint64_t validation_submission_baseline = 0;
  matcore::mdslc::platform::CpuCapabilitiesV2 capabilities;
  // The complete system topology is retained only for diagnostics. Planning
  // and placement use the process-affinity-restricted topology below.
  matcore::mdslc::platform::CpuTopologyV1 topology;
  matcore::mdslc::platform::CpuTopologyV1 available_topology;
  matcore::mdslc::planner::CpuPlannerPlacementEvidenceV1 placement_evidence;
  bool process_affinity_discovery_complete = false;
  std::int32_t process_affinity_platform_error = 0;
  matcore_cpu_affinity_policy_v1 affinity_policy =
      MATCORE_CPU_AFFINITY_NONE_V1;
  matcore_cpu_numa_policy_v1 numa_policy =
      MATCORE_CPU_NUMA_SINGLE_NODE_V1;
  matcore_cpu_smt_policy_v1 smt_policy =
      MATCORE_CPU_SMT_PHYSICAL_CORES_ONLY_V1;
  bool worker_affinity_induced_by_smt_policy = false;
  bool worker_affinity_induced_by_numa_policy = false;
  std::uint32_t creator_logical_cpu =
      matcore::mdslc::platform::kUnknownLogicalCpuV1;
  std::uint32_t selected_numa_node =
      matcore::mdslc::platform::kUnknownTopologyIdV1;
};

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

matcore_status_v0 validate_cpu_policy_v0(
    const matcore_policy_v0 *policy) noexcept {
  if (policy == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "GEMM policy must be non-null");
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
  return status(MATCORE_STATUS_OK_V0, "ok");
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

bool byte_range(const void *pointer, std::size_t bytes,
                std::uintptr_t *begin, std::uintptr_t *end) noexcept {
  if (pointer == nullptr || begin == nullptr || end == nullptr) return false;
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address)
    return false;
  *begin = address;
  *end = address + bytes;
  return true;
}

bool overlaps(std::uintptr_t lhs_begin, std::uintptr_t lhs_end,
              std::uintptr_t rhs_begin, std::uintptr_t rhs_end) noexcept {
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

struct ValidatedGemmV0 {
  matcore::mdslc::planner::CpuGemmProblemV1 problem;
  const float *lhs = nullptr;
  const float *rhs = nullptr;
  float *out = nullptr;
};

matcore_status_v0 validate_gemm_v0(const matcore_tensor_desc_v0 *out,
                                   const matcore_tensor_desc_v0 *lhs,
                                   const matcore_tensor_desc_v0 *rhs,
                                   const matcore_policy_v0 *policy,
                                   ValidatedGemmV0 *validated) noexcept {
  if (out == nullptr || lhs == nullptr || rhs == nullptr || policy == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "GEMM descriptors and policy must be non-null");
  }
  matcore_status_v0 result = validate_cpu_policy_v0(policy);
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  result = validate_tensor(*out, "output data is null");
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

  const std::uint32_t out_alignment =
      matcore::mdslc::planner::pointer_alignment_bytes(out->data);
  const std::uint32_t lhs_alignment =
      matcore::mdslc::planner::pointer_alignment_bytes(lhs->data);
  const std::uint32_t rhs_alignment =
      matcore::mdslc::planner::pointer_alignment_bytes(rhs->data);
  std::uint32_t minimum_alignment =
      out_alignment < lhs_alignment ? out_alignment : lhs_alignment;
  if (rhs_alignment < minimum_alignment) minimum_alignment = rhs_alignment;
  validated->problem = {m,
                        n,
                        k,
                        matcore::mdslc::planner::CpuScalarTypeV1::f32,
                        matcore::mdslc::planner::CpuScalarTypeV1::f32,
                        matcore::mdslc::planner::CpuLayoutV1::row_major_contiguous,
                        minimum_alignment};
  validated->lhs = static_cast<const float *>(lhs->data);
  validated->rhs = static_cast<const float *>(rhs->data);
  validated->out = static_cast<float *>(out->data);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

bool empty_candidate_report(
    const matcore_cpu_gemm_candidate_v1 &candidate) noexcept {
  return candidate.stable_id == nullptr && candidate.legal == 0 &&
         candidate.deterministic_priority == 0 &&
         candidate.estimated_cost == 0 && candidate.reason == nullptr &&
         reserved_is_zero(candidate.reserved);
}

matcore_status_v0 validate_empty_report(
    const matcore_cpu_gemm_plan_report_v1 &report) noexcept {
  if (report.abi_version != MATCORE_RUNTIME_PLAN_ABI_VERSION_V1 ||
      report.struct_size != sizeof(matcore_cpu_gemm_plan_report_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU plan report ABI version or size mismatch");
  }
  if (report.planner_version != 0 || report.request != 0 ||
      report.plan_status != 0 || report.architecture != 0 ||
      report.capability_detection_complete != 0 ||
      report.usable_vector_bits != 0 || report.feature_bits != 0 ||
      report.m != 0 || report.n != 0 || report.k != 0 ||
      report.minimum_alignment_bytes != 0 || report.candidate_count != 0 ||
      report.selected_stable_id != nullptr || report.selection_reason != nullptr ||
      !reserved_is_zero(report.reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan report output fields must be zero");
  }
  for (const auto &candidate : report.candidates) {
    if (!empty_candidate_report(candidate)) {
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "CPU plan report output fields must be zero");
    }
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_cpu_architecture_v1 to_c_architecture(
    matcore::mdslc::planner::CpuArchitectureV1 architecture) noexcept {
  using Architecture = matcore::mdslc::planner::CpuArchitectureV1;
  switch (architecture) {
    case Architecture::x86_64:
      return MATCORE_CPU_ARCHITECTURE_X86_64_V1;
    case Architecture::aarch64:
      return MATCORE_CPU_ARCHITECTURE_AARCH64_V1;
    case Architecture::unknown:
      return MATCORE_CPU_ARCHITECTURE_UNKNOWN_V1;
  }
  return MATCORE_CPU_ARCHITECTURE_UNKNOWN_V1;
}

matcore_cpu_plan_status_v1 to_c_plan_status(
    matcore::mdslc::planner::CpuPlanStatusV1 plan_status) noexcept {
  using Status = matcore::mdslc::planner::CpuPlanStatusV1;
  switch (plan_status) {
    case Status::selected:
      return MATCORE_CPU_PLAN_STATUS_SELECTED_V1;
    case Status::no_legal_variant:
      return MATCORE_CPU_PLAN_STATUS_NO_LEGAL_VARIANT_V1;
    case Status::forced_variant_illegal:
      return MATCORE_CPU_PLAN_STATUS_FORCED_VARIANT_ILLEGAL_V1;
    case Status::invalid_problem:
      return MATCORE_CPU_PLAN_STATUS_INVALID_PROBLEM_V1;
    case Status::invalid_capabilities:
      return MATCORE_CPU_PLAN_STATUS_INVALID_CAPABILITIES_V1;
  }
  return MATCORE_CPU_PLAN_STATUS_INVALID_CAPABILITIES_V1;
}

void populate_report(const matcore::mdslc::planner::CpuGemmPlanV1 &plan,
                     matcore_cpu_gemm_plan_report_v1 *report) noexcept {
  matcore_cpu_gemm_plan_report_v1 output{};
  output.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  output.struct_size = sizeof(output);
  output.planner_version = plan.planner_version;
  output.request = MATCORE_CPU_PLAN_REQUEST_AUTOMATIC_V1;
  output.plan_status = to_c_plan_status(plan.status);
  output.architecture = to_c_architecture(plan.capabilities.architecture);
  output.capability_detection_complete =
      plan.capabilities.detection_complete ? 1U : 0U;
  output.usable_vector_bits = plan.capabilities.usable_vector_bits;
  output.feature_bits = plan.capabilities.features;
  output.m = plan.problem.m;
  output.n = plan.problem.n;
  output.k = plan.problem.k;
  output.minimum_alignment_bytes = plan.problem.minimum_alignment_bytes;
  output.candidate_count = static_cast<uint32_t>(plan.candidates.size());
  for (std::size_t index = 0; index < plan.candidates.size(); ++index) {
    const auto &source = plan.candidates[index];
    auto &destination = output.candidates[index];
    destination.stable_id = source.stable_id.data();
    destination.legal = source.legal ? 1U : 0U;
    destination.deterministic_priority = source.deterministic_priority;
    destination.estimated_cost = source.estimated_cost;
    destination.reason = source.reason.data();
  }
  output.selected_stable_id = plan.selected_id.data();
  output.selection_reason = plan.selection_reason.data();
  *report = output;
}

bool empty_candidate_report_v2(
    const matcore_cpu_gemm_candidate_v2 &candidate) noexcept {
  return candidate.stable_id == nullptr && candidate.legal == 0 &&
         candidate.deterministic_priority == 0 &&
         candidate.actual_threads == 0 &&
         candidate.required_workspace_alignment == 0 &&
         candidate.estimated_cost == 0 &&
         candidate.required_workspace_bytes == 0 && candidate.reason == nullptr &&
         reserved_is_zero(candidate.reserved);
}

matcore_status_v0 validate_empty_report_v2(
    const matcore_cpu_gemm_plan_report_v2 &report) noexcept {
  if (report.abi_version != MATCORE_RUNTIME_PLAN_ABI_VERSION_V2 ||
      report.struct_size != sizeof(matcore_cpu_gemm_plan_report_v2)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU plan v2 report ABI version or size mismatch");
  }
  if (report.planner_version != 0 || report.request != 0 ||
      report.plan_status != 0 || report.architecture != 0 ||
      report.capability_detection_complete != 0 ||
      report.usable_vector_bits != 0 || report.feature_bits != 0 ||
      report.m != 0 || report.n != 0 || report.k != 0 ||
      report.minimum_alignment_bytes != 0 || report.requested_threads != 0 ||
      report.candidate_count != 0 || report.selected_actual_threads != 0 ||
      report.selected_workspace_bytes != 0 ||
      report.selected_workspace_alignment != 0 || report.reserved0 != 0 ||
      report.selected_stable_id != nullptr || report.selection_reason != nullptr ||
      report.external_provider != nullptr ||
      report.external_provider_version != nullptr ||
      report.external_provider_config != nullptr ||
      !reserved_is_zero(report.reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan v2 report output fields must be zero");
  }
  for (const auto &candidate : report.candidates) {
    if (!empty_candidate_report_v2(candidate)) {
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "CPU plan v2 candidate output fields must be zero");
    }
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 validate_execution_options(
    const matcore_cpu_gemm_execution_options_v1 *options,
    matcore::mdslc::planner::CpuGemmRequestV2 *request) noexcept {
  if (options == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM execution options must be non-null");
  }
  if (options->abi_version != MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1 ||
      options->struct_size != sizeof(matcore_cpu_gemm_execution_options_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM execution options ABI version or size mismatch");
  }
  if (options->requested_threads == 0) {
    return status(MATCORE_STATUS_INVALID_THREAD_COUNT_V0,
                  "CPU GEMM requested thread count must be positive");
  }
  if (options->flags != 0 || !reserved_is_zero(options->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM execution option flags and reserved fields must be zero");
  }
  using Request = matcore::mdslc::planner::CpuGemmRequestV2;
  switch (options->request) {
    case MATCORE_CPU_GEMM_REQUEST_AUTOMATIC_V2:
      *request = Request::automatic;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2:
      *request = Request::force_reference;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_TILED_V2:
      *request = Request::force_tiled;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_COMPILER_VECTORIZED_V2:
      *request = Request::force_compiler_vectorized;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V2:
      *request = Request::force_external_openblas;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V2:
      *request = Request::force_native_packed_avx2_fma;
      break;
    default:
      return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                    "CPU GEMM execution requested an unknown variant");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 validate_empty_workspace_requirements(
    const matcore_gemm_workspace_requirements_v1 *requirements) noexcept {
  if (requirements == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM workspace requirements must be non-null");
  }
  if (requirements->abi_version != MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1 ||
      requirements->struct_size !=
          sizeof(matcore_gemm_workspace_requirements_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM workspace requirements ABI version or size mismatch");
  }
  if (requirements->workspace_bytes != 0 ||
      requirements->workspace_alignment != 0 ||
      requirements->actual_threads != 0 ||
      requirements->selected_stable_id != nullptr ||
      !reserved_is_zero(requirements->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM workspace requirement output fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 validate_empty_prepacked_requirements(
    const matcore_gemm_prepacked_b_requirements_v1 *requirements) noexcept {
  if (requirements == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM prepacked-B requirements must be non-null");
  }
  if (requirements->abi_version != MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1 ||
      requirements->struct_size !=
          sizeof(matcore_gemm_prepacked_b_requirements_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM prepacked-B requirements ABI version or size mismatch");
  }
  if (requirements->packed_b_bytes != 0 ||
      requirements->packed_b_alignment != 0 || requirements->reserved0 != 0 ||
      requirements->execution_workspace_bytes != 0 ||
      requirements->execution_workspace_alignment != 0 ||
      requirements->reserved1 != 0 ||
      !reserved_is_zero(requirements->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM prepacked-B requirement output fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 validate_empty_packed_b_desc(
    const matcore_packed_b_desc_v1 *packed_b) noexcept {
  if (packed_b == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM packed-B descriptor must be non-null");
  }
  if (packed_b->abi_version != MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1 ||
      packed_b->struct_size != sizeof(matcore_packed_b_desc_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM packed-B descriptor ABI version or size mismatch");
  }
  if (packed_b->source_data != nullptr || packed_b->packed_data != nullptr ||
      packed_b->storage_bytes != 0 || packed_b->packed_elements != 0 ||
      packed_b->k != 0 || packed_b->n != 0 || packed_b->kc != 0 ||
      packed_b->nc != 0 || packed_b->nr != 0 || packed_b->reserved0 != 0 ||
      packed_b->provenance != 0 || !reserved_is_zero(packed_b->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM packed-B output fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 unpack_packed_b_desc(
    const matcore_packed_b_desc_v1 *packed_b,
    matcore::mdslc::runtime::CpuPackedBViewV1 *view) noexcept {
  if (packed_b == nullptr || view == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM packed-B descriptor must be non-null");
  }
  if (packed_b->abi_version != MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1 ||
      packed_b->struct_size != sizeof(matcore_packed_b_desc_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM packed-B descriptor ABI version or size mismatch");
  }
  if (packed_b->reserved0 != 0 || !reserved_is_zero(packed_b->reserved) ||
      packed_b->storage_bytes > std::numeric_limits<std::size_t>::max() ||
      packed_b->packed_elements > std::numeric_limits<std::size_t>::max()) {
    return status(MATCORE_STATUS_PREPACK_MISMATCH_V0,
                  "CPU GEMM packed-B descriptor is malformed");
  }
  matcore::mdslc::runtime::CpuPackedBViewV1 output;
  output.version = matcore::mdslc::runtime::kCpuPackedGemmBackendVersionV1;
  output.struct_size = sizeof(output);
  output.source_data = static_cast<const float *>(packed_b->source_data);
  output.packed_data = static_cast<const float *>(packed_b->packed_data);
  output.storage_bytes = static_cast<std::size_t>(packed_b->storage_bytes);
  output.packed_elements = static_cast<std::size_t>(packed_b->packed_elements);
  output.k = packed_b->k;
  output.n = packed_b->n;
  output.kc = packed_b->kc;
  output.nc = packed_b->nc;
  output.nr = packed_b->nr;
  output.provenance = packed_b->provenance;
  *view = output;
  return status(MATCORE_STATUS_OK_V0, "ok");
}

bool forced_native_packed_request(
    matcore::mdslc::planner::CpuGemmRequestV2 request) noexcept {
  return request ==
         matcore::mdslc::planner::CpuGemmRequestV2::
             force_native_packed_avx2_fma;
}

matcore_status_v0 packed_execution_status(
    matcore::mdslc::runtime::CpuPackedGemmStatusV1 packed_status) noexcept {
  using PackedStatus = matcore::mdslc::runtime::CpuPackedGemmStatusV1;
  switch (packed_status) {
    case PackedStatus::success:
      return status(MATCORE_STATUS_OK_V0, "ok");
    case PackedStatus::workspace_insufficient:
      return status(MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0,
                    "native packed GEMM workspace is too small");
    case PackedStatus::workspace_misaligned:
    case PackedStatus::invalid_pointer_alignment:
      return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                    "native packed GEMM alignment contract failed");
    case PackedStatus::alias_violation:
      return status(MATCORE_STATUS_ALIAS_VIOLATION_V0,
                    "native packed GEMM buffers overlap illegally");
    case PackedStatus::arithmetic_overflow:
      return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                    "native packed GEMM size arithmetic overflowed");
    case PackedStatus::isa_unavailable:
      return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                    "native packed AVX2/FMA is unavailable on this CPU");
    case PackedStatus::invalid_prepacked_b:
      return status(MATCORE_STATUS_PREPACK_MISMATCH_V0,
                    "native packed GEMM prepacked-B view is incompatible");
    case PackedStatus::invalid_problem:
    case PackedStatus::null_pointer:
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "native packed GEMM received an invalid argument");
  }
  return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                "native packed GEMM returned an unknown status");
}

void populate_report_v2(
    const matcore::mdslc::planner::CpuGemmPlanV2 &plan,
    const matcore::mdslc::runtime::OpenBlasProviderInfoV1 &provider,
    matcore_cpu_gemm_plan_report_v2 *report) noexcept {
  matcore_cpu_gemm_plan_report_v2 output{};
  output.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
  output.struct_size = sizeof(output);
  output.planner_version = plan.planner_version;
  output.request = static_cast<matcore_cpu_gemm_request_v2>(plan.request);
  output.plan_status = to_c_plan_status(plan.status);
  output.architecture = to_c_architecture(plan.capabilities.architecture);
  output.capability_detection_complete =
      plan.capabilities.detection_complete ? 1U : 0U;
  output.usable_vector_bits = plan.capabilities.usable_vector_bits;
  output.feature_bits = plan.capabilities.features;
  output.m = plan.problem.m;
  output.n = plan.problem.n;
  output.k = plan.problem.k;
  output.minimum_alignment_bytes = plan.problem.minimum_alignment_bytes;
  output.requested_threads = plan.resources.requested_threads;
  output.candidate_count = static_cast<std::uint32_t>(plan.candidates.size());
  for (std::size_t index = 0; index < plan.candidates.size(); ++index) {
    const auto &source = plan.candidates[index];
    auto &destination = output.candidates[index];
    destination.stable_id = source.stable_id.data();
    destination.legal = source.legal ? 1U : 0U;
    destination.deterministic_priority = source.deterministic_priority;
    destination.actual_threads = source.actual_threads;
    destination.required_workspace_alignment =
        source.required_workspace_alignment;
    destination.estimated_cost = source.estimated_cost;
    destination.required_workspace_bytes = source.required_workspace_bytes;
    destination.reason = source.reason.data();
  }
  output.selected_stable_id = plan.selected_id.data();
  output.selection_reason = plan.selection_reason.data();
  output.external_provider = provider.linked ? "OpenBLAS" : "unavailable";
  output.external_provider_version = provider.package_version;
  output.external_provider_config = provider.runtime_config;
  if (plan.status == matcore::mdslc::planner::CpuPlanStatusV1::selected) {
    const std::size_t selected =
        static_cast<std::size_t>(plan.selected_variant);
    if (selected < plan.candidates.size()) {
      const auto &candidate = plan.candidates[selected];
      output.selected_actual_threads = candidate.actual_threads;
      output.selected_workspace_bytes = candidate.required_workspace_bytes;
      output.selected_workspace_alignment =
          candidate.required_workspace_alignment;
    }
  }
  *report = output;
}

matcore_status_v0 unavailable_plan_status(
    const matcore::mdslc::planner::CpuGemmPlanV2 &plan) noexcept {
  if (plan.status == matcore::mdslc::planner::CpuPlanStatusV1::invalid_problem ||
      plan.status ==
          matcore::mdslc::planner::CpuPlanStatusV1::invalid_capabilities) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM planner rejected the problem or capability record");
  }
  if (plan.request !=
      matcore::mdslc::planner::CpuGemmRequestV2::automatic) {
    const std::size_t candidate_index =
        static_cast<std::size_t>(plan.request) - 1U;
    if (candidate_index < plan.candidates.size() &&
        !plan.candidates[candidate_index].reason.empty()) {
      return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                    plan.candidates[candidate_index].reason.data());
    }
  }
  return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                "CPU GEMM planner found no legal requested implementation");
}

bool execute_legacy_variant(
    matcore::mdslc::planner::CpuGemmVariantV2 variant,
    const ValidatedGemmV0 &validated) noexcept {
  using RequestV1 = matcore::mdslc::planner::CpuGemmRequestV1;
  RequestV1 request = RequestV1::force_reference;
  switch (variant) {
    case matcore::mdslc::planner::CpuGemmVariantV2::reference:
      request = RequestV1::force_reference;
      break;
    case matcore::mdslc::planner::CpuGemmVariantV2::tiled:
      request = RequestV1::force_tiled;
      break;
    case matcore::mdslc::planner::CpuGemmVariantV2::compiler_vectorized:
      request = RequestV1::force_compiler_vectorized;
      break;
    case matcore::mdslc::planner::CpuGemmVariantV2::external_openblas:
    case matcore::mdslc::planner::CpuGemmVariantV2::native_packed_avx2_fma:
      return false;
  }
  const auto legacy_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1(), request);
  return matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
      legacy_plan, validated.lhs, validated.rhs, validated.out);
}

std::size_t dtype_size_v0(matcore_dtype_v0 dtype) noexcept {
  switch (dtype) {
    case MATCORE_DTYPE_BF16_V0:
      return sizeof(matcore_bf16_v1);
    case MATCORE_DTYPE_I8_V0:
      return sizeof(std::int8_t);
    case MATCORE_DTYPE_F32_V0:
      return sizeof(float);
    case MATCORE_DTYPE_I32_V0:
      return sizeof(std::int32_t);
    default:
      return 0;
  }
}

matcore_status_v0 validate_typed_tensor_v1(
    const matcore_tensor_desc_v0 &tensor, matcore_dtype_v0 expected_dtype,
    const char *null_message) noexcept {
  if (!valid_header(tensor.abi_version, tensor.struct_size,
                    sizeof(matcore_tensor_desc_v0))) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "typed tensor descriptor ABI version or size mismatch");
  }
  if (!reserved_is_zero(tensor.reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "typed tensor descriptor reserved fields must be zero");
  }
  if (tensor.data == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0, null_message);
  if (tensor.dtype != expected_dtype) {
    return status(MATCORE_STATUS_UNSUPPORTED_DTYPE_V0,
                  "typed GEMM tensor dtype does not match the entry point");
  }
  if (tensor.rank != 2) {
    return status(MATCORE_STATUS_UNSUPPORTED_RANK_V0,
                  "typed GEMM requires rank-2 tensors");
  }
  if (tensor.dims[0] <= 0 || tensor.dims[1] <= 0) {
    return status(MATCORE_STATUS_INVALID_SHAPE_V0,
                  "typed GEMM dimensions must be positive");
  }
  if (tensor.strides[1] != 1 || tensor.strides[0] != tensor.dims[1]) {
    return status(MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0,
                  "typed GEMM requires row-major contiguous tensors");
  }
  if (!valid_memory_space(tensor.memory_space) ||
      tensor.memory_space != MATCORE_MEMORY_SPACE_HOST_V0) {
    return status(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0,
                  "typed CPU GEMM accepts host tensors only");
  }
  if (!valid_mutability(tensor.mutability)) {
    return status(MATCORE_STATUS_UNSUPPORTED_MUTABILITY_V0,
                  "typed tensor descriptor has invalid mutability");
  }
  const std::size_t alignment = dtype_size_v0(expected_dtype);
  if (alignment == 0 ||
      reinterpret_cast<std::uintptr_t>(tensor.data) % alignment != 0) {
    return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                  "typed tensor data violates its natural alignment");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

bool typed_byte_range_v1(const matcore_tensor_desc_v0 &tensor,
                         std::uintptr_t *begin,
                         std::uintptr_t *end) noexcept {
  const auto rows = static_cast<std::uint64_t>(tensor.dims[0]);
  const auto columns = static_cast<std::uint64_t>(tensor.dims[1]);
  const auto element_bytes = static_cast<std::uint64_t>(dtype_size_v0(tensor.dtype));
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (element_bytes == 0 || rows > maximum / columns) return false;
  const std::uint64_t elements = rows * columns;
  if (elements > maximum / element_bytes) return false;
  const std::uint64_t bytes = elements * element_bytes;
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address)
    return false;
  *begin = address;
  *end = address + static_cast<std::uintptr_t>(bytes);
  return true;
}

struct ValidatedTypedGemmV1 {
  matcore::mdslc::runtime::CpuTypedGemmShapeV1 shape;
  const void *lhs = nullptr;
  const void *rhs = nullptr;
  void *out = nullptr;
};

matcore_status_v0 validate_typed_gemm_v1(
    const matcore_tensor_desc_v0 *out,
    const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy,
    matcore_dtype_v0 output_dtype,
    matcore_dtype_v0 input_dtype,
    ValidatedTypedGemmV1 *validated) noexcept {
  if (out == nullptr || lhs == nullptr || rhs == nullptr || validated == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "typed GEMM descriptors must be non-null");
  }
  matcore_status_v0 result = validate_cpu_policy_v0(policy);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_typed_tensor_v1(*out, output_dtype, "typed output data is null");
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_typed_tensor_v1(*lhs, input_dtype, "typed left input data is null");
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_typed_tensor_v1(*rhs, input_dtype, "typed right input data is null");
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (out->mutability != MATCORE_MUTABILITY_READ_WRITE_V0) {
    return status(MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0,
                  "typed GEMM output must be mutable");
  }
  const std::int64_t m = lhs->dims[0];
  const std::int64_t k = lhs->dims[1];
  const std::int64_t n = rhs->dims[1];
  if (rhs->dims[0] != k || out->dims[0] != m || out->dims[1] != n) {
    return status(MATCORE_STATUS_SHAPE_MISMATCH_V0,
                  "typed GEMM shapes must be lhs[M,K], rhs[K,N], out[M,N]");
  }
  std::uintptr_t out_begin = 0;
  std::uintptr_t out_end = 0;
  std::uintptr_t lhs_begin = 0;
  std::uintptr_t lhs_end = 0;
  std::uintptr_t rhs_begin = 0;
  std::uintptr_t rhs_end = 0;
  if (!typed_byte_range_v1(*out, &out_begin, &out_end) ||
      !typed_byte_range_v1(*lhs, &lhs_begin, &lhs_end) ||
      !typed_byte_range_v1(*rhs, &rhs_begin, &rhs_end)) {
    return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                  "typed tensor byte range overflows the address space");
  }
  if (overlaps(out_begin, out_end, lhs_begin, lhs_end) ||
      overlaps(out_begin, out_end, rhs_begin, rhs_end)) {
    return status(MATCORE_STATUS_ALIAS_VIOLATION_V0,
                  "typed GEMM output must not overlap either input");
  }
  validated->shape = {matcore::mdslc::runtime::kCpuNumericReferenceVersionV1,
                      m, n, k};
  validated->lhs = lhs->data;
  validated->rhs = rhs->data;
  validated->out = out->data;
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 numeric_reference_status_v1(
    matcore::mdslc::runtime::CpuNumericReferenceStatusV1 result) noexcept {
  using NumericStatus =
      matcore::mdslc::runtime::CpuNumericReferenceStatusV1;
  switch (result) {
    case NumericStatus::success:
      return status(MATCORE_STATUS_OK_V0, "ok");
    case NumericStatus::invalid_problem:
    case NumericStatus::null_pointer:
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "typed reference GEMM received an invalid argument");
    case NumericStatus::invalid_pointer_alignment:
      return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                    "typed reference GEMM alignment contract failed");
    case NumericStatus::alias_violation:
      return status(MATCORE_STATUS_ALIAS_VIOLATION_V0,
                    "typed reference GEMM buffers overlap illegally");
    case NumericStatus::arithmetic_overflow:
      return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                    "typed reference GEMM size arithmetic overflowed");
  }
  return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                "typed reference GEMM returned an unknown status");
}

inline constexpr std::uint64_t kCpuExecutionContextMagicV1 =
    UINT64_C(0x4d44534c43545831);

using PackedMicrokernelV1 = void (*)(
    const float *, const float *, std::size_t, float *, std::size_t,
    std::uint32_t, std::uint32_t, bool) noexcept;

bool run_packed_microkernel_self_test_v1(PackedMicrokernelV1 microkernel,
                                         bool runtime_usable) noexcept {
  if (microkernel == nullptr || !runtime_usable) return false;
  constexpr std::size_t rows = matcore::mdslc::runtime::kCpuPackedGemmMrV1;
  constexpr std::size_t columns = matcore::mdslc::runtime::kCpuPackedGemmNrV1;
  constexpr std::size_t depth = 3;
  alignas(64) std::array<float, rows * depth> packed_a{};
  alignas(64) std::array<float, depth * columns> packed_b{};
  alignas(64) std::array<float, rows * columns> output{};
  for (std::size_t p = 0; p < depth; ++p) {
    for (std::size_t row = 0; row < rows; ++row)
      packed_a[p * rows + row] = static_cast<float>(row + 1U);
    for (std::size_t column = 0; column < columns; ++column)
      packed_b[p * columns + column] = static_cast<float>(p + 1U);
  }
  microkernel(packed_a.data(), packed_b.data(), depth, output.data(), columns,
              static_cast<std::uint32_t>(rows),
              static_cast<std::uint32_t>(columns), false);
  for (std::size_t row = 0; row < rows; ++row) {
    const float expected = static_cast<float>(6U * (row + 1U));
    for (std::size_t column = 0; column < columns; ++column) {
      if (output[row * columns + column] != expected) return false;
    }
  }
  return true;
}

struct CpuRuntimeSelfTestsV1 {
  bool packed_avx2_f32 = false;
  bool packed_avx512_f32 = false;
};

const CpuRuntimeSelfTestsV1 &cpu_runtime_self_tests_v1() noexcept {
  static const CpuRuntimeSelfTestsV1 tests = []() noexcept {
    CpuRuntimeSelfTestsV1 result;
    result.packed_avx2_f32 = run_packed_microkernel_self_test_v1(
        matcore::mdslc::runtime::detail::
            matcore_cpu_packed_avx2_4x16_microkernel_f32_v1,
        matcore::mdslc::runtime::cpu_packed_avx2_build_available_v1() &&
            matcore::mdslc::runtime::cpu_packed_avx2_runtime_usable_v1());
    result.packed_avx512_f32 = run_packed_microkernel_self_test_v1(
        matcore::mdslc::runtime::detail::
            matcore_cpu_packed_avx512_4x16_microkernel_f32_v1,
        matcore::mdslc::runtime::cpu_packed_avx512_build_available_v1() &&
            matcore::mdslc::runtime::cpu_packed_avx512_runtime_usable_v1());
    return result;
  }();
  return tests;
}

matcore::mdslc::platform::CpuImplementationAvailabilityV2
cpu_implementation_availability_v2() noexcept {
  using matcore::mdslc::platform::CpuFeatureV2;
  using matcore::mdslc::platform::feature_bit;
  matcore::mdslc::platform::CpuImplementationAvailabilityV2 result;
#if defined(__x86_64__) || defined(_M_X64)
  result.compiled.known = matcore::mdslc::platform::kKnownCpuFeatureBitsV2;
  result.runtime_validated.known =
      matcore::mdslc::platform::kKnownCpuFeatureBitsV2;
#elif defined(__aarch64__) || defined(_M_ARM64)
  result.compiled.known = feature_bit(CpuFeatureV2::portable_scalar_f32);
  result.runtime_validated.known =
      feature_bit(CpuFeatureV2::portable_scalar_f32);
#endif
  const std::uint64_t portable =
      feature_bit(CpuFeatureV2::portable_scalar_f32);
  result.compiled.available |= portable;
  result.runtime_validated.available |= portable;

  const bool avx2_compiled =
      matcore::mdslc::runtime::cpu_packed_avx2_build_available_v1();
  const bool avx512_compiled =
      matcore::mdslc::runtime::cpu_packed_avx512_build_available_v1();
  if (avx2_compiled) {
    result.compiled.available |= feature_bit(CpuFeatureV2::avx2) |
                                 feature_bit(CpuFeatureV2::fma);
  }
  if (avx512_compiled) {
    result.compiled.available |= feature_bit(CpuFeatureV2::avx512f) |
                                 feature_bit(CpuFeatureV2::fma);
  }
  const CpuRuntimeSelfTestsV1 &tests = cpu_runtime_self_tests_v1();
  if (tests.packed_avx2_f32) {
    result.runtime_validated.available |= feature_bit(CpuFeatureV2::avx2) |
                                          feature_bit(CpuFeatureV2::fma);
  }
  if (tests.packed_avx512_f32) {
    result.runtime_validated.available |=
        feature_bit(CpuFeatureV2::avx512f) |
        feature_bit(CpuFeatureV2::fma);
  }
  return result;
}

matcore::mdslc::platform::CpuCapabilitiesV2
discover_runtime_cpu_capabilities_v2() noexcept {
  // Freeze one fail-closed process record. In particular, Linux AMX state is
  // thread-scoped; silently changing a public capability answer when a later
  // thread gains permission would make planning nondeterministic. A future
  // AMX execution context can own an explicit permission transition/version.
  static const matcore::mdslc::platform::CpuCapabilitiesV2 capabilities =
      matcore::mdslc::platform::discover_cpu_capabilities_v2(
          cpu_implementation_availability_v2());
  return capabilities;
}

matcore_cpu_architecture_v1 to_c_platform_architecture(
    matcore::mdslc::platform::ArchitectureKindV1 architecture) noexcept {
  using Architecture = matcore::mdslc::platform::ArchitectureKindV1;
  switch (architecture) {
    case Architecture::x86_64:
      return MATCORE_CPU_ARCHITECTURE_X86_64_V1;
    case Architecture::aarch64:
      return MATCORE_CPU_ARCHITECTURE_AARCH64_V1;
    case Architecture::unknown:
      return MATCORE_CPU_ARCHITECTURE_UNKNOWN_V1;
  }
  return MATCORE_CPU_ARCHITECTURE_UNKNOWN_V1;
}

bool capability_detection_complete_v2(
    const matcore::mdslc::platform::CpuCapabilitiesV2 &capabilities) noexcept {
  using matcore::mdslc::platform::domain_complete_v2;
  return matcore::mdslc::platform::validate_cpu_capabilities_v2(capabilities)
             .valid &&
         capabilities.architecture !=
             matcore::mdslc::platform::ArchitectureKindV1::unknown &&
         domain_complete_v2(capabilities.hardware, capabilities.architecture) &&
         domain_complete_v2(capabilities.os_enabled,
                            capabilities.architecture) &&
         domain_complete_v2(capabilities.compiler, capabilities.architecture);
}

matcore_cpu_capabilities_v2 to_c_capabilities_v2(
    const matcore::mdslc::platform::CpuCapabilitiesV2 &source) noexcept {
  matcore_cpu_capabilities_v2 output{};
  output.abi_version = MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2;
  output.struct_size = sizeof(output);
  output.architecture = to_c_platform_architecture(source.architecture);
  output.detection_complete = capability_detection_complete_v2(source) ? 1U : 0U;
  output.hardware_known_features = source.hardware.known;
  output.hardware_available_features = source.hardware.available;
  output.os_known_features = source.os_enabled.known;
  output.os_available_features = source.os_enabled.available;
  output.compiler_known_features = source.compiler.known;
  output.compiler_available_features = source.compiler.available;
  output.implementation_known_features = source.implementation.known;
  output.implementation_available_features = source.implementation.available;
  output.runtime_validation_known_features = source.runtime_validation.known;
  output.runtime_validated_features = source.runtime_validation.available;
  output.os_xstate_mask = source.os_xstate_mask;
  output.usable_vector_bits = source.usable_vector_bits;
  output.os_xstate_mask_known = source.os_xstate_mask_known ? 1U : 0U;
  output.amx_permission_known = source.amx_permission_known ? 1U : 0U;
  output.amx_permission_granted = source.amx_permission_granted ? 1U : 0U;
  return output;
}

bool zero_capabilities_v2(const matcore_cpu_capabilities_v2 &value) noexcept {
  return value.abi_version == 0 && value.struct_size == 0 &&
         value.architecture == 0 && value.detection_complete == 0 &&
         value.hardware_known_features == 0 &&
         value.hardware_available_features == 0 &&
         value.os_known_features == 0 && value.os_available_features == 0 &&
         value.compiler_known_features == 0 &&
         value.compiler_available_features == 0 &&
         value.implementation_known_features == 0 &&
         value.implementation_available_features == 0 &&
         value.runtime_validation_known_features == 0 &&
         value.runtime_validated_features == 0 && value.os_xstate_mask == 0 &&
         value.usable_vector_bits == 0 && value.os_xstate_mask_known == 0 &&
         value.amx_permission_known == 0 &&
         value.amx_permission_granted == 0 && value.reserved0 == 0 &&
         reserved_is_zero(value.reserved);
}

matcore_status_v0 validate_empty_capabilities_v2(
    const matcore_cpu_capabilities_v2 *capabilities) noexcept {
  if (capabilities == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU capability v2 output must be non-null");
  if (capabilities->abi_version !=
          MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2 ||
      capabilities->struct_size != sizeof(matcore_cpu_capabilities_v2)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU capability v2 ABI version or size mismatch");
  }
  matcore_cpu_capabilities_v2 payload = *capabilities;
  payload.abi_version = 0;
  payload.struct_size = 0;
  if (!zero_capabilities_v2(payload)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU capability v2 output fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

bool valid_execution_context_v1(
    const matcore_cpu_execution_context_v1 *context) noexcept {
  return context != nullptr && context->magic == kCpuExecutionContextMagicV1 &&
         context->workers != nullptr;
}

matcore_status_v0 validate_context_options_v1(
    const matcore_cpu_execution_context_options_v1 *options) noexcept {
  if (options == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context options must be non-null");
  if (options->abi_version !=
          MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1 ||
      options->struct_size !=
          sizeof(matcore_cpu_execution_context_options_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU execution-context option ABI version or size mismatch");
  }
  if (options->requested_threads == 0)
    return status(MATCORE_STATUS_INVALID_THREAD_COUNT_V0,
                  "CPU execution-context thread count must be positive");
  if (options->affinity_policy != MATCORE_CPU_AFFINITY_NONE_V1 &&
      options->affinity_policy != MATCORE_CPU_AFFINITY_COMPACT_V1 &&
      options->affinity_policy != MATCORE_CPU_AFFINITY_SCATTER_V1) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context affinity policy is unsupported");
  }
  if (options->numa_policy != MATCORE_CPU_NUMA_SINGLE_NODE_V1 &&
      options->numa_policy != MATCORE_CPU_NUMA_LOCAL_FIRST_V1) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context NUMA policy is unsupported");
  }
  if (options->smt_policy != MATCORE_CPU_SMT_PHYSICAL_CORES_ONLY_V1 &&
      options->smt_policy != MATCORE_CPU_SMT_ALLOW_V1) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context SMT policy is unsupported");
  }
  if (options->flags != 0 ||
      !reserved_is_zero(options->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context flags and reserved fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 validate_empty_context_report_v1(
    const matcore_cpu_execution_context_report_v1 *report) noexcept {
  if (report == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context report must be non-null");
  if (report->abi_version !=
          MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1 ||
      report->struct_size != sizeof(matcore_cpu_execution_context_report_v1)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU execution-context report ABI version or size mismatch");
  }
  if (report->requested_threads != 0 || report->actual_worker_count != 0 ||
      report->persistent_worker_count != 0 || report->affinity_policy != 0 ||
      report->numa_policy != 0 || report->smt_policy != 0 ||
      report->topology_discovery_complete != 0 ||
      report->logical_cpu_count != 0 || report->physical_core_count != 0 ||
      report->socket_count != 0 || report->numa_node_count != 0 ||
      report->available_logical_cpu_count != 0 ||
      report->available_physical_core_count != 0 ||
      report->process_affinity_discovery_complete != 0 ||
      report->process_affinity_platform_error != 0 ||
      report->worker_affinity_status != 0 ||
      report->affinity_requested_workers != 0 ||
      report->affinity_applied_workers != 0 ||
      report->affinity_first_failed_worker != 0 ||
      report->affinity_first_failed_cpu != 0 ||
      report->affinity_platform_error != 0 ||
      report->affinity_complete != 0 ||
      report->numa_memory_placement_applied != 0 ||
      report->worker_affinity_induced_by_smt_policy != 0 ||
      report->worker_affinity_induced_by_numa_policy != 0 ||
      report->creator_logical_cpu != 0 || report->selected_numa_node != 0 ||
      report->execution_generation != 0 ||
      !reserved_is_zero(report->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context report output fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_cpu_affinity_status_v1 to_c_affinity_status_v1(
    matcore::mdslc::runtime::CpuWorkerAffinityStatusV1 source) noexcept {
  using Status = matcore::mdslc::runtime::CpuWorkerAffinityStatusV1;
  switch (source) {
    case Status::not_requested:
      return MATCORE_CPU_AFFINITY_STATUS_NOT_REQUESTED_V1;
    case Status::complete:
      return MATCORE_CPU_AFFINITY_STATUS_COMPLETE_V1;
    case Status::invalid_configuration:
      return MATCORE_CPU_AFFINITY_STATUS_INVALID_CONFIGURATION_V1;
    case Status::unavailable:
      return MATCORE_CPU_AFFINITY_STATUS_UNAVAILABLE_V1;
    case Status::application_failed:
      return MATCORE_CPU_AFFINITY_STATUS_APPLICATION_FAILED_V1;
    case Status::partially_applied:
      return MATCORE_CPU_AFFINITY_STATUS_PARTIALLY_APPLIED_V1;
  }
  return MATCORE_CPU_AFFINITY_STATUS_APPLICATION_FAILED_V1;
}

void populate_context_report_v1(
    const matcore_cpu_execution_context_v1 &context,
    matcore_cpu_execution_context_report_v1 *report) noexcept {
  const auto info = context.workers->info();
  matcore_cpu_execution_context_report_v1 output{};
  output.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  output.struct_size = sizeof(output);
  output.requested_threads = info.requested_threads;
  output.actual_worker_count = info.actual_worker_count;
  output.persistent_worker_count = info.actual_worker_count;
  output.affinity_policy = context.affinity_policy;
  output.numa_policy = context.numa_policy;
  output.smt_policy = context.smt_policy;
  output.topology_discovery_complete =
      context.topology.discovery_complete ? 1U : 0U;
  output.logical_cpu_count =
      matcore::mdslc::platform::logical_cpu_count_v1(context.topology);
  output.physical_core_count =
      matcore::mdslc::platform::physical_core_count_v1(context.topology);
  output.socket_count =
      matcore::mdslc::platform::socket_count_v1(context.topology);
  output.numa_node_count =
      matcore::mdslc::platform::numa_node_count_v1(context.topology);
  output.available_logical_cpu_count =
      matcore::mdslc::platform::logical_cpu_count_v1(
          context.available_topology);
  output.available_physical_core_count =
      matcore::mdslc::platform::physical_core_count_v1(
          context.available_topology);
  output.process_affinity_discovery_complete =
      context.process_affinity_discovery_complete ? 1U : 0U;
  output.process_affinity_platform_error =
      context.process_affinity_platform_error;
  output.worker_affinity_status =
      to_c_affinity_status_v1(info.affinity.status);
  output.affinity_requested_workers = info.affinity.requested_workers;
  output.affinity_applied_workers = info.affinity.applied_workers;
  output.affinity_first_failed_worker = info.affinity.first_failed_worker;
  output.affinity_first_failed_cpu = info.affinity.first_failed_cpu;
  output.affinity_platform_error = info.affinity.platform_error;
  output.affinity_complete = info.affinity.complete ? 1U : 0U;
  output.numa_memory_placement_applied =
      info.affinity.numa_memory_placement_applied ? 1U : 0U;
  output.worker_affinity_induced_by_smt_policy =
      context.worker_affinity_induced_by_smt_policy ? 1U : 0U;
  output.worker_affinity_induced_by_numa_policy =
      context.worker_affinity_induced_by_numa_policy ? 1U : 0U;
  output.creator_logical_cpu = context.creator_logical_cpu;
  output.selected_numa_node = context.selected_numa_node;
  output.execution_generation =
      info.completed_submissions >= context.validation_submission_baseline
          ? info.completed_submissions - context.validation_submission_baseline
          : 0;
  *report = output;
}

void populate_failed_affinity_report_v1(
    const matcore_cpu_execution_context_options_v1 &options,
    const matcore::mdslc::platform::CpuTopologyV1 &topology,
    const matcore::mdslc::platform::CpuTopologyV1 &available_topology,
    const matcore::mdslc::platform::ThreadAffinityInventoryV1 &inventory,
    const matcore::mdslc::runtime::CpuWorkerAffinityReportV1 &affinity,
    bool induced_by_smt, bool induced_by_numa,
    std::uint32_t creator_logical_cpu, std::uint32_t selected_numa_node,
    matcore_cpu_execution_context_report_v1 *report) noexcept {
  matcore_cpu_execution_context_report_v1 output{};
  output.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  output.struct_size = sizeof(output);
  output.requested_threads = options.requested_threads;
  output.affinity_policy = options.affinity_policy;
  output.numa_policy = options.numa_policy;
  output.smt_policy = options.smt_policy;
  output.topology_discovery_complete = topology.discovery_complete ? 1U : 0U;
  output.logical_cpu_count =
      matcore::mdslc::platform::logical_cpu_count_v1(topology);
  output.physical_core_count =
      matcore::mdslc::platform::physical_core_count_v1(topology);
  output.socket_count = matcore::mdslc::platform::socket_count_v1(topology);
  output.numa_node_count =
      matcore::mdslc::platform::numa_node_count_v1(topology);
  output.available_logical_cpu_count =
      matcore::mdslc::platform::logical_cpu_count_v1(available_topology);
  output.available_physical_core_count =
      matcore::mdslc::platform::physical_core_count_v1(available_topology);
  output.process_affinity_discovery_complete =
      inventory.discovery_complete ? 1U : 0U;
  output.process_affinity_platform_error = inventory.platform_error;
  output.worker_affinity_status = to_c_affinity_status_v1(affinity.status);
  output.affinity_requested_workers = affinity.requested_workers;
  output.affinity_applied_workers = affinity.applied_workers;
  output.affinity_first_failed_worker = affinity.first_failed_worker;
  output.affinity_first_failed_cpu = affinity.first_failed_cpu;
  output.affinity_platform_error = affinity.platform_error;
  output.affinity_complete = affinity.complete ? 1U : 0U;
  output.numa_memory_placement_applied = 0;
  output.worker_affinity_induced_by_smt_policy = induced_by_smt ? 1U : 0U;
  output.worker_affinity_induced_by_numa_policy = induced_by_numa ? 1U : 0U;
  output.creator_logical_cpu = creator_logical_cpu;
  output.selected_numa_node = selected_numa_node;
  *report = output;
}

matcore::mdslc::platform::CpuAffinityPolicyV1 placement_affinity_v1(
    matcore_cpu_affinity_policy_v1 affinity,
    matcore_cpu_numa_policy_v1 numa) noexcept {
  if (numa == MATCORE_CPU_NUMA_LOCAL_FIRST_V1 &&
      affinity == MATCORE_CPU_AFFINITY_NONE_V1) {
    return matcore::mdslc::platform::CpuAffinityPolicyV1::local_first;
  }
  return affinity == MATCORE_CPU_AFFINITY_SCATTER_V1
             ? matcore::mdslc::platform::CpuAffinityPolicyV1::scatter
             : matcore::mdslc::platform::CpuAffinityPolicyV1::compact;
}

struct ContextPlacementV1 {
  matcore::mdslc::platform::CpuTopologyV1 available_topology;
  matcore::mdslc::platform::CpuPlacementPlanV1 plan;
  std::uint32_t worker_count = 0;
  std::vector<std::uint32_t> worker_cpu_ids;
  bool affinity_induced_by_smt_policy = false;
  bool affinity_induced_by_numa_policy = false;
  std::uint32_t creator_logical_cpu =
      matcore::mdslc::platform::kUnknownLogicalCpuV1;
  std::uint32_t selected_numa_node =
      matcore::mdslc::platform::kUnknownTopologyIdV1;
};

bool logical_cpu_numa_node_v1(
    const matcore::mdslc::platform::CpuTopologyV1 &topology,
    std::uint32_t logical_cpu, std::uint32_t *node) noexcept {
  if (node == nullptr) return false;
  const auto processor = std::find_if(
      topology.logical_processors.begin(), topology.logical_processors.end(),
      [logical_cpu](const auto &entry) {
        return entry.logical_cpu == logical_cpu && entry.online;
      });
  if (processor == topology.logical_processors.end() ||
      processor->numa_node_id ==
          matcore::mdslc::platform::kUnknownTopologyIdV1) {
    return false;
  }
  *node = processor->numa_node_id;
  return true;
}

matcore_status_v0 select_context_placement_v1(
    const matcore_cpu_execution_context_options_v1 &options,
    const matcore::mdslc::platform::CpuTopologyV1 &topology,
    const matcore::mdslc::platform::ThreadAffinityInventoryV1 &inventory,
    const matcore::mdslc::platform::CurrentLogicalCpuV1 &current_cpu,
    ContextPlacementV1 *selection) {
  if (selection == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context placement output must be non-null");
  const auto topology_validation =
      matcore::mdslc::platform::validate_cpu_topology_v1(topology);
  if (!topology_validation.valid || !topology.discovery_complete ||
      topology.numa_nodes.empty()) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "complete CPU topology is required to create a context");
  }
  if (!inventory.backend_available || !inventory.discovery_complete ||
      inventory.platform_error != 0 ||
      inventory.allowed_logical_cpus.empty()) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "complete process CPU-affinity discovery is required to create a context");
  }
  auto restricted = matcore::mdslc::platform::restrict_cpu_topology_v1(
      topology, inventory.allowed_logical_cpus);
  if (!restricted) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "process CPU-affinity mask cannot be projected onto the discovered topology");
  }

  const std::uint32_t available =
      options.smt_policy == MATCORE_CPU_SMT_ALLOW_V1
          ? matcore::mdslc::platform::logical_cpu_count_v1(
                restricted.topology)
          : matcore::mdslc::platform::physical_core_count_v1(
                restricted.topology);
  const std::uint32_t candidate =
      std::min(options.requested_threads, available);
  if (candidate == 0)
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "process CPU-affinity mask exposes no legal worker capacity");

  const bool explicit_worker_affinity =
      options.affinity_policy != MATCORE_CPU_AFFINITY_NONE_V1 ||
      options.numa_policy == MATCORE_CPU_NUMA_LOCAL_FIRST_V1 ||
      options.smt_policy == MATCORE_CPU_SMT_PHYSICAL_CORES_ONLY_V1;
  if (!explicit_worker_affinity && restricted.topology.numa_nodes.size() > 1) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "a multi-node process mask requires explicit worker affinity; NUMA page placement is not implemented");
  }

  matcore::mdslc::platform::CpuPlacementRequestV1 request;
  request.affinity =
      placement_affinity_v1(options.affinity_policy, options.numa_policy);
  request.smt =
      options.smt_policy == MATCORE_CPU_SMT_ALLOW_V1
          ? matcore::mdslc::platform::CpuSmtPolicyV1::allow_smt
          : matcore::mdslc::platform::CpuSmtPolicyV1::physical_cores_only;
  std::uint32_t creator_node = restricted.topology.numa_nodes.front().node_id;
  if (options.numa_policy == MATCORE_CPU_NUMA_LOCAL_FIRST_V1) {
    if (!current_cpu.backend_available || !current_cpu.discovery_complete ||
        current_cpu.platform_error != 0 ||
        current_cpu.logical_cpu ==
            matcore::mdslc::platform::kUnknownLogicalCpuV1) {
      return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                    "local-first placement requires authenticated current-CPU discovery");
    }
    if (!logical_cpu_numa_node_v1(restricted.topology,
                                  current_cpu.logical_cpu, &creator_node)) {
      return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                    "local-first placement cannot authenticate the creator CPU inside the process-affinity topology");
    }
  }
  request.preferred_numa_node = creator_node;
  request.allow_cross_numa = false;
  request.requested_workers = candidate;
  auto placement = matcore::mdslc::platform::plan_cpu_placement_v1(
      restricted.topology, request);
  if (placement.status !=
      matcore::mdslc::platform::CpuPlacementStatusV1::selected) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "requested worker policy has no legal process-mask-local single-node placement");
  }
  ContextPlacementV1 output;
  output.available_topology = std::move(restricted.topology);
  output.worker_count = placement.actual_workers;
  if (explicit_worker_affinity) output.worker_cpu_ids = placement.logical_cpus;
  output.affinity_induced_by_smt_policy =
      options.affinity_policy == MATCORE_CPU_AFFINITY_NONE_V1 &&
      options.smt_policy == MATCORE_CPU_SMT_PHYSICAL_CORES_ONLY_V1;
  output.affinity_induced_by_numa_policy =
      options.affinity_policy == MATCORE_CPU_AFFINITY_NONE_V1 &&
      options.numa_policy == MATCORE_CPU_NUMA_LOCAL_FIRST_V1;
  output.creator_logical_cpu =
      current_cpu.discovery_complete
          ? current_cpu.logical_cpu
          : matcore::mdslc::platform::kUnknownLogicalCpuV1;
  output.selected_numa_node = creator_node;
  output.plan = std::move(placement);
  *selection = std::move(output);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore::mdslc::planner::CpuPlannerPlacementEvidenceV1
planner_placement_evidence_v1(
    const matcore::mdslc::platform::CpuTopologyV1 &available_topology,
    const matcore::mdslc::platform::CpuPlacementPlanV1 &placement,
    matcore_cpu_numa_policy_v1 numa_policy, bool affinity_requested,
    const matcore::mdslc::runtime::CpuWorkerAffinityReportV1
        &affinity_report) noexcept {
  namespace platform = matcore::mdslc::platform;
  namespace planner = matcore::mdslc::planner;
  planner::CpuPlannerPlacementEvidenceV1 evidence;
  const auto topology_validation =
      platform::validate_cpu_topology_v1(available_topology);
  if (!topology_validation.valid || !available_topology.discovery_complete ||
      placement.status != platform::CpuPlacementStatusV1::selected ||
      placement.numa_nodes.empty() ||
      placement.numa_nodes.size() > evidence.selected_numa_nodes.size()) {
    return evidence;
  }

  evidence.affinity_requested = affinity_requested;
  evidence.affinity_applied =
      affinity_requested && affinity_report.complete &&
      affinity_report.status ==
          matcore::mdslc::runtime::CpuWorkerAffinityStatusV1::complete;
  evidence.affinity = placement.affinity;
  evidence.numa = numa_policy == MATCORE_CPU_NUMA_LOCAL_FIRST_V1
                      ? planner::CpuPlannerNumaPolicyV1::local_first
                      : planner::CpuPlannerNumaPolicyV1::single_node;
  evidence.selected_numa_node_count =
      static_cast<std::uint32_t>(placement.numa_nodes.size());
  for (std::size_t index = 0; index < placement.numa_nodes.size(); ++index)
    evidence.selected_numa_nodes[index] = placement.numa_nodes[index];
  evidence.crosses_numa_nodes = placement.crosses_numa_nodes;
  evidence.caller_first_touch_required =
      placement.caller_first_touch_required;

  const std::uint32_t local_node = evidence.selected_numa_nodes.front();
  const auto node = std::find_if(
      available_topology.numa_nodes.begin(),
      available_topology.numa_nodes.end(),
      [local_node](const platform::CpuNumaNodeV1 &candidate) {
        return candidate.node_id == local_node;
      });
  if (node == available_topology.numa_nodes.end()) return {};
  const auto local = platform::restrict_cpu_topology_v1(
      available_topology, node->logical_cpus);
  if (!local) return {};
  evidence.local_logical_processor_capacity =
      platform::logical_cpu_count_v1(local.topology);
  evidence.local_physical_core_capacity =
      platform::physical_core_count_v1(local.topology);
  evidence.evidence_complete =
      evidence.local_logical_processor_capacity != 0 &&
      evidence.local_physical_core_capacity != 0 &&
      (!affinity_requested || evidence.affinity_applied);
  return evidence;
}

matcore_status_v0 validate_execution_options_v2(
    const matcore_cpu_gemm_execution_options_v2 *options,
    const matcore_cpu_execution_context_v1 &context,
    matcore::mdslc::planner::CpuGemmRequestV3 *request) noexcept {
  if (options == nullptr || request == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM v2 execution options must be non-null");
  if (options->abi_version !=
          MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2 ||
      options->struct_size != sizeof(matcore_cpu_gemm_execution_options_v2)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM v2 execution option ABI version or size mismatch");
  }
  if (options->requested_threads == 0)
    return status(MATCORE_STATUS_INVALID_THREAD_COUNT_V0,
                  "CPU GEMM v2 requested thread count must be positive");
  if (options->affinity_policy != context.affinity_policy ||
      options->numa_policy != context.numa_policy ||
      options->smt_policy != context.smt_policy) {
    return status(MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
                  "CPU GEMM affinity, NUMA, or SMT policy does not match the fixed execution context");
  }
  if (options->reserved0 != 0 || options->flags != 0 ||
      !reserved_is_zero(options->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM v2 flags and reserved fields must be zero");
  }
  using Request = matcore::mdslc::planner::CpuGemmRequestV3;
  switch (options->request) {
    case MATCORE_CPU_GEMM_REQUEST_AUTOMATIC_V3:
      *request = Request::automatic;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V3:
      *request = Request::force_reference;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_TILED_V3:
      *request = Request::force_tiled;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_COMPILER_VECTORIZED_V3:
      *request = Request::force_compiler_vectorized;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V3:
      *request = Request::force_external_openblas;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V3:
      *request = Request::force_native_packed_avx2_fma;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX512_FMA_V3:
      *request = Request::force_native_packed_avx512_fma;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX2_FMA_V3:
      *request = Request::force_native_parallel_avx2_fma;
      break;
    case MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX512_FMA_V3:
      *request = Request::force_native_parallel_avx512_fma;
      break;
    default:
      return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                    "CPU GEMM v2 requested an unknown variant");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

bool empty_candidate_report_v3(
    const matcore_cpu_gemm_candidate_v3 &candidate) noexcept {
  return candidate.stable_id == nullptr && candidate.legal == 0 &&
         candidate.deterministic_priority == 0 &&
         candidate.actual_threads == 0 &&
         candidate.required_workspace_alignment == 0 &&
         candidate.estimated_cost == 0 &&
         candidate.required_workspace_bytes == 0 &&
         candidate.shared_workspace_bytes == 0 &&
         candidate.per_worker_workspace_bytes == 0 &&
         candidate.required_hardware_features == 0 &&
         candidate.required_os_features == 0 &&
         candidate.required_compiler_features == 0 &&
         candidate.required_implementation_features == 0 &&
         candidate.runtime_validated == 0 && candidate.reserved0 == 0 &&
         candidate.reason == nullptr && reserved_is_zero(candidate.reserved);
}

matcore_status_v0 validate_empty_report_v3(
    const matcore_cpu_gemm_plan_report_v3 *report) noexcept {
  if (report == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan v3 report must be non-null");
  if (report->abi_version != MATCORE_RUNTIME_PLAN_ABI_VERSION_V3 ||
      report->struct_size != sizeof(matcore_cpu_gemm_plan_report_v3)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU plan v3 report ABI version or size mismatch");
  }
  if (report->planner_version != 0 || report->request != 0 ||
      report->plan_status != 0 || report->architecture != 0 ||
      report->capability_detection_complete != 0 ||
      report->usable_vector_bits != 0 ||
      !zero_capabilities_v2(report->capabilities) || report->m != 0 ||
      report->n != 0 || report->k != 0 ||
      report->minimum_alignment_bytes != 0 ||
      report->requested_threads != 0 || report->candidate_count != 0 ||
      report->selected_actual_threads != 0 ||
      report->selected_workspace_bytes != 0 ||
      report->selected_workspace_alignment != 0 ||
      report->selected_affinity_policy != 0 ||
      report->selected_numa_policy != 0 ||
      report->selected_smt_policy != 0 ||
      report->numa_memory_placement_applied != 0 ||
      report->topology_discovery_complete != 0 ||
      report->logical_cpu_count != 0 || report->physical_core_count != 0 ||
      report->socket_count != 0 || report->numa_node_count != 0 ||
      report->available_logical_cpu_count != 0 ||
      report->available_physical_core_count != 0 ||
      report->worker_affinity_induced_by_smt_policy != 0 ||
      report->worker_affinity_induced_by_numa_policy != 0 ||
      report->creator_logical_cpu != 0 || report->selected_numa_node != 0 ||
      report->selected_stable_id != nullptr ||
      report->selection_reason != nullptr ||
      report->external_provider != nullptr ||
      report->external_provider_version != nullptr ||
      report->external_provider_config != nullptr ||
      !reserved_is_zero(report->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan v3 report output fields must be zero");
  }
  for (const auto &candidate : report->candidates) {
    if (!empty_candidate_report_v3(candidate)) {
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "CPU plan v3 candidate output fields must be zero");
    }
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 validate_empty_workspace_requirements_v2(
    const matcore_gemm_workspace_requirements_v2 *requirements) noexcept {
  if (requirements == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM v2 workspace requirements must be non-null");
  if (requirements->abi_version !=
          MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2 ||
      requirements->struct_size !=
          sizeof(matcore_gemm_workspace_requirements_v2)) {
    return status(MATCORE_STATUS_ABI_MISMATCH_V0,
                  "CPU GEMM v2 workspace ABI version or size mismatch");
  }
  if (requirements->workspace_bytes != 0 ||
      requirements->workspace_alignment != 0 ||
      requirements->actual_threads != 0 ||
      requirements->per_worker_workspace_bytes != 0 ||
      requirements->shared_workspace_bytes != 0 ||
      requirements->selected_stable_id != nullptr ||
      requirements->affinity_policy != 0 || requirements->numa_policy != 0 ||
      requirements->smt_policy != 0 || requirements->reserved0 != 0 ||
      !reserved_is_zero(requirements->reserved)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM v2 workspace output fields must be zero");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

void populate_report_v3(
    const matcore::mdslc::planner::CpuGemmPlanV3 &plan,
    const matcore::mdslc::platform::CpuCapabilitiesV2 &capabilities,
    const matcore::mdslc::platform::CpuTopologyV1 &topology,
    const matcore::mdslc::platform::CpuTopologyV1 &available_topology,
    matcore_cpu_affinity_policy_v1 affinity,
    matcore_cpu_numa_policy_v1 numa,
    matcore_cpu_smt_policy_v1 smt,
    bool affinity_induced_by_smt, bool affinity_induced_by_numa,
    std::uint32_t creator_logical_cpu, std::uint32_t selected_numa_node,
    const matcore::mdslc::runtime::OpenBlasProviderInfoV1 &provider,
    matcore_cpu_gemm_plan_report_v3 *report) noexcept {
  matcore_cpu_gemm_plan_report_v3 output{};
  output.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V3;
  output.struct_size = sizeof(output);
  output.planner_version = plan.planner_version;
  output.request = static_cast<matcore_cpu_gemm_request_v3>(plan.request);
  output.plan_status = to_c_plan_status(plan.status);
  output.architecture = to_c_platform_architecture(capabilities.architecture);
  output.capability_detection_complete =
      capability_detection_complete_v2(capabilities) ? 1U : 0U;
  output.usable_vector_bits = capabilities.usable_vector_bits;
  output.capabilities = to_c_capabilities_v2(capabilities);
  output.m = plan.problem.m;
  output.n = plan.problem.n;
  output.k = plan.problem.k;
  output.minimum_alignment_bytes = plan.problem.minimum_alignment_bytes;
  output.requested_threads = plan.thread_policy.requested_threads;
  output.candidate_count = static_cast<std::uint32_t>(plan.candidates.size());
  output.selected_affinity_policy = affinity;
  output.selected_numa_policy = numa;
  output.selected_smt_policy = smt;
  // v1 applies only scheduler affinity. It performs no NUMA allocation,
  // binding, migration, first-touch, or interleaving operation.
  output.numa_memory_placement_applied = 0;
  output.topology_discovery_complete = topology.discovery_complete ? 1U : 0U;
  output.logical_cpu_count =
      matcore::mdslc::platform::logical_cpu_count_v1(topology);
  output.physical_core_count =
      matcore::mdslc::platform::physical_core_count_v1(topology);
  output.socket_count = matcore::mdslc::platform::socket_count_v1(topology);
  output.numa_node_count =
      matcore::mdslc::platform::numa_node_count_v1(topology);
  output.available_logical_cpu_count =
      matcore::mdslc::platform::logical_cpu_count_v1(available_topology);
  output.available_physical_core_count =
      matcore::mdslc::platform::physical_core_count_v1(available_topology);
  output.worker_affinity_induced_by_smt_policy =
      affinity_induced_by_smt ? 1U : 0U;
  output.worker_affinity_induced_by_numa_policy =
      affinity_induced_by_numa ? 1U : 0U;
  output.creator_logical_cpu = creator_logical_cpu;
  output.selected_numa_node = selected_numa_node;
  for (std::size_t index = 0; index < plan.candidates.size(); ++index) {
    const auto &source = plan.candidates[index];
    auto &destination = output.candidates[index];
    destination.stable_id = source.stable_id.data();
    destination.legal = source.legal ? 1U : 0U;
    destination.deterministic_priority = source.deterministic_priority;
    destination.actual_threads = source.actual_threads;
    destination.required_workspace_alignment =
        source.required_workspace_alignment;
    destination.estimated_cost = source.estimated_cost;
    destination.required_workspace_bytes = source.required_workspace_bytes;
    destination.shared_workspace_bytes = source.shared_workspace_bytes;
    destination.per_worker_workspace_bytes =
        source.per_worker_workspace_bytes;
    destination.required_hardware_features =
        source.required_hardware_features;
    destination.required_os_features = source.required_os_features;
    destination.required_compiler_features =
        source.required_compiler_features;
    destination.required_implementation_features =
        source.required_implementation_features;
    destination.runtime_validated = source.runtime_validated ? 1U : 0U;
    destination.reason = source.reason.data();
  }
  output.selected_stable_id =
      plan.selected_id.empty() ? nullptr : plan.selected_id.data();
  output.selection_reason = plan.selection_reason.empty()
                                ? nullptr
                                : plan.selection_reason.data();
  output.external_provider = provider.linked ? "OpenBLAS" : "unavailable";
  output.external_provider_version = provider.package_version;
  output.external_provider_config = provider.runtime_config;
  if (plan.status == matcore::mdslc::planner::CpuPlanStatusV1::selected) {
    const std::size_t selected = static_cast<std::size_t>(plan.selected_variant);
    if (selected < plan.candidates.size()) {
      const auto &candidate = plan.candidates[selected];
      output.selected_actual_threads = candidate.actual_threads;
      output.selected_workspace_bytes = candidate.required_workspace_bytes;
      output.selected_workspace_alignment =
          candidate.required_workspace_alignment;
    }
  }
  *report = output;
}

matcore_status_v0 unavailable_plan_status_v3(
    const matcore::mdslc::planner::CpuGemmPlanV3 &plan) noexcept {
  if (plan.status == matcore::mdslc::planner::CpuPlanStatusV1::invalid_problem)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM planner v3 rejected the problem");
  if (plan.status ==
      matcore::mdslc::planner::CpuPlanStatusV1::invalid_capabilities) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  plan.selection_reason.empty()
                      ? "CPU GEMM planner v3 rejected capability/topology state"
                      : plan.selection_reason.data());
  }
  if (plan.request != matcore::mdslc::planner::CpuGemmRequestV3::automatic) {
    const std::size_t candidate_index =
        static_cast<std::size_t>(plan.request) - 1U;
    if (candidate_index < plan.candidates.size() &&
        !plan.candidates[candidate_index].reason.empty()) {
      return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                    plan.candidates[candidate_index].reason.data());
    }
  }
  return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                "CPU GEMM planner v3 found no legal requested implementation");
}

struct AdvancedGemmPlanV3 {
  matcore::mdslc::planner::CpuGemmPlanV3 plan;
  matcore::mdslc::planner::CpuGemmImplementationResourcesV2 resources;
};

AdvancedGemmPlanV3 make_advanced_gemm_plan_v3(
    matcore_cpu_execution_context_v1 &context,
    const ValidatedGemmV0 &validated,
    const matcore_cpu_gemm_execution_options_v2 &options,
    matcore::mdslc::planner::CpuGemmRequestV3 request) noexcept {
  auto baseline =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, options.requested_threads);
  const CpuRuntimeSelfTestsV1 &self_tests = cpu_runtime_self_tests_v1();
  baseline.native_packed_avx2_fma_compiled =
      baseline.native_packed_avx2_fma_compiled &&
      self_tests.packed_avx2_f32;
  auto resources =
      matcore::mdslc::runtime::augment_cpu_gemm_implementation_resources_v2(
          validated.problem, baseline, context.workers.get(),
          context.validation_evidence);
  resources.native_parallel_avx2_fma_compiled =
      resources.native_parallel_avx2_fma_compiled &&
      self_tests.packed_avx2_f32;
  matcore::mdslc::planner::CpuThreadPolicyV1 thread_policy;
  thread_policy.requested_threads = options.requested_threads;
  thread_policy.maximum_threads = context.workers->info().actual_worker_count;
  thread_policy.allow_smt =
      context.smt_policy == MATCORE_CPU_SMT_ALLOW_V1;
  thread_policy.external_provider_parallelism_active = false;
  thread_policy.worker_affinity_active =
      context.affinity_policy != MATCORE_CPU_AFFINITY_NONE_V1 ||
      context.worker_affinity_induced_by_smt_policy ||
      context.worker_affinity_induced_by_numa_policy;
  auto plan = matcore::mdslc::planner::plan_cpu_gemm_v3(
      validated.problem, context.capabilities, context.available_topology,
      thread_policy, resources, request, 0, context.placement_evidence);
  return {plan, resources};
}

struct WorkspaceBreakdownV2 {
  std::uint64_t total = 0;
  std::uint64_t shared = 0;
  std::uint64_t per_worker = 0;
  std::uint32_t alignment = 0;
  std::uint32_t threads = 0;
};

bool selected_workspace_breakdown_v2(
    const AdvancedGemmPlanV3 &advanced,
    WorkspaceBreakdownV2 *breakdown) noexcept {
  if (breakdown == nullptr ||
      advanced.plan.status !=
          matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return false;
  const std::size_t index =
      static_cast<std::size_t>(advanced.plan.selected_variant);
  if (index >= advanced.plan.candidates.size()) return false;
  const auto &candidate = advanced.plan.candidates[index];
  WorkspaceBreakdownV2 result;
  result.total = candidate.required_workspace_bytes;
  result.shared = candidate.shared_workspace_bytes;
  result.per_worker = candidate.per_worker_workspace_bytes;
  result.alignment = candidate.required_workspace_alignment;
  result.threads = candidate.actual_threads;
  if (result.per_worker != 0 &&
      result.threads >
          (std::numeric_limits<std::uint64_t>::max() - result.shared) /
              result.per_worker) {
    return false;
  }
  if (result.shared + result.per_worker * result.threads != result.total)
    return false;
  *breakdown = result;
  return true;
}

void populate_workspace_requirements_v2(
    const matcore::mdslc::planner::CpuGemmPlanV3 &plan,
    const WorkspaceBreakdownV2 &breakdown,
    matcore_cpu_affinity_policy_v1 affinity,
    matcore_cpu_numa_policy_v1 numa,
    matcore_cpu_smt_policy_v1 smt,
    matcore_gemm_workspace_requirements_v2 *requirements) noexcept {
  matcore_gemm_workspace_requirements_v2 output{};
  output.abi_version = MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2;
  output.struct_size = sizeof(output);
  output.workspace_bytes = breakdown.total;
  output.workspace_alignment = breakdown.alignment;
  output.actual_threads = breakdown.threads;
  output.per_worker_workspace_bytes = breakdown.per_worker;
  output.shared_workspace_bytes = breakdown.shared;
  output.selected_stable_id = plan.selected_id.data();
  output.affinity_policy = affinity;
  output.numa_policy = numa;
  output.smt_policy = smt;
  *requirements = output;
}

matcore_status_v0 validate_selected_workspace_v2(
    const WorkspaceBreakdownV2 &requirements, void *workspace,
    std::size_t workspace_bytes, const matcore_tensor_desc_v0 &out,
    const matcore_tensor_desc_v0 &lhs,
    const matcore_tensor_desc_v0 &rhs) noexcept {
  if (requirements.total == 0) return status(MATCORE_STATUS_OK_V0, "ok");
  if (requirements.total > workspace_bytes || workspace == nullptr)
    return status(MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0,
                  "CPU GEMM v2 workspace is smaller than the selected plan requires");
  if (requirements.total > std::numeric_limits<std::size_t>::max())
    return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                  "CPU GEMM v2 workspace exceeds the addressable size");
  if (requirements.alignment == 0 ||
      (requirements.alignment & (requirements.alignment - 1U)) != 0 ||
      reinterpret_cast<std::uintptr_t>(workspace) % requirements.alignment !=
          0) {
    return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                  "CPU GEMM v2 workspace does not satisfy selected alignment");
  }
  std::uintptr_t workspace_begin = 0;
  std::uintptr_t workspace_end = 0;
  std::uintptr_t out_begin = 0;
  std::uintptr_t out_end = 0;
  std::uintptr_t lhs_begin = 0;
  std::uintptr_t lhs_end = 0;
  std::uintptr_t rhs_begin = 0;
  std::uintptr_t rhs_end = 0;
  if (!byte_range(workspace, static_cast<std::size_t>(requirements.total),
                  &workspace_begin, &workspace_end) ||
      !byte_count(out, &out_begin, &out_end) ||
      !byte_count(lhs, &lhs_begin, &lhs_end) ||
      !byte_count(rhs, &rhs_begin, &rhs_end)) {
    return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                  "CPU GEMM v2 workspace range overflows the address space");
  }
  if (overlaps(workspace_begin, workspace_end, out_begin, out_end) ||
      overlaps(workspace_begin, workspace_end, lhs_begin, lhs_end) ||
      overlaps(workspace_begin, workspace_end, rhs_begin, rhs_end)) {
    return status(MATCORE_STATUS_ALIAS_VIOLATION_V0,
                  "CPU GEMM v2 workspace must not overlap any tensor");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

matcore_status_v0 parallel_execution_status_v1(
    matcore::mdslc::runtime::CpuParallelGemmStatusV1 result) noexcept {
  using ParallelStatus = matcore::mdslc::runtime::CpuParallelGemmStatusV1;
  switch (result) {
    case ParallelStatus::success:
      return status(MATCORE_STATUS_OK_V0, "ok");
    case ParallelStatus::invalid_thread_count:
      return status(MATCORE_STATUS_INVALID_THREAD_COUNT_V0,
                    "parallel CPU GEMM thread request is invalid");
    case ParallelStatus::workspace_insufficient:
      return status(MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0,
                    "parallel CPU GEMM workspace is too small");
    case ParallelStatus::workspace_misaligned:
    case ParallelStatus::invalid_pointer_alignment:
      return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                    "parallel CPU GEMM alignment contract failed");
    case ParallelStatus::alias_violation:
      return status(MATCORE_STATUS_ALIAS_VIOLATION_V0,
                    "parallel CPU GEMM buffers overlap illegally");
    case ParallelStatus::arithmetic_overflow:
      return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                    "parallel CPU GEMM size arithmetic overflowed");
    case ParallelStatus::isa_unavailable:
      return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                    "parallel CPU GEMM ISA is unavailable");
    case ParallelStatus::context_unavailable:
      return status(MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
                    "parallel CPU execution context is unavailable");
    case ParallelStatus::nested_parallelism_rejected:
      return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                    "nested native/provider parallelism is prohibited");
    case ParallelStatus::worker_task_failed:
      return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                    "parallel CPU GEMM worker task failed");
    case ParallelStatus::invalid_problem:
    case ParallelStatus::null_pointer:
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "parallel CPU GEMM received an invalid argument");
  }
  return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                "parallel CPU GEMM returned an unknown status");
}

bool execute_legacy_variant_v3(
    matcore::mdslc::planner::CpuGemmVariantV3 variant,
    const ValidatedGemmV0 &validated) noexcept {
  using VariantV3 = matcore::mdslc::planner::CpuGemmVariantV3;
  using VariantV2 = matcore::mdslc::planner::CpuGemmVariantV2;
  switch (variant) {
    case VariantV3::reference:
      return execute_legacy_variant(VariantV2::reference, validated);
    case VariantV3::tiled:
      return execute_legacy_variant(VariantV2::tiled, validated);
    case VariantV3::compiler_vectorized:
      return execute_legacy_variant(VariantV2::compiler_vectorized, validated);
    case VariantV3::external_openblas:
    case VariantV3::native_packed_avx2_fma:
    case VariantV3::native_packed_avx512_fma:
    case VariantV3::native_parallel_avx2_fma:
    case VariantV3::native_parallel_avx512_fma:
      return false;
  }
  return false;
}

matcore_status_v0 execute_single_variant_v3(
    matcore::mdslc::planner::CpuGemmVariantV3 variant,
    const ValidatedGemmV0 &validated,
    const matcore::mdslc::planner::CpuCandidateDecisionV3 &candidate,
    void *workspace, std::size_t workspace_bytes) noexcept {
  using Variant = matcore::mdslc::planner::CpuGemmVariantV3;
  switch (variant) {
    case Variant::reference:
    case Variant::tiled:
    case Variant::compiler_vectorized:
      return execute_legacy_variant_v3(variant, validated)
                 ? status(MATCORE_STATUS_OK_V0, "ok")
                 : status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                          "selected legacy CPU GEMM implementation did not execute");
    case Variant::external_openblas: {
      std::uint32_t actual_threads = 0;
      const auto provider_status =
          matcore::mdslc::runtime::execute_openblas_gemm_f32_v1(
              validated.problem, validated.lhs, validated.rhs, validated.out,
              candidate.actual_threads, &actual_threads);
      if (provider_status !=
              matcore::mdslc::runtime::OpenBlasExecutionStatusV1::success ||
          actual_threads != candidate.actual_threads) {
        return status(
            MATCORE_STATUS_EXTERNAL_PROVIDER_FAILURE_V0,
            "OpenBLAS SGEMM failed or violated the selected thread policy");
      }
      return status(MATCORE_STATUS_OK_V0, "ok");
    }
    case Variant::native_packed_avx2_fma:
      return packed_execution_status(
          matcore::mdslc::runtime::cpu_execute_packed_avx2_v1(
              validated.problem, validated.lhs, validated.rhs, validated.out,
              workspace, workspace_bytes));
    case Variant::native_packed_avx512_fma:
      return packed_execution_status(
          matcore::mdslc::runtime::cpu_execute_packed_avx512_v1(
              validated.problem, validated.lhs, validated.rhs, validated.out,
              workspace, workspace_bytes));
    case Variant::native_parallel_avx2_fma:
    case Variant::native_parallel_avx512_fma:
      return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                    "parallel CPU GEMM cannot use the single-worker path");
  }
  return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                "selected CPU GEMM variant is unknown");
}

struct SingleVariantTaskV3 {
  matcore::mdslc::planner::CpuGemmVariantV3 variant =
      matcore::mdslc::planner::CpuGemmVariantV3::reference;
  const ValidatedGemmV0 *validated = nullptr;
  const matcore::mdslc::planner::CpuCandidateDecisionV3 *candidate = nullptr;
  void *workspace = nullptr;
  std::size_t workspace_bytes = 0;
  matcore_status_v0 result =
      status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
             "single-worker CPU GEMM did not execute");
};

matcore::mdslc::runtime::CpuExecutionStatusV1 single_variant_task_v3(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept {
  if (task_index != 0 || worker_index != 0 || user_data == nullptr) {
    return matcore::mdslc::runtime::CpuExecutionStatusV1::
        invalid_configuration;
  }
  auto &task = *static_cast<SingleVariantTaskV3 *>(user_data);
  if (task.validated == nullptr || task.candidate == nullptr) {
    return matcore::mdslc::runtime::CpuExecutionStatusV1::
        invalid_configuration;
  }
  task.result = execute_single_variant_v3(
      task.variant, *task.validated, *task.candidate, task.workspace,
      task.workspace_bytes);
  return task.result.code == MATCORE_STATUS_OK_V0
             ? matcore::mdslc::runtime::CpuExecutionStatusV1::success
             : matcore::mdslc::runtime::CpuExecutionStatusV1::callback_failed;
}

matcore_status_v0 execute_single_variant_in_context_v3(
    matcore_cpu_execution_context_v1 &context,
    matcore::mdslc::planner::CpuGemmVariantV3 variant,
    const ValidatedGemmV0 &validated,
    const matcore::mdslc::planner::CpuCandidateDecisionV3 &candidate,
    void *workspace, std::size_t workspace_bytes) noexcept {
  const auto context_info = context.workers->info();
  if (context_info.affinity.requested_workers == 0) {
    return execute_single_variant_v3(variant, validated, candidate, workspace,
                                     workspace_bytes);
  }
  if (variant ==
          matcore::mdslc::planner::CpuGemmVariantV3::external_openblas &&
      candidate.actual_threads != 1) {
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "multi-thread OpenBLAS cannot execute inside one bound worker");
  }
  SingleVariantTaskV3 task;
  task.variant = variant;
  task.validated = &validated;
  task.candidate = &candidate;
  task.workspace = workspace;
  task.workspace_bytes = workspace_bytes;
  const auto nesting =
      variant == matcore::mdslc::planner::CpuGemmVariantV3::external_openblas
          ? matcore::mdslc::runtime::CpuProviderNestingPolicyV1::
                external_provider_active
          : matcore::mdslc::runtime::CpuProviderNestingPolicyV1::native_only;
  const auto execution = context.workers->run_tasks(
      1, 1, nesting, single_variant_task_v3, &task);
  if (task.result.code != MATCORE_STATUS_OK_V0) return task.result;
  if (execution != matcore::mdslc::runtime::CpuExecutionStatusV1::success) {
    return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                  matcore::mdslc::runtime::cpu_execution_status_message_v1(
                      execution));
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

}  // namespace

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_query_cpu_capabilities_v2(
    matcore_cpu_capabilities_v2 *capabilities) noexcept {
  const matcore_status_v0 validation =
      validate_empty_capabilities_v2(capabilities);
  if (validation.code != MATCORE_STATUS_OK_V0) return validation;
  const auto discovered = discover_runtime_cpu_capabilities_v2();
  if (!matcore::mdslc::platform::validate_cpu_capabilities_v2(discovered)
           .valid) {
    return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                  "detected CPU capability v2 record is invalid");
  }
  *capabilities = to_c_capabilities_v2(discovered);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_create_v1(
    const matcore_cpu_execution_context_options_v1 *options,
    matcore_cpu_execution_context_v1 **context,
    matcore_cpu_execution_context_report_v1 *report) noexcept {
  const matcore_status_v0 options_status = validate_context_options_v1(options);
  if (options_status.code != MATCORE_STATUS_OK_V0) return options_status;
  if (context == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context output handle must be non-null");
  if (*context != nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU execution-context output handle must initially be null");
  const matcore_status_v0 report_status =
      validate_empty_context_report_v1(report);
  if (report_status.code != MATCORE_STATUS_OK_V0) return report_status;

  try {
    const auto capabilities = discover_runtime_cpu_capabilities_v2();
    if (!matcore::mdslc::platform::validate_cpu_capabilities_v2(capabilities)
             .valid) {
      return status(MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
                    "CPU execution context requires valid capability discovery");
    }
    auto topology =
        matcore::mdslc::platform::discover_linux_cpu_topology_v1();
    const auto process_affinity =
        matcore::mdslc::platform::discover_current_thread_affinity_v1();
    const auto current_cpu =
        matcore::mdslc::platform::discover_current_logical_cpu_v1();
    ContextPlacementV1 selected_placement;
    matcore_status_v0 placement = select_context_placement_v1(
        *options, topology, process_affinity, current_cpu,
        &selected_placement);
    if (placement.code != MATCORE_STATUS_OK_V0) return placement;

    matcore::mdslc::runtime::CpuExecutionStatusV1 worker_status{};
    matcore::mdslc::runtime::CpuWorkerAffinityReportV1 worker_affinity;
    matcore::mdslc::runtime::CpuExecutionContextConfigV1 worker_config;
    worker_config.requested_threads = options->requested_threads;
    worker_config.maximum_threads = selected_placement.worker_count;
    worker_config.worker_cpu_ids = selected_placement.worker_cpu_ids;
    auto workers = matcore::mdslc::runtime::CpuExecutionContextV1::create(
        worker_config, &worker_status, &worker_affinity);
    if (workers == nullptr ||
        worker_status != matcore::mdslc::runtime::CpuExecutionStatusV1::success) {
      if (worker_status ==
              matcore::mdslc::runtime::CpuExecutionStatusV1::affinity_unavailable ||
          worker_status == matcore::mdslc::runtime::CpuExecutionStatusV1::
                               affinity_application_failed) {
        populate_failed_affinity_report_v1(
            *options, topology, selected_placement.available_topology,
            process_affinity, worker_affinity,
            selected_placement.affinity_induced_by_smt_policy,
            selected_placement.affinity_induced_by_numa_policy,
            selected_placement.creator_logical_cpu,
            selected_placement.selected_numa_node, report);
      }
      return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                    matcore::mdslc::runtime::cpu_execution_status_message_v1(
                      worker_status));
    }
    const auto placement_evidence = planner_placement_evidence_v1(
        selected_placement.available_topology, selected_placement.plan,
        options->numa_policy, !selected_placement.worker_cpu_ids.empty(),
        worker_affinity);
    if (!placement_evidence.evidence_complete) {
      return status(
          MATCORE_STATUS_UNSUPPORTED_CAPABILITY_V0,
          "CPU execution context could not authenticate planner placement evidence");
    }
    const auto validation_evidence =
        matcore::mdslc::runtime::validate_cpu_runtime_variants_v1(*workers);
    const std::uint64_t validation_submission_baseline =
        workers->info().completed_submissions;
    auto *created = new (std::nothrow) matcore_cpu_execution_context_v1;
    if (created == nullptr)
      return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                    "CPU execution-context handle allocation failed");
    created->magic = kCpuExecutionContextMagicV1;
    created->workers = std::move(workers);
    created->validation_evidence = validation_evidence;
    created->validation_submission_baseline =
        validation_submission_baseline;
    created->capabilities = capabilities;
    created->topology = std::move(topology);
    created->available_topology =
        std::move(selected_placement.available_topology);
    created->placement_evidence = placement_evidence;
    created->process_affinity_discovery_complete =
        process_affinity.discovery_complete;
    created->process_affinity_platform_error = process_affinity.platform_error;
    created->affinity_policy = options->affinity_policy;
    created->numa_policy = options->numa_policy;
    created->smt_policy = options->smt_policy;
    created->worker_affinity_induced_by_smt_policy =
        selected_placement.affinity_induced_by_smt_policy;
    created->worker_affinity_induced_by_numa_policy =
        selected_placement.affinity_induced_by_numa_policy;
    created->creator_logical_cpu = selected_placement.creator_logical_cpu;
    created->selected_numa_node = selected_placement.selected_numa_node;
    populate_context_report_v1(*created, report);
    *context = created;
    return status(MATCORE_STATUS_OK_V0, "ok");
  } catch (...) {
    return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                  "CPU execution-context discovery or creation failed");
  }
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_query_v1(
    matcore_cpu_execution_context_v1 *context,
    matcore_cpu_execution_context_report_v1 *report) noexcept {
  if (!valid_execution_context_v1(context))
    return status(MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
                  "CPU execution-context handle is invalid");
  const matcore_status_v0 validation =
      validate_empty_context_report_v1(report);
  if (validation.code != MATCORE_STATUS_OK_V0) return validation;
  populate_context_report_v1(*context, report);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_destroy_v1(
    matcore_cpu_execution_context_v1 *context) noexcept {
  if (!valid_execution_context_v1(context))
    return status(MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
                  "CPU execution-context handle is invalid");
  context->magic = 0;
  context->workers->shutdown();
  delete context;
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_context_workspace_size_v2(
    matcore_cpu_execution_context_v1 *context,
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v2 *options,
    matcore_gemm_workspace_requirements_v2 *requirements,
    matcore_cpu_gemm_plan_report_v3 *report) noexcept {
  if (!valid_execution_context_v1(context))
    return status(MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
                  "CPU execution-context handle is invalid");
  matcore::mdslc::planner::CpuGemmRequestV3 request;
  matcore_status_v0 result =
      validate_execution_options_v2(options, *context, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_empty_workspace_requirements_v2(requirements);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_empty_report_v3(report);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  const AdvancedGemmPlanV3 advanced =
      make_advanced_gemm_plan_v3(*context, validated, *options, request);
  populate_report_v3(advanced.plan, context->capabilities, context->topology,
                     context->available_topology, context->affinity_policy,
                     context->numa_policy, context->smt_policy,
                     context->worker_affinity_induced_by_smt_policy,
                     context->worker_affinity_induced_by_numa_policy,
                     context->creator_logical_cpu,
                     context->selected_numa_node,
                     matcore::mdslc::runtime::openblas_provider_info_v1(),
                     report);
  if (advanced.plan.status !=
      matcore::mdslc::planner::CpuPlanStatusV1::selected) {
    return unavailable_plan_status_v3(advanced.plan);
  }
  WorkspaceBreakdownV2 breakdown;
  if (!selected_workspace_breakdown_v2(advanced, &breakdown))
    return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                  "CPU GEMM v2 workspace breakdown is inconsistent");
  populate_workspace_requirements_v2(
      advanced.plan, breakdown, context->affinity_policy,
      context->numa_policy, context->smt_policy, requirements);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_execute_context_v2(
    matcore_cpu_execution_context_v1 *context,
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v2 *options, void *workspace,
    std::size_t workspace_bytes,
    matcore_cpu_gemm_plan_report_v3 *report) noexcept {
  if (!valid_execution_context_v1(context))
    return status(MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
                  "CPU execution-context handle is invalid");
  matcore::mdslc::planner::CpuGemmRequestV3 request;
  matcore_status_v0 result =
      validate_execution_options_v2(options, *context, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_empty_report_v3(report);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  const AdvancedGemmPlanV3 advanced =
      make_advanced_gemm_plan_v3(*context, validated, *options, request);
  populate_report_v3(advanced.plan, context->capabilities, context->topology,
                     context->available_topology, context->affinity_policy,
                     context->numa_policy, context->smt_policy,
                     context->worker_affinity_induced_by_smt_policy,
                     context->worker_affinity_induced_by_numa_policy,
                     context->creator_logical_cpu,
                     context->selected_numa_node,
                     matcore::mdslc::runtime::openblas_provider_info_v1(),
                     report);
  if (advanced.plan.status !=
      matcore::mdslc::planner::CpuPlanStatusV1::selected) {
    return unavailable_plan_status_v3(advanced.plan);
  }
  WorkspaceBreakdownV2 breakdown;
  if (!selected_workspace_breakdown_v2(advanced, &breakdown))
    return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                  "CPU GEMM v2 workspace breakdown is inconsistent");
  result = validate_selected_workspace_v2(
      breakdown, workspace, workspace_bytes, *out, *lhs, *rhs);
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  const std::size_t selected =
      static_cast<std::size_t>(advanced.plan.selected_variant);
  if (selected >= advanced.plan.candidates.size())
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "CPU GEMM v2 selected variant index is invalid");
  const auto &candidate = advanced.plan.candidates[selected];
  using Variant = matcore::mdslc::planner::CpuGemmVariantV3;
  switch (advanced.plan.selected_variant) {
    case Variant::reference:
    case Variant::tiled:
    case Variant::compiler_vectorized:
    case Variant::external_openblas:
    case Variant::native_packed_avx2_fma:
    case Variant::native_packed_avx512_fma:
      result = execute_single_variant_in_context_v3(
          *context, advanced.plan.selected_variant, validated, candidate,
          workspace, workspace_bytes);
      if (result.code != MATCORE_STATUS_OK_V0) return result;
      break;
    case Variant::native_parallel_avx2_fma: {
      matcore::mdslc::runtime::CpuParallelGemmReportV1 parallel_report;
      result = parallel_execution_status_v1(
          matcore::mdslc::runtime::cpu_execute_parallel_packed_avx2_v1(
              *context->workers, validated.problem, validated.lhs,
              validated.rhs, validated.out, workspace, workspace_bytes,
              candidate.actual_threads,
              matcore::mdslc::runtime::CpuProviderNestingPolicyV1::native_only,
              &parallel_report));
      if (result.code != MATCORE_STATUS_OK_V0) return result;
      if (parallel_report.actual_threads != candidate.actual_threads)
        return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                      "parallel AVX2 execution violated selected thread count");
      break;
    }
    case Variant::native_parallel_avx512_fma: {
      matcore::mdslc::runtime::CpuParallelGemmReportV1 parallel_report;
      result = parallel_execution_status_v1(
          matcore::mdslc::runtime::cpu_execute_parallel_packed_avx512_v1(
              *context->workers, validated.problem, validated.lhs,
              validated.rhs, validated.out, workspace, workspace_bytes,
              candidate.actual_threads,
              matcore::mdslc::runtime::CpuProviderNestingPolicyV1::native_only,
              &parallel_report));
      if (result.code != MATCORE_STATUS_OK_V0) return result;
      if (parallel_report.actual_threads != candidate.actual_threads)
        return status(MATCORE_STATUS_EXECUTOR_FAILURE_V0,
                      "parallel AVX-512 execution violated selected thread count");
      break;
    }
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_plan_gemm_f32_v1(const matcore_tensor_desc_v0 *out,
                                 const matcore_tensor_desc_v0 *lhs,
                                 const matcore_tensor_desc_v0 *rhs,
                                 const matcore_policy_v0 *policy,
                                 matcore_cpu_gemm_plan_report_v1 *report) noexcept {
  if (report == nullptr) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan report must be non-null");
  }
  matcore_status_v0 result = validate_empty_report(*report);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1());
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM planner found no legal implementation");
  }
  populate_report(plan, report);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_workspace_size_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    matcore_gemm_workspace_requirements_v1 *requirements,
    matcore_cpu_gemm_plan_report_v2 *report) noexcept {
  matcore::mdslc::planner::CpuGemmRequestV2 request;
  matcore_status_v0 result = validate_execution_options(options, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  result = validate_empty_workspace_requirements(requirements);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (report == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan v2 report must be non-null");
  result = validate_empty_report_v2(*report);
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  const auto resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, options->requested_threads);
  const auto capabilities =
      matcore::mdslc::planner::discover_cpu_capabilities_v1();
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v2(
      validated.problem, capabilities, resources, request);
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return unavailable_plan_status(plan);

  const std::size_t selected = static_cast<std::size_t>(plan.selected_variant);
  const auto &candidate = plan.candidates[selected];
  matcore_gemm_workspace_requirements_v1 output{};
  output.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  output.struct_size = sizeof(output);
  output.workspace_bytes = candidate.required_workspace_bytes;
  output.workspace_alignment = candidate.required_workspace_alignment;
  output.actual_threads = candidate.actual_threads;
  output.selected_stable_id = candidate.stable_id.data();
  *requirements = output;
  populate_report_v2(
      plan, matcore::mdslc::runtime::openblas_provider_info_v1(), report);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_execute_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options, void *workspace,
    std::size_t workspace_bytes, matcore_cpu_gemm_plan_report_v2 *report) noexcept {
  matcore::mdslc::planner::CpuGemmRequestV2 request;
  matcore_status_v0 result = validate_execution_options(options, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (report == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan v2 report must be non-null");
  result = validate_empty_report_v2(*report);
  if (result.code != MATCORE_STATUS_OK_V0) return result;

  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  const auto resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, options->requested_threads);
  const auto capabilities =
      matcore::mdslc::planner::discover_cpu_capabilities_v1();
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v2(
      validated.problem, capabilities, resources, request);
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return unavailable_plan_status(plan);
  const std::size_t selected = static_cast<std::size_t>(plan.selected_variant);
  const auto &candidate = plan.candidates[selected];
  if (candidate.required_workspace_bytes > workspace_bytes ||
      (candidate.required_workspace_bytes != 0 && workspace == nullptr)) {
    return status(MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0,
                  "CPU GEMM workspace is smaller than the selected plan requires");
  }
  if (candidate.required_workspace_bytes != 0 &&
      (reinterpret_cast<std::uintptr_t>(workspace) %
       candidate.required_workspace_alignment) != 0) {
    return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                  "CPU GEMM workspace does not satisfy the selected alignment");
  }

  bool executed = false;
  switch (plan.selected_variant) {
    case matcore::mdslc::planner::CpuGemmVariantV2::reference:
    case matcore::mdslc::planner::CpuGemmVariantV2::tiled:
    case matcore::mdslc::planner::CpuGemmVariantV2::compiler_vectorized:
      executed = execute_legacy_variant(plan.selected_variant, validated);
      break;
    case matcore::mdslc::planner::CpuGemmVariantV2::external_openblas: {
      std::uint32_t actual_threads = 0;
      const auto provider_status =
          matcore::mdslc::runtime::execute_openblas_gemm_f32_v1(
              validated.problem, validated.lhs, validated.rhs, validated.out,
              options->requested_threads, &actual_threads);
      if (provider_status !=
              matcore::mdslc::runtime::OpenBlasExecutionStatusV1::success ||
          actual_threads != options->requested_threads) {
        return status(MATCORE_STATUS_EXTERNAL_PROVIDER_FAILURE_V0,
                      "OpenBLAS SGEMM failed or did not honor the thread policy");
      }
      executed = true;
      break;
    }
    case matcore::mdslc::planner::CpuGemmVariantV2::native_packed_avx2_fma:
      result = packed_execution_status(
          matcore::mdslc::runtime::cpu_execute_packed_avx2_v1(
              validated.problem, validated.lhs, validated.rhs, validated.out,
              workspace, workspace_bytes));
      if (result.code != MATCORE_STATUS_OK_V0) return result;
      executed = true;
      break;
  }
  if (!executed)
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "selected CPU GEMM implementation did not execute");

  populate_report_v2(
      plan, matcore::mdslc::runtime::openblas_provider_info_v1(), report);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_prepacked_b_size_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    matcore_gemm_prepacked_b_requirements_v1 *requirements) noexcept {
  matcore::mdslc::planner::CpuGemmRequestV2 request;
  matcore_status_v0 result = validate_execution_options(options, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (!forced_native_packed_request(request)) {
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "prepacked-B requires the forced native packed AVX2/FMA variant");
  }
  result = validate_empty_prepacked_requirements(requirements);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  const auto resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, options->requested_threads);
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v2(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1(), resources,
      request);
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return unavailable_plan_status(plan);

  matcore::mdslc::runtime::CpuPackedGemmWorkspaceRequirementsV1 packed_storage;
  auto packed_status =
      matcore::mdslc::runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
          validated.problem, &packed_storage);
  if (packed_status !=
      matcore::mdslc::runtime::CpuPackedGemmStatusV1::success)
    return packed_execution_status(packed_status);
  matcore::mdslc::runtime::CpuPackedGemmWorkspaceRequirementsV1 execution;
  packed_status =
      matcore::mdslc::runtime::cpu_packed_avx2_workspace_requirements_v1(
          validated.problem,
          matcore::mdslc::runtime::CpuPackedGemmWorkspaceModeV1::
              transient_a_with_prepacked_b,
          &execution);
  if (packed_status !=
      matcore::mdslc::runtime::CpuPackedGemmStatusV1::success)
    return packed_execution_status(packed_status);

  matcore_gemm_prepacked_b_requirements_v1 output{};
  output.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  output.struct_size = sizeof(output);
  output.packed_b_bytes = packed_storage.total_bytes;
  output.packed_b_alignment =
      static_cast<std::uint32_t>(packed_storage.alignment_bytes);
  output.execution_workspace_bytes = execution.total_bytes;
  output.execution_workspace_alignment =
      static_cast<std::uint32_t>(execution.alignment_bytes);
  *requirements = output;
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_prepack_b_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options, void *packed_storage,
    std::size_t packed_storage_bytes,
    matcore_packed_b_desc_v1 *packed_b) noexcept {
  matcore::mdslc::planner::CpuGemmRequestV2 request;
  matcore_status_v0 result = validate_execution_options(options, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (!forced_native_packed_request(request)) {
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "prepacked-B requires the forced native packed AVX2/FMA variant");
  }
  result = validate_empty_packed_b_desc(packed_b);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  const auto resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, options->requested_threads);
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v2(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1(), resources,
      request);
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return unavailable_plan_status(plan);

  matcore::mdslc::runtime::CpuPackedGemmWorkspaceRequirementsV1
      packed_requirements;
  const auto requirements_status =
      matcore::mdslc::runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
          validated.problem, &packed_requirements);
  if (requirements_status !=
      matcore::mdslc::runtime::CpuPackedGemmStatusV1::success)
    return packed_execution_status(requirements_status);
  if (packed_storage != nullptr) {
    std::uintptr_t storage_begin = 0;
    std::uintptr_t storage_end = 0;
    std::uintptr_t out_begin = 0;
    std::uintptr_t out_end = 0;
    std::uintptr_t lhs_begin = 0;
    std::uintptr_t lhs_end = 0;
    std::uintptr_t rhs_begin = 0;
    std::uintptr_t rhs_end = 0;
    std::uintptr_t descriptor_begin = 0;
    std::uintptr_t descriptor_end = 0;
    if (!byte_range(packed_storage, packed_requirements.total_bytes,
                    &storage_begin, &storage_end) ||
        !byte_count(*out, &out_begin, &out_end) ||
        !byte_count(*lhs, &lhs_begin, &lhs_end) ||
        !byte_count(*rhs, &rhs_begin, &rhs_end) ||
        !byte_range(packed_b, sizeof(*packed_b), &descriptor_begin,
                    &descriptor_end)) {
      return status(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0,
                    "packed-B storage byte range overflows the address space");
    }
    if (overlaps(storage_begin, storage_end, out_begin, out_end) ||
        overlaps(storage_begin, storage_end, lhs_begin, lhs_end) ||
        overlaps(storage_begin, storage_end, rhs_begin, rhs_end) ||
        overlaps(storage_begin, storage_end, descriptor_begin,
                 descriptor_end) ||
        overlaps(descriptor_begin, descriptor_end, out_begin, out_end) ||
        overlaps(descriptor_begin, descriptor_end, lhs_begin, lhs_end) ||
        overlaps(descriptor_begin, descriptor_end, rhs_begin, rhs_end)) {
      return status(
          MATCORE_STATUS_ALIAS_VIOLATION_V0,
          "packed-B storage and descriptor must not overlap GEMM tensors or each other");
    }
  }

  matcore::mdslc::runtime::CpuPackedBViewV1 view;
  const auto packed_status =
      matcore::mdslc::runtime::cpu_prepare_packed_b_avx2_v1(
          validated.problem, validated.rhs, packed_storage,
          packed_storage_bytes, &view);
  if (packed_status !=
      matcore::mdslc::runtime::CpuPackedGemmStatusV1::success)
    return packed_execution_status(packed_status);

  matcore_packed_b_desc_v1 output{};
  output.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  output.struct_size = sizeof(output);
  output.source_data = view.source_data;
  output.packed_data = view.packed_data;
  output.storage_bytes = view.storage_bytes;
  output.packed_elements = view.packed_elements;
  output.k = view.k;
  output.n = view.n;
  output.kc = view.kc;
  output.nc = view.nc;
  output.nr = view.nr;
  output.provenance = view.provenance;
  *packed_b = output;
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_execute_prepacked_b_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *policy,
    const matcore_cpu_gemm_execution_options_v1 *options,
    const matcore_packed_b_desc_v1 *packed_b, void *workspace,
    std::size_t workspace_bytes, matcore_cpu_gemm_plan_report_v2 *report) noexcept {
  matcore::mdslc::planner::CpuGemmRequestV2 request;
  matcore_status_v0 result = validate_execution_options(options, &request);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (!forced_native_packed_request(request)) {
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "prepacked-B requires the forced native packed AVX2/FMA variant");
  }
  if (report == nullptr)
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU plan v2 report must be non-null");
  result = validate_empty_report_v2(*report);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  matcore::mdslc::runtime::CpuPackedBViewV1 view;
  result = unpack_packed_b_desc(packed_b, &view);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  ValidatedGemmV0 validated;
  result = validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  if (view.source_data != validated.rhs) {
    return status(MATCORE_STATUS_PREPACK_MISMATCH_V0,
                  "packed-B source identity does not match the supplied right input");
  }
  const auto resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, options->requested_threads);
  auto plan = matcore::mdslc::planner::plan_cpu_gemm_v2(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1(), resources,
      request);
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return unavailable_plan_status(plan);

  matcore::mdslc::runtime::CpuPackedGemmWorkspaceRequirementsV1 execution;
  const auto query_status =
      matcore::mdslc::runtime::cpu_packed_avx2_workspace_requirements_v1(
          validated.problem,
          matcore::mdslc::runtime::CpuPackedGemmWorkspaceModeV1::
              transient_a_with_prepacked_b,
          &execution);
  if (query_status !=
      matcore::mdslc::runtime::CpuPackedGemmStatusV1::success)
    return packed_execution_status(query_status);
  if (execution.total_bytes > workspace_bytes ||
      (execution.total_bytes != 0 && workspace == nullptr)) {
    return status(MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0,
                  "prepacked-B execution workspace is too small");
  }
  if (execution.total_bytes != 0 &&
      reinterpret_cast<std::uintptr_t>(workspace) %
              execution.alignment_bytes !=
          0) {
    return status(MATCORE_STATUS_INVALID_ALIGNMENT_V0,
                  "prepacked-B execution workspace is misaligned");
  }
  const auto execute_status =
      matcore::mdslc::runtime::cpu_execute_packed_avx2_prepacked_b_v1(
          validated.problem, validated.lhs, validated.out, view, workspace,
          workspace_bytes);
  if (execute_status !=
      matcore::mdslc::runtime::CpuPackedGemmStatusV1::success)
    return packed_execution_status(execute_status);

  const std::size_t packed_index = static_cast<std::size_t>(
      matcore::mdslc::planner::CpuGemmVariantV2::native_packed_avx2_fma);
  plan.candidates[packed_index].required_workspace_bytes = execution.total_bytes;
  plan.candidates[packed_index].required_workspace_alignment =
      static_cast<std::uint32_t>(execution.alignment_bytes);
  populate_report_v2(
      plan, matcore::mdslc::runtime::openblas_provider_info_v1(), report);
  return status(MATCORE_STATUS_OK_V0, "ok");
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_bf16_f32_reference_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) noexcept {
  ValidatedTypedGemmV1 validated;
  const matcore_status_v0 validation = validate_typed_gemm_v1(
      out, lhs, rhs, policy, MATCORE_DTYPE_F32_V0,
      MATCORE_DTYPE_BF16_V0, &validated);
  if (validation.code != MATCORE_STATUS_OK_V0) return validation;
  return numeric_reference_status_v1(
      matcore::mdslc::runtime::cpu_reference_gemm_bf16_storage_f32_v1(
          validated.shape,
          static_cast<const std::uint16_t *>(validated.lhs),
          static_cast<const std::uint16_t *>(validated.rhs),
          static_cast<float *>(validated.out)));
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_i8_i32_reference_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs,
    const matcore_policy_v0 *policy) noexcept {
  ValidatedTypedGemmV1 validated;
  const matcore_status_v0 validation = validate_typed_gemm_v1(
      out, lhs, rhs, policy, MATCORE_DTYPE_I32_V0, MATCORE_DTYPE_I8_V0,
      &validated);
  if (validation.code != MATCORE_STATUS_OK_V0) return validation;
  return numeric_reference_status_v1(
      matcore::mdslc::runtime::cpu_reference_gemm_i8_i32_v1(
          validated.shape,
          static_cast<const std::int8_t *>(validated.lhs),
          static_cast<const std::int8_t *>(validated.rhs),
          static_cast<std::int32_t *>(validated.out)));
}

extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_gemm_f32_v0(const matcore_tensor_desc_v0 *out,
                            const matcore_tensor_desc_v0 *lhs,
                            const matcore_tensor_desc_v0 *rhs,
                            const matcore_policy_v0 *policy) noexcept {
  ValidatedGemmV0 validated;
  const matcore_status_v0 result =
      validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  auto resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          validated.problem, 1);
  // The original one-shot ABI cannot receive caller-owned packing storage.
  // Keep it allocation-free by removing workspace-requiring candidates while
  // still allowing zero-workspace external and native variants.
  resources.native_packed_avx2_fma_compiled = false;
  resources.native_packed_workspace_size_valid = false;
  resources.native_packed_workspace_bytes = 0;
  resources.native_packed_workspace_alignment = 0;
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v2(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1(), resources);
  if (plan.status != matcore::mdslc::planner::CpuPlanStatusV1::selected)
    return unavailable_plan_status(plan);
  if (plan.selected_variant ==
      matcore::mdslc::planner::CpuGemmVariantV2::external_openblas) {
    std::uint32_t actual_threads = 0;
    const auto provider_status =
        matcore::mdslc::runtime::execute_openblas_gemm_f32_v1(
            validated.problem, validated.lhs, validated.rhs, validated.out, 1,
            &actual_threads);
    if (provider_status !=
            matcore::mdslc::runtime::OpenBlasExecutionStatusV1::success ||
        actual_threads != 1) {
      return status(MATCORE_STATUS_EXTERNAL_PROVIDER_FAILURE_V0,
                    "OpenBLAS SGEMM failed or did not honor one-shot thread policy");
    }
  } else if (!execute_legacy_variant(plan.selected_variant, validated)) {
    return status(MATCORE_STATUS_UNAVAILABLE_VARIANT_V0,
                  "selected zero-workspace CPU GEMM implementation did not execute");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}
