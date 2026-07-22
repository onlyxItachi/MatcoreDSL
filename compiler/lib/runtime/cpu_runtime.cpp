#include "matcore/runtime_c.h"

#include "cpu_backend_registry.h"
#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_planner.h"
#include "cpu_planner_v2.h"

#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(matcore::mdslc::planner::kCpuGemmCandidateCountV1 ==
                  MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1,
              "CPU planner registry and C plan report must stay in sync");
static_assert(matcore::mdslc::planner::kCpuGemmCandidateCountV2 ==
                  MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V2,
              "CPU planner v2 registry and C plan report must stay in sync");

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

}  // namespace

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
matcore_runtime_gemm_f32_v0(const matcore_tensor_desc_v0 *out,
                            const matcore_tensor_desc_v0 *lhs,
                            const matcore_tensor_desc_v0 *rhs,
                            const matcore_policy_v0 *policy) noexcept {
  ValidatedGemmV0 validated;
  const matcore_status_v0 result =
      validate_gemm_v0(out, lhs, rhs, policy, &validated);
  if (result.code != MATCORE_STATUS_OK_V0) return result;
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      validated.problem,
      matcore::mdslc::planner::discover_cpu_capabilities_v1());
  if (!matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
          plan, validated.lhs, validated.rhs, validated.out)) {
    return status(MATCORE_STATUS_INVALID_ARGUMENT_V0,
                  "CPU GEMM planner found no legal implementation");
  }
  return status(MATCORE_STATUS_OK_V0, "ok");
}
