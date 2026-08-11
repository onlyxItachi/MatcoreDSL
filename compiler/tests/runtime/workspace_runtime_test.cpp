#include "matcore/runtime_c.h"

#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

matcore_tensor_desc_v0 tensor(void *data, std::int64_t rows,
                              std::int64_t columns,
                              matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 value{};
  value.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  value.struct_size = sizeof(value);
  value.data = data;
  value.dtype = MATCORE_DTYPE_F32_V0;
  value.rank = 2;
  value.dims[0] = rows;
  value.dims[1] = columns;
  value.strides[0] = columns;
  value.strides[1] = 1;
  value.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  value.mutability = mutability;
  return value;
}

matcore_policy_v0 policy() {
  matcore_policy_v0 value{};
  value.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  value.struct_size = sizeof(value);
  value.target = MATCORE_TARGET_CPU_V0;
  value.fallback = MATCORE_FALLBACK_ERROR_V0;
  return value;
}

matcore_cpu_gemm_execution_options_v1 options(
    matcore_cpu_gemm_request_v2 request, std::uint32_t threads = 1) {
  matcore_cpu_gemm_execution_options_v1 value{};
  value.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  value.struct_size = sizeof(value);
  value.request = request;
  value.requested_threads = threads;
  return value;
}

matcore_gemm_workspace_requirements_v1 empty_requirements() {
  matcore_gemm_workspace_requirements_v1 value{};
  value.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  value.struct_size = sizeof(value);
  return value;
}

matcore_cpu_gemm_plan_report_v2 empty_report() {
  matcore_cpu_gemm_plan_report_v2 value{};
  value.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
  value.struct_size = sizeof(value);
  return value;
}

bool correct(const std::array<float, 4> &out) {
  const std::array<double, 4> expected{19.0, 22.0, 43.0, 50.0};
  for (std::size_t index = 0; index < out.size(); ++index) {
    if (std::fabs(static_cast<double>(out[index]) - expected[index]) > 1.0e-5)
      return false;
  }
  return true;
}

}  // namespace

int main() {
  std::array<float, 4> lhs{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> rhs{5.0F, 6.0F, 7.0F, 8.0F};
  std::array<float, 4> out{-1.0F, -1.0F, -1.0F, -1.0F};
  matcore_tensor_desc_v0 lhs_desc =
      tensor(lhs.data(), 2, 2, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 rhs_desc =
      tensor(rhs.data(), 2, 2, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 out_desc =
      tensor(out.data(), 2, 2, MATCORE_MUTABILITY_READ_WRITE_V0);
  const matcore_policy_v0 cpu_policy = policy();

  auto reference_options =
      options(MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2);
  auto requirements = empty_requirements();
  auto report = empty_report();
  matcore_status_v0 result = matcore_runtime_gemm_f32_workspace_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &reference_options,
      &requirements, &report);
  if (result.code != MATCORE_STATUS_OK_V0 ||
      requirements.workspace_bytes != 0 ||
      requirements.workspace_alignment != 1 ||
      requirements.actual_threads != 1 ||
      std::strcmp(requirements.selected_stable_id,
                  "cpu.reference.f32.v1") != 0 ||
      report.candidate_count != MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V2) {
    std::cerr << "reference workspace query failed: " << result.message << '\n';
    return 1;
  }
  if (result.message == nullptr || report.external_provider == nullptr ||
      report.external_provider_version == nullptr ||
      report.external_provider_config == nullptr) {
    std::cerr << "workspace query omitted borrowed diagnostic strings\n";
    return 1;
  }
  const char *borrowed_status_message = result.message;
  const char *borrowed_provider = report.external_provider;
  const char *borrowed_provider_version = report.external_provider_version;
  const char *borrowed_provider_config = report.external_provider_config;
  const std::string status_message_copy(borrowed_status_message);
  const std::string provider_copy(borrowed_provider);
  const std::string provider_version_copy(borrowed_provider_version);
  const std::string provider_config_copy(borrowed_provider_config);

  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &reference_options,
      nullptr, 0, &report);
  if (result.code != MATCORE_STATUS_OK_V0 || !correct(out) ||
      std::strcmp(report.selected_stable_id, "cpu.reference.f32.v1") != 0) {
    std::cerr << "workspace reference execution failed: " << result.message
              << '\n';
    return 1;
  }

  out.fill(-3.0F);
  auto bad_threads =
      options(MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2, 0);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &bad_threads, nullptr, 0,
      &report);
  if (result.code != MATCORE_STATUS_INVALID_THREAD_COUNT_V0 ||
      out != std::array<float, 4>{-3.0F, -3.0F, -3.0F, -3.0F}) {
    std::cerr << "invalid thread count was accepted or mutated output\n";
    return 1;
  }

  auto openblas_options =
      options(MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V2);
  requirements = empty_requirements();
  report = empty_report();
  result = matcore_runtime_gemm_f32_workspace_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &openblas_options,
      &requirements, &report);
  if (result.code == MATCORE_STATUS_OK_V0) {
    if (requirements.workspace_bytes != 0 ||
        std::strcmp(requirements.selected_stable_id,
                    "cpu.external.openblas.f32.v1") != 0 ||
        std::strcmp(report.external_provider, "OpenBLAS") != 0) {
      std::cerr << "linked OpenBLAS workspace report is inconsistent\n";
      return 1;
    }
    out.fill(-5.0F);
    report = empty_report();
    result = matcore_runtime_gemm_f32_execute_v1(
        &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &openblas_options,
        nullptr, 0, &report);
    if (result.code != MATCORE_STATUS_OK_V0 || !correct(out)) {
      std::cerr << "workspace OpenBLAS execution failed\n";
      return 1;
    }
  } else if (result.code != MATCORE_STATUS_UNAVAILABLE_VARIANT_V0) {
    std::cerr << "unlinked OpenBLAS failed with the wrong status\n";
    return 1;
  }

  out.fill(-4.0F);
  auto excessive_openblas_options = options(
      MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V2,
      std::numeric_limits<std::uint32_t>::max());
  requirements = empty_requirements();
  report = empty_report();
  const auto requirements_before = requirements;
  const auto report_before = report;
  result = matcore_runtime_gemm_f32_workspace_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy,
      &excessive_openblas_options, &requirements, &report);
  if (result.code != MATCORE_STATUS_UNAVAILABLE_VARIANT_V0 ||
      std::memcmp(&requirements, &requirements_before,
                  sizeof(requirements)) != 0 ||
      std::memcmp(&report, &report_before, sizeof(report)) != 0 ||
      out != std::array<float, 4>{-4.0F, -4.0F, -4.0F, -4.0F}) {
    std::cerr << "excessive OpenBLAS thread plan mutated output reports or data\n";
    return 1;
  }

  auto packed_options = options(
      MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V2);
  requirements = empty_requirements();
  report = empty_report();
  result = matcore_runtime_gemm_f32_workspace_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      &requirements, &report);
  if (result.code != MATCORE_STATUS_OK_V0 ||
      requirements.workspace_bytes == 0 ||
      requirements.workspace_alignment != 64 ||
      std::strcmp(requirements.selected_stable_id,
                  "cpu.native-packed.avx2-fma.f32.v1") != 0) {
    std::cerr << "native packed workspace query failed: " << result.message
              << '\n';
    return 1;
  }
  std::vector<std::byte> workspace_storage(
      static_cast<std::size_t>(requirements.workspace_bytes) + 63);
  const auto raw =
      reinterpret_cast<std::uintptr_t>(workspace_storage.data());
  void *workspace = reinterpret_cast<void *>((raw + 63U) & ~uintptr_t{63U});

  out.fill(-11.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options, workspace,
      static_cast<std::size_t>(requirements.workspace_bytes - 1), &report);
  if (result.code != MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0 ||
      out != std::array<float, 4>{-11.0F, -11.0F, -11.0F, -11.0F}) {
    std::cerr << "insufficient packed workspace mutated output\n";
    return 1;
  }

  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options, workspace,
      static_cast<std::size_t>(requirements.workspace_bytes), &report);
  if (result.code != MATCORE_STATUS_OK_V0 || !correct(out) ||
      std::strcmp(report.selected_stable_id,
                  "cpu.native-packed.avx2-fma.f32.v1") != 0) {
    std::cerr << "workspace native packed execution failed: " << result.message
              << '\n';
    return 1;
  }

  matcore_gemm_prepacked_b_requirements_v1 prepacked_requirements{};
  prepacked_requirements.abi_version =
      MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  prepacked_requirements.struct_size = sizeof(prepacked_requirements);
  result = matcore_runtime_gemm_f32_prepacked_b_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      &prepacked_requirements);
  if (result.code != MATCORE_STATUS_OK_V0 ||
      prepacked_requirements.packed_b_bytes == 0 ||
      prepacked_requirements.packed_b_alignment != 64 ||
      prepacked_requirements.execution_workspace_bytes == 0 ||
      prepacked_requirements.execution_workspace_alignment != 64) {
    std::cerr << "prepacked-B requirement query failed\n";
    return 1;
  }
  std::vector<std::byte> packed_storage_bytes(
      static_cast<std::size_t>(prepacked_requirements.packed_b_bytes) + 63);
  const auto packed_raw =
      reinterpret_cast<std::uintptr_t>(packed_storage_bytes.data());
  void *packed_storage =
      reinterpret_cast<void *>((packed_raw + 63U) & ~uintptr_t{63U});
  matcore_packed_b_desc_v1 packed_b{};
  packed_b.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  packed_b.struct_size = sizeof(packed_b);
  result = matcore_runtime_gemm_f32_prepack_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      packed_storage,
      static_cast<std::size_t>(prepacked_requirements.packed_b_bytes),
      &packed_b);
  if (result.code != MATCORE_STATUS_OK_V0 ||
      packed_b.source_data != rhs.data() || packed_b.packed_data == nullptr ||
      packed_b.provenance == 0) {
    std::cerr << "prepacked-B preparation failed\n";
    return 1;
  }
  const matcore_packed_b_desc_v1 valid_packed_b = packed_b;

  alignas(64) std::array<std::byte, 256> aliased_output_storage{};
  auto *aliased_output =
      reinterpret_cast<float *>(aliased_output_storage.data());
  for (std::size_t index = 0; index < out.size(); ++index)
    aliased_output[index] = -29.0F;
  matcore_tensor_desc_v0 aliased_out_desc =
      tensor(aliased_output, 2, 2, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_packed_b_desc_v1 rejected_packed_b{};
  rejected_packed_b.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  rejected_packed_b.struct_size = sizeof(rejected_packed_b);
  result = matcore_runtime_gemm_f32_prepack_b_v1(
      &aliased_out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      aliased_output_storage.data(),
      static_cast<std::size_t>(prepacked_requirements.packed_b_bytes),
      &rejected_packed_b);
  if (result.code != MATCORE_STATUS_ALIAS_VIOLATION_V0 ||
      aliased_output[0] != -29.0F || aliased_output[1] != -29.0F ||
      aliased_output[2] != -29.0F || aliased_output[3] != -29.0F) {
    std::cerr << "packed-B storage overlapped and mutated output\n";
    return 1;
  }

  alignas(64) std::array<std::byte, 256> aliased_lhs_storage{};
  auto *aliased_lhs = reinterpret_cast<float *>(aliased_lhs_storage.data());
  for (std::size_t index = 0; index < lhs.size(); ++index)
    aliased_lhs[index] = lhs[index];
  matcore_tensor_desc_v0 aliased_lhs_desc =
      tensor(aliased_lhs, 2, 2, MATCORE_MUTABILITY_READ_ONLY_V0);
  rejected_packed_b = {};
  rejected_packed_b.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  rejected_packed_b.struct_size = sizeof(rejected_packed_b);
  result = matcore_runtime_gemm_f32_prepack_b_v1(
      &out_desc, &aliased_lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      aliased_lhs_storage.data(),
      static_cast<std::size_t>(prepacked_requirements.packed_b_bytes),
      &rejected_packed_b);
  if (result.code != MATCORE_STATUS_ALIAS_VIOLATION_V0 ||
      aliased_lhs[0] != lhs[0] || aliased_lhs[1] != lhs[1] ||
      aliased_lhs[2] != lhs[2] || aliased_lhs[3] != lhs[3]) {
    std::cerr << "packed-B storage overlapped and mutated left input\n";
    return 1;
  }

  std::vector<std::byte> prepacked_workspace_storage(
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes) +
      63);
  const auto prepacked_workspace_raw =
      reinterpret_cast<std::uintptr_t>(prepacked_workspace_storage.data());
  void *prepacked_workspace = reinterpret_cast<void *>(
      (prepacked_workspace_raw + 63U) & ~uintptr_t{63U});
  out.fill(-13.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options, &packed_b,
      prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_OK_V0 || !correct(out) ||
      report.selected_workspace_bytes !=
          prepacked_requirements.execution_workspace_bytes) {
    std::cerr << "prepacked-B execution failed: " << result.message << '\n';
    return 1;
  }

  out.fill(-15.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      &valid_packed_b, prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_OK_V0 || !correct(out)) {
    std::cerr << "serial prepacked-B reuse failed: " << result.message << '\n';
    return 1;
  }

  auto two_thread_packed_options = options(
      MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V2, 2);
  matcore_gemm_prepacked_b_requirements_v1 two_thread_requirements{};
  two_thread_requirements.abi_version =
      MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  two_thread_requirements.struct_size = sizeof(two_thread_requirements);
  const auto two_thread_requirements_before = two_thread_requirements;
  out.fill(-16.0F);
  result = matcore_runtime_gemm_f32_prepacked_b_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy,
      &two_thread_packed_options, &two_thread_requirements);
  if (result.code != MATCORE_STATUS_UNAVAILABLE_VARIANT_V0 ||
      std::memcmp(&two_thread_requirements, &two_thread_requirements_before,
                  sizeof(two_thread_requirements)) != 0 ||
      out != std::array<float, 4>{-16.0F, -16.0F, -16.0F, -16.0F}) {
    std::cerr << "prepacked-B v1 accepted non-serial execution\n";
    return 1;
  }

  std::vector<std::byte> relocated_storage_bytes(
      static_cast<std::size_t>(prepacked_requirements.packed_b_bytes) + 63);
  const auto relocated_raw =
      reinterpret_cast<std::uintptr_t>(relocated_storage_bytes.data());
  void *relocated_storage =
      reinterpret_cast<void *>((relocated_raw + 63U) & ~uintptr_t{63U});
  std::memcpy(relocated_storage, valid_packed_b.packed_data,
              static_cast<std::size_t>(prepacked_requirements.packed_b_bytes));
  auto relocated_packed_b = valid_packed_b;
  relocated_packed_b.packed_data = relocated_storage;
  out.fill(-18.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      &relocated_packed_b, prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_PREPACK_MISMATCH_V0 ||
      out != std::array<float, 4>{-18.0F, -18.0F, -18.0F, -18.0F}) {
    std::cerr << "relocated packed-B storage retained stale authentication\n";
    return 1;
  }

  std::array<float, 4> other_rhs{2.0F, 0.0F, 0.0F, 2.0F};
  matcore_tensor_desc_v0 other_rhs_desc =
      tensor(other_rhs.data(), 2, 2, MATCORE_MUTABILITY_READ_ONLY_V0);
  out.fill(-19.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &lhs_desc, &other_rhs_desc, &cpu_policy, &packed_options,
      &valid_packed_b, prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_PREPACK_MISMATCH_V0 ||
      out != std::array<float, 4>{-19.0F, -19.0F, -19.0F, -19.0F}) {
    std::cerr << "prepacked-B execution accepted a different right input\n";
    return 1;
  }

  matcore_tensor_desc_v0 packed_storage_lhs_desc = tensor(
      const_cast<void *>(valid_packed_b.packed_data), 2, 2,
      MATCORE_MUTABILITY_READ_ONLY_V0);
  out.fill(-23.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &packed_storage_lhs_desc, &rhs_desc, &cpu_policy,
      &packed_options, &valid_packed_b, prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_ALIAS_VIOLATION_V0 ||
      out != std::array<float, 4>{-23.0F, -23.0F, -23.0F, -23.0F}) {
    std::cerr << "prepacked-B execution accepted packed storage as left input\n";
    return 1;
  }

  out.fill(-17.0F);
  packed_b.provenance ^= UINT64_C(1);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options, &packed_b,
      prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_PREPACK_MISMATCH_V0 ||
      out != std::array<float, 4>{-17.0F, -17.0F, -17.0F, -17.0F}) {
    std::cerr << "corrupt prepacked-B descriptor was accepted or mutated output\n";
    return 1;
  }

  rhs = {1.0F, 0.0F, 0.0F, 1.0F};
  packed_b = {};
  packed_b.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  packed_b.struct_size = sizeof(packed_b);
  result = matcore_runtime_gemm_f32_prepack_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options,
      packed_storage,
      static_cast<std::size_t>(prepacked_requirements.packed_b_bytes),
      &packed_b);
  if (result.code != MATCORE_STATUS_OK_V0 ||
      packed_b.provenance != valid_packed_b.provenance) {
    std::cerr << "explicit repack after source mutation failed\n";
    return 1;
  }
  out.fill(-21.0F);
  report = empty_report();
  result = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &cpu_policy, &packed_options, &packed_b,
      prepacked_workspace,
      static_cast<std::size_t>(
          prepacked_requirements.execution_workspace_bytes),
      &report);
  if (result.code != MATCORE_STATUS_OK_V0 || out != lhs) {
    std::cerr << "repacked mutated source did not produce current contents\n";
    return 1;
  }

  if (std::strcmp(borrowed_status_message, status_message_copy.c_str()) != 0 ||
      std::strcmp(borrowed_provider, provider_copy.c_str()) != 0 ||
      std::strcmp(borrowed_provider_version, provider_version_copy.c_str()) !=
          0 ||
      std::strcmp(borrowed_provider_config, provider_config_copy.c_str()) != 0) {
    std::cerr << "borrowed runtime/provider string expired during runtime use\n";
    return 1;
  }
  return 0;
}
