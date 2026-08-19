#include <matcore/runtime_c.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  } else {
    std::cout << "PASS: " << message << "\n";
  }
}

matcore_tensor_desc_v0 make_desc(float *data, std::int64_t rows,
                                 std::int64_t cols,
                                 matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 desc{};
  desc.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc.struct_size = sizeof(matcore_tensor_desc_v0);
  desc.data = data;
  desc.dtype = MATCORE_DTYPE_F32_V0;
  desc.rank = 2;
  desc.dims[0] = rows;
  desc.dims[1] = cols;
  desc.strides[0] = cols;
  desc.strides[1] = 1;
  desc.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc.mutability = mutability;
  return desc;
}

void test_loop_invariant_b_prepacking() {
  const std::int64_t m = 64;
  const std::int64_t n = 64;
  const std::int64_t k = 64;

  std::vector<float> lhs(m * k, 1.0f);
  std::vector<float> rhs(k * n, 2.0f);
  std::vector<float> out(m * n, 0.0f);

  matcore_tensor_desc_v0 out_desc = make_desc(out.data(), m, n, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 lhs_desc = make_desc(lhs.data(), m, k, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 rhs_desc = make_desc(rhs.data(), k, n, MATCORE_MUTABILITY_READ_ONLY_V0);

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  matcore_cpu_gemm_execution_options_v1 options{};
  options.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  options.struct_size = sizeof(matcore_cpu_gemm_execution_options_v1);
  options.request = MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V2;
  options.requested_threads = 1;

  matcore_gemm_prepacked_b_requirements_v1 req{};
  req.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  req.struct_size = sizeof(matcore_gemm_prepacked_b_requirements_v1);

  matcore_status_v0 sz_status = matcore_runtime_gemm_f32_prepacked_b_size_v1(
      &out_desc, &lhs_desc, &rhs_desc, &policy, &options, &req);

  if (sz_status.code != MATCORE_STATUS_OK_V0) {
    std::cout << "Packed AVX2/FMA prepacked B size query returned status code " << sz_status.code << " (expected on non-AVX2 fallback lanes)\n";
    return;
  }

  expect(req.packed_b_bytes > 0, "prepacked B storage requirement is positive");

  std::vector<uint8_t> packed_storage_raw(req.packed_b_bytes + req.packed_b_alignment);
  const auto packed_uptr = reinterpret_cast<std::uintptr_t>(packed_storage_raw.data());
  void *packed_storage = reinterpret_cast<void *>((packed_uptr + 63U) & ~uintptr_t{63U});

  std::vector<uint8_t> workspace_raw(req.execution_workspace_bytes + req.execution_workspace_alignment);
  const auto ws_uptr = reinterpret_cast<std::uintptr_t>(workspace_raw.data());
  void *workspace = reinterpret_cast<void *>((ws_uptr + 63U) & ~uintptr_t{63U});

  matcore_packed_b_desc_v1 packed_b{};
  packed_b.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  packed_b.struct_size = sizeof(matcore_packed_b_desc_v1);

  matcore_status_v0 prep_status = matcore_runtime_gemm_f32_prepack_b_v1(
      &out_desc, &lhs_desc, &rhs_desc, &policy, &options,
      packed_storage, static_cast<std::size_t>(req.packed_b_bytes), &packed_b);
  expect(prep_status.code == MATCORE_STATUS_OK_V0, "prepack B preparation succeeds");

  // Repeated execution loop using hoisted prepacked B
  const int loop_count = 50;
  for (int iter = 0; iter < loop_count; ++iter) {
    matcore_cpu_gemm_plan_report_v2 report{};
    report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
    report.struct_size = sizeof(matcore_cpu_gemm_plan_report_v2);
    matcore_status_v0 exec_status = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
        &out_desc, &lhs_desc, &rhs_desc, &policy, &options, &packed_b,
        workspace, static_cast<std::size_t>(req.execution_workspace_bytes), &report);
    expect(exec_status.code == MATCORE_STATUS_OK_V0, "prepacked B execution succeeds in loop iteration");
  }

  expect(out[0] == 128.0f, "prepacked B GEMM output equals 64 * 2.0 = 128.0");
}

} // namespace

int main() {
  std::cout << "Starting loop-invariant prepacked B hoisting tests...\n";
  test_loop_invariant_b_prepacking();

  if (failures != 0) {
    std::cerr << "Prepacked B hoisting tests: " << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "Prepacked B hoisting tests: all checks passed successfully\n";
  return 0;
}
