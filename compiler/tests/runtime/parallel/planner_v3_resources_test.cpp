#include "cpu_planner_v3_resources.h"

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace planner = matcore::mdslc::planner;
namespace runtime = matcore::mdslc::runtime;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  const planner::CpuGemmProblemV1 problem{
      1024, 1024, 1024, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  planner::CpuGemmImplementationResourcesV1 baseline;
  baseline.native_packed_avx2_fma_compiled = true;
  baseline.native_packed_workspace_size_valid = true;
  baseline.native_packed_workspace_bytes = 393216;
  baseline.native_packed_workspace_alignment = 64;

  runtime::CpuExecutionStatusV1 context_status{};
  auto context = runtime::CpuExecutionContextV1::create(
      {runtime::kCpuExecutionContextVersionV1, 8, 4}, &context_status);
  expect(context_status == runtime::CpuExecutionStatusV1::success &&
             context != nullptr,
         "resource fixture creates a topology-capped context");

  runtime::CpuRuntimeValidationEvidenceV1 validation;
  validation.packed_avx512_f32_runtime_validated =
      runtime::cpu_packed_avx512_runtime_usable_v1();
  const auto resources =
      runtime::augment_cpu_gemm_implementation_resources_v2(
          problem, baseline, context.get(), validation);
  expect(resources.execution_context_available &&
             resources.execution_context_worker_capacity == 4,
         "resource projection reports persistent worker capacity");
  expect(resources.native_parallel_avx2_fma_compiled &&
             resources.native_parallel_avx2_workspace_size_valid &&
             resources.native_parallel_avx2_shared_workspace_bytes != 0 &&
             resources.native_parallel_avx2_per_worker_workspace_bytes != 0 &&
             resources.native_parallel_avx2_workspace_alignment == 64,
         "resource projection reports AVX2 shared and per-worker workspace");
  expect(resources.native_packed_avx512_fma_compiled &&
             resources.native_packed_avx512_workspace_size_valid &&
             resources.native_parallel_avx512_fma_compiled &&
             resources.native_parallel_avx512_workspace_size_valid,
         "resource projection reports compiled AVX-512 implementation shape");
  expect(resources.native_packed_avx512_fma_runtime_validated ==
             runtime::cpu_packed_avx512_runtime_usable_v1(),
         "AVX-512 runtime validation requires injected evidence and usable hardware");

  const auto without_context =
      runtime::augment_cpu_gemm_implementation_resources_v2(
          problem, baseline, nullptr, validation);
  expect(!without_context.execution_context_available &&
             without_context.execution_context_worker_capacity == 0 &&
             without_context.native_parallel_avx2_fma_compiled,
         "implementation availability stays distinct from context availability");

  auto invalid_validation = validation;
  ++invalid_validation.version;
  const auto future_evidence =
      runtime::augment_cpu_gemm_implementation_resources_v2(
          problem, baseline, context.get(), invalid_validation);
  expect(!future_evidence.native_packed_avx512_fma_runtime_validated,
         "future validation-evidence version fails closed");

  if (failures != 0) {
    std::cerr << failures << " planner-v3 resource checks failed\n";
    return 1;
  }
  std::cout << "planner v3 runtime resources PASS\n";
  return 0;
}
