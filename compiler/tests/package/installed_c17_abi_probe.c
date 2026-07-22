#include <matcore/runtime_c.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect_true(int condition, const char *message) {
  if (!condition) {
    (void)fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

static void expect_ok(matcore_status_v0 status, const char *operation) {
  if (status.code != MATCORE_STATUS_OK_V0) {
    (void)fprintf(stderr, "FAIL: %s: %s\n", operation,
                  status.message != NULL ? status.message : "no diagnostic");
    ++failures;
  }
}

static void expect_rejected(matcore_status_v0 status, const char *operation) {
  if (status.code == MATCORE_STATUS_OK_V0) {
    (void)fprintf(stderr, "FAIL: %s accepted null arguments\n", operation);
    ++failures;
  }
}

static matcore_tensor_desc_v0 matrix(void *data, matcore_dtype_v0 dtype,
                                     int64_t rows, int64_t columns,
                                     matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 result = {0};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = (uint32_t)sizeof(result);
  result.data = data;
  result.dtype = dtype;
  result.rank = 2;
  result.dims[0] = rows;
  result.dims[1] = columns;
  result.strides[0] = columns;
  result.strides[1] = 1;
  result.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  result.mutability = mutability;
  return result;
}

static matcore_policy_v0 cpu_policy(void) {
  matcore_policy_v0 result = {0};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = (uint32_t)sizeof(result);
  result.target = MATCORE_TARGET_CPU_V0;
  result.fallback = MATCORE_FALLBACK_ERROR_V0;
  return result;
}

static void exercise_every_export_rejection(void) {
  expect_rejected(matcore_runtime_query_cpu_capabilities_v2(NULL),
                  "query capabilities");
  expect_rejected(matcore_runtime_plan_gemm_f32_v1(NULL, NULL, NULL, NULL,
                                                   NULL),
                  "legacy plan");
  expect_rejected(matcore_runtime_gemm_f32_workspace_size_v1(
                      NULL, NULL, NULL, NULL, NULL, NULL, NULL),
                  "workspace query v1");
  expect_rejected(matcore_runtime_gemm_f32_execute_v1(
                      NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL),
                  "workspace execution v1");
  expect_rejected(matcore_runtime_gemm_f32_prepacked_b_size_v1(
                      NULL, NULL, NULL, NULL, NULL, NULL),
                  "prepacked B query v1");
  expect_rejected(matcore_runtime_gemm_f32_prepack_b_v1(
                      NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL),
                  "prepacked B preparation v1");
  expect_rejected(matcore_runtime_gemm_f32_execute_prepacked_b_v1(
                      NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL),
                  "prepacked B execution v1");
  expect_rejected(matcore_runtime_cpu_execution_context_create_v1(NULL, NULL,
                                                                  NULL),
                  "context create v1");
  expect_rejected(matcore_runtime_cpu_execution_context_query_v1(NULL, NULL),
                  "context query v1");
  expect_rejected(matcore_runtime_cpu_execution_context_destroy_v1(NULL),
                  "context destroy v1");
  expect_rejected(matcore_runtime_gemm_f32_context_workspace_size_v2(
                      NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
                  "context workspace query v2");
  expect_rejected(matcore_runtime_gemm_f32_execute_context_v2(
                      NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL),
                  "context execution v2");
  expect_rejected(matcore_runtime_gemm_bf16_f32_reference_v1(
                      NULL, NULL, NULL, NULL),
                  "BF16 reference v1");
  expect_rejected(matcore_runtime_gemm_i8_i32_reference_v1(
                      NULL, NULL, NULL, NULL),
                  "INT8 reference v1");
  expect_rejected(matcore_runtime_gemm_f32_v0(NULL, NULL, NULL, NULL),
                  "F32 execution v0");
}

static void exercise_f32_and_planning(void) {
  float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  float rhs[4] = {5.0F, 6.0F, 7.0F, 8.0F};
  float out[4] = {-1.0F, -1.0F, -1.0F, -1.0F};
  matcore_tensor_desc_v0 lhs_desc =
      matrix(lhs, MATCORE_DTYPE_F32_V0, 2, 2,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 rhs_desc =
      matrix(rhs, MATCORE_DTYPE_F32_V0, 2, 2,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 out_desc =
      matrix(out, MATCORE_DTYPE_F32_V0, 2, 2,
             MATCORE_MUTABILITY_READ_WRITE_V0);
  const matcore_policy_v0 policy = cpu_policy();
  matcore_cpu_gemm_plan_report_v1 legacy_report = {0};
  matcore_cpu_gemm_execution_options_v1 options_v1 = {0};
  matcore_gemm_workspace_requirements_v1 requirements_v1 = {0};
  matcore_cpu_gemm_plan_report_v2 report_v2 = {0};

  expect_ok(matcore_runtime_gemm_f32_v0(&out_desc, &lhs_desc, &rhs_desc,
                                        &policy),
            "F32 execution v0");
  expect_true(out[0] == 19.0F && out[1] == 22.0F && out[2] == 43.0F &&
                  out[3] == 50.0F,
              "F32 execution v0 produced the expected matrix");

  legacy_report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  legacy_report.struct_size = (uint32_t)sizeof(legacy_report);
  expect_ok(matcore_runtime_plan_gemm_f32_v1(
                &out_desc, &lhs_desc, &rhs_desc, &policy, &legacy_report),
            "legacy plan v1");
  expect_true(legacy_report.selected_stable_id != NULL,
              "legacy plan selected a stable variant");

  options_v1.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  options_v1.struct_size = (uint32_t)sizeof(options_v1);
  options_v1.request = MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2;
  options_v1.requested_threads = 1;
  requirements_v1.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  requirements_v1.struct_size = (uint32_t)sizeof(requirements_v1);
  report_v2.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
  report_v2.struct_size = (uint32_t)sizeof(report_v2);
  expect_ok(matcore_runtime_gemm_f32_workspace_size_v1(
                &out_desc, &lhs_desc, &rhs_desc, &policy, &options_v1,
                &requirements_v1, &report_v2),
            "workspace query v1");
  expect_true(requirements_v1.workspace_bytes == 0 &&
                  requirements_v1.actual_threads == 1 &&
                  requirements_v1.selected_stable_id != NULL &&
                  strcmp(requirements_v1.selected_stable_id,
                         "cpu.reference.f32.v1") == 0,
              "forced reference reports zero caller workspace");

  (void)memset(out, 0, sizeof(out));
  (void)memset(&report_v2, 0, sizeof(report_v2));
  report_v2.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
  report_v2.struct_size = (uint32_t)sizeof(report_v2);
  expect_ok(matcore_runtime_gemm_f32_execute_v1(
                &out_desc, &lhs_desc, &rhs_desc, &policy, &options_v1, NULL, 0,
                &report_v2),
            "workspace execution v1");
  expect_true(out[0] == 19.0F && out[1] == 22.0F && out[2] == 43.0F &&
                  out[3] == 50.0F,
              "workspace execution v1 produced the expected matrix");
}

static void exercise_context(void) {
  float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  float rhs[4] = {5.0F, 6.0F, 7.0F, 8.0F};
  float out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  matcore_tensor_desc_v0 lhs_desc =
      matrix(lhs, MATCORE_DTYPE_F32_V0, 2, 2,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 rhs_desc =
      matrix(rhs, MATCORE_DTYPE_F32_V0, 2, 2,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 out_desc =
      matrix(out, MATCORE_DTYPE_F32_V0, 2, 2,
             MATCORE_MUTABILITY_READ_WRITE_V0);
  const matcore_policy_v0 policy = cpu_policy();
  matcore_cpu_execution_context_options_v1 context_options = {0};
  matcore_cpu_execution_context_report_v1 created = {0};
  matcore_cpu_execution_context_report_v1 queried = {0};
  matcore_cpu_execution_context_v1 *context = NULL;
  matcore_cpu_gemm_execution_options_v2 options_v2 = {0};
  matcore_gemm_workspace_requirements_v2 requirements_v2 = {0};
  matcore_cpu_gemm_plan_report_v3 report_v3 = {0};

  context_options.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  context_options.struct_size = (uint32_t)sizeof(context_options);
  context_options.requested_threads = 1;
  context_options.affinity_policy = MATCORE_CPU_AFFINITY_NONE_V1;
  context_options.numa_policy = MATCORE_CPU_NUMA_SINGLE_NODE_V1;
  context_options.smt_policy = MATCORE_CPU_SMT_ALLOW_V1;
  created.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  created.struct_size = (uint32_t)sizeof(created);
  expect_ok(matcore_runtime_cpu_execution_context_create_v1(
                &context_options, &context, &created),
            "context create v1");
  if (context == NULL) {
    expect_true(0, "context create returned a null handle");
    return;
  }
  expect_true(created.actual_worker_count == 1 &&
                  created.persistent_worker_count == 1,
              "context owns exactly one persistent worker");

  queried.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  queried.struct_size = (uint32_t)sizeof(queried);
  expect_ok(matcore_runtime_cpu_execution_context_query_v1(context, &queried),
            "context query v1");
  expect_true(queried.actual_worker_count == created.actual_worker_count,
              "context query preserves fixed worker state");

  options_v2.abi_version = MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2;
  options_v2.struct_size = (uint32_t)sizeof(options_v2);
  options_v2.request = MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V3;
  options_v2.requested_threads = 1;
  options_v2.affinity_policy = MATCORE_CPU_AFFINITY_NONE_V1;
  options_v2.numa_policy = MATCORE_CPU_NUMA_SINGLE_NODE_V1;
  options_v2.smt_policy = MATCORE_CPU_SMT_ALLOW_V1;
  requirements_v2.abi_version = MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2;
  requirements_v2.struct_size = (uint32_t)sizeof(requirements_v2);
  report_v3.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V3;
  report_v3.struct_size = (uint32_t)sizeof(report_v3);
  expect_ok(matcore_runtime_gemm_f32_context_workspace_size_v2(
                context, &out_desc, &lhs_desc, &rhs_desc, &policy, &options_v2,
                &requirements_v2, &report_v3),
            "context workspace query v2");
  expect_true(requirements_v2.workspace_bytes == 0 &&
                  requirements_v2.selected_stable_id != NULL &&
                  strcmp(requirements_v2.selected_stable_id,
                         "cpu.reference.f32.v1") == 0,
              "context planner selected the forced reference variant");

  (void)memset(&report_v3, 0, sizeof(report_v3));
  report_v3.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V3;
  report_v3.struct_size = (uint32_t)sizeof(report_v3);
  expect_ok(matcore_runtime_gemm_f32_execute_context_v2(
                context, &out_desc, &lhs_desc, &rhs_desc, &policy, &options_v2,
                NULL, 0, &report_v3),
            "context execution v2");
  expect_true(out[0] == 19.0F && out[1] == 22.0F && out[2] == 43.0F &&
                  out[3] == 50.0F,
              "context execution v2 produced the expected matrix");

  expect_ok(matcore_runtime_cpu_execution_context_destroy_v1(context),
            "context destroy v1");
}

static void exercise_capabilities_and_typed_references(void) {
  matcore_cpu_capabilities_v2 capabilities = {0};
  matcore_bf16_v1 bf16_lhs[1] = {UINT16_C(0x3f80)};
  matcore_bf16_v1 bf16_rhs[1] = {UINT16_C(0x4000)};
  float bf16_out[1] = {0.0F};
  int8_t i8_lhs[1] = {INT8_C(3)};
  int8_t i8_rhs[1] = {INT8_C(4)};
  int32_t i32_out[1] = {0};
  matcore_tensor_desc_v0 bf16_lhs_desc =
      matrix(bf16_lhs, MATCORE_DTYPE_BF16_V0, 1, 1,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 bf16_rhs_desc =
      matrix(bf16_rhs, MATCORE_DTYPE_BF16_V0, 1, 1,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 bf16_out_desc =
      matrix(bf16_out, MATCORE_DTYPE_F32_V0, 1, 1,
             MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 i8_lhs_desc =
      matrix(i8_lhs, MATCORE_DTYPE_I8_V0, 1, 1,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 i8_rhs_desc =
      matrix(i8_rhs, MATCORE_DTYPE_I8_V0, 1, 1,
             MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 i32_out_desc =
      matrix(i32_out, MATCORE_DTYPE_I32_V0, 1, 1,
             MATCORE_MUTABILITY_READ_WRITE_V0);
  const matcore_policy_v0 policy = cpu_policy();

  capabilities.abi_version = MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2;
  capabilities.struct_size = (uint32_t)sizeof(capabilities);
  expect_ok(matcore_runtime_query_cpu_capabilities_v2(&capabilities),
            "query capabilities v2");
  expect_true((capabilities.implementation_available_features &
               MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V2) != 0,
              "capability query reports the portable implementation");

  expect_ok(matcore_runtime_gemm_bf16_f32_reference_v1(
                &bf16_out_desc, &bf16_lhs_desc, &bf16_rhs_desc, &policy),
            "BF16 reference v1");
  expect_true(bf16_out[0] == 2.0F,
              "BF16 reference v1 produced the expected value");

  expect_ok(matcore_runtime_gemm_i8_i32_reference_v1(
                &i32_out_desc, &i8_lhs_desc, &i8_rhs_desc, &policy),
            "INT8 reference v1");
  expect_true(i32_out[0] == INT32_C(12),
              "INT8 reference v1 produced the expected value");
}

int main(void) {
  exercise_every_export_rejection();
  exercise_capabilities_and_typed_references();
  exercise_f32_and_planning();
  exercise_context();
  if (failures != 0) {
    return 1;
  }
  (void)puts("installed strict C17 ABI probe: PASS");
  return 0;
}
