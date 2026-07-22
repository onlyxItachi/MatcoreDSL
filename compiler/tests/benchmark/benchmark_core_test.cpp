#include "benchmark.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace bench = matcore::mdslc::bench;
int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

}  // namespace

int main() {
  const auto quick = bench::quick_profile_v1();
  expect(quick.size() == 8, "quick profile has the declared eight shapes");
  expect(quick.front().m == 1 && quick.front().n == 1 && quick.front().k == 1,
         "quick profile begins with 1x1x1");
  expect(quick[1].m == 2 && quick[1].n == 3 && quick[1].k == 2,
         "profile dimensions use unambiguous M,N,K order");
  expect(quick.back().m == 256 && quick.back().n == 256 &&
             quick.back().k == 256,
         "quick profile ends at 256 cubed");

  const auto standard = bench::standard_profile_v1();
  expect(standard.size() == 30,
         "standard profile has 17 square, 8 rectangular, and 5 tail shapes");
  const auto full = bench::full_profile_v1();
  expect(full.size() == standard.size() + 2 &&
             full[full.size() - 2].m == 4096 && full.back().m == 8192,
         "full profile adds only the opt-in 4096 and 8192 squares");

  bench::BenchmarkOptionsV1 options;
  options.profile = bench::ProfileV1::custom;
  options.shapes = {{4, 5, 6}};
  std::string error;
  expect(bench::validate_options_v1(options, error),
         "valid custom benchmark options pass");
  auto invalid_alignment = options;
  invalid_alignment.alignment_bytes = 6;
  expect(!bench::validate_options_v1(invalid_alignment, error) &&
             error.find("power of two") != std::string::npos,
         "non-power-of-two alignment is rejected");
  auto invalid_modes = options;
  invalid_modes.packing_mode = bench::PackingModeV1::prepack_b;
  invalid_modes.allocation_mode = bench::AllocationModeV1::include_allocation;
  expect(!bench::validate_options_v1(invalid_modes, error) &&
             error.find("requires reusable workspace") != std::string::npos,
         "prepack-B and one-shot allocation cannot be conflated");
  auto invalid_compute_allocation = options;
  invalid_compute_allocation.packing_mode = bench::PackingModeV1::exclude;
  invalid_compute_allocation.allocation_mode =
      bench::AllocationModeV1::include_allocation;
  expect(!bench::validate_options_v1(invalid_compute_allocation, error) &&
             error.find("compute diagnostics require reusable workspace") !=
                 std::string::npos,
         "compute-only packing exclusion cannot include one-shot allocation");

  bench::RunnerPlanV1 zero_workspace_plan;
  zero_workspace_plan.legal = true;
  zero_workspace_plan.selected_variant = "cpu.reference.f32.v1";
  std::uint64_t bytes = 0;
  expect(bench::checked_memory_requirement_v1({4, 5, 6}, zero_workspace_plan,
                                               options, bytes, error) &&
             bytes >= (4 * 6 + 6 * 5 + 4 * 5) * sizeof(float),
         "memory accounting includes all matrices");
  auto tiny_limit = options;
  tiny_limit.maximum_memory_bytes = 16;
  expect(!bench::checked_memory_requirement_v1(
             {4, 5, 6}, zero_workspace_plan, tiny_limit, bytes, error) &&
             error.find("exceeds --max-memory-mib") != std::string::npos,
         "memory cap rejects before allocation");
  expect(!bench::checked_memory_requirement_v1(
             {std::numeric_limits<std::int64_t>::max(),
              std::numeric_limits<std::int64_t>::max(), 2},
             zero_workspace_plan, options, bytes, error),
         "overflowing matrix storage is rejected");

  bench::RunnerPlanV1 workspace_plan = zero_workspace_plan;
  workspace_plan.workspace_bytes = 1024;
  workspace_plan.workspace_alignment = 64;
  auto include_allocation = options;
  include_allocation.allocation_mode =
      bench::AllocationModeV1::include_allocation;
  const std::uint64_t matrix_payload = (4 * 6 + 6 * 5 + 4 * 5) * sizeof(float);
  const std::uint64_t aligned_buffer_overhead =
      2 * UINT64_C(64) + alignof(std::max_align_t);
  const std::uint64_t exact_peak =
      matrix_payload + 3 * aligned_buffer_overhead + 1024 +
      aligned_buffer_overhead;
  expect(bench::checked_memory_requirement_v1(
             {4, 5, 6}, workspace_plan, include_allocation, bytes, error) &&
             bytes == exact_peak,
         "one-shot memory accounting equals one live output/workspace pair and "
         "all AlignedBuffer slack");
  include_allocation.maximum_memory_bytes = exact_peak;
  expect(bench::checked_memory_requirement_v1(
             {4, 5, 6}, workspace_plan, include_allocation, bytes, error),
         "exact one-shot peak is accepted by the memory cap");
  include_allocation.maximum_memory_bytes = exact_peak - 1;
  expect(!bench::checked_memory_requirement_v1(
             {4, 5, 6}, workspace_plan, include_allocation, bytes, error) &&
             error.find("exceeds --max-memory-mib") != std::string::npos,
         "one byte below the exact one-shot peak is rejected");

  const auto statistics =
      bench::summarize_timings_v1({0.002, 0.001, 0.004, 0.003}, 1, 1000);
  expect(statistics.valid && statistics.minimum_seconds == 0.001 &&
             statistics.median_seconds == 0.003 &&
             statistics.p95_seconds == 0.004,
         "timing statistics use sorted minimum, median, and nearest-rank p95");
  const auto too_short =
      bench::summarize_timings_v1({1.0e-7}, 1, 1'000'000);
  expect(!too_short.valid && too_short.minimum_seconds == 1.0e-7 &&
             too_short.rejection_reason.find("timer floor") != std::string::npos,
         "sub-floor timing is retained but rejected for performance claims");

  auto runner = bench::make_planner_runner_v1();
  const auto registered_variants = runner->variant_ids();
  expect(registered_variants.size() == 8 &&
             registered_variants[5] ==
                 "cpu.native-packed.avx512-fma.f32.v1" &&
             registered_variants[6] ==
                 "cpu.native-parallel.avx2-fma.f32.v1" &&
             registered_variants[7] ==
                 "cpu.native-parallel.avx512-fma.f32.v1",
         "benchmark exposes the stable planner-v3 eight-variant registry");
  const auto runner_environment = runner->environment();
  expect(runner_environment.capability_record_version == 2 &&
             runner_environment.topology_record_version == 1 &&
             runner_environment.available_processors > 0 &&
             !runner_environment.worker_affinity_applied &&
             !runner_environment.capability_record.empty() &&
             !runner_environment.topology_record.empty() &&
             !runner_environment.capability_runtime_validation_source.empty(),
         "runner exposes versioned capability, topology, and validation-source metadata");
  const auto unsupported_affinity = runner->plan(
      {256, 128, 128}, 64, 2, "auto", bench::PackingModeV1::include,
      bench::SmtPolicyV2::physical_cores_only,
      bench::AffinityPolicyV2::compact);
  expect(!unsupported_affinity.legal &&
             unsupported_affinity.reason.find("worker-affinity policy") !=
                 std::string::npos,
         "unimplemented worker binding is rejected instead of conflating it with inherited affinity");
  bench::BenchmarkOptionsV1 run_options;
  run_options.profile = bench::ProfileV1::custom;
  run_options.shapes = {{2, 3, 2}};
  run_options.requested_variant = "cpu.reference.f32.v1";
  run_options.warmup_iterations = 0;
  run_options.measured_iterations = 3;
  // This exercises runner correctness and metadata under instrumented builds,
  // not performance acceptance. The dedicated summarize_timings checks above
  // retain explicit timer-floor rejection coverage.
  run_options.timer_floor_nanoseconds = 1;
  run_options.maximum_memory_bytes = 16 * 1024 * 1024;
  bench::BenchmarkReportV1 report;
  expect(bench::run_benchmarks_v1(run_options, *runner, report, error),
         "reference runner completes a bounded benchmark");
  expect(report.results.size() == 1 &&
             report.results[0].plan.selected_variant ==
                 "cpu.reference.f32.v1" &&
             report.results[0].correctness.passed &&
             report.results[0].correctness.oracle_mode == "full-double" &&
             report.results[0].timing.valid &&
             report.results[0].plan.planner_version == 3 &&
             report.results[0].plan.smt_policy == "physical-cores-only" &&
             !report.results[0].plan.worker_affinity_applied,
         "benchmark result identifies the variant and passes independent oracle");

  auto scaling_options = run_options;
  scaling_options.compare_one_thread = true;
  bench::BenchmarkReportV1 scaling_report;
  expect(bench::run_benchmarks_v1(scaling_options, *runner, scaling_report,
                                  error) &&
             scaling_report.results[0].scaling.valid &&
             scaling_report.results[0].scaling.speedup_over_one_thread == 1.0 &&
             scaling_report.results[0].scaling.parallel_efficiency == 1.0,
         "explicit one-thread comparison records a deterministic unit baseline");

  auto regret_options = run_options;
  regret_options.requested_variant = "auto";
  regret_options.planner_regret = true;
  regret_options.measured_iterations = 1;
  bench::BenchmarkReportV1 regret_report;
  expect(bench::run_benchmarks_v1(regret_options, *runner, regret_report,
                                  error) &&
             regret_report.results[0].planner_regret.valid &&
             regret_report.results[0].planner_regret.candidates.size() == 8 &&
             regret_report.results[0].planner_regret.regret >= 1.0,
         "planner regret measures the stable registry and reports selected over fastest");

  const auto rejected_reference_compute = runner->plan(
      {16, 16, 16}, 64, 1, "cpu.reference.f32.v1",
      bench::PackingModeV1::exclude);
  expect(!rejected_reference_compute.legal &&
             rejected_reference_compute.reason.find(
                 "diagnostic implemented only") != std::string::npos,
         "exclude-packing rejects variants without an explicit compute-only "
         "diagnostic path");

  const auto packed_compute_plan = runner->plan(
      {33, 35, 37}, 64, 1, "cpu.native-packed.avx2-fma.f32.v1",
      bench::PackingModeV1::exclude);
  if (packed_compute_plan.legal) {
    expect(packed_compute_plan.workspace_bytes > 0 &&
               !packed_compute_plan.complete_implementation_comparison &&
               packed_compute_plan.timing_scope.find("packed-compute-only") !=
                   std::string::npos &&
               packed_compute_plan.diagnostic.find(
                   "A and B packing prepared before timing") !=
                   std::string::npos,
           "native compute diagnostic declares storage and its non-comparable "
           "timing scope");
    bench::BenchmarkOptionsV1 compute_options = run_options;
    compute_options.shapes = {{1, 1, 1}, {2, 3, 2}, {5, 17, 19},
                              {33, 35, 37}, {127, 129, 131}};
    compute_options.requested_variant =
        "cpu.native-packed.avx2-fma.f32.v1";
    compute_options.packing_mode = bench::PackingModeV1::exclude;
    bench::BenchmarkReportV1 compute_report;
    const bool compute_ran = bench::run_benchmarks_v1(
        compute_options, *runner, compute_report, error);
    bool every_compute_result_passed =
        compute_ran && compute_report.results.size() == 5;
    for (const auto &compute_result : compute_report.results) {
      every_compute_result_passed =
          every_compute_result_passed && compute_result.correctness.passed &&
          compute_result.timing.valid &&
          !compute_result.plan.complete_implementation_comparison;
    }
    expect(every_compute_result_passed,
           "native compute-only diagnostic executes tiny, rectangular, and "
           "tail shapes through the independent oracle");
  }

  const auto parallel_plan = runner->plan(
      {256, 128, 128}, 64, 2,
      "cpu.native-parallel.avx2-fma.f32.v1",
      bench::PackingModeV1::include);
  if (parallel_plan.legal) {
    expect(parallel_plan.actual_threads == 2 &&
               parallel_plan.persistent_execution_context &&
               parallel_plan.shared_workspace_bytes > 0 &&
               parallel_plan.per_worker_workspace_bytes > 0 &&
               parallel_plan.workspace_bytes >=
                   parallel_plan.shared_workspace_bytes +
                       2 * parallel_plan.per_worker_workspace_bytes,
           "parallel AVX2 plan exposes persistent-context and split workspace metadata");
    auto parallel_options = run_options;
    parallel_options.shapes = {{256, 128, 128}};
    parallel_options.requested_variant =
        "cpu.native-parallel.avx2-fma.f32.v1";
    parallel_options.requested_threads = 2;
    parallel_options.measured_iterations = 1;
    parallel_options.compare_one_thread = true;
    bench::BenchmarkReportV1 parallel_report;
    expect(bench::run_benchmarks_v1(parallel_options, *runner,
                                    parallel_report, error) &&
               parallel_report.results[0].correctness.passed &&
               parallel_report.results[0].scaling.valid &&
               parallel_report.results[0].scaling.baseline_variant ==
                   "cpu.native-packed.avx2-fma.f32.v1" &&
               parallel_report.environment.runner.execution_context_submissions >
                   0,
           "persistent parallel AVX2 dispatch is oracle-checked against a same-family one-thread timing");
  }

  std::ostringstream encoded;
  bench::write_json_v1(report, encoded);
  const std::string json = encoded.str();
  expect(json.find("\"schema\": \"matcore.benchmark.cpu.gemm\"") !=
                 std::string::npos &&
             json.find("\"version\": 2") != std::string::npos &&
             json.find("\"correctness\": true") != std::string::npos &&
             json.find("\"planner_version\": 3") != std::string::npos &&
             json.find("\"shared_workspace_bytes\"") != std::string::npos &&
             json.find("\"planner_regret\"") != std::string::npos &&
             json.find("\"timing_scope\"") != std::string::npos &&
             json.find("\"timer_resolution_ns\"") != std::string::npos,
         "JSON output carries schema, correctness, and timer metadata");

  if (failures != 0) return 1;
  std::cout << "matcore benchmark contract PASS\n";
  return 0;
}
