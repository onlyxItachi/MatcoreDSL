#include "benchmark.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
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

class RecordingRunner final : public bench::GemmRunnerV1 {
 private:
  struct ExecutionState final : bench::RunnerPlanStateV1 {
    bool fail = false;
    bool corrupt_intermediate_measurement = false;
    mutable std::uint32_t executions = 0;
  };

 public:
  enum class PlanDrift {
    none,
    legal,
    selected_variant,
    reason,
    diagnostic,
    timing_scope,
    comparable,
    planner_version,
    actual_threads,
    workspace_bytes,
    shared_workspace_bytes,
    per_worker_workspace_bytes,
    workspace_alignment,
    prepacked_b_bytes,
    packing_required,
    supports_prepacked_b,
    persistent_context,
    smt_policy,
    affinity_policy,
    worker_affinity_applied,
    worker_affinity_user_requested,
    worker_affinity_policy_induced,
    affinity_diagnostic,
  };

  explicit RecordingRunner(bool fail_reverse_selected = false,
                           PlanDrift drift = PlanDrift::none,
                           bool corrupt_intermediate_measurement = false)
      : fail_reverse_selected_(fail_reverse_selected),
        drift_(drift),
        corrupt_intermediate_measurement_(
            corrupt_intermediate_measurement) {}

  bench::RunnerEnvironmentV1 environment() const override { return {}; }

  std::vector<std::string> variant_ids() const override {
    return {"test.forward-first", "test.selected", "test.reverse-first"};
  }

  bench::RunnerPlanV1 plan(
      const bench::GemmShapeV1 &, std::uint32_t, std::uint32_t,
      std::string_view requested_variant, bench::PackingModeV1,
      bench::SmtPolicyV2, bench::AffinityPolicyV2) const override {
    plan_requests_.emplace_back(requested_variant);
    bench::RunnerPlanV1 result;
    result.legal = true;
    result.selected_variant = requested_variant == "auto"
                                  ? "test.selected"
                                  : std::string(requested_variant);
    result.reason = "recording runner accepts the requested variant";
    result.diagnostic = "deterministic benchmark-order test double";
    result.timing_scope = "complete-call";
    result.complete_implementation_comparison = true;
    result.planner_version = 3;
    result.actual_threads = 1;
    result.workspace_alignment = 1;
    if (requested_variant == "test.selected") {
      ++selected_plan_count_;
      if (selected_plan_count_ == 2) {
        switch (drift_) {
          case PlanDrift::none: break;
          case PlanDrift::legal: result.legal = false; break;
          case PlanDrift::selected_variant:
            result.selected_variant = "test.misattributed";
            break;
          case PlanDrift::reason: result.reason += " drift"; break;
          case PlanDrift::diagnostic: result.diagnostic += " drift"; break;
          case PlanDrift::timing_scope: result.timing_scope += "-drift"; break;
          case PlanDrift::comparable:
            result.complete_implementation_comparison = false;
            break;
          case PlanDrift::planner_version: result.planner_version = 4; break;
          case PlanDrift::actual_threads: result.actual_threads = 2; break;
          case PlanDrift::workspace_bytes: result.workspace_bytes = 64; break;
          case PlanDrift::shared_workspace_bytes:
            result.shared_workspace_bytes = 16;
            break;
          case PlanDrift::per_worker_workspace_bytes:
            result.per_worker_workspace_bytes = 16;
            break;
          case PlanDrift::workspace_alignment:
            result.workspace_alignment = 2;
            break;
          case PlanDrift::prepacked_b_bytes:
            result.prepacked_b_bytes = 16;
            break;
          case PlanDrift::packing_required: result.packing_required = true; break;
          case PlanDrift::supports_prepacked_b:
            result.supports_prepacked_b = true;
            break;
          case PlanDrift::persistent_context:
            result.persistent_execution_context = true;
            break;
          case PlanDrift::smt_policy: result.smt_policy = "drift"; break;
          case PlanDrift::affinity_policy:
            result.affinity_policy = "drift";
            break;
          case PlanDrift::worker_affinity_applied:
            result.worker_affinity_applied = true;
            break;
          case PlanDrift::worker_affinity_user_requested:
            result.worker_affinity_user_requested = true;
            break;
          case PlanDrift::worker_affinity_policy_induced:
            result.worker_affinity_policy_induced = true;
            break;
          case PlanDrift::affinity_diagnostic:
            result.affinity_diagnostic = "drift";
            break;
        }
      }
      auto state = std::make_shared<ExecutionState>();
      state->fail = fail_reverse_selected_ && selected_plan_count_ == 3;
      state->corrupt_intermediate_measurement =
          corrupt_intermediate_measurement_ && selected_plan_count_ == 2;
      result.state = std::move(state);
    }
    return result;
  }

  bool prepare(const bench::RunnerPlanV1 &, const bench::GemmShapeV1 &,
               const float *, const float *, std::span<std::byte>,
               std::span<std::byte>, bool, std::string &) const override {
    return true;
  }

  bool execute(const bench::RunnerPlanV1 &plan,
               const bench::GemmShapeV1 &shape, const float *lhs,
               const float *rhs, float *output, std::span<std::byte>,
               std::span<const std::byte>, bool,
               std::string &error) const override {
    const auto state =
        std::dynamic_pointer_cast<const ExecutionState>(plan.state);
    if (state != nullptr && state->fail) {
      error = "recording runner injected reverse-pass failure";
      return false;
    }
    for (std::int64_t row = 0; row < shape.m; ++row) {
      for (std::int64_t column = 0; column < shape.n; ++column) {
        float sum = 0.0F;
        for (std::int64_t inner = 0; inner < shape.k; ++inner) {
          sum += lhs[row * shape.k + inner] *
                 rhs[inner * shape.n + column];
        }
        output[row * shape.n + column] = sum;
      }
    }
    if (state != nullptr) {
      ++state->executions;
      // With no warmups and a one-nanosecond floor, execution 1 is the probe;
      // execution 3 is the second measured sample. A later sample restores the
      // correct output, so final-output-only validation would miss this.
      if (state->corrupt_intermediate_measurement && state->executions == 3)
        output[0] += 100.0F;
    }
    return true;
  }

  bool synchronize(std::string &) const override { return true; }

  const std::vector<std::string> &plan_requests() const noexcept {
    return plan_requests_;
  }

 private:
  bool fail_reverse_selected_ = false;
  PlanDrift drift_ = PlanDrift::none;
  bool corrupt_intermediate_measurement_ = false;
  mutable std::uint32_t selected_plan_count_ = 0;
  mutable std::vector<std::string> plan_requests_;
};

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
             runner_environment.worker_affinity_applied &&
             !runner_environment.worker_affinity_user_requested &&
             runner_environment.worker_affinity_policy_induced &&
             !runner_environment.capability_record.empty() &&
             !runner_environment.topology_record.empty() &&
             runner_environment.capability_runtime_validation_source.find(
                 "authenticated independently") != std::string::npos,
         "runner exposes versioned capability, topology, and validation-source metadata");
  if (runner_environment.physical_cores >= 2) {
    for (const auto policy : {bench::AffinityPolicyV2::compact,
                              bench::AffinityPolicyV2::scatter,
                              bench::AffinityPolicyV2::local_first}) {
      const auto affinity_plan = runner->plan(
          {256, 128, 128}, 64, 2,
          "cpu.native-parallel.avx2-fma.f32.v1",
          bench::PackingModeV1::include,
          bench::SmtPolicyV2::physical_cores_only, policy);
      expect(affinity_plan.legal && affinity_plan.worker_affinity_applied &&
                 affinity_plan.affinity_diagnostic.find("cpu_ids=[") !=
                     std::string::npos &&
                 affinity_plan.affinity_diagnostic.find(
                     "numa_memory_placement=false") != std::string::npos,
             "explicit affinity uses a strict persistent worker context and "
             "does not claim NUMA memory placement");
    }
    const auto affinity_serial_plan = runner->plan(
        {64, 64, 64}, 64, 2, "cpu.reference.f32.v1",
        bench::PackingModeV1::include,
        bench::SmtPolicyV2::physical_cores_only,
        bench::AffinityPolicyV2::compact);
    expect(affinity_serial_plan.legal &&
               affinity_serial_plan.worker_affinity_applied &&
               affinity_serial_plan.worker_affinity_user_requested &&
               !affinity_serial_plan.worker_affinity_policy_induced &&
               affinity_serial_plan.timing_scope.find(
                   "pinned persistent worker 0") != std::string::npos,
           "serial variants dispatch through pinned worker zero when affinity "
           "is explicitly requested");
    const auto bound_auto_plan = runner->plan(
        {64, 64, 64}, 64, 2, "auto", bench::PackingModeV1::include,
        bench::SmtPolicyV2::physical_cores_only,
        bench::AffinityPolicyV2::compact);
    expect(bound_auto_plan.legal &&
               (bound_auto_plan.selected_variant !=
                    "cpu.external.openblas.f32.v1" ||
                bound_auto_plan.actual_threads == 1),
           "automatic planning permits only single-thread OpenBLAS when "
           "exact bound-worker placement is active");
  }
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
             report.results[0].plan.worker_affinity_applied &&
             !report.results[0].plan.worker_affinity_user_requested &&
             report.results[0].plan.worker_affinity_policy_induced &&
             report.results[0].correctness.measured_executions_checked >= 3 &&
             report.results[0].correctness.validation_scope.find(
                 "every measured execution") != std::string::npos,
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
             regret_report.results[0].planner_regret.regret >= 1.0 &&
             bench::regret_aggregation_method_name_v3(
                 regret_report.results[0].planner_regret.aggregation_method) ==
                 "arithmetic-mean-of-forward-and-reverse-pass-medians",
         "planner regret balances forward/reverse registry passes and reports "
         "selected over fastest");

  RecordingRunner recording_runner;
  auto balanced_options = run_options;
  balanced_options.requested_variant = "auto";
  balanced_options.planner_regret = true;
  balanced_options.measured_iterations = 1;
  bench::BenchmarkReportV1 balanced_report;
  expect(bench::run_benchmarks_v1(balanced_options, recording_runner,
                                  balanced_report, error),
         "recording runner completes balanced planner-regret measurement");
  const std::vector<std::string> expected_plan_order = {
      "auto", "test.forward-first", "test.selected", "test.reverse-first",
      "test.forward-first", "test.selected", "test.reverse-first",
      "test.reverse-first", "test.selected", "test.forward-first"};
  const auto &balanced_regret = balanced_report.results[0].planner_regret;
  const auto selected_candidate = std::find_if(
      balanced_regret.candidates.begin(), balanced_regret.candidates.end(),
      [](const bench::RegretCandidateResultV3 &candidate) {
        return candidate.variant == "test.selected";
      });
  expect(recording_runner.plan_requests() == expected_plan_order,
         "after stable candidate preflight, regret measurements run in a "
         "complete forward pass followed by the exact reverse pass");
  expect(balanced_regret.valid &&
             balanced_regret.candidates.size() == 3 &&
             balanced_regret.candidates[0].variant == "test.forward-first" &&
             balanced_regret.candidates[1].variant == "test.selected" &&
             balanced_regret.candidates[2].variant == "test.reverse-first" &&
             selected_candidate != balanced_regret.candidates.end() &&
             selected_candidate->selected_variant == "test.selected" &&
             selected_candidate->timing_scope == "complete-call" &&
             selected_candidate->actual_threads == 1 &&
             selected_candidate->workspace_alignment == 1 &&
             selected_candidate->plan_authenticated &&
             balanced_regret.selected_balanced_estimate_seconds ==
                 selected_candidate->balanced_estimate_seconds &&
             selected_candidate->forward_pass_median_seconds > 0.0 &&
             selected_candidate->reverse_pass_median_seconds > 0.0 &&
             selected_candidate->balanced_estimate_seconds ==
                 std::midpoint(
                     selected_candidate->forward_pass_median_seconds,
                     selected_candidate->reverse_pass_median_seconds) &&
             selected_candidate->forward_pass_measured_executions_checked >= 1 &&
             selected_candidate->reverse_pass_measured_executions_checked >= 1 &&
             selected_candidate->measurement_reason.find(
                 "forward and reverse stable-registry-pass medians") !=
                 std::string::npos,
         "regret output keeps stable registry order and derives the selected "
         "timing from the same balanced passes as every alternative");
  std::ostringstream balanced_encoded;
  bench::write_json_v1(balanced_report, balanced_encoded);
  const std::string balanced_json = balanced_encoded.str();
  expect(balanced_json.find(
             "\"aggregation_method\": "
             "\"arithmetic-mean-of-forward-and-reverse-pass-medians\"") !=
                 std::string::npos &&
             balanced_json.find("\"forward_pass_median_seconds\"") !=
                 std::string::npos &&
             balanced_json.find("\"reverse_pass_median_seconds\"") !=
                 std::string::npos &&
             balanced_json.find("\"balanced_estimate_seconds\"") !=
                 std::string::npos &&
             balanced_json.find("\"selected_median_seconds\"") ==
                 std::string::npos &&
             balanced_json.find("\"fastest_legal_median_seconds\"") ==
                 std::string::npos,
         "schema v3 names pass medians and the derived balanced estimate "
         "without representing an arithmetic mean as a median");

  RecordingRunner reverse_failure_runner(/*fail_reverse_selected=*/true);
  bench::BenchmarkReportV1 rejected_balanced_report;
  std::string rejected_balanced_error;
  expect(!bench::run_benchmarks_v1(balanced_options, reverse_failure_runner,
                                   rejected_balanced_report,
                                   rejected_balanced_error) &&
             rejected_balanced_error.find(
                 "planner-regret reverse pass candidate failed for "
                 "test.selected") != std::string::npos,
         "a failed reverse candidate pass rejects the planner-regret run with "
         "the pass and variant identified");

  const std::array plan_drift_cases = {
      std::pair{RecordingRunner::PlanDrift::reason, "selection_reason"},
      std::pair{RecordingRunner::PlanDrift::diagnostic, "plan_diagnostic"},
      std::pair{RecordingRunner::PlanDrift::timing_scope, "timing_scope"},
      std::pair{RecordingRunner::PlanDrift::comparable,
                "complete_implementation_comparison"},
      std::pair{RecordingRunner::PlanDrift::planner_version,
                "planner_version"},
      std::pair{RecordingRunner::PlanDrift::actual_threads, "actual_threads"},
      std::pair{RecordingRunner::PlanDrift::workspace_bytes,
                "workspace_bytes"},
      std::pair{RecordingRunner::PlanDrift::shared_workspace_bytes,
                "shared_workspace_bytes"},
      std::pair{RecordingRunner::PlanDrift::per_worker_workspace_bytes,
                "per_worker_workspace_bytes"},
      std::pair{RecordingRunner::PlanDrift::workspace_alignment,
                "workspace_alignment"},
      std::pair{RecordingRunner::PlanDrift::prepacked_b_bytes,
                "prepacked_b_bytes"},
      std::pair{RecordingRunner::PlanDrift::packing_required,
                "packing_required"},
      std::pair{RecordingRunner::PlanDrift::supports_prepacked_b,
                "supports_prepacked_b"},
      std::pair{RecordingRunner::PlanDrift::persistent_context,
                "persistent_execution_context"},
      std::pair{RecordingRunner::PlanDrift::smt_policy, "smt_policy"},
      std::pair{RecordingRunner::PlanDrift::affinity_policy,
                "affinity_policy"},
      std::pair{RecordingRunner::PlanDrift::worker_affinity_applied,
                "worker_affinity_applied"},
      std::pair{RecordingRunner::PlanDrift::worker_affinity_user_requested,
                "worker_affinity_user_requested"},
      std::pair{RecordingRunner::PlanDrift::worker_affinity_policy_induced,
                "worker_affinity_policy_induced"},
      std::pair{RecordingRunner::PlanDrift::affinity_diagnostic,
                "affinity_diagnostic"},
  };
  for (const auto &[drift, field] : plan_drift_cases) {
    RecordingRunner drift_runner(/*fail_reverse_selected=*/false, drift);
    bench::BenchmarkReportV1 drift_report;
    std::string drift_error;
    expect(!bench::run_benchmarks_v1(balanced_options, drift_runner,
                                     drift_report, drift_error) &&
               drift_error.find("pass plan authentication failed") !=
                   std::string::npos &&
               drift_error.find(field) != std::string::npos,
           std::string("recursive regret measurement rejects preflight drift in ") +
               field);
  }

  for (const auto &[drift, expected_error] :
       std::array{
           std::pair{RecordingRunner::PlanDrift::legal,
                     "variant planning failed"},
           std::pair{RecordingRunner::PlanDrift::selected_variant,
                     "instead of requested test.selected"},
       }) {
    RecordingRunner attribution_runner(/*fail_reverse_selected=*/false, drift);
    bench::BenchmarkReportV1 attribution_report;
    std::string attribution_error;
    expect(!bench::run_benchmarks_v1(balanced_options, attribution_runner,
                                     attribution_report, attribution_error) &&
               attribution_error.find(expected_error) != std::string::npos,
           "recursive regret measurement rejects legal/selection "
           "misattribution before accepting a timing");
  }

  auto corruption_options = balanced_options;
  corruption_options.measured_iterations = 3;
  RecordingRunner corruption_runner(
      /*fail_reverse_selected=*/false, RecordingRunner::PlanDrift::none,
      /*corrupt_intermediate_measurement=*/true);
  bench::BenchmarkReportV1 corruption_report;
  std::string corruption_error;
  expect(!bench::run_benchmarks_v1(corruption_options, corruption_runner,
                                   corruption_report, corruption_error) &&
             corruption_error.find("correctness failed for test.selected at "
                                   "measured iteration 1") != std::string::npos,
         "an intermediate corrupt execution is rejected before a later "
         "execution can overwrite it with a correct final output");

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
               parallel_report.environment.runner.worker_affinity_applied &&
               parallel_report.environment.runner
                   .worker_affinity_policy_induced &&
               parallel_report.environment.runner.execution_context_submissions >
                   0,
           "persistent parallel AVX2 dispatch is oracle-checked against a same-family one-thread timing");
  }

  std::ostringstream encoded;
  bench::write_json_v1(report, encoded);
  const std::string json = encoded.str();
  expect(json.find("\"schema\": \"matcore.benchmark.cpu.gemm\"") !=
                 std::string::npos &&
             json.find("\"version\": 3") != std::string::npos &&
             json.find("\"correctness\": true") != std::string::npos &&
             json.find("\"measured_executions_checked\"") !=
                 std::string::npos &&
             json.find("\"correctness_validation_scope\"") !=
                 std::string::npos &&
             json.find("\"planner_version\": 3") != std::string::npos &&
             json.find("\"shared_workspace_bytes\"") != std::string::npos &&
             json.find("\"worker_affinity_policy_induced\"") !=
                 std::string::npos &&
             json.find("\"planner_regret\"") != std::string::npos &&
             json.find("\"timing_scope\"") != std::string::npos &&
             json.find("\"timer_resolution_ns\"") != std::string::npos,
         "JSON output carries schema, correctness, and timer metadata");

  if (failures != 0) return 1;
  std::cout << "matcore benchmark contract PASS\n";
  return 0;
}
