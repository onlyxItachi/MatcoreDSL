#ifndef MATCORE_MDSLC_TOOLS_MATCORE_BENCH_BENCHMARK_H
#define MATCORE_MDSLC_TOOLS_MATCORE_BENCH_BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace matcore::mdslc::bench {

inline constexpr std::uint32_t kBenchmarkSchemaVersionV2 = 2;
inline constexpr std::uint64_t kDefaultMaximumMemoryBytes =
    UINT64_C(2) * 1024 * 1024 * 1024;
inline constexpr std::uint64_t kDefaultTimerFloorNanoseconds = 1'000'000;

enum class CacheModeV1 : std::uint8_t { hot = 0, cold = 1 };
enum class AllocationModeV1 : std::uint8_t {
  reuse_workspace = 0,
  include_allocation = 1,
};
enum class PackingModeV1 : std::uint8_t {
  include = 0,
  exclude = 1,
  prepack_b = 2,
};
enum class ProfileV1 : std::uint8_t { custom = 0, quick = 1, standard = 2, full = 3 };
enum class SmtPolicyV2 : std::uint8_t {
  physical_cores_only = 0,
  allow_smt = 1,
};
enum class AffinityPolicyV2 : std::uint8_t {
  none = 0,
  compact = 1,
  scatter = 2,
  local_first = 3,
};

struct GemmShapeV1 {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
};

struct BenchmarkOptionsV1 {
  std::vector<GemmShapeV1> shapes;
  ProfileV1 profile = ProfileV1::quick;
  std::string requested_variant = "auto";
  std::uint32_t requested_threads = 1;
  std::uint32_t warmup_iterations = 2;
  std::uint32_t measured_iterations = 9;
  std::uint32_t alignment_bytes = 64;
  CacheModeV1 cache_mode = CacheModeV1::hot;
  AllocationModeV1 allocation_mode = AllocationModeV1::reuse_workspace;
  PackingModeV1 packing_mode = PackingModeV1::include;
  SmtPolicyV2 smt_policy = SmtPolicyV2::physical_cores_only;
  AffinityPolicyV2 affinity_policy = AffinityPolicyV2::none;
  std::uint64_t maximum_memory_bytes = kDefaultMaximumMemoryBytes;
  std::uint64_t timer_floor_nanoseconds = kDefaultTimerFloorNanoseconds;
  std::uint64_t seed = UINT64_C(0x4d4154434f524531);
  bool guard = false;
  bool compare_one_thread = false;
  bool planner_regret = false;
  std::string json_output;
};

struct RunnerPlanStateV1 {
  virtual ~RunnerPlanStateV1() = default;
};

struct RunnerPlanV1 {
  bool legal = false;
  std::string selected_variant;
  std::string reason;
  std::string diagnostic;
  std::string timing_scope;
  bool complete_implementation_comparison = true;
  std::uint32_t planner_version = 0;
  std::uint32_t actual_threads = 1;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t shared_workspace_bytes = 0;
  std::uint64_t per_worker_workspace_bytes = 0;
  std::uint32_t workspace_alignment = 1;
  std::uint64_t prepacked_b_bytes = 0;
  bool packing_required = false;
  bool supports_prepacked_b = false;
  bool persistent_execution_context = false;
  std::string smt_policy;
  std::string affinity_policy;
  bool worker_affinity_applied = false;
  bool worker_affinity_user_requested = false;
  bool worker_affinity_policy_induced = false;
  std::string affinity_diagnostic;
  std::shared_ptr<const RunnerPlanStateV1> state;
};

struct RunnerEnvironmentV1 {
  std::string capability_record;
  std::string capability_runtime_validation_source;
  std::string topology_record;
  std::uint32_t capability_record_version = 0;
  std::uint32_t topology_record_version = 0;
  bool topology_discovery_complete = false;
  std::uint32_t logical_processors = 0;
  std::uint32_t physical_cores = 0;
  std::uint32_t numa_nodes = 0;
  bool persistent_execution_context = false;
  std::uint32_t execution_context_workers = 0;
  std::uint64_t execution_context_workers_started = 0;
  std::uint64_t execution_context_submissions = 0;
  std::uint32_t available_processors = 0;
  bool worker_affinity_applied = false;
  bool worker_affinity_user_requested = false;
  bool worker_affinity_policy_induced = false;
  std::string worker_affinity_source;
  std::string provider_name;
  std::string provider_version;
  std::string provider_config;
};

/*
 * Internal benchmark hook. It is deliberately not part of the installed C
 * ABI. Implementations own no caller buffers and must not allocate inside
 * execute(). A packed implementation exposes all storage through RunnerPlanV1.
 */
class GemmRunnerV1 {
 public:
  virtual ~GemmRunnerV1() = default;
  virtual RunnerEnvironmentV1 environment() const = 0;
  virtual std::vector<std::string> variant_ids() const = 0;
  virtual RunnerPlanV1 plan(const GemmShapeV1 &shape,
                            std::uint32_t minimum_alignment,
                            std::uint32_t requested_threads,
                            std::string_view requested_variant,
                            PackingModeV1 packing_mode,
                            SmtPolicyV2 smt_policy =
                                SmtPolicyV2::physical_cores_only,
                            AffinityPolicyV2 affinity_policy =
                                AffinityPolicyV2::none) const = 0;
  virtual bool prepare(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
                       const float *lhs, const float *rhs,
                       std::span<std::byte> workspace,
                       std::span<std::byte> prepacked_b_storage,
                       bool prepack_b, std::string &error) const = 0;
  virtual bool execute(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
                       const float *lhs, const float *rhs, float *output,
                       std::span<std::byte> workspace,
                       std::span<const std::byte> prepacked_b_storage,
                       bool packing_is_prepared,
                       std::string &error) const = 0;
  virtual bool synchronize(std::string &error) const = 0;
};

struct TimingStatisticsV1 {
  bool valid = false;
  std::string rejection_reason;
  std::uint64_t aggregate_repetitions = 1;
  double minimum_seconds = 0.0;
  double median_seconds = 0.0;
  double p95_seconds = 0.0;
};

struct CorrectnessResultV1 {
  bool passed = false;
  std::string oracle_mode;
  std::uint64_t checked_elements = 0;
  double checksum = 0.0;
  double expected_checksum = 0.0;
  double maximum_absolute_error = 0.0;
  double maximum_allowed_error = 0.0;
  std::string reason;
};

struct ScalingResultV2 {
  bool requested = false;
  bool valid = false;
  std::string baseline_variant;
  double one_thread_median_seconds = 0.0;
  double speedup_over_one_thread = 0.0;
  double parallel_efficiency = 0.0;
  std::string reason;
};

struct RegretCandidateResultV2 {
  std::string variant;
  bool legal = false;
  std::string reason;
  bool complete_implementation_comparison = false;
  bool timing_valid = false;
  bool correctness_passed = false;
  double median_seconds = 0.0;
  std::string measurement_reason;
};

struct PlannerRegretResultV2 {
  bool requested = false;
  bool valid = false;
  std::string fastest_legal_variant;
  double fastest_legal_median_seconds = 0.0;
  double selected_median_seconds = 0.0;
  double regret = 0.0;
  std::string reason;
  std::vector<RegretCandidateResultV2> candidates;
};

struct BenchmarkResultV1 {
  GemmShapeV1 shape;
  std::string requested_variant;
  RunnerPlanV1 plan;
  std::uint64_t estimated_memory_bytes = 0;
  CacheModeV1 cache_mode = CacheModeV1::hot;
  AllocationModeV1 allocation_mode = AllocationModeV1::reuse_workspace;
  PackingModeV1 packing_mode = PackingModeV1::include;
  TimingStatisticsV1 timing;
  CorrectnessResultV1 correctness;
  double gflops = 0.0;
  ScalingResultV2 scaling;
  PlannerRegretResultV2 planner_regret;
};

struct BenchmarkEnvironmentV1 {
  std::string os_family;
  std::string architecture;
  std::string compiler;
  std::string compiler_flags;
  std::string build_type;
  std::string cpu_model;
  std::string governor;
  std::string frequency_policy;
  std::string boost_state;
  std::string cpu_affinity;
  std::uint32_t hardware_threads = 0;
  std::string source_commit;
  std::string timer_source;
  std::uint64_t timer_resolution_nanoseconds = 0;
  std::int64_t timestamp_unix_seconds = 0;
  RunnerEnvironmentV1 runner;
};

struct BenchmarkReportV1 {
  std::uint32_t schema_version = kBenchmarkSchemaVersionV2;
  std::string operation = "matcore.gemm";
  std::string dtype = "f32";
  std::string accumulation_dtype = "f32";
  std::string layout = "row-major-contiguous";
  BenchmarkOptionsV1 options;
  BenchmarkEnvironmentV1 environment;
  std::vector<BenchmarkResultV1> results;
};

std::vector<GemmShapeV1> quick_profile_v1();
std::vector<GemmShapeV1> standard_profile_v1();
std::vector<GemmShapeV1> full_profile_v1();

bool validate_options_v1(BenchmarkOptionsV1 &options, std::string &error);
bool checked_memory_requirement_v1(const GemmShapeV1 &shape,
                                   const RunnerPlanV1 &plan,
                                   const BenchmarkOptionsV1 &options,
                                   std::uint64_t &required_bytes,
                                   std::string &error);
TimingStatisticsV1 summarize_timings_v1(std::vector<double> samples_seconds,
                                        std::uint64_t aggregate_repetitions,
                                        std::uint64_t timer_floor_nanoseconds);
BenchmarkEnvironmentV1 discover_benchmark_environment_v1(
    const RunnerEnvironmentV1 &runner_environment);
bool run_benchmarks_v1(const BenchmarkOptionsV1 &options,
                       const GemmRunnerV1 &runner,
                       BenchmarkReportV1 &report, std::string &error);
void write_json_v1(const BenchmarkReportV1 &report, std::ostream &output);

std::unique_ptr<GemmRunnerV1> make_planner_runner_v1();

std::string_view cache_mode_name_v1(CacheModeV1 mode) noexcept;
std::string_view allocation_mode_name_v1(AllocationModeV1 mode) noexcept;
std::string_view packing_mode_name_v1(PackingModeV1 mode) noexcept;
std::string_view profile_name_v1(ProfileV1 profile) noexcept;
std::string_view smt_policy_name_v2(SmtPolicyV2 policy) noexcept;
std::string_view affinity_policy_name_v2(AffinityPolicyV2 policy) noexcept;

}  // namespace matcore::mdslc::bench

#endif
