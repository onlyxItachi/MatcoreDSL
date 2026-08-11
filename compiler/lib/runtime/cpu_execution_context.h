#ifndef MATCORE_MDSLC_RUNTIME_CPU_EXECUTION_CONTEXT_H
#define MATCORE_MDSLC_RUNTIME_CPU_EXECUTION_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuExecutionContextVersionV1 = 1;

enum class CpuExecutionStatusV1 : std::uint32_t {
  success = 0,
  invalid_configuration = 1,
  context_stopping = 2,
  nested_parallelism_rejected = 3,
  callback_failed = 4,
  resource_exhausted = 5,
  affinity_unavailable = 6,
  affinity_application_failed = 7,
};

enum class CpuProviderNestingPolicyV1 : std::uint32_t {
  native_only = 0,
  external_provider_active = 1,
};

struct CpuExecutionContextConfigV1 {
  CpuExecutionContextConfigV1() = default;
  CpuExecutionContextConfigV1(std::uint32_t input_version,
                              std::uint32_t input_requested_threads,
                              std::uint32_t input_maximum_threads) noexcept
      : version(input_version),
        requested_threads(input_requested_threads),
        maximum_threads(input_maximum_threads) {}

  std::uint32_t version = kCpuExecutionContextVersionV1;
  std::uint32_t requested_threads = 1;
  // Zero means the caller did not supply a topology ceiling. In that case the
  // explicit request is honored. Callers with topology data should pass the
  // physical-core or policy ceiling here.
  std::uint32_t maximum_threads = 0;

  // Empty means scheduler affinity is not requested. Otherwise this must
  // contain exactly one unique logical CPU ID per actual worker, in worker
  // index order. Explicit affinity is strict: context creation fails if any
  // worker cannot apply its assigned CPU. The list describes CPU scheduling
  // only; it does not request or imply NUMA memory placement.
  std::vector<std::uint32_t> worker_cpu_ids;
};

enum class CpuWorkerAffinityStatusV1 : std::uint8_t {
  not_requested = 0,
  complete = 1,
  invalid_configuration = 2,
  unavailable = 3,
  application_failed = 4,
  partially_applied = 5,
};

struct CpuWorkerAffinityReportV1 {
  std::uint32_t version = kCpuExecutionContextVersionV1;
  CpuWorkerAffinityStatusV1 status =
      CpuWorkerAffinityStatusV1::not_requested;
  std::uint32_t requested_workers = 0;
  std::uint32_t applied_workers = 0;
  std::uint32_t first_failed_worker =
      std::numeric_limits<std::uint32_t>::max();
  std::uint32_t first_failed_cpu =
      std::numeric_limits<std::uint32_t>::max();
  std::int32_t platform_error = 0;
  bool complete = true;
  // Always false in v1. Worker affinity never claims first-touch, allocation,
  // migration, binding, or interleaving of NUMA pages.
  bool numa_memory_placement_applied = false;
};

struct CpuExecutionContextInfoV1 {
  std::uint32_t version = kCpuExecutionContextVersionV1;
  std::uint32_t requested_threads = 0;
  std::uint32_t actual_worker_count = 0;
  std::uint64_t workers_started = 0;
  std::uint64_t completed_submissions = 0;
  bool accepting_work = false;
  CpuWorkerAffinityReportV1 affinity;
};

using CpuExecutionTaskV1 = CpuExecutionStatusV1 (*)(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept;

using CpuExecutionPreflightV1 = CpuExecutionStatusV1 (*)(
    std::size_t worker_index, void *user_data) noexcept;

// A context owns a fixed set of persistent workers. It deliberately owns no
// GEMM packing buffers: per-worker workspaces remain explicit and caller-owned.
// Concurrent submissions to one context are serialized. Independent contexts
// may execute concurrently.
class CpuExecutionContextV1 final {
 public:
  static std::unique_ptr<CpuExecutionContextV1> create(
      const CpuExecutionContextConfigV1 &config,
      CpuExecutionStatusV1 *status,
      CpuWorkerAffinityReportV1 *affinity_report = nullptr) noexcept;

  CpuExecutionContextV1(const CpuExecutionContextV1 &) = delete;
  CpuExecutionContextV1 &operator=(const CpuExecutionContextV1 &) = delete;
  CpuExecutionContextV1(CpuExecutionContextV1 &&) = delete;
  CpuExecutionContextV1 &operator=(CpuExecutionContextV1 &&) = delete;

  ~CpuExecutionContextV1();

  CpuExecutionContextInfoV1 info() const noexcept;

  CpuExecutionStatusV1 run_tasks(
      std::size_t task_count, std::uint32_t active_threads,
      CpuProviderNestingPolicyV1 nesting_policy,
      CpuExecutionTaskV1 task, void *user_data) noexcept;

  // Every active worker runs preflight exactly once. No worker invokes task
  // until every active worker has reached the barrier and every preflight has
  // succeeded. A failed preflight suppresses all task callbacks while keeping
  // the context reusable. This is the execution-legality boundary for
  // thread-local state such as the floating-point environment.
  CpuExecutionStatusV1 run_tasks_with_preflight(
      std::size_t task_count, std::uint32_t active_threads,
      CpuProviderNestingPolicyV1 nesting_policy,
      CpuExecutionPreflightV1 preflight,
      CpuExecutionTaskV1 task, void *user_data) noexcept;

  // Idempotent. A submission already in progress is allowed to finish before
  // worker shutdown. No later submission is accepted.
  void shutdown() noexcept;

 private:
  struct Impl;
  explicit CpuExecutionContextV1(std::unique_ptr<Impl> implementation) noexcept;
  std::unique_ptr<Impl> impl_;
};

const char *cpu_execution_status_message_v1(
    CpuExecutionStatusV1 status) noexcept;

const char *cpu_worker_affinity_status_message_v1(
    CpuWorkerAffinityStatusV1 status) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
