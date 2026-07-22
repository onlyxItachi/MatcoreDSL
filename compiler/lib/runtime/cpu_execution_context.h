#ifndef MATCORE_MDSLC_RUNTIME_CPU_EXECUTION_CONTEXT_H
#define MATCORE_MDSLC_RUNTIME_CPU_EXECUTION_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <memory>

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuExecutionContextVersionV1 = 1;

enum class CpuExecutionStatusV1 : std::uint32_t {
  success = 0,
  invalid_configuration = 1,
  context_stopping = 2,
  nested_parallelism_rejected = 3,
  callback_failed = 4,
  resource_exhausted = 5,
};

enum class CpuProviderNestingPolicyV1 : std::uint32_t {
  native_only = 0,
  external_provider_active = 1,
};

struct CpuExecutionContextConfigV1 {
  std::uint32_t version = kCpuExecutionContextVersionV1;
  std::uint32_t requested_threads = 1;
  // Zero means the caller did not supply a topology ceiling. In that case the
  // explicit request is honored. Callers with topology data should pass the
  // physical-core or policy ceiling here.
  std::uint32_t maximum_threads = 0;
};

struct CpuExecutionContextInfoV1 {
  std::uint32_t version = kCpuExecutionContextVersionV1;
  std::uint32_t requested_threads = 0;
  std::uint32_t actual_worker_count = 0;
  std::uint64_t workers_started = 0;
  std::uint64_t completed_submissions = 0;
  bool accepting_work = false;
};

using CpuExecutionTaskV1 = CpuExecutionStatusV1 (*)(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept;

// A context owns a fixed set of persistent workers. It deliberately owns no
// GEMM packing buffers: per-worker workspaces remain explicit and caller-owned.
// Concurrent submissions to one context are serialized. Independent contexts
// may execute concurrently.
class CpuExecutionContextV1 final {
 public:
  static std::unique_ptr<CpuExecutionContextV1> create(
      const CpuExecutionContextConfigV1 &config,
      CpuExecutionStatusV1 *status) noexcept;

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

}  // namespace matcore::mdslc::runtime

#endif
