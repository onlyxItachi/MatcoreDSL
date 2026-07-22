#include "cpu_execution_context.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace matcore::mdslc::runtime {
namespace {

inline constexpr std::size_t kWorkerCacheLineBytes = 64;

struct alignas(kWorkerCacheLineBytes) WorkerResultV1 {
  CpuExecutionStatusV1 status = CpuExecutionStatusV1::success;
  std::size_t first_failed_task = std::numeric_limits<std::size_t>::max();
};

static_assert(sizeof(WorkerResultV1) % kWorkerCacheLineBytes == 0);

struct SubmissionV1 {
  std::size_t task_count = 0;
  std::uint32_t active_threads = 0;
  CpuExecutionTaskV1 task = nullptr;
  void *user_data = nullptr;
  std::uint32_t completed_workers = 0;
};

}  // namespace

struct CpuExecutionContextV1::Impl {
  explicit Impl(CpuExecutionContextConfigV1 input_config,
                std::uint32_t worker_count)
      : config(input_config), results(worker_count), worker_count(worker_count) {}

  CpuExecutionContextConfigV1 config;
  std::vector<WorkerResultV1> results;
  std::vector<std::thread> workers;
  std::uint32_t worker_count = 0;

  mutable std::mutex state_mutex;
  std::condition_variable work_available;
  std::condition_variable submission_complete;
  SubmissionV1 *submission = nullptr;
  std::uint64_t epoch = 0;
  bool stopping = false;

  // Holding this for a full submission intentionally serializes callers and
  // makes the borrowed callback/user-data lifetime unambiguous.
  std::mutex submission_mutex;
  std::atomic<std::uint64_t> completed_submissions{0};
  std::atomic<std::uint64_t> workers_started{0};

  void worker_loop(std::size_t worker_index) noexcept {
    workers_started.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t observed_epoch = 0;
    for (;;) {
      SubmissionV1 *current = nullptr;
      {
        std::unique_lock lock(state_mutex);
        work_available.wait(lock, [&] {
          return stopping || epoch != observed_epoch;
        });
        if (stopping) return;
        observed_epoch = epoch;
        current = submission;
      }

      if (current == nullptr || worker_index >= current->active_threads) {
        continue;
      }

      WorkerResultV1 &result = results[worker_index];
      for (std::size_t task_index = worker_index;
           task_index < current->task_count;
           task_index += current->active_threads) {
        const CpuExecutionStatusV1 status =
            current->task(task_index, worker_index, current->user_data);
        if (status != CpuExecutionStatusV1::success) {
          result.status = status;
          result.first_failed_task = task_index;
          break;
        }
      }

      {
        std::lock_guard lock(state_mutex);
        ++current->completed_workers;
        if (current->completed_workers == current->active_threads) {
          submission_complete.notify_one();
        }
      }
    }
  }
};

CpuExecutionContextV1::CpuExecutionContextV1(
    std::unique_ptr<Impl> implementation) noexcept
    : impl_(std::move(implementation)) {}

std::unique_ptr<CpuExecutionContextV1> CpuExecutionContextV1::create(
    const CpuExecutionContextConfigV1 &config,
    CpuExecutionStatusV1 *status) noexcept {
  if (status != nullptr) *status = CpuExecutionStatusV1::invalid_configuration;
  if (config.version != kCpuExecutionContextVersionV1 ||
      config.requested_threads == 0) {
    return nullptr;
  }
  const std::uint32_t ceiling =
      config.maximum_threads == 0 ? config.requested_threads
                                  : config.maximum_threads;
  const std::uint32_t worker_count =
      std::min(config.requested_threads, ceiling);
  if (worker_count == 0) return nullptr;

  try {
    auto implementation = std::make_unique<Impl>(config, worker_count);
    try {
      implementation->workers.reserve(worker_count);
      for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        implementation->workers.emplace_back(
            [raw = implementation.get(), worker] { raw->worker_loop(worker); });
      }
    } catch (...) {
      {
        std::lock_guard lock(implementation->state_mutex);
        implementation->stopping = true;
      }
      implementation->work_available.notify_all();
      for (std::thread &worker : implementation->workers) {
        if (worker.joinable()) worker.join();
      }
      throw;
    }
    auto result = std::unique_ptr<CpuExecutionContextV1>(
        new CpuExecutionContextV1(std::move(implementation)));
    if (status != nullptr) *status = CpuExecutionStatusV1::success;
    return result;
  } catch (...) {
    if (status != nullptr) *status = CpuExecutionStatusV1::resource_exhausted;
    return nullptr;
  }
}

CpuExecutionContextV1::~CpuExecutionContextV1() { shutdown(); }

CpuExecutionContextInfoV1 CpuExecutionContextV1::info() const noexcept {
  CpuExecutionContextInfoV1 result;
  if (impl_ == nullptr) return result;
  result.requested_threads = impl_->config.requested_threads;
  result.actual_worker_count = impl_->worker_count;
  result.workers_started =
      impl_->workers_started.load(std::memory_order_relaxed);
  result.completed_submissions =
      impl_->completed_submissions.load(std::memory_order_relaxed);
  {
    std::lock_guard lock(impl_->state_mutex);
    result.accepting_work = !impl_->stopping;
  }
  return result;
}

CpuExecutionStatusV1 CpuExecutionContextV1::run_tasks(
    std::size_t task_count, std::uint32_t active_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuExecutionTaskV1 task, void *user_data) noexcept {
  if (impl_ == nullptr || task_count == 0 || active_threads == 0 ||
      active_threads > impl_->worker_count || active_threads > task_count ||
      task == nullptr) {
    return CpuExecutionStatusV1::invalid_configuration;
  }
  if (nesting_policy == CpuProviderNestingPolicyV1::external_provider_active &&
      active_threads > 1) {
    return CpuExecutionStatusV1::nested_parallelism_rejected;
  }
  if (nesting_policy != CpuProviderNestingPolicyV1::native_only &&
      nesting_policy != CpuProviderNestingPolicyV1::external_provider_active) {
    return CpuExecutionStatusV1::invalid_configuration;
  }

  try {
    std::unique_lock submission_lock(impl_->submission_mutex);
    SubmissionV1 submission;
    submission.task_count = task_count;
    submission.active_threads = active_threads;
    submission.task = task;
    submission.user_data = user_data;

    {
      std::lock_guard state_lock(impl_->state_mutex);
      if (impl_->stopping) return CpuExecutionStatusV1::context_stopping;
      for (std::uint32_t worker = 0; worker < active_threads; ++worker) {
        impl_->results[worker].status = CpuExecutionStatusV1::success;
        impl_->results[worker].first_failed_task =
            std::numeric_limits<std::size_t>::max();
      }
      impl_->submission = &submission;
      ++impl_->epoch;
    }
    impl_->work_available.notify_all();

    {
      std::unique_lock state_lock(impl_->state_mutex);
      impl_->submission_complete.wait(state_lock, [&] {
        return submission.completed_workers == submission.active_threads;
      });
      impl_->submission = nullptr;
    }
    impl_->completed_submissions.fetch_add(1, std::memory_order_relaxed);

    CpuExecutionStatusV1 result = CpuExecutionStatusV1::success;
    std::size_t first_failed_task = std::numeric_limits<std::size_t>::max();
    for (std::uint32_t worker = 0; worker < active_threads; ++worker) {
      if (impl_->results[worker].first_failed_task < first_failed_task) {
        first_failed_task = impl_->results[worker].first_failed_task;
        result = impl_->results[worker].status;
      }
    }
    return result;
  } catch (...) {
    return CpuExecutionStatusV1::resource_exhausted;
  }
}

void CpuExecutionContextV1::shutdown() noexcept {
  if (impl_ == nullptr) return;
  try {
    std::unique_lock submission_lock(impl_->submission_mutex);
    {
      std::lock_guard state_lock(impl_->state_mutex);
      if (impl_->stopping && impl_->workers.empty()) return;
      impl_->stopping = true;
    }
    impl_->work_available.notify_all();
    for (std::thread &worker : impl_->workers) {
      if (worker.joinable()) worker.join();
    }
    impl_->workers.clear();
  } catch (...) {
    // std::thread::join cannot fail for a joinable thread in this ownership
    // model. The catch keeps destruction noexcept if a platform reports a
    // system error nevertheless.
  }
}

const char *cpu_execution_status_message_v1(
    CpuExecutionStatusV1 status) noexcept {
  switch (status) {
    case CpuExecutionStatusV1::success:
      return "success";
    case CpuExecutionStatusV1::invalid_configuration:
      return "invalid CPU execution-context configuration";
    case CpuExecutionStatusV1::context_stopping:
      return "CPU execution context is stopping";
    case CpuExecutionStatusV1::nested_parallelism_rejected:
      return "nested native/provider parallelism is prohibited";
    case CpuExecutionStatusV1::callback_failed:
      return "CPU execution task failed";
    case CpuExecutionStatusV1::resource_exhausted:
      return "CPU execution-context resource allocation failed";
  }
  return "unknown CPU execution-context status";
}

}  // namespace matcore::mdslc::runtime
