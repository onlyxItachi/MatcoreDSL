#include "cpu_execution_context.h"

#include "thread_affinity_v1.h"

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
thread_local const void *active_execution_context = nullptr;

struct alignas(kWorkerCacheLineBytes) WorkerResultV1 {
  CpuExecutionStatusV1 status = CpuExecutionStatusV1::success;
  std::size_t first_failed_task = std::numeric_limits<std::size_t>::max();
};

static_assert(sizeof(WorkerResultV1) % kWorkerCacheLineBytes == 0);

struct SubmissionV1 {
  std::size_t task_count = 0;
  std::uint32_t active_threads = 0;
  CpuExecutionPreflightV1 preflight = nullptr;
  CpuExecutionTaskV1 task = nullptr;
  void *user_data = nullptr;
  std::uint32_t completed_preflights = 0;
  bool preflight_failed = false;
  std::uint32_t completed_workers = 0;
};

}  // namespace

struct CpuExecutionContextV1::Impl {
  explicit Impl(CpuExecutionContextConfigV1 input_config,
                std::uint32_t worker_count)
      : config(std::move(input_config)),
        results(worker_count),
        affinity_results(worker_count),
        worker_count(worker_count) {
    affinity_report.requested_workers =
        static_cast<std::uint32_t>(config.worker_cpu_ids.size());
    if (!config.worker_cpu_ids.empty()) {
      affinity_report.complete = false;
      affinity_report.status =
          CpuWorkerAffinityStatusV1::application_failed;
    }
  }

  CpuExecutionContextConfigV1 config;
  std::vector<WorkerResultV1> results;
  std::vector<platform::ThreadAffinityApplicationV1> affinity_results;
  std::vector<std::thread> workers;
  std::uint32_t worker_count = 0;
  CpuWorkerAffinityReportV1 affinity_report;

  mutable std::mutex state_mutex;
  std::condition_variable work_available;
  std::condition_variable preflight_complete;
  std::condition_variable submission_complete;
  std::condition_variable worker_startup_complete;
  SubmissionV1 *submission = nullptr;
  std::uint64_t epoch = 0;
  std::uint32_t workers_initialized = 0;
  bool stopping = false;

  // Holding this for a full submission intentionally serializes callers and
  // makes the borrowed callback/user-data lifetime unambiguous.
  std::mutex submission_mutex;
  std::atomic<std::uint64_t> completed_submissions{0};
  std::atomic<std::uint64_t> workers_started{0};

  void worker_loop(std::size_t worker_index) noexcept {
    active_execution_context = this;

    platform::ThreadAffinityApplicationV1 affinity;
    if (!config.worker_cpu_ids.empty()) {
      affinity = platform::apply_current_thread_affinity_v1(
          config.worker_cpu_ids[worker_index]);
    }
    {
      std::lock_guard lock(state_mutex);
      affinity_results[worker_index] = affinity;
      ++workers_initialized;
    }
    workers_started.fetch_add(1, std::memory_order_relaxed);
    worker_startup_complete.notify_one();
    std::uint64_t observed_epoch = 0;
    for (;;) {
      SubmissionV1 *current = nullptr;
      {
        std::unique_lock lock(state_mutex);
        work_available.wait(lock, [&] {
          return stopping || epoch != observed_epoch;
        });
        const bool has_unseen_submission = epoch != observed_epoch;
        if (stopping && !has_unseen_submission) {
          active_execution_context = nullptr;
          return;
        }
        if (has_unseen_submission) observed_epoch = epoch;
        // `submission` is borrowed from the submitter's stack. Only active
        // workers participate in its lifetime barrier. Decide participation
        // while holding state_mutex and never retain the pointer on an
        // inactive worker: the submitter may return as soon as all active
        // workers complete.
        if (has_unseen_submission && submission != nullptr &&
            worker_index < submission->active_threads) {
          current = submission;
        }
        if (stopping && current == nullptr) {
          active_execution_context = nullptr;
          return;
        }
      }

      if (current == nullptr) continue;

      WorkerResultV1 &result = results[worker_index];
      bool execute_tasks = true;
      if (current->preflight != nullptr) {
        const CpuExecutionStatusV1 preflight_status =
            current->preflight(worker_index, current->user_data);
        std::unique_lock lock(state_mutex);
        if (preflight_status != CpuExecutionStatusV1::success) {
          result.status = preflight_status;
          result.first_failed_task = 0;
          current->preflight_failed = true;
        }
        ++current->completed_preflights;
        if (current->completed_preflights == current->active_threads) {
          preflight_complete.notify_all();
        } else {
          preflight_complete.wait(lock, [&] {
            return current->completed_preflights == current->active_threads;
          });
        }
        execute_tasks = !current->preflight_failed;
      }

      if (execute_tasks) {
        for (std::size_t task_index = worker_index;
             task_index < current->task_count;
             task_index += current->active_threads) {
          const CpuExecutionStatusV1 task_status =
              current->task(task_index, worker_index, current->user_data);
          if (task_status != CpuExecutionStatusV1::success) {
            result.status = task_status;
            result.first_failed_task = task_index;
            break;
          }
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

namespace {

bool affinity_configuration_valid(const CpuExecutionContextConfigV1 &config,
                                  std::uint32_t worker_count) {
  if (config.worker_cpu_ids.empty()) return true;
  if (config.worker_cpu_ids.size() != worker_count) return false;
  std::vector<std::uint32_t> ordered = config.worker_cpu_ids;
  std::sort(ordered.begin(), ordered.end());
  return std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end();
}

template <class Implementation>
CpuWorkerAffinityReportV1 summarize_affinity(
    const Implementation &implementation) noexcept {
  CpuWorkerAffinityReportV1 report;
  if (implementation.config.worker_cpu_ids.empty()) return report;

  report.requested_workers = implementation.worker_count;
  report.complete = false;
  bool all_failures_unavailable = true;
  for (std::uint32_t worker = 0; worker < implementation.worker_count;
       ++worker) {
    const platform::ThreadAffinityApplicationV1 &result =
        implementation.affinity_results[worker];
    if (result.status == platform::ThreadAffinityStatusV1::applied) {
      ++report.applied_workers;
      continue;
    }
    if (result.status != platform::ThreadAffinityStatusV1::unavailable) {
      all_failures_unavailable = false;
    }
    if (report.first_failed_worker ==
        std::numeric_limits<std::uint32_t>::max()) {
      report.first_failed_worker = worker;
      report.first_failed_cpu = result.requested_logical_cpu;
      report.platform_error = result.platform_error;
    }
  }

  if (report.applied_workers == report.requested_workers) {
    report.status = CpuWorkerAffinityStatusV1::complete;
    report.complete = true;
  } else if (report.applied_workers != 0) {
    report.status = CpuWorkerAffinityStatusV1::partially_applied;
  } else if (all_failures_unavailable) {
    report.status = CpuWorkerAffinityStatusV1::unavailable;
  } else {
    report.status = CpuWorkerAffinityStatusV1::application_failed;
  }
  return report;
}

}  // namespace

CpuExecutionContextV1::CpuExecutionContextV1(
    std::unique_ptr<Impl> implementation) noexcept
    : impl_(std::move(implementation)) {}

std::unique_ptr<CpuExecutionContextV1> CpuExecutionContextV1::create(
    const CpuExecutionContextConfigV1 &config,
    CpuExecutionStatusV1 *status,
    CpuWorkerAffinityReportV1 *affinity_report) noexcept {
  if (status != nullptr) *status = CpuExecutionStatusV1::invalid_configuration;
  if (affinity_report != nullptr) *affinity_report = {};
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
    if (!affinity_configuration_valid(config, worker_count)) {
      if (affinity_report != nullptr) {
        affinity_report->status =
            CpuWorkerAffinityStatusV1::invalid_configuration;
        affinity_report->requested_workers =
            static_cast<std::uint32_t>(config.worker_cpu_ids.size());
        affinity_report->complete = false;
      }
      return nullptr;
    }
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

    {
      std::unique_lock lock(implementation->state_mutex);
      implementation->worker_startup_complete.wait(lock, [&] {
        return implementation->workers_initialized == worker_count;
      });
    }
    implementation->affinity_report = summarize_affinity(*implementation);
    if (affinity_report != nullptr) {
      *affinity_report = implementation->affinity_report;
    }
    if (!implementation->affinity_report.complete) {
      {
        std::lock_guard lock(implementation->state_mutex);
        implementation->stopping = true;
      }
      implementation->work_available.notify_all();
      for (std::thread &worker : implementation->workers) {
        if (worker.joinable()) worker.join();
      }
      if (status != nullptr) {
        *status = implementation->affinity_report.status ==
                          CpuWorkerAffinityStatusV1::unavailable
                      ? CpuExecutionStatusV1::affinity_unavailable
                      : CpuExecutionStatusV1::affinity_application_failed;
      }
      return nullptr;
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
  result.affinity = impl_->affinity_report;
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
  return run_tasks_with_preflight(task_count, active_threads, nesting_policy,
                                  nullptr, task, user_data);
}

CpuExecutionStatusV1 CpuExecutionContextV1::run_tasks_with_preflight(
    std::size_t task_count, std::uint32_t active_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuExecutionPreflightV1 preflight,
    CpuExecutionTaskV1 task, void *user_data) noexcept {
  if (impl_ == nullptr || task_count == 0 || active_threads == 0 ||
      active_threads > impl_->worker_count || active_threads > task_count ||
      task == nullptr) {
    return CpuExecutionStatusV1::invalid_configuration;
  }
  if (active_execution_context == impl_.get())
    return CpuExecutionStatusV1::invalid_configuration;
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
    submission.preflight = preflight;
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
  if (active_execution_context == impl_.get()) {
    // A task may request stop, but a worker must never join itself or wait on
    // the submission mutex held by its submitter. The owning caller can later
    // call shutdown again to join the now-stopped workers.
    {
      std::lock_guard state_lock(impl_->state_mutex);
      impl_->stopping = true;
    }
    impl_->work_available.notify_all();
    return;
  }
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
    case CpuExecutionStatusV1::affinity_unavailable:
      return "CPU worker affinity is unavailable on this platform";
    case CpuExecutionStatusV1::affinity_application_failed:
      return "CPU worker affinity could not be applied completely";
  }
  return "unknown CPU execution-context status";
}

const char *cpu_worker_affinity_status_message_v1(
    CpuWorkerAffinityStatusV1 status) noexcept {
  switch (status) {
    case CpuWorkerAffinityStatusV1::not_requested:
      return "worker affinity was not requested";
    case CpuWorkerAffinityStatusV1::complete:
      return "worker affinity was applied completely";
    case CpuWorkerAffinityStatusV1::invalid_configuration:
      return "worker affinity configuration is invalid";
    case CpuWorkerAffinityStatusV1::unavailable:
      return "worker affinity is unavailable on this platform";
    case CpuWorkerAffinityStatusV1::application_failed:
      return "worker affinity application failed";
    case CpuWorkerAffinityStatusV1::partially_applied:
      return "worker affinity was only partially applied";
  }
  return "unknown worker-affinity status";
}

}  // namespace matcore::mdslc::runtime
