#include "cpu_execution_context.h"
#include "thread_affinity_v1.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace runtime = matcore::mdslc::runtime;
namespace platform = matcore::mdslc::platform;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

struct AssignmentState {
  std::vector<std::size_t> owners;
  std::size_t fail_task = std::numeric_limits<std::size_t>::max();
};

runtime::CpuExecutionStatusV1 record_assignment(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept {
  auto &state = *static_cast<AssignmentState *>(user_data);
  state.owners[task_index] = worker_index;
  return task_index == state.fail_task
             ? runtime::CpuExecutionStatusV1::callback_failed
             : runtime::CpuExecutionStatusV1::success;
}

struct ReentrantState {
  runtime::CpuExecutionContextV1 *context = nullptr;
  runtime::CpuExecutionStatusV1 nested =
      runtime::CpuExecutionStatusV1::success;
  bool request_shutdown = false;
};

struct ConcurrentShutdownState {
  runtime::CpuExecutionContextV1 *context = nullptr;
  std::atomic<std::uint32_t> callbacks{0};
};

struct BorrowedStackState {
  std::uint64_t expected = 0;
  std::uint64_t observed = 0;
};

runtime::CpuExecutionStatusV1 observe_borrowed_stack_state(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept {
  if (task_index != 0 || worker_index != 0 || user_data == nullptr)
    return runtime::CpuExecutionStatusV1::invalid_configuration;
  auto &state = *static_cast<BorrowedStackState *>(user_data);
  state.observed = state.expected;
  return runtime::CpuExecutionStatusV1::success;
}

#if defined(__linux__)
struct AffinityObservationState {
  explicit AffinityObservationState(std::uint32_t workers)
      : observed_cpus(workers,
                      std::numeric_limits<std::uint32_t>::max()) {}

  std::vector<std::uint32_t> observed_cpus;
  std::atomic<bool> complete{true};
};

runtime::CpuExecutionStatusV1 observe_worker_affinity(
    std::size_t, std::size_t worker_index, void *user_data) noexcept {
  auto &state = *static_cast<AffinityObservationState *>(user_data);
  try {
    const auto inventory = platform::discover_current_thread_affinity_v1();
    if (!inventory.discovery_complete ||
        inventory.allowed_logical_cpus.size() != 1) {
      state.complete.store(false, std::memory_order_relaxed);
      return runtime::CpuExecutionStatusV1::callback_failed;
    }
    state.observed_cpus[worker_index] =
        inventory.allowed_logical_cpus.front();
    return runtime::CpuExecutionStatusV1::success;
  } catch (...) {
    state.complete.store(false, std::memory_order_relaxed);
    return runtime::CpuExecutionStatusV1::resource_exhausted;
  }
}
#endif

runtime::CpuExecutionStatusV1 reentrant_callback(
    std::size_t, std::size_t, void *user_data) noexcept {
  auto &state = *static_cast<ReentrantState *>(user_data);
  if (state.request_shutdown) {
    state.context->shutdown();
  } else {
    AssignmentState nested_state{std::vector<std::size_t>(1)};
    state.nested = state.context->run_tasks(
        1, 1, runtime::CpuProviderNestingPolicyV1::native_only,
        record_assignment, &nested_state);
  }
  return runtime::CpuExecutionStatusV1::success;
}

runtime::CpuExecutionStatusV1 concurrent_shutdown_callback(
    std::size_t, std::size_t, void *user_data) noexcept {
  auto &state = *static_cast<ConcurrentShutdownState *>(user_data);
  state.context->shutdown();
  state.callbacks.fetch_add(1, std::memory_order_relaxed);
  return runtime::CpuExecutionStatusV1::success;
}

bool wait_for_workers(runtime::CpuExecutionContextV1 &context,
                      std::uint32_t count) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    if (context.info().workers_started == count) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

std::unique_ptr<runtime::CpuExecutionContextV1> make_context(
    std::uint32_t requested, std::uint32_t maximum = 0) {
  runtime::CpuExecutionStatusV1 status{};
  auto context = runtime::CpuExecutionContextV1::create(
      {runtime::kCpuExecutionContextVersionV1, requested, maximum}, &status);
  expect(status == runtime::CpuExecutionStatusV1::success && context != nullptr,
         "execution context creation succeeds");
  return context;
}

void configuration_and_static_distribution() {
  runtime::CpuExecutionStatusV1 status{};
  auto invalid = runtime::CpuExecutionContextV1::create(
      {runtime::kCpuExecutionContextVersionV1, 0, 0}, &status);
  expect(invalid == nullptr &&
             status == runtime::CpuExecutionStatusV1::invalid_configuration,
         "zero-thread context is rejected");

  auto context = make_context(8, 4);
  expect(context != nullptr && wait_for_workers(*context, 4),
         "topology ceiling creates exactly four persistent workers");
  const auto initial = context->info();
  expect(initial.requested_threads == 8 && initial.actual_worker_count == 4 &&
             initial.workers_started == 4 && initial.accepting_work,
         "context reports requested and actual worker counts");
  expect(initial.affinity.status ==
                 runtime::CpuWorkerAffinityStatusV1::not_requested &&
             initial.affinity.complete &&
             initial.affinity.requested_workers == 0 &&
             initial.affinity.applied_workers == 0 &&
             !initial.affinity.numa_memory_placement_applied,
         "unbound context reports no affinity and no NUMA memory placement");

  AssignmentState state{std::vector<std::size_t>(13,
                                                  std::numeric_limits<std::size_t>::max())};
  const auto result = context->run_tasks(
      state.owners.size(), 4,
      runtime::CpuProviderNestingPolicyV1::native_only, record_assignment,
      &state);
  expect(result == runtime::CpuExecutionStatusV1::success,
         "static task submission succeeds");
  for (std::size_t task = 0; task < state.owners.size(); ++task) {
    expect(state.owners[task] == task % 4,
           "task ownership is deterministic cyclic distribution");
  }

  const auto workers_started = context->info().workers_started;
  for (int repetition = 0; repetition < 50; ++repetition) {
    std::fill(state.owners.begin(), state.owners.end(),
              std::numeric_limits<std::size_t>::max());
    expect(context->run_tasks(
               state.owners.size(), 4,
               runtime::CpuProviderNestingPolicyV1::native_only,
               record_assignment, &state) ==
               runtime::CpuExecutionStatusV1::success,
           "repeated submission succeeds");
  }
  const auto repeated = context->info();
  expect(repeated.workers_started == workers_started,
         "repeated execution does not recreate workers");
  expect(repeated.completed_submissions == 51,
         "context counts completed submissions");

  state.fail_task = 6;
  expect(context->run_tasks(13, 4,
                            runtime::CpuProviderNestingPolicyV1::native_only,
                            record_assignment, &state) ==
             runtime::CpuExecutionStatusV1::callback_failed,
         "worker callback error propagates without deadlock");
  state.fail_task = std::numeric_limits<std::size_t>::max();
  expect(context->run_tasks(13, 4,
                            runtime::CpuProviderNestingPolicyV1::native_only,
                            record_assignment, &state) ==
             runtime::CpuExecutionStatusV1::success,
         "context remains reusable after task error");

  expect(context->run_tasks(
             4, 4,
             runtime::CpuProviderNestingPolicyV1::external_provider_active,
             record_assignment, &state) ==
             runtime::CpuExecutionStatusV1::nested_parallelism_rejected,
         "native/provider nested parallelism is rejected");
  expect(context->run_tasks(
             4, 1,
             runtime::CpuProviderNestingPolicyV1::external_provider_active,
             record_assignment, &state) ==
             runtime::CpuExecutionStatusV1::success,
         "single worker remains legal with external provider active");

  context->shutdown();
  context->shutdown();
  expect(!context->info().accepting_work,
         "shutdown is idempotent and reported");
  expect(context->run_tasks(1, 1,
                            runtime::CpuProviderNestingPolicyV1::native_only,
                            record_assignment, &state) ==
             runtime::CpuExecutionStatusV1::context_stopping,
         "stopped context rejects later submissions");
}

void inactive_workers_do_not_retain_borrowed_submissions() {
  auto context = make_context(8, 8);
  expect(context != nullptr && wait_for_workers(*context, 8),
         "lifetime stress starts eight persistent workers");
  if (context == nullptr) return;

  constexpr std::uint64_t repetitions = 4096;
  bool completed = true;
  for (std::uint64_t generation = 1; generation <= repetitions;
       ++generation) {
    BorrowedStackState state{generation, 0};
    const auto result = context->run_tasks(
        1, 1, runtime::CpuProviderNestingPolicyV1::native_only,
        observe_borrowed_stack_state, &state);
    if (result != runtime::CpuExecutionStatusV1::success ||
        state.observed != generation) {
      completed = false;
      break;
    }

    // Periodically activate every worker. This both exercises alternating
    // participant counts and forces workers that were inactive in prior
    // submissions to observe a later epoch while previous stack payloads are
    // already out of scope.
    if ((generation % 64U) == 0U) {
      AssignmentState barrier{std::vector<std::size_t>(8)};
      if (context->run_tasks(
              barrier.owners.size(), 8,
              runtime::CpuProviderNestingPolicyV1::native_only,
              record_assignment, &barrier) !=
          runtime::CpuExecutionStatusV1::success) {
        completed = false;
        break;
      }
    }
  }
  expect(completed,
         "inactive workers never retain stack submissions past run_tasks");
  expect(context->info().completed_submissions ==
             repetitions + repetitions / 64U,
         "alternating active-thread stress completes every submission");
}

void explicit_worker_affinity_is_strict_and_reported() {
  const auto inventory = platform::discover_current_thread_affinity_v1();
#if defined(__linux__) || defined(_WIN32)
  expect(inventory.backend_available && inventory.discovery_complete &&
             !inventory.allowed_logical_cpus.empty(),
         "execution-context affinity test has a complete host CPU mask");
  if (!inventory.discovery_complete ||
      inventory.allowed_logical_cpus.empty()) {
    return;
  }

  const std::uint32_t successful_workers =
      inventory.allowed_logical_cpus.size() >= 2 ? 2U : 1U;
  runtime::CpuExecutionContextConfigV1 config;
  config.requested_threads = successful_workers;
  config.maximum_threads = successful_workers;
  config.worker_cpu_ids.assign(inventory.allowed_logical_cpus.begin(),
                               inventory.allowed_logical_cpus.begin() +
                                   successful_workers);

  runtime::CpuExecutionStatusV1 status{};
  runtime::CpuWorkerAffinityReportV1 creation_report;
  auto context = runtime::CpuExecutionContextV1::create(
      config, &status, &creation_report);
  expect(context != nullptr && status == runtime::CpuExecutionStatusV1::success,
         "strict explicit worker affinity creates a persistent context");
  if (context == nullptr) return;

  const auto info = context->info();
  expect(creation_report.status ==
                 runtime::CpuWorkerAffinityStatusV1::complete &&
             creation_report.complete &&
             creation_report.requested_workers == successful_workers &&
             creation_report.applied_workers == successful_workers &&
             !creation_report.numa_memory_placement_applied &&
             info.affinity.status == creation_report.status &&
             info.affinity.applied_workers == successful_workers,
         "creation and context diagnostics report complete scheduler affinity only");

  AffinityObservationState observed(successful_workers);
  expect(context->run_tasks(
             successful_workers, successful_workers,
             runtime::CpuProviderNestingPolicyV1::native_only,
             observe_worker_affinity, &observed) ==
                 runtime::CpuExecutionStatusV1::success &&
             observed.complete.load(std::memory_order_relaxed),
         "persistent workers retain their assigned one-CPU masks");
  for (std::uint32_t worker = 0; worker < successful_workers; ++worker) {
    expect(observed.observed_cpus[worker] == config.worker_cpu_ids[worker],
           "worker index maps deterministically to the requested logical CPU");
  }

  runtime::CpuExecutionContextConfigV1 wrong_count;
  wrong_count.requested_threads = 2;
  wrong_count.maximum_threads = 2;
  wrong_count.worker_cpu_ids = {inventory.allowed_logical_cpus.front()};
  creation_report = {};
  auto rejected = runtime::CpuExecutionContextV1::create(
      wrong_count, &status, &creation_report);
  expect(rejected == nullptr &&
             status == runtime::CpuExecutionStatusV1::invalid_configuration &&
             creation_report.status ==
                 runtime::CpuWorkerAffinityStatusV1::invalid_configuration &&
             !creation_report.complete,
         "CPU-ID list must exactly cover the actual persistent worker count");

  runtime::CpuExecutionContextConfigV1 duplicate;
  duplicate.requested_threads = 2;
  duplicate.maximum_threads = 2;
  duplicate.worker_cpu_ids = {inventory.allowed_logical_cpus.front(),
                              inventory.allowed_logical_cpus.front()};
  rejected = runtime::CpuExecutionContextV1::create(
      duplicate, &status, &creation_report);
  expect(rejected == nullptr &&
             status == runtime::CpuExecutionStatusV1::invalid_configuration,
         "duplicate CPU IDs are rejected instead of oversubscribing silently");

  runtime::CpuExecutionContextConfigV1 partial;
  partial.requested_threads = 2;
  partial.maximum_threads = 2;
  partial.worker_cpu_ids = {
      inventory.allowed_logical_cpus.front(),
      std::numeric_limits<std::uint32_t>::max()};
  creation_report = {};
  rejected = runtime::CpuExecutionContextV1::create(
      partial, &status, &creation_report);
  expect(rejected == nullptr &&
             status ==
                 runtime::CpuExecutionStatusV1::affinity_application_failed &&
             creation_report.status ==
                 runtime::CpuWorkerAffinityStatusV1::partially_applied &&
             creation_report.requested_workers == 2 &&
             creation_report.applied_workers == 1 &&
             creation_report.first_failed_worker == 1 &&
             creation_report.first_failed_cpu ==
                 std::numeric_limits<std::uint32_t>::max() &&
             !creation_report.complete &&
             !creation_report.numa_memory_placement_applied,
         "partial affinity application tears down the context and reports exact failure");
#else
  runtime::CpuExecutionContextConfigV1 config;
  config.worker_cpu_ids = {0};
  runtime::CpuExecutionStatusV1 status{};
  runtime::CpuWorkerAffinityReportV1 report;
  auto context = runtime::CpuExecutionContextV1::create(config, &status, &report);
  expect(context == nullptr &&
             status == runtime::CpuExecutionStatusV1::affinity_unavailable &&
             report.status == runtime::CpuWorkerAffinityStatusV1::unavailable &&
             !report.complete,
         "unsupported-platform explicit affinity fails closed during context creation");
#endif
}

void independent_contexts_execute_concurrently() {
  auto first = make_context(2, 2);
  auto second = make_context(2, 2);
  expect(wait_for_workers(*first, 2) && wait_for_workers(*second, 2),
         "independent contexts start their own fixed workers");
  AssignmentState first_state{std::vector<std::size_t>(200)};
  AssignmentState second_state{std::vector<std::size_t>(200)};
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  auto launch = [&](runtime::CpuExecutionContextV1 &context,
                    AssignmentState &state) {
    ready.fetch_add(1, std::memory_order_release);
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    return context.run_tasks(
        state.owners.size(), 2,
        runtime::CpuProviderNestingPolicyV1::native_only, record_assignment,
        &state);
  };
  runtime::CpuExecutionStatusV1 first_result{};
  runtime::CpuExecutionStatusV1 second_result{};
  std::thread first_submitter(
      [&] { first_result = launch(*first, first_state); });
  std::thread second_submitter(
      [&] { second_result = launch(*second, second_state); });
  while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
  go.store(true, std::memory_order_release);
  first_submitter.join();
  second_submitter.join();
  expect(first_result == runtime::CpuExecutionStatusV1::success &&
             second_result == runtime::CpuExecutionStatusV1::success,
         "independent execution contexts complete concurrently");
}

void reentrant_operations_fail_or_stop_without_deadlock() {
  auto context = make_context(2, 2);
  ReentrantState nested{context.get()};
  expect(context->run_tasks(
             1, 1, runtime::CpuProviderNestingPolicyV1::native_only,
             reentrant_callback, &nested) ==
             runtime::CpuExecutionStatusV1::success &&
             nested.nested ==
                 runtime::CpuExecutionStatusV1::invalid_configuration,
         "worker reentrant submission fails fast without deadlock");

  ReentrantState stop{context.get(), runtime::CpuExecutionStatusV1::success,
                      true};
  expect(context->run_tasks(
             1, 1, runtime::CpuProviderNestingPolicyV1::native_only,
             reentrant_callback, &stop) ==
             runtime::CpuExecutionStatusV1::success,
         "worker stop request lets active submission complete");
  expect(!context->info().accepting_work,
         "worker stop request is observable by the owner");
  context->shutdown();
}

void multi_worker_shutdown_completes_active_barrier() {
#if defined(__linux__)
  // Make the wake-up order deterministic: with all workers inheriting one CPU,
  // the first callback requests shutdown before the remaining workers can
  // consume the epoch. They must still join the active submission barrier.
  const auto inventory = platform::discover_current_thread_affinity_v1();
  expect(inventory.discovery_complete &&
             !inventory.allowed_logical_cpus.empty(),
         "multi-worker shutdown test can select one allowed Linux CPU");
  if (!inventory.discovery_complete || inventory.allowed_logical_cpus.empty())
    return;
  const auto binding = platform::apply_current_thread_affinity_v1(
      inventory.allowed_logical_cpus.front());
  expect(binding.status == platform::ThreadAffinityStatusV1::applied,
         "multi-worker shutdown test binds its inherited worker mask");
  if (binding.status != platform::ThreadAffinityStatusV1::applied) return;
#endif

  constexpr std::uint32_t worker_count = 16;
  auto context = make_context(worker_count, worker_count);
  if (context == nullptr) return;
  ConcurrentShutdownState state{context.get()};
  expect(context->run_tasks(
             worker_count, worker_count,
             runtime::CpuProviderNestingPolicyV1::native_only,
             concurrent_shutdown_callback, &state) ==
             runtime::CpuExecutionStatusV1::success,
         "callback shutdown lets every active worker complete the barrier");
  expect(state.callbacks.load(std::memory_order_relaxed) == worker_count,
         "callback shutdown does not drop an active worker");
  const auto info = context->info();
  expect(info.completed_submissions == 1 && !info.accepting_work,
         "multi-worker callback shutdown completes and stops the context");
  context->shutdown();
}

}  // namespace

int main() {
  configuration_and_static_distribution();
  inactive_workers_do_not_retain_borrowed_submissions();
  explicit_worker_affinity_is_strict_and_reported();
  independent_contexts_execute_concurrently();
  reentrant_operations_fail_or_stop_without_deadlock();
  multi_worker_shutdown_completes_active_barrier();
  if (failures != 0) {
    std::cerr << failures << " execution-context checks failed\n";
    return 1;
  }
  std::cout << "persistent execution context PASS\n";
  return 0;
}
