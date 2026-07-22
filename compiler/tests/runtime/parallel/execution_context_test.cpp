#include "cpu_execution_context.h"

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

}  // namespace

int main() {
  configuration_and_static_distribution();
  independent_contexts_execute_concurrently();
  reentrant_operations_fail_or_stop_without_deadlock();
  if (failures != 0) {
    std::cerr << failures << " execution-context checks failed\n";
    return 1;
  }
  std::cout << "persistent execution context PASS\n";
  return 0;
}
