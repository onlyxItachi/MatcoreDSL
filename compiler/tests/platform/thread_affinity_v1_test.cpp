#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "thread_affinity_v1.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

namespace platform = matcore::mdslc::platform;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void status_contract() {
  expect(platform::to_string(platform::ThreadAffinityStatusV1::not_requested) ==
             "not-requested" &&
             platform::to_string(platform::ThreadAffinityStatusV1::applied) ==
                 "applied" &&
             platform::to_string(platform::ThreadAffinityStatusV1::unavailable) ==
                 "unavailable" &&
             platform::to_string(
                 platform::ThreadAffinityStatusV1::invalid_cpu_id) ==
                 "invalid-cpu-id" &&
             platform::to_string(
                 platform::ThreadAffinityStatusV1::system_error) ==
                 "system-error",
         "thread-affinity status diagnostics are stable");
}

void native_backend_contract() {
  const auto inventory = platform::discover_current_thread_affinity_v1();
  const auto current = platform::discover_current_logical_cpu_v1();
#if defined(__linux__) || defined(_WIN32)
#if defined(__linux__)
  constexpr std::string_view kPlatformName = "Linux";
  const bool supported_processor_group_model = true;
#else
  constexpr std::string_view kPlatformName = "Windows";
  const bool supported_processor_group_model =
      ::GetActiveProcessorGroupCount() == 1;
#endif
  if (!supported_processor_group_model) {
    expect(inventory.backend_available && !inventory.discovery_complete &&
               inventory.platform_error != 0 &&
               inventory.allowed_logical_cpus.empty(),
           "multi-group Windows affinity discovery fails closed");
    expect(current.backend_available && !current.discovery_complete &&
               current.platform_error != 0,
           "multi-group Windows current-CPU discovery fails closed");
    const auto unavailable = platform::apply_current_thread_affinity_v1(0);
    expect(unavailable.status ==
               platform::ThreadAffinityStatusV1::unavailable,
           "multi-group Windows affinity application is explicitly unavailable");
    return;
  }
  expect(inventory.backend_available && inventory.discovery_complete &&
             inventory.platform_error == 0 &&
             !inventory.allowed_logical_cpus.empty(),
         "native affinity backend discovers the calling thread's allowed CPUs");
  expect(current.backend_available && current.discovery_complete &&
             current.platform_error == 0 &&
             std::find(inventory.allowed_logical_cpus.begin(),
                       inventory.allowed_logical_cpus.end(),
                       current.logical_cpu) !=
                 inventory.allowed_logical_cpus.end(),
         "native current-CPU discovery authenticates a CPU in the caller's allowed mask");
  if (!inventory.discovery_complete ||
      inventory.allowed_logical_cpus.empty()) {
    return;
  }

  platform::ThreadAffinityApplicationV1 application;
  platform::ThreadAffinityInventoryV1 bound_inventory;
  const std::uint32_t selected = inventory.allowed_logical_cpus.front();
  std::thread worker([&] {
    application = platform::apply_current_thread_affinity_v1(selected);
    bound_inventory = platform::discover_current_thread_affinity_v1();
  });
  worker.join();

  expect(application.status == platform::ThreadAffinityStatusV1::applied &&
             application.requested_logical_cpu == selected &&
             application.platform_error == 0,
         "native backend applies an allowed logical CPU to a child thread");
  expect(bound_inventory.discovery_complete &&
             bound_inventory.allowed_logical_cpus.size() == 1 &&
             bound_inventory.allowed_logical_cpus.front() == selected,
         "applied child-thread mask contains exactly the requested CPU");

  const auto invalid = platform::apply_current_thread_affinity_v1(
      std::numeric_limits<std::uint32_t>::max());
  expect(invalid.status ==
                 platform::ThreadAffinityStatusV1::invalid_cpu_id &&
             invalid.platform_error != 0,
         "out-of-range logical CPU fails before changing caller affinity");
  std::cout << kPlatformName << " affinity backend runtime-validated\n";
#else
  expect(!inventory.backend_available && !inventory.discovery_complete &&
             inventory.allowed_logical_cpus.empty(),
         "unsupported platform reports no affinity backend");
  expect(!current.backend_available && !current.discovery_complete &&
             current.logical_cpu == platform::kUnknownLogicalCpuV1,
         "unsupported platform cannot fabricate a current logical CPU");
  const auto unavailable = platform::apply_current_thread_affinity_v1(0);
  expect(unavailable.status ==
             platform::ThreadAffinityStatusV1::unavailable,
         "unsupported platform fails closed instead of reporting no-op success");
#endif
}

}  // namespace

int main() {
  status_contract();
  native_backend_contract();
  if (failures != 0) {
    std::cerr << failures << " thread-affinity checks failed\n";
    return 1;
  }
  std::cout << "thread affinity v1 PASS\n";
  return 0;
}
