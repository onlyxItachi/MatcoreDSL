#include "thread_affinity_v1.h"

#if defined(__linux__)
#include <cerrno>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace matcore::mdslc::platform {

ThreadAffinityApplicationV1 apply_current_thread_affinity_v1(
    std::uint32_t logical_cpu) noexcept {
  ThreadAffinityApplicationV1 result;
  result.requested_logical_cpu = logical_cpu;

#if defined(__linux__)
  if (logical_cpu >= static_cast<std::uint32_t>(CPU_SETSIZE)) {
    result.status = ThreadAffinityStatusV1::invalid_cpu_id;
    result.platform_error = EINVAL;
    return result;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<int>(logical_cpu), &set);
  const int error = ::pthread_setaffinity_np(::pthread_self(), sizeof(set),
                                             &set);
  if (error != 0) {
    result.status = error == EINVAL
                        ? ThreadAffinityStatusV1::invalid_cpu_id
                        : ThreadAffinityStatusV1::system_error;
    result.platform_error = error;
    return result;
  }
  result.status = ThreadAffinityStatusV1::applied;
  return result;
#else
  (void)logical_cpu;
  result.status = ThreadAffinityStatusV1::unavailable;
  return result;
#endif
}

ThreadAffinityInventoryV1 discover_current_thread_affinity_v1() {
  ThreadAffinityInventoryV1 result;
#if defined(__linux__)
  result.backend_available = true;

  const long configured = ::sysconf(_SC_NPROCESSORS_CONF);
  if (configured < 0 ||
      static_cast<unsigned long>(configured) >
          static_cast<unsigned long>(CPU_SETSIZE)) {
    result.platform_error = configured < 0 ? errno : EOVERFLOW;
    return result;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  const int error =
      ::pthread_getaffinity_np(::pthread_self(), sizeof(set), &set);
  if (error != 0) {
    result.platform_error = error;
    return result;
  }

  try {
    result.allowed_logical_cpus.reserve(static_cast<std::size_t>(configured));
    for (int cpu = 0; cpu < configured; ++cpu) {
      if (CPU_ISSET(cpu, &set)) {
        result.allowed_logical_cpus.push_back(static_cast<std::uint32_t>(cpu));
      }
    }
  } catch (...) {
    result.allowed_logical_cpus.clear();
    result.platform_error = ENOMEM;
    return result;
  }
  result.discovery_complete = !result.allowed_logical_cpus.empty();
  if (!result.discovery_complete) result.platform_error = ENODEV;
#endif
  return result;
}

CurrentLogicalCpuV1 discover_current_logical_cpu_v1() noexcept {
  CurrentLogicalCpuV1 result;
#if defined(__linux__)
  result.backend_available = true;
  errno = 0;
  const int logical_cpu = ::sched_getcpu();
  if (logical_cpu < 0) {
    result.platform_error = errno != 0 ? errno : ENODEV;
    return result;
  }
  result.logical_cpu = static_cast<std::uint32_t>(logical_cpu);
  result.discovery_complete = true;
#endif
  return result;
}

std::string_view to_string(ThreadAffinityStatusV1 status) noexcept {
  switch (status) {
    case ThreadAffinityStatusV1::not_requested:
      return "not-requested";
    case ThreadAffinityStatusV1::applied:
      return "applied";
    case ThreadAffinityStatusV1::unavailable:
      return "unavailable";
    case ThreadAffinityStatusV1::invalid_cpu_id:
      return "invalid-cpu-id";
    case ThreadAffinityStatusV1::system_error:
      return "system-error";
  }
  return "unknown";
}

}  // namespace matcore::mdslc::platform
