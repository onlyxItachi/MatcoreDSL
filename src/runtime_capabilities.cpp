#include "matcore/runtime_capabilities.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>
#include <regex>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "llvm/ADT/StringMap.h"
#include "llvm/TargetParser/Host.h"

namespace matcore {
namespace {

using CUresult = int;
using CUdevice = int;
using hipError_t = int;

using CuInitFn = CUresult (*)(unsigned int);
using CuDeviceGetCountFn = CUresult (*)(int *);
using CuDeviceGetFn = CUresult (*)(CUdevice *, int);
using CuDeviceGetAttributeFn = CUresult (*)(int *, int, CUdevice);
using HipInitFn = hipError_t (*)(unsigned int);
using HipGetDeviceCountFn = hipError_t (*)(int *);

constexpr int kCudaSuccess = 0;
constexpr hipError_t kHipSuccess = 0;
constexpr int kCudaComputeCapabilityMajorAttr = 75;
constexpr int kCudaComputeCapabilityMinorAttr = 76;

struct RocmRuntimeCapabilities {
  bool library_available = false;
  bool device_present = false;
};

enum class GpuBackendClaim : int { kNone = 0, kNvidia = 1, kAmd = 2 };
static std::atomic<int> g_gpu_backend_claim{0};

struct TimedCommandResult {
  bool completed = false;
  bool timed_out = false;
  int exit_code = -1;
  std::string output;
};

GpuBackendClaim claimForTarget(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kNvidiaDGPU:
      return GpuBackendClaim::kNvidia;
    case TargetKind::kAmdIGPU:
      return GpuBackendClaim::kAmd;
    default:
      return GpuBackendClaim::kNone;
  }
}

std::string backendName(GpuBackendClaim claim) {
  switch (claim) {
    case GpuBackendClaim::kNvidia:
      return "NVIDIA";
    case GpuBackendClaim::kAmd:
      return "AMD HIP";
    case GpuBackendClaim::kNone:
      return "none";
  }
  return "unknown";
}

bool commandSucceeded(const TimedCommandResult &result) {
  return result.completed && !result.timed_out && result.exit_code == 0;
}

TimedCommandResult runPopenCommandWithDeadline(
    std::string_view command, std::chrono::steady_clock::time_point deadline) {
  TimedCommandResult result;
  int output_pipe[2] = {-1, -1};
  if (pipe(output_pipe) != 0) {
    return result;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(output_pipe[0]);
    close(output_pipe[1]);
    return result;
  }

  if (pid == 0) {
    close(output_pipe[0]);
    setpgid(0, 0);
    const std::string command_string(command);
    FILE *stream = popen(command_string.c_str(), "r");
    std::string output;
    int status = -1;
    if (stream != nullptr) {
      char buffer[512];
      while (fgets(buffer, sizeof(buffer), stream) != nullptr) {
        output.append(buffer);
      }
      status = pclose(stream);
    }
    int exit_code = -1;
    if (status != -1 && WIFEXITED(status)) {
      exit_code = WEXITSTATUS(status);
    } else if (status != -1 && WIFSIGNALED(status)) {
      exit_code = 128 + WTERMSIG(status);
    }
    const std::string payload = std::to_string(exit_code) + "\n" + output;
    ssize_t written = 0;
    while (written < static_cast<ssize_t>(payload.size())) {
      const ssize_t chunk = write(output_pipe[1], payload.data() + written,
                                  payload.size() - written);
      if (chunk <= 0) {
        break;
      }
      written += chunk;
    }
    close(output_pipe[1]);
    _exit(0);
  }

  close(output_pipe[1]);
  while (true) {
    int wait_status = 0;
    const pid_t wait_result = waitpid(pid, &wait_status, WNOHANG);
    if (wait_result == pid) {
      break;
    }
    if (wait_result < 0) {
      close(output_pipe[0]);
      return result;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(-pid, SIGKILL);
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
      close(output_pipe[0]);
      result.timed_out = true;
      return result;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::string payload;
  char buffer[512];
  ssize_t bytes = 0;
  while ((bytes = read(output_pipe[0], buffer, sizeof(buffer))) > 0) {
    payload.append(buffer, static_cast<size_t>(bytes));
  }
  close(output_pipe[0]);

  result.completed = true;
  const size_t newline = payload.find('\n');
  if (newline == std::string::npos) {
    result.exit_code = -1;
    result.output = payload;
    return result;
  }
  try {
    result.exit_code = std::stoi(payload.substr(0, newline));
  } catch (...) {
    result.exit_code = -1;
  }
  result.output = payload.substr(newline + 1);
  return result;
}

TimedCommandResult runPopenCommandWithTimeout(std::string_view command,
                                              int timeout_seconds) {
  return runPopenCommandWithDeadline(
      command,
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds));
}

bool kernelLogPermissionDenied(const std::string &output) {
  return output.find("Operation not permitted") != std::string::npos ||
         output.find("Permission denied") != std::string::npos ||
         output.find("read kernel buffer failed") != std::string::npos;
}

GpuPreflightResult kernelLogAdvisory(TargetKind target) {
  GpuPreflightResult result{GpuPreflightStatus::kPass, ""};
  const TargetKind normalized = normalizeTarget(target);
  if (normalized != TargetKind::kNvidiaDGPU &&
      normalized != TargetKind::kAmdIGPU) {
    return result;
  }

  const TimedCommandResult dmesg_result =
      runPopenCommandWithTimeout("dmesg --color=never 2>&1 | tail -n 200", 3);
  if (dmesg_result.timed_out) {
    return {GpuPreflightStatus::kAdvisoryWarning,
            "advisory: timed out while checking recent kernel messages"};
  }
  if (kernelLogPermissionDenied(dmesg_result.output)) {
    return {GpuPreflightStatus::kAdvisoryWarning,
            "advisory: kernel log access denied while checking GPU driver health"};
  }
  if (!commandSucceeded(dmesg_result)) {
    return result;
  }

  if (normalized == TargetKind::kNvidiaDGPU) {
    if (std::regex_search(dmesg_result.output, std::regex("Xid"))) {
      return {GpuPreflightStatus::kAdvisoryWarning,
              "advisory: recent kernel logs include NVIDIA Xid errors"};
    }
    return result;
  }

  if (std::regex_search(dmesg_result.output,
                        std::regex("amdgpu.*error",
                                   std::regex_constants::icase))) {
    return {GpuPreflightStatus::kAdvisoryWarning,
            "advisory: recent kernel logs include amdgpu error messages"};
  }
  return result;
}

std::optional<void *> openSharedLibrary(const char *primary,
                                        const char *fallback = nullptr) {
  if (void *handle = dlopen(primary, RTLD_NOW | RTLD_LOCAL)) {
    return handle;
  }
  if (fallback != nullptr) {
    if (void *handle = dlopen(fallback, RTLD_NOW | RTLD_LOCAL)) {
      return handle;
    }
  }
  return std::nullopt;
}

void closeSharedLibrary(std::optional<void *> handle) {
  if (handle.has_value()) {
    dlclose(*handle);
  }
}

NvidiaRuntimeCapabilities detectNvidiaRuntime() {
  NvidiaRuntimeCapabilities out;

  std::optional<void *> libcuda =
      openSharedLibrary("libcuda.so", "/lib/x86_64-linux-gnu/libcuda.so");
  if (!libcuda.has_value()) {
    return out;
  }
  out.driver_available = true;

  auto cuInit = reinterpret_cast<CuInitFn>(dlsym(*libcuda, "cuInit"));
  auto cuDeviceGetCount =
      reinterpret_cast<CuDeviceGetCountFn>(dlsym(*libcuda, "cuDeviceGetCount"));
  auto cuDeviceGet =
      reinterpret_cast<CuDeviceGetFn>(dlsym(*libcuda, "cuDeviceGet"));
  auto cuDeviceGetAttribute =
      reinterpret_cast<CuDeviceGetAttributeFn>(dlsym(*libcuda, "cuDeviceGetAttribute"));

  if (cuInit == nullptr || cuDeviceGetCount == nullptr || cuDeviceGet == nullptr ||
      cuDeviceGetAttribute == nullptr) {
    closeSharedLibrary(libcuda);
    return out;
  }
  if (cuInit(0) != kCudaSuccess) {
    closeSharedLibrary(libcuda);
    return out;
  }

  int device_count = 0;
  if (cuDeviceGetCount(&device_count) != kCudaSuccess || device_count <= 0) {
    closeSharedLibrary(libcuda);
    return out;
  }
  out.device_present = true;

  CUdevice device = 0;
  if (cuDeviceGet(&device, /*ordinal=*/0) != kCudaSuccess) {
    closeSharedLibrary(libcuda);
    return out;
  }

  int major = 0;
  int minor = 0;
  if (cuDeviceGetAttribute(&major, kCudaComputeCapabilityMajorAttr, device) ==
          kCudaSuccess &&
      cuDeviceGetAttribute(&minor, kCudaComputeCapabilityMinorAttr, device) ==
          kCudaSuccess) {
    out.compute_major = major;
    out.compute_minor = minor;
    out.chip = "sm_" + std::to_string(major) + std::to_string(minor);
  }

  closeSharedLibrary(libcuda);
  return out;
}

RocmRuntimeCapabilities detectRocmRuntime() {
  RocmRuntimeCapabilities out;
  std::optional<void *> handle =
      openSharedLibrary("libamdhip64.so", "/lib/x86_64-linux-gnu/libamdhip64.so");
  if (!handle.has_value()) {
    return out;
  }
  out.library_available = true;

  auto hipInit = reinterpret_cast<HipInitFn>(dlsym(*handle, "hipInit"));
  auto hipGetDeviceCount =
      reinterpret_cast<HipGetDeviceCountFn>(dlsym(*handle, "hipGetDeviceCount"));
  if (hipInit == nullptr || hipGetDeviceCount == nullptr) {
    closeSharedLibrary(handle);
    return out;
  }
  if (hipInit(0) != kHipSuccess) {
    closeSharedLibrary(handle);
    return out;
  }

  int device_count = 0;
  if (hipGetDeviceCount(&device_count) == kHipSuccess && device_count > 0) {
    out.device_present = true;
  }
  closeSharedLibrary(handle);
  return out;
}

bool detectNpuRuntime() {
  for (const char *candidate :
       {"libamdxdna.so", "libxaiengine.so", "libxrt_coreutil.so"}) {
    std::optional<void *> handle = openSharedLibrary(candidate);
    if (handle.has_value()) {
      closeSharedLibrary(handle);
      return true;
    }
  }
  return false;
}

bool compareComputeCapability(const NvidiaRuntimeCapabilities &runtime,
                             int min_major, int min_minor) {
  if (!runtime.device_present) {
    return false;
  }
  if (runtime.compute_major > min_major) {
    return true;
  }
  if (runtime.compute_major == min_major && runtime.compute_minor >= min_minor) {
    return true;
  }
  return false;
}

}  // namespace

RuntimeCapabilities DetectRuntimeCapabilities() {
  RuntimeCapabilities runtime;

  llvm::StringMap<bool> host_features;
  runtime.x86.detected = llvm::sys::getHostCPUFeatures(host_features);
  if (runtime.x86.detected) {
    for (const auto &entry : host_features) {
      if (entry.second) {
        runtime.x86.enabled_features.insert(entry.first().str());
      }
    }
  }

  // SAFETY: Do NOT eagerly probe GPU runtimes here.
  // On dual-GPU systems (e.g. NVIDIA dGPU + AMD iGPU), calling cuInit()
  // or hipInit() prematurely can interfere with the other vendor's driver.
  // GPU capabilities are probed lazily on first use via probeNvidiaIfNeeded()
  // and probeRocmIfNeeded() — called from CanExecuteOnHost() only when
  // the target actually requires that backend.
  runtime.nvidia_probed = false;
  runtime.rocm_probed = false;
  runtime.rocm_runtime_available = false;
  runtime.npu_runtime_available = detectNpuRuntime();
  return runtime;
}

void probeNvidiaIfNeeded(RuntimeCapabilities &runtime) {
  if (!runtime.nvidia_probed) {
    if (!acquireGpuBackendClaim(TargetKind::kNvidiaDGPU, nullptr)) {
      runtime.nvidia = NvidiaRuntimeCapabilities{};
      runtime.nvidia_probed = true;
      return;
    }
    runtime.nvidia = detectNvidiaRuntime();
    runtime.nvidia_probed = true;
  }
}

void probeRocmIfNeeded(RuntimeCapabilities &runtime) {
  if (!runtime.rocm_probed) {
    if (!acquireGpuBackendClaim(TargetKind::kAmdIGPU, nullptr)) {
      runtime.rocm_library_available = false;
      runtime.rocm_device_present = false;
      runtime.rocm_runtime_available = false;
      runtime.rocm_probed = true;
      return;
    }
    const RocmRuntimeCapabilities rocm = detectRocmRuntime();
    runtime.rocm_library_available = rocm.library_available;
    runtime.rocm_device_present = rocm.device_present;
    runtime.rocm_runtime_available = rocm.library_available && rocm.device_present;
    runtime.rocm_probed = true;
  }
}

bool SupportsX86Feature(const RuntimeCapabilities &runtime,
                        std::string_view feature) {
  if (!runtime.x86.detected) {
    return false;
  }
  return runtime.x86.enabled_features.find(std::string(feature)) !=
         runtime.x86.enabled_features.end();
}

GpuPreflightResult gpuPreflightCheck(TargetKind target) {
  const char *disable_gpu = std::getenv("MATCORE_DISABLE_GPU");
  if (disable_gpu != nullptr && std::string_view(disable_gpu) == "1") {
    return {GpuPreflightStatus::kFail,
            "GPU execution disabled by MATCORE_DISABLE_GPU=1"};
  }

  const TargetKind normalized = normalizeTarget(target);
  if (normalized != TargetKind::kNvidiaDGPU &&
      normalized != TargetKind::kAmdIGPU) {
    return {GpuPreflightStatus::kPass, ""};
  }

  // Cache preflight results to avoid spawning nvidia-smi / dmesg subprocesses
  // on every mc.launch() call.  Re-check periodically (every 30s) to detect
  // hot-unplug or driver recovery.  Failures are never cached so the next
  // call retries immediately.
  static std::mutex preflight_mutex;
  static std::unordered_map<int, GpuPreflightResult> preflight_cache;
  static std::unordered_map<int, std::chrono::steady_clock::time_point> preflight_timestamps;
  constexpr auto kPreflightCacheDuration = std::chrono::seconds(30);

  {
    std::lock_guard<std::mutex> lock(preflight_mutex);
    const int key = static_cast<int>(normalized);
    auto ts_it = preflight_timestamps.find(key);
    if (ts_it != preflight_timestamps.end()) {
      const auto elapsed = std::chrono::steady_clock::now() - ts_it->second;
      if (elapsed < kPreflightCacheDuration) {
        auto cached_it = preflight_cache.find(key);
        if (cached_it != preflight_cache.end() &&
            cached_it->second.status != GpuPreflightStatus::kFail) {
          return cached_it->second;
        }
      }
    }
  }

  GpuPreflightResult result;

  if (normalized == TargetKind::kNvidiaDGPU) {
    const TimedCommandResult nvidia_smi =
        runPopenCommandWithTimeout("nvidia-smi -L 2>&1", 3);
    if (nvidia_smi.timed_out) {
      return {GpuPreflightStatus::kFail,
              "nvidia-smi -L timed out after 3 seconds"};
    }
    if (!commandSucceeded(nvidia_smi)) {
      return {GpuPreflightStatus::kFail,
              "nvidia-smi -L failed; NVIDIA runtime preflight did not pass"};
    }
    const GpuPreflightResult advisory = kernelLogAdvisory(normalized);
    if (advisory.status != GpuPreflightStatus::kPass) {
      result = advisory;
    } else {
      result = {GpuPreflightStatus::kPass, ""};
    }
  } else {
    const auto amd_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    const TimedCommandResult rocm_smi =
        runPopenCommandWithDeadline("rocm-smi 2>&1", amd_deadline);
    TimedCommandResult rocminfo;
    if (!commandSucceeded(rocm_smi)) {
      rocminfo = runPopenCommandWithDeadline("rocminfo 2>&1", amd_deadline);
    }
    if (!commandSucceeded(rocm_smi) && !commandSucceeded(rocminfo)) {
      if (rocm_smi.timed_out && rocminfo.timed_out) {
        return {GpuPreflightStatus::kFail,
                "rocm-smi and rocminfo timed out after 3 seconds"};
      }
      return {GpuPreflightStatus::kFail,
              "neither rocm-smi nor rocminfo completed successfully"};
    }
    const GpuPreflightResult advisory = kernelLogAdvisory(normalized);
    if (advisory.status != GpuPreflightStatus::kPass) {
      result = advisory;
    } else {
      result = {GpuPreflightStatus::kPass, ""};
    }
  }

  // Cache successful / advisory results (not failures — those retry next call).
  if (result.status != GpuPreflightStatus::kFail) {
    std::lock_guard<std::mutex> lock(preflight_mutex);
    const int key = static_cast<int>(normalized);
    preflight_cache[key] = result;
    preflight_timestamps[key] = std::chrono::steady_clock::now();
  }

  return result;
}

bool acquireGpuBackendClaim(TargetKind target, std::string *denial_reason) {
  const GpuBackendClaim desired_claim = claimForTarget(target);
  if (desired_claim == GpuBackendClaim::kNone) {
    if (denial_reason != nullptr) {
      denial_reason->clear();
    }
    return true;
  }

  int expected = static_cast<int>(GpuBackendClaim::kNone);
  const int desired = static_cast<int>(desired_claim);
  if (g_gpu_backend_claim.compare_exchange_strong(expected, desired) ||
      expected == desired) {
    if (denial_reason != nullptr) {
      denial_reason->clear();
    }
    return true;
  }

  if (denial_reason != nullptr) {
    *denial_reason = "SAFETY: Cannot use " + backendName(desired_claim) +
                     " GPU backend — " +
                     backendName(static_cast<GpuBackendClaim>(expected)) +
                     " backend already claimed this process. Restart the process "
                     "to switch GPU backends safely.";
  }
  return false;
}

void releaseGpuBackendClaim(TargetKind /*target*/) {
  // V2 SAFETY: Backend claim is process-sticky. Once a GPU backend is claimed,
  // it remains claimed for the entire process lifetime. Releasing and
  // re-acquiring risks cross-backend symbol collision on dual-GPU systems
  // (e.g. NVIDIA RTX 4060 + AMD 890M iGPU sharing mgpu* symbols).
  // This is a deliberate no-op.
}

bool CanExecuteOnHost(const RuntimeCapabilities &runtime,
                      const ExecutionRequirements &requirements,
                      std::string *denial_reason) {
  // Safe mutable ref for lazy probing — cachedRuntimeCapabilities() is non-const
  auto &mutable_runtime = const_cast<RuntimeCapabilities &>(runtime);
  auto deny = [&](const std::string &message) {
    if (denial_reason != nullptr) {
      *denial_reason = message;
    }
    return false;
  };

  for (const std::string &feature : requirements.required_x86_features) {
    if (!SupportsX86Feature(runtime, feature)) {
      return deny("host CPU does not provide required feature '" + feature + "'");
    }
  }

  if (requirements.requires_nvidia_device) {
    const char *disable_gpu = std::getenv("MATCORE_DISABLE_GPU");
    if (disable_gpu != nullptr && std::string_view(disable_gpu) == "1") {
      return deny("GPU execution disabled by MATCORE_DISABLE_GPU=1");
    }
    if (!acquireGpuBackendClaim(TargetKind::kNvidiaDGPU, denial_reason)) {
      return false;
    }
    // Lazy probe: only touch CUDA when NVIDIA is actually requested.
    probeNvidiaIfNeeded(mutable_runtime);
    if (!runtime.nvidia.driver_available) {
      return deny("NVIDIA driver library is unavailable");
    }
    if (!runtime.nvidia.device_present) {
      return deny("no NVIDIA device is visible to the runtime");
    }
    if (requirements.min_nvidia_sm_major.has_value() &&
        requirements.min_nvidia_sm_minor.has_value() &&
        !compareComputeCapability(runtime.nvidia, *requirements.min_nvidia_sm_major,
                                  *requirements.min_nvidia_sm_minor)) {
      return deny("device capability " + runtime.nvidia.chip +
                  " is below required sm_" +
                  std::to_string(*requirements.min_nvidia_sm_major) +
                  std::to_string(*requirements.min_nvidia_sm_minor));
    }
  }

  if (requirements.requires_rocm_runtime) {
    const char *disable_gpu = std::getenv("MATCORE_DISABLE_GPU");
    if (disable_gpu != nullptr && std::string_view(disable_gpu) == "1") {
      return deny("GPU execution disabled by MATCORE_DISABLE_GPU=1");
    }
    if (!acquireGpuBackendClaim(TargetKind::kAmdIGPU, denial_reason)) {
      return false;
    }
    // Lazy probe: only touch HIP when AMD is actually requested.
    probeRocmIfNeeded(mutable_runtime);
    if (!runtime.rocm_library_available) {
      return deny("ROCm runtime library is unavailable");
    }
    if (!runtime.rocm_device_present) {
      return deny("ROCm runtime is present but no AMD HIP device is available");
    }
  }
  if (requirements.requires_npu_runtime && !runtime.npu_runtime_available) {
    return deny("NPU runtime library is unavailable");
  }

  if (denial_reason != nullptr) {
    denial_reason->clear();
  }
  return true;
}

std::string FormatExecutionDeniedMessage(const RequestedTargetProfile &profile,
                                         const std::string &denial_reason) {
  return "Execution Denied: Hardware limits for target '" + profile.canonical +
         "': " + denial_reason;
}

}  // namespace matcore
