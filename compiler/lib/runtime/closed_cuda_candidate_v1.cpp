#include "closed_cuda_candidate_v1.h"
#include "closed_gpu_images_v1.h"
#include "closed_gpu_capability_v1.h"
#include <cuda.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#if defined(__FAST_MATH__)
#error "strict CUDA adapter must not be compiled with fast math"
#endif

namespace matcore::mdslc::runtime::closed_host_v1::detail {
namespace {
// A device/driver failure which leaves resource quiescence unprovable disables
// further attempts in this adapter instance. It must not cause repeated leaks
// or an unsafe free of storage potentially still in use by a GPU.
std::atomic<bool> poisoned{false};

struct ErrnoRestore {
  int value = errno;
  ~ErrnoRestore() { errno = value; }
};

template <typename Work> Code isolated(Work &&work) noexcept {
  ErrnoRestore preserve;
  if (poisoned.load(std::memory_order_acquire)) return Code::candidate_failure;
  Code result = Code::candidate_failure;
  try {
    // No CUDA API runs on the caller thread. This preserves its context stack,
    // errno, FP environment and unrelated thread-local driver state even when a
    // driver context-pop operation fails. The worker has no caller GPU state.
    std::thread worker([&]() noexcept {
      if (std::fesetenv(FE_DFL_ENV) != 0) { result = Code::unsupported_fp_environment; return; }
      try { result = work(); }
      catch (const std::bad_alloc &) { result = Code::allocation_failure; }
      catch (...) { result = Code::candidate_failure; }
    });
    // Failure to join our own live worker is fail-stop through the joinable
    // thread destructor, not a normal return exposing captured stack objects.
    worker.join();
  } catch (const std::bad_alloc &) { result = Code::allocation_failure; }
  catch (...) { result = Code::candidate_failure; }
  return result;
}

Code discover(CUdevice &selected) noexcept {
  if (poisoned.load(std::memory_order_acquire)) return Code::candidate_unavailable;
  if (cuInit(0) != CUDA_SUCCESS) return Code::candidate_unavailable;
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0)
    return Code::candidate_unavailable;
  for (int index = 0; index < count; ++index) {
    CUdevice device;
    int major = 0, minor = 0;
    if (cuDeviceGet(&device, index) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS)
      return Code::candidate_unavailable;
    if (major == 8 && minor == 9) { selected = device; return Code::ok; }
  }
  return Code::candidate_incompatible;
}

bool bytes(std::uint64_t rows, std::uint64_t columns, std::size_t &result) noexcept {
  constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / sizeof(float);
  if (rows > limit || columns > limit || (rows && columns > limit / rows)) return false;
  result = static_cast<std::size_t>(rows * columns * sizeof(float));
  return true;
}

std::array<std::uint64_t, 7> descriptor(CUdeviceptr pointer, std::uint64_t rows,
                                       std::uint64_t columns) noexcept {
  return {pointer, pointer, 0, rows, columns, columns, 1};
}

struct Resources {
  CUcontext context = nullptr, previous = nullptr;
  CUmodule fill = nullptr, gemm = nullptr;
  std::array<CUdeviceptr, 3> memory{};
  std::vector<float> hostA, hostB, hostC;
  Resources *quarantineNext = nullptr;

  // GPU transfers refer only to heap-owned staging, never caller memory. On
  // unknown quiescence retain this frame, the private context, device memory,
  // and host staging; an API failure does not prove that no work was queued.
  bool finish() noexcept {
    if (!context) return true;
    bool okay = true;
    const bool quiescent = cuCtxSynchronize() == CUDA_SUCCESS;
    if (quiescent) {
      for (auto pointer : memory)
        if (pointer && cuMemFree(pointer) != CUDA_SUCCESS) okay = false;
      if (gemm && cuModuleUnload(gemm) != CUDA_SUCCESS) okay = false;
      if (fill && cuModuleUnload(fill) != CUDA_SUCCESS) okay = false;
    } else {
      okay = false;
      poisoned.store(true, std::memory_order_release);
    }
    CUcontext popped = nullptr;
    if (cuCtxPopCurrent(&popped) != CUDA_SUCCESS || popped != context) okay = false;
    CUcontext restored = nullptr;
    if (cuCtxGetCurrent(&restored) != CUDA_SUCCESS || restored != previous) okay = false;
    if (quiescent && cuCtxDestroy(context) != CUDA_SUCCESS) okay = false;
    if (okay) context = nullptr;
    if (!okay) poisoned.store(true, std::memory_order_release);
    return okay;
  }
};

std::atomic<Resources *> quarantineHead{nullptr};
void quarantine(std::unique_ptr<Resources> &resources) noexcept {
  poisoned.store(true, std::memory_order_release);
  auto *retained = resources.release();
  retained->quarantineNext = quarantineHead.load(std::memory_order_relaxed);
  while (!quarantineHead.compare_exchange_weak(retained->quarantineNext, retained,
      std::memory_order_release, std::memory_order_relaxed)) {}
}

bool range(const void *pointer, std::size_t size, std::uintptr_t &begin,
           std::uintptr_t &end) noexcept {
  begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (size && (!pointer || begin % alignof(float))) return false;
  if (begin > std::numeric_limits<std::uintptr_t>::max() - size) return false;
  end = begin + size;
  return true;
}

Code execute(CandidateInput a, CandidateInput b, CandidateOutput c, CUdevice device) {
  std::array<std::size_t, 3> sizes;
  if (!bytes(a.rows, a.columns, sizes[0]) || !bytes(b.rows, b.columns, sizes[1]) ||
      !bytes(c.rows, c.columns, sizes[2])) return Code::extent_overflow;
  if (a.columns != b.rows || c.rows != a.rows || c.columns != b.columns)
    return Code::shape_mismatch;
  if ((sizes[0] && !a.data) || (sizes[1] && !b.data) || (sizes[2] && !c.data))
    return Code::invalid_view;
  // Qualification envelope for this deliberately scalar device realization,
  // not a semantic tensor limit, tuning threshold, or cross-target policy.
  if (!closedGpuShapeCompatibleV1(a.rows, b.columns, a.columns))
    return Code::candidate_incompatible;
  std::uintptr_t ab, ae, bb, be, cb, ce;
  if (!range(a.data, sizes[0], ab, ae) || !range(b.data, sizes[1], bb, be) ||
      !range(c.data, sizes[2], cb, ce) ||
      (sizes[2] && ((sizes[0] && cb < ae && ab < ce) || (sizes[1] && cb < be && bb < ce))))
    return Code::invalid_view;
  int maxX = 0, maxY = 0;
  if (cuDeviceGetAttribute(&maxX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&maxY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device) != CUDA_SUCCESS)
    return Code::candidate_unavailable;
  // gpu.block_id lowers through signed i32 here; guard before narrowing/launch.
  if (maxX <= 0 || maxY <= 0 || c.rows > static_cast<std::uint64_t>(maxX) ||
      c.columns > static_cast<std::uint64_t>(maxY) ||
      c.rows > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      c.columns > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    return Code::candidate_incompatible;
  if (!sizes[2]) return Code::ok;
  if (mdslc_nvvm_fill_image_v1_size < 64 || mdslc_nvvm_gemm_image_v1_size < 64)
    return Code::candidate_unavailable;

  auto owned = std::make_unique<Resources>();
  auto &resources = *owned;
  if (sizes[0]) resources.hostA.assign(a.data, a.data + sizes[0] / sizeof(float));
  if (sizes[1]) resources.hostB.assign(b.data, b.data + sizes[1] / sizeof(float));
  resources.hostC.resize(sizes[2] / sizeof(float));
  if (cuCtxGetCurrent(&resources.previous) != CUDA_SUCCESS) return Code::candidate_failure;
  if (cuCtxCreate(&resources.context, nullptr, 0, device) != CUDA_SUCCESS)
    return Code::candidate_failure;
  const auto run = [&]() noexcept -> Code {
    if (cuModuleLoadData(&resources.fill, mdslc_nvvm_fill_image_v1) != CUDA_SUCCESS ||
        cuModuleLoadData(&resources.gemm, mdslc_nvvm_gemm_image_v1) != CUDA_SUCCESS)
      return Code::candidate_failure;
    CUfunction fill = nullptr, gemm = nullptr;
    if (cuModuleGetFunction(&fill, resources.fill, kGpuStrictGemmKernelV1) != CUDA_SUCCESS ||
        cuModuleGetFunction(&gemm, resources.gemm, kGpuStrictGemmKernelV1) != CUDA_SUCCESS)
      return Code::candidate_failure;
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      const auto status = cuMemAlloc(&resources.memory[index], sizes[index] ? sizes[index] : sizeof(float));
      if (status != CUDA_SUCCESS)
        return status == CUDA_ERROR_OUT_OF_MEMORY ? Code::allocation_failure : Code::candidate_failure;
    }
    if ((sizes[0] && cuMemcpyHtoD(resources.memory[0], resources.hostA.data(), sizes[0]) != CUDA_SUCCESS) ||
        (sizes[1] && cuMemcpyHtoD(resources.memory[1], resources.hostB.data(), sizes[1]) != CUDA_SUCCESS))
      return Code::candidate_failure;
    auto da = descriptor(resources.memory[0], a.rows, a.columns);
    auto db = descriptor(resources.memory[1], b.rows, b.columns);
    auto dc = descriptor(resources.memory[2], c.rows, c.columns);
    std::array<void *, 7> fillArgs;
    std::array<void *, 21> gemmArgs;
    for (unsigned field = 0; field < 7; ++field) {
      fillArgs[field] = &dc[field];
      gemmArgs[field] = &da[field];
      gemmArgs[field + 7] = &db[field];
      gemmArgs[field + 14] = &dc[field];
    }
    const auto m = static_cast<unsigned>(c.rows), n = static_cast<unsigned>(c.columns);
    if (cuLaunchKernel(fill, m, n, 1, 1, 1, 1, 0, nullptr, fillArgs.data(), nullptr) != CUDA_SUCCESS ||
        cuLaunchKernel(gemm, m, n, 1, 1, 1, 1, 0, nullptr, gemmArgs.data(), nullptr) != CUDA_SUCCESS ||
        cuCtxSynchronize() != CUDA_SUCCESS) return Code::candidate_failure;
    if (cuMemcpyDtoH(resources.hostC.data(), resources.memory[2], sizes[2]) != CUDA_SUCCESS)
      return Code::candidate_failure;
    return Code::ok;
  };
  auto status = run();
  if (!resources.finish()) {
    quarantine(owned);
    return Code::candidate_failure;
  }
  // All fallible preparation, execution, transfers and cleanup are complete.
  if (status == Code::ok) std::memcpy(c.data, resources.hostC.data(), sizes[2]);
  return status;
}
} // namespace

Code cudaCandidateAvailable() noexcept {
  return isolated([] { CUdevice device; return discover(device); });
}

Code cudaGemmCandidate(CandidateInput a, CandidateInput b, CandidateOutput c) noexcept {
  return isolated([&] {
    CUdevice device;
    const auto status = discover(device);
    return status == Code::ok ? execute(a, b, c, device) : status;
  });
}
} // namespace matcore::mdslc::runtime::closed_host_v1::detail
