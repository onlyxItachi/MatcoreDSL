#include "closed_rocdl_candidate_v1.h"
#include "closed_gpu_images_v1.h"
#include "closed_gpu_capability_v1.h"

#include <hip/hip_runtime_api.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#if defined(__FAST_MATH__)
#error "strict ROCDL adapter must not be compiled with fast math"
#endif

namespace matcore::mdslc::runtime::closed_host_v1::detail {
namespace {
constexpr std::uint64_t maxExtent = 65535;
constexpr const char *kernelName = "__matcore_strict_gemm_f32_v1_kernel";
std::atomic<bool> poisoned{false};

struct Frame {
  hipStream_t stream = nullptr;
  hipModule_t fillModule = nullptr, gemmModule = nullptr;
  hipFunction_t fill = nullptr, gemm = nullptr;
  float *a = nullptr, *b = nullptr, *c = nullptr;
  std::vector<float> hostA, hostB, hostC;
  bool workMayBePending = false;
  Frame *quarantineNext = nullptr;
};
// Reachable process-lifetime quarantine is intentional after unproven device
// completion, not an unreachable accidental leak or permission to reuse memory.
std::atomic<Frame *> quarantineHead{nullptr};
void quarantine(std::unique_ptr<Frame> &frame) noexcept {
  poisoned.store(true, std::memory_order_release);
  auto *owned = frame.release();
  owned->quarantineNext = quarantineHead.load(std::memory_order_relaxed);
  while (!quarantineHead.compare_exchange_weak(owned->quarantineNext, owned,
      std::memory_order_release, std::memory_order_relaxed)) {}
}

struct ErrnoRestore {
  int value = errno;
  ~ErrnoRestore() { errno = value; }
};

template <typename Work> Code isolated(Work &&work) noexcept {
  ErrnoRestore preserve;
  if (poisoned.load(std::memory_order_acquire)) return Code::candidate_failure;
  if (std::getenv("HSA_OVERRIDE_GFX_VERSION")) return Code::candidate_unavailable;
  Code result = Code::candidate_failure;
  try {
    // HIP current-device/context and last-error are thread-local. No HIP call
    // executes on the caller thread, which may have unrelated live HIP state.
    std::thread worker([&]() noexcept {
      if (std::fesetenv(FE_DFL_ENV) != 0) {
        result = Code::unsupported_fp_environment;
        return;
      }
      try { result = work(); }
      catch (const std::bad_alloc &) { result = Code::allocation_failure; }
      catch (...) { result = Code::candidate_failure; }
    });
    // A valid join is not a fallible semantic frontier. If the platform cannot
    // join its own live worker, normal return would expose live stack captures;
    // the standard joinable-thread destructor therefore terminates fail-stop.
    worker.join();
  } catch (const std::bad_alloc &) { result = Code::allocation_failure; }
  catch (...) { result = Code::candidate_failure; }
  return result;
}

Code discover(int &selected) noexcept {
  if (poisoned.load(std::memory_order_acquire)) return Code::candidate_failure;
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) return Code::candidate_unavailable;
  for (int device = 0; device < count; ++device) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) != hipSuccess)
      return Code::candidate_unavailable;
    if (std::strncmp(properties.gcnArchName, "gfx1150", 7) != 0 ||
        (properties.gcnArchName[7] != '\0' && properties.gcnArchName[7] != ':'))
      continue;
    if (properties.maxGridSize[0] < static_cast<int>(maxExtent) ||
        properties.maxGridSize[1] < static_cast<int>(maxExtent) ||
        properties.maxGridSize[2] < 1 || properties.maxThreadsPerBlock < 1)
      return Code::candidate_unavailable;
    selected = device;
    return Code::ok;
  }
  return Code::candidate_unavailable;
}

bool footprint(std::uint64_t rows, std::uint64_t cols, std::size_t &count) noexcept {
  if (rows > maxExtent || cols > maxExtent) return false;
  constexpr auto limit = static_cast<std::uint64_t>(
      std::numeric_limits<std::ptrdiff_t>::max() / sizeof(float));
  if (cols && rows > limit / cols) return false;
  count = static_cast<std::size_t>(rows * cols);
  return true;
}
bool range(const float *pointer, std::size_t count,
           std::uintptr_t &begin, std::uintptr_t &end) noexcept {
  begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto bytes = count * sizeof(float);
  if (count && (!pointer || begin % alignof(float))) return false;
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  end = begin + bytes;
  return true;
}

// This function is called only after queue completion has been proved. A
// failed resource release poisons the candidate; no success value is issued.
bool release(Frame &frame) noexcept {
  bool okay = true;
  for (auto **pointer : {&frame.a, &frame.b, &frame.c}) {
    if (*pointer && hipFree(*pointer) != hipSuccess) okay = false;
    else *pointer = nullptr;
  }
  for (auto *module : {&frame.fillModule, &frame.gemmModule}) {
    if (*module && hipModuleUnload(*module) != hipSuccess) okay = false;
    else *module = nullptr;
  }
  if (frame.stream && hipStreamDestroy(frame.stream) != hipSuccess) okay = false;
  else frame.stream = nullptr;
  return okay;
}

struct Descriptor {
  float *allocated, *aligned;
  std::int64_t offset = 0, rows, columns, rowStride, columnStride = 1;
  Descriptor(float *data, std::uint64_t m, std::uint64_t n)
      : allocated(data), aligned(data), rows(m), columns(n), rowStride(n) {}
  void append(void **args) noexcept {
    args[0] = &allocated; args[1] = &aligned; args[2] = &offset;
    args[3] = &rows; args[4] = &columns; args[5] = &rowStride;
    args[6] = &columnStride;
  }
};
static_assert(sizeof(Descriptor) == 56 && alignof(Descriptor) == 8);

Code execute(CandidateInput lhs, CandidateInput rhs, CandidateOutput output,
             std::size_t ac, std::size_t bc, std::size_t cc) {
  int selected = -1;
  if (auto code = discover(selected); code != Code::ok) return code;
  if (hipSetDevice(selected) != hipSuccess) return Code::candidate_unavailable;
  if (!cc) return Code::ok;

  auto frame = std::make_unique<Frame>();
  // All host staging allocation/copies precede the first asynchronous device
  // use. Failure completion may retain this frame after the caller returns.
  if (ac) frame->hostA.assign(lhs.data, lhs.data + ac);
  if (bc) frame->hostB.assign(rhs.data, rhs.data + bc);
  frame->hostC.resize(cc);

  auto failed = [&](Code code = Code::candidate_failure) noexcept {
    if (frame->workMayBePending &&
        hipStreamSynchronize(frame->stream) != hipSuccess) {
      quarantine(frame);
      return code;
    }
    frame->workMayBePending = false;
    if (!release(*frame)) quarantine(frame);
    return code;
  };
  if (hipStreamCreateWithFlags(&frame->stream, hipStreamNonBlocking) != hipSuccess)
    return failed();
  if (hipModuleLoadData(&frame->fillModule, mdslc_rocdl_fill_image_v1) != hipSuccess ||
      hipModuleLoadData(&frame->gemmModule, mdslc_rocdl_gemm_image_v1) != hipSuccess ||
      hipModuleGetFunction(&frame->fill, frame->fillModule, kernelName) != hipSuccess ||
      hipModuleGetFunction(&frame->gemm, frame->gemmModule, kernelName) != hipSuccess)
    return failed();
  auto allocate = [](float **pointer, std::size_t count) {
    return count ? hipMalloc(reinterpret_cast<void **>(pointer), count * sizeof(float))
                 : hipSuccess;
  };
  if (allocate(&frame->a, ac) != hipSuccess ||
      allocate(&frame->b, bc) != hipSuccess ||
      allocate(&frame->c, cc) != hipSuccess)
    return failed(Code::allocation_failure);

  // An error result is not proof that a driver queued nothing. Mark pending
  // before every asynchronous operation and establish completion before free.
  frame->workMayBePending = true;
  if ((ac && hipMemcpyAsync(frame->a, frame->hostA.data(), ac * sizeof(float),
                            hipMemcpyHostToDevice, frame->stream) != hipSuccess) ||
      (bc && hipMemcpyAsync(frame->b, frame->hostB.data(), bc * sizeof(float),
                            hipMemcpyHostToDevice, frame->stream) != hipSuccess))
    return failed();
  Descriptor a(frame->a, lhs.rows, lhs.columns);
  Descriptor b(frame->b, rhs.rows, rhs.columns);
  Descriptor c(frame->c, output.rows, output.columns);
  std::array<void *, 7> fillArgs{};
  c.append(fillArgs.data());
  std::array<void *, 21> gemmArgs{};
  a.append(gemmArgs.data()); b.append(gemmArgs.data() + 7);
  c.append(gemmArgs.data() + 14);
  const auto m = static_cast<unsigned>(output.rows);
  const auto n = static_cast<unsigned>(output.columns);
  if (hipModuleLaunchKernel(frame->fill, m, n, 1, 1, 1, 1, 0, frame->stream,
                            fillArgs.data(), nullptr) != hipSuccess ||
      hipModuleLaunchKernel(frame->gemm, m, n, 1, 1, 1, 1, 0, frame->stream,
                            gemmArgs.data(), nullptr) != hipSuccess ||
      hipMemcpyAsync(frame->hostC.data(), frame->c, cc * sizeof(float),
                       hipMemcpyDeviceToHost, frame->stream) != hipSuccess)
    return failed();
  if (hipStreamSynchronize(frame->stream) != hipSuccess) return failed();
  frame->workMayBePending = false;
  if (!release(*frame)) {
    quarantine(frame);
    return Code::candidate_failure;
  }
  // No recoverable operation follows the first private host-output byte write.
  std::memcpy(output.data, frame->hostC.data(), cc * sizeof(float));
  return Code::ok;
}
} // namespace

Code rocdlCandidateAvailable() noexcept {
  return isolated([] { int device = -1; return discover(device); });
}

Code rocdlGemmCandidate(CandidateInput lhs, CandidateInput rhs,
                        CandidateOutput output) noexcept {
  ErrnoRestore preserve;
  if (lhs.columns != rhs.rows || output.rows != lhs.rows ||
      output.columns != rhs.columns) return Code::shape_mismatch;
  std::size_t ac = 0, bc = 0, cc = 0;
  if (!footprint(lhs.rows, lhs.columns, ac) ||
      !footprint(rhs.rows, rhs.columns, bc) ||
      !footprint(output.rows, output.columns, cc))
    return Code::candidate_incompatible;
  if (!closedGpuShapeCompatibleV1(lhs.rows, rhs.columns, lhs.columns))
    return Code::candidate_incompatible;
  std::uintptr_t ab, ae, bb, be, cb, ce;
  if (!range(lhs.data, ac, ab, ae) || !range(rhs.data, bc, bb, be) ||
      !range(output.data, cc, cb, ce) ||
      (cc && ((ac && cb < ae && ab < ce) || (bc && cb < be && bb < ce))))
    return Code::invalid_view;
  return isolated([&] { return execute(lhs, rhs, output, ac, bc, cc); });
}
} // namespace matcore::mdslc::runtime::closed_host_v1::detail
