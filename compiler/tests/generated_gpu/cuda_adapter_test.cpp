// Links fake Driver API definitions against the real adapter, never into the
// production runtime. This tests recoverable API failure and TLS containment.
#include "closed_cuda_candidate_v1.h"
#include "closed_gpu_images_v1.h"
#include <cuda.h>
#include <array>
#include <cerrno>
#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace detail = matcore::mdslc::runtime::closed_host_v1::detail;
using matcore::mdslc::runtime::closed_host_v1::Code;
namespace matcore::mdslc::runtime::closed_host_v1::detail {
extern const unsigned char mdslc_nvvm_fill_image_v1[64] = {};
extern const unsigned char mdslc_nvvm_gemm_image_v1[64] = {};
extern const std::size_t mdslc_nvvm_fill_image_v1_size = 64;
extern const std::size_t mdslc_nvvm_gemm_image_v1_size = 64;
}
namespace {
std::thread::id caller;
thread_local CUcontext current = nullptr;
unsigned calls = 0, failedAt = 0, frees = 0;
bool neverComplete = false, pending = false, pendingDownload = false;
bool downloadStarted = false;
const float *retainedTransfer = nullptr;
std::vector<void *> allocations;
CUresult enter() {
  if (std::this_thread::get_id() == caller) std::abort();
  errno = EDOM;
  std::fesetround(FE_UPWARD);
  std::feraiseexcept(FE_INVALID);
  ++calls;
  return calls == failedAt ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
#define ENTRY do { auto result = enter(); if (result != CUDA_SUCCESS) return result; } while (false)
}
extern "C" {
CUresult CUDAAPI cuInit(unsigned) { ENTRY; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGetCount(int *count) { ENTRY; *count = 1; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGet(CUdevice *device, int) { ENTRY; *device = 0; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDeviceGetAttribute(int *value, CUdevice_attribute attribute, CUdevice) {
  ENTRY;
  *value = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 8 :
           attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR ? 9 : 65535;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *context) { ENTRY; *context = current; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxCreate(CUcontext *context, CUctxCreateParams *, unsigned, CUdevice) {
  ENTRY; *context = current = reinterpret_cast<CUcontext>(0x1230); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxSynchronize() {
  ENTRY;
  if (neverComplete && pending && (!pendingDownload || downloadStarted)) return CUDA_ERROR_UNKNOWN;
  pending = false; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *context) {
  ENTRY; *context = current; current = nullptr; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxDestroy(CUcontext) { ENTRY; return CUDA_SUCCESS; }
CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  ENTRY; *module = reinterpret_cast<CUmodule>(const_cast<void *>(image)); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleGetFunction(CUfunction *function, CUmodule module, const char *) {
  ENTRY; *function = reinterpret_cast<CUfunction>(module); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleUnload(CUmodule) { ENTRY; return CUDA_SUCCESS; }
CUresult CUDAAPI cuMemAlloc(CUdeviceptr *pointer, std::size_t bytes) {
  ENTRY;
  auto *data = std::malloc(bytes);
  if (!data) return CUDA_ERROR_OUT_OF_MEMORY;
  allocations.push_back(data);
  *pointer = reinterpret_cast<CUdeviceptr>(data);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemFree(CUdeviceptr pointer) {
  ENTRY;
  if (pending) std::abort();
  ++frees;
  for (auto &allocation : allocations) if (allocation == reinterpret_cast<void *>(pointer)) {
    std::free(allocation); allocation = nullptr; return CUDA_SUCCESS;
  }
  std::abort();
}
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr target, const void *source, std::size_t bytes) {
  // Error postcondition deliberately leaves a transfer referencing host storage.
  pending = true;
  retainedTransfer = static_cast<const float *>(source);
  ENTRY;
  if (neverComplete && !pendingDownload) return CUDA_ERROR_UNKNOWN;
  std::memcpy(reinterpret_cast<void *>(target), source, bytes);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH(void *target, CUdeviceptr source, std::size_t bytes) {
  pending = true;
  downloadStarted = true;
  retainedTransfer = static_cast<const float *>(target);
  ENTRY;
  std::memcpy(target, reinterpret_cast<void *>(source), bytes);
  if (neverComplete && pendingDownload) return CUDA_ERROR_UNKNOWN;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned m, unsigned n, unsigned,
    unsigned, unsigned, unsigned, unsigned, CUstream, void **params, void **) {
  pending = true;
  ENTRY;
  auto field = [&](unsigned index) { return *static_cast<std::uint64_t *>(params[index]); };
  const bool fill = reinterpret_cast<const void *>(function) == detail::mdslc_nvvm_fill_image_v1;
  if (fill) {
    auto *c = reinterpret_cast<float *>(field(1));
    for (unsigned i = 0; i < m * n; ++i) c[i] = 0.0f;
  } else {
    const auto *a = reinterpret_cast<const float *>(field(1));
    const auto *b = reinterpret_cast<const float *>(field(8));
    auto *c = reinterpret_cast<float *>(field(15));
    const auto k = field(4);
    for (unsigned i = 0; i < m; ++i) for (unsigned j = 0; j < n; ++j)
      for (unsigned p = 0; p < k; ++p) c[i * n + j] += a[i * k + p] * b[p * n + j];
  }
  return CUDA_SUCCESS;
}
}

int main(int argc, char **argv) {
  caller = std::this_thread::get_id();
  if (argc == 2 && std::strcmp(argv[1], "pending") == 0) neverComplete = true;
  else if (argc == 2 && std::strcmp(argv[1], "pending-download") == 0)
    neverComplete = pendingDownload = true;
  else if (argc == 2) failedAt = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
  current = reinterpret_cast<CUcontext>(0xbeef);
  std::fesetround(FE_DOWNWARD);
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_DIVBYZERO);
  errno = EBUSY;
  std::array<float, 6> a{1, 1, 1, 1, 1, 1};
  std::array<float, 12> b{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::array<float, 8> output;
  output.fill(9876.0f);
  const auto result = detail::cudaGemmCandidate({a.data(), 2, 3}, {b.data(), 3, 4}, {output.data(), 2, 4});
  if (current != reinterpret_cast<CUcontext>(0xbeef) || errno != EBUSY ||
      std::fegetround() != FE_DOWNWARD || std::fetestexcept(FE_ALL_EXCEPT) != FE_DIVBYZERO)
    return 1;
  const bool fault = failedAt || neverComplete;
  if ((result == Code::ok) == fault) return 2;
  for (float value : output) if (value != (fault ? 9876.0f : 6.0f)) return 3;
  if (neverComplete) {
    if (frees != 0 || !retainedTransfer || retainedTransfer[0] != (pendingDownload ? 6.0f : 1.0f)) return 4;
    const auto previousCalls = calls;
    if (detail::cudaCandidateAvailable() == Code::ok || calls != previousCalls) return 5;
  }
  // Fake device allocations are test-owned; adapter quarantine deliberately
  // retains host staging. Reclaim fake device bytes after checking its contract.
  for (auto allocation : allocations) std::free(allocation);
  std::cout << "PASS CUDA adapter fault=" << (pendingDownload ? "pending-download" :
              neverComplete ? "pending" : std::to_string(failedAt))
            << " calls=" << calls << " frees=" << frees << '\n';
}
