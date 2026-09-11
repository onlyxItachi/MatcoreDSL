// A fake HIP ABI checks adapter state transitions. This is not GPU evidence.
// No injection hook exists in the production runtime; this executable links
// substitute HIP functions instead of libamdhip64.
#include "closed_rocdl_candidate_v1.h"
#include "closed_gpu_images_v1.h"
#include <hip/hip_runtime_api.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <xmmintrin.h>

namespace ch = matcore::mdslc::runtime::closed_host_v1;
namespace detail = ch::detail;
namespace {
int steps = 0, failStep = 0, failures = 0, allocations = 0, checks = 0;
bool unknownCompletion = false, pending = false, cleanupBeforeCompletion = false;
bool wrongThread = false, unavailable = false;
const std::thread::id owner = std::this_thread::get_id();
thread_local int currentDevice = 123;
thread_local hipError_t lastError = hipErrorInvalidValue;
void check(bool pass, const char *what) {
  ++checks;
  if (!pass) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
hipError_t step() {
  ++steps;
  if (std::this_thread::get_id() == owner) wrongThread = true;
  errno = ERANGE;
  lastError = steps == failStep ? hipErrorUnknown : hipSuccess;
  return lastError;
}
template <typename T> T fake(std::uintptr_t id) { return reinterpret_cast<T>(id); }
void numericalKernel(void **args, bool fill, unsigned m, unsigned n) {
  auto data = [&](int offset) { return *static_cast<float **>(args[offset + 1]); };
  auto dim = [&](int offset) { return *static_cast<std::int64_t *>(args[offset]); };
  if (fill) {
    for (unsigned i = 0; i < m*n; ++i) data(0)[i] = 0.0f;
  } else {
    const auto k = dim(4);
    for (unsigned i = 0; i < m; ++i)
      for (unsigned j = 0; j < n; ++j)
        for (std::int64_t p = 0; p < k; ++p) {
          volatile float product = data(0)[i*k+p] * data(7)[p*n+j];
          data(14)[i*n+j] = data(14)[i*n+j] + product;
        }
  }
}
} // namespace

namespace matcore::mdslc::runtime::closed_host_v1::detail {
const unsigned char mdslc_rocdl_fill_image_v1[] = {1};
const unsigned char mdslc_rocdl_gemm_image_v1[] = {2};
const std::size_t mdslc_rocdl_fill_image_v1_size = 1;
const std::size_t mdslc_rocdl_gemm_image_v1_size = 1;
}
extern "C" {
hipError_t hipGetDeviceCount(int *count) {
  auto e = step(); if (e == hipSuccess) *count = unavailable ? 0 : 1; return e;
}
hipError_t hipGetDeviceProperties(hipDeviceProp_t *p, int) {
  auto e = step(); if (e != hipSuccess) return e;
  *p = {}; std::strcpy(p->gcnArchName, "gfx1150");
  p->maxGridSize[0] = p->maxGridSize[1] = 65535;
  p->maxGridSize[2] = p->maxThreadsPerBlock = 1; return e;
}
hipError_t hipSetDevice(int device) {
  auto e = step(); if (e == hipSuccess) currentDevice = device; return e;
}
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned flags) {
  auto e = step(); if (e != hipSuccess) return e;
  if (flags != hipStreamNonBlocking) return hipErrorInvalidValue;
  *stream = fake<hipStream_t>(1); return e;
}
hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
  auto e = step(); if (e != hipSuccess) return e;
  *module = fake<hipModule_t>(image == detail::mdslc_rocdl_fill_image_v1 ? 2 : 3);
  return e;
}
hipError_t hipModuleGetFunction(hipFunction_t *fn, hipModule_t module, const char *name) {
  auto e = step(); if (e != hipSuccess) return e;
  if (std::strcmp(name, "__matcore_strict_gemm_f32_v1_kernel")) return hipErrorInvalidValue;
  *fn = fake<hipFunction_t>(reinterpret_cast<std::uintptr_t>(module)); return e;
}
hipError_t hipMalloc(void **out, std::size_t size) {
  auto e = step(); if (e != hipSuccess) return e;
  *out = std::malloc(size); if (!*out) return hipErrorOutOfMemory;
  ++allocations; return e;
}
hipError_t hipMemcpyAsync(void *to, const void *from, std::size_t size,
                          hipMemcpyKind, hipStream_t stream) {
  auto e = step(); pending = true;
  if (!stream) return hipErrorInvalidValue;
  if (e == hipSuccess) std::memcpy(to, from, size);
  return e;
}
hipError_t hipModuleLaunchKernel(hipFunction_t fn, unsigned x, unsigned y,
    unsigned z, unsigned tx, unsigned ty, unsigned tz, unsigned shared,
    hipStream_t stream, void **args, void **) {
  auto e = step(); pending = true;
  if (e != hipSuccess) return e;
  if (z != 1 || tx != 1 || ty != 1 || tz != 1 || shared || !stream)
    return hipErrorInvalidValue;
  numericalKernel(args, fn == fake<hipFunction_t>(2), x, y); return e;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  auto e = step();
  if (unknownCompletion || e != hipSuccess) return hipErrorUnknown;
  pending = false; return e;
}
hipError_t hipFree(void *pointer) {
  auto e = step();
  if (pending) cleanupBeforeCompletion = true;
  if (e == hipSuccess) { std::free(pointer); --allocations; }
  return e;
}
hipError_t hipModuleUnload(hipModule_t) {
  auto e = step(); if (pending) cleanupBeforeCompletion = true; return e;
}
hipError_t hipStreamDestroy(hipStream_t) {
  auto e = step(); if (pending) cleanupBeforeCompletion = true; return e;
}
} // extern C

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const char *mode = argv[1];
  if (std::strncmp(mode, "fault-", 6) == 0) failStep = std::atoi(mode + 6);
  if (!std::strcmp(mode, "unknown-completion")) unknownCompletion = true;
  if (!std::strcmp(mode, "unavailable")) unavailable = true;
  std::array<float, 6> a{1,2,3,4,5,6};
  std::array<float, 8> b{1,2,3,4,5,6,7,8};
  std::array<float, 12> output; output.fill(-77);
  const auto originalA = a;
  const auto originalB = b;
  std::fenv_t original;
  std::fegetenv(&original);
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT | FE_UNDERFLOW);
  _mm_setcsr(_mm_getcsr() | 0x8040u);
  const auto originalMxcsr = _mm_getcsr();
  const int originalFlags = std::fetestexcept(FE_ALL_EXCEPT);
  errno = EDOM;

  ch::Code code;
  if (!std::strcmp(mode, "shape")) {
    code = detail::rocdlGemmCandidate({a.data(),3,2}, {b.data(),3,4},
                                     {output.data(),3,4});
    check(code == ch::Code::shape_mismatch && steps == 0, "shape rejected before HIP");
  } else if (!std::strcmp(mode, "oversize")) {
    code = detail::rocdlGemmCandidate({a.data(),65536,2}, {b.data(),2,4},
                                     {output.data(),65536,4});
    check(code == ch::Code::candidate_incompatible && steps == 0,
          "schedule extent rejected before storage access");
  } else if (!std::strcmp(mode, "output-alias")) {
    code = detail::rocdlGemmCandidate({a.data(),3,2}, {b.data(),2,4},
                                     {a.data(),3,4});
    check(code == ch::Code::invalid_view && steps == 0, "overlapping destination rejected");
  } else if (!std::strcmp(mode, "output-limit") || !std::strcmp(mode, "zero-K-limit")) {
    const auto k = !std::strcmp(mode, "zero-K-limit") ? 0U : 1U;
    code = detail::rocdlGemmCandidate({a.data(),1025,k}, {b.data(),k,1025},
                                     {output.data(),1025,1025});
    check(code == ch::Code::candidate_incompatible && steps == 0,
          "bounded output-grid qualification includes zero-K fill");
  } else if (!std::strcmp(mode, "work-limit")) {
    code = detail::rocdlGemmCandidate({a.data(),1024,65}, {b.data(),65,1024},
                                     {output.data(),1024,1024});
    check(code == ch::Code::candidate_incompatible && steps == 0,
          "serial-K work limit checked before allocation or launch");
  } else if (!std::strcmp(mode, "null")) {
    code = detail::rocdlGemmCandidate({nullptr,3,2}, {b.data(),2,4},
                                     {output.data(),3,4});
    check(code == ch::Code::invalid_view && steps == 0, "nonempty null rejected");
  } else if (!std::strcmp(mode, "environment")) {
    setenv("HSA_OVERRIDE_GFX_VERSION", "11.5.0", 1);
    code = detail::rocdlCandidateAvailable();
    unsetenv("HSA_OVERRIDE_GFX_VERSION");
    check(code == ch::Code::candidate_unavailable && steps == 0,
          "spoofing cannot authorize even matching target");
  } else {
    code = detail::rocdlGemmCandidate({a.data(),3,2}, {b.data(),2,4},
                                     {output.data(),3,4});
    if (!std::strcmp(mode, "good")) {
      check(code == ch::Code::ok, "mock numerical request succeeds");
      const std::array<float,12> expected{11,14,17,20,23,30,37,44,35,46,57,68};
      check(output == expected, "correct private output issued after cleanup");
      check(allocations == 0, "normal path releases all buffers");
    } else {
      check(code != ch::Code::ok, "injected missing capability or driver error fails");
      for (float value : output) check(value == -77, "failure leaves output unchanged");
    }
  }
  if (unknownCompletion) {
    check(allocations == 3 && pending, "unknown completion retains all device buffers");
    const auto oldSteps = steps;
    check(detail::rocdlCandidateAvailable() == ch::Code::candidate_failure,
          "quarantine poisons future availability");
    check(steps == oldSteps, "poisoned candidate never touches HIP again");
  }
  check(!cleanupBeforeCompletion, "never release resources before completion proof");
  check(!wrongThread, "HIP calls run only on isolated worker thread");
  check(currentDevice == 123 && lastError == hipErrorInvalidValue,
        "caller HIP device and error TLS remain unchanged");
  check(errno == EDOM && _mm_getcsr() == originalMxcsr &&
        std::fegetround() == FE_DOWNWARD &&
        std::fetestexcept(FE_ALL_EXCEPT) == originalFlags,
        "caller errno and complete FP state preserved");
  check(a == originalA && b == originalB, "inputs remain immutable");
  std::fesetenv(&original);
  std::printf("MOCK %s checks=%d failures=%d api_steps=%d retained_buffers=%d\n",
               mode, checks, failures, steps, allocations);
  return failures ? 1 : 0;
}
