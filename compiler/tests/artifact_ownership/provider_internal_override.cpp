// Manual falsification instrument, not a supported provider adapter. Run only
// against the selected Linux OpenBLAS with OPENBLAS_NUM_THREADS=1. The host hook
// preserves provider allocation/results while demonstrating a hidden host effect.
#include <cblas.h>
#include <dlfcn.h>
#include <cstdio>
#include <vector>

using Allocate = void *(*)(int);
static Allocate provider_allocate;
static unsigned host_observations;

extern "C" void *blas_memory_alloc(int slot) {
  // OpenBLAS may allocate during initialization, before main. Resolving only in
  // main was an invalid first harness and is not the successful counterexample.
  if (!provider_allocate)
    provider_allocate = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "blas_memory_alloc"));
  ++host_observations;
  return provider_allocate(slot);
}

int main() {
  provider_allocate = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "blas_memory_alloc"));
  if (!provider_allocate) return 2;
  constexpr int extent = 512;
  std::vector<float> a(extent * extent, 1.0f), b(extent * extent, 1.0f), c(extent * extent, 0.0f);
  const auto before = host_observations;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, extent, extent,
              extent, 1.0f, a.data(), extent, b.data(), extent, 0.0f,
              c.data(), extent);
  bool correct = true;
  for (const auto value : c) correct = correct && value == extent;
  std::printf("correct=%d hidden_host_calls=%u\n", correct, host_observations - before);
  return correct && host_observations > before ? 0 : 1;
}
