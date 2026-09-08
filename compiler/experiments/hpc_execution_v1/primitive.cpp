#include "harness.h"
#ifndef MDSLC_EXPERIMENT_ENTRY
#define MDSLC_EXPERIMENT_ENTRY _mlir_ciface___matcore_strict_gemm_f32_v1
#endif
struct Memref {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
static_assert(sizeof(Memref) == 56 && alignof(Memref) == 8);
extern "C" void MDSLC_EXPERIMENT_ENTRY(Memref *, Memref *, Memref *);
namespace hpc {
const bool is_region = false;
const char *backend = "standalone_primitive_compute";
Outcome invoke(float *a, float *b, float *c, unsigned long long m,
               unsigned long long n, unsigned long long k, int) {
  Memref av{a, a, 0, {std::int64_t(m), std::int64_t(k)}, {std::int64_t(k), 1}};
  Memref bv{b, b, 0, {std::int64_t(k), std::int64_t(n)}, {std::int64_t(n), 1}};
  Memref cv{c, c, 0, {std::int64_t(m), std::int64_t(n)}, {std::int64_t(n), 1}};
  MDSLC_EXPERIMENT_ENTRY(&av, &bv, &cv);
  return {true, Error::none, 0, 0, 0, 0, 0};
}
}
int main(int argc, char **argv) { return hpc::main(argc, argv); }
