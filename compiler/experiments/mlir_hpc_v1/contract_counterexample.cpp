#include <bit>
#include <cstdint>
#include <cstdio>
struct Memref {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_research_gemm(Memref *, Memref *, Memref *);
int main() {
  float a[2]{-1.0f, 0x1.000002p0f}, b[2]{1.0f, 0x1.fffffep-1f}, c = 17;
  Memref av{a, a, 0, {1, 2}, {2, 1}}, bv{b, b, 0, {2, 1}, {1, 1}},
      cv{&c, &c, 0, {1, 1}, {1, 1}};
  _mlir_ciface_research_gemm(&av, &bv, &cv);
  volatile float p0 = a[0] * b[0];
  volatile float s0 = 0.0f + p0;
  volatile float p1 = a[1] * b[1];
  volatile float expected = s0 + p1;
  std::printf("contract candidate bits=%08x strict expected=%08x\n",
              std::bit_cast<unsigned>(c), std::bit_cast<unsigned>(float(expected)));
  if (std::bit_cast<unsigned>(float(expected)) != 0) return 2;
  const auto actual = std::bit_cast<unsigned>(c);
  return actual == 0 ? 0 : actual == 0x337ffffe ? 1 : 2;
}
