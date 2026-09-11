// Standalone physical-execution falsifier. Not a production registry adapter.
#include <hip/hip_runtime_api.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
bool cleanupFailed = false;
void cleanupCheck(hipError_t error, const char *operation) noexcept {
  if (error != hipSuccess) {
    cleanupFailed = true;
    std::fprintf(stderr, "cleanup failure: %s: %s\n", operation,
                 hipGetErrorString(error));
  }
}
void hipCheck(hipError_t error, const char *operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}
struct Module {
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
  explicit Module(const char *path) {
    hipCheck(hipModuleLoad(&module, path), "hipModuleLoad");
    try {
      hipCheck(hipModuleGetFunction(&function, module,
               "__matcore_strict_gemm_f32_v1_kernel"), "hipModuleGetFunction");
    } catch (...) {
      cleanupCheck(hipModuleUnload(module), "hipModuleUnload failed module");
      throw;
    }
  }
  void close() noexcept {
    if (module) cleanupCheck(hipModuleUnload(module), "hipModuleUnload");
    module = nullptr;
  }
  ~Module() { close(); }
};
struct Buffer {
  float *base = nullptr;
  std::size_t count;
  static constexpr float guard = 13579.25f;
  explicit Buffer(std::size_t size) : count(size) {
    hipCheck(hipMalloc(reinterpret_cast<void **>(&base),
                       (count + 2) * sizeof(float)), "hipMalloc");
    try {
      std::vector<float> initial(count + 2, guard);
      hipCheck(hipMemcpy(base, initial.data(), initial.size() * sizeof(float),
                         hipMemcpyHostToDevice), "hipMemcpy guards");
    } catch (...) {
      cleanupCheck(hipFree(base), "hipFree failed allocation initialization");
      throw;
    }
  }
  ~Buffer() { if (base) cleanupCheck(hipFree(base), "hipFree"); }
  Buffer(const Buffer &) = delete;
  float *data() { return base + 1; }
  void upload(const std::vector<float> &v) {
    if (v.size() != count) throw std::runtime_error("input shape mismatch");
    if (count) hipCheck(hipMemcpy(data(), v.data(), count * sizeof(float),
                                 hipMemcpyHostToDevice), "hipMemcpy input");
  }
  std::vector<float> download() {
    std::vector<float> v(count + 2);
    hipCheck(hipMemcpy(v.data(), base, v.size() * sizeof(float),
                        hipMemcpyDeviceToHost), "hipMemcpy result");
    if (v.front() != guard || v.back() != guard)
      throw std::runtime_error("GPU crossed guarded allocation boundary");
    return {v.begin() + 1, v.end() - 1};
  }
};
// MLIR's canonical flattened rank-2 descriptor ABI: not a public Matcore ABI.
struct Descriptor {
  float *allocated, *aligned;
  std::int64_t offset = 0, rows, columns, rowStride, columnStride = 1;
  Descriptor(Buffer &buffer, std::int64_t m, std::int64_t n)
      : allocated(buffer.base), aligned(buffer.data()), rows(m), columns(n),
        rowStride(n) {}
  void append(std::vector<void *> &args) {
    args.insert(args.end(), {&allocated, &aligned, &offset, &rows, &columns,
                             &rowStride, &columnStride});
  }
};
bool same(float x, float y) {
  return (std::isnan(x) && std::isnan(y)) ||
         std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
}
std::vector<float> oracle(const std::vector<float> &a,
                          const std::vector<float> &b,
                          std::size_t m, std::size_t n, std::size_t k) {
  std::vector<float> result(m * n, 0.0f);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      volatile float sum = 0.0f;
      for (std::size_t p = 0; p < k; ++p) {
        volatile float product = a[i * k + p] * b[p * n + j];
        sum = sum + product;
      }
      result[i * n + j] = sum;
    }
  return result;
}
struct Checks { unsigned cases = 0, values = 0, launches = 0; };
void runCase(Module &fill, Module &gemm, Checks &checks, const char *name,
             std::size_t m, std::size_t n, std::size_t k,
             std::vector<float> a, std::vector<float> b, bool alias = false) {
  if (a.size() != m * k || b.size() != k * n ||
      (alias && a != b)) throw std::runtime_error("bad fixture");
  // This specimen maps output rows/columns directly to GPU grid x/y. This bound
  // is deliberately conservative, and checked before any allocation or launch.
  if (m > 65535 || n > 65535 || k > 65535)
    throw std::runtime_error("prototype launch extent outside bounded contract");
  const auto expected = oracle(a, b, m, n, k);
  Buffer ad(a.size()), bd(b.size()), cd(m * n);
  ad.upload(a);
  if (!alias) bd.upload(b);
  Descriptor da(ad, m, k), db(alias ? ad : bd, k, n), dc(cd, m, n);
  if (m && n) {
    std::vector<void *> fillArgs;
    dc.append(fillArgs);
    hipCheck(hipModuleLaunchKernel(fill.function, m, n, 1, 1, 1, 1, 0,
                                   nullptr, fillArgs.data(), nullptr),
             "hipModuleLaunchKernel fill");
    ++checks.launches;
    std::vector<void *> gemmArgs;
    da.append(gemmArgs); db.append(gemmArgs); dc.append(gemmArgs);
    hipCheck(hipModuleLaunchKernel(gemm.function, m, n, 1, 1, 1, 1, 0,
                                   nullptr, gemmArgs.data(), nullptr),
             "hipModuleLaunchKernel gemm");
    ++checks.launches;
    hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
  }
  auto actual = cd.download();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    ++checks.values;
    if (!same(actual[i], expected[i])) {
      std::fprintf(stderr, "%s[%zu]: actual=%a (%08x) expected=%a (%08x)\n",
          name, i, actual[i], std::bit_cast<std::uint32_t>(actual[i]),
          expected[i], std::bit_cast<std::uint32_t>(expected[i]));
      throw std::runtime_error("strict numerical mismatch");
    }
  }
  auto afterA = ad.download();
  auto afterB = alias ? afterA : bd.download();
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!same(a[i], afterA[i])) throw std::runtime_error("lhs mutated");
  for (std::size_t i = 0; i < b.size(); ++i)
    if (!same(b[i], afterB[i])) throw std::runtime_error("rhs mutated");
  ++checks.cases;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: runner fill.hsaco gemm.hsaco");
    if (std::getenv("HSA_OVERRIDE_GFX_VERSION"))
      throw std::runtime_error("gfx override is forbidden in physical evidence");
    if (std::fegetround() != FE_TONEAREST)
      throw std::runtime_error("oracle requires nearest-even rounding");
    int count = 0, selected = -1;
    hipCheck(hipGetDeviceCount(&count), "hipGetDeviceCount");
    hipDeviceProp_t properties{};
    for (int i = 0; i < count; ++i) {
      hipCheck(hipGetDeviceProperties(&properties, i), "hipGetDeviceProperties");
      const std::string arch = properties.gcnArchName;
      if (arch == "gfx1150" || arch.starts_with("gfx1150:")) {
        selected = i; break;
      }
    }
    if (selected < 0) throw std::runtime_error("physical gfx1150 is unavailable");
    hipCheck(hipSetDevice(selected), "hipSetDevice");
    std::printf("PHYSICAL device=%s arch=%s gridLimit=%d,%d,%d\n",
        properties.name, properties.gcnArchName, properties.maxGridSize[0],
        properties.maxGridSize[1], properties.maxGridSize[2]);
    if (properties.maxGridSize[0] < 65535 || properties.maxGridSize[1] < 65535)
      throw std::runtime_error("device grid does not satisfy prototype contract");
    Module fill(argv[1]), gemm(argv[2]);
    Checks checks;
    std::uint32_t random = 0x8f23a519;
    auto next = [&] { random ^= random << 13; random ^= random >> 17;
                      random ^= random << 5; return random; };
    for (unsigned test = 0; test < 64; ++test) {
      std::size_t m = next() % 34, n = next() % 38, k = next() % 36;
      std::vector<float> a(m * k), b(k * n);
      for (auto &value : a) value = (static_cast<int>(next() % 129) - 64) / 16.f;
      for (auto &value : b) value = (static_cast<int>(next() % 129) - 64) / 16.f;
      runCase(fill, gemm, checks, "rectangular-random", m, n, k, a, b);
    }
    runCase(fill, gemm, checks, "lhs-rhs-alias", 3, 3, 3,
            {1,2,3,4,5,6,7,8,9}, {1,2,3,4,5,6,7,8,9}, true);
    runCase(fill, gemm, checks, "zero-reduction", 3, 7, 0, {}, {});
    runCase(fill, gemm, checks, "zero-M", 0, 7, 3, {}, std::vector<float>(21));
    runCase(fill, gemm, checks, "zero-N", 3, 0, 7, std::vector<float>(21), {});
    runCase(fill, gemm, checks, "fma-discriminator", 1, 1, 2,
            {-1.f, 1.f + 0x1p-23f}, {1.f, 1.f - 0x1p-23f});
    runCase(fill, gemm, checks, "reduction-order", 1, 1, 4,
            {0x1p24f, 1.f, -0x1p24f, 1.f}, {1,1,1,1});
    runCase(fill, gemm, checks, "subnormal-output", 1, 1, 1, {0x1p-126f}, {.5f});
    runCase(fill, gemm, checks, "subnormal-input", 1, 1, 1, {0x1p-149f}, {1.f});
    runCase(fill, gemm, checks, "signed-zero", 1, 1, 1, {-0.f}, {1.f});
    runCase(fill, gemm, checks, "nan", 1, 1, 1,
            {std::numeric_limits<float>::quiet_NaN()}, {1.f});
    runCase(fill, gemm, checks, "infinity", 1, 1, 1,
            {std::numeric_limits<float>::infinity()}, {1.f});
    runCase(fill, gemm, checks, "infinity-times-zero", 1, 1, 1,
            {std::numeric_limits<float>::infinity()}, {0.f});
    gemm.close(); fill.close();
    if (cleanupFailed) throw std::runtime_error("resource cleanup failed");
    std::printf("PASS cases=%u compared_values=%u physical_kernel_launches=%u "
                "strict_bitwise_or_nan_class=1 guarded_allocations=1\n",
                checks.cases, checks.values, checks.launches);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
