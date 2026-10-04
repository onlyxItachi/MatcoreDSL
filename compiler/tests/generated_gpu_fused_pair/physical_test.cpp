// Actual device launch, no device arithmetic supplied by this harness.
// Research-only fail-stop protocol: unknown completion does not expose output
// or reclaim possibly-live buffers. It proves no source/frontier status law.
#if defined(RESEARCH_CUDA)
#include <cuda.h>
#else
#include <hip/hip_runtime_api.h>
#endif
#include <array>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <xmmintrin.h>

constexpr char kernelName[] = "__matcore_strict_fused_gemm_f32_v1_kernel";
constexpr uint32_t canaryBits = 0x4a792d63;
static uint64_t outputChecks = 0, canaryChecks = 0, inputChecks = 0;
static unsigned launched = 0, bypassed = 0;
static bool negativeWorkspace = false;
static void require(bool yes, const std::string &message) {
  if (!yes) throw std::runtime_error(message);
}
// Exiting without cleanup deliberately leaves unresolved resources live for
// process lifetime. This is not recoverable runtime quarantine or authority.
[[noreturn]] static void failStop(const char *operation, int status) {
  std::fprintf(stderr, "FAIL GPU research fail-stop op=%s status=%d; "
      "no output exposed, no possibly-live resource reclaimed\n", operation, status);
  std::fflush(stderr);
  std::_Exit(2);
}
#if defined(RESEARCH_CUDA)
using DevicePointer = CUdeviceptr;
using Module = CUmodule;
using Function = CUfunction;
using Context = CUcontext;
static void check(CUresult status, const char *operation) {
  if (status != CUDA_SUCCESS) failStop(operation, status);
}
static Context context;
static void initialize() {
  check(cuInit(0), "cuInit");
  int count = 0, driver = 0;
  check(cuDeviceGetCount(&count), "cuDeviceGetCount");
  check(cuDriverGetVersion(&driver), "cuDriverGetVersion");
  bool found = false;
  for (int i = 0; i < count; ++i) {
    CUdevice device;
    check(cuDeviceGet(&device, i), "cuDeviceGet");
    int major = 0, minor = 0;
    char name[256]{};
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                               device), "cuDeviceGetAttribute major");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               device), "cuDeviceGetAttribute minor");
    check(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    if (major != 8 || minor != 9 || !std::strstr(name, "RTX 4060")) continue;
    check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");
    std::printf("physical=NVVM device=%s sm=%d%d driver=%d\n", name, major, minor, driver);
    found = true; break;
  }
  require(found, "physical RTX4060 sm89 unavailable: no fallback/skip");
}
static void allocate(DevicePointer &pointer, size_t bytes) {
  check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
}
static void upload(DevicePointer pointer, const void *host, size_t bytes) {
  check(cuMemcpyHtoD(pointer, host, bytes), "cuMemcpyHtoD");
}
static void download(void *host, DevicePointer pointer, size_t bytes) {
  check(cuMemcpyDtoH(host, pointer, bytes), "cuMemcpyDtoH");
}
static void release(DevicePointer pointer) { check(cuMemFree(pointer), "cuMemFree"); }
static void load(Module &module, Function &function, const void *image) {
  check(cuModuleLoadData(&module, image), "cuModuleLoadData");
  check(cuModuleGetFunction(&function, module, kernelName), "cuModuleGetFunction");
}
static void launch(Function function, void **arguments) {
  check(cuLaunchKernel(function, 1,1,1, 1,1,1, 0,nullptr, arguments,nullptr),
        "cuLaunchKernel");
  check(cuCtxSynchronize(), "cuCtxSynchronize");
}
static void shutdown(Module module) {
  check(cuModuleUnload(module), "cuModuleUnload");
  check(cuCtxDestroy(context), "cuCtxDestroy");
}
#else
using DevicePointer = void *;
using Module = hipModule_t;
using Function = hipFunction_t;
static void check(hipError_t status, const char *operation) {
  if (status != hipSuccess) failStop(operation, status);
}
static void initialize() {
  require(std::getenv("HSA_OVERRIDE_GFX_VERSION") == nullptr,
          "HSA_OVERRIDE_GFX_VERSION present: physical qualification refused");
  int count = 0, runtime = 0;
  check(hipGetDeviceCount(&count), "hipGetDeviceCount");
  check(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
  bool found = false;
  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, i), "hipGetDeviceProperties");
    std::string arch=props.gcnArchName;
    if (!arch.starts_with("gfx1150") || (arch.size()>7 && arch[7]!=':')) continue;
    check(hipSetDevice(i), "hipSetDevice");
    std::printf("physical=ROCDL device=%s arch=%s runtime=%d HSA_OVERRIDE_GFX_VERSION=absent\n",
                props.name, props.gcnArchName, runtime);
    found = true; break;
  }
  require(found, "physical Radeon890M gfx1150 unavailable: no fallback/skip");
}
static void allocate(DevicePointer &pointer, size_t bytes) {
  check(hipMalloc(&pointer, bytes), "hipMalloc");
}
static void upload(DevicePointer pointer, const void *host, size_t bytes) {
  check(hipMemcpyHtoD(pointer, const_cast<void *>(host), bytes), "hipMemcpyHtoD");
}
static void download(void *host, DevicePointer pointer, size_t bytes) {
  check(hipMemcpyDtoH(host, pointer, bytes), "hipMemcpyDtoH");
}
static void release(DevicePointer pointer) { check(hipFree(pointer), "hipFree"); }
static void load(Module &module, Function &function, const void *image) {
  check(hipModuleLoadData(&module, image), "hipModuleLoadData");
  check(hipModuleGetFunction(&function, module, kernelName), "hipModuleGetFunction");
}
static void launch(Function function, void **arguments) {
  check(hipModuleLaunchKernel(function, 1,1,1, 1,1,1, 0,nullptr, arguments,nullptr),
        "hipModuleLaunchKernel");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}
static void shutdown(Module module) { check(hipModuleUnload(module), "hipModuleUnload"); }
#endif

static uint64_t address(DevicePointer pointer) {
#if defined(RESEARCH_CUDA)
  return pointer;
#else
  return reinterpret_cast<uintptr_t>(pointer);
#endif
}
static DevicePointer offset(DevicePointer pointer, size_t bytes) {
#if defined(RESEARCH_CUDA)
  return pointer + bytes;
#else
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(pointer) + bytes);
#endif
}
// Both targets' audited kernel ABI is ptr,ptr,i64,i64,i64,i64,i64 per memref.
struct Descriptor {
  uint64_t allocated, aligned;
  int64_t offset, rows, columns, rowStride, columnStride;
};
static_assert(sizeof(Descriptor) == 56);
struct Shape { int64_t m, k, n, p; };
static size_t extent(int64_t rows, int64_t columns) {
  require(rows >= 0 && columns >= 0, "negative dimension");
  constexpr auto maximum = static_cast<uint64_t>(PTRDIFF_MAX) / sizeof(float);
  require(!columns || static_cast<uint64_t>(rows) <= maximum / columns,
          "extent_overflow");
  return static_cast<size_t>(rows) * static_cast<size_t>(columns);
}
static void guards(Shape shape) {
  // Isolated interface order only, not original source guards/status protocol.
  extent(shape.m, shape.k); extent(shape.k, shape.n);
  extent(shape.m, shape.n); // Full logical C, even empty final E.
  extent(shape.n, shape.p); extent(shape.m, shape.p);
  extent(std::min<int64_t>(shape.m,4), shape.n);
}
static void workBound(Shape s) {
  require(s.m<=65535 && s.k<=65535 && s.n<=65535 && s.p<=65535,
          "research display-GPU shape bound");
  require(static_cast<uint64_t>(s.m)*s.n*(s.k+s.p)<=(uint64_t{1}<<18),
          "research pair-work bound");
}
static std::vector<float> oracle(Shape s, const std::vector<float> &a,
                                const std::vector<float> &b,
                                const std::vector<float> &d) {
  // Full host C is intentional independent arithmetic reference, never GPU
  // workspace. Each volatile assignment enforces the source f32 boundary.
  std::vector<float> c(extent(s.m,s.n)), e(extent(s.m,s.p));
  for (int64_t i=0;i<s.m;++i) for (int64_t n=0;n<s.n;++n) {
    volatile float sum = 0.0f;
    for (int64_t k=0;k<s.k;++k) {
      volatile float product = a[i*s.k+k]*b[k*s.n+n];
      volatile float next = sum + product; sum = next;
    }
    c[i*s.n+n] = sum;
  }
  for (int64_t i=0;i<s.m;++i) for (int64_t p=0;p<s.p;++p) {
    volatile float sum = 0.0f;
    for (int64_t n=0;n<s.n;++n) {
      volatile float product = c[i*s.n+n]*d[n*s.p+p];
      volatile float next = sum + product; sum = next;
    }
    e[i*s.p+p] = sum;
  }
  return e;
}
struct Case {
  Shape s;
  std::vector<float> a,b,d;
  std::string name;
  bool aliasAll = false;
};
static void runCase(Function function, const Case &test) {
  const auto s = test.s;
  guards(s);
  require(extent(s.m,s.k) == test.a.size() && extent(s.k,s.n) == test.b.size() &&
          extent(s.n,s.p) == test.d.size(), "input extent differs: " + test.name);
  auto expected = oracle(s,test.a,test.b,test.d);
  if (expected.empty() || s.n == 0) {
    // Empty E: no loop. N=0: strict +0 second empty reduction, no leaf.
    for (float value : expected) require(std::bit_cast<uint32_t>(value)==0,
                                        "N0 oracle did not return +0");
    ++bypassed; return;
  }
  // Hard diagnostic work bounds; never launch enormous cases on display GPUs.
  workBound(s);
  std::array<DevicePointer,5> storage{};
  std::array<Descriptor,5> descriptors{};
  const std::array<std::vector<float>,3> inputs{test.a,test.b,test.d};
  const std::array<int64_t,5> rows{s.m,s.k,s.n,s.m,std::min<int64_t>(4,s.m)};
  const std::array<int64_t,5> cols{s.k,s.n,s.p,s.p,s.n};
  std::array<std::vector<float>,5> host;
  for (unsigned i=0;i<5;++i) {
    size_t count = extent(rows[i],cols[i]);
    if (i<3) host[i]=inputs[i];
    else host[i]=std::vector<float>(count,std::bit_cast<float>(canaryBits));
    if (i<3 && !count) {
      descriptors[i]={0,0,0,rows[i],cols[i],cols[i],1};
      continue; // K=0 A/B are genuinely null empty inputs.
    }
    if (i<3 && i>0 && test.aliasAll) {
      require(inputs[i]==inputs[0] && count==inputs[0].size(), "bad alias fixture");
      storage[i]=storage[0]; descriptors[i]=descriptors[0]; continue;
    }
    // One leading/trailing canary around every nonempty immutable/private span.
    size_t physicalCount = negativeWorkspace && i==4 ? 1 : count;
    std::vector<float> initial(physicalCount+2,std::bit_cast<float>(canaryBits));
    std::copy_n(host[i].begin(),physicalCount,initial.begin()+1);
    allocate(storage[i],initial.size()*sizeof(float));
    upload(storage[i],initial.data(),initial.size()*sizeof(float));
    descriptors[i]={address(storage[i]),address(offset(storage[i],sizeof(float))),
                    0,rows[i],cols[i],cols[i],1};
  }
  std::array<void *,35> arguments{};
  for (unsigned i=0;i<5;++i) {
    auto &d=descriptors[i];
    std::array<void *,7> fields{&d.allocated,&d.aligned,&d.offset,&d.rows,
                               &d.columns,&d.rowStride,&d.columnStride};
    std::copy(fields.begin(),fields.end(),arguments.begin()+7*i);
  }
  launch(function,arguments.data()); // Completion must precede any output use.
  ++launched;
  for (unsigned i=0;i<5;++i) {
    if (!address(storage[i])) continue;
    size_t count=extent(rows[i],cols[i]);
    std::vector<float> actual(count+2);
    download(actual.data(),storage[i],actual.size()*sizeof(float));
    require(std::bit_cast<uint32_t>(actual.front())==canaryBits &&
            std::bit_cast<uint32_t>(actual.back())==canaryBits,
            "device write escaped private span: " + test.name);
    canaryChecks+=2;
    if (i<3) for (size_t j=0;j<count;++j) {
      require(std::bit_cast<uint32_t>(actual[j+1])==std::bit_cast<uint32_t>(host[i][j]),
              "immutable input modified: " + test.name); ++inputChecks;
    }
    if (i==3) for (size_t j=0;j<count;++j) {
      const bool match=std::isnan(expected[j]) ? std::isnan(actual[j+1]) :
          std::bit_cast<uint32_t>(actual[j+1])==std::bit_cast<uint32_t>(expected[j]);
      require(match,"strict output mismatch " + test.name + " index=" +
          std::to_string(j) + " expected_bits=" +
          std::to_string(std::bit_cast<uint32_t>(expected[j])) + " actual_bits=" +
          std::to_string(std::bit_cast<uint32_t>(actual[j+1]))); ++outputChecks;
    }
  }
  // All kernels are known complete; only then reclaim storage.
  for (unsigned i=0;i<5;++i)
    if (address(storage[i]) && !(test.aliasAll && i>0 && i<3)) release(storage[i]);
}

static std::vector<Case> cases() {
  std::vector<Case> result;
  std::mt19937 rng(0x684197);
  auto randomValues=[&](size_t n) {
    std::vector<float> values(n);
    for (auto &v:values) v=(static_cast<int>(rng()%129)-64)*0.03125f;
    return values;
  };
  for (int m=0;m<=9;++m) for (int k: {0,1,3,5})
    for (int n: {0,1,2,5}) for (int p: {0,1,3}) {
      Shape s{m,k,n,p}; result.push_back({s,randomValues(extent(m,k)),
          randomValues(extent(k,n)),randomValues(extent(n,p)),"tiny/tail"});
    }
  for (unsigned i=0;i<32;++i) {
    Shape s{static_cast<int64_t>(1+rng()%33), static_cast<int64_t>(rng()%36),
            static_cast<int64_t>(1+rng()%37), static_cast<int64_t>(1+rng()%31)};
    result.push_back({s,randomValues(extent(s.m,s.k)),randomValues(extent(s.k,s.n)),
                       randomValues(extent(s.n,s.p)),"seeded-mixed"});
  }
  for (Shape s: {Shape{8,128,128,128},Shape{64,32,64,32},Shape{65,31,64,32},\n                Shape{1,65535,1,1},Shape{1,0,65535,1},Shape{65535,1,1,1},\n                Shape{1,0,1,65535}})
    result.push_back({s,randomValues(extent(s.m,s.k)),randomValues(extent(s.k,s.n)),
                       randomValues(extent(s.n,s.p)),"pair-work-cap-boundary/tail"});
  float inf=std::numeric_limits<float>::infinity(),nan=std::numeric_limits<float>::quiet_NaN();
  float denorm=std::bit_cast<float>(uint32_t{1});
  for (float value: {inf,-inf,nan,0.0f,-0.0f,denorm,-denorm}) {
    result.push_back({{5,0,1,1},{},{},{value},"K0-nonempty-consumer"});
    result.push_back({{5,1,1,1},std::vector<float>(5,value),{1},{1},"IEEE-first"});
    result.push_back({{5,1,1,1},std::vector<float>(5,1),{1},{value},"IEEE-second"});
  }
  result.push_back({{1,2,1,1},{-1,1+0x1p-23f},{1,1-0x1p-23f},{1},"producer-FMA"});
  result.push_back({{1,1,2,1},{1},{-1,1+0x1p-23f},{1,1-0x1p-23f},"consumer-FMA"});
  result.push_back({{1,3,1,1},{0x1p24f,1,-0x1p24f},{1,1,1},{1},"producer-order"});
  result.push_back({{1,1,3,1},{1},{0x1p24f,1,-0x1p24f},{1,1,1},"consumer-order"});
  result.push_back({{1,2,2,1},{1+0x1p-23f,1},{1-0x1p-23f,0,0,1},{1,-1},
                    "producer-rounded-C-boundary"});
  result.push_back({{1,1,1,1},{std::bit_cast<float>(uint32_t{0x00800000})},{0.5f},{1},
                    "producer-gradual-underflow"});
  result.push_back({{1,1,1,1},{std::bit_cast<float>(uint32_t{0x00800000})},{1},{0.5f},
                    "consumer-gradual-underflow"});
  result.push_back({{1,1,1,1},{denorm},{1},{2},"subnormal-input-not-DAZ"});
  auto same=randomValues(9);
  result.push_back({{3,3,3,3},same,same,same,"A-B-D-MAY-alias",true});
  return result;
}

static void hostFalsifiers() {
  bool overflow=false;
  try { guards({INT64_MAX,0,2,0}); }
  catch (const std::exception &e) { overflow=std::string(e.what())=="extent_overflow"; }
  require(overflow,"full C overflow was hidden by empty E/panel");
  // Empty giant E bypass is allowed only in this isolated arithmetic harness;
  // never send its overflowing serial IV or unbounded work to the device.
  guards({INT64_MAX,0,0,0});
  auto zeroInf=oracle({1,0,1,1},{},{},{std::numeric_limits<float>::infinity()});
  require(std::isnan(zeroInf[0]),"invalid generic K0 finalzero oracle");
  volatile float fused=std::fma(1+0x1p-23f,1-0x1p-23f,-1);
  auto separate=oracle({1,2,1,1},{-1,1+0x1p-23f},{1,1-0x1p-23f},{1});
  require(fused!=separate[0],"FMA negative control does not distinguish");
  auto increasing=oracle({1,3,1,1},{0x1p24f,1,-0x1p24f},{1,1,1},{1});
  volatile float reverseFirst=-0x1p24f+1;
  volatile float reverse=reverseFirst+0x1p24f;
  require(increasing[0]!=reverse,"order negative control does not distinguish");
  bool workRejected=false;
  try { workBound({65,32,64,32}); }
  catch (const std::exception &e) { workRejected=std::string(e.what())=="research pair-work bound"; }
  require(workRejected,"pair work over-cap was accepted");\n  for (Shape shape: {Shape{65536,1,1,1},Shape{1,65536,1,1},\n                      Shape{1,0,65536,1},Shape{1,0,1,65536}}) {\n    bool refused=false;\n    try { workBound(shape); }\n    catch (const std::exception &) { refused=true; }\n    require(refused,"dimension65536 was accepted");\n  }
  auto rounded=oracle({1,2,2,1},{1+0x1p-23f,1},{1-0x1p-23f,0,0,1},{1,-1});
  volatile double exactC=(1+0x1p-23)*(1-0x1p-23);
  volatile double unrounded=exactC-1;
  require(rounded[0]==0 && unrounded!=0,"f32 C boundary control does not distinguish");
  std::printf("host-falsifiers=full-C-overflow,K0-Inf-not-finalzero,FMA,order; "
              "f32-C-boundary,pair-work-overcap; no huge shape enters GPU\n");
}

int main(int argc,char **argv) {
  if (argc!=2 && argc!=3) { std::fprintf(stderr,"usage: execute IMAGE [--negative-workspace-under-memcheck]\n"); return 1; }
  if (argc==3) {
#if defined(RESEARCH_CUDA)
    auto *permission=std::getenv("MDSLC_RESEARCH_CUDA_MEMCHECK_NEGATIVE_V1");
    if (std::string(argv[2])!="--negative-workspace-under-memcheck" || !permission ||
        std::string(permission)!="1") return 1;
    negativeWorkspace=true;
#else
    std::fprintf(stderr,"malformed AMD launch is explicitly unsupported\n"); return 1;
#endif
  }
  std::ifstream file(argv[1],std::ios::binary);
  std::vector<char> image((std::istreambuf_iterator<char>(file)),{});
  if (image.empty()) { std::fprintf(stderr,"image missing/empty\n"); return 1; }
  // Keep GPU and oracle effects on a joined worker, while the caller has hostile
  // FP controls/flags. This is a bounded harness observation, not runtime proof.
  fenv_t initial; std::fegetenv(&initial); unsigned initialMxcsr=_mm_getcsr();
  std::fesetround(FE_DOWNWARD); std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_DIVBYZERO); _mm_setcsr(_mm_getcsr()|0x8040u);
  unsigned callerMxcsr=_mm_getcsr(); int callerFlags=std::fetestexcept(FE_ALL_EXCEPT);
  int callerRound=std::fegetround(); errno=EDOM;
  bool success=false;
  std::thread worker([&] {
    try {
      std::fesetround(FE_TONEAREST); _mm_setcsr(_mm_getcsr()&~0x8040u);
      hostFalsifiers(); initialize();
      Module module{}; Function function{}; load(module,function,image.data());
      if (negativeWorkspace) {
        std::printf("NEGATIVE CUDA memcheck-only: forged4x3scratch descriptor, "
                    "physical1float+2canaries; separate bounded context/process\n");
        Case bad{{5,1,3,2},std::vector<float>(5,1),std::vector<float>(3,1),
                 std::vector<float>(6,1),"intentional-undercapacity-scratch"};
        runCase(function,bad);
        throw std::runtime_error("negative workspace escaped all instrumentation/checks");
      } else for (const auto &test:cases()) runCase(function,test);
      shutdown(module); success=true;
    } catch (const std::exception &error) {
      std::fprintf(stderr,"FAIL research arithmetic/ABI: %s\n",error.what());
      // No exception cleanup frees possibly-live device objects.
    }
  });
  worker.join();
  bool callerPreserved=_mm_getcsr()==callerMxcsr && std::fegetround()==callerRound &&
      std::fetestexcept(FE_ALL_EXCEPT)==callerFlags && errno==EDOM;
  std::fesetenv(&initial); _mm_setcsr(initialMxcsr);
  if (!success || !callerPreserved) return 1;
  std::printf("PASS launched=%u host-bypassed=%u strict-output-checks=%llu "
      "canary-checks=%llu immutable-input-checks=%llu caller-FP-errno=preserved; "
      "NaN-classification otherwise bitwise; no source-authority/speed claim\n",
      launched,bypassed,static_cast<unsigned long long>(outputChecks),
      static_cast<unsigned long long>(canaryChecks),static_cast<unsigned long long>(inputChecks));
  return 0;
}
