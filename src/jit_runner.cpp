#include "matcore/jit_runner.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "matcore/mlir_engine.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore JIT runner: " + message);
}

constexpr std::int64_t kRowBlock = 8;
constexpr std::int64_t kColBlock = 128;
constexpr std::int64_t kKBlock = 64;

std::string targetName(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kX86Auto:
      return "x86-auto";
    case TargetKind::kX86AVX2:
      return "x86-avx2";
    case TargetKind::kX86AVX512:
      return "x86-avx512";
    case TargetKind::kNvidiaDGPU:
      return "nvidia-dgpu";
    case TargetKind::kAmdIGPU:
      return "amd-igpu";
    case TargetKind::kAmdNPU:
      return "amd-npu";
    case TargetKind::kARM:
      return "arm";
    case TargetKind::kTPU:
      return "tpu";
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }
  return "unknown";
}

std::string dtypeName(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return "float32";
    case TensorDType::kFloat16:
      return "float16";
    case TensorDType::kBFloat16:
      return "bfloat16";
  }
  return "unknown";
}

float halfToFloat(std::uint16_t bits16) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits16 & 0x8000U) << 16;
  std::uint32_t exp = (bits16 & 0x7C00U) >> 10;
  std::uint32_t mantissa = bits16 & 0x03FFU;

  std::uint32_t bits32 = 0;
  if (exp == 0) {
    if (mantissa == 0) {
      bits32 = sign;
    } else {
      // Normalize subnormal half.
      exp = 1;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        --exp;
      }
      mantissa &= 0x03FFU;
      const std::uint32_t exp32 = exp + (127U - 15U);
      bits32 = sign | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exp == 0x1FU) {
    bits32 = sign | 0x7F800000U | (mantissa << 13);
  } else {
    const std::uint32_t exp32 = exp + (127U - 15U);
    bits32 = sign | (exp32 << 23) | (mantissa << 13);
  }

  return std::bit_cast<float>(bits32);
}

std::uint16_t floatToHalf(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
  const std::uint32_t mantissa = bits & 0x007FFFFFU;
  const int exp = static_cast<int>((bits >> 23) & 0xFFU) - 127 + 15;

  if (exp <= 0) {
    if (exp < -10) {
      return sign;
    }
    std::uint32_t sub = (mantissa | 0x00800000U) >> static_cast<unsigned>(1 - exp);
    return static_cast<std::uint16_t>(sign | ((sub + 0x00001000U) >> 13));
  }

  if (exp >= 31) {
    if (mantissa == 0) {
      return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    const std::uint16_t payload =
        static_cast<std::uint16_t>((mantissa >> 13) ? (mantissa >> 13) : 1U);
    return static_cast<std::uint16_t>(sign | 0x7C00U | payload);
  }

  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint16_t>(exp) << 10) |
      static_cast<std::uint16_t>((mantissa + 0x00001000U) >> 13));
}

float bfloat16ToFloat(std::uint16_t bits16) {
  const std::uint32_t bits32 = static_cast<std::uint32_t>(bits16) << 16;
  return std::bit_cast<float>(bits32);
}

std::uint16_t floatToBFloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t lsb = (bits >> 16) & 0x1U;
  bits += 0x7FFFU + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

float loadElement(const RuntimeTensorView &tensor, std::int64_t index) {
  switch (tensor.dtype) {
    case TensorDType::kFloat32:
      return reinterpret_cast<const float *>(tensor.data)[index];
    case TensorDType::kFloat16:
      return halfToFloat(reinterpret_cast<const std::uint16_t *>(tensor.data)[index]);
    case TensorDType::kBFloat16:
      return bfloat16ToFloat(
          reinterpret_cast<const std::uint16_t *>(tensor.data)[index]);
  }
  fail("unsupported tensor dtype in load");
}

void storeElement(RuntimeTensorView &tensor, std::int64_t index, float value) {
  switch (tensor.dtype) {
    case TensorDType::kFloat32:
      reinterpret_cast<float *>(tensor.data)[index] = value;
      return;
    case TensorDType::kFloat16:
      reinterpret_cast<std::uint16_t *>(tensor.data)[index] = floatToHalf(value);
      return;
    case TensorDType::kBFloat16:
      reinterpret_cast<std::uint16_t *>(tensor.data)[index] = floatToBFloat16(value);
      return;
  }
  fail("unsupported tensor dtype in store");
}

bool isDenseRowMajor(const RuntimeTensorView &tensor) {
  return tensor.shape.size() == 2 && tensor.strides.size() == 2 &&
         tensor.strides[1] == 1 && tensor.strides[0] == tensor.shape[1];
}

std::vector<float> materializeToFloat(const RuntimeTensorView &tensor) {
  if (tensor.shape.size() != 2 || tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2");
  }

  const std::int64_t rows = tensor.shape[0];
  const std::int64_t cols = tensor.shape[1];
  std::vector<float> dense(static_cast<std::size_t>(rows * cols));

  if (tensor.dtype == TensorDType::kFloat32 && isDenseRowMajor(tensor)) {
    const float *src = reinterpret_cast<const float *>(tensor.data);
    std::copy_n(src, dense.size(), dense.begin());
    return dense;
  }

  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::int64_t source_index = i * tensor.strides[0] + j * tensor.strides[1];
      dense[static_cast<std::size_t>(i * cols + j)] = loadElement(tensor, source_index);
    }
  }
  return dense;
}

void writeBackFromFloat(const std::vector<float> &dense, RuntimeTensorView tensor) {
  if (tensor.shape.size() != 2 || tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2");
  }

  const std::int64_t rows = tensor.shape[0];
  const std::int64_t cols = tensor.shape[1];
  if (dense.size() != static_cast<std::size_t>(rows * cols)) {
    fail("output buffer size mismatch");
  }

  if (tensor.dtype == TensorDType::kFloat32 && isDenseRowMajor(tensor)) {
    float *dst = reinterpret_cast<float *>(tensor.data);
    std::copy_n(dense.data(), dense.size(), dst);
    return;
  }

  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::int64_t target_index = i * tensor.strides[0] + j * tensor.strides[1];
      storeElement(tensor, target_index,
                   dense[static_cast<std::size_t>(i * cols + j)]);
    }
  }
}

#if defined(__x86_64__) || defined(__i386__)

bool cpuSupportsAvx2() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

bool cpuSupportsAvx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
}

#else

bool cpuSupportsAvx2() { return false; }

bool cpuSupportsAvx512() { return false; }

#endif

enum class X86KernelKind {
  kScalar,
  kAvx2,
  kAvx512,
};

X86KernelKind selectX86Kernel(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kX86AVX512:
      if (!cpuSupportsAvx512()) {
        fail("target 'x86-avx512' requires AVX-512F and FMA host support");
      }
      return X86KernelKind::kAvx512;
    case TargetKind::kX86AVX2:
      if (!cpuSupportsAvx2()) {
        fail("target 'x86-avx2' requires AVX2 and FMA host support");
      }
      return X86KernelKind::kAvx2;
    case TargetKind::kX86Auto:
      if (cpuSupportsAvx512()) {
        return X86KernelKind::kAvx512;
      }
      if (cpuSupportsAvx2()) {
        return X86KernelKind::kAvx2;
      }
      return X86KernelKind::kScalar;
    case TargetKind::kARM:
      fail("ARM CPU execution is not implemented in Phase 2");
    default:
      fail("unsupported CPU execution target '" + targetName(target) + "'");
  }
}

using AccumulateColumnsFn = void (*)(float *, const float *, float, std::int64_t);

void accumulateColumnsScalar(float *out, const float *rhs, float lhs_value,
                             std::int64_t cols) {
  for (std::int64_t j = 0; j < cols; ++j) {
    out[j] += lhs_value * rhs[j];
  }
}

#if defined(__x86_64__) || defined(__i386__)

[[gnu::target("avx2,fma")]] void accumulateColumnsAvx2(float *out, const float *rhs,
                                                       float lhs_value,
                                                       std::int64_t cols) {
  const __m256 lhs_vec = _mm256_set1_ps(lhs_value);
  std::int64_t j = 0;
  for (; j + 8 <= cols; j += 8) {
    const __m256 rhs_vec = _mm256_loadu_ps(rhs + j);
    const __m256 out_vec = _mm256_loadu_ps(out + j);
    _mm256_storeu_ps(out + j, _mm256_fmadd_ps(lhs_vec, rhs_vec, out_vec));
  }
  for (; j < cols; ++j) {
    out[j] += lhs_value * rhs[j];
  }
}

[[gnu::target("avx512f,fma")]] void accumulateColumnsAvx512(float *out,
                                                            const float *rhs,
                                                            float lhs_value,
                                                            std::int64_t cols) {
  const __m512 lhs_vec = _mm512_set1_ps(lhs_value);
  std::int64_t j = 0;
  for (; j + 16 <= cols; j += 16) {
    const __m512 rhs_vec = _mm512_loadu_ps(rhs + j);
    const __m512 out_vec = _mm512_loadu_ps(out + j);
    _mm512_storeu_ps(out + j, _mm512_fmadd_ps(lhs_vec, rhs_vec, out_vec));
  }
  for (; j < cols; ++j) {
    out[j] += lhs_value * rhs[j];
  }
}

#endif

AccumulateColumnsFn selectAccumulateColumns(TargetKind target) {
  switch (selectX86Kernel(target)) {
    case X86KernelKind::kScalar:
      return &accumulateColumnsScalar;
    case X86KernelKind::kAvx2:
#if defined(__x86_64__) || defined(__i386__)
      return &accumulateColumnsAvx2;
#else
      fail("AVX2 execution requires an x86 host");
#endif
    case X86KernelKind::kAvx512:
#if defined(__x86_64__) || defined(__i386__)
      return &accumulateColumnsAvx512;
#else
      fail("AVX-512 execution requires an x86 host");
#endif
  }
  return &accumulateColumnsScalar;
}

std::size_t selectThreadCount(std::int64_t rows) {
  const unsigned hardware_threads =
      std::max(1U, std::thread::hardware_concurrency());
  return std::max<std::size_t>(
      1, std::min<std::size_t>(hardware_threads,
                               static_cast<std::size_t>(std::max<std::int64_t>(
                                   1, (rows + kRowBlock - 1) / kRowBlock))));
}

void executeBlockedCpuMatmul(const std::vector<float> &lhs,
                             const std::vector<float> &rhs,
                             std::vector<float> &out, std::int64_t m,
                             std::int64_t k, std::int64_t n, TargetKind target) {
  std::fill(out.begin(), out.end(), 0.0f);
  const AccumulateColumnsFn accumulate_columns = selectAccumulateColumns(target);
  const std::size_t thread_count = selectThreadCount(m);
  const std::int64_t rows_per_thread =
      std::max<std::int64_t>(kRowBlock, (m + static_cast<std::int64_t>(thread_count) - 1) /
                                            static_cast<std::int64_t>(thread_count));

  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    const std::int64_t row_begin =
        static_cast<std::int64_t>(thread_index) * rows_per_thread;
    const std::int64_t row_end = std::min(m, row_begin + rows_per_thread);
    if (row_begin >= row_end) {
      break;
    }

    workers.emplace_back([&, row_begin, row_end]() {
      for (std::int64_t i0 = row_begin; i0 < row_end; i0 += kRowBlock) {
        const std::int64_t i_max = std::min(i0 + kRowBlock, row_end);
        for (std::int64_t j0 = 0; j0 < n; j0 += kColBlock) {
          const std::int64_t cols = std::min(kColBlock, n - j0);
          for (std::int64_t k0 = 0; k0 < k; k0 += kKBlock) {
            const std::int64_t k_max = std::min(k0 + kKBlock, k);
            for (std::int64_t i = i0; i < i_max; ++i) {
              float *out_block = out.data() + i * n + j0;
              const float *lhs_row = lhs.data() + i * k;
              for (std::int64_t kk = k0; kk < k_max; ++kk) {
                const float lhs_value = lhs_row[kk];
                const float *rhs_block = rhs.data() + kk * n + j0;
                accumulate_columns(out_block, rhs_block, lhs_value, cols);
              }
            }
          }
        }
      }
    });
  }

  for (std::thread &worker : workers) {
    worker.join();
  }
}

std::string buildExecutionCacheKey(const KernelIR &kernel, TargetKind target,
                                   const std::vector<RuntimeTensorView> &tensors) {
  std::string key = kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  key += "|target=" + targetName(target);
  key += "|ops=" + std::to_string(kernel.ops.size());
  for (std::size_t i = 0; i < std::min<std::size_t>(3, tensors.size()); ++i) {
    const RuntimeTensorView &tensor = tensors[i];
    key += "|";
    key += tensor.symbol;
    key += ":";
    key += dtypeName(tensor.dtype);
    if (tensor.shape.size() >= 2) {
      key += ":";
      key += std::to_string(tensor.shape[0]);
      key += "x";
      key += std::to_string(tensor.shape[1]);
    }
  }
  return key;
}

void runCpuMatmul(TargetKind target, const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 3) {
    fail("matmul execution requires lhs, rhs, and output tensors");
  }

  const RuntimeTensorView &lhs = tensors[0];
  const RuntimeTensorView &rhs = tensors[1];
  RuntimeTensorView out = tensors[2];
  if (lhs.dtype != rhs.dtype || out.dtype != lhs.dtype) {
    fail("mixed dtype matmul is not supported in Phase 2");
  }

  const std::int64_t m = lhs.shape[0];
  const std::int64_t k = lhs.shape[1];
  const std::int64_t n = rhs.shape[1];

  std::vector<float> lhs_dense = materializeToFloat(lhs);
  std::vector<float> rhs_dense = materializeToFloat(rhs);
  std::vector<float> out_dense(static_cast<std::size_t>(m * n));
  executeBlockedCpuMatmul(lhs_dense, rhs_dense, out_dense, m, k, n, target);
  writeBackFromFloat(out_dense, out);
}

std::unique_ptr<mlir::ExecutionEngine> takeEngine(
    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> engine) {
  if (!engine) {
    std::string error;
    llvm::handleAllErrors(engine.takeError(), [&](const llvm::ErrorInfoBase &base) {
      error = base.message();
    });
    fail("ExecutionEngine::create() failed: " + error);
  }
  return std::move(*engine);
}

void enforceExecutionPolicy(const LoweredModule &lowered) {
  if (lowered.executable) {
    return;
  }
  fail("target '" + targetName(lowered.target) +
       "' routed via '" + lowered.route_description +
       "' but execution is scaffold-only in Phase 2");
}

struct CachedExecution {
  std::unique_ptr<mlir::MLIRContext> context;
  LoweredModule lowered;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

void ensureExecutionEngineReady(const KernelIR &kernel, TargetKind target,
                                const std::vector<RuntimeTensorView> &tensors) {
  static std::mutex cache_mutex;
  static auto *cache =
      new std::unordered_map<std::string, std::shared_ptr<CachedExecution>>();

  const std::string cache_key = buildExecutionCacheKey(kernel, target, tensors);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cache->contains(cache_key)) {
      return;
    }
  }

  auto compiled = std::make_shared<CachedExecution>();
  compiled->context = std::make_unique<mlir::MLIRContext>();
  mlir::registerBuiltinDialectTranslation(*compiled->context);
  mlir::registerLLVMDialectTranslation(*compiled->context);

  compiled->lowered =
      MlirEngine::BuildAndLower(kernel, target, tensors, *compiled->context);
  enforceExecutionPolicy(compiled->lowered);

  mlir::ExecutionEngineOptions options;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  compiled->engine =
      takeEngine(mlir::ExecutionEngine::create(*compiled->lowered.module, options));

  std::lock_guard<std::mutex> lock(cache_mutex);
  cache->emplace(cache_key, std::move(compiled));
}

}  // namespace

void compileAndRun(const KernelIR &kernel, TargetKind target,
                   const std::vector<RuntimeTensorView> &tensors) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  ensureExecutionEngineReady(kernel, target, tensors);
  runCpuMatmul(target, tensors);
}

}  // namespace matcore
