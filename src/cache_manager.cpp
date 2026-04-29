#include "cache_manager.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <variant>

#include <dlfcn.h>
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MD5.h"
#include "matcore/gpu_runtime_symbols.h"
#include "matcore/kernel_ir.h"

namespace matcore {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore JIT runner: " + message);
}

std::string targetName(const RequestedTargetProfile &target_profile) {
  if (!target_profile.canonical.empty()) {
    return target_profile.canonical;
  }
  return CanonicalTargetString(target_profile);
}

std::string dtypeName(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return "float32";
    case TensorDType::kFloat16:
      return "float16";
    case TensorDType::kBFloat16:
      return "bfloat16";
    case TensorDType::kInt8:
      return "int8";
    case TensorDType::kInt32:
      return "int32";
    case TensorDType::kFloat8E4M3FN:
      return "float8_e4m3fn";
  }
  return "unknown";
}

std::string buildStableCacheHash(const std::string &key) {
  llvm::MD5 hasher;
  hasher.update(llvm::StringRef(kDiskCacheVersion.data(), kDiskCacheVersion.size()));
  hasher.update(key);
  llvm::MD5::MD5Result result;
  hasher.final(result);
  return result.digest().str().str();
}

fs::path currentExtensionPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&registerGpuRuntimeSymbols), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::error_code ec;
  fs::path path = fs::weakly_canonical(fs::path(info.dli_fname), ec);
  if (ec) {
    return fs::path(info.dli_fname);
  }
  return path;
}

fs::path cacheRootPath() {
  if (const char *override_dir = std::getenv("MATCORE_CACHE_DIR")) {
    if (*override_dir != '\0') {
      return fs::path(override_dir);
    }
  }
  const fs::path extension_path = currentExtensionPath();
  if (!extension_path.empty()) {
    return extension_path.parent_path().parent_path() / ".matcore_cache";
  }
  return fs::current_path() / ".matcore_cache";
}

void appendKernelStructure(std::string &key, const KernelIR &kernel) {
  if (kernel.version == KernelIRVersion::kGraphV2 && kernel.graph.has_value()) {
    const KernelGraphIR &g = *kernel.graph;
    key += "|gv2";
    key += "|nv=" + std::to_string(g.values.size());
    for (const TensorDesc &v : g.values) {
      key += ";v:";
      key += v.symbol;
      key += ":";
      key += std::to_string(static_cast<int>(v.dtype));
      key += ":";
      key += std::to_string(static_cast<int>(v.value_kind));
      key += ":";
      key += std::to_string(static_cast<int>(v.storage_hint));
      key += ":";
      key += std::to_string(static_cast<int>(v.escape));
      key += ":s=";
      for (auto d : v.shape) {
        key += std::to_string(d);
        key += ",";
      }
      key += ":t=";
      for (auto d : v.strides) {
        key += std::to_string(d);
        key += ",";
      }
      if (v.is_parameter) key += ":p";
      if (v.is_output) key += ":o";
      if (v.is_device_resident) key += ":dev";
    }
    key += "|iv=";
    for (auto id : g.input_values) {
      key += std::to_string(id);
      key += ",";
    }
    key += "|ov=";
    for (auto id : g.output_values) {
      key += std::to_string(id);
      key += ",";
    }
    key += "|topo=";
    for (auto id : g.topo_order) {
      key += std::to_string(id);
      key += ",";
    }
    key += "|nn=" + std::to_string(g.nodes.size());
    for (const KernelNode &n : g.nodes) {
      key += ";n:";
      key += std::to_string(n.id);
      key += ":";
      key += std::to_string(static_cast<int>(n.kind));
      key += ":in=";
      for (auto id : n.inputs) {
        key += std::to_string(id);
        key += ",";
      }
      key += ":out=";
      for (auto id : n.outputs) {
        key += std::to_string(id);
        key += ",";
      }
      key += ":sef=";
      key += (n.side_effect_free ? "1" : "0");
      key += ":attr=";
      key += std::to_string(n.attrs.index());
      key += ":";
      std::visit(
          [&key](const auto &a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, MatMulAttrs>) {
              key += "mm:";
              key += std::to_string(a.lhs.value_id) + "/" +
                     (a.lhs.transpose_last2 ? "1" : "0") + ",";
              key += std::to_string(a.rhs.value_id) + "/" +
                     (a.rhs.transpose_last2 ? "1" : "0") + ",";
              key += std::to_string(static_cast<int>(a.accumulate_dtype));
              key += ",";
              key += (a.allow_split_k ? "1" : "0");
            } else if constexpr (std::is_same_v<T, ElementwiseAttrs>) {
              key += "ew:";
              key += std::to_string(static_cast<int>(a.kind));
              key += ",in=";
              for (auto id : a.inputs) {
                key += std::to_string(id);
                key += "/";
              }
              key += ",im=";
              for (auto s : a.scalar_immediates) {
                key += std::to_string(s);
                key += "/";
              }
              key += (a.allow_inplace ? "1" : "0");
            } else if constexpr (std::is_same_v<T, TransposeAttrs>) {
              key += "tp:";
              key += std::to_string(a.input);
            } else if constexpr (std::is_same_v<T, CastAttrs>) {
              key += "cast:";
              key += std::to_string(a.input);
              key += ",";
              key += std::to_string(static_cast<int>(a.target_dtype));
            } else if constexpr (std::is_same_v<T, ReduceAttrs>) {
              key += "red:";
              key += std::to_string(static_cast<int>(a.kind));
              key += ",";
              key += std::to_string(a.input);
              key += ",";
              key += std::to_string(static_cast<int>(a.axis));
              key += ",";
              key += (a.keepdim ? "1" : "0");
            } else if constexpr (std::is_same_v<T, SoftmaxAttrs>) {
              key += "sm:";
              key += std::to_string(a.input);
              key += ",";
              key += std::to_string(a.axis);
              key += ",";
              key += (a.stable ? "1" : "0");
              key += ",";
              key += (a.causal ? "1" : "0");
            } else if constexpr (std::is_same_v<T, StoreAttrs>) {
              key += "st:";
              key += std::to_string(a.input);
              key += ",";
              key += std::to_string(a.output_tensor);
            }
          },
          n.attrs);
    }
  } else {
    key += "|ops=" + std::to_string(kernel.ops.size());
    for (const KernelOp &op : kernel.ops) {
      key += ";op";
      key += std::to_string(op.index());
    }
  }
}

}  // namespace

std::string buildExecutionCacheKey(
    const KernelIR &kernel, const RequestedTargetProfile &target_profile,
    const std::vector<RuntimeTensorView> &tensors,
    const std::optional<std::string_view> &x86_cache_tag,
    bool graph_mode) {
  std::string key = kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  key += "|target=" + targetName(target_profile);
  if (graph_mode) {
    key += "|graph_mode";
  }
  if (x86_cache_tag.has_value()) {
    key += "|";
    key += std::string(*x86_cache_tag);
  }
  appendKernelStructure(key, kernel);
  if (kernel.global_quantization.enabled) {
    key += "|gq=" + std::to_string(kernel.global_quantization.scale);
    key += ":";
    key += std::to_string(kernel.global_quantization.zero_point);
  }
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
    if (tensor.is_device_resident) {
      key += ":dev";
    }
    if (tensor.quantization.enabled) {
      key += ":q=";
      key += std::to_string(tensor.quantization.scale);
      key += ":";
      key += std::to_string(tensor.quantization.zero_point);
    }
  }
  return key;
}

DiskCacheArtifacts buildDiskCacheArtifacts(const std::string &cache_key) {
  DiskCacheArtifacts artifacts;
  artifacts.root_dir = cacheRootPath();
  artifacts.artifact_dir = artifacts.root_dir / buildStableCacheHash(cache_key);
  artifacts.shared_object_path = artifacts.artifact_dir / "kernel.so";
  artifacts.object_path = artifacts.artifact_dir / "kernel.o";
  return artifacts;
}

bool isDiskCacheSupported(const RequestedTargetProfile &target_profile) {
  switch (normalizeTarget(target_profile.kind)) {
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
    case TargetKind::kNvidiaDGPU:
      return true;
    default:
      return false;
  }
}

void ensureCacheDirectory(const fs::path &path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  if (ec) {
    fail("failed to create cache directory '" + path.string() + "': " +
         ec.message());
  }
}

}  // namespace matcore
