#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace matcore {

enum class TargetKind {
  kX86Auto,
  kX86AVX2,
  kX86AVX512,
  kNvidiaDGPU,
  kAmdIGPU,
  kAmdNPU,
  kARM,
  // Legacy aliases kept for compatibility with older bridge payloads.
  kNVPTX,
  kAMDGCN,
  kNPU,
  kTPU,
};

enum class TensorDType {
  kFloat32,
  kFloat16,
  kBFloat16,
};

struct LoopRange {
  std::string var;
  std::int64_t lower = 0;
  std::int64_t upper = 0;
  std::int64_t step = 1;
};

struct LoadOp {
  std::string output;
  std::string tensor;
  std::vector<std::string> indices;
};

struct MatMulOp {
  std::string output;
  std::string lhs;
  std::string rhs;
};

struct StoreOp {
  std::string tensor;
  std::string value;
  std::vector<std::string> indices;
};

struct AssignOp {
  std::string output;
  std::string value;
};

using KernelOp = std::variant<LoadOp, MatMulOp, StoreOp, AssignOp>;

struct KernelIR {
  std::string kernel_name;
  std::vector<std::string> params;
  std::vector<LoopRange> loops;
  std::vector<KernelOp> ops;
};

struct RuntimeTensorView {
  std::string symbol;
  void *data = nullptr;
  TensorDType dtype = TensorDType::kFloat32;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
  bool c_contiguous = false;
};

inline TargetKind normalizeTarget(TargetKind target) {
  switch (target) {
    case TargetKind::kNVPTX:
      return TargetKind::kNvidiaDGPU;
    case TargetKind::kAMDGCN:
      return TargetKind::kAmdIGPU;
    case TargetKind::kNPU:
      return TargetKind::kAmdNPU;
    default:
      return target;
  }
}

inline bool isCpuTarget(TargetKind target) {
  target = normalizeTarget(target);
  return target == TargetKind::kX86Auto || target == TargetKind::kX86AVX2 ||
         target == TargetKind::kX86AVX512 || target == TargetKind::kARM;
}

inline std::optional<std::size_t> findTensorIndex(const KernelIR &kernel,
                                                  const std::string &symbol) {
  for (std::size_t i = 0; i < kernel.params.size(); ++i) {
    if (kernel.params[i] == symbol) {
      return i;
    }
  }
  return std::nullopt;
}

}  // namespace matcore
