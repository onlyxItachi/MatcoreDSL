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
  kInt8,
  kInt32,
  kFloat8E4M3FN,
};

struct QuantizationParams {
  bool enabled = false;
  float scale = 1.0f;
  std::int32_t zero_point = 0;
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

// Transpose operation: swap last two dimensions
struct TransposeOp {
  std::string result;
  std::string input;
};

enum class ElementwiseKind {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kReLU,
  kGELU,
  kSigmoid,
  kNeg,
  kAbs,
  kSqrt,
  kExp,
  kLog,
  kTanh,
  kSoftmax,
  kMin,
  kMax,
  kSin,
  kCos,
  kRsqrt,
};

enum class KernelIRVersion {
  kLinearV1,  // Current linear IR
  kGraphV2,   // New graph IR for fusion
};

enum class OpKind {
  kMatMul,
  kElementwise,
  kTranspose,
  kCast,
  kReduce,
  kSoftmax,
  kStore,
};

enum class ReductionKind {
  kSum,
  kMax,
  kMin,
};

enum class ReductionAxisKind {
  kAxis0,
  kAxis1,
  kLast,
};

enum class ValueKind {
  kInput,
  kOutput,
  kIntermediate,
};

enum class StorageHint {
  kAuto,       // Compiler decides
  kRegister,   // Prefer registers
  kSharedMem,  // Prefer shared memory
  kVRAM,       // Force VRAM
};

enum class EscapeKind {
  kNoEscape,      // Intermediate, can stay in registers/smem
  kEscapeToVRAM,  // Must be written to VRAM (output or multi-use)
};

struct TensorDesc {
  std::string symbol;
  TensorDType dtype = TensorDType::kFloat32;
  std::vector<std::int64_t> shape;     // -1 means dynamic
  std::vector<std::int64_t> strides;   // optional; empty for logical values
  ValueKind value_kind = ValueKind::kIntermediate;
  StorageHint storage_hint = StorageHint::kAuto;
  EscapeKind escape = EscapeKind::kNoEscape;
  bool is_parameter = false;
  bool is_output = false;
  bool is_device_resident = false;
  std::optional<std::uint32_t> producer;
  std::vector<std::uint32_t> consumers;
};

struct OperandRef {
  std::uint32_t value_id = 0;
  bool transpose_last2 = false;
};

struct MatMulAttrs {
  OperandRef lhs;
  OperandRef rhs;
  TensorDType accumulate_dtype = TensorDType::kFloat32;
  bool allow_split_k = true;
};

struct ElementwiseAttrs {
  ElementwiseKind kind = ElementwiseKind::kAdd;
  std::vector<std::uint32_t> inputs;
  std::vector<double> scalar_immediates;
  bool allow_inplace = true;
};

struct TransposeAttrs {
  std::uint32_t input = 0;
};

struct CastAttrs {
  std::uint32_t input = 0;
  TensorDType target_dtype = TensorDType::kFloat32;
};

struct ReduceAttrs {
  ReductionKind kind = ReductionKind::kSum;
  std::uint32_t input = 0;
  ReductionAxisKind axis = ReductionAxisKind::kLast;
  bool keepdim = false;
};

struct SoftmaxAttrs {
  std::uint32_t input = 0;
  int axis = -1;
  bool stable = true;
  bool causal = false;
};

struct StoreAttrs {
  std::uint32_t input = 0;
  std::uint32_t output_tensor = 0;
};

using NodeAttrs = std::variant<MatMulAttrs, ElementwiseAttrs, TransposeAttrs,
                               CastAttrs, ReduceAttrs, SoftmaxAttrs, StoreAttrs>;

struct KernelNode {
  std::uint32_t id = 0;
  OpKind kind = OpKind::kMatMul;
  std::string debug_name;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  NodeAttrs attrs;
  bool side_effect_free = true;
};

struct KernelGraphIR {
  std::vector<TensorDesc> values;
  std::vector<KernelNode> nodes;
  std::vector<std::uint32_t> input_values;
  std::vector<std::uint32_t> output_values;
  std::vector<std::uint32_t> topo_order;
};

// Elementwise operation: apply unary/binary op to tiles
struct ElementwiseOp {
  std::string result;
  ElementwiseKind kind = ElementwiseKind::kAdd;
  std::string lhs;
  std::string rhs;
};

// Cast operation: explicit dtype conversion
struct CastOp {
  std::string result;
  std::string input;
  TensorDType target_dtype = TensorDType::kFloat32;
};

using KernelOp = std::variant<LoadOp, MatMulOp, StoreOp, AssignOp, TransposeOp,
                              ElementwiseOp, CastOp>;

struct KernelIR {
  KernelIRVersion version = KernelIRVersion::kLinearV1;
  std::string kernel_name;

  // V1 fields (kept for compatibility)
  std::vector<std::string> params;
  std::vector<LoopRange> loops;
  std::vector<KernelOp> ops;

  // V2 graph form (optional — used when version == kGraphV2)
  std::optional<KernelGraphIR> graph;

  QuantizationParams global_quantization;
};

struct RuntimeTensorView {
  std::string symbol;
  void *data = nullptr;
  TensorDType dtype = TensorDType::kFloat32;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
  bool c_contiguous = false;
  bool is_device_resident = false;  // True when backed by DeviceTensor (GPU ptr)
  QuantizationParams quantization;
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
