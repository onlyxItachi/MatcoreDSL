#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "matcore/kernel_ir.h"

namespace matcore {

struct CachedExecution;
class ObservabilityContext;

/// Frozen tensor metadata — immutable after plan creation.
/// Used to validate that execute_plan() tensors match the plan.
struct FrozenTensorMeta {
  std::string symbol;
  TensorDType dtype = TensorDType::kFloat32;
  int64_t rank = 2;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  bool is_device_resident = false;
  bool is_output = false;
};

/// Pre-compiled execution plan. Python calls create_plan() once (expensive),
/// then execute_plan() repeatedly (near-zero overhead).
///
/// Owns the full compiled bundle: MLIRContext + LoweredModule + ExecutionEngine.
/// This prevents dangling MLIR state that would occur if only the engine were shared.
///
/// Plans are shape-locked: tensors passed to execute_plan() must match the
/// frozen metadata exactly (dtype, rank, shape, strides, residency).
class MatcorePlan {
public:
  ~MatcorePlan();

  // Not copyable. Custom move nulls source graph resources.
  MatcorePlan(const MatcorePlan &) = delete;
  MatcorePlan &operator=(const MatcorePlan &) = delete;
  MatcorePlan(MatcorePlan &&) noexcept;
  MatcorePlan &operator=(MatcorePlan &&) noexcept;

  /// Create a plan from kernel IR + tensor metadata + target.
  /// Performs full JIT compilation. Expensive — call once.
  /// Set graph_mode=true to enable CUDA graph capture on first execute.
  static std::unique_ptr<MatcorePlan>
  create(const KernelIR &kernel,
         const std::vector<RuntimeTensorView> &template_tensors,
         const std::string &target_str, ObservabilityContext *obs,
         bool graph_mode = false);

  /// Execute the plan with the given tensors. Near-zero overhead.
  /// Validates tensor metadata matches frozen plan. Zeros device-resident
  /// outputs on the execution stream. Invokes cached engine.
  void execute(const std::vector<RuntimeTensorView> &tensors);

  /// Check if tensors are compatible with this plan.
  bool validateTensors(const std::vector<RuntimeTensorView> &tensors,
                       std::string *error_msg) const;

  /// Plan generation ID for stale-handle detection.
  uint64_t generationId() const { return generation_id_; }

  /// Whether this plan has device-resident tensors.
  bool hasDeviceTensors() const { return has_device_tensors_; }

  /// Number of tensors expected.
  size_t numTensors() const { return frozen_meta_.size(); }

  /// Frozen metadata for each tensor.
  const std::vector<FrozenTensorMeta> &frozenMeta() const {
    return frozen_meta_;
  }

private:
  MatcorePlan() = default;

  static uint64_t nextGenerationId();

  // Zero device-resident output tensors (null stream)
  void zeroOutputs(const std::vector<RuntimeTensorView> &tensors);
  // Zero device-resident output tensors on a specific stream (for graph capture)
  void zeroOutputsOnStream(const std::vector<RuntimeTensorView> &tensors,
                           void *stream);

  uint64_t generation_id_ = 0;
  std::shared_ptr<CachedExecution> execution_;
  std::vector<FrozenTensorMeta> frozen_meta_;
  bool has_device_tensors_ = false;
  std::string cache_key_;

  // V2 Pillar 2: CUDA Graph support
  bool graph_mode_ = false;        // capture/replay enabled
  void *graph_stream_ = nullptr;   // dedicated capture stream (CUstream)
  void *graph_exec_ = nullptr;     // instantiated graph (CUgraphExec)
  bool graph_captured_ = false;    // true after first successful capture
  // Captured tensor data pointers — replay rejects pointer changes
  std::vector<void *> captured_ptrs_;
};

}  // namespace matcore
