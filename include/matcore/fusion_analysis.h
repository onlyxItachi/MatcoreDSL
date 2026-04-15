#pragma once

#include "matcore/kernel_ir.h"
#include "matcore/mlir_engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace matcore {

enum class FusionPatternKind {
  kNone,
  kMatmulElementwise,
  kElementwiseMatmul,
  kMatmulElementwiseMatmul,
  kMatmulSoftmaxMatmul,
  kGenericTileChain,
};

struct TileShape {
  int br = 0;
  int bc = 0;
  int d = 0;
  int k_step = 0;
  int stages = 1;
  int num_warps = 1;
  int threads_per_block = 32;
};

struct SharedBufferDesc {
  std::string name;
  TensorDType dtype = TensorDType::kFloat16;
  std::vector<int64_t> shape;
  std::size_t bytes = 0;
  bool double_buffered = false;
  bool padded = false;
};

struct RegisterEstimate {
  int base_regs = 0;
  int accum_regs = 0;
  int fragment_regs = 0;
  int elementwise_regs = 0;
  int reduction_regs = 0;
  int indexing_regs = 0;
  int total_regs = 0;
  bool may_spill = false;
};

struct OccupancyEstimate {
  int blocks_per_sm = 0;
  int warps_per_sm = 0;
  float occupancy = 0.0f;
  int limited_by_regs = 0;
  int limited_by_smem = 0;
};

struct FusedKernelPlan {
  FusionPatternKind pattern = FusionPatternKind::kNone;
  std::vector<std::uint32_t> node_ids;
  std::vector<std::uint32_t> input_value_ids;
  std::vector<std::uint32_t> output_value_ids;
  TileShape tile;
  std::vector<SharedBufferDesc> shared_buffers;
  std::size_t shared_bytes = 0;
  RegisterEstimate regs;
  OccupancyEstimate occupancy;
  bool use_online_softmax = false;
  bool use_split_k = false;
  bool uses_tensor_cores = true;
  std::string rejection_reason;
  std::string debug_summary;

  bool is_accepted() const { return rejection_reason.empty(); }
};

struct FusionAnalysisResult {
  std::vector<FusedKernelPlan> accepted_plans;
  std::vector<FusedKernelPlan> rejected_plans;
  std::string debug_log;
};

class FusionAnalyzer {
 public:
  static constexpr int kSmemBytesPerSM = 102400;
  static constexpr int kSmemSoftBudget = 40960;
  static constexpr int kSmemHardBudget = 49152;
  static constexpr int kRegsPerSM = 65536;
  static constexpr int kMaxThreadsPerSM = 1536;
  static constexpr int kMaxWarpsPerSM = 48;
  static constexpr int kMaxBlocksPerSM = 16;
  static constexpr int kRegSoftCap = 160;
  static constexpr int kRegHardCap = 192;
  static constexpr int kNumSMs = 24;

  static FusionAnalysisResult Analyze(const KernelGraphIR &graph,
                                      const RequestedTargetProfile &target);

 private:
  static void annotateEscapes(KernelGraphIR &graph);
  static std::vector<std::vector<uint32_t>> discoverCandidates(
      const KernelGraphIR &graph);
  static FusionPatternKind classifyRegion(const std::vector<uint32_t> &region,
                                          const KernelGraphIR &graph);
  static std::vector<TileShape> enumerateTileCandidates(
      FusionPatternKind pattern, const KernelGraphIR &graph,
      const std::vector<uint32_t> &region);
  static std::size_t estimateSharedMemory(
      FusionPatternKind pattern, const TileShape &tile,
      const KernelGraphIR &graph, const std::vector<uint32_t> &region);
  static RegisterEstimate estimateRegisters(
      FusionPatternKind pattern, const TileShape &tile,
      const KernelGraphIR &graph, const std::vector<uint32_t> &region);
  static OccupancyEstimate estimateOccupancy(const TileShape &tile,
                                             std::size_t shared_bytes,
                                             int total_regs);
  static double scorePlan(const FusedKernelPlan &plan,
                          const KernelGraphIR &graph);
};

}  // namespace matcore
