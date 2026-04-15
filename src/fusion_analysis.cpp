#include "matcore/fusion_analysis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace matcore {
namespace {

const KernelNode &nodeById(const KernelGraphIR &graph, std::uint32_t id) {
  return graph.nodes.at(id);
}

KernelNode &nodeById(KernelGraphIR &graph, std::uint32_t id) {
  return graph.nodes.at(id);
}

const TensorDesc &valueById(const KernelGraphIR &graph, std::uint32_t id) {
  return graph.values.at(id);
}

TensorDesc &valueById(KernelGraphIR &graph, std::uint32_t id) {
  return graph.values.at(id);
}

bool isPointwiseLike(OpKind kind) {
  return kind == OpKind::kElementwise || kind == OpKind::kCast ||
         kind == OpKind::kTranspose;
}

std::size_t dtypeSize(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return 2;
    case TensorDType::kInt8:
    case TensorDType::kFloat8E4M3FN:
      return 1;
    case TensorDType::kFloat32:
    case TensorDType::kInt32:
      return 4;
  }
  return 4;
}

std::string patternKindToString(FusionPatternKind pattern) {
  switch (pattern) {
    case FusionPatternKind::kNone:
      return "none";
    case FusionPatternKind::kMatmulElementwise:
      return "matmul_elementwise";
    case FusionPatternKind::kElementwiseMatmul:
      return "elementwise_matmul";
    case FusionPatternKind::kMatmulElementwiseMatmul:
      return "matmul_elementwise_matmul";
    case FusionPatternKind::kMatmulSoftmaxMatmul:
      return "matmul_softmax_matmul";
    case FusionPatternKind::kGenericTileChain:
      return "generic_tile_chain";
  }
  return "unknown";
}

std::vector<std::uint32_t> topoOrderOrDefault(const KernelGraphIR &graph) {
  if (!graph.topo_order.empty()) {
    return graph.topo_order;
  }
  std::vector<std::uint32_t> order(graph.nodes.size());
  std::iota(order.begin(), order.end(), 0U);
  return order;
}

TensorDType regionDType(const KernelGraphIR &graph,
                        const std::vector<std::uint32_t> &region) {
  for (std::uint32_t node_id : region) {
    const auto &node = nodeById(graph, node_id);
    if (!node.outputs.empty()) {
      return valueById(graph, node.outputs.front()).dtype;
    }
  }
  return TensorDType::kFloat16;
}

TensorDType softmaxScoreDType(const KernelGraphIR &graph,
                              const std::vector<std::uint32_t> &region) {
  for (std::uint32_t node_id : region) {
    const auto &node = nodeById(graph, node_id);
    if (node.kind == OpKind::kSoftmax && !node.inputs.empty()) {
      return valueById(graph, node.inputs.front()).dtype;
    }
  }
  return regionDType(graph, region);
}

std::vector<SharedBufferDesc> buildSharedBuffers(
    FusionPatternKind pattern, const TileShape &tile, const KernelGraphIR &graph,
    const std::vector<std::uint32_t> &region) {
  const TensorDType dtype = regionDType(graph, region);
  const std::size_t elem_bytes = dtypeSize(dtype);
  std::vector<SharedBufferDesc> buffers;

  auto add_buffer = [&](std::string name, TensorDType buffer_dtype,
                        std::vector<int64_t> shape, bool double_buffered = false,
                        bool padded = false) {
    std::size_t elements = 1;
    for (int64_t dim : shape) {
      elements *= static_cast<std::size_t>(std::max<int64_t>(dim, 1));
    }
    SharedBufferDesc desc;
    desc.name = std::move(name);
    desc.dtype = buffer_dtype;
    desc.shape = std::move(shape);
    desc.bytes = elements * dtypeSize(buffer_dtype) *
                 (double_buffered ? static_cast<std::size_t>(tile.stages) : 1U);
    desc.double_buffered = double_buffered;
    desc.padded = padded;
    buffers.push_back(std::move(desc));
  };

  switch (pattern) {
    case FusionPatternKind::kMatmulElementwise:
    case FusionPatternKind::kElementwiseMatmul:
    case FusionPatternKind::kGenericTileChain:
      add_buffer("lhs_tile", dtype, {tile.br, tile.k_step}, tile.stages > 1);
      add_buffer("rhs_tile", dtype, {tile.k_step, tile.bc}, tile.stages > 1);
      break;
    case FusionPatternKind::kMatmulElementwiseMatmul:
      add_buffer("lhs_tile", dtype, {tile.br, tile.k_step}, tile.stages > 1);
      add_buffer("rhs_tile", dtype, {tile.k_step, tile.bc}, tile.stages > 1);
      add_buffer("intermediate_tile", dtype, {tile.br, tile.bc});
      break;
    case FusionPatternKind::kMatmulSoftmaxMatmul: {
      (void)softmaxScoreDType(graph, region);
      add_buffer("accum_tile", TensorDType::kFloat32, {tile.br, tile.d});
      break;
    }
    case FusionPatternKind::kNone:
      break;
  }

  return buffers;
}

int countElementwiseNodes(const KernelGraphIR &graph,
                          const std::vector<std::uint32_t> &region) {
  int count = 0;
  for (std::uint32_t node_id : region) {
    if (isPointwiseLike(nodeById(graph, node_id).kind)) {
      ++count;
    }
  }
  return count;
}

bool anyMatmulAllowsSplitK(const KernelGraphIR &graph,
                           const std::vector<std::uint32_t> &region) {
  for (std::uint32_t node_id : region) {
    const auto &node = nodeById(graph, node_id);
    if (node.kind != OpKind::kMatMul) {
      continue;
    }
    if (const auto *attrs = std::get_if<MatMulAttrs>(&node.attrs)) {
      if (attrs->allow_split_k) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

FusionAnalysisResult FusionAnalyzer::Analyze(
    const KernelGraphIR &graph, const RequestedTargetProfile &target) {
  FusionAnalysisResult result;
  KernelGraphIR annotated = graph;
  annotateEscapes(annotated);

  const bool tensor_core_target =
      normalizeTarget(target.kind) == TargetKind::kNvidiaDGPU;
  auto candidates = discoverCandidates(annotated);

  std::ostringstream debug;
  debug << "fusion_candidates=" << candidates.size() << '\n';

  for (const auto &region : candidates) {
    if (region.size() <= 1) {
      continue;
    }

    const auto pattern = classifyRegion(region, annotated);
    if (pattern == FusionPatternKind::kNone) {
      debug << "region skipped: no supported pattern\n";
      continue;
    }

    auto tile_candidates = enumerateTileCandidates(pattern, annotated, region);
    FusedKernelPlan best_plan;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto &tile : tile_candidates) {
      FusedKernelPlan plan;
      plan.pattern = pattern;
      plan.node_ids = region;
      plan.tile = tile;
      plan.use_online_softmax =
          pattern == FusionPatternKind::kMatmulSoftmaxMatmul;
      plan.use_split_k = anyMatmulAllowsSplitK(annotated, region) &&
                         tile.k_step <= 32 &&
                         pattern != FusionPatternKind::kMatmulSoftmaxMatmul;
      plan.uses_tensor_cores = tensor_core_target;

      std::unordered_set<std::uint32_t> region_nodes(region.begin(), region.end());
      std::unordered_set<std::uint32_t> seen_inputs;
      std::unordered_set<std::uint32_t> seen_outputs;

      for (std::uint32_t node_id : region) {
        const auto &node = nodeById(annotated, node_id);
        for (std::uint32_t input_id : node.inputs) {
          const auto &value = valueById(annotated, input_id);
          const bool boundary_input =
              !value.producer.has_value() ||
              region_nodes.count(*value.producer) == 0;
          if (boundary_input && seen_inputs.insert(input_id).second) {
            plan.input_value_ids.push_back(input_id);
          }
        }
        for (std::uint32_t output_id : node.outputs) {
          const auto &value = valueById(annotated, output_id);
          bool escapes_region = value.value_kind == ValueKind::kOutput;
          for (std::uint32_t consumer : value.consumers) {
            if (region_nodes.count(consumer) == 0) {
              escapes_region = true;
              break;
            }
          }
          if (escapes_region && seen_outputs.insert(output_id).second) {
            plan.output_value_ids.push_back(output_id);
          }
        }
      }

      plan.shared_bytes = estimateSharedMemory(pattern, tile, annotated, region);
      plan.shared_buffers = buildSharedBuffers(pattern, tile, annotated, region);
      plan.regs = estimateRegisters(pattern, tile, annotated, region);
      plan.occupancy =
          estimateOccupancy(tile, plan.shared_bytes, plan.regs.total_regs);

      if (plan.shared_bytes > kSmemHardBudget) {
        plan.rejection_reason = "exceeds smem hard budget";
        result.rejected_plans.push_back(plan);
        continue;
      }
      if (plan.regs.total_regs > kRegHardCap) {
        plan.rejection_reason = "exceeds register hard cap";
        result.rejected_plans.push_back(plan);
        continue;
      }
      if (plan.occupancy.occupancy < 0.125f) {
        plan.rejection_reason = "occupancy too low";
        result.rejected_plans.push_back(plan);
        continue;
      }

      const double score = scorePlan(plan, annotated);
      if (score > best_score) {
        best_score = score;
        best_plan = plan;
      }
    }

    if (best_score > -std::numeric_limits<double>::infinity()) {
      best_plan.debug_summary =
          "pattern=" + patternKindToString(best_plan.pattern) + " tile=" +
          std::to_string(best_plan.tile.br) + "x" +
          std::to_string(best_plan.tile.bc) + " smem=" +
          std::to_string(best_plan.shared_bytes) + " regs=" +
          std::to_string(best_plan.regs.total_regs) + " occ=" +
          std::to_string(best_plan.occupancy.occupancy);
      result.accepted_plans.push_back(best_plan);
      debug << "accepted " << best_plan.debug_summary << '\n';
    }
  }

  result.debug_log = debug.str();
  return result;
}

void FusionAnalyzer::annotateEscapes(KernelGraphIR &graph) {
  for (auto &value : graph.values) {
    const bool is_external = value.value_kind == ValueKind::kInput ||
                             value.value_kind == ValueKind::kOutput;
    const bool multi_use = value.consumers.size() > 1;
    const bool no_single_region =
        !value.producer.has_value() || value.consumers.empty();
    value.escape = (is_external || multi_use || no_single_region)
                       ? EscapeKind::kEscapeToVRAM
                       : EscapeKind::kNoEscape;
  }
}

std::vector<std::vector<uint32_t>> FusionAnalyzer::discoverCandidates(
    const KernelGraphIR &graph) {
  std::vector<std::vector<uint32_t>> candidates;
  std::unordered_set<std::string> seen;

  auto maybe_add = [&](std::vector<std::uint32_t> region) {
    if (region.size() <= 1) {
      return;
    }
    std::ostringstream key;
    for (std::uint32_t id : region) {
      key << id << ',';
    }
    if (seen.insert(key.str()).second) {
      candidates.push_back(std::move(region));
    }
  };

  for (std::uint32_t node_id : topoOrderOrDefault(graph)) {
    const auto &start = nodeById(graph, node_id);
    const bool start_is_matmul = start.kind == OpKind::kMatMul;
    const bool start_is_pointwise = isPointwiseLike(start.kind);
    if (!start_is_matmul && !start_is_pointwise) {
      continue;
    }

    std::vector<std::uint32_t> region;
    std::uint32_t current_id = node_id;
    int matmul_count = 0;

    while (true) {
      const auto &current = nodeById(graph, current_id);
      if (current.kind == OpKind::kMatMul) {
        ++matmul_count;
      } else if (!isPointwiseLike(current.kind) &&
                 current.kind != OpKind::kSoftmax) {
        break;
      }

      region.push_back(current_id);

      if (start_is_pointwise && matmul_count >= 1) {
        break;
      }
      if (matmul_count >= 2) {
        break;
      }
      if (current.outputs.size() != 1) {
        break;
      }

      const auto &out = valueById(graph, current.outputs.front());
      if (out.escape != EscapeKind::kNoEscape || out.consumers.size() != 1) {
        break;
      }

      const std::uint32_t next_id = out.consumers.front();
      const auto &next = nodeById(graph, next_id);
      if (start_is_matmul) {
        if (!isPointwiseLike(next.kind) && next.kind != OpKind::kSoftmax &&
            next.kind != OpKind::kMatMul) {
          break;
        }
      } else if (!isPointwiseLike(next.kind) && next.kind != OpKind::kMatMul) {
        break;
      }

      current_id = next_id;
    }

    maybe_add(std::move(region));
  }

  return candidates;
}

FusionPatternKind FusionAnalyzer::classifyRegion(
    const std::vector<uint32_t> &region, const KernelGraphIR &graph) {
  if (region.empty()) {
    return FusionPatternKind::kNone;
  }

  int matmul_count = 0;
  int softmax_count = 0;
  int pointwise_count = 0;
  for (std::uint32_t node_id : region) {
    const auto &node = nodeById(graph, node_id);
    if (node.kind == OpKind::kMatMul) {
      ++matmul_count;
    } else if (node.kind == OpKind::kSoftmax) {
      ++softmax_count;
    } else if (isPointwiseLike(node.kind)) {
      ++pointwise_count;
    }
  }

  const OpKind first_kind = nodeById(graph, region.front()).kind;
  const OpKind last_kind = nodeById(graph, region.back()).kind;

  if (matmul_count == 1 && softmax_count == 0) {
    if (first_kind == OpKind::kMatMul &&
        pointwise_count == static_cast<int>(region.size()) - 1) {
      return FusionPatternKind::kMatmulElementwise;
    }
    if (last_kind == OpKind::kMatMul &&
        pointwise_count == static_cast<int>(region.size()) - 1) {
      return FusionPatternKind::kElementwiseMatmul;
    }
    return FusionPatternKind::kGenericTileChain;
  }

  if (matmul_count == 2 && softmax_count == 0) {
    return FusionPatternKind::kMatmulElementwiseMatmul;
  }
  if (matmul_count == 2 && softmax_count > 0) {
    return FusionPatternKind::kMatmulSoftmaxMatmul;
  }
  if (matmul_count > 0) {
    return FusionPatternKind::kGenericTileChain;
  }
  return FusionPatternKind::kNone;
}

std::vector<TileShape> FusionAnalyzer::enumerateTileCandidates(
    FusionPatternKind pattern, const KernelGraphIR &graph,
    const std::vector<uint32_t> &region) {
  (void)graph;
  (void)region;
  switch (pattern) {
    case FusionPatternKind::kMatmulElementwise:
    case FusionPatternKind::kElementwiseMatmul:
      return {{128, 128, 0, 64, 2, 4, 128},
              {64, 64, 0, 32, 2, 4, 128},
              {64, 64, 0, 64, 1, 2, 64}};
    case FusionPatternKind::kMatmulElementwiseMatmul:
      return {{64, 64, 64, 32, 2, 4, 128},
              {128, 64, 64, 32, 2, 8, 256}};
    case FusionPatternKind::kMatmulSoftmaxMatmul:
      return {{32, 32, 16, 16, 1, 2, 64},
              {32, 16, 16, 16, 1, 2, 64},
              {16, 16, 8, 8, 1, 1, 32}};
    case FusionPatternKind::kGenericTileChain:
      return {{64, 64, 0, 32, 1, 2, 64}, {64, 128, 0, 32, 1, 4, 128}};
    case FusionPatternKind::kNone:
      return {};
  }
  return {};
}

std::size_t FusionAnalyzer::estimateSharedMemory(
    FusionPatternKind pattern, const TileShape &tile, const KernelGraphIR &graph,
    const std::vector<uint32_t> &region) {
  const std::size_t dtype_bytes = dtypeSize(regionDType(graph, region));

  switch (pattern) {
    case FusionPatternKind::kMatmulElementwise:
    case FusionPatternKind::kElementwiseMatmul:
    case FusionPatternKind::kGenericTileChain:
      return static_cast<std::size_t>(tile.stages) *
             static_cast<std::size_t>(tile.br * tile.k_step +
                                      tile.k_step * tile.bc) *
             dtype_bytes;
    case FusionPatternKind::kMatmulElementwiseMatmul:
      return static_cast<std::size_t>(tile.stages) *
                 static_cast<std::size_t>(tile.br * tile.k_step +
                                          tile.k_step * tile.bc) *
                 dtype_bytes +
             static_cast<std::size_t>(tile.br * tile.bc) * dtype_bytes;
    case FusionPatternKind::kMatmulSoftmaxMatmul: {
      (void)softmaxScoreDType(graph, region);
      return static_cast<std::size_t>(tile.br * tile.d) * sizeof(float);
    }
    case FusionPatternKind::kNone:
      return 0;
  }
  return 0;
}

RegisterEstimate FusionAnalyzer::estimateRegisters(
    FusionPatternKind pattern, const TileShape &tile, const KernelGraphIR &graph,
    const std::vector<uint32_t> &region) {
  RegisterEstimate regs;
  regs.base_regs = 32;
  regs.indexing_regs = 8;
  regs.accum_regs = std::max(1, (tile.br * tile.bc) /
                                    std::max(1, 32 * tile.num_warps) * 2);
  regs.fragment_regs =
      std::clamp(16 + ((tile.k_step >= 64) ? 12 : 8) + (tile.d > 0 ? 4 : 0), 16,
                 32);
  regs.elementwise_regs = 4 * countElementwiseNodes(graph, region);
  regs.reduction_regs =
      (pattern == FusionPatternKind::kMatmulSoftmaxMatmul) ? 16 : 0;

  const int subtotal = regs.base_regs + regs.accum_regs + regs.fragment_regs +
                       regs.elementwise_regs + regs.reduction_regs +
                       regs.indexing_regs;
  regs.total_regs = subtotal + std::max(4, static_cast<int>(std::ceil(subtotal * 0.1)));
  regs.may_spill = regs.total_regs > kRegSoftCap;
  return regs;
}

OccupancyEstimate FusionAnalyzer::estimateOccupancy(const TileShape &tile,
                                                    std::size_t shared_bytes,
                                                    int total_regs) {
  OccupancyEstimate occ;
  const int threads_per_block = std::max(32, tile.threads_per_block);
  const int smem_block_bytes = static_cast<int>(std::max<std::size_t>(shared_bytes, 1));
  const int regs_per_block = std::max(1, total_regs * threads_per_block);
  const int blocks_by_smem = kSmemBytesPerSM / smem_block_bytes;
  const int blocks_by_regs = kRegsPerSM / regs_per_block;
  const int blocks_by_threads = kMaxThreadsPerSM / threads_per_block;

  occ.limited_by_smem = blocks_by_smem;
  occ.limited_by_regs = blocks_by_regs;
  occ.blocks_per_sm = std::min(
      {blocks_by_smem, blocks_by_regs, blocks_by_threads, kMaxBlocksPerSM});
  occ.warps_per_sm = occ.blocks_per_sm * (threads_per_block / 32);
  occ.occupancy =
      static_cast<float>(occ.warps_per_sm) / static_cast<float>(kMaxWarpsPerSM);
  return occ;
}

double FusionAnalyzer::scorePlan(const FusedKernelPlan &plan,
                                 const KernelGraphIR &graph) {
  double bytes_avoided = 0.0;
  std::unordered_set<std::uint32_t> region_nodes(plan.node_ids.begin(),
                                                 plan.node_ids.end());
  for (std::uint32_t node_id : plan.node_ids) {
    const auto &node = nodeById(graph, node_id);
    for (std::uint32_t output_id : node.outputs) {
      const auto &value = valueById(graph, output_id);
      if (value.value_kind == ValueKind::kOutput ||
          value.escape == EscapeKind::kEscapeToVRAM) {
        continue;
      }
      bool consumed_inside = false;
      for (std::uint32_t consumer : value.consumers) {
        if (region_nodes.count(consumer) != 0) {
          consumed_inside = true;
          break;
        }
      }
      if (!consumed_inside) {
        continue;
      }

      std::size_t elements = 1;
      for (std::int64_t dim : value.shape) {
        elements *= static_cast<std::size_t>(std::max<std::int64_t>(dim, 1));
      }
      bytes_avoided +=
          (2.0 * static_cast<double>(elements * dtypeSize(value.dtype))) / 1024.0;
    }
  }

  const double tc_bonus = plan.uses_tensor_cores ? 1.0 : 0.0;
  const double occ_bonus = static_cast<double>(plan.occupancy.occupancy) * 10.0;
  const double smem_penalty =
      std::max(0.0, static_cast<double>(plan.shared_bytes - kSmemSoftBudget)) /
      1024.0;
  const double reg_penalty =
      std::max(0.0, static_cast<double>(plan.regs.total_regs - kRegSoftCap)) /
      16.0;
  const double spill_penalty = plan.regs.may_spill ? 1.0 : 0.0;

  return 4.0 * bytes_avoided + 2.0 * tc_bonus + 1.0 * occ_bonus -
         2.5 * smem_penalty - 3.0 * reg_penalty - 5.0 * spill_penalty;
}

}  // namespace matcore
