#include "matcore/region_verifier.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("RegionV1 verifier: " + message);
}

void requireValueId(const RegionIR &region, std::uint32_t value_id,
                    const std::string &context) {
  if (value_id >= region.values.size()) {
    fail(context + " references missing value id " + std::to_string(value_id));
  }
}

const TensorDesc &value(const RegionIR &region, std::uint32_t value_id,
                        const std::string &context) {
  requireValueId(region, value_id, context);
  return region.values[value_id];
}

void requireRank(const TensorDesc &desc, std::size_t rank,
                 const std::string &name) {
  if (desc.shape.size() != rank) {
    fail(name + " must be rank " + std::to_string(rank));
  }
}

void requireFloat32(const TensorDesc &desc, const std::string &name) {
  if (desc.dtype != TensorDType::kFloat32) {
    fail(name + " must be float32");
  }
}

void verifyPositiveStaticShape(const TensorDesc &desc, const std::string &name) {
  for (std::size_t i = 0; i < desc.shape.size(); ++i) {
    if (desc.shape[i] <= 0) {
      fail(name + " dimension " + std::to_string(i) +
           " must be a positive static size");
    }
  }
}

void verifyRuntimeTensorMatches(const RuntimeTensorView &runtime,
                                const TensorDesc &desc,
                                const std::string &name) {
  if (runtime.dtype != desc.dtype) {
    fail(name + " runtime dtype does not match region descriptor");
  }
  if (runtime.shape.size() != desc.shape.size()) {
    fail(name + " runtime rank does not match region descriptor");
  }
  if (runtime.strides.size() != runtime.shape.size()) {
    fail(name + " runtime shape/stride rank mismatch");
  }
  for (std::size_t i = 0; i < desc.shape.size(); ++i) {
    if (runtime.shape[i] != desc.shape[i]) {
      fail(name + " runtime shape does not match region descriptor");
    }
  }
}

std::unordered_map<std::uint32_t, const RegionNode *> buildNodeMap(
    const RegionIR &region) {
  std::unordered_map<std::uint32_t, const RegionNode *> nodes;
  for (const RegionNode &node : region.nodes) {
    if (!nodes.emplace(node.id, &node).second) {
      fail("duplicate node id " + std::to_string(node.id));
    }
  }
  return nodes;
}

void verifyTopoOrder(
    const RegionIR &region,
    const std::unordered_map<std::uint32_t, const RegionNode *> &nodes) {
  if (region.topo_order.size() != region.nodes.size()) {
    fail("topo_order must contain every region node exactly once");
  }

  std::unordered_set<std::uint32_t> seen;
  std::unordered_set<std::uint32_t> topo_ids;
  for (std::uint32_t node_id : region.topo_order) {
    auto it = nodes.find(node_id);
    if (it == nodes.end()) {
      fail("topo_order references missing node id " + std::to_string(node_id));
    }
    if (!topo_ids.insert(node_id).second) {
      fail("topo_order contains duplicate node id " + std::to_string(node_id));
    }

    const RegionNode &node = *it->second;
    for (std::uint32_t input_id : node.inputs) {
      const TensorDesc &input = value(region, input_id, "node input");
      if (input.producer.has_value() && !seen.count(*input.producer)) {
        fail("node " + std::to_string(node.id) +
             " appears before producer " + std::to_string(*input.producer));
      }
    }
    seen.insert(node_id);
  }
}

void verifyInputsAndOutputs(const RegionIR &region) {
  if (region.input_values.empty()) {
    fail("region must declare input_values");
  }
  if (region.output_values.size() != 1) {
    fail("region_v1 currently supports exactly one output value");
  }

  std::unordered_set<std::uint32_t> input_ids;
  for (std::uint32_t value_id : region.input_values) {
    const TensorDesc &desc = value(region, value_id, "input_values");
    if (!input_ids.insert(value_id).second) {
      fail("input_values contains duplicate value id " + std::to_string(value_id));
    }
    if (desc.value_kind != ValueKind::kInput) {
      fail("input value " + std::to_string(value_id) + " must have kind=input");
    }
  }

  const std::uint32_t output_id = region.output_values.front();
  const TensorDesc &output = value(region, output_id, "output_values");
  if (output.value_kind != ValueKind::kOutput) {
    fail("output value " + std::to_string(output_id) + " must have kind=output");
  }
  if (!output.producer.has_value()) {
    fail("output value " + std::to_string(output_id) + " must have a producer");
  }
}

void verifyNodeReferences(const RegionIR &region) {
  std::unordered_set<std::uint32_t> produced_values;
  for (const RegionNode &node : region.nodes) {
    if (node.outputs.empty()) {
      fail("node " + std::to_string(node.id) + " must produce at least one value");
    }
    for (std::uint32_t input_id : node.inputs) {
      requireValueId(region, input_id, "node " + std::to_string(node.id) +
                                           " input");
    }
    for (std::uint32_t output_id : node.outputs) {
      const TensorDesc &output =
          value(region, output_id, "node " + std::to_string(node.id) +
                                     " output");
      if (!produced_values.insert(output_id).second) {
        fail("value " + std::to_string(output_id) +
             " is produced by multiple nodes");
      }
      if (!output.producer.has_value() || *output.producer != node.id) {
        fail("value " + std::to_string(output_id) +
             " producer must match node " + std::to_string(node.id));
      }
    }
  }
}

void verifyBlockAttnRes(const RegionIR &region, const RegionNode &node) {
  if (node.inputs.size() != 3) {
    fail("block_attn_res node must have exactly three inputs");
  }
  if (node.outputs.size() != 1) {
    fail("block_attn_res node must have exactly one output");
  }

  const auto &attrs = std::get<BlockAttnResAttrs>(node.attrs);
  if (attrs.blocks != node.inputs[0] || attrs.partial != node.inputs[1] ||
      attrs.query != node.inputs[2]) {
    fail("block_attn_res attrs must reference node inputs in ABI order");
  }

  const TensorDesc &blocks = value(region, attrs.blocks, "block_attn_res blocks");
  const TensorDesc &partial =
      value(region, attrs.partial, "block_attn_res partial");
  const TensorDesc &query = value(region, attrs.query, "block_attn_res query");
  const TensorDesc &output =
      value(region, node.outputs.front(), "block_attn_res output");

  requireFloat32(blocks, "block_attn_res blocks");
  requireFloat32(partial, "block_attn_res partial");
  requireFloat32(query, "block_attn_res query");
  requireFloat32(output, "block_attn_res output");
  requireRank(blocks, 4, "block_attn_res blocks");
  requireRank(partial, 3, "block_attn_res partial");
  requireRank(query, 1, "block_attn_res query");
  requireRank(output, 3, "block_attn_res output");
  verifyPositiveStaticShape(blocks, "block_attn_res blocks");
  verifyPositiveStaticShape(partial, "block_attn_res partial");
  verifyPositiveStaticShape(query, "block_attn_res query");
  verifyPositiveStaticShape(output, "block_attn_res output");

  const std::int64_t max_blocks = blocks.shape[0];
  const std::int64_t batch = blocks.shape[1];
  const std::int64_t tokens = blocks.shape[2];
  const std::int64_t width = blocks.shape[3];
  if (max_blocks > 32) {
    fail("block_attn_res MAX_BLOCKS must be <= 32");
  }
  if (partial.shape[0] != batch || partial.shape[1] != tokens ||
      partial.shape[2] != width || output.shape[0] != batch ||
      output.shape[1] != tokens || output.shape[2] != width ||
      query.shape[0] != width) {
    fail("block_attn_res shapes must match packed ABI");
  }
  if (attrs.block_count < 0 || attrs.block_count > max_blocks) {
    fail("block_attn_res block_count must be in [0, MAX_BLOCKS]");
  }
  if (!std::isfinite(attrs.eps) || attrs.eps <= 0.0f) {
    fail("block_attn_res eps must be finite and positive");
  }
}

void verifyOps(const RegionIR &region) {
  for (const RegionNode &node : region.nodes) {
    switch (node.kind) {
      case RegionOpKind::kBlockAttnRes:
        verifyBlockAttnRes(region, node);
        break;
    }
  }
}

void verifyRuntimeTensors(const RegionIR &region,
                          const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.empty()) {
    return;
  }
  const std::size_t expected =
      region.input_values.size() + region.output_values.size();
  if (tensors.size() != expected) {
    fail("runtime tensor count must match region inputs plus outputs");
  }
  for (std::size_t i = 0; i < region.input_values.size(); ++i) {
    verifyRuntimeTensorMatches(
        tensors[i], value(region, region.input_values[i], "runtime input"),
        "runtime input " + std::to_string(i));
  }
  verifyRuntimeTensorMatches(
      tensors.back(), value(region, region.output_values.front(), "runtime output"),
      "runtime output");
}

}  // namespace

void ValidateRegionIR(const KernelIR &kernel,
                      const std::vector<RuntimeTensorView> &tensors) {
  if (kernel.version != KernelIRVersion::kRegionV1) {
    fail("kernel version is not region_v1");
  }
  if (!kernel.region.has_value()) {
    fail("missing region_v1 body");
  }

  const RegionIR &region = *kernel.region;
  if (region.values.empty()) {
    fail("region must contain values");
  }
  if (region.nodes.empty()) {
    fail("region must contain at least one node");
  }

  const auto nodes = buildNodeMap(region);
  verifyInputsAndOutputs(region);
  verifyNodeReferences(region);
  verifyTopoOrder(region, nodes);
  verifyOps(region);
  verifyRuntimeTensors(region, tensors);
}

}  // namespace matcore
