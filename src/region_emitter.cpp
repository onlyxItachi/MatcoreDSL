#include "matcore/region_emitter.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace matcore {
namespace {

std::vector<const RegionNode *> orderedRegionNodes(const RegionIR &region) {
  std::unordered_map<std::uint32_t, const RegionNode *> by_id;
  for (const RegionNode &node : region.nodes) {
    by_id.emplace(node.id, &node);
  }

  std::vector<const RegionNode *> ordered;
  ordered.reserve(region.nodes.size());
  if (!region.topo_order.empty()) {
    for (std::uint32_t node_id : region.topo_order) {
      auto it = by_id.find(node_id);
      if (it == by_id.end()) {
        throw std::runtime_error(
            "RegionMlirEmitter: topo_order references missing node");
      }
      ordered.push_back(it->second);
    }
    return ordered;
  }

  for (const RegionNode &node : region.nodes) {
    ordered.push_back(&node);
  }
  return ordered;
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> RegionMlirEmitter::Emit(
    const KernelIR &kernel, const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  if (!kernel.region.has_value()) {
    throw std::runtime_error("RegionMlirEmitter: missing region_v1 IR");
  }
  const RegionIR &region = *kernel.region;
  const std::vector<const RegionNode *> ordered_nodes = orderedRegionNodes(region);
  if (ordered_nodes.empty()) {
    throw std::runtime_error(
        "RegionMlirEmitter: region_v1 requires at least one op");
  }
  if (ordered_nodes.size() != 1) {
    throw std::runtime_error(
        "RegionMlirEmitter: verified multi-op regions are accepted, but "
        "multi-op lowering is not implemented yet");
  }

  const RegionNode &node = *ordered_nodes.front();
  switch (node.kind) {
    case RegionOpKind::kBlockAttnRes:
      return emitBlockAttnRes(kernel, node, tensors, target, context);
  }
  throw std::runtime_error("RegionMlirEmitter: unsupported region op");
}

}  // namespace matcore
