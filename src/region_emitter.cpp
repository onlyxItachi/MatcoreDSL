#include "matcore/region_emitter.h"

#include <stdexcept>

namespace matcore {

mlir::OwningOpRef<mlir::ModuleOp> RegionMlirEmitter::Emit(
    const KernelIR &kernel, const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  if (!kernel.region.has_value()) {
    throw std::runtime_error("RegionMlirEmitter: missing region_v1 IR");
  }
  const RegionIR &region = *kernel.region;
  if (region.nodes.size() != 1) {
    throw std::runtime_error(
        "RegionMlirEmitter: region_v1 currently supports exactly one op");
  }
  const RegionNode &node = region.nodes.front();
  switch (node.kind) {
    case RegionOpKind::kBlockAttnRes:
      return emitBlockAttnRes(kernel, node, tensors, target, context);
  }
  throw std::runtime_error("RegionMlirEmitter: unsupported region op");
}

}  // namespace matcore
