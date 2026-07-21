#include "fusion_emitter_internal.h"

#include <stdexcept>
#include <vector>

namespace matcore {

mlir::OwningOpRef<mlir::ModuleOp> FusionMlirEmitter::Emit(
    const KernelIR &kernel, const FusedKernelPlan &plan,
    const std::vector<RuntimeTensorView> &tensors,
    const RequestedTargetProfile &target, mlir::MLIRContext &context) {
  fusion_emit::validateFusionPlanAgainstEmitter(kernel, plan, target);
  switch (plan.pattern) {
    case FusionPatternKind::kMatmulElementwise:
      return emitFamilyA(kernel, plan, tensors, target, context);
    case FusionPatternKind::kMatmulElementwiseMatmul:
      return emitFamilyB(kernel, plan, tensors, target, context);
    case FusionPatternKind::kMatmulSoftmaxMatmul:
      return emitFamilyC(kernel, plan, tensors, target, context);
    default:
      throw std::runtime_error("FusionMlirEmitter: unsupported pattern");
  }
}

}  // namespace matcore
