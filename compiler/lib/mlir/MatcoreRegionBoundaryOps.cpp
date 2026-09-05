#include "MatcoreOps.h"
#include "MatcoreRegionBoundaryOps.h"

namespace matcore::mdslc::mlir_dialect {
namespace {
bool descriptor(mlir::Value value) {
  return mlir::isa<RegionDescriptorType>(value.getType());
}
bool order(mlir::Value value) {
  return mlir::isa<RegionOrderType>(value.getType());
}
bool tensor(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  return type && type.getRank() == 2 && type.getElementType().isF32() &&
         !type.getEncoding();
}
void observable(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &e) {
  // Failures, fenv transitions, and writes have observable order. Unscoped
  // effects conservatively alias all descriptor handles and ordinary memory.
  e.emplace_back(mlir::MemoryEffects::Read::get());
  e.emplace_back(mlir::MemoryEffects::Write::get());
}
} // namespace

mlir::LogicalResult RegionBeginOp::verify() {
  return order(getOrder()) ? mlir::success()
                          : emitOpError("requires region_order result");
}
mlir::LogicalResult RegionGuardOp::verify() {
  if (!order(getOrder()) || !order(getChecked()) || !descriptor(getLhs()) ||
      !descriptor(getRhs()) || !descriptor(getOutput()) ||
      (getStage() != 0 && getStage() != 1))
    return emitOpError("requires descriptor bindings, order and stage 0 or 1");
  return mlir::success();
}
mlir::LogicalResult RegionReadOp::verify() {
  if (!descriptor(getDescriptor()) || !order(getChecked()) ||
      !tensor(getValue()) || (getStage() != 0 && getStage() != 1) ||
      (getRole() != "lhs" && getRole() != "rhs"))
    return emitOpError("requires guarded rank-2 f32 input snapshot");
  return mlir::success();
}
mlir::LogicalResult RegionCommitOp::verify() {
  if (!descriptor(getDescriptor()) || !order(getChecked()) ||
      !order(getOrder()) || !tensor(getValue()) ||
      getValue().getType() != getCommitted().getType() ||
      getFailureBehavior() != "may_write_output_before_failure_no_rollback" ||
      (getStage() != 0 && getStage() != 1))
    return emitOpError("requires checked output commit and equal tensor types");
  return mlir::success();
}
mlir::LogicalResult RegionEndOp::verify() {
  return order(getOrder()) ? mlir::success()
                          : emitOpError("requires region_order operand");
}
void RegionBeginOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &e) {
  observable(e);
}
void RegionGuardOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &e) {
  observable(e);
}
void RegionReadOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &e) {
  e.emplace_back(mlir::MemoryEffects::Read::get());
}
void RegionCommitOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &e) {
  observable(e);
}
void RegionEndOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &e) {
  observable(e);
}
} // namespace matcore::mdslc::mlir_dialect
