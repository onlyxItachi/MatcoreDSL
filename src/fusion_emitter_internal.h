#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "matcore/fusion_emitter.h"
#include "matcore/fusion_analysis.h"
#include "matcore/kernel_ir.h"
#include "matcore/mlir_engine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace matcore::fusion_emit {

struct FamilyAEpilogueSpec {
  ElementwiseKind kind = ElementwiseKind::kAdd;
  int boundary_arg_index = -1;
  int value_input_pos = 0;
};

mlir::Type getElementType(TensorDType dtype, mlir::OpBuilder &builder);
mlir::Value emitElementwiseOnValue(mlir::OpBuilder &builder, mlir::Location loc,
                                   ElementwiseKind kind, mlir::Value val,
                                   mlir::Type f32);
mlir::Value emitBinaryElementwiseOnValues(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          ElementwiseKind kind,
                                          mlir::Value lhs, mlir::Value rhs);
mlir::Value castToF32(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type elem_type,
                      mlir::Type f32);
mlir::Value castFromF32(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, mlir::Type elem_type,
                        mlir::Type f32);
bool isUnaryElementwiseKind(ElementwiseKind kind);

llvm::SmallVector<FamilyAEpilogueSpec, 4>
decodeFamilyAEpilogueSpecs(mlir::ModuleOp module);
unsigned countGpuLaunches(mlir::func::FuncOp func);
bool inlineLastKIterationTileEpilogue(
    mlir::func::FuncOp func, mlir::Value out_arg,
    const llvm::SmallVector<FamilyAEpilogueSpec, 4> &epilogue_specs);
bool inlineSingleTileLaunchEpilogue(
    mlir::func::FuncOp func, mlir::Value out_arg,
    const llvm::SmallVector<FamilyAEpilogueSpec, 4> &epilogue_specs);

void validateFusionPlanAgainstEmitter(const KernelIR &kernel,
                                      const FusedKernelPlan &plan,
                                      const RequestedTargetProfile &target);

}  // namespace matcore::fusion_emit
