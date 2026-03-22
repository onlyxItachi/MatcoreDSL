#pragma once

#include "matcore/kernel_ir.h"

namespace mlir {
class ExecutionEngine;
}

namespace matcore {

void registerGpuRuntimeSymbols(mlir::ExecutionEngine &engine, TargetKind target);

}  // namespace matcore
