#pragma once

#include "matcore/kernel_ir.h"

namespace mlir {
class ExecutionEngine;
}

namespace matcore {

class ObservabilityContext;

void setGpuRuntimeObservabilityContext(ObservabilityContext *obs);
void registerGpuRuntimeSymbols(mlir::ExecutionEngine &engine, TargetKind target);

}  // namespace matcore
