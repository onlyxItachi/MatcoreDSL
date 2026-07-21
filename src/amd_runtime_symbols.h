#pragma once

namespace mlir {
class ExecutionEngine;
}

namespace matcore {

bool registerAmdRuntimeSymbols(mlir::ExecutionEngine &engine);

}  // namespace matcore
