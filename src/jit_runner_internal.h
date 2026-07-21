#pragma once

#include <memory>

#include "matcore/mlir_engine.h"

namespace mlir {
class MLIRContext;
class ExecutionEngine;
}  // namespace mlir

namespace matcore {

enum class ExecutionBackend {
  kExecutionEngine,
  kSharedLibrary,
};

struct CachedExecution {
  ~CachedExecution();

  ExecutionBackend backend = ExecutionBackend::kExecutionEngine;
  std::unique_ptr<mlir::MLIRContext> context;
  LoweredModule lowered;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  void *shared_library_handle = nullptr;
  void *ciface_entrypoint = nullptr;
};

}  // namespace matcore
