#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "matcore/gpu_mapping.h"
#include "matcore/lowering_pipeline.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

namespace matcore {

class TransformBuilder {
 public:
  TransformBuilder(mlir::MLIRContext *ctx, mlir::Location loc);

  mlir::OwningOpRef<mlir::ModuleOp> build();

  void matchOp(llvm::StringRef opName);

  void tileLinalg(mlir::ArrayRef<int64_t> tileSizes);
  void tileLinalgToGpuBlocks(mlir::ArrayRef<int64_t> tileSizes);
  void tileLinalgToGpuThreads(mlir::ArrayRef<int64_t> numThreads);
  void tileLinalgWithFor(mlir::ArrayRef<int64_t> tileSizes);

  void vectorize();

  void mapToGpuThreads(mlir::ArrayRef<int64_t> blockDims);
  void mapToGpuBlocks(mlir::ArrayRef<int64_t> gridDims);

  void bufferize();

  void promoteTensorToSharedMemory();
  void rewriteMatmulAsMmaSync();

 private:
  using StepFn = std::function<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                           mlir::Value)>;

  void addTileUsingForallStep(mlir::ArrayRef<int64_t> sizes, bool useNumThreads,
                              bool useBlockMapping);

  mlir::MLIRContext *ctx_;
  mlir::Location loc_;
  mlir::OpBuilder builder_;
  std::vector<StepFn> steps_;
};

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaTransformMappingModule(
    mlir::MLIRContext *ctx, mlir::Location loc,
    const MatmulLoweringSignature &signature,
    const NvidiaMappingConfig &config);

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaThreadMappingModule(
    mlir::MLIRContext *ctx, mlir::Location loc,
    const NvidiaMappingConfig &config);

mlir::OwningOpRef<mlir::ModuleOp> BuildNvidiaMmaRewriteModule(
    mlir::MLIRContext *ctx, mlir::Location loc);

}  // namespace matcore
