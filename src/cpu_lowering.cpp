#include "matcore/cpu_lowering.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace matcore {
namespace {

std::int64_t pickTilingFactor(std::int64_t dim, std::int64_t preferred) {
  if (preferred <= 0) {
    return 1;
  }
  if (dim <= 0) {
    return preferred;
  }

  std::int64_t tile = std::min(dim, preferred);
  while (tile > 1 && (dim % tile) != 0) {
    tile /= 2;
  }
  return std::max<std::int64_t>(tile, 1);
}

llvm::SmallVector<std::int64_t, 3> selectCpuMatmulTileSizes(
    mlir::linalg::MatmulOp matmul) {
  auto lhs_type = llvm::dyn_cast<mlir::ShapedType>(matmul.getInputs()[0].getType());
  auto rhs_type = llvm::dyn_cast<mlir::ShapedType>(matmul.getInputs()[1].getType());

  const bool low_precision = lhs_type && lhs_type.getElementType().isIntOrFloat() &&
                             (lhs_type.getElementType().isF16() ||
                              lhs_type.getElementType().isBF16());

  const std::int64_t preferred_m = low_precision ? 8 : 8;
  const std::int64_t preferred_n = low_precision ? 8 : 8;
  const std::int64_t preferred_k = low_precision ? 16 : 8;

  std::int64_t m = mlir::ShapedType::kDynamic;
  std::int64_t n = mlir::ShapedType::kDynamic;
  std::int64_t k = mlir::ShapedType::kDynamic;
  if (lhs_type && lhs_type.hasRank() && lhs_type.getRank() == 2) {
    m = lhs_type.getDimSize(0);
    k = lhs_type.getDimSize(1);
  }
  if (rhs_type && rhs_type.hasRank() && rhs_type.getRank() == 2) {
    n = rhs_type.getDimSize(1);
  }

  return {pickTilingFactor(m, preferred_m), pickTilingFactor(n, preferred_n),
          pickTilingFactor(k, preferred_k)};
}

struct TileAndVectorizeMatmulPass
    : public mlir::PassWrapper<TileAndVectorizeMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    std::vector<mlir::Operation *> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op.getOperation()); });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::Operation *matmul : matmuls) {
      auto typed_matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(matmul);
      if (!typed_matmul) {
        matmul->emitError("MatCore expected linalg.matmul while tiling");
        signalPassFailure();
        return;
      }

      mlir::linalg::LinalgTilingOptions tiling_options;
      tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
      tiling_options.setTileSizes(selectCpuMatmulTileSizes(typed_matmul));

      rewriter.setInsertionPoint(matmul);
      mlir::FailureOr<mlir::linalg::TiledLinalgOp> tiled =
          mlir::linalg::tileLinalgOp(rewriter, typed_matmul, tiling_options);
      if (mlir::failed(tiled)) {
        matmul->emitError("MatCore failed to tile linalg.matmul for x86 lowering");
        signalPassFailure();
        return;
      }

      mlir::Operation *vectorize_target = tiled->op.getOperation();
      if (mlir::failed(mlir::linalg::vectorizeOpPrecondition(vectorize_target))) {
        vectorize_target->emitError(
            "MatCore x86 lowering requires vectorizable tiled linalg.matmul");
        signalPassFailure();
        return;
      }
      if (mlir::failed(mlir::linalg::vectorize(rewriter, vectorize_target))) {
        vectorize_target->emitError("MatCore failed to vectorize tiled linalg.matmul");
        signalPassFailure();
        return;
      }
      if (typed_matmul.getOperation() != vectorize_target &&
          typed_matmul.getOperation()->getBlock() != nullptr) {
        rewriter.eraseOp(typed_matmul.getOperation());
      }
      if (vectorize_target->getBlock() != nullptr) {
        rewriter.eraseOp(vectorize_target);
      }
    }
  }
};

struct LowerResidualVectorOpsPass
    : public mlir::PassWrapper<LowerResidualVectorOpsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns, mlir::vector::VectorMultiReductionLowering::InnerReduction);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("MatCore failed to lower residual vector ops");
      signalPassFailure();
    }
  }
};

void addCommonLLVMLoweringPasses(mlir::PassManager &pm, bool enable_x86vector) {
  mlir::ConvertVectorToLLVMPassOptions vector_to_llvm_opts;
  vector_to_llvm_opts.x86Vector = enable_x86vector;

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass(vector_to_llvm_opts));
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool useCpuVectorPipeline(const MatmulLoweringSignature &signature) {
  if (signature.quantized_i8) {
    return false;
  }
  if (signature.lhs_dtype != TensorDType::kFloat32 ||
      signature.rhs_dtype != TensorDType::kFloat32 ||
      signature.out_dtype != TensorDType::kFloat32) {
    return false;
  }
  return true;
}

}  // namespace

void configureCpuPassPipeline(mlir::PassManager &pm,
                              const MatmulLoweringSignature &signature) {
  if (useCpuVectorPipeline(signature)) {
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<TileAndVectorizeMatmulPass>());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::memref::createExpandOpsPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<LowerResidualVectorOpsPass>());
    mlir::VectorTransferToSCFOptions vector_to_scf_options;
    vector_to_scf_options.setTargetRank(1);
    pm.addPass(mlir::createConvertVectorToSCFPass(vector_to_scf_options));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/true);
    return;
  }

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/false);
}

}  // namespace matcore
