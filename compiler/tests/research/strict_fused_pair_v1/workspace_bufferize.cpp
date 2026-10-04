// Research tool, not an importer, candidate issuer or execution authority.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Config/llvm-config.h"

int main(int argc, char **argv) {
  if (argc != 2 || llvm::StringRef(LLVM_VERSION_STRING) != "21.1.8")
    return 1;
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::bufferization::BufferizationDialect, mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
      mlir::transform::TransformDialect>();
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module || !llvm::hasSingleElement(module->getOps<mlir::func::FuncOp>()))
    return 2;
  auto fn = *module->getOps<mlir::func::FuncOp>().begin();
  if (fn.getNumArguments() != 4 || fn.getNumResults() != 1 ||
      fn.getName() != "research_strict_fused_pair")
    return 3;
  auto workspaceType = mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
      mlir::Float32Type::get(&context));
  if (mlir::failed(fn.insertArgument(4, workspaceType, {}, fn.getLoc())))
    return 3;
  unsigned mapped = 0;
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
  options.allocationFn = [&](mlir::OpBuilder &builder, mlir::Location loc,
      mlir::MemRefType type, mlir::ValueRange sizes, unsigned) -> mlir::FailureOr<mlir::Value> {
    // The only permitted allocation request is the exact serial row panel.
    // Full source C representability and actual workspace checks remain callers'
    // obligations; this research tool supplies no runtime guards.
    if (++mapped != 1 || type != workspaceType || sizes.size() != 2) {
      llvm::errs() << "rejected request type/count: " << type << ", " << sizes.size() << '\n';
      return mlir::failure();
    }
    auto rows = sizes[0].getDefiningOp<mlir::affine::AffineMinOp>();
    auto cols = sizes[1].getDefiningOp<mlir::memref::DimOp>();
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(builder.getInsertionBlock()->getParentOp());
    auto colSource = cols ? cols.getSource() : mlir::Value{};
    if (auto conversion = colSource.getDefiningOp<mlir::bufferization::ToBufferOp>())
      colSource = conversion.getTensor();
    if (auto conversion = colSource.getDefiningOp<mlir::bufferization::ToTensorOp>())
      colSource = conversion.getBuffer();
    auto lower = loop ? loop.getLowerBound().getDefiningOp<mlir::arith::ConstantOp>() : mlir::arith::ConstantOp{};
    auto step = loop ? loop.getStep().getDefiningOp<mlir::arith::ConstantOp>() : mlir::arith::ConstantOp{};
    if (!rows || !cols || !loop || rows->getParentOp() != loop ||
        loop->getParentOp() != fn || !lower || !step ||
        !mlir::isa<mlir::IntegerAttr>(lower.getValue()) ||
        !mlir::isa<mlir::IntegerAttr>(step.getValue()) ||
        mlir::cast<mlir::IntegerAttr>(lower.getValue()).getInt() != 0 ||
        mlir::cast<mlir::IntegerAttr>(step.getValue()).getInt() != 4 ||
        rows.getAffineMap().getNumDims() != 1 || rows.getAffineMap().getNumSymbols() != 1 ||
        rows.getAffineMap().getNumResults() != 2 ||
        rows.getAffineMap().getResult(1) != mlir::getAffineConstantExpr(4, &context) ||
        rows.getAffineMap().getResult(0) !=
          -mlir::getAffineDimExpr(0, &context) + mlir::getAffineSymbolExpr(0, &context) ||
        rows.getOperand(0) != loop.getInductionVar() ||
        rows.getOperand(1) != loop.getUpperBound() ||
        colSource != fn.getArgument(1) || cols.getConstantIndex() != 1) {
      llvm::errs() << "rejected request shape/origin\n";
      return mlir::failure();
    }
    llvm::SmallVector<mlir::OpFoldResult> shape{sizes[0], sizes[1]};
    llvm::SmallVector<mlir::OpFoldResult> strides{sizes[1], builder.getIndexAttr(1)};
    return mlir::memref::ReinterpretCastOp::create(builder, loc, type,
        fn.getArgument(4), builder.getIndexAttr(0), shape, strides).getResult();
  };
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics stats;
  if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(*module, options, state, &stats)) ||
      mapped != 1 || mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "workspace mapping count: " << mapped << '\n';
    return 4;
  }
  bool unexpected = false;
  module->walk([&](mlir::Operation *op) {
    auto name = op->getName().getStringRef();
    unexpected |= name == "memref.alloc" || name == "memref.alloca" || name == "memref.realloc" ||
        name == "memref.dealloc" || name == "bufferization.alloc_tensor" ||
        name == "bufferization.clone";
  });
  if (unexpected)
    return 5;
  module->print(llvm::outs());
  llvm::outs() << '\n';
}
