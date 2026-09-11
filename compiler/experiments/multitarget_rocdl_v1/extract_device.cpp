// Research-only operation-preserving extraction, not a semantic issuer.
// Same extraction boundary as the sibling NVVM experiment; production should
// share an issuer-owned derivation instead of accepting caller-provided MLIR.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  mlir::MLIRContext context;
  context.loadDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::func::FuncDialect, mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
      mlir::ROCDL::ROCDLDialect, mlir::memref::MemRefDialect>();
  auto source = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!source || mlir::failed(mlir::verify(*source))) return 1;
  unsigned count = 0;
  for (auto gpu : source->getOps<mlir::gpu::GPUModuleOp>()) {
    mlir::OwningOpRef<mlir::ModuleOp> device =
        mlir::ModuleOp::create(gpu.getLoc());
    if (auto layout = gpu->getAttr("llvm.data_layout"))
      (*device)->setAttr("llvm.data_layout", layout);
    (*device)->setAttr("llvm.target_triple",
        mlir::StringAttr::get(&context, "amdgcn-amd-amdhsa"));
    for (auto &operation : gpu.getBody()->getOperations()) {
      if (operation.getName().getDialectNamespace() != "llvm") return 1;
      device->push_back(operation.clone());
    }
    if (mlir::failed(mlir::verify(*device))) return 1;
    std::error_code error;
    const std::string path = std::string(argv[2]) + "/" +
                             gpu.getName().str() + ".device.mlir";
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
    if (error) return 1;
    device->print(output);
    output.close();
    if (output.has_error()) return 1;
    llvm::outs() << path << '\n';
    ++count;
  }
  return count == 2 ? 0 : 1;
}
