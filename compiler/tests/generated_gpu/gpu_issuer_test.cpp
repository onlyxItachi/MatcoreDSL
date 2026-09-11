#include "MatcoreGpuGemmCandidate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <string>

int main(int argc, char **argv) {
  namespace candidate = matcore::mdslc::gpu_candidate;
  using namespace mlir;
  MLIRContext context;
  unsigned checks = 0;
  auto require = [&](bool condition, llvm::StringRef name) {
    ++checks;
    if (!condition) { llvm::errs() << "FAIL " << name << '\n'; return false; }
    return true;
  };
  auto nvvm = candidate::issueStrictGpuGemmArtifactV1(context, candidate::TargetV1::NvvmSm89);
  if (!require(bool(nvvm), nvvm.error)) return 1;
  auto rocdl = candidate::issueStrictGpuGemmArtifactV1(context, candidate::TargetV1::RocdlGfx1150);
  if (!require(bool(rocdl), rocdl.error)) return 1;
  if (!require(nvvm.semantic_ir == rocdl.semantic_ir && nvvm.structured_ir == rocdl.structured_ir &&
               nvvm.bufferized_ir == rocdl.bufferized_ir && nvvm.outlined_ir == rocdl.outlined_ir,
               "machine selection must not change semantics or common schedule")) return 1;
  auto again = candidate::issueStrictGpuGemmArtifactV1(context, candidate::TargetV1::NvvmSm89);
  if (!require(again.manifest == nvvm.manifest && again.fill_llvm_ir == nvvm.fill_llvm_ir &&
               again.gemm_llvm_ir == nvvm.gemm_llvm_ir, "deterministic issuer")) return 1;
  if (!require(!candidate::issueStrictGpuGemmArtifactV1(context, static_cast<candidate::TargetV1>(99)),
               "unknown target rejected")) return 1;
  auto parse = [&]() { return parseSourceString<ModuleOp>(nvvm.outlined_ir, &context); };
  std::string error;
  auto correct = parse();
  if (!require(correct && candidate::verifyStrictGpuGemmOutlinedV1(*correct, error), "outlined positive")) return 1;
  const auto attack = [&](llvm::StringRef name, std::function<void(ModuleOp)> mutate) {
    auto changed = parse();
    mutate(*changed);
    return require(!candidate::verifyStrictGpuGemmOutlinedV1(*changed, error), name);
  };
  if (!attack("negative zero initialization", [&](ModuleOp module) {
    module.walk([&](arith::ConstantOp op) {
      if (op.getType().isF32()) op.setValueAttr(FloatAttr::get(op.getType(), -0.0));
    });
  })) return 1;
  if (!attack("unknown execution attribute", [&](ModuleOp module) {
    module->setAttr("mdsl.execution_authority", UnitAttr::get(&context));
  })) return 1;
  if (!attack("reassociation flag", [&](ModuleOp module) {
    module.walk([&](arith::AddFOp op) { op.setFastmath(arith::FastMathFlags::reassoc); });
  })) return 1;
  if (!attack("FMA permission flag", [&](ModuleOp module) {
    module.walk([&](arith::MulFOp op) { op.setFastmath(arith::FastMathFlags::contract); });
  })) return 1;
  if (!attack("reduction starts at one", [&](ModuleOp module) {
    module.walk([&](scf::ForOp op) { op.getLowerBoundMutable().assign(op.getStep()); });
  })) return 1;
  if (!attack("reduction uses wrong dimension", [&](ModuleOp module) {
    module.walk([&](scf::ForOp op) {
      auto dimension = op.getUpperBound().getDefiningOp<memref::DimOp>();
      dimension.getIndexMutable().assign(op.getLowerBound());
    });
  })) return 1;
  if (!attack("A transposed indexing", [&](ModuleOp module) {
    bool first = true;
    module.walk([&](memref::LoadOp op) {
      if (!first) return;
      first = false;
      auto indices = op.getIndices();
      op.getIndicesMutable().assign(ValueRange{indices[1], indices[0]});
    });
  })) return 1;
  if (!attack("destination redirected to input", [&](ModuleOp module) {
    module.walk([&](scf::ForOp op) {
      auto gpu = op->getParentOfType<gpu::GPUFuncOp>();
      op.walk([&](memref::StoreOp store) { store.getMemrefMutable().assign(gpu.getArgument(0)); });
    });
  })) return 1;
  if (!attack("launch order reversed", [&](ModuleOp module) {
    auto host = *module.getOps<func::FuncOp>().begin();
    auto launches = host.getOps<gpu::LaunchFuncOp>();
    auto it = launches.begin();
    auto fill = *it++;
    auto gemm = *it;
    fill->moveAfter(gemm);
  })) return 1;
  if (!attack("output result redirected", [&](ModuleOp module) {
    auto host = *module.getOps<func::FuncOp>().begin();
    auto ret = cast<func::ReturnOp>(host.getBody().front().getTerminator());
    ret->setOperand(0, host.getArgument(0));
  })) return 1;
  if (!attack("wrong thread count", [&](ModuleOp module) {
    module.walk([&](gpu::GPUFuncOp op) {
      op->setAttr("known_block_size", DenseI32ArrayAttr::get(&context, {2, 1, 1}));
    });
  })) return 1;
  if (!attack("forged input noalias", [&](ModuleOp module) {
    module.walk([&](gpu::GPUFuncOp op) { op.setArgAttr(0, "llvm.noalias", UnitAttr::get(&context)); });
  })) return 1;
  if (argc == 2) {
    const auto save = [&](std::string name, const std::string &text) {
      std::error_code ec;
      llvm::raw_fd_ostream output(std::string(argv[1]) + "/" + name, ec, llvm::sys::fs::OF_None);
      if (ec) return false;
      output << text;
      output.close();
      return !output.has_error();
    };
    if (!save("nvvm.fill.ll", nvvm.fill_llvm_ir) || !save("nvvm.gemm.ll", nvvm.gemm_llvm_ir) ||
        !save("rocdl.fill.ll", rocdl.fill_llvm_ir) || !save("rocdl.gemm.ll", rocdl.gemm_llvm_ir) ||
        !save("nvvm.manifest", nvvm.manifest) || !save("rocdl.manifest", rocdl.manifest)) return 1;
  }
  llvm::outs() << "PASS " << checks << " shared strict GPU issuer checks\n";
}
