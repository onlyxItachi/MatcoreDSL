#include "MatcoreCpuFusedGemmCandidate.h"
#include "MatcoreCpuGemmCandidate.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <vector>

namespace candidate = matcore::mdslc::cpu_candidate;
int checks = 0, failures = 0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) { ++failures; std::cerr << "FAIL " << label << '\n'; }
}
int main() {
  mlir::MLIRContext context;
  auto stages = candidate::buildStrictFusedGemmStagesV1(context);
  check(bool(stages), "closed pair derivation: " + stages.error);
  if (!stages) return 1;
  std::string error;
  check(candidate::verifyStrictFusedGemmStructuredV1(*stages.structured, error), "exact structured pair");
  check(candidate::verifyStrictFusedGemmScheduledV1(*stages.scheduled, error), "exact row-panel tensor schedule");
  check(candidate::verifyStrictFusedGemmBufferizedV1(*stages.bufferized, error), "exact caller workspace buffer schedule");
  check(!candidate::verifyStrictFusedGemmStructuredV1(*stages.semantic, error), "semantic inspection is not a candidate");
  check(!candidate::verifyStrictGemmStructuredV1(*stages.structured, error), "old single-GEMM verifier is not relaxed");
  check(!candidate::verifyStrictFusedGemmStructuredV1(*stages.scheduled, error), "scheduled graph cannot impersonate original pair");
  auto verify = [&](int stage, mlir::ModuleOp m) {
    return stage == 0 ? candidate::verifyStrictFusedGemmStructuredV1(m, error) :
        stage == 1 ? candidate::verifyStrictFusedGemmScheduledV1(m, error) : candidate::verifyStrictFusedGemmBufferizedV1(m, error);
  };
  auto clone = [&](int stage) { return mlir::OwningOpRef<mlir::ModuleOp>((stage == 0 ? stages.structured : stage == 1 ? stages.scheduled : stages.bufferized)->clone()); };
  auto reject = [&](int stage, auto mutate, const std::string &label) {
    auto bad = clone(stage); mutate(*bad);
    check(!verify(stage, *bad) && !error.empty(), label);
  };
  for (int stage = 0; stage < 3; ++stage) {
    reject(stage, [&](mlir::ModuleOp m) { m->setAttr("mdsl.execution_authority", mlir::StringAttr::get(&context, "trusted")); }, "forged authority");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::linalg::MatmulOp op) { op->setAttr("mdsl.noalias", mlir::UnitAttr::get(&context)); }); }, "forged nested alias evidence");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::arith::ConstantOp op) { if (op.getResult().getType().isF32()) op.setValueAttr(mlir::FloatAttr::get(mlir::Float32Type::get(&context), -0.0)); }); }, "negative-zero seed");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::arith::MulFOp op) { op.setFastmath(mlir::arith::FastMathFlags::contract); }); }, "FMA permission");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::arith::AddFOp op) { op.setFastmath(mlir::arith::FastMathFlags::reassoc); }); }, "reduction reassociation");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::arith::AddFOp op) { auto lhs = op.getLhs(); op->setOperand(0, op.getRhs()); op->setOperand(1, lhs); }); }, "commuted accumulator cannot hide in graph equivalence");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::linalg::MatmulOp op) { auto lhs = op.getDpsInputs()[0]; op->setOperand(0, op.getDpsInputs()[1]); op->setOperand(1, lhs); }); }, "lhs/rhs orientation");
    reject(stage, [&](mlir::ModuleOp m) { auto fn = *m.getOps<mlir::func::FuncOp>().begin(); fn.setName("foreign_pair"); }, "foreign symbol");
    reject(stage, [&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin(); mlir::OpBuilder b(&context);
      b.setInsertionPointToStart(&fn.getBody().front()); mlir::arith::ConstantOp::create(b, fn.getLoc(), b.getF32FloatAttr(9.0));
    }, "extra operation");
  }
  reject(0, [&](mlir::ModuleOp m) {
    std::vector<mlir::linalg::MatmulOp> products; m.walk([&](mlir::linalg::MatmulOp op) { products.push_back(op); });
    products[1]->setOperand(1, products[0].getResult(0));
  }, "C*C is not a single-use lhs chain");
  for (int stage : {1, 2}) {
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::arith::ConstantIndexOp op) { if (op.value() == 4) op.setValueAttr(mlir::IntegerAttr::get(mlir::IndexType::get(&context), 8)); }); }, "changed row tile size");
    reject(stage, [&](mlir::ModuleOp m) { m.walk([&](mlir::affine::AffineMinOp op) { op->setAttr("unexpected", mlir::UnitAttr::get(&context)); }); }, "unreviewed affine footprint");
  }
  reject(1, [&](mlir::ModuleOp m) {
    mlir::Value rows; m.walk([&](mlir::tensor::DimOp op) { if (op.getConstantIndex() == 0) rows = op.getResult(); });
    m.walk([&](mlir::tensor::EmptyOp op) { op->setOperand(0, rows); });
  }, "full intermediate instead of row panel");
  reject(2, [&](mlir::ModuleOp m) {
    auto fn = *m.getOps<mlir::func::FuncOp>().begin();
    m.walk([&](mlir::memref::ReinterpretCastOp op) { op->setOperand(0, fn.getArgument(3)); });
  }, "scratch substituted by output");
  reject(2, [&](mlir::ModuleOp m) { m.walk([&](mlir::memref::ReinterpretCastOp op) { op->setOperand(2, op->getOperand(1)); }); }, "wrong panel column extent");
  reject(2, [&](mlir::ModuleOp m) {
    auto fn = *m.getOps<mlir::func::FuncOp>().begin(); mlir::OpBuilder b(&context); b.setInsertionPointToStart(&fn.getBody().front());
    mlir::memref::AllocaOp::create(b, fn.getLoc(), mlir::MemRefType::get({4}, b.getF32Type()));
  }, "hidden stack tensor allocation");

  auto first = candidate::issueStrictFusedGemmArtifactV1(context, false);
  auto second = candidate::issueStrictFusedGemmArtifactV1(context, false);
  auto sanitized = candidate::issueStrictFusedGemmArtifactV1(context, true);
  check(bool(first), "LLVM artifact issued: " + first.error);
  check(second && first.llvm_ir == second.llvm_ir && first.manifest == second.manifest, "deterministic closed issuance");
  check(sanitized && sanitized.llvm_ir.find("sanitize_address") != std::string::npos, "actual generated-function sanitizer attribute");
  check(first.semantic_ir.find("inspection_only_no_execution") != std::string::npos && first.manifest.find("source_authority=none_builtin_primitive_only") != std::string::npos, "no source authority");
  if (!first) return 1;
  llvm::LLVMContext llvmContext;
  auto parse = [&]() { llvm::SMDiagnostic diagnostic; return llvm::parseAssemblyString(first.llvm_ir, diagnostic, llvmContext); };
  auto llvmReject = [&](auto mutate, const std::string &label) {
    auto bad = parse(); if (!bad) { check(false, "cannot parse issued LLVM"); return; }
    mutate(*bad); check(!candidate::verifyStrictFusedGemmLLVMV1(*bad, false, error) && !error.empty(), label);
  };
  llvmReject([&](llvm::Module &m) { m.getFunction(candidate::kStrictFusedGemmSymbolV1)->setName("foreign"); }, "foreign LLVM leaf");
  llvmReject([&](llvm::Module &m) { m.getFunction(candidate::kStrictFusedGemmSymbolV1)->setLinkage(llvm::GlobalValue::WeakAnyLinkage); }, "replaceable weak LLVM leaf");
  llvmReject([&](llvm::Module &m) { m.getFunction(candidate::kStrictFusedGemmSymbolV1)->addParamAttr(1, llvm::Attribute::NoAlias); }, "input-input alias contract not fabricated");
  llvmReject([&](llvm::Module &m) { m.getFunction(candidate::kStrictFusedGemmSymbolV1)->addFnAttr("target-features", "+avx2"); }, "unissued target features");
  llvmReject([&](llvm::Module &m) { m.getFunction(candidate::kStrictFusedGemmSymbolV1)->addFnAttr("no-nans-fp-math", "true"); }, "unsafe non-instruction FP permission");
  llvmReject([&](llvm::Module &m) {
    for (auto &b : *m.getFunction(candidate::kStrictFusedGemmSymbolV1)) for (auto &i : b)
      if (i.getOpcode() == llvm::Instruction::Add && i.getOperand(1)->getType()->isIntegerTy(64) && llvm::isa<llvm::ConstantInt>(i.getOperand(1))) {
        i.setOperand(1, llvm::ConstantInt::get(i.getOperand(1)->getType(), 9)); return;
      }
  }, "integer loop increment/bound corruption");
  llvmReject([&](llvm::Module &m) {
    for (auto &b : *m.getFunction(candidate::kStrictFusedGemmSymbolV1)) for (auto &i : b)
      if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&i); gep && gep->getNumIndices() == 1) {
        gep->setOperand(1, llvm::ConstantInt::get(llvm::Type::getInt64Ty(llvmContext), 0)); return;
      }
  }, "float element address corruption");
  llvmReject([&](llvm::Module &m) {
    for (auto &b : *m.getFunction(candidate::kStrictFusedGemmSymbolV1)) for (auto &i : b) if (i.getOpcode() == llvm::Instruction::FMul) i.setHasAllowContract(true);
  }, "LLVM FMA permission");
  llvmReject([&](llvm::Module &m) {
    for (auto &b : *m.getFunction(candidate::kStrictFusedGemmSymbolV1)) for (auto &i : b) if (i.getOpcode() == llvm::Instruction::FAdd) { auto *lhs = i.getOperand(0); i.setOperand(0, i.getOperand(1)); i.setOperand(1, lhs); }
  }, "LLVM accumulator orientation");
  llvmReject([&](llvm::Module &m) {
    auto *leaf = m.getFunction(candidate::kStrictFusedGemmSymbolV1);
    for (auto &b : *leaf) for (auto &i : b) if (auto *s = llvm::dyn_cast<llvm::StoreInst>(&i); s && s->getValueOperand()->getType()->isFloatTy()) { s->setOperand(1, leaf->getArg(1)); return; }
  }, "LLVM store into input");
  llvmReject([&](llvm::Module &m) {
    for (auto &b : *m.getFunction(candidate::kStrictFusedGemmSymbolV1)) for (auto &i : b) if (auto *s = llvm::dyn_cast<llvm::StoreInst>(&i); s && llvm::isa<llvm::ConstantFP>(s->getValueOperand())) { s->setOperand(0, llvm::ConstantFP::get(llvmContext, llvm::APFloat(-0.0f))); return; }
  }, "LLVM negative-zero fill");
  llvmReject([&](llvm::Module &m) {
    auto *wrapper = m.getFunction(candidate::kStrictFusedGemmCInterfaceV1);
    for (auto &b : *wrapper) for (auto &i : b) if (auto *call = llvm::dyn_cast<llvm::CallInst>(&i)) { auto *a = call->getArgOperand(1); call->setArgOperand(1, call->getArgOperand(8)); call->setArgOperand(8, a); return; }
  }, "wrapper input binding swap");
  llvmReject([&](llvm::Module &m) {
    auto *leaf = m.getFunction(candidate::kStrictFusedGemmSymbolV1); llvm::IRBuilder<> b(&leaf->getEntryBlock().front());
    auto foreign = m.getOrInsertFunction("malloc", llvm::FunctionType::get(b.getPtrTy(), {b.getInt64Ty()}, false)); b.CreateCall(foreign, {b.getInt64(16)});
  }, "hidden allocator/provider call");
  llvmReject([&](llvm::Module &m) {
    auto *leaf = m.getFunction(candidate::kStrictFusedGemmSymbolV1); llvm::IRBuilder<> b(&leaf->getEntryBlock().front()); b.CreateAlloca(b.getFloatTy(), b.getInt64(32));
  }, "LLVM tensor stack allocation");
  std::cout << checks << " checks, " << failures << " failures\n"; return failures ? 1 : 0;
}
