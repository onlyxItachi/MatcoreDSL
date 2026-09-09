#include "MatcoreCpuReassociateGemmCandidate.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>

namespace cc = matcore::mdslc::cpu_candidate;
unsigned checks = 0, failures = 0;
void check(bool ok, const std::string &label) {
  ++checks;
  if (!ok) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
template<class Op> Op first(mlir::ModuleOp module) {
  Op found;
  module.walk([&](Op op) { if (!found) found = op; });
  return found;
}
mlir::scf::ForOp carried(mlir::ModuleOp module) {
  mlir::scf::ForOp found;
  module.walk([&](mlir::scf::ForOp op) { if (op.getNumRegionIterArgs()) found = op; });
  return found;
}
int main() {
  mlir::MLIRContext context;
  auto stages = cc::buildReassociateGemmStagesV1(context);
  auto strict = cc::buildStrictGemmStagesV1(context);
  check(bool(stages) && bool(strict), "separate verified semantic primitives");
  if (!stages || !strict) return 1;
  std::string error;
  check(cc::verifyReassociateGemmStructuredV1(*stages.structured, error), "reassociate structured");
  check(cc::verifyReassociateGemmBufferizedV1(*stages.bufferized, error), "reassociate buffer");
  check(!cc::verifyStrictGemmStructuredV1(*stages.structured, error), "reassociate is not strict identity");
  check(!cc::verifyReassociateGemmStructuredV1(*strict.structured, error), "strict is not reassociate identity");
  check(!cc::deriveReassociateGemmRegisterV1(*strict.bufferized, error), "strict buffer cannot acquire vector FMA schedule");
  unsigned profiles = 0;
  stages.semantic->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "mdsl_admission.gemm") return;
    ++profiles;
    check(op->getAttrOfType<mlir::StringAttr>("numerical_profile").getValue() == "reassociate_f32" &&
          op->getAttrOfType<mlir::StringAttr>("multiply_add_contraction").getValue() == "permitted" &&
          op->getAttrOfType<mlir::StringAttr>("cross_operation_reassociation").getValue() == "forbidden",
          "authoritative mathematical permission, not scheduling relabel");
  });
  check(profiles == 1, "exactly one reassociate GEMM");
  auto stageReject = [&](bool buffer, auto mutate, const char *label) {
    auto bad = buffer ? stages.bufferized->clone() : stages.structured->clone();
    mlir::OwningOpRef<mlir::ModuleOp> owned = bad;
    mutate(*owned);
    check(!(buffer ? cc::verifyReassociateGemmBufferizedV1(*owned, error)
                   : cc::verifyReassociateGemmStructuredV1(*owned, error)) && !error.empty(), label);
  };
  for (bool buffer : {false, true}) {
    stageReject(buffer, [&](mlir::ModuleOp m) {
      auto op = first<mlir::linalg::MatmulOp>(m);
      auto a = op->getOperand(0); op->setOperand(0, op->getOperand(1)); op->setOperand(1, a);
    }, "wrong noncommuting operands");
    stageReject(buffer, [&](mlir::ModuleOp m) {
      first<mlir::arith::ConstantOp>(m).setValueAttr(mlir::FloatAttr::get(mlir::Float32Type::get(&context), -0.0));
    }, "wrong zero overwrite");
    stageReject(buffer, [&](mlir::ModuleOp m) {
      first<mlir::arith::MulFOp>(m).setFastmath(mlir::arith::FastMathFlags::fast);
    }, "reassociate does not mean blanket fast math");
    stageReject(buffer, [&](mlir::ModuleOp m) {
      m->setAttr("execution_authority", mlir::UnitAttr::get(&context));
    }, "serialized authority rejected");
  }
  auto scheduled = cc::deriveReassociateGemmRegisterV1(*stages.bufferized, error);
  check(bool(scheduled), "real upstream tiled/vector derivation: " + error);
  if (!scheduled) return 1;
  check(cc::verifyReassociateGemmRegisterV1(*scheduled, error), "checked exact register derivation");
  check(cc::verifyReassociateGemmBufferizedV1(*stages.bufferized, error), "derivation preserves input buffer stage");
  auto reject = [&](auto mutate, const char *label) {
    mlir::OwningOpRef<mlir::ModuleOp> bad = scheduled->clone();
    mutate(*bad);
    check(!cc::verifyReassociateGemmRegisterV1(*bad, error) && !error.empty(), label);
  };
  reject([&](mlir::ModuleOp m) { carried(m).getUpperBoundMutable().assign(carried(m).getLowerBound()); }, "K terms omitted");
  reject([&](mlir::ModuleOp m) {
    auto k = carried(m); mlir::OpBuilder b(k);
    auto two = mlir::arith::ConstantOp::create(b, k.getLoc(), b.getIndexAttr(2));
    k.getStepMutable().assign(two);
  }, "K terms skipped by step");
  reject([&](mlir::ModuleOp m) {
    auto k = carried(m); auto read = k.getInitArgs()[0].getDefiningOp<mlir::vector::TransferReadOp>();
    read.getBaseMutable().assign(first<mlir::func::FuncOp>(m).getArgument(0));
  }, "accumulator seed aliases A");
  reject([&](mlir::ModuleOp m) {
    auto k = carried(m); auto read = k.getInitArgs()[0].getDefiningOp<mlir::vector::TransferReadOp>();
    read->moveBefore(k.getBody(), k.getBody()->begin());
  }, "destination reset/read moved inside K");
  reject([&](mlir::ModuleOp m) {
    auto outer = first<mlir::vector::OuterProductOp>(m); mlir::OpBuilder b(outer);
    auto zero = mlir::arith::ConstantOp::create(b, outer.getLoc(),
      mlir::DenseElementsAttr::get(outer.getResult().getType(), b.getF32FloatAttr(0.0)));
    outer.getAccMutable().assign(zero);
  }, "well-typed zero reset every K");
  reject([&](mlir::ModuleOp m) {
    auto fn = first<mlir::func::FuncOp>(m);
    auto fill = *fn.front().getOps<mlir::scf::ForOp>().begin();
    fill->moveBefore(fn.front().getTerminator());
  }, "well-formed late initialization after contraction");
  reject([&](mlir::ModuleOp m) {
    first<mlir::vector::TransferWriteOp>(m).getBaseMutable().assign(first<mlir::func::FuncOp>(m).getArgument(0));
  }, "vector publication into input");
  reject([&](mlir::ModuleOp m) {
    auto write = first<mlir::vector::TransferWriteOp>(m);
    write.getIndicesMutable()[1].set(write.getIndices()[0]);
  }, "wrong output tile index");
  reject([&](mlir::ModuleOp m) {
    auto fn = first<mlir::func::FuncOp>(m); mlir::OpBuilder b(&fn.front(), fn.front().begin());
    mlir::memref::AllocOp::create(b, fn.getLoc(), mlir::MemRefType::get({1}, b.getF32Type()));
  }, "extra hidden allocation");
  reject([&](mlir::ModuleOp m) {
    auto fn = first<mlir::func::FuncOp>(m); mlir::OpBuilder b(&fn.front(), fn.front().begin());
    mlir::memref::CopyOp::create(b, fn.getLoc(), fn.getArgument(0), fn.getArgument(2));
  }, "extra copy");
  reject([&](mlir::ModuleOp m) { first<mlir::arith::MulFOp>(m).setFastmath(mlir::arith::FastMathFlags::nnan); }, "finite-only tail assumption");
  reject([&](mlir::ModuleOp m) {
    auto add = first<mlir::arith::AddFOp>(m); add->setOperand(0, add.getRhs());
  }, "lost scalar-tail accumulator");
  reject([&](mlir::ModuleOp m) {
    auto load = first<mlir::memref::LoadOp>(m);
    load.getIndicesMutable()[0].set(load.getIndices()[1]);
  }, "well-typed wrong scalar-tail lhs row");
  reject([&](mlir::ModuleOp m) {
    auto apply = first<mlir::affine::AffineApplyOp>(m);
    auto s = mlir::getAffineSymbolExpr(0, &context);
    apply.setMapAttr(mlir::AffineMapAttr::get(mlir::AffineMap::get(0, 1, s.floorDiv(8) * 8)));
  }, "incorrect full-M bound");
  reject([&](mlir::ModuleOp m) {
    first<mlir::vector::OuterProductOp>(m).setKind(mlir::vector::CombiningKind::MUL);
  }, "outerproduct accumulator kind");
  reject([&](mlir::ModuleOp m) {
    first<mlir::vector::TransferReadOp>(m)->setAttr("in_bounds", mlir::ArrayAttr::get(&context,
      {mlir::BoolAttr::get(&context, false), mlir::BoolAttr::get(&context, true)}));
  }, "unsupported padded/masked transfer permission");
  reject([&](mlir::ModuleOp m) { first<mlir::scf::ForOp>(m)->setAttr("distribution", mlir::UnitAttr::get(&context)); }, "distributed-loop hoist unsupported");

  auto artifact = cc::issueReassociateGemmArtifactV1(context, false);
  auto sanitized = cc::issueReassociateGemmArtifactV1(context, true);
  auto strictArtifact = cc::issueStrictGemmArtifactV1(context, false);
  check(bool(artifact) && bool(sanitized) && bool(strictArtifact), "closed issuance: " + artifact.error);
  if (!artifact || !sanitized || !strictArtifact) return 1;
  check(artifact.semantic_ir != strictArtifact.semantic_ir, "separate semantic identity");
  check(artifact.manifest.find("profile=reassociate_f32") != std::string::npos &&
        artifact.manifest.find("target_features=+avx2,+fma") != std::string::npos &&
        artifact.manifest.find("source_authority=none_builtin_primitive_only") != std::string::npos,
        "manifest retains profile, target, and authority distinctions");
  for (const auto *text : {&artifact.llvm_ir, &sanitized.llvm_ir}) {
    llvm::LLVMContext llvmContext; llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseAssemblyString(*text, diagnostic, llvmContext);
    check(module && cc::verifyReassociateGemmLLVMV1(*module, error), "issued LLVM exact replay: " + error);
  }
  auto llvmReject = [&](auto mutate, const char *label) {
    llvm::LLVMContext llvmContext; llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseAssemblyString(artifact.llvm_ir, diagnostic, llvmContext);
    if (!module) { check(false, "parse LLVM mutation base"); return; }
    mutate(*module);
    check(!cc::verifyReassociateGemmLLVMV1(*module, error) && !error.empty(), label);
  };
  llvmReject([](llvm::Module &m) { m.getFunction(cc::kReassociateGemmSymbolV1)->addParamAttr(1, llvm::Attribute::NoAlias); }, "extra input noalias");
  llvmReject([](llvm::Module &m) { m.getFunction(cc::kReassociateGemmSymbolV1)->removeParamAttr(15, llvm::Attribute::NoAlias); }, "lost private output fact");
  llvmReject([](llvm::Module &m) { m.getFunction(cc::kReassociateGemmCInterfaceV1)->addParamAttr(2, llvm::Attribute::NoAlias); }, "descriptor noalias is not data isolation");
  llvmReject([](llvm::Module &m) { m.getFunction(cc::kReassociateGemmSymbolV1)->addFnAttr("target-cpu", "x86-64-v3"); }, "ISA widened beyond runtime gate");
  llvmReject([](llvm::Module &m) { m.getFunction(cc::kReassociateGemmSymbolV1)->addFnAttr("target-features", "+avx2,+fma,+bmi2"); }, "hidden target feature");
  llvmReject([](llvm::Module &m) { m.setTargetTriple(llvm::Triple("aarch64-unknown-linux-gnu")); }, "foreign target");
  llvmReject([](llvm::Module &m) { m.setDataLayout("e-p:32:32"); }, "different pointer ABI");
  llvmReject([](llvm::Module &m) { m.getFunction(cc::kReassociateGemmSymbolV1)->setCallingConv(llvm::CallingConv::Fast); }, "wrong calling convention");
  llvmReject([](llvm::Module &m) {
    for (auto &block : *m.getFunction(cc::kReassociateGemmSymbolV1)) for (auto &op : block)
      if (auto *call = llvm::dyn_cast<llvm::IntrinsicInst>(&op)) {
        call->setArgOperand(2, llvm::Constant::getNullValue(call->getType())); return;
      }
  }, "K accumulator reset in LLVM");
  llvmReject([](llvm::Module &m) {
    for (auto &block : *m.getFunction(cc::kReassociateGemmSymbolV1)) for (auto &op : block)
      if (auto *call = llvm::dyn_cast<llvm::IntrinsicInst>(&op)) {
        llvm::FastMathFlags flags; flags.setFast(); call->setFastMathFlags(flags); return;
      }
  }, "LLVM blanket fast math");
  llvmReject([](llvm::Module &m) { m.setModuleInlineAsm("nop"); }, "unreviewed machine effects");
  llvmReject([](llvm::Module &m) {
    auto *callee = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(m.getContext()), false),
        llvm::GlobalValue::ExternalLinkage, "untrusted_provider", m);
    llvm::IRBuilder<> b(&*m.getFunction(cc::kReassociateGemmSymbolV1)->getEntryBlock().getFirstInsertionPt());
    b.CreateCall(callee);
  }, "unknown external effect added to leaf");
  std::cout << "reassociate CPU candidate: " << checks << " checks, " << failures << " failures\n";
  return failures != 0;
}
