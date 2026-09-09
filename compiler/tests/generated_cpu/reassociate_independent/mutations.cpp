#include "MatcoreCpuReassociateGemmCandidate.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>

namespace cc=matcore::mdslc::cpu_candidate;
int checks=0,failures=0;
void check(bool condition,const char *what) {
  ++checks;
  if(!condition) {++failures; std::cerr<<"FAIL: "<<what<<'\n';}
}
template<class T> T first(mlir::ModuleOp module) {
  T result;
  module.walk([&](T op){if(!result) result=op;});
  return result;
}
int main() {
  mlir::MLIRContext context;
  std::string error;
  auto stages=cc::buildReassociateGemmStagesV1(context);
  auto strict=cc::buildStrictGemmStagesV1(context);
  check(bool(stages)&&bool(strict),"construct two real semantic primitives");
  if(!stages||!strict) return 1;
  for(auto *stage : {&stages,&strict}) {
    unsigned count=0;
    stage->semantic->walk([&](mlir::Operation *op) {
      if(op->getName().getStringRef()!="mdsl_admission.gemm") return;
      ++count;
      const bool relaxed=stage==&stages;
      check(op->getAttrOfType<mlir::StringAttr>("numerical_profile").getValue()==
            (relaxed?"reassociate_f32":"strict_f32"),"true per-operation semantic profile");
      check(op->getAttrOfType<mlir::StringAttr>("cross_operation_reassociation").getValue()=="forbidden",
            "complete per-GEMM f32 boundaries retained");
    });
    check(count==1,"one GEMM per builtin");
  }
  // Deliberate authority-boundary positive: identical base arithmetic is legal
  // under both profiles. A self-consistency verifier is not a source admission
  // seal, and changing an internal symbol must not be reported as authentication.
  auto renamed=mlir::OwningOpRef<mlir::ModuleOp>(strict.bufferized->clone());
  first<mlir::func::FuncOp>(*renamed).setName(cc::kReassociateGemmSymbolV1);
  check(cc::verifyReassociateGemmBufferizedV1(*renamed,error),
        "renamed shared arithmetic form passes only self-consistency");
  auto scheduled=cc::deriveReassociateGemmRegisterV1(*stages.bufferized,error);
  check(bool(scheduled),"derive independently reviewed real schedule");
  if(!scheduled) return 1;
  auto badScheduled=[&](auto mutate,const char *label,bool replayOnly=false) {
    auto bad=mlir::OwningOpRef<mlir::ModuleOp>(scheduled->clone());
    mutate(*bad);
    check(mlir::succeeded(mlir::verify(*bad)),"mutation remains structurally valid MLIR");
    const bool accepted=cc::verifyReassociateGemmRegisterV1(*bad,error);
    check(!accepted&&!error.empty(),label);
    if(replayOnly) check(error.find("exact trusted upstream replay")!=std::string::npos,
                        "rejection really uses exact replay beyond narrow envelope");
  };
  badScheduled([&](mlir::ModuleOp m) {
    auto load=first<mlir::arith::MulFOp>(m).getLhs().getDefiningOp<mlir::memref::LoadOp>();
    load.getIndicesMutable()[0].set(load.getIndices()[1]);
  },"wrong scalar-tail A row rejected",true);
  badScheduled([&](mlir::ModuleOp m) {
    auto load=first<mlir::arith::MulFOp>(m).getRhs().getDefiningOp<mlir::memref::LoadOp>();
    load.getIndicesMutable()[1].set(load.getIndices()[0]);
  },"wrong scalar-tail B column rejected",true);
  badScheduled([&](mlir::ModuleOp m) {
    auto firstLoop=first<mlir::scf::ForOp>(m);
    firstLoop.getUpperBoundMutable().assign(firstLoop.getLowerBound());
  },"zero-trip initialization cannot seed poison into vector carry");
  badScheduled([&](mlir::ModuleOp m) {
    mlir::vector::TransferReadOp a;
    auto fn=first<mlir::func::FuncOp>(m);
    m.walk([&](mlir::vector::TransferReadOp op){if(op.getBase()==fn.getArgument(0)) a=op;});
    a.setPermutationMapAttr(mlir::AffineMapAttr::get(
      mlir::AffineMap::getPermutationMap(llvm::ArrayRef<unsigned>{1,0},&context)));
  },"transposed full-tile footprint rejected");
  badScheduled([&](mlir::ModuleOp m) {
    auto fn=first<mlir::func::FuncOp>(m);
    auto read=first<mlir::vector::TransferReadOp>(m);
    mlir::OpBuilder builder(&fn.front(),fn.front().begin());
    auto zero=mlir::arith::ConstantOp::create(builder,fn.getLoc(),builder.getF32FloatAttr(0));
    read.getPaddingMutable().assign(zero);
  },"even harmless in-bounds padding changes are outside exact replay",true);
  {
    auto locations=mlir::OwningOpRef<mlir::ModuleOp>(scheduled->clone());
    locations->walk([&](mlir::Operation *op) {
      op->setLoc(mlir::FileLineColLoc::get(&context,"review-only.mlir",7,9));
    });
    check(cc::verifyReassociateGemmRegisterV1(*locations,error),
          "scheduled replay deliberately ignores diagnostic locations");
  }
  const auto artifact=cc::issueReassociateGemmArtifactV1(context,false);
  check(bool(artifact),"closed issuer creates canonical LLVM");
  if(!artifact) return 1;
  auto llvmCase=[&](auto mutate,const char *label,bool accept=false) {
    llvm::LLVMContext llvmContext;
    llvm::SMDiagnostic diagnostic;
    auto module=llvm::parseAssemblyString(artifact.llvm_ir,diagnostic,llvmContext);
    check(bool(module),"parse issued LLVM");
    if(!module) return;
    mutate(*module);
    check(!llvm::verifyModule(*module),"mutation remains structurally valid LLVM");
    check(cc::verifyReassociateGemmLLVMV1(*module,error)==accept,label);
    if(!accept) check(!error.empty(),"LLVM refusal is diagnosed");
  };
  llvmCase([](llvm::Module &m) {
    auto *wrapper=m.getFunction(cc::kReassociateGemmCInterfaceV1);
    for(auto &block:*wrapper) for(auto &op:block)
      if(auto *call=llvm::dyn_cast<llvm::CallInst>(&op)) {
        call->setArgOperand(15,call->getArgOperand(1)); return;
      }
  },"wrapper redirected aligned C to A despite unchanged noalias position");
  llvmCase([](llvm::Module &m) {
    auto *wrapper=m.getFunction(cc::kReassociateGemmCInterfaceV1);
    for(auto &block:*wrapper) for(auto &op:block)
      if(auto *call=llvm::dyn_cast<llvm::CallInst>(&op)) {
        call->setArgOperand(15,call->getArgOperand(14)); return;
      }
  },"allocated C is not the exact aligned-C descriptor mapping");
  llvmCase([](llvm::Module &m) {
    auto *leaf=m.getFunction(cc::kReassociateGemmSymbolV1);
    leaf->addParamAttr(15,llvm::Attribute::getWithAlignment(m.getContext(),llvm::Align(64)));
  },"private noalias does not invent alignment");
  llvmCase([](llvm::Module &m) {
    m.getFunction(cc::kReassociateGemmCInterfaceV1)->addFnAttr(llvm::Attribute::SanitizeAddress);
  },"wrapper-only sanitizer attribute cannot fake generated-leaf instrumentation");
  llvmCase([](llvm::Module &m) {m.setSourceFileName("different-source-file");},
           "LLVM source filename is inside exact replay");
  llvmCase([](llvm::Module &m) {m.setModuleIdentifier("different-diagnostic-comment");},
           "only LLVM diagnostic module identifier may vary",true);
  std::cout<<"Independent reassociate issuer mutations: "<<checks<<" checks; "<<failures<<" failures\n";
  return failures?1:0;
}
