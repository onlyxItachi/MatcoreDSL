#include "MatcoreGpuFusedGemmCandidate.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>
namespace candidate=matcore::mdslc::gpu_candidate;
static unsigned checks=0, corruptions=0;
static void require(bool yes,const std::string &message) {
  ++checks; if (!yes) { llvm::errs()<<"FAIL "<<message<<'\n'; std::exit(1); }
}
static std::string changed(std::string input,const std::string &before,const std::string &after) {
  auto position=input.find(before); require(position!=std::string::npos,"mutation anchor missing: "+before);
  input.replace(position,before.size(),after); return input;
}
int main() {
  mlir::MLIRContext context; context.disableMultithreading();
  for (auto target:{candidate::TargetV1::NvvmSm89,candidate::TargetV1::RocdlGfx1150}) {
    auto artifact=candidate::issueStrictGpuFusedGemmArtifactV1(context,target);
    require(bool(artifact),artifact.error);
    require(artifact.manifest.find("source_authority=none_builtin_primitive_only")!=std::string::npos,
            "primitive manifest incorrectly grants source authority");
    std::string error;
    auto outlined=mlir::parseSourceString<mlir::ModuleOp>(artifact.outlined_ir,&context);
    require(outlined && candidate::verifyStrictGpuFusedGemmOutlinedV1(*outlined,error),error);
    auto outlineNegative=[&](const char *label,const char *before,const char *after) {
      auto text=changed(artifact.outlined_ir,before,after);
      auto module=mlir::parseSourceString<mlir::ModuleOp>(text,&context);
      require(module && mlir::succeeded(mlir::verify(*module)),std::string(label)+" was not valid MLIR");
      require(!candidate::verifyStrictGpuFusedGemmOutlinedV1(*module,error),std::string(label)+" corruption accepted");
      ++corruptions;
    };
    outlineNegative("row-step","%c4 = arith.constant 4","%c4 = arith.constant 2");
    outlineNegative("zero-seed","arith.constant 0.000000e+00 : f32","arith.constant -0.000000e+00 : f32");
    outlineNegative("multiply-order","arith.mulf %3, %4","arith.mulf %4, %3");
    outlineNegative("consumer-bound","to %dim_0 step %c1 {\n              %3 = memref.load",
                                     "to %dim_4 step %c1 {\n              %3 = memref.load");
    outlineNegative("workspace-alias-output","memref.reinterpret_cast %arg4","memref.reinterpret_cast %arg3");
    outlineNegative("consumer-transpose","%reinterpret_cast_10[%arg6, %arg8]","%reinterpret_cast_10[%arg8, %arg6]");
    outlineNegative("D-orientation","memref.reinterpret_cast %arg2","memref.reinterpret_cast %arg0");
    outlineNegative("extra-effect","gpu.return","gpu.barrier\n      gpu.return");
    outlineNegative("threads","known_block_size = array<i32: 1, 1, 1>","known_block_size = array<i32: 2, 1, 1>");
    outlineNegative("grid-race","known_grid_size = array<i32: 1, 1, 1>","known_grid_size = array<i32: 2, 1, 1>");
    outlineNegative("hidden-alloc","gpu.return","%extra = memref.alloc() : memref<4x4xf32>\n      gpu.return");
    outlineNegative("nontemporal-true","memref.store %cst, %arg3[%arg5, %arg6] :",
                     "memref.store %cst, %arg3[%arg5, %arg6] {nontemporal = true} :");
    llvm::LLVMContext llvmContext; llvm::SMDiagnostic diagnostic;
    auto module=llvm::parseAssemblyString(artifact.device_llvm_ir,diagnostic,llvmContext);
    require(module && candidate::verifyStrictGpuFusedGemmLLVMV1(*module,target,error),error);
    auto llvmNegative=[&](const char *label,const char *before,const char *after) {
      auto text=changed(artifact.device_llvm_ir,before,after);
      if(std::string(label)=="unknown-call") text+="\ndeclare void @external_effect()\n";
      auto corrupt=llvm::parseAssemblyString(text,diagnostic,llvmContext);
      require(bool(corrupt),std::string(label)+" was not valid LLVM");
      require(!candidate::verifyStrictGpuFusedGemmLLVMV1(*corrupt,target,error),std::string(label)+" corruption accepted");
      ++corruptions;
    };
    llvmNegative("contraction","fmul float","fmul contract float");
    llvmNegative("addition","fadd float","fsub float");
    llvmNegative("input-noalias","ptr %0,","ptr noalias %0,");
    llvmNegative("negative-zero","store float 0.000000e+00","store float -0.000000e+00");
    llvmNegative("row-active-extent","i64 %56, i64 4","i64 %56, i64 3");
    llvmNegative("IV","add i64","sub i64");
    llvmNegative("GEP-permission","getelementptr float","getelementptr inbounds float");
    llvmNegative("workspace-to-output","getelementptr inbounds nuw float, ptr %29",
                                      "getelementptr inbounds nuw float, ptr %22");
    llvmNegative("FP-context","\"denormal-fp-math-f32\"=\"ieee,ieee\"","\"denormal-fp-math-f32\"=\"preserve-sign,preserve-sign\"");
    llvmNegative("unknown-call","  br label %36","  call void @external_effect()\n  br label %36");
    // The unknown call needs a declaration, so separately authenticate extra
    // declarations as an import corruption rather than an ill-formed fixture.
    auto extra=artifact.device_llvm_ir+"\ndeclare void @external_effect()\n";
    auto imported=llvm::parseAssemblyString(extra,diagnostic,llvmContext);
    require(imported && !candidate::verifyStrictGpuFusedGemmLLVMV1(*imported,target,error),"unknown import accepted");
    ++corruptions;
    auto other=target==candidate::TargetV1::NvvmSm89 ? candidate::TargetV1::RocdlGfx1150 : candidate::TargetV1::NvvmSm89;
    require(!candidate::verifyStrictGpuFusedGemmLLVMV1(*module,other,error),"wrong target graph accepted"); ++corruptions;
    auto repeat=candidate::issueStrictGpuFusedGemmArtifactV1(context,target);
    require(repeat && repeat.device_llvm_ir==artifact.device_llvm_ir && repeat.manifest==artifact.manifest,
            "closed issuer was nondeterministic");
  }
  auto invalid=candidate::issueStrictGpuFusedGemmArtifactV1(context,static_cast<candidate::TargetV1>(99));
  require(!invalid && !invalid.error.empty(),"unknown target admitted");
  llvm::outs()<<"PASS strict GPU pair closed issuer checks="<<checks<<" valid-IR corruptions="<<corruptions<<'\n';
}
