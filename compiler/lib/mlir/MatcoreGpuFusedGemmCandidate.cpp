#include "MatcoreGpuFusedGemmCandidate.h"
#include "MatcoreCpuFusedGemmCandidate.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore::mdslc::gpu_candidate {
namespace {
using namespace mlir;
bool fail(std::string &error, llvm::StringRef message) {
  error = "strict GPU fused GEMM: " + message.str(); return false;
}
std::string print(ModuleOp module) {
  std::string output; llvm::raw_string_ostream stream(output);
  module.print(stream, OpPrintingFlags().useLocalScope()); return output;
}
std::string digest(llvm::StringRef text) {
  constexpr char hex[] = "0123456789abcdef";
  std::string output;
  for (auto byte : llvm::SHA256::hash(llvm::arrayRefFromStringRef(text))) {
    output += hex[byte >> 4]; output += hex[byte & 15];
  }
  return output;
}
bool validTarget(TargetV1 target) {
  return target == TargetV1::NvvmSm89 || target == TargetV1::RocdlGfx1150;
}
// Golden complete outlined graph is intentionally reviewed as a pinned21.1.8
// compiler fixture. Not a user-replaceable file or a second lowering strategy.
#include "MatcoreGpuFusedGemmOutlinedV1.inc"

OwningOpRef<ModuleOp> outline(MLIRContext &context, ModuleOp input, std::string &error) {
  OwningOpRef<ModuleOp> module(cast<ModuleOp>(input->clone()));
  PassManager serial(&context);
  serial.addPass(createConvertLinalgToLoopsPass());
  serial.addPass(memref::createExpandStridedMetadataPass());
  serial.addPass(createCanonicalizerPass());
  if (failed(serial.run(*module))) { fail(error,"fixed serial Linalg lowering failed"); return {}; }
  auto host = module->lookupSymbol<func::FuncOp>(cpu_candidate::kStrictFusedGemmSymbolV1);
  if (!host || host.getNumArguments()!=5 || host.getNumResults()!=0 ||
      !llvm::hasSingleElement(host.getBody())) { fail(error,"closed pair signature drifted"); return {}; }
  auto &body=host.getBody().front(); SmallVector<Operation *> original;
  for (auto &op:body) if (!isa<func::ReturnOp>(op)) original.push_back(&op);
  OpBuilder builder(&body,body.begin());
  auto one=builder.create<arith::ConstantIndexOp>(host.getLoc(),1);
  auto launch=builder.create<gpu::LaunchOp>(host.getLoc(),one,one,one,one,one,one);
  builder.setInsertionPointToEnd(&launch.getBody().front());
  auto end=builder.create<gpu::TerminatorOp>(host.getLoc());
  // Structural enclosure ONLY: loops/indexing/arithmetic all come from the
  // verified pair and upstream serial lowering, never handwritten GPU math.
  for (auto *op:original) op->moveBefore(end);
  if (failed(verify(*module))) { fail(error,"structural launch verification failed"); return {}; }
  SmallVector<Value> captures(host.getArguments());
  auto kernel=outlineKernelFunc(launch,kFusedGemmKernelV1,captures);
  if (captures.size()!=5 || !llvm::equal(captures,host.getArguments())) {
    kernel.erase(); fail(error,"upstream outline changed A/B/D/E/workspace ABI"); return {};
  }
  builder.setInsertionPointToEnd(module->getBody());
  auto device=builder.create<gpu::GPUModuleOp>(host.getLoc(),"strict_pair");
  device.getBody()->push_back(kernel); host.erase();
  module->getOperation()->setAttr("gpu.container_module",builder.getUnitAttr());
  PassManager cleanup(&context); cleanup.addPass(createCanonicalizerPass());
  if (failed(cleanup.run(*module)) || !verifyStrictGpuFusedGemmOutlinedV1(*module,error)) return {};
  return module;
}
std::unique_ptr<llvm::Module> lower(MLIRContext &context, ModuleOp outlined,
                                  TargetV1 target, llvm::LLVMContext &llvmContext,
                                  std::string &error) {
  OwningOpRef<ModuleOp> module(cast<ModuleOp>(outlined->clone()));
  PassManager passes(&context); auto &device=passes.nest<gpu::GPUModuleOp>();
  device.addPass(createLowerAffinePass()); device.addPass(createSCFToControlFlowPass());
  if (target==TargetV1::NvvmSm89) {
    ConvertGpuOpsToNVVMOpsOptions options; options.indexBitwidth=64;
    device.addPass(createConvertGpuOpsToNVVMOps(options));
  } else device.addPass(createLowerGpuOpsToROCDLOpsPass("gfx1150",64,false,gpu::amd::Runtime::HIP));
  device.addPass(createCanonicalizerPass()); device.addPass(createReconcileUnrealizedCastsPass());
  if (failed(passes.run(*module))) { fail(error,"fixed GPU conversion failed"); return {}; }
  auto gpuModule=module->lookupSymbol<gpu::GPUModuleOp>("strict_pair");
  if (!gpuModule) { fail(error,"GPU module identity lost"); return {}; }
  OwningOpRef<ModuleOp> exported=ModuleOp::create(gpuModule.getLoc());
  if (auto layout=gpuModule->getAttr("llvm.data_layout"))
    exported->getOperation()->setAttr("llvm.data_layout",layout);
  for (auto &op:gpuModule.getBody()->getOperations()) {
    if (op.getName().getDialectNamespace()!="llvm") { fail(error,"unlowered device op"); return {}; }
    exported->push_back(op.clone());
  }
  auto llvmModule=translateModuleToLLVMIR(*exported,llvmContext);
  if (!llvmModule) { fail(error,"GPU LLVM translation failed"); return {}; }
  llvmModule->setTargetTriple(llvm::Triple(target==TargetV1::NvvmSm89 ?
                                         "nvptx64-nvidia-cuda" : "amdgcn-amd-amdhsa"));
  auto *function=llvmModule->getFunction(kFusedGemmKernelV1);
  if (!function) { fail(error,"GPU function identity lost"); return {}; }
  function->addFnAttr("denormal-fp-math","ieee,ieee");
  function->addFnAttr("denormal-fp-math-f32","ieee,ieee");
  function->addFnAttr("target-cpu",target==TargetV1::NvvmSm89 ? "sm_89" : "gfx1150");
  if (target==TargetV1::NvvmSm89) {
    function->addFnAttr("nvptx-f32ftz","false"); function->addFnAttr("target-features","+ptx80");
  }
  if (!verifyStrictGpuFusedGemmLLVMV1(*llvmModule,target,error)) return {};
  return llvmModule;
}
} // namespace

bool verifyStrictGpuFusedGemmOutlinedV1(mlir::ModuleOp module, std::string &error) {
  error.clear();
  if (!module || failed(verify(module))) return fail(error,"invalid outlined module");
  auto reference=parseSourceString<ModuleOp>(kOutlinedReference,module.getContext());
  OwningOpRef<ModuleOp> normalized(cast<ModuleOp>(module->clone()));
  // Upstream load/store builders encode nontemporal=false, while the custom
  // parser omits the equivalent optional default. Normalize ONLY this missing
  // false representation on clones; true and every other property remain exact.
  auto defaults=[&](ModuleOp value) {
    value.walk([&](Operation *op) {
      if(auto load=dyn_cast<memref::LoadOp>(op)) {
        if(!load.getNontemporalAttr()) load.setNontemporalAttr(BoolAttr::get(module.getContext(),false));
      } else if(auto store=dyn_cast<memref::StoreOp>(op)) {
        if(!store.getNontemporalAttr()) store.setNontemporalAttr(BoolAttr::get(module.getContext(),false));
      }
    });
  };
  defaults(*normalized); if(reference) defaults(*reference);
  if (!reference || !OperationEquivalence::isEquivalentTo(*normalized,*reference,
                                         OperationEquivalence::IgnoreLocations)) {
    return fail(error,"pinned full GPU graph changed (ABI, shapes, loops, +0, f32, workspace or effects)");
  }
  return true;
}

bool verifyStrictGpuFusedGemmLLVMV1(const llvm::Module &module, TargetV1 target,
                                 std::string &error) {
  error.clear();
  if (!validTarget(target) || llvm::verifyModule(module)) return fail(error,"invalid GPU target/LLVM");
  auto *function=module.getFunction(kFusedGemmKernelV1);
  if (!function || function->isDeclaration() || function->arg_size()!=35 ||
      !function->getReturnType()->isVoidTy()) return fail(error,"exact35-field ABI changed");
  unsigned index=0,mul=0,add=0;
  for (auto &argument:function->args()) {
    unsigned field=index++%7;
    if (field<2 ? !argument.getType()->isPointerTy() : !argument.getType()->isIntegerTy(64))
      return fail(error,"expanded memref field type changed");
    if (argument.hasNoAliasAttr()) return fail(error,"input MAY-alias permission changed");
  }
  if (!module.global_empty() || !module.alias_empty() || !module.ifunc_empty() ||
      !module.getModuleInlineAsm().empty()) return fail(error,"unexpected global/alias/inline assembly");
  for (auto &f:module) {
    if (&f!=function && (!f.isDeclaration() || !f.isIntrinsic())) return fail(error,"extra function/import");
    for (auto &block:f) for (auto &inst:block) {
      if (auto *fp=llvm::dyn_cast<llvm::FPMathOperator>(&inst))
        if (fp->getFastMathFlags().any()) return fail(error,"LLVM fastmath appeared");
      mul+=inst.getOpcode()==llvm::Instruction::FMul;
      add+=inst.getOpcode()==llvm::Instruction::FAdd;
      if (llvm::isa<llvm::AllocaInst>(inst)) return fail(error,"device allocation appeared");
      if (auto *call=llvm::dyn_cast<llvm::CallBase>(&inst)) {
        auto *callee=call->getCalledFunction();
        if (!callee || callee->getName()!="llvm.smin.i64") return fail(error,"unexpected device call");
      }
    }
  }
  if (mul!=2 || add!=2) return fail(error,"two strict scalar reduction sites changed");
  // Deliberately small pinned drift gate, not a general LLVM proof checker.
  // It authenticates every IV/GEP/branch/attribute/metadata of this build's
  // closed preoptimization graph. Ignore diagnostic ModuleID only.
  std::string graph; llvm::raw_string_ostream output(graph);
  module.print(output,nullptr); output.flush();
  if (llvm::StringRef(graph).starts_with("; ModuleID = ")) graph.erase(0,graph.find('\n')+1);
  const auto expected=target==TargetV1::NvvmSm89 ?
      "561394a4bf68154b455da4d7315ed18e1ed9e7aecc7cdc246ce98f6a80afb666" :
      "870ce35559e60007d9107bffa0b9a5d10f84af88a94f37510bab1d03ff9151c9";
  return digest(graph)==expected ? true : fail(error,"pinned complete GPU LLVM graph changed");
}

GpuFusedGemmArtifactV1 issueStrictGpuFusedGemmArtifactV1(mlir::MLIRContext &context,
                                                     TargetV1 target) {
  GpuFusedGemmArtifactV1 artifact;
  if (llvm::StringRef(LLVM_VERSION_STRING)!="21.1.8" || !validTarget(target)) {
    fail(artifact.error,"requires exact21.1.8 and explicit sm89/gfx1150 target"); return artifact;
  }
  DialectRegistry conversions;
  arith::registerConvertArithToLLVMInterface(conversions);
  cf::registerConvertControlFlowToLLVMInterface(conversions);
  registerConvertMemRefToLLVMInterface(conversions); context.appendDialectRegistry(conversions);
  context.loadDialect<gpu::GPUDialect,NVVM::NVVMDialect,ROCDL::ROCDLDialect,
                     LLVM::LLVMDialect,scf::SCFDialect,affine::AffineDialect>();
  registerBuiltinDialectTranslation(context); registerLLVMDialectTranslation(context);
  registerNVVMDialectTranslation(context); registerROCDLDialectTranslation(context);
  auto stages=cpu_candidate::buildStrictFusedGemmStagesV1(context);
  if (!stages) { artifact.error=stages.error; return artifact; }
  artifact.semantic_ir=print(*stages.semantic); artifact.structured_ir=print(*stages.structured);
  artifact.scheduled_ir=print(*stages.scheduled); artifact.bufferized_ir=print(*stages.bufferized);
  auto outlined=outline(context,*stages.bufferized,artifact.error);
  if (!outlined) return artifact;
  artifact.outlined_ir=print(*outlined); llvm::LLVMContext llvmContext;
  auto llvmModule=lower(context,*outlined,target,llvmContext,artifact.error);
  if (!llvmModule) return artifact;
  llvm::raw_string_ostream out(artifact.device_llvm_ir); llvmModule->print(out,nullptr); out.flush();
  artifact.manifest="schema=matcore-builtin-strict-fused-gpu-gemm-v1\nsource_authority=none_builtin_primitive_only\n"
      "toolchain=21.1.8\nprofile=strict_f32_both\nchain=C=A*B;E=C*D\n"
      "schedule=one-kernel;grid1;block1;serial-row-panel4\nreduction_order=increasing_K_then_increasing_N\n"
      "intermediate_round=f32\nC_workspace_elements=min(4,M)*N\nlogical_C_extent_check=M*N_retained_caller_obligation\n"
      "expanded_abi=A,B,D,E,workspace;35_fields\nleaf_precondition=M,N,P_positive;K_may0\n"
      "tensor_allocations=0\ncopies=0\ninput_input_alias=permitted\n"
      "writable_storage=private_E_and_workspace_disjoint_from_each_other_and_inputs\n"
      "caller_guards=not_discharged\nsource_frontiers=not_implemented_by_leaf\n"
      "completion_and_failure=explicit_combined_adapter_obligation\nsource_residency=none\nperformance_claim=none\n"
      "qualification_pair_work_bound=M*N*(K+P)<=262144\n"
      "llvm_calls=smin_i64_only\nmachine_imports=none_required_actual_image_gate\n";
  artifact.manifest+=target==TargetV1::NvvmSm89 ? "target=nvvm-sm89\n" : "target=rocdl-gfx1150\n";
  artifact.manifest+="semantic_sha256="+digest(artifact.semantic_ir)+"\nstructured_sha256="+digest(artifact.structured_ir)+
      "\nscheduled_sha256="+digest(artifact.scheduled_ir)+"\nbufferized_sha256="+digest(artifact.bufferized_ir)+
      "\noutlined_sha256="+digest(artifact.outlined_ir)+"\nllvm_sha256="+digest(artifact.device_llvm_ir)+"\n";
  return artifact;
}
} // namespace matcore::mdslc::gpu_candidate
