// Isolated feasibility experiment, not a production issuer or source authority.
// Arithmetic comes only from the closed existing strict pair stage builder.
#include "MatcoreCpuFusedGemmCandidate.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
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
#include "llvm/Support/raw_ostream.h"
#include <stdexcept>

using namespace mlir;
namespace pair = matcore::mdslc::cpu_candidate;
constexpr char kernelName[] = "__matcore_research_strict_fused_pair_kernel";

static void require(bool yes, llvm::StringRef error) {
  if (!yes) throw std::runtime_error(error.str());
}
static void save(llvm::StringRef directory, llvm::StringRef name, ModuleOp module) {
  std::error_code error;
  llvm::raw_fd_ostream out((directory + "/" + name).str(), error);
  require(!error, error.message());
  module.print(out, OpPrintingFlags().useLocalScope());
  out << '\n';
}
static void saveText(llvm::StringRef directory, llvm::StringRef name,
                     llvm::StringRef value) {
  std::error_code error;
  llvm::raw_fd_ostream out((directory + "/" + name).str(), error);
  require(!error, error.message());
  out << value;
}

// Keep the tempting existing mapper's result, even when it cannot outline all
// nested launches. This is a losing option, not a selected fallback.
static void naiveMapping(MLIRContext &context, ModuleOp input, llvm::StringRef dir) {
  OwningOpRef<ModuleOp> module(cast<ModuleOp>(input->clone()));
  PassManager map(&context);
  map.addPass(createConvertLinalgToParallelLoopsPass());
  map.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  map.addPass(createConvertParallelLoopToGpuPass());
  map.addPass(createCanonicalizerPass());
  bool okay = succeeded(map.run(*module));
  save(dir, "naive-parallel-mapping.mlir", *module);
  unsigned launches = 0, loops = 0;
  module->walk([&](gpu::LaunchOp) { ++launches; });
  module->walk([&](scf::ForOp) { ++loops; });
  saveText(dir, "naive-parallel-mapping.txt",
      "passes=" + std::string(okay ? "PASS" : "FAIL") +
      "\ngpu.launch-sites=" + std::to_string(launches) +
      "\nscf.for-sites=" + std::to_string(loops) +
      "\nselected=no; separate kernels inside host row loop, not one fused GPU kernel\n");
}

static OwningOpRef<ModuleOp> wrapAndOutline(MLIRContext &context, ModuleOp input,
                                          llvm::StringRef dir) {
  OwningOpRef<ModuleOp> module(cast<ModuleOp>(input->clone()));
  PassManager serial(&context);
  serial.addPass(createConvertLinalgToLoopsPass());
  serial.addPass(memref::createExpandStridedMetadataPass());
  serial.addPass(createCanonicalizerPass());
  require(succeeded(serial.run(*module)), "serial Linalg/metadata lowering failed");
  save(dir, "serial.mlir", *module);
  auto host = module->lookupSymbol<func::FuncOp>(pair::kStrictFusedGemmSymbolV1);
  require(host && host.getNumArguments() == 5 && host.getNumResults() == 0 &&
          llvm::hasSingleElement(host.getBody()), "closed pair signature drifted");
  auto &body = host.getBody().front();
  SmallVector<Operation *> original;
  for (auto &op : body) if (!isa<func::ReturnOp>(op)) original.push_back(&op);
  OpBuilder builder(&body, body.begin());
  auto one = builder.create<arith::ConstantIndexOp>(host.getLoc(), 1);
  auto launch = builder.create<gpu::LaunchOp>(host.getLoc(), one, one, one,
                                            one, one, one);
  // This only encloses the already lowered closed pair. No loop, arithmetic,
  // reduction, buffer or indexing expression is synthesized here.
  auto *launchBody = &launch.getBody().front();
  builder.setInsertionPointToEnd(launchBody);
  auto end = builder.create<gpu::TerminatorOp>(host.getLoc());
  for (auto *op : original) op->moveBefore(end);
  require(succeeded(verify(*module)), "invalid structural gpu.launch wrapper");
  save(dir, "wrapped.mlir", *module);

  // Upstream outlineKernelFunc explicitly supports preseeded capture order.
  // Seed exactly A,B,D,E,scratch instead of inferring ABI from first-use order.
  SmallVector<Value> captures(host.getArguments());
  auto kernel = outlineKernelFunc(launch, kernelName, captures);
  require(captures.size() == 5 && llvm::equal(captures, host.getArguments()),
          "outlining added or reordered a captured value");
  builder.setInsertionPointToEnd(module->getBody());
  auto device = builder.create<gpu::GPUModuleOp>(host.getLoc(), "strict_pair");
  device.getBody()->push_back(kernel);
  host.erase();
  module->getOperation()->setAttr("gpu.container_module", builder.getUnitAttr());
  PassManager cleanup(&context);
  cleanup.addPass(createCanonicalizerPass());
  require(succeeded(cleanup.run(*module)), "outlined cleanup failed");
  require(succeeded(verify(*module)) && kernel.isKernel() &&
          kernel.getNumArguments() == 5 && kernel.getNumResults() == 0 &&
          kernel.getWorkgroupAttributions().empty() &&
          kernel.getPrivateAttributions().empty(), "outlined kernel contract drifted");
  auto block = kernel->getAttrOfType<DenseI32ArrayAttr>("known_block_size");
  auto grid = kernel->getAttrOfType<DenseI32ArrayAttr>("known_grid_size");
  require(block && grid && block.asArrayRef() == llvm::ArrayRef<int32_t>({1,1,1}) &&
          grid.asArrayRef() == llvm::ArrayRef<int32_t>({1,1,1}), "not exactly one thread");
  unsigned mul = 0, add = 0;
  kernel.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    require(name != "memref.alloc" && name != "memref.alloca" &&
            name != "memref.dealloc" && name != "memref.copy" &&
            name != "func.call" && name != "gpu.barrier", "unexpected device effect");
    if (auto m = dyn_cast<arith::MulFOp>(op)) {
      require(m.getType().isF32() && m.getFastmath() == arith::FastMathFlags::none,
              "producer/consumer multiply changed"); ++mul;
    }
    if (auto a = dyn_cast<arith::AddFOp>(op)) {
      require(a.getType().isF32() && a.getFastmath() == arith::FastMathFlags::none,
              "producer/consumer addition changed"); ++add;
    }
  });
  require(mul == 2 && add == 2, "not exactly two strict reduction sites");
  save(dir, "outlined.mlir", *module);
  return module;
}

static void lower(MLIRContext &context, ModuleOp outlined, bool nvvm,
                  llvm::StringRef dir) {
  OwningOpRef<ModuleOp> module(cast<ModuleOp>(outlined->clone()));
  PassManager passes(&context);
  auto &device = passes.nest<gpu::GPUModuleOp>();
  device.addPass(createLowerAffinePass());
  device.addPass(createSCFToControlFlowPass());
  if (nvvm) {
    ConvertGpuOpsToNVVMOpsOptions options;
    options.indexBitwidth = 64;
    device.addPass(createConvertGpuOpsToNVVMOps(options));
  } else {
    device.addPass(createLowerGpuOpsToROCDLOpsPass("gfx1150", 64, false,
                                                  gpu::amd::Runtime::HIP));
  }
  device.addPass(createCanonicalizerPass());
  device.addPass(createReconcileUnrealizedCastsPass());
  require(succeeded(passes.run(*module)), "GPU target conversion failed");
  save(dir, nvvm ? "nvvm.mlir" : "rocdl.mlir", *module);
  auto gpuModule = module->lookupSymbol<gpu::GPUModuleOp>("strict_pair");
  require(bool(gpuModule), "missing unique GPU module");
  OwningOpRef<ModuleOp> exported = ModuleOp::create(gpuModule.getLoc());
  if (auto layout = gpuModule->getAttr("llvm.data_layout"))
    exported->getOperation()->setAttr("llvm.data_layout", layout);
  for (auto &op : gpuModule.getBody()->getOperations()) {
    require(op.getName().getDialectNamespace() == "llvm", "unlowered top-level op");
    exported->push_back(op.clone());
  }
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*exported, llvmContext);
  require(bool(llvmModule), "LLVM translation failed");
  llvmModule->setTargetTriple(llvm::Triple(nvvm ? "nvptx64-nvidia-cuda" :
                                                   "amdgcn-amd-amdhsa"));
  auto *function = llvmModule->getFunction(kernelName);
  require(function && !function->isDeclaration() && function->arg_size() == 35 &&
          function->getReturnType()->isVoidTy(), "not the exact 35-field GPU ABI");
  unsigned argumentIndex = 0, mul = 0, add = 0;
  for (auto &argument : function->args()) {
    unsigned field = argumentIndex++ % 7;
    require(field < 2 ? argument.getType()->isPointerTy() :
                       argument.getType()->isIntegerTy(64), "descriptor field drift");
    require(!argument.hasNoAliasAttr(), "input alias permission appeared");
  }
  for (auto &f : *llvmModule) {
    require(&f == function || (f.isDeclaration() && f.isIntrinsic()),
            "extra definition or unknown imported function");
    for (auto &block : f) for (auto &inst : block) {
      if (auto *fp = llvm::dyn_cast<llvm::FPMathOperator>(&inst))
        require(!fp->getFastMathFlags().any(), "LLVM fastmath appeared");
      mul += inst.getOpcode() == llvm::Instruction::FMul;
      add += inst.getOpcode() == llvm::Instruction::FAdd;
      require(!llvm::isa<llvm::AllocaInst>(inst), "device stack allocation appeared");
      if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
        auto *callee = call->getCalledFunction();
        require(callee && callee->getName().starts_with("llvm.smin.i64"),
                "unexpected device intrinsic or indirect call");
      }
    }
  }
  require(mul == 2 && add == 2, "LLVM strict scalar arithmetic sites changed");
  function->addFnAttr("denormal-fp-math", "ieee,ieee");
  function->addFnAttr("denormal-fp-math-f32", "ieee,ieee");
  function->addFnAttr("target-cpu", nvvm ? "sm_89" : "gfx1150");
  if (nvvm) {
    function->addFnAttr("nvptx-f32ftz", "false");
    function->addFnAttr("target-features", "+ptx80");
  }
  require(!llvm::verifyModule(*llvmModule), "invalid exported LLVM");
  std::string text;
  llvm::raw_string_ostream output(text);
  llvmModule->print(output, nullptr);
  saveText(dir, nvvm ? "nvvm.ll" : "rocdl.ll", text);
}

int main(int argc, char **argv) {
  try {
    require(argc == 2 && llvm::StringRef(LLVM_VERSION_STRING) == "21.1.8",
            "usage: exact LLVM21.1.8 issue OUTPUT_DIRECTORY");
    MLIRContext context;
    context.disableMultithreading();
    DialectRegistry conversions;
    arith::registerConvertArithToLLVMInterface(conversions);
    cf::registerConvertControlFlowToLLVMInterface(conversions);
    registerConvertMemRefToLLVMInterface(conversions);
    context.appendDialectRegistry(conversions);
    context.loadDialect<gpu::GPUDialect, NVVM::NVVMDialect, ROCDL::ROCDLDialect,
        LLVM::LLVMDialect, scf::SCFDialect, affine::AffineDialect>();
    registerBuiltinDialectTranslation(context);
    registerLLVMDialectTranslation(context);
    registerNVVMDialectTranslation(context);
    registerROCDLDialectTranslation(context);
    auto stages = pair::buildStrictFusedGemmStagesV1(context);
    require(bool(stages), stages.error);
    save(argv[1], "semantic.mlir", *stages.semantic);
    save(argv[1], "structured.mlir", *stages.structured);
    save(argv[1], "scheduled.mlir", *stages.scheduled);
    save(argv[1], "bufferized.mlir", *stages.bufferized);
    naiveMapping(context, *stages.bufferized, argv[1]);
    auto outlined = wrapAndOutline(context, *stages.bufferized, argv[1]);
    lower(context, *outlined, true, argv[1]);
    lower(context, *outlined, false, argv[1]);
    llvm::outs() << "PASS closed upstream pair -> serial GPU kernel -> NVVM/ROCDL; "
                    "five memrefs A,B,D,E,scratch; 35 expanded ABI fields\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL isolated pair GPU experiment: " << error.what() << '\n';
    return 1;
  }
}
