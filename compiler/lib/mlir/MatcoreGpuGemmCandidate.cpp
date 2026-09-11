#include "MatcoreGpuGemmCandidate.h"
#include "MatcoreCpuGemmCandidate.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <array>

namespace matcore::mdslc::gpu_candidate {
namespace {
using namespace mlir;

bool reject(std::string &error, llvm::StringRef message) {
  error = "strict GPU GEMM: " + message.str();
  return false;
}
std::string print(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, OpPrintingFlags().useLocalScope());
  return text;
}
std::string hash(llvm::StringRef text) {
  auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(text));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : bytes) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}
bool indexConstant(Value value, int64_t expected) {
  auto op = value.getDefiningOp<arith::ConstantIndexOp>();
  return op && op.value() == expected;
}
bool dimension(Value value, Value memref, int64_t dimension) {
  auto op = value.getDefiningOp<memref::DimOp>();
  return op && op.getSource() == memref && indexConstant(op.getIndex(), dimension);
}
bool matrix(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 2 && memref.isDynamicDim(0) &&
         memref.isDynamicDim(1) && memref.getElementType().isF32() &&
         memref.getLayout().isIdentity() && !memref.getMemorySpace();
}
bool blockIndex(Value value, gpu::Dimension dimension) {
  auto op = value.getDefiningOp<gpu::BlockIdOp>();
  return op && op.getDimension() == dimension;
}
bool indices(ValueRange values, Value first, Value second) {
  return values.size() == 2 && values[0] == first && values[1] == second;
}
bool kernel(gpu::GPUFuncOp function, unsigned arguments) {
  if (!function || function.getName() != kGemmKernelV1 || !function.isKernel() ||
      function.getNumArguments() != arguments || function.getNumResults() != 0 ||
      !llvm::hasSingleElement(function.getBody()) || function.getAllArgAttrs() ||
      function.getAllResultAttrs() || !function.getWorkgroupAttributions().empty() ||
      !function.getPrivateAttributions().empty()) return false;
  for (auto argument : function.getArgumentTypes()) if (!matrix(argument)) return false;
  auto block = function->getAttrOfType<DenseI32ArrayAttr>("known_block_size");
  return block && block.asArrayRef() == llvm::ArrayRef<int32_t>({1, 1, 1});
}
bool verifyFill(gpu::GPUFuncOp function, std::string &error) {
  if (!kernel(function, 1)) return reject(error, "fill kernel signature/block size changed");
  auto &body = function.getBody().front();
  if (body.getOperations().size() != 5) return reject(error, "fill kernel has extra operations");
  auto it = body.begin();
  auto zero = dyn_cast<arith::ConstantOp>(&*it++);
  auto x = dyn_cast<gpu::BlockIdOp>(&*it++);
  auto y = dyn_cast<gpu::BlockIdOp>(&*it++);
  auto store = dyn_cast<memref::StoreOp>(&*it++);
  auto end = dyn_cast<gpu::ReturnOp>(&*it);
  auto value = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr{};
  if (!value || !value.getType().isF32() || !value.getValue().isZero() ||
      value.getValue().isNegative() || !x || !y || !store || !end ||
      x.getDimension() != gpu::Dimension::x || y.getDimension() != gpu::Dimension::y ||
      store.getValue() != zero || store.getMemRef() != body.getArgument(0) ||
      !indices(store.getIndices(), x, y))
    return reject(error, "fill is not exact positive-zero overwrite at block (M,N)");
  return true;
}
bool verifyGemm(gpu::GPUFuncOp function, std::string &error) {
  if (!kernel(function, 3)) return reject(error, "GEMM kernel signature/block size changed");
  auto &body = function.getBody().front();
  if (body.getOperations().size() != 7) return reject(error, "GEMM kernel has extra operations");
  auto it = body.begin();
  auto one = dyn_cast<arith::ConstantIndexOp>(&*it++);
  auto zero = dyn_cast<arith::ConstantIndexOp>(&*it++);
  auto x = dyn_cast<gpu::BlockIdOp>(&*it++);
  auto y = dyn_cast<gpu::BlockIdOp>(&*it++);
  auto k = dyn_cast<memref::DimOp>(&*it++);
  auto loop = dyn_cast<scf::ForOp>(&*it++);
  auto end = dyn_cast<gpu::ReturnOp>(&*it);
  if (!one || !zero || !x || !y || !k || !loop || !end ||
      !indexConstant(one, 1) || !indexConstant(zero, 0) ||
      !blockIndex(x, gpu::Dimension::x) || !blockIndex(y, gpu::Dimension::y) ||
      !dimension(k, body.getArgument(0), 1) || loop.getLowerBound() != zero ||
      loop.getUpperBound() != k || loop.getStep() != one ||
      loop.getNumResults() || !loop.getInitArgs().empty() ||
      loop.getBody()->getOperations().size() != 7)
    return reject(error, "GEMM lost increasing 0..A.K scalar reduction or M/N ownership");
  auto inner = loop.getBody()->begin();
  auto a = dyn_cast<memref::LoadOp>(&*inner++);
  auto b = dyn_cast<memref::LoadOp>(&*inner++);
  auto c = dyn_cast<memref::LoadOp>(&*inner++);
  auto mul = dyn_cast<arith::MulFOp>(&*inner++);
  auto add = dyn_cast<arith::AddFOp>(&*inner++);
  auto store = dyn_cast<memref::StoreOp>(&*inner++);
  auto yield = dyn_cast<scf::YieldOp>(&*inner);
  if (!a || !b || !c || !mul || !add || !store || !yield ||
      a.getMemRef() != body.getArgument(0) || b.getMemRef() != body.getArgument(1) ||
      c.getMemRef() != body.getArgument(2) || store.getMemRef() != body.getArgument(2) ||
      !indices(a.getIndices(), x, loop.getInductionVar()) ||
      !indices(b.getIndices(), loop.getInductionVar(), y) ||
      !indices(c.getIndices(), x, y) || !indices(store.getIndices(), x, y) ||
      mul.getLhs() != a || mul.getRhs() != b || add.getLhs() != c || add.getRhs() != mul ||
      mul.getFastmath() != arith::FastMathFlags::none ||
      add.getFastmath() != arith::FastMathFlags::none || store.getValue() != add ||
      yield.getNumOperands() != 0)
    return reject(error, "GEMM changed input orientation, C identity, scalar arithmetic or numerics");
  return true;
}
bool launch(gpu::LaunchFuncOp operation, llvm::StringRef module, ValueRange arguments,
            Value m, Value n) {
  if (!operation || operation.getKernelModuleName().getValue() != module ||
      operation.getKernelName().getValue() != kGemmKernelV1 ||
      operation.getKernelOperands() != arguments ||
      operation.getGridSizeX() != m || operation.getGridSizeY() != n ||
      !indexConstant(operation.getGridSizeZ(), 1) ||
      !indexConstant(operation.getBlockSizeX(), 1) ||
      !indexConstant(operation.getBlockSizeY(), 1) ||
      !indexConstant(operation.getBlockSizeZ(), 1) ||
      !operation.getAsyncDependencies().empty() || operation.getNumResults() ||
      operation.getDynamicSharedMemorySize() || operation.getClusterSizeX() ||
      operation.getClusterSizeY() || operation.getClusterSizeZ() || operation.getAsyncObject()) return false;
  return true;
}

std::string exportDevice(gpu::GPUModuleOp gpuModule, TargetV1 target,
                         unsigned arguments, std::string &error) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(gpuModule.getLoc());
  if (auto layout = gpuModule->getAttr("llvm.data_layout"))
    module->getOperation()->setAttr("llvm.data_layout", layout);
  for (auto &operation : gpuModule.getBody()->getOperations()) {
    if (operation.getName().getDialectNamespace() != "llvm") {
      reject(error, "unlowered top-level device operation"); return {};
    }
    module->push_back(operation.clone());
  }
  if (failed(verify(*module))) { reject(error, "invalid extracted device module"); return {}; }
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) { reject(error, "device LLVM translation failed"); return {}; }
  llvmModule->setTargetTriple(llvm::Triple(target == TargetV1::NvvmSm89 ?
      "nvptx64-nvidia-cuda" : "amdgcn-amd-amdhsa"));
  auto *function = llvmModule->getFunction(kGemmKernelV1);
  if (!function || function->isDeclaration() || !function->getReturnType()->isVoidTy() ||
      function->arg_size() != arguments) { reject(error, "expanded device ABI differs"); return {}; }
  unsigned index = 0;
  for (auto &argument : function->args()) {
    const unsigned field = index++ % 7;
    if (field < 2 ? !argument.getType()->isPointerTy() : !argument.getType()->isIntegerTy(64)) {
      reject(error, "expanded memref field type differs"); return {};
    }
  }
  for (auto &f : *llvmModule) {
    if (!f.isDeclaration() && &f != function) { reject(error, "extra device function"); return {}; }
    if (f.isDeclaration() && !f.isIntrinsic()) { reject(error, "external device call"); return {}; }
    for (auto &block : f) for (auto &instruction : block) {
      if (auto *fp = llvm::dyn_cast<llvm::FPMathOperator>(&instruction))
        if (fp->getFastMathFlags().any()) { reject(error, "LLVM fast math appeared"); return {}; }
    }
  }
  function->addFnAttr("denormal-fp-math", "ieee,ieee");
  function->addFnAttr("denormal-fp-math-f32", "ieee,ieee");
  function->addFnAttr("target-cpu", target == TargetV1::NvvmSm89 ? "sm_89" : "gfx1150");
  if (target == TargetV1::NvvmSm89) {
    function->addFnAttr("nvptx-f32ftz", "false");
    function->addFnAttr("target-features", "+ptx80");
  }
  if (llvm::verifyModule(*llvmModule)) { reject(error, "invalid exported LLVM device module"); return {}; }
  std::string output;
  llvm::raw_string_ostream stream(output);
  llvmModule->print(stream, nullptr);
  return output;
}
} // namespace

bool verifyStrictGpuGemmOutlinedV1(mlir::ModuleOp module, std::string &error) {
  using namespace mlir;
  error.clear();
  if (!module || failed(verify(module)) || module.getBody()->getOperations().size() != 3)
    return reject(error, "expected one host function and exactly two GPU modules");
  bool unsupportedAttribute = false;
  module.walk([&](Operation *op) {
    for (auto attribute : op->getDiscardableAttrs()) {
      const auto name = attribute.getName().strref();
      if ((op == module.getOperation() && name == "gpu.container_module") ||
          (isa<gpu::GPUFuncOp>(op) && (name == "known_block_size" ||
            name == "gpu.kernel" || name == "workgroup_attributions" ||
            name == "sym_name"))) continue;
      error += "unsupported attribute " + name.str() + " on " + op->getName().getStringRef().str() + "; ";
      unsupportedAttribute = true;
    }
  });
  if (unsupportedAttribute) return false;
  auto host = module.lookupSymbol<func::FuncOp>(cpu_candidate::kStrictGemmSymbolV1);
  auto fillModule = module.lookupSymbol<gpu::GPUModuleOp>(kGemmKernelV1);
  auto gemmModule = module.lookupSymbol<gpu::GPUModuleOp>(std::string(kGemmKernelV1) + "_0");
  if (!host || !fillModule || !gemmModule || host.getNumArguments() != 3 ||
      host.getNumResults() != 1 || !llvm::hasSingleElement(host.getBody()) ||
      host.getAllArgAttrs() || host.getAllResultAttrs() || fillModule.getTargets() ||
      gemmModule.getTargets() ||
      fillModule.getBody()->getOperations().size() != 1 ||
      gemmModule.getBody()->getOperations().size() != 1)
    return reject(error, "GPU module/function identity differs");
  for (auto type : host.getArgumentTypes()) if (!matrix(type)) return reject(error, "host matrix type differs");
  if (host.getResultTypes()[0] != host.getArgumentTypes()[2])
    return reject(error, "host result lost destination type");
  auto &body = host.getBody().front();
  if (body.getOperations().size() != 9) return reject(error, "host launch schedule has extra operations");
  auto it = body.begin();
  auto one = dyn_cast<arith::ConstantIndexOp>(&*it++);
  auto zero = dyn_cast<arith::ConstantIndexOp>(&*it++);
  auto cm = dyn_cast<memref::DimOp>(&*it++);
  auto cn = dyn_cast<memref::DimOp>(&*it++);
  auto fill = dyn_cast<gpu::LaunchFuncOp>(&*it++);
  auto am = dyn_cast<memref::DimOp>(&*it++);
  auto bn = dyn_cast<memref::DimOp>(&*it++);
  auto gemm = dyn_cast<gpu::LaunchFuncOp>(&*it++);
  auto ret = dyn_cast<func::ReturnOp>(&*it);
  if (!one || !zero || !cm || !cn || !am || !bn || !ret ||
      !indexConstant(one, 1) || !indexConstant(zero, 0) ||
      !dimension(cm, body.getArgument(2), 0) || !dimension(cn, body.getArgument(2), 1) ||
      !dimension(am, body.getArgument(0), 0) || !dimension(bn, body.getArgument(1), 1) ||
      !launch(fill, kGemmKernelV1, ValueRange{body.getArgument(2)}, cm, cn) ||
      !launch(gemm, std::string(kGemmKernelV1) + "_0", body.getArguments(), am, bn) ||
      ret.getNumOperands() != 1 || ret.getOperand(0) != body.getArgument(2))
    return reject(error, "fill/GEMM launch order, shape source or destination changed");
  return verifyFill(dyn_cast<gpu::GPUFuncOp>(fillModule.getBody()->front()), error) &&
         verifyGemm(dyn_cast<gpu::GPUFuncOp>(gemmModule.getBody()->front()), error);
}

GpuGemmArtifactV1 issueStrictGpuGemmArtifactV1(mlir::MLIRContext &context, TargetV1 target) {
  using namespace mlir;
  GpuGemmArtifactV1 artifact;
  if (llvm::StringRef(LLVM_VERSION_STRING) != "21.1.8" ||
      (target != TargetV1::NvvmSm89 && target != TargetV1::RocdlGfx1150)) {
    reject(artifact.error, "issuer requires exact 21.1.8 and an explicit reviewed target");
    return artifact;
  }
  // mlir-opt registers these external conversion interfaces globally. An
  // embedded compiler must register the exact payload owners explicitly;
  // loading a dialect alone does not supply its LLVM conversion patterns.
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
  auto stages = cpu_candidate::buildStrictGemmStagesV1(context);
  if (!stages) { artifact.error = stages.error; return artifact; }
  artifact.semantic_ir = print(*stages.semantic);
  artifact.structured_ir = print(*stages.structured);
  artifact.bufferized_ir = print(*stages.bufferized);
  auto outlined = mlir::cast<ModuleOp>(stages.bufferized->clone());
  OwningOpRef<ModuleOp> owner(outlined);
  PassManager schedule(&context);
  schedule.addPass(createConvertLinalgToParallelLoopsPass());
  schedule.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  schedule.addPass(createConvertParallelLoopToGpuPass());
  schedule.addPass(createCanonicalizerPass());
  schedule.addPass(createGpuLaunchSinkIndexComputationsPass());
  schedule.addPass(createGpuKernelOutliningPass());
  schedule.addPass(createCanonicalizerPass());
  if (failed(schedule.run(outlined)) || !verifyStrictGpuGemmOutlinedV1(outlined, artifact.error)) {
    if (artifact.error.empty()) reject(artifact.error, "structured GPU scheduling failed");
    return artifact;
  }
  artifact.outlined_ir = print(outlined);
  PassManager lower(&context);
  auto &device = lower.nest<gpu::GPUModuleOp>();
  device.addPass(createLowerAffinePass());
  device.addPass(createSCFToControlFlowPass());
  if (target == TargetV1::NvvmSm89) {
    ConvertGpuOpsToNVVMOpsOptions options;
    options.indexBitwidth = 64;
    device.addPass(createConvertGpuOpsToNVVMOps(options));
  } else {
    device.addPass(createLowerGpuOpsToROCDLOpsPass("gfx1150", 64, false, gpu::amd::Runtime::HIP));
  }
  device.addPass(createCanonicalizerPass());
  device.addPass(createReconcileUnrealizedCastsPass());
  if (failed(lower.run(outlined))) {
    reject(artifact.error, target == TargetV1::NvvmSm89 ? "NVVM conversion failed" : "ROCDL conversion failed");
    return artifact;
  }
  auto fill = outlined.lookupSymbol<gpu::GPUModuleOp>(kGemmKernelV1);
  auto gemm = outlined.lookupSymbol<gpu::GPUModuleOp>(std::string(kGemmKernelV1) + "_0");
  artifact.fill_llvm_ir = exportDevice(fill, target, 7, artifact.error);
  if (artifact.fill_llvm_ir.empty()) return artifact;
  artifact.gemm_llvm_ir = exportDevice(gemm, target, 21, artifact.error);
  if (artifact.gemm_llvm_ir.empty()) return artifact;
  artifact.manifest = "matcore-gpu-strict-gemm-v1\ncompiler=" LLVM_VERSION_STRING "\n";
  artifact.manifest += target == TargetV1::NvvmSm89 ? "target=nvvm-sm89\n" : "target=rocdl-gfx1150\n";
  artifact.manifest += "semantics=positive-zero;increasing-k;separate-f32;ieee-denormals\n"
      "schedule=two-ordered-kernels;one-block-per-output;one-thread-per-block\n"
      "machine-contract=fp-contract-off;ieee-denormals;nvvm-ptxas-fmad-false\n"
      "runtime-obligations=shape-and-grid-guards;explicit-transfer;private-output;checked-completion\n"
      "imported-artifact-authority=none\n";
  artifact.manifest += "semantic-sha256=" + hash(artifact.semantic_ir) + "\n";
  artifact.manifest += "structured-sha256=" + hash(artifact.structured_ir) + "\n";
  artifact.manifest += "outlined-sha256=" + hash(artifact.outlined_ir) + "\n";
  artifact.manifest += "fill-llvm-sha256=" + hash(artifact.fill_llvm_ir) + "\n";
  artifact.manifest += "gemm-llvm-sha256=" + hash(artifact.gemm_llvm_ir) + "\n";
  return artifact;
}
} // namespace matcore::mdslc::gpu_candidate
