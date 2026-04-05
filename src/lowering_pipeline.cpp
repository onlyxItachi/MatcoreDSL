#include "matcore/lowering_pipeline.h"

#include "fp8_wgmma.h"
#include "gpu_amd_lowering.h"
#include "gpu_data_staging.h"
#include "matcore/cpu_lowering.h"
#include "matcore/diagnostics.h"
#include "matcore/gpu_mapping.h"
#include "matcore/gpu_nvvm_lowering.h"
#include "matcore/gpu_tiling.h"
#include "matcore/observability.h"
#include "matcore/target_registry.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore lowering pipeline: " + message);
}

std::string dtypeName(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return "float32";
    case TensorDType::kFloat16:
      return "float16";
    case TensorDType::kBFloat16:
      return "bfloat16";
    case TensorDType::kInt8:
      return "int8";
    case TensorDType::kInt32:
      return "int32";
    case TensorDType::kFloat8E4M3FN:
      return "float8_e4m3fn";
  }
  return "unknown";
}

std::string signatureName(const MatmulLoweringSignature &signature) {
  return "lhs=" + dtypeName(signature.lhs_dtype) +
         ", rhs=" + dtypeName(signature.rhs_dtype) +
         ", out=" + dtypeName(signature.out_dtype);
}

bool usesFp8Operands(const MatmulLoweringSignature &signature) {
  return signature.lhs_dtype == TensorDType::kFloat8E4M3FN ||
         signature.rhs_dtype == TensorDType::kFloat8E4M3FN;
}

struct MatmulDims {
  int m = -1;
  int n = -1;
  int k = -1;
};

MatmulDims inferMatmulDimsFromModule(mlir::ModuleOp module) {
  MatmulDims dims;
  auto m_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_m");
  auto n_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_n");
  auto k_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_k");
  if (m_attr && n_attr && k_attr) {
    dims.m = static_cast<int>(m_attr.getInt());
    dims.n = static_cast<int>(n_attr.getInt());
    dims.k = static_cast<int>(k_attr.getInt());
    return dims;
  }
  mlir::Operation *first_matmul = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (first_matmul != nullptr) {
      return;
    }
    if (llvm::isa<mlir::linalg::MatmulOp>(op)) {
      first_matmul = op;
    }
  });
  if (first_matmul == nullptr) {
    return dims;
  }
  auto linalg_op = llvm::cast<mlir::linalg::LinalgOp>(first_matmul);
  llvm::SmallVector<mlir::Value> inputs = linalg_op.getDpsInputs();
  if (inputs.size() < 2) {
    return dims;
  }
  auto lhs_type = llvm::dyn_cast<mlir::ShapedType>(inputs[0].getType());
  auto rhs_type = llvm::dyn_cast<mlir::ShapedType>(inputs[1].getType());
  if (!lhs_type || !rhs_type || !lhs_type.hasRank() || !rhs_type.hasRank() ||
      lhs_type.getRank() != 2 || rhs_type.getRank() != 2) {
    return dims;
  }
  const auto toInt = [](std::int64_t dim) -> int {
    return dim == mlir::ShapedType::kDynamic ? -1 : static_cast<int>(dim);
  };
  dims.m = toInt(lhs_type.getDimSize(0));
  dims.k = toInt(lhs_type.getDimSize(1));
  dims.n = toInt(rhs_type.getDimSize(1));
  return dims;
}

MatmulDims inferMatmulDims(const MatmulLoweringSignature &signature,
                           mlir::ModuleOp module = mlir::ModuleOp()) {
  if (signature.matmul_m > 0 && signature.matmul_n > 0 && signature.matmul_k > 0) {
    return {.m = signature.matmul_m, .n = signature.matmul_n, .k = signature.matmul_k};
  }
  if (module) {
    return inferMatmulDimsFromModule(module);
  }
  return {};
}

bool isFp8TargetCapable(const LoweringPlan &plan,
                        const MatmulLoweringSignature &signature) {
  return plan.route == LoweringRoute::kNvidiaNvptx &&
         normalizeTarget(signature.target_kind) == TargetKind::kNvidiaDGPU &&
         signature.nvidia_sm_major >= 9;
}

std::string fp8WgmmaIneligibilityReason(const LoweringPlan &plan,
                                        const MatmulLoweringSignature &signature,
                                        const MatmulDims &dims) {
  if (!usesFp8Operands(signature)) {
    return {};
  }
  if (signature.out_dtype != TensorDType::kFloat32) {
    return "float8_e4m3fn matmul requires float32 output/accumulation for "
           "MLIR 18.1.3 FP8 WGMMA";
  }
  if (!isFp8TargetCapable(plan, signature) &&
      (plan.route != LoweringRoute::kNvidiaNvptx ||
       normalizeTarget(signature.target_kind) != TargetKind::kNvidiaDGPU)) {
    return "float8_e4m3fn matmul is currently limited to nvidia-dgpu";
  }
  if (signature.nvidia_sm_major < 9) {
    return "float8_e4m3fn matmul requires native NVIDIA FP8 tensor-core "
           "support (sm_90+ WGMMA); request nvidia-dgpu:sm_90 or newer";
  }
  if (!isLegalFp8WgmmaShape(dims.m, dims.n, dims.k)) {
    return "float8_e4m3fn matmul is not eligible for NVIDIA FP8 WGMMA: "
           "requires static positive shapes with M multiple of 64, K multiple "
           "of 32, and N in [8..256] step 8";
  }
  MatmulLoweringSignature shaped_signature = signature;
  shaped_signature.matmul_m = dims.m;
  shaped_signature.matmul_n = dims.n;
  shaped_signature.matmul_k = dims.k;
  if (!isEligibleForFp8Wgmma(shaped_signature)) {
    return "float8_e4m3fn matmul is not eligible for NVIDIA FP8 WGMMA";
  }
  return {};
}

std::string normalizeFailureMessage(const std::string &message) {
  constexpr llvm::StringLiteral kPrefix = "MatCore lowering pipeline: ";
  if (llvm::StringRef(message).starts_with(kPrefix)) {
    return message.substr(kPrefix.size());
  }
  return message;
}

std::string diagnosticSeverityName(mlir::DiagnosticSeverity severity) {
  switch (severity) {
    case mlir::DiagnosticSeverity::Error:
      return "error";
    case mlir::DiagnosticSeverity::Warning:
      return "warning";
    case mlir::DiagnosticSeverity::Remark:
      return "remark";
    case mlir::DiagnosticSeverity::Note:
      return "note";
  }
  return "error";
}

std::string captureIrForDiagnostics(mlir::ModuleOp module) {
  std::string ir;
  if (!module) {
    return ir;
  }
  llvm::raw_string_ostream stream(ir);
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm().elideLargeElementsAttrs();
  module.print(stream, flags);
  stream.flush();
  return ir;
}

class FailedPassCaptureInstrumentation final : public mlir::PassInstrumentation {
 public:
  explicit FailedPassCaptureInstrumentation(std::string *failing_pass)
      : failing_pass_(failing_pass) {}

  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *) override {
    if (failing_pass_ == nullptr || pass == nullptr || !failing_pass_->empty()) {
      return;
    }
    *failing_pass_ = pass->getArgument().str();
    if (failing_pass_->empty()) {
      *failing_pass_ = pass->getName().str();
    }
  }

 private:
  std::string *failing_pass_ = nullptr;
};

TensorDType decodeTensorDType(mlir::Type type) {
  if (type.isF32()) {
    return TensorDType::kFloat32;
  }
  if (type.isF16()) {
    return TensorDType::kFloat16;
  }
  if (type.isBF16()) {
    return TensorDType::kBFloat16;
  }
  if (type.isInteger(8)) {
    return TensorDType::kInt8;
  }
  if (type.isInteger(32)) {
    return TensorDType::kInt32;
  }
  if (type.isFloat8E4M3FN()) {
    return TensorDType::kFloat8E4M3FN;
  }
  fail("module carries unsupported dtype attribute");
}

LoweringRoute selectRoute(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
      return LoweringRoute::kCpuVector;
    case TargetKind::kNvidiaDGPU:
      return LoweringRoute::kNvidiaNvptx;
    case TargetKind::kAmdIGPU:
      return LoweringRoute::kAmdRocdl;
    case TargetKind::kAmdNPU:
      return LoweringRoute::kAmdNpuScaffold;
    case TargetKind::kARM:
      fail("ARM route exists but is not implemented in Phase 2");
    case TargetKind::kTPU:
      fail("TPU route exists but is not implemented in Phase 2");
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }
  fail("unsupported target route");
}

void addGpuCommonModulePasses(mlir::PassManager &pm, std::int64_t index_bitwidth) {
  mlir::ConvertIndexToLLVMPassOptions index_to_llvm_opts;
  index_to_llvm_opts.indexBitwidth = index_bitwidth;

  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass(index_to_llvm_opts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void addGpuHostPostPasses(mlir::PassManager &pm,
                          const std::string &binary_target,
                          const std::string &toolkit_path = {}) {
  mlir::GpuToLLVMConversionPassOptions gpu_to_llvm_opts;
  gpu_to_llvm_opts.hostBarePtrCallConv = false;
  gpu_to_llvm_opts.kernelBarePtrCallConv = false;
  pm.addPass(mlir::createGpuToLLVMConversionPass(gpu_to_llvm_opts));

  mlir::GpuModuleToBinaryPassOptions binary_opts;
  binary_opts.compilationTarget = binary_target;
  binary_opts.toolkitPath = toolkit_path;
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary_opts));

  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

std::string rocdlToolkitPath() {
  constexpr std::string_view kPrimary = "/usr/lib/llvm-18/lib/clang/18";
  if (std::filesystem::exists(kPrimary)) {
    return std::string(kPrimary);
  }
  return {};
}

std::string requestedAmdChip(mlir::ModuleOp module) {
  auto chip_from_target_attr = [&](llvm::StringRef attr_name) -> std::string {
    auto attr = module->getAttrOfType<mlir::StringAttr>(attr_name);
    if (!attr) {
      return {};
    }
    const std::string target = attr.getValue().str();
    const std::size_t split = target.find(':');
    if (split == std::string::npos || split + 1 >= target.size()) {
      return {};
    }
    return target.substr(split + 1);
  };

  auto chip_attr = module->getAttrOfType<mlir::StringAttr>("matcore.amd_chip");
  if (chip_attr && !chip_attr.getValue().empty()) {
    return chip_attr.getValue().str();
  }
  if (std::string chip = chip_from_target_attr("matcore.requested_target_raw");
      !chip.empty()) {
    return chip;
  }
  if (std::string chip = chip_from_target_attr("matcore.requested_target");
      !chip.empty()) {
    return chip;
  }
  return "gfx90a";
}

void runFp8WgmmaPreflight(mlir::ModuleOp module, const LoweringPlan &plan,
                          const MatmulLoweringSignature &signature,
                          ObservabilityContext *obs) {
  if (!usesFp8Operands(signature) || plan.route != LoweringRoute::kNvidiaNvptx) {
    return;
  }

  const MatmulDims dims = inferMatmulDims(signature, module);
  const Fp8WgmmaConfig config =
      getFp8WgmmaTileConfig(dims.m, dims.n, dims.k);
  const std::string ineligible_reason =
      fp8WgmmaIneligibilityReason(plan, signature, dims);
  const bool eligible = ineligible_reason.empty();
  if (obs != nullptr) {
    std::string details;
    const bool legal_shape = isLegalFp8WgmmaShape(dims.m, dims.n, dims.k);
    details += "eligible=" + std::string(eligible ? "true" : "false") + "\n";
    details +=
        "reason=" +
        (eligible ? "eligible (infrastructure-only path)" : ineligible_reason) +
        "\n";
    details += "shape_m=" + std::to_string(dims.m) + "\n";
    details += "shape_n=" + std::to_string(dims.n) + "\n";
    details += "shape_k=" + std::to_string(dims.k) + "\n";
    details += "tile_config_status=" +
               std::string(legal_shape ? "shape_aware" : "provisional_illegal_shape") +
               "\n";
    details += "tile_m=" + std::to_string(config.M_tile) + "\n";
    details += "tile_n=" + std::to_string(config.N_tile) + "\n";
    details += "tile_k=" + std::to_string(config.K_tile) + "\n";
    details += "use_tma=" + std::string(config.use_tma ? "true" : "false") + "\n";
    details += "sm=" + std::to_string(signature.nvidia_sm_major) + "." +
               std::to_string(signature.nvidia_sm_minor) + "\n";
    obs->snapshotText("fp8_wgmma_preflight", details);
  }
  if (!eligible) {
    fail(ineligible_reason);
  }
  fail("float8_e4m3fn matmul is eligible for NVIDIA FP8 WGMMA on sm_90+, "
       "but MatCore does not implement that path yet (TODO: custom "
       "linalg.matmul -> nvgpu.warpgroup.mma rewrite)");
}

NvidiaMappingConfig selectNvidiaMappingForModule(
    mlir::ModuleOp module, const MatmulLoweringSignature &signature) {
  mlir::Operation *first_matmul = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (first_matmul != nullptr) {
      return;
    }
    if (!llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::QuantizedMatmulOp>(op)) {
      return;
    }
    first_matmul = op;
  });
  if (first_matmul == nullptr) {
    fail("NVIDIA lowering expected a linalg matmul op before transform");
  }
  return SelectNvidiaMappingConfig(
      llvm::cast<mlir::linalg::LinalgOp>(first_matmul), signature);
}

}  // namespace

LoweringPlan selectLoweringPlan(TargetKind target) {
  LoweringPlan plan;
  plan.route = selectRoute(target);
  plan.route_name = routeName(plan.route);
  plan.route_description = routeDescription(plan.route);
  plan.executable = plan.route != LoweringRoute::kAmdNpuScaffold;
  return plan;
}

const char *routeName(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      return "x86-vector";
    case LoweringRoute::kNvidiaNvptx:
      return "nvidia-dgpu";
    case LoweringRoute::kAmdRocdl:
      return "amd-igpu";
    case LoweringRoute::kAmdNpuScaffold:
      return "amd-npu";
  }
  return "unknown";
}

std::string routeDescription(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      return "func+memref+linalg.matmul -> loops/vector -> llvm(x86vector)";
    case LoweringRoute::kNvidiaNvptx:
      return "linalg.matmul -> tiled gpu mapping -> workgroup promotion -> mma sync -> gpu.launch -> nvvm -> llvm";
    case LoweringRoute::kAmdRocdl:
      return "linalg -> scf.parallel -> gpu.launch -> rocdl -> llvm";
    case LoweringRoute::kAmdNpuScaffold:
      return "aie/xdna scaffold route (external toolchain required)";
  }
  return "unknown";
}

MatmulLoweringSignature decodeMatmulSignatureFromModule(mlir::ModuleOp module) {
  MatmulLoweringSignature signature;
  auto lhs_attr = module->getAttrOfType<mlir::TypeAttr>("matcore.lhs_dtype");
  auto rhs_attr = module->getAttrOfType<mlir::TypeAttr>("matcore.rhs_dtype");
  auto out_attr = module->getAttrOfType<mlir::TypeAttr>("matcore.out_dtype");
  auto requested_target_attr =
      module->getAttrOfType<mlir::StringAttr>("matcore.requested_target");
  auto nvidia_chip_attr =
      module->getAttrOfType<mlir::StringAttr>("matcore.nvidia_chip");
  auto target_attr =
      module->getAttrOfType<mlir::IntegerAttr>("matcore.target_kind");
  auto sm_major_attr =
      module->getAttrOfType<mlir::IntegerAttr>("matcore.nvidia_sm_major");
  auto sm_minor_attr =
      module->getAttrOfType<mlir::IntegerAttr>("matcore.nvidia_sm_minor");
  auto m_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_m");
  auto n_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_n");
  auto k_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.matmul_k");
  if (lhs_attr && rhs_attr && out_attr) {
    signature.lhs_dtype = decodeTensorDType(lhs_attr.getValue());
    signature.rhs_dtype = decodeTensorDType(rhs_attr.getValue());
    signature.out_dtype = decodeTensorDType(out_attr.getValue());
    signature.quantized_i8 =
        signature.lhs_dtype == TensorDType::kInt8 &&
        signature.out_dtype == TensorDType::kInt32;
  }
  if (target_attr) {
    const std::int64_t target_value = target_attr.getInt();
    if (target_value >= static_cast<std::int64_t>(TargetKind::kX86Auto) &&
        target_value <= static_cast<std::int64_t>(TargetKind::kTPU)) {
      signature.target_kind = static_cast<TargetKind>(target_value);
    }
  }
  if (requested_target_attr) {
    const RequestedTargetProfile profile =
        ParseRequestedTargetProfile(requested_target_attr.getValue().str());
    signature.target_kind = profile.kind;
    signature.nvidia_sm_major = profile.nvidia_sm_major.value_or(0);
    signature.nvidia_sm_minor = profile.nvidia_sm_minor.value_or(0);
  }
  if (nvidia_chip_attr &&
      (signature.nvidia_sm_major == 0 || signature.nvidia_sm_minor == 0)) {
    const RequestedTargetProfile chip_profile =
        ParseRequestedTargetProfile(nvidia_chip_attr.getValue().str());
    if (chip_profile.nvidia_sm_major.has_value() &&
        chip_profile.nvidia_sm_minor.has_value()) {
      signature.target_kind = TargetKind::kNvidiaDGPU;
      signature.nvidia_sm_major = *chip_profile.nvidia_sm_major;
      signature.nvidia_sm_minor = *chip_profile.nvidia_sm_minor;
    }
  }
  if (sm_major_attr) {
    signature.nvidia_sm_major = sm_major_attr.getInt();
  }
  if (sm_minor_attr) {
    signature.nvidia_sm_minor = sm_minor_attr.getInt();
  }
  if (m_attr) {
    signature.matmul_m = m_attr.getInt();
  }
  if (n_attr) {
    signature.matmul_n = n_attr.getInt();
  }
  if (k_attr) {
    signature.matmul_k = k_attr.getInt();
  }
  return signature;
}

void configureLoweringPipeline(mlir::PassManager &pm, const LoweringPlan &plan,
                               const MatmulLoweringSignature &signature,
                               llvm::StringRef nvidia_chip,
                               llvm::StringRef amd_chip,
                               mlir::ModuleOp module,
                               ObservabilityContext *obs) {
  if (obs) {
    attachObservability(pm, obs, std::string(routeName(plan.route)));
  }

  switch (plan.route) {
    case LoweringRoute::kCpuVector:
      configureCpuPassPipeline(pm, signature);
      return;
    case LoweringRoute::kNvidiaNvptx:
      ConfigureNvidiaGenericGpuStage(pm);
      ConfigureNvidiaNvvmStage(pm, nvidia_chip, obs);
      return;
    case LoweringRoute::kAmdRocdl: {
      const std::string resolved_amd_chip =
          amd_chip.empty() ? requestedAmdChip(module) : amd_chip.str();
      const AmdGpuConfig amd_config = detectAmdGpuConfig(resolved_amd_chip);
      configureAmdLoweringPipeline(pm, amd_config);
      addGpuCommonModulePasses(pm, /*index_bitwidth=*/64);
      addGpuHostPostPasses(pm, "fatbin", rocdlToolkitPath());
      return;
    }
    case LoweringRoute::kAmdNpuScaffold:
      fail("amd-npu lowering remains unavailable without an external AIE/XDNA toolchain");
      return;
  }
}

void runLoweringPipeline(mlir::ModuleOp module, const LoweringPlan &plan,
                         const MatmulLoweringSignature &signature,
                         llvm::StringRef nvidia_chip,
                         llvm::StringRef amd_chip,
                         ObservabilityContext *obs) {
  runFp8WgmmaPreflight(module, plan, signature, obs);
  const std::string resolved_amd_chip =
      amd_chip.empty() ? requestedAmdChip(module) : amd_chip.str();
  auto create_stage_trace = [&](const std::string &stage_name) {
    if (obs == nullptr) {
      return std::unique_ptr<ObservabilityContext::TraceScope>();
    }
    return std::make_unique<ObservabilityContext::TraceScope>(
        *obs, TraceEventKind::kPassStageStart, TraceEventKind::kPassStageEnd,
        stage_name);
  };
  int stage_counter = 0;
  auto nextStageIndex = [&]() { return ++stage_counter; };
  const std::string route_name = routeName(plan.route);
  const std::string dtype_signature = signatureName(signature);
  const std::string target_profile = [&]() -> std::string {
    if (plan.route == LoweringRoute::kNvidiaNvptx) {
      return nvidia_chip.str();
    }
    if (plan.route == LoweringRoute::kAmdRocdl) {
      return resolved_amd_chip;
    }
    return {};
  }();
  auto fail_with_report = [&](const std::string &stage_name, int stage_index,
                              const std::string &raw_diagnostics,
                              const std::string &pass_name = {},
                              std::vector<CapturedDiagnostic> captured = {},
                              const std::string &ir_before = {}) -> void {
    for (CapturedDiagnostic &entry : captured) {
      if (entry.pass_name.empty()) {
        entry.pass_name = pass_name.empty() ? stage_name : pass_name;
      }
    }
    StructuredDiagnosticReport report = buildDiagnosticReport(
        route_name, stage_name, stage_index, raw_diagnostics, module,
        target_profile, dtype_signature, captured, ir_before);
    if (obs != nullptr) {
      obs->snapshotText(stage_name + "_diagnostic",
                        formatDiagnosticReportJson(report), ".json");
    }
    fail(formatDiagnosticReport(report));
  };
  auto run_stage = [&](llvm::StringRef stage_name, auto &&configure_stage) {
    const std::string stage = stage_name.str();
    [[maybe_unused]] auto stage_trace = create_stage_trace(stage);
    const int stage_index = nextStageIndex();
    const std::string ir_before = captureIrForDiagnostics(module);
    std::string failing_pass_name;
    mlir::PassManager pm(module.getContext());
    pm.addInstrumentation(
        std::make_unique<FailedPassCaptureInstrumentation>(&failing_pass_name));
    if (obs) {
      attachObservability(pm, obs, stage);
    }
    configure_stage(pm);
    std::string diagnostics;
    std::vector<CapturedDiagnostic> captured_diagnostics;
    mlir::ScopedDiagnosticHandler diag_handler(
        module.getContext(), [&](mlir::Diagnostic &diag) {
          CapturedDiagnostic captured;
          captured.severity = diagnosticSeverityName(diag.getSeverity());
          captured.pass_name = failing_pass_name;
          llvm::raw_string_ostream message_stream(captured.message);
          diag.print(message_stream);
          message_stream.flush();
          captured_diagnostics.push_back(captured);
          llvm::raw_string_ostream stream(diagnostics);
          diag.print(stream);
          stream << '\n';
          stream.flush();
          return mlir::success();
        });
    if (mlir::failed(pm.run(module))) {
      fail_with_report(stage, stage_index, diagnostics, failing_pass_name,
                       std::move(captured_diagnostics), ir_before);
    }
  };

  if (plan.route == LoweringRoute::kNvidiaNvptx) {
    run_stage("nvidia-tensor-bufferize", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
      pm.addPass(mlir::bufferization::createOneShotBufferizePass());
      pm.addPass(mlir::createBufferizationToMemRefPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    });
    const int mapping_stage_index = nextStageIndex();
    [[maybe_unused]] auto mapping_trace =
        create_stage_trace("nvidia-select-mapping");
    const std::string mapping_ir_before = captureIrForDiagnostics(module);
    NvidiaMappingConfig mapping = [&]() -> NvidiaMappingConfig {
      try {
        return selectNvidiaMappingForModule(module, signature);
      } catch (const std::exception &exc) {
        std::vector<CapturedDiagnostic> captured = {{
            .pass_name = "nvidia-select-mapping",
            .severity = "error",
            .message = normalizeFailureMessage(exc.what()),
        }};
        fail_with_report("nvidia-select-mapping", mapping_stage_index,
                         captured[0].message, "nvidia-select-mapping",
                         std::move(captured), mapping_ir_before);
      }
      llvm_unreachable("nvidia-select-mapping should fail through fail_with_report");
    }();
    if (obs) {
      obs->snapshot("nvidia-apply-transform_pre", module);
    }
    [[maybe_unused]] auto apply_transform_trace =
        create_stage_trace("nvidia-apply-transform");
    const int stage_index = nextStageIndex();
    const std::string ir_before = captureIrForDiagnostics(module);
    try {
      ApplyNvidiaMmaTransformToModule(module, signature, mapping);
    } catch (const std::exception &exc) {
      std::string normalized = normalizeFailureMessage(exc.what());
      std::vector<CapturedDiagnostic> captured = {{
          .pass_name = "nvidia-apply-transform",
          .severity = "error",
          .message = normalized,
      }};
      fail_with_report("nvidia-apply-transform", stage_index,
                       normalized, "nvidia-apply-transform",
                       std::move(captured), ir_before);
    }
    if (obs) {
      obs->snapshot("nvidia-apply-transform_post", module);
    }
    run_stage("nvidia-dynamic-macro-topology", [&](mlir::PassManager &pm) {
      AddNvidiaDynamicMacroGridMappingPasses(pm, mapping);
    });

    if (mapping.rewrite_to_mma_sync) {
      run_stage("nvidia-post-transform-canonicalize",
                [&](mlir::PassManager &pm) {
                  pm.addPass(mlir::createCanonicalizerPass());
                  pm.addPass(mlir::createCSEPass());
                });
      run_stage("nvidia-mma-preparation", [&](mlir::PassManager &pm) {
        AddNvidiaMmaPreparationPasses(pm);
      });
      if (obs) {
        obs->snapshot("nvidia-rewrite-mma-sync_pre", module);
      }
      [[maybe_unused]] auto rewrite_trace =
          create_stage_trace("nvidia-rewrite-mma-sync");
      const int stage_index = nextStageIndex();
      const std::string ir_before = captureIrForDiagnostics(module);
      try {
        ApplyNvidiaMmaRewriteToModule(module);
      } catch (const std::exception &exc) {
        std::string normalized = normalizeFailureMessage(exc.what());
        std::vector<CapturedDiagnostic> captured = {{
            .pass_name = "nvidia-rewrite-mma-sync",
            .severity = "error",
            .message = normalized,
        }};
        fail_with_report("nvidia-rewrite-mma-sync", stage_index,
                         normalized, "nvidia-rewrite-mma-sync",
                         std::move(captured), ir_before);
      }
      if (obs) {
        obs->snapshot("nvidia-rewrite-mma-sync_post", module);
      }
      [[maybe_unused]] auto verify_trace =
          create_stage_trace("nvidia-verify-no-residual-matmul");
      const int verify_stage_index = nextStageIndex();
      const std::string verify_ir_before = captureIrForDiagnostics(module);
      try {
        VerifyNoResidualNvidiaMatmulOnModule(module);
      } catch (const std::exception &exc) {
        std::string normalized = normalizeFailureMessage(exc.what());
        std::vector<CapturedDiagnostic> captured = {{
            .pass_name = "nvidia-verify-no-residual-matmul",
            .severity = "error",
            .message = normalized,
        }};
        fail_with_report("nvidia-verify-no-residual-matmul", verify_stage_index,
                         normalized, "nvidia-verify-no-residual-matmul",
                         std::move(captured), verify_ir_before);
      }
      run_stage("nvidia-launch-config", [&](mlir::PassManager &pm) {
        AddNvidiaLaunchConfigurationPasses(pm);
      });
      run_stage("nvidia-loop-materialization", [&](mlir::PassManager &pm) {
        AddNvidiaLoopMaterializationPasses(pm);
      });
      run_stage("nvidia-vector-to-gpu", [&](mlir::PassManager &pm) {
        ConfigureNvidiaVectorToGpuStage(pm);
      });
    } else {
      if (obs) {
        obs->snapshot("nvidia-map-threads_pre", module);
      }
      [[maybe_unused]] auto map_threads_trace =
          create_stage_trace("nvidia-map-threads");
      const int stage_index = nextStageIndex();
      const std::string ir_before = captureIrForDiagnostics(module);
      try {
        ApplyNvidiaThreadMappingToModule(module, mapping);
      } catch (const std::exception &exc) {
        std::string normalized = normalizeFailureMessage(exc.what());
        std::vector<CapturedDiagnostic> captured = {{
            .pass_name = "nvidia-map-threads",
            .severity = "error",
            .message = normalized,
        }};
        fail_with_report("nvidia-map-threads", stage_index,
                         normalized, "nvidia-map-threads", std::move(captured),
                         ir_before);
      }
      if (obs) {
        obs->snapshot("nvidia-map-threads_post", module);
      }
      run_stage("nvidia-post-thread-map-canonicalize",
                [&](mlir::PassManager &pm) {
                  pm.addPass(mlir::createCanonicalizerPass());
                  pm.addPass(mlir::createCSEPass());
                });
      run_stage("nvidia-launch-config", [&](mlir::PassManager &pm) {
        AddNvidiaLaunchConfigurationPasses(pm);
      });
      run_stage("nvidia-loop-materialization", [&](mlir::PassManager &pm) {
        AddNvidiaLoopMaterializationPasses(pm);
      });
      run_stage("nvidia-vector-to-gpu", [&](mlir::PassManager &pm) {
        ConfigureNvidiaVectorToGpuStage(pm);
      });
    }
    run_stage("nvidia-gpu-data-staging", [&](mlir::PassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(CreateGpuDataStagingPass());
    });
    run_stage("nvidia-nvvm", [&](mlir::PassManager &pm) {
      ConfigureNvidiaNvvmStage(pm, nvidia_chip, obs);
    });
    return;
  }

  if (plan.route == LoweringRoute::kAmdRocdl) {
    const AmdGpuConfig amd_config = detectAmdGpuConfig(resolved_amd_chip);
    const std::string ineligible = amdIneligibilityReason(signature, amd_config);
    if (!ineligible.empty()) {
      fail("matmul signature " + signatureName(signature) +
           " is not supported on AMD chip '" + amd_config.chip + "': " + ineligible);
    }
  }

  mlir::PassManager pm(module.getContext());
  std::string failing_pass_name;
  pm.addInstrumentation(
      std::make_unique<FailedPassCaptureInstrumentation>(&failing_pass_name));
  if (obs) {
    obs->snapshot("lowering_pre", module);
  }
  configureLoweringPipeline(pm, plan, signature, nvidia_chip, resolved_amd_chip,
                            module, obs);
  std::string diagnostics;
  std::vector<CapturedDiagnostic> captured_diagnostics;
  const std::string ir_before = captureIrForDiagnostics(module);
  [[maybe_unused]] auto lowering_trace = create_stage_trace("lowering");
  mlir::ScopedDiagnosticHandler diag_handler(
      module.getContext(), [&](mlir::Diagnostic &diag) {
        CapturedDiagnostic captured;
        captured.severity = diagnosticSeverityName(diag.getSeverity());
        captured.pass_name = failing_pass_name;
        llvm::raw_string_ostream message_stream(captured.message);
        diag.print(message_stream);
        message_stream.flush();
        captured_diagnostics.push_back(captured);
        llvm::raw_string_ostream stream(diagnostics);
        diag.print(stream);
        stream << '\n';
        stream.flush();
        return mlir::success();
      });
  if (mlir::failed(pm.run(module))) {
    fail_with_report("lowering", nextStageIndex(), diagnostics, failing_pass_name,
                     std::move(captured_diagnostics), ir_before);
  }
  if (obs) {
    obs->snapshot("lowering_post", module);
  }
}

}  // namespace matcore
