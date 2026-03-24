#include "matcore/lowering_pipeline.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore lowering pipeline: " + message);
}

std::int64_t pickTilingFactor(std::int64_t dim, std::int64_t preferred) {
  if (preferred <= 0) {
    return 1;
  }
  if (dim <= 0) {
    return preferred;
  }

  std::int64_t tile = std::min(dim, preferred);
  while (tile > 1 && (dim % tile) != 0) {
    tile /= 2;
  }
  return std::max<std::int64_t>(tile, 1);
}

llvm::SmallVector<std::int64_t, 3> selectCpuMatmulTileSizes(
    mlir::linalg::MatmulOp matmul) {
  auto lhs_type = llvm::dyn_cast<mlir::ShapedType>(matmul.getInputs()[0].getType());
  auto rhs_type = llvm::dyn_cast<mlir::ShapedType>(matmul.getInputs()[1].getType());

  const bool low_precision = lhs_type && lhs_type.getElementType().isIntOrFloat() &&
                             (lhs_type.getElementType().isF16() ||
                              lhs_type.getElementType().isBF16());

  const std::int64_t preferred_m = low_precision ? 8 : 8;
  const std::int64_t preferred_n = low_precision ? 8 : 8;
  const std::int64_t preferred_k = low_precision ? 16 : 8;

  std::int64_t m = mlir::ShapedType::kDynamic;
  std::int64_t n = mlir::ShapedType::kDynamic;
  std::int64_t k = mlir::ShapedType::kDynamic;
  if (lhs_type && lhs_type.hasRank() && lhs_type.getRank() == 2) {
    m = lhs_type.getDimSize(0);
    k = lhs_type.getDimSize(1);
  }
  if (rhs_type && rhs_type.hasRank() && rhs_type.getRank() == 2) {
    n = rhs_type.getDimSize(1);
  }

  return {pickTilingFactor(m, preferred_m), pickTilingFactor(n, preferred_n),
          pickTilingFactor(k, preferred_k)};
}

mlir::Type getMatmulInputElementType(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<mlir::Value> inputs = op.getDpsInputs();
  if (inputs.empty()) {
    fail("matmul op is missing inputs");
  }
  auto shaped_type = llvm::dyn_cast<mlir::ShapedType>(inputs.front().getType());
  if (!shaped_type) {
    fail("matmul op input is not a shaped type");
  }
  return shaped_type.getElementType();
}

std::string nvidiaMatmulOpName(const MatmulLoweringSignature &signature) {
  if (signature.quantized_i8) {
    return "linalg.quantized_matmul";
  }
  return "linalg.matmul";
}

bool isLowPrecisionTensorType(TensorDType dtype) {
  return dtype == TensorDType::kFloat16 || dtype == TensorDType::kBFloat16;
}

bool isTensorCoreMmaSyncType(const MatmulLoweringSignature &signature) {
  return signature.lhs_dtype == TensorDType::kFloat16 &&
         signature.rhs_dtype == TensorDType::kFloat16 &&
         signature.out_dtype == TensorDType::kFloat16;
}

bool isWorkgroupMemorySpace(mlir::Attribute memory_space) {
  if (!memory_space) {
    return false;
  }
  if (auto gpu_space = llvm::dyn_cast<mlir::gpu::AddressSpaceAttr>(memory_space)) {
    return gpu_space.getValue() == mlir::gpu::AddressSpace::Workgroup;
  }
  if (auto int_space = llvm::dyn_cast<mlir::IntegerAttr>(memory_space)) {
    return int_space.getInt() ==
           static_cast<std::int64_t>(mlir::gpu::AddressSpace::Workgroup);
  }
  return false;
}

bool isWorkgroupMemRefType(mlir::MemRefType type) {
  return type && isWorkgroupMemorySpace(type.getMemorySpace());
}

std::optional<std::int64_t> matchConstantIndex(mlir::Value value) {
  llvm::APInt constant;
  if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant))) {
    return std::nullopt;
  }
  return constant.getSExtValue();
}

std::optional<mlir::MemRefType> inferStaticViewType(mlir::memref::ViewOp view) {
  auto view_type = llvm::dyn_cast<mlir::MemRefType>(view.getType());
  if (!view_type || !view_type.hasRank()) {
    return std::nullopt;
  }
  if (view_type.hasStaticShape()) {
    return view_type;
  }

  auto byte_shift = matchConstantIndex(view.getByteShift());
  if (!byte_shift.has_value() || *byte_shift != 0) {
    return std::nullopt;
  }

  llvm::SmallVector<std::int64_t, 4> shape;
  shape.reserve(view_type.getRank());
  for (mlir::Value size : view.getSizes()) {
    auto constant_size = matchConstantIndex(size);
    if (!constant_size.has_value() || *constant_size < 0) {
      return std::nullopt;
    }
    shape.push_back(*constant_size);
  }

  return mlir::MemRefType::get(shape, view_type.getElementType(),
                               mlir::AffineMap(), view_type.getMemorySpace());
}

mlir::BlockArgument addLaunchWorkgroupAttribution(mlir::gpu::LaunchOp launch,
                                                  mlir::Type type,
                                                  mlir::Location loc) {
  mlir::Block &body = launch.getBody().front();
  const unsigned index = launch.getNumConfigRegionAttributes() +
                         launch.getNumWorkgroupAttributions();
  mlir::BlockArgument argument =
      body.insertArgument(index, type, loc);

  auto existing_count = launch->getAttrOfType<mlir::IntegerAttr>(
      mlir::gpu::LaunchOp::getNumWorkgroupAttributionsAttrName());
  const std::int64_t next_count =
      (existing_count ? existing_count.getInt() : 0) + 1;
  auto count_type = existing_count
                        ? existing_count.getType()
                        : mlir::IntegerType::get(launch.getContext(), 64);
  launch->setAttr(mlir::gpu::LaunchOp::getNumWorkgroupAttributionsAttrName(),
                  mlir::IntegerAttr::get(count_type, next_count));
  return argument;
}

struct NvidiaTileConfig {
  std::int64_t grid_y = 1;
  std::int64_t grid_x = 1;
  std::int64_t block_tile_m = 128;
  std::int64_t block_tile_n = 128;
  std::int64_t thread_tile_m = 16;
  std::int64_t thread_tile_n = 8;
  std::int64_t block_threads_y = 8;
  std::int64_t block_threads_x = 16;
  std::int64_t k_tile = 16;
  bool rewrite_to_mma_sync = true;
};

std::int64_t ceilDiv(std::int64_t lhs, std::int64_t rhs) {
  if (rhs <= 0) {
    return 1;
  }
  if (lhs <= 0) {
    return 1;
  }
  return (lhs + rhs - 1) / rhs;
}

NvidiaTileConfig selectNvidiaTileConfig(
    mlir::linalg::LinalgOp op, const MatmulLoweringSignature &signature) {
  llvm::SmallVector<mlir::Value> inputs = op.getDpsInputs();
  auto lhs_type = inputs.size() > 0
                      ? llvm::dyn_cast<mlir::ShapedType>(inputs[0].getType())
                      : mlir::ShapedType();
  auto rhs_type = inputs.size() > 1
                      ? llvm::dyn_cast<mlir::ShapedType>(inputs[1].getType())
                      : mlir::ShapedType();

  std::int64_t m = mlir::ShapedType::kDynamic;
  std::int64_t n = mlir::ShapedType::kDynamic;
  std::int64_t k = mlir::ShapedType::kDynamic;
  if (lhs_type && lhs_type.hasRank() && lhs_type.getRank() == 2) {
    m = lhs_type.getDimSize(0);
    k = lhs_type.getDimSize(1);
  }
  if (rhs_type && rhs_type.hasRank() && rhs_type.getRank() == 2) {
    n = rhs_type.getDimSize(1);
  }

  NvidiaTileConfig config;
  config.block_tile_m = pickTilingFactor(m, 128);
  config.block_tile_n = pickTilingFactor(n, 128);
  config.grid_y = ceilDiv(m, config.block_tile_m);
  config.grid_x = ceilDiv(n, config.block_tile_n);
  config.k_tile = pickTilingFactor(k, signature.quantized_i8 ? 32 : 16);

  const bool statically_compatible_mma =
      m != mlir::ShapedType::kDynamic && n != mlir::ShapedType::kDynamic &&
      k != mlir::ShapedType::kDynamic && m >= 16 && n >= 8 && k >= 16 &&
      (m % 16) == 0 && (n % 8) == 0 && (k % 16) == 0 &&
      isTensorCoreMmaSyncType(signature);
  config.rewrite_to_mma_sync =
      !signature.quantized_i8 && statically_compatible_mma;
  if (config.rewrite_to_mma_sync) {
    config.block_tile_m = 16;
    config.block_tile_n = 8;
    config.thread_tile_m = 16;
    config.thread_tile_n = 8;
    config.block_threads_y = 1;
    config.block_threads_x = 32;
    config.grid_y = ceilDiv(m, config.block_tile_m);
    config.grid_x = ceilDiv(n, config.block_tile_n);
    config.k_tile = 16;
    return config;
  }
  if (!config.rewrite_to_mma_sync) {
    config.thread_tile_m = std::max<std::int64_t>(
        1, pickTilingFactor(config.block_tile_m, 16));
    config.thread_tile_n = std::max<std::int64_t>(
        1, pickTilingFactor(config.block_tile_n, 8));
  }
  config.block_threads_y = std::max<std::int64_t>(
      1, ceilDiv(config.block_tile_m, config.thread_tile_m));
  config.block_threads_x = std::max<std::int64_t>(
      1, ceilDiv(config.block_tile_n, config.thread_tile_n));
  return config;
}

void registerTransformDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::transform::TransformDialect>();
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::nvgpu::registerTransformDialectExtension(registry);
  mlir::transform::registerDebugExtension(registry);
}

std::string buildNvidiaTransformSequence(
    const MatmulLoweringSignature &signature, const NvidiaTileConfig &config) {
  const std::string op_name = nvidiaMatmulOpName(signature);
  std::ostringstream ir;
  ir << "module attributes {transform.with_named_sequence} {\n";
  ir << "  transform.named_sequence @__transform_main"
        "(%root: !transform.any_op) {\n";
  ir << "    %func = transform.structured.match ops{[\"func.func\"]} in %root"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    %matmul = transform.structured.match ops{[\"" << op_name
     << "\"]} in %func : (!transform.any_op) -> !transform.any_op\n";
  ir << "    %block_tiled, %block_forall = transform.structured.tile_using_forall"
     << " %matmul tile_sizes [" << config.block_tile_m << ", "
     << config.block_tile_n
     << ", 0](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op)"
        " -> (!transform.any_op, !transform.any_op)\n";
  ir << "    %k_tiled, %k_loop = transform.structured.tile_using_for"
     << " %block_tiled [0, 0, " << config.k_tile
     << "] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n";
  ir << "    %promoted = transform.structured.promote %k_tiled"
        " {operands_to_promote = [0, 1], use_full_tiles_by_default,"
        " memory_space = #gpu.address_space<workgroup>} :"
        " (!transform.any_op) -> !transform.any_op\n";
  if (config.rewrite_to_mma_sync) {
    ir << "    %launch = transform.gpu.map_forall_to_blocks %func"
          " generate_gpu_launch grid_dims = [" << config.grid_x << ", "
       << config.grid_y << ", 1] : (!transform.any_op)"
          " -> !transform.any_op\n";
  } else {
    ir << "    %warp_tiled, %thread_forall = transform.structured.tile_using_forall"
       << " %promoted num_threads [" << config.block_threads_y << ", "
       << config.block_threads_x
       << "](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op)"
          " -> (!transform.any_op, !transform.any_op)\n";
    ir << "    %launch = transform.gpu.map_forall_to_blocks %func"
          " generate_gpu_launch grid_dims = [" << config.grid_x << ", "
       << config.grid_y << ", 1] : (!transform.any_op)"
          " -> !transform.any_op\n";
    ir << "    %launch_threads = transform.gpu.map_nested_forall_to_threads"
          " %launch block_dims = [" << config.block_threads_x << ", "
       << config.block_threads_y
       << ", 1] sync_after_distribute = true warp_size = 32"
          " : (!transform.any_op) -> !transform.any_op\n";
  }
  ir << "    transform.yield\n";
  ir << "  }\n";
  ir << "}\n";
  return ir.str();
}

std::string buildNvidiaMmaRewriteSequence() {
  std::ostringstream ir;
  ir << "module attributes {transform.with_named_sequence} {\n";
  ir << "  transform.named_sequence @__transform_main"
        "(%root: !transform.any_op) {\n";
  ir << "    %launch = transform.structured.match ops{[\"gpu.launch\"]} in %root"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    %matmul = transform.structured.match ops{[\"linalg.matmul\"]} in %launch"
        " : (!transform.any_op) -> !transform.any_op\n";
  ir << "    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul"
        " : (!transform.any_op) -> ()\n";
  ir << "    transform.yield\n";
  ir << "  }\n";
  ir << "}\n";
  return ir.str();
}

std::string dumpModuleIR(mlir::ModuleOp module) {
  std::string module_ir;
  llvm::raw_string_ostream stream(module_ir);
  module.print(stream);
  stream.flush();
  return module_ir;
}

void applyNvidiaMmaTransformToModule(mlir::ModuleOp module,
                                     const MatmulLoweringSignature &signature) {
  mlir::DialectRegistry registry;
  registerTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  mlir::Operation *first_matmul = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (first_matmul == nullptr &&
        llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::QuantizedMatmulOp>(op)) {
      first_matmul = op;
    }
  });
  if (first_matmul == nullptr) {
    return;
  }

  auto linalg_op = llvm::cast<mlir::linalg::LinalgOp>(first_matmul);
  const NvidiaTileConfig tile_config = selectNvidiaTileConfig(linalg_op, signature);
  const std::string transform_ir =
      buildNvidiaTransformSequence(signature, tile_config);

  auto transform_module =
      mlir::parseSourceString<mlir::ModuleOp>(transform_ir, module.getContext());
  if (!transform_module) {
    fail("failed to parse NVIDIA transform sequence:\n" + transform_ir);
  }

  auto entry_point = transform_module->lookupSymbol<mlir::transform::NamedSequenceOp>(
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  if (!entry_point) {
    fail("failed to build NVIDIA transform entry point");
  }

  module->setAttr(mlir::transform::TransformDialect::kWithNamedSequenceAttrName,
                  mlir::UnitAttr::get(module.getContext()));
  {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    builder.clone(*entry_point);
  }

  mlir::PassManager pm(module.getContext());
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = mlir::transform::TransformDialect::kTransformEntryPointSymbolName.str();
  options.disableExpensiveChecks = false;
  pm.addPass(mlir::transform::createInterpreterPass(options));
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diag_handler(
      module.getContext(), [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream stream(diagnostics);
        diag.print(stream);
        stream << '\n';
        stream.flush();
        return mlir::success();
      });
  if (mlir::failed(pm.run(module))) {
    fail("failed to apply NVIDIA MMA transform sequence" +
         (diagnostics.empty() ? std::string() : ":\n" + diagnostics));
  }

  llvm::SmallVector<mlir::Operation *, 2> transform_symbols;
  module.walk([&](mlir::transform::NamedSequenceOp op) {
    transform_symbols.push_back(op.getOperation());
  });
  for (mlir::Operation *op : llvm::reverse(transform_symbols)) {
    op->erase();
  }
  module->removeAttr(mlir::transform::TransformDialect::kWithNamedSequenceAttrName);
}

void applyNvidiaMmaRewriteToModule(mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  registerTransformDialects(registry);
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadDialect<mlir::transform::TransformDialect>();

  bool has_matmul = false;
  module.walk([&](mlir::Operation *op) {
    if (!has_matmul && llvm::isa<mlir::linalg::MatmulOp>(op)) {
      has_matmul = true;
    }
  });
  if (!has_matmul) {
    return;
  }

  const std::string transform_ir = buildNvidiaMmaRewriteSequence();
  auto transform_module =
      mlir::parseSourceString<mlir::ModuleOp>(transform_ir, module.getContext());
  if (!transform_module) {
    fail("failed to parse NVIDIA MMA rewrite sequence:\n" + transform_ir);
  }

  auto entry_point = transform_module->lookupSymbol<mlir::transform::NamedSequenceOp>(
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
  if (!entry_point) {
    fail("failed to build NVIDIA MMA rewrite entry point");
  }

  module->setAttr(mlir::transform::TransformDialect::kWithNamedSequenceAttrName,
                  mlir::UnitAttr::get(module.getContext()));
  {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    builder.clone(*entry_point);
  }

  mlir::PassManager pm(module.getContext());
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint =
      mlir::transform::TransformDialect::kTransformEntryPointSymbolName.str();
  options.disableExpensiveChecks = false;
  pm.addPass(mlir::transform::createInterpreterPass(options));
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diag_handler(
      module.getContext(), [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream stream(diagnostics);
        diag.print(stream);
        stream << '\n';
        stream.flush();
        return mlir::success();
      });
  if (mlir::failed(pm.run(module))) {
    fail("failed to apply NVIDIA MMA rewrite sequence" +
         (diagnostics.empty() ? std::string() : ":\n" + diagnostics));
  }

  llvm::SmallVector<mlir::Operation *, 2> transform_symbols;
  module.walk([&](mlir::transform::NamedSequenceOp op) {
    transform_symbols.push_back(op.getOperation());
  });
  for (mlir::Operation *op : llvm::reverse(transform_symbols)) {
    op->erase();
  }
  module->removeAttr(mlir::transform::TransformDialect::kWithNamedSequenceAttrName);
}

void verifyNoResidualNvidiaMatmulOnModule(mlir::ModuleOp module) {
  bool residual_matmul = false;
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::linalg::MatmulOp>(op)) {
      residual_matmul = true;
    }
  });
  if (residual_matmul) {
    fail("NVIDIA lowering left a residual linalg matmul");
  }
}

struct TileAndVectorizeMatmulPass
    : public mlir::PassWrapper<TileAndVectorizeMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    std::vector<mlir::Operation *> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op.getOperation()); });

    mlir::IRRewriter rewriter(&getContext());
    for (mlir::Operation *matmul : matmuls) {
      auto typed_matmul = llvm::dyn_cast<mlir::linalg::MatmulOp>(matmul);
      if (!typed_matmul) {
        matmul->emitError("MatCore expected linalg.matmul while tiling");
        signalPassFailure();
        return;
      }

      mlir::linalg::LinalgTilingOptions tiling_options;
      tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
      tiling_options.setTileSizes(selectCpuMatmulTileSizes(typed_matmul));

      rewriter.setInsertionPoint(matmul);
      mlir::FailureOr<mlir::linalg::TiledLinalgOp> tiled =
          mlir::linalg::tileLinalgOp(rewriter, typed_matmul, tiling_options);
      if (mlir::failed(tiled)) {
        matmul->emitError("MatCore failed to tile linalg.matmul for x86 lowering");
        signalPassFailure();
        return;
      }

      mlir::Operation *vectorize_target = tiled->op.getOperation();
      if (mlir::failed(mlir::linalg::vectorizeOpPrecondition(vectorize_target))) {
        vectorize_target->emitError(
            "MatCore x86 lowering requires vectorizable tiled linalg.matmul");
        signalPassFailure();
        return;
      }
      if (mlir::failed(mlir::linalg::vectorize(rewriter, vectorize_target))) {
        vectorize_target->emitError("MatCore failed to vectorize tiled linalg.matmul");
        signalPassFailure();
        return;
      }
      if (typed_matmul.getOperation() != vectorize_target &&
          typed_matmul.getOperation()->getBlock() != nullptr) {
        rewriter.eraseOp(typed_matmul.getOperation());
      }
      if (vectorize_target->getBlock() != nullptr) {
        rewriter.eraseOp(vectorize_target);
      }
    }
  }
};

struct LowerResidualVectorOpsPass
    : public mlir::PassWrapper<LowerResidualVectorOpsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns, mlir::vector::VectorMultiReductionLowering::InnerReduction);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("MatCore failed to lower residual vector ops");
      signalPassFailure();
    }
  }
};

struct ApplyNvidiaMmaTransformPass
    : public mlir::PassWrapper<ApplyNvidiaMmaTransformPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  explicit ApplyNvidiaMmaTransformPass(MatmulLoweringSignature signature)
      : signature(signature) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registerTransformDialects(registry);
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::Operation *first_matmul = nullptr;
    module.walk([&](mlir::Operation *op) {
      if (first_matmul == nullptr &&
          llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::QuantizedMatmulOp>(op)) {
        first_matmul = op;
      }
    });
    if (first_matmul == nullptr) {
      return;
    }

    auto linalg_op = llvm::cast<mlir::linalg::LinalgOp>(first_matmul);
    const NvidiaTileConfig tile_config = selectNvidiaTileConfig(linalg_op, signature);
    const std::string transform_ir =
        buildNvidiaTransformSequence(signature, tile_config);

    auto transform_module =
        mlir::parseSourceString<mlir::ModuleOp>(transform_ir, &getContext());
    if (!transform_module) {
      module.emitError("MatCore failed to parse NVIDIA transform sequence:\n")
          << transform_ir;
      signalPassFailure();
      return;
    }

    auto entry_point = transform_module->lookupSymbol<mlir::transform::NamedSequenceOp>(
        mlir::transform::TransformDialect::kTransformEntryPointSymbolName);
    if (!entry_point) {
      module.emitError("MatCore failed to build NVIDIA transform entry point");
      signalPassFailure();
      return;
    }

    mlir::transform::TransformOptions options;
    options.enableExpensiveChecks(true);
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler diag_handler(
        &getContext(), [&](mlir::Diagnostic &diag) {
          llvm::raw_string_ostream stream(diagnostics);
          diag.print(stream);
          stream << '\n';
          stream.flush();
          return mlir::success();
        });
    if (mlir::failed(mlir::transform::applyTransformNamedSequence(
            module, entry_point.getOperation(), *transform_module, options))) {
      module.emitError("MatCore failed to apply NVIDIA MMA transform sequence")
          << (diagnostics.empty() ? "" : (":\n" + diagnostics));
      signalPassFailure();
    }
  }

 private:
  MatmulLoweringSignature signature;
};

struct ConfigureNvidiaLaunchPass
    : public mlir::PassWrapper<ConfigureNvidiaLaunchPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::OpBuilder builder(&getContext());
    func.walk([&](mlir::gpu::LaunchOp launch) {
      builder.setInsertionPoint(launch);
      auto c1 = builder.create<mlir::arith::ConstantIndexOp>(launch.getLoc(), 1);
      auto c32 =
          builder.create<mlir::arith::ConstantIndexOp>(launch.getLoc(), 32);
      launch.getBlockSizeXMutable().set(c32.getResult());
      launch.getBlockSizeYMutable().set(c1.getResult());
      launch.getBlockSizeZMutable().set(c1.getResult());
    });
  }
};

struct PromoteGpuWorkgroupAllocationsPass
    : public mlir::PassWrapper<PromoteGpuWorkgroupAllocationsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::gpu::LaunchOp, 4> launches;
    func.walk([&](mlir::gpu::LaunchOp launch) { launches.push_back(launch); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::gpu::LaunchOp launch : launches) {
      llvm::SmallVector<mlir::memref::AllocOp, 4> workgroup_allocs;
      launch.walk([&](mlir::memref::AllocOp alloc) {
        if (alloc->getParentOfType<mlir::gpu::LaunchOp>() != launch) {
          return;
        }
        if (isWorkgroupMemRefType(alloc.getType())) {
          workgroup_allocs.push_back(alloc);
        }
      });

      if (workgroup_allocs.empty()) {
        continue;
      }

      builder.setInsertionPointToStart(&launch.getBody().front());
      for (mlir::memref::AllocOp alloc : workgroup_allocs) {
        if (!alloc->use_empty() && alloc->hasOneUse() &&
            llvm::isa<mlir::memref::DeallocOp>(*alloc->user_begin())) {
          alloc.emitError("MatCore cannot legalize a dead workgroup allocation");
          signalPassFailure();
          return;
        }

        llvm::SmallVector<mlir::memref::DeallocOp, 2> deallocs;
        mlir::memref::ViewOp view;
        for (mlir::Operation *user : alloc->getUsers()) {
          if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(user)) {
            deallocs.push_back(dealloc);
            continue;
          }
          if (auto candidate_view = llvm::dyn_cast<mlir::memref::ViewOp>(user)) {
            if (view && candidate_view != view) {
              alloc.emitError("MatCore expected at most one workgroup memref.view");
              signalPassFailure();
              return;
            }
            view = candidate_view;
            continue;
          }
        }

        if (view) {
          auto static_view_type = inferStaticViewType(view);
          if (!static_view_type.has_value()) {
            view.emitError("MatCore requires statically sized promoted workgroup views");
            signalPassFailure();
            return;
          }

          mlir::BlockArgument attribution =
              addLaunchWorkgroupAttribution(launch, *static_view_type,
                                            view.getLoc());
          mlir::Value replacement = attribution;
          if (attribution.getType() != view.getType()) {
            replacement =
                builder.create<mlir::memref::CastOp>(view.getLoc(), view.getType(),
                                                     attribution);
          }
          view.getResult().replaceAllUsesWith(replacement);
          view.erase();
        } else {
          if (!alloc.getType().hasStaticShape()) {
            alloc.emitError("MatCore requires statically shaped workgroup allocations");
            signalPassFailure();
            return;
          }
          mlir::BlockArgument attribution =
              addLaunchWorkgroupAttribution(launch, alloc.getType(),
                                            alloc.getLoc());
          llvm::SmallVector<mlir::OpOperand *, 8> non_dealloc_uses;
          for (mlir::OpOperand &use : alloc->getUses()) {
            if (!llvm::isa<mlir::memref::DeallocOp>(use.getOwner())) {
              non_dealloc_uses.push_back(&use);
            }
          }
          for (mlir::OpOperand *use : non_dealloc_uses) {
            use->set(attribution);
          }
        }

        for (mlir::memref::DeallocOp dealloc : deallocs) {
          dealloc.erase();
        }
        if (alloc->use_empty()) {
          alloc.erase();
        }
      }
    }
  }
};

struct SpecializeNvidiaWorkgroupMatmulOperandsPass
    : public mlir::PassWrapper<SpecializeNvidiaWorkgroupMatmulOperandsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<mlir::linalg::MatmulOp, 4> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    mlir::OpBuilder builder(&getContext());
    for (mlir::linalg::MatmulOp op : matmuls) {
      llvm::SmallVector<mlir::Value, 2> specialized_inputs;
      bool changed = false;
      for (mlir::Value input : op.getInputs()) {
        mlir::Value replacement = input;
        if (auto cast = input.getDefiningOp<mlir::memref::CastOp>()) {
          auto source_type = llvm::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
          auto result_type = llvm::dyn_cast<mlir::MemRefType>(cast.getType());
          if (source_type && result_type && source_type.hasStaticShape() &&
              !result_type.hasStaticShape() &&
              isWorkgroupMemorySpace(source_type.getMemorySpace())) {
            replacement = cast.getSource();
            changed = true;
          }
        }
        specialized_inputs.push_back(replacement);
      }

      if (!changed) {
        continue;
      }

      builder.setInsertionPoint(op);
      auto replacement = builder.create<mlir::linalg::MatmulOp>(
          op.getLoc(), mlir::ValueRange(specialized_inputs), op.getOutputs());
      op->replaceAllUsesWith(replacement->getResults());
      op.erase();
    }
  }
};

struct VerifyNoResidualNvidiaMatmulPass
    : public mlir::PassWrapper<VerifyNoResidualNvidiaMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    bool residual_matmul = false;
    func.walk([&](mlir::Operation *op) {
      if (llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::QuantizedMatmulOp>(op)) {
        op->emitError("MatCore NVIDIA lowering left a residual linalg matmul");
        residual_matmul = true;
      }
    });
    if (residual_matmul) {
      signalPassFailure();
    }
  }
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

void addCommonLLVMLoweringPasses(mlir::PassManager &pm, bool enable_x86vector) {
  mlir::ConvertVectorToLLVMPassOptions vector_to_llvm_opts;
  vector_to_llvm_opts.x86Vector = enable_x86vector;

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass(vector_to_llvm_opts));
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void addLinalgToGpuLaunchPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
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

void configureNvidiaGenericGpuStage(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);
  addGpuCommonModulePasses(pm, /*index_bitwidth=*/64);
  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/true));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::nvgpu::createOptimizeSharedMemoryPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void addGpuHostPostPasses(mlir::PassManager &pm, const std::string &binary_target,
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

bool useCpuVectorPipeline(const MatmulLoweringSignature &signature) {
  if (signature.quantized_i8) {
    return false;
  }
  if (signature.lhs_dtype != TensorDType::kFloat32 ||
      signature.rhs_dtype != TensorDType::kFloat32 ||
      signature.out_dtype != TensorDType::kFloat32) {
    return false;
  }
  return true;
}

void configureCpuPassPipeline(mlir::PassManager &pm,
                              const MatmulLoweringSignature &signature) {
  if (useCpuVectorPipeline(signature)) {
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<TileAndVectorizeMatmulPass>());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::memref::createExpandOpsPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        std::make_unique<LowerResidualVectorOpsPass>());
    mlir::VectorTransferToSCFOptions vector_to_scf_options;
    vector_to_scf_options.setTargetRank(1);
    pm.addPass(mlir::createConvertVectorToSCFPass(vector_to_scf_options));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/true);
    return;
  }

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  addCommonLLVMLoweringPasses(pm, /*enable_x86vector=*/false);
}

void configureNvidiaPassPipeline(mlir::PassManager &pm,
                                 const MatmulLoweringSignature &signature,
                                 llvm::StringRef cubin_chip) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ApplyNvidiaMmaTransformPass>(signature));
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ConfigureNvidiaLaunchPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<VerifyNoResidualNvidiaMatmulPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/true));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::nvgpu::createOptimizeSharedMemoryPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  mlir::gpu::GPUToNVVMPipelineOptions nvvm_opts;
  nvvm_opts.indexBitWidth = 64;
  nvvm_opts.cubinTriple = "nvptx64-nvidia-cuda";
  nvvm_opts.cubinChip = cubin_chip.str();
  nvvm_opts.cubinFormat = "fatbin";
  nvvm_opts.optLevel = 2;
  nvvm_opts.kernelUseBarePtrCallConv = false;
  nvvm_opts.hostUseBarePtrCallConv = false;
  mlir::gpu::buildLowerToNVVMPassPipeline(pm, nvvm_opts);
}

void configureNvidiaTransformStage(mlir::PassManager &pm,
                                   const MatmulLoweringSignature &signature) {
  pm.addPass(std::make_unique<ApplyNvidiaMmaTransformPass>(signature));
}

void configureNvidiaLoopMaterializationStage(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<PromoteGpuWorkgroupAllocationsPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void configureNvidiaMmaPreparationStage(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<PromoteGpuWorkgroupAllocationsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<SpecializeNvidiaWorkgroupMatmulOperandsPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void configureNvidiaVectorToGpuStage(mlir::PassManager &pm) {
  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/true));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::nvgpu::createOptimizeSharedMemoryPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void configureNvidiaNvvmStage(mlir::PassManager &pm, llvm::StringRef cubin_chip) {
  mlir::gpu::GPUToNVVMPipelineOptions nvvm_opts;
  nvvm_opts.indexBitWidth = 64;
  nvvm_opts.cubinTriple = "nvptx64-nvidia-cuda";
  nvvm_opts.cubinChip = cubin_chip.str();
  nvvm_opts.cubinFormat = "fatbin";
  nvvm_opts.optLevel = 2;
  nvvm_opts.kernelUseBarePtrCallConv = false;
  nvvm_opts.hostUseBarePtrCallConv = false;
  mlir::gpu::buildLowerToNVVMPassPipeline(pm, nvvm_opts);
}

void configureAmdPassPipeline(mlir::PassManager &pm) {
  addLinalgToGpuLaunchPasses(pm);
  addGpuCommonModulePasses(pm, /*index_bitwidth=*/64);

  mlir::GpuROCDLAttachTargetOptions rocdl_target_opts;
  rocdl_target_opts.triple = "amdgcn-amd-amdhsa";
  rocdl_target_opts.chip = "gfx1150";
  pm.addPass(mlir::createGpuROCDLAttachTarget(rocdl_target_opts));

  pm.addPass(mlir::createConvertVectorToGPUPass(/*useNvGpu=*/false));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createLowerGpuOpsToROCDLOpsPass(
          rocdl_target_opts.chip, /*indexBitwidth=*/64,
          /*useBarePtrCallConv=*/false, mlir::gpu::amd::Runtime::HIP));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());
  // Ubuntu packages ROCm device bitcode under the LLVM 17 Clang resource dir.
  addGpuHostPostPasses(pm, "bin", "/usr/lib/llvm-17/lib/clang/17");
}

void validateFp8Support(const LoweringPlan &plan,
                        const MatmulLoweringSignature &signature) {
  if (signature.lhs_dtype != TensorDType::kFloat8E4M3FN &&
      signature.rhs_dtype != TensorDType::kFloat8E4M3FN) {
    return;
  }
  if (plan.route != LoweringRoute::kNvidiaNvptx) {
    fail("float8_e4m3fn matmul is currently limited to nvidia-dgpu");
  }
  fail("float8_e4m3fn matmul requires a dedicated native NVIDIA FP8 WGMMA "
       "lowering path, and MatCore does not implement that path yet");
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
      return "linalg.matmul -> block tiling -> workgroup promotion -> mma sync -> gpu.launch -> nvvm -> llvm";
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
  if (lhs_attr && rhs_attr && out_attr) {
    signature.lhs_dtype = decodeTensorDType(lhs_attr.getValue());
    signature.rhs_dtype = decodeTensorDType(rhs_attr.getValue());
    signature.out_dtype = decodeTensorDType(out_attr.getValue());
    signature.quantized_i8 =
        signature.lhs_dtype == TensorDType::kInt8 &&
        signature.out_dtype == TensorDType::kInt32;
  }
  return signature;
}

void configureLoweringPipeline(mlir::PassManager &pm, const LoweringPlan &plan,
                               const MatmulLoweringSignature &signature,
                               llvm::StringRef nvidia_chip) {
  validateFp8Support(plan, signature);

  switch (plan.route) {
    case LoweringRoute::kCpuVector:
      configureCpuPassPipeline(pm, signature);
      return;
    case LoweringRoute::kNvidiaNvptx:
      configureNvidiaPassPipeline(pm, signature, nvidia_chip);
      return;
    case LoweringRoute::kAmdRocdl:
      configureAmdPassPipeline(pm);
      return;
    case LoweringRoute::kAmdNpuScaffold:
      fail("amd-npu lowering remains unavailable without an external AIE/XDNA toolchain");
      return;
  }
}

void runLoweringPipeline(mlir::ModuleOp module, const LoweringPlan &plan,
                         const MatmulLoweringSignature &signature,
                         llvm::StringRef nvidia_chip) {
  auto run_stage = [&](llvm::StringRef stage_name,
                       auto &&configure_stage) {
    mlir::PassManager pm(module.getContext());
    configure_stage(pm);
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler diag_handler(
        module.getContext(), [&](mlir::Diagnostic &diag) {
          llvm::raw_string_ostream stream(diagnostics);
          diag.print(stream);
          stream << '\n';
          stream.flush();
          return mlir::success();
        });
    if (mlir::failed(pm.run(module))) {
      std::string module_ir;
      llvm::raw_string_ostream stream(module_ir);
      module.print(stream);
      stream.flush();
      fail("failed to run lowering pipeline for route " +
           std::string(routeName(plan.route)) + " at stage '" +
           stage_name.str() + "'" +
           (diagnostics.empty() ? "" : ("\n" + diagnostics)) + "\n" + module_ir);
    }
  };

  if (plan.route == LoweringRoute::kNvidiaNvptx) {
    bool use_mma_path = false;
    module.walk([&](mlir::Operation *op) {
      if (use_mma_path) {
        return;
      }
      auto linalg_op = llvm::dyn_cast<mlir::linalg::LinalgOp>(op);
      if (!linalg_op) {
        return;
      }
      if (!llvm::isa<mlir::linalg::MatmulOp>(op)) {
        return;
      }
      use_mma_path = selectNvidiaTileConfig(linalg_op, signature).rewrite_to_mma_sync;
    });

    try {
      applyNvidiaMmaTransformToModule(module, signature);
    } catch (const std::exception &exc) {
      fail("failed to run lowering pipeline for route " +
           std::string(routeName(plan.route)) +
           " at stage 'nvidia-apply-mma-transform'\n" + exc.what() + "\n" +
           dumpModuleIR(module));
    }
    if (use_mma_path) {
      run_stage("nvidia-post-transform-canonicalize", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });
      run_stage("nvidia-mma-preparation", [&](mlir::PassManager &pm) {
        configureNvidiaMmaPreparationStage(pm);
      });
      try {
        applyNvidiaMmaRewriteToModule(module);
      } catch (const std::exception &exc) {
        fail("failed to run lowering pipeline for route " +
             std::string(routeName(plan.route)) +
             " at stage 'nvidia-rewrite-mma-sync'\n" + exc.what() + "\n" +
             dumpModuleIR(module));
      }
      verifyNoResidualNvidiaMatmulOnModule(module);
      run_stage("nvidia-configure-launch", [&](mlir::PassManager &pm) {
        pm.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<ConfigureNvidiaLaunchPass>());
      });
      run_stage("nvidia-loop-materialization", [&](mlir::PassManager &pm) {
        configureNvidiaLoopMaterializationStage(pm);
      });
      run_stage("nvidia-vector-to-gpu", [&](mlir::PassManager &pm) {
        configureNvidiaVectorToGpuStage(pm);
      });
    } else {
      run_stage("nvidia-generic-gpu", [&](mlir::PassManager &pm) {
        configureNvidiaGenericGpuStage(pm);
      });
    }
    run_stage("nvidia-nvvm", [&](mlir::PassManager &pm) {
      configureNvidiaNvvmStage(pm, nvidia_chip);
    });
    return;
  }

  mlir::PassManager pm(module.getContext());
  configureLoweringPipeline(pm, plan, signature, nvidia_chip);
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diag_handler(
      module.getContext(), [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream stream(diagnostics);
        diag.print(stream);
        stream << '\n';
        stream.flush();
        return mlir::success();
      });
  if (mlir::failed(pm.run(module))) {
    fail("failed to run lowering pipeline for route " +
         std::string(routeName(plan.route)) +
         (diagnostics.empty() ? "" : ("\n" + diagnostics)) + "\n" +
         dumpModuleIR(module));
  }
}

}  // namespace matcore
