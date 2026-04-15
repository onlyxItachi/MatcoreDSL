#include "gpu_data_staging.h"

#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace matcore {
namespace {

/// Trace a memref Value back through view-like ops (subview, cast,
/// reinterpret_cast, expand_shape, collapse_shape, view) to find the root
/// source. If the root is a BlockArgument of the enclosing func's ENTRY block,
/// return it. Returns nullptr for non-func block args (e.g. scf.for iter args).
static mlir::BlockArgument traceToFuncArg(mlir::Value memref) {
  mlir::Value current = memref;
  for (;;) {
    if (auto subview = current.getDefiningOp<mlir::memref::SubViewOp>()) {
      current = subview.getSource();
    } else if (auto cast = current.getDefiningOp<mlir::memref::CastOp>()) {
      current = cast.getSource();
    } else if (auto ri = current.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      current = ri.getSource();
    } else if (auto expand = current.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
      current = expand.getSrc();
    } else if (auto collapse = current.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
      current = collapse.getSrc();
    } else if (auto view = current.getDefiningOp<mlir::memref::ViewOp>()) {
      current = view.getSource();
    } else {
      break;
    }
  }
  auto arg = llvm::dyn_cast<mlir::BlockArgument>(current);
  if (!arg)
    return {};
  // Only accept args from the function's entry block — not scf.for iter args
  // or other region args which would give false positives.
  auto func = llvm::dyn_cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func || arg.getOwner() != &func.getBody().front())
    return {};
  return arg;
}

/// Check if a captured memref traces back to a func argument tagged with
/// the "matcore.device_resident" attribute.
static bool isDeviceResident(mlir::Value memref, mlir::func::FuncOp func) {
  mlir::BlockArgument arg = traceToFuncArg(memref);
  if (!arg)
    return false;
  unsigned idx = arg.getArgNumber();
  return func.getArgAttr(idx, "matcore.device_resident") != nullptr;
}

/// Collect every memref-typed Value that is *used* inside the gpu.launch body
/// but *defined* outside it (i.e. host-resident buffers captured by the kernel).
llvm::SetVector<mlir::Value> collectCapturedMemrefs(mlir::gpu::LaunchOp launch) {
  llvm::SetVector<mlir::Value> captured;
  launch.getBody().walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (!llvm::isa<mlir::MemRefType>(operand.getType()))
        continue;
      // Skip values defined inside the launch body.
      if (launch.getBody().isAncestor(operand.getParentRegion()))
        continue;
      captured.insert(operand);
    }
  });
  return captured;
}

/// Build a set of all memref Values that alias `root` inside the launch body.
/// Traces through view-like ops: subview, cast, reinterpret_cast,
/// expand_shape, collapse_shape, view.  Uses a worklist to reach transitive
/// aliases (e.g. subview-of-subview).
llvm::DenseSet<mlir::Value> buildAliasSet(mlir::gpu::LaunchOp launch,
                                           mlir::Value root) {
  llvm::DenseSet<mlir::Value> aliases;
  aliases.insert(root);

  // Worklist-based fixed-point: keep expanding until no new aliases found.
  bool changed = true;
  while (changed) {
    changed = false;
    launch.getBody().walk([&](mlir::Operation *op) {
      mlir::Value source;
      mlir::Value result;

      if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(op)) {
        source = subview.getSource();
        result = subview.getResult();
      } else if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(op)) {
        source = cast.getSource();
        result = cast.getResult();
      } else if (auto ri = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
        source = ri.getSource();
        result = ri.getResult();
      } else if (auto expand = llvm::dyn_cast<mlir::memref::ExpandShapeOp>(op)) {
        source = expand.getSrc();
        result = expand.getResult();
      } else if (auto collapse = llvm::dyn_cast<mlir::memref::CollapseShapeOp>(op)) {
        source = collapse.getSrc();
        result = collapse.getResult();
      } else if (auto view = llvm::dyn_cast<mlir::memref::ViewOp>(op)) {
        source = view.getSource();
        result = view.getResult();
      } else if (auto esm = llvm::dyn_cast<mlir::memref::ExtractStridedMetadataOp>(op)) {
        // extract_strided_metadata returns (base_buffer, offset, sizes..., strides...).
        // The base_buffer (result #0) aliases the source memref.
        source = esm.getSource();
        result = esm.getBaseBuffer();
      } else {
        return;
      }

      if (source && result && aliases.count(source) && !aliases.count(result)) {
        aliases.insert(result);
        changed = true;
      }
    });
  }
  return aliases;
}

/// Conservative check: does the launch body potentially *write* to `memref`
/// or any of its aliases (subviews, casts, etc.)?
bool isWrittenInsideLaunch(mlir::gpu::LaunchOp launch, mlir::Value memref) {
  llvm::DenseSet<mlir::Value> aliases = buildAliasSet(launch, memref);
  bool written = false;

  launch.getBody().walk([&](mlir::Operation *op) {
    if (written)
      return mlir::WalkResult::interrupt();
    // memref.store
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
      if (aliases.count(store.getMemRef()))
        written = true;
    }
    // vector.transfer_write (MMA path writes through this)
    if (auto xfer = llvm::dyn_cast<mlir::vector::TransferWriteOp>(op)) {
      if (aliases.count(xfer.getSource()))
        written = true;
    }
    // gpu.subgroup_mma_store_matrix
    if (auto mma_store = llvm::dyn_cast<mlir::gpu::SubgroupMmaStoreMatrixOp>(op)) {
      if (aliases.count(mma_store.getDstMemref()))
        written = true;
    }
    // linalg output operands
    if (auto linalg_op = llvm::dyn_cast<mlir::linalg::LinalgOp>(op)) {
      for (mlir::Value init : linalg_op.getDpsInits()) {
        if (aliases.count(init))
          written = true;
      }
    }
    // memref.copy destination
    if (auto copy = llvm::dyn_cast<mlir::memref::CopyOp>(op)) {
      if (aliases.count(copy.getTarget()))
        written = true;
    }
    return mlir::WalkResult::advance();
  });
  return written;
}

/// The pass itself.
struct GpuDataStagingPass
    : public mlir::PassWrapper<GpuDataStagingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuDataStagingPass)

  llvm::StringRef getArgument() const override {
    return "matcore-gpu-data-staging";
  }
  llvm::StringRef getDescription() const override {
    return "Stage host memrefs to GPU device memory around gpu.launch ops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    // Collect all gpu.launch ops (snapshot to avoid iterator invalidation).
    llvm::SmallVector<mlir::gpu::LaunchOp, 4> launches;
    func.walk([&](mlir::gpu::LaunchOp launch) {
      launches.push_back(launch);
    });

    for (auto launch : launches) {
      if (mlir::failed(stageDataForLaunch(launch))) {
        signalPassFailure();
        return;
      }
    }
  }

  mlir::LogicalResult stageDataForLaunch(mlir::gpu::LaunchOp launch) {
    llvm::SetVector<mlir::Value> captured = collectCapturedMemrefs(launch);
    if (captured.empty())
      return mlir::success();

    // Get enclosing FuncOp for device-residency checks.
    auto func = launch->getParentOfType<mlir::func::FuncOp>();

    // Partition: device-resident memrefs are already on GPU — skip them.
    llvm::SetVector<mlir::Value> host_captured;
    for (mlir::Value memref : captured) {
      if (func && isDeviceResident(memref, func)) {
        // Already on device — no alloc, no H→D copy, no D→H copy needed.
        continue;
      }
      host_captured.insert(memref);
    }

    // If all captured memrefs are device-resident, nothing to stage.
    if (host_captured.empty())
      return mlir::success();

    mlir::Location loc = launch.getLoc();
    auto token_type =
        mlir::gpu::AsyncTokenType::get(launch->getContext());

    // Phase 1: Before the launch — allocate device memory and copy H→D.
    // All ops use async form (gpu-to-llvm only lowers async gpu ops).
    mlir::OpBuilder pre_builder(launch);
    llvm::DenseMap<mlir::Value, mlir::Value> host_to_device;
    llvm::SmallVector<mlir::Value> all_device_memrefs;
    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> output_pairs;

    // Create initial async token via gpu.wait async.
    auto initial_wait = pre_builder.create<mlir::gpu::WaitOp>(
        loc, token_type, mlir::ValueRange{});
    mlir::Value current_token = initial_wait.getAsyncToken();

    for (mlir::Value host_memref : host_captured) {
      auto memref_type = llvm::cast<mlir::MemRefType>(host_memref.getType());

      // gpu.alloc async [token] → (memref, token)
      auto alloc = pre_builder.create<mlir::gpu::AllocOp>(
          loc,
          /*memref=*/memref_type,
          /*asyncToken=*/token_type,
          /*asyncDependencies=*/mlir::ValueRange{current_token},
          /*dynamicSizes=*/mlir::ValueRange{},
          /*symbolOperands=*/mlir::ValueRange{},
          /*hostShared=*/false);
      mlir::Value device_memref = alloc.getMemref();
      current_token = alloc.getAsyncToken();

      // gpu.memcpy async [token] host → device → token
      auto memcpy = pre_builder.create<mlir::gpu::MemcpyOp>(
          loc,
          /*asyncToken=*/token_type,
          /*asyncDependencies=*/mlir::ValueRange{current_token},
          /*dst=*/device_memref,
          /*src=*/host_memref);
      current_token = memcpy.getAsyncToken();

      host_to_device[host_memref] = device_memref;
      all_device_memrefs.push_back(device_memref);

      // Track outputs for D→H copy after the launch.
      if (isWrittenInsideLaunch(launch, host_memref)) {
        output_pairs.emplace_back(host_memref, device_memref);
      }
    }

    // Synchronize: wait for all H→D copies before kernel launch.
    pre_builder.create<mlir::gpu::WaitOp>(
        loc, mlir::Type(), mlir::ValueRange{current_token});

    // Phase 2: Replace host memref uses inside the launch body with device
    // counterparts.  We walk every operation in the launch and rewrite operands.
    launch.getBody().walk([&](mlir::Operation *op) {
      for (mlir::OpOperand &operand : op->getOpOperands()) {
        auto it = host_to_device.find(operand.get());
        if (it != host_to_device.end()) {
          operand.set(it->second);
        }
      }
    });

    // Phase 3: After the launch — copy written memrefs D→H, then dealloc all.
    // Again use async form for gpu-to-llvm compatibility.
    mlir::OpBuilder post_builder(launch->getBlock(),
                                 std::next(launch->getIterator()));

    auto post_wait = post_builder.create<mlir::gpu::WaitOp>(
        loc, token_type, mlir::ValueRange{});
    current_token = post_wait.getAsyncToken();

    for (auto [host_memref, device_memref] : output_pairs) {
      auto memcpy = post_builder.create<mlir::gpu::MemcpyOp>(
          loc,
          /*asyncToken=*/token_type,
          /*asyncDependencies=*/mlir::ValueRange{current_token},
          /*dst=*/host_memref,
          /*src=*/device_memref);
      current_token = memcpy.getAsyncToken();
    }

    // Synchronize before dealloc to ensure D→H copies complete.
    post_builder.create<mlir::gpu::WaitOp>(
        loc, mlir::Type(), mlir::ValueRange{current_token});

    // Dealloc all device buffers (async chain, then final wait).
    auto dealloc_wait = post_builder.create<mlir::gpu::WaitOp>(
        loc, token_type, mlir::ValueRange{});
    current_token = dealloc_wait.getAsyncToken();

    for (mlir::Value device_memref : all_device_memrefs) {
      auto dealloc = post_builder.create<mlir::gpu::DeallocOp>(
          loc,
          /*asyncToken=*/token_type,
          /*asyncDependencies=*/mlir::ValueRange{current_token},
          /*memref=*/device_memref);
      current_token = dealloc.getAsyncToken();
    }

    // Final wait for all deallocations.
    post_builder.create<mlir::gpu::WaitOp>(
        loc, mlir::Type(), mlir::ValueRange{current_token});

    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateGpuDataStagingPass() {
  return std::make_unique<GpuDataStagingPass>();
}

}  // namespace matcore
