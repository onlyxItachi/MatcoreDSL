#include "matcore/lowering_pipeline.h"

#include "fp8_wgmma.h"
#include "gpu_amd_lowering.h"
#include "gpu_data_staging.h"
#include "matcore/cpu_lowering.h"
#include "matcore/diagnostics.h"
#include "matcore/fusion_analysis.h"
#include "matcore/fusion_emitter.h"
#include "matcore/gpu_mapping.h"
#include "matcore/gpu_nvvm_lowering.h"
#include "matcore/gpu_tiling.h"
#include "matcore/observability.h"
#include "matcore/target_registry.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore {
namespace {

// Lowers residual vector ops (multi_reduction, broadcast, transpose, shape_cast)
// BEFORE ConvertGpuOpsToNVVMOps. This is critical because ConvertVectorToLLVMPass
// (which would normally handle these) ALSO converts the vector<2xf16> operands of
// nvvm.mma.sync, destroying the MMA intrinsic and causing fallback to scalar FP16.
// By decomposing these ops before NVVM conversion, we avoid needing
// ConvertVectorToLLVMPass entirely, preserving Tensor Core emission.
struct LowerResidualVectorOpsPass
    : public mlir::PassWrapper<LowerResidualVectorOpsPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerResidualVectorOpsPass)

  llvm::StringRef getArgument() const override {
    return "matcore-lower-residual-vector-ops";
  }
  llvm::StringRef getDescription() const override {
    return "Lower vector.multi_reduction/broadcast/transpose before NVVM conversion";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    // Multi-reduction: 3D accumulator reduction → 1D vector.reduction + extract/insert
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns,
        mlir::vector::VectorMultiReductionLowering::InnerReduction);
    // Broadcast: vector.broadcast → insert/extract chains
    mlir::vector::populateVectorBroadcastLoweringPatterns(patterns);
    // Transpose: vector.transpose → shufflevector/insert/extract
    mlir::vector::populateVectorTransposeLoweringPatterns(
        patterns, mlir::vector::VectorTransformsOptions());
    // Shape cast: vector.shape_cast → insert/extract
    mlir::vector::populateVectorShapeCastLoweringPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore lowering pipeline: " + message);
}

std::optional<int> getFusionRegisterCap(mlir::ModuleOp module) {
  auto max_regs_attr = module->getAttrOfType<mlir::IntegerAttr>("matcore.max_regs");
  if (!max_regs_attr) {
    return std::nullopt;
  }
  const int requested = static_cast<int>(max_regs_attr.getInt());
  if (requested <= 0) {
    return std::nullopt;
  }
  return std::min(requested, FusionAnalyzer::kRegHardCap);
}

std::string buildMaxRegisterCommandOptions(std::optional<int> max_regs) {
  if (!max_regs) {
    return {};
  }
  return "--maxrregcount=" + std::to_string(*max_regs);
}

std::optional<int> parsePtxMaxRegisterCount(llvm::StringRef ptx) {
  constexpr llvm::StringLiteral marker = ".maxnreg";
  size_t offset = ptx.find(marker);
  if (offset == llvm::StringRef::npos) {
    return std::nullopt;
  }

  offset += marker.size();
  while (offset < ptx.size() &&
         std::isspace(static_cast<unsigned char>(ptx[offset]))) {
    ++offset;
  }

  size_t start = offset;
  while (offset < ptx.size() &&
         std::isdigit(static_cast<unsigned char>(ptx[offset]))) {
    ++offset;
  }
  if (start == offset) {
    return std::nullopt;
  }

  return std::stoi(ptx.substr(start, offset - start).str());
}

std::optional<std::string> extractPtxAssembly(mlir::ModuleOp module) {
  std::optional<std::string> ptx;
  module.walk([&](mlir::gpu::BinaryOp binary_op) {
    for (mlir::Attribute object_attr : binary_op.getObjects()) {
      auto object = llvm::dyn_cast<mlir::gpu::ObjectAttr>(object_attr);
      if (!object ||
          object.getFormat() != mlir::gpu::CompilationTarget::Assembly) {
        continue;
      }
      ptx = object.getObject().getValue().str();
      return;
    }
  });
  return ptx;
}

void annotateFusionRegisterUsageFromPtx(mlir::ModuleOp module,
                                        llvm::StringRef cmd_options,
                                        ObservabilityContext *obs) {
  auto cloned = mlir::OwningOpRef<mlir::ModuleOp>(
      llvm::cast<mlir::ModuleOp>(module->clone()));
  auto *ctx = module.getContext();
  mlir::PassManager pm(ctx);
  if (obs) {
    attachObservability(pm, obs, "fusion-nvvm-ptx-probe");
  }

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler diag_handler(ctx, [&](mlir::Diagnostic &diag) {
    llvm::raw_string_ostream stream(diagnostics);
    diag.print(stream);
    stream << '\n';
    stream.flush();
    return mlir::success();
  });

  mlir::GpuModuleToBinaryPassOptions ptx_opts;
  ptx_opts.compilationTarget = "assembly";
  ptx_opts.cmdOptions = cmd_options.str();
  pm.addPass(mlir::createGpuModuleToBinaryPass(ptx_opts));
  if (mlir::failed(pm.run(*cloned))) {
    (void)diagnostics;
    return;
  }

  const auto ptx = extractPtxAssembly(*cloned);
  const auto actual_reg_count =
      ptx ? parsePtxMaxRegisterCount(*ptx) : std::nullopt;
  if (!actual_reg_count) {
    return;
  }

  mlir::Builder builder(ctx);
  module->setAttr("matcore.actual_reg_count",
                  builder.getI32IntegerAttr(*actual_reg_count));
  module->setAttr(
      "matcore.reg_budget_exceeded",
      builder.getBoolAttr(*actual_reg_count > FusionAnalyzer::kRegHardCap));
}

static mlir::Value buildCeilDivIndex(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value lhs,
                                     mlir::Value rhs) {
  return builder.create<mlir::arith::CeilDivUIOp>(loc, lhs, rhs);
}

// ============================================================================
// Phase A (MW-7): Sub-tile loop unrolling.
// After MMA rewrite, the K-loop body contains nested M/N sub-tile loops.
// Unrolling them exposes 8 sequential MMA ops for Phase B accumulator hoisting.
// ============================================================================
struct SubTileUnrollPass
    : public mlir::PassWrapper<SubTileUnrollPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubTileUnrollPass)

  llvm::StringRef getArgument() const override {
    return "matcore-sub-tile-unroll";
  }
  llvm::StringRef getDescription() const override {
    return "Unroll M/N sub-tile loops to expose all MMA ops in K-loop body";
  }
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    // Fixpoint re-walk: unroll one loop per iteration to avoid stale ForOp
    // pointers after inner loop unrolling invalidates outer loop ops.
    int totalUnrolled = 0;
    bool changed = true;
    while (changed) {
      changed = false;
      op->walk([&](mlir::scf::ForOp forOp) -> mlir::WalkResult {
        auto lb = forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto ub = forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto step = forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
        if (!lb || !ub || !step || step.value() == 0)
          return mlir::WalkResult::advance();
        // Skip the K-loop: it contains linalg.copy ops for A/B tile loads.
        // Only unroll sub-tile loops (m_sub, n_sub, k_sub) which have MMA ops.
        bool hasCopy = false;
        forOp->walk([&](mlir::linalg::CopyOp) { hasCopy = true; });
        if (hasCopy)
          return mlir::WalkResult::advance();
        int64_t trip = (ub.value() - lb.value()) / step.value();
        if (trip >= 2 && trip <= 4) {
          auto result = mlir::loopUnrollByFactor(forOp, trip);
          fprintf(stderr, "[SubTileUnroll] Unroll factor %ld: %s\n",
                  trip, mlir::succeeded(result) ? "OK" : "FAILED");
          fflush(stderr);
          if (mlir::succeeded(result)) {
            totalUnrolled++;
            changed = true;
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      });
    }
    fprintf(stderr, "[SubTileUnroll] Total loops unrolled: %d\n", totalUnrolled);
    fflush(stderr);
  }
};

// ============================================================================
// Phase K1: TagKLoopPass — tag the K-reduction loop with a marker attribute.
// Runs after transform application, before DynamicMacroGridMappingPass.
// The attribute survives cloning (forall→launch) and is used by AccHoist,
// DoubleBuffer, and SplitK passes to reliably identify the K-loop.
// ============================================================================
struct TagKLoopPass
    : public mlir::PassWrapper<TagKLoopPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TagKLoopPass)

  explicit TagKLoopPass(int64_t k_tile) : k_tile_(k_tile) {}

  llvm::StringRef getArgument() const override {
    return "matcore-tag-k-loop";
  }
  llvm::StringRef getDescription() const override {
    return "Tag the K-reduction loop with matcore.k_loop attribute";
  }
  void runOnOperation() override {
    mlir::Operation *top = getOperation();
    int tagged = 0;

    // Strategy: find outermost scf.for that contains linalg.matmul (or
    // nvgpu.mma.sync after MMA rewrite) and has step matching k_tile.
    // "Outermost" = no ancestor scf.for also contains matmul/MMA ops.
    llvm::SmallVector<mlir::scf::ForOp, 4> candidates;
    top->walk([&](mlir::scf::ForOp forOp) {
      // Check step matches k_tile
      auto stepCst = forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!stepCst || stepCst.value() != k_tile_)
        return;

      // Check contains matmul or MMA ops (K-loop signature)
      bool hasMatmul = false;
      forOp.getBody()->walk([&](mlir::Operation *op) {
        if (llvm::isa<mlir::linalg::MatmulOp>(op) ||
            llvm::isa<mlir::nvgpu::MmaSyncOp>(op))
          hasMatmul = true;
      });
      if (!hasMatmul)
        return;

      // Check this is outermost: no ancestor scf.for with same properties
      bool hasMatchingAncestor = false;
      mlir::Operation *parent = forOp->getParentOp();
      while (parent) {
        if (auto parentFor = llvm::dyn_cast<mlir::scf::ForOp>(parent)) {
          bool parentHasMatmul = false;
          parentFor.getBody()->walk([&](mlir::Operation *op) {
            if (llvm::isa<mlir::linalg::MatmulOp>(op) ||
                llvm::isa<mlir::nvgpu::MmaSyncOp>(op))
              parentHasMatmul = true;
          });
          auto pStep = parentFor.getStep()
                           .getDefiningOp<mlir::arith::ConstantIndexOp>();
          if (parentHasMatmul && pStep && pStep.value() == k_tile_) {
            hasMatchingAncestor = true;
            break;
          }
        }
        parent = parent->getParentOp();
      }
      if (!hasMatchingAncestor)
        candidates.push_back(forOp);
    });

    for (auto forOp : candidates) {
      forOp->setAttr("matcore.k_loop",
                      mlir::UnitAttr::get(forOp->getContext()));
      tagged++;
    }
    fprintf(stderr, "[TagKLoop] Tagged %d K-loop(s) (k_tile=%lld)\n",
            tagged, (long long)k_tile_);
    fflush(stderr);
  }

  int64_t k_tile_;
};

// ============================================================================
// SplitKPartitionPass — partition K-loop by blockIdx.z.
// Must run BEFORE DoubleBuffer so the prologue/epilogue see correct bounds.
// Each block in gridDim.z processes K/split_k elements of the K dimension.
// ============================================================================
struct SplitKPartitionPass
    : public mlir::PassWrapper<SplitKPartitionPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitKPartitionPass)

  explicit SplitKPartitionPass(int64_t split_k) : split_k_factor(split_k) {}

  llvm::StringRef getArgument() const override {
    return "matcore-splitk-partition";
  }
  llvm::StringRef getDescription() const override {
    return "Partition K-loop bounds by blockIdx.z for split-K";
  }
  void runOnOperation() override {
    if (split_k_factor <= 1)
      return;

    mlir::Operation *top = getOperation();

    // Find the GEMM launch (first gpu.launch)
    mlir::gpu::LaunchOp gemmLaunch;
    top->walk([&](mlir::gpu::LaunchOp launch) {
      if (!gemmLaunch)
        gemmLaunch = launch;
      return mlir::WalkResult::interrupt();
    });
    if (!gemmLaunch)
      return;

    // Find tagged K-loop inside launch
    mlir::scf::ForOp kLoop;
    gemmLaunch.getBody().walk([&](mlir::scf::ForOp forOp) {
      if (forOp->hasAttr("matcore.k_loop")) {
        kLoop = forOp;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (!kLoop) {
      fprintf(stderr, "[SplitKPartition] No matcore.k_loop found, skipping\n");
      fflush(stderr);
      return;
    }

    mlir::OpBuilder kb(kLoop);
    mlir::Location loc = kLoop.getLoc();

    // Original K-loop: for k = 0 to K step k_tile
    // New bounds:  k_start = blockIdx.z * chunk
    //              k_end   = (blockIdx.z + 1) * chunk
    //   where chunk = K / split_k_factor
    mlir::Value origUB = kLoop.getUpperBound();
    mlir::Value splitKVal =
        kb.create<mlir::arith::ConstantIndexOp>(loc, split_k_factor);
    mlir::Value chunk =
        kb.create<mlir::arith::DivUIOp>(loc, origUB, splitKVal);

    mlir::Value blockIdxZ = gemmLaunch.getBlockIds().z;
    mlir::Value kStart =
        kb.create<mlir::arith::MulIOp>(loc, blockIdxZ, chunk);

    mlir::Value oneK =
        kb.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value bzPlusOne =
        kb.create<mlir::arith::AddIOp>(loc, blockIdxZ, oneK);
    mlir::Value kEnd =
        kb.create<mlir::arith::MulIOp>(loc, bzPlusOne, chunk);

    kLoop.setLowerBound(kStart);
    kLoop.setUpperBound(kEnd);

    fprintf(stderr, "[SplitKPartition] K-loop partitioned by blockIdx.z "
            "(split_k=%lld)\n", (long long)split_k_factor);
    fflush(stderr);
  }

  int64_t split_k_factor;
};

// ============================================================================
// Phase B (MW-7): K-loop accumulator hoisting.
// Promotes MMA C accumulators from global memory to scf.for iter_args
// (registers). Eliminates N×8 global C loads/stores per K-iteration.
// ============================================================================
struct AccumulatorHoistPass
    : public mlir::PassWrapper<AccumulatorHoistPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccumulatorHoistPass)

  llvm::StringRef getArgument() const override {
    return "matcore-accumulator-hoist";
  }
  llvm::StringRef getDescription() const override {
    return "Hoist MMA C accumulators out of K-loop as iter_args";
  }

  // Check if a value depends on anything defined inside the given block
  // or on the loop induction variable.
  static bool dependsOnLoop(mlir::Value val, mlir::Block *loopBody,
                            mlir::Value iv,
                            llvm::SmallPtrSetImpl<mlir::Value> &visited) {
    if (!visited.insert(val).second)
      return false;
    if (val == iv)
      return true;
    auto *defOp = val.getDefiningOp();
    if (!defOp)
      return false; // block argument (not IV) — loop-invariant
    if (defOp->getBlock() == loopBody) {
      // Defined inside loop — check if its operands depend on loop
      for (mlir::Value operand : defOp->getOperands()) {
        if (dependsOnLoop(operand, loopBody, iv, visited))
          return true;
      }
    }
    return false;
  }

  // Clone a value's definition chain outside the loop.
  // Only works for values that are loop-invariant (don't depend on IV).
  static mlir::Value cloneOutsideLoop(mlir::Value val, mlir::Block *loopBody,
                                      mlir::Value iv,
                                      mlir::OpBuilder &builder,
                                      mlir::IRMapping &mapping) {
    if (mlir::Value mapped = mapping.lookupOrNull(val))
      return mapped;
    auto *defOp = val.getDefiningOp();
    if (!defOp || defOp->getBlock() != loopBody) {
      // Defined outside loop — use directly
      mapping.map(val, val);
      return val;
    }
    // Clone operands first (recursive)
    for (mlir::Value operand : defOp->getOperands()) {
      if (!cloneOutsideLoop(operand, loopBody, iv, builder, mapping))
        return nullptr;
    }
    mlir::Operation *cloned = builder.clone(*defOp, mapping);
    for (auto [oldRes, newRes] :
         llvm::zip(defOp->getResults(), cloned->getResults())) {
      mapping.map(oldRes, newRes);
    }
    return mapping.lookup(val);
  }

  // Represents one MMA result → extract → store pattern.
  struct MmaStorePattern {
    mlir::vector::ExtractOp extract;
    mlir::memref::StoreOp store;
  };

  void runOnOperation() override {
    mlir::Operation *top = getOperation();
    // Find the K-loop: prefer matcore.k_loop attribute (set by TagKLoopPass),
    // fall back to heuristic for legacy non-split paths.
    mlir::scf::ForOp kLoop;
    top->walk([&](mlir::scf::ForOp forOp) {
      if (forOp->hasAttr("matcore.k_loop")) {
        kLoop = forOp;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (!kLoop) {
      // Heuristic fallback: outermost loop with MMA ops (legacy V3 path)
      llvm::SmallVector<mlir::scf::ForOp, 4> allLoops;
      top->walk([&](mlir::scf::ForOp forOp) { allLoops.push_back(forOp); });
      for (auto &f : allLoops) {
        bool hasMma = false;
        f.getBody()->walk([&](mlir::nvgpu::MmaSyncOp) { hasMma = true; });
        if (!hasMma) continue;
        if (!kLoop) { kLoop = f; continue; }
        auto curUb = kLoop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        auto newUb = f.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
        if (curUb && newUb && newUb.value() > curUb.value())
          kLoop = f;
        else if (!curUb && !newUb) {} // both dynamic — keep first
        else if (curUb && !newUb)
          kLoop = f; // prefer dynamic upper bound (split-K K-loop)
      }
    }
    fprintf(stderr, "[AccHoist] K-loop=%s (attr=%s)\n",
            kLoop ? "YES" : "NO",
            kLoop && kLoop->hasAttr("matcore.k_loop") ? "tagged" : "heuristic");
    fflush(stderr);
    if (!kLoop)
      return;

    mlir::Block *oldBody = kLoop.getBody();
    mlir::Value oldIv = kLoop.getInductionVar();

    // Collect MMA ops in program order.
    llvm::SmallVector<mlir::nvgpu::MmaSyncOp, 8> oldMmas;
    for (mlir::Operation &op : *oldBody) {
      if (auto mma = llvm::dyn_cast<mlir::nvgpu::MmaSyncOp>(op))
        oldMmas.push_back(mma);
    }
    if (oldMmas.empty())
      return;
    fprintf(stderr, "[AccHoist] Found %lu MMA ops in K-loop\n",
            (unsigned long)oldMmas.size());
    fflush(stderr);

    // Collect extract→store patterns for each MMA (for post-loop stores).
    llvm::SmallVector<llvm::SmallVector<MmaStorePattern, 4>, 8> mmaStorePatterns;
    mmaStorePatterns.reserve(oldMmas.size());
    for (mlir::nvgpu::MmaSyncOp mma : oldMmas) {
      llvm::SmallVector<MmaStorePattern, 4> patterns;
      for (mlir::Operation *user : mma.getResult().getUsers()) {
        auto extract = llvm::dyn_cast<mlir::vector::ExtractOp>(user);
        if (!extract)
          continue;
        for (mlir::Operation *eu : extract.getResult().getUsers()) {
          auto store = llvm::dyn_cast<mlir::memref::StoreOp>(eu);
          if (store && store.getValueToStore() == extract.getResult())
            patterns.push_back({extract, store});
        }
      }
      if (patterns.empty()) {
        fprintf(stderr, "[AccHoist] BAIL: no extract→store pattern for MMA\n");
        // Check what users the MMA result actually has
        for (mlir::Operation *user : mma.getResult().getUsers()) {
          fprintf(stderr, "[AccHoist]   user: %s\n",
                  user->getName().getStringRef().str().c_str());
        }
        fflush(stderr);
        return;
      }
      mmaStorePatterns.push_back(std::move(patterns));
    }
    fprintf(stderr, "[AccHoist] Found store patterns for all MMAs\n");
    fflush(stderr);

    // Verify store addresses are loop-invariant (don't depend on K IV).
    for (size_t si = 0; si < mmaStorePatterns.size(); si++) {
      for (size_t pi = 0; pi < mmaStorePatterns[si].size(); pi++) {
        MmaStorePattern &pat = mmaStorePatterns[si][pi];
        llvm::SmallPtrSet<mlir::Value, 16> vis;
        if (dependsOnLoop(pat.store.getMemRef(), oldBody, oldIv, vis)) {
          fprintf(stderr, "[AccHoist] BAIL: store[%lu][%lu] memref depends on K-loop IV\n",
                  (unsigned long)si, (unsigned long)pi);
          // Trace why
          mlir::Value memref = pat.store.getMemRef();
          auto *defOp = memref.getDefiningOp();
          if (defOp) {
            fprintf(stderr, "[AccHoist]   memref defined by: %s in block=%p, loopBody=%p\n",
                    defOp->getName().getStringRef().str().c_str(),
                    (void*)defOp->getBlock(), (void*)oldBody);
            for (mlir::Value op : defOp->getOperands()) {
              auto *opDef = op.getDefiningOp();
              bool isIv = (op == oldIv);
              fprintf(stderr, "[AccHoist]     operand: %s isIV=%d\n",
                      opDef ? opDef->getName().getStringRef().str().c_str() : "blockarg",
                      isIv);
            }
          } else {
            fprintf(stderr, "[AccHoist]   memref is blockarg, isIV=%d\n",
                    memref == oldIv);
          }
          fflush(stderr);
          return;
        }
        for (size_t ii = 0; ii < pat.store.getIndices().size(); ii++) {
          mlir::Value idx = pat.store.getIndices()[ii];
          llvm::SmallPtrSet<mlir::Value, 16> ivis;
          if (dependsOnLoop(idx, oldBody, oldIv, ivis)) {
            fprintf(stderr, "[AccHoist] BAIL: store[%lu][%lu] index[%lu] depends on K-loop IV\n",
                    (unsigned long)si, (unsigned long)pi, (unsigned long)ii);
            fflush(stderr);
            return;
          }
        }
      }
    }
    fprintf(stderr, "[AccHoist] Store addresses are loop-invariant ✓\n");
    fflush(stderr);

    // Collect the set of ops that form C-load and C-store chains (to skip
    // during cloning). These are: extract→store from MMA result, and the
    // load→insert chain feeding MMA operand 2.
    // IMPORTANT: Only skip ops whose ALL users are also in the C-chain.
    // Ops shared with non-C code (e.g., subviews also used by inner loops)
    // must NOT be skipped.
    llvm::SmallPtrSet<mlir::Operation *, 32> cChainCandidates;
    for (mlir::nvgpu::MmaSyncOp mma : oldMmas) {
      // Output side: extract + store
      for (mlir::Operation *user : mma.getResult().getUsers()) {
        if (auto extract = llvm::dyn_cast<mlir::vector::ExtractOp>(user)) {
          cChainCandidates.insert(extract);
          for (mlir::Operation *eu : extract.getResult().getUsers()) {
            if (llvm::isa<mlir::memref::StoreOp>(eu))
              cChainCandidates.insert(eu);
          }
        }
      }
      // Input side: trace MMA operand 2 (C accumulator input) back to loads.
      // The C input is typically: splat/insert chain from scalar memref.loads
      mlir::Value cInput = mma.getMatrixC();
      llvm::SmallVector<mlir::Value, 16> worklist;
      worklist.push_back(cInput);
      while (!worklist.empty()) {
        mlir::Value v = worklist.pop_back_val();
        auto *def = v.getDefiningOp();
        if (!def || def->getBlock() != oldBody)
          continue;
        if (cChainCandidates.insert(def).second) {
          for (mlir::Value operand : def->getOperands())
            worklist.push_back(operand);
        }
      }
    }
    
    // Filter: only skip ops whose ALL result users are also safe to skip.
    // Iterate until convergence to handle transitive dependencies.
    llvm::SmallPtrSet<mlir::Operation *, 32> cChainOps;
    for (mlir::Operation *op : cChainCandidates)
      cChainOps.insert(op);
    
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<mlir::Operation *, 16> toRemove;
      for (mlir::Operation *op : cChainOps) {
        for (mlir::Value result : op->getResults()) {
          for (mlir::Operation *user : result.getUsers()) {
            if (!cChainOps.contains(user)) {
              toRemove.push_back(op);
              goto next_op;
            }
          }
        }
        next_op:;
      }
      for (mlir::Operation *op : toRemove) {
        cChainOps.erase(op);
        changed = true;
      }
    }
    // Also exclude the MMA ops themselves from skip set (we MUST clone them)
    for (mlir::nvgpu::MmaSyncOp mma : oldMmas)
      cChainOps.erase(mma);

    fprintf(stderr, "[AccHoist] C-chain: %lu candidates, %lu safe to skip\n",
            (unsigned long)cChainCandidates.size(),
            (unsigned long)cChainOps.size());
    fflush(stderr);

    // ── GROUP MMAs by store target (same C output = same accumulator chain) ──
    // With k_sub>1 (after sub-tile unrolling), MMAs at same (m,n) position but
    // different k offsets must chain: mma_k1.C = mma_k0.result.
    llvm::SmallVector<int, 16> groupOf(oldMmas.size(), -1);
    int numGroups = 0;
    for (size_t i = 0; i < oldMmas.size(); i++) {
      if (groupOf[i] >= 0) continue;
      groupOf[i] = numGroups;
      auto &patsI = mmaStorePatterns[i];
      for (size_t j = i + 1; j < oldMmas.size(); j++) {
        if (groupOf[j] >= 0) continue;
        auto &patsJ = mmaStorePatterns[j];
        if (patsI[0].store.getMemRef() == patsJ[0].store.getMemRef()) {
          auto indicesI = patsI[0].store.getIndices();
          auto indicesJ = patsJ[0].store.getIndices();
          bool match = (indicesI.size() == indicesJ.size());
          for (size_t k = 0; match && k < indicesI.size(); k++)
            match = (indicesI[k] == indicesJ[k]);
          if (match)
            groupOf[j] = numGroups;
        }
      }
      numGroups++;
    }

    llvm::SmallVector<llvm::SmallVector<size_t, 4>, 8> mmaGroups(numGroups);
    for (size_t i = 0; i < oldMmas.size(); i++)
      mmaGroups[groupOf[i]].push_back(i);

    fprintf(stderr, "[AccHoist] Grouped %lu MMAs into %d accumulator chains\n",
            (unsigned long)oldMmas.size(), numGroups);
    fflush(stderr);

    // Create zero-initialized accumulators (one per GROUP, not per MMA).
    mlir::IRRewriter rewriter(kLoop.getContext());
    rewriter.setInsertionPoint(kLoop);
    llvm::SmallVector<mlir::Value, 8> zeroInits;
    for (int g = 0; g < numGroups; g++) {
      size_t firstMma = mmaGroups[g][0];
      auto vecTy = llvm::dyn_cast<mlir::VectorType>(
          oldMmas[firstMma].getResult().getType());
      if (!vecTy)
        return;
      auto zeroAttr = rewriter.getZeroAttr(vecTy);
      zeroInits.push_back(
          rewriter.create<mlir::arith::ConstantOp>(
              oldMmas[firstMma].getLoc(), vecTy, zeroAttr));
    }

    // Build new init args: old iter_args + zero accumulators (numGroups).
    llvm::SmallVector<mlir::Value, 16> newInitArgs(kLoop.getInitArgs());
    newInitArgs.append(zeroInits.begin(), zeroInits.end());
    const unsigned oldIterCount = kLoop.getNumRegionIterArgs();

    // Capture cloned MMA ops during body building.
    llvm::SmallVector<mlir::nvgpu::MmaSyncOp, 16> newMmas;
    auto newFor = rewriter.create<mlir::scf::ForOp>(
        kLoop.getLoc(), kLoop.getLowerBound(), kLoop.getUpperBound(),
        kLoop.getStep(), newInitArgs,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value iv,
            mlir::ValueRange iterArgs) {
          mlir::IRMapping bodyMapping;
          bodyMapping.map(oldIv, iv);
          for (auto [oldArg, newArg] :
               llvm::zip(kLoop.getRegionIterArgs(),
                         iterArgs.take_front(oldIterCount))) {
            bodyMapping.map(oldArg, newArg);
          }

          // Clone body ops, SKIPPING C-chain ops (loads/stores for C).
          for (mlir::Operation &op : oldBody->without_terminator()) {
            if (cChainOps.contains(&op))
              continue;
            mlir::Operation *cloned = b.clone(op, bodyMapping);
            if (auto mma = llvm::dyn_cast<mlir::nvgpu::MmaSyncOp>(cloned))
              newMmas.push_back(mma);
          }

          // Rewire MMA operand 2 (C input) with accumulator chaining.
          // First MMA in group -> group's iter_arg (zero-init on first K-iter).
          // Subsequent MMAs -> predecessor's result (chain accumulation).
          for (int g = 0; g < numGroups; g++) {
            for (size_t pos = 0; pos < mmaGroups[g].size(); pos++) {
              size_t mmaIdx = mmaGroups[g][pos];
              if (mmaIdx >= newMmas.size()) continue;
              if (pos == 0) {
                newMmas[mmaIdx]->setOperand(
                    2, iterArgs[oldIterCount + g]);
              } else {
                size_t prevIdx = mmaGroups[g][pos - 1];
                newMmas[mmaIdx]->setOperand(
                    2, newMmas[prevIdx].getResult());
              }
            }
          }

          // Build yield: old yield values + LAST MMA result per group.
          llvm::SmallVector<mlir::Value, 16> yieldVals;
          auto oldYield =
              llvm::cast<mlir::scf::YieldOp>(oldBody->getTerminator());
          for (mlir::Value v : oldYield.getOperands())
            yieldVals.push_back(bodyMapping.lookup(v));
          for (int g = 0; g < numGroups; g++) {
            size_t lastIdx = mmaGroups[g].back();
            yieldVals.push_back(newMmas[lastIdx].getResult());
          }
          b.create<mlir::scf::YieldOp>(loc, yieldVals);
        });

    if (newMmas.size() != oldMmas.size())
      return; // Safety: mismatch means something went wrong

    // Post-loop stores from LAST MMA in each group (fully accumulated result).
    rewriter.setInsertionPointAfter(newFor);
    mlir::IRMapping postLoopMapping;
    for (int g = 0; g < numGroups; g++) {
      size_t lastMmaIdx = mmaGroups[g].back();
      mlir::Value finalAcc =
          newFor.getResult(kLoop.getNumResults() + g);
      for (MmaStorePattern &pat : mmaStorePatterns[lastMmaIdx]) {
        auto pos = pat.extract.getStaticPosition();
        mlir::Value extracted = rewriter.create<mlir::vector::ExtractOp>(
            pat.store.getLoc(), finalAcc, pos);
        mlir::Value memref = cloneOutsideLoop(
            pat.store.getMemRef(), oldBody, oldIv, rewriter, postLoopMapping);
        if (!memref)
          return;
        llvm::SmallVector<mlir::Value, 4> indices;
        for (mlir::Value oldIdx : pat.store.getIndices()) {
          mlir::Value newIdx = cloneOutsideLoop(oldIdx, oldBody, oldIv,
                                                rewriter, postLoopMapping);
          if (!newIdx)
            return;
          indices.push_back(newIdx);
        }
        auto storeOp = rewriter.create<mlir::memref::StoreOp>(
            pat.store.getLoc(), extracted, memref, indices);
        // Tag for SplitKEpiloguePass to find epilogue stores reliably
        storeOp->setAttr("matcore.epilogue_store",
                         mlir::UnitAttr::get(storeOp->getContext()));
      }
    }

    // Propagate K-loop tag to the new loop
    if (kLoop->hasAttr("matcore.k_loop"))
      newFor->setAttr("matcore.k_loop",
                      mlir::UnitAttr::get(newFor->getContext()));

    // Replace old loop results and erase.
    for (auto [oldRes, newRes] :
         llvm::zip(kLoop.getResults(),
                   newFor.getResults().take_front(kLoop.getNumResults()))) {
      oldRes.replaceAllUsesWith(newRes);
    }
    rewriter.eraseOp(kLoop);
    fprintf(stderr, "[AccHoist] SUCCESS: hoisted %d accumulator groups "
            "(%lu total MMAs) as iter_args\n",
            numGroups, (unsigned long)oldMmas.size());
    fflush(stderr);

  }
};

// ============================================================================
// Phase C (MW-7): Insert gpu.barrier after shared memory copy ops.
// Ensures all warps finish writing smem A/B before any warp reads from them.
// ============================================================================
struct InsertSmemBarriersPass
    : public mlir::PassWrapper<InsertSmemBarriersPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertSmemBarriersPass)

  llvm::StringRef getArgument() const override {
    return "matcore-insert-smem-barriers";
  }
  llvm::StringRef getDescription() const override {
    return "Insert gpu.barrier after shared memory copy ops";
  }
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    llvm::SmallVector<mlir::Operation *, 8> copyOps;
    op->walk([&](mlir::linalg::CopyOp copyOp) {
      copyOps.push_back(copyOp.getOperation());
    });
    for (mlir::Operation *copyOp : copyOps) {
      mlir::OpBuilder builder(copyOp->getContext());
      builder.setInsertionPointAfter(copyOp);
      builder.create<mlir::gpu::BarrierOp>(copyOp->getLoc());
    }
  }
};

// ============================================================================
// Phase F (MW-7): Async vectorized A/B tile copies (global → shared memory).
// Replaces linalg.copy with nvgpu.device_async_copy (CP.ASYNC.CG.SHARED.GLOBAL).
// Each of 128 threads issues one 16-byte async copy per tile, bypassing
// registers entirely (data flows global → L2 → shared memory directly).
// Combines Phase F (vectorize) + Phase D (cp.async) in one pass.
// MUST run after Phase C (barriers already inserted; this pass replaces them
// with proper async commit/wait + barrier).
// ============================================================================
struct VectorizeTileCopyPass
    : public mlir::PassWrapper<VectorizeTileCopyPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeTileCopyPass)

  llvm::StringRef getArgument() const override {
    return "matcore-vectorize-tile-copy";
  }
  llvm::StringRef getDescription() const override {
    return "Replace linalg.copy with nvgpu.device_async_copy (CP.ASYNC)";
  }

  void runOnOperation() override {
    mlir::Operation *top = getOperation();

    // Collect global→workgroup linalg.copy ops
    llvm::SmallVector<mlir::linalg::CopyOp, 4> copyOps;
    top->walk([&](mlir::linalg::CopyOp copy) {
      auto dstType = mlir::cast<mlir::MemRefType>(copy.getOutputs()[0].getType());
      auto dstAS = dstType.getMemorySpace();
      if (!dstAS) return;
      auto gpuAS = mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(dstAS);
      if (!gpuAS || gpuAS.getValue() != mlir::gpu::AddressSpace::Workgroup)
        return;
      copyOps.push_back(copy);
    });
    if (copyOps.empty()) return;

    auto getConstIndex = [](mlir::Value v) -> int64_t {
      if (auto cst = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
        return cst.value();
      return -1;
    };

    // cp.async copies 16 bytes (8 f16) or 8 bytes (4 f16) per instruction.
    // Prefer 16B (.cg, L1 bypass) but fall back to 8B (.ca) when 16B
    // doesn't divide evenly into the number of threads.
    const int64_t preferredVecSize = 8;  // 8 f16 = 16 bytes
    const int64_t fallbackVecSize = 4;   // 4 f16 = 8 bytes
    auto &ctx = *top->getContext();
    auto tokenType = mlir::nvgpu::DeviceAsyncTokenType::get(&ctx);

    llvm::SmallVector<mlir::Value, 4> asyncTokens;
    llvm::SmallVector<mlir::linalg::CopyOp, 4> replacedCopies;
    mlir::Operation *lastNewOp = nullptr;
    int replaced = 0;

    for (mlir::linalg::CopyOp copy : copyOps) {
      // Resolve enclosing gpu.launch for this specific copy
      auto launch = copy->getParentOfType<mlir::gpu::LaunchOp>();
      if (!launch) continue;
      int64_t blockDimX = getConstIndex(launch.getBlockSizeX());
      int64_t blockDimY = getConstIndex(launch.getBlockSizeY());
      if (blockDimX < 0 || blockDimY < 0) continue;
      int64_t numThreads = blockDimX * blockDimY;
      mlir::Value src = copy.getInputs()[0];
      mlir::Value dst = copy.getOutputs()[0];

      auto srcType = mlir::cast<mlir::MemRefType>(src.getType());
      auto dstType = mlir::cast<mlir::MemRefType>(dst.getType());

      // Validate shape: 2D, f16, inner stride 1
      auto shape = srcType.getShape();
      if (shape.size() != 2 || !srcType.getElementType().isF16())
        continue;

      int64_t rows = shape[0], cols = shape[1];

      // Verify inner stride is 1 (contiguous elements for cp.async)
      int64_t offset;
      llvm::SmallVector<int64_t, 2> srcStrides;
      if (mlir::failed(mlir::getStridesAndOffset(srcType, srcStrides, offset)))
        continue;
      if (srcStrides.back() != 1) continue;

      // Try preferred 16B vec, fall back to 8B if it doesn't divide evenly
      int64_t vecSize = preferredVecSize;
      if (cols % vecSize != 0 || (rows * (cols / vecSize)) % numThreads != 0) {
        vecSize = fallbackVecSize;
      }
      if (cols % vecSize != 0) continue;

      int64_t vecsPerRow = cols / vecSize;
      int64_t totalVecs = rows * vecsPerRow;
      if (totalVecs % numThreads != 0) continue;
      int64_t vecsPerThread = totalVecs / numThreads;

      fprintf(stderr, "[VecCopy] Async copy %ldx%ld: %ld vecs/thread, "
              "vec_size=%ld (cp.async 16B)\n",
              rows, cols, vecsPerThread, vecSize);
      fflush(stderr);

      mlir::OpBuilder b(copy);
      mlir::Location loc = copy.getLoc();

      // Thread IDs → linear thread index
      auto tidX = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
      auto tidY = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::y);
      auto blockDimXVal = b.create<mlir::arith::ConstantIndexOp>(loc, blockDimX);
      auto linearTid = b.create<mlir::arith::AddIOp>(loc,
          b.create<mlir::arith::MulIOp>(loc, tidY, blockDimXVal), tidX);

      auto vecsPerRowVal = b.create<mlir::arith::ConstantIndexOp>(loc, vecsPerRow);
      auto vecSizeVal = b.create<mlir::arith::ConstantIndexOp>(loc, vecSize);
      auto dstElements = b.getIntegerAttr(b.getIndexType(), vecSize);

      for (int64_t v = 0; v < vecsPerThread; ++v) {
        mlir::Value vecIdx;
        if (vecsPerThread == 1) {
          vecIdx = linearTid;
        } else {
          auto off = b.create<mlir::arith::ConstantIndexOp>(loc, v * numThreads);
          vecIdx = b.create<mlir::arith::AddIOp>(loc, linearTid, off);
        }

        // row = vecIdx / vecsPerRow, col = (vecIdx % vecsPerRow) * vecSize
        auto row = b.create<mlir::arith::DivUIOp>(loc, vecIdx, vecsPerRowVal);
        auto colVec = b.create<mlir::arith::RemUIOp>(loc, vecIdx, vecsPerRowVal);
        auto col = b.create<mlir::arith::MulIOp>(loc, colVec, vecSizeVal);

        // nvgpu.device_async_copy: src[row,col] → dst[row,col]
        // 16B: CP.ASYNC.CG.SHARED.GLOBAL (L1 bypass)
        //  8B: CP.ASYNC.CA.SHARED.GLOBAL (cache all)
        bool useBypassL1 = (vecSize == 8);  // CG mode only for 16-byte copies
        auto token = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc, tokenType,
            dst, mlir::ValueRange{row, col},   // shared mem destination
            src, mlir::ValueRange{row, col},   // global mem source
            dstElements,
            /*srcElements=*/mlir::Value(),     // no partial copy
            /*bypassL1=*/useBypassL1 ? mlir::UnitAttr::get(b.getContext())
                                     : mlir::UnitAttr());

        asyncTokens.push_back(token.getAsyncToken());
        lastNewOp = token.getOperation();
      }

      ++replaced;
      replacedCopies.push_back(copy);
    }

    if (replaced == 0) return;

    // After all async copies: commit group → wait → barrier
    mlir::OpBuilder b(lastNewOp);
    b.setInsertionPointAfter(lastNewOp);
    auto loc = lastNewOp->getLoc();

    // Commit all pending async copies into one group
    auto groupToken = b.create<mlir::nvgpu::DeviceAsyncCreateGroupOp>(
        loc, tokenType, asyncTokens);

    // Wait for the group to complete (numGroups=0 means wait for ALL pending)
    b.create<mlir::nvgpu::DeviceAsyncWaitOp>(
        loc, groupToken.getResult(),
        b.getI32IntegerAttr(0));

    // Barrier: all threads must see complete shared memory before MMA reads
    b.create<mlir::gpu::BarrierOp>(loc);

    // Collect Phase C barriers adjacent to successfully replaced copies only
    llvm::SmallVector<mlir::gpu::BarrierOp, 4> barriersToRemove;
    for (auto copy : replacedCopies) {
      auto *next = copy->getNextNode();
      if (next) {
        if (auto barrier = mlir::dyn_cast<mlir::gpu::BarrierOp>(next))
          barriersToRemove.push_back(barrier);
      }
    }

    // Remove Phase C barriers (now superseded by async wait + new barrier)
    for (auto barrier : barriersToRemove)
      barrier->erase();

    // Erase only successfully replaced linalg.copy ops
    for (auto copy : replacedCopies)
      copy->erase();

    fprintf(stderr, "[VecCopy] Replaced %d linalg.copy ops with "
            "nvgpu.device_async_copy (CP.ASYNC)\n", replaced);
    fflush(stderr);
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// MW-7 Phase H: Shared Memory Bank Conflict Avoidance (Padding)
//
// Adds +1 column padding to shared memory buffers to break bank conflict
// stride patterns. For memref<64x16xf16, wg>, every 4 rows (32 bytes stride
// × 4 = 128 = 32 banks cycle) map to the same banks. With padding to
// memref<64x17xf16, wg>, the stride becomes 34 bytes, breaking the cycle.
//
// This pass modifies workgroup attribution types and propagates the type
// change through all SubViewOp chains. Must run BEFORE Phase E (which
// creates smemA1/smemB1 based on existing attribution types).
// ═══════════════════════════════════════════════════════════════════════════
struct PadSharedMemoryPass
    : public mlir::PassWrapper<PadSharedMemoryPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PadSharedMemoryPass)

  llvm::StringRef getArgument() const override {
    return "matcore-pad-shared-memory";
  }
  llvm::StringRef getDescription() const override {
    return "Pad shared memory to avoid bank conflicts (+1 column)";
  }

  void runOnOperation() override {
    auto *top = getOperation();

    // Find the single gpu.launch (assert uniqueness for safety)
    mlir::gpu::LaunchOp launch;
    top->walk([&](mlir::gpu::LaunchOp op) {
      assert(!launch && "PadSharedMemoryPass: multiple gpu.launch ops");
      launch = op;
    });
    if (!launch) return;

    auto wgAttrs = launch.getWorkgroupAttributions();
    if (wgAttrs.empty()) return;

    // Guard: only pad when we see exactly 2 rank-2 attrs (smemA + smemB)
    unsigned rank2Count = 0;
    for (auto attr : wgAttrs) {
      if (mlir::cast<mlir::MemRefType>(attr.getType()).getRank() == 2)
        ++rank2Count;
    }
    if (rank2Count != 2) {
      fprintf(stderr, "[PadSmem] Expected 2 rank-2 attrs, found %u — skipping\n",
              rank2Count);
      fflush(stderr);
      return;
    }

    bool changed = false;
    for (auto attr : wgAttrs) {
      auto memType = mlir::cast<mlir::MemRefType>(attr.getType());
      if (memType.getRank() != 2) continue;
      auto shape = memType.getShape();
      int64_t lastDim = shape.back();

      // Compute element size in bytes for alignment
      unsigned elemBits = memType.getElementTypeBitWidth();
      unsigned elemBytes = (elemBits + 7) / 8;

      // cp.async requires 16-byte aligned destinations.
      // Pad by the minimum aligned amount that also breaks bank conflicts.
      // For f16 (2B): 16/2 = 8 elements alignment unit.
      unsigned alignElements = 16 / elemBytes;
      if (alignElements < 1) alignElements = 1;

      // Pad by one alignment unit (e.g., 8 f16 elements = 16 bytes)
      // This breaks the bank conflict stride while maintaining cp.async alignment.
      int64_t padAmount = alignElements;
      int64_t paddedDim = lastDim + padAmount;

      llvm::SmallVector<int64_t, 2> paddedShape{shape[0], paddedDim};
      auto paddedType = mlir::MemRefType::get(
          paddedShape, memType.getElementType(),
          mlir::AffineMap(), memType.getMemorySpace());

      attr.setType(paddedType);

      // Propagate type change through all SubViewOp chains rooted at this attr
      propagateTypeChange(attr);

      fprintf(stderr, "[PadSmem] %ldx%ld -> %ldx%ld (pad=%ld, %dB-aligned)\n",
              shape[0], shape[1], paddedShape[0], paddedShape[1],
              padAmount, alignElements * elemBytes);
      fflush(stderr);
      changed = true;
    }

    if (!changed) {
      fprintf(stderr, "[PadSmem] No workgroup attrs to pad\n");
      fflush(stderr);
    }
  }

private:
  void propagateTypeChange(mlir::Value root) {
    // BFS: find all SubViewOps that transitively depend on root as source,
    // and update their result types to reflect the new source strides.
    llvm::SmallVector<mlir::Value, 8> worklist;
    worklist.push_back(root);

    while (!worklist.empty()) {
      auto val = worklist.pop_back_val();
      for (auto &use : val.getUses()) {
        auto *user = use.getOwner();
        if (auto sv = mlir::dyn_cast<mlir::memref::SubViewOp>(user)) {
          auto sourceType =
              mlir::cast<mlir::MemRefType>(sv.getSource().getType());
          auto newResultType = mlir::cast<mlir::MemRefType>(
              mlir::memref::SubViewOp::inferResultType(
                  sourceType, sv.getStaticOffsets(), sv.getStaticSizes(),
                  sv.getStaticStrides()));
          sv.getResult().setType(newResultType);
          // Continue propagation through this subview's users
          worklist.push_back(sv.getResult());
        }
      }
    }
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// MW-7 Phase E: Double-Buffer K-Loop (Software Pipelining)
//
// Transforms the K-loop to use ping-pong shared memory buffers with cp.async
// overlap. After this pass, each iteration's MMA compute overlaps with the
// NEXT iteration's memory prefetch, hiding global memory latency.
//
// Input pattern  (post-Phase-F):
//   gpu.launch workgroup(smemA0, smemB0)
//     scf.for k = 0 to K step 16 iter_args(acc0..3)
//       srcA = subview globalA[0, k]; srcB = subview globalB[k, 0]
//       async_copy srcA → smemA0; async_copy srcB → smemB0
//       create_group; wait(group, 0); barrier
//       ... MMA ops using smemA0, smemB0 ...
//       barrier; yield acc0..3
//
// Output pattern (double-buffered):
//   gpu.launch workgroup(smemA0, smemB0, smemA1, smemB1)
//     // Prologue: prefetch k=0 (no wait)
//     async_copy globalA[0,0] → smemA0; async_copy globalB[0,0] → smemB0
//     create_group → prologue_token
//
//     scf.for k iter_args(acc0..3, readA, readB, writeA, writeB, groupToken)
//       // TOP: wait for previous prefetch
//       wait(groupToken, 0); barrier
//       // PREFETCH: issue next tile (overlapped with compute below)
//       k_next = k + step
//       token = scf.if(k_next < K) {
//         async_copy globalA[0,k_next] → writeA
//         async_copy globalB[k_next,0] → writeB
//         create_group → tok; yield tok
//       } else { empty_group → dummy; yield dummy }
//       // COMPUTE: MMA on readA/readB (overlaps with prefetch!)
//       ... MMA ops using readA, readB ...
//       yield acc, writeA, writeB, readA, readB, token  // swap buffers
// ═══════════════════════════════════════════════════════════════════════════
struct DoubleBufferKLoopPass
    : public mlir::PassWrapper<DoubleBufferKLoopPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DoubleBufferKLoopPass)

  llvm::StringRef getArgument() const override {
    return "matcore-double-buffer-kloop";
  }
  llvm::StringRef getDescription() const override {
    return "Double-buffer K-loop: ping-pong shared memory with cp.async overlap";
  }

  void runOnOperation() override {
    auto *top = getOperation();
    auto *ctx = top->getContext();

    // ── 1. Find gpu.launch ──────────────────────────────────────────────
    mlir::gpu::LaunchOp launch;
    top->walk([&](mlir::gpu::LaunchOp op) { launch = op; });
    if (!launch) return;

    // ── 2. Get existing workgroup attributions (smemA0, smemB0) ─────────
    auto wgAttrs = launch.getWorkgroupAttributions();
    if (wgAttrs.size() < 2) return;
    auto smemA0 = wgAttrs[0];
    auto smemB0 = wgAttrs[1];
    auto smemAType = mlir::cast<mlir::MemRefType>(smemA0.getType());
    auto smemBType = mlir::cast<mlir::MemRefType>(smemB0.getType());
    auto loc = launch.getLoc();

    // ── 3. Add new workgroup attributions (smemA1, smemB1) ──────────────
    auto smemA1 = launch.addWorkgroupAttribution(smemAType, loc);
    auto smemB1 = launch.addWorkgroupAttribution(smemBType, loc);

    fprintf(stderr, "[DoubleBuffer] Added smemA1(%ldx%ld), smemB1(%ldx%ld) "
            "workgroup attrs\n",
            smemAType.getShape()[0], smemAType.getShape()[1],
            smemBType.getShape()[0], smemBType.getShape()[1]);
    fflush(stderr);

    // ── 4. Find K-loop: outermost scf.for containing async copies ──
    // After AccHoist, the K-loop carries accumulator iter_args. The count
    // depends on configuration (8 for 4-warp, up to 32 for 16-warp).
    // Prefer matcore.k_loop attribute (set by TagKLoopPass); fall back to
    // async-copy heuristic for legacy paths.
    mlir::scf::ForOp kLoop;
    launch.getBody().walk([&](mlir::scf::ForOp forOp) {
      if (forOp->hasAttr("matcore.k_loop")) {
        kLoop = forOp;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (!kLoop) {
      // Heuristic fallback: outermost loop with async copies
      launch.getBody().walk([&](mlir::scf::ForOp forOp) {
        bool hasAsyncCopy = false;
        forOp.getBody()->walk([&](mlir::nvgpu::DeviceAsyncCopyOp) {
          hasAsyncCopy = true;
        });
        if (hasAsyncCopy && forOp.getNumResults() > 0) {
          if (!kLoop) {
            kLoop = forOp;
          } else {
            auto curUb = kLoop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
            auto newUb = forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
            if (curUb && newUb && newUb.value() > curUb.value())
              kLoop = forOp;
            else if (curUb && !newUb)
              kLoop = forOp; // prefer dynamic upper bound (split-K)
          }
        }
        return mlir::WalkResult::advance();
      });
    }
    if (!kLoop) {
      fprintf(stderr, "[DoubleBuffer] No K-loop with async copies found\n");
      fflush(stderr);
      return;
    }
    fprintf(stderr, "[DoubleBuffer] Found K-loop with %u iter_args\n",
            kLoop.getNumResults());
    fflush(stderr);

    // ── 5. Collect ops in K-loop body ───────────────────────────────────
    auto *kBody = kLoop.getBody();
    auto kIV = kLoop.getInductionVar();

    llvm::SmallVector<mlir::memref::SubViewOp, 2> srcSubviews;
    llvm::SmallVector<mlir::nvgpu::DeviceAsyncCopyOp, 2> asyncCopies;
    mlir::nvgpu::DeviceAsyncCreateGroupOp createGroupOp;
    mlir::nvgpu::DeviceAsyncWaitOp waitOp;
    llvm::SmallVector<mlir::gpu::BarrierOp, 4> barriers;

    for (auto &op : *kBody) {
      if (auto sv = mlir::dyn_cast<mlir::memref::SubViewOp>(&op)) {
        for (auto operand : sv->getOperands())
          if (operand == kIV) { srcSubviews.push_back(sv); break; }
      }
      if (auto ac = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncCopyOp>(&op))
        asyncCopies.push_back(ac);
      if (auto cg = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncCreateGroupOp>(&op))
        createGroupOp = cg;
      if (auto w = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncWaitOp>(&op))
        waitOp = w;
      if (auto b = mlir::dyn_cast<mlir::gpu::BarrierOp>(&op))
        barriers.push_back(b);
    }

    if (srcSubviews.size() < 2 || asyncCopies.size() < 2 ||
        !createGroupOp || !waitOp || barriers.size() < 2) {
      fprintf(stderr, "[DoubleBuffer] Unexpected loop body structure "
              "(srcSV=%lu, asyncCP=%lu, barriers=%lu), skipping\n",
              srcSubviews.size(), asyncCopies.size(), barriers.size());
      fflush(stderr);
      return;
    }

    fprintf(stderr, "[DoubleBuffer] Loop body: %lu srcSubviews, %lu asyncCopies, "
            "%lu barriers\n", srcSubviews.size(), asyncCopies.size(),
            barriers.size());
    fflush(stderr);

    auto postCopyBarrier = barriers[0];
    auto endBarrier = barriers.back();

    // ── 6. BUILD PROLOGUE (before K-loop, NO wait/barrier) ──────────────
    // Clone the prefetch sequence (srcSubviews → asyncCopies → createGroup)
    // with K mapped to the loop's lower bound. No wait or barrier — the
    // first loop iteration's "wait at top" covers the prologue copies.
    // Guard: skip if loop is zero-trip (defensive — MW path guarantees K≥384).
    mlir::IRRewriter rewriter(ctx);

    // Verify loop is non-empty (lb < ub) — required for prologue correctness
    {
      auto lbCst = kLoop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto ubCst = kLoop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (lbCst && ubCst && lbCst.value() >= ubCst.value()) {
        fprintf(stderr, "[DoubleBuffer] Zero-trip K-loop (lb=%ld >= ub=%ld), skipping\n",
                (long)lbCst.value(), (long)ubCst.value());
        fflush(stderr);
        return;
      }
    }

    rewriter.setInsertionPoint(kLoop);

    mlir::IRMapping prologueMap;
    prologueMap.map(kIV, kLoop.getLowerBound());

    auto startIt = mlir::Block::iterator(srcSubviews[0].getOperation());
    auto endIt = std::next(
        mlir::Block::iterator(createGroupOp.getOperation()));
    for (auto it = startIt; it != endIt; ++it) {
      rewriter.clone(*it, prologueMap);
    }
    // Retrieve prologue's create_group token via the IRMapping
    auto prologueGroupToken =
        prologueMap.lookupOrDefault(createGroupOp.getAsyncToken());

    fprintf(stderr, "[DoubleBuffer] Prologue: k=lb prefetch issued (no wait)\n");
    fflush(stderr);

    // ── 7. replaceWithAdditionalYields ──────────────────────────────────
    // Add 5 new iter_args:
    //   readA(init=smemA0), readB(init=smemB0),
    //   writeA(init=smemA1), writeB(init=smemB1),
    //   groupToken(init=prologueGroupToken)
    //
    // replaceInitOperandUsesInLoop=true: all uses of smemA0→readA, smemB0→readB
    // inside the loop body (compute subviews will read from correct buffer).
    //
    // yieldFn: swap read/write buffers; token is a placeholder (replaced later).
    auto yieldFn = [](mlir::OpBuilder &b, mlir::Location loc,
                      mlir::ArrayRef<mlir::BlockArgument> newBBArgs)
        -> mlir::SmallVector<mlir::Value> {
      // [0]=readA, [1]=readB, [2]=writeA, [3]=writeB, [4]=groupToken
      // Swap: next readA←writeA, readB←writeB, writeA←readA, writeB←readB
      // Token: yield back same (placeholder — replaced after scf.if creation)
      return {newBBArgs[2], newBBArgs[3], newBBArgs[0], newBBArgs[1],
              newBBArgs[4]};
    };

    auto result = kLoop.replaceWithAdditionalYields(
        rewriter,
        mlir::ValueRange{smemA0, smemB0, smemA1, smemB1, prologueGroupToken},
        /*replaceInitOperandUsesInLoop=*/true,
        yieldFn);

    if (mlir::failed(result)) {
      fprintf(stderr, "[DoubleBuffer] replaceWithAdditionalYields FAILED\n");
      fflush(stderr);
      signalPassFailure();
      return;
    }

    auto newKLoop = mlir::cast<mlir::scf::ForOp>((*result).getOperation());
    auto *newBody = newKLoop.getBody();
    auto newIV = newKLoop.getInductionVar();

    // Propagate K-loop tag for SplitKEpiloguePass
    if (kLoop->hasAttr("matcore.k_loop"))
      newKLoop->setAttr("matcore.k_loop",
                         mlir::UnitAttr::get(newKLoop->getContext()));

    // Block args: [0]=IV, [1..N]=acc0..acc(N-1), [N+1..N+5]=readA,readB,writeA,writeB,groupToken
    unsigned numAccIters = newKLoop.getNumRegionIterArgs() - 5;
    unsigned base = 1 + numAccIters;
    auto readA      = newBody->getArgument(base + 0);
    auto readB      = newBody->getArgument(base + 1);
    auto writeA     = newBody->getArgument(base + 2);
    auto writeB     = newBody->getArgument(base + 3);
    auto groupToken = newBody->getArgument(base + 4);

    fprintf(stderr, "[DoubleBuffer] replaceWithAdditionalYields OK, "
            "%d iter_args\n", (int)newKLoop.getNumResults());
    fflush(stderr);

    // ── 8. Re-collect ops in NEW loop body ──────────────────────────────
    // After replaceWithAdditionalYields, ops were moved (pointers valid) but
    // we re-collect to be safe and use the new induction variable.
    srcSubviews.clear();
    asyncCopies.clear();
    createGroupOp = nullptr;
    waitOp = nullptr;
    barriers.clear();

    for (auto &op : *newBody) {
      if (auto sv = mlir::dyn_cast<mlir::memref::SubViewOp>(&op)) {
        for (auto operand : sv->getOperands())
          if (operand == newIV) { srcSubviews.push_back(sv); break; }
      }
      if (auto ac = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncCopyOp>(&op))
        asyncCopies.push_back(ac);
      if (auto cg = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncCreateGroupOp>(&op))
        createGroupOp = cg;
      if (auto w = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncWaitOp>(&op))
        waitOp = w;
      if (auto b = mlir::dyn_cast<mlir::gpu::BarrierOp>(&op))
        barriers.push_back(b);
    }

    if (srcSubviews.size() < 2 || asyncCopies.size() < 2 ||
        !createGroupOp || !waitOp || barriers.size() < 2) {
      fprintf(stderr, "[DoubleBuffer] Post-rewrite: unexpected structure "
              "(srcSV=%lu, asyncCP=%lu, barriers=%lu)\n",
              srcSubviews.size(), asyncCopies.size(), barriers.size());
      fflush(stderr);
      signalPassFailure();
      return;
    }
    postCopyBarrier = barriers[0];
    endBarrier = barriers.back();

    // ── 9. Insert wait + barrier at TOP of loop body ────────────────────
    // This waits for the previous iteration's prefetch (or the prologue
    // on the first iteration). Placed before any other ops.
    rewriter.setInsertionPointToStart(newBody);
    rewriter.create<mlir::nvgpu::DeviceAsyncWaitOp>(
        loc, groupToken,
        rewriter.getI32IntegerAttr(0));  // wait_group(0) = wait for ALL
    rewriter.create<mlir::gpu::BarrierOp>(loc);

    // ── 10. Find first compute op (warp subview using readA/readB) ──────
    mlir::Operation *firstComputeOp = nullptr;
    for (auto &op : *newBody) {
      if (auto sv = mlir::dyn_cast<mlir::memref::SubViewOp>(&op)) {
        if (sv.getSource() == readA || sv.getSource() == readB) {
          firstComputeOp = &op;
          break;
        }
      }
    }
    if (!firstComputeOp) {
      fprintf(stderr, "[DoubleBuffer] Cannot find compute section "
              "(no warp subview of readA/readB)\n");
      fflush(stderr);
      signalPassFailure();
      return;
    }

    // ── 11. Insert scf.if (conditional prefetch) before compute ─────────
    // This is placed BEFORE the MMA section so the async copies can overlap
    // with the MMA compute phase that follows.
    rewriter.setInsertionPoint(firstComputeOp);

    auto kStep  = newKLoop.getStep();
    auto kUpper = newKLoop.getUpperBound();
    auto kNext  = rewriter.create<mlir::arith::AddIOp>(loc, newIV, kStep);
    auto hasNext = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, kNext, kUpper);

    auto tokenType = mlir::nvgpu::DeviceAsyncTokenType::get(ctx);
    auto ifOp = rewriter.create<mlir::scf::IfOp>(
        loc, mlir::TypeRange{tokenType}, hasNext,
        /*withElseRegion=*/true);

    // ── 11a. THEN block: prefetch next K-tile ───────────────────────────
    // Clone the entire prefetch sequence (srcSubviews → address math →
    // asyncCopies → createGroup) with k → k_next and read→write buffers.
    // This handles any number of async copies per tile (1, 2, etc.).
    {
      auto &thenBlock = ifOp.getThenRegion().front();
      rewriter.setInsertionPointToStart(&thenBlock);

      mlir::IRMapping ifMap;
      ifMap.map(newIV, kNext.getResult());
      ifMap.map(readA, writeA);
      ifMap.map(readB, writeB);

      auto startIt = mlir::Block::iterator(srcSubviews[0].getOperation());
      auto endIt = std::next(
          mlir::Block::iterator(createGroupOp.getOperation()));
      mlir::Value newGroupToken;
      for (auto it = startIt; it != endIt; ++it) {
        auto *cloned = rewriter.clone(*it, ifMap);
        if (auto cg = mlir::dyn_cast<mlir::nvgpu::DeviceAsyncCreateGroupOp>(cloned))
          newGroupToken = cg.getAsyncToken();
      }

      if (!newGroupToken) {
        fprintf(stderr, "[DoubleBuffer] THEN block: no create_group in cloned ops\n");
        fflush(stderr);
        signalPassFailure();
        return;
      }

      rewriter.create<mlir::scf::YieldOp>(
          loc, mlir::ValueRange{newGroupToken});
    }

    // ── 11b. ELSE block: empty group (no pending copies, harmless token) ─
    {
      auto &elseBlock = ifOp.getElseRegion().front();
      rewriter.setInsertionPointToStart(&elseBlock);

      auto dummyGroup = rewriter.create<mlir::nvgpu::DeviceAsyncCreateGroupOp>(
          loc, tokenType, mlir::ValueRange{});

      rewriter.create<mlir::scf::YieldOp>(
          loc, mlir::ValueRange{dummyGroup.getAsyncToken()});
    }

    // ── 12. Replace yield's token placeholder with scf.if result ────────
    // The yield has: [0..N-1]=acc, [N..N+3]=buffer swap, [N+4]=token placeholder
    auto *yieldOp = newBody->getTerminator();
    yieldOp->setOperand(numAccIters + 4, ifOp.getResult(0));

    // ── 13. Erase old prefetch ops (reverse dependency order) ───────────
    // wait → postCopyBarrier → createGroup → asyncCopies → srcSubviews
    // + endBarrier (replaced by the new wait+barrier at top of next iter)
    rewriter.eraseOp(waitOp);
    rewriter.eraseOp(postCopyBarrier);
    rewriter.eraseOp(createGroupOp);
    for (auto &ac : asyncCopies) rewriter.eraseOp(ac);
    for (auto &sv : srcSubviews) rewriter.eraseOp(sv);
    rewriter.eraseOp(endBarrier);

    fprintf(stderr, "[DoubleBuffer] Phase E: double-buffering applied\n"
            "  - 4 workgroup attrs (2× ping-pong buffers)\n"
            "  - %d iter_args (%d acc + 4 memref + 1 token)\n"
            "  - %lu async copies per K-step\n"
            "  - wait-at-top for compute/prefetch overlap\n"
            "  - scf.if conditional last-iteration guard\n",
            (int)newKLoop.getNumResults(), (int)numAccIters,
            asyncCopies.size());
    fflush(stderr);
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// LdMatrixRewritePass — Replace scalar shared memory loads + vector.insert
// chains with warp-cooperative nvgpu.ldmatrix operations.
//
// Before: 64 × LDS.U16 (scalar) + 600+ address math ops
// After:  ~12 × LDSM (ldmatrix.sync.aligned) + ~20 address math ops
//
// Pattern matched (per MMA operand):
//   A fragment: 8 memref.load → vector.splat → 8 vector.insert → vector<4x2xf16>
//     → replaced by nvgpu.ldmatrix(x4, transpose=false)
//   B fragment: 4 memref.load → vector.splat → 4 vector.insert → vector<2x2xf16>
//     → replaced by nvgpu.ldmatrix(x2, transpose=true)
//
// Alignment guarantee (verified for padded smem layout):
//   smemA stride=40 → row=80 bytes (5×16B), col offset 0 or 8 (0 or 16B)
//   smemB stride=136 → row=272 bytes (17×16B), col offset always 0
//   All per-thread ldmatrix addresses are 16-byte aligned.
//
// Must run AFTER DoubleBufferKLoopPass + CSE (scalar loads have final
// subview references). Subsequent CSE+canonicalize cleans dead code.
// ═══════════════════════════════════════════════════════════════════════════
struct LdMatrixRewritePass
    : public mlir::PassWrapper<LdMatrixRewritePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LdMatrixRewritePass)

  llvm::StringRef getArgument() const override {
    return "matcore-ldmatrix-rewrite";
  }
  llvm::StringRef getDescription() const override {
    return "Replace scalar smem loads with nvgpu.ldmatrix for MMA operands";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::nvgpu::NVGPUDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto *rootOp = getOperation();
    mlir::DenseMap<mlir::Value, mlir::Value> cache;
    unsigned numReplaced = 0;

    rootOp->walk([&](mlir::nvgpu::MmaSyncOp mmaOp) {
      if (auto newA = getOrCreateLdMatrix(mmaOp.getMatrixA(), /*isA=*/true,
                                          cache, mmaOp)) {
        mmaOp.getMatrixAMutable().set(newA);
        ++numReplaced;
      }
      if (auto newB = getOrCreateLdMatrix(mmaOp.getMatrixB(), /*isA=*/false,
                                          cache, mmaOp)) {
        mmaOp.getMatrixBMutable().set(newB);
        ++numReplaced;
      }
    });

    if (numReplaced > 0)
      llvm::errs() << "[LdMatrixRewrite] Replaced " << numReplaced
                    << " MMA operand(s) with ldmatrix\n";
  }

private:
  // Find gpu.thread_id x in the enclosing gpu.launch
  mlir::Value findThreadIdX(mlir::Operation *op) {
    auto launch = op->getParentOfType<mlir::gpu::LaunchOp>();
    if (!launch) return nullptr;
    mlir::Value result = nullptr;
    launch.getBody().walk([&](mlir::gpu::ThreadIdOp tidOp) {
      if (tidOp.getDimension() == mlir::gpu::Dimension::x) {
        result = tidOp.getResult();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return result;
  }

  // Trace a vector value back through vector.insert chain to the vector.splat.
  // Returns the SplatOp if found, nullptr otherwise.
  mlir::vector::SplatOp traceSplat(mlir::Value vec) {
    mlir::Operation *cur = vec.getDefiningOp();
    while (cur) {
      auto insertOp = mlir::dyn_cast_or_null<mlir::vector::InsertOp>(cur);
      if (!insertOp) return nullptr;
      auto dest = insertOp.getDest();
      if (auto splat = dest.getDefiningOp<mlir::vector::SplatOp>())
        return splat;
      cur = dest.getDefiningOp();
    }
    return nullptr;
  }

  // Get the source shared memory subview from a scalar load feeding the splat
  mlir::memref::LoadOp getWorkgroupLoad(mlir::vector::SplatOp splatOp) {
    auto loadOp =
        splatOp.getInput().getDefiningOp<mlir::memref::LoadOp>();
    if (!loadOp) return nullptr;
    auto memrefType = loadOp.getMemRefType();
    auto gpuAS = mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(
        memrefType.getMemorySpace());
    if (!gpuAS || gpuAS.getValue() != mlir::gpu::AddressSpace::Workgroup)
      return nullptr;
    return loadOp;
  }

  mlir::Value getOrCreateLdMatrix(mlir::Value operand, bool isA,
                                  mlir::DenseMap<mlir::Value, mlir::Value> &cache,
                                  mlir::nvgpu::MmaSyncOp mmaOp) {
    auto it = cache.find(operand);
    if (it != cache.end())
      return it->second;

    // Trace insert chain → splat → load
    auto splatOp = traceSplat(operand);
    if (!splatOp) return nullptr;

    auto loadOp = getWorkgroupLoad(splatOp);
    if (!loadOp) return nullptr;

    mlir::Value threadIdX = findThreadIdX(mmaOp);
    if (!threadIdX) return nullptr;

    mlir::OpBuilder builder(splatOp);
    auto loc = splatOp.getLoc();

    // Create lane_id = thread_id_x % 32 (CSE will merge with existing)
    auto c32 = builder.create<mlir::arith::ConstantIndexOp>(loc, 32);
    mlir::Value laneId =
        builder.create<mlir::arith::RemUIOp>(loc, threadIdX, c32);

    // row = lane_id % 16 (shared by both A and B)
    auto c16 = builder.create<mlir::arith::ConstantIndexOp>(loc, 16);
    mlir::Value row =
        builder.create<mlir::arith::RemUIOp>(loc, laneId, c16);

    mlir::Value srcMemref = loadOp.getMemRef();
    mlir::Value result;

    if (isA) {
      // A fragment: ldmatrix.x4, transpose=false
      // Address: [lane_id % 16, (lane_id / 16) * 8]
      // Each thread loads 16 bytes (8 f16) from one row of the 16×16 tile
      auto c8 = builder.create<mlir::arith::ConstantIndexOp>(loc, 8);
      mlir::Value halfWarp =
          builder.create<mlir::arith::DivUIOp>(loc, laneId, c16);
      mlir::Value col =
          builder.create<mlir::arith::MulIOp>(loc, halfWarp, c8);
      auto resType = mlir::VectorType::get({4, 2}, builder.getF16Type());
      result = builder.create<mlir::nvgpu::LdMatrixOp>(
          loc, resType, srcMemref, mlir::ValueRange{row, col},
          /*transpose=*/false, /*numTiles=*/static_cast<uint32_t>(4));
    } else {
      // B fragment: ldmatrix.x2, transpose=true
      // Address: [lane_id % 16, 0]
      // Transpose needed: B is stored [K][N] row-major, MMA needs
      // K-indexed register layout
      auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto resType = mlir::VectorType::get({2, 2}, builder.getF16Type());
      result = builder.create<mlir::nvgpu::LdMatrixOp>(
          loc, resType, srcMemref, mlir::ValueRange{row, c0},
          /*transpose=*/true, /*numTiles=*/static_cast<uint32_t>(2));
    }

    cache[operand] = result;
    return result;
  }
};

// ============================================================================
// Phase K: SplitKEpiloguePass — rewrite epilogue stores for Split-K.
// When split_k > 1, each K-partition block computes a partial result.
// This pass:
//   1. Allocates a workspace memref<split_k × M × N × f32> via gpu.alloc
//   2. Rewrites tagged epilogue stores: C[m,n] (f16) → workspace[bz, m, n] (f32)
//   3. Generates a reduction kernel summing workspace slices → C[m,n] (f16)
//   4. Deallocates workspace
// Runs AFTER all MW optimization passes and vector-to-gpu, BEFORE staging.
// ============================================================================
struct SplitKEpiloguePass
    : public mlir::PassWrapper<SplitKEpiloguePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitKEpiloguePass)

  explicit SplitKEpiloguePass(const NvidiaMappingConfig &config)
      : split_k_factor(config.split_k_factor),
        block_tile_m(config.block_tile_m),
        block_tile_n(config.block_tile_n) {}

  llvm::StringRef getArgument() const override {
    return "matcore-splitk-epilogue";
  }
  llvm::StringRef getDescription() const override {
    return "Rewrite epilogue stores for Split-K workspace + reduction";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (split_k_factor <= 1)
      return;

    mlir::Operation *top = getOperation();

    // Find the GEMM launch (first gpu.launch)
    mlir::gpu::LaunchOp gemmLaunch;
    top->walk([&](mlir::gpu::LaunchOp launch) {
      if (!gemmLaunch)
        gemmLaunch = launch;
      return mlir::WalkResult::interrupt();
    });
    if (!gemmLaunch) {
      fprintf(stderr, "[SplitKEpilogue] No gpu.launch found, skipping\n");
      fflush(stderr);
      return;
    }

    // Find tagged epilogue stores inside the GEMM launch
    llvm::SmallVector<mlir::memref::StoreOp, 32> epilogueStores;
    gemmLaunch.getBody().walk([&](mlir::memref::StoreOp store) {
      if (store->hasAttr("matcore.epilogue_store"))
        epilogueStores.push_back(store);
    });
    if (epilogueStores.empty()) {
      fprintf(stderr, "[SplitKEpilogue] No tagged epilogue stores, skipping\n");
      fflush(stderr);
      return;
    }
    fprintf(stderr, "[SplitKEpilogue] Found %lu tagged epilogue stores\n",
            (unsigned long)epilogueStores.size());
    fflush(stderr);

    // Determine C memref from first epilogue store
    mlir::Value cMemref = epilogueStores[0].getMemRef();
    auto cType = llvm::dyn_cast<mlir::MemRefType>(cMemref.getType());
    if (!cType || cType.getRank() != 2) {
      fprintf(stderr, "[SplitKEpilogue] C memref is not rank-2, aborting\n");
      fflush(stderr);
      return;
    }

    // Find the func.func to get M, N dimensions and insert workspace ops
    mlir::func::FuncOp func;
    top->walk([&](mlir::func::FuncOp f) { func = f; });
    if (!func)
      return;

    mlir::OpBuilder funcBuilder(func.getBody().front().getTerminator());
    mlir::Location loc = func.getLoc();

    // Get M, N from the function argument C (last arg = output tensor).
    // We use the func arg directly — NOT tracing from inside gpu.launch,
    // because epilogue store memrefs are launch-body block arguments.
    unsigned numArgs = func.getNumArguments();
    mlir::Value cArg = func.getArgument(numArgs - 1); // C is the last arg
    auto cArgType = llvm::dyn_cast<mlir::MemRefType>(cArg.getType());
    if (!cArgType || cArgType.getRank() != 2) {
      fprintf(stderr, "[SplitKEpilogue] Last func arg is not rank-2 memref\n");
      fflush(stderr);
      return;
    }

    // M and N are compile-time known (static memref shapes)
    int64_t staticM = cArgType.getDimSize(0);
    int64_t staticN = cArgType.getDimSize(1);
    if (staticM == mlir::ShapedType::kDynamic ||
        staticN == mlir::ShapedType::kDynamic) {
      fprintf(stderr,
              "[SplitKEpilogue] Dynamic C dims not supported, skipping\n");
      fflush(stderr);
      return;
    }

    mlir::OpBuilder preBuilder(&func.getBody().front().front());
    mlir::Value dimM = preBuilder.create<mlir::arith::ConstantIndexOp>(
        loc, staticM);
    mlir::Value dimN = preBuilder.create<mlir::arith::ConstantIndexOp>(
        loc, staticN);

    // Allocate workspace with STATIC shape using ASYNC ops.
    // GpuToLLVMConversionPass only converts gpu.alloc with async tokens
    // (non-async, non-shared allocs are silently skipped by the pattern).
    auto f32Type = mlir::Float32Type::get(func.getContext());
    auto workspaceType = mlir::MemRefType::get(
        {split_k_factor, staticM, staticN}, f32Type);
    auto tokenType = mlir::gpu::AsyncTokenType::get(func.getContext());

    // Create initial async token
    auto initialWait = preBuilder.create<mlir::gpu::WaitOp>(
        loc, tokenType, mlir::ValueRange{});
    mlir::Value currentToken = initialWait.getAsyncToken();

    auto workspaceAlloc = preBuilder.create<mlir::gpu::AllocOp>(
        loc, workspaceType, /*asyncToken=*/tokenType,
        /*asyncDeps=*/mlir::ValueRange{currentToken},
        /*dynSizes=*/mlir::ValueRange(),
        /*symbolOperands=*/mlir::ValueRange(),
        /*hostShared=*/false);
    mlir::Value workspace = workspaceAlloc.getMemref();
    currentToken = workspaceAlloc.getAsyncToken();

    // Zero the workspace (async)
    mlir::Value zeroF32 = preBuilder.create<mlir::arith::ConstantOp>(
        loc, f32Type, preBuilder.getFloatAttr(f32Type, 0.0));
    auto memsetOp = preBuilder.create<mlir::gpu::MemsetOp>(
        loc, /*asyncToken=*/tokenType,
        /*asyncDeps=*/mlir::ValueRange{currentToken}, workspace, zeroF32);
    currentToken = memsetOp.getAsyncToken();

    // Synchronize: wait for alloc+memset to complete before kernel launch
    preBuilder.create<mlir::gpu::WaitOp>(loc, mlir::Type(),
                                          mlir::ValueRange{currentToken});

    // NOTE: K-loop partitioning is done by SplitKPartitionPass (runs before
    // DoubleBuffer) so the prologue/epilogue see correct per-block K bounds.

    // Rewrite epilogue stores: C[m,n] (f16) → workspace[blockIdx.z, m, n] (f32)
    //
    // AccHoist creates stores like: memref.store %val, %subview[lane_r, lane_c]
    // where %subview = memref.subview %C[group_m_off, group_n_off][16,8][1,1].
    // The store indices are subview-relative.  We must resolve through the
    // subview chain to get absolute C coordinates, otherwise all 8 MMA groups
    // overwrite the same workspace positions (the lane-relative indices are
    // identical across groups; only the subview offsets differ).
    mlir::Value blockIdxZ = gemmLaunch.getBlockIds().z;
    for (auto store : epilogueStores) {
      mlir::OpBuilder sb(store);
      mlir::Value val = store.getValueToStore();

      // Extend f16 → f32
      mlir::Value valF32 = sb.create<mlir::arith::ExtFOp>(
          store.getLoc(), f32Type, val);

      // Resolve absolute indices by walking the memref def chain.
      // After canonicalization, subview chains may be folded into
      // reinterpret_cast ops. We handle both SubViewOp (per-dim offsets)
      // and ReinterpretCastOp (single linear offset decomposed via N).
      llvm::SmallVector<mlir::Value> absIndices(store.getIndices().begin(),
                                                 store.getIndices().end());
      mlir::Value curMemref = store.getMemRef();

      // Walk SubViewOp chain (original path)
      while (auto svOp =
                 curMemref.getDefiningOp<mlir::memref::SubViewOp>()) {
        auto mixedOffsets = svOp.getMixedOffsets();
        for (unsigned i = 0; i < absIndices.size() && i < mixedOffsets.size();
             ++i) {
          mlir::Value offset;
          if (auto attr = llvm::dyn_cast<mlir::Attribute>(mixedOffsets[i])) {
            int64_t sv = llvm::cast<mlir::IntegerAttr>(attr).getInt();
            if (sv != 0)
              offset = sb.create<mlir::arith::ConstantIndexOp>(
                  store.getLoc(), sv);
          } else {
            offset = llvm::cast<mlir::Value>(mixedOffsets[i]);
          }
          if (offset)
            absIndices[i] = sb.create<mlir::arith::AddIOp>(
                store.getLoc(), absIndices[i], offset);
        }
        curMemref = svOp.getSource();
      }

      // Handle ReinterpretCastOp: canonicalization folds subview chains into
      // reinterpret_cast with a single linear offset and original C strides.
      // Decompose: row_off = linear_off / N, col_off = linear_off % N.
      if (auto rcOp =
              curMemref.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
        auto mixedOffsets = rcOp.getMixedOffsets();
        // Use the row stride from the reinterpret_cast itself (more robust)
        auto mixedStrides = rcOp.getMixedStrides();
        int64_t rowStride = -1;
        if (mixedStrides.size() >= 1) {
          if (auto attr = llvm::dyn_cast<mlir::Attribute>(mixedStrides[0]))
            rowStride = llvm::cast<mlir::IntegerAttr>(attr).getInt();
        }
        if (mixedOffsets.size() == 1 && absIndices.size() == 2) {
          mlir::Value linearOff;
          if (auto attr = llvm::dyn_cast<mlir::Attribute>(mixedOffsets[0])) {
            int64_t sv = llvm::cast<mlir::IntegerAttr>(attr).getInt();
            if (sv != 0)
              linearOff = sb.create<mlir::arith::ConstantIndexOp>(
                  store.getLoc(), sv);
          } else {
            linearOff = llvm::cast<mlir::Value>(mixedOffsets[0]);
          }
          if (linearOff) {
            // Use row stride from reinterpret_cast for decomposition
            int64_t divisor = (rowStride > 0) ? rowStride : staticN;
            mlir::Value nVal = sb.create<mlir::arith::ConstantIndexOp>(
                store.getLoc(), divisor);
            mlir::Value rowOff = sb.create<mlir::arith::DivUIOp>(
                store.getLoc(), linearOff, nVal);
            mlir::Value colOff = sb.create<mlir::arith::RemUIOp>(
                store.getLoc(), linearOff, nVal);
            absIndices[0] = sb.create<mlir::arith::AddIOp>(
                store.getLoc(), absIndices[0], rowOff);
            absIndices[1] = sb.create<mlir::arith::AddIOp>(
                store.getLoc(), absIndices[1], colOff);
          }
        }
      }

      // Warn if chain walk stopped at an unrecognized view-like op
      if (!llvm::isa<mlir::BlockArgument>(curMemref) &&
          !curMemref.getDefiningOp<mlir::memref::SubViewOp>() &&
          !curMemref.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
        if (auto *defOp = curMemref.getDefiningOp())
          fprintf(stderr, "[SplitKEpilogue] WARNING: chain walk stopped at "
                  "unrecognized op '%s'\n",
                  defOp->getName().getStringRef().str().c_str());
      }

      // Build workspace indices: [blockIdx.z, abs_m, abs_n]
      llvm::SmallVector<mlir::Value, 3> wsIndices;
      wsIndices.push_back(blockIdxZ);
      for (auto idx : absIndices)
        wsIndices.push_back(idx);

      sb.create<mlir::memref::StoreOp>(store.getLoc(), valF32,
                                        workspace, wsIndices);
      store.erase();
    }

    // Generate reduction kernel: sum workspace slices → C
    // Insert AFTER the GEMM launch. Sequential launches on same stream are
    // implicitly ordered — no explicit sync needed between separate kernels.
    mlir::OpBuilder afterGemm(gemmLaunch->getNextNode());
    loc = gemmLaunch.getLoc();

    // Reduction kernel: 1D grid, 256 threads per block
    int64_t reduceThreads = 256;
    int64_t totalElemsConst = staticM * staticN;
    int64_t gridDimRedConst =
        (totalElemsConst + reduceThreads - 1) / reduceThreads;
    mlir::Value totalElems =
        afterGemm.create<mlir::arith::ConstantIndexOp>(loc, totalElemsConst);
    mlir::Value gridDimRed =
        afterGemm.create<mlir::arith::ConstantIndexOp>(loc, gridDimRedConst);
    mlir::Value one = afterGemm.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value blockDimRed =
        afterGemm.create<mlir::arith::ConstantIndexOp>(loc, reduceThreads);

    auto reduceLaunch = afterGemm.create<mlir::gpu::LaunchOp>(
        loc, gridDimRed, one, one, blockDimRed, one, one);

    // Reduction kernel body
    mlir::Block &reduceBody = reduceLaunch.getBody().front();
    mlir::OpBuilder rb = mlir::OpBuilder::atBlockEnd(&reduceBody);

    // global_tid = blockIdx.x * blockDim + threadIdx.x
    mlir::Value blockIdx = reduceLaunch.getBlockIds().x;
    mlir::Value threadIdx = reduceLaunch.getThreadIds().x;
    mlir::Value blockDimVal =
        rb.create<mlir::arith::ConstantIndexOp>(loc, reduceThreads);
    mlir::Value globalTid =
        rb.create<mlir::arith::MulIOp>(loc, blockIdx, blockDimVal);
    globalTid =
        rb.create<mlir::arith::AddIOp>(loc, globalTid, threadIdx);

    // Bounds check: if (global_tid < M * N)
    mlir::Value inBounds = rb.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, globalTid, totalElems);

    auto ifOp = rb.create<mlir::scf::IfOp>(loc, inBounds, /*withElse=*/false);
    rb = mlir::OpBuilder::atBlockBegin(&ifOp.getThenRegion().front());

    // row = global_tid / N, col = global_tid % N
    mlir::Value row = rb.create<mlir::arith::DivUIOp>(loc, globalTid, dimN);
    mlir::Value col = rb.create<mlir::arith::RemUIOp>(loc, globalTid, dimN);

    // Sum workspace[z, row, col] for z in 0..split_k
    mlir::Value zeroIdx =
        rb.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value splitKLoop =
        rb.create<mlir::arith::ConstantIndexOp>(loc, split_k_factor);
    mlir::Value oneIdx =
        rb.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value initSum = rb.create<mlir::arith::ConstantOp>(
        loc, f32Type, rb.getFloatAttr(f32Type, 0.0));

    // scf.for z = 0 to split_k step 1: accumulate f32
    auto sumLoop = rb.create<mlir::scf::ForOp>(
        loc, zeroIdx, splitKLoop, oneIdx, mlir::ValueRange({initSum}),
        [&](mlir::OpBuilder &lb, mlir::Location lloc, mlir::Value zIv,
            mlir::ValueRange iterArgs) {
          mlir::Value acc = iterArgs[0];
          mlir::Value elem = lb.create<mlir::memref::LoadOp>(
              lloc, workspace, mlir::ValueRange({zIv, row, col}));
          mlir::Value newAcc =
              lb.create<mlir::arith::AddFOp>(lloc, acc, elem);
          lb.create<mlir::scf::YieldOp>(lloc, mlir::ValueRange({newAcc}));
        });

    // Truncate f32 → f16 and store to C
    mlir::Value finalSum = sumLoop.getResult(0);
    auto f16Type = mlir::Float16Type::get(func.getContext());
    mlir::Value truncated =
        rb.create<mlir::arith::TruncFOp>(loc, f16Type, finalSum);
    rb.create<mlir::memref::StoreOp>(loc, truncated, cArg,
                                      mlir::ValueRange({row, col}));

    // Add terminator to reduction launch
    rb = mlir::OpBuilder::atBlockEnd(&reduceBody);
    rb.create<mlir::gpu::TerminatorOp>(loc);

    // Deallocate workspace after reduction (async, matching alloc pattern)
    mlir::OpBuilder deallocBuilder(reduceLaunch->getNextNode());
    auto deallocTokenType = mlir::gpu::AsyncTokenType::get(func.getContext());
    auto deallocWait = deallocBuilder.create<mlir::gpu::WaitOp>(
        loc, deallocTokenType, mlir::ValueRange{});
    mlir::Value deallocToken = deallocWait.getAsyncToken();
    auto deallocOp = deallocBuilder.create<mlir::gpu::DeallocOp>(
        loc, /*asyncToken=*/deallocTokenType,
        /*asyncDeps=*/mlir::ValueRange{deallocToken}, workspace);
    deallocBuilder.create<mlir::gpu::WaitOp>(
        loc, mlir::Type(), mlir::ValueRange{deallocOp.getAsyncToken()});

    // Remove any memref.copy from the old C buffer to the output argument.
    // The original non-split-K path copies the C alloc to cArg after the
    // GEMM launch.  With split-K the reduction kernel writes directly to
    // cArg, so the stale copy would overwrite correct results with zeros.
    llvm::SmallVector<mlir::memref::CopyOp> copiesToRemove;
    func.walk([&](mlir::memref::CopyOp copyOp) {
      if (copyOp.getTarget() == cArg)
        copiesToRemove.push_back(copyOp);
    });
    for (auto copy : copiesToRemove)
      copy.erase();

    fprintf(stderr, "[SplitKEpilogue] Inserted workspace alloc + reduction "
            "kernel (split_k=%lld, threads=%lld), removed %lu stale copies\n",
            (long long)split_k_factor, (long long)reduceThreads,
            (unsigned long)copiesToRemove.size());
    fflush(stderr);
  }

  int64_t split_k_factor;
  int64_t block_tile_m;
  int64_t block_tile_n;
};

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

void runElementwiseLoweringPipeline(mlir::ModuleOp module,
                                    const RequestedTargetProfile &target_profile,
                                    llvm::StringRef nvidia_chip,
                                    ObservabilityContext *obs) {
  (void)target_profile;
  auto *ctx = module.getContext();

  auto run_stage = [&](const char *name, auto &&configure) {
    mlir::PassManager pm(ctx);
    if (obs) {
      attachObservability(pm, obs, name);
    }
    configure(pm);
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler diag_handler(ctx, [&](mlir::Diagnostic &diag) {
      llvm::raw_string_ostream stream(diagnostics);
      diag.print(stream);
      stream << '\n';
      stream.flush();
      return mlir::success();
    });
    if (mlir::failed(pm.run(module))) {
      fail(std::string("elementwise lowering stage '") + name +
           "' failed: " + diagnostics);
    }
  };

  std::string libdevice_path;
  for (const auto &candidate : {
           "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
           "/usr/local/cuda-13.2/nvvm/libdevice/libdevice.10.bc",
           "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc",
       }) {
    if (std::filesystem::exists(candidate)) {
      libdevice_path = candidate;
      break;
    }
  }

  // Phase 1: GPU data staging
  run_stage("elem-gpu-data-staging", [&](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(CreateGpuDataStagingPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  });

  // Phase 2: Kernel outlining
  run_stage("elem-nvvm-outline", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createGpuKernelOutliningPass());
  });

  // Phase 3: Host-side scalar lowering (func-scoped conversions only)
  run_stage("elem-nvvm-host-scalar", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    auto &func_pm = pm.nest<mlir::func::FuncOp>();
    func_pm.addPass(mlir::createArithToLLVMConversionPass());
    mlir::ConvertIndexToLLVMPassOptions idx_opts;
    idx_opts.indexBitwidth = 64;
    func_pm.addPass(mlir::createConvertIndexToLLVMPass(idx_opts));
  });

  // Phase 4: Attach NVVM target (with libdevice if available)
  run_stage("elem-nvvm-attach-target", [&](mlir::PassManager &pm) {
    mlir::GpuNVVMAttachTargetOptions target_opts;
    target_opts.triple = "nvptx64-nvidia-cuda";
    target_opts.chip = nvidia_chip.str();
    target_opts.optLevel = 2;
    std::vector<std::string> link_libs;
    if (!libdevice_path.empty()) {
      link_libs.push_back(libdevice_path);
      target_opts.linkLibs = link_libs;
    }
    pm.addPass(mlir::createGpuNVVMAttachTarget(target_opts));
  });

  // Phase 5: GPU module internal lowering (includes math->NVVM patterns)
  run_stage("elem-nvvm-gpu-module-lower", [&](mlir::PassManager &pm) {
    auto &gpu_pm = pm.nest<mlir::gpu::GPUModuleOp>();
    gpu_pm.addPass(mlir::createStripDebugInfoPass());
    gpu_pm.addPass(mlir::createConvertSCFToCFPass());
    mlir::ConvertGpuOpsToNVVMOpsOptions nvvm_conv_opts;
    nvvm_conv_opts.indexBitwidth = 64;
    nvvm_conv_opts.useBarePtrCallConv = false;
    gpu_pm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvm_conv_opts));
    gpu_pm.addPass(mlir::createCanonicalizerPass());
    gpu_pm.addPass(mlir::createCSEPass());
    gpu_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  });

  // Phase 6: GPU -> LLVM runtime calls
  run_stage("elem-nvvm-gpu-to-llvm", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createGpuToLLVMConversionPass());
  });

  // Phase 7: Host finalize
  run_stage("elem-nvvm-host-finalize", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  });

  // Phase 8: Serialize gpu.binary
  run_stage("elem-nvvm-binary", [&](mlir::PassManager &pm) {
    mlir::GpuModuleToBinaryPassOptions bin_opts;
    bin_opts.compilationTarget = "fatbin";
    pm.addPass(mlir::createGpuModuleToBinaryPass(bin_opts));
  });

  // Phase 9: Cleanup
  run_stage("elem-nvvm-cleanup", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  });
}

void runFusionLoweringPipeline(mlir::ModuleOp module,
                               const RequestedTargetProfile &target_profile,
                               llvm::StringRef nvidia_chip,
                               ObservabilityContext *obs) {
  auto kernel_type_attr =
      module->getAttrOfType<mlir::StringAttr>("matcore.kernel_type");
  if (!kernel_type_attr || !kernel_type_attr.getValue().starts_with("fused_")) {
    fail("fusion lowering currently expects matcore.kernel_type to start with fused_");
  }

  auto *ctx = module.getContext();
  auto run_stage = [&](const char *name, auto &&configure) {
    mlir::PassManager pm(ctx);
    if (obs) {
      attachObservability(pm, obs, name);
    }
    configure(pm);
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler diag_handler(ctx, [&](mlir::Diagnostic &diag) {
      llvm::raw_string_ostream stream(diagnostics);
      diag.print(stream);
      stream << '\n';
      stream.flush();
      return mlir::success();
    });
    if (mlir::failed(pm.run(module))) {
      fail(std::string("fusion lowering stage '") + name + "' failed: " +
           diagnostics);
    }
  };

  std::string libdevice_path;
  for (const auto &candidate : {
           "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
           "/usr/local/cuda-13.2/nvvm/libdevice/libdevice.10.bc",
           "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc",
       }) {
    if (std::filesystem::exists(candidate)) {
      libdevice_path = candidate;
      break;
    }
  }

  (void)target_profile;
  const std::optional<int> max_regs = getFusionRegisterCap(module);
  if (max_regs) {
    mlir::Builder builder(ctx);
    module->setAttr("matcore.max_regs", builder.getI32IntegerAttr(*max_regs));
  }
  const std::string nvvm_cmd_options = buildMaxRegisterCommandOptions(max_regs);

  // Check if this is a family_a or family_b MMA kernel (has linalg.matmul).
  auto fusion_pattern_attr =
      module->getAttrOfType<mlir::StringAttr>("matcore.fusion_pattern");
  const bool is_mma_family =
      fusion_pattern_attr &&
      (fusion_pattern_attr.getValue() == "family_a");

  if (is_mma_family) {
    // === MMA FUSION PATH ===
    // The emitter produced linalg.matmul on memrefs. We reuse the standard
    // MMA transform pipeline to tile and rewrite to tensor cores, then
    // run the FusionEpiloguePass for elementwise ops, then data-staging.

    // 1. Decode signature + select mapping (same as standard matmul path).
    const MatmulLoweringSignature signature =
        decodeMatmulSignatureFromModule(module);
    NvidiaMappingConfig mapping =
        selectNvidiaMappingForModule(module, signature);

    // Disable split-K for fusion patterns: the fusion pipeline doesn't have
    // the SplitKEpiloguePass (workspace alloc + reduction kernel). Without it,
    // split-K partitions race-write partial results to the output buffer,
    // corrupting the result. Force split_k=1 so the full K range is computed
    // in a single pass per block.
    if (mapping.split_k_factor > 1) {
      fprintf(stderr,
              "[FusionMMA] Disabling split_k=%lld for fusion pattern "
              "(no reduction kernel in fusion pipeline)\n",
              (long long)mapping.split_k_factor);
      mapping.split_k_factor = 1;
    }

    // 2. Apply tiling transform.
    if (UsesMultiWarpMmaSync(mapping)) {
      ApplyNvidiaMultiWarpTransformToModule(module, signature, mapping);
    } else {
      ApplyNvidiaMmaTransformToModule(module, signature, mapping);
    }

    // 3. Tag K-loop (needed by AccHoist/DoubleBuffer).
    run_stage("fusion-mma-tag-k-loop", [&](mlir::PassManager &pm) {
      pm.addPass(std::make_unique<TagKLoopPass>(mapping.k_tile));
    });

    // 4. Map to GPU grid.
    run_stage("fusion-mma-grid-mapping", [&](mlir::PassManager &pm) {
      AddNvidiaDynamicMacroGridMappingPasses(pm, mapping);
    });

    // 5. Canonicalize + CSE after transform.
    run_stage("fusion-mma-post-transform", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    });

    // 6. MMA rewrite or thread mapping, depending on dtype support.
    if (mapping.rewrite_to_mma_sync) {
      if (UsesMultiWarpMmaSync(mapping)) {
        // === MULTI-WARP MMA PATH ===
        // Must match the standalone multi-warp pipeline exactly.
        // Multi-warp thread mapping
        ApplyNvidiaMultiWarpThreadMappingToModule(module, mapping);
        run_stage("fusion-mw-post-thread-map", [&](mlir::PassManager &pm) {
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
        // Sub-tile unroll
        run_stage("fusion-mw-sub-tile-unroll", [&](mlir::PassManager &pm) {
          pm.addPass(std::make_unique<SubTileUnrollPass>());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
        // Address math simplification
        run_stage("fusion-mw-address-math", [&](mlir::PassManager &pm) {
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
        // MMA preparation + rewrite
        run_stage("fusion-mma-preparation", [&](mlir::PassManager &pm) {
          AddNvidiaMmaPreparationPasses(pm);
        });
        // Collect pre-MMA thread IDs for lane fix
        llvm::SmallPtrSet<mlir::Operation *, 16> pre_mma_thread_ids;
        module->walk([&](mlir::gpu::ThreadIdOp op) {
          pre_mma_thread_ids.insert(op.getOperation());
        });
        ApplyNvidiaMmaRewriteToModule(module);
        // Fix MMA lane IDs for multi-warp: thread_id_x % 32
        if (mapping.block_threads_x > 32) {
          llvm::SmallVector<mlir::gpu::ThreadIdOp, 8> mma_thread_ids;
          module->walk([&](mlir::gpu::ThreadIdOp op) {
            if (!pre_mma_thread_ids.contains(op.getOperation()) &&
                op.getDimension() == mlir::gpu::Dimension::x) {
              mma_thread_ids.push_back(op);
            }
          });
          for (auto op : mma_thread_ids) {
            mlir::OpBuilder builder(op->getContext());
            builder.setInsertionPointAfter(op);
            auto loc = op.getLoc();
            auto c32 = builder.create<mlir::arith::ConstantIndexOp>(loc, 32);
            auto lane =
                builder.create<mlir::arith::RemUIOp>(loc, op.getResult(), c32);
            llvm::SmallPtrSet<mlir::Operation *, 1> except;
            except.insert(lane.getOperation());
            op.getResult().replaceAllUsesExcept(lane.getResult(), except);
          }
        }
        VerifyNoResidualNvidiaMatmulOnModule(module);
        // Accumulator hoisting
        run_stage("fusion-mw-accumulator-hoist", [&](mlir::PassManager &pm) {
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
          pm.addPass(std::make_unique<AccumulatorHoistPass>());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
        // Shared memory barriers
        run_stage("fusion-mw-smem-barriers", [&](mlir::PassManager &pm) {
          pm.addPass(std::make_unique<InsertSmemBarriersPass>());
        });
        // Vectorize tile copies
        run_stage("fusion-mw-vectorize-tile-copy", [&](mlir::PassManager &pm) {
          pm.addPass(std::make_unique<VectorizeTileCopyPass>());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
        // Pad shared memory
        run_stage("fusion-mw-pad-smem", [&](mlir::PassManager &pm) {
          pm.addPass(std::make_unique<PadSharedMemoryPass>());
        });
        // Double-buffer K-loop
        run_stage("fusion-mw-double-buffer", [&](mlir::PassManager &pm) {
          pm.addPass(std::make_unique<DoubleBufferKLoopPass>());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
        // LdMatrix rewrite
        run_stage("fusion-mw-ldmatrix", [&](mlir::PassManager &pm) {
          pm.addPass(std::make_unique<LdMatrixRewritePass>());
          pm.addPass(mlir::createCanonicalizerPass());
          pm.addPass(mlir::createCSEPass());
        });
      } else {
        // Single-warp MMA path
        run_stage("fusion-mma-preparation", [&](mlir::PassManager &pm) {
          AddNvidiaMmaPreparationPasses(pm);
        });
        ApplyNvidiaMmaRewriteToModule(module);
        VerifyNoResidualNvidiaMatmulOnModule(module);
      }
    } else {
      // F32 path: thread mapping (tiled, coalesced, but no tensor cores).
      ApplyNvidiaThreadMappingToModule(module, mapping);
      run_stage("fusion-thread-post-map", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });
    }

    // 7. Launch configuration + loop materialization + vector→gpu.
    run_stage("fusion-mma-launch-config", [&](mlir::PassManager &pm) {
      AddNvidiaLaunchConfigurationPasses(pm);
    });
    run_stage("fusion-mma-loop-materialization", [&](mlir::PassManager &pm) {
      AddNvidiaLoopMaterializationPasses(pm);
    });
    run_stage("fusion-mma-vector-to-gpu", [&](mlir::PassManager &pm) {
      ConfigureNvidiaVectorToGpuStage(pm);
    });

    // 7. Loop materialization (linalg→loops, scf→cf, etc.).

    // 8. Fusion epilogue: second gpu.launch for relu/gelu/exp if needed.
    run_stage("fusion-epilogue", [&](mlir::PassManager &pm) {
      pm.addPass(CreateFusionEpiloguePass());
    });

    // 9. Data staging for ALL gpu.launch ops (matmul + epilogue).
    run_stage("fusion-mma-data-staging", [&](mlir::PassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(CreateGpuDataStagingPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    });

    // 10. Full NVVM lowering (with NVGPU→NVVM for tensor core ops).
    run_stage("fusion-mma-nvvm-outline", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::createConvertNVGPUToNVVMPass());
      pm.addPass(mlir::createGpuKernelOutliningPass());
    });
    run_stage("fusion-mma-nvvm-host-scalar", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::createConvertVectorToSCFPass());
      pm.addPass(mlir::createConvertSCFToCFPass());
      pm.addPass(mlir::createConvertNVVMToLLVMPass());
      pm.addPass(mlir::memref::createExpandStridedMetadataPass());
      pm.addPass(mlir::createLowerAffinePass());
      // Scope math/arith/index → LLVM to host func only.
      // GPU module has its own math → NVVM lowering via createConvertGpuOpsToNVVMOps.
      auto &func_pm = pm.nest<mlir::func::FuncOp>();
      func_pm.addPass(mlir::createConvertMathToLLVMPass());
      func_pm.addPass(mlir::createArithToLLVMConversionPass());
      mlir::ConvertIndexToLLVMPassOptions idx_opts;
      idx_opts.indexBitwidth = 64;
      func_pm.addPass(mlir::createConvertIndexToLLVMPass(idx_opts));
    });
    run_stage("fusion-mma-nvvm-attach-target", [&](mlir::PassManager &pm) {
      mlir::GpuNVVMAttachTargetOptions target_opts;
      target_opts.triple = "nvptx64-nvidia-cuda";
      target_opts.chip = nvidia_chip.str();
      target_opts.optLevel = 2;
      std::vector<std::string> link_libs;
      if (!libdevice_path.empty()) {
        link_libs.push_back(libdevice_path);
        target_opts.linkLibs = link_libs;
      }
      pm.addPass(mlir::createGpuNVVMAttachTarget(target_opts));
    });
    run_stage("fusion-mma-nvvm-gpu-module", [&](mlir::PassManager &pm) {
      auto &gpu_pm = pm.nest<mlir::gpu::GPUModuleOp>();
      gpu_pm.addPass(mlir::createStripDebugInfoPass());
      mlir::ConvertGpuOpsToNVVMOpsOptions nvvm_conv_opts;
      nvvm_conv_opts.indexBitwidth = 64;
      nvvm_conv_opts.useBarePtrCallConv = false;
      gpu_pm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvm_conv_opts));
      gpu_pm.addPass(mlir::createCanonicalizerPass());
      gpu_pm.addPass(mlir::createCSEPass());
      gpu_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    });
    run_stage("fusion-mma-nvvm-gpu-to-llvm", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::createGpuToLLVMConversionPass());
    });
    run_stage("fusion-mma-nvvm-host-finalize", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(mlir::createConvertControlFlowToLLVMPass());
      pm.addPass(mlir::createConvertFuncToLLVMPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    });
    annotateFusionRegisterUsageFromPtx(module, nvvm_cmd_options, obs);
    run_stage("fusion-mma-nvvm-binary", [&](mlir::PassManager &pm) {
      mlir::GpuModuleToBinaryPassOptions bin_opts;
      bin_opts.compilationTarget = "fatbin";
      bin_opts.cmdOptions = nvvm_cmd_options;
      pm.addPass(mlir::createGpuModuleToBinaryPass(bin_opts));
    });
    run_stage("fusion-mma-nvvm-cleanup", [&](mlir::PassManager &pm) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    });
    return;
  }

  // === ORIGINAL FUSION PATH (family_c and fallback) ===
  // The emitter produced explicit gpu.launch with scalar ops.
  run_stage("fusion-gpu-data-staging", [&](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(CreateGpuDataStagingPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  });
  run_stage("fusion-nvvm-outline", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createGpuKernelOutliningPass());
  });
  run_stage("fusion-nvvm-host-scalar", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    auto &func_pm = pm.nest<mlir::func::FuncOp>();
    func_pm.addPass(mlir::createArithToLLVMConversionPass());
    mlir::ConvertIndexToLLVMPassOptions idx_opts;
    idx_opts.indexBitwidth = 64;
    func_pm.addPass(mlir::createConvertIndexToLLVMPass(idx_opts));
  });
  run_stage("fusion-nvvm-attach-target", [&](mlir::PassManager &pm) {
    mlir::GpuNVVMAttachTargetOptions target_opts;
    target_opts.triple = "nvptx64-nvidia-cuda";
    target_opts.chip = nvidia_chip.str();
    target_opts.optLevel = 2;
    std::vector<std::string> link_libs;
    if (!libdevice_path.empty()) {
      link_libs.push_back(libdevice_path);
      target_opts.linkLibs = link_libs;
    }
    pm.addPass(mlir::createGpuNVVMAttachTarget(target_opts));
  });
  run_stage("fusion-nvvm-gpu-module-lower", [&](mlir::PassManager &pm) {
    auto &gpu_pm = pm.nest<mlir::gpu::GPUModuleOp>();
    gpu_pm.addPass(mlir::createStripDebugInfoPass());
    gpu_pm.addPass(mlir::createConvertSCFToCFPass());
    mlir::ConvertGpuOpsToNVVMOpsOptions nvvm_conv_opts;
    nvvm_conv_opts.indexBitwidth = 64;
    nvvm_conv_opts.useBarePtrCallConv = false;
    gpu_pm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvm_conv_opts));
    gpu_pm.addPass(mlir::createCanonicalizerPass());
    gpu_pm.addPass(mlir::createCSEPass());
    gpu_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  });
  run_stage("fusion-nvvm-gpu-to-llvm", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createGpuToLLVMConversionPass());
  });
  run_stage("fusion-nvvm-host-finalize", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  });
  annotateFusionRegisterUsageFromPtx(module, nvvm_cmd_options, obs);
  run_stage("fusion-nvvm-binary", [&](mlir::PassManager &pm) {
    mlir::GpuModuleToBinaryPassOptions bin_opts;
    bin_opts.compilationTarget = "fatbin";
    bin_opts.cmdOptions = nvvm_cmd_options;
    pm.addPass(mlir::createGpuModuleToBinaryPass(bin_opts));
  });
  run_stage("fusion-nvvm-cleanup", [&](mlir::PassManager &pm) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  });
}

void runLoweringPipeline(mlir::ModuleOp module, const LoweringPlan &plan,
                         const MatmulLoweringSignature &signature,
                         llvm::StringRef nvidia_chip,
                         llvm::StringRef amd_chip,
                         ObservabilityContext *obs) {
  if (auto kernel_type_attr =
          module->getAttrOfType<mlir::StringAttr>("matcore.kernel_type");
      kernel_type_attr && kernel_type_attr.getValue() == "elementwise") {
    if (plan.route != LoweringRoute::kNvidiaNvptx) {
      fail("elementwise GPU lowering currently requires nvidia-dgpu target route");
    }
    RequestedTargetProfile target_profile;
    if (auto requested_target =
            module->getAttrOfType<mlir::StringAttr>("matcore.requested_target")) {
      target_profile = ParseRequestedTargetProfile(requested_target.getValue().str());
    }
    runElementwiseLoweringPipeline(module, target_profile, nvidia_chip, obs);
    return;
  }
  if (auto kernel_type_attr =
          module->getAttrOfType<mlir::StringAttr>("matcore.kernel_type");
      kernel_type_attr && kernel_type_attr.getValue().starts_with("fused_")) {
    if (plan.route != LoweringRoute::kNvidiaNvptx) {
      fail("fusion GPU lowering currently requires nvidia-dgpu target route");
    }
    RequestedTargetProfile target_profile;
    if (auto requested_target =
            module->getAttrOfType<mlir::StringAttr>("matcore.requested_target")) {
      target_profile =
          ParseRequestedTargetProfile(requested_target.getValue().str());
    }
    runFusionLoweringPipeline(module, target_profile, nvidia_chip, obs);
    return;
  }

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
      if (UsesMultiWarpMmaSync(mapping)) {
        ApplyNvidiaMultiWarpTransformToModule(module, signature, mapping);
      } else {
        ApplyNvidiaMmaTransformToModule(module, signature, mapping);
      }
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

    // Tag the K-loop before DynamicMacroGridMappingPass clones body
    // (StringAttrs survive Operation::clone)
    if (mapping.split_k_factor > 1 || true) { // always tag for AccHoist/DoubleBuffer
      run_stage("nvidia-tag-k-loop", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<TagKLoopPass>(mapping.k_tile));
      });
    }

    run_stage("nvidia-dynamic-macro-topology", [&](mlir::PassManager &pm) {
      AddNvidiaDynamicMacroGridMappingPasses(pm, mapping);
    });

    if (UsesMultiWarpMmaSync(mapping)) {
      // === MULTI-WARP VECTORIZE PATH (V4) ===
      // Transform already created: blocks → K-loop → smem → warps → sub-tile → vector.contract
      run_stage("nvidia-post-transform-canonicalize",
                [&](mlir::PassManager &pm) {
                  pm.addPass(mlir::createCanonicalizerPass());
                  pm.addPass(mlir::createCSEPass());
                });
      // Map warp-level scf.forall to thread indices within gpu.launch
      if (obs) {
        obs->snapshot("nvidia-multiwarp-thread-mapping_pre", module);
      }
      {
        [[maybe_unused]] auto mw_map_trace =
            create_stage_trace("nvidia-multiwarp-thread-mapping");
        const int mw_stage_index = nextStageIndex();
        const std::string mw_ir_before = captureIrForDiagnostics(module);
        try {
          ApplyNvidiaMultiWarpThreadMappingToModule(module, mapping);
        } catch (const std::exception &exc) {
          std::string normalized = normalizeFailureMessage(exc.what());
          std::vector<CapturedDiagnostic> captured = {{
              .pass_name = "nvidia-multiwarp-thread-mapping",
              .severity = "error",
              .message = normalized,
          }};
          fail_with_report("nvidia-multiwarp-thread-mapping", mw_stage_index,
                           normalized, "nvidia-multiwarp-thread-mapping",
                           std::move(captured), mw_ir_before);
        }
      }
      if (obs) {
        obs->snapshot("nvidia-multiwarp-thread-mapping_post", module);
      }
      run_stage("nvidia-post-thread-map-canonicalize",
                [&](mlir::PassManager &pm) {
                  pm.addPass(mlir::createCanonicalizerPass());
                  pm.addPass(mlir::createCSEPass());
                });
      // === MW-7 Phase A: Sub-tile loop unrolling ===
      // Unroll M/N sub-tile loops to expose all 8 MMA ops in K-loop body.
      run_stage("nvidia-sub-tile-unroll", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<SubTileUnrollPass>());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });

      // === MW-7 Phase G: Address math simplification ===
      // After unrolling, many address computations become constant-foldable.
      run_stage("nvidia-address-math-simplify", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });

      run_stage("nvidia-mma-preparation", [&](mlir::PassManager &pm) {
        AddNvidiaMmaPreparationPasses(pm);
      });
      // Rewrite 16×8×16 linalg.matmul → nvvm.mma.sync (same as V3 single-warp)
      // The vectorize path was abandoned because transform.structured.vectorize
      // produces arith.mulf + vector.multi_reduction instead of vector.contract,
      // which ConvertVectorToGPU cannot convert to nvgpu.mma.sync.
      {
        if (obs) {
          obs->snapshot("nvidia-rewrite-mma-sync_pre", module);
        }
        [[maybe_unused]] auto rewrite_trace =
            create_stage_trace("nvidia-rewrite-mma-sync");
        const int stage_index = nextStageIndex();
        const std::string ir_before = captureIrForDiagnostics(module);

        // Collect existing gpu.thread_id ops BEFORE MMA rewrite
        llvm::SmallPtrSet<mlir::Operation *, 16> pre_mma_thread_ids;
        module->walk([&](mlir::gpu::ThreadIdOp op) {
          pre_mma_thread_ids.insert(op.getOperation());
        });

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

        // Fix MMA lane IDs for multi-warp:
        // RewriteMatmulAsMmaSyncOp generates gpu.thread_id x for MMA fragment
        // positions, assuming thread_id_x == lane_id (true for single-warp).
        // In multi-warp with block_threads_x > 32, thread_id_x ranges beyond
        // 0-31, causing threads in warp 1+ to compute wrong MMA positions.
        // Fix: replace NEW thread_id_x ops (from MMA rewrite) with
        // thread_id_x % warp_size to get the correct lane_id.
        if (mapping.block_threads_x > 32) {
          llvm::SmallVector<mlir::gpu::ThreadIdOp, 8> mma_thread_ids;
          module->walk([&](mlir::gpu::ThreadIdOp op) {
            if (!pre_mma_thread_ids.contains(op.getOperation()) &&
                op.getDimension() == mlir::gpu::Dimension::x) {
              mma_thread_ids.push_back(op);
            }
          });

          for (auto op : mma_thread_ids) {
            mlir::OpBuilder builder(op->getContext());
            builder.setInsertionPointAfter(op);
            auto loc = op.getLoc();
            auto c32 = builder.create<mlir::arith::ConstantIndexOp>(loc, 32);
            auto lane =
                builder.create<mlir::arith::RemUIOp>(loc, op.getResult(), c32);
            llvm::SmallPtrSet<mlir::Operation *, 1> except;
            except.insert(lane.getOperation());
            op.getResult().replaceAllUsesExcept(lane.getResult(), except);
          }
        }

        if (obs) {
          obs->snapshot("nvidia-rewrite-mma-sync_post", module);
        }
      }

      // === MW-7 Phase B: Accumulator hoisting ===
      // Hoist MMA C accumulators from global memory to K-loop iter_args.
      run_stage("nvidia-accumulator-hoist", [&](mlir::PassManager &pm) {
        // CSE first: after MMA rewrite + k_sub unrolling, stores at same
        // (m,n) position use structurally identical but separate SSA values.
        // CSE merges them so AccHoist can group by SSA equality.
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(std::make_unique<AccumulatorHoistPass>());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });

      // === MW-7 Phase C: Shared memory barriers ===
      // Ensure all warps finish writing smem A/B before any warp reads.
      run_stage("nvidia-smem-barriers", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<InsertSmemBarriersPass>());
      });

      // === MW-7 Phase F: Vectorize A/B tile copies ===
      // Replace linalg.copy with cooperative vector.load/vector.store (LDG.128).
      // Must run after Phase C (barriers in place) and before loop materialization.
      run_stage("nvidia-vectorize-tile-copy", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<VectorizeTileCopyPass>());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });

      // Phase H: Pad shared memory (+1 col) to break bank conflict stride patterns
      run_stage("nvidia-pad-shared-memory", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<PadSharedMemoryPass>());
      });

      // NOTE: K-loop partitioning for split-K is done by
      // DynamicMacroGridMappingPass (gpu_mapping.cpp:447) during grid setup,
      // BEFORE DoubleBuffer runs, so prologue/epilogue see correct bounds.

      // Phase E: Double-buffer the K-loop (ping-pong shared memory + cp.async overlap)
      run_stage("nvidia-double-buffer-kloop", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<DoubleBufferKLoopPass>());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });

      // Phase L1: Replace scalar smem loads with warp-cooperative ldmatrix
      run_stage("nvidia-ldmatrix-rewrite", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<LdMatrixRewritePass>());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
      });

      run_stage("nvidia-launch-config", [&](mlir::PassManager &pm) {
        AddNvidiaLaunchConfigurationPasses(pm);
      });
      // Keep loop materialization: legalizes residual linalg.fill/copy from smem promotion
      run_stage("nvidia-loop-materialization", [&](mlir::PassManager &pm) {
        AddNvidiaLoopMaterializationPasses(pm);
      });
      // VectorToGPU: converts any remaining vector.contract → nvgpu.mma.sync
      run_stage("nvidia-vector-to-gpu", [&](mlir::PassManager &pm) {
        ConfigureNvidiaVectorToGpuStage(pm);
      });
    } else if (mapping.rewrite_to_mma_sync) {
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
    // Verify no residual linalg.matmul — catches silent vectorize failures
    // that would produce garbage code or crash during NVVM lowering.
    {
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
    }

    // Split-K epilogue: rewrite C stores to workspace + generate reduction kernel
    if (mapping.split_k_factor > 1) {
      run_stage("nvidia-splitk-epilogue", [&](mlir::PassManager &pm) {
        pm.addPass(std::make_unique<SplitKEpiloguePass>(mapping));
      });
    }

    run_stage("nvidia-gpu-data-staging", [&](mlir::PassManager &pm) {
      pm.addNestedPass<mlir::func::FuncOp>(CreateGpuDataStagingPass());
    });
    if (UsesMultiWarpMmaSync(mapping)) {
      // Split NVVM into sub-stages for multi-warp debugging
      run_stage("nvvm-phase1-outline", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createConvertNVGPUToNVVMPass());
        pm.addPass(mlir::createGpuKernelOutliningPass());
      });
      // Phase 2: Host-side scalar lowering.
      // Structural passes run at module level (VectorToSCF, SCFToCF, etc.).
      // ArithToLLVM/IndexToLLVM are scoped to func::FuncOp ONLY to avoid
      // contaminating the gpu.module kernel with i64↔index casts that
      // ReconcileUnrealizedCasts can't resolve for complex multi-warp kernels.
      run_stage("nvvm-phase2-host-scalar", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createConvertVectorToSCFPass());
        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addPass(mlir::createConvertNVVMToLLVMPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::createLowerAffinePass());
        // Scope to host function only — gpu.module is untouched
        auto &func_pm = pm.nest<mlir::func::FuncOp>();
        func_pm.addPass(mlir::createArithToLLVMConversionPass());
        mlir::ConvertIndexToLLVMPassOptions idx_opts;
        idx_opts.indexBitwidth = 64;
        func_pm.addPass(mlir::createConvertIndexToLLVMPass(idx_opts));
      });
      run_stage("nvvm-phase3-attach-target", [&](mlir::PassManager &pm) {
        mlir::GpuNVVMAttachTargetOptions target_opts;
        target_opts.triple = "nvptx64-nvidia-cuda";
        target_opts.chip = nvidia_chip.str();
        target_opts.optLevel = 2;
        pm.addPass(mlir::createGpuNVVMAttachTarget(target_opts));
      });
      run_stage("nvvm-phase4-gpu-module", [&](mlir::PassManager &pm) {
        // Match single-warp Phase 4 (gpu_nvvm_lowering.cpp:312-323).
        // ConvertGpuOpsToNVVM is a mega-pass that internally includes
        // arith-to-LLVM, vector-to-LLVM, memref-to-LLVM patterns — all
        // sharing a single LLVMTypeConverter where NVVM ops are legal.
        // Do NOT add standalone ConvertVectorToLLVM or ArithToLLVM here:
        // they create separate type converters that corrupt nvvm.mma.sync
        // operand types (vector<2xf16> → llvm.vec<2xf16>), causing 0 HMMA.
        auto &gpu_pm = pm.nest<mlir::gpu::GPUModuleOp>();
        gpu_pm.addPass(mlir::createStripDebugInfoPass());
        mlir::ConvertGpuOpsToNVVMOpsOptions nvvm_conv_opts;
        nvvm_conv_opts.indexBitwidth = 64;
        nvvm_conv_opts.useBarePtrCallConv = false;
        gpu_pm.addPass(mlir::createConvertGpuOpsToNVVMOps(nvvm_conv_opts));
        gpu_pm.addPass(mlir::createCanonicalizerPass());
        gpu_pm.addPass(mlir::createCSEPass());
        gpu_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });
      run_stage("nvvm-phase5-gpu-to-llvm", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createGpuToLLVMConversionPass());
      });
      // Phase 5b: Same as single-warp — only FinalizeMemRef + CF + Func.
      // ArithToLLVM/IndexToLLVM already ran in Phase 2 (func-scoped).
      run_stage("nvvm-phase5b-host-lowering", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });
      run_stage("nvvm-phase6-binary", [&](mlir::PassManager &pm) {
        mlir::GpuModuleToBinaryPassOptions bin_opts;
        bin_opts.compilationTarget = "fatbin";
        pm.addPass(mlir::createGpuModuleToBinaryPass(bin_opts));
      });
      run_stage("nvvm-phase7-cleanup", [&](mlir::PassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });
    } else {
      run_stage("nvidia-nvvm", [&](mlir::PassManager &pm) {
        ConfigureNvidiaNvvmStage(pm, nvidia_chip, obs);
      });
    }
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
