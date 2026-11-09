// Independent regression controls for otherwise valid, non-closed shaders.
// Never imports an executable artifact, dispatches, or grants runtime authority.
#include "MatcoreSpirvGemmCandidate.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <functional>
#include <map>
#include <string>

namespace {
using namespace mlir;
namespace candidate = matcore::mdslc::spirv_candidate;
using Counts = std::map<std::string, unsigned>;

Counts counts(Operation *module) {
  Counts result;
  module->walk([&](Operation *op) { ++result[op->getName().getStringRef().str()]; });
  return result;
}

bool decorateBufferMember(spirv::ModuleOp module, unsigned binding,
                          spirv::Decoration decoration) {
  for (auto global : module.getOps<spirv::GlobalVariableOp>()) {
    if (global.getBinding() != binding) continue;
    auto pointer = dyn_cast<spirv::PointerType>(global.getType());
    auto structure = pointer ? dyn_cast<spirv::StructType>(pointer.getPointeeType())
                             : spirv::StructType{};
    if (!structure || structure.getNumElements() != 1 || !structure.hasOffset())
      return false;
    SmallVector<spirv::StructType::MemberDecorationInfo> decorations;
    structure.getMemberDecorations(decorations);
    decorations.emplace_back(0, 0, decoration, 0);
    auto changed = spirv::StructType::get(
        {structure.getElementType(0)},
        {static_cast<spirv::StructType::OffsetInfo>(structure.getMemberOffset(0))},
        decorations);
    auto changedPointer = spirv::PointerType::get(changed, pointer.getStorageClass());
    global.setTypeAttr(TypeAttr::get(changedPointer));
    module.walk([&](spirv::AddressOfOp address) {
      if (address.getVariable() == global.getSymName())
        address->getResult(0).setType(changedPointer);
    });
    return true;
  }
  return false;
}

bool writeWords(const std::filesystem::path &path, ArrayRef<std::uint32_t> words) {
  std::error_code error;
  llvm::raw_fd_ostream output(path.string(), error, llvm::sys::fs::CD_CreateNew,
                             llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
  if (error) return false;
  for (auto word : words)
    for (unsigned shift = 0; shift != 32; shift += 8)
      output << static_cast<char>((word >> shift) & 255);
  output.close();
  return !output.has_error();
}

spirv::FuncOp decodedFunction(spirv::ModuleOp module) {
  auto functions = module.getOps<spirv::FuncOp>();
  return llvm::hasSingleElement(functions) ? *functions.begin() : spirv::FuncOp{};
}
spirv::LoopOp decodedLoop(spirv::ModuleOp module) {
  spirv::LoopOp loop;
  unsigned count = 0;
  module.walk([&](spirv::LoopOp candidate) { loop = candidate; ++count; });
  return count == 1 ? loop : spirv::LoopOp{};
}
} // namespace

int main(int argc, char **argv) {
  // Optional export is for a separately recorded upstream spirv-val gate.
  // The directory must not exist; ordinary CTest invocation writes no files.
  if (argc > 2) return 2;
  std::filesystem::path directory;
  if (argc == 2) {
    directory = argv[1];
    std::error_code error;
    if (!std::filesystem::create_directory(directory, error)) return 2;
  }
  MLIRContext context;
  unsigned checks = 0, failures = 0;
  std::string error;
  const auto require = [&](bool condition, llvm::StringRef name) {
    ++checks;
    if (!condition) {
      ++failures;
      llvm::errs() << "FAIL " << name << ": " << error << '\n';
    }
    return condition;
  };
  auto artifact = candidate::issueStrictGemmArtifactV1(context, {2, 3, 4});
  error = artifact.error;
  if (!require(bool(artifact), "closed control issuance")) return 1;
  auto control = parseSourceString<spirv::ModuleOp>(artifact.gemm_spirv_ir, &context);
  if (!require(control && succeeded(verify(*control)), "control MLIR validity") ||
      !require(candidate::verifyStrictGemmShaderV1(*control, {2, 3, 4}, false, error),
               "control structural acceptance")) return 1;
  const Counts expectedCounts = counts(*control);

  const auto attack = [&](const char *name, const char *file,
                          const std::function<bool(spirv::ModuleOp)> &mutate,
                          bool expectedVulkanValid = true) {
    auto changed = parseSourceString<spirv::ModuleOp>(artifact.gemm_spirv_ir, &context);
    error.clear();
    if (!require(changed && mutate(*changed), std::string(name) + " constructed")) return;
    if (!require(counts(*changed) == expectedCounts,
                 std::string(name) + " unchanged operation inventory")) return;
    // A generic-MLIR failure must never be reported as a closed-graph refusal.
    if (!require(succeeded(verify(*changed)), std::string(name) + " MLIR valid")) return;
    SmallVector<std::uint32_t> words;
    spirv::SerializationOptions options;
    options.emitSymbolName = true;
    options.emitDebugInfo = false;
    if (!require(succeeded(spirv::serialize(*changed, words, options)),
                 std::string(name) + " production serialization")) return;
    if (!directory.empty() && expectedVulkanValid)
      if (!require(writeWords(directory / file, words), std::string(name) + " export")) return;
    // No artifact replay is involved: this tests the independent semantic/ABI
    // classifier directly, rather than a string/word identity mismatch.
    require(!candidate::verifyStrictGemmShaderV1(*changed, {2, 3, 4}, false, error) &&
                !error.empty(), std::string(name) + " closed classifier refusal");
  };

  attack("load Aligned8 overclaim", "load-aligned8.spv", [](auto module) {
    bool changed = false;
    module.walk([&](spirv::LoadOp load) {
      if (!changed && load->getResult(0).getType().isF32()) {
        load.setMemoryAccess(spirv::MemoryAccess::Aligned);
        load.setAlignment(8);
        changed = true;
      }
    });
    return changed;
  });
  attack("store Aligned8 overclaim", "store-aligned8.spv", [](auto module) {
    bool changed = false;
    module.walk([&](spirv::StoreOp store) {
      store.setMemoryAccess(spirv::MemoryAccess::Aligned);
      store.setAlignment(8);
      changed = true;
    });
    return changed;
  });
  attack("buffer member RelaxedPrecision", "member-relaxed.spv", [](auto module) {
    return decorateBufferMember(module, 0, spirv::Decoration::RelaxedPrecision);
  });
  attack("foreign entry interface", "not-exported.spv", [&](auto module) {
    spirv::GlobalVariableOp storage;
    for (auto global : module.template getOps<spirv::GlobalVariableOp>())
      if (global.getBinding() == 0) storage = global;
    if (!storage) return false;
    for (auto entry : module.template getOps<spirv::EntryPointOp>()) {
      entry.setInterfaceAttr(ArrayAttr::get(&context,
          {FlatSymbolRefAttr::get(&context, storage.getSymName())}));
      return true;
    }
    return false;
  }, false); // MLIR valid, but invalid Vulkan1.2/SPIR-V1.3 interface storage class.

  const auto outlinedAttack = [&](const char *name,
                                  const std::function<bool(ModuleOp)> &mutate) {
    auto changed = parseSourceString<ModuleOp>(artifact.outlined_ir, &context);
    error.clear();
    if (!changed) { require(false, "outlined control parse"); return; }
    const auto controlCounts = counts(*changed);
    if (!require(mutate(*changed), std::string(name) + " constructed")) return;
    if (!require(counts(*changed) == controlCounts,
                 std::string(name) + " unchanged operation inventory")) return;
    if (!require(succeeded(verify(*changed)), std::string(name) + " MLIR valid")) return;
    require(!candidate::verifyStrictGemmOutlinedV1(*changed, {2, 3, 4}, error) &&
                !error.empty(), std::string(name) + " closed classifier refusal");
  };
  outlinedAttack("unproved block-id upper bound", [&](auto module) {
    bool changed = false;
    module.walk([&](gpu::BlockIdOp block) {
      if (!changed && block.getDimension() == gpu::Dimension::x) {
        // M=2: x=1 is a legal dispatched block. An explicit upper_bound=1
        // overrides known_grid_size and makes that execution undefined.
        block.setUpperBoundAttr(Builder(&context).getIndexAttr(1));
        changed = true;
      }
    });
    return changed;
  });
  outlinedAttack("foreign nontemporal input hint", [](auto module) {
    bool changed = false;
    module.walk([&](memref::LoadOp load) {
      if (!changed) { load.setNontemporal(true); changed = true; }
    });
    return changed;
  });

  // Exercise the actual decoded graph, rather than editing the pre-serialized
  // module or bypassing the new two-block acceptance path with artifact replay.
  auto decoded = spirv::deserialize(artifact.gemm_words, &context);
  if (!require(decoded && succeeded(verify(*decoded)), "native decoded control validity")) return 1;
  auto decodedEntry = decodedFunction(*decoded);
  if (!require(decodedEntry && decodedEntry.getBody().getBlocks().size() == 2 &&
                   decodedEntry.getBody().back().getNumArguments() == 1,
               "pinned native decoded two-block control") ||
      !require(candidate::verifyStrictGemmShaderV1(*decoded, {2, 3, 4}, false, error),
               "native decoded structural acceptance")) return 1;
  const Counts decodedCounts = counts(*decoded);
  const auto decodedAttack = [&](const char *name,
                                 const std::function<bool(spirv::ModuleOp)> &mutate,
                                 bool sameInventory = true) {
    auto changed = spirv::deserialize(artifact.gemm_words, &context);
    error.clear();
    if (!require(changed && mutate(*changed), std::string(name) + " constructed")) return;
    if (sameInventory && !require(counts(*changed) == decodedCounts,
                                  std::string(name) + " unchanged operation inventory")) return;
    if (!require(succeeded(verify(*changed)), std::string(name) + " MLIR valid")) return;
    require(!candidate::verifyStrictGemmShaderV1(*changed, {2, 3, 4}, false, error) &&
                !error.empty(), std::string(name) + " closed classifier refusal");
  };
  decodedAttack("decoded nonzero forwarded seed", [&](auto module) {
    auto function = decodedFunction(module);
    if (!function || function.getBody().getBlocks().size() != 2) return false;
    auto branch = dyn_cast<spirv::BranchOp>(function.getBody().front().getTerminator());
    auto seed = branch && branch->getNumOperands() == 1
                    ? branch->getOperand(0).template getDefiningOp<spirv::ConstantOp>()
                    : spirv::ConstantOp{};
    if (!seed) return false;
    seed.setValueAttr(Builder(&context).getI32IntegerAttr(1));
    return true;
  });
  decodedAttack("decoded builtin-controlled forwarded seed", [](auto module) {
    auto function = decodedFunction(module);
    if (!function || function.getBody().getBlocks().size() != 2) return false;
    auto branch = dyn_cast<spirv::BranchOp>(function.getBody().front().getTerminator());
    if (!branch || branch->getNumOperands() != 1) return false;
    for (auto &operation : function.getBody().front())
      if (auto index = dyn_cast<spirv::CompositeExtractOp>(operation)) {
        branch->setOperand(0, index->getResult(0));
        return true;
      }
    return false;
  });
  decodedAttack("decoded header K constant changed", [&](auto module) {
    auto loop = decodedLoop(module);
    auto constant = loop ? dyn_cast<spirv::ConstantOp>(loop.getHeaderBlock()->front())
                         : spirv::ConstantOp{};
    if (!constant) return false;
    constant.setValueAttr(Builder(&context).getI32IntegerAttr(2));
    return true;
  });
  decodedAttack("decoded reduction no-progress increment", [&](auto module) {
    auto loop = decodedLoop(module);
    if (!loop) return false;
    auto branch = dyn_cast<spirv::BranchOp>(loop.getContinueBlock()->getTerminator());
    auto next = branch && branch->getNumOperands() == 1
                    ? branch->getOperand(0).template getDefiningOp<spirv::IAddOp>()
                    : spirv::IAddOp{};
    auto one = next ? next->getOperand(1).template getDefiningOp<spirv::ConstantOp>()
                    : spirv::ConstantOp{};
    if (!one) return false;
    one.setValueAttr(Builder(&context).getI32IntegerAttr(0));
    return true;
  });
  decodedAttack("decoded extra predecessor block", [&](auto module) {
    auto function = decodedFunction(module);
    if (!function || function.getBody().getBlocks().size() != 2) return false;
    Block *body = &function.getBody().back();
    auto *extra = new Block;
    function.getBody().push_back(extra);
    OpBuilder builder = OpBuilder::atBlockEnd(extra);
    auto zero = spirv::ConstantOp::create(builder, function.getLoc(),
        builder.getI32Type(), builder.getI32IntegerAttr(0));
    spirv::BranchOp::create(builder, function.getLoc(), body, ValueRange{zero});
    return true;
  }, false);
  decodedAttack("decoded wrapper argument bypassed by concrete zero", [](auto module) {
    auto function = decodedFunction(module);
    auto loop = decodedLoop(module);
    if (!function || !loop || function.getBody().getBlocks().size() != 2) return false;
    auto branch = dyn_cast<spirv::BranchOp>(function.getBody().front().getTerminator());
    auto initial = dyn_cast<spirv::BranchOp>(loop.getEntryBlock()->getTerminator());
    auto &body = function.getBody().back();
    if (!branch || !initial || branch->getNumOperands() != 1 ||
        initial->getNumOperands() != 1 || body.getNumArguments() != 1) return false;
    Value zero = branch->getOperand(0), forwarded = body.getArgument(0);
    initial->setOperand(0, zero);
    // Keep one use of the wrapper argument, but not as the loop's incoming IV.
    // c6eb716's constant-zero OR forwarded-zero condition accepted this foreign
    // form; exact forwarding must use the body argument, as 77fac52 requires.
    OpBuilder builder(loop.getContinueBlock()->getTerminator());
    spirv::IAddOp::create(builder, function.getLoc(), forwarded, zero);
    return true;
  }, false);

  if (failures) return 1;
  llvm::outs() << "PASS " << checks
               << " independent structural adversarial checks; no dispatch/import authority\n";
  return 0;
}
