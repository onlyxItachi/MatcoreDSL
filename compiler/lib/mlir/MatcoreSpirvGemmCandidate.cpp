#include "MatcoreSpirvGemmCandidate.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>

namespace matcore::mdslc::spirv_candidate {
namespace {
using namespace mlir;

bool reject(std::string &error, llvm::StringRef message) {
  error = "strict static SPIR-V GEMM issuer: " + message.str();
  return false;
}
bool attrsOnly(Operation *op, std::initializer_list<llvm::StringRef> allowed,
                std::string &error) {
  for (auto attr : op->getAttrs())
    if (!llvm::is_contained(allowed, attr.getName().strref()))
      return reject(error, "unsupported attribute " + attr.getName().str() +
          " on " + op->getName().getStringRef().str());
  return true;
}
std::string print(Operation *operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation->print(stream, OpPrintingFlags().useLocalScope());
  return result;
}
std::string hash(llvm::StringRef text) {
  const auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(text));
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : bytes) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}
std::string wordIdentity(const std::vector<std::uint32_t> &words) {
  // Explicit byte order, independent of the compiler host's endian convention.
  std::string bytes;
  for (auto word : words)
    for (unsigned shift = 0; shift != 32; shift += 8)
      bytes += char((word >> shift) & 255);
  return hash(bytes);
}
bool integer(Value value, std::int64_t expected) {
  auto op = value.getDefiningOp<arith::ConstantOp>();
  auto attr = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
  return attr && attr.getInt() == expected;
}
bool positiveZero(Value value, bool spv = false) {
  FloatAttr attr;
  if (spv) {
    auto op = value.getDefiningOp<spirv::ConstantOp>();
    if (op) attr = dyn_cast<FloatAttr>(op.getValue());
  } else {
    auto op = value.getDefiningOp<arith::ConstantOp>();
    if (op) attr = dyn_cast<FloatAttr>(op.getValue());
  }
  return attr && attr.getType().isF32() && attr.getValue().isZero() &&
         !attr.getValue().isNegative();
}
bool outputIndex(Value value, gpu::Dimension dimension, std::int64_t extent) {
  if (extent == 1) return integer(value, 0);
  auto op = value.getDefiningOp<gpu::BlockIdOp>();
  return op && op.getDimension() == dimension;
}
bool matrix(Type type, std::int64_t rows, std::int64_t columns) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 2 &&
         memref.getShape() == llvm::ArrayRef<std::int64_t>({rows, columns}) &&
         memref.getElementType().isF32() && memref.getLayout().isIdentity() &&
         !memref.getMemorySpace();
}
bool kernelBody(gpu::GPUFuncOp function, StaticGemmShapeV1 shape, bool fill,
                std::string &error) {
  if (!function || function.getName() != kEntryPointV1 ||
      function.getNumArguments() != (fill ? 1 : 3) ||
      function.getNumResults() || !llvm::hasSingleElement(function.getBody()) ||
      function.getAllArgAttrs() || function.getAllResultAttrs() ||
      !function.getWorkgroupAttributions().empty() ||
      !function.getPrivateAttributions().empty())
    return reject(error, "outlined kernel signature differs");
  const auto local = function->getAttrOfType<DenseI32ArrayAttr>("known_block_size");
  const auto grid = function->getAttrOfType<DenseI32ArrayAttr>("known_grid_size");
  if (!function->hasAttr("gpu.kernel") || !local || !grid ||
      local.asArrayRef() != llvm::ArrayRef<std::int32_t>({1, 1, 1}) ||
      grid.asArrayRef() != llvm::ArrayRef<std::int32_t>(
          {std::int32_t(shape.m), std::int32_t(shape.n), 1}))
    return reject(error, "outlined kernel local/dispatch geometry differs");
  if (fill ? !matrix(function.getArgument(0).getType(), shape.m, shape.n)
           : (!matrix(function.getArgument(0).getType(), shape.m, shape.k) ||
              !matrix(function.getArgument(1).getType(), shape.k, shape.n) ||
              !matrix(function.getArgument(2).getType(), shape.m, shape.n)))
    return reject(error, "outlined kernel static operand types differ");
  bool unknown = false;
  std::string unexpected;
  unsigned loops = 0, returns = 0;
  SmallVector<memref::LoadOp> loads;
  SmallVector<memref::StoreOp> stores;
  SmallVector<arith::MulFOp> multiplies;
  SmallVector<arith::AddFOp> adds;
  scf::ForOp loop;
  function.walk([&](Operation *op) {
    if (!isa<gpu::GPUFuncOp, arith::ConstantOp, gpu::BlockIdOp,
             memref::LoadOp, memref::StoreOp, arith::MulFOp, arith::AddFOp,
             scf::ForOp, scf::YieldOp, gpu::ReturnOp>(op)) unknown = true;
    if (auto candidate = dyn_cast<scf::ForOp>(op)) { ++loops; loop = candidate; }
    if (isa<gpu::ReturnOp>(op)) ++returns;
    if (auto candidate = dyn_cast<memref::LoadOp>(op)) loads.push_back(candidate);
    if (auto candidate = dyn_cast<memref::StoreOp>(op)) stores.push_back(candidate);
    if (auto candidate = dyn_cast<arith::MulFOp>(op)) multiplies.push_back(candidate);
    if (auto candidate = dyn_cast<arith::AddFOp>(op)) adds.push_back(candidate);
    for (auto attr : op->getDiscardableAttrDictionary()) {
      auto name = attr.getName().strref();
      if (isa<gpu::GPUFuncOp>(op) &&
          (name == "known_block_size" || name == "known_grid_size" ||
           name == "gpu.kernel" || name == "workgroup_attributions" ||
           name == "sym_name")) continue;
      unexpected = "unsupported attribute " + name.str() + " on " + op->getName().getStringRef().str();
      unknown = true;
    }
  });
  if (unknown || returns != 1 || stores.size() != 1 ||
      !isa<gpu::ReturnOp>(function.getBody().front().back()))
    return reject(error, "outlined kernel has extra computation/attributes/effects: " + unexpected);
  auto store = stores[0];
  auto indices = store.getIndices();
  if (indices.size() != 2 ||
      !outputIndex(indices[0], gpu::Dimension::x, shape.m) ||
      !outputIndex(indices[1], gpu::Dimension::y, shape.n) ||
      store.getMemRef() != function.getArgument(fill ? 0 : 2))
    return reject(error, "outlined writable range/output indices differ");
  if (fill) {
    if (loops || !loads.empty() || !multiplies.empty() || !adds.empty() ||
        !positiveZero(store.getValue()))
      return reject(error, "outlined fill lost positive-zero overwrite");
    return true;
  }
  if (loads.size() != 3 || multiplies.size() != 1 || adds.size() != 1 ||
      loops != unsigned(shape.k != 1))
    return reject(error, "outlined GEMM scalar reduction graph differs");
  Value reduction;
  if (shape.k != 1) {
    if (!integer(loop.getLowerBound(), 0) ||
        !integer(loop.getUpperBound(), shape.k) ||
        !integer(loop.getStep(), 1) || loop.getNumResults() ||
        loop.getNumRegionIterArgs() || store->getBlock() != loop.getBody() ||
        loop->getBlock() != &function.getBody().front())
      return reject(error, "outlined GEMM changed increasing complete scalar K");
    reduction = loop.getInductionVar();
  } else {
    reduction = loads[0].getIndices()[1];
    if (!integer(reduction, 0) || store->getBlock() != &function.getBody().front())
      return reject(error, "outlined singleton K index differs");
  }
  const SmallVector<Value> expected[] = {
      {indices[0], reduction}, {reduction, indices[1]}, {indices[0], indices[1]}};
  for (unsigned i = 0; i != 3; ++i)
    if (loads[i].getMemRef() != function.getArgument(i) ||
        loads[i].getIndices() != ValueRange(expected[i]) ||
        loads[i]->getBlock() != store->getBlock())
      return reject(error, "outlined GEMM operand/index binding differs");
  auto multiply = multiplies[0];
  auto add = adds[0];
  if (multiply.getLhs() != loads[0].getResult() ||
      multiply.getRhs() != loads[1].getResult() ||
      add.getLhs() != loads[2].getResult() ||
      add.getRhs() != multiply.getResult() || store.getValue() != add.getResult() ||
      multiply.getFastmath() != arith::FastMathFlags::none ||
      add.getFastmath() != arith::FastMathFlags::none)
    return reject(error, "outlined GEMM lost ordered separate strict f32 arithmetic");
  return true;
}

// Affine symbolic indices [constant, block-x, block-y, scalar-k]. The only
// accepted multiplication has a constant operand; no nonlinear/wrapping index
// expression is accepted. Issuance bounds prove these exact expressions safe.
using Linear = std::array<std::int64_t, 4>;
std::optional<Linear> linear(Value value, Value reduction,
                              spirv::ModuleOp module, unsigned depth = 0) {
  if (depth > 32 || !value.getType().isInteger(32)) return {};
  if (value == reduction) return Linear{0, 0, 0, 1};
  if (auto op = value.getDefiningOp<spirv::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return Linear{attr.getInt(), 0, 0, 0};
    return {};
  }
  if (auto op = value.getDefiningOp<spirv::CompositeExtractOp>()) {
    auto load = op->getOperand(0).getDefiningOp<spirv::LoadOp>();
    auto address = load ? load->getOperand(0).getDefiningOp<spirv::AddressOfOp>()
                        : spirv::AddressOfOp{};
    auto global = address ? module.lookupSymbol<spirv::GlobalVariableOp>(address.getVariable())
                          : spirv::GlobalVariableOp{};
    auto builtin = global ? global.getBuiltin() : std::optional<llvm::StringRef>{};
    if (!builtin || *builtin != "WorkgroupId" || op.getIndices().size() != 1)
      return {};
    auto axis = dyn_cast<IntegerAttr>(op.getIndices()[0]);
    if (!axis || axis.getInt() < 0 || axis.getInt() > 1) return {};
    Linear result{0, 0, 0, 0};
    result[axis.getInt() + 1] = 1;
    return result;
  }
  auto *op = value.getDefiningOp();
  if (!op || !isa<spirv::IAddOp, spirv::IMulOp>(op)) return {};
  auto lhs = linear(op->getOperand(0), reduction, module, depth + 1);
  auto rhs = linear(op->getOperand(1), reduction, module, depth + 1);
  if (!lhs || !rhs) return {};
  Linear result{};
  if (isa<spirv::IAddOp>(op)) {
    for (unsigned i = 0; i != 4; ++i) {
      if (__builtin_add_overflow((*lhs)[i], (*rhs)[i], &result[i])) return {};
    }
  } else {
    const auto constant = [](Linear candidate) {
      return candidate[1] == 0 && candidate[2] == 0 && candidate[3] == 0;
    };
    if (!constant(*lhs) && !constant(*rhs)) return {};
    if (!constant(*lhs)) std::swap(lhs, rhs);
    for (unsigned i = 0; i != 4; ++i) {
      if (__builtin_mul_overflow((*lhs)[0], (*rhs)[i], &result[i])) return {};
    }
  }
  return result;
}
bool spvInteger(Value value, std::int64_t expected) {
  auto op = value.getDefiningOp<spirv::ConstantOp>();
  auto attr = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
  return attr && attr.getType().isInteger(32) && attr.getInt() == expected;
}
bool address(Value pointer, unsigned binding, Linear expected, Value reduction,
             spirv::ModuleOp module) {
  auto chain = pointer.getDefiningOp<spirv::AccessChainOp>();
  if (!chain || chain.getIndices().size() != 2 ||
      !spvInteger(chain.getIndices()[0], 0)) return false;
  auto base = chain->getOperand(0).getDefiningOp<spirv::AddressOfOp>();
  auto global = base ? module.lookupSymbol<spirv::GlobalVariableOp>(base.getVariable())
                     : spirv::GlobalVariableOp{};
  auto index = linear(chain.getIndices()[1], reduction, module);
  return global && global.getDescriptorSet() == 0 && global.getBinding() == binding &&
         index && *index == expected;
}

ShaderAbiV1 abi(StaticGemmShapeV1 shape, bool fill) {
  const auto bytes = [](std::int64_t rows, std::int64_t columns) {
    return std::uint64_t(rows) * std::uint64_t(columns) * 4;
  };
  ShaderAbiV1 result{kEntryPointV1, {1, 1, 1},
      {std::uint32_t(shape.m), std::uint32_t(shape.n), 1}, {}};
  if (fill) result.buffers = {{BufferRoleV1::Output, 0, 0, bytes(shape.m, shape.n), 4, 0, true}};
  else result.buffers = {{BufferRoleV1::Lhs, 0, 0, bytes(shape.m, shape.k), 4, 0, false},
      {BufferRoleV1::Rhs, 0, 1, bytes(shape.k, shape.n), 4, 0, false},
      {BufferRoleV1::Output, 0, 2, bytes(shape.m, shape.n), 4, 0, true}};
  return result;
}
std::string abiIdentity(const ShaderAbiV1 &value) {
  std::string result = value.entry_point + ";Logical;GLSL450;GLCompute;local=";
  for (auto extent : value.local_size) result += std::to_string(extent) + ",";
  result += ";dispatch=";
  for (auto extent : value.dispatch) result += std::to_string(extent) + ",";
  for (auto buffer : value.buffers)
    result += ";buffer=" + std::to_string(unsigned(buffer.role)) + "," +
        std::to_string(buffer.descriptor_set) + "," + std::to_string(buffer.binding) +
        "," + std::to_string(buffer.bytes) + "," + std::to_string(buffer.element_stride) +
        "," + std::to_string(buffer.member_offset) + "," + std::to_string(buffer.writable);
  return result;
}
} // namespace

bool validateStaticGemmShapeV1(StaticGemmShapeV1 shape, std::string &error) {
  error.clear();
  if (shape.m <= 0 || shape.k <= 0 || shape.n <= 0)
    return reject(error, "requires fully-static positive M/K/N; zero/dynamic unsupported");
  if (shape.m > kMaxDispatchAxisV1 || shape.n > kMaxDispatchAxisV1)
    return reject(error, "M/N exceed guaranteed Vulkan 1.2 dispatch-axis limit 65535");
  if (shape.k >= std::numeric_limits<std::int32_t>::max())
    return reject(error, "K-loop next induction value must fit signed i32");
  const auto fits = [](std::int64_t rows, std::int64_t columns) {
    return std::uint64_t(rows) <= kMaxStorageBufferElementsV1 / std::uint64_t(columns);
  };
  if (!fits(shape.m, shape.k) || !fits(shape.k, shape.n) || !fits(shape.m, shape.n))
    return reject(error, "MK/KN/MN exceed guaranteed 128MiB storage-buffer range");
  return true;
}

bool verifyStrictGemmOutlinedV1(mlir::ModuleOp module, StaticGemmShapeV1 shape,
                               std::string &error) {
  using namespace mlir;
  if (!validateStaticGemmShapeV1(shape, error)) return false;
  if (!module || failed(verify(module)) || module.getBody()->getOperations().size() != 3)
    return reject(error, "expected one host function and exactly two outlined modules");
  bool extraAttributes = false;
  module.walk([&](Operation *op) {
    if (isa<gpu::GPUFuncOp>(op) || op->getParentOfType<gpu::GPUFuncOp>()) return;
    for (auto attr : op->getDiscardableAttrDictionary()) {
      auto name = attr.getName().strref();
      if (op == module.getOperation() &&
          (name == "gpu.container_module" || name == spirv::getTargetEnvAttrName())) continue;
      extraAttributes = true;
    }
  });
  if (extraAttributes) return reject(error, "outlined host/module acquired extra attributes");
  auto host = module.lookupSymbol<func::FuncOp>(cpu_candidate::kStrictGemmSymbolV1);
  auto fill = module.lookupSymbol<gpu::GPUModuleOp>(kEntryPointV1);
  auto gemm = module.lookupSymbol<gpu::GPUModuleOp>(std::string(kEntryPointV1) + "_0");
  if (!host || !fill || !gemm || host.getNumArguments() != 3 ||
      host.getNumResults() != 1 || !llvm::hasSingleElement(host.getBody()) ||
      !matrix(host.getArgument(0).getType(), shape.m, shape.k) ||
      !matrix(host.getArgument(1).getType(), shape.k, shape.n) ||
      !matrix(host.getArgument(2).getType(), shape.m, shape.n) ||
      host.getResultTypes()[0] != host.getArgument(2).getType() ||
      host.getAllArgAttrs() || host.getAllResultAttrs() ||
      fill.getTargets() || gemm.getTargets() ||
      fill.getBody()->getOperations().size() != 1 ||
      gemm.getBody()->getOperations().size() != 1 ||
      !llvm::hasSingleElement(fill.getOps<gpu::GPUFuncOp>()) ||
      !llvm::hasSingleElement(gemm.getOps<gpu::GPUFuncOp>()))
    return reject(error, "outlined static host/module signature differs");
  SmallVector<gpu::LaunchFuncOp> launches;
  for (auto &op : host.getBody().front()) {
    if (auto launch = dyn_cast<gpu::LaunchFuncOp>(op)) launches.push_back(launch);
    else if (!isa<arith::ConstantOp, func::ReturnOp>(op))
      return reject(error, "outlined host has an unexpected effect or computation");
  }
  if (launches.size() != 2) return reject(error, "fill/GEMM dispatch count differs");
  for (unsigned i = 0; i != 2; ++i) {
    auto launch = launches[i];
    auto kernel = launch.getKernel();
    if (kernel.getRootReference() != (i ? gemm.getName() : fill.getName()) ||
        kernel.getLeafReference() != kEntryPointV1 ||
        !integer(launch.getGridSizeX(), shape.m) ||
        !integer(launch.getGridSizeY(), shape.n) || !integer(launch.getGridSizeZ(), 1) ||
        !integer(launch.getBlockSizeX(), 1) || !integer(launch.getBlockSizeY(), 1) ||
        !integer(launch.getBlockSizeZ(), 1) || launch.getAsyncToken() ||
        !launch.getAsyncDependencies().empty() || launch.getDynamicSharedMemorySize())
      return reject(error, "outlined ordered launch/geometry differs");
    auto arguments = launch.getKernelOperands();
    if (arguments.size() != (i ? 3 : 1)) return reject(error, "launch argument count differs");
    for (unsigned j = 0; j != arguments.size(); ++j)
      if (arguments[j] != host.getArgument(i ? j : 2))
        return reject(error, "fill/GEMM launch storage binding differs");
  }
  auto ret = dyn_cast<func::ReturnOp>(host.getBody().front().back());
  if (!ret || ret.getNumOperands() != 1 || ret.getOperand(0) != host.getArgument(2))
    return reject(error, "outlined host lost destination identity");
  return kernelBody(*fill.getOps<gpu::GPUFuncOp>().begin(), shape, true, error) &&
         kernelBody(*gemm.getOps<gpu::GPUFuncOp>().begin(), shape, false, error);
}

bool verifyStrictGemmShaderV1(mlir::spirv::ModuleOp module,
                             StaticGemmShapeV1 shape, bool fill,
                             std::string &error) {
  using namespace mlir;
  if (!validateStaticGemmShapeV1(shape, error)) return false;
  if (!module || failed(verify(module)) ||
      module.getAddressingModel() != spirv::AddressingModel::Logical ||
      module.getMemoryModel() != spirv::MemoryModel::GLSL450)
    return reject(error, "final shader is not verified Logical/GLSL450");
  if (!attrsOnly(module, {"addressing_model", "memory_model", "vce_triple", "sym_name"}, error))
    return false;
  const auto wantedCaps = std::set<spirv::Capability>{spirv::Capability::Shader,
      spirv::Capability::DenormPreserve, spirv::Capability::SignedZeroInfNanPreserve,
      spirv::Capability::RoundingModeRTE};
  const auto wantedExtensions = std::set<spirv::Extension>{
      spirv::Extension::SPV_KHR_storage_buffer_storage_class,
      spirv::Extension::SPV_KHR_float_controls};
  auto triple = module.getVceTriple();
  if (!triple || triple->getVersion() != spirv::Version::V_1_3 ||
      std::set<spirv::Capability>(triple->getCapabilities().begin(), triple->getCapabilities().end()) != wantedCaps ||
      std::set<spirv::Extension>(triple->getExtensions().begin(), triple->getExtensions().end()) != wantedExtensions)
    return reject(error, "final shader strict capability/extension/version set differs");
  auto functions = module.getOps<spirv::FuncOp>();
  auto entries = module.getOps<spirv::EntryPointOp>();
  if (!llvm::hasSingleElement(functions) || !llvm::hasSingleElement(entries))
    return reject(error, "final shader requires exactly one function and entry point");
  auto function = *functions.begin();
  auto entry = *entries.begin();
  if (function.getName() != kEntryPointV1 || function.getNumArguments() ||
      function.getNumResults() || !llvm::hasSingleElement(function.getBody()) ||
      function.getFunctionControl() != spirv::FunctionControl::None ||
      entry.getFn() != kEntryPointV1 || entry.getExecutionModel() != spirv::ExecutionModel::GLCompute)
    return reject(error, "final shader entry/function identity differs");
  // These analysis-only properties are not serialized. When retained on the
  // compiler-owned lowering, bind their values as well; a native binary round
  // trip legitimately has none of them.
  if (auto local = function->getAttrOfType<DenseI32ArrayAttr>("known_block_size"))
    if (local.asArrayRef() != llvm::ArrayRef<std::int32_t>({1, 1, 1}))
      return reject(error, "final shader retained local-size analysis differs");
  if (auto grid = function->getAttrOfType<DenseI32ArrayAttr>("known_grid_size"))
    if (grid.asArrayRef() != llvm::ArrayRef<std::int32_t>(
            {std::int32_t(shape.m), std::int32_t(shape.n), 1}))
      return reject(error, "final shader retained dispatch analysis differs");
  if (auto workgroup = function->getAttrOfType<IntegerAttr>("workgroup_attributions"))
    if (workgroup.getInt() != 0)
      return reject(error, "final shader acquired workgroup storage");
  const auto shaderAbi = abi(shape, fill);
  for (auto &operation : *module.getBody())
    if (!isa<spirv::FuncOp, spirv::GlobalVariableOp, spirv::EntryPointOp,
             spirv::ExecutionModeOp>(operation))
      return reject(error, "final shader contains an unexpected import/declaration/operation");
  if (!attrsOnly(entry, {"execution_model", "fn", "interface"}, error)) return false;
  unsigned storageCount = 0, builtinCount = 0;
  std::set<unsigned> bindings;
  for (auto global : module.getOps<spirv::GlobalVariableOp>()) {
    if (!attrsOnly(global, {"type", "sym_name", "descriptor_set", "binding", "builtin"}, error))
      return false;
    auto pointer = dyn_cast<spirv::PointerType>(global.getType());
    if (!pointer) return reject(error, "final global is not a logical pointer");
    if (pointer.getStorageClass() == spirv::StorageClass::Input) {
      ++builtinCount;
      auto builtin = global.getBuiltin();
      auto type = dyn_cast<VectorType>(pointer.getPointeeType());
      if (!builtin || *builtin != "WorkgroupId" || !type ||
          type.getShape() != llvm::ArrayRef<std::int64_t>({3}) ||
          !type.getElementType().isInteger(32) || global.getBinding() || global.getDescriptorSet())
        return reject(error, "final shader unexpected builtin input");
      auto interface = entry.getInterface().size() == 1
          ? dyn_cast<FlatSymbolRefAttr>(entry.getInterface()[0]) : FlatSymbolRefAttr{};
      if (!interface || interface.getValue() != global.getSymName())
        return reject(error, "final shader interface is not the exact WorkgroupId variable");
      continue;
    }
    ++storageCount;
    if (pointer.getStorageClass() != spirv::StorageClass::StorageBuffer ||
        global.getDescriptorSet() != 0 || !global.getBinding() ||
        *global.getBinding() >= shaderAbi.buffers.size() ||
        !bindings.insert(*global.getBinding()).second || global.getBuiltin())
      return reject(error, "final shader storage-class/set/binding differs");
    auto structure = dyn_cast<spirv::StructType>(pointer.getPointeeType());
    if (!structure || structure.getNumElements() != 1 || !structure.hasOffset() ||
        structure.getMemberOffset(0) != 0)
      return reject(error, "final shader buffer member layout differs");
    auto array = dyn_cast<spirv::ArrayType>(structure.getElementType(0));
    if (!array || !array.getElementType().isF32() || array.getArrayStride() != 4 ||
        std::uint64_t(array.getNumElements()) * 4 != shaderAbi.buffers[*global.getBinding()].bytes)
      return reject(error, "final shader f32 static array extent/stride/range differs");
    SmallVector<spirv::StructType::MemberDecorationInfo> decorations;
    structure.getMemberDecorations(decorations);
    if (!decorations.empty())
      return reject(error, "final shader buffer member acquired relaxed/foreign decorations");
  }
  if (storageCount != shaderAbi.buffers.size() ||
      builtinCount != unsigned(shape.m != 1 || shape.n != 1) ||
      entry.getInterface().size() != builtinCount)
    return reject(error, "final shader global/interface set differs");
  std::set<spirv::ExecutionMode> modes;
  for (auto mode : module.getOps<spirv::ExecutionModeOp>()) {
    if (!attrsOnly(mode, {"fn", "execution_mode", "values"}, error)) return false;
    if (mode.getFn() != kEntryPointV1 || !modes.insert(mode.getExecutionMode()).second)
      return reject(error, "final shader duplicate/foreign execution mode");
    SmallVector<std::int64_t> values;
    for (auto attr : mode.getValues()) {
      auto integer = dyn_cast<IntegerAttr>(attr);
      if (!integer) return reject(error, "final shader noninteger execution mode");
      values.push_back(integer.getInt());
    }
    if (mode.getExecutionMode() == spirv::ExecutionMode::LocalSize) {
      if (values != SmallVector<std::int64_t>({1, 1, 1}))
        return reject(error, "final shader local geometry differs");
    } else if (values != SmallVector<std::int64_t>({32})) {
      return reject(error, "final shader float-control width differs");
    }
  }
  if (modes != std::set<spirv::ExecutionMode>{spirv::ExecutionMode::LocalSize,
      spirv::ExecutionMode::DenormPreserve, spirv::ExecutionMode::RoundingModeRTE,
      spirv::ExecutionMode::SignedZeroInfNanPreserve})
    return reject(error, "final shader missing/extra strict float controls");

  bool unknown = false;
  unsigned loops = 0, returns = 0;
  spirv::LoopOp loop;
  SmallVector<spirv::LoadOp> loads;
  SmallVector<spirv::StoreOp> stores;
  SmallVector<spirv::FMulOp> multiplies;
  SmallVector<spirv::FAddOp> adds;
  function.walk([&](Operation *op) {
    if (!isa<spirv::FuncOp, spirv::AddressOfOp, spirv::ConstantOp, spirv::LoadOp,
             spirv::CompositeExtractOp, spirv::IMulOp, spirv::IAddOp,
             spirv::AccessChainOp, spirv::StoreOp, spirv::ReturnOp,
             spirv::LoopOp, spirv::BranchOp, spirv::BranchConditionalOp,
             spirv::SLessThanOp, spirv::MergeOp, spirv::FMulOp, spirv::FAddOp>(op)) unknown = true;
    if (auto candidate = dyn_cast<spirv::LoopOp>(op)) { ++loops; loop = candidate; }
    if (isa<spirv::ReturnOp>(op)) ++returns;
    if (auto candidate = dyn_cast<spirv::LoadOp>(op))
      if (candidate->getResult(0).getType().isF32()) loads.push_back(candidate);
    if (auto candidate = dyn_cast<spirv::StoreOp>(op)) stores.push_back(candidate);
    if (auto candidate = dyn_cast<spirv::FMulOp>(op)) multiplies.push_back(candidate);
    if (auto candidate = dyn_cast<spirv::FAddOp>(op)) adds.push_back(candidate);
    bool allowed = false;
    if (isa<spirv::FuncOp>(op))
      allowed = attrsOnly(op, {"sym_name", "function_type", "function_control",
          "known_block_size", "known_grid_size", "workgroup_attributions"}, error);
    else if (isa<spirv::AddressOfOp>(op)) allowed = attrsOnly(op, {"variable"}, error);
    else if (isa<spirv::ConstantOp>(op)) allowed = attrsOnly(op, {"value"}, error);
    else if (isa<spirv::CompositeExtractOp>(op)) allowed = attrsOnly(op, {"indices"}, error);
    else if (isa<spirv::LoopOp>(op)) allowed = attrsOnly(op, {"loop_control"}, error);
    else if (isa<spirv::BranchConditionalOp>(op)) allowed = attrsOnly(op, {"operandSegmentSizes"}, error);
    else if (isa<spirv::FMulOp, spirv::FAddOp>(op)) allowed = attrsOnly(op, {"no_contraction"}, error);
    else allowed = attrsOnly(op, {}, error);
    unknown |= !allowed;
  });
  if (unknown || returns != 1 || stores.size() != 1 ||
      !isa<spirv::ReturnOp>(function.getBody().front().back()) ||
      loops != unsigned(!fill && shape.k != 1) ||
      loads.size() != (fill ? 0U : 3U) || multiplies.size() != (fill ? 0U : 1U) ||
      adds.size() != (fill ? 0U : 1U))
    return reject(error, "final shader extra computation/call/permission/effect: " + error);
  auto store = stores[0];
  Value reduction;
  Block *body = &function.getBody().front();
  if (!fill && shape.k != 1) {
    if (loop.getBody().getBlocks().size() != 4 || loop.getNumResults() ||
        loop.getLoopControl() != spirv::LoopControl::None ||
        loop->getBlock() != body)
      return reject(error, "final shader reduction CFG structure differs");
    auto *entryBlock = loop.getEntryBlock();
    auto *header = loop.getHeaderBlock();
    auto *merge = loop.getMergeBlock();
    auto *continueBlock = loop.getContinueBlock();
    if (!entryBlock || !header || !merge || !continueBlock ||
        entryBlock->getOperations().size() != 1 || header->getOperations().size() != 2 ||
        header->getNumArguments() != 1 || merge->getOperations().size() != 1 ||
        !isa<spirv::MergeOp>(merge->front()))
      return reject(error, "final shader reduction CFG blocks differ");
    reduction = header->getArgument(0);
    auto *initial = entryBlock->getTerminator();
    auto *condition = &header->front();
    auto *branch = header->getTerminator();
    auto *next = continueBlock->getTerminator();
    if (!isa<spirv::BranchOp>(initial) || initial->getNumOperands() != 1 ||
        !spvInteger(initial->getOperand(0), 0) || initial->getSuccessor(0) != header ||
        !isa<spirv::SLessThanOp>(condition) || condition->getOperand(0) != reduction ||
        !spvInteger(condition->getOperand(1), shape.k) ||
        !isa<spirv::BranchConditionalOp>(branch) || branch->getNumOperands() != 1 ||
        branch->getOperand(0) != condition->getResult(0) ||
        branch->getSuccessor(0) != continueBlock || branch->getSuccessor(1) != merge ||
        !isa<spirv::BranchOp>(next) || next->getNumOperands() != 1 ||
        next->getSuccessor(0) != header)
      return reject(error, "final shader reduction CFG edges/bound differ");
    auto increment = next->getOperand(0).getDefiningOp<spirv::IAddOp>();
    if (!increment || increment->getOperand(0) != reduction ||
        !spvInteger(increment->getOperand(1), 1))
      return reject(error, "final shader reduction next IV is not k+1");
    body = continueBlock;
  }
  const Linear output{0, shape.m == 1 ? 0 : shape.n, shape.n == 1 ? 0 : 1, 0};
  if (store->getBlock() != body ||
      !address(store->getOperand(0), fill ? 0 : 2, output, reduction, module))
    return reject(error, "final shader output AccessChain/writable range differs");
  if (fill) {
    if (!positiveZero(store->getOperand(1), true))
      return reject(error, "final shader fill is not positive zero");
    return true;
  }
  const Linear lhs{0, shape.m == 1 ? 0 : shape.k, 0, shape.k == 1 ? 0 : 1};
  const Linear rhs{0, 0, shape.n == 1 ? 0 : 1, shape.k == 1 ? 0 : shape.n};
  const Linear indices[] = {lhs, rhs, output};
  for (unsigned i = 0; i != 3; ++i)
    if (loads[i]->getBlock() != body ||
        !address(loads[i]->getOperand(0), i, indices[i], reduction, module))
      return reject(error, "final shader input/accumulator AccessChain differs");
  auto multiply = multiplies[0];
  auto add = adds[0];
  if (!multiply->hasAttrOfType<UnitAttr>("no_contraction") ||
      !add->hasAttrOfType<UnitAttr>("no_contraction") ||
      multiply->getOperand(0) != loads[0]->getResult(0) ||
      multiply->getOperand(1) != loads[1]->getResult(0) ||
      add->getOperand(0) != loads[2]->getResult(0) ||
      add->getOperand(1) != multiply->getResult(0) ||
      store->getOperand(1) != add->getResult(0) ||
      !multiply->getResult(0).getType().isF32() || !add->getResult(0).getType().isF32())
    return reject(error, "final shader lost separate noncontracting f32 reduction");
  return true;
}

StrictGemmArtifactV1 issueStrictGemmArtifactV1(mlir::MLIRContext &context,
                                              StaticGemmShapeV1 shape) {
  using namespace mlir;
  StrictGemmArtifactV1 result;
  result.shape = shape;
  if (!validateStaticGemmShapeV1(shape, result.error)) return result;
  if (llvm::StringRef(LLVM_VERSION_STRING) != "21.1.8") {
    reject(result.error, "requires exact coherent LLVM/MLIR 21.1.8");
    return result;
  }
  context.loadDialect<gpu::GPUDialect, scf::SCFDialect, spirv::SPIRVDialect>();
  auto stages = cpu_candidate::buildStrictGemmStagesV1(context, shape);
  if (!stages) { result.error = stages.error; return result; }
  result.semantic_ir = print(stages.semantic->getOperation());
  result.structured_ir = print(stages.structured->getOperation());
  result.bufferized_ir = print(stages.bufferized->getOperation());
  auto &module = stages.bufferized;
  const auto target = spirv::TargetEnvAttr::get(spirv::VerCapExtAttr::get(
      spirv::Version::V_1_3, {spirv::Capability::Shader, spirv::Capability::DenormPreserve,
          spirv::Capability::SignedZeroInfNanPreserve, spirv::Capability::RoundingModeRTE},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class,
          spirv::Extension::SPV_KHR_float_controls}, &context),
      spirv::getDefaultResourceLimits(&context), spirv::ClientAPI::Vulkan);
  module->getOperation()->setAttr(spirv::getTargetEnvAttrName(), target);
  PassManager schedule(&context);
  schedule.addPass(createConvertLinalgToParallelLoopsPass());
  schedule.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
  schedule.addPass(createConvertParallelLoopToGpuPass());
  schedule.addPass(createCanonicalizerPass());
  schedule.addPass(createGpuLaunchSinkIndexComputationsPass());
  schedule.addPass(createGpuKernelOutliningPass());
  schedule.addPass(createCanonicalizerPass());
  if (failed(schedule.run(*module)) || !verifyStrictGemmOutlinedV1(*module, shape, result.error)) {
    if (result.error.empty()) reject(result.error, "upstream structured GPU scheduling failed");
    return result;
  }
  result.outlined_ir = print(module->getOperation());
  module->walk([&](gpu::GPUFuncOp kernel) {
    kernel->setAttr(spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr(&context, {1, 1, 1}, std::nullopt, 32));
    for (unsigned i = 0; i != kernel.getNumArguments(); ++i)
      kernel.setArgAttr(i, spirv::getInterfaceVarABIAttrName(),
          spirv::getInterfaceVarABIAttr(0, i, spirv::StorageClass::StorageBuffer, &context));
  });
  PassManager convert(&context);
  convert.addPass(createConvertGPUToSPIRVPass());
  convert.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  if (failed(convert.run(*module))) {
    reject(result.error, "upstream GPU-to-SPIR-V/production ABI lowering failed");
    return result;
  }
  SmallVector<spirv::ModuleOp> shaders;
  for (auto shader : module->getOps<spirv::ModuleOp>()) shaders.push_back(shader);
  if (shaders.size() != 2) {
    reject(result.error, "upstream lowering did not produce exactly fill and GEMM");
    return result;
  }
  for (unsigned index = 0; index != 2; ++index) {
    auto shader = shaders[index];
    auto functions = shader.getOps<spirv::FuncOp>();
    if (!llvm::hasSingleElement(functions)) {
      reject(result.error, "upstream shader lost exact entry function");
      return result;
    }
    auto function = *functions.begin();
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(shader.getBody());
    // EntryPointABI targetWidth emits SignedZeroInfNanPreserve only in 21.1.8.
    // Add the other two required controls via production operations, never
    // patch serialized assembly or invent arithmetic/addresses/control flow.
    spirv::ExecutionModeOp::create(builder, function.getLoc(), function,
        spirv::ExecutionMode::DenormPreserve, llvm::ArrayRef<std::int32_t>({32}));
    spirv::ExecutionModeOp::create(builder, function.getLoc(), function,
        spirv::ExecutionMode::RoundingModeRTE, llvm::ArrayRef<std::int32_t>({32}));
    shader.walk([&](Operation *op) {
      if (isa<spirv::FMulOp, spirv::FAddOp>(op))
        op->setAttr("no_contraction", builder.getUnitAttr());
    });
  }
  PassManager requirements(&context);
  requirements.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
  if (failed(requirements.run(*module))) {
    reject(result.error, "SPIR-V version/capability/extension deduction failed");
    return result;
  }
  for (unsigned index = 0; index != 2; ++index) {
    auto shader = shaders[index];
    auto required = *shader.getVceTriple();
    // Keep the selected target's explicit version, not the pass's minimum
    // version; extension/capability deduction remains independently checked.
    SmallVector<spirv::Capability> capabilities(required.getCapabilities());
    SmallVector<spirv::Extension> extensions(required.getExtensions());
    shader.setVceTripleAttr(spirv::VerCapExtAttr::get(
        spirv::Version::V_1_3, capabilities, extensions, &context));
    if (!verifyStrictGemmShaderV1(shader, shape, index == 0, result.error)) return result;
    SmallVector<std::uint32_t> words;
    spirv::SerializationOptions options;
    options.emitSymbolName = true;
    options.emitDebugInfo = false;
    if (failed(spirv::serialize(shader, words, options)) || words.size() < 5 ||
        words[0] != 0x07230203 || words[1] != 0x00010300 || words[4] != 0) {
      reject(result.error, "production SPIR-V serialization/version failed");
      return result;
    }
    // Inspect the actual serialized word stream through upstream's native
    // deserializer, not just the pre-serialization module. This is an internal
    // round trip of freshly compiler-issued bytes, never an import entry point.
    auto decoded = spirv::deserialize(words, &context);
    if (!decoded || !verifyStrictGemmShaderV1(*decoded, shape, index == 0, result.error)) {
      if (result.error.empty()) reject(result.error, "serialized shader native round-trip inspection failed");
      return result;
    }
    (index ? result.gemm_spirv_ir : result.fill_spirv_ir) = print(shader.getOperation());
    (index ? result.gemm_words : result.fill_words).assign(words.begin(), words.end());
  }
  result.fill_abi = abi(shape, true);
  result.gemm_abi = abi(shape, false);
  result.manifest = "matcore-closed-static-spirv-gemm-v1\ncompiler=" LLVM_VERSION_STRING
      "\nauthority=compiler-private-issuance-only;no-source-runtime-candidate\n"
      "target=spirv1.3-vulkan1.2-logical-glsl450;device-availability-unproven\n"
      "semantics=positive-zero;increasing-scalar-k;separate-f32-mul-add;no-contraction;"
      "rte;denorm-preserve;signed-zero-inf-nan-preserve\n"
      "pipeline=closed-semantic-static;structured;one-shot-bufferize;"
      "linalg-parallel;gpu-map;parallel-to-gpu;canonicalize;sink-index;outline;"
      "canonicalize;production-abi;gpu-to-spirv;lower-abi;strict-controls;update-vce;serialize;native-deserialize-check\n"
      "dependency=fill-output-to-gemm-output;ordered-dispatch;external-memory-barrier-required\n"
      "family=positive;M,N<=65535;K<INT32_MAX;MK,KN,MN<=33554432\n";
  result.manifest += "MKN=" + std::to_string(shape.m) + "," + std::to_string(shape.k) + "," +
      std::to_string(shape.n) + "\nfill-abi=" + abiIdentity(result.fill_abi) +
      "\ngemm-abi=" + abiIdentity(result.gemm_abi) + "\n";
  const std::pair<const char *, const std::string *> stagesToBind[] = {
      {"semantic", &result.semantic_ir}, {"structured", &result.structured_ir},
      {"bufferized", &result.bufferized_ir}, {"outlined", &result.outlined_ir},
      {"fill-spirv", &result.fill_spirv_ir}, {"gemm-spirv", &result.gemm_spirv_ir}};
  for (auto [name, text] : stagesToBind)
    result.manifest += std::string(name) + "-sha256=" + hash(*text) + "\n";
  result.manifest += "fill-words-sha256=" + wordIdentity(result.fill_words) +
      "\ngemm-words-sha256=" + wordIdentity(result.gemm_words) + "\n";
  return result;
}

bool verifyStrictGemmArtifactV1(mlir::MLIRContext &context,
                               const StrictGemmArtifactV1 &artifact,
                               std::string &error) {
  error.clear();
  if (!artifact) return reject(error, "artifact is empty or already failed");
  auto replay = issueStrictGemmArtifactV1(context, artifact.shape);
  if (!replay) { error = replay.error; return false; }
  if (artifact.semantic_ir != replay.semantic_ir || artifact.structured_ir != replay.structured_ir ||
      artifact.bufferized_ir != replay.bufferized_ir || artifact.outlined_ir != replay.outlined_ir ||
      artifact.fill_spirv_ir != replay.fill_spirv_ir || artifact.gemm_spirv_ir != replay.gemm_spirv_ir ||
      artifact.fill_words != replay.fill_words || artifact.gemm_words != replay.gemm_words ||
      artifact.fill_abi != replay.fill_abi || artifact.gemm_abi != replay.gemm_abi ||
      artifact.manifest != replay.manifest)
    return reject(error, "complete closed graph/binary/ABI replay differs; supplied artifacts confer no authority");
  return true;
}
} // namespace matcore::mdslc::spirv_candidate
