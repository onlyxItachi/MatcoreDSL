#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "matcore/jit_runner.h"
#include "matcore/kernel_ir.h"

namespace nb = nanobind;

namespace {

struct Invocation {
  matcore::KernelIR kernel;
  matcore::TargetKind target;
  std::vector<matcore::RuntimeTensorView> tensors;
  std::vector<nb::object> keepalive_tensors;
  std::vector<std::string> tensor_dtypes;
};

struct ParsedTensor {
  std::uintptr_t data_ptr = 0;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> element_strides;
  bool c_contiguous = false;
  std::string dtype_name;
};

std::string toString(nb::handle value) {
  return nb::cast<std::string>(nb::str(value));
}

std::string toLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

std::int64_t toInt64(nb::handle value, std::string_view field_name) {
  try {
    return nb::cast<std::int64_t>(value);
  } catch (const nb::cast_error &) {
    throw std::runtime_error("Expected integer field '" + std::string(field_name) +
                             "'.");
  }
}

bool hasKey(const nb::dict &obj, const char *key) {
  return obj.contains(nb::str(key));
}

nb::object requireKey(const nb::dict &obj, const char *key) {
  nb::str py_key(key);
  if (!obj.contains(py_key)) {
    throw std::runtime_error("Missing required key '" + std::string(key) + "'.");
  }
  return obj[py_key];
}

std::vector<std::int64_t> parseInt64Sequence(nb::handle value,
                                             std::string_view field_name) {
  std::vector<std::int64_t> result;

  if (nb::isinstance<nb::list>(value)) {
    for (nb::handle item : nb::borrow<nb::list>(value)) {
      result.push_back(toInt64(item, field_name));
    }
    return result;
  }

  if (nb::isinstance<nb::tuple>(value)) {
    for (nb::handle item : nb::borrow<nb::tuple>(value)) {
      result.push_back(toInt64(item, field_name));
    }
    return result;
  }

  throw std::runtime_error("Field '" + std::string(field_name) +
                           "' must be a list or tuple.");
}

std::vector<std::string> parseStringSequence(nb::handle value,
                                             std::string_view field_name) {
  std::vector<std::string> result;

  if (nb::isinstance<nb::list>(value)) {
    for (nb::handle item : nb::borrow<nb::list>(value)) {
      result.push_back(toString(item));
    }
    return result;
  }

  if (nb::isinstance<nb::tuple>(value)) {
    for (nb::handle item : nb::borrow<nb::tuple>(value)) {
      result.push_back(toString(item));
    }
    return result;
  }

  throw std::runtime_error("Field '" + std::string(field_name) +
                           "' must be a list or tuple.");
}

std::vector<nb::handle> parseObjectSequence(nb::handle value,
                                            std::string_view field_name) {
  std::vector<nb::handle> result;

  if (nb::isinstance<nb::list>(value)) {
    for (nb::handle item : nb::borrow<nb::list>(value)) {
      result.push_back(item);
    }
    return result;
  }

  if (nb::isinstance<nb::tuple>(value)) {
    for (nb::handle item : nb::borrow<nb::tuple>(value)) {
      result.push_back(item);
    }
    return result;
  }

  throw std::runtime_error("Field '" + std::string(field_name) +
                           "' must be a list or tuple.");
}

std::vector<std::int64_t> computeContiguousElementStrides(
    const std::vector<std::int64_t> &shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }

  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * std::max<std::int64_t>(shape[i], 1);
  }
  return strides;
}

std::string normalizeDTypeName(std::string name) {
  name = toLower(std::move(name));
  if (name == "single") {
    return "float32";
  }
  if (name == "half") {
    return "float16";
  }
  if (name == "bf16") {
    return "bfloat16";
  }
  return name;
}

std::string parseSupportedDType(const nb::object &tensor, std::size_t index) {
  std::string dtype_name =
      normalizeDTypeName(toString(tensor.attr("dtype").attr("name")));
  if (dtype_name != "float32" && dtype_name != "float16" &&
      dtype_name != "bfloat16") {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " uses unsupported dtype '" + dtype_name +
                             "'. Supported dtypes: float32, float16, bfloat16.");
  }
  return dtype_name;
}

matcore::TensorDType parseRuntimeTensorDType(const std::string &dtype_name) {
  if (dtype_name == "float32") {
    return matcore::TensorDType::kFloat32;
  }
  if (dtype_name == "float16") {
    return matcore::TensorDType::kFloat16;
  }
  if (dtype_name == "bfloat16") {
    return matcore::TensorDType::kBFloat16;
  }
  throw std::runtime_error("Unsupported runtime tensor dtype '" + dtype_name + "'.");
}

ParsedTensor parseTensorArgument(const nb::object &tensor, std::size_t index) {
  ParsedTensor parsed;
  parsed.dtype_name = parseSupportedDType(tensor, index);
  parsed.c_contiguous = nb::cast<bool>(tensor.attr("flags").attr("c_contiguous"));
  if (!parsed.c_contiguous) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " must be C-contiguous for zero-copy execution.");
  }

  parsed.shape = parseInt64Sequence(tensor.attr("shape"), "shape");

  const std::int64_t item_size =
      nb::cast<std::int64_t>(tensor.attr("dtype").attr("itemsize"));
  if (item_size <= 0) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " has invalid itemsize.");
  }

  nb::object strides_obj = tensor.attr("strides");
  if (strides_obj.is_none()) {
    parsed.element_strides = computeContiguousElementStrides(parsed.shape);
  } else {
    std::vector<std::int64_t> byte_strides =
        parseInt64Sequence(strides_obj, "strides");
    parsed.element_strides.reserve(byte_strides.size());
    for (std::int64_t byte_stride : byte_strides) {
      if (byte_stride % item_size != 0) {
        throw std::runtime_error("Tensor argument " + std::to_string(index) +
                                 " has non-integral element stride.");
      }
      parsed.element_strides.push_back(byte_stride / item_size);
    }
  }

  nb::object array_interface_obj = tensor.attr("__array_interface__");
  if (!nb::isinstance<nb::dict>(array_interface_obj)) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " must expose a valid __array_interface__ dict.");
  }
  nb::dict array_interface = nb::cast<nb::dict>(array_interface_obj);
  if (!array_interface.contains(nb::str("data"))) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " is missing __array_interface__['data'].");
  }

  nb::object data_obj = array_interface[nb::str("data")];
  if (!nb::isinstance<nb::tuple>(data_obj)) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " has malformed __array_interface__['data'].");
  }
  nb::tuple data_tuple = nb::cast<nb::tuple>(data_obj);
  if (data_tuple.size() == 0) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " has empty __array_interface__['data'] tuple.");
  }

  parsed.data_ptr = nb::cast<std::uintptr_t>(data_tuple[0]);
  if (parsed.data_ptr == 0) {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " has null data pointer.");
  }
  return parsed;
}

std::unordered_map<std::string, std::string> parseKernelDeclaredDtypes(
    const nb::dict &kernel_obj) {
  std::unordered_map<std::string, std::string> out;
  if (!hasKey(kernel_obj, "tensor_dtypes")) {
    return out;
  }
  for (nb::handle entry : parseObjectSequence(kernel_obj["tensor_dtypes"],
                                              "tensor_dtypes")) {
    if (!nb::isinstance<nb::dict>(entry)) {
      throw std::runtime_error("Each tensor_dtypes entry must be a dict.");
    }
    nb::dict entry_dict = nb::borrow<nb::dict>(entry);
    std::string symbol = toString(requireKey(entry_dict, "symbol"));
    std::string dtype = normalizeDTypeName(toString(requireKey(entry_dict, "dtype")));
    out[symbol] = dtype;
  }
  return out;
}

void appendTensorDtypeMetadata(
    matcore::KernelIR &kernel, const std::vector<matcore::RuntimeTensorView> &tensors,
    const std::vector<std::string> &dtypes) {
  const std::size_t count = std::min(tensors.size(), dtypes.size());
  for (std::size_t i = 0; i < count; ++i) {
    matcore::AssignOp op;
    op.output = "__dtype__" + tensors[i].symbol;
    op.value = dtypes[i];
    kernel.ops.emplace_back(std::move(op));
  }
}

matcore::LoopRange parseLoop(const nb::dict &obj) {
  matcore::LoopRange loop;
  loop.var = toString(requireKey(obj, "var"));
  loop.lower = hasKey(obj, "lower") ? toInt64(obj["lower"], "lower") : 0;
  loop.upper = hasKey(obj, "upper") ? toInt64(obj["upper"], "upper") : 0;
  loop.step = hasKey(obj, "step") ? toInt64(obj["step"], "step") : 1;
  return loop;
}

matcore::KernelOp parseOp(const nb::dict &op_obj) {
  std::string kind;
  if (hasKey(op_obj, "op")) {
    kind = toLower(toString(op_obj["op"]));
  } else if (hasKey(op_obj, "kind")) {
    kind = toLower(toString(op_obj["kind"]));
  } else {
    throw std::runtime_error("Kernel op is missing 'op'/'kind' discriminator.");
  }

  if (kind == "load") {
    matcore::LoadOp op;
    op.output = toString(requireKey(op_obj, "output"));
    op.tensor = toString(requireKey(op_obj, "tensor"));
    op.indices = hasKey(op_obj, "indices")
                     ? parseStringSequence(op_obj["indices"], "indices")
                     : std::vector<std::string>{};
    return op;
  }

  if (kind == "store") {
    matcore::StoreOp op;
    op.tensor = toString(requireKey(op_obj, "tensor"));
    if (hasKey(op_obj, "value")) {
      op.value = toString(op_obj["value"]);
    } else if (hasKey(op_obj, "tile")) {
      op.value = toString(op_obj["tile"]);
    } else {
      throw std::runtime_error("Store op must include 'value' (or legacy 'tile').");
    }
    op.indices = hasKey(op_obj, "indices")
                     ? parseStringSequence(op_obj["indices"], "indices")
                     : std::vector<std::string>{};
    return op;
  }

  if (kind == "matmul" || kind == "mma") {
    matcore::MatMulOp op;
    op.output = toString(requireKey(op_obj, "output"));
    op.lhs = hasKey(op_obj, "lhs") ? toString(op_obj["lhs"]) : toString(requireKey(op_obj, "a"));
    op.rhs = hasKey(op_obj, "rhs") ? toString(op_obj["rhs"]) : toString(requireKey(op_obj, "b"));
    return op;
  }

  if (kind == "assign" || kind == "bind") {
    matcore::AssignOp op;
    op.output = toString(requireKey(op_obj, "output"));
    if (hasKey(op_obj, "value")) {
      op.value = toString(op_obj["value"]);
    } else if (hasKey(op_obj, "expr")) {
      op.value = toString(op_obj["expr"]);
    } else {
      throw std::runtime_error("Assign op must include 'value' (or legacy 'expr').");
    }
    return op;
  }

  throw std::runtime_error("Unsupported kernel op kind '" + kind + "'.");
}

matcore::KernelIR parseKernelIR(const nb::dict &kernel_obj) {
  matcore::KernelIR kernel;
  if (hasKey(kernel_obj, "kernel_name")) {
    kernel.kernel_name = toString(kernel_obj["kernel_name"]);
  } else {
    kernel.kernel_name = toString(requireKey(kernel_obj, "name"));
  }

  if (hasKey(kernel_obj, "params")) {
    kernel.params = parseStringSequence(kernel_obj["params"], "params");
  }

  if (hasKey(kernel_obj, "loops")) {
    for (nb::handle loop_handle : parseObjectSequence(kernel_obj["loops"], "loops")) {
      if (!nb::isinstance<nb::dict>(loop_handle)) {
        throw std::runtime_error("Each loop entry must be a dict.");
      }
      kernel.loops.push_back(parseLoop(nb::borrow<nb::dict>(loop_handle)));
    }
  }

  if (hasKey(kernel_obj, "ops")) {
    for (nb::handle op_handle : parseObjectSequence(kernel_obj["ops"], "ops")) {
      if (!nb::isinstance<nb::dict>(op_handle)) {
        throw std::runtime_error("Each op entry must be a dict.");
      }
      kernel.ops.push_back(parseOp(nb::borrow<nb::dict>(op_handle)));
    }
  }

  return kernel;
}

matcore::TargetKind parseTarget(const std::string &target) {
  const std::string normalized = toLower(target);

  if (normalized == "x86-auto" || normalized == "x86auto" || normalized == "x86") {
    return matcore::TargetKind::kX86Auto;
  }
  if (normalized == "x86-avx2" || normalized == "x86_avx2") {
    return matcore::TargetKind::kX86AVX2;
  }
  if (normalized == "x86-avx512" || normalized == "x86_avx512") {
    return matcore::TargetKind::kX86AVX512;
  }
  if (normalized == "nvidia-dgpu" || normalized == "nvidia_dgpu" ||
      normalized == "nvptx") {
    return matcore::TargetKind::kNVPTX;
  }
  if (normalized == "amd-igpu" || normalized == "amd_igpu" ||
      normalized == "amdgcn") {
    return matcore::TargetKind::kAMDGCN;
  }
  if (normalized == "amd-npu" || normalized == "amd_npu" ||
      normalized == "npu") {
    return matcore::TargetKind::kNPU;
  }
  if (normalized == "arm") {
    return matcore::TargetKind::kARM;
  }
  if (normalized == "tpu") {
    return matcore::TargetKind::kTPU;
  }

  throw std::runtime_error(
      "Unsupported target '" + target +
      "'. Supported targets: x86-auto, x86-avx2, x86-avx512, amd-igpu, nvidia-dgpu, amd-npu.");
}

Invocation buildInvocation(const nb::dict &kernel_obj, const std::string &target,
                           const nb::args &tensor_args) {
  Invocation invocation;
  invocation.kernel = parseKernelIR(kernel_obj);
  invocation.target = parseTarget(target);
  const auto kernel_declared_dtypes = parseKernelDeclaredDtypes(kernel_obj);

  invocation.keepalive_tensors.reserve(tensor_args.size());
  invocation.tensors.reserve(tensor_args.size());
  invocation.tensor_dtypes.reserve(tensor_args.size());

  for (std::size_t i = 0; i < tensor_args.size(); ++i) {
    nb::object tensor = nb::borrow<nb::object>(tensor_args[i]);
    ParsedTensor parsed = parseTensorArgument(tensor, i);
    invocation.keepalive_tensors.push_back(tensor);

    matcore::RuntimeTensorView view;
    if (i < invocation.kernel.params.size()) {
      view.symbol = invocation.kernel.params[i];
    } else {
      view.symbol = "arg" + std::to_string(i);
    }

    if (auto it = kernel_declared_dtypes.find(view.symbol);
        it != kernel_declared_dtypes.end() && it->second != parsed.dtype_name) {
      throw std::runtime_error("Tensor dtype mismatch for symbol '" + view.symbol +
                               "': frontend declared '" + it->second +
                               "' but runtime received '" + parsed.dtype_name + "'.");
    }

    view.data = reinterpret_cast<decltype(view.data)>(parsed.data_ptr);
    view.dtype = parseRuntimeTensorDType(parsed.dtype_name);
    view.c_contiguous = parsed.c_contiguous;
    view.shape = std::move(parsed.shape);
    view.strides = std::move(parsed.element_strides);

    invocation.tensor_dtypes.push_back(parsed.dtype_name);
    invocation.tensors.push_back(std::move(view));
  }

  appendTensorDtypeMetadata(invocation.kernel, invocation.tensors,
                            invocation.tensor_dtypes);
  return invocation;
}

void triggerCompilation(Invocation &invocation) {
  nb::gil_scoped_release release;
  matcore::compileAndRun(invocation.kernel, invocation.target, invocation.tensors);
}

}  // namespace

NB_MODULE(_matcore_native, m) {
  m.doc() = "MatCore nanobind bridge";

  m.def("compile_and_run",
        [](nb::dict kernel_ir, const std::string &target, nb::args tensors) {
          Invocation invocation = buildInvocation(kernel_ir, target, tensors);
          triggerCompilation(invocation);
        },
        "Compile and run a kernel IR with zero-copy tensor views.");
}
