#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "matcore/jit_runner.h"
#include "matcore/kernel_ir.h"
#include "matcore/device_buffer.h"
#include "matcore/observability.h"
#include "matcore/plan.h"
#include "matcore/target_registry.h"

namespace nb = nanobind;

namespace {

struct Invocation {
  matcore::KernelIR kernel;
  matcore::RequestedTargetProfile target_profile;
  std::vector<matcore::RuntimeTensorView> tensors;
  std::vector<nb::object> keepalive_tensors;
  std::vector<std::string> tensor_dtypes;
};

struct ParsedTensor {
  std::uintptr_t data_ptr = 0;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> element_strides;
  bool c_contiguous = false;
  bool is_device_resident = false;
  std::string dtype_name;
  matcore::QuantizationParams quantization;
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

float toFloat(nb::handle value, std::string_view field_name) {
  try {
    return static_cast<float>(nb::cast<double>(value));
  } catch (const nb::cast_error &) {
    throw std::runtime_error("Expected floating-point field '" +
                             std::string(field_name) + "'.");
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
  if (name == "i8") {
    return "int8";
  }
  if (name == "float8e4m3fn" || name == "f8e4m3fn" || name == "fp8_e4m3fn" ||
      name == "fp8-e4m3fn" || name == "e4m3fn") {
    return "float8_e4m3fn";
  }
  return name;
}

std::int64_t expectedStorageBytesForDType(const std::string &dtype_name) {
  if (dtype_name == "float32") {
    return 4;
  }
  if (dtype_name == "int32") {
    return 4;
  }
  if (dtype_name == "float16" || dtype_name == "bfloat16") {
    return 2;
  }
  if (dtype_name == "int8" || dtype_name == "float8_e4m3fn") {
    return 1;
  }
  throw std::runtime_error("Unsupported dtype '" + dtype_name + "'.");
}

bool dtypeSupportsQuantization(const std::string &dtype_name) {
  return dtype_name == "int8";
}

std::string parseSupportedDType(const nb::object &tensor, std::size_t index) {
  std::string dtype_name;
  if (nb::hasattr(tensor, "matcore_dtype")) {
    dtype_name = normalizeDTypeName(toString(tensor.attr("matcore_dtype")));
  } else {
    dtype_name = normalizeDTypeName(toString(tensor.attr("dtype").attr("name")));
  }
  if (dtype_name != "float32" && dtype_name != "float16" &&
      dtype_name != "bfloat16" && dtype_name != "int8" &&
      dtype_name != "int32" &&
      dtype_name != "float8_e4m3fn") {
    throw std::runtime_error("Tensor argument " + std::to_string(index) +
                             " uses unsupported dtype '" + dtype_name +
                             "'. Supported dtypes: float32, float16, bfloat16, "
                             "int8, int32, float8_e4m3fn.");
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
  if (dtype_name == "int8") {
    return matcore::TensorDType::kInt8;
  }
  if (dtype_name == "int32") {
    return matcore::TensorDType::kInt32;
  }
  if (dtype_name == "float8_e4m3fn") {
    return matcore::TensorDType::kFloat8E4M3FN;
  }
  throw std::runtime_error("Unsupported runtime tensor dtype '" + dtype_name + "'.");
}

matcore::QuantizationParams parseQuantizationConfig(const nb::dict &obj,
                                                    bool default_enabled) {
  matcore::QuantizationParams quant;
  quant.enabled = default_enabled;
  if (hasKey(obj, "enabled")) {
    quant.enabled = nb::cast<bool>(obj["enabled"]);
  }
  if (hasKey(obj, "scale")) {
    quant.scale = toFloat(obj["scale"], "scale");
    quant.enabled = true;
  }
  if (hasKey(obj, "zero_point")) {
    quant.zero_point =
        static_cast<std::int32_t>(toInt64(obj["zero_point"], "zero_point"));
    quant.enabled = true;
  }
  return quant;
}

matcore::QuantizationParams parseTensorQuantization(const nb::object &tensor,
                                                    const std::string &dtype_name) {
  matcore::QuantizationParams quant;
  quant.enabled = false;

  if (nb::hasattr(tensor, "matcore_quantization")) {
    nb::object quant_obj = tensor.attr("matcore_quantization");
    if (!nb::isinstance<nb::dict>(quant_obj)) {
      throw std::runtime_error(
          "Tensor matcore_quantization must be a dict when provided.");
    }
    quant = parseQuantizationConfig(nb::cast<nb::dict>(quant_obj),
                                    /*default_enabled=*/false);
  }

  if (nb::hasattr(tensor, "matcore_quant_enabled")) {
    quant.enabled = nb::cast<bool>(tensor.attr("matcore_quant_enabled"));
  }
  if (nb::hasattr(tensor, "matcore_scale")) {
    quant.scale = toFloat(tensor.attr("matcore_scale"), "matcore_scale");
    quant.enabled = true;
  }
  if (nb::hasattr(tensor, "matcore_zero_point")) {
    quant.zero_point = static_cast<std::int32_t>(
        toInt64(tensor.attr("matcore_zero_point"), "matcore_zero_point"));
    quant.enabled = true;
  }

  if (!dtypeSupportsQuantization(dtype_name)) {
    quant.enabled = false;
    quant.scale = 1.0f;
    quant.zero_point = 0;
  }
  return quant;
}

ParsedTensor parseTensorArgument(const nb::object &tensor, std::size_t index) {
  ParsedTensor parsed;

  // Check if this is a DeviceTensor (GPU-resident).
  if (nb::hasattr(tensor, "_matcore_device_tensor")) {
    nb::object handle_obj = tensor.attr("device_ptr");
    auto &handle = nb::cast<matcore::DeviceBufferHandle &>(handle_obj);
    if (!matcore::matcore_device_is_valid(handle)) {
      throw std::runtime_error("Tensor argument " + std::to_string(index) +
                               " is a DeviceTensor with an invalid or freed handle.");
    }
    parsed.data_ptr = static_cast<std::uintptr_t>(handle.ptr);
    parsed.is_device_resident = true;
    parsed.dtype_name = nb::cast<std::string>(tensor.attr("dtype"));
    parsed.shape = parseInt64Sequence(tensor.attr("shape"), "shape");
    parsed.element_strides = parseInt64Sequence(tensor.attr("strides"), "strides");
    parsed.c_contiguous = true;  // DeviceTensor enforces contiguous in to_device()
    return parsed;
  }

  // Standard numpy array path.
  parsed.dtype_name = parseSupportedDType(tensor, index);
  parsed.quantization = parseTensorQuantization(tensor, parsed.dtype_name);
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
  const std::int64_t expected_item_size =
      expectedStorageBytesForDType(parsed.dtype_name);
  if (item_size != expected_item_size) {
    throw std::runtime_error(
        "Tensor argument " + std::to_string(index) + " declared logical dtype '" +
        parsed.dtype_name + "' expects itemsize " +
        std::to_string(expected_item_size) + " but runtime storage itemsize is " +
        std::to_string(item_size) + ".");
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
    expectedStorageBytesForDType(dtype);
    out[symbol] = dtype;
  }
  return out;
}

std::unordered_map<std::string, matcore::QuantizationParams>
parseKernelDeclaredTensorQuantization(const nb::dict &kernel_obj) {
  std::unordered_map<std::string, matcore::QuantizationParams> out;
  if (!hasKey(kernel_obj, "tensor_quantization")) {
    return out;
  }

  for (nb::handle entry : parseObjectSequence(kernel_obj["tensor_quantization"],
                                              "tensor_quantization")) {
    if (!nb::isinstance<nb::dict>(entry)) {
      throw std::runtime_error("Each tensor_quantization entry must be a dict.");
    }
    nb::dict entry_dict = nb::borrow<nb::dict>(entry);
    std::string symbol = toString(requireKey(entry_dict, "symbol"));
    out[symbol] = parseQuantizationConfig(entry_dict, /*default_enabled=*/false);
  }
  return out;
}

matcore::QuantizationParams parseKernelGlobalQuantization(const nb::dict &kernel_obj) {
  matcore::QuantizationParams quant;
  quant.enabled = false;

  if (hasKey(kernel_obj, "global_quantization")) {
    nb::object global_obj = kernel_obj["global_quantization"];
    if (!nb::isinstance<nb::dict>(global_obj)) {
      throw std::runtime_error("global_quantization must be a dict.");
    }
    quant = parseQuantizationConfig(nb::cast<nb::dict>(global_obj),
                                    /*default_enabled=*/false);
  }

  // Legacy compatibility: top-level quant keys can still configure global params.
  if (hasKey(kernel_obj, "quant_scale")) {
    quant.scale = toFloat(kernel_obj["quant_scale"], "quant_scale");
    quant.enabled = true;
  }
  if (hasKey(kernel_obj, "quant_zero_point")) {
    quant.zero_point =
        static_cast<std::int32_t>(toInt64(kernel_obj["quant_zero_point"],
                                          "quant_zero_point"));
    quant.enabled = true;
  }
  if (hasKey(kernel_obj, "quant_enabled")) {
    quant.enabled = nb::cast<bool>(kernel_obj["quant_enabled"]);
  }

  return quant;
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

  for (const matcore::RuntimeTensorView &tensor : tensors) {
    if (!tensor.quantization.enabled) {
      continue;
    }
    matcore::AssignOp scale_op;
    scale_op.output = "__qscale__" + tensor.symbol;
    scale_op.value = std::to_string(tensor.quantization.scale);
    kernel.ops.emplace_back(std::move(scale_op));

    matcore::AssignOp zp_op;
    zp_op.output = "__qzp__" + tensor.symbol;
    zp_op.value = std::to_string(tensor.quantization.zero_point);
    kernel.ops.emplace_back(std::move(zp_op));
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

bool isOpDiscriminator(const std::string &value) {
  return value == "load" || value == "store" || value == "matmul" ||
         value == "mma" || value == "assign" || value == "bind" ||
         value == "transpose" || value == "elementwise" || value == "cast";
}

matcore::ElementwiseKind parseElementwiseKind(const nb::dict &op_obj) {
  std::string raw_kind;
  if (hasKey(op_obj, "elementwise_kind")) {
    raw_kind = toLower(toString(op_obj["elementwise_kind"]));
  } else if (hasKey(op_obj, "kind")) {
    raw_kind = toLower(toString(op_obj["kind"]));
  } else {
    throw std::runtime_error(
        "Elementwise op must include 'kind' (or 'elementwise_kind').");
  }

  if (raw_kind == "add") {
    return matcore::ElementwiseKind::kAdd;
  }
  if (raw_kind == "sub") {
    return matcore::ElementwiseKind::kSub;
  }
  if (raw_kind == "mul") {
    return matcore::ElementwiseKind::kMul;
  }
  if (raw_kind == "div") {
    return matcore::ElementwiseKind::kDiv;
  }
  if (raw_kind == "relu") {
    return matcore::ElementwiseKind::kReLU;
  }
  if (raw_kind == "gelu") {
    return matcore::ElementwiseKind::kGELU;
  }
  if (raw_kind == "sigmoid") {
    return matcore::ElementwiseKind::kSigmoid;
  }
  if (raw_kind == "neg") {
    return matcore::ElementwiseKind::kNeg;
  }
  if (raw_kind == "abs") {
    return matcore::ElementwiseKind::kAbs;
  }
  if (raw_kind == "sqrt") {
    return matcore::ElementwiseKind::kSqrt;
  }
  if (raw_kind == "exp") {
    return matcore::ElementwiseKind::kExp;
  }

  throw std::runtime_error("Unsupported elementwise kind '" + raw_kind + "'.");
}

bool isUnaryElementwise(matcore::ElementwiseKind kind) {
  return kind == matcore::ElementwiseKind::kReLU ||
         kind == matcore::ElementwiseKind::kGELU ||
         kind == matcore::ElementwiseKind::kSigmoid ||
         kind == matcore::ElementwiseKind::kNeg ||
         kind == matcore::ElementwiseKind::kAbs ||
         kind == matcore::ElementwiseKind::kSqrt ||
         kind == matcore::ElementwiseKind::kExp;
}

std::optional<std::string> parseOptionalString(const nb::dict &obj,
                                               const char *field_name) {
  if (!hasKey(obj, field_name)) {
    return std::nullopt;
  }
  nb::handle value = obj[nb::str(field_name)];
  if (value.is_none()) {
    return std::nullopt;
  }
  return toString(value);
}

matcore::KernelOp parseOp(const nb::dict &op_obj) {
  std::string kind;
  if (hasKey(op_obj, "op")) {
    kind = toLower(toString(op_obj["op"]));
  } else if (hasKey(op_obj, "kind")) {
    const std::string fallback_kind = toLower(toString(op_obj["kind"]));
    if (isOpDiscriminator(fallback_kind)) {
      kind = fallback_kind;
    }
  } else {
    throw std::runtime_error("Kernel op is missing 'op'/'kind' discriminator.");
  }
  if (kind.empty()) {
    throw std::runtime_error("Kernel op is missing 'op' discriminator.");
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

  if (kind == "transpose") {
    matcore::TransposeOp op;
    if (hasKey(op_obj, "result")) {
      op.result = toString(op_obj["result"]);
    } else {
      op.result = toString(requireKey(op_obj, "output"));
    }
    if (hasKey(op_obj, "input")) {
      op.input = toString(op_obj["input"]);
    } else {
      op.input = toString(requireKey(op_obj, "source"));
    }
    return op;
  }

  if (kind == "elementwise") {
    matcore::ElementwiseOp op;
    if (hasKey(op_obj, "result")) {
      op.result = toString(op_obj["result"]);
    } else {
      op.result = toString(requireKey(op_obj, "output"));
    }
    op.kind = parseElementwiseKind(op_obj);
    op.lhs = hasKey(op_obj, "lhs") ? toString(op_obj["lhs"])
                                   : toString(requireKey(op_obj, "input"));
    std::optional<std::string> rhs = parseOptionalString(op_obj, "rhs");
    if (!rhs.has_value()) {
      rhs = parseOptionalString(op_obj, "input_rhs");
    }
    if (isUnaryElementwise(op.kind)) {
      if (rhs.has_value() && !rhs->empty()) {
        throw std::runtime_error(
            "Unary elementwise op must not include a non-empty 'rhs'.");
      }
      op.rhs.clear();
    } else {
      if (!rhs.has_value() || rhs->empty()) {
        throw std::runtime_error(
            "Binary elementwise op must include non-empty 'rhs' (or 'input_rhs').");
      }
      op.rhs = *rhs;
    }
    return op;
  }

  if (kind == "cast") {
    matcore::CastOp op;
    if (hasKey(op_obj, "result")) {
      op.result = toString(op_obj["result"]);
    } else {
      op.result = toString(requireKey(op_obj, "output"));
    }
    if (hasKey(op_obj, "input")) {
      op.input = toString(op_obj["input"]);
    } else {
      op.input = toString(requireKey(op_obj, "source"));
    }
    const std::string dtype_name = normalizeDTypeName(
        hasKey(op_obj, "target_dtype") ? toString(op_obj["target_dtype"])
                                        : toString(requireKey(op_obj, "dtype")));
    op.target_dtype = parseRuntimeTensorDType(dtype_name);
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

  kernel.global_quantization = parseKernelGlobalQuantization(kernel_obj);

  return kernel;
}

Invocation buildInvocation(const nb::dict &kernel_obj, const std::string &target,
                           const nb::args &tensor_args) {
  Invocation invocation;
  invocation.kernel = parseKernelIR(kernel_obj);
  invocation.target_profile = matcore::ParseRequestedTargetProfile(target);
  const auto kernel_declared_dtypes = parseKernelDeclaredDtypes(kernel_obj);
  const auto kernel_declared_tensor_quantization =
      parseKernelDeclaredTensorQuantization(kernel_obj);

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

    if (auto it = kernel_declared_tensor_quantization.find(view.symbol);
        it != kernel_declared_tensor_quantization.end()) {
      if (parsed.dtype_name != "int8") {
        parsed.quantization.enabled = false;
        parsed.quantization.scale = 1.0f;
        parsed.quantization.zero_point = 0;
      } else if (parsed.quantization.enabled && it->second.enabled &&
          (parsed.quantization.scale != it->second.scale ||
           parsed.quantization.zero_point != it->second.zero_point)) {
        throw std::runtime_error(
            "Tensor quantization mismatch for symbol '" + view.symbol + "'.");
      } else if (!parsed.quantization.enabled) {
        parsed.quantization = it->second;
      }
    }
    if (!parsed.quantization.enabled && parsed.dtype_name == "int8" &&
        invocation.kernel.global_quantization.enabled) {
      parsed.quantization = invocation.kernel.global_quantization;
    }

    view.data = reinterpret_cast<decltype(view.data)>(parsed.data_ptr);
    view.dtype = parseRuntimeTensorDType(parsed.dtype_name);
    view.c_contiguous = parsed.c_contiguous;
    view.is_device_resident = parsed.is_device_resident;
    view.shape = std::move(parsed.shape);
    view.strides = std::move(parsed.element_strides);
    view.quantization = parsed.quantization;

    invocation.tensor_dtypes.push_back(parsed.dtype_name);
    invocation.tensors.push_back(std::move(view));
  }

  // Reject mixed host/device tensor sets.
  {
    bool has_device = false, has_host = false;
    for (const auto &tv : invocation.tensors) {
      if (tv.is_device_resident) has_device = true;
      else has_host = true;
    }
    if (has_device && has_host) {
      throw std::runtime_error(
          "mc.launch() does not support mixed host/device tensors. "
          "Use mc.to_device() on all tensors or none.");
    }
  }

  appendTensorDtypeMetadata(invocation.kernel, invocation.tensors,
                            invocation.tensor_dtypes);
  return invocation;
}

void triggerCompilation(const Invocation &invocation,
                        matcore::ObservabilityContext *obs = nullptr) {
  if (obs) {
    const std::size_t count =
        std::min(invocation.tensors.size(), invocation.tensor_dtypes.size());
    for (std::size_t i = 0; i < count; ++i) {
      obs->traceEvent(matcore::TraceEventKind::kTensorViewBind,
                      invocation.tensors[i].symbol, invocation.tensor_dtypes[i]);
    }
  }
  nb::gil_scoped_release release;
  matcore::compileAndRun(invocation.kernel, invocation.target_profile,
                         invocation.tensors, obs);
}

}  // namespace

NB_MODULE(_matcore_native, m) {
  m.doc() = "MatCore nanobind bridge";

  m.def("compile_and_run",
        [](nb::dict kernel_ir, const std::string &target, nb::args tensors,
           nb::kwargs kwargs) {
          Invocation invocation = buildInvocation(kernel_ir, target, tensors);
          std::unique_ptr<matcore::ObservabilityContext> obs;
          const bool has_observability_kwargs =
              kwargs.contains(nb::str("debug")) || kwargs.contains(nb::str("trace")) ||
              kwargs.contains(nb::str("session_id")) ||
              kwargs.contains(nb::str("debug_dir"));
          if (has_observability_kwargs) {
            matcore::ObservabilityOptions opts;
            if (kwargs.contains(nb::str("debug"))) {
              opts.debug_enabled = nb::cast<bool>(kwargs[nb::str("debug")]);
            }
            if (kwargs.contains(nb::str("trace"))) {
              const std::string trace_str =
                  nb::cast<std::string>(kwargs[nb::str("trace")]);
              if (trace_str == "summary") {
                opts.trace_mode = matcore::TraceMode::kSummary;
              } else if (trace_str == "verbose") {
                opts.trace_mode = matcore::TraceMode::kVerbose;
              } else if (trace_str == "json") {
                opts.trace_mode = matcore::TraceMode::kJson;
              } else if (trace_str == "chrome") {
                opts.trace_mode = matcore::TraceMode::kChrome;
              } else {
                opts.trace_mode = matcore::TraceMode::kNone;
              }
            }
            if (kwargs.contains(nb::str("session_id"))) {
              opts.session_id =
                  nb::cast<std::string>(kwargs[nb::str("session_id")]);
            }
            if (kwargs.contains(nb::str("debug_dir"))) {
              opts.output_dir = nb::cast<std::string>(kwargs[nb::str("debug_dir")]);
            }
            opts.force_recompile = opts.debug_enabled;
            obs = std::make_unique<matcore::ObservabilityContext>(opts);
          }

          triggerCompilation(invocation, obs.get());
        },
        "Compile and run a kernel IR with zero-copy tensor views.");

  // DeviceBufferHandle type exposed to Python as opaque handle.
  nb::class_<matcore::DeviceBufferHandle>(m, "DeviceBufferHandle")
      .def_ro("ptr", &matcore::DeviceBufferHandle::ptr)
      .def_ro("size_bytes", &matcore::DeviceBufferHandle::size_bytes)
      .def_ro("alloc_id", &matcore::DeviceBufferHandle::alloc_id);

  m.def("matcore_device_alloc",
        [](std::uint64_t size_bytes) {
          nb::gil_scoped_release release;
          return matcore::matcore_device_alloc(size_bytes);
        },
        "Allocate GPU device memory via the memory pool.");

  m.def("matcore_device_free",
        [](matcore::DeviceBufferHandle handle) {
          nb::gil_scoped_release release;
          matcore::matcore_device_free(handle);
        },
        "Release GPU device memory back to the pool.");

  m.def("matcore_device_upload",
        [](matcore::DeviceBufferHandle dst, std::uintptr_t host_src,
           std::uint64_t size_bytes) {
          nb::gil_scoped_release release;
          matcore::matcore_device_upload(
              dst, reinterpret_cast<const void *>(host_src), size_bytes);
        },
        "Copy data from host to device buffer.");

  m.def("matcore_device_download",
        [](std::uintptr_t host_dst, matcore::DeviceBufferHandle src,
           std::uint64_t size_bytes) {
          nb::gil_scoped_release release;
          matcore::matcore_device_download(
              reinterpret_cast<void *>(host_dst), src, size_bytes);
        },
        "Copy data from device buffer to host.");

  m.def("matcore_device_zero",
        [](matcore::DeviceBufferHandle handle) {
          nb::gil_scoped_release release;
          matcore::matcore_device_zero(handle);
        },
        "Zero-fill a device buffer.");

  // V2: MatcorePlan — pre-compiled execution plan for near-zero dispatch
  nb::class_<matcore::MatcorePlan>(m, "MatcorePlan")
      .def_prop_ro("generation_id", &matcore::MatcorePlan::generationId)
      .def_prop_ro("has_device_tensors", &matcore::MatcorePlan::hasDeviceTensors)
      .def_prop_ro("num_tensors", &matcore::MatcorePlan::numTensors);

  m.def("create_plan",
        [](nb::dict kernel_ir, const std::string &target, nb::args tensor_args,
           nb::kwargs kwargs)
            -> matcore::MatcorePlan * {
          bool graph_mode = false;
          if (kwargs.contains("graph_mode")) {
            graph_mode = nb::cast<bool>(kwargs["graph_mode"]);
          }
          // Reuse existing buildInvocation for parsing
          Invocation invocation = buildInvocation(kernel_ir, target, tensor_args);
          nb::gil_scoped_release release;
          auto plan = matcore::MatcorePlan::create(
              invocation.kernel, invocation.tensors, target, nullptr,
              graph_mode);
          return plan.release();  // Transfer ownership to nanobind
        },
        nb::rv_policy::take_ownership,
        "Create a pre-compiled execution plan (expensive, call once).");

  m.def("execute_plan",
        [](matcore::MatcorePlan &plan, nb::args tensor_args) {
          // Parse tensors using the same path as compile_and_run
          std::vector<matcore::RuntimeTensorView> tensors;
          std::vector<nb::object> keepalive;
          tensors.reserve(tensor_args.size());
          keepalive.reserve(tensor_args.size());
          for (std::size_t i = 0; i < tensor_args.size(); ++i) {
            nb::object tensor = nb::borrow<nb::object>(tensor_args[i]);
            ParsedTensor parsed = parseTensorArgument(tensor, i);
            keepalive.push_back(tensor);
            matcore::RuntimeTensorView view;
            // Use frozen meta symbols if available
            if (i < plan.numTensors()) {
              view.symbol = plan.frozenMeta()[i].symbol;
            } else {
              view.symbol = "arg" + std::to_string(i);
            }
            view.data = reinterpret_cast<void *>(parsed.data_ptr);
            view.dtype = parseRuntimeTensorDType(parsed.dtype_name);
            view.c_contiguous = parsed.c_contiguous;
            view.is_device_resident = parsed.is_device_resident;
            view.shape = std::move(parsed.shape);
            view.strides = std::move(parsed.element_strides);
            view.quantization = parsed.quantization;
            tensors.push_back(std::move(view));
          }
          nb::gil_scoped_release release;
          plan.execute(tensors);
        },
        "Execute a pre-compiled plan with near-zero overhead.");
}
