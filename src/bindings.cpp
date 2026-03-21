#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "matcore/jit_runner.h"
#include "matcore/kernel_ir.h"

namespace nb = nanobind;

namespace {

using CpuTensor = nb::ndarray<float, nb::c_contig, nb::device::cpu>;

struct Invocation {
  matcore::KernelIR kernel;
  matcore::TargetKind target;
  std::vector<matcore::RuntimeTensorView> tensors;
  std::vector<CpuTensor> keepalive_tensors;
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
  if (normalized == "arm") {
    return matcore::TargetKind::kARM;
  }
  if (normalized == "nvptx") {
    return matcore::TargetKind::kNVPTX;
  }
  if (normalized == "amdgcn") {
    return matcore::TargetKind::kAMDGCN;
  }
  if (normalized == "npu") {
    return matcore::TargetKind::kNPU;
  }
  if (normalized == "tpu") {
    return matcore::TargetKind::kTPU;
  }

  throw std::runtime_error(
      "Unsupported target '" + target +
      "'. Supported targets: x86-auto, x86-avx2, x86-avx512, arm, nvptx, amdgcn, npu, tpu.");
}

Invocation buildInvocation(const nb::dict &kernel_obj, const std::string &target,
                           const nb::args &tensor_args) {
  Invocation invocation;
  invocation.kernel = parseKernelIR(kernel_obj);
  invocation.target = parseTarget(target);

  invocation.keepalive_tensors.reserve(tensor_args.size());
  invocation.tensors.reserve(tensor_args.size());

  for (std::size_t i = 0; i < tensor_args.size(); ++i) {
    CpuTensor tensor = nb::cast<CpuTensor>(tensor_args[i]);
    invocation.keepalive_tensors.push_back(tensor);

    matcore::RuntimeTensorView view;
    if (i < invocation.kernel.params.size()) {
      view.symbol = invocation.kernel.params[i];
    } else {
      view.symbol = "arg" + std::to_string(i);
    }

    view.data = const_cast<float *>(tensor.data());
    view.c_contiguous = true;

    const std::size_t dims = tensor.ndim();
    view.shape.reserve(dims);
    view.strides.reserve(dims);
    for (std::size_t dim = 0; dim < dims; ++dim) {
      view.shape.push_back(static_cast<std::int64_t>(tensor.shape(dim)));
      view.strides.push_back(static_cast<std::int64_t>(tensor.stride(dim)));
    }

    invocation.tensors.push_back(std::move(view));
  }

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
