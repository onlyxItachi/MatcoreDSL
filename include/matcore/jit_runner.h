#pragma once

#include <vector>

#include "matcore/kernel_ir.h"

namespace matcore {

void compileAndRun(const KernelIR &kernel, TargetKind target,
                   const std::vector<RuntimeTensorView> &tensors);

}  // namespace matcore
