#pragma once

#include <vector>

#include "matcore/kernel_ir.h"

namespace matcore {

void ValidateRegionIR(const KernelIR &kernel,
                      const std::vector<RuntimeTensorView> &tensors = {});

}  // namespace matcore
