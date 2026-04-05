#pragma once

#include <vector>

#include "matcore/kernel_ir.h"
#include "matcore/target_registry.h"

namespace matcore {

class ObservabilityContext;

void compileAndRun(const KernelIR &kernel,
                   const RequestedTargetProfile &target_profile,
                   const std::vector<RuntimeTensorView> &tensors,
                   ObservabilityContext *obs = nullptr);

}  // namespace matcore
