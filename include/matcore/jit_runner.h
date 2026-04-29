#pragma once

#include <string>
#include <vector>

#include "matcore/kernel_ir.h"
#include "matcore/target_registry.h"

namespace matcore {

class ObservabilityContext;

struct CompilationStats {
  int actual_reg_count = 0;
  bool reg_budget_exceeded = false;
  std::string route;
  int fusion_launch_count = 0;
  std::string family_c_strategy;
  int family_c_dtile = 0;
  bool available = false;
};

void compileAndRun(const KernelIR &kernel,
                   const RequestedTargetProfile &target_profile,
                   const std::vector<RuntimeTensorView> &tensors,
                   ObservabilityContext *obs = nullptr);
CompilationStats getLastCompilationStats();

}  // namespace matcore
