#ifndef MATCORE_MDSLC_RUNTIME_CPU_RUNTIME_VALIDATION_H
#define MATCORE_MDSLC_RUNTIME_CPU_RUNTIME_VALIDATION_H

#include "cpu_execution_context.h"
#include "cpu_planner_v3_resources.h"

namespace matcore::mdslc::runtime {

/*
 * Execute one deterministic numerical self-test for every stable F32 CPU
 * implementation. Serial/provider tests run on worker zero, so evidence for
 * a bound context authenticates the same placement used by later execution.
 * Parallel evidence requires at least two persistent workers. A false field
 * means unavailable or failed; no ISA/provider support is inferred.
 */
CpuRuntimeValidationEvidenceV1 validate_cpu_runtime_variants_v1(
    CpuExecutionContextV1 &context) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
