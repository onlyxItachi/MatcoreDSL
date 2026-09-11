#pragma once
#ifdef MDSLC_TEST_STRICT_ISA
#include "../../lib/platform/closed_cpu_isa_v1.h"
#include <cstdio>
#define _mlir_ciface___matcore_strict_gemm_f32_v1 MDSLC_TEST_STRICT_SYMBOL
inline bool strictIsaTestAvailable() {
  namespace p = matcore::mdslc::platform;
  constexpr auto isa = static_cast<p::StrictCpuIsaV1>(MDSLC_TEST_STRICT_ISA);
  const bool available = p::closedCpuIsaSupportedV1(isa, p::discoverClosedCpuIsaFactsV1());
  if (!available) std::printf("SKIP strict ISA %s: hardware/OS unavailable\n", p::strictCpuIsaNameV1(isa));
  return available;
}
#else
inline bool strictIsaTestAvailable() { return true; }
#endif
