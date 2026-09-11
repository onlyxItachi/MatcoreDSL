#include "closed_cpu_isa_v1.h"
#include "cpu_x86_probe_internal.h"

namespace matcore::mdslc::platform {
ClosedCpuIsaFactsV1 discoverClosedCpuIsaFactsV1() noexcept {
  ClosedCpuIsaFactsV1 facts;
#if defined(__linux__) && defined(__x86_64__)
  using namespace x86_probe_internal;
  facts.native_linux_x86 = true;
  CpuidResult leaf1, leaf7;
  facts.leaf1_known = read_cpuid(1, 0, &leaf1);
  if (!facts.leaf1_known) return facts;
  facts.leaf1_ecx = leaf1.ecx;
  facts.leaf1_edx = leaf1.edx;
  facts.leaf7_known = read_cpuid(7, 0, &leaf7);
  if (facts.leaf7_known) facts.leaf7_ebx = leaf7.ebx;
  constexpr auto xsave_osxsave_avx = (1U << 26) | (1U << 27) | (1U << 28);
  if ((leaf1.ecx & xsave_osxsave_avx) == xsave_osxsave_avx)
    facts.xcr0_known = read_xcr0(&facts.xcr0);
#endif
  return facts;
}
} // namespace matcore::mdslc::platform
