#include "../../lib/platform/closed_cpu_isa_v1.h"
#include <cerrno>
#include <iostream>
namespace p = matcore::mdslc::platform;
int main() {
  unsigned checks=0, failures=0;
  auto check=[&](bool okay) { ++checks; failures+=!okay; };
  p::ClosedCpuIsaFactsV1 good;
  good.native_linux_x86=good.leaf1_known=good.leaf7_known=good.xcr0_known=true;
  good.leaf1_ecx=p::kClosedAvxEcxV1|p::kClosedAvx512EcxV1;
  good.leaf1_edx=p::kClosedX86EdxV1; good.leaf7_ebx=(1U<<5)|(1U<<16); good.xcr0=0xe6;
  for(auto isa:{p::StrictCpuIsaV1::AVX,p::StrictCpuIsaV1::AVX2,p::StrictCpuIsaV1::AVX512F}) {
    check(p::closedCpuIsaSupportedV1(isa,good));
    for(unsigned bit=0;bit<32;++bit) {
      if(good.leaf1_ecx&(1U<<bit)) {
        auto bad=good; bad.leaf1_ecx&=~(1U<<bit);
        const bool required=(p::kClosedAvxEcxV1&(1U<<bit)) || isa==p::StrictCpuIsaV1::AVX512F;
        check(p::closedCpuIsaSupportedV1(isa,bad)!=required);
      }
      if(good.leaf1_edx&(1U<<bit)) {
        auto bad=good; bad.leaf1_edx&=~(1U<<bit); check(!p::closedCpuIsaSupportedV1(isa,bad));
      }
    }
    for(unsigned mutation=0;mutation<9;++mutation) {
      auto bad=good;
      switch(mutation) {
      case 0: bad.version=0; break; case 1: bad.version=2; break;
      case 2: bad.native_linux_x86=false; break; case 3: bad.leaf1_known=false; break;
      case 4: bad.xcr0_known=false; break; case 5: bad.xcr0=2; break;
      case 6: bad.xcr0=4; break; case 7: bad.xcr0=0; break;
      case 8: bad.leaf1_ecx=0; break;
      }
      check(!p::closedCpuIsaSupportedV1(isa,bad));
    }
    auto partial=good; partial.leaf7_known=false;
    check(p::closedCpuIsaSupportedV1(isa,partial)==(isa==p::StrictCpuIsaV1::AVX));
    partial=good; partial.leaf7_ebx&=~(1U<<5);
    check(p::closedCpuIsaSupportedV1(isa,partial)==(isa==p::StrictCpuIsaV1::AVX));
    for(auto mask:{0x6U,0x26U,0x46U,0xa6U,0xc6U}) {
      partial=good; partial.xcr0=mask;
      check(p::closedCpuIsaSupportedV1(isa,partial)==(isa!=p::StrictCpuIsaV1::AVX512F));
    }
  }
  check(!p::closedCpuIsaSupportedV1(p::StrictCpuIsaV1::Baseline,good));
  check(!p::closedCpuIsaSupportedV1(static_cast<p::StrictCpuIsaV1>(77),good));
  errno=EDOM; const auto native=p::discoverClosedCpuIsaFactsV1(); check(errno==EDOM);
  for(auto isa:{p::StrictCpuIsaV1::AVX,p::StrictCpuIsaV1::AVX2,p::StrictCpuIsaV1::AVX512F})
    std::cout << p::strictCpuIsaNameV1(isa) << " hardware_os=" << p::closedCpuIsaSupportedV1(isa,native) << '\n';
  std::cout << "Strict ISA facts: " << checks << " checks; " << failures << " failures\n";
  return failures!=0;
}
