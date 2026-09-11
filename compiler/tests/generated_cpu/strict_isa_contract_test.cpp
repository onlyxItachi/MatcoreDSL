#include "MatcoreCpuGemmCandidate.h"
#include <iostream>
#include <string>
namespace c = matcore::mdslc::cpu_candidate;
namespace p = matcore::mdslc::platform;
int main() {
  unsigned checks=0, failures=0;
  auto check=[&](bool okay, const char *what) {
    ++checks; if (!okay) { ++failures; std::cerr << "FAIL " << what << '\n'; }
  };
  mlir::MLIRContext context;
  constexpr auto row=c::StrictGemmScheduleV1::RowContiguousMKN;
  constexpr auto x86=c::CpuTargetV1::LinuxX86_64;
  auto baseline=c::issueStrictGemmArtifactV1(context,false,row);
  check(bool(baseline), "baseline issued");
  for (auto isa : {c::StrictCpuIsaV1::AVX,c::StrictCpuIsaV1::AVX2,c::StrictCpuIsaV1::AVX512F}) {
    for (bool asan : {false,true}) {
      auto issued=c::issueStrictGemmArtifactV1(context,asan,row,x86,isa);
      check(bool(issued), "closed ISA issued");
      check(issued.semantic_ir==baseline.semantic_ir && issued.structured_ir==baseline.structured_ir &&
            issued.bufferized_ir==baseline.bufferized_ir && issued.scheduled_ir==baseline.scheduled_ir &&
            issued.transform_ir==baseline.transform_ir, "same exact authenticated MLIR recipe");
      check(issued.llvm_ir.find(c::strictCpuLeafSymbolV1(isa))!=std::string::npos &&
            issued.llvm_ir.find(std::string("\"target-features\"=\"")+c::strictCpuFeaturesV1(isa)+"\"")!=std::string::npos &&
            issued.llvm_ir.find(std::string("\"prefer-vector-width\"=\"")+c::strictCpuVectorWidthV1(isa)+"\"")!=std::string::npos,
            "distinct symbol and exact closed LLVM realization attributes");
      check(issued.llvm_ir.find("fmul float")!=std::string::npos && issued.llvm_ir.find("fadd float")!=std::string::npos &&
            issued.llvm_ir.find(" fast ")==std::string::npos && issued.llvm_ir.find(" contract ")==std::string::npos &&
            issued.llvm_ir.find("@llvm.fma")==std::string::npos, "no arithmetic fusion permission");
      check(issued.manifest.find("arithmetic_fma_permission=none\n")!=std::string::npos &&
            issued.manifest.find(std::string("isa=")+p::strictCpuIsaNameV1(isa)+"\n")!=std::string::npos,
            "manifest binds precise ISA, not a performance or execution claim");
      check((issued.llvm_ir.find("sanitize_address")!=std::string::npos)==asan, "instrumentation is issued");
    }
    check(!c::issueStrictGemmArtifactV1(context,false,c::StrictGemmScheduleV1::ScalarMNK,x86,isa), "scalar ISA request refused");
    check(!c::issueStrictGemmArtifactV1(context,false,row,c::CpuTargetV1::LinuxAArch64,isa), "ARM ISA request refused");
  }
  check(!c::issueStrictGemmArtifactV1(context,false,row,x86,static_cast<c::StrictCpuIsaV1>(90)), "unknown ISA refused");
  auto explicitBaseline=c::issueStrictGemmArtifactV1(context,false,row,x86,c::StrictCpuIsaV1::Baseline);
  check(explicitBaseline.llvm_ir==baseline.llvm_ir && explicitBaseline.manifest==baseline.manifest, "default byte identity");
  std::cout << "Strict ISA issuer: " << checks << " checks; " << failures << " failures\n";
  return failures!=0;
}
