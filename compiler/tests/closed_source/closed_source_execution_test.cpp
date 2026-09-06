#include "ClosedHostEmitter.h"
#include "frontend.h"
#include "../../lib/support/platform_support.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fe = matcore::mdslc::frontend;
namespace cr = matcore::mdslc::closed_region;
namespace cg = matcore::mdslc::codegen;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
static_assert(!std::is_default_constructible_v<fe::AuthenticatedClosedRegionEvidence>);
static_assert(!std::is_constructible_v<fe::AuthenticatedClosedRegionEvidence, cr::Program>);

namespace {
unsigned checks=0, failures=0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
void write(const fs::path &path, const std::string &bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (!stream) throw std::runtime_error("could not write owned fixture: " + path.string());
}
std::string read(const fs::path &path) {
  std::ifstream stream(path,std::ios::binary);
  if(!stream) throw std::runtime_error("could not read generated artifact: "+path.string());
  return {std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
}
std::string region(const std::string &body, const std::string &name="region",
                   const std::string &helpers="") {
  return "#include <cstdint>\nusing namespace mdsl_probe;\n" + helpers +
    "\n[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\nvoid " + name +
    "(Storage A,Shape m,Storage B,Shape k,Storage C,Shape n,"
    "Storage D,Shape p,Storage E,Storage F) {\n" + body + "\n}\n";
}
struct Fixture {
  support::TempDirectoryV1 temporary;
  fe::Options options;
  Fixture(const std::string &clang, const std::string &resource,
          const std::string &source) {
    std::string error;
    auto directory=support::create_temp_directory_v1("mdslc-source-execution-test",error);
    if (!directory) throw std::runtime_error(error);
    temporary=std::move(*directory);
    options.input_path=(path()/"math.mdsl").string();
    options.clang_path=clang;
    options.clang_resource_directory=resource;
    options.compiler_arguments={"-std=c++20", "-I"+path().string()};
    write(options.input_path,source);
  }
  const fs::path &path() const { return temporary.path(); }
  fe::ClosedRegionAdmissionResult admit(const std::string &name="region") const {
    return fe::admitClosedRegionHost(options,path().string(),name);
  }
};
struct Example { std::string name, source, invocation; };

const std::string childPrelude=R"child(
#include "closed_host_v1.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#if defined(MDSLC_SOURCE_TRUSTED_REGISTRY)
#include "matcore/runtime_c.h"
#endif
namespace h=matcore::mdslc::runtime::closed_host_v1;
static unsigned checks=0, failures=0;
static void expect(bool c,const char *m) {
  ++checks; if(!c) {++failures; std::fprintf(stderr,"FAIL: %s\n",m);}
}
static h::Session makeSession() {
#if defined(MDSLC_SOURCE_TRUSTED_REGISTRY)
  return h::Session(h::Options{h::Candidate::generated_strict});
#else
  return h::Session();
#endif
}
static void expectMathCandidate(const h::Session &session) {
  const auto report=session.candidateReport();
#if defined(MDSLC_SOURCE_TRUSTED_REGISTRY)
  const auto expected=h::Implementation::generated_strict;
#else
  const auto expected=h::Implementation::native_strict;
#endif
  expect(report.actual==expected && report.invocation_attempted && report.value_issued &&
         report.actual_threads==1 && report.code==h::Code::ok,
         "compiled source actually executed the requested mathematical implementation");
}
static h::ResourceView view(float *p,std::uint64_t m,std::uint64_t n,std::uint64_t cap) {
  return {p,m,n,cap,h::Access::read_write};
}
static std::vector<float> oracle(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 unsigned m,unsigned k,unsigned n) {
  std::vector<float> c(m*n);
  for(unsigned i=0;i<m;++i) for(unsigned j=0;j<n;++j) {
    double sum=0;
    for(unsigned q=0;q<k;++q) sum+=static_cast<double>(a[i*k+q])*b[q*n+j];
    c[i*n+j]=static_cast<float>(sum);
  }
  return c;
}
template<std::size_t N>
static std::array<float,N> observed(const h::Session &s,std::uint64_t index) {
  auto value=s.observation(index);
  std::array<float,N> out{};
  h::Session copy;
  expect(value.valid() && static_cast<bool>(copy.publish(1,value,view(out.data(),value.rows(),value.columns(),N))),
         "owning observation can be inspected after generated return");
  return out;
}
)child";

std::string invocation(std::string text,const std::string &symbol) {
  const std::string marker="@ENTRY@";
  std::size_t at=0;
  while((at=text.find(marker,at))!=std::string::npos) {
    const auto name="matcore::mdslc::generated_closed_host_v1::"+symbol;
    text.replace(at,marker.size(),name); at+=name.size();
  }
  return text;
}
void verifyFrontiers(const cg::ClosedHostEmission &out,const std::string &source) {
  std::uint64_t previous=0;
  for(const auto &site:out.frontiers) {
    check(site.id>previous,"unique ordered source frontier"); previous=site.id;
    check(site.source.length && site.source.offset<=source.size() &&
          site.source.length<=source.size()-site.source.offset,
          "frontier retains original source span");
    for(const auto &helper:site.helper_calls)
      check(helper.length && helper.offset<=source.size() &&
            helper.length<=source.size()-helper.offset,"helper call source retained");
  }
  check(out.completion_frontier>previous,"completion frontier follows all branch IDs");
  check(out.implementation.find("session.gemm(")!=std::string::npos ||
        out.implementation.find("session.read(")!=std::string::npos,
        "actual static adapter orchestration emitted");
  check(out.implementation.find("MatcoreClosedRegion")==std::string::npos &&
        out.implementation.find("mlir::")==std::string::npos &&
        out.implementation.find("parse(")==std::string::npos,
        "runtime source contains no semantic IR or AST interpreter");
}
} // namespace

int main(int argc,char **argv) {
  if(argc<4) return 2;
  const std::string clang=argv[1], resource=argv[2];
  const fs::path compiler=fs::weakly_canonical(argv[3]);
  try {
    std::string registry,runtimeLibrary,asanControl,kernel,asanKernel;
    std::vector<std::string> childFlags;
    for(int i=4;i<argc;++i) {
      const std::string option=argv[i];
      auto take=[&]() -> std::string {
        if(++i==argc) throw std::runtime_error("missing value after "+option);
        return argv[i];
      };
      if(option=="--registry") registry=take();
      else if(option=="--runtime") runtimeLibrary=take();
      else if(option=="--asan-control") asanControl=take();
      else if(option=="--kernel") kernel=take();
      else if(option=="--asan-kernel") asanKernel=take();
      else childFlags.push_back(option);
    }
    const bool generated=!registry.empty();
    if(generated && (runtimeLibrary.empty() || asanControl.empty() || kernel.empty() || asanKernel.empty()))
      throw std::runtime_error("generated source test requires registry, runtime and kernel controls");
    const std::string lhs=
      "auto a=read(A,m,k); auto b=read(B,k,n); auto c=gemm(a,b,Numerics::strict_f32); "
      "publish(c,C); observe(C); auto d=read(D,n,p); "
      "auto e=gemm(c,d,Numerics::strict_f32); publish(e,E); observe(E);";
    const std::string rhs=
      "auto a=read(A,m,k); auto b=read(B,k,n); auto c=gemm(a,b,Numerics::strict_f32); "
      "publish(c,C); observe(C); auto d=read(D,p,m); "
      "auto e=gemm(d,c,Numerics::strict_f32); publish(e,E); observe(E);";
    const std::string simple=
      "auto a=read(A,m,k); auto b=read(B,k,n); "
      "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C);";
    const std::vector<Example> examples={
      {"lhs_rectangular",region(lhs),R"child({
        std::vector<float>a{1,2,3,4,5,6},b{1,2,3,4,5,6},d{2,0,1,3,4,1};
        std::vector<float>c(4,-1),e(6,-2); auto s=makeSession();
        auto status=@ENTRY@(s,view(a.data(),2,3,6),2,view(b.data(),3,2,6),3,
          view(c.data(),2,2,4),2,view(d.data(),2,3,6),3,view(e.data(),2,3,6),{});
        auto cexpected=oracle(a,b,2,3,2),eexpected=oracle(cexpected,d,2,2,3);
        expect(status && status.completed && status.publications==2 && status.observations==2,
               "lhs source completes exact publication/observation trace");
        expect(c==cexpected && e==eexpected,"lhs rectangular source matches independent oracle");
        expect(observed<4>(s,0)==std::array<float,4>{22,28,49,64},"first observation stores C");
        expect(s.observationFrontier(0)<s.observationFrontier(1),"observation order survives execution");
        expectMathCandidate(s);
      })child"},
      {"rhs_rectangular",region(rhs),R"child({
        std::vector<float>a{1,2,3,4,5,6},b{1,2,3,4,5,6},d{2,0,1,3,4,1};
        std::vector<float>c(4,-1),e(6,-2); auto s=makeSession();
        auto status=@ENTRY@(s,view(a.data(),2,3,6),2,view(b.data(),3,2,6),3,
          view(c.data(),2,2,4),2,view(d.data(),3,2,6),3,view(e.data(),3,2,6),{});
        expect(status && status.completed,"rhs source executes");
        expect(c==oracle(a,b,2,3,2) && e==oracle(d,c,3,2,2),
               "rhs rectangular source never commutes multiplication");
        expect(observed<6>(s,1)==std::array<float,6>{44,56,169,220,137,176},"rhs observation is exact");
        expectMathCandidate(s);
      })child"},
      {"alias_old_late",region(
        "auto old=read(C,m,n); auto a=read(A,m,k); auto b=read(B,k,n); "
        "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C); "
        "auto late=read(D,m,n); auto e=gemm(old,late,Numerics::strict_f32); "
        "publish(e,E); observe(C); publish(old,F);"),R"child({
        float a[4]{1,0,0,1},b[4]{5,6,7,8},storage[8]{1,2,3,4,5,6,7,8},saved[4]{};
        auto s=makeSession();
        auto status=@ENTRY@(s,view(a,2,2,4),2,view(b,2,2,4),2,
          view(storage,2,2,8),2,view(storage,2,2,8),0,view(storage+1,2,2,7),view(saved,2,2,4));
        expect(status && status.completed,"aliased source executes with owned snapshots");
        expect(std::array<float,4>{saved[0],saved[1],saved[2],saved[3]}==std::array<float,4>{1,2,3,4},
               "old logical contents survive both publications");
        expect(std::array<float,5>{storage[0],storage[1],storage[2],storage[3],storage[4]}==
               std::array<float,5>{5,19,22,43,50},"late read sees first publication and shifted output order");
        expect(observed<4>(s,0)==std::array<float,4>{5,6,7,8},"earlier observation immutable");
        expect(observed<4>(s,1)==std::array<float,4>{5,19,22,43},"later observation sees overlapping write");
        expect(storage[5]==6 && storage[7]==8,"out-of-footprint sentinels survive");
        expectMathCandidate(s);
      })child"},
      {"late_shape_failure",region(lhs),R"child({
        float a[4]{1,2,3,4},b[4]{2,0,1,3},c[4]{-1,-1,-1,-1},d[6]{},e[4]{-2,-2,-2,-2};
        auto s=makeSession();
        auto status=@ENTRY@(s,view(a,2,2,4),2,view(b,2,2,4),2,
          view(c,2,2,4),2,view(d,3,2,6),2,view(e,2,2,4),{});
        expect(status.code==h::Code::shape_mismatch && status.publications==1 && status.observations==1,
               "late read shape mismatch retires after first publication and observation");
        expect(std::array<float,4>{c[0],c[1],c[2],c[3]}==std::array<float,4>{4,6,10,12},
               "second-operation failure cannot suppress or roll back first output");
        expect(e[0]==-2 && e[3]==-2,"later output untouched after failure");
        expectMathCandidate(s);
      })child"},
      {"dead_result_guard",region(
        "auto old=read(C,m,n); publish(old,E); auto a=read(A,m,k); auto b=read(B,p,n); "
        "auto dead=gemm(a,b,Numerics::strict_f32); observe(C);"),R"child({
        float a[4]{1,2,3,4},b[6]{},c[4]{5,6,7,8},e[4]{}; auto s=makeSession();
        auto status=@ENTRY@(s,view(a,2,2,4),2,view(b,3,2,6),2,
          view(c,2,2,4),2,{},3,view(e,2,2,4),{});
        expect(status.code==h::Code::shape_mismatch && status.publications==1 && status.observations==0,
               "dead GEMM keeps dynamic check and prevents later observation");
        expect(e[0]==5 && e[3]==8,"dead-check failure preserves earlier publication");
      })child"},
      {"unsigned_branch",region(
        "if (p > m) { auto a=read(A,m,k); auto b=read(B,k,n); "
        "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C); } "
        "else { auto d=read(D,m,n); publish(d,E); }"),R"child({
        float a[4]{1,2,3,4},b[4]{2,0,1,3},c[4]{},e[4]{-1,-1,-1,-1}; auto s=makeSession();
        auto invalid=view(nullptr,UINT64_MAX,UINT64_MAX,0);
        auto status=@ENTRY@(s,view(a,2,2,4),2,view(b,2,2,4),2,
          view(c,2,2,4),2,invalid,UINT64_MAX,view(e,2,2,4),{});
        expect(status && status.completed && status.publications==1,
               "unsigned high-bit shape compare takes correct branch without narrowing");
        expect(c[0]==4 && c[3]==12 && e[0]==-1,"untaken invalid resource never prevalidated/read");
        expectMathCandidate(s);
        auto other=makeSession();
        status=@ENTRY@(other,view(a,2,2,4),2,view(b,2,2,4),2,
          view(c,2,2,4),2,view(e,2,2,4),0,view(c,2,2,4),{});
        expect(status && status.completed && c[0]==-1 && c[3]==-1,
               "opposite branch executes its own frontier IDs despite static gaps");
      })child"},
      {"zero_and_reuse",region(simple),R"child({
        float c[6]{9,9,9,9,9,9}; auto s=makeSession();
        auto status=@ENTRY@(s,view(nullptr,2,0,0),2,view(nullptr,0,3,0),0,
          view(c,2,3,6),3,{},0,{},{});
        expect(status && status.completed && c[0]==0 && c[5]==0,"zero-K source initializes nonempty result");
        expect(s.candidateReport().actual==h::Implementation::zero_reduction &&
               !s.candidateReport().invocation_attempted,
               "zero reduction reports local semantic realization, not generated invocation");
        c[0]=77;
        status=@ENTRY@(s,view(nullptr,2,0,0),2,view(nullptr,0,3,0),0,
          view(c,2,3,6),3,{},0,{},{});
        expect(!status && c[0]==77,"completed Session reuse rejected before effects");
        auto failed=makeSession(); h::Value unused;
        auto first=failed.read(1,view(nullptr,1,1,1),unused);
        status=@ENTRY@(failed,view(nullptr,2,0,0),2,view(nullptr,0,3,0),0,
          view(c,2,3,6),3,{},0,{},{});
        expect(status.code==first.code && status.failed_frontier==first.failed_frontier && c[0]==77,
               "failed Session preserves first error and cannot publish");
        auto partial=makeSession(); float input[1]{1};
        partial.read(1,view(input,1,1,1),unused);
        status=@ENTRY@(partial,view(nullptr,2,0,0),2,view(nullptr,0,3,0),0,
          view(c,2,3,6),3,{},0,{},{});
        expect(!status && c[0]==77,"partially used Session rejected before effects");
        auto empty=makeSession();
        status=@ENTRY@(empty,view(nullptr,0,0,0),0,view(nullptr,0,3,0),0,
          view(nullptr,0,3,0),3,{},0,{},{});
        expect(status && status.completed && status.publications==1 && status.observations==1,
               "zero-footprint publication and observation never dereference null");
      })child"},
      {"reusable_helper",region(
        "auto a=read(A,m,k); auto b=read(B,k,n); auto c=product(a,b); "
        "Shape width=cols(c); auto d=read(D,width,p); auto e=product(c,d); publish(e,E); observe(E);",
        "region", "template<class T> T product(T a,T b) { auto c=gemm(a,b,Numerics::reassociate_f32); return c; }"),R"child({
        float a[4]{1,2,3,4},b[4]{2,0,1,3},d[2]{1,2},e[2]{}; auto s=makeSession();
        auto status=@ENTRY@(s,view(a,2,2,4),2,view(b,2,2,4),2,{},2,
          view(d,2,1,2),1,view(e,2,1,2),{});
        expect(status && status.completed && e[0]==16 && e[1]==34,
               "admitted template helper and value-shape query execute without runtime interpretation");
        expectMathCandidate(s);
      })child"},
      {"strict_numeric_environment",region(simple),R"child({
        float a[2]{-1.0f,0x1.000002p0f},b[2]{1.0f,0x1.fffffep-1f},c[1]{77};
        fenv_t saved{};
        expect(fegetenv(&saved)==0 && fesetround(FE_DOWNWARD)==0 &&
               feclearexcept(FE_ALL_EXCEPT)==0 && feraiseexcept(FE_INVALID)==0,
               "construct nondefault caller FP state for generated entry");
        const auto flags=fetestexcept(FE_ALL_EXCEPT);
        auto s=makeSession();
        auto status=@ENTRY@(s,view(a,1,2,2),1,view(b,2,1,2),2,
          view(c,1,1,1),1,{},0,{},{});
        expect(status && std::bit_cast<std::uint32_t>(c[0])==0,
               "strict source preserves separate multiply/add FMA discriminator");
        expect(fegetround()==FE_DOWNWARD && fetestexcept(FE_ALL_EXCEPT)==flags,
               "generated orchestration restores complete caller FP state");
        expect(fesetenv(&saved)==0,"restore numerical fixture environment");
        expectMathCandidate(s);
      })child"}
    };

    std::string implementations=childPrelude, mainBody="int main() {\n";
    for(const auto &example:examples) {
      Fixture fixture(clang,resource,example.source);
      auto admitted=fixture.admit();
      check(admitted && admitted.syntax_valid,example.name+" real-host admission: "+admitted.error);
      if(!admitted) continue;
      auto emitted=cg::emitClosedHostV1(*admitted.evidence);
      check(static_cast<bool>(emitted),example.name+" emission: "+emitted.error);
      if(!emitted) continue;
      auto repeated=cg::emitClosedHostV1(*admitted.evidence);
      check(repeated && repeated.emission->implementation==emitted.emission->implementation,
            example.name+" deterministic immutable-evidence emission");
      verifyFrontiers(*emitted.emission,example.source);
      if(example.name=="reusable_helper")
        check(std::any_of(emitted.emission->frontiers.begin(),emitted.emission->frontiers.end(),
                         [](const auto &f){return !f.helper_calls.empty();}),
              "expanded helper retains source call provenance");
      implementations+=emitted.emission->implementation;
      mainBody+=invocation(example.invocation,emitted.emission->symbol)+"\n";
    }

    // Two independently selected regions in one immutable host context must not
    // collide even though both source and host-context hashes are identical.
    const std::string two=region(simple,"alpha")+region(
      "auto a=read(A,m,k); auto b=read(B,k,n); auto c=gemm(a,b,Numerics::strict_f32); publish(c,E);",
      "beta");
    Fixture multiple(clang,resource,two);
    auto alpha=multiple.admit("alpha"), beta=multiple.admit("beta");
    check(alpha && beta,"two separately selected regions admit from same physical TU");
    if(alpha && beta) {
      check(alpha.evidence->program().source_sha256==beta.evidence->program().source_sha256 &&
            alpha.evidence->hostContextIdentity()==beta.evidence->hostContextIdentity(),
            "collision falsifier actually has identical source/context identities");
      auto left=cg::emitClosedHostV1(*alpha.evidence),right=cg::emitClosedHostV1(*beta.evidence);
      check(left && right && left.emission->symbol!=right.emission->symbol,
            "different selected regions have distinct generated entry symbols");
      if(left && right && left.emission->symbol!=right.emission->symbol) {
        implementations+=left.emission->implementation+right.emission->implementation;
        mainBody+=invocation(R"child({
          float a[1]{2},b[1]{3},c[1]{},e[1]{}; auto s=makeSession();
          auto status=@ENTRY@(s,view(a,1,1,1),1,view(b,1,1,1),1,view(c,1,1,1),1,{},0,view(e,1,1,1),{});
          expect(status && c[0]==6 && e[0]==0,"same-TU alpha links and executes only its publication");
          expectMathCandidate(s);
        })child",left.emission->symbol);
        mainBody+=invocation(R"child({
          float a[1]{2},b[1]{3},c[1]{},e[1]{}; auto s=makeSession();
          auto status=@ENTRY@(s,view(a,1,1,1),1,view(b,1,1,1),1,view(c,1,1,1),1,{},0,view(e,1,1,1),{});
          expect(status && c[0]==0 && e[0]==6,"same-TU beta links and executes distinct publication");
          expectMathCandidate(s);
        })child",right.emission->symbol);
      }
    }

    Fixture authority(clang,resource,region(simple));
    auto seal=authority.admit();
    const auto physical=region(simple);
    const auto hermeticSource=physical.substr(physical.find('\n')+1);
    auto hermetic=fe::admitClosedRegionSource(hermeticSource,authority.options.input_path);
    check(seal && hermetic,"positive real-host and hermetic authority controls");
    if(seal && hermetic) {
      check(!cg::emitClosedHostV1(*hermetic.evidence),"hermetic inspection evidence never authorizes execution emitter");
      auto before=cg::emitClosedHostV1(*seal.evidence);
      write(authority.options.input_path,region(simple)+"\n// changed source identity\n");
      auto after=cg::emitClosedHostV1(*seal.evidence);
      check(before && after && before.emission->implementation==after.emission->implementation,
            "historical seal emits its exact old program rather than rereading changed source");
      auto fresh=authority.admit();
      auto newer=fresh ? cg::emitClosedHostV1(*fresh.evidence) : cg::ClosedHostEmissionResult{};
      check(newer && before && newer.emission->symbol!=before.emission->symbol,
            "fresh changed source gets distinct generated identity");
      auto options=authority.options;
      options.compiler_arguments.push_back("-DUNUSED_HOST_CONTEXT=1");
      auto flagged=fe::admitClosedRegionHost(options,authority.path().string());
      auto flaggedEmission=flagged ? cg::emitClosedHostV1(*flagged.evidence) : cg::ClosedHostEmissionResult{};
      check(flaggedEmission && newer && flaggedEmission.emission->host_context_sha256!=newer.emission->host_context_sha256 &&
            flaggedEmission.emission->symbol!=newer.emission->symbol,
            "context-only change cannot reuse a differently authenticated artifact identity");
      mlir::MLIRContext context;
      auto witness=cr::buildModule(seal.evidence->program(),context);
      if(witness) {
        auto movedFrom=*seal.evidence;
        auto movedTo=std::move(movedFrom);
        check(!movedFrom.hasHostContext() && movedFrom.hostContextIdentity().empty(),
              "moved-from evidence has no usable authority identity");
        check(!cg::emitClosedHostV1(movedFrom),
              "moved-from evidence fails closed rather than dereferencing an absent seal");
        check(static_cast<bool>(cg::emitClosedHostV1(movedTo)),
              "moved-to evidence retains its exact authenticated execution authority");
        std::string movedError;
        check(!fe::verifyClosedRegionMatchesEvidence(movedFrom,*witness.module,movedError) &&
              !movedError.empty(),"moved-from source pairing rejects before payload access");
        auto throwsLogicError=[](auto access) {
          try { access(); } catch(const std::logic_error &) { return true; }
          return false;
        };
        check(throwsLogicError([&] { (void)movedFrom.program(); }),
              "moved-from program accessor diagnoses invalid use");
        check(throwsLogicError([&] { (void)movedFrom.sourceSnapshot(); }),
              "moved-from source accessor diagnoses invalid use");
        check(throwsLogicError([&] { (void)movedFrom.regionName(); }),
              "moved-from region accessor diagnoses invalid use");
        (*witness.module)->setAttr("forged.execution.authority",mlir::UnitAttr::get(&context));
        std::string error;
        check(!fe::verifyClosedRegionMatchesEvidence(*seal.evidence,*witness.module,error),
              "modified editable semantic witness does not pair with source authority");
      } else check(false,"authoritative witness builds for tamper falsifier");
      const char *prior=std::getenv("CPATH");
      const std::optional<std::string> saved=prior?std::optional<std::string>(prior):std::nullopt;
      setenv("CPATH",authority.path().c_str(),1);
      auto poisoned=cg::emitClosedHostV1(*seal.evidence);
      if(saved) setenv("CPATH",saved->c_str(),1); else unsetenv("CPATH");
      check(!poisoned,"host environment poisoning rejects evidence replay before emission");
    }
    for(const auto &bad : {"unknown();", "throw 1;", "volatile int x=0; (void)x;"}) {
      Fixture rejected(clang,resource,region(bad,"region","void unknown();"));
      auto result=rejected.admit();
      check(!result,"unclosed host semantics cannot reach executable emitter");
    }

    Fixture providerFailure(clang,resource,region(
      "auto old=read(C,m,n); publish(old,E); observe(E); "
      "auto a=read(A,m,k); auto b=read(B,k,n); "
      "auto c=gemm(a,b,Numerics::strict_f32); publish(c,F);"));
    auto providerSeal=providerFailure.admit();
    auto providerEmission=providerSeal ? cg::emitClosedHostV1(*providerSeal.evidence) : cg::ClosedHostEmissionResult{};
    check(static_cast<bool>(providerEmission),"source with late forced-provider numerical failure emits");
    if(providerEmission) {
      implementations+=providerEmission.emission->implementation;
      mainBody+=invocation(R"child({
        float a[1]{2},b[1]{3},old[1]{17},published[1]{},later[1]{-1};
        h::Session s(h::Options{h::Candidate::authenticated_openblas});
        auto status=@ENTRY@(s,view(a,1,1,1),1,view(b,1,1,1),1,view(old,1,1,1),1,
                            {},0,view(published,1,1,1),view(later,1,1,1));
        expect(status.code==h::Code::candidate_incompatible && status.publications==1 &&
               status.observations==1 && status.failed_frontier==6 &&
               status.completed_frontier==5 && status.completed_effect_frontier==3 &&
               published[0]==17 && later[0]==-1,
               "forced provider cannot execute strict profile or erase earlier source effects");
        expect(!s.candidateReport().invocation_attempted && !s.candidateReport().provider_contract_checked &&
               !s.candidateReport().value_issued,"incompatible provider rejected before probing or computing");
        expect(observed<1>(s,0)[0]==17,"observation survives forced-provider failure");
      })child",providerEmission.emission->symbol);
    }
    if(generated) {
      mainBody+=R"child({
        float a[1]{2},b[1]{3},c[1]{-1};
        auto descriptor=[](float *p,bool writable) {
          matcore_tensor_desc_v0 d{}; d.struct_size=sizeof(d); d.data=p;
          d.dtype=MATCORE_DTYPE_F32_V0; d.rank=2; d.dims[0]=1; d.dims[1]=1;
          d.strides[0]=1; d.strides[1]=1; d.memory_space=MATCORE_MEMORY_SPACE_HOST_V0;
          d.mutability=writable?MATCORE_MUTABILITY_READ_WRITE_V0:MATCORE_MUTABILITY_READ_ONLY_V0;
          return d;
        };
        auto lhs=descriptor(a,false),rhs=descriptor(b,false),out=descriptor(c,true);
        matcore_policy_v0 policy{}; policy.struct_size=sizeof(policy);
        policy.target=MATCORE_TARGET_CPU_V0; policy.fallback=MATCORE_FALLBACK_ERROR_V0;
        matcore_cpu_gemm_execution_options_v1 options{};
        options.abi_version=MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;options.struct_size=sizeof(options);
        options.requested_threads=1;options.request=MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2;
        matcore_cpu_gemm_plan_report_v2 report{};
        report.abi_version=MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;report.struct_size=sizeof(report);
        const auto status=matcore_runtime_gemm_f32_execute_v1(&out,&lhs,&rhs,&policy,&options,nullptr,0,&report);
        expect(status.code==MATCORE_STATUS_OK_V0 && c[0]==6 && report.selected_stable_id &&
               std::strcmp(report.selected_stable_id,"cpu.reference.f32.v1")==0,
               "ordinary legacy C ABI coexists in the exact generated-source executable");
      })child";
    }

    if(failures) {
      std::cout<<"Closed source seam: "<<checks<<" checks, "<<failures<<" failures before child compilation\n";
      return 1;
    }
    mainBody+="std::printf(\"Generated orchestration: %u checks, %u failures\\n\",checks,failures); return failures?1:0; }\n";
    Fixture output(clang,resource,region(simple));
    const auto cpp=output.path()/"generated.cpp", object=output.path()/"generated.o",
               executable=output.path()/"generated-executable";
    write(cpp,implementations+mainBody);
    const std::string runtime=(compiler/"lib/runtime").string();
    support::ProcessRequestV1 compile;
    compile.working_directory=output.path();
    compile.environment=support::compiler_environment_sanitization_v1();
    compile.argv={clang,"-std=c++20","-O1","-g","-Wall","-Wextra","-Werror",
                  "-frounding-math","-ftrapping-math","-ffp-contract=off","-I",runtime};
    compile.argv.insert(compile.argv.end(),childFlags.begin(),childFlags.end());
    if(generated) compile.argv.insert(compile.argv.end(),{"-DMDSLC_SOURCE_TRUSTED_REGISTRY=1",
                                                        "-I",(compiler/"include").string()});
    const auto base=compile.argv;
    compile.argv.insert(compile.argv.end(),{"-c",cpp.string(),"-o",object.string()});
    auto compiled=support::run_process_v1(compile);
    check(compiled.launched && compiled.exit_code==0,"ordinary generated object compile: "+compiled.error+compiled.stderr_text);
    if(compiled.launched && compiled.exit_code==0) {
      compile.argv=base;
      compile.argv.push_back(object.string());
      if(generated) {
        compile.argv.insert(compile.argv.end(),{registry,runtimeLibrary,"-Wl,-rpath,"+fs::path(runtimeLibrary).parent_path().string()});
      } else compile.argv.push_back((compiler/"lib/runtime/closed_host_v1.cpp").string());
      compile.argv.insert(compile.argv.end(),{"-o",executable.string()});
      auto linked=support::run_process_v1(compile);
      check(linked.launched && linked.exit_code==0,"ordinary final link with production adapter: "+linked.error+linked.stderr_text);
      if(linked.launched && linked.exit_code==0) {
        support::ProcessRequestV1 execute;
        execute.argv={executable.string()}; execute.working_directory=output.path();
        auto ran=support::run_process_v1(execute);
        check(ran.launched && ran.exit_code==0,"actual generated orchestration execution: "+ran.error+ran.stderr_text);
        check(ran.stdout_text.find("0 failures")!=std::string::npos,"child reports actual executed assertions");
        std::cout<<ran.stdout_text;
      }
    }
    if(generated) {
      support::ProcessRequestV1 negative;
      negative.argv={asanControl,"--oob"}; negative.working_directory=output.path();
      // Match the established kernel control: symbolization must stay local and
      // the deliberately failing process must stop at its first memory error.
      negative.environment={{"DEBUGINFOD_URLS",std::string()},
                            {"ASAN_OPTIONS",std::string("halt_on_error=1:detect_leaks=0")}};
      auto rejected=support::run_process_v1(negative);
      check(rejected.launched && rejected.exit_code!=0 &&
            rejected.stderr_text.find("AddressSanitizer: heap-buffer-overflow")!=std::string::npos &&
            rejected.stderr_text.find("__matcore_strict_gemm_f32_v1")!=std::string::npos,
            "negative control detects actual generated-kernel out-of-bounds memory access");
      const bool instrumented=std::any_of(childFlags.begin(),childFlags.end(),[](const std::string &flag) {
        return flag.starts_with("-fsanitize=") && flag.find("address")!=std::string::npos;
      });
      if(instrumented) {
        const auto actual=read(kernel),control=read(asanKernel);
        check(!actual.empty() && actual==control,
              "source-linked LLVM object exactly matches the generated-memory ASan negative control object");
      }
    }
  } catch(const std::exception &error) { check(false,error.what()); }
  std::cout<<"Closed source seam: "<<checks<<" checks, "<<failures<<" failures\n";
  return failures?1:0;
}
