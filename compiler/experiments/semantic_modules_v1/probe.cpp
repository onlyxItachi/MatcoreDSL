// Research-only consumer of the existing, noninstalled admission test seam.
// Generates ordinary C++ fixtures; does not parse or interpret C++ itself.
#include "ClosedRegionAdmissionInternal.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace fe = matcore::mdslc::frontend;
namespace cr = matcore::mdslc::closed_region;
unsigned checks = 0, failures = 0;
void check(bool good, const std::string &label) {
  ++checks;
  if (!good) ++failures;
  std::cout << (good ? "PASS " : "FAIL ") << label << '\n';
}
void write(const fs::path &path, const std::string &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << bytes;
  out.close();
  if (!out) throw std::runtime_error("cannot write owned fixture: " + path.string());
}
const std::string preamble =
    "#include <matcore/region.h>\nusing namespace matcore::mdsl;\n";
const std::string strict_helper = R"cpp(
Value two_gemm(Value a, Value b, Value d) {
  auto c = gemm(a, b, Numerics::strict_f32);
  return gemm(c, d, Numerics::strict_f32);
}
)cpp";
const std::string entry = R"cpp(
Shape identity_shape(Shape n) { return n; }
MATCORE_REGION Result region(Storage A, Storage B, Storage D, Storage E,
                            Shape m, Shape k, Shape n, Shape p) noexcept {
  auto a = read(A, m, k);
  auto b = read(B, k, n);
  auto original_width = cols(b);
  auto width = identity_shape(original_width);
  auto d = read(D, width, p);
  auto e = two_gemm(a, b, d);
  publish(e, E);
  observe(E);
  return complete();
}
)cpp";
const std::string host = R"cpp(
#include <iostream>
#include <string>
#include <vector>
int main(int argc, char **argv) {
  const bool refusal = argc > 1 && std::string(argv[1]) == "refusal";
  const bool square = argc > 1 && std::string(argv[1]) == "square";
  const bool shape_fail = argc > 1 && std::string(argv[1]) == "shape";
  Shape m=2, k=square?2:4, n=square?2:3, p=2;
  std::vector<float> a(m*k),b(k*n),d(n*p),e(m*p,-777);
  for (Shape i=0;i<a.size();++i) a[i]=float(int(i%7)-3);
  for (Shape i=0;i<b.size();++i) b[i]=float(int((i*3+1)%9)-4);
  for (Shape i=0;i<d.size();++i) d[i]=float(int((i*5+2)%11)-5);
  // Small integer products/sums are exact in both f32 and f64. This oracle
  // explicitly rounds the intermediate result and never commutes operands.
  std::vector<float> c(m*n),expected(m*p);
  for(Shape i=0;i<m;++i) for(Shape j=0;j<n;++j) {
    double v=0; for(Shape x=0;x<k;++x) v+=double(a[i*k+x])*b[x*n+j];
    c[i*n+j]=float(v);
  }
  for(Shape i=0;i<m;++i) for(Shape j=0;j<p;++j) {
    double v=0; for(Shape x=0;x<n;++x) v+=double(c[i*n+x])*d[x*p+j];
    expected[i*p+j]=float(v);
  }
  auto result=region({a.data(),m,k,m*k},{b.data(),k,n,k*n},
      {d.data(),n+shape_fail,p,n*p},
      {e.data(),m,p,m*p,Access::read_write},m,k,n,p);
  if(refusal || shape_fail) {
    const Shape failed=shape_fail?3:4, completed=failed-1;
    bool good=!result && result.error()==(shape_fail?Error::shape_mismatch:Error::candidate_incompatible)
        && result.failed_frontier()==failed && result.completed_frontier()==completed
        && result.completed_effect_frontier()==0 && result.publication_count()==0
        && result.observation_count()==0;
    for(float x:e) good &= x==-777;
    std::cout << "PREFIX failed=" << result.failed_frontier()
              << " completed=" << result.completed_frontier()
              << " effects=" << result.completed_effect_frontier()
              << " publications=" << result.publication_count() << '\n';
    return good?0:1;
  }
  bool good=result.ok() && result.completed_frontier()==8
      && result.completed_effect_frontier()==7 && result.publication_count()==1
      && result.observation_count()==1;
  auto observation=result.observation(0);
  good &= observation.valid() && observation.rows()==m && observation.columns()==p;
  for(Shape i=0;i<e.size();++i)
    good &= e[i]==expected[i] && observation.valid() && observation.data()[i]==expected[i];
  std::cout << "MATH " << (good?"PASS":"FAIL") << " expected=";
  for(float x:expected) std::cout << x << ',';
  std::cout << " actual="; for(float x:e) std::cout << x << ',';
  std::cout << '\n';
  return good?0:1;
}
)cpp";
std::string replace(std::string text, const std::string &from, const std::string &to) {
  std::size_t at=0;
  while((at=text.find(from,at))!=std::string::npos) {
    text.replace(at,from.size(),to); at+=to.size();
  }
  return text;
}
struct Fixture {
  fs::path directory;
  fe::Options options;
  fe::ExperimentalRegionHeaders headers;
  Fixture(const fs::path &root, const std::string &name, const std::string &source,
          const std::string &library, char **argv) : directory(root/name) {
    fs::create_directory(directory);
    write(directory/"source.mdsl",source+host);
    write(directory/"math_library.h",library);
    write(directory/"dependency.h","inline constexpr int host_tag = 1;\n");
    const fs::path include=fs::absolute(argv[3]);
    options.input_path=(directory/"source.mdsl").string();
    options.clang_path=argv[1]; options.clang_resource_directory=argv[2];
    options.compiler_arguments={"-std=c++20","-I"+directory.string(),"-I"+include.string()};
    headers={(include/"matcore/region.h").string(),(include/"matcore/detail/region_storage.h").string()};
  }
  fe::ClosedRegionAdmissionResult admit() {
    return fe::admitExperimentalRegionHost(options,directory.string(),headers,"region");
  }
};
void paired(fe::ClosedRegionAdmissionResult &result, const std::string &label) {
  if(!result) return;
  mlir::MLIRContext context;
  auto module=cr::buildModule(result.evidence->program(),context);
  std::string error;
  check(bool(module) && fe::verifyClosedRegionMatchesEvidence(*result.evidence,*module.module,error),
        label+" exact source/witness pairing: "+error);
}
int main(int argc,char **argv) {
  if(argc!=5) return 2;
  try {
    const fs::path root=fs::absolute(argv[4]);
    fs::create_directories(root);
    Fixture strict(root,"main_strict",preamble+strict_helper+entry,"",argv);
    auto baseline=strict.admit();
    check(baseline.syntax_valid && bool(baseline),"main-owned pure helper admitted: "+baseline.error);
    paired(baseline,"main-owned");
    if(!baseline) return 1;
    const auto &body=baseline.evidence->program().regions.at(0).body;
    check(body.size()==7 && body[3].kind==cr::Operation::Kind::Gemm &&
          body[4].kind==cr::Operation::Kind::Gemm && body[3].helper_calls.size()==1 &&
          body[4].helper_calls.size()==1,"two GEMMs expanded with helper call provenance, no semantic call");
    check(body[2].rows.kind==cr::Dimension::Kind::ValueColumns &&
          body[2].rows.reference==body[1].result,"shape helper expands ValueColumns dependence");
    const auto helper=baseline.evidence->entryBinding()->value_helpers.at(0).mangled_name;
    Fixture entry_header(root,"entry_header",preamble+"#include \"math_library.h\"\n"+strict_helper+entry,
        "Result region(Storage,Storage,Storage,Storage,Shape,Shape,Shape,Shape) noexcept;\n",argv);
    auto declaration=entry_header.admit();
    check(declaration.syntax_valid && bool(declaration),
          "entry-only declaration header exception is admitted: "+declaration.error);
    paired(declaration,"entry prototype header");
    const auto relaxed=replace(strict_helper,"strict_f32","reassociate_f32");
    Fixture permissions(root,"main_reassociate",preamble+relaxed+entry,"",argv);
    auto permitted=permissions.admit();
    check(bool(permitted),"same-ABI broader numerical source freshly admitted: "+permitted.error);
    paired(permitted,"reassociate");
    if(permitted) {
      check(permitted.evidence->entryBinding()->value_helpers.at(0).mangled_name==helper,
            "strict and reassociated helper have the same C++ ABI symbol");
      check(permitted.evidence->program().source_sha256!=baseline.evidence->program().source_sha256 &&
            permitted.evidence->program().regions[0].body[3].numerical_profile==cr::NumericalProfile::ReassociateF32,
            "same ABI does not erase distinct body identity or numerical permission");
    }
    Fixture wrong(root,"main_wrong_math",preamble+replace(strict_helper,"gemm(c, d,","gemm(d, c,")+entry,"",argv);
    auto changed=wrong.admit();
    check(bool(changed),"same-ABI different math is admissible as a DIFFERENT program: "+changed.error);
    if(changed) check(changed.evidence->entryBinding()->value_helpers.at(0).mangled_name==helper &&
        changed.evidence->program().regions[0].body[4].lhs!=body[4].lhs,
        "same symbol does not prove operand order or mathematical equivalence");
    const auto hidden=replace(strict_helper,"auto c =","observe(hidden_storage); auto c =");
    struct Rejected { const char *name; std::string source, library, diagnostic; };
    const Rejected rejected[]={
      {"header_strict",preamble+"#include \"math_library.h\"\n"+entry,strict_helper,"source-owned free"},
      {"header_redecl",preamble+"#include \"math_library.h\"\n"+strict_helper+entry,
       "Value two_gemm(Value, Value, Value);\n","redeclaration"},
      {"unknown_external",preamble+"Value two_gemm(Value,Value,Value);\n"+entry,"","unknown host call"},
      {"hidden_header",preamble+"Storage hidden_storage;\n#include \"math_library.h\"\n"+entry,hidden,"source-owned free"},
      {"hidden_main",preamble+"Storage hidden_storage;\n"+hidden+entry,"","pure helper cannot"},
      {"nested_shape_call",preamble+strict_helper+
          replace(entry,"identity_shape(original_width)","identity_shape(cols(b))"),"","nested calls"},
      {"numerics_parameter",preamble+replace(strict_helper,"Value d)","Value d, Numerics mode)")+
          replace(entry,"two_gemm(a, b, d)","two_gemm(a, b, d, Numerics::strict_f32)"),"","parameter"}
    };
    for(const auto &test:rejected) {
      Fixture fixture(root,test.name,test.source,test.library,argv);
      auto result=fixture.admit();
      check(result.syntax_valid,std::string(test.name)+" ordinary C++ well-formed");
      check(!result && result.error.find(test.diagnostic)!=std::string::npos,
            std::string(test.name)+" refuses at expected boundary: "+result.error);
    }
    // Mutate only files owned by this experiment. The existing test callback
    // is deterministic: no race timing, linker substitution or production hook.
    for(const std::string target:{"source.mdsl","dependency.h"}) {
      const std::string source=preamble+"#include \"dependency.h\"\n"+strict_helper+entry;
      Fixture fixture(root,"replace_"+target,source,"",argv);
      auto result=fe::detail::admitExperimentalRegionHostForTesting(
          fixture.options,fixture.directory.string(),fixture.headers,"region",[&]{
            write(fixture.directory/target,target=="source.mdsl"?
                  preamble+relaxed+entry+host:"inline constexpr int host_tag = 2;\n");
          });
      check(!result,"parse/freeze replacement cannot issue evidence ("+target+"): "+result.error);
    }
    Fixture frozen(root,"frozen_dependency",preamble+"#include \"dependency.h\"\n"+strict_helper+entry,"",argv);
    auto sealed=frozen.admit();
    check(bool(sealed),"dependency positive control sealed");
    if(sealed) {
      auto inputs=fe::detail::ClosedRegionCompilationAccess::host(*sealed.evidence);
      std::string error;
      check(inputs->unchanged(error),"live closure initially unchanged: "+error);
      write(frozen.directory/"dependency.h","inline constexpr int host_tag = 3;\n");
      check(!inputs->unchanged(error),"live replaced header is rejected: "+error);
      paired(sealed,"historical frozen dependency replay (not current-file authorization)");
    }
    std::cout << checks << " semantic module checks, " << failures << " failures\n";
    return failures?1:0;
  } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
