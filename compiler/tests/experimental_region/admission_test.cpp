#include "ClosedRegionAdmissionInternal.h"
#include "../../lib/support/platform_support.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fe = matcore::mdslc::frontend;
namespace cr = matcore::mdslc::closed_region;
namespace fs = std::filesystem;
namespace support = matcore::mdslc::support;
namespace {
unsigned checks = 0, failures = 0;
void check(bool good, const std::string &label) {
  ++checks;
  if (!good) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
void write(const fs::path &path, const std::string &bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << bytes;
  file.close();
  if (!file) throw std::runtime_error("cannot write test fixture");
}
std::string load(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), {}};
}
struct Fixture {
  support::TempDirectoryV1 temporary;
  fe::Options options;
  fe::ExperimentalRegionHeaders headers;
  Fixture(const std::string &clang, const std::string &resource,
          const fs::path &include, const std::string &source) {
    std::string error;
    auto owned = support::create_temp_directory_v1("mdslc-experimental-region", error);
    if (!owned) throw std::runtime_error(error);
    temporary = std::move(*owned);
    options.input_path = (temporary.path() / "source.mdsl").string();
    options.clang_path = clang;
    options.clang_resource_directory = resource;
    options.compiler_arguments = {"-std=c++20", "-I" + temporary.path().string(),
                                  "-I" + include.string()};
    headers = {(include / "matcore/region.h").string(),
               (include / "matcore/detail/region_storage.h").string()};
    write(options.input_path, source);
  }
  fe::ClosedRegionAdmissionResult admit(const std::string &name = "region") {
    return fe::admitExperimentalRegionHost(options, temporary.path().string(), headers, name);
  }
};
const std::string preamble = "#include <matcore/region.h>\nusing namespace matcore::mdsl;\n";
const std::string signature = "MATCORE_REGION\nResult region(Storage A, Storage B, Storage C, Shape m, Shape k, Shape n) noexcept";
const std::string math = "auto a=read(A,m,k); auto b=read(B,k,n); auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C);";
std::string program(const std::string &body) { return preamble + signature + " {\n" + body + "\n}\n"; }
void paired(fe::ClosedRegionAdmissionResult &result, const std::string &label) {
  if (!result) return;
  mlir::MLIRContext context;
  auto module = cr::buildModule(result.evidence->program(), context);
  check(bool(module), label + " semantic graph builds: " + module.error);
  if (!module) return;
  std::string error;
  check(fe::verifyClosedRegionMatchesEvidence(*result.evidence, *module.module, error),
        label + " paired frozen replay: " + error);
}
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  const fs::path include = fs::absolute(argv[3]).lexically_normal();
  try {
    Fixture first(argv[1], argv[2], include, program(math + "return complete();"));
    auto initial = first.admit();
    check(initial.syntax_valid, "ordinary Clang C++ accepts public source: " + initial.error);
    check(bool(initial), "public mathematical source admitted: " + initial.error);
    paired(initial, "public");
    if (initial) {
      const auto &binding = *initial.evidence->entryBinding();
      check(binding.qualified_name == "region" && !binding.mangled_name.empty() &&
                binding.parameters.size() == 6 && binding.namespaces.empty(),
            "entry has exact host signature and named parameter witness");
      const auto source = initial.evidence->sourceSnapshot();
      check(source.substr(binding.body.offset, 1) == "{" &&
                source.substr(binding.completion.offset, binding.completion.length) == "return complete()",
            "body and terminal completion source ranges are sealed");
      check(bool(fe::detail::ClosedRegionCompilationAccess::host(*initial.evidence)),
            "compiler-only frozen host access exists");
    }
    const std::vector<std::pair<std::string, std::string>> admitted = {
      {"rhs carried rectangular", program("auto a=read(A,2,3);auto b=read(B,3,4);auto c=gemm(a,b,Numerics::strict_f32);publish(c,C);auto d=read(A,5,2);auto e=gemm(d,c,Numerics::reassociate_f32);publish(e,C);return complete();")},
      {"lhs carried late read and branch", program(math + "auto late=read(C,m,n); if(m<n){auto e=gemm(c,late,Numerics::strict_f32);publish(e,C);}else{observe(C);} return complete();")},
      {"old API header first", "#include <matcore/mdsl.h>\n" + program(math + "return complete();")},
      {"host standard-library debug configuration", "#define _GLIBCXX_DEBUG 1\n#define _GLIBCXX_USE_CXX11_ABI 0\n#include <vector>\n" + program(math + "return complete();")},
      {"old API header second", preamble + "#include <matcore/mdsl.h>\n" + signature + "{" + math + "return complete();}"},
      {"literal marker", preamble + "[[clang::annotate(\"matcore.experimental.region.v1\")]] Result region(Storage A,Storage B,Storage C,Shape m,Shape k,Shape n) noexcept{" + math + "return complete();}"},
      {"pure helper", preamble + "Value product(Value a,Value b){return gemm(a,b,Numerics::strict_f32);}\n" + signature + "{auto a=read(A,m,k);auto b=read(B,k,n);auto c=product(a,b);publish(c,C);return complete();}"},
      {"shape helper", preamble + "Shape first(Shape n){return n;}\n" + signature + "{auto dim=first(m);auto a=read(A,dim,k);auto b=read(B,k,n);auto c=gemm(a,b,Numerics::strict_f32);publish(c,C);return complete();}"},
      {"template helper", preamble + "template<class T> T product(T a,T b){return gemm(a,b,Numerics::strict_f32);}\n" + signature + "{auto a=read(A,m,k);auto b=read(B,k,n);auto c=product(a,b);publish(c,C);return complete();}"}
    };
    for (const auto &[name, source] : admitted) {
      Fixture fixture(argv[1], argv[2], include, source);
      auto result = fixture.admit();
      check(bool(result), name + ": " + result.error);
      paired(result, name);
      if (result && (name == "pure helper" || name == "template helper"))
        check(result.evidence->entryBinding()->value_helpers.size() == 1 &&
              !result.evidence->entryBinding()->value_helpers[0].mangled_name.empty(),
              name + " exact instantiated source Value helper symbol bound");
      if (result && name == "shape helper")
        check(result.evidence->entryBinding()->value_helpers.empty(), "shape-only helper remains ordinary host code");
    }
    const std::vector<std::pair<std::string, std::string>> rejected = {
      {"return comma", program(math + "return (observe(C),complete());")},
      {"early return", program("return complete();" + math)},
      {"branch return", program("if(m<n){return complete();}else{observe(C);}return complete();")},
      {"completion statement", program(math + "complete();return complete();")},
      {"callee comma", program(math + "return (observe(C),complete)();")},
      {"host call", preamble + "void effect();\n" + signature + "{effect();return complete();}"},
      {"hidden destructor", preamble + "struct Guard{~Guard();};\n" + signature + "{Guard guard;return complete();}"},
      {"conversion", preamble + "struct X{operator Result();};\n" + signature + "{return X{};}"},
      {"fake completion", preamble + "Result other()noexcept;\n" + signature + "{return other();}"},
      {"completion redefinition", preamble + "namespace matcore::mdsl{Result complete()noexcept{return complete();}}\n" + signature + "{return complete();}"},
      {"missing noexcept", preamble + "MATCORE_REGION Result region(Storage A){return complete();}"},
      {"forged marker macro", preamble + "#undef MATCORE_REGION\n#define MATCORE_REGION [[clang::annotate(\"matcore.experimental.region.v1\")]]\n" + signature + "{return complete();}"},
      {"nested marker macro", preamble + "#define REGION MATCORE_REGION\nREGION Result region(Storage A)noexcept{return complete();}"},
      {"private grammar is not public", "using namespace mdsl_probe; [[clang::annotate(\"mdsl.private.closed_region.v1\")]]void region(Storage A){}"},
      {"volatile", program("volatile Shape x=m;return complete();")},
      {"consteval placeholder evaluation", preamble + "consteval Shape folded(){return sizeof(Value);}\n" + signature + "{auto dim=folded();auto a=read(A,dim,k);return complete();}"},
      {"constexpr placeholder evaluation", preamble + "constexpr Shape folded(){return sizeof(Value);}\n" + signature + "{auto dim=folded();auto a=read(A,dim,k);return complete();}"},
      {"throw", program("throw 1;return complete();")},
      {"mutable pointer", program("auto p=&A;return complete();")}
    };
    for (const auto &[name, source] : rejected) {
      Fixture fixture(argv[1], argv[2], include, source);
      auto result = fixture.admit();
      check(result.syntax_valid, name + " is ordinary well-formed C++: " + result.error);
      check(!result && !result.error.empty(), name + " fail closed: " + result.error);
    }
    for (const auto *specifier : {"constexpr", "consteval"}) {
      Fixture constant(argv[1], argv[2], include,
          preamble + "MATCORE_REGION " + specifier + " Result region() noexcept{return complete();}");
      auto result = constant.admit();
      check(!result, std::string(specifier) + " entry cannot be issued as a runtime region");
      // With the actual owning nonliteral Result, C++20 itself rejects these
      // declarations. Do not report this as a well-formed admission rejection.
      check(!result.syntax_valid, std::string(specifier) + " owning Result is not a C++20 constant-evaluation type");
    }
    Fixture namespaced(argv[1], argv[2], include, preamble + "namespace math {inline namespace v1 {" + signature + "{" + math + "return complete();}}}");
    auto ns = namespaced.admit("math::v1::region");
    check(bool(ns), "named inline namespace entry: " + ns.error);
    if (ns) check(ns.evidence->entryBinding()->namespaces.size() == 2 && ns.evidence->entryBinding()->namespaces[1].is_inline,
                  "namespace binding preserves inline namespace");
    Fixture prototype(argv[1], argv[2], include, preamble + "#include \"api.h\"\n" + signature + "{" + math + "return complete();}");
    write(prototype.temporary.path() / "api.h", "#include <matcore/region.h>\nmatcore::mdsl::Result region(matcore::mdsl::Storage,matcore::mdsl::Storage,matcore::mdsl::Storage,matcore::mdsl::Shape,matcore::mdsl::Shape,matcore::mdsl::Shape)noexcept;\n");
    auto proto = prototype.admit();
    check(bool(proto), "ordinary declaration-only header prototype: " + proto.error);
    paired(proto, "prototype");
    Fixture shadow(argv[1], argv[2], include, program(math + "return complete();"));
    write(shadow.temporary.path() / "matcore/region.h", load(include / "matcore/region.h"));
    check(!shadow.admit(), "copied identical region header cannot forge installation FileID");
    Fixture packed(argv[1], argv[2], include, "#pragma pack(push,1)\n" + program(math + "return complete();") + "#pragma pack(pop)\n");
    check(!packed.admit(), "ambient packing cannot change canonical resource/result layout");
    Fixture attributed(argv[1], argv[2], include, "#pragma clang attribute push(__attribute__((trivial_abi)), apply_to=record)\n" + program(math + "return complete();") + "#pragma clang attribute pop\n");
    check(!attributed.admit(), "ambient ownership ABI attribute cannot change canonical result contract");
    Fixture member_attribute(argv[1], argv[2], include,
        "#pragma clang attribute push([[noreturn]], apply_to=function(is_member))\n" +
        preamble + "#pragma clang attribute pop\n" + signature + "{" + math + "return complete();}");
    auto member_result = member_attribute.admit();
    check(member_result.syntax_valid && !member_result,
          "well-formed ambient member-only noreturn cannot alter Result destruction: " + member_result.error);
    Fixture enum_attribute(argv[1], argv[2], include,
        "#pragma clang attribute push(__attribute__((enum_extensibility(closed))), apply_to=enum)\n" +
        preamble + "#pragma clang attribute pop\n" + signature + "{" + math + "return complete();}");
    auto enum_result = enum_attribute.admit();
    check(enum_result.syntax_valid && !enum_result,
          "well-formed ambient enum-only attributes cannot change canonical enum contract: " + enum_result.error);
    for (const auto &definition : {
           std::string("inline matcore::mdsl::Observation::~Observation() noexcept {extern void host_effect();host_effect();}\n"),
           std::string("inline matcore::mdsl::Result::Result(Result&&) noexcept :status_(),observations_(nullptr),failure_(){extern void host_effect();host_effect();}\n"),
           std::string("namespace matcore::mdslc::runtime::closed_host_v1 {class Session{public:static matcore::mdsl::Result forge(){Status s;s.completed=true;return matcore::mdsl::Result(s,nullptr,{});}};}\n"),
           std::string("namespace matcore::mdslc::runtime::closed_host_v1 {struct ObservationBlock{int forged;~ObservationBlock(){extern void host_effect();host_effect();}};}\n"),
           std::string("namespace matcore::mdslc::runtime::closed_host_v1 {const char* message(Code) noexcept {extern void host_effect();host_effect();return \"forged\";}}\n")}) {
      Fixture redefined(argv[1], argv[2], include,
          preamble + definition + signature + "{" + math + "return complete();}");
      auto result = redefined.admit();
      check(result.syntax_valid && !result,
            "well-formed out-of-line ownership/runtime definition cannot acquire canonical authority: " + result.error);
    }
    Fixture fp(argv[1], argv[2], include, preamble + "#pragma clang fp reassociate(on)\n" + signature + "{" + math + "return complete();}");
    check(!fp.admit(), "ambient floating-point permissions are not numerical authority");
    Fixture modified(argv[1], argv[2], include, program(math + "return complete();"));
    write(modified.temporary.path() / "matcore/region.h", load(include / "matcore/region.h") + "\n// changed installed prototype\n");
    modified.headers.region_path = (modified.temporary.path() / "matcore/region.h").string();
    check(!modified.admit(), "configured header path cannot authorize changed canonical bytes");
    Fixture race(argv[1], argv[2], include, program(math + "return complete();"));
    auto raced = fe::detail::admitExperimentalRegionHostForTesting(
        race.options, race.temporary.path().string(), race.headers, "region", [&] {
          write(race.options.input_path, program(math + "return (observe(C),complete());"));
        });
    check(!raced, "source change between admission and freeze cannot issue evidence");
  } catch (const std::exception &error) {
    std::cerr << "EXCEPTION: " << error.what() << '\n';
    return 2;
  }
  std::cout << checks << " experimental admission checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
