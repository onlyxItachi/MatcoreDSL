#include "ClosedRegionAdmissionInternal.h"
#include "../../lib/support/platform_support.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>

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
      const auto snapshot = fe::detail::ClosedRegionCompilationAccess::host(*initial.evidence);
      check(snapshot && snapshot->sourceSnapshot() == load(first.options.input_path) &&
                snapshot->arguments().back() == first.options.input_path &&
                std::find(snapshot->arguments().begin(), snapshot->arguments().end(), "-include") ==
                    snapshot->arguments().end(),
            "public admission freezes original main source without an inspection prelude");
    }
    const std::vector<std::pair<std::string, std::string>> admitted = {
      {"unpolluted ordinary host spelling", "int mdsl_probe = 9;\n" + program(math + "return complete();")},
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
      {"private grammar is not public", "namespace mdsl_probe { struct Storage {}; }\nusing namespace mdsl_probe; [[clang::annotate(\"mdsl.private.closed_region.v1\")]]void region(Storage A){}"},
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
           std::string("namespace matcore::mdslc::runtime::closed_host_v1 {class SessionAbiV2{public:static matcore::mdsl::Result forge(){Status s;s.completed=true;return matcore::mdsl::Result(s,nullptr,{});}};}\n"),
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

    const std::string leaf =
        "inline Value leaf(Value a,Value b){return gemm(a,b,Numerics::strict_f32);}\n"
        "inline Shape dimension(Shape n){return n;}\n";
    const std::string chain =
        "#include \"a_leaf.h\"\n"
        "inline Value two_gemm(Value a,Value b,Value d){auto c=leaf(a,b);return leaf(c,d);}\n";
    const std::string library_source = preamble + "#include \"z_chain.h\"\n"
        "MATCORE_REGION Result region(Storage A,Storage B,Storage D,Storage E,"
        "Shape m,Shape k,Shape n,Shape p) noexcept {"
        "auto width=dimension(n);auto a=read(A,m,k);auto b=read(B,k,width);"
        "auto d=read(D,width,p);auto e=two_gemm(a,b,d);publish(e,E);observe(E);return complete();}\n";
    Fixture library(argv[1], argv[2], include, library_source);
    write(library.temporary.path() / "a_leaf.h", leaf);
    write(library.temporary.path() / "z_chain.h", chain);
    auto imported = library.admit();
    check(imported.syntax_valid && bool(imported), "transitive source-visible pure math library: " + imported.error);
    paired(imported, "source-visible library");
    if (imported) {
      const auto &program = imported.evidence->program();
      check(program.source_files.size() == 3 && program.source_files[0].id == 1 &&
            program.source_files[0].path == program.source_identity &&
            program.source_files[0].sha256 == program.source_sha256 &&
            program.source_files[1].path == (library.temporary.path()/"a_leaf.h").string() &&
            program.source_files[2].path == (library.temporary.path()/"z_chain.h").string(),
            "explicit main plus deterministic path-sorted semantic file table");
      check(program.source_files[1].sha256 == fe::detail::closedRegionDigest(leaf) &&
            program.source_files[1].byte_size == leaf.size() &&
            program.source_files[2].sha256 == fe::detail::closedRegionDigest(chain),
            "library file digests and byte bounds match independently supplied bytes");
      const auto &ops = program.regions[0].body;
      check(ops.size() == 7 && ops[3].site.file_id == 2 && ops[4].site.file_id == 2 &&
            ops[3].helper_calls.size() == 2 && ops[4].helper_calls.size() == 2 &&
            ops[3].helper_calls[0].file_id == 1 && ops[3].helper_calls[1].file_id == 3,
            "transitive expansion retains leaf origin and outermost-first cross-file call chain");
      const auto &helpers = imported.evidence->entryBinding()->value_helpers;
      check(helpers.size() == 2 && helpers[0].body.file_id == 3 && helpers[1].body.file_id == 2,
            "retired Value helpers retain exact definition files; Shape-only helper is not retired");
      auto inputs = fe::detail::ClosedRegionCompilationAccess::host(*imported.evidence);
      std::string error;
      check(inputs->unchanged(error), "library closure initially fresh: " + error);
      write(library.temporary.path()/"a_leaf.h", leaf + "// replaced after sealing\n");
      check(!inputs->unchanged(error), "changed transitive math header rejects live freshness");
      paired(imported, "historical library replay after live replacement");
    }
    Fixture library_race(argv[1], argv[2], include, library_source);
    write(library_race.temporary.path()/"a_leaf.h", leaf);
    write(library_race.temporary.path()/"z_chain.h", chain);
    auto header_raced = fe::detail::admitExperimentalRegionHostForTesting(
        library_race.options, library_race.temporary.path().string(), library_race.headers, "region", [&] {
          write(library_race.temporary.path()/"a_leaf.h", leaf + "// changed during admission\n");
        });
    check(!header_raced, "math-header parse/freeze replacement cannot issue evidence");
    Fixture helper_prototype(argv[1], argv[2], include,
        preamble + "#include \"helper.h\"\n"
        "Value product(Value a,Value b){return gemm(a,b,Numerics::strict_f32);}\n" +
        signature + "{auto a=read(A,m,k);auto b=read(B,k,n);auto c=product(a,b);publish(c,C);return complete();}");
    write(helper_prototype.temporary.path()/"helper.h", "Value product(Value,Value);\n");
    auto helper_decl = helper_prototype.admit();
    check(bool(helper_decl), "captured header redeclaration with main helper definition: " + helper_decl.error);
    paired(helper_decl, "helper redeclaration");
    const std::string primary_template =
        "template<class Tag,class T> T select_product(T a,T b){"
        "return gemm(b,a,Numerics::strict_f32);}\n";
    const std::string explicit_specialization =
        "template<> Value select_product<int,Value>(Value a,Value b){"
        "return gemm(a,b,Numerics::strict_f32);}\n";
    const std::string specialized_source = preamble +
        "#include \"a_primary.h\"\n#include \"z_specialization.h\"\n" + signature +
        "{auto a=read(A,m,k);auto b=read(B,k,n);"
        "auto first=select_product<int>(a,b);auto second=select_product<long>(a,b);"
        "publish(first,C);publish(second,C);return complete();}";
    Fixture specialized(argv[1], argv[2], include, specialized_source);
    write(specialized.temporary.path()/"a_primary.h", primary_template);
    write(specialized.temporary.path()/"z_specialization.h", explicit_specialization);
    auto selected = specialized.admit();
    check(selected.syntax_valid && bool(selected),
          "cross-header explicit specialization and generic primary instantiation: " + selected.error);
    paired(selected, "cross-header concrete template bodies");
    if (selected) {
      const auto &program = selected.evidence->program();
      const auto &ops = program.regions[0].body;
      const auto &helpers = selected.evidence->entryBinding()->value_helpers;
      check(program.source_files.size() == 3 &&
            program.source_files[1].sha256 == fe::detail::closedRegionDigest(primary_template) &&
            program.source_files[2].sha256 == fe::detail::closedRegionDigest(explicit_specialization),
            "synthetic redeclaration and selected definition retain both exact source identities");
      check(ops.size() == 6 && ops[2].site.file_id == 3 && ops[3].site.file_id == 2 &&
            ops[2].lhs == ops[3].rhs && ops[2].rhs == ops[3].lhs &&
            ops[2].helper_calls.size() == 1 && ops[2].helper_calls[0].file_id == 1 &&
            ops[3].helper_calls.size() == 1 && ops[3].helper_calls[0].file_id == 1,
            "selected specialization and generic primary preserve distinct bodies, operand order and callers");
      check(helpers.size() == 2 && helpers[0].body.file_id == 3 && helpers[1].body.file_id == 2 &&
            helpers[0].mangled_name != helpers[1].mangled_name,
            "both concrete template symbols retire with their actual definition owners");
    }
    Fixture impure_specialization(argv[1], argv[2], include, specialized_source);
    write(impure_specialization.temporary.path()/"a_primary.h", primary_template);
    write(impure_specialization.temporary.path()/"z_specialization.h",
          "Storage hidden;\ntemplate<> Value select_product<int,Value>(Value a,Value b){"
          "observe(hidden);return a;}\n");
    auto impure_selected = impure_specialization.admit();
    check(impure_selected.syntax_valid && !impure_selected &&
          impure_selected.error.find("pure helper cannot") != std::string::npos &&
          impure_selected.error.find("z_specialization.h:2:") != std::string::npos,
          "selected cross-header specialization rejects its actual hidden effect: " + impure_selected.error);
    const std::string one_product = preamble + "#include \"helper.h\"\n" + signature +
        "{auto a=read(A,m,k);auto b=read(B,k,n);auto c=product(a,b);publish(c,C);return complete();}";
    for (const auto &[name, bytes, reason] : std::vector<std::tuple<std::string,std::string,std::string>>{
        {"hidden observation", "Storage hidden;\nValue product(Value a,Value b){observe(hidden);return a;}\n", "pure helper cannot"},
        {"unknown body", "Value product(Value,Value);\n", "unknown host call"},
        {"inherited FP", "#pragma clang fp reassociate(on)\nValue product(Value a,Value b){return gemm(a,b,Numerics::strict_f32);}\n", "floating-point policy"},
        {"macro body", "#define BODY return gemm(a,b,Numerics::strict_f32)\nValue product(Value a,Value b){BODY;}\n", "preprocessing"},
        {"nested call", "Value product(Value a,Value b){return gemm(gemm(a,b,Numerics::strict_f32),b,Numerics::strict_f32);}\n", "nested calls"}}) {
      Fixture hostile(argv[1], argv[2], include, one_product);
      write(hostile.temporary.path()/"helper.h", bytes);
      auto result = hostile.admit();
      check(result.syntax_valid && !result && result.error.find(reason) != std::string::npos,
            "header " + name + " rejects its actual closed grammar: " + result.error);
    }
    Fixture spoof(argv[1], argv[2], include, one_product);
    write(spoof.temporary.path()/"helper.h",
        "#line 900 \"forged-main.mdsl\"\nStorage hidden;\n"
        "Value product(Value a,Value b){observe(hidden);return a;}\n");
    auto spoofed = spoof.admit();
    check(spoofed.syntax_valid && !spoofed &&
          spoofed.error.find((spoof.temporary.path()/"helper.h").string()+":3:") != std::string::npos &&
          spoofed.error.find("pure helper cannot") != std::string::npos,
          "header diagnostic uses physical spelling file/line, never presumed #line identity: " + spoofed.error);
  } catch (const std::exception &error) {
    std::cerr << "EXCEPTION: " << error.what() << '\n';
    return 2;
  }
  std::cout << checks << " experimental admission checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
