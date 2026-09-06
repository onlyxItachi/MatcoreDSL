#include "AuthenticatedHostThunk.h"
#include "platform_support.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cg=matcore::mdslc::codegen;
namespace support=matcore::mdslc::support;
namespace fs=std::filesystem;
unsigned checks=0,failures=0;
void check(bool condition,const std::string &name) {
  ++checks;if(!condition){++failures;std::cerr<<"FAIL: "<<name<<'\n';}
}
void write(const fs::path &path,const std::string &text) {
  std::ofstream stream(path);stream<<text;stream.close();
  if(!stream) throw std::runtime_error("fixture write failed");
}
std::string print(const llvm::Module &module) {
  std::string text;llvm::raw_string_ostream output(text);module.print(output,nullptr);return text;
}
llvm::Function *fixtureFunction(llvm::Module &module,const std::string &fragment) {
  llvm::Function *found=nullptr;
  for(auto &function:module) if(function.getName().contains(fragment)) {
    if(found) throw std::runtime_error("ambiguous fixture function: "+fragment);
    found=&function;
  }
  if(!found) throw std::runtime_error("missing fixture function: "+fragment);
  return found;
}
struct Fixture {
  support::TempDirectoryV1 temporary;
  llvm::LLVMContext context;
  std::string clang;
  Fixture(std::string clang):clang(std::move(clang)) {
    std::string error;
    auto made=support::create_temp_directory_v1("mdslc-abi-thunk",error);
    if(!made) throw std::runtime_error(error);
    temporary=std::move(*made);
  }
  support::ProcessResultV1 run(std::vector<std::string> argv) {
    support::ProcessRequestV1 request;request.argv=std::move(argv);
    request.working_directory=temporary.path();
    request.environment=support::compiler_environment_sanitization_v1();
    request.environment.push_back({"DEBUGINFOD_URLS",std::string()});
    return support::run_process_v1(request);
  }
  void sanitizerFlags(std::vector<std::string> &argv) {
#ifdef MDSLC_ABI_THUNK_SANITIZE
    argv.push_back("-fsanitize=address,undefined");
    argv.push_back("-fno-omit-frame-pointer");
#else
    (void)argv;
#endif
  }
  std::unique_ptr<llvm::Module> compile(const std::string &name,const std::string &source,bool debug=true) {
    const auto cpp=temporary.path()/(name+".cpp"),ir=temporary.path()/(name+".ll");
    write(cpp,source);
    std::vector<std::string> argv={clang,"-std=c++20","-O0","-Xclang","-disable-llvm-passes",
      "-Wno-return-type-c-linkage","-S","-emit-llvm",cpp.string(),"-o",ir.string()};
    if(debug) argv.push_back("-g");
    sanitizerFlags(argv);
    auto compiled=run(std::move(argv));
    check(compiled.launched && compiled.exit_code==0,"actual Clang fixture: "+name+compiled.stderr_text);
    llvm::SMDiagnostic diagnostic;
    auto result=llvm::parseIRFile(ir.string(),diagnostic,context);
    if(!result) { diagnostic.print("abi fixture",llvm::errs());throw std::runtime_error("fixture LLVM parse failed"); }
    return result;
  }
  std::string execute(const llvm::Module &module,const std::string &name,bool expectAsan=false) {
    auto ir=temporary.path()/(name+".ll"),exe=temporary.path()/name;
    write(ir,print(module));
    std::vector<std::string> argv={clang,"-O2",ir.string(),"-o",exe.string()};
    sanitizerFlags(argv);
    auto compiled=run(std::move(argv));
    check(compiled.launched && compiled.exit_code==0,"real final link: "+name+compiled.stderr_text);
    if(compiled.exit_code) return {};
    auto executed=run({exe.string()});
    if(expectAsan)check(executed.launched && executed.exit_code!=0 &&
      executed.stderr_text.find("AddressSanitizer: stack-buffer-overflow")!=std::string::npos,
      "actual thunked helper load is instrumented by ASan: "+executed.stderr_text);
    else check(executed.launched && executed.exit_code==0,"real executable: "+name+executed.stderr_text);
    return executed.stdout_text;
  }
};

int main(int argc,char **argv) {
  if(argc!=2) return 2;
  try {
    Fixture fixture(argv[1]);
    write(fixture.temporary.path()/"facade.h",R"cpp(
#pragma once
#include <cstdint>
#include <vector>
struct alignas(16) Storage {float *data; std::uint64_t rows,columns,capacity;};
struct Shape {std::uint64_t value;};
extern int result_destructors;
struct Result {
  std::vector<float> values;
  std::uint64_t tag;
  ~Result() noexcept {++result_destructors;}
};
inline Shape ShapeHelper(Shape value) {return {value.value+1};}
)cpp");
    const std::string hostSource=R"cpp(
#include "facade.h"
#include <cstdio>
#include <source_location>
int result_destructors=0;
static int ValueLeaf(int n) {return n+1;}
static int ValueRoot(int n) {return ValueLeaf(n);}
namespace mathematical {
__attribute__((visibility("hidden"))) Result region(Storage storage,Shape shape) noexcept { (void)storage; return Result{{float(ValueRoot(int(shape.value)))},0}; } constexpr unsigned location_column=__builtin_COLUMN();
constexpr auto location=std::source_location::current();
}
struct Guard {int &trace; Guard(int &t):trace(t){trace=trace*10+1;} ~Guard(){trace=trace*10+2;}};
int main() {
  float input=5;Storage storage{&input,1,1,1};int trace=0;
  using Function=Result(*)(Storage,Shape) noexcept;
  Function volatile pointer=&mathematical::region;
  {
    Guard guard(trace);
    auto direct=mathematical::region(storage,ShapeHelper({6}));
    auto indirect=pointer(storage,Shape{8});
    std::printf("math %.0f %.0f %llu\n",direct.values[0],indirect.values[0],
                static_cast<unsigned long long>(direct.tag));
    std::printf("host %u %u %u %d %s\n",mathematical::location_column,
                mathematical::location.line(),mathematical::location.column(),trace,
                mathematical::location.file_name());
  }
  std::printf("done %d %d\n",trace,result_destructors);
}
)cpp";
    auto host=fixture.compile("host",hostSource);
    const std::string helperSource=R"cpp(
#include "facade.h"
extern "C" Result __matcore_owned_helper(Storage storage,Shape shape) noexcept {
  return Result{{storage.data[0]+float(shape.value)},42};
}
)cpp";
    auto helper=fixture.compile("helper",helperSource);
    auto *original=fixtureFunction(*host,"6region"),*generated=fixtureFunction(*helper,"__matcore_owned_helper");
    cg::HostThunkRequest request{original->getName().str(),generated->getName().str(),
      {fixtureFunction(*host,"ValueRoot")->getName().str(),fixtureFunction(*host,"ValueLeaf")->getName().str()}};
    bool sret=false,byval=false;
    for(unsigned i=0;i<original->arg_size();++i) {
      sret|=original->hasParamAttribute(i,llvm::Attribute::StructRet);
      byval|=original->hasParamAttribute(i,llvm::Attribute::ByVal);
    }
    check(sret && byval,"real Clang uses Result sret and Storage byval");
    const auto before=print(*host),helperBefore=print(*helper);
    auto transformed=cg::linkAuthenticatedHostThunk(*host,*helper,request);
    check(static_cast<bool>(transformed),"matched real Clang aggregate ABI links: "+transformed.error);
    check(print(*host)==before,"host input module unchanged after transformation");
    check(print(*helper)==helperBefore,"helper input module unchanged after transformation");
    if(transformed) {
      auto *thunk=transformed.module->getFunction(request.host_symbol);
      check(thunk && thunk->getVisibility()==original->getVisibility() &&
            thunk->getLinkage()==original->getLinkage(),"original symbol visibility and linkage preserved");
      check(!transformed.module->getFunction(request.retired_value_functions[0]) &&
            !transformed.module->getFunction(request.retired_value_functions[1]),"exact sealed Value subgraph retired");
      check(fixtureFunction(*transformed.module,"ShapeHelper")!=nullptr,"ordinary Shape helper retained");
      const auto old=fixture.execute(*host,"original"),now=fixture.execute(*transformed.module,"thunked");
      check(now.starts_with("math 12 13 42\n"),"direct and function-pointer calls execute generated Result");
      const auto oldHost=old.find("host "),newHost=now.find("host ");
      check(oldHost!=std::string::npos && newHost!=std::string::npos && old.substr(oldHost)==now.substr(newHost),
            "ordinary RAII, source file, source line and same-line column are preserved");
      check(now.ends_with("done 12 2\n"),"Result and host RAII destruction occur exactly once per object");
    }
    auto rejectHelper=[&](const std::string &label,auto mutate) {
      auto changed=llvm::CloneModule(*helper);mutate(*changed,*changed->getFunction(request.helper_symbol));
      auto rejected=cg::linkAuthenticatedHostThunk(*host,*changed,request);
      check(!rejected && !rejected.error.empty(),label);
      check(print(*host)==before,"host unchanged after rejected "+label);
    };
    rejectHelper("changed target triple rejected",[](auto &m,auto &){m.setTargetTriple(llvm::Triple("aarch64-unknown-linux-gnu"));});
    rejectHelper("changed data layout rejected",[](auto &m,auto &){m.setDataLayout("e-p:32:32");});
    rejectHelper("changed calling convention rejected",[](auto &,auto &f){f.setCallingConv(llvm::CallingConv::Fast);});
    rejectHelper("changed target feature ABI rejected",[](auto &,auto &f){f.addFnAttr("target-features","+avx2");});
    rejectHelper("missing noexcept realization rejected",[](auto &,auto &f){f.removeFnAttr(llvm::Attribute::NoUnwind);});
    rejectHelper("changed sret alignment rejected",[](auto &m,auto &f){f.addParamAttr(0,llvm::Attribute::getWithAlignment(m.getContext(),llvm::Align(32)));});
    rejectHelper("missing sret contract rejected",[](auto &,auto &f){f.removeParamAttr(0,llvm::Attribute::StructRet);});
    for(bool sameSize:{false,true}) rejectHelper(sameSize?"same-size record permutation rejected":"changed nested record size rejected",
      [sameSize](auto &m,auto &f){
        auto *old=f.getParamStructRetType(0);auto *record=llvm::cast<llvm::StructType>(old);
        std::vector<llvm::Type*> elements(record->element_begin(),record->element_end());
        if(sameSize) std::reverse(elements.begin(),elements.end());else elements.push_back(llvm::Type::getInt64Ty(m.getContext()));
        auto *changed=llvm::StructType::create(m.getContext(),elements,"ChangedResult");
        f.addParamAttr(0,llvm::Attribute::getWithStructRetType(m.getContext(),changed));
      });
    rejectHelper("byval Storage field permutation rejected",[](auto &m,auto &f){
      auto *record=llvm::cast<llvm::StructType>(f.getParamByValType(1));
      std::vector<llvm::Type*> elements(record->element_begin(),record->element_end());
      std::swap(elements.front(),elements.back());
      auto *changed=llvm::StructType::create(m.getContext(),elements,"ChangedStorage");
      f.addParamAttr(1,llvm::Attribute::getWithByValType(m.getContext(),changed));
    });
    rejectHelper("byval Storage alignment mismatch rejected",[](auto &m,auto &f){
      f.addParamAttr(1,llvm::Attribute::getWithAlignment(m.getContext(),llvm::Align(32)));
    });
    auto debugHelper=fixture.compile("debug-helper","#define _GLIBCXX_DEBUG 1\n"+helperSource);
    check(!cg::linkAuthenticatedHostThunk(*host,*debugHelper,request),
          "real stdlib debug Result layout mismatch rejected despite identical facade bytes");
#ifdef MDSLC_ABI_THUNK_SANITIZE
    auto unsafeSource=helperSource;
    unsafeSource.replace(unsafeSource.find("storage.data[0]"),std::string("storage.data[0]").size(),
                         "storage.data[storage.capacity]");
    auto unsafeHelper=fixture.compile("unsafe-helper",unsafeSource);
    auto unsafe=cg::linkAuthenticatedHostThunk(*host,*unsafeHelper,request);
    check(static_cast<bool>(unsafe),"instrumentation negative control has identical ABI: "+unsafe.error);
    if(unsafe)(void)fixture.execute(*unsafe.module,"asan-negative",true);
#endif
    auto prior=std::move(transformed);
    prior=cg::linkAuthenticatedHostThunk(*host,*helper,request);
    check(static_cast<bool>(prior),"populated result move assignment preserves owned context lifetime");
    // An owning opaque result has no shared vector ODR implementation to
    // incidentally force LLVM Linker's identified sret types to be uniqued.
    // Compile the same real C++ record in two modules, including actual ASan
    // and UBSan instrumentation in that build, and execute the linked result.
    write(fixture.temporary.path()/"opaque.h",R"cpp(
#pragma once
#include <cstdint>
struct OpaqueStorage {float *data; std::uint64_t rows,columns,capacity;};
struct OpaqueShape {std::uint64_t value;};
struct OpaqueStatus {
  unsigned char code;
  std::uint64_t failed,frontier,effect,publications,observations;
  bool completed;
};
struct OpaqueLocation {const char *file; unsigned line,column;};
class OpaqueResult {
public:
  OpaqueResult(std::uint64_t value,void *owner) noexcept;
  OpaqueResult(const OpaqueResult &)=delete;
  ~OpaqueResult() noexcept;
  std::uint64_t value() const noexcept {return status_.frontier;}
  void *owner() const noexcept {return owner_;}
private:
  OpaqueStatus status_;
  void *owner_;
  OpaqueLocation location_{};
};
)cpp");
    auto opaqueHost=fixture.compile("opaque-host",R"cpp(
#include "opaque.h"
#include <cstdio>
int opaque_destructors=0;
OpaqueResult::OpaqueResult(std::uint64_t value,void *owner) noexcept
    :status_{0,0,value,0,0,0,true},owner_(owner) {}
OpaqueResult::~OpaqueResult() noexcept {++opaque_destructors;}
namespace opaque_math {
OpaqueResult region(OpaqueStorage storage,OpaqueShape shape) noexcept {
  return OpaqueResult(shape.value,storage.data);
}
}
int main() {
  float source=4;
  using Function=OpaqueResult(*)(OpaqueStorage,OpaqueShape) noexcept;
  Function volatile pointer=&opaque_math::region;
  {
    auto value=pointer(OpaqueStorage{&source,1,1,1},OpaqueShape{7});
    std::printf("opaque %llu %d\n",static_cast<unsigned long long>(value.value()),value.owner()==&source);
  }
  std::printf("destroyed %d\n",opaque_destructors);
}
)cpp",false);
    auto opaqueHelper=fixture.compile("opaque-helper",R"cpp(
#include "opaque.h"
extern "C" OpaqueResult __matcore_opaque_helper(OpaqueStorage storage,OpaqueShape shape) noexcept {
  return OpaqueResult(shape.value+static_cast<unsigned>(*storage.data),storage.data);
}
)cpp",false);
    cg::HostThunkRequest opaqueRequest{
      fixtureFunction(*opaqueHost,"6region")->getName().str(),"__matcore_opaque_helper",{}};
    auto opaqueThunk=cg::linkAuthenticatedHostThunk(*opaqueHost,*opaqueHelper,opaqueRequest);
    check(static_cast<bool>(opaqueThunk),"opaque real Clang sret ABI does not require identified-type uniquing: "+opaqueThunk.error);
    if(opaqueThunk) {
#ifdef MDSLC_ABI_THUNK_SANITIZE
      auto *hostEntry=opaqueThunk.module->getFunction(opaqueRequest.host_symbol);
      auto *helperEntry=opaqueThunk.module->getFunction(opaqueRequest.helper_symbol);
      check(hostEntry->getParamStructRetType(0)!=helperEntry->getParamStructRetType(0),
            "exact Clang21 ASan fixture retains distinct isomorphic sret type identities");
#endif
      check(fixture.execute(*opaqueThunk.module,"opaque-thunk")=="opaque 11 1\ndestroyed 1\n",
            "opaque owning Result executes through preserved function pointer and destroys once");
    }
    llvm::LLVMContext separateContext;llvm::SMDiagnostic separateDiagnostic;
    auto separate=llvm::parseIRFile((fixture.temporary.path()/"helper.ll").string(),separateDiagnostic,separateContext);
    check(separate && !cg::linkAuthenticatedHostThunk(*host,*separate,request),"different LLVM contexts rejected");
    auto aliasHost=llvm::CloneModule(*host);
    auto *aliasTarget=aliasHost->getFunction(request.host_symbol);
    llvm::GlobalAlias::create(aliasTarget->getValueType(),aliasTarget->getAddressSpace(),
      llvm::GlobalValue::ExternalLinkage,"original_region_alias",aliasTarget,aliasHost.get());
    auto aliased=cg::linkAuthenticatedHostThunk(*aliasHost,*helper,request);
    check(aliased && aliased.module->getNamedAlias("original_region_alias")->getAliaseeObject()==
          aliased.module->getFunction(request.host_symbol),"ordinary original function alias preserves identity");
    auto metadataHost=llvm::CloneModule(*host);
    auto *metadataTarget=metadataHost->getFunction(request.host_symbol);
    for(const char *name:{"first_cfi_identity","second_cfi_identity"}) {
      llvm::Metadata *entries[]={llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(fixture.context),0)),llvm::MDString::get(fixture.context,name)};
      metadataTarget->addMetadata(llvm::LLVMContext::MD_type,*llvm::MDNode::get(fixture.context,entries));
    }
    auto metadataResult=cg::linkAuthenticatedHostThunk(*metadataHost,*helper,request);
    llvm::SmallVector<llvm::MDNode*,2> types;
    if(metadataResult)metadataResult.module->getFunction(request.host_symbol)->getMetadata(llvm::LLVMContext::MD_type,types);
    check(metadataResult && types.size()==2,"all function CFI identity records preserved");
    auto escaped=llvm::CloneModule(*host);
    auto *root=escaped->getFunction(request.retired_value_functions[0]);
    new llvm::GlobalVariable(*escaped,root->getType(),true,llvm::GlobalValue::ExternalLinkage,root,"escaped_value_helper");
    check(!cg::linkAuthenticatedHostThunk(*escaped,*helper,request),"escaped Value helper address blocks retirement");
    auto outside=llvm::CloneModule(*host);
    auto *leaf=outside->getFunction(request.retired_value_functions[1]);
    auto *extra=llvm::Function::Create(leaf->getFunctionType(),llvm::GlobalValue::ExternalLinkage,"unadmitted_host",*outside);
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(fixture.context,"entry",extra));
    std::vector<llvm::Value*> arguments;for(auto &arg:extra->args())arguments.push_back(&arg);
    builder.CreateRet(builder.CreateCall(leaf,arguments));
    check(!cg::linkAuthenticatedHostThunk(*outside,*helper,request),"outside host call blocks Value helper retirement");
    auto collisionHost=llvm::CloneModule(*host),collisionHelper=llvm::CloneModule(*helper);
    for(auto *module:{collisionHost.get(),collisionHelper.get()})
      new llvm::GlobalVariable(*module,llvm::Type::getInt32Ty(fixture.context),false,
        llvm::GlobalValue::ExternalLinkage,llvm::ConstantInt::get(llvm::Type::getInt32Ty(fixture.context),1),"strong_collision");
    auto isolatedCollision=cg::linkAuthenticatedHostThunk(*collisionHost,*collisionHelper,request);
    check(isolatedCollision &&
          isolatedCollision.module->getNamedGlobal("strong_collision")->hasExternalLinkage(),
          "unrelated strong host global survives isolated private helper definition");
    auto weakHost=fixture.compile("weak-host",R"cpp(
#include <cstdio>
extern "C" __attribute__((weak)) int private_inline(){return 91;}
extern "C" int region(int x) noexcept {return x;}
int main(){std::printf("isolated %d host %d\n",region(3),private_inline());}
)cpp",false);
    auto weakHelper=fixture.compile("weak-helper",R"cpp(
extern "C" __attribute__((weak)) int private_inline(){return 7;}
extern "C" int owned_helper(int x) noexcept {return private_inline()+x;}
)cpp",false);
    auto weakResult=cg::linkAuthenticatedHostThunk(*weakHost,*weakHelper,
                                                  {"region","owned_helper",{}});
    check(weakResult && fixture.execute(*weakResult.module,"isolated-weak")==
          "isolated 10 host 91\n",
          "host weak definition cannot replace compiler-issued private inline body");
    auto missing=request;missing.retired_value_functions.push_back("not_a_sealed_helper");
    check(!cg::linkAuthenticatedHostThunk(*host,*helper,missing),"missing helper binding rejected");
    auto duplicate=request;duplicate.retired_value_functions.push_back(duplicate.retired_value_functions.front());
    check(!cg::linkAuthenticatedHostThunk(*host,*helper,duplicate),"duplicate retirement rejected");
    auto stale=llvm::CloneModule(*host);
    auto *staleTarget=stale->getFunction(request.host_symbol);
    staleTarget->setMemoryEffects(llvm::MemoryEffects::argMemOnly());
    staleTarget->addFnAttr(llvm::Attribute::NoFree);
    for(auto &function:*stale)for(auto &block:function)for(auto &instruction:block)
      if(auto *call=llvm::dyn_cast<llvm::CallBase>(&instruction))
        if(call->getCalledFunction()==staleTarget)call->setMemoryEffects(llvm::MemoryEffects::argMemOnly());
    auto cleared=cg::linkAuthenticatedHostThunk(*stale,*helper,request);
    check(cleared && !cleared.module->getFunction(request.host_symbol)->hasFnAttribute(llvm::Attribute::Memory) &&
          !cleared.module->getFunction(request.host_symbol)->hasFnAttribute(llvm::Attribute::NoFree),
          "thunk discards now-false original function memory facts: "+cleared.error);
    if(cleared)for(auto &function:*cleared.module)for(auto &block:function)for(auto &instruction:block)
      if(auto *call=llvm::dyn_cast<llvm::CallBase>(&instruction))
        if(call->getCalledFunction()==cleared.module->getFunction(request.host_symbol))
          check(!call->hasFnAttr(llvm::Attribute::Memory),"existing direct callsite discards old memory facts");
  } catch(const std::exception &error) {check(false,error.what());}
  std::cout<<checks<<" authenticated ABI thunk checks, "<<failures<<" failures\n";
  return failures?1:0;
}
