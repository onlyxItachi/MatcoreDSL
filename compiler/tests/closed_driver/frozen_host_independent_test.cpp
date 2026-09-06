#include "FrozenHostCodegen.h"
#include "../../lib/support/platform_support.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace fe=matcore::mdslc::frontend;
namespace h=fe::closed_region_host;
namespace s=matcore::mdslc::support;
namespace fs=std::filesystem;
unsigned checks=0,bad=0;
void check(bool okay,const std::string &label){++checks;if(!okay){++bad;std::cerr<<label<<'\n';}}
void write(const fs::path&p,const std::string&t){std::ofstream f(p);f<<t;f.close();if(!f)throw std::runtime_error("write");}
s::ProcessResultV1 run(std::vector<std::string> args,const fs::path&cwd){s::ProcessRequestV1 p;p.argv=std::move(args);p.working_directory=cwd;p.environment=s::compiler_environment_sanitization_v1();return s::run_process_v1(p);}
int main(){
  std::string error;
  auto inputs=s::create_temp_directory_v1("mdslc-independent-frozen-input",error);
  auto outputs=s::create_temp_directory_v1("mdslc-independent-frozen-output",error);
  if(!inputs||!outputs)return 2;
  const auto src=inputs->path()/"original.cpp",header=inputs->path()/"host.h";
  write(header,"#pragma once\n#define HOST_TERM 17\n");
  write(src,
    "#include <cstdio>\n#include <source_location>\n#include \"host.h\"\n"
    "#line 200 \"logical.mdsl\"\n"
    "__attribute__((always_inline)) inline int must_remain(int x){return x+HOST_TERM;}\n"
    "int wrapper(int x){return must_remain(x);}\n"
    "int effects; struct C{C(){effects=effects*10+1;}~C(){effects=effects*10+2;}};\n"
    "int region(int n){C c;try{if(n==1)throw 7;}catch(int x){effects+=x;}return wrapper(n);}\n"
    "int main(int argc,char**){auto fn=&region;auto x=fn(argc);std::printf(\"%d %d %u %u %s\\n\",x,effects,__builtin_LINE(),std::source_location::current().column(),__FILE__);}\n");
  fe::Options options;options.input_path=src.string();options.clang_path="/usr/bin/clang++-21";
  options.clang_resource_directory="/usr/lib/llvm-21/lib/clang/21";
  options.compiler_arguments={"-I"+inputs->path().string()};
  auto capture=h::prepareHostInputs(options,inputs->path().string(),{{"/__mdsl_private__/fixture.h","#pragma once\n"}},error);
  check(bool(capture),"capture "+error);if(!capture)return 1;
  clang::FileSystemOptions settings;settings.WorkingDir=inputs->path().string();
  auto files=llvm::makeIntrusiveRefCnt<clang::FileManager>(settings,capture->fileSystem());
  clang::tooling::ToolInvocation parse(capture->arguments(),std::make_unique<clang::SyntaxOnlyAction>(),files.get());
  check(parse.run(),"original host sema");
  auto snapshot=capture->freeze(error);check(bool(snapshot),"freeze "+error);if(!snapshot)return 1;
  std::string ir="stale";
  check(matcore::mdslc::codegen::detail::compileFrozenHostToLLVM(*snapshot,ir,error),"frozen codegen "+error);
  check(ir.find("call noundef i32 @_Z11must_remaini")!=std::string::npos,
        "always_inline call must remain before LLVM transformations");
  const auto ll=outputs->path()/"original.ll",obj=outputs->path()/"from_ir",direct=outputs->path()/"direct";
  write(ll,ir);
  auto a=run({options.clang_path,"-x","ir",ll.string(),"-o",obj.string()},inputs->path());
  auto b=run({options.clang_path,"-std=c++20",src.string(),"-o",direct.string()},inputs->path());
  check(a.launched&&a.exit_code==0,"IR link "+a.stderr_text);
  check(b.launched&&b.exit_code==0,"ordinary link "+b.stderr_text);
  auto x=run({obj.string()},inputs->path()),y=run({direct.string()},inputs->path());
  check(x.exit_code==0&&y.exit_code==0&&x.stdout_text==y.stdout_text&&x.stdout_text.starts_with("18 82 ")&&x.stdout_text.find("logical.mdsl")!=std::string::npos,
        "line directives, file/column builtins, function pointer, EH and RAII preserve ordinary output: "+x.stdout_text+" vs "+y.stdout_text);
  check(snapshot->unchanged(error),"outputs did not change input closure");
  write(src,"int main(){return 19;}\n");
  check(!matcore::mdslc::codegen::detail::compileFrozenHostToLLVM(*snapshot,ir,error)&&ir.empty(),"mutated main refuses publication and clears stale output");
  std::cout<<checks<<" independent frozen host checks, "<<bad<<" failures\n";
  return bad?1:0;
}
