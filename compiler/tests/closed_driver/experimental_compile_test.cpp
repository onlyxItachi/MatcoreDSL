#include "ExperimentalRegionCompiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "../../lib/support/platform_support.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fe = matcore::mdslc::frontend;
namespace cg = matcore::mdslc::codegen;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
namespace {
unsigned checks = 0, failures = 0;
void check(bool okay, const std::string &why) {
  ++checks;
  if (!okay) { ++failures; std::cerr << "FAIL: " << why << '\n'; }
}
void write(const fs::path &path, const std::string &bytes) {
  std::ofstream output(path, std::ios::binary);
  output << bytes;
  output.close();
  if (!output) throw std::runtime_error("fixture output failed");
}
support::ProcessResultV1 run(std::vector<std::string> args, const fs::path &cwd) {
  support::ProcessRequestV1 request;
  request.argv = std::move(args);
  request.working_directory = cwd;
  request.environment = support::compiler_environment_sanitization_v1();
  return support::run_process_v1(request);
}
const char *source = R"cpp(#include <matcore/region.h>
#include <cstdio>
namespace mdsl = matcore::mdsl;
namespace example {
mdsl::Value product(mdsl::Value a, mdsl::Value b) noexcept {
  return mdsl::gemm(a, b, mdsl::Numerics::strict_f32);
}
MATCORE_REGION
mdsl::Result pipeline(mdsl::Storage a, mdsl::Storage b, mdsl::Storage d,
                      mdsl::Storage c, mdsl::Storage e,
                      mdsl::Shape m, mdsl::Shape n, mdsl::Shape k) noexcept {
  const auto av = mdsl::read(a, m, k);
  const auto bv = mdsl::read(b, k, n);
  const auto cv = product(av, bv);
  mdsl::publish(cv, c);
  mdsl::observe(c);
  const auto dv = mdsl::read(d, m, m);
  const auto ev = product(dv, cv);
  mdsl::publish(ev, e);
  mdsl::observe(e);
  return mdsl::complete();
}
}
int host_effects = 0;
struct Host { Host(){ ++host_effects; } ~Host(){ ++host_effects; } };
int main(int argc, char**) {
  float a[6]{1,2,3,4,5,6}, b[6]{1,2,3,4,5,6}, d[4]{1,2,0,1};
  float c[4]{}, e[4]{};
  const auto ro = mdsl::Access::read_only, rw = mdsl::Access::read_write;
  auto call = &example::pipeline;
  { Host host;
    auto result = call({a,2,3,6,ro},{b,3,2,6,ro},{d,2,2,4,ro},
                       {c,2,2,4,rw},{e,2,2,4,rw},2,2,3);
    if (!result || result.publication_count()!=2 || result.observation_count()!=2) return 10;
    const float expected_c[4]{22,28,49,64}, expected_e[4]{120,156,49,64};
    for (int i=0;i<4;++i)
      if(c[i]!=expected_c[i] || e[i]!=expected_e[i]) return 11;
    auto observed = result.observation(0);
    c[0]=999;
    if(!observed.valid() || observed.data()[0]!=22 || result.failure_location().file) return 12;
  }
  e[0]=-99;
  auto failed = example::pipeline({a,2,3,6,ro},{b,3,2,6,ro},{d,1,2,4,ro},
                                  {c,2,2,4,rw},{e,2,2,4,rw},2,2,3);
  if(failed || failed.error()!=mdsl::Error::shape_mismatch ||
     failed.publication_count()!=1 || failed.observation_count()!=1 ||
     c[0]!=22 || e[0]!=-99 || !failed.failure_location().file ||
     !failed.failure_location().line || !failed.failure_location().column) return 13;
  if(argc>1) { volatile int out_of_bounds=9; float *pointer=a; return int(pointer[out_of_bounds]); }
#if __has_feature(address_sanitizer)
  constexpr int instrumented_host=1;
#else
  constexpr int instrumented_host=0;
#endif
  std::printf("rectangular rhs carry; ordered prefix; owning observation; host=%d san=%d\n",
              host_effects, instrumented_host);
  return host_effects==2 ? 0 : 14;
}
)cpp";
} // namespace

int main(int argc, char **argv) {
  if (argc != 6 && argc != 7) return 2;
  const bool sanitized = argc == 7 && std::string(argv[6]) == "--asan";
  std::string error;
  auto sources = support::create_temp_directory_v1("mdslc-public-input", error);
  auto staging = support::create_temp_directory_v1("mdslc-public-helper", error);
  auto substituted = support::create_temp_directory_v1("mdslc-public-substituted", error);
  auto shadowed = support::create_temp_directory_v1("mdslc-public-shadowed", error);
  auto outputs = support::create_temp_directory_v1("mdslc-public-output", error);
  if (!sources || !staging || !substituted || !shadowed || !outputs) throw std::runtime_error(error);
  const fs::path compiler = argv[3];
  const auto input = sources->path() / "program.mdsl";
  write(input, source);
  fe::Options options;
  options.input_path = input.string();
  options.clang_path = argv[1];
  options.clang_resource_directory = argv[2];
  options.compiler_arguments = {"-I" + (compiler / "include").string()};
  if (sanitized) options.compiler_arguments.push_back("-fsanitize=address,undefined");
  const fe::ExperimentalRegionHeaders headers{
      (compiler / "include/matcore/region.h").string(),
      (compiler / "include/matcore/detail/region_storage.h").string()};
  auto admitted = fe::admitExperimentalRegionHost(options, sources->path().string(),
                                                  headers, "example::pipeline");
  check(bool(admitted) && admitted.syntax_valid, "real public source admission: " + admitted.error);
  if (!admitted) return 1;
  auto runtime_artifact = llvm::MemoryBuffer::getFile(argv[5]);
  if (!runtime_artifact) throw std::runtime_error("missing trusted test runtime artifact");
  cg::ExperimentalCompilerInputs inputs{argv[1], argv[2], compiler / "include",
      compiler / "lib/runtime/closed_host_v1.h", staging->path(), sanitized,
      {{cg::SymbolArtifactOwner::MatcoreRuntime, (*runtime_artifact)->getMemBufferRef()}}};
  auto changed_inputs = inputs;
  changed_inputs.staging_directory = substituted->path();
  auto substitution = cg::compileExperimentalRegionToLLVMForTesting(*admitted.evidence,
      changed_inputs, cg::ClosedCpuPolicy::GeneratedStrict, [&] {
        const auto helper = changed_inputs.staging_directory / "region-helper.cpp";
        std::ifstream input_file(helper, std::ios::binary);
        std::string replacement((std::istreambuf_iterator<char>(input_file)), {});
        const auto at = replacement.find("Candidate::generated_strict");
        if (at == std::string::npos) throw std::runtime_error("missing issued candidate selection");
        replacement.replace(at, std::string("Candidate::generated_strict").size(),
                            "Candidate::native_strict");
        write(helper, replacement);
      });
  check(!substitution && substitution.error.find("differs from compiler-issued bytes") != std::string::npos,
        "a self-consistent staged helper cannot substitute its candidate policy");
  auto shadow_inputs = inputs;
  shadow_inputs.staging_directory = shadowed->path();
  auto shadow = cg::compileExperimentalRegionToLLVMForTesting(*admitted.evidence,
      shadow_inputs, cg::ClosedCpuPolicy::GeneratedStrict, [&] {
        write(shadow_inputs.staging_directory / "closed_host_v1.h",
              "#error mutable staging header must not define the runtime contract\n");
      });
  check(bool(shadow), "compiler-owned VFS headers exclude mutable helper-directory shadows: " + shadow.error);
  auto compiled = cg::compileExperimentalRegionToLLVM(*admitted.evidence, inputs,
                                                     cg::ClosedCpuPolicy::GeneratedStrict);
  check(bool(compiled), "connected original-host/helper ABI compilation: " + compiled.error);
  if (!compiled) return 1;
  check(compiled.compilation->llvm_ir.find(compiled.compilation->emission.host_symbol) != std::string::npos,
        "original callable host symbol retained");
  if (sanitized)
    check(compiled.compilation->llvm_ir.find("sanitize_address") != std::string::npos,
          "actual Clang-generated functions carry address instrumentation intent");
  const auto llvm_file = outputs->path() / "program.ll";
  const auto executable = outputs->path() / "program";
  write(llvm_file, compiled.compilation->llvm_ir);
  std::vector<std::string> args{argv[1], "-x", "ir", llvm_file.string(), "-x", "none",
      argv[4], argv[5], "-pthread", "-Wl,-rpath," + fs::path(argv[5]).parent_path().string(),
      "-o", executable.string()};
  if (sanitized) args.push_back("-fsanitize=address,undefined");
  auto link = run(args, outputs->path());
  check(link.launched && link.exit_code == 0, "ordinary final link with actual registry: " + link.stderr_text);
  if (link.exit_code != 0) return 1;
  check(compiled.compilation->inputsUnchanged(error), "input closure unchanged after final link: " + error);
  const auto execution = run({executable.string()}, outputs->path());
  const std::string expected = "rectangular rhs carry; ordered prefix; owning observation; host=2 san=" +
                               std::string(sanitized ? "1\n" : "0\n");
  check(execution.launched && execution.exit_code == 0 && execution.stdout_text == expected,
        "actual generated region with ordinary direct/function-pointer host calls: " + execution.stderr_text);
  if (sanitized) {
    const auto negative = run({executable.string(), "negative-control"}, outputs->path());
    check(negative.exit_code != 0 && negative.stderr_text.find("AddressSanitizer") != std::string::npos,
          "positive control proves actual original-host memory access is instrumented");
  }
  write(input, "int main(){return 23;}\n");
  check(!compiled.compilation->inputsUnchanged(error), "changed original host blocks output publication");
  std::cout << checks << " public connected compiler checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
