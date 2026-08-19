#include "codegen.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

} // namespace

int main() {
  using matcore::mdslc::codegen::RuntimeDispatchBackendEntryV1;
  using matcore::mdslc::codegen::RuntimeDispatchBackendProducerV1;
  using matcore::mdslc::codegen::generateRuntimeDispatchBackendV1;

  const std::string site = "mc_0123456789abcdef0123456789abcdef";
  std::string source;
  std::string error;
  expect(generateRuntimeDispatchBackendV1(
             {RuntimeDispatchBackendEntryV1{site}},
             RuntimeDispatchBackendProducerV1::CaptureV0, source, error),
         "capture-v0 backend generation succeeds");
  expect(error.empty(), "successful generation clears the error");
  expect(source.find("Producer: Matcore MLIR") == std::string::npos,
         "capture-v0 bytes do not acquire a semantic producer marker");
  expect(source.find("matcore_generated_backend_" + site + "_v0") !=
             std::string::npos,
         "generated backend uses the canonical site symbol");
  expect(source.find("matcore_runtime_gemm_f32_v0") != std::string::npos,
         "generated backend retains the stable runtime dispatch boundary");

  expect(generateRuntimeDispatchBackendV1(
             {RuntimeDispatchBackendEntryV1{site}},
             RuntimeDispatchBackendProducerV1::MatcoreMlirCpuV1, source,
             error),
         "Matcore MLIR backend generation succeeds");
  expect(source.find(
             "// Producer: Matcore MLIR CPU runtime-dispatch lowering v1.") !=
             std::string::npos,
         "semantic backend identifies its authenticated producer");

  // Test static AOT specialization generation
  RuntimeDispatchBackendEntryV1 static_entry{
      .site_id = site,
      .static_m = 16,
      .static_n = 64,
      .static_k = 32,
      .alignment = 32,
      .no_alias = true,
  };
  expect(generateRuntimeDispatchBackendV1(
             {static_entry},
             RuntimeDispatchBackendProducerV1::MatcoreMlirCpuV1, source,
             error),
         "Matcore MLIR static specialized backend generation succeeds");
  expect(source.find("direct_static_microkernel_" + site) != std::string::npos,
         "static specialized backend emits direct in-register microkernel");
  expect(source.find("output->dims[0] == 16") != std::string::npos,
         "static specialized backend includes shape guard check");
  expect(source.find("matcore_runtime_gemm_f32_v0") != std::string::npos,
         "static specialized backend retains runtime fallback path");

  // Test degenerate M=1, N=1 dot product specialization
  RuntimeDispatchBackendEntryV1 dot_entry{
      .site_id = site,
      .static_m = 1,
      .static_n = 1,
      .static_k = 128,
  };
  expect(generateRuntimeDispatchBackendV1(
             {dot_entry},
             RuntimeDispatchBackendProducerV1::MatcoreMlirCpuV1, source,
             error),
         "dot product static specialization succeeds");
  expect(source.find("for (std::int64_t p = 0; p < 128; ++p)") != std::string::npos,
         "dot product specialization generates 1D reduction loop");

  source = "sentinel";
  error = "stale";
  expect(!generateRuntimeDispatchBackendV1(
             {RuntimeDispatchBackendEntryV1{"not-a-site"}},
             RuntimeDispatchBackendProducerV1::MatcoreMlirCpuV1, source,
             error),
         "noncanonical site is rejected");
  expect(source.empty() && !error.empty(),
         "rejected generation publishes no partial backend source");

  source = "sentinel";
  expect(!generateRuntimeDispatchBackendV1(
             {RuntimeDispatchBackendEntryV1{site},
              RuntimeDispatchBackendEntryV1{site}},
             RuntimeDispatchBackendProducerV1::MatcoreMlirCpuV1, source,
             error),
         "duplicate semantic backend site is rejected");
  expect(source.empty(), "duplicate rejection is transactional");

  if (failures != 0) {
    std::cerr << "Runtime-dispatch backend tests: " << failures
              << " failure(s)\n";
    return 1;
  }
  std::cout << "Runtime-dispatch backend tests: 17 checks, 0 failures\n";
  return 0;
}
