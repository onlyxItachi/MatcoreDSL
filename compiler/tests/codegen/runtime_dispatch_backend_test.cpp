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
  std::cout << "Runtime-dispatch backend tests: 11 checks, 0 failures\n";
  return 0;
}
