// Research host for MLIR/SPIRV-Cross-emitted, completely reviewed MSL goldens.
// Never links Matcore runtime, publishes caller storage or authorizes a target.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

struct Case {
  const char *name;
  std::array<float, 6> a;
  std::array<float, 12> b;
};

static float f(std::uint32_t bits) { return std::bit_cast<float>(bits); }
static std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

// Volatile separates rounding points; host compile uses no-fast-math/contract off.
static float reference(const Case &c, unsigned i, unsigned j) {
  volatile float sum = 0.0f;
  for (unsigned k = 0; k != 3; ++k) {
    volatile float product = c.a[i * 3 + k] * c.b[k * 4 + j];
    sum = sum + product;
  }
  return sum;
}

static float wrongOrderControl(const Case &c) {
  volatile float sum = 0.0f;
  for (unsigned k : {0u, 2u, 1u}) {
    volatile float product = c.a[k] * c.b[k * 4];
    sum = sum + product;
  }
  return sum;
}

static Case dotCase(const char *name, std::array<float, 3> a,
                    std::array<float, 3> b) {
  Case result{name, {}, {}};
  for (unsigned i = 0; i != 2; ++i)
    for (unsigned k = 0; k != 3; ++k) result.a[i * 3 + k] = a[k];
  for (unsigned k = 0; k != 3; ++k)
    for (unsigned j = 0; j != 4; ++j) result.b[k * 4 + j] = b[k];
  return result;
}

static bool save(NSMutableDictionary *report, NSString *path) {
  NSError *error = nil;
  NSData *json = [NSJSONSerialization dataWithJSONObject:report
      options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
  if (!json || ![json writeToFile:path options:NSDataWritingAtomic error:&error]) {
    fprintf(stderr, "could not write evidence: %s\n", error.description.UTF8String);
    return false;
  }
  puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
  return true;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 4) return 2; // fill.metal gemm.metal evidence.json
    NSString *path = @(argv[3]);
    NSMutableDictionary *report = [@{
      @"classification": @"research-only; no MDSLC execution authority",
      @"strict_contract_qualified": @NO,
      @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
      @"arithmetic_flags": @"fastMathEnabled=NO; pragma METAL fp contract(off)",
      @"msl_language_version": @"2.1 (explicit for offline and runtime compilation)",
      @"geometry": @"M=2 N=4 K=3; 1 thread/group; 2x4 groups",
      @"storage": @"explicit shared A/B/private-result staging; host access after completion"
    } mutableCopy];
    auto fail = [&](NSString *phase, NSError *error) {
      report[@"status"] = @"PROBE_ERROR";
      report[@"phase"] = phase;
      report[@"error"] = error ? error.description : @"API returned nil/invalid state";
      save(report, path);
      return 1;
    };
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    NSMutableArray *names = [NSMutableArray array];
    for (id<MTLDevice> found in devices) [names addObject:found.name];
    report[@"enumerated_devices"] = names;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      report[@"status"] = @"SKIP_NO_METAL_DEVICE";
      return save(report, path) ? 77 : 1;
    }
    report[@"device"] = device.name;
    report[@"registry_id"] = @(device.registryID);
    report[@"unified_memory"] = @(device.hasUnifiedMemory);
    // This probe only implements Apple Silicon-style shared CPU/GPU storage.
    if (!device.hasUnifiedMemory) {
      report[@"status"] = @"SKIP_UNSUPPORTED_STORAGE_MODEL";
      return save(report, path) ? 77 : 1;
    }
    if (fesetround(FE_TONEAREST) != 0 || fegetround() != FE_TONEAREST)
      return fail(@"host oracle rounding", nil);
    std::vector<Case> cases;
    cases.push_back({"rectangular", {1, -2, 3, -4, 5, 6},
                                    {2, 3, -4, 5, 6, 7, 8, -9, 10, -11, 12, 13}});
    cases.push_back(dotCase("subnormal-input", {f(1), 0, 0}, {1, 1, 1}));
    cases.push_back(dotCase("subnormal-result", {f(0x00800000), 0, 0}, {0.5f, 1, 1}));
    cases.push_back(dotCase("negative-subnormal", {f(0x80000001), 0, 0}, {1, 1, 1}));
    cases.push_back(dotCase("signed-zero", {-0.0f, -0.0f, -0.0f}, {1, 1, 1}));
    cases.push_back(dotCase("fma-discriminator", {-1, f(0x3f800001), 0},
                                                 {1, f(0x3f7ffffe), 1}));
    cases.push_back(dotCase("increasing-k", {16777216.0f, 1, -16777216.0f}, {1, 1, 1}));
    cases.push_back(dotCase("rte-not-rtz", {1, f(0x33c00000), 0}, {1, 1, 1}));
    cases.push_back(dotCase("infinity", {INFINITY, 1, 1}, {1, 1, 1}));
    cases.push_back(dotCase("quiet-nan", {f(0x7fc00001), 1, 1}, {1, 1, 1}));
    if (bits(reference(cases[1], 0, 0)) != 1 ||
        bits(reference(cases[2], 0, 0)) != 0x00400000 ||
        bits(reference(cases[3], 0, 0)) != 0x80000001 ||
        bits(reference(cases[4], 0, 0)) != 0 ||
        bits(reference(cases[5], 0, 0)) != 0 ||
        std::fma(f(0x3f800001), f(0x3f7ffffe), -1.0f) == 0 ||
        reference(cases[6], 0, 0) != 0 ||
        wrongOrderControl(cases[6]) == reference(cases[6], 0, 0) ||
        bits(reference(cases[7], 0, 0)) != 0x3f800001)
      return fail(@"host oracle adversarial controls", nil);
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion2_1;
    options.fastMathEnabled = NO;
    auto pipeline = [&](const char *file) -> id<MTLComputePipelineState> {
      NSString *text = [NSString stringWithContentsOfFile:@(file)
          encoding:NSUTF8StringEncoding error:&error];
      if (!text) return nil;
      text = [@"#pragma METAL fp contract(off)\n" stringByAppendingString:text];
      id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
      if (!library) return nil;
      id<MTLFunction> fn = [library newFunctionWithName:@"metal_static_specimen_kernel"];
      if (!fn) return nil;
      return [device newComputePipelineStateWithFunction:fn error:&error];
    };
    id<MTLComputePipelineState> fill = pipeline(argv[1]);
    if (!fill) return fail(@"compile MLIR-derived fill MSL", error);
    id<MTLComputePipelineState> gemm = pipeline(argv[2]);
    if (!gemm) return fail(@"compile MLIR-derived GEMM MSL", error);
    report[@"runtime_shader_compilation"] = @"PASS";
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return fail(@"command queue", nil);
    NSMutableArray *results = [NSMutableArray array];
    unsigned comparisons = 0, mismatches = 0;
    for (const Case &c : cases) {
      id<MTLBuffer> a = [device newBufferWithBytes:c.a.data() length:sizeof(c.a)
          options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [device newBufferWithBytes:c.b.data() length:sizeof(c.b)
          options:MTLResourceStorageModeShared];
      // 256-byte aligned output offset avoids device binding alignment guesses.
      std::array<float, 73> sentinel;
      sentinel.fill(f(0x42f60000));
      id<MTLBuffer> output = [device newBufferWithBytes:sentinel.data() length:sizeof(sentinel)
          options:MTLResourceStorageModeShared];
      if (!a || !b || !output) return fail(@"shared staging allocation", nil);
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (!command) return fail(@"command buffer", nil);
      id<MTLComputeCommandEncoder> init = [command computeCommandEncoder];
      if (!init) return fail(@"fill encoder", nil);
      [init setComputePipelineState:fill];
      [init setBuffer:output offset:256 atIndex:0];
      [init dispatchThreadgroups:MTLSizeMake(2, 4, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [init endEncoding];
      id<MTLComputeCommandEncoder> multiply = [command computeCommandEncoder];
      if (!multiply) return fail(@"GEMM encoder", nil);
      [multiply setComputePipelineState:gemm];
      [multiply setBuffer:a offset:0 atIndex:0];
      [multiply setBuffer:b offset:0 atIndex:1];
      [multiply setBuffer:output offset:256 atIndex:2];
      [multiply dispatchThreadgroups:MTLSizeMake(2, 4, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [multiply endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        return fail(@"checked GPU completion", command.error);
      const float *actual = static_cast<const float *>(output.contents);
      for (unsigned i = 0; i != 73; ++i)
        if ((i < 64 || i >= 72) && bits(actual[i]) != 0x42f60000)
          return fail(@"output guard overwritten", nil);
      if (std::memcmp(a.contents, c.a.data(), sizeof(c.a)) ||
          std::memcmp(b.contents, c.b.data(), sizeof(c.b)))
        return fail(@"input unexpectedly modified", nil);
      NSMutableArray *want = [NSMutableArray array], *got = [NSMutableArray array];
      unsigned failed = 0;
      for (unsigned i = 0; i != 2; ++i) for (unsigned j = 0; j != 4; ++j) {
        float expected = reference(c, i, j), observed = actual[64 + i * 4 + j];
        [want addObject:@(bits(expected))]; [got addObject:@(bits(observed))];
        bool equal = std::isnan(expected) ? std::isnan(observed) : bits(expected) == bits(observed);
        ++comparisons;
        if (!equal) { ++failed; ++mismatches; }
      }
      [results addObject:@{@"case": @(c.name), @"expected_bits": want,
                          @"actual_bits": got, @"mismatches": @(failed)}];
    }
    report[@"results"] = results;
    report[@"comparisons"] = @(comparisons);
    report[@"mismatches"] = @(mismatches);
    report[@"status"] = mismatches ? @"EXECUTED_STRICT_COUNTEREXAMPLE" : @"EXECUTED_BOUNDED_SAMPLE_MATCH";
    // Successful observation is not production qualification, even if all match.
    return save(report, path) ? 0 : 1;
  }
}
