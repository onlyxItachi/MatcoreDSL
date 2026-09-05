#include "matcore/runtime_c.h"
#include "fp_environment_v1.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace platform = matcore::mdslc::platform;

namespace {
unsigned checks = 0;
unsigned failures = 0;

// Deliberately not assert(): these oracles must still run in Release/NDEBUG.
void expect(bool condition, std::string_view label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << label << '\n';
  }
}

matcore_tensor_desc_v0 descriptor(void *data, std::int64_t rows,
                                 std::int64_t columns, bool output = false) {
  matcore_tensor_desc_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.data = data;
  result.dtype = MATCORE_DTYPE_F32_V0;
  result.rank = 2;
  result.dims[0] = rows;
  result.dims[1] = columns;
  result.strides[0] = columns;
  result.strides[1] = 1;
  result.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  result.mutability = output ? MATCORE_MUTABILITY_READ_WRITE_V0
                             : MATCORE_MUTABILITY_READ_ONLY_V0;
  return result;
}

matcore_policy_v0 cpu_policy() {
  matcore_policy_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.target = MATCORE_TARGET_CPU_V0;
  result.fallback = MATCORE_FALLBACK_ERROR_V0;
  return result;
}

matcore_cpu_gemm_plan_report_v1 empty_report() {
  matcore_cpu_gemm_plan_report_v1 result{};
  result.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  result.struct_size = sizeof(result);
  return result;
}

struct Fixture {
  alignas(64) std::array<float, 16> a;
  alignas(64) std::array<float, 16> b;
  alignas(64) std::array<float, 16> c;
  matcore_tensor_desc_v0 lhs;
  matcore_tensor_desc_v0 rhs;
  matcore_tensor_desc_v0 out;
  matcore_policy_v0 policy = cpu_policy();

  Fixture() : lhs(descriptor(a.data(), 2, 2)),
              rhs(descriptor(b.data(), 2, 2)),
              out(descriptor(c.data(), 2, 2, true)) {
    a.fill(2.0F);
    b.fill(3.0F);
    c.fill(-17.0F);
  }
};

template <typename Mutation>
void plan_case(std::string_view predicate, matcore_status_code_v0 expected,
               Mutation mutation) {
  Fixture fixture;
  const auto a_before = fixture.a;
  const auto b_before = fixture.b;
  const auto c_before = fixture.c;
  mutation(fixture);
  auto report = empty_report();
  std::array<unsigned char, sizeof(report)> report_before{};
  std::memcpy(report_before.data(), &report, sizeof(report));
  // The real existing C ABI shares validate_gemm_v0 with one-shot execution.
  // It does not execute GEMM, probe OpenBLAS, or inspect the FP environment.
  const auto result = matcore_runtime_plan_gemm_f32_v1(
      &fixture.out, &fixture.lhs, &fixture.rhs, &fixture.policy, &report);
  expect(result.code == expected, predicate);
  expect(fixture.a == a_before && fixture.b == b_before && fixture.c == c_before,
         std::string(predicate) + ": planner preserves all tensor canaries");
  if (expected != MATCORE_STATUS_OK_V0)
    expect(std::memcmp(&report, report_before.data(), sizeof(report)) == 0,
           std::string(predicate) + ": failure preserves report");
}

void descriptor_predicate_oracles() {
  plan_case("matching positive descriptors", MATCORE_STATUS_OK_V0,
            [](Fixture &) {});
  for (unsigned role = 0; role != 3; ++role) {
    const std::string suffix = " role=" + std::to_string(role);
    plan_case("data_nonnull" + suffix, MATCORE_STATUS_INVALID_ARGUMENT_V0,
              [role](Fixture &f) {
                std::array tensors{&f.out, &f.lhs, &f.rhs};
                tensors[role]->data = nullptr;
              });
    for (unsigned axis = 0; axis != 2; ++axis)
      for (const std::int64_t value : {0, -1})
        plan_case("positive_dimensions" + suffix + " axis=" + std::to_string(axis) +
                      " value=" + std::to_string(value),
                  MATCORE_STATUS_INVALID_SHAPE_V0, [=](Fixture &f) {
                    std::array tensors{&f.out, &f.lhs, &f.rhs};
                    tensors[role]->dims[axis] = value;
                  });
    plan_case("pointer_alignment_required" + suffix,
              MATCORE_STATUS_INVALID_ALIGNMENT_V0, [role](Fixture &f) {
                std::array tensors{&f.out, &f.lhs, &f.rhs};
                auto *bytes = static_cast<unsigned char *>(tensors[role]->data);
                tensors[role]->data = bytes + 1; // Live object byte, never dereferenced as float.
              });
  }
  plan_case("contraction_dimension_equal", MATCORE_STATUS_SHAPE_MISMATCH_V0,
            [](Fixture &f) { f.rhs.dims[0] = 3; });
  plan_case("output_rows_equal", MATCORE_STATUS_SHAPE_MISMATCH_V0,
            [](Fixture &f) { f.out.dims[0] = 3; });
  plan_case("output_columns_equal", MATCORE_STATUS_SHAPE_MISMATCH_V0,
            [](Fixture &f) { f.out.dims[1] = f.out.strides[0] = 3; });
  plan_case("output_input_no_overlap output/lhs", MATCORE_STATUS_ALIAS_VIOLATION_V0,
            [](Fixture &f) { f.out.data = f.a.data(); });
  plan_case("output_input_no_overlap output/rhs partial", MATCORE_STATUS_ALIAS_VIOLATION_V0,
            [](Fixture &f) { f.out.data = f.b.data() + 1; });
  plan_case("output_input_no_overlap permits equal input descriptors", MATCORE_STATUS_OK_V0,
            [](Fixture &f) { f.rhs.data = f.a.data(); });
  plan_case("output_input_no_overlap permits partial input overlap", MATCORE_STATUS_OK_V0,
            [](Fixture &f) { f.rhs.data = f.a.data() + 1; });

  // These descriptors always contain real live object pointers. Their claimed
  // extents are intentionally invalid; no pointer arithmetic/dereference is
  // performed beyond the real allocations by this test or the plan API.
  plan_case("byte_range_representable element-product overflow",
            MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0, [](Fixture &f) {
              constexpr auto huge = std::numeric_limits<std::int64_t>::max();
              for (auto *tensor : {&f.out, &f.lhs, &f.rhs}) {
                tensor->dims[0] = tensor->dims[1] = huge;
                tensor->strides[0] = huge;
              }
            });
  plan_case("byte_range_representable byte-product overflow",
            MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0, [](Fixture &f) {
              f.out.dims[0] = f.lhs.dims[0] = std::numeric_limits<std::int64_t>::max();
            });
  plan_case("byte_range_representable address-plus-extent overflow",
            MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0, [](Fixture &f) {
              const auto address = reinterpret_cast<std::uintptr_t>(f.out.data);
              const auto remaining = std::numeric_limits<std::uintptr_t>::max() - address;
              const auto rows = static_cast<std::int64_t>(remaining / sizeof(float) + 1U);
              f.out = descriptor(f.c.data(), rows, 1, true);
              f.lhs = descriptor(f.a.data(), rows, 1);
              f.rhs = descriptor(f.b.data(), 1, 1);
            });
}

void defensive_abi_oracles() {
  // Wrong C-ABI records exercise existing defensive checks. They are not
  // asserted to be reachable from the authenticated source adapter.
  plan_case("source_tensor_contract ABI defense: rank", MATCORE_STATUS_UNSUPPORTED_RANK_V0,
            [](Fixture &f) { f.lhs.rank = 1; });
  plan_case("source_tensor_contract ABI defense: dtype", MATCORE_STATUS_UNSUPPORTED_DTYPE_V0,
            [](Fixture &f) { f.lhs.dtype = MATCORE_DTYPE_F64_V0; });
  plan_case("source_layout_contract ABI defense", MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0,
            [](Fixture &f) { f.rhs.strides[1] = 2; });
  plan_case("source_host_designation ABI defense: mixed", MATCORE_STATUS_MIXED_MEMORY_SPACES_V0,
            [](Fixture &f) { f.rhs.memory_space = MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0; });
  plan_case("source_host_designation ABI defense: nonhost", MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0,
            [](Fixture &f) {
              f.out.memory_space = f.lhs.memory_space = f.rhs.memory_space =
                  MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0;
            });
  plan_case("source_access_designation ABI defense", MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0,
            [](Fixture &f) { f.out.mutability = MATCORE_MUTABILITY_READ_ONLY_V0; });
  plan_case("source_policy_intent ABI defense: target", MATCORE_STATUS_UNSUPPORTED_TARGET_V0,
            [](Fixture &f) { f.policy.target = MATCORE_TARGET_CUDA_V0; });
  plan_case("source_policy_intent ABI defense: fallback", MATCORE_STATUS_UNSUPPORTED_FALLBACK_V0,
            [](Fixture &f) { f.policy.fallback = MATCORE_FALLBACK_ALLOW_V0; });
  plan_case("adapter ABI header defense", MATCORE_STATUS_ABI_MISMATCH_V0,
            [](Fixture &f) { f.out.struct_size = 0; });
  plan_case("adapter ABI reserved defense", MATCORE_STATUS_INVALID_ARGUMENT_V0,
            [](Fixture &f) { f.policy.reserved[0] = 1; });
  plan_case("first failure: policy precedes descriptors", MATCORE_STATUS_UNSUPPORTED_TARGET_V0,
            [](Fixture &f) { f.policy.target = MATCORE_TARGET_CUDA_V0; f.out.data = nullptr; });
  plan_case("first failure: complete output validation precedes lhs", MATCORE_STATUS_UNSUPPORTED_RANK_V0,
            [](Fixture &f) { f.out.rank = 1; f.lhs.data = nullptr; });
}

void capacity_remains_a_caller_obligation() {
  // Each actual float array has capacity ONE. Alignment gives the distinct
  // objects nonoverlapping 64-byte claimed intervals, but does not enlarge the
  // arrays. A 4x4 descriptor is therefore unsafe to execute even if plan accepts.
  // NEVER pass these descriptors to any execution entry point.
  alignas(64) std::array<float, 1> a{2.0F};
  alignas(64) std::array<float, 1> b{3.0F};
  alignas(64) std::array<float, 1> c{-17.0F};
  const auto lhs = descriptor(a.data(), 4, 4);
  const auto rhs = descriptor(b.data(), 4, 4);
  const auto out = descriptor(c.data(), 4, 4, true);
  const auto policy = cpu_policy();
  auto report = empty_report();
  expect(matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &policy, &report).code ==
             MATCORE_STATUS_OK_V0,
         "backing_capacity_sufficient remains unproved after successful plan");
  expect(a[0] == 2.0F && b[0] == 3.0F && c[0] == -17.0F,
         "undersized capacity inspection left actual elements unchanged");
  // descriptor_object_valid, backing_host_accessible, backing_lifetime_valid,
  // backing_access_permitted and no_conflicting_concurrent_access likewise
  // remain caller obligations. Do not manufacture dangling or inaccessible
  // references to 'test' these by dereferencing invalid objects.
}

void synthetic_fp_oracles() {
  constexpr std::uint32_t mxcsr = 0x1f80U;
  constexpr std::uint16_t x87 = 0x037fU;
  const auto compatible = [](std::uint32_t simd, std::uint16_t control) {
    return platform::decode_linux_x86_fp_environment_v1(simd, control)
        .explicit_gemm_f32_v1_compatible;
  };
  expect(compatible(mxcsr, x87), "floating_environment_compatible baseline decoder");
  expect(!compatible(mxcsr | (1U << 15U), x87), "floating_environment_compatible rejects FTZ");
  expect(!compatible(mxcsr | (1U << 6U), x87), "floating_environment_compatible rejects DAZ");
  for (unsigned rounding = 1; rounding != 4; ++rounding) {
    expect(!compatible(mxcsr | (rounding << 13U), x87),
           "floating_environment_compatible rejects non-nearest SIMD rounding");
    expect(!compatible(mxcsr, static_cast<std::uint16_t>(x87 | (rounding << 10U))),
           "floating_environment_compatible rejects non-nearest x87 rounding");
  }
  for (unsigned exception = 0; exception != 6; ++exception) {
    expect(!compatible(mxcsr & ~(1U << (exception + 7U)), x87),
           "floating_environment_compatible rejects unmasked SIMD exception");
    expect(!compatible(mxcsr, static_cast<std::uint16_t>(x87 & ~(1U << exception))),
           "floating_environment_compatible rejects unmasked x87 exception");
  }
  for (std::uint32_t status = 0; status != 64; ++status)
    expect(compatible(mxcsr | status, x87),
           "floating_environment_compatible does not require cleared sticky status flags");
  for (unsigned precision = 0; precision != 4; ++precision)
    expect(compatible(mxcsr, static_cast<std::uint16_t>((x87 & ~0x0300U) | (precision << 8U))),
           "floating_environment_compatible does not invent an x87 precision predicate");
  const auto current = platform::inspect_current_fp_environment_v1();
  expect(current.version == platform::kFpEnvironmentVersionV1,
         "current floating environment report has its existing version");
  if (!current.discovery_complete)
    expect(!current.explicit_gemm_f32_v1_compatible,
           "unknown floating environment cannot discharge compatibility");
  expect((std::string_view(platform::fp_environment_rejection_reason_v1(current)) == "ok") ==
             current.explicit_gemm_f32_v1_compatible,
         "current floating environment diagnostic agrees with compatibility");
}

void planner_does_not_discharge_fp_or_execute() {
#if defined(__linux__) && defined(__x86_64__)
  const auto saved = _mm_getcsr();
  {
    struct RestoreMxcsr {
      std::uint32_t saved;
      ~RestoreMxcsr() { _mm_setcsr(saved); }
    } restore{saved};
    _mm_setcsr(saved | (1U << 15U));
    expect(!platform::inspect_current_fp_environment_v1().explicit_gemm_f32_v1_compatible,
           "floating_environment_compatible observes actual FTZ");
    plan_case("plan success does not discharge floating_environment_compatible",
              MATCORE_STATUS_OK_V0, [](Fixture &) {});
    Fixture fixture;
    const auto c_before = fixture.c;
    // This is the ONLY execution-API invocation in this file. A null descriptor
    // must fail before descriptor reads, provider discovery or GEMM execution.
    expect(matcore_runtime_gemm_f32_v0(nullptr, &fixture.lhs, &fixture.rhs,
                                      &fixture.policy).code == MATCORE_STATUS_INVALID_ARGUMENT_V0,
           "descriptor validation retains priority over actual incompatible FP state");
    expect(fixture.c == c_before, "pre-FP invalid descriptor leaves output canaries unchanged");
  }
  expect(_mm_getcsr() == saved, "actual FP test restores original MXCSR exactly");
#else
  std::cout << "SKIP: actual FTZ manipulation is Linux x86-64 only; synthetic FP oracle ran\n";
#endif
}
} // namespace

int main() {
  descriptor_predicate_oracles();
  defensive_abi_oracles();
  capacity_remains_a_caller_obligation();
  synthetic_fp_oracles();
  planner_does_not_discharge_fp_or_execute();
  std::cout << "region guard runtime/FP oracle: " << checks << " checks, " << failures
            << " failures; no GEMM executed\n";
  return failures == 0 ? 0 : 1;
}
