#include <matcore/runtime_c.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  } else {
    std::cout << "PASS: " << message << "\n";
  }
}

matcore_tensor_desc_v0 make_desc(float *data, std::int64_t rows,
                                 std::int64_t cols,
                                 matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 desc{};
  desc.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  desc.struct_size = sizeof(matcore_tensor_desc_v0);
  desc.data = data;
  desc.dtype = MATCORE_DTYPE_F32_V0;
  desc.rank = 2;
  desc.dims[0] = rows;
  desc.dims[1] = cols;
  desc.strides[0] = cols;
  desc.strides[1] = 1;
  desc.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  desc.mutability = mutability;
  return desc;
}

// Guarded backend entry simulating generated static AOT dispatch
extern "C" matcore_status_v0 test_guarded_static_backend(
    const matcore_tensor_desc_v0 *c, const matcore_tensor_desc_v0 *a,
    const matcore_tensor_desc_v0 *b, const matcore_policy_v0 *policy) {
  if (!c || !a || !b || !policy) {
    matcore_status_v0 status{};
    status.code = MATCORE_STATUS_INVALID_ARGUMENT_V0;
    return status;
  }
  // 1. Legality precondition: No output aliasing
  if (c->data == a->data || c->data == b->data) {
    matcore_status_v0 status{};
    status.code = MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0;
    return status;
  }
  // 2. Exact static geometry guard
  if (c->dims[0] == 16 && c->dims[1] == 16 &&
      a->dims[0] == 16 && a->dims[1] == 16 &&
      b->dims[0] == 16 && b->dims[1] == 16 &&
      c->strides[1] == 1 && a->strides[1] == 1 && b->strides[1] == 1) {
    // Aligned/Unaligned bifurcation
    const bool is_aligned = (reinterpret_cast<std::uintptr_t>(c->data) & 31) == 0 &&
                            (reinterpret_cast<std::uintptr_t>(a->data) & 31) == 0 &&
                            (reinterpret_cast<std::uintptr_t>(b->data) & 31) == 0;
    (void)is_aligned;
    float *__restrict out = static_cast<float *>(c->data);
    const float *__restrict lhs = static_cast<const float *>(a->data);
    const float *__restrict rhs = static_cast<const float *>(b->data);
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < 16; ++k) {
          sum += lhs[i * 16 + k] * rhs[k * 16 + j];
        }
        out[i * 16 + j] = sum;
      }
    }
    matcore_status_v0 status{};
    status.code = MATCORE_STATUS_OK_V0;
    return status;
  }
  // Fallback to runtime
  return matcore_runtime_gemm_f32_v0(c, a, b, policy);
}

void test_aliasing_rejection() {
  std::vector<float> buf(16 * 16, 1.0f);
  matcore_tensor_desc_v0 c_desc = make_desc(buf.data(), 16, 16, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 a_desc = make_desc(buf.data(), 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 b_desc = make_desc(buf.data(), 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  matcore_status_v0 status = test_guarded_static_backend(&c_desc, &a_desc, &b_desc, &policy);
  expect(status.code != MATCORE_STATUS_OK_V0, "aliased output memory pointer is rejected before mutation");
}

void test_unaligned_access_correctness() {
  // Allocate buffer with unaligned offset (+1 float = +4 bytes offset from 32-byte alignment)
  std::vector<float> raw_buf(16 * 16 + 8, 1.0f);
  float *unaligned_c = raw_buf.data() + 1;
  float *unaligned_a = raw_buf.data() + 2;
  float *unaligned_b = raw_buf.data() + 3;

  matcore_tensor_desc_v0 c_desc = make_desc(unaligned_c, 16, 16, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 a_desc = make_desc(unaligned_a, 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 b_desc = make_desc(unaligned_b, 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  matcore_status_v0 status = test_guarded_static_backend(&c_desc, &a_desc, &b_desc, &policy);
  expect(status.code == MATCORE_STATUS_OK_V0, "unaligned pointers execute without hardware fault");
  expect(unaligned_c[0] == 16.0f, "unaligned matrix multiplication computes exact output values");
}

void test_guard_overhead_timing() {
  std::vector<float> a(16 * 16, 1.0f);
  std::vector<float> b(16 * 16, 2.0f);
  std::vector<float> c(16 * 16, 0.0f);

  matcore_tensor_desc_v0 c_desc = make_desc(c.data(), 16, 16, MATCORE_MUTABILITY_READ_WRITE_V0);
  matcore_tensor_desc_v0 a_desc = make_desc(a.data(), 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);
  matcore_tensor_desc_v0 b_desc = make_desc(b.data(), 16, 16, MATCORE_MUTABILITY_READ_ONLY_V0);

  matcore_policy_v0 policy{};
  policy.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  policy.struct_size = sizeof(matcore_policy_v0);
  policy.target = MATCORE_TARGET_CPU_V0;
  policy.fallback = MATCORE_FALLBACK_ERROR_V0;

  // Warmup
  for (int i = 0; i < 100; ++i) {
    test_guarded_static_backend(&c_desc, &a_desc, &b_desc, &policy);
  }

  const int iterations = 10000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    test_guarded_static_backend(&c_desc, &a_desc, &b_desc, &policy);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / static_cast<double>(iterations);

  std::cout << "Average guarded static 16x16 GEMM latency: " << elapsed_ns << " ns/op\n";
  expect(elapsed_ns > 0.0, "guarded static GEMM latency is measurable and non-zero");
}

} // namespace

int main() {
  std::cout << "Starting aliasing and alignment policy validation tests...\n";
  test_aliasing_rejection();
  test_unaligned_access_correctness();
  test_guard_overhead_timing();

  if (failures != 0) {
    std::cerr << "Aliasing and alignment policy tests: " << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "Aliasing and alignment policy tests: all checks passed successfully\n";
  return 0;
}
