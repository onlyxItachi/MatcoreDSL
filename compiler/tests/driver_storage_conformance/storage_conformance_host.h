// Ordinary C++ callers. This oracle shares no compiler-private runtime code.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {
using Shape = mdsl::Shape;
using Matrix = std::vector<float>;
unsigned checks = 0, failures = 0;
const char *current_case = nullptr;
bool expect_policy_rejection = false;
void check(bool condition, const char *message) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL %s/%s: %s\n", lhs_carry ? "lhs" : "rhs",
                 current_case, message);
  }
}
bool exact(float a, float b) {
  std::uint32_t x = 0, y = 0;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}
Matrix snapshot(const std::array<float, 192> &arena, Shape offset, Shape size) {
  return Matrix(arena.begin() + offset, arena.begin() + offset + size);
}
Matrix multiply(const Matrix &a, const Matrix &b, Shape m, Shape n, Shape k) {
  Matrix out(m * n, 0.0f);
  for (Shape i = 0; i < m; ++i)
    for (Shape j = 0; j < n; ++j) {
      double total = 0;
      for (Shape t = 0; t < k; ++t)
        total += static_cast<double>(a[i * k + t]) * b[t * n + j];
      out[i * n + j] = static_cast<float>(total);
    }
  return out;
}
void observation(const mdsl::Observation &handle, const Matrix &expected,
                 Shape rows, Shape columns) {
  check(handle.valid(), "observation remains valid");
  check(handle.rows() == rows && handle.columns() == columns,
        "observation retains exact rectangular dimensions");
  if (!handle.valid() || !handle.data()) {
    check(expected.empty(), "nonempty observation has data");
    return;
  }
  for (Shape i = 0; i < expected.size(); ++i)
    check(exact(handle.data()[i], expected[i]), "observation exact f32 snapshot");
}
enum class Alias { none, late_exact, late_partial, publish_over_a,
                   publish_over_b, outputs_overlap, all_same };
enum class Failure { none, late_descriptor_shape, late_capacity,
                     second_gemm_shape, first_read_shape,
                     first_publish_access, second_publish_shape,
                     second_publish_access };
struct Case {
  const char *name;
  Shape m = 2, n = 3, k = 4, p = 2;
  Alias alias = Alias::none;
  Failure failure = Failure::none;
};

void run(const Case &test) {
  current_case = test.name;
  std::array<float, 192> arena{};
  for (Shape i = 0; i < arena.size(); ++i)
    arena[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 13) - 6);
  auto expected_arena = arena;
  Shape a_offset = 0, b_offset = 24, c_offset = 48, d_offset = 72, e_offset = 96;
  switch (test.alias) {
  case Alias::none: break;
  case Alias::late_exact: d_offset = c_offset; break;
  case Alias::late_partial: d_offset = c_offset + 2; break;
  case Alias::publish_over_a: c_offset = a_offset; break;
  case Alias::publish_over_b: c_offset = b_offset; break;
  case Alias::outputs_overlap: e_offset = c_offset + 1; break;
  case Alias::all_same: a_offset = b_offset = c_offset = d_offset = e_offset = 0; break;
  }
  const auto m = test.m, n = test.n, k = test.k, p = test.p;
  const Shape q = (lhs_carry ? n : m) +
                 (test.failure == Failure::second_gemm_shape ? 1 : 0);
  const Shape dr = lhs_carry ? q : p, dc = lhs_carry ? p : q;
  const Shape er = lhs_carry ? m : p, ec = lhs_carry ? p : n;
  auto storage = [&](Shape offset, Shape rows, Shape columns) {
    // Even mutable input descriptors are allowed to alias. Null empty buffers
    // exercise public descriptor semantics without manufacturing invalid data.
    return mdsl::Storage{rows * columns ? arena.data() + offset : nullptr,
                         rows, columns, rows * columns, mdsl::Access::read_write};
  };
  auto A = storage(a_offset, m, k), B = storage(b_offset, k, n);
  auto C = storage(c_offset, m, n), D = storage(d_offset, dr, dc);
  auto E = storage(e_offset, er, ec);
  Shape expected_failed = 0, expected_completed = 10, expected_effect = 9;
  Shape publications = 2, observations = 2;
  auto expected_error = mdsl::Error::ok;
  unsigned expected_line = 0;
  switch (test.failure) {
  case Failure::none: break;
  case Failure::late_descriptor_shape:
    ++D.rows;
    D.capacity_elements = D.rows * D.columns;
    expected_failed = 6; expected_completed = 5; expected_effect = 5;
    publications = observations = 1;
    expected_error = mdsl::Error::shape_mismatch; expected_line = 14; break;
  case Failure::late_capacity:
    D.capacity_elements = 0;
    expected_failed = 6; expected_completed = 5; expected_effect = 5;
    publications = observations = 1;
    expected_error = mdsl::Error::insufficient_capacity; expected_line = 14; break;
  case Failure::second_gemm_shape:
    expected_failed = 7; expected_completed = 6; expected_effect = 5;
    publications = observations = 1;
    expected_error = mdsl::Error::shape_mismatch; expected_line = 15; break;
  case Failure::first_read_shape:
    ++A.rows;
    A.capacity_elements = A.rows * A.columns;
    expected_failed = 1; expected_completed = expected_effect = 0;
    publications = observations = 0;
    expected_error = mdsl::Error::shape_mismatch; expected_line = 9; break;
  case Failure::first_publish_access:
    C.access = mdsl::Access::read_only;
    expected_failed = 4; expected_completed = 3; expected_effect = 0;
    publications = observations = 0;
    expected_error = mdsl::Error::access_denied; expected_line = 12; break;
  case Failure::second_publish_shape:
    ++E.rows;
    E.capacity_elements = E.rows * E.columns;
    expected_failed = 8; expected_completed = 7; expected_effect = 5;
    publications = observations = 1;
    expected_error = mdsl::Error::shape_mismatch; expected_line = 16; break;
  case Failure::second_publish_access:
    E.access = mdsl::Access::read_only;
    expected_failed = 8; expected_completed = 7; expected_effect = 5;
    publications = observations = 1;
    expected_error = mdsl::Error::access_denied; expected_line = 16; break;
  }
  if (expect_policy_rejection && test.failure != Failure::first_read_shape) {
    // existing-native is not silently allowed to weaken strict_f32, including
    // empty and zero-reduction mathematical cases.
    expected_failed = 3; expected_completed = 2; expected_effect = 0;
    publications = observations = 0;
    expected_error = mdsl::Error::candidate_incompatible; expected_line = 11;
  }
  const auto a = snapshot(expected_arena, a_offset, m * k);
  const auto b = snapshot(expected_arena, b_offset, k * n);
  const auto c = multiply(a, b, m, n, k);
  if (std::strcmp(test.name, "rectangular") == 0)
    check(c == Matrix({0, -24, 4, 64, -21, 63}), "independent first-GEMM literal oracle");
  if (publications >= 1)
    std::copy(c.begin(), c.end(), expected_arena.begin() + c_offset);
  // This read is intentionally AFTER the first publication in the oracle.
  const auto d = snapshot(expected_arena, d_offset, dr * dc);
  Matrix e;
  if (publications >= 2) {
    e = lhs_carry ? multiply(c, d, m, p, n) : multiply(d, c, p, n, m);
    if (std::strcmp(test.name, "rectangular") == 0)
      check(e == (lhs_carry ? Matrix({104, -36, -531, 211})
                            : Matrix({64, 123, 39, 128, 78, 106})),
            "independent noncommuting second-GEMM literal oracle");
    if (test.alias == Alias::late_exact || test.alias == Alias::late_partial) {
      const auto stale_d = snapshot(arena, d_offset, dr * dc);
      const auto hoisted_e = lhs_carry ? multiply(c, stale_d, m, p, n)
                                     : multiply(stale_d, c, p, n, m);
      check(e != hoisted_e, "alias fixture distinguishes a wrongly hoisted late read");
    }
    std::copy(e.begin(), e.end(), expected_arena.begin() + e_offset);
  }
  mdsl::Observation first, second;
  {
    auto function = &pipeline;
    auto result = function(A, B, D, C, E, m, n, k, p, q);
    check(result.ok() == (expected_error == mdsl::Error::ok), "checked success/failure");
    check(result.error() == expected_error, "exact error category");
    check(result.publication_count() == publications, "exact publication prefix");
    check(result.observation_count() == observations, "exact observation prefix");
    check(result.failed_frontier() == expected_failed, "failed operation frontier");
    check(result.completed_frontier() == expected_completed, "completed operation frontier");
    check(result.completed_effect_frontier() == expected_effect, "completed effect frontier");
    const auto location = result.failure_location();
    check(location.line == expected_line, "failure names original operation line");
    check((location.file != nullptr) == (expected_error != mdsl::Error::ok), "failure filename presence");
    if (location.file) {
      check(std::strstr(location.file, lhs_carry ? "lhs.mdsl" : "rhs.mdsl") != nullptr,
            "failure names original public source");
      check(location.column != 0, "failure source column is nonzero");
    }
    check(!result.observation(observations).valid(), "out of range observation is invalid");
    if (observations >= 1) first = result.observation(0);
    if (observations >= 2) second = result.observation(1);
    auto moved = std::move(result);
    check(!result.ok() && result.observation_count() == 0, "moved-from result is empty");
    check(moved.publication_count() == publications, "move retains exact status");
    std::printf("%s %s error=%u failed=%llu completed=%llu effect=%llu publications=%llu observations=%llu\n",
                lhs_carry ? "lhs" : "rhs", test.name, static_cast<unsigned>(moved.error()),
                moved.failed_frontier(), moved.completed_frontier(),
                moved.completed_effect_frontier(), moved.publication_count(), moved.observation_count());
  }
  // Full-arena equality also detects writes beyond the retired publication
  // extents, writes to a failed destination, and changes to unrelated storage.
  for (Shape i = 0; i < arena.size(); ++i)
    check(exact(arena[i], expected_arena[i]), "entire mutable arena matches effect-prefix oracle");
  std::fill(arena.begin(), arena.end(), 9999.0f);
  if (observations >= 1) {
    observation(first, c, m, n);
    auto copy = first;
    mdsl::Observation moved = std::move(copy);
    first = {};
    check(!copy.valid(), "moved-from observation is invalid");
    observation(moved, c, m, n);
  }
  if (observations >= 2) observation(second, e, er, ec);
}
} // namespace

int main(int argc, char **argv) {
  expect_policy_rejection = argc == 2 && std::strcmp(argv[1], "--expect-policy-rejection") == 0;
  const Case cases[]{
    {"rectangular"},
    {"late_read_exact_alias", 2, 3, 4, 2, Alias::late_exact},
    {"late_read_partial_alias", 2, 3, 4, 2, Alias::late_partial},
    {"first_publish_over_a", 2, 3, 4, 2, Alias::publish_over_a},
    {"first_publish_over_b", 2, 3, 4, 2, Alias::publish_over_b},
    {"overlapping_outputs", 2, 3, 4, 2, Alias::outputs_overlap},
    {"all_mutable_resources_same_pointer", 2, 3, 4, 2, Alias::all_same},
    {"empty_m", 0, 3, 4, 2},
    {"empty_n", 2, 0, 4, 2},
    {"zero_first_reduction", 2, 3, 0, 2},
    {"empty_second_output_p", 2, 3, 4, 0},
    {"all_zero_dimensions", 0, 0, 0, 0},
    {"late_descriptor_shape", 2, 3, 4, 2, Alias::none, Failure::late_descriptor_shape},
    {"late_capacity", 2, 3, 4, 2, Alias::none, Failure::late_capacity},
    {"dynamic_second_gemm_shape", 2, 3, 4, 2, Alias::none, Failure::second_gemm_shape},
    {"first_read_shape", 2, 3, 4, 2, Alias::none, Failure::first_read_shape},
    {"first_publish_read_only", 2, 3, 4, 2, Alias::none, Failure::first_publish_access},
    {"second_publish_shape", 2, 3, 4, 2, Alias::none, Failure::second_publish_shape},
    {"second_publish_read_only", 2, 3, 4, 2, Alias::none, Failure::second_publish_access},
    {"late_shape_overlapping_outputs", 2, 3, 4, 2, Alias::outputs_overlap, Failure::late_descriptor_shape},
    {"dynamic_shape_all_same_pointer", 2, 3, 4, 2, Alias::all_same, Failure::second_gemm_shape},
    {"second_publish_shape_all_same_pointer", 2, 3, 4, 2, Alias::all_same, Failure::second_publish_shape},
  };
  for (const auto &test : cases) run(test);
  std::printf("%s %u checks, %u failures, %zu cases\n", lhs_carry ? "lhs" : "rhs",
              checks, failures, std::size(cases));
  return failures ? 1 : 0;
}
