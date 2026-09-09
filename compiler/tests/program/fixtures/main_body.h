#pragma once
#include <cstdio>
static unsigned checks = 0, failures = 0;
static void check(bool good) {
  ++checks;
  if (!good) { ++failures; std::fprintf(stderr, "FAIL multi-TU check %u\n", checks); }
}
int main() {
  check(region_unit_identity() == 37);
  float a[] = {1, 2, 3, 4}, b[] = {0, 1, 2, 0}, c[4] = {}, d[4] = {};
  const float ab[] = {4, 1, 8, 3}, aba[] = {7, 12, 17, 28};
  const md::Storage A{a, 2, 2, 4, md::Access::read_only};
  const md::Storage B{b, 2, 2, 4, md::Access::read_only};
  const md::Storage C{c, 2, 2, 4, md::Access::read_write};
  const md::Storage D{d, 2, 2, 4, md::Access::read_write};
  auto one = first(A, B, C, 2, 2, 2);
  check(one.ok());
  check(one.publication_count() == 1 && one.observation_count() == 1);
  auto old = one.observation(0);
  check(old.valid());
  for (unsigned i = 0; i < 4; ++i) {
    check(c[i] == ab[i]);
    if (old.valid()) check(old.data()[i] == ab[i]);
  }
  auto two = second(C, A, D, 2, 2, 2);
  check(two.ok());
  check(two.publication_count() == 1 && two.observation_count() == 1);
  auto newer = two.observation(0);
  check(newer.valid());
  for (unsigned i = 0; i < 4; ++i) {
    check(d[i] == aba[i]);
    if (newer.valid()) check(newer.data()[i] == aba[i]);
    c[i] = d[i] = -91;
  }
  for (unsigned i = 0; i < 4; ++i) {
    if (old.valid()) check(old.data()[i] == ab[i]);
    if (newer.valid()) check(newer.data()[i] == aba[i]);
  }
  auto bad = second(C, A, D, 2, 3, 2);
  check(!bad && bad.error() == md::Error::shape_mismatch);
  check(bad.publication_count() == 0 && bad.observation_count() == 0);
  for (unsigned i = 0; i < 4; ++i)
    if (old.valid()) check(old.data()[i] == ab[i]);
  if (!failures) std::printf("PASS authenticated_multi_tu %u checks\n", checks);
  return failures ? 1 : 0;
}
