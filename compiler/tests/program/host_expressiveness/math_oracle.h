#pragma once
#include "../fixtures/api.h"
#include <cstdio>
static unsigned checks = 0, failures = 0;
static void check(bool value) { ++checks; if (!value) ++failures; }
static void exercise_regions() {
  float a[] = {1, 2, 3, 4}, b[] = {0, 1, 2, 0}, c[4] = {}, d[4] = {};
  const float ab[] = {4, 1, 8, 3}, aba[] = {7, 12, 17, 28};
  const md::Storage A{a, 2, 2, 4, md::Access::read_only};
  const md::Storage B{b, 2, 2, 4, md::Access::read_only};
  const md::Storage C{c, 2, 2, 4, md::Access::read_write};
  const md::Storage D{d, 2, 2, 4, md::Access::read_write};
  const auto one = first(A, B, C, 2, 2, 2);
  const auto two = second(C, A, D, 2, 2, 2);
  check(one.ok()); check(two.ok());
  for (unsigned i = 0; i < 4; ++i) { check(c[i] == ab[i]); check(d[i] == aba[i]); }
}
