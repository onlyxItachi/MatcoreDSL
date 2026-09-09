#include "math_oracle.h"
int first_host_value();
int second_host_value();
int main() {
  check(first_host_value() == 31); check(second_host_value() == 47);
  exercise_regions();
  if (!failures) std::printf("PASS ordinary_internal_linkage %u checks\n", checks);
  return failures ? 1 : 0;
}
