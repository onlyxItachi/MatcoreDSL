#include "math_oracle.h"
int host_helper(int);
int main() {
  check(host_helper(5) == 14);
  exercise_regions();
  if (!failures) std::printf("PASS ordinary_host_utility %u checks\n", checks);
  return failures ? 1 : 0;
}
