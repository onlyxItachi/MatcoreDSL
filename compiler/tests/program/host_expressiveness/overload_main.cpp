#include "math_oracle.h"
int first(int);
int main() {
  exercise_regions();
  check(first(31) == 42);
  if (!failures) std::printf("PASS ordinary_overload %u checks\n", checks);
  return failures ? 1 : 0;
}
