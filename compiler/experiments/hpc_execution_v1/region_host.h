#pragma once
#include "harness.h"
namespace hpc {
const bool is_region = true;
const char *backend = "public_region_end_to_end";
Outcome invoke(float *a, float *b, float *c, unsigned long long m,
               unsigned long long n, unsigned long long k, int fault) {
  auto result = hpc_region({a, m + (fault == 1), k, m * k}, {b, k, n, k * n},
      {c, m, n, m * n, fault == 2 ? mdsl::Access::read_only : mdsl::Access::read_write}, m, n, k);
  const auto code = result.error();
  const auto error = code == mdsl::Error::ok ? Error::none :
      code == mdsl::Error::shape_mismatch ? Error::shape :
      code == mdsl::Error::access_denied ? Error::access :
      code == mdsl::Error::candidate_incompatible ? Error::incompatible : Error::other;
  return {result.ok(), error, result.failed_frontier(), result.completed_frontier(),
          result.completed_effect_frontier(), result.publication_count(), result.observation_count()};
}
}
int main(int argc, char **argv) { return hpc::main(argc, argv); }
