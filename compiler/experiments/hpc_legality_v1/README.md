# Independent HPC legality fixtures v1

These are bounded compiler/candidate contract experiments, not a benchmark or a
new test framework. They introduce no production hooks or execution authority.
The initial base is canonical `0e8c040809ae1fe6b0f1c01e41fc509b07212be7`.

## Files and expected outcomes

| Fixture | Independent oracle |
| --- | --- |
| `strict_counterexamples.cpp` | Eight exact checks distinguish separate f32 multiply/add from FMA, increasing K from split partial sums, and two GEMM rounding boundaries from cross-GEMM reassociation. |
| [`schedule_source_contract.mdsl`](../../tests/generated_cpu/schedule_source_contract.mdsl) | 4,140 checks through one real admitted region: strict arithmetic in scalar tails and full output blocks, lane-distinct rectangular geometry, special values, two GEMMs, old immutable values, late partial-alias reads, owning observations and earlier effect prefixes after late read/publication failures. |
| [`schedule_same_value_contract.mdsl`](../../tests/generated_cpu/schedule_same_value_contract.mdsl) | 1,503 checks pass exactly the same immutable Value to both GEMM operands. Destination isolation must not become an unsupported claim that the inputs are mutually disjoint. |

Both `.mdsl` fixtures compare every non-NaN output bit against an independently
written increasing-K f32 oracle. NaN classification is checked without requiring
a particular payload. Volatile f32 intermediates preserve separate multiply/add
in the oracle. Source fixtures use only the existing public region API.

The numerical cases deliberately include full 5x33 output geometry, not only
N=1: a broken SIMD main body can coexist with a correct scalar remainder.
Finite lane-distinct data use non-symmetric shapes around 4/8/16/32-column tails.
The two-GEMM fixture exercises current semantics; it does not implement or claim
a whole-region transformation or private-storage reuse.

## Run

From the repository root, use a fresh ignored output directory and the exact
build/install driver under review. The driver does not overwrite existing output.

```sh
mkdir -p build-hpc-legality-v1
/usr/bin/clang++-21 -std=c++20 -O2 -ffp-contract=off \
  compiler/experiments/hpc_legality_v1/strict_counterexamples.cpp \
  -o build-hpc-legality-v1/strict_counterexamples
build-hpc-legality-v1/strict_counterexamples

MDSLC_HPC_REGION_DRIVER=/absolute/path/to/reviewed/bin/mdslc-region
"$MDSLC_HPC_REGION_DRIVER" compiler/tests/generated_cpu/schedule_source_contract.mdsl \
  --region contract_region --candidate generated-strict \
  -o build-hpc-legality-v1/source_contract
build-hpc-legality-v1/source_contract
"$MDSLC_HPC_REGION_DRIVER" compiler/tests/generated_cpu/schedule_same_value_contract.mdsl \
  --region same_value_region --candidate generated-strict \
  -o build-hpc-legality-v1/same_value_contract
build-hpc-legality-v1/same_value_contract
```

Repeat both source fixtures with the affected ASan+UBSan **driver/build**, using
different output filenames. Linking an ASan runtime is not evidence that the
generated leaf is instrumented: retain the existing deliberate generated OOB
negative control and add a full/tail-vector OOB control when that form is issued.
These fixtures alone are not a proof of absence of vector overreads.

The integration owner may register
[`schedule_source_contract.cmake`](../../tests/generated_cpu/schedule_source_contract.cmake)
with `-DDRIVER=/absolute/reviewed/mdslc-region` and `-DCASE=source` or
`-DCASE=same_value`. It compiles the exact adjacent source, requires exit zero
and the exact successful check count, and retains failed artifacts. This lane
does not modify the owning `CMakeLists.txt` registration.

## Baseline versus transformed evidence

Initial experiments used the already-built unchanged scalar generated candidate
from implementation head `b0e7ac9ac67c3c314a44e5db108be390485404eb` (normally
merged before the canonical base). That establishes executable oracle readiness,
not successful transformed candidate integration. Exact driver hashes, observed
results and required adversarial stage mutations are in the
[independent report](../../../docs/mdslc/agent-reports/mlir-hpc-legality-v1.md).
