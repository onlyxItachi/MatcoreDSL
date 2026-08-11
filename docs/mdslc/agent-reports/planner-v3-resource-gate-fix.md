# Planner v3 resource gate fix

## Scope

This lane investigated the exact-head Release failure in
`runtime.cpu.planner_v3_resources`. It changed only the focused runtime resource
test; planner selection, runtime validation, numerical semantics, and production
execution behavior are unchanged.

## Root cause

The failure was deterministic, not a host transient. Runtime numerical
conformance work added by `715936e` expanded each available parallel ISA probe
into four persistent-context submissions: worker floating-point-state capture,
finite GEMM, special-value GEMM, and floating-point-state verification and
restoration. The test still counted one submission per parallel ISA. It also
described OpenBLAS execution as conditional even though all six serial variants
are submitted unconditionally and availability is evaluated on the bound
worker.

## Correction

The exact dispatch oracle now requires:

```text
6 + 4 * avx2_runtime_usable + 4 * avx512_runtime_usable
```

The numerical-evidence assertion is separate from the dispatch-count assertion,
and OpenBLAS evidence is compared with the immutable provider conformance result
rather than linkage alone. This preserves fail-closed behavior and makes a
future numerical failure distinct from a worker-dispatch regression.

## Validation

On exact base `6b828c9a733bba4d89d6adb7dec6a7d0b284d290`, using the
Release/OpenBLAS/MLIR build that originally failed, the two focused gates were
rebuilt with low-priority two-way parallelism and then each executed ten
consecutive times:

```sh
nice -n 10 cmake --build <r1-build> \
  --target matcore_planner_v3_resources_test \
           matcore_cpu_runtime_validation_test --parallel 2
ctest --test-dir <r1-build> \
  -R '^runtime\.cpu\.(planner_v3_resources|variant_conformance\.v1)$' \
  --output-on-failure --repeat until-fail:10 -j1
```

```text
runtime.cpu.planner_v3_resources: 10/10 passed
runtime.cpu.variant_conformance.v1: 10/10 passed
```

The unchanged conformance gate confirms that the corrected dispatch oracle did
not weaken numerical, floating-point-environment, provider, or ISA validation.

Independent adversarial rereview accepted the change with no high or medium
finding. One low, pre-existing coverage limitation remains: the exact bound
worker accounting block is skipped when affinity discovery is incomplete or
the process has fewer than two allowed logical CPUs. The broader portable
validation fixture still runs in that environment; this patch does not broaden
the platform contract.
