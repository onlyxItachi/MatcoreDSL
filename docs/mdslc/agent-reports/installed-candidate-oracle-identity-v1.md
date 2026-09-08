# Installed candidate oracle identity and CMake policy fix

## Observed counterexample

The exact `c3ffafb556d7a693430e2648088b922b8fa18ac3` hosted
[Release provider-ON job](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34243275177/job/102118643505)
reported CMP0054 unset at the install script's `elseif(test STREQUAL
"candidates")`. The existing `candidates` variable held the installed DSO path;
OLD policy dereferenced that quoted comparison operand. The loop therefore
selected the private-Value fallback while keeping the candidates label:

```text
Installed candidates: Independent private Value: 85 checks, 32000 ownership cycles, 0 failures
```

This is an actual hosted skipped-oracle counterexample, not a compiler numerical
failure. A successful process status and the caller-generated label did not
prove that `candidate_test.cpp` was executed. The failed hosted installation
record remains valid historical evidence; this change does not turn that run
into a success or repair its separate installed-driver provider-link failure.

## Scoped test-only change

- The installer script now starts with `cmake_minimum_required(VERSION 3.24)`,
  setting the intended NEW policies when invoked directly through `cmake -P`.
- After each of the result/candidates/private_value processes exits zero, an
  independent identity check matches its complete stripped output to the
  expected suite marker, a positive check count, and zero failures. The mapping
  is keyed by the requested label, not by the source-selection branch. Thus the
  wrong source cannot also change the expected identity and disguise the skip.
- The helper accepts the candidate marker for both provider-ON and provider-OFF
  counts without freezing the count against future legitimate added checks.
  It does not accept zero checks, failed checks, another suite's marker, extra
  output summaries, or unknown labels.
- `PROVIDER` is forwarded unchanged to the installed `driver_contract.cmake`
  child, as requested by the integration owner. Production provider linking and
  the child's ELF provider checks remain owned by that separate fix.

## Bounded validation on 2026-09-08

The independent worktree
`/home/hamza-usta/MatcoreDSL-wt-installed-candidate-oracle-identity-v1` is based on
`c3ffafb556d7a693430e2648088b922b8fa18ac3`, branch
`test/installed-candidate-oracle-identity-v1`.

The only available local CMake is 4.3.2. Its `--help-policy CMP0054` documents
that OLD behavior was removed in CMake 4.0; no older tool was installed or
downloaded. The tiny regression therefore demonstrates the corrected actual
NEW-policy branch and explicitly emulates the documented OLD quoted-operand
lookup. It does not claim a local CMake 3.28 execution. The real hosted warning
and mislabeled output above provide the actual OLD-policy evidence.

The integration owner approved read-only execution of three existing installed
consumers from the stable prefix
`/home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release/experimental-region-package-7cee80301558`.
No rebuild or reinstall was performed:

```sh
cmake \
  -DINSTALLED_PREFIX=/home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release/experimental-region-package-7cee80301558 \
  -P compiler/tests/cmake/experimental_regions_install_identity_test.cmake
```

Observed results:

```text
NEW-policy installer dispatch selects the candidate oracle
Documented OLD lookup emulation selects the private_value fallback
Four positive output identities and six rejection controls passed
Verified existing installed result: Experimental owning Result: 37 checks, 0 failures
Verified existing installed candidates: 151358 candidate checks, 0 failures
Verified existing installed private_value: Independent private Value: 85 checks, 32000 ownership cycles, 0 failures
```

The exact existing executable SHA256 values were:

| Executable | SHA256 |
| --- | --- |
| result | `83f3450819fa6f40d783c45752d2b8946b1c9c74226b2d70112436b23649463a` |
| candidates | `15faa2e43d9fb65e3067c1f8571be84d5a7173dcd1d387781cabe4c4890897da` |
| private_value | `55df519d76937f548f8fb8a86ac5033fbdf274e83ee66b275d41d25f00276925` |

The standalone regression does not require installed executables when
`INSTALLED_PREFIX` is omitted. It is not separately registered in CTest, so it
does not change the integration owner's test-count contracts. The identity
guard itself runs in every feature-ON install-contract test.

This bounded validation proves the corrected dispatch and identity guard and
checks real existing consumer outputs. It is not a fresh full install-contract
run against the production provider fix. In particular, archive-consumer linking
with a nonstandard provider path remains an explicit integration check now that
the candidate oracle is no longer silently skipped. No root worktree writes,
compiler/runtime edits, full builds, package/toolchain changes, or merges were
performed by this lane.
