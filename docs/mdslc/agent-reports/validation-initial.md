# Initial validation and adversarial review

## Scope and baseline

- Role: validation and adversarial review (agent E).
- Worktree: `/home/hamza-usta/MatcoreDSL-wt-tests`.
- Branch: `mdslc/validation`.
- Original branch SHA: `351075e4d8af1880330b7c0474d701ca76776dfa`.
- Reviewed integration SHA: `f054c45` (`mdslc/bootstrap-v0`).
- Write ownership: `compiler/tests/integration/**`,
  `compiler/tests/fixtures/**`, and this report only.
- Reviewed milestone: standalone skeleton and valid-C++ Goal 2 driver. The
  extractor, rewrite, IR verifier, generated stubs, CPU runtime, packaging, and
  consumer project were not integrated and are not claimed as validated.

## Independent build and artifact evidence

The following clean out-of-tree build used the selected coherent executable:

```sh
cmake -S compiler -B /tmp/matcoredsl-validation-goal2.Rve2IL -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21
cmake --build /tmp/matcoredsl-validation-goal2.Rve2IL -- -j2
```

CMake identified Clang 21.1.8. Configuration and the two-step driver build
completed without Python, nanobind, LLVM libraries, MLIR, or legacy targets.

Independent executable and object checks passed:

```sh
/tmp/matcoredsl-validation-goal2.Rve2IL/bin/mdslc++ --verbose \
  -std=c++20 compiler/examples/hello_host.mdsl \
  -o /tmp/matcoredsl-validation-goal2.Rve2IL/hello_host
/tmp/matcoredsl-validation-goal2.Rve2IL/hello_host

/tmp/matcoredsl-validation-goal2.Rve2IL/bin/mdslc++ --verbose --save-temps \
  -std=c++20 -c compiler/examples/hello_host.mdsl \
  -o /tmp/matcoredsl-validation-goal2.Rve2IL/hello_host.o
file /tmp/matcoredsl-validation-goal2.Rve2IL/hello_host.o
readelf -h /tmp/matcoredsl-validation-goal2.Rve2IL/hello_host.o
nm -C /tmp/matcoredsl-validation-goal2.Rve2IL/hello_host.o
```

Evidence:

- verbose argv showed `'/usr/bin/clang++-21' ... '-x' 'c++'` before the
  `.mdsl` input;
- executable exit status was zero and stdout was exactly `5`;
- `file` reported ELF 64-bit x86-64 relocatable;
- `readelf` reported `Type: REL (Relocatable file)`;
- `nm -C` reported defined `main` and weak `host_add<int>`;
- `hello_host.ii`, `.bc`, `.s`, and `.o` were emitted;
- a missing `.mdsl` input returned exactly status 1 and preserved Clang's path
  and `no such file or directory` diagnostic;
- a source filename containing `;touch ...` compiled and ran while no marker
  appeared, confirming the current `fork`/`execv` argv path does not evaluate
  shell metacharacters.

The committed data-driven smoke run was:

```sh
python3 compiler/tests/integration/run_validation_matrix.py \
  --build-dir /tmp/matcoredsl-validation-goal2.Rve2IL
```

Result: 5 active passes, 0 failures, 39 explicitly pending cases, and 1 known
failure. Pending cases are not executed or counted as passes. Running with
`--include-known-failures` reproduced the known failure as one `XFAIL`.

## Findings

### Medium: `--` breaks mandatory language injection

Reproduction:

```sh
/tmp/matcoredsl-validation-goal2.Rve2IL/bin/mdslc++ --verbose \
  -std=c++20 \
  -o /tmp/matcoredsl-validation-goal2.Rve2IL/hello_after_double_dash \
  -- compiler/examples/hello_host.mdsl
```

Observed status: 1. The printed argv was conceptually:

```text
clang++ -std=c++20 -o ... -- -x c++ compiler/examples/hello_host.mdsl
```

Clang diagnosed both `-x` and `c++` as nonexistent files because the driver
inserted them after the compiler option terminator. This violates normal
argument forwarding and the rule that every `.mdsl` input be forced to C++.
The regression is `G02_DOUBLE_DASH` and remains `known_failure`; it must be
fixed before final acceptance.

### Residual argument-classification risk

`OptionConsumesNextArgument` is a manually maintained allow-list. A supported
Clang option omitted from that list can cause an option value ending in
`.mdsl` to be mistaken for an input. This is not a demonstrated failure in the
required Goal 2 command set, but later driver work should either narrow and
document the forwarded option grammar or add table-driven coverage for every
accepted separated-value option.

No high-severity shell-injection, artifact-shape, language-selection (without
`--`), exit-propagation, or diagnostic-loss finding was observed in Goals 1-2.

## Validation matrix policy

`validation_matrix.json` enumerates all 20 requested positive requirements and
all requested negative requirements (malformed and version-mismatched IR are
separate N20 variants). Representative source fixtures cover host-only C++,
canonical/aliased GEMM, free-function/class contexts, stable multi-site input,
and the initial forbidden recognition/output contexts.

Cases that require extraction, rewrite, runtime descriptors, packaging,
sanitizers, or an API not yet declared are marked `pending` with their future
milestone or fixture plan. `--require-all` intentionally returns nonzero while
any pending or known-failure case remains. Full execution and adversarial diff
review must be repeated after the CPU vertical slice is integrated.
