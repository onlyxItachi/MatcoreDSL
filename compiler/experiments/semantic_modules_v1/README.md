# Semantic module feasibility controls

Research-only; see the [evidence and decision](../../../docs/mdslc/agent-reports/semantic-modules-feasibility-v1.md).
No production changes, import syntax, alternate C++ parser or interpreter.

Use an existing coherent **Release, tests-ON** Ninja build with
`matcore_experimental_region_admission_tests` configured. The runner reads its
link command inventory, then compiles/links just this small probe in a fresh
output directory. It never invokes a Ninja/CMake build or overwrites old outputs.
This experiment currently selects the repository's Clang 21 executable path;
it is not a generic cross-platform build harness.

```sh
python3 compiler/experiments/semantic_modules_v1/run.py \
  --build /home/hamza-usta/MatcoreDSL-wt-private-candidate-dso-v1/build-isolated-release \
  --mlir-prefix /home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21 \
  --out /tmp/mdslc-semantic-modules-new-run \
  --has-openblas
```

Omit `--has-openblas` to omit the two provider-specific driver lanes. Doing so
does **not** change the existing build's provider configuration or prove a
provider-OFF build. No tests-disabled, sanitizer or timing mode is implemented.
The output directory must not already exist; it contains generated fixtures,
objects, executables, MLIR bytecode and complete command/output/hash evidence.
Do not commit these generated outputs.

`probe.cpp` invokes the existing compiler's deterministic noninstalled
parse/freeze test callback, mutating only files in its own scratch directory.
Its seven well-formed refusal fixtures remain distinct from positive helper
admission, source/witness pairing and live versus historical closure checks.
The public host uses a small exact-integer double oracle, preserving intermediate
f32 rounding; wrong math must fail, strict OpenBLAS must refuse without effects.
The MLIR fixtures test only upstream symbols, inlining, type verification and
bytecode round-trip—not Matcore module execution authority.
