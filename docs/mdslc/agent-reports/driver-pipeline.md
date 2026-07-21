# Driver Pipeline Agent Report

## Scope and ownership

- Worktree: `/home/hamza-usta/MatcoreDSL-wt-pipeline`
- Branch: `mdslc/driver-pipeline-v0`
- Integrated starting SHA: `06e99b0593793b5fd83ccb441305eb72c45cddf3`
- Owned files: `compiler/CMakeLists.txt`, `compiler/cmake/**`,
  `compiler/tools/mdslc/**`, `compiler/examples/gemm_v0.mdsl`, and this report.
- No public header, frontend, IR, codegen, runtime, legacy, or integration
  branch source was edited as part of the driver changes.

The code-generation handoff
`b290ca05c744a5fd762861c5a43f374b89c91061` was cherry-picked into this
worktree as `1617ace` solely to perform the end-to-end validation. The
integration owner already owns the original handoff; it is not a new driver
change to cherry-pick.

## Build wiring

The standalone top-level CMake project now builds the existing IR, frontend,
codegen, extractor, runtime, runtime test, and driver targets. Executables and
libraries are placed in `bin/` and `lib/`. `mdsl.h` and `runtime_c.h` are
mirrored with `configure_file(COPYONLY)` into `include/matcore/`, giving the
driver a relocatable build-tree layout. Python is optional and used only to
register the existing test scripts when an interpreter is available; no
compiler or runtime target links or executes Python.

## Driver pipeline

The old direct Clang path remains the default for host-only `.mdsl` sources.
Only explicit `--matcore-target=cpu` selects the new one-source bootstrap path:

```text
input.mdsl
  -> matcore-extract (IR plus all-or-none generated outputs)
  -> input.host.cpp + input.matcore.json + input.sites.h
  -> input.stubs.cpp + input.backend.cpp
  -> three independent clang++ compilations
  -> clang++ -r combined relocatable object, or normal final link
  -> relative libmatcore_runtime with a build-tree rpath
```

The extractor, public include directory, and runtime library directory are
resolved relative to the running `mdslc++` binary. Commands are executed only
with `fork`/`execv` argv vectors. No shell or Python process is involved.

`--save-temps` uses deterministic names next to the requested output. Normal
runs use `mkdtemp` and remove only that exact temporary directory through a
scope guard. Unsupported targets fail explicitly and never fall back. The v0
pipeline requires one `.mdsl` input and an explicit `-o`, rejects conflicting
compile modes, checks the extractor's complete output contract, and refuses to
overwrite its input source.

## Reproducible validation

Fresh configure, build, and test:

```text
cmake -S compiler -B /tmp/matcore-pipeline-clean -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21
cmake --build /tmp/matcore-pipeline-clean -- -j2
ctest --test-dir /tmp/matcore-pipeline-clean --output-on-failure -j1
```

Result: clean 15-step Ninja build with Clang 21.1.8; all 3 registered tests
passed (`frontend.bootstrap`, `integration.validation_matrix`, and
`runtime.cpu.gemm_v0`).

Saved-artifact relocatable proof:

```text
/tmp/matcore-pipeline-clean/bin/mdslc++ \
  -std=c++20 --matcore-target=cpu --save-temps -c \
  compiler/examples/gemm_v0.mdsl \
  -o /tmp/matcore-pipeline-clean/gemm_v0.o
```

Generated files:

```text
gemm_v0.host.cpp
gemm_v0.matcore.json
gemm_v0.sites.h
gemm_v0.stubs.cpp
gemm_v0.backend.cpp
gemm_v0.host.o
gemm_v0.stubs.o
gemm_v0.backend.o
gemm_v0.o
```

Artifact inspection:

```text
file /tmp/matcore-pipeline-clean/gemm_v0.o
readelf -h /tmp/matcore-pipeline-clean/gemm_v0.o
nm -C /tmp/matcore-pipeline-clean/gemm_v0.o
```

- `file`: ELF 64-bit LSB relocatable, x86-64, not stripped.
- `readelf`: `Type: REL (Relocatable file)` and x86-64 machine.
- `nm -C`: defined `main`, one stable `__matcore_call_site_*`, and one
  `matcore_generated_backend_*_v0`; unresolved
  `matcore_runtime_gemm_f32_v0` remains for the ordinary final link.

Ordinary external link and execution:

```text
/usr/bin/clang++-21 /tmp/matcore-pipeline-clean/gemm_v0.o \
  -L/tmp/matcore-pipeline-clean/lib -lmatcore_runtime \
  -Wl,-rpath,/tmp/matcore-pipeline-clean/lib \
  -o /tmp/matcore-pipeline-clean/gemm_v0
/tmp/matcore-pipeline-clean/gemm_v0
ldd /tmp/matcore-pipeline-clean/gemm_v0
```

Program output was exactly:

```text
host-before
MDSLC CPU GEMM PASS
```

The example compares the runtime result with a distinct oracle using double
accumulation in `k -> column -> row` traversal. `ldd` resolved
`libmatcore_runtime.so.0` from `/tmp/matcore-pipeline-clean/lib`.

The driver's normal final-link mode was also executed successfully. Its
verbose trace showed a private `/tmp/mdslc-XXXXXX` workspace, three separate
compilations, `-L`/rpath relative to the driver layout, and the runtime
library. The workspace no longer existed after the command returned.

Sanitizer propagation and execution:

```text
/tmp/matcore-pipeline-clean/bin/mdslc++ --verbose \
  --matcore-target=cpu -std=c++20 \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  compiler/examples/gemm_v0.mdsl \
  -o /tmp/matcore-pipeline-clean/gemm_v0_sanitized
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
  /tmp/matcore-pipeline-clean/gemm_v0_sanitized
```

The verbose trace contained `-fsanitize=address,undefined` on all three
compilations and on the final link. Execution printed the same pass output
with no sanitizer finding. Sanitizer-enabled `-c` also produced a normal ELF
relocatable.

Additional checks passed:

- Host-only direct fast path still compiled and printed `5`.
- CPU pipeline input after `--` produced a valid combined relocatable.
- `--matcore-target=cuda` returned status 2 with an explicit no-fallback
  diagnostic and emitted no CUDA or CPU artifact.
- Attempting `-o compiler/examples/gemm_v0.mdsl` returned status 2 and a
  before/after SHA-256 comparison confirmed the source was unchanged.

## Known limitations

- Bootstrap pipeline is Linux, Clang, one-source, CPU, rank-2 row-major f32
  GEMM only.
- It requires explicit `-o`; additional source/object inputs and compile modes
  such as preprocessing or assembly output are rejected.
- Shared-library orchestration, install-tree discovery, external consumer
  CMake helpers, CUDA, and other operations remain later milestones.
