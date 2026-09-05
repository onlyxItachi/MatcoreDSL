# Prebuilt MLIR allocator instrumentation control

This small diagnostic is for the pinned, non-ASan LLVM/MLIR 21.1.8 package
tuple. It does not build LLVM, disable process-wide sanitizer checks, or grant
the compiler an exemption from semantic memory checks.

`mlir_allocator_registration.cpp` introduces the same upstream singleton-type
registration as the Matcore descriptor/order types. Its header-instantiated,
weak `BumpPtrAllocator::AllocateSlow` can replace the packaged implementation.
Under ASan it poisons an entire slab, whereas the package's already-inlined
fast allocation path cannot unpoison subsequent allocations. The mixed binary
therefore fails while constructing the builtin MLIR context, before its custom
dialect is loaded. The compatible binary compiles only the registration
translation unit with the package's non-ASan protocol; UBSan remains enabled.

The checker requires all four outcomes:

1. Mixed registration fails with use-after-poison before builtin initialization
   completes.
2. Compatible registration and the fully instrumented client succeed.
3. The compatible binary still detects an intentional ordinary heap overflow.
4. The compatible binary still detects an intentional manually poisoned access.

The last two prevent disguising the failure by disabling ASan or user poisoning.
`symbolize=0` avoids local network-debuginfo delays; it changes diagnostic
formatting only. The expected mixed failure is specific to this pinned
prebuilt-package experiment, not a portable requirement for coherently
ASan-built upstream libraries. A future fully instrumented dependency tuple
must revisit the registration boundary instead of inheriting it silently.

From the repository root, with the inspected local package paths (adjust the
two package prefixes only if the chosen exact tuple lives elsewhere):

```bash
mdslc_repro=$(mktemp -d /tmp/mdslc-allocator-control.XXXXXX)
mdslc_cxx=/usr/bin/clang++-21
mdslc_llvm=/usr/lib/llvm-21
mdslc_mlir=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21
mdslc_fixture=compiler/tests/sanitizer
mdslc_flags=(-std=c++20 -O1 -g -fsanitize=address,undefined
             -I"$mdslc_llvm/include" -I"$mdslc_mlir/include")
"$mdslc_cxx" "${mdslc_flags[@]}" -c "$mdslc_fixture/mlir_allocator_client.cpp" \
  -o "$mdslc_repro/client.o"
"$mdslc_cxx" "${mdslc_flags[@]}" -c "$mdslc_fixture/mlir_allocator_registration.cpp" \
  -o "$mdslc_repro/mixed.o"
"$mdslc_cxx" "${mdslc_flags[@]}" -fno-sanitize=address \
  -c "$mdslc_fixture/mlir_allocator_registration.cpp" \
  -o "$mdslc_repro/compatible.o"
for mdslc_mode in mixed compatible; do
  "$mdslc_cxx" -fsanitize=address,undefined \
    "$mdslc_repro/$mdslc_mode.o" "$mdslc_repro/client.o" \
    "$mdslc_mlir/lib/libMLIRIR.a" "$mdslc_mlir/lib/libMLIRSupport.a" \
    -L"$mdslc_llvm/lib" -lLLVM-21 -o "$mdslc_repro/$mdslc_mode"
done
python3 "$mdslc_fixture/check_mlir_allocator_sanitizer.py" \
  --mixed "$mdslc_repro/mixed" --compatible "$mdslc_repro/compatible"
```

The equivalent compilation executed locally on 2026-09-05 against exact
21.1.8 packages: **4/4 controls passed**. Normal repository sanitizer regression
tests must still pass; this fixture is diagnostic evidence, not their substitute.

Relevant upstream definitions:
[LLVM 21.1.8 allocator poison/unpoison protocol](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/include/llvm/Support/Allocator.h),
[MLIR singleton storage registration](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/include/mlir/Support/StorageUniquer.h).
