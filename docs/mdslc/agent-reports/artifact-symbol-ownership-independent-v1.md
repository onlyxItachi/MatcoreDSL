# Independent artifact symbol ownership review

**Verdict: ACCEPT within the documented exact-artifact Linux x86-64 link
contract.** The reviewer did not author or change the utility, its CMake, or
its production fixtures. This report is not hosted CI or driver-installation
evidence.

Reviewed SHA-256 identities:

- `ArtifactSymbolOwnership.cpp`:
  `5a19b042678354a8bd3e3b92ffe92f0bc7e836aa8e459e5cc6890e6f25fc4f38`.
- `ArtifactSymbolOwnership.h`:
  `c441aee25a3d62b1534db696cbb90a8590866b97ef55b75b94d029d28b929b99`.
- Permanent test:
  `4287c41914df7ea094d172c2f90a9f5e555607e26f2dd684968646fbd609d14d`.

## Rejected alternative, actually falsified

The initial namespace classifier admitted an ordinary Clang-emitted host
definition of an exported runtime function template. Its demangled spelling,
`int matcore::owned_template<int>(int)`, begins with a return type rather than
`matcore::`. An independently compiled DSO exported that weak definition plus
one `matcore_runtime_owned` function. The original checker incorrectly returned
`admitted=1 runtime_exports=1` for a host providing a different template body.

The same unchanged host IR and DSO now produce `admitted=0 runtime_exports=0`
with the exact owned template symbol in the diagnostic. The correction reserves
every defined nonlocal dynamic export, uniformly, without demangling. Permanent
tests also cover templated-function static values and guards. The simpler rule
avoids replacing one incomplete namespace heuristic with another parser.

Preserved local counterexample identities:

- DSO: `e8bb428c795bd0ce6fb082abced5a1bc5c54cb7936ba6ee67f1b39d83214b608`.
- Host LLVM IR: `f3a8de066c0efdc06e4438054b3de3f25a1977744b63bdce3ca4cf7863214e32`.
- Old checker executable: `8d49a1eef7119fbec7705416fd28e475d8e9fd942393bd72eac8a75fd5ed9528`.
- Corrected checker executable: `5c32f7e9da767b1c5e7ff26b457576f67f7f50c679dbe0defb707f38b39206bc`.

The scratch directory is `/tmp/mdslc-artifact-ownership-independent.ncxlH9`;
permanent adversarial fixtures, not that temporary path, carry the regression.
The actual current runtime has only 15 public C exports; this synthetic template
case falsified the proposed general ownership rule, not a claimed live template
exploit against that runtime.

## Independently repeated validation

Linux x86-64, Clang/LLVM 21.1.8:

| Scope | Outcome |
| --- | --- |
| Standalone Release, ownership plus existing platform-support test | 2/2 PASS, 0.97 s |
| Debug ASan+UBSan, same tests | 2/2 PASS, 1.02 s |
| Actual runtime/OpenBLAS artifacts, Release inspection | 72 checks, zero failures |
| Actual runtime/OpenBLAS artifacts, instrumented inspection | 72 checks, zero failures |
| Independent old template counterexample against corrected utility | Rejected at intended symbol-ownership boundary |

The actual-artifact invocations used the frontend integration build's
`libmatcore_runtime.so` and
`/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0`. Ordinary host
`std::vector` code remains admitted against the current hidden-implementation
runtime. Defining the actual provider's `blas_memory_alloc` is rejected.
ASan/UBSan instrumented the inspecting utility; child fixtures are compiled
IR/DSOs for inspection, not a new instrumented provider-execution claim.

## Ownership and limits

Declarations and local symbols do not claim externally interposable definitions.
Functions, data, aliases, IFUNCs, weak exports and versioned names are checked
through LLVM's existing object/global-value interfaces. Nonempty module or
instruction assembly is explicitly outside this bounded contract; empty barriers
remain admitted. Failed checks do not issue successful export counts.

An implementation/STL export leak in a future trusted DSO now fails closed when
it collides with host definitions. Fixing that DSO's export visibility is the
appropriate ownership layer, not guessing that the collision is harmless.
The driver still owns original-host timing, immutable artifact authentication,
whole-linking its private candidate archive, closure rechecks and final linking.
Trusted future dynamic loading, conforming allocation hooks and absence of
foreign interposition remain deployment preconditions. This is not source
authentication, an arbitrary-IR execution gateway, or a sandbox.
