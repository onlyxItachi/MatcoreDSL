# CPU kernel review checklist

Status: fail-closed review gate for private MDSLC CPU kernel and planner
changes.

This checklist applies to GEMM implementation work and to any future private
GEMV or GEVM experiment. It does not freeze public APIs, promote design-only
operations, or substitute for correctness, artifact, sanitizer, package, and
performance evidence.

Reviewers should mark every applicable item as pass, fail, or not applicable
with a reason. A claim is not accepted because the code looks plausible.
Evidence must identify the source commit, build configuration, host,
capabilities, command, and result.

## 1. Change identity and scope

- [ ] The source commit and complete reviewed diff are recorded.
- [ ] The change names the exact operation, dtype, layout, ISA, thread mode,
      packing mode, and cache mode in scope.
- [ ] Implemented, runtime-validated, compile-only, synthetic-only, and
      proposed behavior are separated.
- [ ] Measured, derived, source-backed, hypothesis, and proposed claims are
      labeled consistently.
- [ ] No CUDA, GPU, public GEMV/GEVM, runtime autotuning, or API-freeze work is
      hidden in the diff.
- [ ] No third-party kernel implementation was copied.

## 2. Semantics, IR, and verifier

- [ ] Operand roles, ranks, shapes, dtype, accumulation dtype, layout,
      strides, alignment, memory space, mutability, effects, and alias rules
      are preserved.
- [ ] The operation's overwrite/read-modify-write and numerical-order
      semantics are explicit.
- [ ] Unsupported input fails before output mutation and without fallback.
- [ ] Dynamic dimensions, zero dimensions, overflow, and malformed versions
      are either verified or rejected explicitly.
- [ ] A versioned conversion boundary covers any IR or descriptor change.
- [ ] Deterministic serialization and source-location diagnostics remain
      intact.
- [ ] Optimized code cannot bypass the shared semantic verifier.

## 3. Ownership, workspace, and transformed operands

- [ ] Every allocation and copy is visible in the declared measurement and
      execution contract.
- [ ] Required workspace size and alignment use overflow-safe arithmetic.
- [ ] Insufficient or misaligned workspace fails before output mutation.
- [ ] Workspace does not overlap A, B/x, C/y, transformed storage, or another
      worker's live region.
- [ ] Caller and runtime ownership/lifetime boundaries are documented.
- [ ] A transformed operand records format version, logical/padded extents,
      dtype, layout, blocking/ISA parameters, source identity, storage,
      alignment, and provenance.
- [ ] Pointer identity is not treated as proof that source contents are
      unchanged.
- [ ] Invalidation and concurrent-read rules are explicit.
- [ ] No implicit process-global mutable cache was introduced.
- [ ] Prepacking and reuse are benchmarked separately from transient packing.

## 4. Capability and ISA legality

- [ ] Generic translation units remain runnable without optional ISA support.
- [ ] Hardware support, OS architectural state, compiler support,
      implementation availability, and physical runtime validation are
      distinct facts.
- [ ] The variant fails closed when any required fact is absent.
- [ ] Forced unavailable variants fail actionably without changing output.
- [ ] Synthetic capability tests do not masquerade as physical execution.
- [ ] A variant ID names only an implementation whose body contains the
      claimed ISA.
- [ ] Windows and Linux capability paths report their actual completeness.

## 5. Microkernel and artifact proof

- [ ] The full-tile and checked-edge entry contracts are separate and
      documented.
- [ ] Every unchecked call site proves non-null pointers, complete extents,
      authenticated packed intervals, output bounds, legal stride, K-panel
      mode, non-overlap, and ISA entry.
- [ ] Tails do not perform speculative reads or stores beyond live or
      initialized padded storage.
- [ ] Exact Release symbols are retained and isolated for disassembly.
- [ ] AVX2 evidence requires YMM packed FMA operations in the production body.
- [ ] AVX-512 evidence requires ZMM packed FMA operations in the production
      body.
- [ ] Artifact checks reject scalarized, XMM-only, helper-only, or wrapper-only
      false positives.
- [ ] Register spills, stack traffic, and unexpected scalar hot loops were
      inspected.
- [ ] Debug/default supported configuration either has an honest optimized
      body or is reported separately.
- [ ] Private microkernel symbols are absent from the installed public ABI.

## 6. Correctness and adversarial memory tests

- [ ] Independent double-precision or exact integer oracle is used.
- [ ] Tiny, square, rectangular, vector-like, and randomized shapes pass.
- [ ] Every register-, cache-, and page-boundary tail class passes.
- [ ] 4-, 16-, 32-, and 64-byte alignments are tested where legal.
- [ ] Guard regions prove no out-of-bounds output write.
- [ ] Null, negative, zero, overflow, malformed metadata, and unsupported
      semantics fail cleanly.
- [ ] Exact and partial input/output/workspace/transformed-storage overlaps are
      rejected.
- [ ] Output remains byte-identical after every pre-execution rejection.
- [ ] Repeated executions and independent concurrent contexts remain correct.
- [ ] Release, Debug, ASan, and UBSan pass.
- [ ] TSan passes for new shared state, or an exact supported-toolchain blocker
      is recorded.

## 7. Parallel execution

- [ ] Worker threads are persistent rather than recreated per operation.
- [ ] Requested ceiling and actual worker count are reported.
- [ ] Output tasks own disjoint cache-line-safe regions or synchronize an
      explicit reduction.
- [ ] Per-worker workspace is isolated and false-sharing boundaries are
      documented.
- [ ] Shared transformed storage becomes read-only only after a proven
      publication barrier.
- [ ] No worker reads a panel while another worker may still write it.
- [ ] Error/abort paths cannot strand peers at a barrier or deadlock shutdown.
- [ ] K splitting, if any, exposes reduction workspace, order, and
      synchronization rather than changing arithmetic silently.
- [ ] Native and external-provider pools cannot nest or oversubscribe.
- [ ] Task generation is deterministic for an injected problem/topology.
- [ ] Repeated execution does not recreate workers.
- [ ] Shutdown, context-capacity, concurrent-context, and error-propagation
      stress tests pass.

## 8. Topology, affinity, and NUMA

- [ ] Logical CPUs, physical cores, sockets, NUMA nodes, and discovery
      completeness are not conflated.
- [ ] Affinity requests and actual placement are reported separately.
- [ ] Partial affinity availability fails closed or degrades explicitly.
- [ ] Single-node, synthetic multi-node, and physically validated multi-node
      evidence are labeled separately.
- [ ] No hidden page migration, interleaving, or first-touch assumption exists.
- [ ] Cross-node execution appears in plan diagnostics.
- [ ] Linux sysfs and Windows topology backends remain isolated behind the
      platform model.

## 9. Planner and execution-report integrity

- [ ] Every fixed candidate records legal/rejected state and exact reason.
- [ ] Required capabilities, workspace, topology, actual threads, tasks,
      estimated cost, priority, selected state, and selection reason are
      reported.
- [ ] Cost arithmetic is saturating and candidate/tie order is fixed.
- [ ] Planning has no timing-history or runtime-autotuning dependency.
- [ ] Planner and runtime share the same task-plan function.
- [ ] The execution report authenticates the implementation and resources
      actually used, not only the requested variant.
- [ ] Forced legal candidates execute; forced illegal candidates do not
      fallback.
- [ ] Calibration and holdout sets are separate and identified.
- [ ] Threshold changes improve the target region without catastrophic
      holdout regret.

## 10. Benchmark fairness and performance claims

- [ ] Equivalent semantics, dtype, layout, output verification, threads,
      affinity, cache state, packing mode, and repeated-use mode are compared.
- [ ] Provider initialization and actual OpenBLAS thread count are controlled.
- [ ] Native and provider nested threading is disabled.
- [ ] One-shot, allocation-excluded, packing-included, prepacked/reused, and
      compute-only intervals are named separately.
- [ ] Compute-only diagnostics are not compared as end-to-end results.
- [ ] Allocation and initialization are not accidentally inside a claimed
      compute interval.
- [ ] Warmup, timer floor, iteration count, outlier rule, governor/frequency,
      and host-load metadata are recorded.
- [ ] Every timed result has a checksum and correctness status.
- [ ] GFLOP/s uses `2*M*N*K / seconds` for F32 GEMM.
- [ ] Absolute time, throughput, ratio, scaling, and regret are derived from
      authenticated raw evidence.
- [ ] Raw profiler and benchmark bundles remain ignored and untracked.
- [ ] Sanitized summaries include weak regions and excluded/noise cases.
- [ ] Planner results selecting OpenBLAS are not counted as native parity.
- [ ] No universal BLAS, ISA, topology, or hardware claim is made from one
      host.

## 11. ABI, package, and portability

- [ ] Existing exported C symbols and struct layouts remain compatible.
- [ ] Any unavoidable ABI addition is C-only, versioned, and additive.
- [ ] No STL, templates, C++ exceptions, or private compiler ABI types cross
      the runtime boundary.
- [ ] Internal microkernel, packing, planner, topology, and thread-pool headers
      are not installed.
- [ ] Runtime exports contain no accidental private symbols.
- [ ] Build-tree and installed-tree consumers pass with source unavailable.
- [ ] Package files contain no source/build absolute paths.
- [ ] OpenBLAS-disabled configuration remains supported.
- [ ] Linux Release/Debug and sanitizer gates pass.
- [ ] Windows clang-cl, COFF/PE, runtime DLL/import library, paths-with-spaces,
      install, consumer, and ZIP workflow remain green for the validated
      compatibility scope.

## 12. Operation-specific review

### GEMM

- [ ] Blocking and packed geometry match the authenticated format.
- [ ] First K panel overwrites C and later panels accumulate.
- [ ] A/B panel reuse and any repacking are visible.
- [ ] Complete tiles use only full-tile entries and edges use checked entries.
- [ ] Parallel M/N rectangles are disjoint; K is not split unless explicitly
      reviewed.
- [ ] Cooperative B packing uses disjoint final intervals, a publication
      barrier, and the accepted measured envelope.
- [ ] Caller-owned prepacked B is not confused with within-call cooperative B
      preparation.

### GEMV

- [ ] The operation is still private unless a separate language/IR/ABI review
      approved exposure.
- [ ] Direct row streaming is the default one-shot path.
- [ ] Multi-row dot products share x loads without reading phantom rows.
- [ ] Horizontal reductions and K tails are exact and artifact-tested.
- [ ] Row tasks own disjoint y ranges and do not inherit GEMM thresholds.
- [ ] Any transformed-A mode is explicit and authenticated.

### GEVM

- [ ] The operation is still private unless a separate language/IR/ABI review
      approved exposure.
- [ ] Row-major A is streamed contiguously; naive strided column-dot traversal
      is not the hot path.
- [ ] N-block or row-stream mode is identified explicitly.
- [ ] N tasks own disjoint y ranges.
- [ ] Any K split exposes partial-output workspace and deterministic reduction.
- [ ] Any transformed-A mode is explicit and authenticated.

## 13. Review disposition

- [ ] Every finding has severity, reproduction, owner, and resolution.
- [ ] No confirmed high or medium finding remains.
- [ ] Rejected experiments remain documented as rejected and are not promoted
      through a different call path.
- [ ] Hosted checks and local public-surface gates ran on the reviewed commit.
- [ ] Final parity numbers, when reported, come only from the authenticated
      parity summarizer and its declared envelope.
- [ ] The review states supported claims and claims explicitly unsupported.
- [ ] Public API/ABI/backend freeze remains a separate milestone.

Failure of an applicable correctness, legality, artifact, race, ABI, or
benchmark-authentication item blocks promotion. A performance miss may justify
keeping a correct private candidate for further investigation, but it must not
be relabeled or silently selected.
