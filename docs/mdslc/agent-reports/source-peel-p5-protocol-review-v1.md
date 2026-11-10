# Matched source-call peeling: P5 protocol review

Status: **preregistered harness, not a measurement result**.
[Issue #106](https://github.com/onlyxItachi/MatcoreDSL/issues/106) follows the
separate source connection in PR105. Actual collection is forbidden until that
product qualification and this protocol's review gates complete. Product source
for this fixed campaign is `1ad7fe56b5fcb10efe37e523206228e9ef303f6a`; its
clean producer/build must remain frozen independently of this benchmark branch.

The [protocol and usage](../../../compiler/benchmarks/source_pattern_peel/README.md)
define the exact five roles, three shapes, source-call/retained-ownership interval,
fixed repetitions, noise gates, caps, chronological order and primary contrast.
There is no cost model, selected winner, tuning threshold or runtime-policy edit.
The old P1 source, Python harness and retained campaign are unchanged.

## Decision and falsification record

- P1's overall inconclusive timing and bounded 5x3/stock loss motivate a
  *new*, deliberately narrow experiment. They are not rerun or reinterpreted.
  The primary 33/35/37 peeled5x3/old5x3 contrast was selected before P5 timing.
- Changing to 2048 retained results changes allocator/cache pressure. This is
  not a timer-only adjustment; no timing ratio against the older P1 campaign is
  allowed. One source is rendered once for all five roles by exactly three
  SHA-bound substitutions: matrix, floor and receipt schema. Region, wrapper,
  clocks, ownership, checks and noise predicate stay byte-identical.
- An early proposal to accept one involuntary switch was rejected before
  implementation. Observed switch counts and wall-minus-thread gaps do not
  bound interference duration or cache effects. P1's original zero-switch
  predicate remains; even passing it does not prove a quiet machine.
- Reusing P1's high-level controller by changing its globals was rejected:
  schemas, role counts and raw-command indices are closed over P1's experiment.
  P5 reuses its lower process/identity/installation helpers read-only and has a
  separate fixed controller. The stock-profile audit reconstructs retained
  facts without executing the live helper's CMake transformation again.
- Root review found missing immediate measured-process mapping verification,
  incomplete source/helper inventories, partial stock-profile rederivation and
  unvalidated timeout metadata. The final controller closes these gaps.
- Independent testing also challenged omitted host-snapshot/budget validation,
  role substitution, dropped/duplicated chronological records, changed source
  closure, edited output/noise/normalization/ratios and a failed primary being
  replaced by a surviving secondary. Invalid evidence fails rather than being
  treated as noisy performance.

## Independent lanes and completed evidence

Controller-only commit `e610448` was reviewed by the root and a separate
read-only reviewer who did not author P5. No source-level blocking finding
remained after the corrections. Independent tests/documentation are commit
`0fed84fbd18553918595a3e95dffc1969ab172c7`.

The independent lane ran **28 synthetic Python test methods**, all passing.
Its real-file synthetic collection contains 40 commands and 120 raw files;
only the preparation prerequisite is mocked, and subprocess execution is
forbidden. This proves neither honest-path native preparation nor a complete
preparation replay. Actual source compilation, source/end-K controls, DSO
inspection and offline preparation audit must still pass before measurement.
The initial 18-method protocol scope also passed under the current default and
system Python interpreters (live-verified as 3.14.3 and 3.14.4, not the stale
3.12 label from earlier environment notes); the unchanged P1 15-method suite
passed. At `9afff89`, the configured CTest interpreter is separately verified
as Python 3.15.0a7: both P1/P5 synthetic CTests passed, 2/2, zero skips, 5.76s;
the P5 28-method suite also passed directly under system Python 3.14.4.
JUnit SHA256: `9b60fa6e6dca8a1dff70ef477c4e70cbcbcd976c7c18f3e3e11389ded0ed37be`.
The integration owner must record hosted qualification and any real campaign
outcomes separately. No native build or measurement was part of those tests.

The new Linux/Python CTest is a separate synthetic-only test. It does not run
timing on shared hosted CI or change existing sanitizer selection. No compiler,
runtime, source declaration, issuer, provider/default, numerical permission,
GPU/SPIR-V code or execution authority changes in this benchmark increment.

Receipts check integrity and derivations under a trusted collector/host
precondition. They are not hostile-origin attestations, executable artifact
imports or a sandbox. Successful preparation is correctness/identity evidence;
successful timing describes only the predeclared interval on the observed host.
Losses and inconclusive cells remain in the result. Issue #15 remains open.
