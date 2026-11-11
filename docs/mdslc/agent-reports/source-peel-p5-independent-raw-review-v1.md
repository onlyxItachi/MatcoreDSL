# P5 independent retained-raw audit

Disposition: **honest INCONCLUSIVE_TIMING; primary does not qualify.** No primary
or secondary comparison qualifies. All 18 planned comparison ratios must remain
null. This is neither a qualified speedup nor a qualified slowdown, and no
secondary may replace the failed primary. No repeat collection was performed.

## Scope and independence

The audit script imports only Python standard-library modules, independently
encodes exact integer f32 bits, hashes every expected output call, and uses exact
rational timing/pair arithmetic. It does not import benchmark acceptance or
summary functions. Only after reconstruction does it compare the controller's
derived fields. The reviewer authored test-only source/P5 controls, not production
`SourcePattern`, the protocol, controller or timed source.

Source producer remains `1ad7fe56b5fcb10efe37e523206228e9ef303f6a`; the operator
gate retains qualified-unmerged source `cf154dea...`, harness `ea7accfe...` and
canonical main `1f7c491...`. Its fixed CPU is 2. The gate predates command zero;
it is an operator record, not a serialized execution credential or merge claim.

## Raw timing acceptance

Acceptance is recomputed with the unchanged zero-involuntary-switch predicate,
5 ms floor, CPU/wall >=9/10 and CPU<=wall+100 us. Every row below requires 9/9
accepted samples in both directions before spread/order pairing can qualify.

| M/K/N | Role | Forward accepted | Reverse accepted | Pair |
| --- | --- | ---: | ---: | --- |
| 33/35/36 | stock | 6/9 | 6/9 | Inconclusive |
| 33/35/36 | old 3x5 | 5/9 | 7/9 | Inconclusive |
| 33/35/36 | peeled 3x5 | 7/9 | 6/9 | Inconclusive |
| 33/35/36 | old 5x3 | 3/9 | 3/9 | Inconclusive |
| 33/35/36 | peeled 5x3 | 8/9 | 7/9 | Inconclusive |
| 33/35/37 | stock | 9/9 | 6/9 | Inconclusive |
| 33/35/37 | old 3x5 | 6/9 | 9/9 | Inconclusive |
| 33/35/37 | peeled 3x5 | 8/9 | 7/9 | Inconclusive |
| 33/35/37 | old 5x3, primary denominator | 4/9 | 5/9 | Inconclusive |
| 33/35/37 | peeled 5x3, primary numerator | 9/9 | 6/9 | Inconclusive |
| 33/35/38 | stock | 8/9 | 7/9 | Inconclusive |
| 33/35/38 | old 3x5 | 5/9 | 9/9 | Inconclusive |
| 33/35/38 | peeled 3x5 | 8/9 | 7/9 | Inconclusive |
| 33/35/38 | old 5x3 | 4/9 | 3/9 | Inconclusive |
| 33/35/38 | peeled 5x3 | 7/9 | 7/9 | Inconclusive |

Of 270 timed aggregates, 192 pass and 78 fail solely for a nonzero involuntary
switch delta: 75 have one and three have two. No floor, CPU-ratio or positive
clock-boundary-slack failure occurs. Timed wall intervals span 12,958,396 to
53,571,731 ns. The minimum CPU/wall ratio is 0.9986247239262276; CPU-minus-wall
spans -29,266 to +3,253 ns. All voluntary switch deltas are zero.

Warmups separately have 43 accepted and 17 switch-rejected aggregates out of 60;
they are not timing samples. RUSAGE/thread-clock reads bracket a wider interval
than the steady-clock timer in the frozen source. These deltas do not prove a
switch occurred inside the timed loop or bound its interference, and cannot
justify a post-hoc exception to the registered predicate.

## Correctness, identity and completeness

- Exact 40-process order: initial five controls, 15 forward entries, their exact
  reverse, final five controls. Every raw command, argv, exit, stderr and all
  120 raw-file identities agree with the complete catalog. No process timed out.
- All ten controls report the same 17 required categories, 469 checks each and
  zero failures: 4,690 checks total. All actual executable-map inventories join
  each role's pinned binary/DSO/runtime/dependencies.
- All 552,960 timed calls / 675,164,160 elements and 122,880 warmup calls /
  150,036,480 elements match counts and independently derived ordered hashes.
  Largest absolute integer K-prefix is 438, well below the exact f32 bound.
- All 225 prepared pins still match. All ten source bindings are recomputed;
  80 actual stage/object/issuer hash checks include the 48 cross-instance
  timed/control/end-K field joins. Prepared correctness transfer remains bound
  to actual unchanged artifacts, not merely request flags.
- All 30 measured processes and all 330 aggregates' before/after observations
  retain CPU2 affinity and one task. This is not continuous active-thread proof.
- All 62 retained host snapshots pass: load1 0.35009765625..0.73193359375 on 24
  logical CPUs, no named build/test process, available temperature
  50.125..74.875 C, recorded existing performance governor / boost=1.
  These do not establish continuous exclusivity or absence of SMT competition.
- Maximum benchmark-owned payload bound is 10,776,716 bytes; observed process
  peak RSS maximum is 24,711,168 bytes, a distinct quantity. Raw bytes are 219,895;
  no new generated artifacts; collection wall duration is 27.28075297200121 s.
- All 121 collection files and 509 preparation files remain unchanged. No
  compiler, native program, benchmark or inspection process was launched.

The independent reconstruction agrees with every controller summary/null ratio.
A secondary complete controller audit also passes with process creation blocked;
it is corroboration, not the independent mathematical oracle. Its first wrapper
attempt mis-assumed the auditor's return type while printing; correcting that
presentation-only error produced the clean pass without changing any evidence.

## Next-boundary assessment, not a new protocol

A separately preregistered measurement-method qualification is a defensible
narrow next question. P5 shows this host/workload/strict acceptance combination
yielded no eligible pairs; it does not prove clock failure or quantify switch
interference. Do not loosen the zero-switch rule or publish raw median ratios.
An identical-binary null plus controlled comparisons should be able to falsify
repeatability/order/bias claims, and may itself remain infeasible/inconclusive.
Null agreement alone cannot validate comparisons between different allocator,
cache or generated-code paths. Any future method needs ex-ante falsifiers and
separate validation; it cannot reclassify P1/P5 or become performance policy.
No new campaign, implementation or threshold is authorized by this assessment.

## Reproduction artifacts

- `audit.py` SHA256: `eb74199dd01259f5af367ebb8b0b9e3a70b0ae5c860b2cd317c4dfff82fd7696`.
- `audit.json` SHA256: `79a54b4e91e540142167fd11fe3ff7cb93572fa308e9ade606166d16c398366b`.
- Collection SHA256: `7c76073ec7136586bf92a6576cbd7dd995b0159e5ae0a7506a43765bb42965e7`.
- Prepared SHA256: `add6dfba5ea56d46a743479194bb289f631d780bc85dfa59660a6862e5f8afa6`.
- Plan SHA256: `2933f108ba13bb7e7bd417e8f53efe416650d22506b0a7e2e514278e94b42d85`.

The JSON preserves every rejected timed/warmup coordinate and each direction's
accepted/rejected ordinals. Large output hashes there are exact integer JSON,
not rounded JavaScript-number summaries. This is trusted-collector integrity
review, not hostile-origin attestation or universal numerical qualification.
