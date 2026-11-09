# Windows diagnostic transport regression

Observed while qualifying documentation-only PR #95 from canonical
`b2597bed52de46604574f76eb33663788a5ab994`. In hosted Windows run
[42717343483](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42717343483),
`integration.windows.native_pipeline` failed under Python 3.15.2: a subprocess
reader thread could not decode diagnostic byte `0xfc` as UTF-8. The later
reserved-device-output assertion received `stderr=None` and raised `TypeError`.
The compiler tree was unchanged by that documentation PR; 50/51 Release tests
passed, but the run is a failure, not Windows qualification.

The narrow fix makes this test helper's diagnostic decoding explicit:
UTF-8 with `backslashreplace`. Valid UTF-8 remains text; other bytes remain
visible as escapes. No bytes are silently dropped, the subprocess exit code is
unchanged, and every original negative/actionable-diagnostic check remains.
Strict UTF-8 JSON/artifact reads are unchanged. This does not modify compiler,
frontend, runtime, ABI, source semantics or platform support.

The first fix covered `run` but hosted run
[42718828654](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42718828654)
exposed the same decoding problem in the separate asynchronous source-mutation
`Popen` path. Both now share the explicit diagnostic transport policy; the
source-mutation timing, no-output and actionable-rejection assertions remain.
The new `windows.diagnostic_capture` regression runs four real child processes:
valid UTF-8 stdout plus invalid UTF-8 stderr with exit 7, and ordinary success,
through both synchronous and asynchronous helpers. Exact outputs and statuses
are checked. Local Python 3.15 and system Python 3.12 each passed 4/4 controls.
An independent reviewer inspected the first fix and repeated its original 2/2
controls; the asynchronous follow-up requires its own review and hosted run.
`git diff --check` passed. Native hosted Windows validation remains a separate
merge gate; no Linux run substitutes for it.

This is a test-harness compatibility correction, not another compiler milestone.
The operator's engineering frontier remains Issue #89 and broader mathematical
coverage remains a separate correctness-first track.
