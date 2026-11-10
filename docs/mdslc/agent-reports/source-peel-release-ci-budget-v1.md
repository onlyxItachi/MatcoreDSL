# Source peeling: complete hosted Release qualification budget

The `clang-21-release` job budget changes from 60 to 90 minutes. Its matrix,
commands, two-job build limit, serial CTest selection, per-test timeouts and
production-without-tests package proof are unchanged. Debug, sanitizer, ARM and
Windows budgets are unchanged. This is not a performance result or permission
to count cancelled jobs as passes.

At compiler checkpoint `1ad7fe56b5fcb10efe37e523206228e9ef303f6a`, PR105's
[native run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743932236)
retains two deadline cancellations. Both check-run annotations explicitly say
the job exceeded one hour. Both completed their CTest step successfully, then
were cancelled during the final `BUILD_TESTING=OFF` production/package proof:

- Job `129591398741`, OpenBLAS OFF / scalar: CTest took about 49m37s;
  approximately one minute remained for the production step.
- Job `129591399005`, OpenBLAS OFF / output-tiled: CTest took about 49m27s;
  approximately two minutes remained for the production step.

The exact-head [push run](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/42743923523)
completed all jobs. Its corresponding scalar/output jobs finished in 44m50s and
59m40s; their production steps took approximately 1m33s and 2m11s. Even required
OpenBLAS jobs, which exclude that production step, finished in 59m35s (PR) and
59m51s (push). Moving only the production proof would therefore leave the full
test job close to the old deadline. Rerunning on a faster runner would not fix
that budget mismatch. No acceptance scope is removed or retried until passing.

The integration owner and two independent review lanes inspected the retained
annotations, logs, unchanged commands and existing budget-adjustment precedent.
Fresh exact-head hosted qualification is still required after this correction.
The original two cancelled jobs remain cancelled in the evidence record.

The producer worktree and installed SDK stay frozen at `1ad7fe56`; this change
is made in a separate worktree. Local Release 356/356, committed-head focused
ASan 4/4 and earlier hosted evidence remain attributed to that exact compiler
checkpoint. Compiler, runtime, tests, headers and source contracts are byte-
unchanged by this CI/documentation-only commit. The P5 fixed producer identity
does not move; no timing collection is authorized before qualification finishes.
