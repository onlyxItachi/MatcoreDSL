# Follow-up Notes

## Harness correction

- The initial `device_resident` failure in `summary.md` was a wrapper issue, not a benchmark/runtime issue.
- `tests/run_complete_benchmarks.py` was updated afterward to prepend the repo root to `PYTHONPATH` for subprocess benchmarks.

## Manual rerun: `tests/benchmark_device_resident.py`

Executed with:

```bash
PYTHONPATH=/home/hamza-usta/MatcoreDSL ./.venv/bin/python tests/benchmark_device_resident.py
```

Key results:

- `256x256x256`
  - host median: `0.353 ms`
  - device median: `0.129 ms`
  - speedup: `2.7x`
  - device throughput: `0.2603 TFLOP/s`
- `512x512x512`
  - host median: `0.635 ms`
  - device median: `0.085 ms`
  - speedup: `7.4x`
  - device throughput: `3.1454 TFLOP/s`

Target-validation result from the script:

- `<0.05 ms` at `256x256`: `FAIL`
- `>0.5 TFLOP/s` at `256x256`: `FAIL`

## Root cause: `core_suite` failure

`tests/run_benchmarks.py` failed for real backend reasons, not due to the wrapper.

### AMD gladiator

Direct run:

```bash
./.venv/bin/python tests/benchmark_gladiator.py --arena amd-igpu --size 256 --warmup 8 --runs 4
```

Observed failure:

- `amd-igpu` lowering fails at stage `lowering (stage 1)`
- diagnostic: `failed to legalize operation 'builtin.unrealized_conversion_cast' that was explicitly marked illegal`
- reproduced for `float16` and `int8`

### Massive benchmark

Direct run:

```bash
./.venv/bin/python tests/benchmark_massive.py --targets x86-avx2 nvidia-dgpu amd-igpu
```

Observed results:

- `x86-avx2`: `159574.769 ms`, `0.0004 TFLOP/s`
- `nvidia-dgpu`: `496.538 ms`, `0.1384 TFLOP/s`, median step `6.645 ms`, step throughput `1.2927 TFLOP/s`
- `amd-igpu`: same lowering failure as the gladiator AMD path (`builtin.unrealized_conversion_cast`)
