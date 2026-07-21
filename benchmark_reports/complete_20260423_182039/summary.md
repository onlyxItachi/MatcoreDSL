# Complete Benchmark Summary

Generated: 2026-04-23T18:29:47

| Benchmark | Category | Status | Duration (s) | Detail | Stdout | Stderr |
| --- | --- | --- | ---: | --- | --- | --- |
| core_suite | core | FAIL | 206.8 | Overall:      FAIL | benchmark_reports/complete_20260423_182039/core_suite.stdout.log | benchmark_reports/complete_20260423_182039/core_suite.stderr.log |
| device_resident | device | FAIL | 0.1 | ModuleNotFoundError: No module named 'matcore' | benchmark_reports/complete_20260423_182039/device_resident.stdout.log | benchmark_reports/complete_20260423_182039/device_resident.stderr.log |
| graph_vs_torch | comparison | PASS | 2.2 | with multi-warp cooperation and shared-memory double-buffering. | benchmark_reports/complete_20260423_182039/graph_vs_torch.stdout.log | benchmark_reports/complete_20260423_182039/graph_vs_torch.stderr.log |
| full_comparison | comparison | PASS | 3.1 | → Closest gap: 1.2x at 1024x1024 | benchmark_reports/complete_20260423_182039/full_comparison.stdout.log | benchmark_reports/complete_20260423_182039/full_comparison.stderr.log |
| attention | fusion | PASS | 2.1 | 128x128x64   / 2.875          / 0.023            / 0.261           / 0.003 | benchmark_reports/complete_20260423_182039/attention.stdout.log | benchmark_reports/complete_20260423_182039/attention.stderr.log |
| fusion_suite | fusion | PASS | 4.4 | Family C avg speedup: 0.08x | benchmark_reports/complete_20260423_182039/fusion_suite.stdout.log | benchmark_reports/complete_20260423_182039/fusion_suite.stderr.log |
| fusion_fair | fusion | PASS | 33.1 | ================================================================================== | benchmark_reports/complete_20260423_182039/fusion_fair.stdout.log | benchmark_reports/complete_20260423_182039/fusion_fair.stderr.log |
| fusion_devtensor | fusion | PASS | 198.8 | ======================================================================================= | benchmark_reports/complete_20260423_182039/fusion_devtensor.stdout.log | benchmark_reports/complete_20260423_182039/fusion_devtensor.stderr.log |
| three_way | fusion | PASS | 8.1 | ================================================================================ | benchmark_reports/complete_20260423_182039/three_way.stdout.log | benchmark_reports/complete_20260423_182039/three_way.stderr.log |
| absurd_activation | fusion | PASS | 2.5 | ================================================================================ | benchmark_reports/complete_20260423_182039/absurd_activation.stdout.log | benchmark_reports/complete_20260423_182039/absurd_activation.stderr.log |
| rsqrt_sin_softmax | fusion | PASS | 15.3 | ================================================================================ | benchmark_reports/complete_20260423_182039/rsqrt_sin_softmax.stdout.log | benchmark_reports/complete_20260423_182039/rsqrt_sin_softmax.stderr.log |
| pow2_extended | stress | PASS | 71.5 | ================================================================================== | benchmark_reports/complete_20260423_182039/pow2_extended.stdout.log | benchmark_reports/complete_20260423_182039/pow2_extended.stderr.log |
