from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for candidate in (REPO_ROOT, THIS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from gpu_preflight import GpuPreflightReport, run_full_preflight

DEFAULT_SIZE = 256
QUICK_SIZE = 128
DEFAULT_WARMUP = 8
DEFAULT_RUNS = 4
QUICK_WARMUP = 2
QUICK_RUNS = 2
DEFAULT_COMPARE_REPEATS = 8
QUICK_COMPARE_REPEATS = 3
DEFAULT_LOW_PRECISION_REPEATS = 12
QUICK_LOW_PRECISION_REPEATS = 4
CPU_GATE_TIMEOUT_SECONDS = 300.0
CPU_BENCH_TIMEOUT_SECONDS = 600.0
GPU_BENCH_TIMEOUT_SECONDS = 1200.0
MASSIVE_TIMEOUT_SECONDS = 1800.0

GLADIATOR_LINE = re.compile(r"^\[(OK|SKIP|FAIL)\]\s+(.+?)\s+(float16|bfloat16|int8)\s+(.*)$")
NAMED_METRIC_MS = re.compile(r"(cold|warm_mean)=([0-9]+(?:\.[0-9]+)?) ms")
THROUGHPUT = re.compile(r"\[([0-9]+(?:\.[0-9]+)?)\s+(?:TFLOP/s|TOPS)\]")
LOW_PRECISION_LINE = re.compile(r"^(?P<target>[^: ]+)(?::| )(?P<body>.*)$")


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    detail: str


@dataclass
class SummaryEntry:
    status: str
    detail: str = ""


@dataclass
class PreflightEntry:
    summary: SummaryEntry
    run_benchmark: bool
    advisory: bool = False


@dataclass
class GladiatorSummary:
    status: str
    detail: str
    ok_count: int = 0
    cold_ms: float | None = None
    steady_ms: float | None = None
    ongoing_ms: float | None = None
    peak_tflops: float | None = None


def format_summary_line(label: str, entry: SummaryEntry) -> str:
    base = f"{label:<13} {entry.status}"
    if entry.detail:
        return f"{base} ({entry.detail})"
    return base


def format_ms(value: float | None) -> str:
    return f"{value:.3f} ms" if value is not None else "n/a"


def format_tflops(value: float | None) -> str:
    return f"{value:.4f} TFLOP/s" if value is not None else "n/a"


def run_command(command: Sequence[str], *, timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            ok=False,
            command=list(command),
            returncode=-1,
            stdout="",
            stderr="",
            detail=f"timeout after {timeout:.0f}s",
        )
    except Exception as exc:
        return CommandResult(
            ok=False,
            command=list(command),
            returncode=-1,
            stdout="",
            stderr="",
            detail=f"{type(exc).__name__}: {exc}",
        )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    ok = completed.returncode == 0
    if ok:
        detail = ""
    else:
        detail = stderr.splitlines()[-1] if stderr else stdout.splitlines()[-1] if stdout else f"returncode={completed.returncode}"
    return CommandResult(
        ok=ok,
        command=list(command),
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        detail=detail,
    )


def parse_compare(result: CommandResult, target: str) -> SummaryEntry:
    target_prefix = f"matcore[{target}]"
    if not result.ok:
        return SummaryEntry("FAIL", result.detail or f"{target_prefix} exited with {result.returncode}")

    lines = result.stdout.splitlines()
    failures = [line.strip() for line in lines if line.strip().startswith(target_prefix) and "FAILED" in line]
    successes = [line.strip() for line in lines if line.strip().startswith(target_prefix) and "FAILED" not in line]
    if failures:
        return SummaryEntry("FAIL", failures[0].split(":", 1)[-1].strip())
    if not successes:
        return SummaryEntry("FAIL", f"no success line for {target}")
    return SummaryEntry("PASS", successes[-1].split(":", 1)[-1].strip())


def parse_low_precision(result: CommandResult, target: str) -> SummaryEntry:
    if not result.ok:
        return SummaryEntry("FAIL", result.detail or f"returncode={result.returncode}")

    relevant: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        match = LOW_PRECISION_LINE.match(line)
        if match is None or match.group("target") != target:
            continue
        relevant.append(line)

    failures = [line for line in relevant if "FAILED" in line]
    skips = [line for line in relevant if "SKIP" in line]
    successes = [line for line in relevant if line not in failures and line not in skips]
    if failures:
        return SummaryEntry("FAIL", failures[0].split(":", 1)[-1].strip())
    if successes:
        return SummaryEntry("PASS", f"{len(successes)} low-precision runs")
    if skips:
        return SummaryEntry("SKIP", skips[0].split(":", 1)[-1].strip())
    return SummaryEntry("FAIL", f"no output for {target}")


def _extract_metric(metrics_text: str, name: str) -> float | None:
    for metric_name, value in NAMED_METRIC_MS.findall(metrics_text):
        if metric_name == name:
            return float(value)
    return None


def parse_gladiator(result: CommandResult) -> GladiatorSummary:
    if not result.ok:
        return GladiatorSummary("FAIL", result.detail or f"returncode={result.returncode}")

    fail_lines: list[str] = []
    ok_count = 0
    cold_ms: float | None = None
    steady_ms: float | None = None
    ongoing_ms: float | None = None
    peak_tflops: float | None = None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        match = GLADIATOR_LINE.match(line)
        if match is None:
            continue
        status, prefix, _dtype, metrics = match.groups()
        if status == "FAIL":
            fail_lines.append(f"{prefix}: {metrics}")
            continue
        if status != "OK":
            continue
        ok_count += 1
        if not prefix.startswith("MatCore["):
            continue
        if prefix == "MatCore[cold]":
            cold_ms = _extract_metric(metrics, "cold")
        elif prefix == "MatCore[steady]":
            steady_ms = _extract_metric(metrics, "warm_mean")
        elif prefix == "MatCore[ongoing]":
            ongoing_ms = _extract_metric(metrics, "warm_mean")
        throughput_match = THROUGHPUT.search(metrics)
        if throughput_match is not None:
            parsed = float(throughput_match.group(1))
            peak_tflops = parsed if peak_tflops is None else max(peak_tflops, parsed)

    if fail_lines:
        return GladiatorSummary("FAIL", fail_lines[0], ok_count=ok_count)
    if cold_ms is None or steady_ms is None or ongoing_ms is None:
        if ok_count == 0:
            return GladiatorSummary("SKIP", "no successful gladiator results")
        return GladiatorSummary("FAIL", "missing MatCore cold/steady/ongoing metrics", ok_count=ok_count)
    return GladiatorSummary(
        "PASS",
        f"cold={format_ms(cold_ms)}, steady={format_ms(steady_ms)}, ongoing={format_ms(ongoing_ms)}, peak={format_tflops(peak_tflops)}",
        ok_count=ok_count,
        cold_ms=cold_ms,
        steady_ms=steady_ms,
        ongoing_ms=ongoing_ms,
        peak_tflops=peak_tflops,
    )


def parse_massive(result: CommandResult) -> SummaryEntry:
    if not result.ok:
        return SummaryEntry("FAIL", result.detail or f"returncode={result.returncode}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    failures = [line for line in lines if ": FAILED" in line]
    successes = [line for line in lines if "(ok, checksum=" in line]
    if failures:
        return SummaryEntry("FAIL", failures[0].split(": ", 1)[-1])
    if not successes:
        return SummaryEntry("FAIL", "no successful target results")
    return SummaryEntry("PASS", f"{len(successes)} target(s)")


def preflight_entry(label: str, report: GpuPreflightReport) -> PreflightEntry:
    detail_parts: list[str] = []
    device_name = ""
    driver = ""
    if label == "NVIDIA":
        device_name = report.nvidia.device_name if report.nvidia else ""
        driver = report.nvidia.driver_version if report.nvidia else ""
    elif label == "AMD":
        device_name = report.amd.device_name if report.amd else ""
    if device_name:
        detail_parts.append(device_name)
    if driver:
        detail_parts.append(f"driver {driver}")

    advisory = bool(report.dmesg and (not report.dmesg.clean or report.dmesg.permission_denied))
    if advisory:
        advisory_note = "kernel log advisory"
        if report.dmesg and report.dmesg.permission_denied and report.dmesg.clean:
            advisory_note = "kernel log access advisory"
        detail_parts.append(advisory_note)

    if report.overall_ok:
        detail = ", ".join(detail_parts) if detail_parts else report.summary
        return PreflightEntry(SummaryEntry("PASS", detail), run_benchmark=True, advisory=advisory)

    failure_detail = report.summary
    if label == "NVIDIA" and report.nvidia and report.nvidia.diagnostics:
        failure_detail = report.nvidia.diagnostics
    if label == "AMD" and report.amd and report.amd.diagnostics:
        failure_detail = report.amd.diagnostics
    return PreflightEntry(SummaryEntry("SKIP", failure_detail), run_benchmark=False, advisory=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MatcoreDSL benchmark suite")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip NVIDIA and AMD GPU benchmarks")
    parser.add_argument("--skip-massive", action="store_true", help="Skip the massive benchmark")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"Gladiator matrix size (default: {DEFAULT_SIZE})")
    parser.add_argument("--quick", action="store_true", help="Use smaller/faster benchmark settings for CI")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bench_size = args.size if not args.quick else min(args.size, QUICK_SIZE)
    warmup = QUICK_WARMUP if args.quick else DEFAULT_WARMUP
    runs = QUICK_RUNS if args.quick else DEFAULT_RUNS
    compare_repeats = QUICK_COMPARE_REPEATS if args.quick else DEFAULT_COMPARE_REPEATS
    low_precision_repeats = QUICK_LOW_PRECISION_REPEATS if args.quick else DEFAULT_LOW_PRECISION_REPEATS
    python_exe = sys.executable

    print("== Phase 1: CPU Gate ==")
    cpu_gate_cmd = [
        python_exe,
        str(THIS_DIR / "benchmark_compare.py"),
        "--targets",
        "x86-avx2",
        "--m",
        str(min(bench_size, 128)),
        "--k",
        str(min(bench_size, 128)),
        "--n",
        str(min(bench_size, 128)),
        "--repeats",
        str(compare_repeats),
    ]
    cpu_gate_result = run_command(cpu_gate_cmd, timeout=CPU_GATE_TIMEOUT_SECONDS)
    cpu_gate_entry = parse_compare(cpu_gate_result, "x86-avx2")
    print(format_summary_line("CPU Gate:", cpu_gate_entry))
    if cpu_gate_entry.status != "PASS":
        print("CPU gate failed; aborting remaining benchmarks.", file=sys.stderr)
        print("\n=== MatcoreDSL Benchmark Suite ===")
        print(format_summary_line("CPU Gate:", cpu_gate_entry))
        print(format_summary_line("Overall:", SummaryEntry("FAIL", "CPU gate failed")))
        return 1

    advisory = False
    if args.skip_gpu:
        nvidia_pre = PreflightEntry(SummaryEntry("SKIP", "disabled by --skip-gpu"), run_benchmark=False)
        amd_pre = PreflightEntry(SummaryEntry("SKIP", "disabled by --skip-gpu"), run_benchmark=False)
    else:
        print("\n== Phase 2: GPU Preflight ==")
        nvidia_pre = preflight_entry("NVIDIA", run_full_preflight("nvidia"))
        amd_pre = preflight_entry("AMD", run_full_preflight("amd-igpu"))
        print(format_summary_line("NVIDIA Pre:", nvidia_pre.summary))
        print(format_summary_line("AMD Pre:", amd_pre.summary))
        advisory = nvidia_pre.advisory or amd_pre.advisory

    print("\n== Phase 3: Benchmarks ==")
    failures = False

    cpu_low_cmd = [
        python_exe,
        str(THIS_DIR / "benchmark_low_precision.py"),
        "--targets",
        "x86-avx2",
        "--m",
        str(min(bench_size, 64)),
        "--k",
        str(min(bench_size, 64)),
        "--n",
        str(min(bench_size, 64)),
        "--repeats",
        str(low_precision_repeats),
    ]
    cpu_gladiator_cmd = [
        python_exe,
        str(THIS_DIR / "benchmark_gladiator.py"),
        "--arena",
        "cpu",
        "--matcore-target",
        "x86-avx2",
        "--size",
        str(bench_size),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
    ]
    cpu_low_entry = parse_low_precision(run_command(cpu_low_cmd, timeout=CPU_BENCH_TIMEOUT_SECONDS), "x86-avx2")
    cpu_gladiator_entry = parse_gladiator(run_command(cpu_gladiator_cmd, timeout=CPU_BENCH_TIMEOUT_SECONDS))
    if cpu_low_entry.status == "FAIL" or cpu_gladiator_entry.status == "FAIL":
        failures = True
        detail = cpu_low_entry.detail if cpu_low_entry.status == "FAIL" else cpu_gladiator_entry.detail
        cpu_bench_entry = SummaryEntry("FAIL", detail)
    elif cpu_low_entry.status == "PASS" and cpu_gladiator_entry.status == "PASS":
        cpu_bench_entry = SummaryEntry("PASS", "2 commands")
    else:
        skip_detail = cpu_low_entry.detail if cpu_low_entry.status == "SKIP" else cpu_gladiator_entry.detail
        cpu_bench_entry = SummaryEntry("SKIP", skip_detail or "CPU benchmark coverage incomplete")

    if nvidia_pre.run_benchmark:
        nvidia_cmd = [
            python_exe,
            str(THIS_DIR / "benchmark_gladiator.py"),
            "--arena",
            "nvidia",
            "--size",
            str(bench_size),
            "--warmup",
            str(warmup),
            "--runs",
            str(runs),
        ]
        nvidia_gladiator = parse_gladiator(run_command(nvidia_cmd, timeout=GPU_BENCH_TIMEOUT_SECONDS))
        nvidia_entry = SummaryEntry(nvidia_gladiator.status, nvidia_gladiator.detail)
        failures = failures or nvidia_entry.status == "FAIL"
    else:
        nvidia_entry = SummaryEntry("SKIP", nvidia_pre.summary.detail or "preflight failed")

    if amd_pre.run_benchmark:
        amd_cmd = [
            python_exe,
            str(THIS_DIR / "benchmark_gladiator.py"),
            "--arena",
            "amd-igpu",
            "--size",
            str(bench_size),
            "--warmup",
            str(warmup),
            "--runs",
            str(runs),
        ]
        amd_gladiator = parse_gladiator(run_command(amd_cmd, timeout=GPU_BENCH_TIMEOUT_SECONDS))
        amd_entry = SummaryEntry(amd_gladiator.status, amd_gladiator.detail)
        failures = failures or amd_entry.status == "FAIL"
    else:
        amd_entry = SummaryEntry("SKIP", amd_pre.summary.detail or "preflight failed")

    if args.skip_massive:
        massive_entry = SummaryEntry("SKIP", "disabled by --skip-massive")
    else:
        massive_targets = ["x86-avx2"]
        if not args.skip_gpu and nvidia_pre.run_benchmark:
            massive_targets.append("nvidia-dgpu")
        if not args.skip_gpu and amd_pre.run_benchmark:
            massive_targets.append("amd-igpu")
        massive_cmd = [python_exe, str(THIS_DIR / "benchmark_massive.py"), "--targets", *massive_targets]
        massive_entry = parse_massive(run_command(massive_cmd, timeout=MASSIVE_TIMEOUT_SECONDS))
        failures = failures or massive_entry.status == "FAIL"

    overall_entry = SummaryEntry("PASS")
    exit_code = 0
    if failures or cpu_bench_entry.status == "FAIL":
        overall_entry = SummaryEntry("FAIL")
        exit_code = 1
    elif advisory:
        overall_entry = SummaryEntry("PASS", "advisory warnings only")
        exit_code = 2

    print("\n=== MatcoreDSL Benchmark Suite ===")
    print(format_summary_line("CPU Gate:", cpu_gate_entry))
    print(format_summary_line("NVIDIA Pre:", nvidia_pre.summary))
    print(format_summary_line("AMD Pre:", amd_pre.summary))
    print(format_summary_line("CPU Bench:", cpu_bench_entry))
    print(format_summary_line("NVIDIA Bench:", nvidia_entry))
    print(format_summary_line("AMD Bench:", amd_entry))
    print(format_summary_line("Massive:", massive_entry))
    print(format_summary_line("Overall:", overall_entry))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
