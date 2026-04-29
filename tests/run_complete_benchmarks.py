from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_REPORT_ROOT = REPO_ROOT / "benchmark_reports"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    relpath: str
    timeout_seconds: float
    args: tuple[str, ...] = ()
    category: str = "misc"
    quick_skip: bool = False


@dataclass
class BenchmarkResult:
    name: str
    category: str
    relpath: str
    command: list[str]
    timeout_seconds: float
    started_at: str
    duration_seconds: float
    status: str
    returncode: int | None
    detail: str
    stdout_log: str
    stderr_log: str


SUITE: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="core_suite",
        relpath="tests/run_benchmarks.py",
        timeout_seconds=3600.0,
        category="core",
    ),
    BenchmarkSpec(
        name="device_resident",
        relpath="tests/benchmark_device_resident.py",
        timeout_seconds=1800.0,
        category="device",
    ),
    BenchmarkSpec(
        name="graph_vs_torch",
        relpath="tests/bench_vs_torch.py",
        timeout_seconds=1800.0,
        category="comparison",
    ),
    BenchmarkSpec(
        name="full_comparison",
        relpath="tests/bench_full_comparison.py",
        timeout_seconds=1800.0,
        category="comparison",
    ),
    BenchmarkSpec(
        name="attention",
        relpath="tests/bench_attention.py",
        timeout_seconds=1800.0,
        category="fusion",
    ),
    BenchmarkSpec(
        name="fusion_suite",
        relpath="tests/bench_fusion_suite.py",
        timeout_seconds=2400.0,
        category="fusion",
    ),
    BenchmarkSpec(
        name="fusion_fair",
        relpath="tests/bench_fusion_fair.py",
        timeout_seconds=2400.0,
        category="fusion",
    ),
    BenchmarkSpec(
        name="fusion_devtensor",
        relpath="tests/bench_fusion_devtensor.py",
        timeout_seconds=2400.0,
        category="fusion",
    ),
    BenchmarkSpec(
        name="three_way",
        relpath="tests/benchmark_3way.py",
        timeout_seconds=2400.0,
        category="fusion",
        quick_skip=True,
    ),
    BenchmarkSpec(
        name="absurd_activation",
        relpath="tests/benchmark_absurd_activation.py",
        timeout_seconds=2400.0,
        category="fusion",
        quick_skip=True,
    ),
    BenchmarkSpec(
        name="rsqrt_sin_softmax",
        relpath="tests/benchmark_rsqrt_sin_softmax.py",
        timeout_seconds=2400.0,
        category="fusion",
        quick_skip=True,
    ),
    BenchmarkSpec(
        name="pow2_extended",
        relpath="tests/bench_pow2_extended.py",
        timeout_seconds=3600.0,
        category="stress",
        quick_skip=True,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the broad MatcoreDSL benchmark sweep and archive logs."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the longest standalone scripts and pass --quick to run_benchmarks.py.",
    )
    parser.add_argument(
        "--report-root",
        default=str(DEFAULT_REPORT_ROOT),
        help=f"Directory that will receive the timestamped benchmark report (default: {DEFAULT_REPORT_ROOT}).",
    )
    return parser.parse_args(argv)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def run_spec(
    spec: BenchmarkSpec,
    *,
    python_exe: str,
    quick: bool,
    report_dir: pathlib.Path,
) -> BenchmarkResult:
    script_path = REPO_ROOT / spec.relpath
    args = list(spec.args)
    if spec.name == "core_suite" and quick:
        args.append("--quick")

    command = [python_exe, str(script_path), *args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(REPO_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    started = datetime.now().isoformat(timespec="seconds")
    started_perf = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            check=False,
            env=env,
        )
        duration = time.perf_counter() - started_perf
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode == 0:
            status = "PASS"
            detail = last_nonempty_line(stdout) or "completed"
        elif spec.name == "core_suite" and completed.returncode == 2:
            status = "PASS"
            detail = "completed with advisory warnings"
        else:
            status = "FAIL"
            detail = last_nonempty_line(stderr) or last_nonempty_line(stdout) or f"returncode={completed.returncode}"

        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started_perf
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = "TIMEOUT"
        detail = f"exceeded {spec.timeout_seconds:.0f}s"
        returncode = None

    stdout_log = report_dir / f"{spec.name}.stdout.log"
    stderr_log = report_dir / f"{spec.name}.stderr.log"
    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")

    return BenchmarkResult(
        name=spec.name,
        category=spec.category,
        relpath=spec.relpath,
        command=command,
        timeout_seconds=spec.timeout_seconds,
        started_at=started,
        duration_seconds=duration,
        status=status,
        returncode=returncode,
        detail=detail,
        stdout_log=str(stdout_log.relative_to(REPO_ROOT)),
        stderr_log=str(stderr_log.relative_to(REPO_ROOT)),
    )


def write_summary(report_dir: pathlib.Path, results: list[BenchmarkResult]) -> None:
    summary_json = report_dir / "summary.json"
    summary_md = report_dir / "summary.md"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": [asdict(result) for result in results],
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Complete Benchmark Summary",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Benchmark | Category | Status | Duration (s) | Detail | Stdout | Stderr |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.name} | {result.category} | {result.status} | "
            f"{result.duration_seconds:.1f} | {result.detail.replace('|', '/')} | "
            f"{result.stdout_log} | {result.stderr_log} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_root = pathlib.Path(args.report_root).resolve()
    report_dir = report_root / f"complete_{timestamp_slug()}"
    report_dir.mkdir(parents=True, exist_ok=False)

    print(f"Benchmark report directory: {report_dir}")
    results: list[BenchmarkResult] = []
    python_exe = sys.executable

    for spec in SUITE:
        if args.quick and spec.quick_skip:
            print(f"[SKIP] {spec.name:<20} quick mode")
            results.append(
                BenchmarkResult(
                    name=spec.name,
                    category=spec.category,
                    relpath=spec.relpath,
                    command=[python_exe, str(REPO_ROOT / spec.relpath)],
                    timeout_seconds=spec.timeout_seconds,
                    started_at=datetime.now().isoformat(timespec="seconds"),
                    duration_seconds=0.0,
                    status="SKIP",
                    returncode=None,
                    detail="skipped in quick mode",
                    stdout_log="",
                    stderr_log="",
                )
            )
            continue

        print(f"[RUN ] {spec.name:<20} {spec.relpath}")
        result = run_spec(
            spec,
            python_exe=python_exe,
            quick=args.quick,
            report_dir=report_dir,
        )
        results.append(result)
        print(
            f"[{result.status:<5}] {spec.name:<20} "
            f"{result.duration_seconds:8.1f}s  {result.detail}"
        )

    write_summary(report_dir, results)

    pass_like = {"PASS", "SKIP"}
    if all(result.status in pass_like for result in results):
        print(f"Complete benchmark sweep finished. Summary: {report_dir / 'summary.md'}")
        return 0

    print(f"Complete benchmark sweep finished with failures. Summary: {report_dir / 'summary.md'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
