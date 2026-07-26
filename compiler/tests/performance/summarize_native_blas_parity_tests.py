#!/usr/bin/env python3

"""Adversarial contract tests for the native-BLAS parity summarizer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import tempfile


def load_module(name: str, path: pathlib.Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run(
    command: list[str], expected: int = 0
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: "
            f"{command}\nstdout:\n{completed.stdout}\nstderr:\n"
            f"{completed.stderr}"
        )
    return completed


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(seconds: float) -> list[float]:
    return (
        [
            seconds * factor
            for factor in (0.97, 0.98, 0.99, 0.995, 1.0)
        ]
        + [seconds]
        + [
            seconds * factor
            for factor in (1.005, 1.01, 1.02, 1.03, 1.04)
        ]
    )


def variant_seconds(
    summary: object,
    case: object,
    order_factor: float,
) -> float:
    m, n, k = case.shape
    base = max(2.0 * m * n * k / 100.0e9, 1.0e-5)
    threads = case.threads
    if threads == 1:
        parallel_speedup = 1.0
    elif threads == 4:
        parallel_speedup = 3.5
    else:
        parallel_speedup = max(2.0, min(float(threads) * 0.72, 9.0))
    provider = base / parallel_speedup
    ratios = {
        "cpu.native-packed.avx2-fma.f32.v1": 0.96,
        "cpu.native-packed.avx512-fma.f32.v1": 1.05,
        "cpu.native-parallel.avx2-fma.f32.v1": 1.00,
        "cpu.native-parallel.avx512-fma.f32.v1": 1.10,
        summary.OPENBLAS: 1.0,
        summary.AUTO: 1.0,
    }
    return provider / ratios[case.variant] * order_factor


def planner_regret(
    summary: object,
    selected_variant: str,
    seconds: float,
    requested_threads: int,
) -> dict:
    selected_samples = samples(seconds)
    slower_samples = samples(seconds * 1.08)

    def candidate(variant: str, values: list[float]) -> dict:
        ordered = sorted(values)
        middle = ordered[len(ordered) // 2]
        return {
            "variant": variant,
            "selected_variant": variant,
            "legal": True,
            "plan_authenticated": True,
            "complete_implementation_comparison": True,
            "timing_valid": True,
            "correctness": True,
            "actual_threads": requested_threads,
            "smt_policy": (
                "allow-smt"
                if requested_threads > 1
                else "physical-cores-only"
            ),
            "affinity_policy": (
                "none" if requested_threads > 1 else "compact"
            ),
            "worker_affinity_applied": requested_threads == 1,
            "worker_affinity_user_requested": False,
            "worker_affinity_policy_induced": requested_threads == 1,
            "forward_pass_normalized_samples_seconds": values,
            "reverse_pass_normalized_samples_seconds": values,
            "forward_pass_median_seconds": middle,
            "reverse_pass_median_seconds": middle,
            "forward_pass_untimed_validation_executions_checked": 1,
            "reverse_pass_untimed_validation_executions_checked": 1,
            "balanced_estimate_seconds": middle,
        }

    candidates = [
        candidate(selected_variant, selected_samples),
        candidate(
            "cpu.native-packed.avx2-fma.f32.v1",
            slower_samples,
        ),
    ]
    return {
        "requested": True,
        "valid": True,
        "aggregation_method":
        "arithmetic-mean-of-forward-and-reverse-pass-medians",
        "fastest_legal_variant": selected_variant,
        "fastest_legal_balanced_estimate_seconds": seconds,
        "selected_balanced_estimate_seconds": seconds,
        "regret": 1.0,
        "reason": "synthetic authenticated contract fixture",
        "candidates": candidates,
    }


def environment(source_commit: str, physical_cores: int) -> dict:
    return {
        "source_provenance_state": "clean",
        "source_worktree_dirty": False,
        "source_provenance_origin": "git-worktree",
        "source_commit": source_commit,
        "os_family": "linux",
        "architecture": "x86_64",
        "topology_discovery_complete": True,
        "physical_cores": physical_cores,
        "compiler": "Clang 21.1.8",
        "compiler_flags": "-O3",
        "build_type": "Release",
        "cpu_model": "contract-test-cpu",
        "governor": "performance",
        "frequency_policy": "contract-test",
        "boost_state": "enabled",
        "cpu_affinity": "contract-test",
        "hardware_threads": physical_cores * 2,
        "timer_source": "steady_clock",
        "timer_resolution_ns": 1,
        "capability_record": "matcore.cpu.capabilities.v2",
        "capability_runtime_validation_source": "contract-test",
        "capability_record_version": 2,
        "topology_record_version": 1,
        "logical_processors": physical_cores * 2,
        "numa_nodes": 1,
        "persistent_execution_context": True,
        "available_processors": physical_cores * 2,
        "provider_name": "OpenBLAS",
        "provider_version": "0.3.32",
        "provider_config": "USE_THREAD=1",
    }


def raw_report(
    summary: object,
    runner: object,
    case: object,
    source_commit: str,
    physical_cores: int,
    order_factor: float,
) -> dict:
    seconds = variant_seconds(summary, case, order_factor)
    ordered_samples = samples(seconds)
    ordered = sorted(ordered_samples)
    actual_threads = case.threads
    selected_variant = (
        summary.OPENBLAS if case.variant == summary.AUTO else case.variant
    )
    requested_regret = case.mode == "planner-regret-hot"
    preparation_seconds = seconds * 0.25
    preparation = {
        "requested": case.mode == "prepacked-b-hot",
        "measured": case.mode == "prepacked-b-hot",
        "authenticated": case.mode == "prepacked-b-hot",
        "preparation_calls": 1 if case.mode == "prepacked-b-hot" else 0,
        "amortization_executions": case.lhs_sequence,
        "amortized_total_valid": case.mode == "prepacked-b-hot",
        "input_state": (
            "caller-storage-allocated-unprepared"
            if case.mode == "prepacked-b-hot"
            else "not-requested"
        ),
        "output_state": (
            "prepared-authenticated"
            if case.mode == "prepacked-b-hot"
            else "not-requested"
        ),
        "preparation_seconds": (
            preparation_seconds
            if case.mode == "prepacked-b-hot"
            else 0.0
        ),
        "steady_state_sequence_seconds": (
            seconds * case.lhs_sequence
            if case.mode == "prepacked-b-hot"
            else 0.0
        ),
        "amortized_total_sequence_seconds": (
            preparation_seconds + seconds * case.lhs_sequence
            if case.mode == "prepacked-b-hot"
            else 0.0
        ),
        "amortized_per_execution_seconds": (
            seconds + preparation_seconds / case.lhs_sequence
            if case.mode == "prepacked-b-hot"
            else 0.0
        ),
    }
    task_count = (
        summary.parallel_task_capacity(case.shape, case.threads)
        if case.variant in summary.PARALLEL_NATIVE_VARIANTS
        else 0
    )
    aggregate_repetitions = max(
        1,
        math.ceil(
            summary.TIMER_FLOOR_US
            * 1000
            / (ordered[0] * 1.0e9)
        ),
    )
    m, n, k = case.shape
    result = {
        "m": m,
        "n": n,
        "k": k,
        "requested_variant": case.variant,
        "selected_variant": selected_variant,
        "planner_mode": (
            "automatic" if case.variant == summary.AUTO else "forced"
        ),
        "complete_implementation_comparison": True,
        "timing_valid": True,
        "correctness": True,
        "timed_final_output_authenticated": True,
        "untimed_validation_executions_checked": 1,
        "smt_policy": (
            "allow-smt" if case.threads > 1 else "physical-cores-only"
        ),
        "affinity_policy": "none" if case.threads > 1 else "compact",
        "worker_affinity_applied": case.threads == 1,
        "worker_affinity_user_requested": False,
        "worker_affinity_policy_induced": case.threads == 1,
        "actual_threads": actual_threads,
        "parallel_row_tasks": task_count if task_count else 0,
        "parallel_column_tasks": 1 if task_count else 0,
        "parallel_task_count": task_count,
        "normalized_samples_seconds": ordered_samples,
        "aggregate_repetitions": aggregate_repetitions,
        "timing_aggregation_boundary":
        "one-clock-pair-per-aggregate-repetition-block",
        "timing_rejection_reason": "",
        "minimum_seconds": ordered[0],
        "median_seconds": ordered[len(ordered) // 2],
        "p95_seconds": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "gflops": 2.0 * m * n * k / seconds / 1.0e9,
        "checksum": float(m + n + k),
        "expected_checksum": float(m + n + k),
        "maximum_absolute_error": 0.0,
        "maximum_allowed_error": 1.0e-3,
        "prepacked_b_preparation": preparation,
        "planner_regret": (
            planner_regret(
                summary, selected_variant, seconds, case.threads
            )
            if requested_regret
            else {"requested": False}
        ),
    }
    return {
        "schema": summary.BENCHMARK_SCHEMA,
        "version": summary.BENCHMARK_VERSION,
        "operation": "matcore.gemm",
        "dtype": "f32",
        "accumulation_dtype": "f32",
        "layout": "row-major-contiguous",
        "environment": environment(source_commit, physical_cores),
        "configuration": runner.expected_configuration(case),
        "results": [result],
    }


def write_bundle(
    root: pathlib.Path,
    order: str,
    summary: object,
    runner: object,
    benchmark: pathlib.Path,
    source_commit: str,
    runner_path: pathlib.Path,
    physical_cores: int,
) -> pathlib.Path:
    directory = root / order
    directory.mkdir()
    cases = summary.expected_cases(runner, physical_cores)
    if order == "stable-reverse":
        cases.reverse()
    records = []
    order_factor = 0.99 if order == "stable-forward" else 1.01
    for index, case in enumerate(cases):
        raw_name = f"{index:04d}__{case.key}.json"
        raw_path = directory / raw_name
        document = raw_report(
            summary,
            runner,
            case,
            source_commit,
            physical_cores,
            order_factor,
        )
        raw_path.write_text(
            json.dumps(document, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "index": index,
                "key": case.key,
                "partition": case.partition,
                "family": case.family,
                "shape": list(case.shape),
                "variant": case.variant,
                "threads": case.threads,
                "mode": case.mode,
                "lhs_sequence": case.lhs_sequence,
                "command": runner.case_command(benchmark, case, raw_path),
                "raw_file": raw_name,
                "state": "passed",
                "sha256": sha256(raw_path),
                "selected_variant": document["results"][0][
                    "selected_variant"
                ],
                "actual_threads": document["results"][0]["actual_threads"],
            }
        )
    parallel_plan = summary.expected_parallel_thread_plan(
        runner, physical_cores
    )
    manifest = {
        "schema": summary.MANIFEST_SCHEMA,
        "version": summary.MANIFEST_VERSION,
        "benchmark_schema_version": summary.BENCHMARK_VERSION,
        "benchmark": str(benchmark),
        "benchmark_binary_sha256": sha256(benchmark),
        "runner": str(runner_path),
        "runner_sha256": sha256(runner_path),
        "runner_git_blob": run(
            ["git", "-C", str(runner_path.parents[3]), "hash-object",
             str(runner_path)]
        ).stdout.strip(),
        "source_commit": source_commit,
        "plan_sha256": summary.reconstructed_plan_digest(
            cases,
            physical_cores,
            order,
            benchmark,
            parallel_plan,
        ),
        "benchmark_seed": summary.BENCHMARK_SEED,
        "started_unix_seconds": 100,
        "finished_unix_seconds": 200,
        "suites": sorted(summary.REQUIRED_SUITES),
        "physical_cores": physical_cores,
        "thread_strata": list(summary.thread_strata(physical_cores)),
        "parallel_thread_plan": parallel_plan,
        "partition_interpretation": summary.PARTITION_INTERPRETATION,
        "case_order": order,
        "warmup": summary.WARMUP_ITERATIONS,
        "iterations": summary.MEASURED_ITERATIONS,
        "timer_floor_us": summary.TIMER_FLOOR_US,
        "max_memory_mib": summary.MAX_MEMORY_MIB,
        "limit": 0,
        "dry_run": False,
        "environment_overrides": summary.PROVIDER_ENVIRONMENT,
        "cases": records,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def execute_summary(
    summarizer: pathlib.Path,
    forward: pathlib.Path,
    reverse: pathlib.Path,
    output_root: pathlib.Path,
    expected: int = 0,
    require_pass: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(summarizer),
        "--forward-manifest",
        str(forward),
        "--reverse-manifest",
        str(reverse),
        "--markdown-out",
        str(output_root / "summary.md"),
        "--json-out",
        str(output_root / "summary.json"),
    ]
    if require_pass:
        command.append("--require-pass")
    return run(command, expected=expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarizer", required=True)
    args = parser.parse_args()
    summarizer_path = pathlib.Path(args.summarizer).resolve()
    summary = load_module("matcore_native_parity_summary", summarizer_path)
    runner, runner_path = summary.load_runner()
    assert summary.MANIFEST_VERSION == 3
    assert summary.SUMMARY_VERSION == 2
    assert runner.PARTITION_INTERPRETATION == (
        summary.PARTITION_INTERPRETATION
    )
    repository = runner_path.parents[3]
    source_commit = run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"]
    ).stdout.strip()
    physical_cores = 12

    assert summary.parallel_task_capacity((512, 512, 512), 12) == 8
    assert summary.exact_parallel_thread_strata(
        (512, 512, 512), 12
    ) == (4, 8)
    assert summary.parallel_task_capacity((384, 384, 384), 12) == 3
    assert summary.parallel_task_capacity((1024, 1024, 1024), 12) == 12
    assert len(summary.expected_cases(runner, 12)) > 300

    with tempfile.TemporaryDirectory(
        prefix="matcore native parity summary "
    ) as temporary:
        root = pathlib.Path(temporary)
        benchmark = root / "matcore-bench"
        benchmark.write_bytes(b"authenticated fake benchmark\n")
        forward = write_bundle(
            root,
            "stable-forward",
            summary,
            runner,
            benchmark,
            source_commit,
            runner_path,
            physical_cores,
        )
        reverse = write_bundle(
            root,
            "stable-reverse",
            summary,
            runner,
            benchmark,
            source_commit,
            runner_path,
            physical_cores,
        )
        output = root / "sanitized"
        output.mkdir()
        execute_summary(
            summarizer_path,
            forward,
            reverse,
            output,
            require_pass=True,
        )
        summary_json = json.loads(
            (output / "summary.json").read_text(encoding="utf-8")
        )
        assert summary_json["schema"] == summary.SUMMARY_SCHEMA
        assert summary_json["version"] == 2
        assert summary_json["partition_interpretation"] == {
            "calibration": "candidate-development-and-validation",
            "holdout": "declared-validation-not-blind",
        }
        assert summary_json["verdict"] == "passed"
        assert all(item["passed"] for item in summary_json["criteria"])
        assert summary_json["coverage"]["missing_comparisons"] == []
        first_markdown = (output / "summary.md").read_bytes()
        assert b"validation-not-blind" in first_markdown
        assert b"No unbiased holdout claim is made" in first_markdown
        assert b"manifest v3; benchmark v6; sanitized summary v2" in first_markdown
        assert b"## Measurement contract" in first_markdown
        assert b"## Planner regret" in first_markdown
        assert b"## Prepacked-B repeated execution" in first_markdown
        assert b"## Milestone 6 comparison" in first_markdown
        assert b"## Weakest measured cells" in first_markdown
        assert b"## Claims supported" in first_markdown
        assert b"## Claims explicitly unsupported" in first_markdown
        first_json = (output / "summary.json").read_bytes()
        execute_summary(
            summarizer_path,
            forward,
            reverse,
            output,
            require_pass=True,
        )
        assert (output / "summary.md").read_bytes() == first_markdown
        assert (output / "summary.json").read_bytes() == first_json

        forward_manifest = json.loads(forward.read_text(encoding="utf-8"))
        reverse_manifest = json.loads(reverse.read_text(encoding="utf-8"))

        # Raw tampering is rejected even when the JSON remains parseable.
        first_raw = forward.parent / forward_manifest["cases"][0]["raw_file"]
        pristine_raw = first_raw.read_bytes()
        first_raw.write_bytes(pristine_raw + b" ")
        tampered = execute_summary(
            summarizer_path, forward, reverse, output, expected=2
        )
        assert "raw parity SHA-256 mismatch" in tampered.stderr
        first_raw.write_bytes(pristine_raw)

        # Updating the manifest's raw digest does not authorize internally
        # inconsistent timing fields.
        timing_tamper = json.loads(pristine_raw.decode("utf-8"))
        timing_tamper["results"][0]["median_seconds"] *= 2.0
        first_raw.write_text(
            json.dumps(timing_tamper, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        forward_manifest["cases"][0]["sha256"] = sha256(first_raw)
        forward.write_text(
            json.dumps(forward_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reconstructed = execute_summary(
            summarizer_path, forward, reverse, output, expected=2
        )
        assert "median_seconds does not reconstruct" in reconstructed.stderr
        first_raw.write_bytes(pristine_raw)
        forward_manifest["cases"][0]["sha256"] = sha256(first_raw)
        forward.write_text(
            json.dumps(forward_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Selective omission remains invalid even if an attacker changes the
        # manifest's plan digest to describe the shortened list.
        omitted = copy.deepcopy(forward_manifest)
        omitted["cases"].pop()
        omitted["plan_sha256"] = summary.canonical_sha256(
            {"attacker": "selective omission"}
        )
        forward.write_text(
            json.dumps(omitted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selective = execute_summary(
            summarizer_path, forward, reverse, output, expected=2
        )
        assert "complete frozen v3 matrix" in selective.stderr
        forward.write_text(
            json.dumps(forward_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # An automatic cell may legally use fewer threads than its ceiling,
        # but the paired order must report the same execution semantics.
        reverse_auto = next(
            record
            for record in reverse_manifest["cases"]
            if record["variant"] == summary.AUTO
            and record["threads"] == 4
            and record["mode"] == "auto-complete-hot"
        )
        reverse_raw = reverse.parent / reverse_auto["raw_file"]
        pristine_reverse_raw = reverse_raw.read_bytes()
        reverse_document = json.loads(
            pristine_reverse_raw.decode("utf-8")
        )
        reverse_document["results"][0]["actual_threads"] = 2
        reverse_raw.write_text(
            json.dumps(reverse_document, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reverse_auto["actual_threads"] = 2
        reverse_auto["sha256"] = sha256(reverse_raw)
        reverse.write_text(
            json.dumps(reverse_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unequal = execute_summary(
            summarizer_path, forward, reverse, output, expected=2
        )
        assert "execution semantics differ" in unequal.stderr, unequal.stderr
        reverse_raw.write_bytes(pristine_reverse_raw)
        reverse_manifest = json.loads(
            (reverse.parent / "manifest.json").read_text(encoding="utf-8")
        )
        reverse_auto = next(
            record
            for record in reverse_manifest["cases"]
            if record["variant"] == summary.AUTO
            and record["threads"] == 4
            and record["mode"] == "auto-complete-hot"
        )
        reverse_auto["actual_threads"] = 4
        reverse_auto["sha256"] = sha256(reverse_raw)
        reverse.write_text(
            json.dumps(reverse_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # A runner-digest claim cannot be changed independently from source.
        runner_tamper = copy.deepcopy(forward_manifest)
        runner_tamper["runner_sha256"] = "0" * 64
        forward.write_text(
            json.dumps(runner_tamper, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_runner = execute_summary(
            summarizer_path, forward, reverse, output, expected=2
        )
        assert "runner SHA-256 differs" in rejected_runner.stderr
        forward.write_text(
            json.dumps(forward_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # The executable bytes remain part of both bundle identities.
        pristine_benchmark = benchmark.read_bytes()
        benchmark.write_bytes(pristine_benchmark + b"tamper\n")
        rejected_bench = execute_summary(
            summarizer_path, forward, reverse, output, expected=2
        )
        assert "benchmark binary SHA-256 differs" in rejected_bench.stderr

    print("matcore native BLAS parity summary contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
