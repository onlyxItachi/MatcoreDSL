#!/usr/bin/env python3

"""Synthetic fail-closed contract tests for the deep-audit summarizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile


SEED = 0x4D4154434F524531
SOURCE_COMMIT = "a" * 40
NATIVE = "cpu.native-packed.avx2-fma.f32.v1"
EXTERNAL = "cpu.external.openblas.f32.v1"
PARALLEL = "cpu.native-parallel.avx2-fma.f32.v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def case_key(record: dict) -> str:
    m, n, k = record["shape"]
    encoded = record["variant"].replace(".", "_").replace("-", "_")
    return (
        f"{record['family']}__{m}x{n}x{k}__{encoded}"
        f"__t{record['threads']}__{record['mode']}__lhs{record['lhs_sequence']}"
    )


def false_preparation() -> dict:
    return {
        "requested": False,
        "measured": False,
        "authenticated": False,
        "preparation_calls": 0,
        "amortization_executions": 0,
        "amortized_total_valid": False,
    }


def candidate(variant: str, seconds: float) -> dict:
    samples = [seconds * 0.9, seconds, seconds * 1.1]
    return {
        "variant": variant,
        "selected_variant": variant,
        "legal": True,
        "timing_valid": True,
        "correctness": True,
        "actual_threads": 1,
        "smt_policy": "physical-cores-only",
        "affinity_policy": "compact",
        "worker_affinity_applied": True,
        "worker_affinity_user_requested": True,
        "worker_affinity_policy_induced": False,
        "forward_pass_normalized_samples_seconds": samples,
        "reverse_pass_normalized_samples_seconds": samples,
        "forward_pass_median_seconds": seconds,
        "reverse_pass_median_seconds": seconds,
        "balanced_estimate_seconds": seconds,
    }


def planner_regret(selected_seconds: float, fastest_seconds: float) -> dict:
    selected = candidate(NATIVE, selected_seconds)
    fastest = candidate(EXTERNAL, fastest_seconds)
    return {
        "requested": True,
        "valid": True,
        "fastest_legal_variant": EXTERNAL,
        "fastest_legal_balanced_estimate_seconds": fastest_seconds,
        "selected_balanced_estimate_seconds": selected_seconds,
        "regret": selected_seconds / fastest_seconds,
        "candidates": [selected, fastest],
    }


def write_raw(
    path: pathlib.Path,
    record: dict,
    seconds: float,
    preparation_seconds: float = 0.0,
) -> None:
    m, n, k = record["shape"]
    samples = [seconds * 0.9, seconds, seconds * 1.1]
    mode = record["mode"]
    cache = "cold" if mode == "complete-cold" else "hot"
    allocation = (
        "include-allocation" if mode == "one-shot-hot" else "reuse-workspace"
    )
    packing = (
        "exclude-packing"
        if mode == "compute-only-hot"
        else "prepacked-b"
        if mode == "prepacked-b-hot"
        else "include-packing"
    )
    if mode == "prepacked-b-hot":
        sequence = record["lhs_sequence"]
        steady = seconds * sequence
        total = preparation_seconds + steady
        preparation = {
            "requested": True,
            "measured": True,
            "authenticated": True,
            "preparation_calls": 1,
            "amortization_executions": sequence,
            "amortized_total_valid": True,
            "preparation_seconds": preparation_seconds,
            "steady_state_sequence_seconds": steady,
            "amortized_total_sequence_seconds": total,
            "amortized_per_execution_seconds": total / sequence,
        }
    else:
        preparation = false_preparation()
    planner = (
        planner_regret(seconds, seconds / 2.0)
        if mode == "planner-regret-hot"
        else {"requested": False}
    )
    report = {
        "schema": "matcore.benchmark.cpu.gemm",
        "version": 6,
        "operation": "matcore.gemm",
        "dtype": "f32",
        "accumulation_dtype": "f32",
        "layout": "row-major-contiguous",
        "environment": {
            "os_family": "Linux",
            "architecture": "x86_64",
            "compiler": "clang 21.1.8 synthetic",
            "compiler_flags": "-O3",
            "build_type": "Release",
            "cpu_model": "Synthetic CPU",
            "governor": "performance",
            "frequency_policy": "fixed",
            "boost_state": "disabled",
            "capability_record": "synthetic-avx2",
            "topology_record": "synthetic-1c",
            "capability_record_version": 2,
            "topology_record_version": 1,
            "source_commit": SOURCE_COMMIT,
            "source_provenance_state": "clean",
            "source_worktree_dirty": False,
        },
        "configuration": {
            "requested_variant": record["variant"],
            "requested_threads": record["threads"],
            "warmup_iterations": 2,
            "measured_iterations": 3,
            "lhs_sequence_length": record["lhs_sequence"],
            "cache_mode": cache,
            "allocation_mode": allocation,
            "packing_mode": packing,
            "smt_policy": (
                "allow-smt"
                if record["variant"] == EXTERNAL and record["threads"] > 1
                else "physical-cores-only"
            ),
            "affinity_policy": (
                "none"
                if record["variant"] == EXTERNAL and record["threads"] > 1
                else "compact"
            ),
            "seed": SEED,
        },
        "results": [
            {
                "m": m,
                "n": n,
                "k": k,
                "requested_variant": record["variant"],
                "selected_variant": (
                    NATIVE if record["variant"] == "auto" else record["variant"]
                ),
                "actual_threads": record["actual_threads"],
                "smt_policy": (
                    "allow-smt"
                    if record["variant"] == EXTERNAL and record["threads"] > 1
                    else "physical-cores-only"
                ),
                "affinity_policy": (
                    "none"
                    if record["variant"] == EXTERNAL and record["threads"] > 1
                    else "compact"
                ),
                "worker_affinity_applied": not (
                    record["variant"] == EXTERNAL and record["threads"] > 1
                ),
                "worker_affinity_user_requested": not (
                    record["variant"] == EXTERNAL and record["threads"] > 1
                ),
                "worker_affinity_policy_induced": False,
                "timing_valid": True,
                "correctness": True,
                "timed_final_output_authenticated": True,
                "normalized_samples_seconds": samples,
                "minimum_seconds": samples[0],
                "median_seconds": samples[1],
                "p95_seconds": samples[2],
                "gflops": (2 * m * n * k) / seconds / 1.0e9,
                "prepacked_b_preparation": preparation,
                "planner_regret": planner,
            }
        ],
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def plan_digest(manifest: dict) -> str:
    cases = [
        {
            "family": record["family"],
            "partition": record["partition"],
            "shape": record["shape"],
            "variant": record["variant"],
            "threads": record["threads"],
            "mode": record["mode"],
            "lhs_sequence": record["lhs_sequence"],
        }
        for record in manifest["cases"]
    ]
    skips = [
        {
            "family": skip["family"],
            "shape": skip["shape"],
            "variant": skip["variant"],
            "threads": skip["threads"],
            "mode": skip["mode"],
            "reason": skip["reason"],
        }
        for skip in manifest["skips"]
    ]
    return canonical_sha256(
        {
            "schema": "matcore.cpu-performance-deep-audit.plan",
            "version": 1,
            "suites": sorted(manifest["suites"]),
            "variants": manifest["variants"],
            "threads": manifest["threads"],
            "warmup": manifest["warmup"],
            "iterations": manifest["iterations"],
            "timer_floor_us": manifest["timer_floor_us"],
            "max_memory_mib": manifest["max_memory_mib"],
            "case_order": manifest["case_order"],
            "seed": manifest["benchmark_seed"],
            "cases": cases,
            "skips": skips,
        }
    )


def build_bundle(directory: pathlib.Path, reverse: bool) -> pathlib.Path:
    direction_scale = 2.0 if reverse else 1.0
    specifications = [
        ("complete-hot", NATIVE, 1, 1, 0.002 * direction_scale),
        ("complete-hot", EXTERNAL, 1, 1, 0.001 * direction_scale),
        ("complete-hot", EXTERNAL, 4, 4, 0.0004 * direction_scale),
        ("one-shot-hot", NATIVE, 1, 1, 0.003 * direction_scale),
        ("prepacked-b-hot", NATIVE, 1, 4, 0.0008 * direction_scale),
        ("complete-cold", NATIVE, 1, 1, 0.004 * direction_scale),
        ("compute-only-hot", NATIVE, 1, 1, 0.0015 * direction_scale),
        ("planner-regret-hot", "auto", 1, 1, 0.002 * direction_scale),
    ]
    if reverse:
        specifications = [
            specification
            for specification in specifications
            if specification[0] in {"complete-hot", "one-shot-hot"}
        ]
    records = []
    for index, (mode, variant, threads, lhs_sequence, seconds) in enumerate(
        specifications
    ):
        record = {
            "index": index,
            "family": "medium-square",
            "partition": "calibration",
            "shape": [128, 128, 128],
            "variant": variant,
            "threads": threads,
            "mode": mode,
            "lhs_sequence": lhs_sequence,
            "state": "passed",
            "actual_threads": threads,
            "thread_count_clamped": False,
        }
        record["key"] = case_key(record)
        record["raw_file"] = f"{index:04d}.json"
        preparation_seconds = (
            0.001 * direction_scale if mode == "prepacked-b-hot" else 0.0
        )
        raw_path = directory / record["raw_file"]
        write_raw(raw_path, record, seconds, preparation_seconds)
        record["sha256"] = file_sha256(raw_path)
        records.append(record)

    rejected = {
        "index": len(records),
        "family": "small-square",
        "partition": "diagnostic",
        "shape": [4, 4, 4],
        "variant": PARALLEL,
        "threads": 2,
        "mode": "complete-hot",
        "lhs_sequence": 1,
        "state": "rejected",
        "actual_threads": 0,
        "thread_count_clamped": False,
        "raw_file": "rejected.json",
        "returncode": 2,
        "rejection_category": "parallel-output-macro-tile-count",
        "stdout": "",
        "stderr": "expected synthetic rejection",
    }
    rejected["key"] = case_key(rejected)
    records.append(rejected)
    if reverse:
        records.reverse()
        next(record for record in records if record["state"] == "passed")[
            "state"
        ] = "reused"
        for index, record in enumerate(records):
            record["index"] = index

    manifest = {
        "schema": "matcore.cpu-performance-deep-audit.manifest",
        "version": 2,
        "benchmark_schema_version": 6,
        "benchmark_binary_sha256": "b" * 64,
        "runner_sha256": "c" * 64,
        "plan_sha256": "",
        "benchmark_source_commit": SOURCE_COMMIT,
        "benchmark_seed": SEED,
        "started_unix_seconds": 100,
        "finished_unix_seconds": 200,
        "benchmark": "/synthetic/absolute/path/matcore-bench",
        "suites": (
            ["complete", "oneshot"]
            if reverse
            else [
                "cold",
                "complete",
                "compute",
                "oneshot",
                "prepacked",
                "regret",
            ]
        ),
        "variants": [NATIVE, PARALLEL, EXTERNAL],
        "threads": [1, 2, 4],
        "case_order": "stable-reverse" if reverse else "stable-forward",
        "warmup": 2,
        "iterations": 3,
        "timer_floor_us": 1000,
        "max_memory_mib": 2048,
        "dry_run": False,
        "environment_overrides": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "cases": records,
        "skips": [
            {
                "family": "large-square",
                "shape": [4096, 4096, 4096],
                "variant": "cpu.reference.f32.v1",
                "threads": 1,
                "mode": "complete-hot",
                "reason": "audit runtime bound: synthetic",
            }
        ],
    }
    manifest["plan_sha256"] = plan_digest(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def summarize(
    summarizer: pathlib.Path,
    forward: pathlib.Path,
    reverse: pathlib.Path,
    output: pathlib.Path,
    expected: int = 0,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            sys.executable,
            str(summarizer),
            "--forward-manifest",
            str(forward),
            "--reverse-manifest",
            str(reverse),
            "--markdown-out",
            str(output),
        ],
        expected,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarizer", required=True)
    args = parser.parse_args()
    summarizer = pathlib.Path(args.summarizer).resolve()

    with tempfile.TemporaryDirectory(prefix="matcore audit summary ") as temporary:
        root = pathlib.Path(temporary)
        forward_dir = root / "forward raw"
        reverse_dir = root / "reverse raw"
        forward_dir.mkdir()
        reverse_dir.mkdir()
        forward = build_bundle(forward_dir, reverse=False)
        reverse = build_bundle(reverse_dir, reverse=True)
        output = root / "sanitized report.md"
        summarize(summarizer, forward, reverse, output)
        report = output.read_text(encoding="utf-8")
        assert "complete/oneshot paired stable-forward/stable-reverse" in report
        assert "diagnostic/prepack/regret stable-forward only" in report
        assert "3.000 [2.000, 4.000]" in report
        assert "| medium-square | 1 | 0.500 |" in report
        assert "| 4 | 1 | 1.000" in report
        assert "median diagnostic/hot ratio: 2.000" in report
        assert "median diagnostic/hot ratio: 0.750" in report
        assert "| 1 | 2.000 | 2.000 | 2.000 | 0 |" in report
        assert "expected rejection: parallel-output-macro-tile-count" in report
        assert "/synthetic/absolute/path" not in report

        forward_only = root / "forward-only.md"
        run(
            [
                sys.executable,
                str(summarizer),
                "--forward-manifest",
                str(forward),
                "--markdown-out",
                str(forward_only),
            ]
        )
        assert "stable-forward only" in forward_only.read_text(encoding="utf-8")

        tampered_dir = root / "tampered"
        shutil.copytree(forward_dir, tampered_dir)
        tampered_manifest = tampered_dir / "manifest.json"
        tampered = json.loads(tampered_manifest.read_text(encoding="utf-8"))
        raw = tampered_dir / tampered["cases"][0]["raw_file"]
        raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")
        rejected = summarize(
            summarizer, tampered_manifest, reverse, root / "tampered.md", expected=2
        )
        assert "SHA-256 mismatch" in rejected.stderr

        mixed_dir = root / "mixed-source"
        shutil.copytree(forward_dir, mixed_dir)
        mixed_manifest = mixed_dir / "manifest.json"
        mixed = json.loads(mixed_manifest.read_text(encoding="utf-8"))
        raw_record = mixed["cases"][0]
        raw_path = mixed_dir / raw_record["raw_file"]
        raw_report = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_report["environment"]["source_commit"] = "d" * 40
        raw_path.write_text(json.dumps(raw_report, indent=2) + "\n", encoding="utf-8")
        raw_record["sha256"] = file_sha256(raw_path)
        mixed_manifest.write_text(json.dumps(mixed, indent=2) + "\n", encoding="utf-8")
        rejected = summarize(
            summarizer, mixed_manifest, reverse, root / "mixed.md", expected=2
        )
        assert "source commit differs" in rejected.stderr

        incomplete_dir = root / "incomplete"
        shutil.copytree(forward_dir, incomplete_dir)
        incomplete_manifest = incomplete_dir / "manifest.json"
        incomplete = json.loads(incomplete_manifest.read_text(encoding="utf-8"))
        incomplete["cases"][0]["state"] = "failed"
        incomplete_manifest.write_text(
            json.dumps(incomplete, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer,
            incomplete_manifest,
            reverse,
            root / "incomplete.md",
            expected=2,
        )
        assert "incomplete bundle" in rejected.stderr

        missing_dir = root / "missing-reverse-case"
        shutil.copytree(reverse_dir, missing_dir)
        missing_manifest = missing_dir / "manifest.json"
        missing = json.loads(missing_manifest.read_text(encoding="utf-8"))
        missing["cases"].pop(
            next(
                index
                for index, record in enumerate(missing["cases"])
                if record["state"] in {"passed", "reused"}
                and record["mode"] == "complete-hot"
            )
        )
        missing["plan_sha256"] = plan_digest(missing)
        missing_manifest.write_text(
            json.dumps(missing, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer, forward, missing_manifest, root / "missing.md", expected=2
        )
        assert "exact complete/oneshot forward subset" in rejected.stderr

        extra_dir = root / "extra-reverse-case"
        shutil.copytree(reverse_dir, extra_dir)
        extra_manifest = extra_dir / "manifest.json"
        extra = json.loads(extra_manifest.read_text(encoding="utf-8"))
        forward_data = json.loads(forward.read_text(encoding="utf-8"))
        extra_record = next(
            dict(record)
            for record in forward_data["cases"]
            if record["mode"] == "prepacked-b-hot"
        )
        source_raw = forward_dir / extra_record["raw_file"]
        extra_record["raw_file"] = "extra-prepacked.json"
        shutil.copy2(source_raw, extra_dir / extra_record["raw_file"])
        extra_record["sha256"] = file_sha256(extra_dir / extra_record["raw_file"])
        extra_record["index"] = len(extra["cases"])
        extra["cases"].append(extra_record)
        extra["plan_sha256"] = plan_digest(extra)
        extra_manifest.write_text(
            json.dumps(extra, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer, forward, extra_manifest, root / "extra.md", expected=2
        )
        assert "executable modes are not the required" in rejected.stderr

        mismatch_dir = root / "mismatched-reverse-status"
        shutil.copytree(reverse_dir, mismatch_dir)
        mismatch_manifest = mismatch_dir / "manifest.json"
        mismatch = json.loads(mismatch_manifest.read_text(encoding="utf-8"))
        mismatch_record = next(
            record
            for record in mismatch["cases"]
            if record["state"] in {"passed", "reused"}
            and record["mode"] == "complete-hot"
        )
        mismatch_record["state"] = "rejected"
        mismatch_record.pop("sha256")
        mismatch_record["returncode"] = 2
        mismatch_record["rejection_category"] = "synthetic-status-mismatch"
        mismatch_manifest.write_text(
            json.dumps(mismatch, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer,
            forward,
            mismatch_manifest,
            root / "mismatch.md",
            expected=2,
        )
        assert "per-case execution status differs" in rejected.stderr

    print("deep-audit summary contract: synthetic authentication checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
