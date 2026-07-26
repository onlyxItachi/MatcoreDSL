#!/usr/bin/env python3

"""Synthetic fail-closed contract tests for the deep-audit summarizer."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
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
PARALLEL_AVX512 = "cpu.native-parallel.avx512-fma.f32.v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner_authority(summarizer: pathlib.Path):
    path = summarizer.with_name("run_cpu_deep_audit.py")
    specification = importlib.util.spec_from_file_location(
        "_matcore_summary_test_plan_authority", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


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


def samples_for(seconds: float) -> list[float]:
    return [
        seconds * 0.7,
        seconds * 0.8,
        seconds * 0.9,
        seconds,
        seconds * 1.1,
        seconds * 1.2,
        seconds * 1.3,
    ]


def candidate(variant: str, seconds: float) -> dict:
    samples = samples_for(seconds)
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
    samples = samples_for(seconds)
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
            "capability_record": "synthetic-avx2\nraw-planner-detail",
            "topology_record": "synthetic-1c\nraw-placement-detail",
            "capability_record_version": 2,
            "topology_record_version": 1,
            "logical_processors": 8,
            "physical_cores": 4,
            "numa_nodes": 1,
            "timer_source": "std::chrono::steady_clock",
            "timer_resolution_ns": 1,
            "cpu_affinity": "0-7",
            "hardware_threads": 8,
            "source_provenance_origin": "git-worktree",
            "capability_runtime_validation_source": "synthetic numerical self-test",
            "topology_discovery_complete": True,
            "persistent_execution_context": True,
            "available_processors": 8,
            "provider_name": "OpenBLAS",
            "provider_version": "0.3.32",
            "provider_config": "USE_THREAD=PTHREAD DYNAMIC_ARCH=1",
            "source_commit": SOURCE_COMMIT,
            "source_provenance_state": "clean",
            "source_worktree_dirty": False,
        },
        "configuration": {
            "profile": "custom",
            "requested_variant": record["variant"],
            "requested_threads": record["threads"],
            "warmup_iterations": 2,
            "measured_iterations": 7,
            "lhs_sequence_length": record["lhs_sequence"],
            "alignment_bytes": 64,
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
            "maximum_memory_bytes": 2048 * 1024 * 1024,
            "timer_floor_ns": (
                1000
                if mode in {"complete-cold", "planner-regret-hot"}
                else 1000 * 1000
            ),
            "seed": SEED,
            "compare_one_thread": False,
            "planner_regret": mode == "planner-regret-hot",
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
                "median_seconds": samples[3],
                "p95_seconds": samples[6],
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


def synthetic_seconds(case, reverse: bool) -> float:
    shape = tuple(case.shape)
    direction_scale = (
        2.0 if reverse and case.mode in {"complete-hot", "one-shot-hot"} else 1.0
    )
    if shape == (128, 128, 128):
        if case.mode == "complete-hot":
            if case.variant == NATIVE:
                return 0.002 * direction_scale
            if case.variant == EXTERNAL:
                return 0.001 * direction_scale
            return 0.004 * direction_scale
        if case.variant == NATIVE and case.mode == "one-shot-hot":
            return 0.003 * direction_scale
        if case.variant == NATIVE and case.mode == "prepacked-b-hot":
            return 0.0008
        if case.variant == NATIVE and case.mode == "complete-cold":
            return 0.004
        if case.variant == NATIVE and case.mode == "compute-only-hot":
            return 0.0015
        if case.mode == "planner-regret-hot":
            return 0.002
    if (
        shape == (512, 512, 512)
        and case.mode == "complete-hot"
        and case.variant == NATIVE
    ):
        return 0.002 * direction_scale
    if (
        shape == (512, 512, 512)
        and case.mode == "complete-hot"
        and case.variant == PARALLEL
        and case.threads == 4
    ):
        return 0.0006 * direction_scale
    rates = {
        "cpu.reference.f32.v1": 5.0,
        "cpu.tiled.f32.v1": 20.0,
        "cpu.compiler-vectorized.avx2-fma.f32.v1": 50.0,
        NATIVE: 100.0,
        "cpu.native-packed.avx512-fma.f32.v1": 105.0,
        PARALLEL: 80.0,
        PARALLEL_AVX512: 84.0,
        EXTERNAL: 150.0,
        "auto": 90.0,
    }
    operations = 2 * case.shape[0] * case.shape[1] * case.shape[2]
    seconds = max(1.0e-6, operations / rates[case.variant] / 1.0e9)
    factors = {
        "one-shot-hot": 1.5,
        "prepacked-b-hot": 0.5,
        "complete-cold": 2.0,
        "compute-only-hot": 0.75,
        "planner-regret-hot": 1.0,
        "complete-hot": 1.0,
    }
    return seconds * factors[case.mode] * direction_scale


def build_bundle(
    directory: pathlib.Path, reverse: bool, authority: object
) -> pathlib.Path:
    suites = (
        {"complete", "oneshot"}
        if reverse
        else {"cold", "complete", "compute", "oneshot", "prepacked", "regret"}
    )
    cases, skips = authority.build_cases(
        suites, tuple(authority.VARIANTS), (1, 2, 4, 12)
    )
    case_order = "stable-reverse" if reverse else "stable-forward"
    if reverse:
        cases.reverse()
    benchmark = "/synthetic/absolute/path/matcore-bench"
    records = []
    for index, case in enumerate(cases):
        record = {
            "index": index,
            **json.loads(json.dumps(dataclasses.asdict(case))),
            "state": "planned",
        }
        record["key"] = case.key
        record["raw_file"] = f"{index:04d}__{case.key}.json"
        raw_path = directory / record["raw_file"]
        record["command"] = authority.case_command(
            pathlib.Path(benchmark),
            case,
            raw_path,
            2,
            7,
            1000,
            2048,
        )
        m, n, k = case.shape
        expected_rejection = (
            case.variant in {PARALLEL, PARALLEL_AVX512}
            and case.threads >= 2
            and (m + 127) // 128 < 2
            and case.mode in {"complete-hot", "complete-cold", "one-shot-hot"}
        )
        if expected_rejection:
            record.update(
                {
                    "state": "rejected",
                    "returncode": 1,
                    "rejection_category": "parallel-output-macro-tile-count",
                    "stdout": "",
                    "stderr": (
                        f"matcore-bench: variant planning failed for {m}x{n}x{k}: "
                        "parallel candidate requires at least two output "
                        "macro-tiles and workers\n"
                    ),
                }
            )
            records.append(record)
            continue
        if case.variant in {PARALLEL, PARALLEL_AVX512}:
            actual_threads = min(case.threads, (m + 127) // 128)
        elif case.variant == EXTERNAL:
            actual_threads = case.threads
        else:
            actual_threads = 1
        record.update(
            {
                "state": "passed",
                "actual_threads": actual_threads,
                "thread_count_clamped": actual_threads != case.threads,
            }
        )
        seconds = synthetic_seconds(case, reverse)
        preparation_seconds = (
            0.001 if case.mode == "prepacked-b-hot" else 0.0
        )
        write_raw(raw_path, record, seconds, preparation_seconds)
        record["sha256"] = file_sha256(raw_path)
        records.append(record)
    if reverse:
        next(record for record in records if record["state"] == "passed")[
            "state"
        ] = "reused"
    manifest = {
        "schema": "matcore.cpu-performance-deep-audit.manifest",
        "version": 2,
        "benchmark_schema_version": 6,
        "benchmark_binary_sha256": "b" * 64,
        "runner_sha256": file_sha256(
            pathlib.Path(authority.__file__).resolve()
        ),
        "plan_sha256": authority.plan_fingerprint(
            cases,
            skips,
            suites,
            tuple(authority.VARIANTS),
            (1, 2, 4, 12),
            2,
            7,
            1000,
            2048,
            case_order,
        ),
        "benchmark_source_commit": SOURCE_COMMIT,
        "benchmark_seed": SEED,
        "started_unix_seconds": 100,
        "finished_unix_seconds": 200,
        "benchmark": benchmark,
        "suites": sorted(suites),
        "variants": list(authority.VARIANTS),
        "threads": [1, 2, 4, 12],
        "case_order": case_order,
        "warmup": 2,
        "iterations": 7,
        "timer_floor_us": 1000,
        "max_memory_mib": 2048,
        "dry_run": False,
        "environment_overrides": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "cases": records,
        "skips": json.loads(
            json.dumps([dataclasses.asdict(skip) for skip in skips])
        ),
    }
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
    authority = load_runner_authority(summarizer)

    with tempfile.TemporaryDirectory(prefix="matcore audit summary ") as temporary:
        root = pathlib.Path(temporary)
        forward_dir = root / "forward raw"
        reverse_dir = root / "reverse raw"
        forward_dir.mkdir()
        reverse_dir.mkdir()
        forward = build_bundle(forward_dir, reverse=False, authority=authority)
        reverse = build_bundle(reverse_dir, reverse=True, authority=authority)
        output = root / "sanitized report.md"
        summarize(summarizer, forward, reverse, output)
        report = output.read_text(encoding="utf-8")
        assert "complete/oneshot paired stable-forward/stable-reverse" in report
        assert "diagnostic/prepack/regret stable-forward only" in report
        assert "3.000 [2.000, 4.000]" in report
        assert "| medium-square | 6 | 0.700 |" in report
        assert (
            "| medium-square | calibration | 128×128×128 | "
            "`cpu.native-packed.avx2-fma.f32.v1` |"
        ) in report
        assert "| 4 | 8 | 1.000" in report
        assert "median diagnostic/hot ratio: 2.000" in report
        assert "median diagnostic/hot ratio: 0.750" in report
        assert "| 24 | 2.000 | 2.000 | 2.000 | 0 |" in report
        assert "| medium-square | 4 | 2 | 1.382 | 0.346 |" in report
        assert "| Configured OpenBLAS threads |" in report
        assert "active OpenBLAS concurrency was not sampled" in report
        assert "- Compiler flags: `-O3`" in report
        assert (
            "- Timer: std::chrono::steady_clock; resolution=1 ns" in report
        )
        assert "- External provider: OpenBLAS 0.3.32" in report
        assert (
            "- External provider config: "
            "`USE_THREAD=PTHREAD DYNAMIC_ARCH=1`"
        ) in report
        assert "- Capability record first line: synthetic-avx2" in report
        assert "- Capability record SHA-256:" in report
        assert "- Topology record first line: synthetic-1c" in report
        assert "- Topology record SHA-256:" in report
        assert "raw-planner-detail" not in report
        assert "raw-placement-detail" not in report
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

        configuration_attacks = (
            ("alignment", "alignment_bytes", 4),
            ("maximum-memory", "maximum_memory_bytes", 4096),
            ("timer-floor", "timer_floor_ns", 1),
            ("compare-flag", "compare_one_thread", True),
            ("planner-flag", "planner_regret", True),
            ("smt-placement", "smt_policy", "allow-smt"),
            ("affinity-placement", "affinity_policy", "none"),
        )
        for name, field, value in configuration_attacks:
            attack_dir = root / f"configuration-{name}"
            shutil.copytree(forward_dir, attack_dir)
            attack_manifest_path = attack_dir / "manifest.json"
            attack_manifest = json.loads(
                attack_manifest_path.read_text(encoding="utf-8")
            )
            attack_record = next(
                record
                for record in attack_manifest["cases"]
                if record["state"] == "passed"
                and record["mode"] == "complete-hot"
                and record["variant"] == NATIVE
            )
            attack_raw = attack_dir / attack_record["raw_file"]
            attack_report = json.loads(attack_raw.read_text(encoding="utf-8"))
            attack_report["configuration"][field] = value
            attack_raw.write_text(
                json.dumps(attack_report, indent=2) + "\n", encoding="utf-8"
            )
            attack_record["sha256"] = file_sha256(attack_raw)
            attack_manifest_path.write_text(
                json.dumps(attack_manifest, indent=2) + "\n", encoding="utf-8"
            )
            rejected = summarize(
                summarizer,
                attack_manifest_path,
                reverse,
                root / f"configuration-{name}.md",
                expected=2,
            )
            assert f"configuration mismatch for {field}" in rejected.stderr

        environment_attacks = (
            ("provider", "provider_config", "tampered-provider"),
            ("timer-source", "timer_source", "tampered-clock"),
            ("timer-resolution", "timer_resolution_ns", 999),
            ("physical-cores", "physical_cores", 999),
            ("available-processors", "available_processors", 999),
        )
        for name, field, value in environment_attacks:
            attack_dir = root / f"environment-{name}"
            shutil.copytree(forward_dir, attack_dir)
            attack_manifest_path = attack_dir / "manifest.json"
            attack_manifest = json.loads(
                attack_manifest_path.read_text(encoding="utf-8")
            )
            attack_record = next(
                record
                for record in attack_manifest["cases"]
                if record["state"] == "passed"
            )
            attack_raw = attack_dir / attack_record["raw_file"]
            attack_report = json.loads(attack_raw.read_text(encoding="utf-8"))
            attack_report["environment"][field] = value
            attack_raw.write_text(
                json.dumps(attack_report, indent=2) + "\n", encoding="utf-8"
            )
            attack_record["sha256"] = file_sha256(attack_raw)
            attack_manifest_path.write_text(
                json.dumps(attack_manifest, indent=2) + "\n", encoding="utf-8"
            )
            rejected = summarize(
                summarizer,
                attack_manifest_path,
                reverse,
                root / f"environment-{name}.md",
                expected=2,
            )
            assert "homogeneous benchmark environment" in rejected.stderr

        fallback_dir = root / "forced-fallback"
        shutil.copytree(forward_dir, fallback_dir)
        fallback_manifest_path = fallback_dir / "manifest.json"
        fallback_manifest = json.loads(
            fallback_manifest_path.read_text(encoding="utf-8")
        )
        fallback_record = next(
            record
            for record in fallback_manifest["cases"]
            if record["state"] == "passed"
            and record["variant"] == NATIVE
            and record["mode"] == "complete-hot"
        )
        fallback_raw = fallback_dir / fallback_record["raw_file"]
        fallback_report = json.loads(fallback_raw.read_text(encoding="utf-8"))
        fallback_report["results"][0]["selected_variant"] = EXTERNAL
        fallback_raw.write_text(
            json.dumps(fallback_report, indent=2) + "\n", encoding="utf-8"
        )
        fallback_record["sha256"] = file_sha256(fallback_raw)
        fallback_manifest_path.write_text(
            json.dumps(fallback_manifest, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer,
            fallback_manifest_path,
            reverse,
            root / "fallback.md",
            expected=2,
        )
        assert "silently selected a different implementation" in rejected.stderr

        auto_dir = root / "auto-selection-mismatch"
        shutil.copytree(forward_dir, auto_dir)
        auto_manifest_path = auto_dir / "manifest.json"
        auto_manifest = json.loads(auto_manifest_path.read_text(encoding="utf-8"))
        auto_record = next(
            record
            for record in auto_manifest["cases"]
            if record["state"] == "passed"
            and record["mode"] == "planner-regret-hot"
        )
        auto_raw = auto_dir / auto_record["raw_file"]
        auto_report = json.loads(auto_raw.read_text(encoding="utf-8"))
        auto_report["results"][0]["selected_variant"] = "cpu.nonexistent.f32.v1"
        auto_raw.write_text(
            json.dumps(auto_report, indent=2) + "\n", encoding="utf-8"
        )
        auto_record["sha256"] = file_sha256(auto_raw)
        auto_manifest_path.write_text(
            json.dumps(auto_manifest, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer,
            auto_manifest_path,
            reverse,
            root / "auto-selection.md",
            expected=2,
        )
        assert "no unique authenticated timing candidate" in rejected.stderr

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
        assert "frozen matrix" in rejected.stderr

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
        assert (
            "not an independently expected parallel legality failure"
            in rejected.stderr
        )

        selective_forward_dir = root / "selective-shape-forward"
        selective_reverse_dir = root / "selective-shape-reverse"
        shutil.copytree(forward_dir, selective_forward_dir)
        shutil.copytree(reverse_dir, selective_reverse_dir)
        for directory in (selective_forward_dir, selective_reverse_dir):
            selective_manifest_path = directory / "manifest.json"
            selective = json.loads(
                selective_manifest_path.read_text(encoding="utf-8")
            )
            selective["cases"] = [
                record
                for record in selective["cases"]
                if record["shape"] != [4096, 64, 4096]
            ]
            selective["skips"] = [
                skip
                for skip in selective["skips"]
                if skip["shape"] != [4096, 64, 4096]
            ]
            selective["plan_sha256"] = plan_digest(selective)
            selective_manifest_path.write_text(
                json.dumps(selective, indent=2) + "\n", encoding="utf-8"
            )
        rejected = summarize(
            summarizer,
            selective_forward_dir / "manifest.json",
            selective_reverse_dir / "manifest.json",
            root / "selective.md",
            expected=2,
        )
        assert "independently reconstructed frozen matrix" in rejected.stderr

        rejection_attack_dir = root / "rejection-attack"
        shutil.copytree(forward_dir, rejection_attack_dir)
        rejection_manifest_path = rejection_attack_dir / "manifest.json"
        rejection_attack = json.loads(
            rejection_manifest_path.read_text(encoding="utf-8")
        )
        rejection_record = next(
            record
            for record in rejection_attack["cases"]
            if record["state"] == "rejected"
        )
        rejection_record["returncode"] = 139
        rejection_record["rejection_category"] = (
            "segmentation-fault-treated-as-expected"
        )
        rejection_record["stderr"] = "segmentation fault\n"
        rejection_manifest_path.write_text(
            json.dumps(rejection_attack, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer,
            rejection_manifest_path,
            reverse,
            root / "rejection-attack.md",
            expected=2,
        )
        assert "unexpected return code" in rejected.stderr

        command_attack_dir = root / "command-attack"
        shutil.copytree(forward_dir, command_attack_dir)
        command_manifest_path = command_attack_dir / "manifest.json"
        command_attack = json.loads(
            command_manifest_path.read_text(encoding="utf-8")
        )
        command_record = next(
            record
            for record in command_attack["cases"]
            if record["state"] == "rejected"
        )
        thread_index = command_record["command"].index("--threads") + 1
        command_record["command"][thread_index] = "99"
        command_manifest_path.write_text(
            json.dumps(command_attack, indent=2) + "\n", encoding="utf-8"
        )
        rejected = summarize(
            summarizer,
            command_manifest_path,
            reverse,
            root / "command-attack.md",
            expected=2,
        )
        assert "command does not reconstruct" in rejected.stderr

    print("deep-audit summary contract: synthetic authentication checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
