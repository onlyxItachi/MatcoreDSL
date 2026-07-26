#!/usr/bin/env python3

"""Run the bounded, provenance-checked Milestone 6 CPU GEMM audit matrix.

Raw JSON and the generated manifest are intentionally written outside the
source tree (or beneath the repository's ignored benchmark_reports directory).
This tool never changes planner behavior and never turns an unavailable
variant into a fallback result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import time
from typing import Iterable


SCHEMA_VERSION = 6
MANIFEST_VERSION = 2
DEFAULT_SEED = 0x4D4154434F524531
VARIANTS = (
    "cpu.reference.f32.v1",
    "cpu.tiled.f32.v1",
    "cpu.compiler-vectorized.avx2-fma.f32.v1",
    "cpu.native-packed.avx2-fma.f32.v1",
    "cpu.native-packed.avx512-fma.f32.v1",
    "cpu.native-parallel.avx2-fma.f32.v1",
    "cpu.native-parallel.avx512-fma.f32.v1",
    "cpu.external.openblas.f32.v1",
)
SERIAL_VARIANTS = VARIANTS[:5]
PARALLEL_VARIANTS = VARIANTS[5:7]
EXTERNAL_VARIANT = VARIANTS[7]
PACKED_VARIANTS = VARIANTS[3:7]
SERIAL_PACKED_VARIANTS = VARIANTS[3:5]

SHAPE_FAMILIES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "small-square": tuple((size, size, size) for size in (4, 8, 16, 24, 32, 48, 64)),
    "medium-square": tuple(
        (size, size, size) for size in (96, 128, 192, 256, 384, 512)
    ),
    "large-square": tuple(
        (size, size, size) for size in (768, 1024, 1536, 2048, 4096)
    ),
    "tall-skinny": (
        (4096, 64, 4096),
        (4096, 128, 1024),
        (8192, 32, 1024),
        (2048, 256, 4096),
    ),
    "short-wide": (
        (64, 4096, 4096),
        (128, 4096, 1024),
        (32, 8192, 1024),
        (256, 2048, 4096),
    ),
    "vector-like": (
        (1, 4096, 4096),
        (8, 4096, 4096),
        (4096, 4096, 1),
        (4096, 4096, 8),
    ),
    "tail-heavy": (
        (31, 33, 35),
        (63, 65, 67),
        (127, 129, 131),
        (255, 257, 259),
        (511, 513, 515),
    ),
}

CALIBRATION_SHAPES = {
    *SHAPE_FAMILIES["medium-square"],
    *SHAPE_FAMILIES["tall-skinny"][:2],
    *SHAPE_FAMILIES["short-wide"][:2],
}
HOLDOUT_SHAPES = {
    *SHAPE_FAMILIES["large-square"],
    *SHAPE_FAMILIES["tall-skinny"][2:],
    *SHAPE_FAMILIES["short-wide"][2:],
}
PREPACKED_SHAPES = (
    (128, 128, 128),
    (512, 512, 512),
    (1, 4096, 4096),
    (8, 4096, 4096),
)
ONE_SHOT_SHAPES = (
    (64, 64, 64),
    (256, 256, 256),
    (1024, 1024, 1024),
    (4096, 64, 4096),
    (64, 4096, 4096),
    (1, 4096, 4096),
    (127, 129, 131),
)
COLD_SHAPES = (
    (64, 64, 64),
    (256, 256, 256),
    (1024, 1024, 1024),
    (4096, 64, 4096),
    (64, 4096, 4096),
    (127, 129, 131),
)
REGRET_SHAPES = (
    (32, 32, 32),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (127, 129, 131),
    (511, 513, 515),
)


@dataclasses.dataclass(frozen=True)
class AuditCase:
    family: str
    partition: str
    shape: tuple[int, int, int]
    variant: str
    threads: int
    mode: str
    lhs_sequence: int = 1

    @property
    def key(self) -> str:
        m, n, k = self.shape
        variant = self.variant.replace(".", "_").replace("-", "_")
        return (
            f"{self.family}__{m}x{n}x{k}__{variant}__t{self.threads}"
            f"__{self.mode}__lhs{self.lhs_sequence}"
        )


@dataclasses.dataclass(frozen=True)
class AuditSkip:
    family: str
    shape: tuple[int, int, int]
    variant: str
    threads: int
    mode: str
    reason: str


def partition_for(shape: tuple[int, int, int]) -> str:
    if shape in CALIBRATION_SHAPES:
        return "calibration"
    if shape in HOLDOUT_SHAPES:
        return "holdout"
    return "diagnostic"


def bounded_variant(shape: tuple[int, int, int], variant: str) -> tuple[bool, str]:
    work = shape[0] * shape[1] * shape[2]
    limits = {
        "cpu.reference.f32.v1": 32 * 1024 * 1024,
        "cpu.tiled.f32.v1": 128 * 1024 * 1024,
        "cpu.compiler-vectorized.avx2-fma.f32.v1": 512 * 1024 * 1024,
    }
    limit = limits.get(variant)
    if limit is not None and work > limit:
        return (
            False,
            f"audit runtime bound: {work} scalar products exceeds {limit}",
        )
    return True, ""


def normalized_threads(encoded: str) -> tuple[int, ...]:
    values = sorted({int(value) for value in encoded.split(",") if value})
    if not values or values[0] <= 0:
        raise ValueError("thread list must contain positive integers")
    return tuple(values)


def all_shapes() -> Iterable[tuple[str, tuple[int, int, int]]]:
    for family, shapes in SHAPE_FAMILIES.items():
        for shape in shapes:
            yield family, shape


def build_cases(
    suites: set[str],
    variants: tuple[str, ...],
    threads: tuple[int, ...],
) -> tuple[list[AuditCase], list[AuditSkip]]:
    cases: list[AuditCase] = []
    skips: list[AuditSkip] = []

    if "complete" in suites:
        for family, shape in all_shapes():
            for variant in variants:
                candidate_threads: tuple[int, ...]
                if variant in SERIAL_VARIANTS:
                    candidate_threads = (1,)
                elif variant in PARALLEL_VARIANTS:
                    candidate_threads = tuple(value for value in threads if value > 1)
                else:
                    candidate_threads = threads
                for thread_count in candidate_threads:
                    legal, reason = bounded_variant(shape, variant)
                    if legal:
                        cases.append(
                            AuditCase(
                                family,
                                partition_for(shape),
                                shape,
                                variant,
                                thread_count,
                                "complete-hot",
                            )
                        )
                    else:
                        skips.append(
                            AuditSkip(
                                family,
                                shape,
                                variant,
                                thread_count,
                                "complete-hot",
                                reason,
                            )
                        )

    if "compute" in suites:
        for family, shape in all_shapes():
            for variant in SERIAL_PACKED_VARIANTS[:1]:
                if variant in variants:
                    cases.append(
                        AuditCase(
                            family,
                            partition_for(shape),
                            shape,
                            variant,
                            1,
                            "compute-only-hot",
                        )
                    )

    if "cold" in suites:
        for shape in COLD_SHAPES:
            family = next(
                name for name, shapes in SHAPE_FAMILIES.items() if shape in shapes
            )
            for variant in variants:
                if variant in PARALLEL_VARIANTS:
                    candidate_threads = tuple(value for value in threads if value > 1)
                elif variant == EXTERNAL_VARIANT:
                    candidate_threads = threads
                else:
                    candidate_threads = (1,)
                for thread_count in candidate_threads:
                    legal, reason = bounded_variant(shape, variant)
                    if legal:
                        cases.append(
                            AuditCase(
                                family,
                                partition_for(shape),
                                shape,
                                variant,
                                thread_count,
                                "complete-cold",
                            )
                        )
                    else:
                        skips.append(
                            AuditSkip(
                                family,
                                shape,
                                variant,
                                thread_count,
                                "complete-cold",
                                reason,
                            )
                        )

    if "prepacked" in suites:
        for shape in PREPACKED_SHAPES:
            family = next(
                name for name, shapes in SHAPE_FAMILIES.items() if shape in shapes
            )
            for variant in SERIAL_PACKED_VARIANTS:
                if variant not in variants:
                    continue
                candidate_threads = (
                    (1,)
                    if variant in SERIAL_VARIANTS
                    else tuple(value for value in threads if value > 1)
                )
                for thread_count in candidate_threads:
                    for sequence in (1, 4, 16, 64):
                        cases.append(
                            AuditCase(
                                family,
                                partition_for(shape),
                                shape,
                                variant,
                                thread_count,
                                "prepacked-b-hot",
                                sequence,
                            )
                        )

    if "oneshot" in suites:
        requested_parallel_threads = tuple(
            value for value in threads if value == 4
        )
        if not requested_parallel_threads:
            requested_parallel_threads = tuple(
                value for value in threads if value > 1
            )[:1]
        provider_threads = tuple(
            value for value in threads if value in {1, 4}
        )
        for shape in ONE_SHOT_SHAPES:
            family = next(
                name for name, shapes in SHAPE_FAMILIES.items() if shape in shapes
            )
            for variant in variants:
                if variant in SERIAL_VARIANTS:
                    candidate_threads = (1,)
                elif variant in PARALLEL_VARIANTS:
                    candidate_threads = requested_parallel_threads
                else:
                    candidate_threads = provider_threads
                for thread_count in candidate_threads:
                    legal, reason = bounded_variant(shape, variant)
                    if legal:
                        cases.append(
                            AuditCase(
                                family,
                                partition_for(shape),
                                shape,
                                variant,
                                thread_count,
                                "one-shot-hot",
                            )
                        )
                    else:
                        skips.append(
                            AuditSkip(
                                family,
                                shape,
                                variant,
                                thread_count,
                                "one-shot-hot",
                                reason,
                            )
                        )

    if "regret" in suites:
        for shape in REGRET_SHAPES:
            family = next(
                name for name, shapes in SHAPE_FAMILIES.items() if shape in shapes
            )
            for thread_count in threads:
                cases.append(
                    AuditCase(
                        family,
                        partition_for(shape),
                        shape,
                        "auto",
                        thread_count,
                        "planner-regret-hot",
                    )
                )

    unique = {case.key: case for case in cases}
    return [unique[key] for key in sorted(unique)], skips


def safe_output_directory(path: pathlib.Path, source_root: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    root = source_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0] != "benchmark_reports":
        raise ValueError(
            "raw audit output inside the repository must be under ignored "
            "benchmark_reports/"
        )
    return resolved


def case_command(
    executable: pathlib.Path,
    case: AuditCase,
    output: pathlib.Path,
    warmup: int,
    iterations: int,
    timer_floor_us: int,
    maximum_memory_mib: int,
) -> list[str]:
    m, n, k = case.shape
    # Cold-cache samples deliberately do not aggregate multiple GEMMs behind
    # one eviction. Planner-regret runs already perform balanced forward and
    # reverse passes and can also reject an otherwise useful audit case when a
    # single fast provider sample straddles the aggregation floor. Keep these
    # explicitly diagnostic modes at one microsecond; ordinary complete-call
    # measurements retain the stricter caller-supplied floor.
    effective_timer_floor_us = (
        min(timer_floor_us, 1)
        if case.mode in {"complete-cold", "planner-regret-hot"}
        else timer_floor_us
    )
    command = [
        str(executable),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--variant",
        case.variant,
        "--threads",
        str(case.threads),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--lhs-sequence",
        str(case.lhs_sequence),
        "--timer-floor-us",
        str(effective_timer_floor_us),
        "--max-memory-mib",
        str(maximum_memory_mib),
        "--seed",
        str(DEFAULT_SEED),
        "--alignment",
        "64",
        "--guard",
        "--json-out",
        str(output),
    ]
    if case.mode == "one-shot-hot":
        command.extend(("--include-allocation", "--include-packing", "--hot-cache"))
    else:
        command.append("--reuse-workspace")
        if case.mode == "compute-only-hot":
            command.extend(("--exclude-packing", "--hot-cache"))
        elif case.mode == "prepacked-b-hot":
            command.extend(("--prepack-b", "--hot-cache"))
        elif case.mode == "complete-cold":
            command.extend(("--include-packing", "--cold-cache"))
        else:
            command.extend(("--include-packing", "--hot-cache"))
    if case.mode == "planner-regret-hot":
        command.append("--planner-regret")
    if case.variant == EXTERNAL_VARIANT and case.threads > 1:
        command.extend(("--allow-smt", "--affinity", "none"))
    else:
        command.extend(("--physical-cores-only", "--affinity", "compact"))
    return command


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_case_configuration(
    case: AuditCase,
    warmup: int,
    iterations: int,
    timer_floor_us: int,
    maximum_memory_mib: int,
) -> dict[str, object]:
    diagnostic_floor = case.mode in {"complete-cold", "planner-regret-hot"}
    cache_mode = "cold" if case.mode == "complete-cold" else "hot"
    packing_mode = (
        "exclude-packing"
        if case.mode == "compute-only-hot"
        else "prepacked-b"
        if case.mode == "prepacked-b-hot"
        else "include-packing"
    )
    provider_parallel = case.variant == EXTERNAL_VARIANT and case.threads > 1
    return {
        "requested_variant": case.variant,
        "requested_threads": case.threads,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "lhs_sequence_length": case.lhs_sequence,
        "alignment_bytes": 64,
        "cache_mode": cache_mode,
        "allocation_mode": (
            "include-allocation"
            if case.mode == "one-shot-hot"
            else "reuse-workspace"
        ),
        "packing_mode": packing_mode,
        "smt_policy": (
            "allow-smt" if provider_parallel else "physical-cores-only"
        ),
        "affinity_policy": "none" if provider_parallel else "compact",
        "maximum_memory_bytes": maximum_memory_mib * 1024 * 1024,
        "timer_floor_ns": (
            min(timer_floor_us, 1) if diagnostic_floor else timer_floor_us
        )
        * 1000,
        "seed": DEFAULT_SEED,
        "compare_one_thread": False,
        "planner_regret": case.mode == "planner-regret-hot",
    }


def plan_fingerprint(
    cases: list[AuditCase],
    skips: list[AuditSkip],
    suites: set[str],
    variants: tuple[str, ...],
    threads: tuple[int, ...],
    warmup: int,
    iterations: int,
    timer_floor_us: int,
    maximum_memory_mib: int,
    case_order: str,
    families: set[str] | None = None,
) -> str:
    selected_families = set(SHAPE_FAMILIES) if families is None else families
    return canonical_sha256(
        {
            "schema": "matcore.cpu-performance-deep-audit.plan",
            "version": 1,
            "suites": sorted(suites),
            "families": sorted(selected_families),
            "variants": list(variants),
            "threads": list(threads),
            "warmup": warmup,
            "iterations": iterations,
            "timer_floor_us": timer_floor_us,
            "max_memory_mib": maximum_memory_mib,
            "case_order": case_order,
            "seed": DEFAULT_SEED,
            "cases": [dataclasses.asdict(case) for case in cases],
            "skips": [dataclasses.asdict(skip) for skip in skips],
        }
    )


def expected_legality_rejection(case: AuditCase, stderr: str) -> str | None:
    if (
        case.variant in PARALLEL_VARIANTS
        and "parallel candidate requires at least two output macro-tiles and workers"
        in stderr
    ):
        return "parallel-output-macro-tile-count"
    return None


def authenticate_report(
    path: pathlib.Path,
    case: AuditCase,
    warmup: int,
    iterations: int,
    timer_floor_us: int,
    maximum_memory_mib: int,
    expected_source_commit: str | None = None,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "matcore.benchmark.cpu.gemm":
        raise ValueError("unexpected benchmark schema")
    if report.get("version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema version {SCHEMA_VERSION}")
    if report["environment"]["source_provenance_state"] != "clean":
        raise ValueError("benchmark source provenance is not clean")
    if report["environment"]["source_worktree_dirty"]:
        raise ValueError("benchmark source worktree was dirty")
    source_commit = report["environment"].get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("benchmark source commit is missing")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ValueError(
            "benchmark source commit does not match the frozen resume identity"
        )
    configuration = report["configuration"]
    expected_configuration = expected_case_configuration(
        case, warmup, iterations, timer_floor_us, maximum_memory_mib
    )
    for field, expected in expected_configuration.items():
        if configuration.get(field) != expected:
            raise ValueError(
                f"benchmark configuration mismatch for {field}: "
                f"expected {expected!r}, found {configuration.get(field)!r}"
            )
    if len(report["results"]) != 1:
        raise ValueError("one-case audit command emitted multiple results")
    result = report["results"][0]
    if (result["m"], result["n"], result["k"]) != case.shape:
        raise ValueError("shape metadata mismatch")
    if result["requested_variant"] != case.variant:
        raise ValueError("variant metadata mismatch")
    if not result["timing_valid"] or not result["correctness"]:
        raise ValueError("guarded benchmark result is invalid")
    if not result["timed_final_output_authenticated"]:
        raise ValueError("timed output was not authenticated")
    if result["actual_threads"] < 1 or result["actual_threads"] > case.threads:
        raise ValueError("actual implementation thread count exceeds request")
    samples = result.get("normalized_samples_seconds")
    if (
        not isinstance(samples, list)
        or len(samples) != iterations
        or any(
            not isinstance(sample, (int, float))
            or not math.isfinite(sample)
            or sample <= 0.0
            for sample in samples
        )
    ):
        raise ValueError("ordered timing samples are missing or invalid")
    ordered_samples = sorted(float(sample) for sample in samples)
    p95_index = math.ceil(len(ordered_samples) * 0.95) - 1
    reconstructed = {
        "minimum_seconds": ordered_samples[0],
        "median_seconds": ordered_samples[len(ordered_samples) // 2],
        "p95_seconds": ordered_samples[p95_index],
    }
    for field, expected in reconstructed.items():
        if not math.isclose(
            float(result[field]), expected, rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            raise ValueError(f"{field} does not match the raw timing samples")

    preparation = result.get("prepacked_b_preparation")
    if not isinstance(preparation, dict):
        raise ValueError("prepacked-B preparation metadata is missing")
    if case.mode == "prepacked-b-hot":
        if not (
            preparation.get("requested")
            and preparation.get("measured")
            and preparation.get("authenticated")
            and preparation.get("preparation_calls") == 1
            and preparation.get("amortization_executions") == case.lhs_sequence
            and preparation.get("amortized_total_valid")
        ):
            raise ValueError("prepacked-B preparation was not authenticated")
        prepare_seconds = float(preparation["preparation_seconds"])
        steady_seconds = float(preparation["steady_state_sequence_seconds"])
        total_seconds = float(preparation["amortized_total_sequence_seconds"])
        per_execution = float(preparation["amortized_per_execution_seconds"])
        expected_steady = float(result["median_seconds"]) * case.lhs_sequence
        if (
            prepare_seconds <= 0.0
            or not math.isclose(
                steady_seconds,
                expected_steady,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                total_seconds,
                prepare_seconds + steady_seconds,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                per_execution,
                total_seconds / case.lhs_sequence,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError("prepacked-B amortized timing arithmetic mismatch")
    elif preparation.get("requested"):
        raise ValueError("unexpected prepacked-B preparation metadata")

    if case.mode == "planner-regret-hot":
        regret = result.get("planner_regret")
        if not isinstance(regret, dict) or not regret.get("valid"):
            raise ValueError("planner-regret result is missing or invalid")
        for candidate in regret.get("candidates", []):
            if not candidate.get("timing_valid"):
                continue
            for pass_name in ("forward", "reverse"):
                pass_samples = candidate.get(
                    f"{pass_name}_pass_normalized_samples_seconds"
                )
                if (
                    not isinstance(pass_samples, list)
                    or len(pass_samples) != iterations
                    or any(
                        not isinstance(sample, (int, float))
                        or not math.isfinite(sample)
                        or sample <= 0.0
                        for sample in pass_samples
                    )
                ):
                    raise ValueError(
                        f"planner-regret {pass_name} samples are invalid"
                    )
                expected_median = sorted(
                    float(sample) for sample in pass_samples
                )[len(pass_samples) // 2]
                if not math.isclose(
                    float(candidate[f"{pass_name}_pass_median_seconds"]),
                    expected_median,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        f"planner-regret {pass_name} median mismatch"
                    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--suites",
        default="complete",
        help="comma list: complete,compute,cold,prepacked,oneshot,regret,all",
    )
    parser.add_argument(
        "--families",
        default="all",
        help="comma list of audited shape families, or all",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated stable IDs",
    )
    parser.add_argument("--threads", default="1,2,4,12")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--timer-floor-us", type=int, default=1000)
    parser.add_argument("--max-memory-mib", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--case-order",
        choices=("stable-forward", "stable-reverse"),
        default="stable-forward",
    )
    args = parser.parse_args()

    executable = pathlib.Path(args.bench).resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable does not exist: {executable}")
    runner_path = pathlib.Path(__file__).resolve()
    source_root = runner_path.parents[3]
    try:
        output_dir = safe_output_directory(
            pathlib.Path(args.output_dir), source_root
        )
    except ValueError as error:
        parser.error(str(error))

    suites = {value for value in args.suites.split(",") if value}
    if "all" in suites:
        suites = {
            "complete",
            "compute",
            "cold",
            "prepacked",
            "oneshot",
            "regret",
        }
    unknown_suites = suites - {
        "complete",
        "compute",
        "cold",
        "prepacked",
        "oneshot",
        "regret",
    }
    if not suites or unknown_suites:
        parser.error(f"unknown or empty suite set: {sorted(unknown_suites)}")
    families = {value for value in args.families.split(",") if value}
    if "all" in families:
        families = set(SHAPE_FAMILIES)
    unknown_families = families - set(SHAPE_FAMILIES)
    if not families or unknown_families:
        parser.error(
            f"unknown or empty shape-family set: {sorted(unknown_families)}"
        )
    variants = tuple(value for value in args.variants.split(",") if value)
    unknown_variants = set(variants) - set(VARIANTS)
    if not variants or unknown_variants:
        parser.error(f"unknown or empty variant set: {sorted(unknown_variants)}")
    try:
        threads = normalized_threads(args.threads)
    except ValueError as error:
        parser.error(str(error))
    if args.warmup < 0 or args.iterations <= 0 or args.timer_floor_us <= 0:
        parser.error("warmup must be nonnegative; iterations and timer floor positive")
    if args.max_memory_mib <= 0 or args.limit < 0:
        parser.error("memory bound must be positive and limit nonnegative")

    cases, skips = build_cases(suites, variants, threads)
    cases = [case for case in cases if case.family in families]
    skips = [skip for skip in skips if skip.family in families]
    if args.case_order == "stable-reverse":
        cases.reverse()
    if args.limit:
        cases = cases[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    benchmark_digest = sha256(executable)
    runner_digest = sha256(runner_path)
    current_plan_digest = plan_fingerprint(
        cases,
        skips,
        suites,
        variants,
        threads,
        args.warmup,
        args.iterations,
        args.timer_floor_us,
        args.max_memory_mib,
        args.case_order,
        families=families,
    )
    if args.resume and args.dry_run:
        parser.error("--resume cannot be combined with --dry-run")
    if not args.resume and any(output_dir.iterdir()):
        parser.error(
            "output directory is not empty; use a fresh directory or an "
            "identity-checked --resume"
        )

    prior_records: dict[str, dict] = {}
    frozen_source_commit: str | None = None
    if args.resume:
        if not manifest_path.is_file():
            parser.error("--resume requires an existing manifest.json")
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read resume manifest: {error}")
        expected_identity = {
            "schema": "matcore.cpu-performance-deep-audit.manifest",
            "version": MANIFEST_VERSION,
            "benchmark_schema_version": SCHEMA_VERSION,
            "benchmark_binary_sha256": benchmark_digest,
            "runner_sha256": runner_digest,
            "plan_sha256": current_plan_digest,
            "benchmark_seed": DEFAULT_SEED,
        }
        for field, expected in expected_identity.items():
            if prior_manifest.get(field) != expected:
                parser.error(
                    f"resume identity mismatch for {field}: expected "
                    f"{expected!r}, found {prior_manifest.get(field)!r}"
                )
        frozen_source_commit = prior_manifest.get("benchmark_source_commit")
        if not isinstance(frozen_source_commit, str) or not frozen_source_commit:
            parser.error("resume manifest has no frozen benchmark source commit")
        for prior in prior_manifest.get("cases", []):
            key = prior.get("key")
            if not isinstance(key, str) or key in prior_records:
                parser.error("resume manifest contains an invalid or duplicate case key")
            prior_records[key] = prior

    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    records: list[dict] = []
    started = int(time.time())

    for index, case in enumerate(cases):
        raw_path = output_dir / f"{index:04d}__{case.key}.json"
        command = case_command(
            executable,
            case,
            raw_path,
            args.warmup,
            args.iterations,
            args.timer_floor_us,
            args.max_memory_mib,
        )
        record = {
            "index": index,
            "key": case.key,
            "family": case.family,
            "partition": case.partition,
            "shape": list(case.shape),
            "variant": case.variant,
            "threads": case.threads,
            "mode": case.mode,
            "lhs_sequence": case.lhs_sequence,
            "command": command,
            "raw_file": raw_path.name,
            "state": "planned",
        }
        if args.dry_run:
            records.append(record)
            continue
        prior_record = prior_records.get(case.key)
        prior_reusable = (
            prior_record is not None
            and prior_record.get("state") in {"passed", "reused"}
        )
        if args.resume and prior_reusable:
            if prior_record.get("raw_file") != raw_path.name:
                parser.error(f"resume raw-file identity mismatch for {case.key}")
            if not raw_path.is_file():
                parser.error(f"resume raw file is missing for {case.key}")
            raw_digest = sha256(raw_path)
            if prior_record.get("sha256") != raw_digest:
                parser.error(f"resume raw-file digest mismatch for {case.key}")
            report = authenticate_report(
                raw_path,
                case,
                args.warmup,
                args.iterations,
                args.timer_floor_us,
                args.max_memory_mib,
                frozen_source_commit,
            )
            record["state"] = "reused"
            record["sha256"] = raw_digest
            record["actual_threads"] = report["results"][0]["actual_threads"]
            record["thread_count_clamped"] = (
                record["actual_threads"] != case.threads
            )
            records.append(record)
            continue
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            rejection_category = expected_legality_rejection(
                case, completed.stderr
            )
            record["state"] = "rejected" if rejection_category else "failed"
            if rejection_category:
                record["rejection_category"] = rejection_category
            record["returncode"] = completed.returncode
            record["stdout"] = completed.stdout
            record["stderr"] = completed.stderr
            records.append(record)
            if record["state"] == "failed":
                break
            continue
        report = authenticate_report(
            raw_path,
            case,
            args.warmup,
            args.iterations,
            args.timer_floor_us,
            args.max_memory_mib,
            frozen_source_commit,
        )
        report_source_commit = report["environment"]["source_commit"]
        if frozen_source_commit is None:
            frozen_source_commit = report_source_commit
        record["state"] = "passed"
        record["sha256"] = sha256(raw_path)
        record["actual_threads"] = report["results"][0]["actual_threads"]
        record["thread_count_clamped"] = (
            record["actual_threads"] != case.threads
        )
        records.append(record)

    manifest = {
        "schema": "matcore.cpu-performance-deep-audit.manifest",
        "version": MANIFEST_VERSION,
        "benchmark_schema_version": SCHEMA_VERSION,
        "benchmark_binary_sha256": benchmark_digest,
        "runner_sha256": runner_digest,
        "plan_sha256": current_plan_digest,
        "benchmark_source_commit": frozen_source_commit,
        "benchmark_seed": DEFAULT_SEED,
        "started_unix_seconds": started,
        "finished_unix_seconds": int(time.time()),
        "benchmark": str(executable),
        "suites": sorted(suites),
        "families": sorted(families),
        "variants": list(variants),
        "threads": list(threads),
        "case_order": args.case_order,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timer_floor_us": args.timer_floor_us,
        "max_memory_mib": args.max_memory_mib,
        "dry_run": args.dry_run,
        "environment_overrides": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "cases": records,
        "skips": [dataclasses.asdict(skip) for skip in skips],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures = [record for record in records if record["state"] == "failed"]
    print(
        f"matcore CPU deep audit: {len(records)} cases, "
        f"{len(skips)} explicit skips, {len(failures)} failures; "
        f"manifest={manifest_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
