#!/usr/bin/env python3

"""Run the authenticated Milestone 7 native/OpenBLAS parity matrix.

The frozen methodology lives in
docs/performance/cpu/native-blas-parity-methodology-v1.md.  Raw benchmark JSON
and this runner's manifest must live outside every Git worktree.  The runner
accepts explicit legality rejection, but it never accepts forced-variant
substitution, clamped primary-comparison thread counts, dirty source
provenance, or unauthenticated timed output.
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
import sys
import time


BENCHMARK_SCHEMA_VERSION = 6
MANIFEST_VERSION = 3
DEFAULT_SEED = 0x4D4154434F524532
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 11
DEFAULT_TIMER_FLOOR_US = 1000
DEFAULT_MAX_MEMORY_MIB = 4096

PACKED_AVX2 = "cpu.native-packed.avx2-fma.f32.v1"
PACKED_AVX512 = "cpu.native-packed.avx512-fma.f32.v1"
PARALLEL_AVX2 = "cpu.native-parallel.avx2-fma.f32.v1"
PARALLEL_AVX512 = "cpu.native-parallel.avx512-fma.f32.v1"
OPENBLAS = "cpu.external.openblas.f32.v1"
AUTO = "auto"

SERIAL_PARITY_VARIANTS = (PACKED_AVX2, PACKED_AVX512, OPENBLAS)
PARALLEL_PARITY_VARIANTS = (PARALLEL_AVX2, PARALLEL_AVX512, OPENBLAS)
PREPACKED_VARIANTS = (PACKED_AVX2, PACKED_AVX512)
REPEATED_VARIANTS = SERIAL_PARITY_VARIANTS
SEQUENCES = (1, 4, 16, 64)


@dataclasses.dataclass(frozen=True)
class ShapeSpec:
    partition: str
    family: str
    shape: tuple[int, int, int]


# This remains explicit rather than being derived from the Milestone 6 matrix.
# The string "holdout" is retained as a stable case-key identifier, but it does
# not mean blind holdout: several shapes had already been used in candidate
# experiments before this contract was committed. Manifest v3 records that
# corrected interpretation and prevents older evidence from being reused.
PARTITION_INTERPRETATION = {
    "calibration": "candidate-development-and-validation",
    "holdout": "declared-validation-not-blind",
}

PARITY_SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec("calibration", "medium-square", (96, 96, 96)),
    ShapeSpec("calibration", "medium-square", (192, 192, 192)),
    ShapeSpec("calibration", "medium-square", (384, 384, 384)),
    ShapeSpec("calibration", "tall-skinny", (4096, 64, 4096)),
    ShapeSpec("calibration", "tall-skinny", (4096, 128, 1024)),
    ShapeSpec("calibration", "short-wide", (64, 4096, 4096)),
    ShapeSpec("calibration", "short-wide", (128, 4096, 1024)),
    ShapeSpec("calibration", "tail-heavy", (63, 65, 67)),
    ShapeSpec("calibration", "tail-heavy", (255, 257, 259)),
    ShapeSpec("holdout", "medium-square", (128, 128, 128)),
    ShapeSpec("holdout", "medium-square", (256, 256, 256)),
    ShapeSpec("holdout", "medium-square", (512, 512, 512)),
    ShapeSpec("holdout", "large-square", (768, 768, 768)),
    ShapeSpec("holdout", "large-square", (1024, 1024, 1024)),
    ShapeSpec("holdout", "large-square", (1536, 1536, 1536)),
    ShapeSpec("holdout", "large-square", (2048, 2048, 2048)),
    ShapeSpec("holdout", "large-square", (4096, 4096, 4096)),
    ShapeSpec("holdout", "tall-skinny", (8192, 32, 1024)),
    ShapeSpec("holdout", "tall-skinny", (2048, 256, 4096)),
    ShapeSpec("holdout", "short-wide", (32, 8192, 1024)),
    ShapeSpec("holdout", "short-wide", (256, 2048, 4096)),
    ShapeSpec("holdout", "tail-heavy", (31, 33, 35)),
    ShapeSpec("holdout", "tail-heavy", (127, 129, 131)),
    ShapeSpec("holdout", "tail-heavy", (511, 513, 515)),
)

DIAGNOSTIC_SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec("diagnostic", "vector-like", (1, 4096, 4096)),
    ShapeSpec("diagnostic", "vector-like", (8, 4096, 4096)),
    ShapeSpec("diagnostic", "vector-like", (4096, 4096, 1)),
    ShapeSpec("diagnostic", "vector-like", (4096, 4096, 8)),
)

# The benchmark's built-in regret measurement times every legal registry
# candidate in both directions. Keep that diagnostic bounded because forcing
# scalar candidates on the largest shapes is not a practical acceptance run.
# Complete forced/automatic parity cases do not reconstruct full-registry
# regret, so the final report must label this scope as bounded diagnostic
# evidence rather than full-envelope planner acceptance.
REGRET_SHAPES = frozenset(
    {
        (96, 96, 96),
        (128, 128, 128),
        (192, 192, 192),
        (256, 256, 256),
        (384, 384, 384),
        (512, 512, 512),
        (31, 33, 35),
        (63, 65, 67),
        (127, 129, 131),
        (255, 257, 259),
        (511, 513, 515),
    }
)

REPEATED_SHAPES = frozenset(
    {
        (128, 128, 128),
        (512, 512, 512),
        (4096, 128, 1024),
        (128, 4096, 1024),
    }
)


@dataclasses.dataclass(frozen=True)
class ParityCase:
    partition: str
    family: str
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
            f"{self.partition}__{self.family}__{m}x{n}x{k}__{variant}"
            f"__t{self.threads}__{self.mode}__lhs{self.lhs_sequence}"
        )


@dataclasses.dataclass(frozen=True)
class SourceIdentity:
    commit: str
    runner_blob: str


def run_git(source_root: pathlib.Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Git source authentication failed: {' '.join(arguments)}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def authenticate_source(
    source_root: pathlib.Path, runner_path: pathlib.Path
) -> SourceIdentity:
    commit = run_git(source_root, "rev-parse", "HEAD")
    if len(commit) not in (40, 64) or any(
        character not in "0123456789abcdefABCDEF" for character in commit
    ):
        raise ValueError("source HEAD is not an exact Git object ID")
    status = run_git(
        source_root, "status", "--porcelain", "--untracked-files=all"
    )
    if status:
        raise ValueError(
            "native BLAS parity evidence requires a completely clean source "
            "worktree"
        )
    try:
        relative = runner_path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError("parity runner is outside its source repository") from error
    relative_text = relative.as_posix()
    run_git(source_root, "ls-files", "--error-unmatch", "--", relative_text)
    runner_blob = run_git(source_root, "rev-parse", f"HEAD:{relative_text}")
    current_blob = run_git(source_root, "hash-object", str(runner_path))
    if runner_blob != current_blob:
        raise ValueError("parity runner bytes do not match the authenticated HEAD")
    return SourceIdentity(commit=commit, runner_blob=runner_blob)


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


def nearest_existing_parent(path: pathlib.Path) -> pathlib.Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def safe_output_directory(path: pathlib.Path, source_root: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    root = source_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "raw native BLAS parity output must be outside the source repository"
        )

    existing_parent = nearest_existing_parent(resolved)
    probe = subprocess.run(
        ["git", "-C", str(existing_parent), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        raise ValueError(
            "raw native BLAS parity output must be outside every Git worktree"
        )
    return resolved


def thread_strata(physical_cores: int) -> tuple[int, ...]:
    if physical_cores < 2:
        raise ValueError("physical core count must be at least two")
    return tuple(dict.fromkeys((1, 4, physical_cores)))


def parallel_task_capacity(
    shape: tuple[int, int, int], requested_threads: int
) -> int:
    """Mirror the versioned planner-v3 disjoint-output task-capacity contract."""
    m, n, k = shape
    if min(m, n, k, requested_threads) <= 0:
        return 0
    macro_tiles = (m + 127) // 128
    row_quantum = math.lcm(4, 16 // math.gcd(n, 16))
    row_groups = (m + row_quantum - 1) // row_quantum
    column_panels = (n + 255) // 256
    maximum_row_tasks = min(requested_threads, macro_tiles, row_groups)
    row_tasks = maximum_row_tasks
    task_count = maximum_row_tasks
    if n % 16 == 0 and column_panels > 1:
        for candidate_rows in range(1, maximum_row_tasks + 1):
            candidate_columns = min(
                column_panels, requested_threads // candidate_rows
            )
            candidate_tasks = candidate_rows * candidate_columns
            if candidate_tasks > task_count or (
                candidate_tasks == task_count and candidate_rows > row_tasks
            ):
                row_tasks = candidate_rows
                task_count = candidate_tasks
    if task_count > maximum_row_tasks:
        work_floor = 1 << (23 if row_tasks == 1 else 25)
        if (2 * m * n * k) // task_count < work_floor:
            task_count = maximum_row_tasks
    return min(requested_threads, task_count)


def exact_parallel_thread_strata(
    shape: tuple[int, int, int], physical_cores: int
) -> tuple[int, ...]:
    """Return fair exact-thread cells derived from 4/physical-core ceilings."""
    exact: list[int] = []
    for ceiling in dict.fromkeys((4, physical_cores)):
        capacity = parallel_task_capacity(shape, ceiling)
        if capacity > 1 and capacity not in exact:
            exact.append(capacity)
    return tuple(exact)


def parallel_thread_plan(physical_cores: int) -> list[dict[str, object]]:
    ceilings = tuple(dict.fromkeys((4, physical_cores)))
    return [
        {
            "partition": spec.partition,
            "family": spec.family,
            "shape": list(spec.shape),
            "requested_ceilings": list(ceilings),
            "task_capacities": [
                parallel_task_capacity(spec.shape, ceiling)
                for ceiling in ceilings
            ],
            "exact_comparison_threads": list(
                exact_parallel_thread_strata(spec.shape, physical_cores)
            ),
        }
        for spec in PARITY_SHAPES
    ]


def build_cases(
    suites: set[str], physical_cores: int
) -> list[ParityCase]:
    cases: list[ParityCase] = []
    threads = thread_strata(physical_cores)

    if "parity" in suites:
        for spec in PARITY_SHAPES:
            for variant in SERIAL_PARITY_VARIANTS:
                cases.append(
                    ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        variant,
                        1,
                        "complete-hot",
                    )
                )
            for thread_count in exact_parallel_thread_strata(
                spec.shape, physical_cores
            ):
                for variant in PARALLEL_PARITY_VARIANTS:
                    cases.append(
                        ParityCase(
                            spec.partition,
                            spec.family,
                            spec.shape,
                            variant,
                            thread_count,
                            "complete-hot",
                        )
                    )

    if "auto" in suites:
        for spec in PARITY_SHAPES:
            for thread_count in threads:
                cases.append(
                    ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        AUTO,
                        thread_count,
                        "auto-complete-hot",
                    )
                )

    if "regret" in suites:
        for spec in PARITY_SHAPES:
            if spec.shape not in REGRET_SHAPES:
                continue
            for thread_count in threads:
                cases.append(
                    ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        AUTO,
                        thread_count,
                        "planner-regret-hot",
                    )
                )

    if "repeated" in suites:
        for spec in PARITY_SHAPES:
            if spec.shape not in REPEATED_SHAPES:
                continue
            for variant in REPEATED_VARIANTS:
                for sequence in SEQUENCES:
                    cases.append(
                        ParityCase(
                            spec.partition,
                            spec.family,
                            spec.shape,
                            variant,
                            1,
                            "repeated-hot",
                            sequence,
                        )
                    )

    if "prepacked" in suites:
        for spec in PARITY_SHAPES:
            if spec.shape not in REPEATED_SHAPES:
                continue
            for variant in PREPACKED_VARIANTS:
                for sequence in SEQUENCES:
                    cases.append(
                        ParityCase(
                            spec.partition,
                            spec.family,
                            spec.shape,
                            variant,
                            1,
                            "prepacked-b-hot",
                            sequence,
                        )
                    )

    if "diagnostic" in suites:
        for spec in DIAGNOSTIC_SHAPES:
            for variant in SERIAL_PARITY_VARIANTS:
                cases.append(
                    ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        variant,
                        1,
                        "diagnostic-complete-hot",
                    )
                )

    keys = [case.key for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("frozen parity matrix produced duplicate case keys")
    return cases


def case_placement(case: ParityCase) -> tuple[str, str]:
    if case.threads > 1:
        return "allow-smt", "none"
    return "physical-cores-only", "compact"


def case_command(
    executable: pathlib.Path,
    case: ParityCase,
    output: pathlib.Path,
) -> list[str]:
    m, n, k = case.shape
    smt_policy, affinity = case_placement(case)
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
        str(DEFAULT_WARMUP),
        "--iterations",
        str(DEFAULT_ITERATIONS),
        "--lhs-sequence",
        str(case.lhs_sequence),
        "--timer-floor-us",
        str(DEFAULT_TIMER_FLOOR_US),
        "--max-memory-mib",
        str(DEFAULT_MAX_MEMORY_MIB),
        "--seed",
        str(DEFAULT_SEED),
        "--alignment",
        "64",
        "--guard",
        "--json-out",
        str(output),
        "--reuse-workspace",
    ]
    if case.mode == "prepacked-b-hot":
        command.append("--prepack-b")
    else:
        command.append("--include-packing")
    command.append("--hot-cache")
    if case.mode == "planner-regret-hot":
        command.append("--planner-regret")
    command.append(
        "--allow-smt" if smt_policy == "allow-smt" else "--physical-cores-only"
    )
    command.extend(("--affinity", affinity))
    return command


def expected_configuration(case: ParityCase) -> dict[str, object]:
    smt_policy, affinity = case_placement(case)
    return {
        "profile": "custom",
        "requested_variant": case.variant,
        "requested_threads": case.threads,
        "warmup_iterations": DEFAULT_WARMUP,
        "measured_iterations": DEFAULT_ITERATIONS,
        "lhs_sequence_length": case.lhs_sequence,
        "alignment_bytes": 64,
        "cache_mode": "hot",
        "allocation_mode": "reuse-workspace",
        "packing_mode": (
            "prepacked-b"
            if case.mode == "prepacked-b-hot"
            else "include-packing"
        ),
        "smt_policy": smt_policy,
        "affinity_policy": affinity,
        "maximum_memory_bytes": DEFAULT_MAX_MEMORY_MIB * 1024 * 1024,
        "timer_floor_ns": DEFAULT_TIMER_FLOOR_US * 1000,
        "seed": DEFAULT_SEED,
        "compare_one_thread": False,
        "planner_regret": case.mode == "planner-regret-hot",
    }


def reconstructed_timing(result: dict) -> dict[str, float]:
    samples = result.get("normalized_samples_seconds")
    if (
        not isinstance(samples, list)
        or len(samples) != DEFAULT_ITERATIONS
        or any(
            not isinstance(sample, (int, float))
            or isinstance(sample, bool)
            or not math.isfinite(float(sample))
            or float(sample) <= 0.0
            for sample in samples
        )
    ):
        raise ValueError("ordered timing samples are missing or invalid")
    ordered = sorted(float(sample) for sample in samples)
    p95_index = math.ceil(len(ordered) * 0.95) - 1
    return {
        "minimum_seconds": ordered[0],
        "median_seconds": ordered[len(ordered) // 2],
        "p95_seconds": ordered[p95_index],
    }


def require_close(actual: object, expected: float, field: str) -> None:
    if (
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not math.isfinite(float(actual))
        or not math.isclose(
            float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-15
        )
    ):
        raise ValueError(f"{field} does not reconstruct from authenticated data")


def authenticate_prepacked(result: dict, case: ParityCase) -> None:
    preparation = result.get("prepacked_b_preparation")
    if not isinstance(preparation, dict):
        raise ValueError("prepacked-B preparation metadata is missing")
    if case.mode != "prepacked-b-hot":
        if (
            preparation.get("requested")
            or preparation.get("measured")
            or preparation.get("authenticated")
            or preparation.get("preparation_calls") != 0
        ):
            raise ValueError("unexpected prepacked-B preparation metadata")
        return
    if not (
        preparation.get("requested")
        and preparation.get("measured")
        and preparation.get("authenticated")
        and preparation.get("preparation_calls") == 1
        and preparation.get("amortization_executions") == case.lhs_sequence
        and preparation.get("amortized_total_valid")
        and preparation.get("input_state")
        == "caller-storage-allocated-unprepared"
        and preparation.get("output_state") == "prepared-authenticated"
    ):
        raise ValueError("prepacked-B preparation was not authenticated")
    prepare = float(preparation.get("preparation_seconds", 0.0))
    steady = float(preparation.get("steady_state_sequence_seconds", 0.0))
    total = float(preparation.get("amortized_total_sequence_seconds", 0.0))
    per_execution = float(preparation.get("amortized_per_execution_seconds", 0.0))
    expected_steady = float(result["median_seconds"]) * case.lhs_sequence
    if prepare <= 0.0:
        raise ValueError("prepacked-B preparation duration is not positive")
    require_close(steady, expected_steady, "prepacked steady-state duration")
    require_close(total, prepare + steady, "prepacked amortized total")
    require_close(
        per_execution,
        total / case.lhs_sequence,
        "prepacked amortized per-execution duration",
    )


def authenticate_regret(result: dict) -> None:
    regret = result.get("planner_regret")
    if not isinstance(regret, dict) or not regret.get("requested"):
        raise ValueError("planner-regret metadata is missing")
    if not regret.get("valid"):
        raise ValueError("planner-regret result is invalid")
    if regret.get("aggregation_method") != (
        "arithmetic-mean-of-forward-and-reverse-pass-medians"
    ):
        raise ValueError("planner-regret aggregation contract changed")
    candidates = regret.get("candidates", [])
    expected_variants = [
        "cpu.reference.f32.v1",
        "cpu.tiled.f32.v1",
        "cpu.compiler-vectorized.avx2-fma.f32.v1",
        "cpu.external.openblas.f32.v1",
        "cpu.native-packed.avx2-fma.f32.v1",
        "cpu.native-packed.avx512-fma.f32.v1",
        "cpu.native-parallel.avx2-fma.f32.v1",
        "cpu.native-parallel.avx512-fma.f32.v1",
    ]
    if (
        not isinstance(candidates, list)
        or [
            candidate.get("variant")
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        != expected_variants
    ):
        raise ValueError(
            "planner-regret candidates differ from the complete ordered "
            "v3 registry"
        )
    valid_candidates = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("planner-regret candidate is malformed")
        if candidate.get("legal"):
            if candidate.get("selected_variant") != candidate.get("variant"):
                raise ValueError("forced regret candidate was substituted")
            if not candidate.get("plan_authenticated"):
                raise ValueError("legal regret candidate plan is unauthenticated")
        comparable = bool(
            candidate.get("legal")
            and candidate.get("complete_implementation_comparison")
        )
        if candidate.get("timing_valid") is not comparable:
            raise ValueError(
                "planner-regret timing validity differs from legal "
                "complete-call comparability"
            )
        if not candidate.get("timing_valid"):
            continue
        valid_candidates += 1
        if not candidate.get("correctness"):
            raise ValueError("timed regret candidate failed correctness")
        pass_medians: list[float] = []
        for pass_name in ("forward", "reverse"):
            samples = candidate.get(
                f"{pass_name}_pass_normalized_samples_seconds"
            )
            if (
                not isinstance(samples, list)
                or len(samples) != DEFAULT_ITERATIONS
                or any(
                    not isinstance(sample, (int, float))
                    or isinstance(sample, bool)
                    or not math.isfinite(float(sample))
                    or float(sample) <= 0.0
                    for sample in samples
                )
            ):
                raise ValueError(
                    f"planner-regret {pass_name} samples are invalid"
                )
            median = sorted(float(sample) for sample in samples)[
                len(samples) // 2
            ]
            require_close(
                candidate.get(f"{pass_name}_pass_median_seconds"),
                median,
                f"planner-regret {pass_name} median",
            )
            pass_medians.append(median)
        balanced = (pass_medians[0] + pass_medians[1]) / 2.0
        require_close(
            candidate.get("balanced_estimate_seconds"),
            balanced,
            "planner-regret balanced estimate",
        )
    if valid_candidates < 2:
        raise ValueError("planner-regret has fewer than two comparable candidates")
    fastest = float(regret.get("fastest_legal_balanced_estimate_seconds", 0.0))
    selected = float(regret.get("selected_balanced_estimate_seconds", 0.0))
    if fastest <= 0.0 or selected <= 0.0:
        raise ValueError("planner-regret timing summary is invalid")
    require_close(regret.get("regret"), selected / fastest, "planner regret")


def authenticate_report(
    path: pathlib.Path,
    case: ParityCase,
    source_commit: str,
    physical_cores: int,
) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"raw benchmark JSON cannot be read: {error}") from error
    if report.get("schema") != "matcore.benchmark.cpu.gemm":
        raise ValueError("unexpected benchmark schema")
    if report.get("version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"expected benchmark schema v{BENCHMARK_SCHEMA_VERSION}"
        )
    for field, expected in {
        "operation": "matcore.gemm",
        "dtype": "f32",
        "accumulation_dtype": "f32",
        "layout": "row-major-contiguous",
    }.items():
        if report.get(field) != expected:
            raise ValueError(f"benchmark semantic field {field} changed")

    environment = report.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("benchmark environment metadata is missing")
    if (
        environment.get("source_provenance_state") != "clean"
        or environment.get("source_worktree_dirty") is not False
        or environment.get("source_provenance_origin") != "git-worktree"
    ):
        raise ValueError(
            "benchmark source provenance is not a clean Git worktree"
        )
    if environment.get("source_commit") != source_commit:
        raise ValueError("benchmark source commit differs from runner HEAD")
    if (
        environment.get("os_family") != "linux"
        or environment.get("architecture") != "x86_64"
    ):
        raise ValueError("native BLAS parity evidence requires Linux x86-64")
    if not environment.get("topology_discovery_complete"):
        raise ValueError("physical topology discovery is incomplete")
    if environment.get("physical_cores") != physical_cores:
        raise ValueError("reported physical cores differ from the frozen request")

    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("benchmark configuration metadata is missing")
    for field, expected in expected_configuration(case).items():
        if configuration.get(field) != expected:
            raise ValueError(
                f"benchmark configuration mismatch for {field}: "
                f"expected {expected!r}, found {configuration.get(field)!r}"
            )

    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("one-case parity command must emit exactly one result")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("benchmark result is malformed")
    if tuple(result.get(field) for field in ("m", "n", "k")) != case.shape:
        raise ValueError("result shape differs from the frozen case")
    if result.get("requested_variant") != case.variant:
        raise ValueError("result requested variant differs from the frozen case")
    if case.variant == AUTO:
        if (
            result.get("planner_mode") != "automatic"
            or not isinstance(result.get("selected_variant"), str)
            or not result.get("selected_variant")
            or result.get("selected_variant") == AUTO
        ):
            raise ValueError("automatic plan selection is malformed")
    elif (
        result.get("planner_mode") != "forced"
        or result.get("selected_variant") != case.variant
    ):
        raise ValueError("forced variant was silently substituted")
    if not result.get("complete_implementation_comparison"):
        raise ValueError("parity result is not a complete implementation call")
    if (
        not result.get("timing_valid")
        or not result.get("correctness")
        or not result.get("timed_final_output_authenticated")
    ):
        raise ValueError("timing, correctness, or final output is unauthenticated")
    if int(result.get("untimed_validation_executions_checked", 0)) < 1:
        raise ValueError("untimed correctness replay is missing")
    for field in ("cache_mode", "allocation_mode", "packing_mode"):
        if result.get(field) != configuration[field]:
            raise ValueError(
                f"result {field} differs from the authenticated configuration"
            )

    smt_policy, affinity = case_placement(case)
    if (
        result.get("smt_policy") != smt_policy
        or result.get("affinity_policy") != affinity
    ):
        raise ValueError("result placement differs from the frozen stratum")
    if case.threads > 1:
        if (
            result.get("worker_affinity_applied")
            or result.get("worker_affinity_user_requested")
            or result.get("worker_affinity_policy_induced")
        ):
            raise ValueError("multi-thread parity cell is not unbound")
    elif not result.get("worker_affinity_applied"):
        raise ValueError("single-thread compact placement was not authenticated")

    actual_threads = result.get("actual_threads")
    if not isinstance(actual_threads, int) or actual_threads < 1:
        raise ValueError("actual thread count is invalid")
    if case.variant != AUTO and actual_threads != case.threads:
        raise ValueError(
            "forced parity candidate did not use the exact requested thread count"
        )
    if case.variant == AUTO and actual_threads > case.threads:
        raise ValueError("automatic plan exceeded the requested thread ceiling")
    if case.variant in (PARALLEL_AVX2, PARALLEL_AVX512):
        expected_capacity = parallel_task_capacity(case.shape, case.threads)
        row_tasks = result.get("parallel_row_tasks")
        column_tasks = result.get("parallel_column_tasks")
        task_count = result.get("parallel_task_count")
        if (
            not isinstance(row_tasks, int)
            or not isinstance(column_tasks, int)
            or not isinstance(task_count, int)
            or row_tasks < 1
            or column_tasks < 1
            or task_count != row_tasks * column_tasks
            or task_count != expected_capacity
        ):
            raise ValueError(
                "parallel result task geometry disagrees with the frozen "
                "planner-capacity model"
            )
    elif any(
        result.get(field) != 0
        for field in (
            "parallel_row_tasks",
            "parallel_column_tasks",
            "parallel_task_count",
        )
    ):
        raise ValueError("serial/provider result unexpectedly reports parallel tasks")

    reconstructed = reconstructed_timing(result)
    for field, expected in reconstructed.items():
        require_close(result.get(field), expected, field)
    median = reconstructed["median_seconds"]
    m, n, k = case.shape
    expected_gflops = (2.0 * m * n * k) / median / 1.0e9
    require_close(result.get("gflops"), expected_gflops, "GFLOP/s")
    for field in (
        "checksum",
        "expected_checksum",
        "maximum_absolute_error",
        "maximum_allowed_error",
    ):
        value = result.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{field} is not finite")

    authenticate_prepacked(result, case)
    regret = result.get("planner_regret")
    if case.mode == "planner-regret-hot":
        authenticate_regret(result)
    elif not isinstance(regret, dict) or regret.get("requested"):
        raise ValueError("unexpected planner-regret measurement")
    return report


_EXPECTED_REJECTION_REASONS: tuple[tuple[str, str], ...] = (
    (
        "parallel candidate requires at least two disjoint output tasks and workers",
        "parallel-output-tile-count",
    ),
    (
        "parallel work per thread is below the deterministic threshold",
        "parallel-work-threshold",
    ),
    ("AVX-512F hardware support is unavailable", "avx512-hardware"),
    (
        "AVX-512F architectural state is not OS-enabled",
        "avx512-os-state",
    ),
    (
        "AVX-512F compiler support is unavailable",
        "avx512-compiler",
    ),
    (
        "native packed AVX-512 implementation is not compiled",
        "avx512-not-compiled",
    ),
    (
        "native packed AVX-512 implementation is not runtime-validated",
        "avx512-not-runtime-validated",
    ),
    (
        "parallel AVX-512 implementation is not compiled",
        "parallel-avx512-not-compiled",
    ),
    (
        "parallel AVX-512 implementation is not runtime-validated",
        "parallel-avx512-not-runtime-validated",
    ),
    ("OpenBLAS CBLAS adapter is not linked", "openblas-not-linked"),
    (
        "OpenBLAS F32 implementation is not runtime-validated",
        "openblas-not-runtime-validated",
    ),
    (
        "OpenBLAS requested thread count exceeds provider maximum",
        "openblas-thread-ceiling",
    ),
)


def expected_legality_rejection(case: ParityCase, stderr: str) -> str | None:
    m, n, k = case.shape
    prefix = f"matcore-bench: variant planning failed for {m}x{n}x{k}: "
    if not stderr.strip().startswith(prefix):
        return None
    for reason, category in _EXPECTED_REJECTION_REASONS:
        if stderr.strip() == prefix + reason:
            return category
    return None


def plan_fingerprint(
    cases: list[ParityCase],
    suites: set[str],
    physical_cores: int,
    case_order: str,
    executable: pathlib.Path,
    limit: int,
) -> str:
    return canonical_sha256(
        {
            "schema": "matcore.native-blas-parity.plan",
            "version": 2,
            "suites": sorted(suites),
            "physical_cores": physical_cores,
            "thread_strata": list(thread_strata(physical_cores)),
            "parallel_thread_plan": parallel_thread_plan(physical_cores),
            "partition_interpretation": PARTITION_INTERPRETATION,
            "case_order": case_order,
            "benchmark": str(executable),
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "seed": DEFAULT_SEED,
            "warmup": DEFAULT_WARMUP,
            "iterations": DEFAULT_ITERATIONS,
            "timer_floor_us": DEFAULT_TIMER_FLOOR_US,
            "max_memory_mib": DEFAULT_MAX_MEMORY_MIB,
            "limit": limit,
            "cases": [dataclasses.asdict(case) for case in cases],
        }
    )


def atomic_write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_value(
    *,
    executable: pathlib.Path,
    benchmark_digest: str,
    runner_path: pathlib.Path,
    runner_digest: str,
    source: SourceIdentity,
    plan_digest: str,
    suites: set[str],
    physical_cores: int,
    case_order: str,
    limit: int,
    dry_run: bool,
    started: int,
    records: list[dict],
) -> dict:
    return {
        "schema": "matcore.native-blas-parity.manifest",
        "version": MANIFEST_VERSION,
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": str(executable),
        "benchmark_binary_sha256": benchmark_digest,
        "runner": str(runner_path),
        "runner_sha256": runner_digest,
        "runner_git_blob": source.runner_blob,
        "source_commit": source.commit,
        "plan_sha256": plan_digest,
        "benchmark_seed": DEFAULT_SEED,
        "started_unix_seconds": started,
        "finished_unix_seconds": int(time.time()),
        "suites": sorted(suites),
        "physical_cores": physical_cores,
        "thread_strata": list(thread_strata(physical_cores)),
        "parallel_thread_plan": parallel_thread_plan(physical_cores),
        "partition_interpretation": PARTITION_INTERPRETATION,
        "case_order": case_order,
        "warmup": DEFAULT_WARMUP,
        "iterations": DEFAULT_ITERATIONS,
        "timer_floor_us": DEFAULT_TIMER_FLOOR_US,
        "max_memory_mib": DEFAULT_MAX_MEMORY_MIB,
        "limit": limit,
        "dry_run": dry_run,
        "environment_overrides": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "cases": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--physical-cores", type=int, required=True)
    parser.add_argument(
        "--suites",
        default="all",
        help="comma list: parity,auto,regret,repeated,prepacked,diagnostic,all",
    )
    parser.add_argument(
        "--case-order",
        choices=("stable-forward", "stable-reverse"),
        default="stable-forward",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="test/development prefix bound; final evidence must use zero",
    )
    args = parser.parse_args()

    runner_path = pathlib.Path(__file__).resolve()
    source_root = runner_path.parents[3]
    try:
        source = authenticate_source(source_root, runner_path)
        output_dir = safe_output_directory(
            pathlib.Path(args.output_dir), source_root
        )
        threads = thread_strata(args.physical_cores)
    except ValueError as error:
        parser.error(str(error))

    executable = pathlib.Path(args.bench).resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable does not exist: {executable}")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.resume and args.dry_run:
        parser.error("--resume cannot be combined with --dry-run")

    suites = {value for value in args.suites.split(",") if value}
    known_suites = {
        "parity",
        "auto",
        "regret",
        "repeated",
        "prepacked",
        "diagnostic",
    }
    if "all" in suites:
        suites = set(known_suites)
    unknown_suites = suites - known_suites
    if not suites or unknown_suites:
        parser.error(f"unknown or empty suite set: {sorted(unknown_suites)}")

    try:
        cases = build_cases(suites, args.physical_cores)
    except ValueError as error:
        parser.error(str(error))
    if args.case_order == "stable-reverse":
        cases.reverse()
    if args.limit:
        cases = cases[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if not args.resume and any(output_dir.iterdir()):
        parser.error(
            "output directory is not empty; use a fresh directory or an "
            "identity-checked --resume"
        )

    benchmark_digest = sha256(executable)
    runner_digest = sha256(runner_path)
    plan_digest = plan_fingerprint(
        cases,
        suites,
        args.physical_cores,
        args.case_order,
        executable,
        args.limit,
    )

    prior_records: dict[str, dict] = {}
    started = int(time.time())
    if args.resume:
        if not manifest_path.is_file():
            parser.error("--resume requires an existing manifest.json")
        try:
            prior_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read resume manifest: {error}")
        expected_identity = {
            "schema": "matcore.native-blas-parity.manifest",
            "version": MANIFEST_VERSION,
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark": str(executable),
            "benchmark_binary_sha256": benchmark_digest,
            "runner": str(runner_path),
            "runner_sha256": runner_digest,
            "runner_git_blob": source.runner_blob,
            "source_commit": source.commit,
            "plan_sha256": plan_digest,
            "benchmark_seed": DEFAULT_SEED,
            "physical_cores": args.physical_cores,
            "thread_strata": list(threads),
            "partition_interpretation": PARTITION_INTERPRETATION,
        }
        for field, expected in expected_identity.items():
            if prior_manifest.get(field) != expected:
                parser.error(
                    f"resume identity mismatch for {field}: expected "
                    f"{expected!r}, found {prior_manifest.get(field)!r}"
                )
        started = int(prior_manifest.get("started_unix_seconds", started))
        for record in prior_manifest.get("cases", []):
            key = record.get("key")
            if not isinstance(key, str) or key in prior_records:
                parser.error(
                    "resume manifest contains an invalid or duplicate case key"
                )
            prior_records[key] = record

    records: list[dict] = []
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    def persist() -> None:
        atomic_write_json(
            manifest_path,
            manifest_value(
                executable=executable,
                benchmark_digest=benchmark_digest,
                runner_path=runner_path,
                runner_digest=runner_digest,
                source=source,
                plan_digest=plan_digest,
                suites=suites,
                physical_cores=args.physical_cores,
                case_order=args.case_order,
                limit=args.limit,
                dry_run=args.dry_run,
                started=started,
                records=records,
            ),
        )

    for index, case in enumerate(cases):
        raw_path = output_dir / f"{index:04d}__{case.key}.json"
        command = case_command(executable, case, raw_path)
        record = {
            "index": index,
            "key": case.key,
            "partition": case.partition,
            "family": case.family,
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

        prior = prior_records.get(case.key)
        if (
            args.resume
            and prior is not None
            and prior.get("state") in {"passed", "reused"}
        ):
            if (
                prior.get("raw_file") != raw_path.name
                or prior.get("command") != command
            ):
                parser.error(f"resume raw/command identity mismatch for {case.key}")
            if not raw_path.is_file():
                parser.error(f"resume raw file is missing for {case.key}")
            raw_digest = sha256(raw_path)
            if prior.get("sha256") != raw_digest:
                parser.error(f"resume raw-file digest mismatch for {case.key}")
            try:
                report = authenticate_report(
                    raw_path, case, source.commit, args.physical_cores
                )
            except ValueError as error:
                parser.error(f"resume report authentication failed: {error}")
            record["state"] = "reused"
            record["sha256"] = raw_digest
            record["selected_variant"] = report["results"][0]["selected_variant"]
            record["actual_threads"] = report["results"][0]["actual_threads"]
            records.append(record)
            persist()
            continue

        if sha256(executable) != benchmark_digest or sha256(runner_path) != runner_digest:
            record["state"] = "failed"
            record["error"] = "benchmark or runner bytes changed during execution"
            records.append(record)
            persist()
            break
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            rejection = expected_legality_rejection(case, completed.stderr)
            if raw_path.exists():
                rejection = None
                record["error"] = (
                    "failed benchmark left a raw artifact; rejection is not clean"
                )
            record["state"] = "rejected" if rejection else "failed"
            record["returncode"] = completed.returncode
            record["stdout"] = completed.stdout
            record["stderr"] = completed.stderr
            if rejection:
                record["rejection_category"] = rejection
            records.append(record)
            persist()
            if not rejection:
                break
            continue

        try:
            if not raw_path.is_file():
                raise ValueError("successful benchmark did not create raw JSON")
            if sha256(executable) != benchmark_digest:
                raise ValueError("benchmark bytes changed while a case executed")
            report = authenticate_report(
                raw_path, case, source.commit, args.physical_cores
            )
        except ValueError as error:
            record["state"] = "failed"
            record["returncode"] = completed.returncode
            record["stdout"] = completed.stdout
            record["stderr"] = completed.stderr
            record["error"] = str(error)
            records.append(record)
            persist()
            break
        record["state"] = "passed"
        record["sha256"] = sha256(raw_path)
        record["selected_variant"] = report["results"][0]["selected_variant"]
        record["actual_threads"] = report["results"][0]["actual_threads"]
        records.append(record)
        persist()

    if args.dry_run:
        persist()
    try:
        final_source = authenticate_source(source_root, runner_path)
    except ValueError as error:
        print(f"matcore native BLAS parity: source changed: {error}", file=sys.stderr)
        return 1
    if final_source != source or sha256(executable) != benchmark_digest:
        print(
            "matcore native BLAS parity: source or benchmark identity changed",
            file=sys.stderr,
        )
        return 1

    failures = [record for record in records if record["state"] == "failed"]
    rejections = [record for record in records if record["state"] == "rejected"]
    print(
        f"matcore native BLAS parity: {len(records)} cases, "
        f"{len(rejections)} exact legality rejections, {len(failures)} failures; "
        f"manifest={manifest_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
