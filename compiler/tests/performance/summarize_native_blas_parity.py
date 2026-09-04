#!/usr/bin/env python3

"""Authenticate, pair, and sanitize Milestone 7 native-BLAS parity evidence.

The benchmark runner keeps raw JSON outside Git.  This summarizer verifies the
two complete stable-order bundles against the corrected manifest-v3 authority,
reconstructs every timing and planner-regret aggregate, and emits only
deterministic path-free Markdown/JSON suitable for review.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Iterable


MANIFEST_SCHEMA = "matcore.native-blas-parity.manifest"
MANIFEST_VERSION = 3
BENCHMARK_SCHEMA = "matcore.benchmark.cpu.gemm"
BENCHMARK_VERSION = 6
SUMMARY_SCHEMA = "matcore.native-blas-parity.summary"
SUMMARY_VERSION = 3
PARTITION_INTERPRETATION = {
    "calibration": "candidate-development-and-validation",
    "holdout": "declared-validation-not-blind",
}
BENCHMARK_SEED = 0x4D4154434F524532
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 11
TIMER_FLOOR_US = 1000
MAX_MEMORY_MIB = 4096
REQUIRED_SUITES = {
    "parity",
    "auto",
    "regret",
    "repeated",
    "prepacked",
    "diagnostic",
}
SERIAL_NATIVE_VARIANTS = {
    "cpu.native-packed.avx2-fma.f32.v1",
    "cpu.native-packed.avx512-fma.f32.v1",
}
PARALLEL_NATIVE_VARIANTS = {
    "cpu.native-parallel.avx2-fma.f32.v1",
    "cpu.native-parallel.avx512-fma.f32.v1",
}
NATIVE_VARIANTS = SERIAL_NATIVE_VARIANTS | PARALLEL_NATIVE_VARIANTS
OPENBLAS = "cpu.external.openblas.f32.v1"
AUTO = "auto"
PROVIDER_METADATA_PLACEHOLDERS = {
    "unknown",
    "unavailable",
    "uninspected",
}
PLANNER_V3_VARIANTS = (
    "cpu.reference.f32.v1",
    "cpu.tiled.f32.v1",
    "cpu.compiler-vectorized.avx2-fma.f32.v1",
    OPENBLAS,
    "cpu.native-packed.avx2-fma.f32.v1",
    "cpu.native-packed.avx512-fma.f32.v1",
    "cpu.native-parallel.avx2-fma.f32.v1",
    "cpu.native-parallel.avx512-fma.f32.v1",
)
CORE_FAMILIES = {
    "medium-square",
    "large-square",
    "tall-skinny",
    "short-wide",
    "tail-heavy",
}
MEANINGFUL_PARITY_FAMILIES = CORE_FAMILIES - {"tail-heavy"}
SQUARE_ENVELOPE = {"medium-square", "large-square"}
PROVIDER_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
MILESTONE6_SINGLE_THREAD_FAMILY_MEDIANS = {
    "medium-square": 0.868,
    "large-square": 0.849,
    "tall-skinny": 0.795,
    "short-wide": 0.884,
    "tail-heavy": 0.843,
}


class ParityError(RuntimeError):
    """Evidence is incomplete, unauthenticated, or not comparable."""


@dataclasses.dataclass(frozen=True)
class EvidenceCell:
    semantic_key: tuple
    partition: str
    family: str
    shape: tuple[int, int, int]
    variant: str
    requested_threads: int
    actual_threads: int
    mode: str
    lhs_sequence: int
    selected_variant: str
    placement: tuple
    seconds: float
    seconds_low: float
    seconds_high: float
    preparation_seconds: float
    amortized_seconds: float
    planner_regret: float | None
    checksum: float
    report: dict

    @property
    def operations(self) -> int:
        m, n, k = self.shape
        return 2 * m * n * k

    @property
    def gflops(self) -> float:
        return self.operations / self.seconds / 1.0e9


@dataclasses.dataclass(frozen=True)
class Bundle:
    path: pathlib.Path
    digest: str
    manifest: dict
    cells: dict[tuple, EvidenceCell]
    statuses: dict[tuple, tuple[str, str | None]]
    states: Counter
    rejection_categories: Counter
    environment: dict
    summarizer_sha256: str
    summarizer_git_blob: str


@dataclasses.dataclass(frozen=True)
class Comparison:
    partition: str
    family: str
    shape: tuple[int, int, int]
    threads: int
    native: EvidenceCell
    provider: EvidenceCell

    @property
    def ratio(self) -> float:
        return self.provider.seconds / self.native.seconds


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParityError(message)


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


def require_sha256(value: object, field: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} is not a lowercase SHA-256 digest",
    )
    return value


def load_json(path: pathlib.Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ParityError(f"cannot read {description}: {error}") from error
    require(isinstance(value, dict), f"{description} is not a JSON object")
    return value


def run_git(source_root: pathlib.Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"Git source authentication failed for {' '.join(arguments)}: "
        f"{completed.stderr.strip()}",
    )
    return completed.stdout.strip()


def load_runner():
    runner_path = pathlib.Path(__file__).resolve().with_name(
        "run_native_blas_parity.py"
    )
    specification = importlib.util.spec_from_file_location(
        "_matcore_native_blas_parity_plan_authority", runner_path
    )
    require(
        specification is not None and specification.loader is not None,
        "cannot load the native-BLAS parity plan authority",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module, runner_path


def authenticate_source_and_tools(
    manifest: dict, runner_path: pathlib.Path
) -> tuple[pathlib.Path, str, str]:
    source_root = runner_path.parents[3]
    summarizer_path = pathlib.Path(__file__).resolve()
    source_commit = manifest.get("source_commit")
    require(
        isinstance(source_commit, str)
        and len(source_commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in source_commit),
        "manifest source commit is missing or invalid",
    )
    relative_runner = runner_path.relative_to(source_root).as_posix()
    require(
        sha256(runner_path) == manifest.get("runner_sha256"),
        "local frozen runner SHA-256 differs from manifest authority",
    )
    runner_blob = run_git(source_root, "hash-object", str(runner_path))
    require(
        runner_blob == manifest.get("runner_git_blob"),
        "local frozen runner Git blob differs from manifest authority",
    )
    run_git(source_root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    committed_runner = run_git(
        source_root, "rev-parse", f"{source_commit}:{relative_runner}"
    )
    require(
        committed_runner == runner_blob,
        "manifest source commit does not contain the authenticated runner blob",
    )
    relative_summarizer = summarizer_path.relative_to(source_root).as_posix()
    summarizer_digest = sha256(summarizer_path)
    summarizer_blob = run_git(
        source_root, "hash-object", str(summarizer_path)
    )
    committed_summarizer = run_git(
        source_root, "rev-parse", f"{source_commit}:{relative_summarizer}"
    )
    require(
        committed_summarizer == summarizer_blob,
        "local parity summarizer differs from the declared source commit",
    )

    benchmark = pathlib.Path(str(manifest.get("benchmark", "")))
    require(
        benchmark.is_absolute() and benchmark.is_file(),
        "manifest benchmark executable is absent or is not an absolute file",
    )
    require(
        sha256(benchmark) == manifest.get("benchmark_binary_sha256"),
        "benchmark binary SHA-256 differs from the manifest",
    )
    return benchmark, summarizer_digest, summarizer_blob


def thread_strata(physical_cores: int) -> tuple[int, ...]:
    require(physical_cores >= 2, "physical core count must be at least two")
    return tuple(dict.fromkeys((1, 4, physical_cores)))


def parallel_task_capacity(
    shape: tuple[int, int, int], requested_threads: int
) -> int:
    """Mirror planner-v3's versioned disjoint-output task-capacity contract."""
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
    exact: list[int] = []
    for ceiling in dict.fromkeys((4, physical_cores)):
        capacity = parallel_task_capacity(shape, ceiling)
        if capacity > 1 and capacity not in exact:
            exact.append(capacity)
    return tuple(exact)


def expected_parallel_thread_plan(
    runner: object, physical_cores: int
) -> list[dict[str, object]]:
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
        for spec in runner.PARITY_SHAPES
    ]


def expected_cases(runner: object, physical_cores: int) -> list:
    """Independently reconstruct the complete manifest-v3 matrix."""
    cases = []
    threads = thread_strata(physical_cores)
    for spec in runner.PARITY_SHAPES:
        for variant in runner.SERIAL_PARITY_VARIANTS:
            cases.append(
                runner.ParityCase(
                    spec.partition,
                    spec.family,
                    spec.shape,
                    variant,
                    1,
                    "complete-hot",
                )
            )
        for count in exact_parallel_thread_strata(spec.shape, physical_cores):
            for variant in runner.PARALLEL_PARITY_VARIANTS:
                cases.append(
                    runner.ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        variant,
                        count,
                        "complete-hot",
                    )
                )

    for spec in runner.PARITY_SHAPES:
        for count in threads:
            cases.append(
                runner.ParityCase(
                    spec.partition,
                    spec.family,
                    spec.shape,
                    AUTO,
                    count,
                    "auto-complete-hot",
                )
            )

    for spec in runner.PARITY_SHAPES:
        if spec.shape not in runner.REGRET_SHAPES:
            continue
        for count in threads:
            cases.append(
                runner.ParityCase(
                    spec.partition,
                    spec.family,
                    spec.shape,
                    AUTO,
                    count,
                    "planner-regret-hot",
                )
            )

    for spec in runner.PARITY_SHAPES:
        if spec.shape not in runner.REPEATED_SHAPES:
            continue
        for variant in runner.REPEATED_VARIANTS:
            for sequence in runner.SEQUENCES:
                cases.append(
                    runner.ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        variant,
                        1,
                        "repeated-hot",
                        sequence,
                    )
                )

    for spec in runner.PARITY_SHAPES:
        if spec.shape not in runner.REPEATED_SHAPES:
            continue
        for variant in runner.PREPACKED_VARIANTS:
            for sequence in runner.SEQUENCES:
                cases.append(
                    runner.ParityCase(
                        spec.partition,
                        spec.family,
                        spec.shape,
                        variant,
                        1,
                        "prepacked-b-hot",
                        sequence,
                    )
                )

    for spec in runner.DIAGNOSTIC_SHAPES:
        for variant in runner.SERIAL_PARITY_VARIANTS:
            cases.append(
                runner.ParityCase(
                    spec.partition,
                    spec.family,
                    spec.shape,
                    variant,
                    1,
                    "diagnostic-complete-hot",
                )
            )
    keys = [case.key for case in cases]
    require(
        len(keys) == len(set(keys)),
        "independent manifest-v3 authority produced duplicate case keys",
    )
    return cases


def reconstructed_plan_digest(
    cases: list,
    physical_cores: int,
    case_order: str,
    benchmark: pathlib.Path,
    parallel_plan: list[dict[str, object]],
) -> str:
    return canonical_sha256(
        {
            "schema": "matcore.native-blas-parity.plan",
            "version": 2,
            "suites": sorted(REQUIRED_SUITES),
            "physical_cores": physical_cores,
            "thread_strata": list(thread_strata(physical_cores)),
            "parallel_thread_plan": parallel_plan,
            "partition_interpretation": PARTITION_INTERPRETATION,
            "case_order": case_order,
            "benchmark": str(benchmark),
            "benchmark_schema_version": BENCHMARK_VERSION,
            "seed": BENCHMARK_SEED,
            "warmup": WARMUP_ITERATIONS,
            "iterations": MEASURED_ITERATIONS,
            "timer_floor_us": TIMER_FLOOR_US,
            "max_memory_mib": MAX_MEMORY_MIB,
            "limit": 0,
            "cases": [dataclasses.asdict(case) for case in cases],
        }
    )


def semantic_key(record: dict) -> tuple:
    shape = record.get("shape")
    require(
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(value, int) and value > 0 for value in shape),
        "manifest case has an invalid shape",
    )
    for field in ("partition", "family", "variant", "mode"):
        require(
            isinstance(record.get(field), str) and record[field],
            f"manifest case has an invalid {field}",
        )
    require(
        isinstance(record.get("threads"), int) and record["threads"] > 0,
        "manifest case has an invalid thread count",
    )
    require(
        isinstance(record.get("lhs_sequence"), int)
        and record["lhs_sequence"] > 0,
        "manifest case has an invalid lhs sequence length",
    )
    return (
        record["partition"],
        record["family"],
        tuple(shape),
        record["variant"],
        record["threads"],
        record["mode"],
        record["lhs_sequence"],
    )


def expected_record_fields(case: object) -> dict:
    return {
        "key": case.key,
        "partition": case.partition,
        "family": case.family,
        "shape": list(case.shape),
        "variant": case.variant,
        "threads": case.threads,
        "mode": case.mode,
        "lhs_sequence": case.lhs_sequence,
    }


def authenticate_case_command(
    runner: object,
    benchmark: pathlib.Path,
    case: object,
    record: dict,
) -> pathlib.Path:
    command = record.get("command")
    require(
        isinstance(command, list)
        and command
        and all(isinstance(argument, str) for argument in command),
        "manifest case command is missing or malformed",
    )
    require(
        command.count("--json-out") == 1,
        "manifest case command has an invalid JSON-output argument",
    )
    output_index = command.index("--json-out") + 1
    require(
        output_index < len(command),
        "manifest case command omits the JSON-output path",
    )
    command_output = pathlib.Path(command[output_index])
    require(
        command_output.name == record["raw_file"],
        "manifest command output differs from raw-file identity",
    )
    expected = runner.case_command(benchmark, case, command_output)
    require(
        command == expected,
        f"manifest command does not reconstruct for {record['key']}",
    )
    return command_output


def reconstructed_timing(result: dict) -> tuple[float, float, float]:
    samples = result.get("normalized_samples_seconds")
    require(
        isinstance(samples, list)
        and len(samples) == MEASURED_ITERATIONS
        and all(
            isinstance(sample, (int, float))
            and not isinstance(sample, bool)
            and math.isfinite(float(sample))
            and float(sample) > 0.0
            for sample in samples
        ),
        "raw report has missing or invalid ordered timing samples",
    )
    ordered = sorted(float(sample) for sample in samples)
    aggregate_repetitions = result.get("aggregate_repetitions")
    require(
        isinstance(aggregate_repetitions, int)
        and aggregate_repetitions >= 1
        and result.get("timing_aggregation_boundary")
        == "one-clock-pair-per-aggregate-repetition-block"
        and all(
            sample * aggregate_repetitions * 1.0e9
            >= TIMER_FLOOR_US * 1000
            for sample in ordered
        ),
        "raw aggregate timing does not meet the frozen timer floor",
    )
    values = (
        ordered[0],
        ordered[len(ordered) // 2],
        ordered[math.ceil(len(ordered) * 0.95) - 1],
    )
    for field, expected in zip(
        ("minimum_seconds", "median_seconds", "p95_seconds"),
        values,
    ):
        require(
            isinstance(result.get(field), (int, float))
            and not isinstance(result[field], bool)
            and math.isclose(
                float(result[field]),
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            f"{field} does not reconstruct from authenticated samples",
        )
    return values


def authenticate_prepacked(
    result: dict, mode: str, lhs_sequence: int, median_seconds: float
) -> tuple[float, float]:
    preparation = result.get("prepacked_b_preparation")
    require(isinstance(preparation, dict), "prepacked-B metadata is missing")
    if mode != "prepacked-b-hot":
        require(
            preparation.get("requested") is False
            and preparation.get("measured") is False
            and preparation.get("authenticated") is False
            and preparation.get("preparation_calls") == 0,
            "non-prepacked case contains unexpected preparation evidence",
        )
        return 0.0, 0.0
    require(
        preparation.get("requested") is True
        and preparation.get("measured") is True
        and preparation.get("authenticated") is True
        and preparation.get("preparation_calls") == 1
        and preparation.get("amortization_executions") == lhs_sequence
        and preparation.get("amortized_total_valid") is True
        and preparation.get("input_state")
        == "caller-storage-allocated-unprepared"
        and preparation.get("output_state") == "prepared-authenticated",
        "prepacked-B preparation is not authenticated",
    )
    prepare = float(preparation.get("preparation_seconds", 0.0))
    steady = float(preparation.get("steady_state_sequence_seconds", 0.0))
    total = float(preparation.get("amortized_total_sequence_seconds", 0.0))
    amortized = float(
        preparation.get("amortized_per_execution_seconds", 0.0)
    )
    require(
        prepare > 0.0
        and math.isclose(
            steady,
            median_seconds * lhs_sequence,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            total,
            prepare + steady,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            amortized,
            total / lhs_sequence,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        "prepacked-B preparation/amortization arithmetic does not reconstruct",
    )
    return prepare, amortized


def authenticate_planner_regret(
    result: dict,
    requested: bool,
    requested_threads: int,
    smt_policy: str,
    affinity_policy: str,
) -> float | None:
    planner = result.get("planner_regret")
    require(isinstance(planner, dict), "planner-regret metadata is missing")
    if not requested:
        require(
            planner.get("requested") is False,
            "non-regret case contains planner-regret evidence",
        )
        return None
    require(
        planner.get("requested") is True
        and planner.get("valid") is True
        and planner.get("aggregation_method")
        == "arithmetic-mean-of-forward-and-reverse-pass-medians",
        "planner-regret evidence is invalid",
    )
    candidates = planner.get("candidates")
    require(
        isinstance(candidates, list)
        and [candidate.get("variant") for candidate in candidates
             if isinstance(candidate, dict)]
        == list(PLANNER_V3_VARIANTS),
        "planner-regret candidates differ from the complete ordered v3 "
        "registry",
    )
    reconstructed: list[tuple[dict, float]] = []
    for candidate in candidates:
        require(
            isinstance(candidate, dict),
            "planner-regret candidate is malformed",
        )
        if candidate.get("legal"):
            require(
                candidate.get("selected_variant") == candidate.get("variant")
                and candidate.get("plan_authenticated") is True,
                "forced planner-regret candidate was substituted",
            )
        comparable = bool(
            candidate.get("legal")
            and candidate.get("complete_implementation_comparison")
        )
        require(
            candidate.get("timing_valid") is comparable,
            "planner-regret timing validity differs from legal complete-call "
            "comparability",
        )
        if not candidate.get("timing_valid"):
            continue
        require(
            candidate.get("correctness") is True,
            "timed planner-regret candidate failed correctness",
        )
        actual_threads = candidate.get("actual_threads")
        require(
            isinstance(actual_threads, int)
            and 1 <= actual_threads <= requested_threads,
            "planner-regret candidate exceeded the requested thread ceiling",
        )
        forward_checks = candidate.get(
            "forward_pass_untimed_validation_executions_checked"
        )
        reverse_checks = candidate.get(
            "reverse_pass_untimed_validation_executions_checked"
        )
        require(
            candidate.get("complete_implementation_comparison") is True
            and candidate.get("smt_policy") == smt_policy
            and candidate.get("affinity_policy") == affinity_policy
            and isinstance(forward_checks, int)
            and forward_checks >= 1
            and isinstance(reverse_checks, int)
            and reverse_checks >= 1,
            "planner-regret candidate is not an equivalent complete-call placement",
        )
        medians = []
        for pass_name in ("forward", "reverse"):
            samples = candidate.get(
                f"{pass_name}_pass_normalized_samples_seconds"
            )
            require(
                isinstance(samples, list)
                and len(samples) == MEASURED_ITERATIONS
                and all(
                    isinstance(sample, (int, float))
                    and not isinstance(sample, bool)
                    and math.isfinite(float(sample))
                    and float(sample) > 0.0
                    for sample in samples
                ),
                f"planner-regret {pass_name} samples are invalid",
            )
            median_seconds = sorted(float(sample) for sample in samples)[
                len(samples) // 2
            ]
            require(
                math.isclose(
                    float(candidate[f"{pass_name}_pass_median_seconds"]),
                    median_seconds,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ),
                f"planner-regret {pass_name} median does not reconstruct",
            )
            medians.append(median_seconds)
        balanced = sum(medians) / 2.0
        require(
            math.isclose(
                float(candidate["balanced_estimate_seconds"]),
                balanced,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            "planner-regret balanced estimate does not reconstruct",
        )
        reconstructed.append((candidate, balanced))
    require(
        len(reconstructed) >= 2,
        "planner-regret has fewer than two comparable timed candidates",
    )
    fastest = min(reconstructed, key=lambda item: item[1])
    selected = [
        item
        for item in reconstructed
        if item[0]["variant"] == result.get("selected_variant")
    ]
    require(
        len(selected) == 1,
        "selected planner variant has no unique authenticated timing candidate",
    )
    require(
        selected[0][0]["actual_threads"] == result.get("actual_threads"),
        "selected planner candidate thread count differs from the timed result",
    )
    regret = selected[0][1] / fastest[1]
    require(
        planner.get("fastest_legal_variant") == fastest[0]["variant"]
        and math.isclose(
            float(planner["fastest_legal_balanced_estimate_seconds"]),
            fastest[1],
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            float(planner["selected_balanced_estimate_seconds"]),
            selected[0][1],
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            float(planner["regret"]),
            regret,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        "planner-regret aggregate does not reconstruct",
    )
    return regret


def environment_identity_signature(environment: dict) -> tuple:
    fields = (
        "os_family",
        "architecture",
        "compiler",
        "compiler_flags",
        "build_type",
        "cpu_model",
        "governor",
        "frequency_policy",
        "boost_state",
        "cpu_affinity",
        "hardware_threads",
        "source_provenance_origin",
        "timer_source",
        "timer_resolution_ns",
        "capability_record",
        "capability_runtime_validation_source",
        "capability_record_version",
        "topology_record_version",
        "topology_discovery_complete",
        "logical_processors",
        "physical_cores",
        "numa_nodes",
        "persistent_execution_context",
        "available_processors",
        "provider_name",
        "provider_version",
        "provider_config",
    )
    return tuple(environment.get(field) for field in fields)


def authenticate_report(
    report: dict,
    record: dict,
    manifest: dict,
) -> EvidenceCell:
    require(report.get("schema") == BENCHMARK_SCHEMA, "unexpected raw schema")
    require(report.get("version") == BENCHMARK_VERSION, "unexpected raw version")
    require(
        report.get("operation") == "matcore.gemm"
        and report.get("dtype") == "f32"
        and report.get("accumulation_dtype") == "f32"
        and report.get("layout") == "row-major-contiguous",
        "raw operation contract is not F32 row-major contiguous GEMM",
    )
    environment = report.get("environment")
    configuration = report.get("configuration")
    results = report.get("results")
    require(isinstance(environment, dict), "raw environment is missing")
    require(isinstance(configuration, dict), "raw configuration is missing")
    require(
        isinstance(results, list) and len(results) == 1,
        "raw report must contain exactly one result",
    )
    require(
        environment.get("source_provenance_state") == "clean"
        and environment.get("source_worktree_dirty") is False
        and environment.get("source_provenance_origin") == "git-worktree",
        "raw source provenance is not a clean Git worktree",
    )
    require(
        environment.get("source_commit") == manifest["source_commit"],
        "raw source commit differs from manifest authority",
    )
    require(
        environment.get("os_family") == "linux"
        and environment.get("architecture") == "x86_64"
        and environment.get("topology_discovery_complete") is True
        and environment.get("physical_cores") == manifest["physical_cores"],
        "raw report is not from the declared Linux x86-64 topology",
    )

    key = semantic_key(record)
    partition, family, shape, variant, requested_threads, mode, lhs_sequence = key
    expected_configuration = {
        "profile": "custom",
        "requested_variant": variant,
        "requested_threads": requested_threads,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "lhs_sequence_length": lhs_sequence,
        "alignment_bytes": 64,
        "cache_mode": "hot",
        "allocation_mode": "reuse-workspace",
        "packing_mode": (
            "prepacked-b" if mode == "prepacked-b-hot" else "include-packing"
        ),
        "smt_policy": (
            "allow-smt" if requested_threads > 1 else "physical-cores-only"
        ),
        "affinity_policy": "none" if requested_threads > 1 else "compact",
        "maximum_memory_bytes": MAX_MEMORY_MIB * 1024 * 1024,
        "timer_floor_ns": TIMER_FLOOR_US * 1000,
        "seed": BENCHMARK_SEED,
        "compare_one_thread": False,
        "planner_regret": mode == "planner-regret-hot",
    }
    for field, expected in expected_configuration.items():
        require(
            configuration.get(field) == expected,
            f"raw configuration mismatch for {field}: expected "
            f"{expected!r}, found {configuration.get(field)!r}",
        )

    result = results[0]
    require(isinstance(result, dict), "raw result is malformed")
    require(
        (result.get("m"), result.get("n"), result.get("k")) == shape,
        "raw shape differs from manifest case",
    )
    require(
        result.get("requested_variant") == variant,
        "raw requested variant differs from manifest case",
    )
    if variant == AUTO:
        require(
            result.get("planner_mode") == "automatic"
            and isinstance(result.get("selected_variant"), str)
            and result["selected_variant"]
            and result["selected_variant"] != AUTO,
            "automatic selected variant is malformed",
        )
    else:
        require(
            result.get("planner_mode") == "forced"
            and result.get("selected_variant") == variant,
            "forced raw variant was silently substituted",
        )
    if result.get("selected_variant") == OPENBLAS:
        provider_version = environment.get("provider_version")
        provider_config = environment.get("provider_config")
        require(
            environment.get("provider_name") == "OpenBLAS"
            and isinstance(provider_version, str)
            and bool(provider_version.strip())
            and provider_version == provider_version.strip()
            and provider_version.casefold() not in PROVIDER_METADATA_PLACEHOLDERS
            and isinstance(provider_config, str)
            and bool(provider_config.strip())
            and provider_config == provider_config.strip()
            and provider_config.casefold() not in PROVIDER_METADATA_PLACEHOLDERS,
            "selected OpenBLAS result lacks required provider metadata",
        )
    require(
        result.get("complete_implementation_comparison") is True,
        "raw result is not a complete implementation call",
    )
    validation_checks = result.get("untimed_validation_executions_checked")
    require(
        result.get("timing_valid") is True
        and result.get("correctness") is True
        and result.get("timed_final_output_authenticated") is True
        and isinstance(validation_checks, int)
        and validation_checks >= 1,
        "raw timing, correctness, or final output is unauthenticated",
    )
    require(
        result.get("timing_rejection_reason") in {"", None}
        and result.get("cache_mode") == expected_configuration["cache_mode"]
        and result.get("allocation_mode")
        == expected_configuration["allocation_mode"]
        and result.get("packing_mode")
        == expected_configuration["packing_mode"],
        "raw result timing scope contradicts its configuration",
    )
    actual_threads = result.get("actual_threads")
    require(
        isinstance(actual_threads, int)
        and 1 <= actual_threads <= requested_threads
        and record.get("actual_threads") == actual_threads,
        "raw actual thread count differs from the manifest",
    )
    if variant != AUTO:
        require(
            actual_threads == requested_threads,
            "forced parity result did not use the exact requested thread count",
        )
    expected_smt = expected_configuration["smt_policy"]
    expected_affinity = expected_configuration["affinity_policy"]
    require(
        result.get("smt_policy") == expected_smt
        and result.get("affinity_policy") == expected_affinity,
        "raw result placement differs from its configuration",
    )
    if requested_threads > 1:
        require(
            result.get("worker_affinity_applied") is False
            and result.get("worker_affinity_user_requested") is False
            and result.get("worker_affinity_policy_induced") is False,
            "multi-thread parity result is not in the frozen unbound stratum",
        )
    else:
        require(
            result.get("worker_affinity_applied") is True,
            "single-thread compact placement was not authenticated",
        )

    parallel_fields = (
        "parallel_row_tasks",
        "parallel_column_tasks",
        "parallel_task_count",
    )
    if variant in PARALLEL_NATIVE_VARIANTS:
        row_tasks = result.get(parallel_fields[0])
        column_tasks = result.get(parallel_fields[1])
        task_count = result.get(parallel_fields[2])
        require(
            isinstance(row_tasks, int)
            and isinstance(column_tasks, int)
            and isinstance(task_count, int)
            and row_tasks >= 1
            and column_tasks >= 1
            and task_count == row_tasks * column_tasks
            and task_count
            == parallel_task_capacity(shape, requested_threads),
            "parallel task geometry differs from manifest-v3 capacity",
        )
    else:
        require(
            all(result.get(field, 0) == 0 for field in parallel_fields),
            "serial/provider result unexpectedly reports parallel task geometry",
        )

    minimum, median_seconds, p95 = reconstructed_timing(result)
    operations = 2 * shape[0] * shape[1] * shape[2]
    require(
        isinstance(result.get("gflops"), (int, float))
        and math.isclose(
            float(result["gflops"]),
            operations / median_seconds / 1.0e9,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "raw GFLOP/s does not reconstruct from shape and timing",
    )
    for field in (
        "checksum",
        "expected_checksum",
        "maximum_absolute_error",
        "maximum_allowed_error",
    ):
        require(
            isinstance(result.get(field), (int, float))
            and not isinstance(result[field], bool)
            and math.isfinite(float(result[field])),
            f"raw {field} is not finite",
        )
    require(
        float(result["maximum_absolute_error"])
        <= float(result["maximum_allowed_error"]),
        "raw correctness error exceeds its declared tolerance",
    )
    preparation, amortized = authenticate_prepacked(
        result, mode, lhs_sequence, median_seconds
    )
    regret = authenticate_planner_regret(
        result,
        mode == "planner-regret-hot",
        requested_threads,
        str(expected_smt),
        str(expected_affinity),
    )
    return EvidenceCell(
        semantic_key=key,
        partition=partition,
        family=family,
        shape=shape,
        variant=variant,
        requested_threads=requested_threads,
        actual_threads=actual_threads,
        mode=mode,
        lhs_sequence=lhs_sequence,
        selected_variant=str(result["selected_variant"]),
        placement=(
            actual_threads,
            expected_smt,
            expected_affinity,
            bool(result.get("worker_affinity_applied")),
            bool(result.get("worker_affinity_user_requested")),
            bool(result.get("worker_affinity_policy_induced")),
        ),
        seconds=median_seconds,
        seconds_low=minimum,
        seconds_high=p95,
        preparation_seconds=preparation,
        amortized_seconds=amortized,
        planner_regret=regret,
        checksum=float(result["checksum"]),
        report=report,
    )


def authenticate_rejection(
    runner: object,
    case: object,
    record: dict,
    manifest_directory: pathlib.Path,
    command_output: pathlib.Path,
) -> str:
    require(
        isinstance(record.get("returncode"), int)
        and record["returncode"] != 0,
        "expected legality rejection has an invalid return code",
    )
    require(
        record.get("stdout") == ""
        and isinstance(record.get("stderr"), str)
        and record["stderr"],
        "expected legality rejection lacks an actionable diagnostic",
    )
    category = runner.expected_legality_rejection(case, record["stderr"])
    require(
        category is not None and record.get("rejection_category") == category,
        "rejected case is not an exact frozen legality rejection",
    )
    require(
        "sha256" not in record
        and not (manifest_directory / record["raw_file"]).exists()
        and not command_output.exists(),
        "rejected case unexpectedly produced raw evidence",
    )
    return str(category)


def load_bundle(path: pathlib.Path, expected_order: str) -> Bundle:
    path = path.resolve()
    manifest = load_json(path, "native-BLAS parity manifest")
    require(
        manifest.get("schema") == MANIFEST_SCHEMA,
        "unexpected parity manifest schema",
    )
    require(
        manifest.get("version") == MANIFEST_VERSION,
        "native-BLAS parity summary requires manifest v3",
    )
    require(
        manifest.get("benchmark_schema_version") == BENCHMARK_VERSION,
        "native-BLAS parity summary requires benchmark schema v6",
    )
    for field in ("benchmark_binary_sha256", "runner_sha256", "plan_sha256"):
        require_sha256(manifest.get(field), field)
    require(
        manifest.get("benchmark_seed") == BENCHMARK_SEED
        and manifest.get("warmup") == WARMUP_ITERATIONS
        and manifest.get("iterations") == MEASURED_ITERATIONS
        and manifest.get("timer_floor_us") == TIMER_FLOOR_US
        and manifest.get("max_memory_mib") == MAX_MEMORY_MIB,
        "manifest timing, seed, or memory contract differs from frozen v3",
    )
    require(
        manifest.get("environment_overrides") == PROVIDER_ENVIRONMENT,
        "provider environment differs from the frozen parity contract",
    )
    require(
        manifest.get("case_order") == expected_order,
        f"expected a {expected_order} parity manifest",
    )
    require(
        set(manifest.get("suites", [])) == REQUIRED_SUITES,
        "manifest does not contain the complete required suite set",
    )
    require(
        manifest.get("limit") == 0 and manifest.get("dry_run") is False,
        "limited or dry-run parity manifests are not evidence",
    )
    require(
        isinstance(manifest.get("started_unix_seconds"), int)
        and isinstance(manifest.get("finished_unix_seconds"), int)
        and manifest["finished_unix_seconds"]
        >= manifest["started_unix_seconds"],
        "manifest run interval is invalid",
    )
    physical_cores = manifest.get("physical_cores")
    require(
        isinstance(physical_cores, int) and physical_cores >= 2,
        "manifest physical core count is invalid",
    )
    require(
        manifest.get("thread_strata") == list(thread_strata(physical_cores)),
        "manifest thread strata differ from the frozen contract",
    )

    runner, runner_path = load_runner()
    require(
        getattr(runner, "PARTITION_INTERPRETATION", None)
        == PARTITION_INTERPRETATION,
        "runner partition interpretation differs from the corrected "
        "validation contract",
    )
    require(
        manifest.get("partition_interpretation")
        == PARTITION_INTERPRETATION,
        "manifest partition interpretation differs from the corrected "
        "validation contract",
    )
    benchmark, summarizer_digest, summarizer_blob = (
        authenticate_source_and_tools(manifest, runner_path)
    )
    parallel_plan = expected_parallel_thread_plan(runner, physical_cores)
    require(
        manifest.get("parallel_thread_plan") == parallel_plan,
        "manifest parallel-thread plan differs from v3 task capacity",
    )
    cases = expected_cases(runner, physical_cores)
    if expected_order == "stable-reverse":
        cases.reverse()
    records = manifest.get("cases")
    require(
        isinstance(records, list) and len(records) == len(cases),
        "manifest cases differ from the complete frozen v3 matrix",
    )
    require(
        manifest.get("plan_sha256")
        == reconstructed_plan_digest(
            cases,
            physical_cores,
            expected_order,
            benchmark,
            parallel_plan,
        ),
        "manifest plan SHA-256 does not reconstruct",
    )

    cells: dict[tuple, EvidenceCell] = {}
    statuses: dict[tuple, tuple[str, str | None]] = {}
    states: Counter = Counter()
    rejections: Counter = Counter()
    representative_environment: dict | None = None
    environment_signature: tuple | None = None
    manifest_directory = path.parent
    seen_keys: set[str] = set()
    for index, (case, record) in enumerate(zip(cases, records)):
        require(isinstance(record, dict), "manifest case is not an object")
        for field, expected in expected_record_fields(case).items():
            require(
                record.get(field) == expected,
                f"manifest cases differ from frozen v3 matrix at {case.key}",
            )
        require(
            record["key"] not in seen_keys,
            "manifest contains a duplicate case key",
        )
        seen_keys.add(record["key"])
        require(
            record.get("index") == index,
            "manifest case index differs from stable execution order",
        )
        raw_name = record.get("raw_file")
        require(
            isinstance(raw_name, str)
            and pathlib.PurePath(raw_name).name == raw_name
            and raw_name == f"{index:04d}__{case.key}.json",
            "manifest raw filename differs from stable case identity",
        )
        command_output = authenticate_case_command(
            runner, benchmark, case, record
        )
        key = semantic_key(record)
        state = record.get("state")
        require(
            state in {"passed", "reused", "rejected"},
            f"incomplete parity bundle contains state {state!r}",
        )
        states[state] += 1
        if state == "rejected":
            category = authenticate_rejection(
                runner,
                case,
                record,
                manifest_directory,
                command_output,
            )
            rejections[category] += 1
            statuses[key] = ("rejected", category)
            continue

        raw_path = manifest_directory / raw_name
        require(
            raw_path.is_file(),
            f"raw parity evidence is missing for {case.key}",
        )
        digest = require_sha256(record.get("sha256"), "raw sha256")
        require(
            sha256(raw_path) == digest,
            f"raw parity SHA-256 mismatch for {case.key}",
        )
        report = load_json(raw_path, f"raw parity evidence for {case.key}")
        cell = authenticate_report(report, record, manifest)
        require(
            record.get("selected_variant") == cell.selected_variant
            and record.get("actual_threads") == cell.actual_threads,
            "manifest selected variant or actual threads differ from raw evidence",
        )
        require(key not in cells, "duplicate semantic parity cell")
        cells[key] = cell
        statuses[key] = ("accepted", None)
        signature = environment_identity_signature(report["environment"])
        if environment_signature is None:
            environment_signature = signature
            representative_environment = report["environment"]
        require(
            signature == environment_signature,
            "raw reports do not share one homogeneous benchmark environment",
        )
    require(cells, "parity manifest contains no accepted raw evidence")
    require(
        sum(states.values()) == len(records),
        "parity manifest case accounting is incomplete",
    )
    return Bundle(
        path=path,
        digest=sha256(path),
        manifest=manifest,
        cells=cells,
        statuses=statuses,
        states=states,
        rejection_categories=rejections,
        environment=representative_environment or {},
        summarizer_sha256=summarizer_digest,
        summarizer_git_blob=summarizer_blob,
    )


def pair_bundles(
    forward: Bundle, reverse: Bundle
) -> dict[tuple, EvidenceCell]:
    identity_fields = (
        "benchmark_schema_version",
        "benchmark",
        "benchmark_binary_sha256",
        "runner_sha256",
        "runner_git_blob",
        "source_commit",
        "plan_sha256",
        "benchmark_seed",
        "physical_cores",
        "thread_strata",
        "parallel_thread_plan",
        "warmup",
        "iterations",
        "timer_floor_us",
        "max_memory_mib",
        "limit",
        "dry_run",
        "environment_overrides",
    )
    for field in identity_fields:
        if field == "plan_sha256":
            continue
        require(
            forward.manifest.get(field) == reverse.manifest.get(field),
            f"forward/reverse identity differs for {field}",
        )
    require(
        set(forward.statuses) == set(reverse.statuses),
        "forward/reverse manifests do not contain the same semantic cells",
    )
    require(
        forward.statuses == reverse.statuses,
        "forward/reverse per-cell acceptance status differs",
    )
    require(
        environment_identity_signature(forward.environment)
        == environment_identity_signature(reverse.environment),
        "forward/reverse benchmark environments differ",
    )
    require(
        set(forward.cells) == set(reverse.cells),
        "forward/reverse accepted raw cells differ",
    )
    paired = {}
    for key, first in forward.cells.items():
        second = reverse.cells[key]
        require(
            first.actual_threads == second.actual_threads
            and first.placement == second.placement
            and first.selected_variant == second.selected_variant,
            "forward/reverse case execution semantics differ",
        )
        require(
            math.isclose(
                first.checksum,
                second.checksum,
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            ),
            "forward/reverse correctness checksums differ",
        )
        if first.planner_regret is None:
            require(
                second.planner_regret is None,
                "forward/reverse planner-regret semantics differ",
            )
            regret = None
        else:
            require(
                second.planner_regret is not None,
                "forward/reverse planner-regret semantics differ",
            )
            regret = (first.planner_regret + second.planner_regret) / 2.0
        paired[key] = dataclasses.replace(
            first,
            seconds=(first.seconds + second.seconds) / 2.0,
            seconds_low=min(first.seconds, second.seconds),
            seconds_high=max(first.seconds, second.seconds),
            preparation_seconds=(
                first.preparation_seconds + second.preparation_seconds
            )
            / 2.0,
            amortized_seconds=(
                first.amortized_seconds + second.amortized_seconds
            )
            / 2.0,
            planner_regret=regret,
        )
    return paired


def median(values: Iterable[float]) -> float:
    sequence = list(values)
    require(bool(sequence), "cannot aggregate an empty measurement set")
    return float(statistics.median(sequence))


def percentile_nearest_rank(
    values: Iterable[float], percentile: float
) -> float:
    sequence = sorted(float(value) for value in values)
    require(bool(sequence), "cannot aggregate an empty measurement set")
    return sequence[max(0, math.ceil(len(sequence) * percentile) - 1)]


def build_comparisons(
    cells: dict[tuple, EvidenceCell]
) -> tuple[list[Comparison], list[dict[str, object]]]:
    complete = [cell for cell in cells.values() if cell.mode == "complete-hot"]
    grouped: dict[
        tuple[str, str, tuple[int, int, int], int], list[EvidenceCell]
    ] = defaultdict(list)
    for cell in complete:
        grouped[
            (cell.partition, cell.family, cell.shape, cell.requested_threads)
        ].append(cell)
    comparisons = []
    missing = []
    for (partition, family, shape, threads), group in sorted(grouped.items()):
        providers = [
            cell
            for cell in group
            if cell.variant == OPENBLAS
            and cell.actual_threads == threads
        ]
        eligible_variants = (
            SERIAL_NATIVE_VARIANTS if threads == 1 else PARALLEL_NATIVE_VARIANTS
        )
        natives = [
            cell
            for cell in group
            if cell.variant in eligible_variants
            and cell.actual_threads == threads
        ]
        if len(providers) != 1 or not natives:
            missing.append(
                {
                    "partition": partition,
                    "family": family,
                    "shape": list(shape),
                    "threads": threads,
                    "provider_cells": len(providers),
                    "native_cells": len(natives),
                }
            )
            continue
        provider = providers[0]
        comparable = [
            cell for cell in natives if cell.placement == provider.placement
        ]
        if not comparable:
            missing.append(
                {
                    "partition": partition,
                    "family": family,
                    "shape": list(shape),
                    "threads": threads,
                    "provider_cells": 1,
                    "native_cells": len(natives),
                    "reason": "thread-or-placement-mismatch",
                }
            )
            continue
        native = min(comparable, key=lambda cell: cell.seconds)
        comparisons.append(
            Comparison(partition, family, shape, threads, native, provider)
        )
    return comparisons, missing


def expected_comparison_keys(
    runner: object, physical_cores: int
) -> set[tuple[str, str, tuple[int, int, int], int]]:
    keys = set()
    for spec in runner.PARITY_SHAPES:
        keys.add((spec.partition, spec.family, spec.shape, 1))
        for count in exact_parallel_thread_strata(spec.shape, physical_cores):
            keys.add((spec.partition, spec.family, spec.shape, count))
    return keys


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "median": median(values),
        "p95": percentile_nearest_rank(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def assessment(
    forward: Bundle,
    reverse: Bundle,
    cells: dict[tuple, EvidenceCell],
) -> dict:
    comparisons, missing = build_comparisons(cells)
    runner, _ = load_runner()
    expected_keys = expected_comparison_keys(
        runner, forward.manifest["physical_cores"]
    )
    actual_keys = {
        (
            comparison.partition,
            comparison.family,
            comparison.shape,
            comparison.threads,
        )
        for comparison in comparisons
    }
    missing_keys = {
        (
            row["partition"],
            row["family"],
            tuple(row["shape"]),
            row["threads"],
        )
        for row in missing
    }
    for missing_key in sorted(expected_keys - actual_keys):
        partition, family, shape, threads = missing_key
        if missing_key not in missing_keys:
            missing.append(
                {
                    "partition": partition,
                    "family": family,
                    "shape": list(shape),
                    "threads": threads,
                }
            )
            missing_keys.add(missing_key)

    ratios_by_family_thread: dict[tuple[str, int], list[float]] = defaultdict(
        list
    )
    for comparison in comparisons:
        ratios_by_family_thread[
            (comparison.family, comparison.threads)
        ].append(comparison.ratio)
    family_thread_metrics = [
        {
            "family": family,
            "threads": threads,
            **metric_summary(values),
        }
        for (family, threads), values in sorted(ratios_by_family_thread.items())
    ]

    square_single = [
        comparison.ratio
        for comparison in comparisons
        if comparison.threads == 1
        and comparison.family in SQUARE_ENVELOPE
    ]
    core_family_single = {
        family: [
            comparison.ratio
            for comparison in comparisons
            if comparison.threads == 1 and comparison.family == family
        ]
        for family in sorted(CORE_FAMILIES)
    }
    large_multi = [
        comparison.ratio
        for comparison in comparisons
        if comparison.threads > 1 and comparison.family == "large-square"
    ]

    serial = {
        (cell.shape, cell.variant): cell
        for cell in cells.values()
        if cell.mode == "complete-hot"
        and cell.requested_threads == 1
        and cell.actual_threads == 1
        and cell.variant in SERIAL_NATIVE_VARIANTS
    }
    parallel_map = {
        (cell.shape, cell.variant, cell.actual_threads): cell
        for cell in cells.values()
        if cell.mode == "complete-hot"
        and cell.variant in PARALLEL_NATIVE_VARIANTS
        and cell.actual_threads > 1
    }
    serial_for_parallel = {
        "cpu.native-parallel.avx2-fma.f32.v1":
        "cpu.native-packed.avx2-fma.f32.v1",
        "cpu.native-parallel.avx512-fma.f32.v1":
        "cpu.native-packed.avx512-fma.f32.v1",
    }
    scaling = []
    for (shape, parallel_variant, threads), parallel_cell in sorted(
        parallel_map.items()
    ):
        serial_cell = serial.get(
            (shape, serial_for_parallel[parallel_variant])
        )
        if serial_cell is None:
            continue
        speedup = serial_cell.seconds / parallel_cell.seconds
        scaling.append(
            {
                "shape": list(shape),
                "family": parallel_cell.family,
                "variant": parallel_variant,
                "threads": threads,
                "speedup": speedup,
                "efficiency": speedup / threads,
            }
        )
    large_four_speedups = [
        item["speedup"]
        for item in scaling
        if item["family"] == "large-square" and item["threads"] == 4
    ]

    regrets = [
        float(cell.planner_regret)
        for cell in cells.values()
        if cell.mode == "planner-regret-hot"
        and cell.planner_regret is not None
    ]
    regret_metrics = metric_summary(regrets)
    regret_shapes = sorted(
        {
            cell.shape
            for cell in cells.values()
            if cell.mode == "planner-regret-hot"
            and cell.planner_regret is not None
        }
    )
    declared_parity_shapes = sorted(
        {comparison.shape for comparison in comparisons}
    )
    regret_shape_coverage = {
        "measured_shapes": len(regret_shapes),
        "declared_shapes": len(declared_parity_shapes),
        "missing_shapes": [
            list(shape)
            for shape in declared_parity_shapes
            if shape not in set(regret_shapes)
        ],
        "scope": "bounded-diagnostic-not-full-envelope",
    }
    full_regret_coverage = regret_shapes == declared_parity_shapes
    ceiling_reductions = []
    exact_ceiling_cells = 0
    expected_ceiling_cells = 0
    for row in forward.manifest["parallel_thread_plan"]:
        for ceiling, capacity in zip(
            row["requested_ceilings"],
            row["task_capacities"],
            strict=True,
        ):
            expected_ceiling_cells += 1
            if capacity == ceiling:
                exact_ceiling_cells += 1
                continue
            ceiling_reductions.append(
                {
                    "partition": row["partition"],
                    "family": row["family"],
                    "shape": row["shape"],
                    "requested_ceiling": ceiling,
                    "task_capacity": capacity,
                }
            )
    declared_ceiling_coverage = {
        "expected_cells": expected_ceiling_cells,
        "exact_cells": exact_ceiling_cells,
        "reduced_or_omitted_cells": ceiling_reductions,
        "scope": "declared-4-and-physical-core-ceilings",
    }
    square_metrics = metric_summary(square_single)
    multi_metrics = metric_summary(large_multi)
    four_thread_metrics = metric_summary(large_four_speedups)
    core_family_medians = {
        family: (median(values) if values else None)
        for family, values in core_family_single.items()
    }
    meaningful_medians = [
        value
        for family, value in core_family_medians.items()
        if family in MEANINGFUL_PARITY_FAMILIES and value is not None
    ]

    criteria = [
        {
            "id": "single-square-ratio",
            "threshold": "median >= 0.90",
            "measured": square_metrics["median"],
            "passed": square_metrics["median"] is not None
            and float(square_metrics["median"]) >= 0.90,
        },
        {
            "id": "core-family-floor",
            "threshold": "every core-family single-thread median >= 0.75",
            "measured": core_family_medians,
            "passed": all(
                value is not None and value >= 0.75
                for value in core_family_medians.values()
            ),
        },
        {
            "id": "meaningful-family-parity",
            "threshold": "at least one meaningful family median >= 1.00",
            "measured": max(meaningful_medians)
            if meaningful_medians
            else None,
            "passed": bool(meaningful_medians)
            and max(meaningful_medians) >= 1.00,
        },
        {
            "id": "multi-large-ratio",
            "threshold": "multi-thread large-square median >= 0.85",
            "measured": multi_metrics["median"],
            "passed": multi_metrics["median"] is not None
            and float(multi_metrics["median"]) >= 0.85,
        },
        {
            "id": "four-thread-speedup",
            "threshold": "large-square four-thread median speedup >= 3.00",
            "measured": four_thread_metrics["median"],
            "passed": four_thread_metrics["median"] is not None
            and float(four_thread_metrics["median"]) >= 3.00,
        },
        {
            "id": "planner-regret-bounded-diagnostic",
            "threshold": "median <= 1.05, p95 <= 1.15, maximum <= 1.35",
            "measured": regret_metrics,
            "passed": regret_metrics["median"] is not None
            and float(regret_metrics["median"]) <= 1.05
            and float(regret_metrics["p95"]) <= 1.15
            and float(regret_metrics["maximum"]) <= 1.35,
        },
        {
            "id": "planner-regret-full-envelope-coverage",
            "threshold": (
                "every declared parity shape has complete-registry regret"
            ),
            "measured": regret_shape_coverage,
            "passed": full_regret_coverage,
            "acceptance": False,
        },
        {
            "id": "no-catastrophic-regret-full-envelope",
            "threshold": "maximum <= 2.00 across the declared parity envelope",
            "measured": (
                regret_metrics["maximum"]
                if full_regret_coverage
                else "not established outside bounded diagnostic shapes"
            ),
            "passed": full_regret_coverage
            and regret_metrics["maximum"] is not None
            and float(regret_metrics["maximum"]) <= 2.00,
            "acceptance": False,
        },
        {
            "id": "exact-capacity-comparison-coverage",
            "threshold": "all planned exact task-capacity cells comparable",
            "measured": {
                "expected": len(expected_keys),
                "accepted": len(actual_keys),
                "missing": len(expected_keys - actual_keys),
            },
            "passed": actual_keys == expected_keys,
        },
        {
            "id": "declared-thread-ceiling-coverage",
            "threshold": (
                "every shape compares exact 4-thread and physical-core teams"
            ),
            "measured": {
                "expected": expected_ceiling_cells,
                "exact": exact_ceiling_cells,
                "reduced_or_omitted": len(ceiling_reductions),
            },
            "passed": not ceiling_reductions,
            "acceptance": False,
        },
    ]
    performance_pass = all(
        item["passed"]
        for item in criteria
        if item.get("acceptance", True)
    )
    verdict = (
        "passed"
        if comparisons and square_single and regrets and performance_pass
        else "failed"
    )

    prepacked = []
    for cell in cells.values():
        if cell.mode != "prepacked-b-hot":
            continue
        prepacked.append(
            {
                "partition": cell.partition,
                "family": cell.family,
                "shape": list(cell.shape),
                "variant": cell.variant,
                "lhs_sequence": cell.lhs_sequence,
                "steady_state_gflops": cell.gflops,
                "preparation_seconds": cell.preparation_seconds,
                "amortized_seconds": cell.amortized_seconds,
            }
        )

    comparison_rows = [
        {
            "partition": comparison.partition,
            "family": comparison.family,
            "shape": list(comparison.shape),
            "threads": comparison.threads,
            "native_variant": comparison.native.variant,
            "native_gflops": comparison.native.gflops,
            "openblas_gflops": comparison.provider.gflops,
            "native_openblas_ratio": comparison.ratio,
            "native_seconds": comparison.native.seconds,
            "openblas_seconds": comparison.provider.seconds,
        }
        for comparison in sorted(
            comparisons,
            key=lambda value: (
                value.partition,
                value.family,
                value.shape,
                value.threads,
            ),
        )
    ]
    weakest_comparisons = sorted(
        comparison_rows,
        key=lambda row: (
            row["native_openblas_ratio"],
            row["family"],
            row["shape"],
            row["threads"],
        ),
    )[:5]
    environment = forward.environment
    return {
        "schema": SUMMARY_SCHEMA,
        "version": SUMMARY_VERSION,
        "verdict": verdict,
        "verdict_scope": "bounded-paired-measurement-assessment",
        "milestone_disposition": (
            "requires-manual-full-envelope-and-thread-ceiling-review"
        ),
        "partition_interpretation": PARTITION_INTERPRETATION,
        "evidence_contract": {
            "manifest_version": MANIFEST_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "summary_version": SUMMARY_VERSION,
        },
        "source_commit": forward.manifest["source_commit"],
        "benchmark_binary_sha256": forward.manifest[
            "benchmark_binary_sha256"
        ],
        "runner_sha256": forward.manifest["runner_sha256"],
        "runner_git_blob": forward.manifest["runner_git_blob"],
        "summarizer_sha256": forward.summarizer_sha256,
        "summarizer_git_blob": forward.summarizer_git_blob,
        "forward_manifest_sha256": forward.digest,
        "reverse_manifest_sha256": reverse.digest,
        "physical_cores": forward.manifest["physical_cores"],
        "parallel_thread_plan": forward.manifest["parallel_thread_plan"],
        "started_unix_seconds": {
            "forward": forward.manifest["started_unix_seconds"],
            "reverse": reverse.manifest["started_unix_seconds"],
        },
        "finished_unix_seconds": {
            "forward": forward.manifest["finished_unix_seconds"],
            "reverse": reverse.manifest["finished_unix_seconds"],
        },
        "environment": {
            "cpu_model": environment.get("cpu_model", "unknown"),
            "os_family": environment.get("os_family", "unknown"),
            "architecture": environment.get("architecture", "unknown"),
            "compiler": environment.get("compiler", "unknown"),
            "compiler_flags": environment.get("compiler_flags", "unknown"),
            "build_type": environment.get("build_type", "unknown"),
            "governor": environment.get("governor", "unknown"),
            "frequency_policy": environment.get(
                "frequency_policy", "unknown"
            ),
            "boost_state": environment.get("boost_state", "unknown"),
            "provider_name": environment.get("provider_name", "unknown"),
            "provider_version": environment.get("provider_version", "unknown"),
            "provider_config": environment.get("provider_config", "unknown"),
            "timer_source": environment.get("timer_source", "unknown"),
            "timer_resolution_ns": environment.get(
                "timer_resolution_ns", "unknown"
            ),
        },
        "coverage": {
            "planned_cases": len(forward.manifest["cases"]),
            "paired_accepted_cells": len(cells),
            "forward_states": dict(sorted(forward.states.items())),
            "reverse_states": dict(sorted(reverse.states.items())),
            "rejections": dict(
                sorted(forward.rejection_categories.items())
            ),
            "missing_comparisons": sorted(
                missing,
                key=lambda value: (
                    value["partition"],
                    value["family"],
                    value["shape"],
                    value["threads"],
                ),
            ),
        },
        "criteria": criteria,
        "single_square_ratio": square_metrics,
        "core_family_single_thread_medians": core_family_medians,
        "multi_large_ratio": multi_metrics,
        "four_thread_large_speedup": four_thread_metrics,
        "planner_regret": regret_metrics,
        "planner_regret_coverage": regret_shape_coverage,
        "declared_thread_ceiling_coverage": declared_ceiling_coverage,
        "milestone6_single_thread_family_medians":
            MILESTONE6_SINGLE_THREAD_FAMILY_MEDIANS,
        "family_thread_ratios": family_thread_metrics,
        "comparisons": comparison_rows,
        "weakest_comparisons": weakest_comparisons,
        "scaling": scaling,
        "prepacked_b": prepacked,
    }


def safe_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_text(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    if isinstance(value, dict):
        return ", ".join(
            f"{key}={metric_text(item)}" for key, item in sorted(value.items())
        )
    return safe_text(value)


def render_markdown(summary: dict) -> str:
    environment = summary["environment"]
    evidence_contract = summary["evidence_contract"]
    lines = [
        "# Native BLAS parity v1",
        "",
        "Status: authenticated paired Milestone 7 evidence.",
        "",
        f"**Bounded assessment verdict: {summary['verdict']}**",
        "",
        "Native parity counts only native MDSLC kernels. Automatic plans that "
        "select OpenBLAS do not establish native parity.",
        "",
        "The case-key partition named `holdout` is retained for evidence "
        "compatibility, but it is **validation-not-blind**. Candidate "
        "experiments touched members of that partition before the methodology "
        "freeze. No unbiased holdout claim is made, and the complete declared "
        "matrix remains reported.",
        "",
        "## Evidence identity",
        "",
        (
            "- Contract versions: "
            f"manifest v{evidence_contract['manifest_version']}; "
            f"benchmark v{evidence_contract['benchmark_version']}; "
            f"sanitized summary v{evidence_contract['summary_version']}."
        ),
        f"- Source commit: `{summary['source_commit']}`",
        (
            "- Benchmark binary SHA-256: "
            f"`{summary['benchmark_binary_sha256']}`"
        ),
        f"- Runner SHA-256: `{summary['runner_sha256']}`",
        f"- Runner Git blob: `{summary['runner_git_blob']}`",
        f"- Summarizer SHA-256: `{summary['summarizer_sha256']}`",
        f"- Summarizer Git blob: `{summary['summarizer_git_blob']}`",
        (
            "- Forward manifest SHA-256: "
            f"`{summary['forward_manifest_sha256']}`"
        ),
        (
            "- Reverse manifest SHA-256: "
            f"`{summary['reverse_manifest_sha256']}`"
        ),
        (
            f"- Host: {safe_text(environment['cpu_model'])}; "
            f"{safe_text(environment['os_family'])} "
            f"{safe_text(environment['architecture'])}"
        ),
        (
            f"- Compiler/build: {safe_text(environment['compiler'])}; "
            f"{safe_text(environment['build_type'])}"
        ),
        f"- Compiler flags: `{safe_text(environment['compiler_flags'])}`",
        (
            "- Frequency state: "
            f"governor={safe_text(environment['governor'])}; "
            f"policy={safe_text(environment['frequency_policy'])}; "
            f"boost={safe_text(environment['boost_state'])}"
        ),
        (
            f"- OpenBLAS provider: {safe_text(environment['provider_name'])} "
            f"{safe_text(environment['provider_version'])}; "
            f"`{safe_text(environment['provider_config'])}`"
        ),
        (
            f"- Timer: {safe_text(environment['timer_source'])}; "
            f"resolution={safe_text(environment['timer_resolution_ns'])} ns"
        ),
        (
            "- Collection Unix seconds: "
            f"forward={summary['started_unix_seconds']['forward']}.."
            f"{summary['finished_unix_seconds']['forward']}; "
            f"reverse={summary['started_unix_seconds']['reverse']}.."
            f"{summary['finished_unix_seconds']['reverse']}."
        ),
        f"- Discovered physical cores: {summary['physical_cores']}.",
        "",
        "Raw reports remain outside Git. Forward/reverse cells are paired by "
        "partition, family, shape, requested thread count, mode, variant, and "
        "repeated-input sequence before aggregation.",
        "",
        "## Measurement contract",
        "",
        "- Complete-call parity uses hot cache and caller-owned reused "
        "workspace, excludes allocation, and includes transient packing.",
        "- Single-thread comparisons use authenticated compact placement. "
        "Multi-thread comparisons use the same unbound placement and exact "
        "actual thread count for native and OpenBLAS.",
        "- The requested 4-thread and physical-core ceilings are reduced only "
        "to the deterministic output-task capacity recorded in the manifest; "
        "capacity-limited cells are not relabeled as full-ceiling results.",
        "- Compute-only measurements are diagnostic and never compared with a "
        "complete CBLAS call.",
        "- Every retained sample passed the seeded independent correctness "
        "oracle and final-output authentication.",
        "",
            "## Bounded paired-measurement assessment",
            "",
            "A `passed` summary means the acceptance-enabled rows below pass. "
            "Diagnostic rows remain separate manual Milestone 7 gates; this "
            "summary alone cannot complete the milestone.",
            "",
        "| Criterion | Threshold | Measured | Result |",
        "|---|---|---|---|",
    ]
    for criterion in summary["criteria"]:
        if not criterion.get("acceptance", True):
            result = (
                "DIAGNOSTIC PASS"
                if criterion["passed"]
                else "DIAGNOSTIC NOT MET"
            )
        else:
            result = "PASS" if criterion["passed"] else "NOT MET"
        lines.append(
            f"| `{criterion['id']}` | {criterion['threshold']} | "
            f"{metric_text(criterion['measured'])} | "
            f"{result} |"
        )

    lines.extend(
        [
            "",
            "## Native/OpenBLAS ratios by family and exact thread count",
            "",
            "| Family | Threads | Cells | Median | P95 | Minimum | Maximum |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["family_thread_ratios"]:
        lines.append(
            f"| {row['family']} | {row['threads']} | {row['count']} | "
            f"{metric_text(row['median'])} | {metric_text(row['p95'])} | "
            f"{metric_text(row['minimum'])} | {metric_text(row['maximum'])} |"
        )
    if not summary["family_thread_ratios"]:
        lines.append("| none | — | 0 | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Complete paired shape matrix",
            "",
            "| Partition | Family | M×N×K | Threads | Fastest native | Native GFLOP/s | OpenBLAS GFLOP/s | Ratio |",
            "|---|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['partition']} | {row['family']} | "
            f"{'×'.join(map(str, row['shape']))} | {row['threads']} | "
            f"`{row['native_variant']}` | {row['native_gflops']:.3f} | "
            f"{row['openblas_gflops']:.3f} | "
            f"{row['native_openblas_ratio']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Native scaling",
            "",
            "| Family | M×N×K | Variant | Threads | Speedup | Efficiency |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["scaling"]:
        lines.append(
            f"| {row['family']} | {'×'.join(map(str, row['shape']))} | "
            f"`{row['variant']}` | {row['threads']} | "
            f"{row['speedup']:.3f} | {row['efficiency']:.3f} |"
        )
    if not summary["scaling"]:
        lines.append("| none | — | — | — | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Planner regret",
            "",
            "| Cells | Median | P95 | Minimum | Maximum |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {summary['planner_regret']['count']} | "
                f"{metric_text(summary['planner_regret']['median'])} | "
                f"{metric_text(summary['planner_regret']['p95'])} | "
                f"{metric_text(summary['planner_regret']['minimum'])} | "
                f"{metric_text(summary['planner_regret']['maximum'])} |"
            ),
            "",
            (
                "Coverage: "
                f"{summary['planner_regret_coverage']['measured_shapes']} of "
                f"{summary['planner_regret_coverage']['declared_shapes']} "
                "declared shapes. This is bounded diagnostic evidence, not "
                "full-envelope planner-regret acceptance."
            ),
            "",
            "Automatic plans may select OpenBLAS. Such selections count for "
            "planner regret but never for native parity.",
            "",
            "## Prepacked-B repeated execution",
            "",
            "| Partition | Family | M×N×K | Variant | Inputs | Preparation s | "
            "Steady GFLOP/s | Amortized s |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        summary["prepacked_b"],
        key=lambda item: (
            item["partition"],
            item["family"],
            item["shape"],
            item["variant"],
            item["lhs_sequence"],
        ),
    ):
        lines.append(
            f"| {row['partition']} | {row['family']} | "
            f"{'×'.join(map(str, row['shape']))} | "
            f"`{row['variant']}` | {row['lhs_sequence']} | "
            f"{metric_text(row['preparation_seconds'])} | "
            f"{metric_text(row['steady_state_gflops'])} | "
            f"{metric_text(row['amortized_seconds'])} |"
        )
    if not summary["prepacked_b"]:
        lines.append("| none | — | — | — | — | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Milestone 6 comparison",
            "",
            "The historical values below are the sanitized Milestone 6 "
            "single-thread family medians. They were collected in a separate "
            "run and are not paired frequency controls for the current run.",
            "",
            "| Family | Milestone 6 ratio | Milestone 7 ratio | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    current_single = {
        row["family"]: row["median"]
        for row in summary["family_thread_ratios"]
        if row["threads"] == 1
    }
    for family, baseline in sorted(
        summary["milestone6_single_thread_family_medians"].items()
    ):
        current = current_single.get(family)
        delta = None if current is None else float(current) - baseline
        lines.append(
            f"| {family} | {baseline:.3f} | {metric_text(current)} | "
            f"{metric_text(delta)} |"
        )

    lines.extend(
        [
            "",
            "## Weakest measured cells",
            "",
            "| Family | M×N×K | Threads | Native variant | Ratio |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in summary["weakest_comparisons"]:
        lines.append(
            f"| {row['family']} | {'×'.join(map(str, row['shape']))} | "
            f"{row['threads']} | `{row['native_variant']}` | "
            f"{row['native_openblas_ratio']:.3f} |"
        )

    missing = summary["coverage"]["missing_comparisons"]
    lines.extend(
        [
            "",
            "## Coverage and limitations",
            "",
            (
                f"- Planned cases in each order: "
                f"{summary['coverage']['planned_cases']}."
            ),
            (
                f"- Paired accepted raw cells: "
                f"{summary['coverage']['paired_accepted_cells']}."
            ),
            (
                f"- Missing complete comparison cells: {len(missing)}."
            ),
            (
                "- Forward states: "
                f"{metric_text(summary['coverage']['forward_states'])}."
            ),
            (
                "- Reverse states: "
                f"{metric_text(summary['coverage']['reverse_states'])}."
            ),
            (
                "- Expected rejection categories: "
                f"{metric_text(summary['coverage']['rejections'])}."
            ),
            (
                "- Declared 4-thread/physical-core ceiling cells: "
                f"{summary['declared_thread_ceiling_coverage']['exact_cells']} "
                "exact of "
                f"{summary['declared_thread_ceiling_coverage']['expected_cells']}; "
                f"{len(summary['declared_thread_ceiling_coverage']['reduced_or_omitted_cells'])} "
                "were reduced or omitted by native task capacity."
            ),
            (
                "- Provider thread count is the configured OpenBLAS team size; "
                "active provider concurrency is not sampled."
            ),
            (
                "- Scaling compares compact one-thread native execution with "
                "the frozen unbound multi-thread parity stratum; it is not an "
                "equal-affinity claim."
            ),
            (
                "- Results are host-bounded and do not establish universal BLAS "
                "parity, GPU behavior, or physical multi-node NUMA performance."
            ),
            "",
            "## Claims supported",
            "",
            "- The reported native/OpenBLAS ratios, absolute GFLOP/s, scaling, "
            "prepacked-B measurements, and planner regret are authenticated "
            "and traceable to the exact host, source, binary, and provider.",
            "- Native parity metrics include only native packed or persistent-"
            "parallel MDSLC variants; automatic OpenBLAS selections are not "
            "misclassified as native parity.",
            "- Exact requested/actual thread equality, placement class, "
            "packing visibility, and output correctness are authenticated.",
            "",
            "## Claims explicitly unsupported",
            "",
            "- No universal BLAS-parity, cross-host, cross-provider, GPU, or "
            "physical multi-node NUMA conclusion is made.",
            "- The `holdout` identifier is not blind experimental evidence.",
            "- Provider worker affinity and active concurrency are not sampled; "
            "multi-thread comparisons use the declared unbound stratum.",
            "- These measurements do not freeze the public API, ABI, planner "
            "cost model, private blocking profile, or microkernel symbols.",
            "",
        ]
    )
    if missing:
        lines.extend(
            [
                "### Missing comparable cells",
                "",
                "| Partition | Family | M×N×K | Threads | Reason |",
                "|---|---|---|---:|---|",
            ]
        )
        for row in missing:
            lines.append(
                f"| {row['partition']} | {row['family']} | "
                f"{'×'.join(map(str, row['shape']))} | {row['threads']} | "
                f"{safe_text(row.get('reason', 'candidate unavailable'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def atomic_write(path: pathlib.Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def invalidate_output(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().absolute()
    require(
        not path.exists() or path.is_file() or path.is_symlink(),
        f"summary output path is not a file: {path}",
    )
    if path.exists() or path.is_symlink():
        path.unlink()
    return path


def normalized_path(path: pathlib.Path) -> pathlib.Path:
    return path.expanduser().absolute().resolve(strict=False)


def paths_alias(first: pathlib.Path, second: pathlib.Path) -> bool:
    if normalized_path(first) == normalized_path(second):
        return True
    if first.exists() and second.exists():
        try:
            return os.path.samefile(first, second)
        except OSError:
            return False
    return False


def protected_evidence_paths(manifest_path: pathlib.Path) -> set[pathlib.Path]:
    manifest_path = manifest_path.expanduser().absolute()
    protected = {manifest_path}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return protected
    if not isinstance(manifest, dict):
        return protected
    for field in ("benchmark", "runner"):
        value = manifest.get(field)
        if isinstance(value, str) and value:
            protected.add(pathlib.Path(value).expanduser().absolute())
    records = manifest.get("cases")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            raw_file = record.get("raw_file")
            if isinstance(raw_file, str) and raw_file:
                protected.add(
                    (manifest_path.parent / raw_file).absolute()
                )
    return protected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-manifest", required=True)
    parser.add_argument("--reverse-manifest", required=True)
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--json-out")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help=(
            "return one when authenticated evidence misses any "
            "acceptance-enabled bounded-summary target"
        ),
    )
    args = parser.parse_args()
    try:
        forward_manifest = pathlib.Path(
            args.forward_manifest
        ).expanduser().absolute()
        reverse_manifest = pathlib.Path(
            args.reverse_manifest
        ).expanduser().absolute()
        markdown_out = pathlib.Path(args.markdown_out).expanduser().absolute()
        json_out = (
            pathlib.Path(args.json_out).expanduser().absolute()
            if args.json_out
            else None
        )
        require(
            json_out is None or not paths_alias(json_out, markdown_out),
            "Markdown and JSON outputs must be distinct paths",
        )
        protected = (
            protected_evidence_paths(forward_manifest)
            | protected_evidence_paths(reverse_manifest)
            | {
                pathlib.Path(__file__).resolve(),
                pathlib.Path(__file__).resolve().with_name(
                    "run_native_blas_parity.py"
                ),
            }
        )
        for output in (markdown_out, json_out):
            if output is None:
                continue
            require(
                not any(
                    paths_alias(output, evidence)
                    for evidence in protected
                ),
                "summary output path collides with parity evidence input",
            )
        markdown_out = invalidate_output(markdown_out)
        if json_out is not None:
            json_out = invalidate_output(json_out)
        forward = load_bundle(
            forward_manifest, "stable-forward"
        )
        reverse = load_bundle(
            reverse_manifest, "stable-reverse"
        )
        cells = pair_bundles(forward, reverse)
        summary = assessment(forward, reverse, cells)
        atomic_write(
            markdown_out, render_markdown(summary)
        )
        if json_out is not None:
            atomic_write(
                json_out,
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
            )
    except ParityError as error:
        print(
            f"matcore native-BLAS parity summary rejected: {error}",
            file=sys.stderr,
        )
        return 2
    except (KeyError, TypeError, ValueError) as error:
        print(
            "matcore native-BLAS parity summary rejected: malformed numeric "
            f"or structural evidence: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        f"matcore native-BLAS parity summary: {len(cells)} paired raw cells; "
        f"bounded-verdict={summary['verdict']}; "
        f"markdown={args.markdown_out}"
    )
    if args.require_pass and summary["verdict"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
