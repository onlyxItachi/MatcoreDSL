#!/usr/bin/env python3

"""Authenticate and sanitize Matcore CPU deep-audit evidence.

The input manifests and their raw benchmark JSON files remain outside Git.
This program fails closed on incomplete or internally inconsistent evidence and
writes only aggregate, path-free Markdown suitable for review.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Iterable


MANIFEST_SCHEMA = "matcore.cpu-performance-deep-audit.manifest"
MANIFEST_VERSION = 2
BENCHMARK_SCHEMA = "matcore.benchmark.cpu.gemm"
BENCHMARK_VERSION = 6
BENCHMARK_SEED = 0x4D4154434F524531
REQUIRED_SUITES = {
    "complete",
    "compute",
    "cold",
    "prepacked",
    "oneshot",
    "regret",
}
EXTERNAL_VARIANT = "cpu.external.openblas.f32.v1"


class AuditError(RuntimeError):
    """Evidence is incomplete, unauthenticated, or not comparable."""


@dataclasses.dataclass(frozen=True)
class EvidenceCell:
    semantic_key: tuple
    family: str
    partition: str
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
    comparable_planner_regret: float | None
    planner_exclusions: int
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
    states: Counter
    rejection_categories: Counter
    environment: dict


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def require_sha(value: object, field: str) -> str:
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
        raise AuditError(f"cannot read {description}: {error}") from error
    require(isinstance(value, dict), f"{description} is not a JSON object")
    return value


def median(values: Iterable[float]) -> float:
    sequence = list(values)
    require(bool(sequence), "cannot aggregate an empty measurement set")
    return float(statistics.median(sequence))


def percentile_nearest_rank(values: Iterable[float], percentile: float) -> float:
    sequence = sorted(float(value) for value in values)
    require(bool(sequence), "cannot aggregate an empty measurement set")
    index = max(0, math.ceil(len(sequence) * percentile) - 1)
    return sequence[index]


def case_semantic_key(record: dict) -> tuple:
    shape = record.get("shape")
    require(
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(value, int) and value > 0 for value in shape),
        "manifest case has an invalid shape",
    )
    fields = ("family", "partition", "variant", "mode")
    require(
        all(isinstance(record.get(field), str) and record[field] for field in fields),
        "manifest case has missing semantic fields",
    )
    requested_threads = record.get("threads")
    lhs_sequence = record.get("lhs_sequence")
    require(
        isinstance(requested_threads, int) and requested_threads > 0,
        "manifest case has an invalid requested thread count",
    )
    require(
        isinstance(lhs_sequence, int) and lhs_sequence > 0,
        "manifest case has an invalid lhs sequence length",
    )
    return (
        record["family"],
        record["partition"],
        tuple(shape),
        record["variant"],
        requested_threads,
        record["mode"],
        lhs_sequence,
    )


def expected_case_key(record: dict) -> str:
    m, n, k = record["shape"]
    encoded_variant = record["variant"].replace(".", "_").replace("-", "_")
    return (
        f"{record['family']}__{m}x{n}x{k}__{encoded_variant}"
        f"__t{record['threads']}__{record['mode']}__lhs{record['lhs_sequence']}"
    )


def reconstructed_plan_digest(manifest: dict) -> str:
    cases = []
    for record in manifest["cases"]:
        case_semantic_key(record)
        cases.append(
            {
                "family": record["family"],
                "partition": record["partition"],
                "shape": record["shape"],
                "variant": record["variant"],
                "threads": record["threads"],
                "mode": record["mode"],
                "lhs_sequence": record["lhs_sequence"],
            }
        )
    skips = []
    for skip in manifest["skips"]:
        require(isinstance(skip, dict), "manifest skip is not an object")
        require(
            isinstance(skip.get("shape"), list)
            and len(skip["shape"]) == 3
            and all(isinstance(value, int) and value > 0 for value in skip["shape"]),
            "manifest skip has an invalid shape",
        )
        for field in ("family", "variant", "mode", "reason"):
            require(
                isinstance(skip.get(field), str) and skip[field],
                f"manifest skip has an invalid {field}",
            )
        require(
            isinstance(skip.get("threads"), int) and skip["threads"] > 0,
            "manifest skip has an invalid thread count",
        )
        skips.append(
            {
                "family": skip["family"],
                "shape": skip["shape"],
                "variant": skip["variant"],
                "threads": skip["threads"],
                "mode": skip["mode"],
                "reason": skip["reason"],
            }
        )
    plan = {
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
    return canonical_sha256(plan)


def expected_mode_configuration(mode: str) -> tuple[str, str, str]:
    if mode == "one-shot-hot":
        return ("hot", "include-allocation", "include-packing")
    if mode == "compute-only-hot":
        return ("hot", "reuse-workspace", "exclude-packing")
    if mode == "prepacked-b-hot":
        return ("hot", "reuse-workspace", "prepacked-b")
    if mode == "complete-cold":
        return ("cold", "reuse-workspace", "include-packing")
    if mode in {"complete-hot", "planner-regret-hot"}:
        return ("hot", "reuse-workspace", "include-packing")
    raise AuditError(f"unsupported audit mode {mode!r}")


def reconstructed_samples(result: dict, iterations: int) -> tuple[float, float, float]:
    samples = result.get("normalized_samples_seconds")
    require(
        isinstance(samples, list)
        and len(samples) == iterations
        and all(
            isinstance(sample, (int, float))
            and math.isfinite(sample)
            and sample > 0.0
            for sample in samples
        ),
        "raw report has missing or invalid normalized timing samples",
    )
    ordered = sorted(float(sample) for sample in samples)
    minimum = ordered[0]
    middle = ordered[len(ordered) // 2]
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    for field, expected in (
        ("minimum_seconds", minimum),
        ("median_seconds", middle),
        ("p95_seconds", p95),
    ):
        require(
            isinstance(result.get(field), (int, float))
            and math.isclose(
                float(result[field]), expected, rel_tol=1.0e-12, abs_tol=1.0e-15
            ),
            f"{field} does not reconstruct from normalized timing samples",
        )
    return minimum, middle, p95


def candidate_placement(candidate: dict) -> tuple:
    return (
        int(candidate["actual_threads"]),
        candidate["smt_policy"],
        candidate["affinity_policy"],
        bool(candidate["worker_affinity_applied"]),
        bool(candidate["worker_affinity_user_requested"]),
        bool(candidate["worker_affinity_policy_induced"]),
    )


def authenticate_planner_regret(
    result: dict, iterations: int
) -> tuple[float | None, float | None, int]:
    planner = result.get("planner_regret")
    if not isinstance(planner, dict) or not planner.get("requested"):
        return None, None, 0
    require(planner.get("valid") is True, "planner-regret evidence is invalid")
    candidates = planner.get("candidates")
    require(isinstance(candidates, list) and candidates, "planner candidates missing")
    reconstructed: list[tuple[dict, float]] = []
    for candidate in candidates:
        require(isinstance(candidate, dict), "planner candidate is not an object")
        if not (
            candidate.get("legal")
            and candidate.get("timing_valid")
            and candidate.get("correctness")
        ):
            continue
        pass_medians = []
        for pass_name in ("forward", "reverse"):
            samples = candidate.get(
                f"{pass_name}_pass_normalized_samples_seconds"
            )
            require(
                isinstance(samples, list)
                and len(samples) == iterations
                and all(
                    isinstance(sample, (int, float))
                    and math.isfinite(sample)
                    and sample > 0.0
                    for sample in samples
                ),
                f"planner {pass_name} samples are invalid",
            )
            pass_median = sorted(float(sample) for sample in samples)[
                len(samples) // 2
            ]
            require(
                math.isclose(
                    float(candidate[f"{pass_name}_pass_median_seconds"]),
                    pass_median,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ),
                f"planner {pass_name} median does not reconstruct",
            )
            pass_medians.append(pass_median)
        balanced = sum(pass_medians) / 2.0
        require(
            math.isclose(
                float(candidate["balanced_estimate_seconds"]),
                balanced,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            "planner candidate balanced estimate does not reconstruct",
        )
        reconstructed.append((candidate, balanced))
    require(reconstructed, "planner has no authenticated legal timing candidate")
    fastest = min(reconstructed, key=lambda item: item[1])
    selected = [
        item for item in reconstructed if item[0]["variant"] == result["selected_variant"]
    ]
    require(
        len(selected) == 1,
        "planner selected variant has no unique authenticated timing candidate",
    )
    raw_regret = selected[0][1] / fastest[1]
    require(
        planner.get("fastest_legal_variant") == fastest[0]["variant"],
        "planner fastest variant does not reconstruct",
    )
    require(
        math.isclose(
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
            raw_regret,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        "planner regret aggregate does not reconstruct",
    )

    selected_placement = candidate_placement(selected[0][0])
    comparable = [
        item
        for item in reconstructed
        if candidate_placement(item[0]) == selected_placement
    ]
    comparable_fastest = min(comparable, key=lambda item: item[1])
    comparable_regret = selected[0][1] / comparable_fastest[1]
    return raw_regret, comparable_regret, len(reconstructed) - len(comparable)


def authenticate_report(
    report: dict, record: dict, manifest: dict
) -> EvidenceCell:
    require(report.get("schema") == BENCHMARK_SCHEMA, "unexpected raw schema")
    require(report.get("version") == BENCHMARK_VERSION, "unexpected raw version")
    require(
        report.get("operation") == "matcore.gemm"
        and report.get("dtype") == "f32"
        and report.get("accumulation_dtype") == "f32"
        and report.get("layout") == "row-major-contiguous",
        "raw operation contract is not the audited F32 row-major GEMM contract",
    )
    environment = report.get("environment")
    configuration = report.get("configuration")
    results = report.get("results")
    require(isinstance(environment, dict), "raw environment is missing")
    require(isinstance(configuration, dict), "raw configuration is missing")
    require(isinstance(results, list) and len(results) == 1, "raw result count is not one")
    require(
        environment.get("source_provenance_state") == "clean"
        and environment.get("source_worktree_dirty") is False,
        "raw report was not produced from a clean source worktree",
    )
    require(
        environment.get("source_commit") == manifest["benchmark_source_commit"],
        "raw report source commit differs from manifest identity",
    )
    require(
        configuration.get("seed") == manifest["benchmark_seed"],
        "raw report seed differs from manifest identity",
    )
    semantic_key = case_semantic_key(record)
    family, partition, shape, variant, requested_threads, mode, lhs_sequence = (
        semantic_key
    )
    expected_cache, expected_allocation, expected_packing = (
        expected_mode_configuration(mode)
    )
    require(
        configuration.get("requested_variant") == variant
        and configuration.get("requested_threads") == requested_threads
        and configuration.get("lhs_sequence_length") == lhs_sequence
        and configuration.get("warmup_iterations") == manifest["warmup"]
        and configuration.get("measured_iterations") == manifest["iterations"]
        and configuration.get("cache_mode") == expected_cache
        and configuration.get("allocation_mode") == expected_allocation
        and configuration.get("packing_mode") == expected_packing,
        "raw configuration differs from manifest case semantics",
    )
    result = results[0]
    require(
        (result.get("m"), result.get("n"), result.get("k")) == shape,
        "raw shape differs from manifest case",
    )
    require(
        result.get("requested_variant") == variant,
        "raw requested variant differs from manifest case",
    )
    require(
        result.get("timing_valid") is True
        and result.get("correctness") is True
        and result.get("timed_final_output_authenticated") is True,
        "raw timing or correctness authentication failed",
    )
    _, seconds, _ = reconstructed_samples(result, manifest["iterations"])
    actual_threads = result.get("actual_threads")
    require(
        isinstance(actual_threads, int)
        and 1 <= actual_threads <= requested_threads
        and record.get("actual_threads") == actual_threads
        and record.get("thread_count_clamped") == (actual_threads != requested_threads),
        "raw actual thread count differs from the manifest",
    )
    operations = 2 * shape[0] * shape[1] * shape[2]
    expected_gflops = operations / seconds / 1.0e9
    require(
        isinstance(result.get("gflops"), (int, float))
        and math.isclose(
            float(result["gflops"]),
            expected_gflops,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "raw GFLOP/s does not reconstruct from shape and median",
    )
    placement = (
        actual_threads,
        configuration.get("smt_policy"),
        configuration.get("affinity_policy"),
        result.get("worker_affinity_applied"),
        result.get("worker_affinity_user_requested"),
        result.get("worker_affinity_policy_induced"),
    )
    require(
        result.get("smt_policy") == configuration.get("smt_policy")
        and result.get("affinity_policy") == configuration.get("affinity_policy"),
        "result placement differs from requested placement",
    )

    preparation_seconds = 0.0
    amortized_seconds = 0.0
    preparation = result.get("prepacked_b_preparation")
    require(isinstance(preparation, dict), "prepacked-B metadata is missing")
    if mode == "prepacked-b-hot":
        require(
            preparation.get("requested") is True
            and preparation.get("measured") is True
            and preparation.get("authenticated") is True
            and preparation.get("preparation_calls") == 1
            and preparation.get("amortization_executions") == lhs_sequence
            and preparation.get("amortized_total_valid") is True,
            "prepacked-B evidence is not authenticated",
        )
        preparation_seconds = float(preparation["preparation_seconds"])
        steady = float(preparation["steady_state_sequence_seconds"])
        total = float(preparation["amortized_total_sequence_seconds"])
        amortized_seconds = float(preparation["amortized_per_execution_seconds"])
        require(
            preparation_seconds > 0.0
            and math.isclose(
                steady, seconds * lhs_sequence, rel_tol=1.0e-12, abs_tol=1.0e-15
            )
            and math.isclose(
                total,
                preparation_seconds + steady,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                amortized_seconds,
                total / lhs_sequence,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            "prepacked-B preparation/amortization arithmetic does not reconstruct",
        )
    else:
        require(
            preparation.get("requested") is False,
            "non-prepacked case unexpectedly requested prepacked-B preparation",
        )

    raw_regret, comparable_regret, planner_exclusions = authenticate_planner_regret(
        result, manifest["iterations"]
    )
    if mode == "planner-regret-hot":
        require(raw_regret is not None, "planner-regret case has no regret evidence")
    else:
        require(raw_regret is None, "non-regret case contains planner-regret evidence")

    return EvidenceCell(
        semantic_key=semantic_key,
        family=family,
        partition=partition,
        shape=shape,
        variant=variant,
        requested_threads=requested_threads,
        actual_threads=actual_threads,
        mode=mode,
        lhs_sequence=lhs_sequence,
        selected_variant=str(result["selected_variant"]),
        placement=placement,
        seconds=seconds,
        seconds_low=seconds,
        seconds_high=seconds,
        preparation_seconds=preparation_seconds,
        amortized_seconds=amortized_seconds,
        planner_regret=raw_regret,
        comparable_planner_regret=comparable_regret,
        planner_exclusions=planner_exclusions,
        report=report,
    )


def load_bundle(path: pathlib.Path, expected_order: str) -> Bundle:
    path = path.resolve()
    manifest = load_json(path, "audit manifest")
    require(manifest.get("schema") == MANIFEST_SCHEMA, "unexpected manifest schema")
    require(manifest.get("version") == MANIFEST_VERSION, "expected manifest v2")
    require(
        manifest.get("benchmark_schema_version") == BENCHMARK_VERSION,
        "expected benchmark schema v6",
    )
    for field in ("benchmark_binary_sha256", "runner_sha256", "plan_sha256"):
        require_sha(manifest.get(field), field)
    require(
        manifest.get("benchmark_seed") == BENCHMARK_SEED,
        "unexpected benchmark seed",
    )
    source_commit = manifest.get("benchmark_source_commit")
    require(
        isinstance(source_commit, str)
        and len(source_commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in source_commit),
        "manifest source commit is missing or invalid",
    )
    require(manifest.get("dry_run") is False, "dry-run manifest has no evidence")
    require(
        manifest.get("case_order") == expected_order,
        f"expected {expected_order} manifest",
    )
    require(
        set(manifest.get("suites", [])) == REQUIRED_SUITES,
        "manifest does not cover every required Milestone 6 suite",
    )
    for field in ("cases", "skips", "variants", "threads", "environment_overrides"):
        require(field in manifest, f"manifest is missing {field}")
    require(
        isinstance(manifest["cases"], list) and manifest["cases"],
        "manifest has no executable cases",
    )
    require(isinstance(manifest["skips"], list), "manifest skips are invalid")
    require(
        {
            record.get("mode")
            for record in manifest["cases"]
            if isinstance(record, dict)
        }
        == {
            "complete-hot",
            "compute-only-hot",
            "complete-cold",
            "prepacked-b-hot",
            "one-shot-hot",
            "planner-regret-hot",
        },
        "manifest executable cases do not cover every required audit mode",
    )
    require(
        isinstance(manifest.get("started_unix_seconds"), int)
        and isinstance(manifest.get("finished_unix_seconds"), int)
        and manifest["finished_unix_seconds"] >= manifest["started_unix_seconds"],
        "manifest run interval is invalid",
    )
    require(
        reconstructed_plan_digest(manifest) == manifest["plan_sha256"],
        "manifest plan digest does not reconstruct; bundle is incomplete or altered",
    )

    states: Counter = Counter()
    rejection_categories: Counter = Counter()
    cells: dict[tuple, EvidenceCell] = {}
    environment_signature: tuple | None = None
    representative_environment: dict | None = None
    seen_record_keys: set[str] = set()
    manifest_directory = path.parent
    for record in manifest["cases"]:
        require(isinstance(record, dict), "manifest case is not an object")
        semantic_key = case_semantic_key(record)
        require(
            isinstance(record.get("key"), str)
            and record["key"] == expected_case_key(record)
            and record["key"] not in seen_record_keys,
            "manifest case key is invalid or duplicated",
        )
        seen_record_keys.add(record["key"])
        state = record.get("state")
        require(
            state in {"passed", "reused", "rejected"},
            f"incomplete bundle contains case state {state!r}",
        )
        states[state] += 1
        if state == "rejected":
            require(
                isinstance(record.get("returncode"), int)
                and record["returncode"] != 0
                and isinstance(record.get("rejection_category"), str)
                and record["rejection_category"],
                "rejected case lacks an authenticated rejection category",
            )
            require("sha256" not in record, "rejected case unexpectedly has raw SHA")
            rejection_categories[record["rejection_category"]] += 1
            continue

        raw_name = record.get("raw_file")
        require(
            isinstance(raw_name, str)
            and raw_name
            and pathlib.PurePath(raw_name).name == raw_name,
            "raw evidence path must be a plain relative filename",
        )
        raw_path = manifest_directory / raw_name
        require(raw_path.is_file(), f"raw evidence is missing for {record['key']}")
        expected_digest = require_sha(record.get("sha256"), "raw sha256")
        require(
            sha256(raw_path) == expected_digest,
            f"raw evidence SHA-256 mismatch for {record['key']}",
        )
        report = load_json(raw_path, f"raw evidence for {record['key']}")
        cell = authenticate_report(report, record, manifest)
        require(semantic_key not in cells, "duplicate semantic audit case")
        cells[semantic_key] = cell
        environment = report["environment"]
        signature_fields = (
            "os_family",
            "architecture",
            "compiler",
            "compiler_flags",
            "build_type",
            "cpu_model",
            "governor",
            "frequency_policy",
            "boost_state",
            "capability_record",
            "topology_record",
            "capability_record_version",
            "topology_record_version",
        )
        signature = tuple(environment.get(field) for field in signature_fields)
        if environment_signature is None:
            environment_signature = signature
            representative_environment = environment
        require(
            signature == environment_signature,
            "raw reports do not share one homogeneous benchmark environment",
        )
    require(cells, "manifest contains no accepted raw evidence")
    require(
        states["passed"] + states["reused"] + states["rejected"]
        == len(manifest["cases"]),
        "manifest executable case accounting is incomplete",
    )
    return Bundle(
        path=path,
        digest=sha256(path),
        manifest=manifest,
        cells=cells,
        states=states,
        rejection_categories=rejection_categories,
        environment=representative_environment or {},
    )


def compare_bundle_identity(forward: Bundle, reverse: Bundle) -> None:
    identity_fields = (
        "benchmark_schema_version",
        "benchmark_binary_sha256",
        "runner_sha256",
        "benchmark_source_commit",
        "benchmark_seed",
        "suites",
        "variants",
        "threads",
        "warmup",
        "iterations",
        "timer_floor_us",
        "max_memory_mib",
        "environment_overrides",
    )
    for field in identity_fields:
        require(
            forward.manifest.get(field) == reverse.manifest.get(field),
            f"forward/reverse identity differs for {field}",
        )
    forward_keys = {case_semantic_key(record) for record in forward.manifest["cases"]}
    reverse_keys = {case_semantic_key(record) for record in reverse.manifest["cases"]}
    require(
        forward_keys == reverse_keys,
        "forward/reverse manifests do not contain the same semantic case set",
    )
    normalize_skips = lambda bundle: {
        (
            skip["family"],
            tuple(skip["shape"]),
            skip["variant"],
            skip["threads"],
            skip["mode"],
            skip["reason"],
        )
        for skip in bundle.manifest["skips"]
    }
    require(
        normalize_skips(forward) == normalize_skips(reverse),
        "forward/reverse manifests do not contain the same predeclared skips",
    )
    require(
        forward.states["passed"] + forward.states["reused"]
        == reverse.states["passed"] + reverse.states["reused"]
        and forward.states["rejected"] == reverse.states["rejected"]
        and forward.rejection_categories == reverse.rejection_categories,
        "forward/reverse accepted/rejected accounting differs",
    )
    forward_status = {
        case_semantic_key(record): (
            "accepted" if record["state"] in {"passed", "reused"} else "rejected",
            record.get("rejection_category"),
        )
        for record in forward.manifest["cases"]
    }
    reverse_status = {
        case_semantic_key(record): (
            "accepted" if record["state"] in {"passed", "reused"} else "rejected",
            record.get("rejection_category"),
        )
        for record in reverse.manifest["cases"]
    }
    require(
        forward_status == reverse_status,
        "forward/reverse per-case execution status differs",
    )
    require(
        set(forward.cells) == set(reverse.cells),
        "forward/reverse accepted raw case sets differ",
    )


def pair_cells(forward: Bundle, reverse: Bundle | None) -> dict[tuple, EvidenceCell]:
    if reverse is None:
        return forward.cells
    compare_bundle_identity(forward, reverse)
    paired: dict[tuple, EvidenceCell] = {}
    for key, first in forward.cells.items():
        second = reverse.cells[key]
        require(
            first.actual_threads == second.actual_threads
            and first.placement == second.placement
            and first.selected_variant == second.selected_variant,
            "forward/reverse case execution semantics differ",
        )
        planner_regret = None
        comparable_regret = None
        if first.planner_regret is not None:
            require(
                second.planner_regret is not None
                and first.comparable_planner_regret is not None
                and second.comparable_planner_regret is not None,
                "forward/reverse planner-regret evidence differs",
            )
            planner_regret = (first.planner_regret + second.planner_regret) / 2.0
            comparable_regret = (
                first.comparable_planner_regret
                + second.comparable_planner_regret
            ) / 2.0
        paired[key] = dataclasses.replace(
            first,
            seconds=(first.seconds + second.seconds) / 2.0,
            seconds_low=min(first.seconds, second.seconds),
            seconds_high=max(first.seconds, second.seconds),
            preparation_seconds=(
                first.preparation_seconds + second.preparation_seconds
            )
            / 2.0,
            amortized_seconds=(first.amortized_seconds + second.amortized_seconds)
            / 2.0,
            planner_regret=planner_regret,
            comparable_planner_regret=comparable_regret,
            planner_exclusions=max(
                first.planner_exclusions, second.planner_exclusions
            ),
        )
    return paired


def milliseconds_range(cell: EvidenceCell) -> str:
    return (
        f"{cell.seconds * 1.0e3:.3f} "
        f"[{cell.seconds_low * 1.0e3:.3f}, {cell.seconds_high * 1.0e3:.3f}]"
    )


def safe_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(
    forward: Bundle, reverse: Bundle | None, cells: dict[tuple, EvidenceCell]
) -> str:
    manifest = forward.manifest
    environment = forward.environment
    direction = (
        "paired stable-forward/stable-reverse"
        if reverse is not None
        else "stable-forward only"
    )
    lines = [
        "# MDSLC CPU Performance Deep Audit — Sanitized Evidence",
        "",
        (
            "This report contains authenticated aggregates only; raw benchmark "
            "JSON remains external and untracked."
        ),
        (
            "Timing entries are arithmetic midpoints of run-order medians with "
            "the explicit `[minimum, maximum]` direction range."
            if reverse is not None
            else "Timing entries use one stable-forward run; `[minimum, maximum]` "
            "therefore repeats that single median."
        ),
        "",
        "## Coverage",
        "",
        "| Evidence | Count |",
        "|---|---:|",
        f"| Executable cases | {len(manifest['cases'])} |",
        f"| Passed | {forward.states['passed']} |",
        f"| Reused after authenticated resume | {forward.states['reused']} |",
        f"| Expected legality rejections | {forward.states['rejected']} |",
        f"| Predeclared runtime-bound skips | {len(manifest['skips'])} |",
        f"| Accepted raw reports | {len(forward.cells)} |",
        f"| Direction treatment | {safe_text(direction)} |",
    ]
    if reverse is not None:
        lines.extend(
            [
                f"| Reverse passed | {reverse.states['passed']} |",
                (
                    "| Reverse reused after authenticated resume | "
                    f"{reverse.states['reused']} |"
                ),
                f"| Reverse expected legality rejections | {reverse.states['rejected']} |",
            ]
        )
    lines.extend(
        [
            "",
            "All `failed` and incomplete states are fatal to this summarizer.",
            "",
            "## Evidence identity and environment",
            "",
            f"- Benchmark binary SHA-256: `{manifest['benchmark_binary_sha256']}`",
            f"- Runner SHA-256: `{manifest['runner_sha256']}`",
            f"- Forward manifest SHA-256: `{forward.digest}`",
        ]
    )
    if reverse is not None:
        lines.append(f"- Reverse manifest SHA-256: `{reverse.digest}`")
    lines.extend(
        [
            f"- Source commit: `{manifest['benchmark_source_commit']}`",
            f"- Seed: `{manifest['benchmark_seed']}`",
            f"- Host: {safe_text(environment.get('cpu_model', 'unknown'))}",
            (
                f"- Platform/compiler: {safe_text(environment.get('os_family', 'unknown'))} "
                f"{safe_text(environment.get('architecture', 'unknown'))}; "
                f"{safe_text(environment.get('compiler', 'unknown'))}; "
                f"{safe_text(environment.get('build_type', 'unknown'))}"
            ),
            (
                f"- Frequency metadata: governor="
                f"{safe_text(environment.get('governor', 'unknown'))}, policy="
                f"{safe_text(environment.get('frequency_policy', 'unknown'))}, boost="
                f"{safe_text(environment.get('boost_state', 'unknown'))}"
            ),
            (
                f"- Capability record: "
                f"{safe_text(environment.get('capability_record', 'unknown'))}"
            ),
            (
                f"- Topology record: "
                f"{safe_text(environment.get('topology_record', 'unknown'))}"
            ),
            "",
        ]
    )

    complete = [
        cell
        for cell in cells.values()
        if cell.mode == "complete-hot"
        and cell.requested_threads == 1
        and cell.actual_threads == 1
    ]
    by_shape_variant = {
        (cell.shape, cell.variant): cell for cell in complete
    }
    shapes = sorted({cell.shape for cell in complete})
    comparisons = []
    for shape in shapes:
        provider = by_shape_variant.get((shape, EXTERNAL_VARIANT))
        natives = [
            cell
            for cell in complete
            if cell.shape == shape
            and cell.variant != EXTERNAL_VARIANT
            and provider is not None
            and cell.placement == provider.placement
        ]
        if provider is None or not natives:
            continue
        fastest = min(natives, key=lambda cell: cell.seconds)
        comparisons.append((fastest.family, shape, fastest, provider))

    lines.extend(
        [
            "## Complete-hot single-thread native versus OpenBLAS",
            "",
            (
                "Only requested/actual one-thread cells with identical SMT, "
                "affinity, and applied-affinity metadata are compared."
            ),
            "",
            "| Family | Shapes | Median native/OpenBLAS throughput ratio |",
            "|---|---:|---:|",
        ]
    )
    family_ratios: dict[str, list[float]] = defaultdict(list)
    for family, _, native, provider in comparisons:
        family_ratios[family].append(provider.seconds / native.seconds)
    for family in sorted(family_ratios):
        lines.append(
            f"| {family} | {len(family_ratios[family])} | "
            f"{median(family_ratios[family]):.3f} |"
        )
    if not family_ratios:
        lines.append("| none | 0 | n/a |")

    weak = sorted(
        (
            (provider.seconds / native.seconds, family, shape, native, provider)
            for family, shape, native, provider in comparisons
            if provider.seconds / native.seconds < 0.75
        ),
        key=lambda item: item[0],
    )
    lines.extend(
        [
            "",
            "### Weak shapes (native/OpenBLAS throughput ratio below 0.75)",
            "",
            "| Family | M×N×K | Fastest native | Native ms [range] | OpenBLAS ms [range] | Ratio |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    if weak:
        for ratio, family, shape, native, provider in weak[:16]:
            lines.append(
                f"| {family} | {'×'.join(map(str, shape))} | "
                f"`{native.variant}` | {milliseconds_range(native)} | "
                f"{milliseconds_range(provider)} | {ratio:.3f} |"
            )
    else:
        lines.append("| none | — | — | — | — | — |")

    complete_lookup = {
        (cell.shape, cell.variant, cell.requested_threads): cell
        for cell in cells.values()
        if cell.mode == "complete-hot"
    }
    one_shot_ratios = []
    for cell in cells.values():
        if cell.mode != "one-shot-hot":
            continue
        baseline = complete_lookup.get(
            (cell.shape, cell.variant, cell.requested_threads)
        )
        if (
            baseline is not None
            and cell.actual_threads == baseline.actual_threads
            and cell.placement == baseline.placement
        ):
            one_shot_ratios.append(cell.seconds / baseline.seconds)
    lines.extend(
        [
            "",
            "## Allocation-inclusive one-shot",
            "",
            "Ratios are one-shot time divided by the equivalent reused-workspace complete-call time.",
            "",
            "| Comparable cells | Median ratio | P95 ratio | Maximum ratio |",
            "|---:|---:|---:|---:|",
        ]
    )
    if one_shot_ratios:
        lines.append(
            f"| {len(one_shot_ratios)} | {median(one_shot_ratios):.3f} | "
            f"{percentile_nearest_rank(one_shot_ratios, 0.95):.3f} | "
            f"{max(one_shot_ratios):.3f} |"
        )
    else:
        lines.append("| 0 | n/a | n/a | n/a |")

    prepack_by_sequence: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for cell in cells.values():
        if cell.mode != "prepacked-b-hot":
            continue
        baseline = complete_lookup.get(
            (cell.shape, cell.variant, cell.requested_threads)
        )
        if (
            baseline is not None
            and cell.actual_threads == baseline.actual_threads
            and cell.placement == baseline.placement
        ):
            prepack_by_sequence[cell.lhs_sequence].append(
                (
                    cell.preparation_seconds,
                    cell.seconds,
                    cell.amortized_seconds / baseline.seconds,
                )
            )
    lines.extend(
        [
            "",
            "## Prepacked-B preparation and amortization",
            "",
            (
                "Preparation is measured once outside steady-state execution. "
                "The amortized ratio includes that preparation cost."
            ),
            "",
            "| Repeated A inputs | Cells | Median preparation ms | Median steady-state ms/execution | Median amortized/complete ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for sequence in sorted(prepack_by_sequence):
        values = prepack_by_sequence[sequence]
        lines.append(
            f"| {sequence} | {len(values)} | "
            f"{median(value[0] for value in values) * 1.0e3:.3f} | "
            f"{median(value[1] for value in values) * 1.0e3:.3f} | "
            f"{median(value[2] for value in values):.3f} |"
        )
    if not prepack_by_sequence:
        lines.append("| — | 0 | n/a | n/a | n/a |")

    diagnostic_modes = (
        (
            "complete-cold",
            "Cold-cache diagnostic; each sample is independently evicted and uses the diagnostic timer floor.",
        ),
        (
            "compute-only-hot",
            "Compute-only microkernel diagnostic; packing is excluded and results are not complete BLAS comparisons.",
        ),
    )
    lines.extend(["", "## Cold-cache and compute-only diagnostics", ""])
    for mode, label in diagnostic_modes:
        ratios = []
        mode_cells = [cell for cell in cells.values() if cell.mode == mode]
        for cell in mode_cells:
            baseline = complete_lookup.get(
                (cell.shape, cell.variant, cell.requested_threads)
            )
            if (
                baseline is not None
                and cell.actual_threads == baseline.actual_threads
                and cell.placement == baseline.placement
            ):
                ratios.append(cell.seconds / baseline.seconds)
        ratio_text = f"{median(ratios):.3f}" if ratios else "n/a"
        lines.append(
            f"- **{mode}:** {label} Accepted cells: {len(mode_cells)}; "
            f"comparable-to-hot cells: {len(ratios)}; median diagnostic/hot ratio: "
            f"{ratio_text}."
        )

    regret_cells = [
        cell
        for cell in cells.values()
        if cell.mode == "planner-regret-hot"
        and cell.comparable_planner_regret is not None
    ]
    regrets = [float(cell.comparable_planner_regret) for cell in regret_cells]
    lines.extend(
        [
            "",
            "## Planner regret",
            "",
            (
                "The sanitized regret recomputation compares only candidates "
                "with the selected candidate's actual thread count and placement. "
                "The raw all-legal-candidate regret remains authenticated but is "
                "not used for this cross-placement summary."
            ),
            "",
            "| Cells | Median | P95 | Maximum | Candidate timings excluded for thread/placement mismatch |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    if regrets:
        lines.append(
            f"| {len(regrets)} | {median(regrets):.3f} | "
            f"{percentile_nearest_rank(regrets, 0.95):.3f} | "
            f"{max(regrets):.3f} | "
            f"{sum(cell.planner_exclusions for cell in regret_cells)} |"
        )
    else:
        lines.append("| 0 | n/a | n/a | n/a | 0 |")

    unbound_provider = [
        cell
        for cell in cells.values()
        if cell.mode == "complete-hot"
        and cell.variant == EXTERNAL_VARIANT
        and cell.actual_threads > 1
        and cell.placement[1:3] == ("allow-smt", "none")
    ]
    provider_by_threads: dict[int, list[float]] = defaultdict(list)
    for cell in unbound_provider:
        provider_by_threads[cell.actual_threads].append(cell.gflops)
    clamped = [
        cell
        for cell in cells.values()
        if cell.requested_threads != cell.actual_threads
    ]
    lines.extend(
        [
            "",
            "## Thread and placement exclusions",
            "",
            (
                "Multi-thread OpenBLAS uses unbound `allow-smt/none` placement "
                "in this audit and is retained only as a separate diagnostic; "
                "it is not compared with compact physical-core native runs."
            ),
            "",
            "| Actual OpenBLAS threads | Diagnostic cells | Median GFLOP/s |",
            "|---:|---:|---:|",
        ]
    )
    for thread_count in sorted(provider_by_threads):
        lines.append(
            f"| {thread_count} | {len(provider_by_threads[thread_count])} | "
            f"{median(provider_by_threads[thread_count]):.3f} |"
        )
    if not provider_by_threads:
        lines.append("| — | 0 | n/a |")
    lines.extend(
        [
            "",
            f"- Requested-to-actual thread clamps retained and excluded where "
            f"comparability changed: {len(clamped)} cells.",
            "",
            "## Rejections and predeclared skips",
            "",
            "| Category | Count |",
            "|---|---:|",
        ]
    )
    for category, count in sorted(forward.rejection_categories.items()):
        lines.append(f"| expected rejection: {safe_text(category)} | {count} |")
    skip_counts = Counter(skip["reason"] for skip in manifest["skips"])
    for reason, count in sorted(skip_counts.items()):
        lines.append(f"| predeclared skip: {safe_text(reason)} | {count} |")
    if not forward.rejection_categories and not skip_counts:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "These host-bounded measurements authenticate correctness and "
                "timing provenance. They do not establish universal performance, "
                "physical multi-node NUMA behavior, or hardware-counter claims."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_output(path: pathlib.Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-manifest", required=True)
    parser.add_argument("--reverse-manifest")
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.markdown_out)
    try:
        forward = load_bundle(
            pathlib.Path(args.forward_manifest), "stable-forward"
        )
        reverse = (
            load_bundle(pathlib.Path(args.reverse_manifest), "stable-reverse")
            if args.reverse_manifest
            else None
        )
        cells = pair_cells(forward, reverse)
        rendered = render_report(forward, reverse, cells)
        write_output(output, rendered)
    except AuditError as error:
        print(f"matcore deep-audit summary rejected: {error}", file=sys.stderr)
        return 2
    print(
        f"matcore deep-audit summary: {len(cells)} authenticated raw cells; "
        f"markdown={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
