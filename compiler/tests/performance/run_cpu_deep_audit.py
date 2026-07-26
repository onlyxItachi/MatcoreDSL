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
import os
import pathlib
import subprocess
import sys
import time
from typing import Iterable


SCHEMA_VERSION = 5
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
    (1024, 1024, 1024),
    (4096, 64, 4096),
    (64, 4096, 4096),
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
            for variant in PACKED_VARIANTS[:2]:
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
            for variant in PACKED_VARIANTS:
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
        str(timer_floor_us),
        "--max-memory-mib",
        str(maximum_memory_mib),
        "--alignment",
        "64",
        "--reuse-workspace",
        "--guard",
        "--json-out",
        str(output),
    ]
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


def authenticate_report(path: pathlib.Path, case: AuditCase) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "matcore.benchmark.cpu.gemm":
        raise ValueError("unexpected benchmark schema")
    if report.get("version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema version {SCHEMA_VERSION}")
    if report["environment"]["source_provenance_state"] != "clean":
        raise ValueError("benchmark source provenance is not clean")
    if report["environment"]["source_worktree_dirty"]:
        raise ValueError("benchmark source worktree was dirty")
    configuration = report["configuration"]
    if configuration["lhs_sequence_length"] != case.lhs_sequence:
        raise ValueError("left-input sequence metadata mismatch")
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
    if result["actual_threads"] != case.threads and (
        case.variant in PARALLEL_VARIANTS or case.variant == EXTERNAL_VARIANT
    ):
        raise ValueError("actual implementation thread count differs from request")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--suites",
        default="complete",
        help="comma list: complete,compute,cold,prepacked,regret,all",
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
    args = parser.parse_args()

    executable = pathlib.Path(args.bench).resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable does not exist: {executable}")
    source_root = pathlib.Path(__file__).resolve().parents[3]
    try:
        output_dir = safe_output_directory(
            pathlib.Path(args.output_dir), source_root
        )
    except ValueError as error:
        parser.error(str(error))

    suites = {value for value in args.suites.split(",") if value}
    if "all" in suites:
        suites = {"complete", "compute", "cold", "prepacked", "regret"}
    unknown_suites = suites - {"complete", "compute", "cold", "prepacked", "regret"}
    if not suites or unknown_suites:
        parser.error(f"unknown or empty suite set: {sorted(unknown_suites)}")
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
    if args.limit:
        cases = cases[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
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
        if args.resume and raw_path.exists():
            authenticate_report(raw_path, case)
            record["state"] = "reused"
            record["sha256"] = sha256(raw_path)
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
            record["state"] = "failed"
            record["returncode"] = completed.returncode
            record["stdout"] = completed.stdout
            record["stderr"] = completed.stderr
            records.append(record)
            break
        authenticate_report(raw_path, case)
        record["state"] = "passed"
        record["sha256"] = sha256(raw_path)
        records.append(record)

    manifest = {
        "schema": "matcore.cpu-performance-deep-audit.manifest",
        "version": 1,
        "benchmark_schema_version": SCHEMA_VERSION,
        "started_unix_seconds": started,
        "finished_unix_seconds": int(time.time()),
        "benchmark": str(executable),
        "suites": sorted(suites),
        "variants": list(variants),
        "threads": list(threads),
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
