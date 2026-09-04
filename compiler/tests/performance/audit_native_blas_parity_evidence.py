#!/usr/bin/env python3

"""Diagnose completeness of native/OpenBLAS parity evidence without blessing it.

This tool is deliberately subordinate to ``summarize_native_blas_parity.py``.
It inventories missing cases, unfinished states, raw-file failures, and identity
differences so an interrupted campaign can be understood.  Only the existing
strict bundle and pair authenticators can mark a pair ready for bounded
summarization, and even that readiness is not a parity verdict.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any


AUDIT_SCHEMA = "matcore.native-blas-parity.evidence-audit"
AUDIT_VERSION = 1
FINAL_STATES = {"passed", "reused", "rejected"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: pathlib.Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run_git(repository: pathlib.Path, *arguments: str) -> tuple[bool, str]:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return False, completed.stderr.strip()
    return True, completed.stdout.strip()


def is_exact_object_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def add_gap(gaps: list[dict[str, object]], code: str, detail: str) -> None:
    gaps.append({"code": code, "detail": detail})


def expected_case_data(
    summary: Any,
    runner: Any,
    physical_cores: int,
    order: str,
) -> tuple[list[Any], list[str]]:
    cases = summary.expected_cases(runner, physical_cores)
    if order == "stable-reverse":
        cases.reverse()
    return cases, [case.key for case in cases]


def empty_coverage(expected_keys: list[str]) -> dict[str, object]:
    return {
        "expected_cases": len(expected_keys),
        "observed_records": 0,
        "missing_case_count": len(expected_keys),
        "missing_case_keys": expected_keys,
        "extra_case_count": 0,
        "extra_case_keys": [],
        "duplicate_case_count": 0,
        "duplicate_case_keys": [],
        "case_order_matches": False,
        "record_contract_mismatch_count": 0,
        "record_contract_mismatches": [],
        "state_counts": {},
    }


def grouped_errors(errors: list[tuple[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for key, message in errors:
        grouped[message].append(key)
    return [
        {
            "message": message,
            "count": len(keys),
            "example_case_keys": keys[:5],
        }
        for message, keys in sorted(grouped.items())
    ]


def inspect_tool(path: pathlib.Path, repository: pathlib.Path) -> dict[str, object]:
    relative = path.relative_to(repository).as_posix()
    _, local_blob = run_git(repository, "hash-object", str(path))
    committed, head_blob = run_git(repository, "rev-parse", f"HEAD:{relative}")
    return {
        "path": relative,
        "sha256": sha256(path),
        "git_blob": local_blob,
        "head_git_blob": head_blob if committed else None,
        "matches_head": committed and local_blob == head_blob,
    }


def audit_manifest(
    *,
    path: pathlib.Path | None,
    expected_order: str,
    expected_source_commit: str,
    physical_cores: int,
    runner: Any,
    runner_path: pathlib.Path,
    summary: Any,
) -> tuple[dict[str, object], Any | None]:
    cases, expected_keys = expected_case_data(
        summary, runner, physical_cores, expected_order
    )
    case_by_key = {case.key: case for case in cases}
    gaps: list[dict[str, object]] = []
    result: dict[str, object] = {
        "expected_order": expected_order,
        "path": str(path) if path is not None else None,
        "manifest_sha256": None,
        "status": "missing",
        "gaps": gaps,
        "contract": {},
        "coverage": empty_coverage(expected_keys),
        "raw_artifacts": {
            "required": 0,
            "present": 0,
            "digest_verified": 0,
            "record_authenticated": 0,
            "missing_case_keys": [],
            "digest_mismatch_case_keys": [],
            "unexpected_case_keys": [],
            "authentication_errors": [],
        },
        "provider_metadata": {
            "scope": (
                "record-authenticated raw metadata only; manifest-v3 and "
                "benchmark-v6 do not hash the loaded provider shared object"
            ),
            "values": [],
        },
        "strict_bundle_authentication": {
            "passed": False,
            "error": "manifest is absent",
        },
    }
    if path is None:
        add_gap(
            gaps,
            "manifest-argument-missing",
            f"no {expected_order} manifest was supplied",
        )
        return result, None

    path = path.expanduser().absolute()
    result["path"] = str(path)
    if not path.is_file():
        add_gap(
            gaps,
            "manifest-file-missing",
            f"{expected_order} manifest does not exist",
        )
        return result, None

    result["manifest_sha256"] = sha256(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result["status"] = "invalid"
        result["strict_bundle_authentication"]["error"] = str(error)
        add_gap(gaps, "manifest-json-invalid", str(error))
        return result, None
    if not isinstance(manifest, dict):
        result["status"] = "invalid"
        message = "manifest root is not a JSON object"
        result["strict_bundle_authentication"]["error"] = message
        add_gap(gaps, "manifest-root-invalid", message)
        return result, None

    result["status"] = "inspected"
    contract = result["contract"]
    assert isinstance(contract, dict)

    def expect(field: str, expected: object, code: str) -> None:
        actual = manifest.get(field)
        matches = actual == expected
        contract[field] = {"actual": actual, "expected": expected, "matches": matches}
        if not matches:
            add_gap(gaps, code, f"{field} differs from the campaign contract")

    expect("schema", summary.MANIFEST_SCHEMA, "manifest-schema-mismatch")
    expect("version", summary.MANIFEST_VERSION, "manifest-version-mismatch")
    expect(
        "benchmark_schema_version",
        summary.BENCHMARK_VERSION,
        "benchmark-schema-version-mismatch",
    )
    expect(
        "source_commit",
        expected_source_commit,
        "source-checkpoint-mismatch",
    )
    expect("case_order", expected_order, "case-order-mismatch")
    expect("physical_cores", physical_cores, "physical-core-count-mismatch")
    expect("limit", 0, "limited-manifest")
    expect("dry_run", False, "dry-run-manifest")
    expect(
        "suites",
        sorted(summary.REQUIRED_SUITES),
        "required-suite-set-mismatch",
    )
    expect(
        "thread_strata",
        list(summary.thread_strata(physical_cores)),
        "thread-strata-mismatch",
    )
    expect(
        "parallel_thread_plan",
        summary.expected_parallel_thread_plan(runner, physical_cores),
        "parallel-thread-plan-mismatch",
    )
    expect(
        "partition_interpretation",
        summary.PARTITION_INTERPRETATION,
        "partition-interpretation-mismatch",
    )
    expect("benchmark_seed", summary.BENCHMARK_SEED, "seed-mismatch")
    expect("warmup", summary.WARMUP_ITERATIONS, "warmup-mismatch")
    expect(
        "iterations",
        summary.MEASURED_ITERATIONS,
        "iteration-count-mismatch",
    )
    expect(
        "timer_floor_us",
        summary.TIMER_FLOOR_US,
        "timer-floor-mismatch",
    )
    expect(
        "max_memory_mib",
        summary.MAX_MEMORY_MIB,
        "memory-ceiling-mismatch",
    )
    expect(
        "environment_overrides",
        summary.PROVIDER_ENVIRONMENT,
        "provider-environment-mismatch",
    )

    benchmark_value = manifest.get("benchmark")
    benchmark = (
        pathlib.Path(benchmark_value)
        if isinstance(benchmark_value, str) and benchmark_value
        else None
    )
    benchmark_identity = {
        "path_is_absolute": benchmark is not None and benchmark.is_absolute(),
        "file_exists": benchmark is not None and benchmark.is_file(),
        "declared_sha256": manifest.get("benchmark_binary_sha256"),
        "digest_matches": False,
    }
    if benchmark is not None and benchmark.is_file():
        benchmark_identity["actual_sha256"] = sha256(benchmark)
        benchmark_identity["digest_matches"] = (
            benchmark_identity["actual_sha256"]
            == manifest.get("benchmark_binary_sha256")
        )
    if not benchmark_identity["path_is_absolute"]:
        add_gap(gaps, "benchmark-path-invalid", "benchmark path is not absolute")
    elif not benchmark_identity["file_exists"]:
        add_gap(gaps, "benchmark-file-missing", "benchmark binary is absent")
    elif not benchmark_identity["digest_matches"]:
        add_gap(
            gaps,
            "benchmark-digest-mismatch",
            "benchmark bytes differ from the manifest digest",
        )
    contract["benchmark_identity"] = benchmark_identity
    if benchmark is not None:
        expected_plan_digest = summary.reconstructed_plan_digest(
            cases,
            physical_cores,
            expected_order,
            benchmark,
            summary.expected_parallel_thread_plan(runner, physical_cores),
        )
        expect("plan_sha256", expected_plan_digest, "plan-digest-mismatch")

    local_runner_sha = sha256(runner_path)
    _, local_runner_blob = run_git(
        runner_path.parents[3], "hash-object", str(runner_path)
    )
    runner_identity = {
        "sha256_matches_local": manifest.get("runner_sha256") == local_runner_sha,
        "git_blob_matches_local": (
            manifest.get("runner_git_blob") == local_runner_blob
        ),
    }
    contract["runner_identity"] = runner_identity
    if not runner_identity["sha256_matches_local"]:
        add_gap(
            gaps,
            "runner-sha256-mismatch",
            "manifest runner SHA-256 differs from the local authority",
        )
    if not runner_identity["git_blob_matches_local"]:
        add_gap(
            gaps,
            "runner-git-blob-mismatch",
            "manifest runner Git blob differs from the local authority",
        )

    records = manifest.get("cases")
    if not isinstance(records, list):
        add_gap(gaps, "case-list-invalid", "manifest cases is not a list")
        result["status"] = "incomplete"
        try:
            summary.load_bundle(path, expected_order)
        except (summary.ParityError, KeyError, TypeError, ValueError) as error:
            result["strict_bundle_authentication"]["error"] = str(error)
        return result, None

    valid_keys = [
        record.get("key")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("key"), str)
    ]
    key_counts = collections.Counter(valid_keys)
    observed_key_set = set(valid_keys)
    expected_key_set = set(expected_keys)
    missing_keys = [key for key in expected_keys if key not in observed_key_set]
    extra_keys = sorted(observed_key_set - expected_key_set)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    record_mismatches: list[dict[str, object]] = []
    state_counts: collections.Counter[str] = collections.Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            record_mismatches.append(
                {"index": index, "key": None, "fields": ["record-not-object"]}
            )
            continue
        key = record.get("key")
        state_counts[str(record.get("state"))] += 1
        case = case_by_key.get(key)
        fields: list[str] = []
        if case is None:
            fields.append("unknown-key")
        else:
            for field, expected in summary.expected_record_fields(case).items():
                if record.get(field) != expected:
                    fields.append(field)
        if record.get("index") != index:
            fields.append("index")
        expected_raw = f"{index:04d}__{key}.json" if isinstance(key, str) else None
        if record.get("raw_file") != expected_raw:
            fields.append("raw_file")
        if fields:
            record_mismatches.append(
                {"index": index, "key": key, "fields": sorted(set(fields))}
            )

    coverage = {
        "expected_cases": len(expected_keys),
        "observed_records": len(records),
        "missing_case_count": len(missing_keys),
        "missing_case_keys": missing_keys,
        "extra_case_count": len(extra_keys),
        "extra_case_keys": extra_keys,
        "duplicate_case_count": len(duplicate_keys),
        "duplicate_case_keys": duplicate_keys,
        "case_order_matches": valid_keys == expected_keys,
        "record_contract_mismatch_count": len(record_mismatches),
        "record_contract_mismatches": record_mismatches,
        "state_counts": dict(sorted(state_counts.items())),
    }
    result["coverage"] = coverage
    if missing_keys:
        add_gap(
            gaps,
            "expected-cases-missing",
            f"{len(missing_keys)} of {len(expected_keys)} expected cases are absent",
        )
    if extra_keys:
        add_gap(gaps, "unexpected-cases", f"{len(extra_keys)} case keys are unknown")
    if duplicate_keys:
        add_gap(
            gaps,
            "duplicate-cases",
            f"{len(duplicate_keys)} case keys are duplicated",
        )
    if valid_keys != expected_keys:
        add_gap(
            gaps,
            "case-sequence-mismatch",
            "case keys do not equal the complete stable execution order",
        )
    if record_mismatches:
        add_gap(
            gaps,
            "record-contract-mismatch",
            f"{len(record_mismatches)} records differ from frozen case identity",
        )
    nonfinal = sum(
        count for state, count in state_counts.items() if state not in FINAL_STATES
    )
    if nonfinal:
        add_gap(
            gaps,
            "unfinished-case-states",
            f"{nonfinal} records are not passed, reused, or rejected",
        )

    raw_result = result["raw_artifacts"]
    assert isinstance(raw_result, dict)
    raw_required = 0
    raw_present = 0
    digest_verified = 0
    record_authenticated = 0
    missing_raw: list[str] = []
    digest_mismatch: list[str] = []
    unexpected_raw: list[str] = []
    authentication_errors: list[tuple[str, str]] = []
    provider_metadata: set[tuple[object, object, object]] = set()
    manifest_directory = path.parent
    for record in records:
        if not isinstance(record, dict):
            continue
        key = record.get("key")
        display_key = key if isinstance(key, str) else "<invalid-key>"
        raw_name = record.get("raw_file")
        raw_path = (
            manifest_directory / raw_name
            if isinstance(raw_name, str)
            and pathlib.PurePath(raw_name).name == raw_name
            else None
        )
        state = record.get("state")
        if state in {"passed", "reused"}:
            raw_required += 1
            if raw_path is None or not raw_path.is_file():
                missing_raw.append(display_key)
                continue
            raw_present += 1
            declared_digest = record.get("sha256")
            actual_digest = sha256(raw_path)
            if not is_sha256(declared_digest) or declared_digest != actual_digest:
                digest_mismatch.append(display_key)
                continue
            digest_verified += 1
            case = case_by_key.get(key)
            if case is None or benchmark is None:
                continue
            try:
                summary.authenticate_case_command(runner, benchmark, case, record)
                report = summary.load_json(
                    raw_path, f"raw parity evidence for {display_key}"
                )
                summary.authenticate_report(report, record, manifest)
            except (summary.ParityError, KeyError, TypeError, ValueError) as error:
                authentication_errors.append((display_key, str(error)))
                continue
            record_authenticated += 1
            environment = report.get("environment", {})
            provider_metadata.add(
                (
                    environment.get("provider_name"),
                    environment.get("provider_version"),
                    environment.get("provider_config"),
                )
            )
        elif state == "rejected":
            case = case_by_key.get(key)
            if case is None or benchmark is None:
                continue
            try:
                command_output = summary.authenticate_case_command(
                    runner, benchmark, case, record
                )
                summary.authenticate_rejection(
                    runner,
                    case,
                    record,
                    manifest_directory,
                    command_output,
                )
            except (summary.ParityError, KeyError, TypeError, ValueError) as error:
                authentication_errors.append((display_key, str(error)))
            else:
                record_authenticated += 1
        elif raw_path is not None and raw_path.exists():
            unexpected_raw.append(display_key)

    raw_result.update(
        {
            "required": raw_required,
            "present": raw_present,
            "digest_verified": digest_verified,
            "record_authenticated": record_authenticated,
            "missing_case_keys": missing_raw,
            "digest_mismatch_case_keys": digest_mismatch,
            "unexpected_case_keys": unexpected_raw,
            "authentication_errors": grouped_errors(authentication_errors),
        }
    )
    result["provider_metadata"]["values"] = [
        {"name": name, "version": version, "config": config}
        for name, version, config in sorted(
            provider_metadata, key=lambda value: tuple(str(item) for item in value)
        )
    ]
    if missing_raw:
        add_gap(
            gaps,
            "raw-files-missing",
            f"{len(missing_raw)} accepted records have no raw file",
        )
    if digest_mismatch:
        add_gap(
            gaps,
            "raw-digest-mismatch",
            f"{len(digest_mismatch)} raw files fail SHA-256 verification",
        )
    if unexpected_raw:
        add_gap(
            gaps,
            "unexpected-raw-files",
            f"{len(unexpected_raw)} unfinished records already have raw files",
        )
    if authentication_errors:
        add_gap(
            gaps,
            "record-authentication-failed",
            f"{len(authentication_errors)} final records fail authentication",
        )

    strict_bundle = None
    try:
        strict_bundle = summary.load_bundle(path, expected_order)
    except (summary.ParityError, KeyError, TypeError, ValueError) as error:
        result["strict_bundle_authentication"]["error"] = str(error)
        add_gap(gaps, "strict-bundle-rejected", str(error))
    else:
        result["strict_bundle_authentication"] = {
            "passed": True,
            "error": None,
            "accepted_cells": len(strict_bundle.cells),
            "states": dict(sorted(strict_bundle.states.items())),
            "rejection_categories": dict(
                sorted(strict_bundle.rejection_categories.items())
            ),
        }

    if strict_bundle is not None and not gaps:
        result["status"] = "authenticated"
    else:
        result["status"] = "incomplete"
    return result, strict_bundle


def build_audit(
    *,
    forward_path: pathlib.Path | None,
    reverse_path: pathlib.Path | None,
    expected_source_commit: str,
    physical_cores: int,
    auditor_path: pathlib.Path,
) -> dict[str, object]:
    summary_path = auditor_path.with_name("summarize_native_blas_parity.py")
    summary = load_module("_matcore_native_blas_parity_summary_audit", summary_path)
    runner, runner_path = summary.load_runner()
    repository = runner_path.parents[3]
    _, repository_head = run_git(repository, "rev-parse", "HEAD")
    source_exists, _ = run_git(
        repository, "cat-file", "-e", f"{expected_source_commit}^{{commit}}"
    )
    _, status = run_git(repository, "status", "--porcelain")

    forward, forward_bundle = audit_manifest(
        path=forward_path,
        expected_order="stable-forward",
        expected_source_commit=expected_source_commit,
        physical_cores=physical_cores,
        runner=runner,
        runner_path=runner_path,
        summary=summary,
    )
    reverse, reverse_bundle = audit_manifest(
        path=reverse_path,
        expected_order="stable-reverse",
        expected_source_commit=expected_source_commit,
        physical_cores=physical_cores,
        runner=runner,
        runner_path=runner_path,
        summary=summary,
    )

    pair_gaps: list[dict[str, object]] = []
    pair_authenticated = False
    paired_cells = 0
    pair_error: str | None = None
    if not source_exists:
        pair_error = "expected source commit is not available in the local repository"
        add_gap(pair_gaps, "expected-source-unavailable", pair_error)
    campaign_bundles_match = (
        forward.get("status") == "authenticated"
        and reverse.get("status") == "authenticated"
    )
    if (
        forward_bundle is None
        or reverse_bundle is None
        or not campaign_bundles_match
    ):
        message = "both strict-authenticated bundles are required"
        if pair_error is None:
            pair_error = message
        add_gap(pair_gaps, "authenticated-pair-incomplete", message)
    elif source_exists:
        try:
            cells = summary.pair_bundles(forward_bundle, reverse_bundle)
        except (summary.ParityError, KeyError, TypeError, ValueError) as error:
            pair_error = str(error)
            add_gap(pair_gaps, "strict-pair-rejected", pair_error)
        else:
            pair_authenticated = True
            paired_cells = len(cells)

    expected_cases, _ = expected_case_data(
        summary, runner, physical_cores, "stable-forward"
    )
    report = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "claim_boundary": (
            "diagnostic completeness only; readiness permits the existing "
            "bounded summarizer to run but does not establish native BLAS parity, "
            "performance, scaling, regret acceptance, or provider policy"
        ),
        "campaign_contract": {
            "expected_source_commit": expected_source_commit,
            "expected_source_commit_available_locally": source_exists,
            "physical_cores": physical_cores,
            "expected_cases_per_order": len(expected_cases),
            "expected_orders": ["stable-forward", "stable-reverse"],
            "manifest_version": summary.MANIFEST_VERSION,
            "benchmark_schema_version": summary.BENCHMARK_VERSION,
        },
        "local_inspection_tools": {
            "repository_head": repository_head,
            "working_tree_clean": status == "",
            "auditor": inspect_tool(auditor_path, repository),
            "runner": inspect_tool(runner_path, repository),
            "summarizer": inspect_tool(summary_path, repository),
        },
        "bundles": {
            "stable-forward": forward,
            "stable-reverse": reverse,
        },
        "pair": {
            "authenticated_pair_ready_for_bounded_summary": pair_authenticated,
            "paired_cells": paired_cells,
            "error": pair_error,
            "gaps": pair_gaps,
        },
        "overall_status": (
            "ready-for-bounded-summary" if pair_authenticated else "incomplete"
        ),
    }
    return report


def normalized_path(path: pathlib.Path) -> pathlib.Path:
    return path.expanduser().absolute().resolve(strict=False)


def paths_alias(first: pathlib.Path, second: pathlib.Path) -> bool:
    if normalized_path(first) == normalized_path(second):
        return True
    if first.exists() and second.exists():
        try:
            return first.samefile(second)
        except OSError:
            return False
    return False


def protected_paths(
    auditor_path: pathlib.Path,
    manifests: list[pathlib.Path | None],
) -> set[pathlib.Path]:
    protected = {
        auditor_path,
        auditor_path.with_name("run_native_blas_parity.py"),
        auditor_path.with_name("summarize_native_blas_parity.py"),
    }
    summary = load_module(
        "_matcore_native_blas_parity_summary_paths",
        auditor_path.with_name("summarize_native_blas_parity.py"),
    )
    for manifest in manifests:
        if manifest is not None:
            protected.update(summary.protected_evidence_paths(manifest))
    return protected


def atomic_write_json(path: pathlib.Path, value: object) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = pathlib.Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-manifest")
    parser.add_argument("--reverse-manifest")
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--physical-cores", type=int, required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if not is_exact_object_id(args.expected_source_commit):
        parser.error("--expected-source-commit must be an exact lowercase object ID")
    if args.physical_cores < 2:
        parser.error("--physical-cores must be at least two")

    auditor_path = pathlib.Path(__file__).resolve()
    forward = pathlib.Path(args.forward_manifest) if args.forward_manifest else None
    reverse = pathlib.Path(args.reverse_manifest) if args.reverse_manifest else None
    output = pathlib.Path(args.json_out) if args.json_out else None
    if output is not None:
        for protected in protected_paths(auditor_path, [forward, reverse]):
            if paths_alias(output, protected):
                print(
                    "matcore native-BLAS parity evidence audit rejected: "
                    "JSON output collides with an evidence or tool input",
                    file=sys.stderr,
                )
                return 2
    try:
        report = build_audit(
            forward_path=forward,
            reverse_path=reverse,
            expected_source_commit=args.expected_source_commit,
            physical_cores=args.physical_cores,
            auditor_path=auditor_path,
        )
        if output is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            atomic_write_json(output, report)
            print(
                "matcore native-BLAS parity evidence audit: "
                f"status={report['overall_status']}; json={output}"
            )
    except (OSError, RuntimeError) as error:
        print(
            f"matcore native-BLAS parity evidence audit failed: {error}",
            file=sys.stderr,
        )
        return 2
    return 0 if report["overall_status"] == "ready-for-bounded-summary" else 1


if __name__ == "__main__":
    raise SystemExit(main())
