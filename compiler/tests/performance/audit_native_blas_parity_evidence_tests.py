#!/usr/bin/env python3

"""Adversarial contract tests for the parity evidence completeness auditor."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


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


def load_module(name: str, path: pathlib.Path):
    import importlib.util

    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def execute_audit(
    auditor: pathlib.Path,
    forward: pathlib.Path | None,
    reverse: pathlib.Path | None,
    source_commit: str,
    physical_cores: int,
    output: pathlib.Path,
    expected: int,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(auditor),
        "--expected-source-commit",
        source_commit,
        "--physical-cores",
        str(physical_cores),
        "--json-out",
        str(output),
    ]
    if forward is not None:
        command.extend(("--forward-manifest", str(forward)))
    if reverse is not None:
        command.extend(("--reverse-manifest", str(reverse)))
    return run(command, expected=expected)


def gap_codes(bundle: dict) -> set[str]:
    return {gap["code"] for gap in bundle["gaps"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--summarizer", required=True)
    args = parser.parse_args()
    auditor = pathlib.Path(args.auditor).resolve()
    summarizer = pathlib.Path(args.summarizer).resolve()
    summary = load_module("matcore_native_parity_summary_for_audit", summarizer)
    fixtures = load_module(
        "matcore_native_parity_summary_fixtures",
        summarizer.with_name("summarize_native_blas_parity_tests.py"),
    )
    runner, runner_path = summary.load_runner()
    repository = runner_path.parents[3]
    source_commit = run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"]
    ).stdout.strip()
    physical_cores = 12

    with tempfile.TemporaryDirectory(
        prefix="matcore native parity evidence audit "
    ) as temporary:
        root = pathlib.Path(temporary)
        benchmark = root / "matcore-bench"
        benchmark.write_bytes(b"authenticated fake benchmark\n")
        forward = fixtures.write_bundle(
            root,
            "stable-forward",
            summary,
            runner,
            benchmark,
            source_commit,
            runner_path,
            physical_cores,
        )
        reverse = fixtures.write_bundle(
            root,
            "stable-reverse",
            summary,
            runner,
            benchmark,
            source_commit,
            runner_path,
            physical_cores,
        )
        expected_cases = len(summary.expected_cases(runner, physical_cores))

        ready_output = root / "ready.json"
        execute_audit(
            auditor,
            forward,
            reverse,
            source_commit,
            physical_cores,
            ready_output,
            expected=0,
        )
        ready = json.loads(ready_output.read_text(encoding="utf-8"))
        assert ready["schema"] == "matcore.native-blas-parity.evidence-audit"
        assert ready["version"] == 1
        assert ready["overall_status"] == "ready-for-bounded-summary"
        assert ready["pair"]["authenticated_pair_ready_for_bounded_summary"]
        assert ready["pair"]["paired_cells"] == expected_cases
        assert "does not establish native BLAS parity" in ready["claim_boundary"]
        for order in ("stable-forward", "stable-reverse"):
            bundle = ready["bundles"][order]
            assert bundle["status"] == "authenticated"
            assert bundle["coverage"]["expected_cases"] == expected_cases
            assert bundle["coverage"]["missing_case_count"] == 0
            assert bundle["coverage"]["case_order_matches"]
            assert bundle["raw_artifacts"]["record_authenticated"] == expected_cases
            assert bundle["strict_bundle_authentication"]["passed"]
            assert bundle["provider_metadata"]["values"] == [
                {
                    "name": "OpenBLAS",
                    "version": "0.3.32",
                    "config": "USE_THREAD=1",
                }
            ]

        pristine_forward = forward.read_bytes()
        forward_manifest = json.loads(pristine_forward.decode("utf-8"))

        # Selective truncation is reported as an exact missing key and remains
        # rejected by the strict bundle authenticator.
        omitted_record = forward_manifest["cases"].pop()
        forward.write_text(
            json.dumps(forward_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        omitted_output = root / "omitted.json"
        execute_audit(
            auditor,
            forward,
            reverse,
            source_commit,
            physical_cores,
            omitted_output,
            expected=1,
        )
        omitted = json.loads(omitted_output.read_text(encoding="utf-8"))
        omitted_bundle = omitted["bundles"]["stable-forward"]
        assert omitted["overall_status"] == "incomplete"
        assert omitted_bundle["coverage"]["missing_case_count"] == 1
        assert omitted_bundle["coverage"]["missing_case_keys"] == [
            omitted_record["key"]
        ]
        assert "expected-cases-missing" in gap_codes(omitted_bundle)
        assert "strict-bundle-rejected" in gap_codes(omitted_bundle)
        assert not omitted["pair"][
            "authenticated_pair_ready_for_bounded_summary"
        ]
        forward.write_bytes(pristine_forward)

        # A non-final state cannot be confused with an authenticated result,
        # even when stale raw bytes remain next to the manifest.
        planned_manifest = json.loads(pristine_forward.decode("utf-8"))
        planned_manifest["cases"][0]["state"] = "planned"
        forward.write_text(
            json.dumps(planned_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        planned_output = root / "planned.json"
        execute_audit(
            auditor,
            forward,
            reverse,
            source_commit,
            physical_cores,
            planned_output,
            expected=1,
        )
        planned = json.loads(planned_output.read_text(encoding="utf-8"))
        planned_bundle = planned["bundles"]["stable-forward"]
        assert planned_bundle["coverage"]["state_counts"]["planned"] == 1
        assert "unfinished-case-states" in gap_codes(planned_bundle)
        assert "unexpected-raw-files" in gap_codes(planned_bundle)
        forward.write_bytes(pristine_forward)

        # Raw tampering is exposed independently from the strict summarizer's
        # first-failure diagnostic.
        first_record = json.loads(pristine_forward.decode("utf-8"))["cases"][0]
        first_raw = forward.parent / first_record["raw_file"]
        pristine_raw = first_raw.read_bytes()
        first_raw.write_bytes(pristine_raw + b"tamper\n")
        tampered_output = root / "tampered.json"
        execute_audit(
            auditor,
            forward,
            reverse,
            source_commit,
            physical_cores,
            tampered_output,
            expected=1,
        )
        tampered = json.loads(tampered_output.read_text(encoding="utf-8"))
        tampered_bundle = tampered["bundles"]["stable-forward"]
        assert tampered_bundle["raw_artifacts"][
            "digest_mismatch_case_keys"
        ] == [first_record["key"]]
        assert "raw-digest-mismatch" in gap_codes(tampered_bundle)
        first_raw.write_bytes(pristine_raw)

        provider_manifest = json.loads(pristine_forward.decode("utf-8"))
        provider_record = next(
            record
            for record in provider_manifest["cases"]
            if record["variant"] == summary.OPENBLAS
            and record["state"] == "passed"
        )
        provider_raw = forward.parent / provider_record["raw_file"]
        pristine_provider_raw = provider_raw.read_bytes()
        missing_provider_identity = json.loads(
            pristine_provider_raw.decode("utf-8")
        )
        missing_provider_identity["environment"]["provider_version"] = "UNKNOWN"
        missing_provider_identity["environment"]["provider_config"] = "   "
        provider_raw.write_text(
            json.dumps(missing_provider_identity, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provider_record["sha256"] = fixtures.sha256(provider_raw)
        forward.write_text(
            json.dumps(provider_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provider_output = root / "provider-identity.json"
        execute_audit(
            auditor,
            forward,
            reverse,
            source_commit,
            physical_cores,
            provider_output,
            expected=1,
        )
        provider_audit = json.loads(
            provider_output.read_text(encoding="utf-8")
        )
        provider_bundle = provider_audit["bundles"]["stable-forward"]
        assert "record-authentication-failed" in gap_codes(provider_bundle)
        assert any(
            "provider metadata" in error["message"]
            for error in provider_bundle["raw_artifacts"][
                "authentication_errors"
            ]
        )
        provider_raw.write_bytes(pristine_provider_raw)
        forward.write_bytes(pristine_forward)

        # Strictly valid evidence for another checkpoint is not ready for the
        # explicitly requested campaign checkpoint.
        wrong_source_output = root / "wrong-source.json"
        execute_audit(
            auditor,
            forward,
            reverse,
            "0" * 40,
            physical_cores,
            wrong_source_output,
            expected=1,
        )
        wrong_source = json.loads(
            wrong_source_output.read_text(encoding="utf-8")
        )
        assert wrong_source["bundles"]["stable-forward"][
            "strict_bundle_authentication"
        ]["passed"]
        assert "source-checkpoint-mismatch" in gap_codes(
            wrong_source["bundles"]["stable-forward"]
        )
        assert not wrong_source["pair"][
            "authenticated_pair_ready_for_bounded_summary"
        ]

        # A one-sided campaign reports the entire absent reverse matrix.
        one_sided_output = root / "one-sided.json"
        execute_audit(
            auditor,
            forward,
            None,
            source_commit,
            physical_cores,
            one_sided_output,
            expected=1,
        )
        one_sided = json.loads(one_sided_output.read_text(encoding="utf-8"))
        missing_reverse = one_sided["bundles"]["stable-reverse"]
        assert missing_reverse["coverage"]["missing_case_count"] == expected_cases
        assert "manifest-argument-missing" in gap_codes(missing_reverse)

        # The auditor never overwrites a manifest, raw artifact, executable,
        # runner, or summarizer selected as its JSON output.
        sentinel = forward.read_bytes()
        collision = run(
            [
                sys.executable,
                str(auditor),
                "--forward-manifest",
                str(forward),
                "--reverse-manifest",
                str(reverse),
                "--expected-source-commit",
                source_commit,
                "--physical-cores",
                str(physical_cores),
                "--json-out",
                str(forward),
            ],
            expected=2,
        )
        assert "collides with an evidence or tool input" in collision.stderr
        assert forward.read_bytes() == sentinel

        # Atomic output staging also cannot overwrite a protected input whose
        # name aliases the old deterministic ``.<output>.tmp`` sibling.
        staged_collision_manifest = forward.parent / ".staged-output.json.tmp"
        staged_collision_manifest.write_bytes(sentinel)
        staged_collision_output = forward.parent / "staged-output.json"
        execute_audit(
            auditor,
            staged_collision_manifest,
            reverse,
            source_commit,
            physical_cores,
            staged_collision_output,
            expected=0,
        )
        assert staged_collision_manifest.read_bytes() == sentinel
        staged_report = json.loads(
            staged_collision_output.read_text(encoding="utf-8")
        )
        assert staged_report["overall_status"] == "ready-for-bounded-summary"

        # Output failures use the documented tool/safety status without a
        # traceback and do not remove or replace the selected directory.
        output_directory = root / "directory-is-not-json"
        output_directory.mkdir()
        invalid_output = execute_audit(
            auditor,
            forward,
            reverse,
            source_commit,
            physical_cores,
            output_directory,
            expected=2,
        )
        assert "audit failed:" in invalid_output.stderr
        assert "Traceback" not in invalid_output.stderr
        assert output_directory.is_dir()

    print("matcore native BLAS parity evidence audit contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
