#!/usr/bin/env python3

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--bench", required=True)
    args = parser.parse_args()
    runner = pathlib.Path(args.runner).resolve()
    bench = pathlib.Path(args.bench).resolve()
    repository = runner.parents[3]

    with tempfile.TemporaryDirectory(prefix="matcore deep audit plan ") as temporary:
        output = pathlib.Path(temporary) / "raw output"
        run(
            [
                sys.executable,
                str(runner),
                "--bench",
                str(bench),
                "--output-dir",
                str(output),
                "--suites",
                "all",
                "--threads",
                "1,2,4,12",
                "--dry-run",
            ]
        )
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["schema"] == "matcore.cpu-performance-deep-audit.manifest"
        assert manifest["version"] == 2
        assert manifest["benchmark_schema_version"] == 6
        assert len(manifest["benchmark_binary_sha256"]) == 64
        assert len(manifest["runner_sha256"]) == 64
        assert len(manifest["plan_sha256"]) == 64
        assert manifest["benchmark_seed"] == 0x4D4154434F524531
        assert manifest["benchmark_source_commit"] is None
        assert manifest["dry_run"] is True
        assert manifest["threads"] == [1, 2, 4, 12]
        assert manifest["cases"]
        assert manifest["skips"]

        keys = [case["key"] for case in manifest["cases"]]
        assert len(keys) == len(set(keys))
        assert all("--seed" in case["command"] for case in manifest["cases"])
        families = {case["family"] for case in manifest["cases"]}
        assert families == {
            "small-square",
            "medium-square",
            "large-square",
            "tall-skinny",
            "short-wide",
            "vector-like",
            "tail-heavy",
        }
        partitions = {case["partition"] for case in manifest["cases"]}
        assert partitions == {"diagnostic", "calibration", "holdout"}
        shapes = {tuple(case["shape"]) for case in manifest["cases"]}
        assert (4096, 4096, 4096) in shapes
        assert (4096, 64, 4096) in shapes
        assert (64, 4096, 4096) in shapes
        assert (1, 4096, 4096) in shapes
        assert (511, 513, 515) in shapes

        prepacked = [
            case for case in manifest["cases"] if case["mode"] == "prepacked-b-hot"
        ]
        assert {case["lhs_sequence"] for case in prepacked} == {1, 4, 16, 64}
        assert all(case["variant"].startswith("cpu.native-") for case in prepacked)
        compute = [
            case for case in manifest["cases"] if case["mode"] == "compute-only-hot"
        ]
        assert {
            case["variant"] for case in compute
        } == {"cpu.native-packed.avx2-fma.f32.v1"}
        assert all("--exclude-packing" in case["command"] for case in compute)
        assert {
            case["variant"] for case in prepacked
        } == {
            "cpu.native-packed.avx2-fma.f32.v1",
            "cpu.native-packed.avx512-fma.f32.v1",
        }
        one_shot = [
            case for case in manifest["cases"] if case["mode"] == "one-shot-hot"
        ]
        assert one_shot
        assert all(
            "--include-allocation" in case["command"]
            and "--reuse-workspace" not in case["command"]
            for case in one_shot
        )
        cold = [
            case for case in manifest["cases"] if case["mode"] == "complete-cold"
        ]
        assert cold
        for case in cold:
            floor_index = case["command"].index("--timer-floor-us") + 1
            assert case["command"][floor_index] == "1"
        regret = [
            case
            for case in manifest["cases"]
            if case["mode"] == "planner-regret-hot"
        ]
        assert regret
        for case in regret:
            floor_index = case["command"].index("--timer-floor-us") + 1
            assert case["command"][floor_index] == "1"

        reference_large_skip = [
            skip
            for skip in manifest["skips"]
            if skip["shape"] == [4096, 4096, 4096]
            and skip["variant"] == "cpu.reference.f32.v1"
        ]
        assert reference_large_skip
        assert all(
            "audit runtime bound" in skip["reason"] for skip in reference_large_skip
        )

        parallel = [
            case
            for case in manifest["cases"]
            if case["variant"] == "cpu.native-parallel.avx2-fma.f32.v1"
            and case["mode"] == "complete-hot"
        ]
        assert {case["threads"] for case in parallel} == {2, 4, 12}
        assert all(
            "--physical-cores-only" in case["command"]
            and case["command"][-1] == "compact"
            for case in parallel
        )
        provider_parallel = [
            case
            for case in manifest["cases"]
            if case["variant"] == "cpu.external.openblas.f32.v1"
            and case["threads"] > 1
            and case["mode"] == "complete-hot"
        ]
        assert provider_parallel
        assert all(
            "--allow-smt" in case["command"]
            and case["command"][-1] == "none"
            for case in provider_parallel
        )

        reverse_output = pathlib.Path(temporary) / "reverse raw output"
        run(
            [
                sys.executable,
                str(runner),
                "--bench",
                str(bench),
                "--output-dir",
                str(reverse_output),
                "--suites",
                "all",
                "--threads",
                "1,2,4,12",
                "--case-order",
                "stable-reverse",
                "--dry-run",
            ]
        )
        reverse_manifest = json.loads(
            (reverse_output / "manifest.json").read_text(encoding="utf-8")
        )
        assert reverse_manifest["case_order"] == "stable-reverse"
        assert [case["key"] for case in reverse_manifest["cases"]] == list(
            reversed(keys)
        )

    with tempfile.TemporaryDirectory(prefix="matcore deep audit resume ") as temporary:
        output = pathlib.Path(temporary) / "authenticated"
        base_command = [
            sys.executable,
            str(runner),
            "--bench",
            str(bench),
            "--output-dir",
            str(output),
            "--suites",
            "complete",
            "--variants",
            "cpu.reference.f32.v1",
            "--threads",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--timer-floor-us",
            "1",
            "--limit",
            "1",
        ]
        run(base_command)
        first_manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        assert first_manifest["benchmark_source_commit"]
        assert first_manifest["cases"][0]["state"] == "passed"

        run([*base_command, "--resume"])
        resumed_manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        assert resumed_manifest["cases"][0]["state"] == "reused"
        assert (
            resumed_manifest["benchmark_source_commit"]
            == first_manifest["benchmark_source_commit"]
        )

        changed_contract = base_command.copy()
        iterations_index = changed_contract.index("--iterations") + 1
        changed_contract[iterations_index] = "2"
        rejected_resume = run([*changed_contract, "--resume"], expected=2)
        assert "resume identity mismatch for plan_sha256" in rejected_resume.stderr

        raw_path = output / resumed_manifest["cases"][0]["raw_file"]
        raw_path.write_text(
            raw_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        rejected_digest = run([*base_command, "--resume"], expected=2)
        assert "resume raw-file digest mismatch" in rejected_digest.stderr

    with tempfile.TemporaryDirectory(
        prefix="matcore deep audit rejection "
    ) as temporary:
        output = pathlib.Path(temporary) / "classified"
        run(
            [
                sys.executable,
                str(runner),
                "--bench",
                str(bench),
                "--output-dir",
                str(output),
                "--suites",
                "complete",
                "--families",
                "small-square",
                "--variants",
                "cpu.native-parallel.avx2-fma.f32.v1",
                "--threads",
                "2",
                "--case-order",
                "stable-forward",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--timer-floor-us",
                "1",
                "--limit",
                "1",
            ]
        )
        rejection_manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        assert rejection_manifest["cases"][0]["state"] == "rejected"
        assert (
            rejection_manifest["cases"][0]["rejection_category"]
            == "parallel-output-macro-tile-count"
        )

    rejected = run(
        [
            sys.executable,
            str(runner),
            "--bench",
            str(bench),
            "--output-dir",
            str(repository / "docs" / "forbidden raw audit"),
            "--dry-run",
        ],
        expected=2,
    )
    assert "must be under ignored benchmark_reports/" in rejected.stderr

    print("matcore CPU deep-audit runner contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
