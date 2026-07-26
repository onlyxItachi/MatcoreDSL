#!/usr/bin/env python3

import argparse
import copy
import importlib.util
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
            f"{command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def load_runner(path: pathlib.Path):
    specification = importlib.util.spec_from_file_location(
        "matcore_native_blas_parity_runner", path
    )
    if specification is None or specification.loader is None:
        raise AssertionError("cannot load native BLAS parity runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def discover_physical_cores(bench: pathlib.Path, output: pathlib.Path) -> int:
    run(
        [
            str(bench),
            "--m",
            "1",
            "--n",
            "1",
            "--k",
            "1",
            "--variant",
            "cpu.reference.f32.v1",
            "--threads",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--lhs-sequence",
            "1",
            "--timer-floor-us",
            "1",
            "--max-memory-mib",
            "64",
            "--seed",
            "1",
            "--alignment",
            "64",
            "--json-out",
            str(output),
            "--reuse-workspace",
            "--include-packing",
            "--hot-cache",
            "--allow-smt",
            "--affinity",
            "none",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    physical = report["environment"]["physical_cores"]
    assert isinstance(physical, int) and physical >= 2
    return physical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--bench", required=True)
    args = parser.parse_args()
    runner = pathlib.Path(args.runner).resolve()
    bench = pathlib.Path(args.bench).resolve()
    repository = runner.parents[3]
    module = load_runner(runner)
    assert module.MANIFEST_VERSION == 3
    assert module.PARTITION_INTERPRETATION == {
        "calibration": "candidate-development-and-validation",
        "holdout": "declared-validation-not-blind",
    }

    expected_calibration = {
        (96, 96, 96),
        (192, 192, 192),
        (384, 384, 384),
        (4096, 64, 4096),
        (4096, 128, 1024),
        (64, 4096, 4096),
        (128, 4096, 1024),
        (63, 65, 67),
        (255, 257, 259),
    }
    expected_validation = {
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 32, 1024),
        (2048, 256, 4096),
        (32, 8192, 1024),
        (256, 2048, 4096),
        (31, 33, 35),
        (127, 129, 131),
        (511, 513, 515),
    }
    assert {
        spec.shape
        for spec in module.PARITY_SHAPES
        if spec.partition == "calibration"
    } == expected_calibration
    assert {
        spec.shape
        for spec in module.PARITY_SHAPES
        if spec.partition == "holdout"
    } == expected_validation

    suites = {
        "parity",
        "auto",
        "regret",
        "repeated",
        "prepacked",
        "diagnostic",
    }
    structural = module.build_cases(suites, 12)
    assert len(structural) > 300
    assert len({case.key for case in structural}) == len(structural)
    primary = [case for case in structural if case.mode == "complete-hot"]
    for spec in module.PARITY_SHAPES:
        matching = [case for case in primary if case.shape == spec.shape]
        expected = {
            (module.PACKED_AVX2, 1),
            (module.PACKED_AVX512, 1),
            (module.OPENBLAS, 1),
        }
        for thread_count in module.exact_parallel_thread_strata(
            spec.shape, 12
        ):
            expected.update(
                {
                    (module.PARALLEL_AVX2, thread_count),
                    (module.PARALLEL_AVX512, thread_count),
                    (module.OPENBLAS, thread_count),
                }
            )
            assert (
                module.parallel_task_capacity(spec.shape, thread_count)
                == thread_count
            )
        assert {(case.variant, case.threads) for case in matching} == expected
    assert module.parallel_task_capacity((512, 512, 512), 12) == 8
    assert module.exact_parallel_thread_strata((512, 512, 512), 12) == (4, 8)
    assert module.parallel_task_capacity((384, 384, 384), 12) == 3
    assert module.exact_parallel_thread_strata((384, 384, 384), 12) == (3,)
    assert module.parallel_task_capacity((1024, 1024, 1024), 12) == 12
    assert module.exact_parallel_thread_strata((1024, 1024, 1024), 12) == (
        4,
        12,
    )
    assert {
        case.threads
        for case in structural
        if case.mode in {"auto-complete-hot", "planner-regret-hot"}
    } == {1, 4, 12}
    repeated = [case for case in structural if case.mode == "repeated-hot"]
    prepacked = [
        case for case in structural if case.mode == "prepacked-b-hot"
    ]
    assert {case.shape for case in repeated} == module.REPEATED_SHAPES
    assert {case.shape for case in prepacked} == module.REPEATED_SHAPES
    assert {case.lhs_sequence for case in repeated} == {1, 4, 16, 64}
    assert {case.lhs_sequence for case in prepacked} == {1, 4, 16, 64}
    assert {case.variant for case in repeated} == set(
        module.REPEATED_VARIANTS
    )
    assert {case.variant for case in prepacked} == set(
        module.PREPACKED_VARIANTS
    )
    assert all(case.threads == 1 for case in repeated + prepacked)

    with tempfile.TemporaryDirectory(
        prefix="matcore native parity runner "
    ) as temporary:
        temporary_path = pathlib.Path(temporary)
        discovery = temporary_path / "physical core discovery.json"
        physical_cores = discover_physical_cores(bench, discovery)
        expected_threads = list(module.thread_strata(physical_cores))

        forward_output = temporary_path / "forward raw evidence"
        forward_command = [
            sys.executable,
            str(runner),
            "--bench",
            str(bench),
            "--output-dir",
            str(forward_output),
            "--physical-cores",
            str(physical_cores),
            "--suites",
            "all",
            "--dry-run",
        ]
        run(forward_command)
        forward = json.loads(
            (forward_output / "manifest.json").read_text(encoding="utf-8")
        )
        assert forward["schema"] == "matcore.native-blas-parity.manifest"
        assert forward["version"] == 3
        assert forward["partition_interpretation"] == (
            module.PARTITION_INTERPRETATION
        )
        assert forward["benchmark_schema_version"] == 6
        assert forward["source_commit"]
        assert forward["runner_git_blob"]
        assert len(forward["benchmark_binary_sha256"]) == 64
        assert len(forward["runner_sha256"]) == 64
        assert len(forward["plan_sha256"]) == 64
        assert forward["benchmark_seed"] == 0x4D4154434F524532
        assert forward["dry_run"] is True
        assert forward["limit"] == 0
        assert forward["thread_strata"] == expected_threads
        assert forward["parallel_thread_plan"] == module.parallel_thread_plan(
            physical_cores
        )
        assert len(forward["cases"]) == len(
            module.build_cases(suites, physical_cores)
        )
        keys = [case["key"] for case in forward["cases"]]
        assert len(keys) == len(set(keys))

        for case in forward["cases"]:
            command = case["command"]
            assert "--guard" in command
            assert "--reuse-workspace" in command
            assert "--hot-cache" in command
            assert "--seed" in command
            assert "--json-out" in command
            if case["threads"] > 1:
                assert "--allow-smt" in command
                assert "--physical-cores-only" not in command
                assert command[-2:] == ["--affinity", "none"]
            else:
                assert "--physical-cores-only" in command
                assert "--allow-smt" not in command
                assert command[-2:] == ["--affinity", "compact"]

        reverse_output = temporary_path / "reverse raw evidence"
        run(
            [
                sys.executable,
                str(runner),
                "--bench",
                str(bench),
                "--output-dir",
                str(reverse_output),
                "--physical-cores",
                str(physical_cores),
                "--suites",
                "all",
                "--case-order",
                "stable-reverse",
                "--dry-run",
            ]
        )
        reverse = json.loads(
            (reverse_output / "manifest.json").read_text(encoding="utf-8")
        )
        assert reverse["case_order"] == "stable-reverse"
        assert [case["key"] for case in reverse["cases"]] == list(
            reversed(keys)
        )

        authenticated_output = temporary_path / "authenticated resume"
        authenticated_command = [
            sys.executable,
            str(runner),
            "--bench",
            str(bench),
            "--output-dir",
            str(authenticated_output),
            "--physical-cores",
            str(physical_cores),
            "--suites",
            "regret",
            "--limit",
            "1",
        ]
        run(authenticated_command)
        first = json.loads(
            (authenticated_output / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert first["cases"][0]["state"] == "passed"
        assert first["cases"][0]["variant"] == "auto"
        assert first["cases"][0]["mode"] == "planner-regret-hot"
        raw_path = authenticated_output / first["cases"][0]["raw_file"]
        raw_document = json.loads(raw_path.read_text(encoding="utf-8"))
        case = module.ParityCase(
            partition=first["cases"][0]["partition"],
            family=first["cases"][0]["family"],
            shape=tuple(first["cases"][0]["shape"]),
            variant=first["cases"][0]["variant"],
            threads=first["cases"][0]["threads"],
            mode=first["cases"][0]["mode"],
            lhs_sequence=first["cases"][0]["lhs_sequence"],
        )
        module.authenticate_report(
            raw_path, case, first["source_commit"], physical_cores
        )

        run([*authenticated_command, "--resume"])
        resumed = json.loads(
            (authenticated_output / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert resumed["cases"][0]["state"] == "reused"
        assert resumed["cases"][0]["sha256"] == first["cases"][0]["sha256"]

        changed_order = run(
            [
                *authenticated_command,
                "--case-order",
                "stable-reverse",
                "--resume",
            ],
            expected=2,
        )
        assert "resume identity mismatch for plan_sha256" in changed_order.stderr

        tampered_output = copy.deepcopy(raw_document)
        tampered_output["results"][0]["timed_final_output_authenticated"] = False
        tampered_path = temporary_path / "tampered output.json"
        tampered_path.write_text(
            json.dumps(tampered_output), encoding="utf-8"
        )
        try:
            module.authenticate_report(
                tampered_path, case, first["source_commit"], physical_cores
            )
        except ValueError as error:
            assert "unauthenticated" in str(error)
        else:
            raise AssertionError("unauthenticated final output was accepted")

        substituted = copy.deepcopy(raw_document)
        substituted["results"][0]["planner_mode"] = "forced"
        substituted["results"][0]["requested_variant"] = module.PACKED_AVX2
        substituted["configuration"]["requested_variant"] = module.PACKED_AVX2
        forced_case = module.ParityCase(
            case.partition,
            case.family,
            case.shape,
            module.PACKED_AVX2,
            case.threads,
            "complete-hot",
        )
        substituted["results"][0]["selected_variant"] = (
            "cpu.reference.f32.v1"
        )
        substituted["configuration"]["planner_regret"] = False
        substituted["results"][0]["planner_regret"]["requested"] = False
        substituted_path = temporary_path / "substituted variant.json"
        substituted_path.write_text(
            json.dumps(substituted), encoding="utf-8"
        )
        try:
            module.authenticate_report(
                substituted_path,
                forced_case,
                first["source_commit"],
                physical_cores,
            )
        except ValueError as error:
            assert "silently substituted" in str(error)
        else:
            raise AssertionError("forced variant substitution was accepted")

        raw_path.write_text(
            raw_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        bad_digest = run([*authenticated_command, "--resume"], expected=2)
        assert "resume raw-file digest mismatch" in bad_digest.stderr

        nonempty = temporary_path / "nonempty"
        nonempty.mkdir()
        (nonempty / "unexpected.txt").write_text("x", encoding="utf-8")
        rejected_nonempty = run(
            [
                sys.executable,
                str(runner),
                "--bench",
                str(bench),
                "--output-dir",
                str(nonempty),
                "--physical-cores",
                str(physical_cores),
                "--dry-run",
            ],
            expected=2,
        )
        assert "output directory is not empty" in rejected_nonempty.stderr

        nested_git = temporary_path / "unrelated git worktree"
        nested_git.mkdir()
        run(["git", "-C", str(nested_git), "init", "--quiet"])
        rejected_git_output = run(
            [
                sys.executable,
                str(runner),
                "--bench",
                str(bench),
                "--output-dir",
                str(nested_git / "raw"),
                "--physical-cores",
                str(physical_cores),
                "--dry-run",
            ],
            expected=2,
        )
        assert "outside every Git worktree" in rejected_git_output.stderr

    rejected_source_output = run(
        [
            sys.executable,
            str(runner),
            "--bench",
            str(bench),
            "--output-dir",
            str(repository / "benchmark_reports" / "forbidden parity raw"),
            "--physical-cores",
            "12",
            "--dry-run",
        ],
        expected=2,
    )
    assert "outside the source repository" in rejected_source_output.stderr

    bad_physical = run(
        [
            sys.executable,
            str(runner),
            "--bench",
            str(bench),
            "--output-dir",
            str(pathlib.Path(tempfile.gettempdir()) / "unused parity raw"),
            "--physical-cores",
            "1",
            "--dry-run",
        ],
        expected=2,
    )
    assert "physical core count must be at least two" in bad_physical.stderr

    m, n, k = (31, 33, 35)
    exact_rejection = (
        f"matcore-bench: variant planning failed for {m}x{n}x{k}: "
        "parallel candidate requires at least two disjoint output tasks and workers"
    )
    rejection_case = module.ParityCase(
        "holdout",
        "tail-heavy",
        (m, n, k),
        module.PARALLEL_AVX2,
        4,
        "complete-hot",
    )
    assert (
        module.expected_legality_rejection(rejection_case, exact_rejection)
        == "parallel-output-tile-count"
    )
    assert (
        module.expected_legality_rejection(
            rejection_case, exact_rejection + " (unexpected suffix)"
        )
        is None
    )
    assert (
        module.expected_legality_rejection(
            rejection_case,
            f"matcore-bench: variant planning failed for {m}x{n}x{k}: "
            "arbitrary failure",
        )
        is None
    )

    print("matcore native BLAS parity runner contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
