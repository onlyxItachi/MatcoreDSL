#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import subprocess
import tempfile


def run(command: list[str], *, expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != expected:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected}: {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def require_report_shape(report: dict, schema: dict) -> None:
    required = schema["required"]
    assert all(field in report for field in required)
    assert report["schema"] == "matcore.benchmark.cpu.gemm"
    assert report["version"] == 2
    assert report["operation"] == "matcore.gemm"
    assert report["dtype"] == report["accumulation_dtype"] == "f32"
    assert report["layout"] == "row-major-contiguous"
    environment_required = schema["$defs"]["environment"]["required"]
    configuration_required = schema["$defs"]["configuration"]["required"]
    result_required = schema["$defs"]["result"]["required"]
    assert all(field in report["environment"] for field in environment_required)
    assert all(field in report["configuration"] for field in configuration_required)
    assert report["results"]
    assert all(field in report["results"][0] for field in result_required)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    executable = pathlib.Path(args.bench).resolve()
    schema = json.loads(pathlib.Path(args.schema).read_text(encoding="utf-8"))

    variants = run([str(executable), "--list-variants"]).stdout.splitlines()
    assert variants == [
        "auto",
        "cpu.reference.f32.v1",
        "cpu.tiled.f32.v1",
        "cpu.compiler-vectorized.avx2-fma.f32.v1",
        "cpu.external.openblas.f32.v1",
        "cpu.native-packed.avx2-fma.f32.v1",
        "cpu.native-packed.avx512-fma.f32.v1",
        "cpu.native-parallel.avx2-fma.f32.v1",
        "cpu.native-parallel.avx512-fma.f32.v1",
    ]

    # These are CLI/schema/correctness smoke tests rather than performance
    # acceptance runs. A 1 us floor prevents sanitizer first-probe overhead
    # from making later correct samples look like a timer-contract failure.
    with tempfile.TemporaryDirectory(prefix="matcore bench cli ") as temporary:
        output = pathlib.Path(temporary) / "result with spaces.json"
        run(
            [
                str(executable),
                "--m",
                "33",
                "--n",
                "35",
                "--k",
                "37",
                "--variant",
                "cpu.reference.f32.v1",
                "--warmup",
                "0",
                "--iterations",
                "3",
                "--timer-floor-us",
                "1",
                "--alignment",
                "4",
                "--guard",
                "--json-out",
                str(output),
            ]
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        require_report_shape(report, schema)
        result = report["results"][0]
        assert (result["m"], result["n"], result["k"]) == (33, 35, 37)
        assert result["selected_variant"] == "cpu.reference.f32.v1"
        assert result["correctness"] is True
        assert result["oracle_mode"] == "full-double"
        assert result["timing_valid"] is True
        assert result["aggregate_repetitions"] >= 1
        assert result["minimum_seconds"] > 0
        assert result["median_seconds"] > 0
        assert result["p95_seconds"] >= result["median_seconds"]
        assert math.isfinite(result["gflops"]) and result["gflops"] > 0
        assert report["configuration"]["alignment_bytes"] == 4
        assert report["configuration"]["compare_one_thread"] is False
        assert report["configuration"]["planner_regret"] is False
        assert report["configuration"]["smt_policy"] == "physical-cores-only"
        assert report["configuration"]["affinity_policy"] == "none"
        assert report["environment"]["capability_record_version"] == 2
        assert report["environment"]["topology_record_version"] == 1
        assert report["environment"]["execution_context_workers"] >= 1
        assert report["environment"]["execution_context_workers_started"] >= 1
        assert report["environment"]["available_processors"] >= 1
        assert report["environment"]["worker_affinity_applied"] is False
        assert "not pinned" in report["environment"]["worker_affinity_source"]
        assert "benchmark-process numerical self-test" in report["environment"][
            "capability_runtime_validation_source"
        ]
        assert result["planner_version"] == 3
        assert result["shared_workspace_bytes"] == 0
        assert result["per_worker_workspace_bytes"] == 0
        assert result["persistent_execution_context"] is False
        assert result["smt_policy"] == "physical-cores-only"
        assert result["affinity_policy"] == "none"
        assert result["worker_affinity_applied"] is False
        assert "inherited process mask" in result["affinity_diagnostic"]
        assert result["scaling"]["requested"] is False
        assert result["planner_regret"]["requested"] is False
        assert result["complete_implementation_comparison"] is True
        assert "complete implementation call" in result["timing_scope"]

        allocation_output = pathlib.Path(temporary) / "allocation.json"
        run(
            [
                str(executable),
                "--m",
                "16",
                "--n",
                "16",
                "--k",
                "16",
                "--include-allocation",
                "--warmup",
                "0",
                "--iterations",
                "2",
                "--timer-floor-us",
                "1",
                "--json-out",
                str(allocation_output),
            ]
        )
        allocation_report = json.loads(allocation_output.read_text(encoding="utf-8"))
        assert allocation_report["configuration"]["allocation_mode"] == "include-allocation"
        assert allocation_report["results"][0]["correctness"] is True

        cold_output = pathlib.Path(temporary) / "cold.json"
        run(
            [
                str(executable),
                "--m",
                "1",
                "--n",
                "1",
                "--k",
                "1",
                "--cold-cache",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--timer-floor-us",
                "1000000",
                "--json-out",
                str(cold_output),
            ]
        )
        cold_report = json.loads(cold_output.read_text(encoding="utf-8"))
        assert cold_report["results"][0]["correctness"] is True
        assert cold_report["results"][0]["timing_valid"] is False
        assert "timer floor" in cold_report["results"][0]["timing_rejection_reason"]

        compute_output = pathlib.Path(temporary) / "packed compute only.json"
        packed_compute = subprocess.run(
            [
                str(executable),
                "--m",
                "127",
                "--n",
                "129",
                "--k",
                "131",
                "--variant",
                "cpu.native-packed.avx2-fma.f32.v1",
                "--exclude-packing",
                "--reuse-workspace",
                "--warmup",
                "0",
                "--iterations",
                "2",
                "--timer-floor-us",
                "1",
                "--guard",
                "--json-out",
                str(compute_output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if packed_compute.returncode == 0:
            compute_report = json.loads(compute_output.read_text(encoding="utf-8"))
            compute_result = compute_report["results"][0]
            assert compute_report["configuration"]["packing_mode"] == "exclude-packing"
            assert compute_result["packing_mode"] == "exclude-packing"
            assert compute_result["selected_variant"] == "cpu.native-packed.avx2-fma.f32.v1"
            assert compute_result["correctness"] is True
            assert compute_result["timing_valid"] is True
            assert compute_result["packing_required"] is True
            assert compute_result["workspace_bytes"] > 0
            assert compute_result["complete_implementation_comparison"] is False
            assert "packed-compute-only" in compute_result["timing_scope"]
            assert "A and B packing prepared before timing" in compute_result["plan_diagnostic"]
            assert "comparison=diagnostic-only" in packed_compute.stdout
            assert "timing_scope=\"packed-compute-only:" in packed_compute.stdout

            include_output = pathlib.Path(temporary) / "packed include.json"
            run(
                [
                    str(executable), "--m", "127", "--n", "129", "--k", "131",
                    "--variant", "cpu.native-packed.avx2-fma.f32.v1",
                    "--include-packing", "--warmup", "0", "--iterations", "1",
                    "--timer-floor-us", "1", "--guard", "--json-out",
                    str(include_output),
                ]
            )
            include_result = json.loads(include_output.read_text(encoding="utf-8"))["results"][0]
            assert include_result["packing_mode"] == "include-packing"
            assert "transient A and B packing" in include_result["timing_scope"]

            prepacked_output = pathlib.Path(temporary) / "packed b.json"
            run(
                [
                    str(executable), "--m", "127", "--n", "129", "--k", "131",
                    "--variant", "cpu.native-packed.avx2-fma.f32.v1",
                    "--prepack-b", "--warmup", "0", "--iterations", "1",
                    "--timer-floor-us", "1", "--guard", "--json-out",
                    str(prepacked_output),
                ]
            )
            prepacked_result = json.loads(prepacked_output.read_text(encoding="utf-8"))["results"][0]
            assert prepacked_result["packing_mode"] == "prepacked-b"
            assert "transient A packing" in prepacked_result["timing_scope"]
        else:
            assert "AVX2" in packed_compute.stderr or "unavailable" in packed_compute.stderr

        scaling_output = pathlib.Path(temporary) / "scaling.json"
        run(
            [
                str(executable), "--m", "64", "--n", "64", "--k", "64",
                "--variant", "cpu.reference.f32.v1", "--compare-one-thread",
                "--warmup", "0", "--iterations", "1", "--timer-floor-us", "1",
                "--guard", "--json-out", str(scaling_output),
            ]
        )
        scaling = json.loads(scaling_output.read_text(encoding="utf-8"))["results"][0]["scaling"]
        assert scaling["requested"] is True
        assert scaling["valid"] is True
        assert scaling["baseline_variant"] == "cpu.reference.f32.v1"
        assert scaling["speedup_over_one_thread"] == 1.0
        assert scaling["parallel_efficiency"] == 1.0

        regret_output = pathlib.Path(temporary) / "regret.json"
        run(
            [
                str(executable), "--m", "64", "--n", "64", "--k", "64",
                "--planner-regret", "--warmup", "0", "--iterations", "1",
                "--timer-floor-us", "1", "--guard", "--json-out",
                str(regret_output),
            ]
        )
        regret = json.loads(regret_output.read_text(encoding="utf-8"))["results"][0]["planner_regret"]
        assert regret["requested"] is True
        assert regret["valid"] is True
        assert len(regret["candidates"]) == 8
        assert regret["fastest_legal_variant"]
        assert regret["regret"] >= 1.0

        parallel_output = pathlib.Path(temporary) / "parallel avx2.json"
        parallel = subprocess.run(
            [
                str(executable), "--m", "256", "--n", "128", "--k", "128",
                "--variant", "cpu.native-parallel.avx2-fma.f32.v1",
                "--threads", "2", "--compare-one-thread", "--warmup", "0",
                "--iterations", "1", "--timer-floor-us", "1", "--guard",
                "--json-out", str(parallel_output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if parallel.returncode == 0:
            parallel_result = json.loads(parallel_output.read_text(encoding="utf-8"))["results"][0]
            assert parallel_result["actual_threads"] == 2
            assert parallel_result["persistent_execution_context"] is True
            assert parallel_result["shared_workspace_bytes"] > 0
            assert parallel_result["per_worker_workspace_bytes"] > 0
            assert parallel_result["correctness"] is True
            assert parallel_result["scaling"]["valid"] is True
            parallel_report = json.loads(parallel_output.read_text(encoding="utf-8"))
            assert parallel_report["environment"]["execution_context_submissions"] >= 2
        else:
            assert (
                "runtime-validated" in parallel.stderr
                or "topology" in parallel.stderr
                or "unavailable" in parallel.stderr
            )

        forced_variants = [
            ("cpu.reference.f32.v1", (64, 64, 64), 1),
            ("cpu.tiled.f32.v1", (64, 64, 64), 1),
            ("cpu.compiler-vectorized.avx2-fma.f32.v1", (64, 64, 64), 1),
            ("cpu.external.openblas.f32.v1", (64, 64, 64), 1),
            ("cpu.native-packed.avx2-fma.f32.v1", (64, 64, 64), 1),
            ("cpu.native-packed.avx512-fma.f32.v1", (64, 64, 64), 1),
            ("cpu.native-parallel.avx2-fma.f32.v1", (256, 128, 128), 2),
            ("cpu.native-parallel.avx512-fma.f32.v1", (256, 128, 128), 2),
        ]
        for variant, (m, n, k), threads in forced_variants:
            forced_output = pathlib.Path(temporary) / f"forced-{variant}.json"
            forced = subprocess.run(
                [
                    str(executable), "--m", str(m), "--n", str(n), "--k", str(k),
                    "--variant", variant, "--threads", str(threads), "--warmup", "0",
                    "--iterations", "1", "--timer-floor-us", "1", "--guard",
                    "--json-out", str(forced_output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if forced.returncode == 0:
                forced_report = json.loads(forced_output.read_text(encoding="utf-8"))
                forced_result = forced_report["results"][0]
                assert forced_result["selected_variant"] == variant
                assert forced_result["correctness"] is True
                assert forced_result["planner_version"] == 3
                if "native-parallel" in variant:
                    assert forced_result["persistent_execution_context"] is True
                    assert forced_report["environment"]["execution_context_submissions"] >= 1
            else:
                assert (
                    "unavailable" in forced.stderr
                    or "not runtime-validated" in forced.stderr
                    or "topology" in forced.stderr
                    or "not linked" in forced.stderr
                ), (variant, forced.stderr)
    invalid_modes = run(
        [
            str(executable),
            "--m",
            "2",
            "--n",
            "2",
            "--k",
            "2",
            "--prepack-b",
            "--include-allocation",
        ],
        expected=1,
    )
    assert "prepacked-B mode requires reusable workspace" in invalid_modes.stderr

    invalid_compute_allocation = run(
        [
            str(executable), "--m", "2", "--n", "2", "--k", "2",
            "--variant", "cpu.native-packed.avx2-fma.f32.v1",
            "--exclude-packing", "--include-allocation",
        ],
        expected=1,
    )
    assert "compute diagnostics require reusable workspace" in invalid_compute_allocation.stderr

    misleading_exclusion = run(
        [
            str(executable), "--m", "16", "--n", "16", "--k", "16",
            "--variant", "cpu.reference.f32.v1", "--exclude-packing",
        ],
        expected=1,
    )
    assert "diagnostic implemented only" in misleading_exclusion.stderr

    misleading_openblas_exclusion = subprocess.run(
        [
            str(executable), "--m", "16", "--n", "16", "--k", "16",
            "--variant", "cpu.external.openblas.f32.v1", "--exclude-packing",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert misleading_openblas_exclusion.returncode != 0
    assert (
        "diagnostic implemented only" in misleading_openblas_exclusion.stderr
        or "OpenBLAS CBLAS adapter is not linked"
        in misleading_openblas_exclusion.stderr
    )

    unavailable_prepack = run(
        [
            str(executable),
            "--m",
            "2",
            "--n",
            "2",
            "--k",
            "2",
            "--prepack-b",
        ],
        expected=1,
    )
    assert "does not support prepacked-B" in unavailable_prepack.stderr

    oversized = run(
        [
            str(executable),
            "--m",
            "2048",
            "--n",
            "2048",
            "--k",
            "2048",
            "--max-memory-mib",
            "1",
        ],
        expected=1,
    )
    assert "exceeds --max-memory-mib" in oversized.stderr

    bad_variant = run(
        [
            str(executable),
            "--m",
            "2",
            "--n",
            "2",
            "--k",
            "2",
            "--variant",
            "cpu.fake.f32.v1",
        ],
        expected=1,
    )
    assert "not registered" in bad_variant.stderr

    explicit_single = run(
        [
            str(executable),
            "--m",
            "2",
            "--n",
            "2",
            "--k",
            "2",
            "--threads",
            "2",
            "--variant",
            "cpu.reference.f32.v1",
        ],
    )
    assert "variant=cpu.reference.f32.v1" in explicit_single.stdout
    assert "threads=1" in explicit_single.stdout

    invalid_regret_variant = run(
        [
            str(executable), "--m", "2", "--n", "2", "--k", "2",
            "--variant", "cpu.reference.f32.v1", "--planner-regret",
        ],
        expected=1,
    )
    assert "requires --variant auto" in invalid_regret_variant.stderr

    unsupported_affinity = run(
        [
            str(executable), "--m", "64", "--n", "64", "--k", "64",
            "--threads", "2", "--affinity", "compact",
        ],
        expected=1,
    )
    assert "worker-affinity policy is not implemented" in unsupported_affinity.stderr

    with tempfile.TemporaryDirectory(prefix="matcore smt metadata ") as temporary:
        smt_output = pathlib.Path(temporary) / "smt.json"
        run(
            [
                str(executable), "--m", "64", "--n", "64", "--k", "64",
                "--variant", "cpu.reference.f32.v1", "--allow-smt",
                "--warmup", "0", "--iterations", "1", "--timer-floor-us", "1",
                "--json-out", str(smt_output),
            ]
        )
        smt_report = json.loads(smt_output.read_text(encoding="utf-8"))
        assert smt_report["configuration"]["smt_policy"] == "allow-smt"
        assert smt_report["results"][0]["smt_policy"] == "allow-smt"

    optional_openblas = subprocess.run(
        [
            str(executable),
            "--m",
            "2",
            "--n",
            "2",
            "--k",
            "2",
            "--variant",
            "cpu.external.openblas.f32.v1",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--timer-floor-us",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if optional_openblas.returncode == 0:
        assert "variant=cpu.external.openblas.f32.v1" in optional_openblas.stdout
        assert "correctness=pass" in optional_openblas.stdout
        capped_openblas_threads = run(
            [
                str(executable),
                "--m",
                "2",
                "--n",
                "2",
                "--k",
                "2",
                "--threads",
                "2147483647",
                "--variant",
                "cpu.external.openblas.f32.v1",
            ],
        )
        assert "threads=2147483647" not in capped_openblas_threads.stdout
        assert "variant=cpu.external.openblas.f32.v1" in capped_openblas_threads.stdout
    else:
        assert "OpenBLAS CBLAS adapter is not linked" in optional_openblas.stderr

    print("matcore-bench CLI/JSON contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
