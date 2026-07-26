#!/usr/bin/env python3

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
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
    assert report["version"] == schema["properties"]["version"]["const"]
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


def require_normalized_samples(result: dict, measured_iterations: int) -> None:
    samples = result["normalized_samples_seconds"]
    assert len(samples) == measured_iterations
    assert all(math.isfinite(sample) and sample > 0 for sample in samples)
    ordered = sorted(samples)
    assert result["minimum_seconds"] == ordered[0]
    assert result["median_seconds"] == ordered[len(ordered) // 2]
    p95_index = math.ceil(len(ordered) * 0.95) - 1
    assert result["p95_seconds"] == ordered[p95_index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    executable = pathlib.Path(args.bench).resolve()
    schema_path = pathlib.Path(args.schema)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["version"]["const"] == 6
    help_text = run([str(executable), "--help"]).stdout
    assert "write schema-v6 JSON" in help_text
    assert "schema-v4" not in help_text
    v5_schema = json.loads(
        schema_path.with_name("matcore-bench-v5.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert v5_schema["properties"]["version"]["const"] == 5
    assert "normalized_samples_seconds" not in v5_schema["$defs"]["result"][
        "required"
    ]

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
        assert re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}",
            report["environment"]["source_commit"],
        )
        assert report["environment"]["source_worktree_dirty"] is False
        assert report["environment"]["source_provenance_state"] == "clean"
        assert report["environment"]["source_provenance_origin"] in (
            "git-worktree",
            "explicit-override",
        )
        result = report["results"][0]
        assert (result["m"], result["n"], result["k"]) == (33, 35, 37)
        assert result["selected_variant"] == "cpu.reference.f32.v1"
        assert result["correctness"] is True
        assert result["timed_final_output_authenticated"] is True
        assert result["untimed_validation_executions_checked"] >= 3
        assert "separate untimed validation phase" in result["untimed_validation_scope"]
        assert result["timing_aggregation_boundary"] == (
            "one-clock-pair-per-aggregate-repetition-block"
        )
        assert result["untimed_validation_placement"] == "after-timing"
        assert result["oracle_mode"] == "full-double"
        assert result["timing_valid"] is True
        assert result["aggregate_repetitions"] >= 1
        assert result["minimum_seconds"] > 0
        assert result["median_seconds"] > 0
        assert result["p95_seconds"] >= result["median_seconds"]
        require_normalized_samples(result, measured_iterations=3)
        preparation = result["prepacked_b_preparation"]
        assert preparation["requested"] is False
        assert preparation["measured"] is False
        assert preparation["authenticated"] is False
        assert preparation["preparation_calls"] == 0
        assert preparation["input_state"] == "not-requested"
        assert preparation["output_state"] == "not-requested"
        assert preparation["preparation_seconds"] == 0
        assert preparation["amortization_executions"] == 0
        assert preparation["amortized_total_valid"] is False
        assert math.isfinite(result["gflops"]) and result["gflops"] > 0
        assert report["configuration"]["alignment_bytes"] == 4
        assert report["configuration"]["lhs_sequence_length"] == 1
        assert report["configuration"]["compare_one_thread"] is False
        assert report["configuration"]["planner_regret"] is False
        assert report["configuration"]["smt_policy"] == "physical-cores-only"
        assert report["configuration"]["affinity_policy"] == "none"
        assert report["environment"]["capability_record_version"] == 2
        assert report["environment"]["topology_record_version"] == 1
        assert report["environment"]["execution_context_workers"] >= 1
        assert report["environment"]["execution_context_workers_started"] >= 1
        assert report["environment"]["available_processors"] >= 1
        assert report["environment"]["worker_affinity_applied"] is True
        assert report["environment"]["worker_affinity_user_requested"] is False
        assert report["environment"]["worker_affinity_policy_induced"] is True
        assert "induced by physical-cores-only SMT policy" in report["environment"][
            "worker_affinity_source"
        ]
        assert "restricted to inherited process" in report["environment"][
            "worker_affinity_source"
        ]
        assert "benchmark-process numerical self-test" in report["environment"][
            "capability_runtime_validation_source"
        ]
        assert result["planner_version"] == 3
        assert result["shared_workspace_bytes"] == 0
        assert result["per_worker_workspace_bytes"] == 0
        assert result["persistent_execution_context"] is True
        assert result["smt_policy"] == "physical-cores-only"
        assert result["affinity_policy"] == "none"
        assert result["worker_affinity_applied"] is True
        assert result["worker_affinity_user_requested"] is False
        assert result["worker_affinity_policy_induced"] is True
        assert "one-logical-CPU-per-core" in result["affinity_diagnostic"]
        assert "user_requested=false" in result["affinity_diagnostic"]
        if sys.platform.startswith("linux") and report["environment"]["physical_cores"] >= 2:
            assert "benchmark caller scheduler affinity applied" in result[
                "affinity_diagnostic"
            ]
            assert "caller_cpu_id=" in result["affinity_diagnostic"]
            assert "dedicated_physical_core=true" in result["affinity_diagnostic"]
            assert "benchmark caller scheduler affinity applied" in report[
                "environment"
            ]["worker_affinity_source"]
            assert "benchmark_caller_affinity=" in report["environment"][
                "topology_record"
            ]
        assert "pinned persistent worker 0" in result["timing_scope"]
        assert result["scaling"]["requested"] is False
        assert result["planner_regret"]["requested"] is False
        assert result["complete_implementation_comparison"] is True
        assert "complete implementation call" in result["timing_scope"]

        sequence_output = pathlib.Path(temporary) / "fixed B sequence.json"
        run(
            [
                str(executable),
                "--m",
                "2",
                "--n",
                "3",
                "--k",
                "2",
                "--variant",
                "cpu.reference.f32.v1",
                "--lhs-sequence",
                "4",
                "--warmup",
                "0",
                "--iterations",
                "2",
                "--timer-floor-us",
                "1",
                "--guard",
                "--json-out",
                str(sequence_output),
            ]
        )
        sequence_report = json.loads(
            sequence_output.read_text(encoding="utf-8")
        )
        require_report_shape(sequence_report, schema)
        sequence_result = sequence_report["results"][0]
        assert sequence_report["configuration"]["lhs_sequence_length"] == 4
        assert sequence_result["aggregate_repetitions"] >= 4
        assert sequence_result["aggregate_repetitions"] % 4 == 0
        require_normalized_samples(sequence_result, measured_iterations=2)
        assert sequence_result["untimed_validation_executions_checked"] >= 8
        assert "distinct-left-inputs=4" in sequence_result[
            "untimed_validation_scope"
        ]

        if report["environment"]["physical_cores"] >= 2:
            for policy in ("compact", "scatter", "local-first"):
                affinity_output = pathlib.Path(temporary) / f"affinity-{policy}.json"
                run(
                    [
                        str(executable), "--m", "256", "--n", "128", "--k", "128",
                        "--variant", "cpu.native-parallel.avx2-fma.f32.v1",
                        "--threads", "2",
                        "--affinity", policy, "--warmup", "0", "--iterations", "1",
                        "--timer-floor-us", "1", "--json-out", str(affinity_output),
                    ]
                )
                affinity_report = json.loads(
                    affinity_output.read_text(encoding="utf-8")
                )
                affinity_result = affinity_report["results"][0]
                assert affinity_result["worker_affinity_applied"] is True
                assert affinity_result["worker_affinity_user_requested"] is True
                assert affinity_result["worker_affinity_policy_induced"] is False
                assert affinity_result["affinity_policy"] == policy
                assert "strict per-worker scheduler affinity complete" in affinity_result[
                    "affinity_diagnostic"
                ]
                assert "cpu_ids=[" in affinity_result["affinity_diagnostic"]
                encoded_ids = re.search(
                    r"cpu_ids=\[([0-9,]+)\]", affinity_result["affinity_diagnostic"]
                )
                assert encoded_ids is not None
                worker_ids = {int(value) for value in encoded_ids.group(1).split(",")}
                if sys.platform.startswith("linux"):
                    assert len(worker_ids) == 2
                    assert worker_ids <= set(os.sched_getaffinity(0))
                assert "numa_memory_placement=false" in affinity_result[
                    "affinity_diagnostic"
                ]
                caller_id = re.search(
                    r"caller_cpu_id=([0-9]+)", affinity_result["affinity_diagnostic"]
                )
                if sys.platform.startswith("linux") and len(os.sched_getaffinity(0)) >= 3:
                    assert caller_id is not None
                    assert int(caller_id.group(1)) not in worker_ids
                    assert "benchmark caller scheduler affinity applied" in (
                        affinity_result["affinity_diagnostic"]
                    )
                assert affinity_report["environment"]["worker_affinity_applied"] is True
                assert affinity_report["environment"][
                    "execution_context_workers_started"
                ] >= 2

        affinity_serial_variants = [
            "cpu.reference.f32.v1",
            "cpu.tiled.f32.v1",
            "cpu.compiler-vectorized.avx2-fma.f32.v1",
            "cpu.external.openblas.f32.v1",
            "cpu.native-packed.avx2-fma.f32.v1",
            "cpu.native-packed.avx512-fma.f32.v1",
        ]
        for variant in affinity_serial_variants:
            serial_output = pathlib.Path(temporary) / f"affinity-serial-{variant}.json"
            serial = subprocess.run(
                [
                    str(executable), "--m", "64", "--n", "64", "--k", "64",
                    "--variant", variant, "--threads", "1", "--affinity", "compact",
                    "--warmup", "0", "--iterations", "1", "--timer-floor-us", "1",
                    "--json-out", str(serial_output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if serial.returncode == 0:
                serial_report = json.loads(serial_output.read_text(encoding="utf-8"))
                serial_result = serial_report["results"][0]
                assert serial_result["selected_variant"] == variant
                assert serial_result["correctness"] is True
                assert serial_result["worker_affinity_applied"] is True
                assert serial_result["worker_affinity_user_requested"] is True
                assert serial_result["worker_affinity_policy_induced"] is False
                assert "pinned persistent worker 0" in serial_result["timing_scope"]
                assert serial_report["environment"]["execution_context_submissions"] >= 1
            else:
                assert (
                    "not linked" in serial.stderr
                    or "not runtime-validated" in serial.stderr
                    or "unavailable" in serial.stderr
                ), (variant, serial.stderr)

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
                    "--prepack-b", "--lhs-sequence", "4",
                    "--warmup", "0", "--iterations", "3",
                    "--timer-floor-us", "1", "--guard", "--json-out",
                    str(prepacked_output),
                ]
            )
            prepacked_result = json.loads(prepacked_output.read_text(encoding="utf-8"))["results"][0]
            assert prepacked_result["packing_mode"] == "prepacked-b"
            assert "transient A packing" in prepacked_result["timing_scope"]
            require_normalized_samples(
                prepacked_result, measured_iterations=3
            )
            preparation = prepacked_result["prepacked_b_preparation"]
            assert preparation["requested"] is True
            assert preparation["measured"] is True
            assert preparation["authenticated"] is True
            assert preparation["preparation_calls"] == 1
            assert preparation["input_state"] == (
                "caller-storage-allocated-unprepared"
            )
            assert preparation["output_state"] == "prepared-authenticated"
            assert preparation["preparation_seconds"] > 0
            assert preparation["amortization_executions"] == 4
            assert preparation["amortized_total_valid"] is True
            assert math.isclose(
                preparation["steady_state_sequence_seconds"],
                prepacked_result["median_seconds"] * 4,
                rel_tol=1e-15,
            )
            assert math.isclose(
                preparation["amortized_total_sequence_seconds"],
                preparation["preparation_seconds"]
                + preparation["steady_state_sequence_seconds"],
                rel_tol=1e-15,
            )
            assert math.isclose(
                preparation["amortized_per_execution_seconds"],
                preparation["amortized_total_sequence_seconds"] / 4,
                rel_tol=1e-15,
            )
            assert "exactly one runner.prepare(prepack_b=true) call" in (
                preparation["timing_scope"]
            )
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
        assert len(scaling["one_thread_normalized_samples_seconds"]) == 1
        assert scaling["one_thread_normalized_samples_seconds"][0] == (
            scaling["one_thread_median_seconds"]
        )
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
        regret_result = json.loads(
            regret_output.read_text(encoding="utf-8")
        )["results"][0]
        regret = regret_result["planner_regret"]
        assert regret["requested"] is True
        assert regret["valid"] is True
        assert len(regret["candidates"]) == 8
        assert regret["fastest_legal_variant"]
        assert regret["regret"] >= 1.0
        assert regret["aggregation_method"] == (
            "arithmetic-mean-of-forward-and-reverse-pass-medians"
        )
        assert "equal-weight arithmetic mean" in regret["reason"]
        selected = next(
            candidate
            for candidate in regret["candidates"]
            if candidate["variant"] == regret_result["selected_variant"]
        )
        assert regret["selected_balanced_estimate_seconds"] == (
            selected["balanced_estimate_seconds"]
        )
        assert "selected_median_seconds" not in regret
        assert "fastest_legal_median_seconds" not in regret
        for candidate in regret["candidates"]:
            if candidate["legal"] and candidate["complete_implementation_comparison"]:
                assert candidate["selected_variant"] == candidate["variant"]
                assert candidate["planner_version"] == 3
                assert candidate["timing_scope"]
                assert candidate["actual_threads"] >= 1
                assert candidate["workspace_alignment"] >= 1
                assert candidate["smt_policy"] in (
                    "physical-cores-only",
                    "allow-smt",
                )
                assert candidate["affinity_policy"] in (
                    "none",
                    "compact",
                    "scatter",
                    "local-first",
                )
                assert candidate["timing_valid"] is True
                assert candidate["correctness"] is True
                assert candidate["plan_authenticated"] is True
                assert candidate[
                    "forward_pass_untimed_validation_executions_checked"
                ] >= 1
                assert candidate[
                    "reverse_pass_untimed_validation_executions_checked"
                ] >= 1
                assert candidate[
                    "forward_pass_untimed_validation_placement"
                ] == "after-timing"
                assert candidate[
                    "reverse_pass_untimed_validation_placement"
                ] == "before-timing"
                assert candidate["forward_pass_median_seconds"] > 0
                assert candidate["reverse_pass_median_seconds"] > 0
                assert len(
                    candidate["forward_pass_normalized_samples_seconds"]
                ) == 1
                assert len(
                    candidate["reverse_pass_normalized_samples_seconds"]
                ) == 1
                assert candidate[
                    "forward_pass_normalized_samples_seconds"
                ][0] == candidate["forward_pass_median_seconds"]
                assert candidate[
                    "reverse_pass_normalized_samples_seconds"
                ][0] == candidate["reverse_pass_median_seconds"]
                assert math.isclose(
                    candidate["balanced_estimate_seconds"],
                    (
                        candidate["forward_pass_median_seconds"]
                        + candidate["reverse_pass_median_seconds"]
                    )
                    / 2.0,
                    rel_tol=1.0e-15,
                )
                assert "median_seconds" not in candidate
                assert "forward and reverse stable-registry-pass medians" in (
                    candidate["measurement_reason"]
                )

        parallel_output = pathlib.Path(temporary) / "parallel avx2.json"
        parallel = subprocess.run(
            [
                str(executable), "--m", "256", "--n", "128", "--k", "128",
                "--variant", "cpu.native-parallel.avx2-fma.f32.v1",
                "--threads", "2", "--affinity", "compact",
                "--compare-one-thread", "--warmup", "0",
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
            assert parallel_result["scaling"]["baseline_variant"] == (
                "cpu.native-packed.avx2-fma.f32.v1"
            )
            assert parallel_result["worker_affinity_user_requested"] is True
            assert parallel_result["worker_affinity_policy_induced"] is False
            parallel_report = json.loads(parallel_output.read_text(encoding="utf-8"))
            assert parallel_report["environment"]["execution_context_submissions"] >= 2
        else:
            assert (
                "runtime-validated" in parallel.stderr
                or "topology" in parallel.stderr
                or "unavailable" in parallel.stderr
            )

        parallel_avx512_output = pathlib.Path(temporary) / "parallel avx512.json"
        parallel_avx512 = subprocess.run(
            [
                str(executable), "--m", "256", "--n", "128", "--k", "128",
                "--variant", "cpu.native-parallel.avx512-fma.f32.v1",
                "--threads", "2", "--affinity", "compact",
                "--compare-one-thread", "--warmup", "0", "--iterations", "1",
                "--timer-floor-us", "1", "--guard", "--json-out",
                str(parallel_avx512_output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if parallel_avx512.returncode == 0:
            avx512_result = json.loads(
                parallel_avx512_output.read_text(encoding="utf-8")
            )["results"][0]
            assert avx512_result["correctness"] is True
            assert avx512_result["scaling"]["valid"] is True
            assert avx512_result["scaling"]["baseline_variant"] == (
                "cpu.native-packed.avx512-fma.f32.v1"
            )
            assert avx512_result["worker_affinity_applied"] is True
            assert avx512_result["worker_affinity_user_requested"] is True
        else:
            assert (
                "runtime-validated" in parallel_avx512.stderr
                or "AVX-512" in parallel_avx512.stderr
                or "unavailable" in parallel_avx512.stderr
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

        if sys.platform.startswith("linux") and shutil.which("taskset"):
            allowed = sorted(os.sched_getaffinity(0))
            assert allowed
            cpu = str(allowed[0])
            constrained_output = pathlib.Path(temporary) / "taskset-one-cpu.json"
            run(
                [
                    "taskset", "-c", cpu, str(executable), "--m", "64", "--n", "64",
                    "--k", "64", "--variant", "cpu.reference.f32.v1", "--threads", "1",
                    "--warmup", "0", "--iterations", "1", "--timer-floor-us", "1",
                    "--json-out", str(constrained_output),
                ]
            )
            constrained_report = json.loads(
                constrained_output.read_text(encoding="utf-8")
            )
            assert constrained_report["environment"]["available_processors"] == 1
            assert constrained_report["environment"]["logical_processors"] == 1
            assert constrained_report["environment"]["physical_cores"] == 1
            assert f"scheduler mask [{cpu}]" in constrained_report["environment"][
                "topology_record"
            ]
            assert "benchmark caller isolation unavailable" in constrained_report[
                "environment"
            ]["worker_affinity_source"]
            assert "no spare logical CPU" in constrained_report["results"][0][
                "affinity_diagnostic"
            ]
            assert "caller_scheduler_affinity_applied=false" in constrained_report[
                "results"
            ][0]["affinity_diagnostic"]

            constrained_parallel = subprocess.run(
                [
                    "taskset", "-c", cpu, str(executable), "--m", "256", "--n", "128",
                    "--k", "128", "--variant",
                    "cpu.native-parallel.avx2-fma.f32.v1", "--threads", "2",
                    "--warmup", "0", "--iterations", "1", "--timer-floor-us", "1",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            assert constrained_parallel.returncode != 0
            assert (
                "at least two" in constrained_parallel.stderr
                or "worker" in constrained_parallel.stderr
                or "topology" in constrained_parallel.stderr
                or "not runtime-validated" in constrained_parallel.stderr
            )
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

    invalid_compute_sequence = run(
        [
            str(executable),
            "--m",
            "2",
            "--n",
            "2",
            "--k",
            "2",
            "--exclude-packing",
            "--lhs-sequence",
            "2",
        ],
        expected=1,
    )
    assert "one prepared left input" in invalid_compute_sequence.stderr

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
        assert smt_report["results"][0]["worker_affinity_applied"] is False
        assert smt_report["results"][0][
            "worker_affinity_user_requested"
        ] is False
        assert smt_report["results"][0][
            "worker_affinity_policy_induced"
        ] is False
        assert "inherit the process mask" in smt_report["results"][0][
            "affinity_diagnostic"
        ]
        assert "benchmark caller scheduler affinity not requested" in smt_report[
            "results"
        ][0]["affinity_diagnostic"]
        assert "benchmark caller scheduler affinity not requested" in smt_report[
            "environment"
        ]["worker_affinity_source"]

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
                "--allow-smt",
                "--variant",
                "cpu.external.openblas.f32.v1",
            ],
        )
        assert "threads=2147483647" not in capped_openblas_threads.stdout
        assert "variant=cpu.external.openblas.f32.v1" in capped_openblas_threads.stdout
    else:
        assert "OpenBLAS CBLAS adapter is not linked" in optional_openblas.stderr

    provider_affinity = subprocess.run(
        [
            str(executable), "--m", "64", "--n", "64", "--k", "64",
            "--variant", "cpu.external.openblas.f32.v1", "--threads", "2",
            "--affinity", "compact",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if provider_affinity.returncode == 0:
        assert "variant=cpu.external.openblas.f32.v1" in provider_affinity.stdout
        assert "threads=1" in provider_affinity.stdout
        assert "worker_affinity_applied=true" in provider_affinity.stdout
    else:
        assert (
            "provider-thread affinity is not authenticated"
            in provider_affinity.stderr
            or "multi-thread OpenBLAS is unavailable under bound native workers"
            in provider_affinity.stderr
            or "OpenBLAS CBLAS adapter is not linked" in provider_affinity.stderr
            or "not runtime-validated" in provider_affinity.stderr
        )

    print("matcore-bench CLI/JSON contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
