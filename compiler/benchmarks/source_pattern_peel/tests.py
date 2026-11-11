#!/usr/bin/env python3
"""Synthetic P5 protocol attacks. Never build, prepare or time an executable."""
import copy
import contextlib
import difflib
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import protocol as c
import run as runner
from support import HERE, P1_HERE, ROOT, legacy, p1, p1c, peel


def common():
    return {"schema": c.SCHEMA, "version": 1, "numerics": "strict_f32",
            "timing_scope": p1c.SCOPE, "release_inside_timing": False,
            "chooser_inside_timing": True, "implementation_thread_contract": 1,
            "continuous_thread_proof": False, "seed": p1c.SEED, "input_sequence": 4}


def control():
    return {**common(), "mode": "verify-only", "status": "CORRECTNESS_PASSED",
            "controls": list(p1c.CONTROLS), "failures": 0, "checks": 469}


def sample(cell=1, ordinal=0, cpu=2):
    m, _, n, repetitions = c.MATRIX[cell]
    return {"ordinal": ordinal, "calls": repetitions, "verified_calls": repetitions,
            "verified_elements": repetitions * m * n, "output_hash": c.expected_output_hash(cell),
            "task_count_before": 1, "task_count_after": 1, "cpu_before": cpu, "cpu_after": cpu,
            "affinity_before": [cpu], "affinity_after": [cpu], "wall_ns": 12_000_000,
            "thread_cpu_ns": 12_000_000, "involuntary_switches": 0, "voluntary_switches": 0,
            "rss_max_bytes_before": 40_000_000, "rss_max_bytes_after": 60_000_000,
            "timing_accepted": True}


def receipt(cell=1, cpu=2):
    m, k, n, repetitions = c.MATRIX[cell]
    return {**common(), "mode": "measure", "status": "MEASURED", "cell": cell,
            "m": m, "k": k, "n": n, "repetitions": repetitions, "warmup_aggregates": 2,
            "timer_floor_ns": 5_000_000, "checked_timed_calls": repetitions * 9,
            "checked_timed_elements": repetitions * 9 * m * n, "checked_warmup_calls": repetitions * 2,
            "retained_output_payload_bytes": repetitions * m * n * 4, "sizeof_result": 80,
            "live_payload_bound_bytes": c.payload_bound(cell, 80), "process_task_count_before": 1,
            "process_task_count_after": 1, "process_affinity_before": [cpu], "process_affinity_after": [cpu],
            "loaded_modules": ["/synthetic/p5/executable", "/synthetic/p5/runtime.so"],
            "warmups": [sample(cell, i, cpu) for i in range(2)],
            "samples": [sample(cell, i, cpu) for i in range(9)]}


def cells():
    values = {"stock": 10000, "tiles-3x5": 12000, "peel-3x5": 11000,
              "tiles-5x3": 15000, "peel-5x3": 10000}
    return [{**entry, "timing_accepted": True,
             "normalized_ns": [values[entry["role"]] * (1 if entry["direction"] == "forward" else 1.04)] * 9}
            for entry in c.plan()["order"]]


def block(raw, start, end):
    if raw.count(start) != 1 or raw.count(end) != 1:
        raise AssertionError("test protected-block marker is no longer unique")
    return raw[raw.index(start):raw.index(end)]


def host():
    return {"selected_cpu": 2, "caller_affinity": [2], "build_test_comm_counts": {},
            "loadavg": [0.1,0.1,0.1], "logical_cpus": 24,
            "cpu_temperature_millicelsius": {"synthetic/temp1_input": 45000},
            "exclusivity_proof": "operator acknowledgement plus bounded observations, not continuous proof",
            "os_kernel": {"system": "Linux", "release": "SYNTHETIC-NOT-MEASURED", "machine": "x86_64"},
            "hardware": {"model name": "SYNTHETIC-NOT-MEASURED"}, "smt_siblings": "2,14",
            "cache_topology": [], "governor": "synthetic", "boost": "synthetic"}


class CollectionFixture:
    """Real raw files; preparation authority deliberately NOT modeled.

    Only audit_prepared is mocked. This tests collection derivation, NOT a
    successful full preparation, real SDK, native run or origin attestation.
    """
    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mdslc-p5-SYNTHETIC-")
        base = Path(self.temporary.name).resolve()
        self.root = base / "collection"
        self.root.mkdir()
        (self.root / "raw").mkdir()
        binaries = base / "synthetic-bin"
        binaries.mkdir()
        self.prepared_path = base / "synthetic-prepared.json"
        self.prepared_path.write_text('{"synthetic_unit_prerequisite_only":true}\n')
        self.prepared = {"instances": {}}
        runtime = base / "SYNTHETIC-runtime.so"
        runtime.write_text("SYNTHETIC: not a shared object, never loaded\n")
        for role in c.ROLES:
            binary = binaries / ("synthetic-" + role)
            binary.write_text("SYNTHETIC: not an executable, never launched\n")
            modules = sorted([str(binary), str(runtime)])
            self.prepared["instances"][role] = {"timed": {"binary": str(binary), "loaded_modules": modules}}
        self.report = {"schema": runner.COLLECTION_SCHEMA, "version": c.VERSION, "status": "COMPLETE",
                       "required_product_checkpoint": c.QUALIFIED_SOURCE_COMMIT,
                       "plan_sha256": c.digest(c.plan()), "prepared": str(self.prepared_path),
                       "prepared_identity": p1.file_identity(self.prepared_path), "selected_cpu": 2,
                       "identity_checks_passed": True, "collection_wall_seconds": 30.0,
                       "source_runtime_policy_changed": False, "automatic_selection": False,
                       "performance_claim": "none until reviewed", "host_before": host(), "host_after": host(),
                       "commands": [], "cells": [], "control_runs": []}
        for phase in ("initial", "final"):
            for role in c.ROLES:
                binary = self.prepared["instances"][role]["timed"]["binary"]
                modules = self.prepared["instances"][role]["timed"]["loaded_modules"]
                value = {**control(), "loaded_modules": modules}
                index = self.add([binary, "--verify-only"], value)
                self.report["control_runs"].append({"phase": phase, "role": role, "command_index": index, "receipt": value})
            if phase == "initial":
                for entry in c.plan()["order"]:
                    binary = self.prepared["instances"][entry["role"]]["timed"]["binary"]
                    value = receipt(entry["cell"])
                    value["loaded_modules"] = self.prepared["instances"][entry["role"]]["timed"]["loaded_modules"]
                    index = self.add([binary, "--measure", str(entry["cell"]), "--cpu", "2"], value)
                    self.report["cells"].append({**entry, "receipt": value, "command_index": index,
                        "status": value["status"], "timing_accepted": True,
                        "normalized_ns": [sample["wall_ns"] / value["repetitions"] for sample in value["samples"]],
                        "host_before": host(), "host_after": host()})
        self.report["summary"] = c.summarize(self.report["cells"])
        self.path = self.root / "collection.json"
        self.save()
        return self

    def __exit__(self, *unused):
        self.temporary.cleanup()

    def add(self, argv, value):
        index = len(self.report["commands"])
        directory = self.root / "raw" / f"{index:04d}"
        directory.mkdir()
        record = {"argv": argv, "cwd": str(self.root), "timeout_seconds": 20,
                  "affinity_child": [2], "environment": p1.env_record(), "stripped_environment_names": [],
                  "loader_injection_environment_removed": True, "returncode": 0,
                  "timed_out": False, "directory": f"raw/{index:04d}"}
        self.report["commands"].append(record)
        (directory / "stdout.log").write_text(json.dumps(value) + "\n")
        (directory / "stderr.log").write_text("")
        (directory / "command.json").write_text(json.dumps(record) + "\n")
        return index

    def rewrite_command(self, index):
        (self.root / f"raw/{index:04d}/command.json").write_text(json.dumps(self.report["commands"][index]) + "\n")

    def save(self):
        self.report["raw_catalog"] = runner._catalog(self.root)
        self.report["budget"] = p1.budget(self.root)
        self.path.write_text(json.dumps(self.report) + "\n")

    def audit(self):
        with mock.patch.object(runner, "audit_prepared", return_value=self.prepared) as prerequisite, \
             mock.patch("subprocess.run", side_effect=AssertionError("offline audit attempted process execution")):
            result = runner.audit_collection(self.path)
            prerequisite.assert_called_once_with(self.prepared_path)
            return result


class ProtocolControls(unittest.TestCase):
    def test_frozen_roles_shapes_flags_and_chronology(self):
        roles = ("stock", "tiles-3x5", "peel-3x5", "tiles-5x3", "peel-5x3")
        self.assertEqual(c.ROLES, roles)
        self.assertEqual(c.MATRIX, ((33,35,36,2048), (33,35,37,2048), (33,35,38,2048)))
        self.assertEqual(c.FLAGS, {"stock": None, "tiles-3x5": "--gemm-output-tiles",
                                  "peel-3x5": "--gemm-output-tiles-peel-n", "tiles-5x3": "--gemm-output-tiles",
                                  "peel-5x3": "--gemm-output-tiles-peel-n"})
        plan = c.plan()
        expected = [{"direction": "forward", "cell": cell, "role": role} for cell in range(3) for role in roles]
        self.assertEqual(plan["order"], expected + [{**entry, "direction": "reverse"} for entry in reversed(expected)])
        self.assertEqual(plan["initial_and_final_controls"], list(roles))
        self.assertEqual(len(plan["order"]) + 2 * len(roles), 40)
        self.assertEqual(plan["timer_floor_ns"], 5_000_000)
        self.assertEqual(plan["measured_aggregates"], 9)
        self.assertEqual(plan["warmup_aggregates"], 2)
        self.assertEqual(plan["noise_policy"], p1c.plan()["noise_policy"])
        self.assertEqual(plan["primary"]["cell"], 1)
        self.assertEqual(plan["primary"]["comparison"], "peel-5x3-over-tiles-5x3")
        self.assertEqual((plan["process_seconds"], plan["collection_seconds"]), (20, 180))
        self.assertEqual((plan["live_payload_cap_bytes"], plan["raw_cap_bytes"], plan["generated_artifact_cap_bytes"]),
                         (64 << 20, 64 << 20, 256 << 20))
        self.assertEqual(plan["winner_policy"], "none")
        self.assertIn("forbidden", plan["cross_campaign_timing_comparison"])

    def test_p1_is_read_only_and_distinct(self):
        self.assertIs(p1.c, p1c)
        self.assertIsNot(c, p1c)
        self.assertEqual(p1c.ROLES, ("stock", "tiles-3x5", "tiles-5x3"))
        self.assertEqual(len(p1c.MATRIX), 6)
        self.assertEqual(p1c.FLOOR_NS, 1_000_000)
        self.assertEqual(p1c.digest(p1c.plan()), "dffefb75511adf63afa8ea647da39e04fdbd18c00f6acb01f561732d3c2f8649")
        self.assertNotEqual(c.digest(c.plan()), p1c.digest(p1c.plan()))
        self.assertNotEqual(legacy.__name__, peel.__name__)
        self.assertTrue((ROOT / "compiler/tests/source_pattern_peel/run.py").is_file())

    def test_renderer_exact_three_changes_and_protected_blocks(self):
        raw = (P1_HERE / "call_return.mdsl").read_bytes()
        before = hashlib.sha256(raw).hexdigest()
        self.assertEqual(before, "3717a4de99cca90dd0754a676299e39c2435e76cbd373ef0f3b6bbacf441f7e3")
        rendered = c.render_source(raw)
        self.assertEqual(type(rendered), bytes)
        self.assertEqual(rendered, c.render_source())
        self.assertEqual(rendered, c.render_source(raw))
        for start, end in ((b"MATCORE_REGION\n", b"#pragma STDC FENV_ACCESS"),
                           (b"  [[gnu::noinline]] md::Result call(", b"  void immutable() const"),
                           (b"Sample aggregate(", b"std::string sample_json(")):
            self.assertEqual(block(raw, start, end), block(rendered, start, end))
        changes = list(difflib.unified_diff(raw.decode().splitlines(), rendered.decode().splitlines(), n=0))
        removed = [line[1:] for line in changes if line.startswith("-") and not line.startswith("---")]
        added = [line[1:] for line in changes if line.startswith("+") and not line.startswith("+++")]
        self.assertEqual(len(removed), 4)
        self.assertEqual(len(added), 4)
        self.assertTrue(all("floor_ns =" in line or "matrix{{" in line or
                            "{64,7,19,256}" in line or "matcore.source-pattern.p1" in line for line in removed))
        self.assertTrue(all("floor_ns =" in line or "matrix{{" in line or
                            "{33,35,38,2048}" in line or "matcore.source-pattern.p5" in line for line in added))
        self.assertIn(b"5'000'000", rendered)
        self.assertIn(b"{33,35,37,2048}", rendered)
        self.assertNotIn(b"matcore.source-pattern.p1", rendered)
        self.assertEqual(hashlib.sha256((P1_HERE / "call_return.mdsl").read_bytes()).hexdigest(), before)

    def test_renderer_refuses_foreign_repeated_or_mutated_source(self):
        raw = (P1_HERE / "call_return.mdsl").read_bytes()
        for wrong in (raw + b"\n", raw.replace(b"results.reserve", b"results.resize"),
                      raw.replace(b"involuntary==0", b"involuntary<=1"),
                      raw + b"matcore.source-pattern.p1", b"", raw.decode(), bytearray(raw)):
            with self.subTest(kind=type(wrong)), self.assertRaises(ValueError):
                c.render_source(wrong)

    def test_noise_floor_ratio_slack_and_no_one_switch_rescue(self):
        base = {"wall_ns": 5_000_000, "thread_cpu_ns": 5_000_000,
                "involuntary_switches": 0, "voluntary_switches": 0}
        self.assertTrue(c.timing_accepted(base))
        for changes, expected in (({"wall_ns": 4_999_999, "thread_cpu_ns": 4_999_999}, False),
                                  ({"thread_cpu_ns": 4_500_000}, True),
                                  ({"thread_cpu_ns": 4_499_999}, False),
                                  ({"thread_cpu_ns": 5_100_000}, True),
                                  ({"thread_cpu_ns": 5_100_001}, False),
                                  ({"involuntary_switches": 1}, False),
                                  ({"involuntary_switches": 2}, False)):
            with self.subTest(changes=changes):
                self.assertEqual(c.timing_accepted({**base, **changes}), expected)
        # Original policy records voluntary switches; it does not add a new
        # voluntary-switch gate or reinterpret counts as elapsed interference.
        self.assertTrue(c.timing_accepted({**base, "voluntary_switches": 1}))
        for key in base:
            for bad in (True, -1, 1.5, "0", None):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    c.timing_accepted({**base, key: bad})

    def test_strict_json_and_exact_scalar_types(self):
        self.assertEqual(c.strict_json('{"x":[1,2]}'), {"x": [1,2]})
        for raw in ('{"x":1,"x":2}', '{"x":{"y":1,"y":2}}', '{"x":NaN}', '{"x":Infinity}', '[]'):
            with self.assertRaises(ValueError):
                c.strict_json(raw)
        for bad, wanted in ((True, 1), (1.0, 1), (1, True), ([1], (1,))):
            with self.assertRaises(ValueError):
                c.exact(bad, wanted, "synthetic exact type")

    def test_control_receipt_complete_and_not_p1(self):
        c.check_controls(control(), 0)
        for change in ({"schema": "matcore.source-pattern.p1"}, {"controls": list(c.CONTROLS)[:-1]},
                       {"checks": 0}, {"checks": True}, {"failures": 1}, {"release_inside_timing": True},
                       {"continuous_thread_proof": True}, {"numerics": "reassociate_f32"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.check_controls({**control(), **change}, 0)
        with self.assertRaises(ValueError):
            c.check_controls(control(), 78)

    def test_planning_bound_not_total_rss(self):
        self.assertEqual(c.payload_bound(2, 80), 10_776_716)
        self.assertEqual(c.payload_bound(2, 1024), 12_710_028)
        for cell, (m, k, n, r) in enumerate(c.MATRIX):
            for size in (1, 80, 1024):
                independent = (5*m*k + 2*k*n + (r+8)*m*n)*4 + r*(size+128) + 4096
                self.assertEqual(c.payload_bound(cell, size), independent)
                self.assertLess(independent, 64 << 20)

    def test_independent_integer_prefix_struct_f32_hash(self):
        m, k, n, repetitions = c.MATRIX[1]
        sequences = []
        for sequence in range(4):
            data = bytearray()
            for row in range(m):
                for column in range(n):
                    total = 0
                    for q in range(k):
                        a = (17*row + 5*q + 3*sequence + p1c.SEED % 19) % 19 - 9
                        b = (7*q + 11*column + (p1c.SEED >> 32) % 23) % 23 - 11
                        total += a*b
                        self.assertLess(abs(total), 1 << 24)
                    data.extend(struct.pack("<f", float(total)))
            sequences.append(data)
        expected = 0xCBF29CE484222325
        for call in range(repetitions):
            for byte in sequences[call % 4]:
                expected = ((expected ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
        self.assertEqual(c.expected_output_hash(1), expected)

    def test_full_cell_receipt(self):
        for cell in range(3):
            self.assertTrue(c.check_cell(receipt(cell), 0, cell, 2))

    def test_cell_content_placement_shape_and_count_forgeries(self):
        original = receipt()
        for key, bad in (("schema", "matcore.source-pattern.p1"), ("repetitions", 2047),
                         ("checked_timed_calls", 0), ("checked_timed_elements", 0), ("checked_warmup_calls", 0),
                         ("process_task_count_after", 2), ("process_affinity_after", [2,14]),
                         ("live_payload_bound_bytes", 1), ("sizeof_result", 1025), ("sizeof_result", True),
                         ("loaded_modules", []), ("loaded_modules", ["relative.so"]),
                         ("loaded_modules", ["/duplicate", "/duplicate"]), ("n", 38)):
            changed = copy.deepcopy(original)
            changed[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                c.check_cell(changed, 0, 1, 2)
        for key, bad in (("output_hash", 0), ("calls", 2047), ("verified_calls", 1),
                         ("verified_elements", 0), ("ordinal", 8), ("cpu_after", 14),
                         ("affinity_before", [2,14]), ("task_count_after", 2),
                         ("rss_max_bytes_after", -1), ("timing_accepted", 1)):
            changed = copy.deepcopy(original)
            changed["samples"][0][key] = bad
            with self.subTest(sample_key=key, bad=bad), self.assertRaises(ValueError):
                c.check_cell(changed, 0, 1, 2)
        for group in ("warmups", "samples"):
            for delta in (-1, 1):
                changed = copy.deepcopy(original)
                changed[group] = changed[group][:-1] if delta < 0 else changed[group] + [changed[group][-1]]
                with self.assertRaises(ValueError):
                    c.check_cell(changed, 0, 1, 2)

    def test_one_failed_sample_means_inconclusive_not_rescue(self):
        for index in (0, 8):
            changed = receipt()
            changed["samples"][index].update(involuntary_switches=1, timing_accepted=False)
            changed["status"] = "INCONCLUSIVE_TIMING"
            self.assertFalse(c.check_cell(changed, 78, 1, 2))
            for bad in ("flag", "status", "exit"):
                forged = copy.deepcopy(changed)
                if bad == "flag":
                    forged["samples"][index]["timing_accepted"] = True
                if bad == "status":
                    forged["status"] = "MEASURED"
                with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                    c.check_cell(forged, 0 if bad == "exit" else 78, 1, 2)

    def test_summary_has_one_predeclared_primary_and_correct_ratios(self):
        result = c.summarize(cells())
        primary = [(entry["cell"], label) for entry in result for label, value in entry["comparisons"].items() if value["primary"]]
        self.assertEqual(primary, [(1, "peel-5x3-over-tiles-5x3")])
        comparison = result[1]["comparisons"][c.PRIMARY_COMPARISON]
        self.assertEqual(comparison["status"], "PAIRED")
        self.assertAlmostEqual(comparison["ratio_of_midpoint_medians"], 2/3)
        for ratio in comparison["direction_time_ratios"].values():
            self.assertAlmostEqual(ratio, 2/3)
        self.assertAlmostEqual(result[1]["comparisons"]["peel-3x5-over-tiles-3x5"]["ratio_of_midpoint_medians"], 11/12)
        self.assertEqual(len(result[1]["comparisons"]), 6)

    def test_primary_failure_does_not_promote_secondary(self):
        data = cells()
        next(entry for entry in data if entry["cell"] == 1 and entry["role"] == "tiles-5x3" and entry["direction"] == "reverse")["timing_accepted"] = False
        result = c.summarize(data)
        primary = result[1]["comparisons"][c.PRIMARY_COMPARISON]
        self.assertTrue(primary["primary"])
        self.assertEqual(primary["status"], "INCONCLUSIVE")
        self.assertIsNone(primary["ratio_of_midpoint_medians"])
        self.assertIsNone(primary["direction_time_ratios"])
        self.assertEqual(result[1]["comparisons"]["peel-3x5-over-tiles-3x5"]["status"], "PAIRED")
        self.assertFalse(result[1]["comparisons"]["peel-3x5-over-tiles-3x5"]["primary"])

    def test_valid_loss_and_stock_failure_remain_explicit(self):
        data = cells()
        for entry in data:
            if entry["cell"] == 1 and entry["role"] == "peel-5x3":
                entry["normalized_ns"] = [value * 3 for value in entry["normalized_ns"]]
            if entry["cell"] == 1 and entry["role"] == "stock":
                entry["timing_accepted"] = False
        result = c.summarize(data)[1]["comparisons"]
        self.assertAlmostEqual(result[c.PRIMARY_COMPARISON]["ratio_of_midpoint_medians"], 2)
        self.assertEqual(result[c.PRIMARY_COMPARISON]["status"], "PAIRED")
        for role in c.ROLES[1:]:
            self.assertEqual(result[role + "-over-stock"]["status"], "INCONCLUSIVE")
            self.assertIsNone(result[role + "-over-stock"]["ratio_of_midpoint_medians"])
            self.assertIsNone(result[role + "-over-stock"]["direction_time_ratios"])

    def test_nine_sample_p95_is_max_and_spread_order_limits(self):
        data = cells()
        forward = next(entry for entry in data if entry["cell"] == 1 and entry["role"] == "stock" and entry["direction"] == "forward")
        reverse = next(entry for entry in data if entry["cell"] == 1 and entry["role"] == "stock" and entry["direction"] == "reverse")
        forward["normalized_ns"][-1] = 12500
        self.assertEqual(c.summarize(data)[1]["roles"]["stock"]["status"], "PAIRED")
        self.assertEqual(c.summarize(data)[1]["roles"]["stock"]["within_direction_spread"]["forward"]["nearest_rank_p95_ns"], 12500)
        forward["normalized_ns"][-1] = 12501
        self.assertEqual(c.summarize(data)[1]["roles"]["stock"]["status"], "INCONCLUSIVE_SPREAD_OR_ORDER")
        forward["normalized_ns"] = [10000] * 9
        reverse["normalized_ns"] = [11000] * 9
        self.assertEqual(c.summarize(data)[1]["roles"]["stock"]["status"], "PAIRED")
        reverse["normalized_ns"] = [11001] * 9
        result = c.summarize(data)[1]
        self.assertEqual(result["roles"]["stock"]["status"], "INCONCLUSIVE_SPREAD_OR_ORDER")
        self.assertIsNone(result["comparisons"]["peel-5x3-over-stock"]["direction_time_ratios"])

    def test_counts_preserving_order_and_summary_value_attacks(self):
        for attack in ("swap", "duplicate", "direction", "missing", "extra"):
            data = cells()
            if attack == "swap":
                data[0], data[1] = data[1], data[0]
            elif attack == "duplicate":
                data[1] = copy.deepcopy(data[0])
            elif attack == "direction":
                data[-1]["direction"] = "forward"
            elif attack == "missing":
                data.pop()
            else:
                data.append(copy.deepcopy(data[-1]))
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                c.summarize(data)
        for bad in ([], [1] * 8, [True] * 9, [0] * 9, [-1] * 9, [float("inf")] * 9, [float("nan")] * 9):
            data = cells()
            data[0]["normalized_ns"] = bad
            with self.subTest(samples=bad), self.assertRaises(ValueError):
                c.summarize(data)
        data = cells()
        data[0]["timing_accepted"] = 1
        with self.assertRaises(ValueError):
            c.summarize(data)

    def test_extreme_finite_inputs_never_emit_nonfinite_ratios(self):
        data = cells()
        for entry in data:
            entry["normalized_ns"] = [1e308] * 9
        try:
            result = c.summarize(data)
        except ValueError:
            return  # Explicit rejection is also a valid defensive boundary.
        for entry in result:
            for role in entry["roles"].values():
                if "paired_midpoint_ns" in role:
                    self.assertTrue(math.isfinite(role["paired_midpoint_ns"]))
            for comparison in entry["comparisons"].values():
                if comparison["ratio_of_midpoint_medians"] is not None:
                    self.assertTrue(math.isfinite(comparison["ratio_of_midpoint_medians"]))


class OfflineCollectionControls(unittest.TestCase):
    def test_real_raw_40_command_collection_positive_no_processes(self):
        with CollectionFixture() as fixture:
            self.assertEqual(len(fixture.report["commands"]), 40)
            self.assertEqual(len(fixture.report["raw_catalog"]), 120)
            self.assertEqual(fixture.audit(), fixture.report["summary"])

    def test_recomputed_catalog_cannot_rescue_argv_directory_environment_timeout(self):
        for field, value in (("argv", ["/synthetic/WRONG-ROLE", "--verify-only"]),
                             ("directory", "raw/0001"), ("cwd", "/synthetic/wrong-cwd"),
                             ("affinity_child", [14]), ("environment", {}),
                             ("loader_injection_environment_removed", False),
                             ("timed_out", True), ("timeout_seconds", 999), ("returncode", False)):
            with self.subTest(field=field), CollectionFixture() as fixture:
                fixture.report["commands"][0][field] = value
                fixture.rewrite_command(0)
                fixture.save()
                with self.assertRaises(ValueError):
                    fixture.audit()

    def test_raw_catalog_missing_extra_and_duplicate_join(self):
        with CollectionFixture() as fixture:
            audit = runner.RawAudit(fixture.root, fixture.report)
            record = fixture.report["commands"][0]
            audit.raw(0, record["argv"], cpu=2)
            with self.assertRaises(ValueError):
                audit.raw(0, record["argv"], cpu=2)
            with self.assertRaises(ValueError):
                audit.finish()
        for attack in ("missing", "extra", "catalog-missing"):
            with self.subTest(attack=attack), CollectionFixture() as fixture:
                if attack == "missing":
                    (fixture.root / "raw/0000/stdout.log").unlink()
                elif attack == "extra":
                    (fixture.root / "raw/0000/extra.log").write_text("unjoined\n")
                else:
                    fixture.report["raw_catalog"].pop("raw/0000/stdout.log")
                    fixture.path.write_text(json.dumps(fixture.report) + "\n")
                with self.assertRaises((ValueError, OSError)):
                    fixture.audit()

    def test_replay_exhaustion_and_pinned_bytes_are_offline_only(self):
        with CollectionFixture() as fixture:
            # Replay models a PREPARATION command, whose child affinity is
            # deliberately unspecified, unlike collection's pinned children.
            fixture.report["commands"][0]["affinity_child"] = None
            fixture.rewrite_command(0)
            fixture.save()
            pinned = {str(fixture.prepared_path): p1.file_identity(fixture.prepared_path)}
            audit = runner.RawAudit(fixture.root, fixture.report)
            replay = runner.Replay(audit, [0], pinned)
            self.assertEqual(replay.pin(fixture.prepared_path), pinned[str(fixture.prepared_path)])
            with self.assertRaises(ValueError):
                replay.finish()
            with self.assertRaises(ValueError):
                replay.invoke(fixture.report["commands"][0]["argv"], unrecorded_process_option=True)
            self.assertEqual(replay.cursor, 0)
            with mock.patch("subprocess.run", side_effect=AssertionError("Replay launched a process")):
                result = replay.invoke(fixture.report["commands"][0]["argv"])
            self.assertEqual(result.returncode, 0)
            replay.finish()
            with self.assertRaises(ValueError):
                replay.invoke(fixture.report["commands"][1]["argv"])
            fixture.prepared_path.write_text("changed\n")
            with self.assertRaises(ValueError):
                replay.pin(fixture.prepared_path)

    def test_raw_content_module_and_noise_forgery_not_just_catalog(self):
        for attack in ("stderr", "output-hash", "module", "switch"):
            with self.subTest(attack=attack), CollectionFixture() as fixture:
                entry = fixture.report["cells"][0]
                index = entry["command_index"]
                if attack == "stderr":
                    (fixture.root / f"raw/{index:04d}/stderr.log").write_text("unexpected diagnostic\n")
                else:
                    if attack == "output-hash":
                        entry["receipt"]["samples"][0]["output_hash"] ^= 1
                    elif attack == "module":
                        foreign = fixture.root.parent / "SYNTHETIC-foreign.so"
                        foreign.write_text("wrong existing module, never loaded\n")
                        entry["receipt"]["loaded_modules"] = [str(foreign)]
                    else:
                        entry["receipt"]["samples"][0]["involuntary_switches"] = 1
                    (fixture.root / f"raw/{index:04d}/stdout.log").write_text(json.dumps(entry["receipt"]) + "\n")
                fixture.save()  # Public hashes are internally consistent again.
                with self.assertRaises(ValueError):
                    fixture.audit()

    def test_derived_status_normalization_ratio_and_chronology_attacks(self):
        for attack in ("status", "normalized", "summary", "swap-cells", "control-index", "swap-controls", "schema", "deadline",
                       "source-policy", "auto-policy", "identity", "checkpoint"):
            with self.subTest(attack=attack), CollectionFixture() as fixture:
                if attack == "status":
                    fixture.report["status"] = "INCONCLUSIVE_TIMING"
                elif attack == "normalized":
                    fixture.report["cells"][0]["normalized_ns"][0] += 1
                elif attack == "summary":
                    fixture.report["summary"][1]["comparisons"][c.PRIMARY_COMPARISON]["ratio_of_midpoint_medians"] = 7.0
                elif attack == "swap-cells":
                    fixture.report["cells"][0], fixture.report["cells"][1] = fixture.report["cells"][1], fixture.report["cells"][0]
                elif attack == "control-index":
                    fixture.report["control_runs"][0]["command_index"] = 1
                elif attack == "swap-controls":
                    fixture.report["control_runs"][0], fixture.report["control_runs"][1] = fixture.report["control_runs"][1], fixture.report["control_runs"][0]
                elif attack == "schema":
                    fixture.report["schema"] = "matcore.source-pattern.p1.collection"
                elif attack == "deadline":
                    fixture.report["collection_wall_seconds"] = 180.00001
                elif attack == "source-policy":
                    fixture.report["source_runtime_policy_changed"] = True
                elif attack == "auto-policy":
                    fixture.report["automatic_selection"] = True
                elif attack == "identity":
                    fixture.report["identity_checks_passed"] = False
                else:
                    fixture.report["required_product_checkpoint"] = "0" * 40
                fixture.save()
                with self.assertRaises(ValueError):
                    fixture.audit()

    def test_retained_host_gate_contradictions(self):
        for field, bad in (("selected_cpu", 14), ("caller_affinity", [14]),
                           ("build_test_comm_counts", {"ninja": 1}), ("logical_cpus", 0),
                           ("loadavg", [6.00001,0,0]),
                           ("cpu_temperature_millicelsius", {"synthetic/temp1_input": 85000})):
            for location in ("endpoint", "cell"):
                with self.subTest(field=field, location=location), CollectionFixture() as fixture:
                    record = fixture.report["host_after"] if location == "endpoint" else fixture.report["cells"][0]["host_before"]
                    record[field] = bad
                    fixture.save()
                    with self.assertRaises(ValueError):
                        fixture.audit()

    def test_actual_generated_cap_not_only_reported_budget(self):
        with CollectionFixture() as fixture:
            (fixture.root / "bin").mkdir()
            oversized = fixture.root / "bin/oversized-SYNTHETIC-sparse-data"
            with oversized.open("wb") as stream:
                stream.truncate((256 << 20) + 1)  # Sparse fixture, no large allocation/write.
            with self.assertRaises(ValueError):
                p1.budget(fixture.root)  # Actual sparse size, not reported bytes.
            with self.assertRaises(ValueError):
                fixture.audit()
        with CollectionFixture() as fixture:
            fixture.report["budget"]["raw_bytes"] += 1
            fixture.path.write_text(json.dumps(fixture.report) + "\n")
            with self.assertRaises(ValueError):
                fixture.audit()

    def test_role_flags_argv_and_closed_inventory(self):
        root = Path("/synthetic/p5")
        driver = root / "sdk/bin/mdslc-region"
        for role in c.ROLES:
            for kind in runner._kinds(role):
                argv = runner._argv(driver, root, role, kind)
                self.assertEqual(argv[0], str(driver))
                self.assertEqual(argv[argv.index("--candidate") + 1], "generated-strict")
                self.assertEqual(argv[argv.index("--optimization") + 1], "none")
                self.assertEqual(argv[argv.index("-o") + 1], str(root / "bin" / (kind + "-" + role)))
                if role == "stock":
                    self.assertNotIn("--gemm-output-tiles", argv)
                    self.assertNotIn("--gemm-output-tiles-peel-n", argv)
                else:
                    self.assertEqual(argv[-2:], [c.FLAGS[role], ",".join(map(str,c.TILES[role]))])
                    self.assertEqual(runner._request(role)["schema"],
                                     "matcore-source-output-peel-n-v1" if role.startswith("peel-") else "matcore-source-output-pattern-v1")
        with self.assertRaises(ValueError):
            runner._request("tiles-5x1")

    def test_how_relations_reject_counterfeit_or_stale_eight_hash_binding(self):
        common_keys = ("semantic_sha256", "structured_sha256", "bufferized_sha256",
                       "canonical_runtime_sha256", "stock_candidates_sha256", "clang_sha256", "linker_sha256",
                       "source_context_sha256", "region_name_hex", "candidate", "optimization", "toolchain", "target",
                       "address_sanitizer", "relink_kit_sha256", "exports_sha256", "public_header_sha256",
                       "storage_header_sha256", "private_header_sha256", "provider_sha256", "accelerator_count",
                       "accelerator_0_sha256")
        instances = {}
        for role in c.ROLES:
            manifest = {key: hashlib.sha256(key.encode()).hexdigest() for key in common_keys}
            manifest.update(accelerator_count="1", candidate="generated-strict", optimization="none",
                            target="x86_64-pc-linux-gnu", toolchain="21.1.8", address_sanitizer="off",
                            region_name_hex="p1_pipeline".encode().hex())
            manifest.update({key: hashlib.sha256((role+key).encode()).hexdigest()
                             for key in (*c.ISSUED_HASHES, "dso_sha256", "binding_sha256") if key not in common_keys})
            instances[role] = {kind: {"manifest": copy.deepcopy(manifest)} for kind in runner._kinds(role)}
        runner._relations(instances)
        for attack in ("stale-control", "counterfeit-how", "wrong-runtime", "missing-role", "wrong-kind"):
            changed = copy.deepcopy(instances)
            if attack == "stale-control":
                changed["peel-5x3"]["control"]["manifest"]["llvm_sha256"] = "0" * 64
            elif attack == "counterfeit-how":
                changed["peel-5x3"] = {kind: {"manifest": copy.deepcopy(instances["tiles-5x3"]["timed"]["manifest"])}
                                       for kind in runner._kinds("peel-5x3")}
            elif attack == "wrong-runtime":
                changed["peel-3x5"]["timed"]["manifest"]["canonical_runtime_sha256"] = "0" * 64
            elif attack == "missing-role":
                del changed["stock"]
            else:
                changed["tiles-3x5"]["end_k"] = copy.deepcopy(changed["tiles-3x5"]["control"])
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                runner._relations(changed)
        for field in ("source_context_sha256", "region_name_hex", "candidate", "optimization", "toolchain", "target",
                      "address_sanitizer", "relink_kit_sha256", "exports_sha256", "public_header_sha256",
                      "storage_header_sha256", "private_header_sha256", "provider_sha256", "accelerator_count",
                      "accelerator_0_sha256"):
            changed = copy.deepcopy(instances)
            changed["peel-5x3"]["timed"]["manifest"][field] = "DIFFERENT-SYNTHETIC-OWNER"
            with self.subTest(common_field=field), self.assertRaises(ValueError):
                runner._relations(changed)


if __name__ == "__main__":
    unittest.main()
