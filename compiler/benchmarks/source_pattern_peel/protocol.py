"""Fixed P5 source-call experiment. Pure checks; no compiler or timing runs.

This is a new preregistration, not a repaired/retried P1 campaign or optimizer.
The five roles share one source and one qualified compiler installation.
"""
from functools import lru_cache
import hashlib
import math
import statistics

from support import P1_HERE, p1c

require, exact = p1c.require, p1c.exact
canonical, digest, strict_json = p1c.canonical, p1c.digest, p1c.strict_json
VERSION, SEED, INPUTS = 1, p1c.SEED, p1c.INPUTS
SCOPE, CONTROLS, ISSUED_HASHES = p1c.SCOPE, p1c.CONTROLS, p1c.ISSUED_HASHES
ROLES = ("stock", "tiles-3x5", "peel-3x5", "tiles-5x3", "peel-5x3")
TILES = {"stock": None, "tiles-3x5": (3, 5), "peel-3x5": (3, 5),
         "tiles-5x3": (5, 3), "peel-5x3": (5, 3)}
FLAGS = {role: None if role == "stock" else
         "--gemm-output-tiles-peel-n" if role.startswith("peel-") else
         "--gemm-output-tiles" for role in ROLES}
MATRIX = ((33, 35, 36, 2048), (33, 35, 37, 2048), (33, 35, 38, 2048))
WARMUPS, SAMPLES, FLOOR_NS = 2, 9, 5_000_000
PROCESS_SECONDS, COLLECTION_SECONDS = p1c.PROCESS_SECONDS, p1c.COLLECTION_SECONDS
LIVE_PAYLOAD_CAP, RAW_CAP, ARTIFACT_CAP = p1c.LIVE_PAYLOAD_CAP, p1c.RAW_CAP, p1c.ARTIFACT_CAP
SOURCE_SHA256 = "3717a4de99cca90dd0754a676299e39c2435e76cbd373ef0f3b6bbacf441f7e3"
QUALIFIED_SOURCE_COMMIT = "1ad7fe56b5fcb10efe37e523206228e9ef303f6a"
SCHEMA = "matcore.source-pattern.p5"
PRIMARY_CELL, PRIMARY_COMPARISON = 1, "peel-5x3-over-tiles-5x3"
COMPARISONS = ((PRIMARY_COMPARISON, "peel-5x3", "tiles-5x3"),
               ("peel-3x5-over-tiles-3x5", "peel-3x5", "tiles-3x5"),
               *((role + "-over-stock", role, "stock") for role in ROLES[1:]))


def render_source(raw=None):
    """Three exact replacements, once; every role consumes identical bytes.

    Original region, clocks, ownership, oracles and noise predicate are retained
    byte-for-byte. A changed upstream file fails, not an adaptable text parser.
    """
    if raw is None:
        raw = (P1_HERE / "call_return.mdsl").read_bytes()
    require(type(raw) is bytes, "renderer requires exact bytes")
    exact(hashlib.sha256(raw).hexdigest(), SOURCE_SHA256, "frozen P1 source")
    replacements = (
        (b"constexpr std::uint64_t floor_ns = 1'000'000, payload_cap = 64ULL << 20;",
         b"constexpr std::uint64_t floor_ns = 5'000'000, payload_cap = 64ULL << 20;"),
        (b"constexpr std::array<Cell, 6> matrix{{{32,32,32,128}, {33,35,37,128},\n"
         b"  {64,7,19,256}, {1,257,65,256}, {65,257,1,256}, {128,128,128,8}}};",
         b"constexpr std::array<Cell, 3> matrix{{{33,35,36,2048}, {33,35,37,2048},\n"
         b"  {33,35,38,2048}}};"),
        (b"matcore.source-pattern.p1", SCHEMA.encode()),
    )
    rendered = raw
    for old, new in replacements:
        exact(rendered.count(old), 1, "unique renderer anchor")
        require(new not in rendered, "renderer replacement already exists")
        rendered = rendered.replace(old, new, 1)
    return rendered


def plan():
    forward = [{"direction": "forward", "cell": cell, "role": role}
               for cell in range(len(MATRIX)) for role in ROLES]
    reverse = [{**entry, "direction": "reverse"} for entry in reversed(forward)]
    return {"schema": SCHEMA + ".plan", "version": VERSION,
            "matrix_mkn_repetitions": MATRIX, "roles": ROLES, "tiles": TILES,
            "request_flags": FLAGS, "warmup_aggregates": WARMUPS,
            "measured_aggregates": SAMPLES, "timer_floor_ns": FLOOR_NS,
            "input_sequence": INPUTS, "seed": SEED, "timing_scope": SCOPE,
            "release_inside_timing": False, "chooser_inside_timing": True,
            "owner_vector_reserved_before_timing": True,
            "base_source_sha256": SOURCE_SHA256,
            "required_producer_source_commit": QUALIFIED_SOURCE_COMMIT,
            "qualification_gate": "operator-verified source/CI/install qualification before collection; not an imported authority token",
            "rendered_source_sha256": hashlib.sha256(render_source()).hexdigest(),
            "renderer_changes": ["fixed matrix/repetitions", "5ms floor", "receipt schema"],
            "noise_policy": p1c.plan()["noise_policy"],
            "cache_policy": p1c.plan()["cache_policy"],
            "raw_delivery": "explicit new external directory outside every Git worktree",
            "process_seconds": PROCESS_SECONDS, "collection_seconds": COLLECTION_SECONDS,
            "live_payload_cap_bytes": LIVE_PAYLOAD_CAP, "raw_cap_bytes": RAW_CAP,
            "generated_artifact_cap_bytes": ARTIFACT_CAP, "order": forward + reverse,
            "initial_and_final_controls": list(ROLES), "winner_policy": "none",
            "primary": {"cell": PRIMARY_CELL, "comparison": PRIMARY_COMPARISON,
                        "interpretation": "descriptive same-campaign call interval; not isolated peeling causality"},
            "comparisons": COMPARISONS,
            "cross_campaign_timing_comparison": "forbidden: repetitions and retained ownership changed",
            "selection_basis": "prior P1 pivot loss and P3 mechanism; deliberately narrow, not representative GEMM sampling"}


@lru_cache(maxsize=len(MATRIX))
def expected_output_hash(cell):
    m, k, n, repetitions = MATRIX[cell]
    values = []
    for sequence in range(INPUTS):
        output = []
        for i in range(m):
            for j in range(n):
                total = 0
                for p in range(k):
                    total += ((17 * i + 5 * p + 3 * sequence + SEED % 19) % 19 - 9) * (
                        (7 * p + 11 * j + (SEED >> 32) % 23) % 23 - 11)
                    require(abs(total) < (1 << 24), "integer prefix exceeds exact f32 range")
                output.append(p1c.integer_bits(total))
        values.append(output)
    result = 0xCBF29CE484222325
    for call in range(repetitions):
        for word in values[call % INPUTS]:
            for shift in range(0, 32, 8):
                result = ((result ^ ((word >> shift) & 255)) * 0x100000001B3) & ((1 << 64) - 1)
    return result


def payload_bound(cell, result_size):
    m, k, n, repetitions = MATRIX[cell]
    return (5 * m * k + 2 * k * n + (repetitions + 8) * m * n) * 4 + repetitions * (result_size + 128) + 4096


def check_common(receipt):
    for key, wanted in (("schema", SCHEMA), ("version", VERSION), ("numerics", "strict_f32"),
                        ("timing_scope", SCOPE), ("release_inside_timing", False),
                        ("chooser_inside_timing", True), ("implementation_thread_contract", 1),
                        ("continuous_thread_proof", False), ("seed", SEED), ("input_sequence", INPUTS)):
        exact(receipt.get(key), wanted, key)


def check_controls(receipt, code):
    check_common(receipt)
    for observed, wanted, label in ((code, 0, "control exit"), (receipt.get("mode"), "verify-only", "control mode"),
                                   (receipt.get("status"), "CORRECTNESS_PASSED", "control status"),
                                   (receipt.get("controls"), list(CONTROLS), "complete controls"),
                                   (receipt.get("failures"), 0, "control failures")):
        exact(observed, wanted, label)
    require(type(receipt.get("checks")) is int and receipt["checks"] >= 100, "controls did not check contents")


def timing_accepted(sample):
    for key in ("wall_ns", "thread_cpu_ns", "involuntary_switches", "voluntary_switches"):
        require(type(sample.get(key)) is int and sample[key] >= 0, "invalid clock/switch integer " + key)
    wall, thread = sample["wall_ns"], sample["thread_cpu_ns"]
    require(wall > 0 and thread > 0, "zero timing")
    require(wall <= PROCESS_SECONDS * 1_000_000_000 and thread <= PROCESS_SECONDS * 1_000_000_000 + 100_000,
            "aggregate exceeds complete process deadline")
    return wall >= FLOOR_NS and thread * 10 >= wall * 9 and thread <= wall + 100_000 and sample["involuntary_switches"] == 0


def check_cell(receipt, code, cell, cpu):
    require(type(cell) is int and 0 <= cell < len(MATRIX), "invalid cell")
    require(type(cpu) is int and cpu >= 0, "invalid CPU")
    check_common(receipt)
    exact(receipt.get("mode"), "measure", "measure mode")
    exact(receipt.get("cell"), cell, "requested cell")
    m, k, n, repetitions = MATRIX[cell]
    for key, wanted in (("m", m), ("k", k), ("n", n), ("repetitions", repetitions),
                        ("warmup_aggregates", WARMUPS), ("timer_floor_ns", FLOOR_NS),
                        ("checked_timed_calls", repetitions * SAMPLES),
                        ("checked_timed_elements", repetitions * SAMPLES * m * n),
                        ("checked_warmup_calls", repetitions * WARMUPS),
                        ("retained_output_payload_bytes", repetitions * m * n * 4),
                        ("process_task_count_before", 1), ("process_task_count_after", 1),
                        ("process_affinity_before", [cpu]), ("process_affinity_after", [cpu])):
        exact(receipt.get(key), wanted, key)
    size = receipt.get("sizeof_result")
    require(type(size) is int and 1 <= size <= 1024, "Result size missing")
    exact(receipt.get("live_payload_bound_bytes"), payload_bound(cell, size), "live-payload bound")
    require(receipt["live_payload_bound_bytes"] <= LIVE_PAYLOAD_CAP, "live-payload cap")
    modules = receipt.get("loaded_modules")
    require(type(modules) is list and modules and all(type(path) is str and path.startswith("/") for path in modules)
            and modules == sorted(set(modules)), "loaded-module inventory")
    all_valid = True
    for name, count in (("warmups", WARMUPS), ("samples", SAMPLES)):
        group = receipt.get(name)
        require(type(group) is list and len(group) == count, "missing/extra aggregate " + name)
        for ordinal, sample in enumerate(group):
            require(type(sample) is dict, "sample is not an object")
            for key, wanted in (("ordinal", ordinal), ("calls", repetitions), ("verified_calls", repetitions),
                                ("verified_elements", repetitions * m * n), ("output_hash", expected_output_hash(cell)),
                                ("task_count_before", 1), ("task_count_after", 1), ("cpu_before", cpu), ("cpu_after", cpu),
                                ("affinity_before", [cpu]), ("affinity_after", [cpu])):
                exact(sample.get(key), wanted, key)
            for key in ("rss_max_bytes_before", "rss_max_bytes_after"):
                require(type(sample.get(key)) is int and sample[key] >= 0, "invalid RSS integer")
            valid = timing_accepted(sample)
            exact(sample.get("timing_accepted"), valid, "noise/floor predicate")
            if name == "samples":
                all_valid &= valid
    exact(receipt.get("status"), "MEASURED" if all_valid else "INCONCLUSIVE_TIMING", "cell status")
    exact(code, 0 if all_valid else 78, "cell exit")
    return all_valid


def summarize(cells):
    """Every planned cell stays visible; unqualified ratios are null, never rescued."""
    require(type(cells) is list and len(cells) == len(plan()["order"]), "incomplete collection cannot summarize")
    for expected, result in zip(plan()["order"], cells):
        exact({key: result.get(key) for key in expected}, expected, "frozen global order")
        require(type(result.get("timing_accepted")) is bool, "missing timing acceptance")
        values = result.get("normalized_ns")
        require(type(values) is list and len(values) == SAMPLES and
                all(type(value) in (int, float) and math.isfinite(value) and
                    0 < value <= PROCESS_SECONDS * 1_000_000_000 / MATRIX[expected["cell"]][3]
                    for value in values),
                "invalid normalized sample set")
    summary = []
    for cell in range(len(MATRIX)):
        roles = {}
        for role in ROLES:
            pair = [item for item in cells if item["cell"] == cell and item["role"] == role]
            require(len(pair) == 2 and {item["direction"] for item in pair} == {"forward", "reverse"}, "pairing differs")
            if not all(item["timing_accepted"] for item in pair):
                roles[role] = {"status": "INCONCLUSIVE_TIMING"}
                continue
            medians = {item["direction"]: statistics.median(item["normalized_ns"]) for item in pair}
            spreads = {item["direction"]: {"minimum_ns": min(item["normalized_ns"]),
                       "maximum_ns": max(item["normalized_ns"]),
                       "nearest_rank_p95_ns": sorted(item["normalized_ns"])[math.ceil(.95 * SAMPLES) - 1]}
                       for item in pair}
            stable = (max(medians.values()) <= min(medians.values()) * 1.1 and
                      all(spreads[name]["nearest_rank_p95_ns"] <= medians[name] * 1.25 for name in medians))
            roles[role] = {"status": "PAIRED" if stable else "INCONCLUSIVE_SPREAD_OR_ORDER",
                           "direction_median_ns": medians, "within_direction_spread": spreads,
                           "paired_midpoint_ns": sum(medians.values()) / 2,
                           "direction_range_ns": [min(medians.values()), max(medians.values())]}
        comparisons = {}
        for label, numerator, denominator in COMPARISONS:
            eligible = roles[numerator]["status"] == roles[denominator]["status"] == "PAIRED"
            comparisons[label] = {
                "status": "PAIRED" if eligible else "INCONCLUSIVE",
                "primary": cell == PRIMARY_CELL and label == PRIMARY_COMPARISON,
                "numerator": numerator, "denominator": denominator,
                "ratio_of_midpoint_medians": roles[numerator]["paired_midpoint_ns"] / roles[denominator]["paired_midpoint_ns"] if eligible else None,
                "direction_time_ratios": {direction: roles[numerator]["direction_median_ns"][direction] /
                                           roles[denominator]["direction_median_ns"][direction]
                                           for direction in ("forward", "reverse")} if eligible else None}
        summary.append({"cell": cell, "mkn": list(MATRIX[cell][:3]), "roles": roles, "comparisons": comparisons})
    return summary
