"""Frozen P1 plan and receipt checks; no compiler or benchmark starts here."""
from functools import lru_cache
import hashlib
import json
import math
import statistics

VERSION = 1
SEED = 0x4D44534C50310001
ROLES = ("stock", "tiles-3x5", "tiles-5x3")
TILES = {"stock": None, "tiles-3x5": (3, 5), "tiles-5x3": (5, 3)}
# Explicit M,K,N order, followed by fixed calls per aggregate. Never autotuned.
MATRIX = ((32, 32, 32, 128), (33, 35, 37, 128), (64, 7, 19, 256),
          (1, 257, 65, 256), (65, 257, 1, 256), (128, 128, 128, 8))
WARMUPS, SAMPLES, FLOOR_NS, INPUTS = 2, 9, 1_000_000, 4
PROCESS_SECONDS, COLLECTION_SECONDS = 20, 180
LIVE_PAYLOAD_CAP, RAW_CAP, ARTIFACT_CAP = 64 << 20, 64 << 20, 256 << 20
SCOPE = "source-region-call-return+Result-move;retained-ownership;release-excluded"
CONTROLS = ("integer-rectangle", "non-fma", "increasing-k", "subnormal-input",
            "underflow-rne", "signed-zero", "overflow-order", "infinity", "nan",
            "zero-m", "zero-n", "zero-k", "input-output-alias",
            "owning-observation-survives-next-call", "shape-failure-before-publication",
            "capacity-failure-before-publication", "caller-fp-restore")
ISSUED_HASHES = ("semantic_sha256", "structured_sha256", "bufferized_sha256",
                 "transform_sha256", "scheduled_sha256", "llvm_sha256",
                 "object_sha256", "issuer_manifest_sha256")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key " + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError("nonfinite JSON constant " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(type(value) is dict, "receipt must be one object")
    return value


def exact(value, wanted, label):
    require(type(value) is type(wanted), "wrong type for " + label)
    if type(wanted) in (list, tuple):
        require(len(value) == len(wanted), "wrong length for " + label)
        for i, item in enumerate(wanted):
            exact(value[i], item, label + f"[{i}]")
    elif type(wanted) is dict:
        require(set(value) == set(wanted), "wrong fields for " + label)
        for key in wanted:
            exact(value[key], wanted[key], label + "." + key)
    else:
        require(value == wanted, "wrong " + label)


def plan():
    forward = [{"direction": "forward", "cell": cell, "role": role}
               for cell in range(len(MATRIX)) for role in ROLES]
    reverse = [{**entry, "direction": "reverse"} for entry in reversed(forward)]
    return {"schema": "matcore.source-pattern.p1.plan", "version": VERSION,
            "matrix_mkn_repetitions": MATRIX, "roles": ROLES, "tiles": TILES,
            "warmup_aggregates": WARMUPS, "measured_aggregates": SAMPLES,
            "timer_floor_ns": FLOOR_NS, "input_sequence": INPUTS, "seed": SEED,
            "timing_scope": SCOPE, "release_inside_timing": False,
            "chooser_inside_timing": True, "owner_vector_reserved_before_timing": True,
            "noise_policy": {"max_involuntary_switches": 0, "minimum_thread_cpu_per_wall": "9/10",
                             "thread_clock_boundary_slack_ns": 100_000,
                             "load1_per_logical_cpu_max": "1/4", "observed_cpu_temperature_max_millicelsius": 85000,
                             "known_build_test_processes_max": 0,
                             "sample_p95_over_median_max": "5/4", "forward_reverse_median_ratio_max": "11/10"},
            "cache_policy": "loaded-file audit before process; deterministic input preparation; two verified retention warmups; verification/release between samples; no physical-cache guarantee",
            "raw_delivery": "explicit new external directory outside every Git worktree",
            "process_seconds": PROCESS_SECONDS, "collection_seconds": COLLECTION_SECONDS,
            "live_payload_cap_bytes": LIVE_PAYLOAD_CAP, "raw_cap_bytes": RAW_CAP,
            "generated_artifact_cap_bytes": ARTIFACT_CAP, "order": forward + reverse,
            "initial_and_final_controls": list(ROLES), "winner_policy": "none"}


def integer_bits(value):
    if value == 0:
        return 0
    magnitude = abs(value)
    require(magnitude < (1 << 24), "integer oracle not exactly representable")
    exponent = magnitude.bit_length() - 1
    return ((0x80000000 if value < 0 else 0) | ((exponent + 127) << 23) |
            ((magnitude << (23 - exponent)) & 0x7fffff))


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
                output.append(integer_bits(total))
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
    exact(receipt.get("schema"), "matcore.source-pattern.p1", "receipt schema")
    exact(receipt.get("version"), VERSION, "receipt version")
    exact(receipt.get("numerics"), "strict_f32", "numerical profile")
    exact(receipt.get("timing_scope"), SCOPE, "timing scope")
    exact(receipt.get("release_inside_timing"), False, "release policy")
    exact(receipt.get("chooser_inside_timing"), True, "input chooser policy")
    exact(receipt.get("implementation_thread_contract"), 1, "single-thread implementation contract")
    exact(receipt.get("continuous_thread_proof"), False, "thread-evidence limitation")
    exact(receipt.get("seed"), SEED, "input seed")
    exact(receipt.get("input_sequence"), INPUTS, "input sequence")


def check_controls(receipt, code):
    check_common(receipt)
    exact(code, 0, "control exit")
    exact(receipt.get("mode"), "verify-only", "control mode")
    exact(receipt.get("status"), "CORRECTNESS_PASSED", "control status")
    exact(receipt.get("controls"), list(CONTROLS), "complete ordered control set")
    exact(receipt.get("failures"), 0, "control failures")
    require(type(receipt.get("checks")) is int and receipt["checks"] >= 100, "controls did not check contents")


def check_cell(receipt, code, cell, cpu):
    check_common(receipt)
    exact(receipt.get("mode"), "measure", "measure mode")
    exact(receipt.get("cell"), cell, "requested cell")
    m, k, n, repetitions = MATRIX[cell]
    for key, wanted in (("m", m), ("k", k), ("n", n), ("repetitions", repetitions),
                        ("warmup_aggregates", WARMUPS), ("timer_floor_ns", FLOOR_NS),
                        ("checked_timed_calls", repetitions * SAMPLES),
                        ("checked_timed_elements", repetitions * SAMPLES * m * n),
                        ("checked_warmup_calls", repetitions * WARMUPS),
                        ("retained_output_payload_bytes", repetitions * m * n * 4)):
        exact(receipt.get(key), wanted, key)
    size = receipt.get("sizeof_result")
    require(type(size) is int and 1 <= size <= 1024, "Result size missing")
    exact(receipt.get("live_payload_bound_bytes"), payload_bound(cell, size), "live-payload bound")
    require(receipt["live_payload_bound_bytes"] <= LIVE_PAYLOAD_CAP, "live-payload cap")
    for key in ("process_task_count_before", "process_task_count_after"):
        exact(receipt.get(key), 1, key)
    for key in ("process_affinity_before", "process_affinity_after"):
        exact(receipt.get(key), [cpu], key)
    modules = receipt.get("loaded_modules")
    require(type(modules) is list and modules and modules == sorted(set(modules)) and
            all(type(path) is str and path.startswith("/") for path in modules), "loaded-module inventory")
    all_valid = True
    aggregates = []
    for name, count in (("warmups", WARMUPS), ("samples", SAMPLES)):
        group = receipt.get(name)
        require(type(group) is list and len(group) == count, "missing/extra aggregate " + name)
        aggregates.extend((name, ordinal, sample) for ordinal, sample in enumerate(group))
    for group, ordinal, sample in aggregates:
        require(type(sample) is dict, "sample is not an object")
        for key, wanted in (("ordinal", ordinal), ("calls", repetitions),
                            ("verified_calls", repetitions), ("verified_elements", repetitions * m * n),
                            ("output_hash", expected_output_hash(cell)), ("task_count_before", 1),
                            ("task_count_after", 1), ("cpu_before", cpu), ("cpu_after", cpu)):
            exact(sample.get(key), wanted, key)
        exact(sample.get("affinity_before"), [cpu], "before affinity")
        exact(sample.get("affinity_after"), [cpu], "after affinity")
        for key in ("wall_ns", "thread_cpu_ns", "involuntary_switches", "voluntary_switches",
                    "rss_max_bytes_before", "rss_max_bytes_after"):
            require(type(sample.get(key)) is int and sample[key] >= 0, "invalid integer sample " + key)
        wall, thread = sample["wall_ns"], sample["thread_cpu_ns"]
        require(wall > 0 and thread > 0, "zero timing")
        valid = (wall >= FLOOR_NS and thread * 10 >= wall * 9 and
                 thread <= wall + 100_000 and sample["involuntary_switches"] == 0)
        exact(sample.get("timing_accepted"), valid, "noise/floor predicate")
        if group == "samples":
            all_valid &= valid
    exact(receipt.get("status"), "MEASURED" if all_valid else "INCONCLUSIVE_TIMING", "cell status")
    exact(code, 0 if all_valid else 78, "cell exit")
    return all_valid


def summarize(cells):
    """Paired per-shape medians; no winning role or policy is emitted."""
    require(len(cells) == len(plan()["order"]), "incomplete collection cannot summarize")
    for entry, result in zip(plan()["order"], cells):
        exact({key: result[key] for key in entry}, entry, "frozen global order")
        values = result.get("normalized_ns")
        require(type(values) is list and len(values) == SAMPLES and
                all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
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
            require(all(math.isfinite(value) and value > 0 for value in medians.values()), "invalid normalized timing")
            spreads = {item["direction"]: {"minimum_ns": min(item["normalized_ns"]),
                       "maximum_ns": max(item["normalized_ns"]),
                       "nearest_rank_p95_ns": sorted(item["normalized_ns"])[math.ceil(.95 * SAMPLES) - 1]}
                       for item in pair}
            stable = (max(medians.values()) <= min(medians.values()) * 1.1 and
                      all(spreads[name]["nearest_rank_p95_ns"] <= medians[name] * 1.25 for name in medians))
            roles[role] = {"status": "PAIRED" if stable else "INCONCLUSIVE_SPREAD_OR_ORDER", "direction_median_ns": medians,
                           "within_direction_spread": spreads,
                           "paired_midpoint_ns": sum(medians.values()) / 2,
                           "direction_range_ns": [min(medians.values()), max(medians.values())]}
        baseline = roles["stock"].get("paired_midpoint_ns") if roles["stock"]["status"] == "PAIRED" else None
        for role in ROLES[1:]:
            value = roles[role].get("paired_midpoint_ns")
            roles[role]["ratio_of_midpoint_medians_to_stock"] = value / baseline if value and baseline and roles[role]["status"] == "PAIRED" else None
            roles[role]["direction_time_ratio_to_stock"] = {
                direction: roles[role]["direction_median_ns"][direction] / roles["stock"]["direction_median_ns"][direction]
                for direction in ("forward", "reverse")
            } if "direction_median_ns" in roles[role] and "direction_median_ns" in roles["stock"] else None
        summary.append({"cell": cell, "mkn": list(MATRIX[cell][:3]), "roles": roles})
    return summary
