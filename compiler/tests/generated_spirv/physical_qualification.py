#!/usr/bin/env python3
"""Bounded, explicit-device Vulkan probe receipts; never a benchmark/runtime.

Run only after independent issuer/SPIRV-Tools gates. --expect-refusal is a
separate no-dispatch campaign, not a way to skip a required physical sample.
--self-test uses synthetic receipts only and never starts the probe or Vulkan.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SHAPES = ((2, 3, 4), (1, 1, 1), (1, 7, 9), (9, 7, 1), (3, 5, 7),
          (17, 31, 13), (65, 3, 67), (65535, 1, 1), (1, 1, 65535),
          (3, 2, 5), (1, 257, 1), (5, 1, 7))
# Independent analytic constants, not host floating-point evaluations.
SPECIAL = {
    "subnormal-input": 1, "negative-subnormal": 0x80000001,
    "subnormal-product": 0x00400000, "inexact-underflow-rne": 0x00400002,
    "negative-underflow-rne": 0x80400002, "signed-zero-seed": 0,
    "non-fma": 0, "increasing-k": 0, "overflow-order": 0x7f800000,
    "rte-not-rtz": 0x3f800001, "tie-even-low": 0x3f800000,
    "tie-even-high": 0x3f800002, "positive-infinity": 0x7f800000,
    "negative-infinity": 0xff800000, "infinity-times-zero": None,
    "opposed-infinities": None, "quiet-nan": None,
}
BASE = ("nonuniform-rectangular-integers", "last-k-only-row-column")
COUNTERS = ("dispatches", "checked_completions", "output_checks",
            "input_immutability_checks", "canary_checks", "mismatches")
HASH = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FILES = (
    "compiler/tests/generated_spirv/physical_qualification.py",
    "compiler/tests/generated_spirv/vulkan_probe.cpp",
    "compiler/tests/generated_spirv/strict_oracle.h",
    "compiler/tests/generated_spirv/CMakeLists.txt",
    "compiler/lib/mlir/MatcoreSpirvGemmCandidate.h",
    "compiler/lib/mlir/MatcoreSpirvGemmCandidate.cpp",
    "compiler/lib/mlir/MatcoreCpuGemmCandidate.h",
    "compiler/lib/mlir/MatcoreCpuGemmCandidate.cpp",
)
CAP = 1 << 20
TIMEOUT = 20
FP_CONTROLS = ("denorm_preserve_f32", "rne_f32", "signed_zero_inf_nan_f32")
CONTROL_REFUSAL = "selected device lacks Vulkan1.2/DenormPreserve/RTE/SignedZeroInfNan f32 controls"
PHYSICAL_REFUSAL = "positive sample requires selected physical AMD RADV device"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_int(value, expected, name):
    require(type(value) is int and value == expected, f"{name}: expected {expected}, got {value!r}")


def vector3(value, expected, name):
    require(type(value) is list and len(value) == 3, name + ": expected three integer axes")
    for i, wanted in enumerate(expected):
        exact_int(value[i], wanted, f"{name}[{i}]")


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    def constant(value):
        raise RuntimeError("nonfinite JSON constant: " + value)

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    require(type(value) is dict, "receipt must be one JSON object")
    return value


def common(receipt):
    require(receipt.get("source_runtime_authority") is False, "source/runtime authority must be false")
    require(receipt.get("strict_contract_qualified") is False, "bounded sample is not full qualification")
    require(receipt.get("process_only_qualification") is True, "process-only scope missing")
    require(receipt.get("oracle_uses_host_fp_state") is False, "host-FP-dependent oracle")
    require(receipt.get("nan_policy") == "NaN class only; all other results bit-exact", "NaN comparison policy")
    for key in COUNTERS:
        require(type(receipt.get(key)) is int and receipt[key] >= 0, "missing/noninteger counter " + key)


def no_dispatch(receipt):
    common(receipt)
    for key in COUNTERS:
        exact_int(receipt[key], 0, key)


def software(receipt, code):
    exact_int(code, 0, "exit")
    no_dispatch(receipt)
    require(receipt.get("classification") == "ORACLE_SOFTWARE_ONLY" and receipt.get("status") == "MATCHED", "software-oracle status")
    require(receipt.get("vulkan_loaded") is False, "software mode loaded Vulkan")
    exact_int(receipt.get("oracle_self_controls"), 19, "oracle controls")
    exact_int(receipt.get("oracle_case_count"), 19, "oracle cases")
    require("devices" not in receipt and "fill_sha256" not in receipt, "software mode issued/enumerated")


def devices(receipt):
    values = receipt.get("devices")
    require(type(values) is list, "device inventory missing")
    for index, device in enumerate(values):
        require(type(device) is dict, "device inventory entry")
        exact_int(device.get("index"), index, "device index")
        for key in ("device_uuid", "driver_uuid"):
            require(type(device.get(key)) is str and re.fullmatch(r"[0-9a-f]{32}", device[key]), "invalid " + key)
    return values


def inventory(receipt, code, selected):
    exact_int(code, 0, "inventory exit")
    no_dispatch(receipt)
    require(receipt.get("classification") == "INVENTORY" and receipt.get("status") == "NO_DISPATCH", "inventory status")
    require(receipt.get("expectations_frozen_before_vulkan_load") is True, "oracle not frozen before loader")
    exact_int(receipt.get("oracle_self_controls"), 19, "inventory oracle controls")
    values = devices(receipt)
    require(selected < len(values), "explicit selected device absent; never choose another index")
    return values[selected]


def selected_device(receipt, index, expected):
    exact_int(receipt.get("selected_device_index"), index, "selected device")
    values = devices(receipt)
    require(index < len(values) and values[index] == expected, "selected device identity/properties changed from inventory")


def target_refusal_reason(device):
    # Mirror the probe's ordered device gates. A known reason string alone is
    # not evidence for a refusal contradicted by the selected inventory record.
    for key in ("api_version", "vendor_id", "driver_id", "device_type"):
        require(type(device.get(key)) is int and 0 <= device[key] <= 0xffffffff,
                "missing/noninteger device property " + key)
    for key in FP_CONTROLS:
        require(type(device.get(key)) is bool, "missing/nonboolean device control " + key)
    if device["api_version"] < ((1 << 22) | (2 << 12)) or not all(device[key] for key in FP_CONTROLS):
        return CONTROL_REFUSAL
    if device["vendor_id"] != 0x1002 or device["driver_id"] != 3 or device["device_type"] not in (1, 2):
        return PHYSICAL_REFUSAL
    return None


def empty_stderr(byte_count):
    exact_int(byte_count, 0, "probe stderr diagnostic bytes")


def integer_bits(value):
    if not value:
        return 0
    magnitude = abs(value)
    require(magnitude < (1 << 24), "analytic integer not exactly representable")
    exponent = magnitude.bit_length() - 1
    return ((0x80000000 if value < 0 else 0) | ((exponent + 127) << 23) |
            ((magnitude << (23 - exponent)) & 0x7fffff))


def analytic_hash(shape, last=False):
    m, k, n = shape
    digest = hashlib.sha256()
    for i in range(m):
        for j in range(n):
            value = ((i % 7 + 1) * (j % 11 + 1) if last else
                     sum(((17 * i + 5 * p) % 19 - 9) * ((7 * p + 11 * j) % 23 - 11)
                         for p in range(k)))
            digest.update(integer_bits(value).to_bytes(4, sys.byteorder))
    return digest.hexdigest()


def physical(receipt, code, shape, index, device):
    exact_int(code, 0, "physical exit")
    common(receipt)
    require(receipt.get("classification") == "BOUNDED_PHYSICAL_SAMPLE" and receipt.get("status") == "MATCHED", "not a matched bounded physical sample")
    selected_device(receipt, index, device)
    exact_int(device.get("vendor_id"), 0x1002, "AMD vendor")
    exact_int(device.get("driver_id"), 3, "RADV driver")
    require(type(device.get("device_type")) is int and device["device_type"] in (1, 2), "positive claim requires physical AMD device")
    require(type(device.get("api_version")) is int and device["api_version"] >= ((1 << 22) | (2 << 12)), "Vulkan1.2 device required")
    for key in ("denorm_preserve_f32", "rne_f32", "signed_zero_inf_nan_f32"):
        require(device.get(key) is True, "missing device control " + key)
    vector3(receipt.get("shape_mkn"), shape, "requested shape")
    require(receipt.get("expectations_frozen_before_vulkan_load") is True and
            receipt.get("artifact_replay_verified_before_vulkan") is True, "freeze/replay boundary missing")
    exact_int(receipt.get("oracle_self_controls"), 19, "physical oracle controls")
    m, k, n = shape
    names = [*BASE, *(SPECIAL if k >= 3 else ())]
    cases = receipt.get("cases")
    require(type(cases) is list and [c.get("case") for c in cases] == names, "exact ordered case inventory differs")
    skips = [] if k >= 3 else [{"case": name, "reason": "requires K>=3"} for name in SPECIAL]
    require(receipt.get("skipped_cases") == skips, "special-case skip inventory differs")
    require(receipt.get("special_cases") == ("APPLICABLE_K_GE_3" if k >= 3 else "SKIPPED_K_LT_3; base/last-K still executed"), "special-case applicability label")
    for key in ("fill_sha256", "gemm_sha256"):
        require(type(receipt.get(key)) is str and HASH.fullmatch(receipt[key]), "invalid shader digest " + key)
    require(receipt["fill_sha256"] != receipt["gemm_sha256"], "fill/GEMM byte identities unexpectedly equal")
    vector3(receipt.get("fill_dispatch"), (m, n, 1), "fill dispatch")
    vector3(receipt.get("gemm_dispatch"), (m, n, 1), "GEMM dispatch")
    vector3(receipt.get("local_size"), (1, 1, 1), "local size")
    actual_limit = receipt.get("max_dispatch")
    require(type(actual_limit) is list and len(actual_limit) == 3 and
            all(type(limit) is int and limit >= wanted for limit, wanted in zip(actual_limit, (m, n, 1))), "actual dispatch limit")
    storage_range = receipt.get("max_storage_buffer_range")
    require(type(storage_range) is int and storage_range >= max(m * k, k * n, m * n) * 4, "actual descriptor range")
    require(receipt.get("staging") == "HOST_VISIBLE|HOST_COHERENT only; three disjoint allocations", "staging contract")
    alignment = receipt.get("min_storage_buffer_offset_alignment")
    require(type(alignment) is int and 0 < alignment <= 65536 and alignment & (alignment - 1) == 0, "storage-buffer alignment")
    aligned = max(4, alignment)
    offset = ((256 + aligned - 1) // aligned) * aligned
    exact_int(receipt.get("output_descriptor_offset"), offset, "output guard offset")
    count = len(names)
    operations = count * m * n * (2 * k + 1)
    require(operations <= CAP, "wrapper selected an overwork shape")
    exact_int(receipt.get("test_scalar_operations"), operations, "total scalar operation count")
    exact_int(receipt.get("test_scalar_operation_cap"), CAP, "test work cap")
    expected_counts = (2 * count, count, count * m * n, count * (m * k + k * n),
                       count * (offset // 4 + 64), 0)
    for name, expected in zip(COUNTERS, expected_counts):
        exact_int(receipt[name], expected, name)
    known = {BASE[0]: analytic_hash(shape), BASE[1]: analytic_hash(shape, last=True)}
    for name, value in SPECIAL.items():
        if value is not None:
            known[name] = hashlib.sha256(value.to_bytes(4, sys.byteorder) * (m * n)).hexdigest()
    for case in cases:
        exact_int(case.get("outputs"), m * n, case["case"] + " outputs")
        exact_int(case.get("mismatches"), 0, case["case"] + " mismatches")
        require(case.get("first_output_mismatch") is None, "recorded output counterexample")
        digest = case.get("expected_sha256")
        require(type(digest) is str and HASH.fullmatch(digest), "missing frozen expectation digest")
        if case["case"] in known:
            require(digest == known[case["case"]], "analytic expected bits differ: " + case["case"])


def refusal(receipt, code, before_loader=False, index=None, device=None):
    exact_int(code, 78, "refusal exit")
    no_dispatch(receipt)
    require(receipt.get("classification") == "REFUSED" and receipt.get("status") == "NO_FALLBACK", "wrong refusal classification")
    require(type(receipt.get("reason")) is str and receipt["reason"], "refusal explanation missing")
    if before_loader:
        for field in ("expectations_frozen_before_vulkan_load", "artifact_replay_verified_before_vulkan",
                      "devices", "loader_api_version", "fill_sha256", "gemm_sha256"):
            require(field not in receipt, "invalid/overwork CLI crossed pre-loader boundary: " + field)
    else:
        selected_device(receipt, index, device)
        require(receipt.get("expectations_frozen_before_vulkan_load") is True and
                receipt.get("artifact_replay_verified_before_vulkan") is True, "target refusal did not reach checked device gate")
        expected = target_refusal_reason(device)
        require(expected is not None and receipt["reason"] == expected,
                "target refusal reason is not justified by the selected device properties")


def identity(path):
    before = path.stat()
    require(path.is_file(), "not a regular file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed while hashing: " + str(path))
    return {"sha256": digest.hexdigest(), "bytes": after.st_size, "device": after.st_dev,
            "inode": after.st_ino, "mtime_ns": after.st_mtime_ns}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


class Campaign:
    def __init__(self, output, probe, root, report):
        self.output, self.probe, self.root, self.report = output, probe, root, report
        self.probe_identity = identity(probe)
        self.sources = {name: identity(root / name) for name in SOURCE_FILES}
        report.update(probe={"path": str(probe), **self.probe_identity}, sources_before=self.sources,
                      source_binding="explicit source snapshot; not build-origin attestation")

    def run(self, label, arguments, validate):
        directory = self.output / f"{len(self.report['commands']):03d}-{label}"
        directory.mkdir()
        command = [str(self.probe), *map(str, arguments)]
        record = {"argv": command, "cwd": str(self.root), "started_utc": utc(),
                  "timeout_seconds": TIMEOUT, "returncode": None, "timed_out": False,
                  "validated": False, "artifact_directory": directory.name}
        self.report["commands"].append(record)
        try:
            require(identity(self.probe) == self.probe_identity, "probe changed before invocation")
            with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                try:
                    result = subprocess.run(command, cwd=self.root, stdout=stdout, stderr=stderr,
                                            timeout=TIMEOUT, check=False,
                                            env={**os.environ, "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"})
                    record["returncode"] = result.returncode
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                    raise RuntimeError("probe timeout; no retry or fallback")
            require(identity(self.probe) == self.probe_identity, "probe changed during invocation")
            stdout_path = directory / "stdout.log"
            require(stdout_path.stat().st_size <= 4 * 1024 * 1024, "receipt unexpectedly exceeds 4MiB; inspect raw log")
            receipt = strict_json(stdout_path.read_bytes())
            write_json(directory / "receipt.json", receipt)
            # Keep both raw logs and the decoded receipt on failure, but never
            # let diagnostics (including recoverable sanitizer errors) pass as
            # an otherwise successful or expected-refusal probe invocation.
            empty_stderr((directory / "stderr.log").stat().st_size)
            validate(receipt, record["returncode"])
            record["validated"] = True
            return receipt
        except Exception as error:
            record["validation_error"] = str(error)
            raise
        finally:
            record["ended_utc"] = utc()
            write_json(directory / "command.json", record)

    def finish_identities(self):
        self.report["probe_after"] = identity(self.probe)
        self.report["sources_after"] = {name: identity(self.root / name) for name in SOURCE_FILES}
        require(self.report["probe_after"] == self.probe_identity, "probe identity changed")
        require(self.report["sources_after"] == self.sources, "source snapshot changed during campaign")


def negatives(device):
    shapes = ("0,3,4", "2,0,4", "2,3,0", "-1,3,4", "65536,1,1", "1,1,65536",
              "1,33554433,1", "4096,1,8193", "9223372036854775807,9223372036854775807,2",
              "18446744073709551616,1,1", "2,3", "2,3,4,5", "2,3,x", "2,,4",
              "17,257,17", "1,33554432,1", "4096,1,8192")
    commands = [("invalid-shape-" + str(i), ["--device", device, "--shape", shape])
                for i, shape in enumerate(shapes)]
    return [*commands, ("unknown-shader-option", ["--device", device, "--shape", "2,3,4", "--shader", "untrusted.spv"]),
            ("duplicate-device", ["--device", device, "--device", device, "--shape", "2,3,4"]),
            ("mixed-inventory", ["--inventory", "--device", device]),
            ("missing-shape", ["--device", device])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-refusal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any((args.probe, args.device is not None, args.output, args.expect_refusal)):
            parser.error("--self-test is standalone and never runs the probe")
        return self_test()
    if args.probe is None or args.output is None or args.device is None or not 0 <= args.device <= 0xffffffff:
        parser.error("--probe PATH --device N --output NEWDIR required; N is an unsigned device index")
    probe, root = args.probe.resolve(strict=True), Path(__file__).resolve().parents[3]
    require(os.access(probe, os.X_OK), "probe is not executable")
    output = args.output.absolute()
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        parser.error("output already exists; no-overwrite campaign refused")
    report = {"schema": "matcore-test-vulkan-physical-qualification-v1", "status": "INCOMPLETE",
              "classification": "INCOMPLETE_NO_QUALIFICATION", "source_runtime_authority": False,
              "full_issuer_envelope_physically_qualified": False, "performance_claim": False,
              "upstream_validation_gate_owner": "integration owner; required separately before execution",
              "selected_device_index": args.device, "expect_refusal": args.expect_refusal,
              "started_utc": utc(), "commands": [], "shapes": [], "word_byte_order": sys.byteorder}
    campaign = None
    try:
        campaign = Campaign(output, probe, root, report)
        for name in SOURCE_FILES:
            target = output / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((root / name).read_bytes())
            require(identity(target)["sha256"] == report["sources_before"][name]["sha256"], "source changed while copying")
        for flag in ("HEAD", "--show-toplevel"):
            argv = ["git", "rev-parse", flag]
            result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=TIMEOUT)
            report.setdefault("source_identity_commands", []).append(
                {"argv": argv, "cwd": str(root), "returncode": result.returncode,
                 "stdout": result.stdout, "stderr": result.stderr, "timeout_seconds": TIMEOUT})
            require(result.returncode == 0, "git source identity unavailable")
            report.setdefault("git", {})[flag] = result.stdout.strip()
        campaign.run("oracle-only", ["--oracle-only"], software)
        for label, argv in negatives(args.device):
            campaign.run(label, argv, lambda receipt, code: refusal(receipt, code, before_loader=True))
        listed = campaign.run("inventory", ["--inventory"], lambda receipt, code: inventory(receipt, code, args.device))
        device = devices(listed)[args.device]
        report["selected_device"] = device
        shapes = ((2, 3, 4),) if args.expect_refusal else SHAPES
        for shape in shapes:
            arguments = ["--device", args.device, "--shape", ",".join(map(str, shape))]
            validate = (lambda receipt, code: refusal(receipt, code, index=args.device, device=device)) if args.expect_refusal else (
                lambda receipt, code, shape=shape: physical(receipt, code, shape, args.device, device))
            receipt = campaign.run("shape-" + "-".join(map(str, shape)), arguments, validate)
            report["shapes"].append({"shape_mkn": list(shape), "classification": receipt["classification"],
                                     "shader_sha256": {kind: receipt[kind + "_sha256"] for kind in ("fill", "gemm")},
                                     "dispatches": receipt["dispatches"], "checked_completions": receipt["checked_completions"]})
        report["status"] = "PASSED"
        report["classification"] = "TARGET_REFUSAL_ONLY" if args.expect_refusal else "BOUNDED_PHYSICAL_SAMPLES"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
    finally:
        if campaign is not None:
            try:
                campaign.finish_identities()
            except Exception as error:
                report.update(status="FAILED", classification="INCOMPLETE_NO_QUALIFICATION", identity_error=str(error))
        if report["status"] != "PASSED":
            report["classification"] = "INCOMPLETE_NO_QUALIFICATION"
        report["ended_utc"] = utc()
        report["artifacts"] = {str(path.relative_to(output)): {"sha256": identity(path)["sha256"], "bytes": path.stat().st_size}
                               for path in sorted(output.rglob("*")) if path.is_file()}
        report["artifact_catalog_excludes"] = "summary.json (this document); summary digest printed to stdout"
        write_json(output / "summary.json", report)
    summary = {"status": report["status"], "classification": report["classification"],
               "source_runtime_authority": False, "validated_commands": sum(c["validated"] for c in report["commands"]),
               "shapes": len(report["shapes"]), "summary": str(output / "summary.json"),
               "summary_sha256": identity(output / "summary.json")["sha256"]}
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 1


def self_test():
    """Synthetic parser/validator falsifiers, not probe or physical evidence."""
    basic = dict(source_runtime_authority=False, strict_contract_qualified=False,
                 process_only_qualification=True, oracle_uses_host_fp_state=False,
                 nan_policy="NaN class only; all other results bit-exact", **dict.fromkeys(COUNTERS, 0))
    receipt = {**basic, "classification": "REFUSED", "status": "NO_FALLBACK", "reason": "synthetic control"}
    refusal(receipt, 78, before_loader=True)
    checks = 1
    for key, value in (("dispatches", 1), ("checked_completions", True), ("source_runtime_authority", True),
                       ("expectations_frozen_before_vulkan_load", True), ("devices", []), ("fill_sha256", "0" * 64)):
        bad = {**receipt, key: value}
        try:
            refusal(bad, 78, before_loader=True)
        except RuntimeError:
            checks += 1
        else:
            raise RuntimeError("synthetic refusal mutation was accepted: " + key)
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'{} trailing'):
        try:
            strict_json(raw)
        except (RuntimeError, ValueError):
            checks += 1
        else:
            raise RuntimeError("malformed JSON accepted")
    require(all((2 + (len(SPECIAL) if k >= 3 else 0)) * m * n * (2 * k + 1) <= CAP for m, k, n in SHAPES), "planned shape exceeds cap")
    checks += 1
    require(integer_bits(1) == 0x3f800000 and integer_bits(-1) == 0xbf800000 and integer_bits(123) == 0x42f60000, "integer bit encoder")
    checks += 1
    device = dict(index=0, device_uuid="1" * 32, driver_uuid="2" * 32, vendor_id=0x1002,
                  driver_id=3, device_type=1, api_version=(1 << 22) | (3 << 12),
                  denorm_preserve_f32=True, rne_f32=True, signed_zero_inf_nan_f32=True)
    require(target_refusal_reason(device) is None, "supported AMD device must not justify target refusal")
    empty_stderr(0)
    checks += 2
    for size in (1, True, -1, "0"):
        try:
            empty_stderr(size)
        except RuntimeError:
            checks += 1
        else:
            raise RuntimeError("synthetic nonempty/invalid stderr size accepted")

    def target_fixture(selected, reason):
        return {**basic, "classification": "REFUSED", "status": "NO_FALLBACK", "reason": reason,
                "devices": [copy.deepcopy(selected)], "selected_device_index": 0,
                "expectations_frozen_before_vulkan_load": True,
                "artifact_replay_verified_before_vulkan": True}

    valid_targets = [({key: False}, CONTROL_REFUSAL) for key in FP_CONTROLS]
    valid_targets += [({"api_version": (1 << 22) | (1 << 12)}, CONTROL_REFUSAL),
                      ({"vendor_id": 0x10de}, PHYSICAL_REFUSAL),
                      ({"driver_id": 4}, PHYSICAL_REFUSAL),
                      ({"device_type": 4}, PHYSICAL_REFUSAL)]
    for updates, reason in valid_targets:
        selected = {**device, **updates}
        refusal(target_fixture(selected, reason), 78, index=0, device=selected)
        checks += 1
    invalid_targets = (
        ("supported AMD claims missing FP", {}, CONTROL_REFUSAL),
        ("supported AMD claims wrong device", {}, PHYSICAL_REFUSAL),
        ("missing FP claims later device gate", {"denorm_preserve_f32": False, "vendor_id": 0x10de}, PHYSICAL_REFUSAL),
        ("wrong device claims missing FP", {"vendor_id": 0x10de}, CONTROL_REFUSAL),
        ("nonboolean FP property", {"denorm_preserve_f32": 0}, CONTROL_REFUSAL),
        ("missing FP property", {"denorm_preserve_f32": None}, CONTROL_REFUSAL),
        ("boolean Vulkan version", {"api_version": True}, CONTROL_REFUSAL),
    )
    for label, updates, reason in invalid_targets:
        selected = {**device, **updates}
        try:
            refusal(target_fixture(selected, reason), 78, index=0, device=selected)
        except RuntimeError:
            checks += 1
        else:
            raise RuntimeError("synthetic inconsistent target refusal accepted: " + label)

    def fixture(shape):
        m, k, n = shape
        names = [*BASE, *(SPECIAL if k >= 3 else ())]
        count = len(names)
        hashes = {BASE[0]: analytic_hash(shape), BASE[1]: analytic_hash(shape, last=True)}
        for name, value in SPECIAL.items():
            hashes[name] = (hashlib.sha256(value.to_bytes(4, sys.byteorder) * (m * n)).hexdigest()
                            if value is not None else "c" * 64)
        counters = (2 * count, count, count * m * n, count * (m * k + k * n), count * 128, 0)
        return {**basic, **dict(zip(COUNTERS, counters)), "classification": "BOUNDED_PHYSICAL_SAMPLE", "status": "MATCHED",
                "devices": [device], "selected_device_index": 0, "shape_mkn": list(shape),
                "oracle_self_controls": 19, "expectations_frozen_before_vulkan_load": True,
                "artifact_replay_verified_before_vulkan": True, "fill_sha256": "a" * 64, "gemm_sha256": "b" * 64,
                "fill_dispatch": [m, n, 1], "gemm_dispatch": [m, n, 1], "local_size": [1, 1, 1],
                "max_dispatch": [65535, 65535, 65535], "max_storage_buffer_range": 128 * 1024 * 1024,
                "staging": "HOST_VISIBLE|HOST_COHERENT only; three disjoint allocations",
                "min_storage_buffer_offset_alignment": 16, "output_descriptor_offset": 256,
                "test_scalar_operation_cap": CAP, "test_scalar_operations": count * m * n * (2 * k + 1),
                "special_cases": "APPLICABLE_K_GE_3" if k >= 3 else "SKIPPED_K_LT_3; base/last-K still executed",
                "skipped_cases": [] if k >= 3 else [{"case": name, "reason": "requires K>=3"} for name in SPECIAL],
                "cases": [{"case": name, "outputs": m * n, "mismatches": 0,
                           "first_output_mismatch": None, "expected_sha256": hashes[name]} for name in names]}

    for shape in ((2, 3, 4), (3, 2, 5), (1, 1, 1)):
        physical(fixture(shape), 0, shape, 0, device)
        checks += 1
    good = fixture((2, 3, 4))
    mutations = (
        ("output count", lambda r: r.update(output_checks=r["output_checks"] - 1)),
        ("input count", lambda r: r.update(input_immutability_checks=r["input_immutability_checks"] - 1)),
        ("canary count", lambda r: r.update(canary_checks=r["canary_checks"] - 1)),
        ("completion", lambda r: r.update(checked_completions=0)),
        ("mismatch", lambda r: r.update(mismatches=1)),
        ("counter bool", lambda r: r.update(mismatches=False)),
        ("shape bool", lambda r: r.update(local_size=[True, 1, 1])),
        ("shader identity", lambda r: r.update(fill_sha256=r["gemm_sha256"])),
        ("case inventory", lambda r: r["cases"].pop()),
        ("wrong expected bits", lambda r: r["cases"][0].update(expected_sha256="0" * 64)),
        ("skip", lambda r: r.update(skipped_cases=[{"case": "quiet-nan", "reason": "ignored"}])),
        ("guard offset", lambda r: r.update(output_descriptor_offset=0)),
        ("actual range", lambda r: r.update(max_storage_buffer_range=1)),
        ("actual dispatch", lambda r: r.update(max_dispatch=[1, 1, 1])),
        ("authority", lambda r: r.update(source_runtime_authority=True)),
        ("device identity", lambda r: r["devices"][0].update(driver_uuid="f" * 32)),
        ("device selection", lambda r: r.update(selected_device_index=1)),
        ("classification", lambda r: r.update(classification="COUNTEREXAMPLE")),
        ("host FP oracle", lambda r: r.update(oracle_uses_host_fp_state=True)),
        ("NaN policy", lambda r: r.update(nan_policy="tolerance-only")),
    )
    for label, mutate in mutations:
        bad = copy.deepcopy(good)
        mutate(bad)
        try:
            physical(bad, 0, (2, 3, 4), 0, device)
        except RuntimeError:
            checks += 1
        else:
            raise RuntimeError("synthetic physical mutation accepted: " + label)
    print(json.dumps({"classification": "SYNTHETIC_WRAPPER_SELF_TEST", "checks": checks,
                      "source_runtime_authority": False, "probe_started": False, "physical_execution": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
