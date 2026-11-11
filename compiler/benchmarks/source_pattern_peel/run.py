#!/usr/bin/env python3
"""Separate fixed P5 prepare/collect/audit; never modify or resume P1.

Preparation and collection require the separately qualified product checkpoint
and operator coordination. Inspection receipts are not hostile-origin execution
attestations, compiler inputs, a cache, or schedule-selection authority.
"""
import argparse
import hashlib
import math
from pathlib import Path
import platform
import re
import shlex
import shutil
import sys
import time
from types import SimpleNamespace

from support import p1, p1c, legacy, peel, ROOT, HERE, P1_HERE
import protocol as c

PREPARED_SCHEMA = "matcore.source-pattern.p5.prepared"
COLLECTION_SCHEMA = "matcore.source-pattern.p5.collection"
LEAVES = ("__matcore_strict_gemm_f32_v1", "_mlir_ciface___matcore_strict_gemm_f32_v1")
PRODUCT_VERSION = "MDSLC experimental regions: native Linux x86-64, Clang/MLIR 21.1.8, uninstrumented; API/ABI not frozen\n"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _role(role):
    c.require(role in c.ROLES, "unknown P5 role")
    flag, tiles = c.FLAGS[role], c.TILES[role]
    if flag is None:
        c.require(role == "stock" and tiles is None, "unclosed stock role")
        return None, None
    c.require(flag in ("--gemm-output-tiles", "--gemm-output-tiles-peel-n") and
              tiles in ((3, 5), (5, 3)), "unclosed requested realization")
    return (peel if flag == "--gemm-output-tiles-peel-n" else legacy), tiles


def _kinds(role):
    return ("timed", "control", "end_k") if c.FLAGS[role] == "--gemm-output-tiles-peel-n" else ("timed", "control")


def _source(root, kind):
    return {"timed": (root / "inputs/call_return.mdsl", "p1_pipeline"),
            "control": (root / "inputs/source_pattern/connected.mdsl", "source_pattern_pipeline"),
            "end_k": (root / "inputs/source_pattern_peel/end_k.mdsl", "source_peel_end_k")}[kind]


def _sources(root):
    return ((P1_HERE / "call_return.mdsl", root / "inputs/call_return.mdsl", True),
            (ROOT / "compiler/tests/source_pattern/connected.mdsl", root / "inputs/source_pattern/connected.mdsl", False),
            (ROOT / "compiler/tests/source_pattern/math_helper.h", root / "inputs/source_pattern/math_helper.h", False),
            (ROOT / "compiler/tests/source_pattern_peel/end_k.mdsl", root / "inputs/source_pattern_peel/end_k.mdsl", False))


def _helpers():
    return (HERE / "run.py", HERE / "support.py", HERE / "protocol.py", P1_HERE / "run.py",
            P1_HERE / "contract.py", P1_HERE / "call_return.mdsl",
            ROOT / "compiler/tests/source_pattern/run.py", ROOT / "compiler/tests/source_pattern_peel/run.py")


def _argv(driver, root, role, kind):
    source, region = _source(root, kind)
    argv = [str(driver), str(source), "--region", region, "--candidate", "generated-strict",
            "--optimization", "none", "-o", str(root / "bin" / (kind + "-" + role))]
    if c.FLAGS[role] is not None:
        argv += [c.FLAGS[role], ",".join(map(str, c.TILES[role]))]
    return argv


def _request(role):
    inspect, tiles = _role(role)
    return {"role": role, "flag": c.FLAGS[role], "tiles": None if tiles is None else list(tiles),
            "schema": None if inspect is None else
            "matcore-source-output-peel-n-v1" if inspect is peel else "matcore-source-output-pattern-v1"}


def _event(log, events, label, action):
    begin = len(log.records)
    result = action()
    events.append({"label": label, "indices": list(range(begin, len(log.records)))})
    return result


def _event_labels():
    labels = ["tool-modules:" + name for name in ("driver", "cxx", "linker", "cmake", "objdump", "size")]
    labels += ["driver-version"]
    for role in c.ROLES:
        for kind in _kinds(role):
            labels += ["compile:" + role + ":" + kind]
            if c.FLAGS[role] is not None:
                labels += ["inspect:" + role + ":" + kind]
            labels += ["dependencies:" + role + ":" + kind, "codegen:" + role + ":" + kind]
        labels += ["correctness:" + role + ":" + kind for kind in _kinds(role)]
    return labels + ["stock-identity:timed", "stock-identity:control", "stock-profile"]


def _candidate(instance, role):
    if c.FLAGS[role] is not None:
        return Path(instance["binary"] + ".mdslc-private") / instance["manifest"]["candidate_soname"]
    matches = [Path(path) for path in instance["loaded_modules"]
               if Path(path).name == "libmatcore_closed_candidates_isolated_v1.so"]
    c.require(len(matches) == 1, "stock role lacks its exact candidate owner")
    return matches[0]


def _codegen(log, candidate, tools):
    identity = log.pin(candidate)
    symbols = log.invoke([tools["nm"], "--defined-only", "--print-size", "--format=posix", candidate])
    c.exact(symbols.stderr, "", "codegen symbol stderr")
    sizes = {}
    for symbol in LEAVES:
        rows = [line.split() for line in symbols.stdout.splitlines()
                if line.split() and line.split()[0] == symbol]
        c.require(len(rows) == 1 and len(rows[0]) == 4 and rows[0][1] == "t" and
                  int(rows[0][3], 16) > 0, "codegen capture lacks unique local function " + symbol)
        sizes[symbol] = {"value_hex": rows[0][2], "bytes": int(rows[0][3], 16)}
    assembly = log.invoke([tools["objdump"], "-d", "--no-show-raw-insn",
                           "--disassemble-symbols=" + ",".join(LEAVES), candidate])
    c.exact(assembly.stderr, "", "disassembly stderr")
    c.require(all("<" + symbol + ">:" in assembly.stdout for symbol in LEAVES),
              "disassembly did not consume both actual local functions")
    sections = log.invoke([tools["size"], "--format=sysv", candidate])
    c.exact(sections.stderr, "", "section size stderr")
    c.require(".text" in sections.stdout, "missing actual DSO text section size")
    return {"candidate_path": str(candidate), "candidate_identity": identity, "functions": sizes,
            "assembly_stdout_sha256": _sha(assembly.stdout.encode()),
            "section_sizes_stdout_sha256": _sha(sections.stdout.encode()),
            "scope": "actual owning DSO, local functions and whole-DSO section sizes; no execution or causal speed claim"}


def _correctness(log, instance, role, kind):
    argv = [instance["binary"]]
    if kind == "timed":
        argv += ["--verify-only"]
    elif kind == "end_k":
        argv += list(map(str, c.TILES[role]))
    result = log.invoke(argv)
    c.exact(result.stderr, "", "positive correctness stderr")
    if kind == "timed":
        receipt = c.strict_json(result.stdout)
        c.check_controls(receipt, result.returncode)
        p1.modules_match(receipt, instance["loaded_modules"])
        return receipt
    return peel.end_k_summary(result.stdout) if kind == "end_k" else legacy.semantic_summary(result.stdout, False)


def _relations(instances):
    c.exact(set(instances), set(c.ROLES), "exact five-role inventory")
    for role in c.ROLES:
        c.exact(set(instances[role]), set(_kinds(role)), "role executable-kind inventory")
        if c.FLAGS[role] is not None:
            for kind in _kinds(role)[1:]:
                p1.join_issued(instances[role]["timed"]["manifest"], instances[role][kind]["manifest"])
    baseline = instances["tiles-3x5"]["timed"]["manifest"]
    common = ("source_context_sha256", "region_name_hex", "candidate", "optimization", "toolchain", "target",
              "address_sanitizer", "relink_kit_sha256", "exports_sha256", "public_header_sha256",
              "storage_header_sha256", "private_header_sha256", "provider_sha256", "accelerator_count",
              "semantic_sha256", "structured_sha256", "bufferized_sha256", "canonical_runtime_sha256",
              "stock_candidates_sha256", "clang_sha256", "linker_sha256")
    count = baseline["accelerator_count"]
    c.require(type(count) is str and re.fullmatch(r"[0-2]", count), "closed common accelerator inventory")
    common += tuple("accelerator_" + str(index) + "_sha256" for index in range(int(count)))
    for role in c.ROLES[1:]:
        info = instances[role]["timed"]["manifest"]
        for key in common:
            c.exact(info[key], baseline[key], "common timed source/WHAT/installation/toolchain " + key)
    for a, b in (("tiles-3x5", "peel-3x5"), ("tiles-5x3", "peel-5x3"),
                 ("tiles-3x5", "tiles-5x3"), ("peel-3x5", "peel-5x3")):
        first, second = (instances[role]["timed"]["manifest"] for role in (a, b))
        for key in ("transform_sha256", "scheduled_sha256", "llvm_sha256", "object_sha256",
                    "dso_sha256", "binding_sha256"):
            c.require(first[key] != second[key], "distinct requested HOW not consumed: " + a + "/" + b + "/" + key)


def _catalog(root):
    return {str(path.relative_to(root)): p1.file_identity(path)
            for path in sorted((root / "raw").rglob("*")) if path.is_file()}


def _capsule_inventory(root, prepared):
    expected = {"bin", "inputs", "inspection", "raw", "prepared.json"} if prepared else {"raw", "collection.json"}
    c.exact({path.name for path in root.iterdir()}, expected, "exact capsule top-level inventory")
    if prepared:
        binaries = {kind + "-" + role for role in c.ROLES for kind in _kinds(role)}
        bundles = {kind + "-" + role + ".mdslc-private" for role in c.ROLES
                   if c.FLAGS[role] is not None for kind in _kinds(role)}
        c.exact({path.name for path in (root / "bin").iterdir()}, binaries | bundles, "exact role binary/bundle inventory")
        c.exact({path.name for path in (root / "inputs").iterdir()},
                {"call_return.mdsl", "source_pattern", "source_pattern_peel"}, "exact input root")
        c.exact({path.name for path in (root / "inputs/source_pattern").iterdir()},
                {"connected.mdsl", "math_helper.h"}, "exact unchanged control inputs")
        c.exact({path.name for path in (root / "inputs/source_pattern_peel").iterdir()},
                {"end_k.mdsl"}, "exact end-K input")
        c.exact({path.name for path in (root / "inspection").iterdir()},
                {"remove-rpath.cmake", "stock-rpath-removed.so"}, "exact nonexecuted stock inspection products")


def _host_gate(snapshot, cpu):
    """Recheck recorded bounded predicates only; never sample a live audit host."""
    c.require(type(snapshot) is dict, "missing retained host snapshot")
    c.exact(snapshot.get("selected_cpu"), cpu, "retained selected CPU")
    affinity = snapshot.get("caller_affinity")
    c.require(type(affinity) is list and all(type(value) is int and value >= 0 for value in affinity) and
              affinity == sorted(set(affinity)) and cpu in affinity,
              "retained caller affinity does not admit selected CPU")
    c.exact(snapshot.get("build_test_comm_counts"), {}, "retained known build/test activity")
    logical, load = snapshot.get("logical_cpus"), snapshot.get("loadavg")
    c.require(type(logical) is int and logical > 0 and type(load) is list and len(load) == 3 and
              all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in load) and
              load[0] <= logical / 4, "retained load exceeds unchanged guard")
    temperatures = snapshot.get("cpu_temperature_millicelsius")
    if type(temperatures) is dict:
        c.require(bool(temperatures) and all(type(name) is str and type(value) is int and value < 85000
                  for name, value in temperatures.items()), "retained observed temperature exceeds unchanged guard")
    else:
        c.exact(temperatures, "unavailable; no thermal mechanism claim", "explicit missing temperature observation")


def prepare(args):
    output = p1.new_external_directory(args.output)
    for name in ("bin", "inputs", "inputs/source_pattern", "inputs/source_pattern_peel"):
        (output / name).mkdir()
    log, events = p1.Commands(output), []
    report = {"schema": PREPARED_SCHEMA, "version": c.VERSION, "status": "INCOMPLETE",
              "root": str(output.resolve()), "plan": c.plan(), "plan_sha256": c.digest(c.plan()),
              "required_product_checkpoint": c.QUALIFIED_SOURCE_COMMIT,
              "qualification_gate": "operator prerequisite; not a serialized credential or runtime authority",
              "commands": log.records, "events": events, "instances": {},
              "source_runtime_policy_changed": False, "performance_result": False}
    try:
        prefix, driver = args.prefix.resolve(strict=True), args.driver.resolve(strict=True)
        c.exact(driver, (prefix / "bin/mdslc-region").resolve(strict=True), "installed driver location")
        c.require((prefix / "include/matcore/region.h").is_file() and not (prefix / "CMakeCache.txt").exists(),
                  "provide a qualified relocated SDK, not a build tree")
        report.update(installation_prefix=str(prefix), installation_inventory=p1.inventory(prefix),
                      installation_bytes_excluded_from_generated_cap=p1.tree_bytes(prefix))
        for path in prefix.rglob("*"):
            if path.is_file():
                log.pin(path)
        tools = {name: str(Path(shutil.which(name) or "missing").resolve(strict=True))
                 for name in ("readelf", "nm", "ldd", "cmake", "git")}
        tools.update(cxx=str(args.cxx.resolve(strict=True)), linker=str(args.linker.resolve(strict=True)),
                     objdump=str(args.objdump.resolve(strict=True)), size=str(args.size.resolve(strict=True)))
        own = _helpers()
        for path in (driver, *tools.values(), *own):
            log.pin(path)
        report.update(driver=str(driver), tools=tools, helper_files=[str(path) for path in own])
        report["compilation_tool_modules"] = {}
        for name in ("driver", "cxx", "linker", "cmake", "objdump", "size"):
            path = driver if name == "driver" else Path(tools[name])
            report["compilation_tool_modules"][name] = _event(log, events, "tool-modules:" + name,
                lambda path=path: p1.dependencies(log, path, tools))
        version = _event(log, events, "driver-version", lambda: log.invoke([driver, "--version"]))
        c.exact(version.stdout, PRODUCT_VERSION, "product version/profile")
        c.exact(version.stderr, "", "product version stderr")
        report["sources"] = []
        for original, destination, rendered in _sources(output):
            original_identity = log.pin(original)
            data = c.render_source() if rendered else original.read_bytes()
            c.require(type(data) is bytes, "source renderer must return exact bytes")
            with destination.open("xb") as stream:
                stream.write(data)
            identity = log.pin(destination)
            c.exact(identity["sha256"], _sha(c.render_source()) if rendered else original_identity["sha256"], "one exact source copy")
            report["sources"].append({"original": str(original), "original_identity": original_identity,
                                      "path": str(destination), "identity": identity, "rendered": rendered})
        for role in c.ROLES:
            instances, (inspect, tiles) = {}, _role(role)
            for kind in _kinds(role):
                binary = output / "bin" / (kind + "-" + role)
                _event(log, events, "compile:" + role + ":" + kind,
                       lambda role=role, kind=kind: log.invoke(_argv(driver, output, role, kind), timeout=60))
                instance = {"binary": str(binary), "binary_identity": log.pin(binary), "request": _request(role)}
                if inspect is not None:
                    info, files = _event(log, events, "inspect:" + role + ":" + kind,
                        lambda: inspect.validate_bundle(binary, tiles, _source(output, kind)[1], "none", False, log.invoke))
                    c.exact(info["schema"], _request(role)["schema"], "actual role schema")
                    for key, name in (("clang_sha256", "cxx"), ("linker_sha256", "linker")):
                        c.exact(info[key], log.pin(tools[name])["sha256"], "actual issued tool bytes")
                    instance.update(manifest=info, bundle_files=files)
                    for path in Path(str(binary) + ".mdslc-private").iterdir():
                        if path.is_file():
                            log.pin(path)
                else:
                    c.require(not Path(str(binary) + ".mdslc-private").exists(), "stock role unexpectedly emitted a private bundle")
                instance["loaded_modules"] = _event(log, events, "dependencies:" + role + ":" + kind,
                    lambda: p1.dependencies(log, binary, tools))
                instance["codegen"] = _event(log, events, "codegen:" + role + ":" + kind,
                    lambda: _codegen(log, _candidate(instance, role), tools))
                instances[kind] = instance
            for kind in _kinds(role):
                instances[kind]["correctness"] = _event(log, events, "correctness:" + role + ":" + kind,
                    lambda kind=kind: _correctness(log, instances[kind], role, kind))
            report["instances"][role] = instances
        _relations(report["instances"])
        tiled = report["instances"]["tiles-3x5"]["timed"]["manifest"]
        for kind in ("timed", "control"):
            instance = report["instances"]["stock"][kind]
            instance["stock_identity"] = _event(log, events, "stock-identity:" + kind,
                lambda: p1.stock_identity(log, Path(instance["binary"]), tools,
                    tiled["canonical_runtime_sha256"], tiled["stock_candidates_sha256"], prefix))
        report["stock_profile_catalog"] = str(args.stock_profile.resolve(strict=True))
        report["stock_profile"] = _event(log, events, "stock-profile", lambda: p1.stock_profile(args.stock_profile,
            log, _candidate(report["instances"]["stock"]["timed"], "stock"), tools))
        c.exact(report["stock_profile"]["producer_git"]["head"], c.QUALIFIED_SOURCE_COMMIT, "fixed campaign producer checkpoint")
        c.exact(report["stock_profile"]["selected_schedule"], "scalar-mnk", "fixed stock comparator schedule")
        c.exact([event["label"] for event in events], _event_labels(), "complete ordered preparation")
        log.unchanged()
        c.exact(p1.inventory(prefix), report["installation_inventory"], "installation after preparation")
        report.update(status="PREPARED", budget=p1.budget(output))
    except Exception as error:
        report["failure"] = str(error)
    finally:
        report.update(pinned=log.pinned, raw_catalog=_catalog(output))
        p1.write_json(output / "prepared.json", report)
    print(p1.json.dumps({"status": report["status"], "capsule": str(output), "performance_result": False}))
    return 0 if report["status"] == "PREPARED" else 1


class RawAudit:
    """Integrity/derivation joins for a trusted collector, not origin attestation."""
    def __init__(self, root, report):
        self.root, self.report, self.used = root, report, set()
        records, catalog = report["commands"], report["raw_catalog"]
        c.require(type(records) is list and type(catalog) is dict, "raw command/catalog inventories")
        expected = {f"raw/{index:04d}/{name}" for index in range(len(records))
                    for name in ("stdout.log", "stderr.log", "command.json")}
        c.exact(set(catalog), expected, "complete raw catalog")
        c.exact({str(path.relative_to(root)) for path in (root / "raw").rglob("*") if path.is_file()},
                expected, "extra/missing raw files")
        for relative, identity in catalog.items():
            path = (root / relative).resolve(strict=True)
            c.require(path.is_relative_to(root), "raw catalog escaped root")
            actual = p1.file_identity(path)
            for key in ("sha256", "bytes"):
                c.exact(actual[key], identity[key], "raw bytes " + relative)
        for index, record in enumerate(records):
            c.exact(record["directory"], f"raw/{index:04d}", "chronological raw directory")
            c.exact(record["cwd"], str(root), "raw working directory")
            c.exact(record["environment"], p1.env_record(), "sanitized child environment")
            c.exact(record["loader_injection_environment_removed"], True, "loader environment policy")
            c.exact(record["timed_out"], False, "timed out command cannot qualify")
            c.require(type(record["returncode"]) is int, "missing process return code")
            timeout = record.get("timeout_seconds")
            c.require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 60,
                      "invalid raw command timeout")
            c.exact(p1.read_json(root / record["directory"] / "command.json"), record, "raw command record")

    def raw(self, index, argv, expected=0, cpu=None, maximum_timeout=20):
        c.require(type(index) is int and 0 <= index < len(self.report["commands"]) and index not in self.used,
                  "duplicate/missing raw command join")
        self.used.add(index)
        record = self.report["commands"][index]
        c.exact(record["argv"], list(map(str, argv)), "actual command argv")
        c.exact(record["affinity_child"], None if cpu is None else [cpu], "actual child affinity request")
        c.require(record["timeout_seconds"] <= maximum_timeout, "command exceeds its declared process timeout")
        if expected is not None:
            c.exact(record["returncode"], expected, "actual command exit")
        directory = self.root / record["directory"]
        return SimpleNamespace(stdout=(directory / "stdout.log").read_text(),
                               stderr=(directory / "stderr.log").read_text(), returncode=record["returncode"])

    def finish(self):
        c.exact(self.used, set(range(len(self.report["commands"]))), "unjoined raw process")


class Replay:
    def __init__(self, audit, indices, pinned):
        self.audit, self.indices, self.pinned, self.cursor = audit, indices, pinned, 0

    def pin(self, path):
        path = str(Path(path).resolve(strict=True))
        c.require(path in self.pinned, "offline inspection found an unpinned input")
        actual = p1.file_identity(path)
        c.exact(actual, self.pinned[path], "actual retained input")
        return actual

    def invoke(self, argv, expected=0, timeout=20, **kwargs):
        c.require(not kwargs, "unexpected inspection process options in offline replay")
        c.require(self.cursor < len(self.indices), "extra offline inspection command")
        index = self.indices[self.cursor]
        self.cursor += 1
        return self.audit.raw(index, argv, expected, maximum_timeout=timeout)

    def finish(self):
        c.exact(self.cursor, len(self.indices), "missing offline inspection command")


def _audit_stock_profile(profile, replay, root, tools, installed):
    catalog = profile["catalog"]
    c.exact(catalog.get("schema"), "matcore.source-pattern.p1.stock-profile", "stock catalog schema")
    c.exact(catalog.get("version"), 1, "stock catalog version")
    paths = {key: Path(profile["catalog"][key]).resolve(strict=True)
             for key in ("producer_candidate", "cache", "leaf", "object", "build_ninja")}
    c.exact(catalog["source_commit"], c.QUALIFIED_SOURCE_COMMIT, "fixed stock catalog checkpoint")
    cache = {}
    for line in paths["cache"].read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            name, value = line.split("=", 1)
            cache[name.split(":", 1)[0]] = value
    source = Path(cache["CMAKE_HOME_DIRECTORY"]).resolve(strict=True)
    c.require(source.name == "compiler", "closed stock source directory")
    build = paths["build_ninja"].parent
    c.exact(Path(cache["CMAKE_CACHEFILE_DIR"]).resolve(), build, "stock cache/root join")
    identities = {}
    for key, relative in (("cache", "CMakeCache.txt"), ("leaf", "lib/regions/strict-normal.ll"),
                          ("object", "lib/regions/strict-normal.o"), ("build_ninja", "build.ninja"),
                          ("producer_candidate", "lib/libmatcore_closed_candidates_isolated_v1.so")):
        c.exact(paths[key], (build / relative).resolve(strict=True), "actual stock producer path")
        identities[key] = replay.pin(paths[key])
    installed_identity = replay.pin(installed)
    installed_dynamic = replay.invoke([tools["readelf"], "-d", installed])
    c.require("(RPATH)" not in installed_dynamic.stdout and "(RUNPATH)" not in installed_dynamic.stdout,
              "installed stock has an unexpected loader path")
    script, normalized = root / "inspection/remove-rpath.cmake", root / "inspection/stock-rpath-removed.so"
    equals = "="
    while "]" + equals + "]" in str(normalized):
        equals += "="
    c.exact(script.read_text(), "file(RPATH_REMOVE FILE [" + equals + "[" + str(normalized) + "]" + equals + "])\n",
            "only the bounded standard CMake install transformation")
    replay.pin(script)
    c.exact(replay.invoke([tools["cmake"], "-P", script]).stderr, "", "recorded CMake transform stderr")
    transformed = replay.pin(normalized)
    c.exact(transformed["sha256"], installed_identity["sha256"], "full normalized stock byte join")
    observed = {}
    for key, argv in (("root", ["rev-parse", "--show-toplevel"]), ("head", ["rev-parse", "HEAD"]),
                      ("dirty", ["status", "--porcelain=v1", "--untracked-files=no"])):
        result = replay.invoke([tools["git"], "-C", source, *argv])
        c.exact(result.stderr, "", "producer Git stderr")
        observed[key] = result.stdout.strip()
    c.exact(observed, profile["producer_git"], "stock producer raw Git join")
    c.exact(observed["root"], str(source.parent), "actual producer source root")
    c.exact(observed["head"], c.QUALIFIED_SOURCE_COMMIT, "required product checkpoint")
    c.exact(observed["dirty"], "", "frozen producer source")
    c.exact(cache["MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE"], "scalar", "actual stock cache selection")
    supporting = (build / "CMakeFiles/rules.ninja", source / "lib/regions/private_runtime_v2.exports",
                  source / "tools/mdslc-region/installation_identity.cmake", build / "bin/matcore-cpu-gemm-candidate")
    support_identities = {str(path): replay.pin(path) for path in supporting}
    identity_header = build / "lib/regions/mdslc-region/region_driver_identity.h"
    replay.pin(identity_header)
    for define, wanted in (("REGION_CANDIDATES_SHA", identities["producer_candidate"]["sha256"]),
                           ("REGION_INSTALLED_CANDIDATES_SHA", installed_identity["sha256"])):
        values = re.findall(r'^#define ' + define + r' "([0-9a-f]{64})"$', identity_header.read_text(), re.M)
        c.exact(values, [wanted], "existing stock install-identity header")
    issuer = {}
    manifest_path = Path(str(paths["leaf"]) + ".manifest")
    replay.pin(manifest_path)
    for line in manifest_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        c.require(separator and key not in issuer, "malformed actual stock issuer manifest")
        issuer[key] = value
    for key, wanted in (("schema", "matcore-builtin-strict-cpu-gemm-v1"), ("toolchain", "21.1.8"),
                        ("target", "x86_64-pc-linux-gnu"), ("profile", "strict_f32"),
                        ("address_sanitizer", "off"), ("schedule", "scalar-mnk")):
        c.exact(issuer.get(key), wanted, "actual stock issuer " + key)
    for field, suffix in (("llvm_sha256", ""), ("semantic_sha256", ".semantic.mlir"),
                          ("structured_sha256", ".structured.mlir"), ("bufferized_sha256", ".bufferized.mlir"),
                          ("scheduled_sha256", ".scheduled.mlir"), ("transform_sha256", ".transform.mlir")):
        c.exact(replay.pin(Path(str(paths["leaf"]) + suffix))["sha256"], issuer.get(field), "actual stock stage join")
    ninja = paths["build_ninja"].read_text()
    target = "build " + str(paths["producer_candidate"].relative_to(build)) + ":"
    links = [block for block in re.split(r"(?=^build )", ninja, flags=re.M) if block.startswith(target)]
    c.require(len(links) == 1 and str(paths["object"].relative_to(build)) in links[0].splitlines()[0].split() and
              "-Bsymbolic" in links[0] and str(source / "lib/regions/private_runtime_v2.exports") in links[0],
              "actual stock intrinsic link graph")
    compiles = [line.strip() for line in ninja.splitlines() if "COMMAND =" in line and
                paths["object"].name in line and "-x ir" in line]
    issuers = [line.strip() for line in ninja.splitlines() if "COMMAND =" in line and
               "matcore-cpu-gemm-candidate --output " + str(paths["leaf"]) in line]
    c.require(len(compiles) == len(issuers) == 1, "actual stock command cardinality")
    c.exact(shlex.split(compiles[0].partition("COMMAND =")[2].strip()),
            ["cd", str(paths["leaf"].parent), "&&", cache["CMAKE_CXX_COMPILER"], "-c", "-x", "ir",
             str(paths["leaf"]), "-O2", "-o", str(paths["object"])], "actual stock full-path object command")
    c.exact(shlex.split(issuers[0].partition("COMMAND =")[2].strip()),
            ["cd", str(paths["leaf"].parent), "&&", str(build / "bin/matcore-cpu-gemm-candidate"),
             "--output", str(paths["leaf"])], "actual stock full-path issuer command")
    expected = {"catalog": catalog, "input_identities": identities, "issuer_manifest": issuer,
        "producer_git": observed, "supporting_identities": support_identities,
        "install_transform": {"operation": "CMake file(RPATH_REMOVE) on nonexecuted inspection copy only",
            "raw_producer_sha256": identities["producer_candidate"]["sha256"],
            "installed_sha256": installed_identity["sha256"], "derived_sha256": transformed["sha256"],
            "derived_path": str(normalized), "cmake_sha256": replay.pin(tools["cmake"])["sha256"]},
        "selected_schedule": issuer["schedule"], "compile_command": compiles[0], "issuer_command": issuers[0],
        "link_graph": links[0], "build_flags": {key: cache.get(key, "unavailable") for key in
            ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE")},
        "authority": "producer inspection and same-DSO evidence; not a source/runtime importer or attestation"}
    c.exact(profile, expected, "complete stock-profile reconstruction from retained producer/SDK/raw records")


def audit_prepared(path):
    path = Path(path).resolve(strict=True)
    root, prepared = path.parent, p1.read_json(path)
    for observed, expected, label in ((prepared.get("schema"), PREPARED_SCHEMA, "prepared schema"),
            (prepared.get("version"), c.VERSION, "prepared version"), (prepared.get("status"), "PREPARED", "prepare status"),
            (prepared.get("root"), str(root), "unrelocated capsule"),
            (prepared.get("required_product_checkpoint"), c.QUALIFIED_SOURCE_COMMIT, "required checkpoint"),
            (prepared.get("plan_sha256"), c.digest(c.plan()), "fixed plan"),
            (c.digest(prepared.get("plan")), c.digest(c.plan()), "serialized fixed plan")):
        c.exact(observed, expected, label)
    c.exact(prepared.get("source_runtime_policy_changed"), False, "preparation never changes product policy")
    c.exact(prepared.get("performance_result"), False, "preparation is not performance evidence")
    c.exact(prepared.get("qualification_gate"), "operator prerequisite; not a serialized credential or runtime authority",
            "qualification remains an operator prerequisite")
    _capsule_inventory(root, True)
    c.exact(prepared["budget"], p1.budget(root), "actual retained preparation budgets")
    for pinned, wanted in prepared["pinned"].items():
        c.exact(p1.file_identity(pinned), wanted, "retained preparation input " + pinned)
    c.exact(prepared["helper_files"], [str(path) for path in _helpers()], "exact frozen helper inventory")
    def pin(path):
        resolved = str(Path(path).resolve(strict=True))
        c.require(resolved in prepared["pinned"], "required preparation input is not pinned: " + resolved)
        identity = p1.file_identity(resolved)
        c.exact(identity, prepared["pinned"][resolved], "actual required preparation input")
        return identity
    for helper in _helpers():
        pin(helper)
    for tool in (prepared["driver"], *prepared["tools"].values()):
        pin(tool)
    c.exact(set(prepared["tools"]), {"readelf", "nm", "ldd", "cmake", "git", "cxx", "linker", "objdump", "size"},
            "exact inspection/compilation tool set")
    c.exact(set(prepared["compilation_tool_modules"]), {"driver", "cxx", "linker", "cmake", "objdump", "size"},
            "exact compilation tool module sets")
    c.exact(p1.inventory(Path(prepared["installation_prefix"])), prepared["installation_inventory"], "retained SDK")
    c.exact(prepared["installation_bytes_excluded_from_generated_cap"], p1.tree_bytes(Path(prepared["installation_prefix"])),
            "actual supplied SDK bytes outside generated cap")
    for sdk in Path(prepared["installation_prefix"]).rglob("*"):
        if sdk.is_file():
            pin(sdk)
    c.exact((root / "inputs/call_return.mdsl").read_bytes(), c.render_source(), "actual shared rendered source")
    expected_sources = []
    for original, destination, rendered in _sources(root):
        expected_sources.append({"original": str(original), "original_identity": pin(original),
                                 "path": str(destination), "identity": pin(destination), "rendered": rendered})
        c.exact(destination.read_bytes(), c.render_source() if rendered else original.read_bytes(), "actual fixed compiler source")
    c.exact(prepared["sources"], expected_sources, "exact four-source inventory")
    catalog_path = Path(prepared["stock_profile_catalog"])
    pin(catalog_path)
    c.exact(p1.read_json(catalog_path), prepared["stock_profile"]["catalog"], "actual retained stock catalog")
    audit = RawAudit(root, prepared)
    events = prepared["events"]
    c.exact([event["label"] for event in events], _event_labels(), "ordered preparation event inventory")
    c.exact([index for event in events for index in event["indices"]], list(range(len(prepared["commands"]))),
            "complete contiguous preparation chronology")
    indexed = {event["label"]: event["indices"] for event in events}
    def event(label, action):
        replay = Replay(audit, indexed[label], prepared["pinned"])
        result = action(replay)
        replay.finish()
        return result
    tools, driver = prepared["tools"], Path(prepared["driver"])
    c.exact(driver, (Path(prepared["installation_prefix"]) / "bin/mdslc-region").resolve(strict=True), "actual installed driver")
    for name in ("driver", "cxx", "linker", "cmake", "objdump", "size"):
        modules = event("tool-modules:" + name, lambda replay, name=name:
            p1.dependencies(replay, driver if name == "driver" else Path(tools[name]), tools))
        c.exact(modules, prepared["compilation_tool_modules"][name], "actual preparation tool closure")
    version = event("driver-version", lambda replay: replay.invoke([driver, "--version"]))
    c.exact(version.stdout, PRODUCT_VERSION, "raw product version")
    c.exact(version.stderr, "", "raw version stderr")
    _relations(prepared["instances"])
    for role in c.ROLES:
        inspect, tiles = _role(role)
        for kind in _kinds(role):
            instance = prepared["instances"][role][kind]
            c.exact(instance["binary"], str(root / "bin" / (kind + "-" + role)), "exact role executable")
            c.exact(instance["request"], _request(role), "actual role/flag/schema request")
            result = event("compile:" + role + ":" + kind,
                           lambda replay, role=role, kind=kind: replay.invoke(_argv(driver, root, role, kind), timeout=60))
            c.exact((result.stdout, result.stderr), ("", ""), "successful compiler diagnostics")
            c.exact(pin(instance["binary"]), instance["binary_identity"], "actual role executable bytes")
            if inspect is not None:
                for file in Path(instance["binary"] + ".mdslc-private").iterdir():
                    if file.is_file():
                        pin(file)
                info, files = event("inspect:" + role + ":" + kind, lambda replay:
                    inspect.validate_bundle(Path(instance["binary"]), tiles, _source(root, kind)[1], "none", False, replay.invoke))
                c.exact(info, instance["manifest"], "actual bound source manifest")
                c.exact(files, instance["bundle_files"], "actual bound source files")
                c.exact(info["schema"], _request(role)["schema"], "exact source realization schema")
                for key, name in (("clang_sha256", "cxx"), ("linker_sha256", "linker")):
                    c.exact(info[key], p1.file_identity(tools[name])["sha256"], "actual issued tool bytes")
            else:
                c.require(not Path(instance["binary"] + ".mdslc-private").exists(), "stock role acquired a request bundle")
            modules = event("dependencies:" + role + ":" + kind,
                lambda replay: p1.dependencies(replay, Path(instance["binary"]), tools))
            c.exact(modules, instance["loaded_modules"], "actual role loader closure")
            codegen = event("codegen:" + role + ":" + kind,
                lambda replay: _codegen(replay, _candidate(instance, role), tools))
            c.exact(codegen, instance["codegen"], "actual DSO codegen text/sizes")
        for kind in _kinds(role):
            instance = prepared["instances"][role][kind]
            correctness = event("correctness:" + role + ":" + kind,
                lambda replay: _correctness(replay, instance, role, kind))
            c.exact(correctness, instance["correctness"], "actual untimed correctness receipt")
    tiled = prepared["instances"]["tiles-3x5"]["timed"]["manifest"]
    for kind in ("timed", "control"):
        instance = prepared["instances"]["stock"][kind]
        identity = event("stock-identity:" + kind, lambda replay:
            p1.stock_identity(replay, Path(instance["binary"]), tools, tiled["canonical_runtime_sha256"],
                              tiled["stock_candidates_sha256"], Path(prepared["installation_prefix"])))
        c.exact(identity, instance["stock_identity"], "actual stock identity")
    event("stock-profile", lambda replay: _audit_stock_profile(prepared["stock_profile"], replay, root, tools,
        _candidate(prepared["instances"]["stock"]["timed"], "stock")))
    audit.finish()
    return prepared


def collect(args):
    c.require(args.exclusive_host, "--exclusive-host acknowledgement required; not automatic exclusivity")
    capsule = args.prepared.resolve(strict=True)
    prepared = audit_prepared(capsule / "prepared.json")
    output = p1.new_external_directory(args.output)
    log = p1.Commands(output, dict(prepared["pinned"]))
    log.pin(capsule / "prepared.json")
    for path in (HERE / "run.py", HERE / "protocol.py", HERE / "support.py"):
        log.pin(path)
    report = {"schema": COLLECTION_SCHEMA, "version": c.VERSION, "status": "INCOMPLETE",
              "plan_sha256": c.digest(c.plan()), "required_product_checkpoint": c.QUALIFIED_SOURCE_COMMIT,
              "prepared": str(capsule / "prepared.json"), "prepared_identity": p1.file_identity(capsule / "prepared.json"),
              "commands": log.records, "cells": [{**entry, "status": "NOT_RUN"} for entry in c.plan()["order"]],
              "selected_cpu": args.cpu, "control_runs": [], "source_runtime_policy_changed": False,
              "automatic_selection": False, "performance_claim": "none until reviewed"}
    started = time.monotonic()
    log.deadline = started + c.COLLECTION_SECONDS
    try:
        log.unchanged()
        c.exact(p1.inventory(Path(prepared["installation_prefix"])), prepared["installation_inventory"], "SDK before collection")
        report["host_before"] = p1.host_snapshot(args.cpu)
        for phase in ("initial", "final"):
            for role in c.ROLES:
                instance = prepared["instances"][role]["timed"]
                result = log.invoke([instance["binary"], "--verify-only"], cpu=args.cpu, pins=instance["loaded_modules"])
                c.exact(result.stderr, "", "control stderr")
                receipt = c.strict_json(result.stdout)
                c.check_controls(receipt, result.returncode)
                p1.modules_match(receipt, instance["loaded_modules"])
                report["control_runs"].append({"phase": phase, "role": role,
                    "command_index": len(log.records) - 1, "receipt": receipt})
                log.time_remaining()
            if phase == "final":
                break
            for entry in report["cells"]:
                entry["host_before"] = p1.host_snapshot(args.cpu)
                instance = prepared["instances"][entry["role"]]["timed"]
                result = log.invoke([instance["binary"], "--measure", entry["cell"], "--cpu", args.cpu],
                    timeout=c.PROCESS_SECONDS, cpu=args.cpu, expected=None, pins=instance["loaded_modules"])
                c.exact(result.stderr, "", "measurement stderr")
                receipt = c.strict_json(result.stdout)
                p1.modules_match(receipt, instance["loaded_modules"])
                entry.update(timing_accepted=c.check_cell(receipt, result.returncode, entry["cell"], args.cpu),
                    status=receipt["status"], normalized_ns=[sample["wall_ns"] / receipt["repetitions"] for sample in receipt["samples"]],
                    receipt=receipt, command_index=len(log.records) - 1, host_after=p1.host_snapshot(args.cpu))
                log.time_remaining()
        report["host_after"] = p1.host_snapshot(args.cpu)
        report["summary"] = c.summarize(report["cells"])
        stable = all(role["status"] == "PAIRED" for row in report["summary"] for role in row["roles"].values())
        report["status"] = "COMPLETE" if stable else "INCONCLUSIVE_TIMING"
    except Exception as error:
        report["failure"] = str(error)
    finally:
        try:
            log.unchanged()
            c.exact(p1.inventory(Path(prepared["installation_prefix"])), prepared["installation_inventory"], "SDK after collection")
            log.time_remaining()
            report["identity_checks_passed"] = True
        except Exception as error:
            report.update(status="INCOMPLETE", identity_error=str(error), identity_checks_passed=False)
        report.update(budget=p1.budget(output), raw_catalog=_catalog(output), collection_wall_seconds=time.monotonic() - started)
        if report["collection_wall_seconds"] > c.COLLECTION_SECONDS:
            report.update(status="INCOMPLETE", deadline_error="wall budget includes controls, identities and evidence catalog; no resume")
        p1.write_json(output / "collection.json", report)
    print(p1.json.dumps({"status": report["status"], "evidence": str(output / "collection.json"), "automatic_selection": False}))
    return 0 if report["status"] == "COMPLETE" else 78 if report["status"] == "INCONCLUSIVE_TIMING" else 1


def audit_collection(path):
    path = Path(path).resolve(strict=True)
    root, collection = path.parent, p1.read_json(path)
    c.exact(collection.get("schema"), COLLECTION_SCHEMA, "collection schema")
    c.exact(collection.get("version"), c.VERSION, "collection version")
    c.exact(collection.get("required_product_checkpoint"), c.QUALIFIED_SOURCE_COMMIT, "collection checkpoint")
    c.exact(collection.get("source_runtime_policy_changed"), False, "collection never changes product policy")
    c.exact(collection.get("automatic_selection"), False, "no automatic winner/selection")
    c.exact(collection.get("performance_claim"), "none until reviewed", "unreviewed descriptive evidence only")
    c.require(collection.get("status") in ("COMPLETE", "INCONCLUSIVE_TIMING") and
              collection.get("identity_checks_passed") is True, "incomplete/identity-drift collection")
    c.exact(collection["plan_sha256"], c.digest(c.plan()), "summary plan")
    c.require(type(collection.get("collection_wall_seconds")) in (int, float) and
              0 <= collection["collection_wall_seconds"] <= c.COLLECTION_SECONDS, "collection wall deadline")
    prepared_path = Path(collection["prepared"])
    c.exact(p1.file_identity(prepared_path), collection["prepared_identity"], "actual prepared receipt identity")
    prepared = audit_prepared(prepared_path)
    audit, cpu = RawAudit(root, collection), collection["selected_cpu"]
    c.require(type(cpu) is int and cpu >= 0, "explicit collection CPU")
    _capsule_inventory(root, False)
    c.exact(collection["budget"], p1.budget(root), "actual retained collection budgets")
    _host_gate(collection.get("host_before"), cpu)
    _host_gate(collection.get("host_after"), cpu)
    controls = collection["control_runs"]
    c.exact([(item["phase"], item["role"]) for item in controls],
            [(phase, role) for phase in ("initial", "final") for role in c.ROLES], "complete ordered controls")
    role_count, order = len(c.ROLES), c.plan()["order"]
    for item in controls:
        expected = (0 if item["phase"] == "initial" else role_count + len(order)) + c.ROLES.index(item["role"])
        c.exact(item["command_index"], expected, "control chronological index")
        instance = prepared["instances"][item["role"]]["timed"]
        result = audit.raw(expected, [instance["binary"], "--verify-only"], cpu=cpu)
        c.exact(result.stderr, "", "raw control stderr")
        receipt = c.strict_json(result.stdout)
        c.exact(receipt, item["receipt"], "stored/raw control receipt")
        c.check_controls(receipt, result.returncode)
        p1.modules_match(receipt, instance["loaded_modules"])
    c.exact(len(collection["cells"]), len(order), "complete timed cell inventory")
    for ordinal, (entry, wanted) in enumerate(zip(collection["cells"], order)):
        c.exact({key: entry.get(key) for key in wanted}, wanted, "fixed role/cell/direction order")
        _host_gate(entry.get("host_before"), cpu)
        _host_gate(entry.get("host_after"), cpu)
        index = role_count + ordinal
        c.exact(entry["command_index"], index, "measurement chronological index")
        instance = prepared["instances"][entry["role"]]["timed"]
        result = audit.raw(index, [instance["binary"], "--measure", entry["cell"], "--cpu", cpu], expected=None, cpu=cpu)
        c.exact(result.stderr, "", "raw measured stderr")
        receipt = c.strict_json(result.stdout)
        c.exact(receipt, entry["receipt"], "stored/raw measurement receipt")
        c.exact(c.check_cell(receipt, result.returncode, entry["cell"], cpu), entry["timing_accepted"], "derived acceptance")
        c.exact(entry["status"], receipt["status"], "derived status")
        c.exact(entry["normalized_ns"], [sample["wall_ns"] / receipt["repetitions"] for sample in receipt["samples"]], "derived normalization")
        p1.modules_match(receipt, instance["loaded_modules"])
    audit.finish()
    summary = c.summarize(collection["cells"])
    c.exact(summary, collection["summary"], "all summary fields recomputed from raw clocks")
    stable = all(role["status"] == "PAIRED" for row in summary for role in row["roles"].values())
    c.exact(collection["status"], "COMPLETE" if stable else "INCONCLUSIVE_TIMING", "derived campaign status")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = modes.add_parser("prepare")
    for name in ("driver", "prefix", "output", "stock-profile", "linker"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    prepare_parser.add_argument("--cxx", type=Path, default=Path("/usr/bin/clang++-21"))
    prepare_parser.add_argument("--objdump", type=Path, default=Path("/usr/bin/llvm-objdump-21"))
    prepare_parser.add_argument("--size", type=Path, default=Path("/usr/bin/llvm-size-21"))
    collect_parser = modes.add_parser("collect")
    collect_parser.add_argument("--prepared", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--cpu", type=int, required=True)
    collect_parser.add_argument("--exclusive-host", action="store_true")
    summary_parser = modes.add_parser("summarize")
    summary_parser.add_argument("--collection", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "summarize":
        print(p1.json.dumps(audit_collection(args.collection), indent=2, allow_nan=False))
        return 0
    c.require(platform.system() == "Linux" and platform.machine() == "x86_64", "fixed P5 is native Linux x64 only")
    return prepare(args) if args.mode == "prepare" else collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("P5 refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
