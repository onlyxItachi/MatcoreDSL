#!/usr/bin/env python3
"""P1 prepare/collect/summarize. No automatic compiler builds or tile policy.

Prepare uses an already relocated installation. Collect requires an explicit CPU
and exclusive-host acknowledgement; it never prepares, retunes, retries or resumes.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import contract as c

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SDK_SOURCE = ROOT / "compiler/tests/source_pattern"
BUILD_NAMES = re.compile(r"(?:cmake|ninja|ctest|clang(?:\+\+)?(?:-[0-9]+)?|cc1.*|gcc|g\+\+|ld|ld\.lld|lld|pytest.*)\Z")


def file_identity(path):
    path = Path(path)
    before = path.stat()
    c.require(path.is_file(), "not a regular input file: " + str(path))
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    after = path.stat()
    c.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
              (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed while hashing")
    return {"sha256": result.hexdigest(), "bytes": after.st_size,
            "inode": after.st_ino, "device": after.st_dev, "mtime_ns": after.st_mtime_ns}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return c.strict_json(Path(path).read_text(encoding="utf-8"))


def environment():
    out = dict(os.environ)
    for name in list(out):
        if stripped_variable(name): out.pop(name)
    out.update(LC_ALL="C", PYTHONDONTWRITEBYTECODE="1", DEBUGINFOD_URLS="",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return out


def stripped_variable(name):
    return (name.startswith(("LD_", "MALLOC_", "OMP_", "GOMP_", "KMP_", "OPENBLAS_", "GOTO_", "MKL_")) or
            name in ("GLIBC_TUNABLES", "ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS", "MSAN_OPTIONS"))


def env_record():
    return {key: environment()[key] for key in ("LC_ALL", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}


def inventory(prefix):
    result = {}
    for path in sorted(prefix.rglob("*")):
        if path.is_symlink():
            c.require(not path.is_dir(), "P1 installation cannot contain directory symlinks")
            result[str(path.relative_to(prefix))] = {"symlink": os.readlink(path), "resolved": str(path.resolve(strict=True)),
                                                     **file_identity(path)}
        elif path.is_file():
            result[str(path.relative_to(prefix))] = file_identity(path)
    c.require(result, "empty installation")
    return result


def tree_bytes(path):
    # Count actual files, not each library symlink's target a second time.
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def new_external_directory(path):
    path = path.absolute()
    c.require(not path.exists() and not path.is_symlink(), "output exists; no overwrite")
    parent = path.parent.resolve(strict=True)
    c.require(not any((ancestor / ".git").exists() or (ancestor / ".git").is_symlink()
                      for ancestor in (parent, *parent.parents)), "output has a Git worktree ancestor")
    result = subprocess.run(["git", "-C", str(parent), "rev-parse", "--is-inside-work-tree"],
                            env=environment(), capture_output=True, text=True, timeout=5)
    c.require(result.returncode == 128 and "not a git repository" in result.stderr,
              "raw/artifact output must be outside every Git worktree (use an explicit /var/tmp NEWDIR)")
    path.mkdir(mode=0o700)
    return path


def budget(output):
    raw = tree_bytes(output / "raw") if (output / "raw").exists() else 0
    generated = sum(tree_bytes(output / name) for name in ("bin", "inputs", "inspection") if (output / name).exists())
    c.require(raw <= c.RAW_CAP, "raw receipt cap exceeded; files preserved")
    c.require(generated <= c.ARTIFACT_CAP, "generated binary/bundle cap exceeded; files preserved")
    return {"raw_bytes": raw, "generated_binary_bundle_source_inspection_bytes": generated}


class Commands:
    def __init__(self, output, pinned=None):
        self.output = output
        self.raw = output / "raw"
        self.raw.mkdir()
        self.records = []
        self.pinned = {} if pinned is None else pinned
        self.deadline = None

    def time_remaining(self):
        if self.deadline is None: return 60
        remaining = self.deadline - time.monotonic()
        c.require(remaining > 0, "collection wall deadline exceeded; no retry/resume")
        return remaining

    def pin(self, path):
        path = str(Path(path).resolve(strict=True))
        identity = file_identity(path)
        if path in self.pinned:
            c.exact(identity, self.pinned[path], "pinned input " + path)
        else:
            self.pinned[path] = identity
        return identity

    def unchanged(self, paths=None):
        selected = self.pinned if paths is None else {str(Path(path).resolve(strict=True)): self.pinned[str(Path(path).resolve(strict=True))] for path in paths}
        for path, wanted in selected.items():
            self.time_remaining()
            c.exact(file_identity(path), wanted, "frozen artifact " + path)
        self.time_remaining()

    def invoke(self, argv, timeout=20, cpu=None, expected=0, pins=None):
        argv = list(map(str, argv))
        self.unchanged(pins)
        timeout = min(timeout, self.time_remaining())
        directory = self.raw / f"{len(self.records):04d}"
        directory.mkdir()
        record = {"argv": argv, "cwd": str(self.output), "timeout_seconds": timeout,
                  "affinity_child": None if cpu is None else [cpu], "environment": env_record(),
                  "stripped_environment_names": sorted(name for name in os.environ if stripped_variable(name)),
                  "loader_injection_environment_removed": True, "returncode": None,
                  "timed_out": False, "directory": str(directory.relative_to(self.output))}
        self.records.append(record)
        try:
            with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                try:
                    result = subprocess.run(argv, cwd=self.output, env=environment(), stdout=stdout, stderr=stderr,
                                            check=False, timeout=timeout,
                                            preexec_fn=None if cpu is None else lambda: os.sched_setaffinity(0, {cpu}))
                    record["returncode"] = result.returncode
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
                    raise ValueError("command timed out; no retry; evidence preserved")
            self.unchanged(pins)
            budget(self.output)
            c.require((directory / "stdout.log").stat().st_size <= 8 << 20, "unexpectedly large stdout; raw preserved")
            out = (directory / "stdout.log").read_text(encoding="utf-8")
            err = (directory / "stderr.log").read_text(encoding="utf-8")
            if expected is not None:
                c.exact(result.returncode, expected, "command return code")
            return SimpleNamespace(stdout=out, stderr=err, returncode=result.returncode)
        finally:
            write_json(directory / "command.json", record)


def inspector():
    spec = importlib.util.spec_from_file_location("p1_existing_source_pattern_inspector", SDK_SOURCE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loaded_paths(text):
    c.require("not found" not in text, "unresolved loader dependency")
    paths = []
    for line in text.splitlines():
        match = re.search(r"(?:\S+\s+=>\s+)?(/.+?)\s+\(0x[0-9a-fA-F]+\)\s*$", line.strip())
        if match:
            paths.append(str(Path(match[1]).resolve(strict=True)))
        elif line.strip() and "linux-vdso" not in line:
            raise ValueError("unrecognized loader inspection line: " + line)
    c.require(paths, "empty dynamic loader inventory")
    return sorted(set(paths))


def modules_match(receipt, expected):
    paths = receipt.get("loaded_modules")
    c.require(type(paths) is list and paths and all(type(p) is str for p in paths), "actual loaded mappings missing")
    actual = sorted(set(str(Path(path).resolve(strict=True)) for path in paths))
    c.exact(actual, sorted(expected), "actually loaded executable mappings")


def dependencies(log, binary, tools):
    result = log.invoke([tools["ldd"], binary], expected=0)
    paths = sorted({str(binary.resolve()), *loaded_paths(result.stdout)})
    for path in paths:
        log.pin(path)
    return paths


def stock_identity(log, binary, tools, runtime_sha, stock_sha, prefix):
    paths = dependencies(log, binary, tools)
    stock = [Path(p) for p in paths if Path(p).name == "libmatcore_closed_candidates_isolated_v1.so"]
    runtime = [Path(p) for p in paths if Path(p).name.startswith("libmatcore_runtime.so")]
    c.require(len(stock) == 1 and len(runtime) == 1, "stock/runtime loaded identity missing")
    for path, wanted in ((stock[0], stock_sha), (runtime[0], runtime_sha)):
        c.require(path.is_relative_to(prefix), "stock/runtime escaped relocated installation")
        c.exact(log.pin(path)["sha256"], wanted, "stock/runtime exact bytes")
    symbols = log.invoke([tools["nm"], "--defined-only", "--format=posix", stock[0]], expected=0).stdout
    exports = log.invoke([tools["nm"], "--dynamic", "--defined-only", "--format=posix", stock[0]], expected=0).stdout
    for name in ("__matcore_strict_gemm_f32_v1", "_mlir_ciface___matcore_strict_gemm_f32_v1"):
        matches = [row.split() for row in symbols.splitlines() if row.split() and row.split()[0] == name]
        c.require(len(matches) == 1 and matches[0][1] == "t", "stock leaf not uniquely local")
        c.require(name not in exports, "stock private leaf dynamically exported")
    return {"candidate_path": str(stock[0]), "runtime_path": str(runtime[0]), "loaded_modules": paths}


def join_issued(timed, control):
    for key in c.ISSUED_HASHES:
        c.require(re.fullmatch(r"[0-9a-f]{64}", timed.get(key, "")) and
                  re.fullmatch(r"[0-9a-f]{64}", control.get(key, "")), "missing/malformed issued identity " + key)
        c.exact(timed[key], control[key], "timed/control issued-leaf join " + key)


def stock_profile(path, log, installed_dso, tools):
    """Bound root-provided producer inspection files; never import code or IR."""
    profile = read_json(path); log.pin(path)
    c.exact(profile.get("schema"), "matcore.source-pattern.p1.stock-profile", "stock profile schema")
    c.exact(profile.get("version"), 1, "stock profile version")
    c.require(re.fullmatch(r"[0-9a-f]{40}", profile.get("source_commit", "")), "producer source commit")
    paths = {key: Path(profile[key]).resolve(strict=True) for key in ("producer_candidate", "cache", "leaf", "object", "build_ninja")}
    identities = {key: log.pin(value) for key, value in paths.items()}
    installed = log.pin(installed_dso)
    dynamic = log.invoke([tools["readelf"], "-d", installed_dso], expected=0).stdout
    c.require("(RPATH)" not in dynamic and "(RUNPATH)" not in dynamic, "v1 requires installed stock DSO with no RPATH/RUNPATH")
    # CMake install removes the build RUNPATH. Reproduce only that upstream
    # transformation on a new nonexecuted copy, then compare ALL resulting bytes.
    inspection = log.output / "inspection"; inspection.mkdir()
    normalized = inspection / "stock-rpath-removed.so"
    with normalized.open("xb") as stream: stream.write(paths["producer_candidate"].read_bytes())
    c.exact(file_identity(normalized)["sha256"], identities["producer_candidate"]["sha256"], "producer inspection copy")
    equals = "="
    while "]" + equals + "]" in str(normalized): equals += "="
    script = inspection / "remove-rpath.cmake"
    with script.open("x", encoding="utf-8") as stream:
        stream.write("file(RPATH_REMOVE FILE [" + equals + "[" + str(normalized) + "]" + equals + "])\n")
    log.pin(script)
    log.invoke([tools["cmake"], "-P", script], expected=0)
    transformed = file_identity(normalized)
    c.exact(transformed["sha256"], installed["sha256"], "standard CMake RPATH removal must produce exact installed bytes")
    log.pin(normalized)
    cache = {}
    for line in paths["cache"].read_text().splitlines():
        if not line or line.startswith(("#", "//")): continue
        name, separator, value = line.partition("=")
        if separator: cache[name.partition(":")[0]] = value
    build = paths["build_ninja"].parent
    c.exact(str(Path(cache.get("CMAKE_CACHEFILE_DIR", "missing")).resolve()), str(build), "producer cache directory")
    for key, relative in (("cache", "CMakeCache.txt"), ("build_ninja", "build.ninja"), ("leaf", "lib/regions/strict-normal.ll"),
                          ("object", "lib/regions/strict-normal.o"), ("producer_candidate", "lib/libmatcore_closed_candidates_isolated_v1.so")):
        c.exact(paths[key], (build / relative).resolve(strict=True), "closed producer artifact location " + key)
    source = Path(cache.get("CMAKE_HOME_DIRECTORY", "missing")).resolve(strict=True)
    c.require(source.name == "compiler", "unexpected producer source directory")
    git_results = {}
    for key, args in (("root", ["rev-parse", "--show-toplevel"]), ("head", ["rev-parse", "HEAD"]),
                      ("dirty", ["status", "--porcelain=v1", "--untracked-files=no"])):
        result = log.invoke([tools["git"], "-C", source, *args], expected=0)
        c.exact(result.stderr, "", "producer Git stderr")
        git_results[key] = result.stdout.strip()
    c.exact(git_results["root"], str(source.parent), "actual producer Git root")
    c.exact(git_results["head"], profile["source_commit"], "actual producer Git HEAD")
    c.exact(git_results["dirty"], "", "producer tracked source is frozen")
    supporting = [build / "CMakeFiles/rules.ninja", source / "lib/regions/private_runtime_v2.exports",
                  source / "tools/mdslc-region/installation_identity.cmake",
                  build / "bin/matcore-cpu-gemm-candidate"]
    support_identities = {str(item): log.pin(item) for item in supporting}
    identity_header = build / "lib/regions/mdslc-region/region_driver_identity.h"
    log.pin(identity_header)
    header = identity_header.read_text()
    for define, wanted in (("REGION_CANDIDATES_SHA", identities["producer_candidate"]["sha256"]),
                           ("REGION_INSTALLED_CANDIDATES_SHA", installed["sha256"])):
        values = re.findall(r'^#define ' + define + r' "([0-9a-f]{64})"$', header, re.M)
        c.exact(values, [wanted], "existing compiler install-identity join " + define)
    manifest = {}
    for line in Path(str(paths["leaf"]) + ".manifest").read_text().splitlines():
        key, separator, value = line.partition("=")
        c.require(separator and key not in manifest, "malformed producer leaf manifest")
        manifest[key] = value
    c.exact(manifest.get("schema"), "matcore-builtin-strict-cpu-gemm-v1", "stock issuer schema")
    c.exact(manifest.get("toolchain"), "21.1.8", "stock tuple")
    c.exact(manifest.get("target"), "x86_64-pc-linux-gnu", "stock target")
    c.exact(manifest.get("profile"), "strict_f32", "stock profile")
    c.exact(manifest.get("address_sanitizer"), "off", "stock instrumentation")
    expected = {"scalar": "scalar-mnk", "row-contiguous": "row-contiguous-mkn", "output-tiled": "output-tiled-mkn"}
    selection = cache.get("MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE")
    c.require(selection in expected, "stock build selection missing")
    c.exact(manifest.get("schedule"), expected[selection], "producer schedule")
    stages = {"llvm_sha256": "", "semantic_sha256": ".semantic.mlir", "structured_sha256": ".structured.mlir",
              "bufferized_sha256": ".bufferized.mlir", "scheduled_sha256": ".scheduled.mlir", "transform_sha256": ".transform.mlir"}
    for field, suffix in stages.items():
        artifact = Path(str(paths["leaf"]) + suffix)
        c.exact(log.pin(artifact)["sha256"], manifest.get(field), "producer stage " + field)
    log.pin(Path(str(paths["leaf"]) + ".manifest"))
    if selection == "output-tiled":
        tiles = (int(cache["MDSLC_EXPERIMENTAL_CPU_OUTPUT_TILE_M"]), int(cache["MDSLC_EXPERIMENTAL_CPU_OUTPUT_TILE_N"]))
        c.exact(manifest.get("pattern_parameters"), f"M:{tiles[0]},N:{tiles[1]},K:untiled", "stock tile parameters")
        inspector().output_tile_consumption(Path(str(paths["leaf"]) + ".transform.mlir").read_text(), Path(str(paths["leaf"]) + ".scheduled.mlir").read_text(), tiles)
    ninja = paths["build_ninja"].read_text()
    blocks = re.split(r"(?=^build )", ninja, flags=re.M)
    links = [block for block in blocks if re.match(r"build [^\n]*libmatcore_closed_candidates_isolated_v1\.so:", block)]
    c.require(len(links) == 1 and str(paths["object"].relative_to(build)) in links[0].splitlines()[0].split() and
              links[0].startswith("build " + str(paths["producer_candidate"].relative_to(build)) + ":") and
              "-Bsymbolic" in links[0] and str(source / "lib/regions/private_runtime_v2.exports") in links[0],
              "stock link graph does not bind exact intrinsic leaf/object/export map")
    compile_lines = [line.strip() for line in ninja.splitlines() if "COMMAND =" in line and paths["object"].name in line and "-x ir" in line]
    c.require(len(compile_lines) == 1, "stock object compile command not uniquely recorded")
    c.exact(shlex.split(compile_lines[0].partition("COMMAND =")[2].strip()),
            ["cd", str(paths["leaf"].parent), "&&", cache["CMAKE_CXX_COMPILER"], "-c", "-x", "ir", str(paths["leaf"]), "-O2", "-o", str(paths["object"])],
            "closed stock object compile command and exact input/output paths")
    issuer_lines = [line.strip() for line in ninja.splitlines() if "COMMAND =" in line and "matcore-cpu-gemm-candidate --output " + str(paths["leaf"]) in line]
    c.require(len(issuer_lines) == 1, "stock issuer command not uniquely recorded")
    issuer_args = (["--schedule=row-contiguous"] if selection == "row-contiguous" else
                   ["--output-tiles=" + str(tiles[0]) + "," + str(tiles[1])] if selection == "output-tiled" else [])
    c.exact(shlex.split(issuer_lines[0].partition("COMMAND =")[2].strip()),
            ["cd", str(paths["leaf"].parent), "&&", str(build / "bin/matcore-cpu-gemm-candidate"), "--output", str(paths["leaf"]), *issuer_args],
            "closed stock issuer invocation")
    return {"catalog": profile, "input_identities": identities, "issuer_manifest": manifest,
            "producer_git": git_results, "supporting_identities": support_identities,
            "install_transform": {"operation": "CMake file(RPATH_REMOVE) on nonexecuted inspection copy only",
                                  "raw_producer_sha256": identities["producer_candidate"]["sha256"],
                                  "installed_sha256": installed["sha256"], "derived_sha256": transformed["sha256"],
                                  "derived_path": str(normalized), "cmake_sha256": log.pin(tools["cmake"])["sha256"]},
            "selected_schedule": manifest["schedule"], "compile_command": compile_lines[0], "issuer_command": issuer_lines[0], "link_graph": links[0],
            "build_flags": {key: cache.get(key, "unavailable") for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE")},
            "authority": "producer inspection and same-DSO evidence; not a source/runtime importer or attestation"}


def prepare(args):
    output = new_external_directory(args.output)
    (output / "bin").mkdir()
    (output / "inputs").mkdir()
    log = Commands(output)
    report = {"schema": "matcore.source-pattern.p1.prepared", "version": c.VERSION,
              "status": "INCOMPLETE", "root": str(output.resolve()), "plan": c.plan(),
              "plan_sha256": c.digest(c.plan()), "commands": log.records, "instances": {},
              "source_runtime_policy_changed": False, "performance_result": False}
    try:
        prefix = args.prefix.resolve(strict=True)
        driver = args.driver.resolve(strict=True)
        c.exact(driver, (prefix / "bin/mdslc-region").resolve(strict=True), "installed driver location")
        c.require((prefix / "include/matcore/region.h").is_file() and not (prefix / "CMakeCache.txt").exists(), "provide installed/relocated SDK, not build tree")
        report["installation_prefix"] = str(prefix)
        report["installation_inventory"] = inventory(prefix)
        report["installation_bytes_excluded_from_generated_cap"] = tree_bytes(prefix)
        for path in prefix.rglob("*"):
            if path.is_file(): log.pin(path)
        tools = {name: str(Path(shutil.which(name) or "missing").resolve(strict=True)) for name in ("readelf", "nm", "ldd", "cmake", "git")}
        cxx, linker = args.cxx.resolve(strict=True), args.linker.resolve(strict=True)
        tools.update(cxx=str(cxx), linker=str(linker))
        for path in (driver, *tools.values(), Path(__file__), HERE / "contract.py", SDK_SOURCE / "run.py"):
            log.pin(path)
        report["driver"], report["tools"] = str(driver), tools
        report["compilation_tool_modules"] = {name: dependencies(log, Path(path), tools)
                                               for name, path in (("driver", str(driver)), ("cxx", str(cxx)), ("linker", str(linker)), ("cmake", tools["cmake"]))}
        version = log.invoke([driver, "--version"], expected=0).stdout
        c.exact(version, "MDSLC experimental regions: native Linux x86-64, Clang/MLIR 21.1.8, uninstrumented; API/ABI not frozen\n", "product version/profile")
        for source, name in ((HERE / "call_return.mdsl", "call_return.mdsl"),
                             (SDK_SOURCE / "connected.mdsl", "connected.mdsl"),
                             (SDK_SOURCE / "math_helper.h", "math_helper.h")):
            target = output / "inputs" / name
            original = log.pin(source)
            with target.open("xb") as stream: stream.write(source.read_bytes())
            c.exact(log.pin(target)["sha256"], original["sha256"], "copied source bytes")
        inspect = inspector()
        for role in c.ROLES:
            instances = {}
            for kind, source, region in (("timed", "call_return.mdsl", "p1_pipeline"),
                                         ("control", "connected.mdsl", "source_pattern_pipeline")):
                binary = output / "bin" / (kind + "-" + role)
                argv = [driver, output / "inputs" / source, "--region", region,
                        "--candidate", "generated-strict", "--optimization", "none", "-o", binary]
                if c.TILES[role] is not None: argv += ["--gemm-output-tiles", ",".join(map(str, c.TILES[role]))]
                log.invoke(argv, timeout=60, expected=0)
                log.pin(binary)
                result = {"binary": str(binary), "binary_identity": file_identity(binary)}
                if c.TILES[role] is not None:
                    info, files = inspect.validate_bundle(binary, c.TILES[role], region, "none", False, log.invoke)
                    c.exact(info["clang_sha256"], log.pin(cxx)["sha256"], "each issued compiler identity")
                    c.exact(info["linker_sha256"], log.pin(linker)["sha256"], "each issued linker identity")
                    result.update(manifest=info, bundle_files=files)
                    for path in Path(str(binary) + ".mdslc-private").iterdir():
                        if path.is_file(): log.pin(path)
                else:
                    c.require(not Path(str(binary) + ".mdslc-private").exists(), "stock route unexpectedly emitted a tile bundle")
                result["loaded_modules"] = dependencies(log, binary, tools)
                instances[kind] = result
            if c.TILES[role] is not None:
                join_issued(instances["timed"]["manifest"], instances["control"]["manifest"])
            result = log.invoke([instances["timed"]["binary"], "--verify-only"], expected=0)
            c.exact(result.stderr, "", "timed correctness stderr")
            receipt = c.strict_json(result.stdout)
            c.check_controls(receipt, result.returncode)
            modules_match(receipt, instances["timed"]["loaded_modules"])
            instances["timed"]["correctness"] = receipt
            canonical = log.invoke([instances["control"]["binary"]], expected=0)
            c.exact(canonical.stderr, "", "canonical correctness stderr")
            instances["control"]["correctness"] = inspect.semantic_summary(canonical.stdout, False)
            report["instances"][role] = instances
        tiled = report["instances"]["tiles-3x5"]["timed"]["manifest"]
        for kind in ("timed", "control"):
            binary = Path(report["instances"]["stock"][kind]["binary"])
            report["instances"]["stock"][kind]["stock_identity"] = stock_identity(log, binary, tools, tiled["canonical_runtime_sha256"], tiled["stock_candidates_sha256"], prefix)
        c.exact(log.pin(cxx)["sha256"], tiled["clang_sha256"], "actual configured compiler bytes")
        c.exact(log.pin(linker)["sha256"], tiled["linker_sha256"], "actual configured linker bytes")
        report["stock_profile"] = stock_profile(args.stock_profile, log, Path(report["instances"]["stock"]["timed"]["stock_identity"]["candidate_path"]), tools)
        first, second = (report["instances"][role]["timed"]["manifest"] for role in c.ROLES[1:])
        for key in ("semantic_sha256", "structured_sha256", "bufferized_sha256"):
            c.exact(first[key], second[key], "unchanged original primitive witness " + key)
        for key in ("transform_sha256", "scheduled_sha256", "llvm_sha256", "object_sha256", "dso_sha256", "binding_sha256"):
            c.require(first[key] != second[key], "different tile requests not consumed: " + key)
        log.unchanged()
        c.exact(inventory(prefix), report["installation_inventory"], "installation after prepare")
        report.update(status="PREPARED", pinned=log.pinned, budget=budget(output))
    except Exception as error:
        report["failure"] = str(error)
    finally:
        report["pinned"] = log.pinned
        write_json(output / "prepared.json", report)
    print(json.dumps({"status": report["status"], "capsule": str(output), "performance_result": False}))
    return 0 if report["status"] == "PREPARED" else 1


def host_snapshot(cpu):
    c.require(cpu in os.sched_getaffinity(0), "selected CPU not allowed by caller affinity")
    counts = Counter()
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal(): continue
        try: name = (directory / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if BUILD_NAMES.fullmatch(name): counts[name] += 1
    c.require(not counts, "concurrent build/test process names: " + repr(dict(counts)))
    load = os.getloadavg()
    total = os.cpu_count() or 1
    c.require(load[0] <= total / 4, "load1 exceeds frozen 0.25 per logical CPU guard")
    selected = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    record = {"build_test_comm_counts": dict(counts), "loadavg": load, "logical_cpus": total,
              "caller_affinity": sorted(os.sched_getaffinity(0)), "selected_cpu": cpu,
              "exclusivity_proof": "operator acknowledgement plus bounded observations, not continuous proof"}
    record["os_kernel"] = {"system": platform.system(), "release": platform.release(), "machine": platform.machine()}
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = dict((key.strip(), value.strip()) for line in block.splitlines() if ":" in line for key, value in (line.split(":", 1),))
        if fields.get("processor") == str(cpu):
            record["hardware"] = {key: fields.get(key, "unavailable") for key in ("vendor_id", "model name", "cpu family", "model", "stepping", "flags")}
            break
    for name, path in (("core_id", selected / "topology/core_id"), ("package_id", selected / "topology/physical_package_id"),
                       ("governor", selected / "cpufreq/scaling_governor"), ("scaling_khz", selected / "cpufreq/scaling_cur_freq"),
                       ("boost", Path("/sys/devices/system/cpu/cpufreq/boost"))):
        record[name] = path.read_text().strip() if path.is_file() else "unavailable"
    siblings = selected / "topology/thread_siblings_list"
    record["smt_siblings"] = siblings.read_text().strip() if siblings.is_file() else "unavailable"
    record["cache_topology"] = [{name: (directory / name).read_text().strip() if (directory / name).is_file() else "unavailable"
                                 for name in ("level", "type", "id", "size", "shared_cpu_list")}
                                for directory in sorted((selected / "cache").glob("index*"))]
    temperatures = {}
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        try: name = (directory / "name").read_text().strip()
        except OSError: continue
        if name not in ("k10temp", "coretemp", "zenpower"): continue
        for path in directory.glob("temp*_input"):
            try: temperatures[name + "/" + path.name] = int(path.read_text())
            except (OSError, ValueError): continue
    c.require(all(value < 85_000 for value in temperatures.values()), "observed CPU temperature exceeds frozen 85C guard")
    record["cpu_temperature_millicelsius"] = temperatures or "unavailable; no thermal mechanism claim"
    return record


def collect(args):
    c.require(args.exclusive_host, "--exclusive-host acknowledgement required; not automatic exclusivity")
    capsule = args.prepared.resolve(strict=True)
    prepared = read_json(capsule / "prepared.json")
    c.exact(prepared.get("status"), "PREPARED", "prepare status")
    c.exact(prepared.get("root"), str(capsule), "emitted capsule must not be relocated")
    c.exact(prepared.get("plan_sha256"), c.digest(c.plan()), "frozen plan")
    c.exact(c.digest(prepared.get("plan")), c.digest(c.plan()), "serialized plan")
    output = new_external_directory(args.output)
    log = Commands(output, dict(prepared["pinned"]))
    log.pin(capsule / "prepared.json")
    for name in ("run.py", "contract.py"): log.pin(HERE / name)
    report = {"schema": "matcore.source-pattern.p1.collection", "version": c.VERSION, "status": "INCOMPLETE",
              "plan_sha256": c.digest(c.plan()), "prepared": str(capsule / "prepared.json"),
              "prepared_identity": file_identity(capsule / "prepared.json"),
              "commands": log.records, "cells": [{**entry, "status": "NOT_RUN"} for entry in c.plan()["order"]],
              "selected_cpu": args.cpu, "control_runs": [],
              "source_runtime_policy_changed": False, "automatic_selection": False, "performance_claim": "none until reviewed"}
    started = time.monotonic()
    log.deadline = started + c.COLLECTION_SECONDS
    try:
        log.unchanged()
        c.exact(inventory(Path(prepared["installation_prefix"])), prepared["installation_inventory"], "installation before collection")
        report["host_before"] = host_snapshot(args.cpu)
        for phase in ("initial", "final"):
            for role in c.ROLES:
                binary = prepared["instances"][role]["timed"]["binary"]
                active = prepared["instances"][role]["timed"]["loaded_modules"]
                result = log.invoke([binary, "--verify-only"], cpu=args.cpu, expected=0, pins=active)
                c.exact(result.stderr, "", "control stderr")
                receipt = c.strict_json(result.stdout); c.check_controls(receipt, result.returncode)
                modules_match(receipt, prepared["instances"][role]["timed"]["loaded_modules"])
                report["control_runs"].append({"phase": phase, "role": role, "command_index": len(log.records)-1, "receipt": receipt})
                log.time_remaining()
            if phase == "final": break
            for entry in report["cells"]:
                c.require(time.monotonic() - started < c.COLLECTION_SECONDS, "collection wall budget exceeded; no resume")
                entry["host_before"] = host_snapshot(args.cpu)
                binary = prepared["instances"][entry["role"]]["timed"]["binary"]
                result = log.invoke([binary, "--measure", entry["cell"], "--cpu", args.cpu],
                                    timeout=c.PROCESS_SECONDS, cpu=args.cpu, expected=None,
                                    pins=prepared["instances"][entry["role"]]["timed"]["loaded_modules"])
                c.exact(result.stderr, "", "measurement stderr")
                receipt = c.strict_json(result.stdout)
                entry["timing_accepted"] = c.check_cell(receipt, result.returncode, entry["cell"], args.cpu)
                modules_match(receipt, prepared["instances"][entry["role"]]["timed"]["loaded_modules"])
                entry["status"] = receipt["status"]
                entry["normalized_ns"] = [sample["wall_ns"] / receipt["repetitions"] for sample in receipt["samples"]]
                entry["receipt"] = receipt
                entry["command_index"] = len(log.records)-1
                entry["host_after"] = host_snapshot(args.cpu)
                log.time_remaining()
        report["host_after"] = host_snapshot(args.cpu)
        report["summary"] = c.summarize(report["cells"])
        stable = all(role["status"] == "PAIRED" for item in report["summary"] for role in item["roles"].values())
        report["status"] = "COMPLETE" if stable else "INCONCLUSIVE_TIMING"
        log.time_remaining()
    except Exception as error:
        report["failure"] = str(error)
    finally:
        try:
            log.unchanged()
            c.exact(inventory(Path(prepared["installation_prefix"])), prepared["installation_inventory"], "installation after collection")
            log.time_remaining()
            report["identity_checks_passed"] = True
        except Exception as error:
            report.update(status="INCOMPLETE", identity_error=str(error), identity_checks_passed=False)
        report["budget"] = budget(output)
        report["raw_catalog"] = {str(path.relative_to(output)): file_identity(path) for path in sorted((output / "raw").rglob("*")) if path.is_file()}
        report["collection_wall_seconds"] = time.monotonic() - started
        if report["collection_wall_seconds"] > c.COLLECTION_SECONDS:
            report.update(status="INCOMPLETE", deadline_error="wall budget includes controls, endpoint identities and evidence catalog; no resume")
        write_json(output / "collection.json", report)
    print(json.dumps({"status": report["status"], "evidence": str(output / "collection.json"), "automatic_selection": False}))
    return 0 if report["status"] == "COMPLETE" else 78 if report["status"] == "INCONCLUSIVE_TIMING" else 1


def audit_collection(path):
    """Rejoin stored derivations to raw process stdout/exit before publishing."""
    collection = read_json(path)
    c.require(collection.get("status") in ("COMPLETE", "INCONCLUSIVE_TIMING") and collection.get("identity_checks_passed") is True, "incomplete/identity-drift collection")
    c.exact(collection["plan_sha256"], c.digest(c.plan()), "summary plan")
    c.require(collection.get("collection_wall_seconds", c.COLLECTION_SECONDS + 1) <= c.COLLECTION_SECONDS, "collection exceeded hard wall budget")
    root = path.resolve().parent
    expected_files = {record["directory"] + "/" + name for record in collection["commands"]
                      for name in ("stdout.log", "stderr.log", "command.json")}
    c.exact(set(collection["raw_catalog"]), expected_files, "complete raw artifact catalog")
    c.exact({str(artifact.relative_to(root)) for artifact in (root / "raw").rglob("*") if artifact.is_file()}, expected_files, "extra/missing raw artifact files")
    for relative, identity in collection["raw_catalog"].items():
        artifact = (root / relative).resolve(strict=True)
        c.require(artifact.is_relative_to(root), "raw catalog escaped evidence root")
        actual = file_identity(artifact)
        for key in ("sha256", "bytes"):
            c.exact(actual[key], identity[key], "raw artifact " + relative + " " + key)
    prepared_path = Path(collection["prepared"])
    prepared = read_json(prepared_path)
    c.exact(file_identity(prepared_path), collection["prepared_identity"], "prepared receipt identity")
    cpu = collection["selected_cpu"]
    used = set()

    def raw(index, expected_argv):
        c.require(type(index) is int and index not in used and 0 <= index < len(collection["commands"]), "duplicate/missing command join")
        used.add(index)
        record = collection["commands"][index]
        c.exact(record["directory"], f"raw/{index:04d}", "one-to-one chronological raw directory")
        c.exact(record["argv"], list(map(str, expected_argv)), "raw command argv")
        directory = (root / record["directory"]).resolve(strict=True)
        c.require(directory.is_relative_to(root), "command escaped evidence root")
        c.exact(read_json(directory / "command.json"), record, "command record bytes/metadata")
        c.exact((directory / "stderr.log").read_text(), "", "process stderr")
        receipt = c.strict_json((directory / "stdout.log").read_text())
        return receipt, record["returncode"]

    controls = collection["control_runs"]
    expected_controls = [(phase, role) for phase in ("initial", "final") for role in c.ROLES]
    c.exact([(item["phase"], item["role"]) for item in controls], expected_controls, "initial/final control inventory")
    for item in controls:
        c.exact(item["command_index"], (0 if item["phase"] == "initial" else 3 + len(c.plan()["order"])) + c.ROLES.index(item["role"]), "control chronological command index")
        instance = prepared["instances"][item["role"]]["timed"]
        receipt, code = raw(item["command_index"], [instance["binary"], "--verify-only"])
        c.exact(receipt, item["receipt"], "stored control/raw receipt")
        c.check_controls(receipt, code); modules_match(receipt, instance["loaded_modules"])
    for ordinal, entry in enumerate(collection["cells"]):
        c.exact(entry["command_index"], 3 + ordinal, "measurement chronological command index")
        instance = prepared["instances"][entry["role"]]["timed"]
        receipt, code = raw(entry["command_index"], [instance["binary"], "--measure", entry["cell"], "--cpu", cpu])
        c.exact(receipt, entry["receipt"], "stored measured/raw receipt")
        accepted = c.check_cell(receipt, code, entry["cell"], cpu)
        c.exact(accepted, entry["timing_accepted"], "derived timing acceptance")
        c.exact(entry["status"], receipt["status"], "derived status")
        c.exact(entry["normalized_ns"], [sample["wall_ns"] / receipt["repetitions"] for sample in receipt["samples"]], "normalized timing must derive from raw clocks")
        modules_match(receipt, instance["loaded_modules"])
    c.exact(used, set(range(len(collection["commands"]))), "unjoined extra/missing process")
    summary = c.summarize(collection["cells"])
    c.exact(summary, collection["summary"], "stored summary/raw recomputation")
    stable = all(role["status"] == "PAIRED" for item in summary for role in item["roles"].values())
    c.exact(collection["status"], "COMPLETE" if stable else "INCONCLUSIVE_TIMING", "outer status/raw evidence")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = modes.add_parser("prepare")
    prepare_parser.add_argument("--driver", type=Path, required=True)
    prepare_parser.add_argument("--prefix", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--stock-profile", type=Path, required=True)
    prepare_parser.add_argument("--cxx", type=Path, default=Path("/usr/bin/clang++-21"))
    prepare_parser.add_argument("--linker", type=Path, default=Path("/usr/bin/ld.lld-21"))
    collect_parser = modes.add_parser("collect")
    collect_parser.add_argument("--prepared", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--cpu", type=int, required=True)
    collect_parser.add_argument("--exclusive-host", action="store_true")
    summary_parser = modes.add_parser("summarize")
    summary_parser.add_argument("--collection", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "summarize":
        print(json.dumps(audit_collection(args.collection), indent=2, allow_nan=False))
        return 0
    c.require(platform.system() == "Linux" and platform.machine() == "x86_64", "P1 is bounded to Linux x64")
    return prepare(args) if args.mode == "prepare" else collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("P1 refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
