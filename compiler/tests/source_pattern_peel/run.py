#!/usr/bin/env python3
"""Separate peeled source request controls; never an artifact importer.

Only the existing sibling source_pattern/run.py is a helper dependency. Both
files can be copied with their sibling layout after producer removal. The old
runner, its v1 classifier and the frozen P1 harness remain unmodified.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "source_pattern"
_spec = importlib.util.spec_from_file_location("mdslc_source_peel_legacy_helpers_v1", LEGACY / "run.py")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
SCHEMA = "matcore-source-output-peel-n-v1"
DOMAIN = "matcore-source-output-peel-n-binding-v1"
REALIZATION = "output-tiled-peel-n-mkn"
KEYS = legacy.KEYS | {"realization"}
SSA = r"%[A-Za-z0-9_.$-]+"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def binding_digest(prefix):
    values = (DOMAIN.encode(), prefix.encode())
    return hashlib.sha256(b"".join(str(len(value)).encode() + b":" + value for value in values)).hexdigest()


def manifest(text):
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        require(separator and key and key not in fields, "malformed/duplicate peeled source field")
        fields[key] = value
    require(fields.get("schema") == SCHEMA and fields.get("realization") == REALIZATION,
            "not an explicitly peeled source instance")
    # Reuse the unchanged old field-format and target-independent ownership
    # checks, without changing that module's globals or accepting its schema.
    old_fields = dict(fields)
    del old_fields["realization"]
    old_fields["schema"] = "matcore-source-output-pattern-v1"
    legacy.manifest("".join(key + "=" + value + "\n" for key, value in old_fields.items()))
    marker = "binding_sha256="
    require(text.count(marker) == 1 and text.endswith("\n"), "noncanonical peeled binding record")
    prefix, suffix = text.split(marker)
    require(prefix.startswith("schema=" + SCHEMA + "\n") and
            set(suffix.splitlines()[1:]) == {
                "candidate_soname=" + fields["candidate_soname"],
                "dso_sha256=" + fields["dso_sha256"],
                "manifest_authority=inspection_only_never_imported"},
            "peeled identity fields moved across binding boundary")
    require(binding_digest(prefix) == fields["binding_sha256"], "peeled source binding digest/domain differs")
    return fields


def _lines(text):
    return [line.partition("//")[0].strip() for line in text.splitlines() if line.partition("//")[0].strip()]


class Scope:
    def __init__(self, header="", parent=None, bindings=None):
        self.header, self.parent = header, parent
        self.children, self.defs, self.operations = [], {}, []
        self.bindings = bindings or {}

    def lookup(self, name):
        scope = self
        while scope is not None:
            if name in scope.defs:
                return scope.defs[name]
            scope = scope.parent
        raise ValueError("unbound canonical SSA value: " + name)


def _visible(scope):
    result = {}
    ancestry = []
    while scope is not None:
        ancestry.append(scope)
        scope = scope.parent
    for ancestor in reversed(ancestry):
        result.update(ancestor.defs)
    return result


def _header_value(scope, name):
    require(name in scope.bindings, "loop bound/step is not defined before its use")
    return scope.bindings[name]


def _schedule(text):
    root = Scope()
    stack = [root]
    for line in _lines(text):
        if line == "}":
            require(len(stack) > 1, "unbalanced canonical schedule")
            stack.pop()
            continue
        scope = stack[-1]
        if line.endswith("{"):
            child = Scope(line, scope, _visible(scope))
            scope.children.append(child)
            scope.operations.append(child)
            stack.append(child)
        else:
            scope.operations.append(line)
            definition = re.match(rf"^({SSA}) = (.+)$", line)
            if definition:
                name, expression = definition.groups()
                require(name not in scope.defs, "duplicate canonical SSA definition")
                # Snapshot visible bindings at this statement. Later definitions
                # and sibling-region SSA names must not repair an unbound use.
                scope.defs[name] = (expression, _visible(scope))
    require(len(stack) == 1 and len(root.children) == 1 and root.children[0].header == "module {",
            "not one closed canonical schedule module")
    module = root.children[0]
    require(len(module.children) == 1, "not one canonical scheduled function")
    function = module.children[0]
    signature = re.fullmatch(rf"func.func @__matcore_strict_gemm_f32_v1\(({SSA}): memref<\?x\?xf32>, ({SSA}): memref<\?x\?xf32>, ({SSA}): memref<\?x\?xf32>\) -> memref<\?x\?xf32> \{{", function.header)
    require(signature, "unexpected scheduled function identity/signature")
    return function, signature.groups()


def _constant(definition, expected):
    return definition[0] == f"arith.constant {expected} : index"


def _dimension(definition, matrix, axis):
    match = re.fullmatch(rf"memref.dim ({SSA}), ({SSA}) : memref<\?x\?xf32>", definition[0])
    return bool(match and match[1] == matrix and match[2] in definition[1] and
                _constant(definition[1][match[2]], axis))


def _loop(scope):
    match = re.fullmatch(rf"scf.for ({SSA}) = ({SSA}) to ({SSA}) step ({SSA}) \{{", scope.header)
    require(match, "unexpected canonical output loop")
    return match.groups()


def output_tile_consumption(transform, scheduled, tiles):
    """Bounded canonical-emission discriminator, NOT general MLIR validation.

    C++ issuance independently proves all arithmetic/views/recurrences and exact
    replay. This classifier proves actual request consumption, not legal import.
    """
    tm, tn = tiles
    require(type(tm) is int and type(tn) is int and 1 <= tm <= 64 and 1 <= tn <= 64,
            "unclosed tile request")
    instructions = _lines(transform)
    tile_ops = [line for line in instructions if "transform.structured.tile_using_for" in line]
    require(len(tile_ops) == 1, "not one actual tile Transform")
    tiled = re.fullmatch(rf"({SSA}), ({SSA}), ({SSA}) = transform.structured.tile_using_for ({SSA}) tile_sizes \[{tm}, {tn}, 0\] : .+", tile_ops[0])
    require(tiled, "actual Transform does not consume requested M/N, untiled K")
    require(len(set(tiled.groups()[:3])) == 3, "output tiling handles are not distinct")
    matches = [line for line in instructions if "transform.structured.match" in line]
    entries = [line for line in instructions if line.startswith("transform.named_sequence ")]
    require(len(matches) == len(entries) == 1, "not one actual closed matmul/root match")
    matched = re.fullmatch(rf'({SSA}) = transform.structured.match ops\{{\["linalg.matmul"\]\}} in ({SSA}) : .+', matches[0])
    entry = re.fullmatch(rf"transform.named_sequence @__transform_main\(({SSA}): !transform.any_op(?: \{{transform.readonly\}})?\) \{{", entries[0])
    require(matched and entry and matched[1] == tiled[4] and matched[2] == entry[1] and
            instructions.index(matches[0]) < instructions.index(tile_ops[0]),
            "actual tile does not consume the root's canonical matmul match")
    generalize = [line for line in instructions if "transform.structured.generalize" in line]
    interchange = [line for line in instructions if "transform.structured.interchange" in line]
    require(len(generalize) == len(interchange) == 1, "not one generalize/interchange chain")
    generic = re.fullmatch(rf"({SSA}) = transform.structured.generalize {re.escape(tiled[1])} : .+", generalize[0])
    require(generic and re.fullmatch(rf"{SSA} = transform.structured.interchange {re.escape(generic[1])} iterator_interchange = \[0, 2, 1\] : .+", interchange[0]),
            "actual local M/K/N scheduling chain changed")
    require(instructions.index(tile_ops[0]) < instructions.index(generalize[0]) < instructions.index(interchange[0]),
            "Transform handle consumption order changed")
    peels = [line for line in instructions if "transform.loop.peel" in line]
    require(len(peels) == (tn != 1), "unit-N identity or nonunit peel count differs")
    operations = [line for line in instructions if not line.startswith("transform.named_sequence ") and
                  re.match(rf"^(?:(?:{SSA})(?:, {SSA})* = )?transform\.", line)]
    require(len(operations) == (5 if tn == 1 else 6) and operations[-1] == "transform.yield",
            "unexpected extra Transform program or missing yield")
    if tn != 1:
        require(re.fullmatch(rf"{SSA}, {SSA} = transform.loop.peel {re.escape(tiled[3])} : .+", peels[0]) and
                instructions.index(peels[0]) > instructions.index(interchange[0]),
                "actual peel does not consume the tiled N handle after interchange")
    require(instructions == ["module attributes {transform.with_named_sequence} {", entries[0],
                             matches[0], tile_ops[0], generalize[0], interchange[0],
                             *peels, "transform.yield", "}", "}"],
            "Transform operations escaped the exact named-sequence enclosure/order")

    function, (a, b, c) = _schedule(scheduled)
    require(len(function.children) == 1, "not one shared M output loop")
    outer = function.children[0]
    m, lower, upper, step = _loop(outer)
    require(_constant(_header_value(outer, lower), 0) and _constant(_header_value(outer, step), tm) and
            _dimension(_header_value(outer, upper), a, 0), "M loop does not consume requested step/extent")
    require(len(outer.children) == (1 if tn == 1 else 2), "full/tail loop ownership differs")
    first = outer.children[0]
    n, lower, full_end, step = _loop(first)
    require(_constant(_header_value(first, lower), 0) and _constant(_header_value(first, step), tn),
            "full N lower bound/step changed")
    if tn == 1:
        require(_dimension(_header_value(first, full_end), b, 1), "unit N must cover exactly original N")
    else:
        boundary = _header_value(first, full_end)
        split = re.fullmatch(rf"affine.apply affine_map<\(\)\[s0\] -> \(\(s0 floordiv {tn}\) \* {tn}\)>\(\)\[({SSA})\]", boundary[0])
        require(split and split[1] in boundary[1] and _dimension(boundary[1][split[1]], b, 1),
                "full N split is not tile_n*floor(original_N/tile_n)")
        tail = outer.children[1]
        tail_n, tail_start, tail_end, tail_step = _loop(tail)
        require(_header_value(tail, tail_start) is boundary and _dimension(_header_value(tail, tail_end), b, 1) and
                _constant(_header_value(tail, tail_step), tn), "tail does not cover exactly [F,N) with same step")
    # Actual unique A/B/C slices, full width and remainder width, and one generic
    # per domain. This is intentionally smaller than the production body proof.
    for index, domain in enumerate(outer.children):
        iv = _loop(domain)[0]
        require(len(domain.children) == 1 and domain.children[0].header.startswith("linalg.generic "),
                "not one actual generic in each output domain")
        require(not domain.children[0].children, "unexpected nested output-domain region")
        views = []
        for name, definition in domain.defs.items():
            view = re.fullmatch(rf"memref.subview ({SSA})\[([^]]+)\] \[([^]]+)\] \[1, 1\] : .+", definition[0])
            if view:
                views.append((name, view[1], view[2].split(", "), view[3].split(", "), definition[1], definition))
        require(len(views) == 3 and [item[1] for item in views] == [a, b, c],
                "not unique original A/B/C slices in each domain")
        require(views[0][2] == [m, "0"] and views[1][2] == ["0", iv] and views[2][2] == [m, iv],
                "domain slice origins changed")
        require(len(views[0][3]) == len(views[1][3]) == len(views[2][3]) == 2,
                "slice ranks changed")
        require(views[0][3][0] == views[2][3][0] and views[0][3][1] == views[1][3][0] and
                views[0][3][1] in views[0][4] and
                _dimension(views[0][4][views[0][3][1]], a, 1), "shared M/K slice bounds changed")
        m_width = views[0][3][0]
        if tm == 1 and m_width == "1":
            pass  # Canonical unit-M loop makes its bounded local extent one.
        else:
            require(m_width in views[0][4], "M width is not defined before slice")
            m_bound = views[0][4][m_width]
            minimum = re.fullmatch(rf"affine.min affine_map<\(d0\)\[s0\] -> \(-d0 \+ s0, {tm}\)>\(({SSA})\)\[({SSA})\]", m_bound[0])
            require(minimum and minimum[1] == m and minimum[2] in m_bound[1] and
                    _dimension(m_bound[1][minimum[2]], a, 0), "M slice does not consume bounded requested tile")
        generic = domain.children[0].header
        require(all(domain.children[0].bindings.get(view[0]) is view[5] for view in views),
                "generic slice definition does not dominate actual consumption")
        require(f"ins({views[0][0]}, {views[1][0]} :" in generic and f"outs({views[2][0]} :" in generic,
                "generic does not consume the actual A/B/C slices")
        remaining = views[1][3][1]
        require(views[2][3][1] == remaining, "B/C N slice widths differ")
        if tn == 1 or index == 0:
            require(remaining == str(tn), "full domain N slice is not exactly tile_n")
        else:
            require(remaining in views[1][4], "tail width is not defined before slice")
            definition = views[1][4][remaining]
            extent = re.fullmatch(rf"affine.apply affine_map<\(d0\)\[s0\] -> \(-d0 \+ s0\)>\(({SSA})\)\[({SSA})\]", definition[0])
            require(extent and extent[1] == iv and extent[2] in definition[1] and
                    _dimension(definition[1][extent[2]], b, 1), "tail slice width is not original_N-origin")
    return {"transform_tile_sizes": [tm, tn, 0], "outer_loop_steps": [tm] + [tn] * len(outer.children),
            "n_domains": "unit-identity" if tn == 1 else "disjoint-full-tail"}


def validate_bundle(binary, tiles, region, optimization, sanitized, invoke):
    bundle = Path(str(binary) + ".mdslc-private")
    require(bundle.is_dir() and not bundle.is_symlink() and bundle.stat().st_mode & 0o777 == 0o700,
            "peeled source bundle is not private 0700 directory")
    info = manifest((bundle / "source-pattern.manifest").read_text())
    expected = {"pattern_parameters": f"M:{tiles[0]},N:{tiles[1]},K:untiled",
                "region_name_hex": region.encode().hex(), "candidate": "generated-strict",
                "optimization": optimization, "address_sanitizer": "function_attributes_and_object" if sanitized else "off",
                "target": {"x86_64": "x86_64-pc-linux-gnu", "aarch64": "aarch64-unknown-linux-gnu"}[platform.machine()]}
    require(all(info[key] == value for key, value in expected.items()), "peeled source request/profile binding differs")
    for key, name in legacy.HASH_FILES.items():
        require(legacy.digest(bundle / name) == info[key], "changed issued bytes: " + name)
    consumption = output_tile_consumption((bundle / "gemm.ll.transform.mlir").read_text(),
                                          (bundle / "gemm.ll.scheduled.mlir").read_text(), tiles)
    issuer = dict(line.split("=", 1) for line in (bundle / "gemm.ll.manifest").read_text().splitlines())
    require(issuer.get("schedule") == REALIZATION and issuer.get("pattern") == "matcore-gemm-output-tile-peel-n-v1",
            "source request silently used unpeeled issuer")
    dso = bundle / info["candidate_soname"]
    require(legacy.digest(dso) == info["dso_sha256"], "changed private candidate DSO")
    dependencies = legacy.needed(invoke(["readelf", "-d", binary]).stdout)
    require(dependencies.count(info["candidate_soname"]) == 1 and "libmatcore_runtime.so.0" in dependencies and
            not any(name == "libmatcore_closed_candidates_isolated_v1.so" or
                    (name.startswith("libmdslc-pattern-") and name != info["candidate_soname"]) for name in dependencies),
            "executable does not consume exactly its private DSO and canonical Runtime")
    dynamic = invoke(["readelf", "-d", dso]).stdout
    require("(SYMBOLIC)" in dynamic and "libmatcore_runtime.so.0" in legacy.needed(dynamic),
            "private DSO lost local binding/canonical Runtime")
    require(not any("LLVM" in name or "MLIR" in name or "openblas" in name.lower() for name in legacy.needed(dynamic)),
            "private DSO acquired compiler/provider ownership")
    definitions = [line.split() for line in invoke(["nm", "--defined-only", "--format=posix", dso]).stdout.splitlines()]
    for symbol in ("__matcore_strict_gemm_f32_v1", "_mlir_ciface___matcore_strict_gemm_f32_v1"):
        require([line[1] for line in definitions if line and line[0] == symbol] == ["t"], "leaf/wrapper is not uniquely local")
    exports = invoke(["nm", "--dynamic", "--defined-only", "--format=posix", dso]).stdout
    require("__matcore_strict_gemm" not in exports and "_mlir_ciface_" not in exports, "leaf became substitutable")
    llvm = (bundle / "gemm.ll").read_text()
    require("fmul float" in llvm and "fadd float" in llvm and "llvm.fma" not in llvm and " fast " not in llvm and
            ("sanitize_address" in llvm) == sanitized, "strict arithmetic/instrumentation changed")
    return info, {str(path.relative_to(bundle)): legacy.digest(path) for path in sorted(bundle.iterdir()) if path.is_file()}


def end_k_summary(text):
    match = re.fullmatch(r"Source peel end-K oracle: (\d+) cases; (\d+) checks; 0 failures; full=(\d+) tail1=(\d+) tail2=(\d+); inapplicable-K=(\d+)\n", text)
    require(match and tuple(map(int, (match[1], match[3], match[4], match[5], match[6]))) == (792, 728, 364, 364, 40) and int(match[2]) > 10000,
            "incomplete source-observable end-K/full/tail corpus")
    return {"cases": int(match[1]), "checks": int(match[2]), "full": int(match[3]),
            "tail1": int(match[4]), "tail2": int(match[5]), "inapplicable_K": int(match[6])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sanitized", action="store_true")
    parser.add_argument("--cxx", default="/usr/bin/clang++-21")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--case", choices=("connected", "refusals"), default="connected")
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() in ("x86_64", "aarch64"), "native Linux baseline only")
    driver = args.driver.resolve(strict=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output.resolve()
    else:
        root = Path(tempfile.mkdtemp(prefix="mdslc-source-peel-"))
    old_inputs, new_inputs = root / "inputs/source_pattern", root / "inputs/source_pattern_peel"
    old_inputs.mkdir(parents=True)
    new_inputs.mkdir()
    for name in ("connected.mdsl", "math_helper.h", "forwarding.mdsl", "same_value.mdsl", "rhs.mdsl"):
        shutil.copyfile(LEGACY / name, old_inputs / name)
    shutil.copyfile(HERE / "end_k.mdsl", new_inputs / "end_k.mdsl")
    environment = dict(os.environ, LC_ALL="C", PYTHONDONTWRITEBYTECODE="1",
                       ASAN_OPTIONS="halt_on_error=1:detect_leaks=0:exitcode=1:symbolize=1",
                       UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1", DEBUGINFOD_URLS="")
    for name in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    pinned = {str(driver): legacy.digest(driver)}
    report = {"schema": "source-output-peel-test-v1", "case": args.case, "status": "running",
              "driver": str(driver), "driver_sha256": pinned[str(driver)], "commands": [], "instances": []}
    evidence = root / "evidence.json"

    def save():
        evidence.write_text(json.dumps(report, indent=2) + "\n")

    def invoke(argv, expected=0):
        require(all(legacy.digest(path) == sha for path, sha in pinned.items()), "frozen test artifact changed")
        result = subprocess.run(list(map(str, argv)), cwd=root, env=environment, text=True, capture_output=True, timeout=180)
        report["commands"].append({"argv": list(map(str, argv)), "exit": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        save()
        require(result.returncode == expected, "unexpected command result: " + repr(argv))
        return result

    def compile_one(name, tiles, fixture="connected.mdsl", region="source_pattern_pipeline", old=False, optimization="none"):
        binary = root / name
        source = (new_inputs if fixture == "end_k.mdsl" else old_inputs) / fixture
        invoke([driver, source, "--region", region, "--candidate", "generated-strict",
                "--gemm-output-tiles" if old else "--gemm-output-tiles-peel-n", f"{tiles[0]},{tiles[1]}",
                "--optimization", optimization, "-o", binary])
        info, hashes = (legacy.validate_bundle if old else validate_bundle)(binary, tiles, region, optimization, args.sanitized, invoke)
        pinned[str(binary)] = legacy.digest(binary)
        pinned.update({str(Path(str(binary) + ".mdslc-private") / name): sha for name, sha in hashes.items()})
        report["instances"].append({"tiles": tiles, "old": old, "manifest": info, "bundle_sha256": hashes})
        return binary, info

    try:
        require(("ASan+UBSan" in invoke([driver, "--version"]).stdout) == args.sanitized, "wrong installation profile")
        if args.case == "refusals":
            base = [old_inputs / "connected.mdsl", "--region", "source_pattern_pipeline", "--candidate", "generated-strict"]
            bad = [[*base, "--gemm-output-tiles-peel-n", text] for text in
                   ("0,5", "3,0", "65,5", "3,65", "03,5", "+3,5", "3,-1", "3", "3,5,1", "3,", ",5", "3,5x", "18446744073709551616,5")]
            bad += [[*base, "--gemm-output-tiles-peel-n"],
                    [*base, "--gemm-output-tiles-peel-n", "3,5", "--gemm-output-tiles-peel-n", "5,3"],
                    [*base, "--gemm-output-tiles", "3,5", "--gemm-output-tiles-peel-n", "3,5"],
                    [*base, "--gemm-output-tiles-peel-n", "3,5", "--gemm-output-tiles", "3,5"],
                    [*base, "--gemm-output-tiles-peel-n", "3,5", "-c"],
                    [*base, "--gemm-output-tiles-peel-n", "3,5", "--optimization", "strict-fused-pair"],
                    [old_inputs / "connected.mdsl", "--region", "source_pattern_pipeline", "--gemm-output-tiles-peel-n", "3,5"],
                    [*base, "--candidate", "generated-strict", "--gemm-output-tiles-peel-n", "3,5"],
                    ["--program", "--host", old_inputs / "connected.mdsl", "--region", old_inputs / "connected.mdsl", "source_pattern_pipeline", "--gemm-output-tiles-peel-n", "3,5"]]
            for candidate in ("automatic", "native-strict", "generated-reassociate", "generated-strict-avx", "generated-strict-avx2", "generated-strict-avx512f", "generated-nvvm", "generated-rocdl", "existing-native", "openblas"):
                bad.append([*base[:-1], candidate, "--gemm-output-tiles-peel-n", "3,5"])
            forged = root / "forged.mlir"
            forged.write_text("module attributes {mdsl.execution_authority} {}\n")
            for option in ("--input", "--mlir-input", "--llvm-input", "--gemm-ir", "--gemm-manifest", "--isa", "--target"):
                bad.append([*base, "--gemm-output-tiles-peel-n", "3,5", option, forged])
            for index, arguments in enumerate(bad):
                binary = root / f"refused-{index}"
                result = invoke([driver, *arguments, "-o", binary], expected=1)
                require(result.stderr.startswith("mdslc-region:") and not binary.exists() and
                        not Path(str(binary) + ".mdslc-private").exists(), "refusal published or failed for unrelated reason")
            report["refusals"] = len(bad)
        else:
            probe = root / "raw-asan-probe"
            if args.sanitized:
                require(args.runtime, "actual DSO ASan controls require canonical Runtime path")
                runtime = args.runtime.resolve(strict=True)
                pinned[str(runtime)] = legacy.digest(runtime)
                invoke([args.cxx, HERE / "raw_asan_probe.cpp", "-std=c++20", "-O1", "-g",
                        "-fsanitize=address,undefined", "-Xlinker", "--no-as-needed", runtime,
                        "-ldl", "-pthread", "-Xlinker", "-rpath", "-Xlinker", runtime.parent, "-o", probe])
            records = []
            for tiles in ((3, 5), (5, 3), (5, 1)):
                binary, info = compile_one(f"connected-{tiles[0]}-{tiles[1]}", tiles)
                result = invoke([binary])
                require(not result.stderr, "positive source emitted diagnostics")
                legacy.semantic_summary(result.stdout, args.sanitized)
                require("FAIL source pattern:" in invoke([binary, "--corrupt-output"], expected=1).stderr,
                        "corruption control failed for unrelated reason")
                records.append(info)
                if args.sanitized:
                    dso = Path(str(binary) + ".mdslc-private") / info["candidate_soname"]
                    offset = legacy.private_wrapper_offset(dso.read_bytes(), 62 if platform.machine() == "x86_64" else 183)
                    for width in sorted({1, tiles[1]}):
                        negative = invoke([probe, dso, f"{offset:x}", width], expected=1)
                        require("AddressSanitizer: heap-buffer-overflow" in negative.stderr and
                                re.search(r"#0[^\n]*(_mlir_ciface_)?__matcore_strict_gemm_f32_v1", negative.stderr) and
                                info["candidate_soname"] in negative.stderr,
                                "ASan negative did not reach actual full/tail issued DSO leaf")
                if tiles[1] != 1:
                    for fixture, region, expected in (
                        ("same_value.mdsl", "source_pattern_same_value", "Source pattern same Value: 5 cases; 480 checks; 0 failures\n"),
                        ("rhs.mdsl", "source_pattern_rhs", "Source pattern RHS carry: 8 cases; 256 checks; 0 failures\n"),
                    ):
                        operand, _ = compile_one(f"{fixture[:-5]}-{tiles[0]}-{tiles[1]}", tiles, fixture, region)
                        observed = invoke([operand])
                        require(observed.stdout == expected and not observed.stderr,
                                "same-Value/RHS operand-role source oracle failed")
                    sensitive, _ = compile_one(f"end-k-{tiles[0]}-{tiles[1]}", tiles, "end_k.mdsl", "source_peel_end_k")
                    result = invoke([sensitive, tiles[0], tiles[1]])
                    require(not result.stderr, "end-K source emitted diagnostics")
                    report["instances"][-1]["end_K"] = end_k_summary(result.stdout)
                    require("FAIL source peel end-K:" in invoke([sensitive, tiles[0], tiles[1], "--corrupt-output"], expected=1).stderr,
                            "end-K corruption control failed for unrelated reason")
            _, old_unit = compile_one("old-connected-5-1", (5, 1), old=True)
            old_unit_result = invoke([root / "old-connected-5-1"])
            require(not old_unit_result.stderr, "positive old-unit source emitted diagnostics")
            legacy.semantic_summary(old_unit_result.stdout, args.sanitized)
            for key in ("semantic_sha256", "structured_sha256", "bufferized_sha256", "transform_sha256", "scheduled_sha256", "llvm_sha256", "object_sha256"):
                require(old_unit[key] == records[2][key], "unit-N identity unexpectedly changed computation: " + key)
            for key in ("binding_sha256", "candidate_soname", "dso_sha256"):
                require(old_unit[key] != records[2][key], "old/new unit-N source identity collapsed: " + key)
            for key in ("semantic_sha256", "structured_sha256", "bufferized_sha256"):
                require(records[0][key] == records[1][key], "HOW contaminated unchanged WHAT witness")
            for key in ("transform_sha256", "scheduled_sha256", "llvm_sha256", "object_sha256", "dso_sha256"):
                require(records[0][key] != records[1][key], "different nonunit requests were not consumed: " + key)
            forwarded, _ = compile_one("forwarded-5-3", (5, 3), "forwarding.mdsl", "source_pattern_forwarding", optimization="publication-read-forwarding")
            result = invoke([forwarded])
            require(not result.stderr and re.fullmatch(r"Source pattern eligible forwarding: 8 cases; [0-9]+ checks; 0 failures\n", result.stdout),
                    "checked forwarding source oracle failed")
            require("closed_host_v1::SessionAbiV2::readForwarded" in invoke(["nm", "--undefined-only", "--demangle", forwarded]).stdout,
                    "requested eligible forwarding was not consumed")
        require(all(legacy.digest(path) == sha for path, sha in pinned.items()), "frozen final artifacts changed")
        report["status"] = "pass"
        save()
        print(f"PASS independent source peel {args.case}; evidence={evidence}")
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status="fail", failure=str(error))
        save()
        raise SystemExit(str(error) + "; evidence=" + str(evidence))


if __name__ == "__main__":
    main()
