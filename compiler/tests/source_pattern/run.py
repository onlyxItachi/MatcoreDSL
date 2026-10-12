#!/usr/bin/env python3
"""Independent source-time tile requests, artifact linkage and semantic controls.

This consumes an already-built driver. Manifest inspection is an oracle, not an
IR importer or a substitute for the driver owning and authenticating its bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import struct
import subprocess
import tempfile


KEYS = frozenset((
    "schema", "pattern_parameters", "source_context_sha256", "region_name_hex",
    "source_authority", "manifest_authority", "public_header_sha256", "storage_header_sha256",
    "private_header_sha256", "provider_sha256", "accelerator_count",
    "candidate", "optimization", "toolchain", "target", "address_sanitizer",
    "clang_sha256", "linker_sha256", "relink_kit_sha256", "exports_sha256",
    "canonical_runtime_sha256", "stock_candidates_sha256", "issuer_manifest_sha256",
    "semantic_sha256", "structured_sha256", "bufferized_sha256", "transform_sha256",
    "scheduled_sha256", "llvm_sha256", "object_sha256", "dso_sha256",
    "candidate_soname", "binding_sha256",
))
HASH_FILES = {
    "issuer_manifest_sha256": "gemm.ll.manifest",
    "semantic_sha256": "gemm.ll.semantic.mlir",
    "structured_sha256": "gemm.ll.structured.mlir",
    "bufferized_sha256": "gemm.ll.bufferized.mlir",
    "transform_sha256": "gemm.ll.transform.mlir",
    "scheduled_sha256": "gemm.ll.scheduled.mlir",
    "llvm_sha256": "gemm.ll",
    "object_sha256": "gemm.o",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(text):
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise ValueError("malformed or duplicate source-pattern manifest field")
        result[key] = value
    if not re.fullmatch(r"[0-9]+", result.get("accelerator_count", "")):
        raise ValueError("malformed accelerator count")
    count = int(result["accelerator_count"])
    if count > 2:
        raise ValueError("unclosed accelerator identity set")
    expected_keys = KEYS | {f"accelerator_{index}_sha256" for index in range(count)}
    if set(result) != expected_keys:
        raise ValueError("source-pattern manifest fields differ: " +
                         repr(sorted(set(result) ^ expected_keys)))
    if result["schema"] != "matcore-source-output-pattern-v1":
        raise ValueError("wrong source-pattern inspection schema")
    if result["toolchain"] != "21.1.8":
        raise ValueError("source-pattern inspection toolchain is not exactly the product tuple")
    if result["source_authority"] != "authenticated_in_process_source_closure" or result["manifest_authority"] != "inspection_only_never_imported":
        raise ValueError("inspection manifest falsely claims imported authority")
    for key, value in result.items():
        if key == "provider_sha256" and value == "none":
            continue
        if key == "source_context_sha256":
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise ValueError("malformed authenticated source context identity")
            continue
        if key.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("malformed digest " + key)
    if not re.fullmatch(r"libmdslc-pattern-[0-9a-f]{64}\.so", result["candidate_soname"]):
        raise ValueError("candidate soname is not a closed private instance name")
    if result["candidate_soname"] != "libmdslc-pattern-" + result["binding_sha256"] + ".so":
        raise ValueError("candidate soname is not bound to this exact instance")
    return result


def needed(text):
    return re.findall(r"\(NEEDED\).*?Shared library:\s*\[([^\]]+)\]", text)


def output_tile_consumption(transform, scheduled, tiles):
    """Bounded canonical-emission classifier, never an MLIR parser/importer.

    Check actual output-tile instruction and resolved outer loop increments;
    distinct files or comments claiming parameters cannot satisfy consumption.
    Full IR/numerical legality remains the issuer's independent responsibility.
    """
    ssa = r"%[A-Za-z0-9_.$-]+"
    assignment = rf"{ssa}(?:\s*,\s*{ssa})*\s*=\s*"
    code_lines = lambda value: [line.partition("//")[0] for line in value.splitlines()]
    transforms = [line for line in code_lines(transform) if re.match(
        rf"^\s*{assignment}transform\.structured\.tile_using_for\b", line)]
    if len(transforms) != 1:
        raise ValueError("expected exactly one actual output tile Transform instruction")
    matched = re.fullmatch(
        rf"\s*{assignment}transform\.structured\.tile_using_for\s+{ssa}\s+"
        r"tile_sizes\s*\[\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*\]\s*:\s*.+",
        transforms[0])
    if not matched or tuple(map(int, matched.groups())) != (*tiles, 0):
        raise ValueError("actual Transform tile sizes do not consume the requested M/N and untiled K")
    constants, loops = {}, []
    for index, line in enumerate(code_lines(scheduled)):
        constant = re.fullmatch(rf"\s*({ssa})\s*=\s*arith\.constant\s+(-?[0-9]+)\s*:\s*index\s*", line)
        if constant:
            if constant[1] in constants:
                raise ValueError("duplicate index-constant SSA definition in canonical schedule")
            constants[constant[1]] = (int(constant[2]), index)
        if re.match(rf"^\s*(?:{assignment})?scf\.for\b", line):
            loop = re.fullmatch(rf"(\s*)scf\.for\s+{ssa}\s*=\s*{ssa}\s+to\s+{ssa}\s+step\s+({ssa})\s*\{{\s*", line)
            if not loop:
                raise ValueError("outer output loop differs from canonical scheduled syntax")
            loops.append((loop[2], index, len(loop[1])))
    if len(loops) != 2 or loops[1][2] <= loops[0][2]:
        raise ValueError("expected exactly two ordered nested output-domain SCF loops")
    resolved = []
    for step, index, _ in loops:
        if step not in constants or constants[step][1] >= index:
            raise ValueError("output loop step does not resolve to a preceding index constant")
        resolved.append(constants[step][0])
    if tuple(resolved) != tuple(tiles):
        raise ValueError("actual outer SCF loop steps do not consume requested M/N in order")
    return {"transform_tile_sizes": [*tiles, 0], "outer_loop_steps": resolved}


def negative_dependency_refusal(stderr, bundle):
    # Parent-directory metadata is checked before the new file's former
    # negative status; require the exact captured bundle rather than a crash.
    expected = "captured path identity or metadata changed: " + str(bundle)
    lines = {line.removeprefix("mdslc-region: ") for line in stderr.splitlines()}
    if not stderr.startswith("mdslc-region:") or expected not in lines:
        raise ValueError("negative bundle dependency was not refused at its captured path change")


def private_wrapper_offset(data, machine):
    """Exact native ELF64LE ET_DYN local function in an executable segment.

    Used only by the noninstalled ASan negative control. No public dynamic
    symbol is added and no parsed artifact obtains production authority.
    """
    def unpack(format_string, offset):
        size = struct.calcsize(format_string)
        if offset < 0 or offset + size > len(data):
            raise ValueError("truncated ELF metadata")
        return struct.unpack_from(format_string, data, offset)

    header = unpack("<16sHHIQQQIHHHHHH", 0)
    if (header[0][:7] != b"\x7fELF\x02\x01\x01" or header[1] != 3 or
            header[2] != machine or header[3] != 1 or header[8] != 64 or
            header[9] != 56 or header[11] != 64 or not header[10] or not header[12]):
        raise ValueError("requested DSO is not native ELF64LE ET_DYN")
    programs = [unpack("<IIQQQQQQ", header[5] + i * 56) for i in range(header[10])]
    sections = [unpack("<IIQQQQIIQQ", header[6] + i * 64) for i in range(header[12])]
    matches = []
    wanted = b"_mlir_ciface___matcore_strict_gemm_f32_v1"
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB; never substitute .dynsym.
            continue
        if section[9] != 24 or section[5] % 24 or section[6] >= len(sections):
            raise ValueError("invalid private symbol table")
        names = sections[section[6]]
        if names[1] != 3 or names[4] + names[5] > len(data):
            raise ValueError("invalid private symbol names")
        string_table = data[names[4]:names[4] + names[5]]
        for index in range(section[5] // 24):
            symbol = unpack("<IBBHQQ", section[4] + index * 24)
            if symbol[0] >= len(string_table):
                raise ValueError("invalid ELF symbol name offset")
            end = string_table.find(b"\0", symbol[0])
            if end == -1:
                raise ValueError("unterminated ELF symbol name")
            if string_table[symbol[0]:end] != wanted:
                continue
            if symbol[1] != 2 or not symbol[3] or symbol[3] >= len(sections) or not symbol[5]:
                raise ValueError("requested wrapper is not a defined LOCAL FUNC")
            if not any(segment[0] == 1 and segment[1] & 5 == 5 and
                       segment[3] <= symbol[4] and
                       symbol[4] + symbol[5] <= segment[3] + segment[6]
                       for segment in programs):
                raise ValueError("requested wrapper is not inside a readable executable PT_LOAD")
            matches.append(symbol[4])
    if len(matches) != 1:
        raise ValueError("requested DSO has no unique local wrapper")
    return matches[0]


def semantic_summary(text, sanitized):
    match = re.fullmatch(r"Source pattern semantic oracle: (\d+) cases; (\d+) checks; "
                         r"(\d+) failures; host_sanitized=([01])\n", text)
    if not match or int(match[1]) != 68 or int(match[2]) < 10000 or int(match[3]) != 0:
        raise ValueError("source semantic oracle did not execute its complete independent matrix")
    if (match[4] == "1") != sanitized:
        raise ValueError("original source sanitizer profile differs from installation")
    return {"cases": int(match[1]), "checks": int(match[2]), "failures": int(match[3])}


def validate_bundle(binary, tiles, region, optimization, sanitized, invoke):
    bundle = Path(str(binary) + ".mdslc-private")
    if not bundle.is_dir() or bundle.is_symlink() or stat.S_IMODE(bundle.stat().st_mode) != 0o700:
        raise ValueError("source instance has no real private 0700 bundle")
    info = manifest((bundle / "source-pattern.manifest").read_text())
    expected = {
        "pattern_parameters": f"M:{tiles[0]},N:{tiles[1]},K:untiled",
        "region_name_hex": region.encode().hex(), "candidate": "generated-strict",
        "optimization": optimization,
        "address_sanitizer": "function_attributes_and_object" if sanitized else "off",
    }
    for key, value in expected.items():
        if info[key] != value:
            raise ValueError(f"source instance did not bind {key}: {info[key]!r}")
    target = {"x86_64": "x86_64-pc-linux-gnu", "aarch64": "aarch64-unknown-linux-gnu"}[platform.machine()]
    if info["target"] != target or info["toolchain"] != "21.1.8":
        raise ValueError("source instance did not bind the exact native product tuple")
    for key, name in HASH_FILES.items():
        if digest(bundle / name) != info[key]:
            raise ValueError("retained issued bytes differ from binding: " + name)
    output_tile_consumption((bundle / "gemm.ll.transform.mlir").read_text(),
                            (bundle / "gemm.ll.scheduled.mlir").read_text(), tiles)
    dso = bundle / info["candidate_soname"]
    if digest(dso) != info["dso_sha256"]:
        raise ValueError("source instance DSO differs from bound bytes")
    binary_needed = needed(invoke(["readelf", "-d", binary]).stdout)
    if info["candidate_soname"] not in binary_needed or "libmatcore_runtime.so.0" not in binary_needed:
        raise ValueError("source executable does not directly depend on requested DSO and Runtime")
    if "libmatcore_closed_candidates_isolated_v1.so" in binary_needed:
        raise ValueError("source executable silently uses the stock build-issued candidate DSO")
    dynamic = invoke(["readelf", "-d", dso]).stdout
    if "libmatcore_runtime.so.0" not in needed(dynamic) or "(SYMBOLIC)" not in dynamic:
        raise ValueError("instance DSO did not retain canonical Runtime ownership/local binding")
    if any("LLVM" in item or "MLIR" in item or "openblas" in item.lower() for item in needed(dynamic)):
        raise ValueError("instance DSO leaks compiler/provider ownership")
    definitions = invoke(["nm", "--defined-only", "--format=posix", dso]).stdout.splitlines()
    for symbol in ("__matcore_strict_gemm_f32_v1", "_mlir_ciface___matcore_strict_gemm_f32_v1"):
        matching = [line.split() for line in definitions if line.split() and line.split()[0] == symbol]
        if len(matching) != 1 or matching[0][1] != "t":
            raise ValueError("requested issued leaf/wrapper is not uniquely local: " + symbol)
    exports = invoke(["nm", "--dynamic", "--defined-only", "--format=posix", dso]).stdout
    if "__matcore_strict_gemm" in exports or "_mlir_ciface_" in exports:
        raise ValueError("source private leaf can be dynamically substituted")
    llvm = (bundle / "gemm.ll").read_text()
    if "fmul float" not in llvm or "fadd float" not in llvm or "llvm.fma" in llvm or " fast " in llvm:
        raise ValueError("source leaf does not retain separate strict f32 arithmetic")
    if ("sanitize_address" in llvm) != sanitized:
        raise ValueError("requested leaf instrumentation intent differs from installation")
    return info, {str(path.relative_to(bundle)): digest(path)
                  for path in sorted(bundle.iterdir()) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sanitized", action="store_true")
    parser.add_argument("--cxx", default="/usr/bin/clang++-21")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--case", choices=("connected", "refusals"), default="connected")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        parser.error("source pattern requires native Linux x86-64/AArch64")
    driver = args.driver.resolve(strict=True)
    driver_hash = digest(driver)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        root = args.output.resolve()
    else:
        root = Path(tempfile.mkdtemp(prefix="mdslc-source-pattern-"))
    source = root / "inputs"
    source.mkdir()
    for name in ("connected.mdsl", "same_value.mdsl", "rhs.mdsl", "forwarding.mdsl", "math_helper.h"):
        shutil.copyfile(Path(__file__).resolve().parent / name, source / name)
    environment = dict(os.environ, LC_ALL="C", PYTHONDONTWRITEBYTECODE="1")
    environment["ASAN_OPTIONS"] = "halt_on_error=1:detect_leaks=0:exitcode=1:symbolize=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    environment["DEBUGINFOD_URLS"] = ""
    for name in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    report = {"driver": str(driver), "driver_sha256": driver_hash,
              "sanitized_requested": args.sanitized, "case": args.case,
              "commands": [], "status": "running", "instances": []}
    evidence = root / "evidence.json"
    pinned = {str(driver): driver_hash}

    def save():
        evidence.write_text(json.dumps(report, indent=2) + "\n")

    def invoke(argv, expected=0, stdout=None):
        for name, sha in pinned.items():
            if digest(name) != sha:
                raise ValueError("frozen test artifact changed: " + name)
        result = subprocess.run([str(item) for item in argv], cwd=root, env=environment,
                                text=True, capture_output=True, timeout=180)
        report["commands"].append({"argv": [str(item) for item in argv],
                                   "exit": result.returncode, "stdout": result.stdout,
                                   "stderr": result.stderr})
        save()
        if result.returncode != expected or (stdout is not None and result.stdout != stdout):
            raise ValueError("command/independent oracle failed: " + repr(argv))
        return result

    def compile(binary, fixture="connected.mdsl", region="source_pattern_pipeline", tiles=(3, 5),
                optimization="none", extra=()):
        return invoke([driver, source / fixture, "--region", region,
                       "--candidate", "generated-strict", "--gemm-output-tiles",
                       f"{tiles[0]},{tiles[1]}", "--optimization", optimization,
                       "-o", binary, *extra])

    def reject(label, arguments):
        binary = root / ("refused-" + label)
        result = subprocess.run([str(driver), *map(str, arguments), "-o", str(binary)],
                                cwd=root, env=environment, text=True, capture_output=True, timeout=180)
        report["commands"].append({"argv": [str(driver), *map(str, arguments), "-o", str(binary)],
                                   "exit": result.returncode, "stdout": result.stdout,
                                   "stderr": result.stderr})
        save()
        if result.returncode != 1 or not result.stderr.startswith("mdslc-region:") or binary.exists() or Path(str(binary) + ".mdslc-private").exists():
            raise ValueError("invalid source request did not fail cleanly without publication: " + label)
        return result

    try:
        banner = invoke([driver, "--version"]).stdout
        if ("ASan+UBSan" in banner) != args.sanitized:
            raise ValueError("driver installation instrumentation differs from test selection")
        if args.case == "refusals":
            base = [source / "connected.mdsl", "--region", "source_pattern_pipeline",
                    "--candidate", "generated-strict"]
            bad_tiles = ("0,5", "3,0", "65,5", "3,65", "-1,5", "3,-1", "03,5", "3,05",
                         "+3,5", "3,+5", "3", "3,5,1", ",5", "3,", "3,5x",
                         "18446744073709551616,5", "3,18446744073709551616")
            for index, tiles in enumerate(bad_tiles):
                reject(f"tiles-{index}", [*base, "--gemm-output-tiles", tiles])
            reject("missing-tiles", [*base, "--gemm-output-tiles"])
            reject("duplicate-tiles", [*base, "--gemm-output-tiles", "3,5", "--gemm-output-tiles", "5,3"])
            for candidate in ("automatic", "native-strict", "generated-reassociate", "generated-strict-avx",
                              "generated-strict-avx2", "generated-strict-avx512f", "generated-nvvm",
                              "generated-rocdl", "existing-native", "openblas"):
                reject("candidate-" + candidate, [source / "connected.mdsl", "--region", "source_pattern_pipeline",
                       "--candidate", candidate, "--gemm-output-tiles", "3,5"])
            reject("implicit-candidate", [source / "connected.mdsl", "--region", "source_pattern_pipeline",
                                          "--gemm-output-tiles", "3,5"])
            reject("compile-only", [*base, "--gemm-output-tiles", "3,5", "-c"])
            reject("fused", [*base, "--gemm-output-tiles", "3,5", "--optimization", "strict-fused-pair"])
            reject("program", ["--program", "--host", source / "connected.mdsl", "--region",
                               source / "connected.mdsl", "source_pattern_pipeline", "--candidate",
                               "generated-strict", "--gemm-output-tiles", "3,5"])
            forged_ir = root / "forged.mlir"
            forged_ir.write_text('module attributes {mdsl.execution_authority} {}\n')
            for option in ("--llvm-input", "--mlir-input", "--gemm-ir", "--gemm-manifest"):
                reject(option[2:], [*base, "--gemm-output-tiles", "3,5", option, forged_ir])
            for kind in ("file", "directory", "symlink"):
                binary = root / ("existing-bundle-" + kind)
                bundle = Path(str(binary) + ".mdslc-private")
                if kind == "file":
                    bundle.write_text("must survive refusal\n")
                elif kind == "directory":
                    bundle.mkdir()
                    (bundle / "sentinel").write_text("must survive refusal\n")
                else:
                    target = root / "symlink-target"
                    target.mkdir()
                    (target / "sentinel").write_text("must survive refusal\n")
                    bundle.symlink_to(target, target_is_directory=True)
                result = subprocess.run([str(driver), *map(str, base), "--gemm-output-tiles", "3,5",
                                         "-o", str(binary)], cwd=root, env=environment, text=True,
                                        capture_output=True, timeout=180)
                report["commands"].append({"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
                if result.returncode != 1 or binary.exists() or not bundle.exists():
                    raise ValueError("existing bundle was accepted/deleted: " + kind)
                sentinel = bundle if kind == "file" else bundle / "sentinel"
                if sentinel.read_text() != "must survive refusal\n":
                    raise ValueError("existing bundle content changed: " + kind)
            negative_source = source / "negative_dependency.mdsl"
            generated_path = str(root / "refused-negative-dependency") + ".mdslc-private/gemm.ll"
            negative_source.write_text('#if __has_include("' + generated_path + '")\n'
                                       '#error private artifact changed an admitted negative source dependency\n'
                                       '#endif\n' + (source / "connected.mdsl").read_text())
            dependency = reject("negative-dependency", [negative_source, "--region", "source_pattern_pipeline",
                                "--candidate", "generated-strict", "--gemm-output-tiles", "3,5"])
            negative_dependency_refusal(dependency.stderr, Path(generated_path).parent)
        else:
            probe = root / "raw-asan-probe"
            if args.sanitized:
                if not args.runtime:
                    raise ValueError("actual source-leaf ASan probe requires the canonical Runtime path")
                runtime = args.runtime.resolve(strict=True)
                pinned[str(runtime)] = digest(runtime)
                invoke([args.cxx, Path(__file__).resolve().parent / "raw_asan_probe.cpp",
                        "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
                        "-Xlinker", "--no-as-needed", runtime, "-ldl", "-pthread",
                        "-Xlinker", "-rpath", "-Xlinker", runtime.parent, "-o", probe])
            records = []
            for tiles in ((3, 5), (5, 3), (1, 1), (64, 64)):
                binary = root / f"connected-{tiles[0]}-{tiles[1]}"
                compile(binary, tiles=tiles)
                info, hashes = validate_bundle(binary, tiles, "source_pattern_pipeline", "none", args.sanitized, invoke)
                result = invoke([binary])
                semantic_summary(result.stdout, args.sanitized)
                if result.stderr:
                    raise ValueError("positive source execution emitted diagnostics")
                pinned[str(binary)] = digest(binary)
                bundle = Path(str(binary) + ".mdslc-private")
                pinned.update({str(bundle / name): sha for name, sha in hashes.items()})
                records.append(info)
                report["instances"].append({"tiles": tiles, "manifest": info, "bundle_sha256": hashes,
                                            "semantic_summary": result.stdout.strip()})
                negative = invoke([binary, "--corrupt-output"], expected=1)
                if "FAIL source pattern:" not in negative.stderr:
                    raise ValueError("source corruption control failed for unrelated reasons")
                if args.sanitized:
                    dso = bundle / info["candidate_soname"]
                    wrapper = private_wrapper_offset(dso.read_bytes(),
                        62 if platform.machine() == "x86_64" else 183)
                    negative = invoke([probe, dso, f"{wrapper:x}"], expected=1)
                    if ("AddressSanitizer: heap-buffer-overflow" not in negative.stderr or
                            not re.search(r"#0[^\n]*(_mlir_ciface_)?__matcore_strict_gemm_f32_v1", negative.stderr) or
                            info["candidate_soname"] not in negative.stderr):
                        raise ValueError("ASan negative did not originate in the actual requested source DSO")
            for key in ("semantic_sha256", "structured_sha256", "bufferized_sha256"):
                if records[0][key] != records[1][key]:
                    raise ValueError("HOW parameters contaminated unchanged canonical WHAT witness")
            for key in ("transform_sha256", "scheduled_sha256", "llvm_sha256", "object_sha256",
                        "dso_sha256", "binding_sha256", "candidate_soname"):
                if records[0][key] == records[1][key]:
                    raise ValueError("two explicit requests did not realize distinct bound private artifacts: " + key)
            for tiles in ((3, 5), (5, 3)):
                binary = root / f"same-value-{tiles[0]}-{tiles[1]}"
                compile(binary, "same_value.mdsl", "source_pattern_same_value", tiles)
                validate_bundle(binary, tiles, "source_pattern_same_value", "none", args.sanitized, invoke)
                invoke([binary], stdout="Source pattern same Value: 5 cases; 480 checks; 0 failures\n")
                rhs_binary = root / f"rhs-{tiles[0]}-{tiles[1]}"
                compile(rhs_binary, "rhs.mdsl", "source_pattern_rhs", tiles)
                validate_bundle(rhs_binary, tiles, "source_pattern_rhs", "none", args.sanitized, invoke)
                invoke([rhs_binary], stdout="Source pattern RHS carry: 8 cases; 256 checks; 0 failures\n")
            forwarded = root / "forwarded-3-5"
            compile(forwarded, optimization="publication-read-forwarding")
            validate_bundle(forwarded, (3, 5), "source_pattern_pipeline", "publication-read-forwarding", args.sanitized, invoke)
            semantic_summary(invoke([forwarded]).stdout, args.sanitized)
            # Distinct MAY-alias D and C cannot establish a forwarding fact.
            # Keep their same-source none/forwarding semantic parity without
            # demanding an ineligible transformation. This separate eligible
            # fixture must genuinely connect readForwarded only when requested.
            forwarding_outputs = []
            for optimization in ("none", "publication-read-forwarding"):
                eligible = root / ("eligible-forwarding-" + optimization)
                compile(eligible, "forwarding.mdsl", "source_pattern_forwarding", (3, 5), optimization)
                validate_bundle(eligible, (3, 5), "source_pattern_forwarding", optimization, args.sanitized, invoke)
                output = invoke([eligible]).stdout
                match = re.fullmatch(r"Source pattern eligible forwarding: 8 cases; (\d+) checks; 0 failures\n", output)
                if not match or int(match[1]) < 1000:
                    raise ValueError("eligible forwarding did not execute its independent source/guard oracle")
                symbols = invoke(["nm", "--undefined-only", "--demangle", eligible]).stdout
                if ("closed_host_v1::SessionAbiV2::readForwarded" in symbols) != (optimization == "publication-read-forwarding"):
                    raise ValueError("eligible source forwarding did not match its checked call discriminator")
                forwarding_outputs.append(output)
            if forwarding_outputs[0] != forwarding_outputs[1]:
                raise ValueError("eligible same-source forwarding altered source outcomes/frontiers")
            for candidate in ("automatic", "generated-strict"):
                stock = root / ("unrequested-" + candidate)
                invoke([driver, source / "connected.mdsl", "--region", "source_pattern_pipeline",
                        "--candidate", candidate, "-o", stock])
                semantic_summary(invoke([stock]).stdout, args.sanitized)
                if Path(str(stock) + ".mdslc-private").exists() or "libmatcore_closed_candidates_isolated_v1.so" not in needed(invoke(["readelf", "-d", stock]).stdout):
                    raise ValueError("unrequested/default path acquired source-time issuance or lost stock ownership")
            first = root / "connected-3-5"
            unchanged_binary = digest(first)
            unchanged_bundle = {str(path): digest(path) for path in Path(str(first) + ".mdslc-private").iterdir() if path.is_file()}
            result = subprocess.run([str(driver), str(source / "connected.mdsl"), "--region", "source_pattern_pipeline",
                                     "--candidate", "generated-strict", "--gemm-output-tiles", "5,3", "-o", str(first)],
                                    cwd=root, env=environment, text=True, capture_output=True, timeout=180)
            if result.returncode != 1 or digest(first) != unchanged_binary or any(digest(path) != sha for path, sha in unchanged_bundle.items()):
                raise ValueError("no-clobber request replaced an existing executable/private instance")
            if args.sanitized:
                negative = invoke([first, "--host-oob"], expected=1)
                if "AddressSanitizer: heap-buffer-overflow" not in negative.stderr:
                    raise ValueError("original host sanitizer negative control did not execute")
        report["status"] = "pass"
        save()
        print(f"PASS independent source pattern {args.case}; evidence={evidence}")
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status="fail", failure=str(error))
        save()
        raise SystemExit(str(error) + "; evidence=" + str(evidence))


if __name__ == "__main__":
    main()
