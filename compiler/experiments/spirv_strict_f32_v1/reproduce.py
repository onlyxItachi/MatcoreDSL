#!/usr/bin/env python3
"""Bounded research derivation, not imported-IR or MDSLC execution authority.

Only repository-owned static Linalg supplies arithmetic. The documented
assembly adjustment adds strict controls, never shader operations. No driver,
package, toolchain install, device discovery, or execution occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


MODES = ("DenormPreserve", "RoundingModeRTE", "SignedZeroInfNanPreserve")
FORBIDDEN = ("RelaxedPrecision", "FPRoundingMode", "FPFastMathMode",
             "DenormFlushToZero", "RoundingModeRTZ", " Fma ", "OpFunctionCall")


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def arithmetic_ids(assembly):
    return re.findall(r"^\s*(%\S+) = Op(?:FMul|FAdd)\b", assembly, re.M)


def verify_strict(assembly, kind):
    require(not any(x in assembly for x in FORBIDDEN), "forbidden floating-point permission")
    entries = re.findall(r"OpEntryPoint GLCompute (%\S+) ", assembly)
    require(len(entries) == 1, "one known compute entry required")
    entry = entries[0]
    for mode in MODES:
        require(len(re.findall(r"OpCapability " + mode + r"\b", assembly)) == 1,
                "missing/duplicate capability: " + mode)
        require(len(re.findall(r"OpExecutionMode " + re.escape(entry) + " " + mode + r" 32\b", assembly)) == 1,
                "missing/duplicate f32 execution mode: " + mode)
    require('OpExtension "SPV_KHR_float_controls"' in assembly, "missing float-controls extension")
    require("OpExecutionMode " + entry + " LocalSize 1 1 1" in assembly, "unexpected workgroup geometry")
    ids = arithmetic_ids(assembly)
    require(assembly.count(" OpFMul ") == (1 if kind == "gemm" else 0), "unexpected multiply graph")
    require(assembly.count(" OpFAdd ") == (1 if kind == "gemm" else 0), "unexpected addition graph")
    decorated = re.findall(r"OpDecorate (%\S+) NoContraction\b", assembly)
    require(len(decorated) == len(ids) and set(decorated) == set(ids), "missing/extra NoContraction")


def controlled_assembly(original, kind):
    require(not any(x in original for x in (*MODES, "NoContraction", *FORBIDDEN)),
            "reinspect changed default lowering")
    lines = original.splitlines()
    additions = []
    cap_end = max(i for i, s in enumerate(lines) if "OpCapability " in s) + 1
    caps = ["OpCapability " + x for x in MODES]
    lines[cap_end:cap_end] = caps
    additions.extend(caps)
    ext_end = max(i for i, s in enumerate(lines) if "OpExtension " in s) + 1
    extension = 'OpExtension "SPV_KHR_float_controls"'
    lines.insert(ext_end, extension)
    additions.append(extension)
    modes = [i for i, s in enumerate(lines) if "OpExecutionMode " in s]
    require(len(modes) == 1, "unexpected default execution modes")
    entry = lines[modes[0]].split()[1]
    controls = ["OpExecutionMode " + entry + " " + mode + " 32" for mode in MODES]
    lines[modes[0] + 1:modes[0] + 1] = controls
    additions.extend(controls)
    index = next(i for i, s in enumerate(lines) if "OpDecorate " in s)
    no_contract = ["OpDecorate " + value + " NoContraction" for value in arithmetic_ids(original)]
    lines[index:index] = no_contract
    additions.extend(no_contract)
    # This adjustment has NO arithmetic, indexing, ABI, or control-flow edits.
    require([s for s in lines if s not in additions] == original.splitlines(),
            "adjustment changed generated arithmetic or interface")
    result = "\n".join(lines) + "\n"
    verify_strict(result, kind)
    negative_controls = 0
    for mode in MODES:
        modified = "\n".join(s for s in lines if not ("OpExecutionMode " in s and " " + mode + " 32" in s))
        try:
            verify_strict(modified, kind)
        except RuntimeError:
            negative_controls += 1
        else:
            raise RuntimeError("missing-mode negative was accepted")
    for forbidden in ("RelaxedPrecision", "FPRoundingMode", "FPFastMathMode"):
        try:
            verify_strict(result + "OpDecorate %negative " + forbidden + "\n", kind)
        except RuntimeError:
            negative_controls += 1
        else:
            raise RuntimeError("forbidden-permission negative was accepted")
    if no_contract:
        try:
            verify_strict(result.replace(no_contract[0] + "\n", ""), kind)
        except RuntimeError:
            negative_controls += 1
        else:
            raise RuntimeError("missing-contraction-control negative was accepted")
    return result, negative_controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir-bin", type=Path, required=True)
    ap.add_argument("--spirv-bin", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = []
    specimen = Path(__file__).with_name("static_specimen.mlir")
    specimen_digest = hashlib.sha256(specimen.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text('{"status":"incomplete research derivation"}\n')

    def run(executable, *options):
        argv = [str(executable), *map(str, options)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        commands.append(dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        require(result.returncode == 0, "tool failed: " + repr(argv) + "\n" + result.stderr)
        return result.stdout

    opt, translate = (args.mlir_bin / n for n in ("mlir-opt", "mlir-translate"))
    require("21.1.8" in run(opt, "--version"), "pinned MLIR 21.1.8 required")
    run(args.spirv_bin / "spirv-val", "--version")
    outlined, converted = (args.output / n for n in ("outlined.mlir", "converted.mlir"))
    run(opt, specimen, "--convert-linalg-to-parallel-loops", "--gpu-map-parallel-loops",
        "--convert-parallel-loops-to-gpu", "--canonicalize", "--gpu-launch-sink-index-computations",
        "--gpu-kernel-outlining", "--canonicalize", "-o", outlined)
    pipeline = ("--pass-pipeline=builtin.module(gpu.module("
                "test-spirv-entry-point-abi{workgroup-size=1,1,1}),convert-gpu-to-spirv,"
                "spirv.module(spirv-lower-abi-attrs,spirv-update-vce))")
    run(opt, outlined, pipeline, "-o", converted)
    modules, current, depth = [], [], 0
    for line in converted.read_text().splitlines():
        if not current and line.startswith("  spirv.module @"):
            current = [line]
            depth = line.count("{") - line.count("}")
        elif current:
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                modules.append("\n".join(current) + "\n")
                current = []
    require(len(modules) == 2 and not current, "exactly two printer-produced modules required")
    negatives = 0
    for kind, module in zip(("fill", "gemm"), modules):
        mlir, binary, assembly = (args.output / (kind + s) for s in (".mlir", "-default.spv", "-default.spvasm"))
        mlir.write_text(module)
        run(translate, "--no-implicit-module", "--serialize-spirv", mlir, "-o", binary)
        run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", binary)
        run(args.spirv_bin / "spirv-dis", binary, "-o", assembly)
        # MLIR serializes the default minimal module as SPIR-V1.0. Normalize
        # only its header version so host negatives test missing float controls,
        # not merely the research host's SPIR-V1.3 version pin.
        normalized_default = args.output / (kind + "-default-v13.spv")
        run(args.spirv_bin / "spirv-as", "--target-env", "spv1.3", assembly, "-o", normalized_default)
        run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", normalized_default)
        strict, count = controlled_assembly(assembly.read_text(), kind)
        negatives += count
        strict_assembly, strict_binary = (args.output / (kind + s) for s in ("-strict.spvasm", "-strict.spv"))
        strict_assembly.write_text(strict)
        run(args.spirv_bin / "spirv-as", "--target-env", "spv1.3", strict_assembly, "-o", strict_binary)
        run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", strict_binary)
        final_assembly = args.output / (kind + "-strict-final.spvasm")
        run(args.spirv_bin / "spirv-dis", strict_binary, "-o", final_assembly)
        verify_strict(final_assembly.read_text(), kind)
        if kind == "gemm":
            no_contract_assembly = args.output / "gemm-missing-nocontraction.spvasm"
            no_contract_assembly.write_text("\n".join(s for s in strict.splitlines() if " NoContraction" not in s) + "\n")
            no_contract_binary = args.output / "gemm-missing-nocontraction.spv"
            run(args.spirv_bin / "spirv-as", "--target-env", "spv1.3", no_contract_assembly, "-o", no_contract_binary)
            run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.2", no_contract_binary)
    require(hashlib.sha256(specimen.read_bytes()).hexdigest() == specimen_digest, "specimen changed during derivation")
    tools = [opt, translate, *(args.spirv_bin / n for n in ("spirv-as", "spirv-val", "spirv-dis"))]
    manifest = dict(classification="bounded static research; no MDSLC source/runtime/target authority",
                    status="COMPILED_VALIDATED_ONLY", strict_contract_qualified=False,
                    dimensions=dict(M=2, K=3, N=4), geometry="2x4x1 workgroups, 1x1x1 local size",
                    numerical_adjustment="only generated result decorations/capabilities/execution modes; exact body preserved",
                    required_float32_modes=MODES, required_arithmetic_decoration="NoContraction",
                    negative_controls=negatives, specimen_sha256=specimen_digest,
                    tool_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tools},
                    artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                                     if p.is_file() and p.name != "manifest.json"}, commands=commands)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS bounded static strict SPIR-V derivation;", negatives, "negative controls; NO execution/target authority")


if __name__ == "__main__":
    main()
