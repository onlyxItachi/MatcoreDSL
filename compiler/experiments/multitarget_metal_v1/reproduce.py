#!/usr/bin/env python3
"""Small compile-only upstream Metal experiment, never a product issuer.

All artifacts go to --output; no source build, installation or execution occurs.
The static fixture is an explicit research instantiation, not source authority.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir-bin", type=Path, required=True)
    ap.add_argument("--spirv-bin", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dynamic-outlined", type=Path)
    ap.add_argument("--check-golden", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commands = []
    (args.output / "manifest.json").write_text('{"status":"incomplete research run"}\n')

    def run(executable, *options, expect=0):
        argv = [str(executable), *map(str, options)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        commands.append(dict(argv=argv, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode != expect:
            raise RuntimeError(f"unexpected exit {result.returncode}: {argv}\n{result.stderr}")
        return result.stdout

    opt = args.mlir_bin / "mlir-opt"
    translate = args.mlir_bin / "mlir-translate"
    assert "21.1.8" in run(opt, "--version"), "This reproduction is pinned to MLIR 21.1.8"
    run(args.spirv_bin / "spirv-cross", "--revision")
    run(args.spirv_bin / "spirv-val", "--version")
    outlined = args.output / "static-outlined.mlir"
    converted = args.output / "static-spirv.mlir"
    run(opt, Path(__file__).with_name("static_specimen.mlir"),
        "--convert-linalg-to-parallel-loops", "--gpu-map-parallel-loops",
        "--convert-parallel-loops-to-gpu", "--canonicalize",
        "--gpu-launch-sink-index-computations", "--gpu-kernel-outlining",
        "--canonicalize", "-o", outlined)
    pipeline = ("--pass-pipeline=builtin.module(gpu.module("
                "test-spirv-entry-point-abi{workgroup-size=1,1,1}),"
                "convert-gpu-to-spirv,spirv.module(spirv-lower-abi-attrs,spirv-update-vce))")
    run(opt, outlined, pipeline, "-o", converted)

    # Extract exactly the two known printer-produced SPIR-V modules. This is
    # research artifact selection, NOT a parser/admission path for user input.
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
    assert len(modules) == 2 and not current
    for name, body in zip(("fill", "gemm"), modules):
        mlir = args.output / f"{name}.mlir"
        binary = args.output / f"{name}.spv"
        mlir.write_text(body)
        run(translate, "--no-implicit-module", "--serialize-spirv", mlir, "-o", binary)
        run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.1", binary)
        run(args.spirv_bin / "spirv-dis", binary, "-o", args.output / f"{name}.spvasm")
        run(args.spirv_bin / "spirv-cross", binary, "--msl", "--msl-version", "20100",
            "--output", args.output / f"{name}.metal")

    assembly = (args.output / "gemm.spvasm").read_text()
    assert "OpFMul" in assembly and "OpFAdd" in assembly
    missing = [fact for fact in ("NoContraction", "DenormPreserve", "RoundingModeRTE")
               if fact not in assembly]
    assert len(missing) == 3, "Reinspect newer upstream strictness behavior"

    # Controlled negative: add required SPIR-V modes to the SAME compiled
    # specimen, then see whether the MSL translation implements/rejects them.
    lines = assembly.splitlines()
    cap_end = max(i for i, line in enumerate(lines) if "OpCapability " in line) + 1
    lines[cap_end:cap_end] = [f"OpCapability {x}" for x in
                            ("DenormPreserve", "RoundingModeRTE", "SignedZeroInfNanPreserve")]
    ext_end = max(i for i, line in enumerate(lines) if "OpExtension " in line) + 1
    lines.insert(ext_end, 'OpExtension "SPV_KHR_float_controls"')
    mode_index = next(i for i, line in enumerate(lines) if "OpExecutionMode " in line)
    entry = lines[mode_index].split()[1]
    lines[mode_index + 1:mode_index + 1] = [
        f"OpExecutionMode {entry} {x} 32" for x in
        ("DenormPreserve", "RoundingModeRTE", "SignedZeroInfNanPreserve")]
    decorate_index = next(i for i, line in enumerate(lines) if "OpDecorate " in line)
    fp_ids = [line.split()[0] for line in lines if "OpFMul " in line or "OpFAdd " in line]
    lines[decorate_index:decorate_index] = [f"OpDecorate {x} NoContraction" for x in fp_ids]
    strict_asm = args.output / "gemm-required-fp.spvasm"
    strict_binary = args.output / "gemm-required-fp.spv"
    strict_asm.write_text("\n".join(lines) + "\n")
    run(args.spirv_bin / "spirv-as", "--target-env", "spv1.0", strict_asm, "-o", strict_binary)
    run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.1", strict_binary)
    run(args.spirv_bin / "spirv-cross", strict_binary, "--msl", "--msl-version", "20100",
        "--output", args.output / "gemm-required-fp.metal")
    nocontract = args.output / "gemm-nocontract-only.spvasm"
    nocontract_binary = args.output / "gemm-nocontract-only.spv"
    nocontract.write_text("\n".join(line for line in lines if not any(
        token in line for token in ("DenormPreserve", "RoundingModeRTE",
                                    "SignedZeroInfNanPreserve", "SPV_KHR_float_controls"))) + "\n")
    run(args.spirv_bin / "spirv-as", "--target-env", "spv1.0", nocontract, "-o", nocontract_binary)
    run(args.spirv_bin / "spirv-val", "--target-env", "vulkan1.1", nocontract_binary)
    run(args.spirv_bin / "spirv-cross", nocontract_binary, "--msl", "--msl-version", "20100",
        "--output", args.output / "gemm-nocontract-only.metal")
    assert ((args.output / "gemm-required-fp.metal").read_bytes() ==
            (args.output / "gemm-nocontract-only.metal").read_bytes()), (
                "Reinspect SPIRV-Cross: required FP modes now affect its output")
    # Repository goldens normalize only EOF blank lines; no shader token changes.
    for path in args.output.glob("*.metal"):
        path.write_text(path.read_text().rstrip("\n") + "\n")
    if args.check_golden:
        golden = Path(__file__).with_name("golden")
        for name in ("fill.metal", "gemm.metal", "gemm-required-fp.metal"):
            assert (args.output / name).read_bytes() == (golden / name).read_bytes(), name
    if args.dynamic_outlined:
        dynamic = args.dynamic_outlined.read_text()
        assert dynamic.startswith("module attributes {gpu.container_module}")
        dynamic = dynamic.replace("gpu.container_module}",
            "gpu.container_module, spirv.target_env = #spirv.target_env<"
            "#spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, "
            "#spirv.resource_limits<>>}", 1)
        dynamic_path = args.output / "dynamic-spirv-input.mlir"
        dynamic_path.write_text(dynamic)
        run(opt, dynamic_path, pipeline, "-o", args.output / "dynamic-spirv.mlir", expect=1)
        assert "failed to legalize operation 'memref.store'" in commands[-1]["stderr"]
        assert "memref<?x?xf32>" in commands[-1]["stderr"]

    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.output.iterdir() if p.is_file() and p.name != "manifest.json"}
    (args.output / "manifest.json").write_text(json.dumps(dict(
        classification="compile-only research; not product execution authority",
        tool_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (opt, translate, *(args.spirv_bin / name for name in
                       ("spirv-cross", "spirv-val", "spirv-as", "spirv-dis")))},
        specimen_sha256=hashlib.sha256(Path(__file__).with_name("static_specimen.mlir").read_bytes()).hexdigest(),
        msl_normalization="EOF LF blank lines normalized to one LF; no token changes",
        missing_default_float_controls=missing,
        required_fp_modes_do_not_change_msl=True,
        commands=commands, sha256=hashes), indent=2) + "\n")
    print("PASS: static Linalg -> GPU -> valid SPIR-V -> MSL; strict controls require review")


if __name__ == "__main__":
    main()
