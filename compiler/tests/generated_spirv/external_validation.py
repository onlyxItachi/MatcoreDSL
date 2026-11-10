#!/usr/bin/env python3
"""Independent upstream validation and ABI inspection; never dispatch/import.

The issuer constructs from dimensions only. These reports do not confer source,
runtime, shader-import, performance, or physical-target execution authority.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issuer", type=Path, required=True)
    ap.add_argument("--validator", type=Path, required=True)
    ap.add_argument("--disassembler", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Preserve each invocation's evidence instead of overwriting older receipts.
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    commands = []
    report = {"classification": "COMPILE_AND_VALIDATE_ONLY",
              "source_runtime_authority": False, "physical_execution": False,
              "status": "INCOMPLETE", "shapes": [], "commands": commands}

    def run(argv, success=True):
        argv = list(map(str, argv))
        result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
        commands.append(dict(argv=argv, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
        require((result.returncode == 0) == success, repr(commands[-1]))
        if not success:
            require(result.returncode in (1, 2), "refusal must not be a crash: " + repr(commands[-1]))
            require(result.stderr.strip(), "refusal must explain why")
        return result.stdout

    try:
        report["tools"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.issuer, args.validator, args.disassembler)}
        run([args.validator, "--version"])
        # Axis, storage-range and K-loop bounds tested without allocating buffers
        # or dispatching enormous diagnostic shapes.
        shapes = [(2, 3, 4), (2, 2, 3), (1, 1, 1), (1, 7, 9), (9, 7, 1),
                  (3, 5, 7), (17, 31, 13), (65, 3, 67),
                  (3, 2, 5), (1, 257, 1), (5, 1, 7),
                  (65535, 1, 1), (1, 1, 65535), (1, 33554432, 1),
                  (4096, 1, 8192)]
        for m, k, n in shapes:
            shape = f"{m},{k},{n}"
            directory = output / shape.replace(",", "-")
            run([args.issuer, "--shape", shape, "--output-directory", directory])
            artifacts = {}
            for kind, bindings in (("fill", 1), ("gemm", 3)):
                binary = directory / f"{kind}.spv"
                run([args.validator, "--target-env", "vulkan1.2", binary])
                assembly = run([args.disassembler, binary])
                (directory / f"{kind}.spvasm").write_text(assembly)
                entries = re.findall(r'OpEntryPoint GLCompute (%\S+) "([^"]+)"', assembly)
                require(len(entries) == 1, "exactly one compute entry required")
                entry, name = entries[0]
                require(name == "__matcore_strict_gemm_f32_v1_kernel", "foreign entry")
                require(f"OpExecutionMode {entry} LocalSize 1 1 1" in assembly,
                        "local geometry changed")
                for mode in ("DenormPreserve", "RoundingModeRTE", "SignedZeroInfNanPreserve"):
                    require(len(re.findall(r"OpExecutionMode " + re.escape(entry) +
                                           " " + mode + r" 32\b", assembly)) == 1,
                            "missing/duplicate f32 control " + mode)
                for forbidden in ("RelaxedPrecision", "FPFastMathMode", "FPRoundingMode",
                                  "DenormFlushToZero", "RoundingModeRTZ", "OpFunctionCall",
                                  "OpExtInst ", "OpDot", "OpFma", "OpMatrixTimesMatrix"):
                    require(forbidden not in assembly, "unsupported operation/permission " + forbidden)
                variables = re.findall(r"OpDecorate (%\S+) Binding (\d+)", assembly)
                require(sorted(int(binding) for _, binding in variables) == list(range(bindings)),
                        "binding inventory differs")
                for variable, _ in variables:
                    require(f"OpDecorate {variable} DescriptorSet 0" in assembly, "descriptor set")
                strides = re.findall(r"OpDecorate %\S+ ArrayStride (\d+)", assembly)
                require(strides and all(x == "4" for x in strides), "f32 stride")
                floats = re.findall(r"^\s*(%\S+) = OpF(?:Mul|Add)\b", assembly, re.M)
                controls = re.findall(r"OpDecorate (%\S+) NoContraction\b", assembly)
                require(len(floats) == (2 if kind == "gemm" else 0) and
                        sorted(floats) == sorted(controls), "exact separate controlled multiply/add")
                artifacts[kind] = hashlib.sha256(binary.read_bytes()).hexdigest()
            # Inspection CLI is deliberately no-clobber, not a shader cache.
            run([args.issuer, "--shape", shape, "--output-directory", directory], success=False)
            require(artifacts == {kind: hashlib.sha256((directory / f"{kind}.spv").read_bytes()).hexdigest()
                                  for kind in artifacts}, "existing output modified")
            report["shapes"].append({"M": m, "K": k, "N": n, "sha256": artifacts})

        negatives = ["0,3,4", "2,0,4", "2,3,0", "-1,3,4", "65536,1,1",
                     "1,1,65536", "1,33554433,1", "4096,1,8193",
                     "9223372036854775807,9223372036854775807,2", "2,3",
                     "2,3,4,5", "2,3,x", "2,3,4junk", "2,,4"]
        for index, shape in enumerate(negatives):
            directory = output / f"rejected-{index}"
            run([args.issuer, "--shape", shape, "--output-directory", directory], success=False)
            require(not directory.exists(), "rejected shape published artifacts")
        directory = output / "foreign-ir"
        run([args.issuer, "--input", "untrusted.mlir", "--output-directory", directory], success=False)
        require(not directory.exists(), "foreign IR accepted")
        report["rejections"] = len(negatives) + 1 + len(shapes)
        report["status"] = "PASSED"
        print(f"PASS {len(shapes)} shape pairs; {report['rejections']} CLI refusals; no device execution")
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
