#!/usr/bin/env python3
"""Challenge the actual build-time gate with captured inspector corruption.

No mutated image is executed and no artifact/manifest acquires authority.
Positive cases use actual image/tool output. Each negative changes one observed
metadata/assembly/symbol fact, checking the SAME production CMake gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

KERNEL = "__matcore_strict_fused_gemm_f32_v1_kernel"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", choices=("nvvm", "rocdl"), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    llvm = Path("/usr/lib/llvm-21/bin")
    tools = {"nm": llvm / "llvm-nm", "objdump": llvm / "llvm-objdump",
             "readelf": llvm / "llvm-readelf", "cuobjdump": Path("/usr/local/cuda/bin/cuobjdump")}
    outputs = {}
    commands = {"--undefined-only": (tools["nm"],), "--defined-only": (tools["nm"],)}
    if args.target == "nvvm":
        commands.update({"--dump-sass": (tools["cuobjdump"],), "--dump-elf": (tools["cuobjdump"],)})
    else:
        commands.update({"-d": (tools["objdump"],), "--notes": (tools["readelf"],),
                         "--hex-dump=.rodata": (tools["readelf"],),
                         "--dyn-syms": (tools["readelf"], "--wide"),
                         "--section-headers": (tools["readelf"], "--wide")})
    for option, prefix in commands.items():
        result = subprocess.run([*map(str, prefix), option, str(args.image)],
                                capture_output=True, text=True, timeout=30)
        require(result.returncode == 0, f"real inspection failed {option}: {result.stderr}")
        outputs[option] = result.stdout
    shim = args.build / "captured-inspector.py"
    packet = args.build / "captured-output.json"
    shim.write_text("#!" + sys.executable + "\nimport json,sys\nfrom pathlib import Path\n"
                    "data=json.loads(Path(__file__).with_name('captured-output.json').read_text())\n"
                    "key=next(arg for arg in sys.argv[1:] if arg in data)\n"
                    "sys.stdout.write(data[key])\n")
    shim.chmod(0o755)

    def run_gate(inspectors):
        return subprocess.run(["cmake", f"-DIMAGE={args.image}", f"-DTARGET_KIND={args.target}",
                               f"-DNM={inspectors['nm']}", f"-DOBJDUMP={inspectors['objdump']}",
                               f"-DREADELF={inspectors['readelf']}", f"-DCUOBJDUMP={inspectors['cuobjdump']}",
                               "-P", str(args.gate)], capture_output=True, text=True, timeout=60)

    actual = run_gate(tools)
    require(actual.returncode == 0, f"actual build gate failed: {actual.stderr}")
    packet.write_text(json.dumps(outputs))
    baseline = run_gate(dict.fromkeys(tools, shim))
    require(baseline.returncode == 0, f"unchanged captured gate failed: {baseline.stderr}")
    negatives = []

    def reject(label, key, transform):
        changed = dict(outputs)
        changed[key] = transform(changed[key])
        require(changed[key] != outputs[key], f"negative control ineffective: {label}")
        packet.write_text(json.dumps(changed))
        result = run_gate(dict.fromkeys(tools, shim))
        require(result.returncode != 0, f"production gate accepted {label}")
        negatives.append(label)

    reject("unknown-import", "--undefined-only", lambda text: text + " U arbitrary_device_effect\n")
    reject("extra-executable", "--defined-only", lambda text: text + "0000000000000000 T arbitrary_kernel\n")
    reject("extra-local-executable", "--defined-only", lambda text: text + "0000000000000000 t arbitrary_local\n")
    assembly = "--dump-sass" if args.target == "nvvm" else "-d"
    reject("FMA", assembly, lambda text: text + ("\nFFMA R0,R1,R2,R3;\n" if args.target == "nvvm"
                                                else "\nv_fma_f32 v0,v1,v2,v3\n"))
    reject("missing-multiply", assembly, lambda text: re.sub(r"FMUL|v_mul_f32", "REMOVED", text))
    if args.target == "nvvm":
        reject("FMA32I", assembly, lambda text: text + "\nFFMA32I R0,R1,0.5,R3;\n")
        reject("half-FMA32I", assembly, lambda text: text + "\nHFMA2_32I R0,R1,0.5,R3;\n")
        reject("FTZ", assembly, lambda text: text + "\nFMUL.FTZ R0,R1,R2;\n")
        reject("unknown-call", assembly, lambda text: text + "\nCALL R0;\n")
        reject("relative-call-modifier", assembly, lambda text: text.replace("CALL.REL.NOINC", "CALL.REL", 1))
        reject("relative-call-nonexit", assembly,
               lambda text: re.sub(r"(CALL\.REL\.NOINC +)0x[0-9a-f]+", r"\g<1>0x0", text, count=1))
        reject("extra-disassembled-function", assembly,
               lambda text: text + "\nFunction : arbitrary_local\n/*1f40*/ EXIT ;\n")
        reject("field-size-prefix", "--dump-elf", lambda text: text.replace("Size    : 0x8", "Size    : 0x80", 1))
        reject("cbank-size-prefix", "--dump-elf", lambda text: text.replace("Value:\t0x118\n", "Value:\t0x1180\n", 1))
        reject("field-offset", "--dump-elf", lambda text: text.replace("Offset  : 0x110", "Offset  : 0x118", 1))
        reject("stack", "--dump-elf", lambda text: text.replace("frame size: 0x0", "frame size: 0x10", 1))
    else:
        reject("field-size", "--notes", lambda text: text.replace(".size:           8", ".size:           80", 1))
        reject("field-kind", "--notes", lambda text: text.replace(".value_kind:     global_buffer", ".value_kind:     by_value", 1))
        reject("extra-explicit-field", "--notes", lambda text: text.replace("hidden_block_count_x", "by_value", 1))
        reject("field-offset", "--notes", lambda text: text.replace(".offset:         0\n", ".offset:         8\n", 1))
        reject("workgroup-prefix", "--notes", lambda text: text.replace(".max_flat_workgroup_size: 1", ".max_flat_workgroup_size: 10", 1))
        reject("spill-prefix", "--notes", lambda text: text.replace(".sgpr_spill_count: 0", ".sgpr_spill_count: 01", 1))
        reject("KD-symbol-address", "--dyn-syms", lambda text: re.sub(r"([0-9a-f]+)( +64 OBJECT)", r"0000000000000000\2", text, count=1))
        reject("KD-symbol-section", "--dyn-syms", lambda text: re.sub(r"(DEFAULT +)[0-9]+( +" + KERNEL + r"\.kd)", r"\g<1>1\2", text, count=1))
        text = outputs["--hex-dump=.rodata"]
        lines = [line for line in text.splitlines() if line.startswith("0x")]
        require(len(lines) == 4, "unexpected actual64-byte descriptor dump")
        word = lines[-1].split()[1]
        rsrc1 = int.from_bytes(bytes.fromhex(word), "little")
        for shift, label, value in ((12, "round-mode", 1), (16, "denorm-mode", 0), (23, "IEEE-mode", 0)):
            mask = (1 if shift == 23 else 3) << shift
            forged = ((rsrc1 & ~mask) | (value << shift)).to_bytes(4, "little").hex()
            reject(label, "--hex-dump=.rodata", lambda text, forged=forged: text.replace(word, forged, 1))
    # Restore final manifest/report to the REAL inspector/image, not a synthetic
    # packet. A test report is not an execution authority certificate.
    final = run_gate(tools)
    require(final.returncode == 0, final.stderr)
    data = args.image.read_bytes()
    report = {"status": "PASS", "target": args.target, "image_sha256": hashlib.sha256(data).hexdigest(),
              "negative_controls": negatives, "authority": "none; actual gate falsification only"}
    (args.build / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {args.target} actual machine gate; {len(negatives)} rejected captured-inspector controls")


if __name__ == "__main__":
    main()
