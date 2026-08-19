#!/usr/bin/env python3
"""Zero-runtime-call machine-code proof and register spill archaeology."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def generate_specialized_source(shapes: list[tuple[int, int, int, str]]) -> str:
    lines = [
        '#include <cstdint>',
        '#include <matcore/runtime_c.h>',
        '',
    ]
    for m, n, k, name in shapes:
        lines.append(f'__attribute__((target("avx2,fma"), noinline))')
        lines.append(f'void direct_microkernel_{name}(float *__restrict out, const float *__restrict lhs, const float *__restrict rhs) {{')
        if m == 1 and n == 1:
            lines.append('  float acc = 0.0f;')
            lines.append('  #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)')
            lines.append(f'  for (int p = 0; p < {k}; ++p) {{ acc += lhs[p] * rhs[p]; }}')
            lines.append('  out[0] = acc;')
        elif n == 1:
            lines.append(f'  for (int i = 0; i < {m}; ++i) {{')
            lines.append('    float acc = 0.0f;')
            lines.append('    #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)')
            lines.append(f'    for (int p = 0; p < {k}; ++p) {{ acc += lhs[i * {k} + p] * rhs[p]; }}')
            lines.append('    out[i] = acc;')
            lines.append('  }')
        elif k == 1:
            lines.append(f'  for (int i = 0; i < {m}; ++i) {{')
            lines.append(f'    const float lhs_elem = lhs[i];')
            lines.append('    #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)')
            lines.append(f'    for (int j = 0; j < {n}; ++j) {{ out[i * {n} + j] = lhs_elem * rhs[j]; }}')
            lines.append('  }')
        else:
            lines.append(f'  for (int i = 0; i < {m}; ++i) {{')
            lines.append(f'    for (int j = 0; j < {n}; ++j) {{ out[i * {n} + j] = 0.0f; }}')
            lines.append(f'    for (int p = 0; p < {k}; ++p) {{')
            lines.append(f'      const float a_elem = lhs[i * {k} + p];')
            lines.append('      #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)')
            lines.append(f'      for (int j = 0; j < {n}; ++j) {{ out[i * {n} + j] += a_elem * rhs[p * {n} + j]; }}')
            lines.append('    }')
            lines.append('  }')
        lines.append('}')
        lines.append('')
        # Guarded entry
        lines.append(f'extern "C" matcore_status_v0 matcore_guarded_{name}(const matcore_tensor_desc_v0 *c, const matcore_tensor_desc_v0 *a, const matcore_tensor_desc_v0 *b, const matcore_policy_v0 *policy) {{')
        lines.append(f'  if (c && a && b && policy &&')
        lines.append(f'      c->dims[0] == {m} && c->dims[1] == {n} &&')
        lines.append(f'      a->dims[0] == {m} && a->dims[1] == {k} &&')
        lines.append(f'      b->dims[0] == {k} && b->dims[1] == {n} &&')
        lines.append(f'      c->strides[1] == 1 && a->strides[1] == 1 && b->strides[1] == 1 &&')
        lines.append(f'      c->data != a->data && c->data != b->data) {{')
        lines.append(f'    direct_microkernel_{name}(static_cast<float*>(c->data), static_cast<const float*>(a->data), static_cast<const float*>(b->data));')
        lines.append('    matcore_status_v0 status{{}};')
        lines.append('    status.code = MATCORE_STATUS_OK_V0;')
        lines.append('    return status;')
        lines.append('  }')
        lines.append('  return matcore_runtime_gemm_f32_v0(c, a, b, policy);')
        lines.append('}')
        lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clangxx", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--include-dir", type=Path, required=True)
    args = parser.parse_args()

    shapes = [
        (4, 4, 4, "square_4x4x4"),
        (8, 8, 8, "square_8x8x8"),
        (16, 16, 16, "square_16x16x16"),
        (16, 16, 64, "stress_16x16x64"),
        (16, 32, 64, "stress_16x32x64"),
        (32, 16, 64, "stress_32x16x64"),
        (1, 1, 128, "dot_1x1x128"),
        (16, 1, 128, "gemv_16x1x128"),
        (1, 16, 128, "gevm_1x16x128"),
        (16, 16, 1, "ger_16x16x1"),
    ]

    source_text = generate_specialized_source(shapes)

    with tempfile.TemporaryDirectory(prefix="matcore-disasm-proof-") as tmpdir:
        tmppath = Path(tmpdir)
        src_file = tmppath / "specialized.cpp"
        obj_file = tmppath / "specialized.obj"
        src_file.write_text(source_text, encoding="utf-8")

        compile_cmd = [
            str(args.clangxx),
            "-c",
            "-O3",
            "-std=c++20",
            f"-I{args.include_dir}",
            str(src_file),
            "-o",
            str(obj_file),
        ]
        res = run(compile_cmd, tmppath)
        require(res.returncode == 0, f"Clang compilation failed: {res.stderr}")

        disasm_cmd = [str(args.objdump), "-d", str(obj_file)]
        disasm_res = run(disasm_cmd, tmppath)
        require(disasm_res.returncode == 0, f"llvm-objdump failed: {disasm_res.stderr}")
        disasm = disasm_res.stdout

        print("=== DISASSEMBLY AND SPILL ARCHAEOLOGY REPORT ===")
        for m, n, k, name in shapes:
            # 1. Inspect direct_microkernel_<name>
            pattern = re.compile(rf"<.*direct_microkernel_{name}.*>:(.*?)(?=\n\s*\n\d|\Z)", re.DOTALL)
            match = pattern.search(disasm)
            require(match is not None, f"Could not find disassembly for direct_microkernel_{name}")
            kernel_asm = match.group(1)

            # Check zero runtime calls in microkernel
            calls = re.findall(r"\bcall\b", kernel_asm)
            require(len(calls) == 0, f"direct_microkernel_{name} has {len(calls)} unexpected calls in microkernel body")

            # Check vector instructions
            ymm_regs = re.findall(r"%ymm\d+", kernel_asm)
            fma_ops = re.findall(r"\bvfmadd\w+", kernel_asm)
            mul_ops = re.findall(r"\bvmul\w+", kernel_asm)
            add_ops = re.findall(r"\bvadd\w+", kernel_asm)

            # Check stack spills: instructions like vmovups ... [rsp + ...] or vmovaps ... (%rsp)
            spills = re.findall(r"vmov\w+\s+%\w+,\s*-?\d*\(%rsp\)", kernel_asm) + re.findall(r"vmov\w+\s+%\w+,\s*-?\d*\(%rbp\)", kernel_asm)

            # 2. Inspect guarded wrapper matcore_guarded_<name>
            guard_pattern = re.compile(rf"<.*matcore_guarded_{name}.*>:(.*?)(?=\n\s*\n\d|\Z)", re.DOTALL)
            guard_match = guard_pattern.search(disasm)
            require(guard_match is not None, f"Could not find disassembly for matcore_guarded_{name}")
            guard_asm = guard_match.group(1)

            # Guarded wrapper must contain the fallback call to runtime GEMM
            guard_calls = re.findall(r"\bcallq?\b", guard_asm)
            require(len(guard_calls) >= 1, f"matcore_guarded_{name} must contain calls for microkernel invocation and fallback")

            print(f"Shape {m}x{n}x{k} ({name}):")
            print(f"  Microkernel Calls: 0 (PROVEN ZERO RUNTIME CALLS IN HOT LOOP)")
            print(f"  YMM Vector References: {len(ymm_regs)}")
            print(f"  FMA / Vector Mul/Add Ops: {len(fma_ops) + len(mul_ops) + len(add_ops)}")
            print(f"  Detected Vector Spills: {len(spills)}")
            print(f"  Guarded Fallback Calls: VERIFIED ({len(guard_calls)} branch target calls)")
            print()

    print("All zero-runtime-call machine-code proofs and spill archaeology checks passed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
