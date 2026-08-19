#!/usr/bin/env python3
"""Binary bloat and compile-time stress test across 100+ static GEMM shapes."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import tempfile
import time
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


def generate_shapes() -> list[tuple[int, int, int, str]]:
    shapes: list[tuple[int, int, int, str]] = []
    # 1. Square shapes (16 shapes)
    for dim in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 48]:
        shapes.append((dim, dim, dim, f"sq_{dim}"))

    # 2. Rectangular shapes (30 shapes)
    m_list = [4, 8, 16, 24, 32]
    n_list = [8, 16, 32, 48]
    k_list = [8, 16, 32]
    for m, n, k in itertools.product(m_list, n_list, k_list):
        if (m, n, k) not in [(s[0], s[1], s[2]) for s in shapes]:
            shapes.append((m, n, k, f"rect_{m}x{n}x{k}"))

    # 3. GEMV shapes (20 shapes)
    for m in [2, 4, 8, 12, 16, 20, 24, 28, 32, 48]:
        for k in [16, 64]:
            shapes.append((m, 1, k, f"gemv_{m}x1x{k}"))

    # 4. GEVM shapes (20 shapes)
    for n in [2, 4, 8, 12, 16, 20, 24, 28, 32, 48]:
        for k in [16, 64]:
            shapes.append((1, n, k, f"gevm_1x{n}x{k}"))

    # 5. DOT shapes (16 shapes)
    for k in [4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512]:
        shapes.append((1, 1, k, f"dot_1x1x{k}"))

    # 6. GER rank-1 shapes (18 shapes)
    for m in [4, 8, 12, 16, 24, 32]:
        for n in [4, 8, 16]:
            shapes.append((m, n, 1, f"ger_{m}x{n}x1"))

    return shapes


def generate_stress_source(shapes: list[tuple[int, int, int, str]], aot: bool) -> str:
    lines = [
        '#include <cstdint>',
        '#include <matcore/runtime_c.h>',
        '',
    ]
    if aot:
        for m, n, k, name in shapes:
            lines.append(f'__attribute__((target("avx2,fma"), noinline))')
            lines.append(f'void direct_{name}(float *__restrict out, const float *__restrict lhs, const float *__restrict rhs) {{')
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
                lines.append(f'    const float l_elem = lhs[i];')
                lines.append('    #pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)')
                lines.append(f'    for (int j = 0; j < {n}; ++j) {{ out[i * {n} + j] = l_elem * rhs[j]; }}')
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
            lines.append(f'extern "C" matcore_status_v0 dispatch_{name}(const matcore_tensor_desc_v0 *c, const matcore_tensor_desc_v0 *a, const matcore_tensor_desc_v0 *b, const matcore_policy_v0 *policy) {{')
            lines.append(f'  if (c && a && b && policy &&')
            lines.append(f'      c->dims[0] == {m} && c->dims[1] == {n} &&')
            lines.append(f'      a->dims[0] == {m} && a->dims[1] == {k} &&')
            lines.append(f'      b->dims[0] == {k} && b->dims[1] == {n} &&')
            lines.append(f'      c->strides[1] == 1 && a->strides[1] == 1 && b->strides[1] == 1 &&')
            lines.append(f'      c->data != a->data && c->data != b->data) {{')
            lines.append(f'    direct_{name}(static_cast<float*>(c->data), static_cast<const float*>(a->data), static_cast<const float*>(b->data));')
            lines.append('    matcore_status_v0 status{{}};')
            lines.append('    status.code = MATCORE_STATUS_OK_V0;')
            lines.append('    return status;')
            lines.append('  }')
            lines.append('  return matcore_runtime_gemm_f32_v0(c, a, b, policy);')
            lines.append('}')
            lines.append('')
    else:
        # Runtime-dispatch baseline
        for m, n, k, name in shapes:
            lines.append(f'extern "C" matcore_status_v0 dispatch_{name}(const matcore_tensor_desc_v0 *c, const matcore_tensor_desc_v0 *a, const matcore_tensor_desc_v0 *b, const matcore_policy_v0 *policy) {{')
            lines.append('  return matcore_runtime_gemm_f32_v0(c, a, b, policy);')
            lines.append('}')
            lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clangxx", type=Path, required=True)
    parser.add_argument("--include-dir", type=Path, required=True)
    args = parser.parse_args()

    shapes = generate_shapes()
    require(len(shapes) >= 100, f"Expected at least 100 shapes, got {len(shapes)}")
    print(f"Generated {len(shapes)} distinct static GEMM geometries across Square, Rect, GEMV, GEVM, DOT, and GER.")

    include_dir = args.include_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="matcore-binary-bloat-") as tmpdir:
        tmppath = Path(tmpdir)

        # 1. Compile Runtime-Dispatch Baseline
        baseline_src = tmppath / "baseline.cpp"
        baseline_obj = tmppath / "baseline.obj"
        baseline_src.write_text(generate_stress_source(shapes, aot=False), encoding="utf-8")

        start_time = time.perf_counter()
        res_baseline = run([
            str(args.clangxx),
            "-c",
            "-O3",
            "-std=c++20",
            f"-I{include_dir}",
            str(baseline_src),
            "-o",
            str(baseline_obj),
        ], tmppath)
        baseline_compile_time = time.perf_counter() - start_time
        require(res_baseline.returncode == 0, f"Baseline compile failed: {res_baseline.stderr}")

        baseline_src_size = baseline_src.stat().st_size
        baseline_obj_size = baseline_obj.stat().st_size

        # 2. Compile Static AOT Specialization
        aot_src = tmppath / "aot.cpp"
        aot_obj = tmppath / "aot.obj"
        aot_src.write_text(generate_stress_source(shapes, aot=True), encoding="utf-8")

        start_time = time.perf_counter()
        res_aot = run([
            str(args.clangxx),
            "-c",
            "-O3",
            "-std=c++20",
            f"-I{include_dir}",
            str(aot_src),
            "-o",
            str(aot_obj),
        ], tmppath)
        aot_compile_time = time.perf_counter() - start_time
        require(res_aot.returncode == 0, f"AOT compile failed: {res_aot.stderr}")

        aot_src_size = aot_src.stat().st_size
        aot_obj_size = aot_obj.stat().st_size

        print("=== BINARY BLOAT AND COMPILE-TIME MEASUREMENT REPORT ===")
        print(f"Configurations evaluated: {len(shapes)} static GEMM shapes")
        print(f"Baseline Source Size:   {baseline_src_size / 1024.0:.2f} KB")
        print(f"Baseline Object Size:   {baseline_obj_size / 1024.0:.2f} KB")
        print(f"Baseline Compile Time:  {baseline_compile_time:.3f} s")
        print()
        print(f"Static AOT Source Size: {aot_src_size / 1024.0:.2f} KB")
        print(f"Static AOT Object Size: {aot_obj_size / 1024.0:.2f} KB")
        print(f"Static AOT Compile Time:{aot_compile_time:.3f} s")
        print()
        size_delta_bytes = aot_obj_size - baseline_obj_size
        bytes_per_kernel = size_delta_bytes / len(shapes)
        print(f"Total Object Size Delta: {size_delta_bytes / 1024.0:.2f} KB")
        print(f"Incremental Code Cost:   {bytes_per_kernel:.1f} bytes / specialized kernel")
        print(f"Compile Time Scaling:    {aot_compile_time / baseline_compile_time:.2f}x baseline compile time")
        print()

    print("Binary bloat and compile-time stress test passed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
