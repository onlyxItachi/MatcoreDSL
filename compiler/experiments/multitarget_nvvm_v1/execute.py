#!/usr/bin/env python3
"""Bounded, explicit research execution; never imported by the product runtime.

Launches the actual MLIR-generated fill and GEMM kernels. All allocations and
transfers are test-local. A host output is replaced only after checked device
completion and copy-back. The accepted domain is deliberately bounded.
"""
import argparse
import ctypes as c
import math
from pathlib import Path
import random
import struct


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def f32(value):
    return c.c_float(value).value


def bits(value):
    return struct.pack("f", value)


class Driver:
    def __init__(self, fill, matmul):
        self.cuda = c.CDLL("libcuda.so.1")
        self.context = c.c_void_p()
        self.modules = []
        self.functions = []
        self.call("cuInit", c.c_uint(0))
        self.device = c.c_int()
        self.call("cuDeviceGet", c.byref(self.device), c.c_int(0))
        major, minor = c.c_int(), c.c_int()
        self.call("cuDeviceComputeCapability", c.byref(major), c.byref(minor), self.device)
        require((major.value, minor.value) == (8, 9), "requires physical sm_89")
        name = c.create_string_buffer(256)
        self.call("cuDeviceGetName", name, c.c_int(256), self.device)
        print("device:", name.value.decode(), "compute capability 8.9")
        self.call("cuDevicePrimaryCtxRetain", c.byref(self.context), self.device)
        self.call("cuCtxPushCurrent_v2", self.context)
        for artifact in (fill, matmul):
            module, function = c.c_void_p(), c.c_void_p()
            self.call("cuModuleLoad", c.byref(module), str(artifact).encode())
            self.modules.append(module)
            self.call("cuModuleGetFunction", c.byref(function), module,
                      b"__matcore_strict_gemm_f32_v1_kernel")
            self.functions.append(function)

    def call(self, name, *args):
        code = getattr(self.cuda, name)(*args)
        if code:
            message = c.c_char_p()
            self.cuda.cuGetErrorString(c.c_int(code), c.byref(message))
            raise RuntimeError(f"{name}: CUDA {code}: {message.value!r}")

    def close(self):
        self.call("cuCtxSynchronize")
        for module in reversed(self.modules):
            self.call("cuModuleUnload", module)
        previous = c.c_void_p()
        self.call("cuCtxPopCurrent_v2", c.byref(previous))
        self.call("cuDevicePrimaryCtxRelease_v2", self.device)

    @staticmethod
    def descriptor(pointer, rows, columns):
        return [pointer, pointer, 0, rows, columns, columns, 1]

    def launch(self, function, values, m, n, invalid_block=False):
        fields = [c.c_uint64(value) for value in values]
        args = (c.c_void_p * len(fields))(*[c.addressof(value) for value in fields])
        self.call("cuLaunchKernel", function, c.c_uint(m), c.c_uint(n), c.c_uint(1),
                  c.c_uint(2048 if invalid_block else 1), c.c_uint(1), c.c_uint(1),
                  c.c_uint(0), c.c_void_p(), args, c.c_void_p())

    def gemm(self, lhs, rhs, m, k, n, published, inject_failure=False,
             alias_device_inputs=False):
        require(all(isinstance(x, int) and 0 <= x <= 65535 for x in (m, k, n)),
                "invalid/beyond-research-limit shape")
        require(max(m * k, k * n, m * n) <= 262144, "research allocation bound")
        require(len(lhs) == m * k and len(rhs) == k * n, "shape/storage mismatch")
        require(len(published) == m * n, "output capacity mismatch")
        require(not alias_device_inputs or (lhs == rhs and m * k == k * n),
                "aliased device-input fixture requires equal source bytes")
        if not m or not n:
            return
        allocations = []
        try:
            for count in (m * k, k * n, m * n):
                pointer = c.c_uint64()
                self.call("cuMemAlloc_v2", c.byref(pointer), c.c_size_t((count + 2) * 4))
                allocations.append(pointer)
            for values, pointer in zip((lhs, rhs, published), allocations):
                # Seed output with caller data: upstream linalg.fill must overwrite it.
                host = (c.c_float * (len(values) + 2))(12345.0, *values, -54321.0)
                self.call("cuMemcpyHtoD_v2", pointer, host, c.c_size_t((len(values) + 2) * 4))
            a = self.descriptor(allocations[0].value + 4, m, k)
            b = self.descriptor(allocations[0 if alias_device_inputs else 1].value + 4, k, n)
            out = self.descriptor(allocations[2].value + 4, m, n)
            self.launch(self.functions[0], out, m, n)
            if inject_failure:
                # Real first-kernel completion then real rejected second launch.
                self.call("cuCtxSynchronize")
            self.launch(self.functions[1], a + b + out, m, n, inject_failure)
            self.call("cuCtxSynchronize")
            completed = None
            for index, (original, pointer) in enumerate(zip((lhs, rhs, published), allocations)):
                host = (c.c_float * (len(original) + 2))()
                self.call("cuMemcpyDtoH_v2", host, pointer, c.c_size_t((len(original) + 2) * 4))
                require(host[0] == 12345.0 and host[-1] == -54321.0, "device guard modified")
                if index < 2:
                    require(all(bits(x) == bits(y) for x, y in zip(host[1:-1], original)),
                            "read-only input changed")
                else:
                    completed = list(host)[1:-1]
            published[:] = completed
        finally:
            # Do not release storage used by an outstanding kernel, including errors.
            self.call("cuCtxSynchronize")
            for pointer in reversed(allocations):
                self.call("cuMemFree_v2", pointer)


def reference(a, b, m, k, n):
    result = []
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc = f32(acc + f32(a[i * k + p] * b[p * n + j]))
            result.append(acc)
    return result


def compare(actual, expected, name):
    require(len(actual) == len(expected), f"{name}: result size")
    for index, (a, e) in enumerate(zip(actual, expected)):
        require((math.isnan(a) and math.isnan(e)) or bits(a) == bits(e),
                f"{name}[{index}]: {a!r} != strict reference {e!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_directory", type=Path)
    args = parser.parse_args()
    driver = Driver(args.artifact_directory / "fill.cubin",
                    args.artifact_directory / "matmul.cubin")
    passed = 0

    def case(name, m, k, n, a=None, b=None):
        nonlocal passed
        if a is None:
            a = [f32(((i * 7) % 19 - 9) / 7) for i in range(m * k)]
        if b is None:
            b = [f32(((i * 11) % 23 - 11) / 11) for i in range(k * n)]
        a, b = list(map(f32, a)), list(map(f32, b))
        output = [4711.0] * (m * n)
        driver.gemm(a, b, m, k, n, output)
        compare(output, reference(a, b, m, k, n), name)
        # Independent double-precision arithmetic oracle, separate from bitwise f32.
        for i in range(m):
            for j in range(n):
                products = [float(a[i * k + p]) * float(b[p * n + j]) for p in range(k)]
                if all(math.isfinite(x) for x in products) and math.isfinite(output[i * n + j]):
                    exact = math.fsum(products)
                    bound = 2e-7 * max(k, 1) * math.fsum(abs(x) for x in products) + 1e-44
                    require(abs(output[i * n + j] - exact) <= bound, name + ": double oracle")
        print("PASS", name)
        passed += 1
        return output

    try:
        case("rectangular-noncommuting", 2, 3, 4)
        case("odd-boundaries", 39, 37, 29)
        case("zero-K-overwrites-positive-zero", 3, 0, 5)
        case("zero-M-no-launch", 0, 7, 5)
        case("zero-N-no-launch", 3, 7, 0)
        case("DOT-geometry", 1, 7, 1)
        case("outer-product-geometry", 7, 1, 5)
        case("reject-implicit-FMA", 1, 2, 1, [1.0, 1 + 2**-23], [-1.0, 1 - 2**-23])
        case("reject-reduction-reassociation", 1, 3, 1, [2**24, 1.0, -(2**24)], [1.0] * 3)
        case("gradual-underflow", 1, 1, 2, [2**-126], [0.5, 2**-23])
        case("signed-zero", 1, 1, 2, [-0.0], [1.0, -1.0])
        case("NaN-Inf-no-finite-assumption", 1, 1, 3, [math.inf], [0.0, 1.0, -1.0])
        case("finite-overflow", 1, 1, 1, [3e38], [3e38])
        first = case("two-GEMM-first", 2, 3, 4)
        case("two-GEMM-lhs-carry", 2, 4, 3, first, [f32(i / 3) for i in range(12)])
        case("two-GEMM-rhs-carry", 3, 2, 4, [f32(i / 5) for i in range(6)], first)
        same = [f32(i - 3) for i in range(9)]
        expected = reference(same, same, 3, 3, 3)
        driver.gemm(same, same, 3, 3, 3, same)
        compare(same, expected, "all-host-descriptors-alias")
        print("PASS all-host-descriptors-alias")
        passed += 1
        same = [f32(i - 3) for i in range(9)]
        output = [9876.0] * 9
        driver.gemm(same, same, 3, 3, 3, output, alias_device_inputs=True)
        compare(output, reference(same, same, 3, 3, 3), "actual-device-input-alias")
        print("PASS actual-device-input-alias")
        passed += 1
        random_shapes = random.Random(0x4D44534C)
        for index in range(32):
            m, k, n = [random_shapes.randrange(0, 10) for _ in range(3)]
            case(f"seeded-dynamic-{index}:{m}x{k}x{n}", m, k, n)
        for name, dimensions, injected in (
                ("second-launch-failure-after-fill", (2, 3, 4), True),
                ("negative-shape-fails-before-write", (-1, 3, 4), False),
                ("beyond-grid-limit-fails-before-write", (2, 3, 65536), False)):
            output = [9876.0] * 8
            try:
                driver.gemm([1.0] * 6, [1.0] * 12, *dimensions, output, injected)
                raise AssertionError("failure not observed")
            except RuntimeError as error:
                require(output == [9876.0] * 8, name + ": host output changed")
                if injected:
                    require("cuLaunchKernel" in str(error), "wrong injected failure")
            print("PASS", name)
            passed += 1
        print(f"PASS {passed} bounded NVIDIA cases; performance not measured")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
