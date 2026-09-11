#!/usr/bin/env python3
"""Synthetic classifier controls; never physical execution evidence."""
import unittest
from run import inspect


class InspectionTests(unittest.TestCase):
    symbols = ("0 T __matcore_strict_gemm_f32_v1\n"
               "1 T _mlir_ciface___matcore_strict_gemm_f32_v1\n")
    x86 = "file format elf64-x86-64\n mulps %xmm0, %xmm1\n addss %xmm0, %xmm1\n"
    arm = "file format elf64-littleaarch64\n fmul v0.4s, v1.4s, v2.4s\n fadd s0, s1, s2\n"

    def test_x86_separate(self):
        inspect(self.x86, self.symbols, " U memset\n")

    def test_arm_separate(self):
        inspect(self.arm, self.symbols, "", True)

    def test_x86_fma_rejected(self):
        for mnemonic in ("vfmadd231ps", "vfnmsub132ss", "fmadd213ps"):
            with self.assertRaises(RuntimeError):
                inspect(self.x86 + mnemonic, self.symbols, "")

    def test_arm_fma_rejected(self):
        for mnemonic in ("fmla", "fmls", "fmadd", "fmsub", "fnmadd", "fnmsub"):
            with self.assertRaises(RuntimeError):
                inspect(self.arm + mnemonic, self.symbols, "", True)

    def test_wrong_architecture_rejected(self):
        with self.assertRaises(RuntimeError):
            inspect(self.x86, self.symbols, "", True)

    def test_provider_import_rejected(self):
        with self.assertRaises(RuntimeError):
            inspect(self.x86, self.symbols, " U cblas_sgemm\n")

    def test_extra_function_rejected(self):
        for binding in ("T", "t", "W", "w"):
            with self.assertRaises(RuntimeError):
                inspect(self.x86, self.symbols + f"2 {binding} extra\n", "")

    def test_missing_arithmetic_rejected(self):
        with self.assertRaises(RuntimeError):
            inspect(self.x86.replace("mulps", "movaps"), self.symbols, "")


if __name__ == "__main__":
    unittest.main()
