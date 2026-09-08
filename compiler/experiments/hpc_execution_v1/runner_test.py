#!/usr/bin/env python3
"""Tiny permanent negative controls for evidence parsing and source pairing."""
import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hpc_runner", source / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EvidenceControls(unittest.TestCase):
    header = "Class: ELF64\nMachine: Advanced Micro Devices X86-64\nType: REL (Relocatable file)\n"

    def test_actual_object_kind(self):
        runner.elf_header(self.header, True)
        with self.assertRaises(ValueError):
            runner.elf_header(self.header, False)

    def test_wrong_class_or_architecture(self):
        for old, new in [("ELF64", "ELF32"), ("Advanced Micro Devices X86-64", "AArch64"),
                         ("REL (Relocatable file)", "DYN (Shared object file)")]:
            with self.assertRaises(ValueError):
                runner.elf_header(self.header.replace(old, new), True)

    def test_exact_reassociation_copy(self):
        strict = (source / "strict.mdsl").read_text()
        relaxed = (source / "reassociate.mdsl").read_text()
        self.assertEqual(strict.count("Numerics::strict_f32"), 1)
        self.assertEqual(strict.replace("Numerics::strict_f32", "Numerics::reassociate_f32"), relaxed)


if __name__ == "__main__":
    unittest.main()
