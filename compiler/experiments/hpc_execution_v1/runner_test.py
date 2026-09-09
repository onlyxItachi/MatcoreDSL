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

    def test_numerical_profile_is_explicit(self):
        self.assertEqual(runner.oracle_profile("strict"), ([], 24))
        self.assertEqual(runner.oracle_profile("reassociate"), (["--relaxed"], 16))
        with self.assertRaises(ValueError):
            runner.oracle_profile("fast")
        specs = runner.primitive_specs("baseline.o", [("ordered", "o.o", "ordered")],
                                       [("fused", "f.o", "fused")])
        self.assertEqual([item[3] for item in specs], ["strict", "strict", "reassociate"])

    def test_artifact_labels_cannot_alias(self):
        for label in ("baseline", "same"):
            with self.assertRaises(ValueError):
                runner.primitive_specs("baseline.o", [("same", "a.o", "a")],
                                       [(label, "b.o", "b")])
        for label, entry in (("../escape", "valid"), ("valid", "bad-entry")):
            with self.assertRaises(ValueError):
                runner.primitive_specs("baseline.o", [], [(label, "b.o", entry)])


if __name__ == "__main__":
    unittest.main()
