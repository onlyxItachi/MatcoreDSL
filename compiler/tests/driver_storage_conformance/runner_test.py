#!/usr/bin/env python3
"""Readelf/parser/source-copy unit checks; does not run or build a compiler."""
from pathlib import Path
import tempfile
import unittest

import run


class RunnerChecks(unittest.TestCase):
    def test_sanitizer_link_selection_is_explicit(self):
        plain = run.ordinary_link_command("clang++-21", "source.o", ["candidate.so", "runtime.so"],
                                          "/library", "program", False)
        sanitized = run.ordinary_link_command("clang++-21", "source.o", ["candidate.so", "runtime.so"],
                                              "/library", "program", True)
        self.assertNotIn("-fsanitize=address,undefined", plain)
        self.assertEqual(sanitized.count("-fsanitize=address,undefined"), 1)
        self.assertEqual([arg for arg in sanitized if arg != "-fsanitize=address,undefined"], plain)
        spaced = run.ordinary_link_command("clang++-21", "source.o", [],
                                           "/path with spaces,and-comma", "program", False)
        self.assertEqual(spaced[spaced.index("-rpath") + 1:spaced.index("-o")],
                         ["-Xlinker", "/path with spaces,and-comma"])

    def test_required_shared_dependencies(self):
        text = """
 0x1 (NEEDED) Shared library: [libmatcore_closed_candidates_isolated_v1.so]
 0x1 (NEEDED) Shared library: [libmatcore_runtime.so.0]
 0x1 (NEEDED) Shared library: [libc.so.6]
"""
        self.assertEqual(len(run.dynamic_dependencies(text)["needed"]), 3)
        for name in ("libmatcore_closed_candidates_isolated_v1.so", "libmatcore_runtime.so.0"):
            with self.subTest(missing=name), self.assertRaises(ValueError):
                run.dynamic_dependencies(text.replace(name, "unrelated.so"))

    def test_dependency_names_must_be_needed_records(self):
        with self.assertRaises(ValueError):
            run.dynamic_dependencies("libmatcore_closed_candidates_isolated_v1.so libmatcore_runtime.so.0")

    def test_elf_object_header_and_negative_controls(self):
        text = """ELF Header:
  Class: ELF64
  Type: REL (Relocatable file)
  Machine: Advanced Micro Devices X86-64
"""
        self.assertEqual(run.relocatable_header(text)["Class"], "ELF64")
        arm = text.replace("Advanced Micro Devices X86-64", "AArch64")
        self.assertEqual(run.relocatable_header(arm, "AArch64")["Machine"], "AArch64")
        with self.assertRaises(ValueError):
            run.relocatable_header(text, "AArch64")
        for old, new in (("ELF64", "ELF32"), ("REL (Relocatable file)", "DYN (Shared object file)"),
                         ("Advanced Micro Devices X86-64", "AArch64")):
            with self.subTest(replacement=new), self.assertRaises(ValueError):
                run.relocatable_header(text.replace(old, new))

    def test_reassociate_copies_preserve_original_names_lines_and_host(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="mdslc-runner-unit-") as directory:
            target = Path(directory) / "reassociate-source"
            hashes = run.reassociate_sources(source, target)
            self.assertEqual(len(hashes), 3)
            for carry in ("lhs", "rhs"):
                original = (source / f"{carry}.mdsl").read_text().splitlines()
                changed = (target / f"{carry}.mdsl").read_text().splitlines()
                self.assertEqual(len(changed), len(original))
                differing = [number for number, pair in enumerate(zip(original, changed), 1)
                             if pair[0] != pair[1]]
                self.assertEqual(differing, [11, 15])
                self.assertEqual("\n".join(changed).count("mdsl::Numerics::reassociate_f32"), 2)
            self.assertEqual((target / "storage_conformance_host.h").read_bytes(),
                             (source / "storage_conformance_host.h").read_bytes())


if __name__ == "__main__":
    unittest.main()
