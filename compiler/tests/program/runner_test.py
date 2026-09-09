#!/usr/bin/env python3
"""Classifier controls; these do not substitute for actual source executions."""
from types import SimpleNamespace
import unittest
from run import rejected_before_link


class RefusalClassifier(unittest.TestCase):
    def result(self, code=1, out="", err="OWNERSHIP: expected refusal"):
        return SimpleNamespace(returncode=code, stdout=out, stderr=err)

    def accepted(self, result, exists=False):
        return rejected_before_link(result, "OWNERSHIP: expected refusal", exists)

    def test_exact_checked_refusal(self):
        self.assertTrue(self.accepted(self.result()))

    def test_wrong_exit_after_expected_diagnostic(self):
        for code in (0, -11, -6, 2, 134, 139):
            with self.subTest(code=code):
                self.assertFalse(self.accepted(self.result(code=code)))

    def test_post_gate_phase_never_passes(self):
        for phase in ("VALIDATED program", "CROSS_TU_LINK", "NATIVE_LINK", "PUBLISHED"):
            with self.subTest(phase=phase):
                self.assertFalse(self.accepted(self.result(out=phase)))

    def test_publication_and_wrong_diagnostic(self):
        self.assertFalse(self.accepted(self.result(), exists=True))
        self.assertFalse(self.accepted(self.result(err="another failure")))

    def test_sanitizer_or_crash_is_not_refusal(self):
        for marker in ("AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:",
                       "Segmentation fault", "DEADLYSIGNAL", "core dumped", "Aborted"):
            with self.subTest(marker=marker):
                self.assertFalse(self.accepted(self.result(err="OWNERSHIP: expected refusal\n" + marker)))
                self.assertFalse(self.accepted(self.result(out=marker)))


if __name__ == "__main__":
    unittest.main()
