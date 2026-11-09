#!/usr/bin/env python3
"""Process/diagnostic transport regression; runs on Linux and native Windows."""
from pathlib import Path
import sys
import tempfile
import unittest

from run_windows_native_pipeline import run


class DiagnosticCapture(unittest.TestCase):
    def test_mixed_diagnostic_bytes_preserve_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run([
                sys.executable, "-c",
                "import os; os.write(1, b'valid UTF-8: \\xcf\\x80\\n'); "
                "os.write(2, b'reserved device output path: \\xfc\\xff\\n'); "
                "raise SystemExit(7)",
            ], cwd=Path(directory))
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "valid UTF-8: π\n")
        self.assertEqual(result.stderr, "reserved device output path: \\xfc\\xff\n")

    def test_success_is_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run([sys.executable, "-c", "print('PASS')"], cwd=Path(directory))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "PASS\n", ""))


if __name__ == "__main__":
    unittest.main()
