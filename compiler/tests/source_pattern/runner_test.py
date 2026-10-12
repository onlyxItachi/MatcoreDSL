#!/usr/bin/env python3
"""Synthetic classifier controls, not generated/source execution evidence."""
import struct
import unittest
import run


def manifest_control():
    value = {key: "0" * 64 if key.endswith("_sha256") else "control" for key in run.KEYS}
    value.update(schema="matcore-source-output-pattern-v1", pattern_parameters="M:3,N:5,K:untiled",
                 source_authority="authenticated_in_process_source_closure",
                 manifest_authority="inspection_only_never_imported", accelerator_count="0",
                 source_context_sha256="sha256:" + "0" * 64,
                 binding_sha256="a" * 64, candidate_soname="libmdslc-pattern-" + "a" * 64 + ".so")
    return value


def text(value):
    return "".join(key + "=" + value[key] + "\n" for key in sorted(value))


def elf_control():
    data = bytearray(0x400)
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, ident, 3, 62, 1, 0,
                     64, 0x300, 0, 64, 56, 1, 64, 4, 0)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0, 0, 0, len(data), len(data), 4096)
    names = b"\0_mlir_ciface___matcore_strict_gemm_f32_v1\0"
    data[0x200:0x200 + len(names)] = names
    struct.pack_into("<IBBHQQ", data, 0x160 + 24, 1, 2, 0, 1, 0x100, 16)
    for index, fields in enumerate((
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 6, 0x100, 0x100, 0x40, 0, 0, 16, 0),
        (0, 2, 0, 0, 0x160, 48, 3, 2, 8, 24),
        (0, 3, 0, 0, 0x200, len(names), 0, 0, 1, 0),
    )):
        struct.pack_into("<IIQQQQIIQQ", data, 0x300 + index * 64, *fields)
    return data


class Controls(unittest.TestCase):
    def test_manifest_positive(self):
        value = manifest_control()
        self.assertEqual(run.manifest(text(value)), value)

    def test_manifest_corruptions(self):
        for key, replacement in (("schema", "forged"), ("binding_sha256", "abc"),
                                 ("object_sha256", "G" * 64),
                                 ("candidate_soname", "../foreign.so"),
                                 ("candidate_soname", "libmdslc-pattern-" + "b" * 64 + ".so")):
            with self.subTest(key=key, replacement=replacement):
                value = manifest_control()
                value[key] = replacement
                with self.assertRaises(ValueError):
                    run.manifest(text(value))
        value = manifest_control()
        for bad in (text(value) + "schema=forged\n", text(value) + "forged=authority\n",
                    text({key: val for key, val in value.items() if key != "source_context_sha256"}),
                    "not a manifest\n"):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    run.manifest(bad)

    def test_exact_source_summary(self):
        output = "Source pattern semantic oracle: 68 cases; 10001 checks; 0 failures; host_sanitized=0\n"
        self.assertEqual(run.semantic_summary(output, False)["cases"], 68)
        for bad, sanitized in ((output, True), (output.replace("68 cases", "67 cases"), False),
                               (output.replace("10001 checks", "0 checks"), False),
                               (output.replace("0 failures", "1 failures"), False),
                               (output + "SKIP unavailable\n", False)):
            with self.subTest(output=bad, sanitized=sanitized):
                with self.assertRaises(ValueError):
                    run.semantic_summary(bad, sanitized)

    def test_native_local_wrapper(self):
        self.assertEqual(run.private_wrapper_offset(elf_control(), 62), 0x100)

    def test_wrapper_elf_corruptions(self):
        for label, mutate in (
            ("wrong-machine", lambda data: struct.pack_into("<H", data, 18, 183)),
            ("not-shared", lambda data: struct.pack_into("<H", data, 16, 1)),
            ("global-function", lambda data: data.__setitem__(0x160 + 24 + 4, 0x12)),
            ("undefined", lambda data: struct.pack_into("<H", data, 0x160 + 24 + 6, 0)),
            ("nonexec-segment", lambda data: struct.pack_into("<I", data, 64 + 4, 4)),
            ("out-of-segment", lambda data: struct.pack_into("<Q", data, 0x160 + 24 + 8, 0x500)),
            ("dynsym-only", lambda data: struct.pack_into("<I", data, 0x300 + 2 * 64 + 4, 11)),
        ):
            with self.subTest(label=label):
                data = elf_control()
                mutate(data)
                with self.assertRaises(ValueError):
                    run.private_wrapper_offset(data, 62)
        with self.assertRaises(ValueError):
            run.private_wrapper_offset(elf_control()[:63], 62)

    def test_duplicate_wrapper(self):
        data = elf_control()
        data[0x160 + 48:0x160 + 72] = data[0x160 + 24:0x160 + 48]
        struct.pack_into("<Q", data, 0x300 + 2 * 64 + 32, 72)
        with self.assertRaises(ValueError):
            run.private_wrapper_offset(data, 62)


if __name__ == "__main__":
    unittest.main()
