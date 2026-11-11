#!/usr/bin/env python3
"""Synthetic canonical-receipt attacks only; no compiler or binary execution."""
import hashlib
import unittest
import run


def manifest_control(tm=5, tn=3):
    fields = {key: "0" * 64 if key.endswith("_sha256") else "control" for key in run.legacy.KEYS
              if key not in ("schema", "binding_sha256", "candidate_soname", "dso_sha256", "manifest_authority")}
    fields.update(source_authority="authenticated_in_process_source_closure",
                  source_context_sha256="sha256:" + "0" * 64,
                  realization=run.REALIZATION, accelerator_count="0", toolchain="21.1.8",
                  pattern_parameters=f"M:{tm},N:{tn},K:untiled")
    prefix = "schema=" + run.SCHEMA + "\n" + "".join(key + "=" + value + "\n" for key, value in fields.items())
    identity = run.binding_digest(prefix)
    return prefix + "binding_sha256=" + identity + "\n" + "candidate_soname=libmdslc-pattern-" + identity + ".so\n" + "dso_sha256=" + "0" * 64 + "\nmanifest_authority=inspection_only_never_imported\n"


def transform(tm=5, tn=3):
    peel = "" if tn == 1 else '    %full, %tail = transform.loop.peel %n : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
    return f'''module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%root: !transform.any_op) {{
    %matmul = transform.structured.match ops{{["linalg.matmul"]}} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %m, %n = transform.structured.tile_using_for %matmul tile_sizes [{tm}, {tn}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">)
    %generic = transform.structured.generalize %tiled : (!transform.any_op) -> !transform.any_op
    %scheduled = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
{peel}    transform.yield
  }}
}}
'''


def schedule(tm=5, tn=3):
    def domain(start, end, tail=False):
        width = "%remaining" if tail else str(tn)
        remainder = "        %remaining = affine.apply affine_map<(d0)[s0] -> (-d0 + s0)>(%j)[%N]\n" if tail else ""
        return f'''      scf.for %j = {start} to {end} step %tn {{
        %height = affine.min affine_map<(d0)[s0] -> (-d0 + s0, {tm})>(%i)[%M]
{remainder}        %av = memref.subview %a[%i, 0] [%height, %K] [1, 1] : memref<?x?xf32> to memref<?x?xf32>
        %bv = memref.subview %b[0, %j] [%K, {width}] [1, 1] : memref<?x?xf32> to memref<?x?xf32>
        %cv = memref.subview %c[%i, %j] [%height, {width}] [1, 1] : memref<?x?xf32> to memref<?x?xf32>
        linalg.generic {{indexing_maps = []}} ins(%av, %bv : memref<?x?xf32>, memref<?x?xf32>) outs(%cv : memref<?x?xf32>) {{
          linalg.yield
        }}
      }}
'''
    split = "" if tn == 1 else f"      %F = affine.apply affine_map<()[s0] -> ((s0 floordiv {tn}) * {tn})>()[%N]\n"
    bodies = domain("%zero", "%N" if tn == 1 else "%F")
    if tn != 1:
        bodies += domain("%F", "%N", True)
    # Deliberately synthetic generic body: this is a consumption-classifier
    # fixture, NOT an assertion of MLIR validity or arithmetic authorization.
    return f'''module {{
  func.func @__matcore_strict_gemm_f32_v1(%a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) -> memref<?x?xf32> {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %tm = arith.constant {tm} : index
    %tn = arith.constant {tn} : index
    %M = memref.dim %a, %zero : memref<?x?xf32>
    %K = memref.dim %a, %one : memref<?x?xf32>
    %N = memref.dim %b, %one : memref<?x?xf32>
    scf.for %i = %zero to %M step %tm {{
{split}{bodies}    }}
    return %c : memref<?x?xf32>
  }}
}}
'''


class Controls(unittest.TestCase):
    def test_exact_new_manifest_and_binding(self):
        text = manifest_control()
        value = run.manifest(text)
        self.assertEqual(value["schema"], run.SCHEMA)
        self.assertEqual(value["realization"], run.REALIZATION)
        prefix = text.split("binding_sha256=")[0]
        domain = run.DOMAIN.encode()
        independent = str(len(domain)).encode() + b":" + domain + str(len(prefix.encode())).encode() + b":" + prefix.encode()
        self.assertEqual(value["binding_sha256"], hashlib.sha256(independent).hexdigest())

    def test_old_new_schema_rejection_both_directions(self):
        text = manifest_control()
        with self.assertRaises(ValueError):
            run.legacy.manifest(text)
        old = text.replace(run.SCHEMA, "matcore-source-output-pattern-v1").replace("realization=" + run.REALIZATION + "\n", "")
        run.legacy.manifest(old)  # Unchanged old format accepts its own keys.
        with self.assertRaises(ValueError):
            run.manifest(old)

    def test_manifest_forgeries(self):
        text = manifest_control()
        for label, bad in (
            ("wrong-realization", text.replace(run.REALIZATION, "output-tiled-mkn")),
            ("wrong-domain", text.replace("binding_sha256=", "binding_sha256=" + "a")),
            ("changed-request", text.replace("M:5,N:3", "M:3,N:5")),
            ("changed-authority", text.replace("inspection_only_never_imported", "importable_authority")),
            ("wrong-toolchain", text.replace("toolchain=21.1.8", "toolchain=22.1.8")),
            ("missing-realization", text.replace("realization=" + run.REALIZATION + "\n", "")),
            ("duplicate-schema", text + "schema=" + run.SCHEMA + "\n"),
            ("extra-key", text + "artifact_authority=true\n"),
            ("wrong-dso", text.replace("candidate_soname=libmdslc", "candidate_soname=../libmdslc")),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                run.manifest(bad)

    def test_binding_boundary_and_prefix(self):
        text = manifest_control()
        realization = "realization=" + run.REALIZATION + "\n"
        moved = text.replace(realization, "") + realization
        for bad in (moved, text.rstrip("\n"), "junk\n" + text):
            with self.assertRaises(ValueError):
                run.manifest(bad)

    def test_recomputed_wrong_binding_domain(self):
        text = manifest_control()
        prefix = text.split("binding_sha256=")[0]
        wrong = b"matcore-source-output-pattern-binding-v1"
        material = str(len(wrong)).encode() + b":" + wrong + str(len(prefix.encode())).encode() + b":" + prefix.encode()
        old_identity = run.binding_digest(prefix)
        forged = text.replace(old_identity, hashlib.sha256(material).hexdigest())
        with self.assertRaises(ValueError):
            run.manifest(forged)

    def test_nonunit_actual_domains(self):
        for tm, tn in ((3, 5), (5, 3), (64, 64)):
            result = run.output_tile_consumption(transform(tm, tn), schedule(tm, tn), (tm, tn))
            self.assertEqual(result["outer_loop_steps"], [tm, tn, tn])
            self.assertEqual(result["n_domains"], "disjoint-full-tail")

    def test_unit_graph_identity_and_provenance_limit(self):
        t, s = transform(5, 1), schedule(5, 1)
        self.assertEqual(run.output_tile_consumption(t, s, (5, 1))["n_domains"], "unit-identity")
        # The old helper also accepts the SAME graph; no claim that frozen P1
        # registers a unit-N role or that byte identity reveals request origin.
        self.assertEqual(run.legacy.output_tile_consumption(t, s, (5, 1))["outer_loop_steps"], [5, 1])
        with self.assertRaises(ValueError):
            run.legacy.manifest(manifest_control(5, 1))

    def test_actual_transform_handle_attacks(self):
        t, s = transform(), schedule()
        for bad in (t.replace("peel %n", "peel %m"), t.replace("peel %n", "peel %foreign"),
                    t.replace("tile_using_for %matmul", "tile_using_for %foreign"),
                    t.replace('ops{["linalg.matmul"]}', 'ops{["linalg.fill"]}'),
                    t.replace("generalize %tiled", "generalize %matmul"),
                    t.replace("interchange %generic", "interchange %tiled"),
                    t.replace("[0, 2, 1]", "[0, 1, 2]"), t.replace("[5, 3, 0]", "[5, 3, 1]"),
                    "\n".join("// " + line for line in t.splitlines()),
                    t.replace("    transform.yield", t.splitlines()[-4] + "\n    transform.yield")):
            with self.subTest(transform=bad), self.assertRaises(ValueError):
                run.output_tile_consumption(bad, s, (5, 3))

    def test_unit_peel_and_nonunit_missing_peel(self):
        line = '    %f, %t = transform.loop.peel %n : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        with self.assertRaises(ValueError):
            run.output_tile_consumption(transform(5, 1).replace("    transform.yield", line + "    transform.yield"), schedule(5, 1), (5, 1))
        with self.assertRaises(ValueError):
            run.output_tile_consumption("\n".join(line for line in transform().splitlines() if "transform.loop.peel" not in line), schedule(), (5, 3))

    def test_transform_enclosure_and_yield_order(self):
        text = transform()
        peel = next(line for line in text.splitlines() if "transform.loop.peel" in line)
        for bad in (text.replace(peel + "\n", "") + peel + "\n",
                    text.replace(peel + "\n", "").replace("    transform.yield", "    transform.yield\n" + peel)):
            with self.assertRaises(ValueError):
                run.output_tile_consumption(bad, schedule(), (5, 3))

    def test_counts_preserving_split_and_loop_attacks(self):
        t, s = transform(), schedule()
        for label, bad in (
            ("wrong-split-divisor", s.replace("floordiv 3", "floordiv 5")),
            ("wrong-split-product", s.replace("* 3)", "* 5)")),
            ("split-from-M", s.replace(">()[%N]", ">()[%M]")),
            ("wrong-N-axis", s.replace("%N = memref.dim %b, %one", "%N = memref.dim %b, %zero")),
            ("gap", s.replace("%j = %F to %N", "%j = %one to %N")),
            ("overlap", s.replace("%j = %F to %N", "%j = %zero to %N")),
            ("wrong-tail-step", s.replace("%j = %F to %N step %tn", "%j = %F to %N step %tm")),
            ("wrong-M-step", s.replace("%i = %zero to %M step %tm", "%i = %zero to %M step %tn")),
            ("wrong-M-extent", s.replace("%i = %zero to %M", "%i = %zero to %N")),
            ("wrong-unit-start", s.replace("%j = %zero to %F", "%j = %one to %F")),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                run.output_tile_consumption(t, bad, (5, 3))

    def test_slices_actually_consume_full_tail_bounds(self):
        t, s = transform(), schedule()
        for bad in (s.replace("[%K, 3]", "[%K, 2]"), s.replace("[%height, 3]", "[%height, 2]"),
                    s.replace("(-d0 + s0)>", "(-d0 + s0 + 1)>"),
                    s.replace("(-d0 + s0, 5)", "(-d0 + s0, 4)"),
                    s.replace("subview %b[0, %j]", "subview %b[0, %i]"),
                    s.replace("[%height, %K]", "[%height, %N]"),
                    s.replace("ins(%av, %bv :", "ins(%bv, %av :"),
                    s.replace("outs(%cv :", "outs(%av :")):
            with self.subTest(schedule=bad), self.assertRaises(ValueError):
                run.output_tile_consumption(t, bad, (5, 3))

    def test_scoped_ssa_and_forward_references(self):
        t, s = transform(), schedule()
        # Both sibling domains reuse %height/%av/%bv/%cv legally. A missing
        # tail-only definition cannot borrow one from its full-N sibling.
        self.assertEqual(run.output_tile_consumption(t, s, (5, 3))["outer_loop_steps"], [5, 3, 3])
        start = s.index("      scf.for %j = %F")
        bad = s[:start] + s[start:].replace("        %remaining = affine.apply affine_map<(d0)[s0] -> (-d0 + s0)>(%j)[%N]\n", "")
        with self.assertRaises(ValueError):
            run.output_tile_consumption(t, bad, (5, 3))
        late = s.replace("    %tn = arith.constant 3 : index\n", "").replace("    return %c", "    %tn = arith.constant 3 : index\n    return %c")
        with self.assertRaises(ValueError):
            run.output_tile_consumption(t, late, (5, 3))
        views = [line for line in s.splitlines(keepends=True) if "memref.subview" in line][:3]
        late_views = s
        for line in views:
            late_views = late_views.replace(line, "", 1)
        late_views = late_views.replace("        }\n", "        }\n" + "".join(views), 1)
        with self.assertRaises(ValueError):
            run.output_tile_consumption(t, late_views, (5, 3))

    def test_comments_do_not_supply_or_duplicate_operations(self):
        t, s = transform(), schedule()
        self.assertEqual(run.output_tile_consumption("// " + t.splitlines()[-4] + "\n" + t,
                                                    "// scf.for %fake = %zero to %N step %tn {\n" + s, (5, 3))["n_domains"], "disjoint-full-tail")
        with self.assertRaises(ValueError):
            run.output_tile_consumption(t, "\n".join("// " + line for line in s.splitlines()), (5, 3))

    def test_exact_end_k_receipt(self):
        text = "Source peel end-K oracle: 792 cases; 10001 checks; 0 failures; full=728 tail1=364 tail2=364; inapplicable-K=40\n"
        self.assertEqual(run.end_k_summary(text)["cases"], 792)
        for bad in (text.replace("792", "791"), text.replace("0 failures", "1 failures"), text.replace("tail2=364", "tail2=0"),
                    text.replace("inapplicable-K=40", "inapplicable-K=0"), text + "SKIP unavailable\n"):
            with self.assertRaises(ValueError):
                run.end_k_summary(bad)


if __name__ == "__main__":
    unittest.main()
