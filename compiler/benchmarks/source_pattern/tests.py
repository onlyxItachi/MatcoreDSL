"""Synthetic receipts/command mocks only: no compiler, source executable or timing."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import contract as c
import run


def common():
    return {"schema": "matcore.source-pattern.p1", "version": 1, "numerics": "strict_f32",
            "timing_scope": c.SCOPE, "release_inside_timing": False, "chooser_inside_timing": True,
            "implementation_thread_contract": 1, "continuous_thread_proof": False,
            "seed": c.SEED, "input_sequence": c.INPUTS}


def controls(module):
    return {**common(), "mode": "verify-only", "status": "CORRECTNESS_PASSED", "controls": list(c.CONTROLS),
            "checks": 1000, "failures": 0, "loaded_modules": [str(module)]}


def measured(cell=0, cpu=2, module="/synthetic/p1"):
    m, k, n, repetitions = c.MATRIX[cell]
    sample = dict(calls=repetitions, verified_calls=repetitions, verified_elements=repetitions*m*n,
                  output_hash=c.expected_output_hash(cell), task_count_before=1, task_count_after=1,
                  cpu_before=cpu, cpu_after=cpu, affinity_before=[cpu], affinity_after=[cpu],
                  wall_ns=1_500_000, thread_cpu_ns=1_490_000, involuntary_switches=0, voluntary_switches=0,
                  rss_max_bytes_before=4096, rss_max_bytes_after=8192, timing_accepted=True)
    return {**common(), "mode": "measure", "status": "MEASURED", "cell": cell, "m": m, "k": k, "n": n,
            "repetitions": repetitions, "warmup_aggregates": c.WARMUPS, "timer_floor_ns": c.FLOOR_NS,
            "sizeof_result": 96, "live_payload_bound_bytes": c.payload_bound(cell,96),
            "retained_output_payload_bytes": repetitions*m*n*4,
            "checked_timed_calls": repetitions*c.SAMPLES, "checked_timed_elements": repetitions*c.SAMPLES*m*n,
            "checked_warmup_calls": repetitions*c.WARMUPS, "process_task_count_before": 1, "process_task_count_after": 1,
            "process_affinity_before": [cpu], "process_affinity_after": [cpu], "loaded_modules": [str(module)],
            "warmups": [{**sample, "ordinal": i} for i in range(c.WARMUPS)],
            "samples": [{**sample, "ordinal": i} for i in range(c.SAMPLES)]}


class ContractTests(unittest.TestCase):
    def test_frozen_matrix_lifetimes_and_global_reverse(self):
        self.assertEqual(c.MATRIX, ((32,32,32,128),(33,35,37,128),(64,7,19,256),
                                   (1,257,65,256),(65,257,1,256),(128,128,128,8)))
        plan = c.plan(); self.assertEqual(len(plan["order"]),36)
        forward=plan["order"][:18]; reverse=plan["order"][18:]
        self.assertEqual([(v["cell"],v["role"]) for v in reverse],list(reversed([(v["cell"],v["role"]) for v in forward])))
        self.assertFalse(plan["release_inside_timing"])
        self.assertTrue(plan["chooser_inside_timing"])
        self.assertEqual(plan["winner_policy"],"none")
        for cell in range(6): self.assertLess(c.payload_bound(cell,96),c.LIVE_PAYLOAD_CAP)

    def test_exact_integer_bits(self):
        self.assertEqual(c.integer_bits(0),0)
        self.assertEqual(c.integer_bits(1),0x3f800000)
        self.assertEqual(c.integer_bits(-123),0xc2f60000)
        with self.assertRaises(ValueError): c.integer_bits(1<<24)

    def test_receipt_positive_and_falsifiers(self):
        good=measured(); self.assertTrue(c.check_cell(good,0,0,2))
        mutations=(
            lambda r:r.update(repetitions=256), lambda r:r.update(release_inside_timing=True),
            lambda r:r.update(chooser_inside_timing=False), lambda r:r.update(timing_scope="compute-only"),
            lambda r:r.update(continuous_thread_proof=True), lambda r:r.update(checked_timed_calls=1),
            lambda r:r.update(retained_output_payload_bytes=0), lambda r:r.update(live_payload_bound_bytes=1),
            lambda r:r["samples"].pop(), lambda r:r["warmups"].pop(),
            lambda r:r["samples"][4].update(verified_calls=127),
            lambda r:r["samples"][4].update(output_hash=0),
            lambda r:r["samples"][0].update(wall_ns=0),
            lambda r:r["samples"][0].update(thread_cpu_ns=1.0),
            lambda r:r["samples"][0].update(affinity_after=[2,14]),
            lambda r:r["samples"][0].update(task_count_after=2),
            lambda r:r.update(process_task_count_before=True),
        )
        for mutate in mutations:
            bad=copy.deepcopy(good); mutate(bad)
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError): c.check_cell(bad,0,0,2)

    def test_subfloor_or_noise_is_inconclusive_not_retry(self):
        for changes in (dict(wall_ns=999999,thread_cpu_ns=990000),dict(involuntary_switches=1)):
            receipt=measured(); receipt["samples"][3].update(**changes,timing_accepted=False)
            receipt["status"]="INCONCLUSIVE_TIMING"
            self.assertFalse(c.check_cell(receipt,78,0,2))
            with self.assertRaises(ValueError): c.check_cell(receipt,0,0,2)

    def test_control_inventory_and_json(self):
        receipt=controls("/synthetic/p1"); c.check_controls(receipt,0)
        receipt["controls"]=receipt["controls"][:-1]
        with self.assertRaises(ValueError): c.check_controls(receipt,0)
        for text in ('{"a":1,"a":2}','{"a":NaN}','{} trailing','[]'):
            with self.assertRaises(ValueError): c.strict_json(text)
        with self.assertRaises(ValueError): c.exact([True],[1],"typed list")

    def test_forward_reverse_crossover_and_incomplete(self):
        cells=[{**entry,"timing_accepted":True,"normalized_ns":[100.0]*9} for entry in c.plan()["order"]]
        for item in cells:
            if item["role"]=="tiles-3x5": item["normalized_ns"]=[80.0 if item["direction"]=="forward" else 120.0]*9
        summary=c.summarize(cells)
        role=summary[0]["roles"]["tiles-3x5"]
        self.assertEqual(role["status"],"INCONCLUSIVE_SPREAD_OR_ORDER")
        self.assertEqual(role["direction_time_ratio_to_stock"],{"forward":.8,"reverse":1.2})
        self.assertIsNone(role["ratio_of_midpoint_medians_to_stock"])
        with self.assertRaises(ValueError): c.summarize(cells[:-1])


class RunnerTests(unittest.TestCase):
    def test_all_issued_stage_object_joins(self):
        manifest={key:"a"*64 for key in c.ISSUED_HASHES}
        run.join_issued(manifest,manifest)
        for key in c.ISSUED_HASHES:
            bad={**manifest,key:"b"*64}
            with self.subTest(key=key):
                with self.assertRaises(ValueError): run.join_issued(manifest,bad)
        with self.assertRaises(ValueError): run.join_issued({}, {})

    def test_environment_allowlist_and_loader_tunables(self):
        inherited={"LD_BIND_NOT":"1","LD_PROFILE":"private","GLIBC_TUNABLES":"private",
                   "MALLOC_ARENA_MAX":"128","ASAN_OPTIONS":"private","OMP_DYNAMIC":"true",
                   "OPENBLAS_CORETYPE":"unknown","P1_SYNTHETIC_KEEP":"preserved"}
        with patch.dict(os.environ,inherited,clear=True):
            env=run.environment()
        for key in inherited:
            if key!="P1_SYNTHETIC_KEEP": self.assertNotIn(key,env)
        self.assertEqual(env["P1_SYNTHETIC_KEEP"],"preserved")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"],"1")
        self.assertEqual(env["OMP_NUM_THREADS"],"1")

    def test_safe_command_mock_no_real_process_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="p1-synthetic-") as temporary:
            root=Path(temporary); log=run.Commands(root)
            def fake(argv,**options):
                options["stdout"].write(b'{"synthetic":true}\n')
                return subprocess.CompletedProcess(argv,0)
            with patch.object(run.subprocess,"run",side_effect=fake) as process:
                result=log.invoke(["/synthetic/compiler","--not-executed"],expected=0)
                self.assertEqual(result.stdout,'{"synthetic":true}\n')
                self.assertEqual(process.call_count,1)
            path=root/"receipt.json"; run.write_json(path,{"original":True})
            with self.assertRaises(FileExistsError): run.write_json(path,{"replacement":True})
            self.assertTrue(run.read_json(path)["original"])

    def test_deadline_before_process_or_endpoint_cannot_pass(self):
        with tempfile.TemporaryDirectory(prefix="p1-deadline-synthetic-") as temporary:
            log=run.Commands(Path(temporary)); log.deadline=180
            with patch.object(run.time,"monotonic",return_value=181),patch.object(run.subprocess,"run") as process:
                with self.assertRaises(ValueError): log.invoke(["/never-executed"])
                with self.assertRaises(ValueError): log.unchanged()
                process.assert_not_called()

    def test_raw_directory_git_guard_no_global_changes(self):
        with tempfile.TemporaryDirectory(prefix="p1-path-synthetic-") as temporary:
            output=Path(temporary)/"must-not-exist"
            with patch.object(run.subprocess,"run",return_value=subprocess.CompletedProcess([],0,"true\n","")):
                with self.assertRaises(ValueError): run.new_external_directory(output)
            self.assertFalse(output.exists())

    def test_loaded_dependency_parser_and_retarget(self):
        with tempfile.TemporaryDirectory(prefix="p1-modules-synthetic-") as temporary:
            path=Path(temporary)/"library with space.so"; path.write_bytes(b"synthetic")
            parsed=run.loaded_paths(f'linux-vdso.so.1 (0x1234)\nlibtest.so => {path} (0x1234)\n')
            self.assertEqual(parsed,[str(path)])
            run.modules_match({"loaded_modules":[str(path)]},parsed)
            with self.assertRaises(ValueError): run.modules_match({"loaded_modules":[str(path)]},[])
            with self.assertRaises(ValueError): run.loaded_paths("libtest.so => not found\n")

    def test_upstream_rpath_transform_mock_preserves_inputs_and_rejects_other_changes(self):
        # Toy bytes plus a mocked CMake command, NOT actual ELF transformation.
        for fault in ("none","bytes","wrong-root","wrong-head","foreign-compile-input"):
            with tempfile.TemporaryDirectory(prefix="p1-stock-synthetic-") as temporary:
                root=Path(temporary); output=root/"capsule"; output.mkdir()
                build=root/"build"; (build/"lib/regions/mdslc-region").mkdir(parents=True)
                (build/"CMakeFiles").mkdir(); (build/"bin").mkdir()
                source_root=root/"source"; source=source_root/"compiler"
                (source/"lib/regions").mkdir(parents=True); (source/"tools/mdslc-region").mkdir(parents=True)
                producer=build/"lib/libmatcore_closed_candidates_isolated_v1.so"; installed=root/"installed.so"
                producer_bytes=b"toy-RPATH=build-code"+(b"-corrupt" if fault=="bytes" else b"")
                producer.write_bytes(producer_bytes); installed.write_bytes(b"toy-RPATH=empty-code")
                cache=build/"CMakeCache.txt"
                cache.write_text("MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE:STRING=scalar\nCMAKE_BUILD_TYPE:STRING=Release\n"
                                 f"CMAKE_CACHEFILE_DIR:INTERNAL={build if fault!='wrong-root' else root/'foreign-build'}\nCMAKE_HOME_DIRECTORY:INTERNAL={source}\nCMAKE_CXX_COMPILER:FILEPATH=/synthetic/compiler\n")
                leaf=build/"lib/regions/strict-normal.ll"; leaf.write_text("synthetic nonexecuted leaf")
                manifest={"schema":"matcore-builtin-strict-cpu-gemm-v1","toolchain":"21.1.8","target":"x86_64-pc-linux-gnu",
                          "profile":"strict_f32","address_sanitizer":"off","schedule":"scalar-mnk"}
                for key,suffix in (("llvm_sha256",""),("semantic_sha256",".semantic.mlir"),("structured_sha256",".structured.mlir"),
                                   ("bufferized_sha256",".bufferized.mlir"),("scheduled_sha256",".scheduled.mlir"),("transform_sha256",".transform.mlir")):
                    path=Path(str(leaf)+suffix)
                    if suffix:path.write_text("synthetic "+key)
                    manifest[key]=run.file_identity(path)["sha256"]
                Path(str(leaf)+".manifest").write_text("".join(key+"="+value+"\n" for key,value in manifest.items()))
                obj=build/"lib/regions/strict-normal.o"; obj.write_bytes(b"not an object; never executed")
                ninja=build/"build.ninja"
                export=source/"lib/regions/private_runtime_v2.exports"; export.write_text("synthetic exports")
                script=source/"tools/mdslc-region/installation_identity.cmake"; script.write_text("synthetic existing install script")
                (build/"CMakeFiles/rules.ninja").write_text("synthetic rules")
                issuer=build/"bin/matcore-cpu-gemm-candidate"; issuer.write_bytes(b"not executed")
                (build/"lib/regions/mdslc-region/region_driver_identity.h").write_text(
                    '#define REGION_CANDIDATES_SHA "'+run.file_identity(producer)["sha256"]+'"\n'
                    '#define REGION_INSTALLED_CANDIDATES_SHA "'+run.file_identity(installed)["sha256"]+'"\n')
                compile_input=leaf if fault!="foreign-compile-input" else root/"foreign-build/lib/regions/strict-normal.ll"
                ninja.write_text(f"build lib/libmatcore_closed_candidates_isolated_v1.so: LINK lib/regions/strict-normal.o\n  LINK_FLAGS = -Bsymbolic --version-script={export}\n"
                                 f"build lib/regions/strict-normal.o: CUSTOM_COMMAND lib/regions/strict-normal.ll\n  COMMAND = cd {leaf.parent} && /synthetic/compiler -c -x ir {compile_input} -O2 -o {obj}\n"
                                 f"build lib/regions/strict-normal.ll: CUSTOM_COMMAND\n  COMMAND = cd {leaf.parent} && {issuer} --output {leaf}\n")
                cmake=root/"synthetic-cmake"; cmake.write_bytes(b"never executed")
                readelf=root/"synthetic-readelf"; readelf.write_bytes(b"never executed")
                git=root/"synthetic-git"; git.write_bytes(b"never executed")
                profile=root/"stock-profile.json"
                run.write_json(profile,dict(schema="matcore.source-pattern.p1.stock-profile",version=1,source_commit=("b" if fault=="wrong-head" else "a")*40,
                                           producer_candidate=str(producer),cache=str(cache),leaf=str(leaf),object=str(obj),build_ninja=str(ninja)))
                log=run.Commands(output)
                def fake(argv,**options):
                    if argv[0]==str(cmake):
                        path=output/"inspection/stock-rpath-removed.so"
                        path.write_bytes(path.read_bytes().replace(b"RPATH=build",b"RPATH=empty"))
                    elif argv[0]==str(git):
                        if argv[-1]=="--show-toplevel": options["stdout"].write((str(source_root)+"\n").encode())
                        elif argv[-1]=="HEAD": options["stdout"].write(("a"*40+"\n").encode())
                    return subprocess.CompletedProcess(argv,0)
                with patch.object(run.subprocess,"run",side_effect=fake):
                    if fault!="none":
                        with self.assertRaises(ValueError): run.stock_profile(profile,log,installed,{"cmake":str(cmake),"readelf":str(readelf),"git":str(git)})
                    else:
                        result=run.stock_profile(profile,log,installed,{"cmake":str(cmake),"readelf":str(readelf),"git":str(git)})
                        self.assertEqual(result["selected_schedule"],"scalar-mnk")
                        self.assertEqual(result["install_transform"]["derived_sha256"],run.file_identity(installed)["sha256"])
                self.assertEqual(producer.read_bytes(),producer_bytes)
                self.assertEqual(installed.read_bytes(),b"toy-RPATH=empty-code")

    def test_full_synthetic_collection_json_raw_audit_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="p1-audit-synthetic-") as temporary:
            root=Path(temporary); (root/"raw").mkdir()
            binary=root/"synthetic-never-executed"; binary.write_bytes(b"synthetic fixture, not ELF")
            prepared=root/"prepared.json"
            instances={role:{"timed":{"binary":str(binary),"loaded_modules":[str(binary)]}} for role in c.ROLES}
            run.write_json(prepared,{"instances":instances})
            report={"status":"COMPLETE","identity_checks_passed":True,"plan_sha256":c.digest(c.plan()),
                    "selected_cpu":2,"collection_wall_seconds":10.0,"prepared":str(prepared),
                    "prepared_identity":run.file_identity(prepared),"commands":[],"cells":[],"control_runs":[]}
            def record(argv,receipt):
                index=len(report["commands"]); directory=root/"raw"/f"{index:04d}"; directory.mkdir()
                command={"argv":list(map(str,argv)),"returncode":0,"directory":str(directory.relative_to(root))}
                report["commands"].append(command)
                (directory/"stdout.log").write_text(json.dumps(receipt))
                (directory/"stderr.log").write_text("")
                run.write_json(directory/"command.json",command)
                return index
            for phase in ("initial","final"):
                for role in c.ROLES:
                    receipt=controls(binary); index=record([binary,"--verify-only"],receipt)
                    report["control_runs"].append(dict(phase=phase,role=role,command_index=index,receipt=receipt))
                if phase=="final": break
                for entry in c.plan()["order"]:
                    receipt=measured(entry["cell"],module=binary)
                    index=record([binary,"--measure",entry["cell"],"--cpu",2],receipt)
                    report["cells"].append({**entry,"status":"MEASURED","timing_accepted":True,"receipt":receipt,
                                            "command_index":index,"normalized_ns":[s["wall_ns"]/receipt["repetitions"] for s in receipt["samples"]]})
            report["summary"]=c.summarize(report["cells"])
            report["raw_catalog"]={str(path.relative_to(root)):run.file_identity(path) for path in (root/"raw").rglob("*") if path.is_file()}
            collection=root/"collection.json"; run.write_json(collection,report)
            self.assertEqual(run.audit_collection(collection),report["summary"])
            for label,mutate in (
                ("normalized-only",lambda r:r["cells"][0].update(normalized_ns=[.001]*9)),
                ("catalog-hole",lambda r:r["raw_catalog"].pop(next(iter(r["raw_catalog"])))),
                ("late-deadline",lambda r:r.update(collection_wall_seconds=180.01)),
                ("missing-final-control",lambda r:r["control_runs"].pop()),
                ("swap-initial-final",lambda r:(r["control_runs"][0].update(command_index=39),r["control_runs"][3].update(command_index=0))),
                ("swap-forward-reverse",lambda r:(r["cells"][0].update(command_index=38),r["cells"][-1].update(command_index=3))),
            ):
                bad=copy.deepcopy(report); mutate(bad)
                path=root/(label+".json"); run.write_json(path,bad)
                with self.subTest(label=label):
                    with self.assertRaises(ValueError): run.audit_collection(path)


if __name__=="__main__": unittest.main()
