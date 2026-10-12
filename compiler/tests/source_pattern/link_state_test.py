#!/usr/bin/env python3
"""ELF link-state mechanism controls, not source or target execution evidence."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--linker", required=True)
    args = parser.parse_args()
    environment = dict(os.environ, LC_ALL="C")
    for name in ("COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_RUN_PATH",
                 "LDEMULATION", "GNUTARGET", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"):
        environment.pop(name, None)

    def run(argv):
        result = subprocess.run([str(item) for item in argv], env=environment,
                                text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError("link-state command failed: " + repr(argv) + "\n" + result.stderr)
        return result.stdout

    with tempfile.TemporaryDirectory(prefix="mdslc-link-state-") as temporary:
        root = Path(temporary)
        runtime = root / "runtime.so"
        base = [args.cxx, "--no-default-config", "--ld-path=" + args.linker,
                "-shared", "-x", "c++", "/dev/null", "-x", "none"]
        run([*base, "-Xlinker", "-soname", "-Xlinker", "libmdslc-link-state-probe.so", "-o", runtime])
        variants = {
            "stock": [runtime],
            "old_global_reset": ["-Xlinker", "--no-as-needed", runtime, "-Xlinker", "--as-needed"],
            "fixed": ["-Xlinker", "--push-state", "-Xlinker", "--no-as-needed", runtime,
                      "-Xlinker", "--pop-state"],
            "fixed_incoming_as_needed": ["-Xlinker", "--as-needed", "-Xlinker", "--push-state",
                                         "-Xlinker", "--no-as-needed", runtime, "-Xlinker", "--pop-state"],
        }
        needed = {}
        for name, flags in variants.items():
            output = root / (name + ".so")
            run([*base, *flags, "-lm", "-pthread", "-o", output])
            names = re.findall(r"\(NEEDED\).*?Shared library:\s*\[([^\]]+)\]",
                               run(["readelf", "-d", output]))
            if len(names) != len(set(names)):
                raise RuntimeError("duplicate ELF dependency in mechanism control")
            needed[name] = set(names)
        owner = "libmdslc-link-state-probe.so"
        if owner not in needed["stock"] or "libm.so.6" not in needed["stock"]:
            raise RuntimeError("stock mechanism control did not establish unused libm retention")
        if needed["fixed"] != needed["stock"]:
            raise RuntimeError("scoped Runtime retention changed the original trailing link policy")
        if needed["old_global_reset"] != {owner} or needed["old_global_reset"] == needed["stock"]:
            raise RuntimeError("old global as-needed leak did not falsify the empty-DSO control")
        if needed["fixed_incoming_as_needed"] != {owner}:
            raise RuntimeError("scoped Runtime retention failed to restore incoming as-needed policy")
        for name, names in needed.items():
            print(name + " DT_NEEDED=" + repr(sorted(names)))
        print("PASS ELF link-state controls: stock/fixed equal; old reset differs; incoming policy restored")


if __name__ == "__main__":
    main()
