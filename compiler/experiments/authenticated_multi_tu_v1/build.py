#!/usr/bin/env python3
"""Build only the isolated research executable against a recorded H1 build."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_root.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("output directory must be new")
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    repo = source.parents[2]
    lines = (build / "build.ninja").read_text().splitlines()
    def variables(prefix):
        start = next(i for i, row in enumerate(lines) if row.startswith(prefix))
        found = {}
        for row in lines[start + 1:]:
            if not row.startswith("  "):
                break
            key, value = row.strip().split(" = ", 1)
            found[key] = value
        return found
    compile_vars = variables("build lib/regions/mdslc-region/CMakeFiles/mdslc-region.dir/main.cpp.o:")
    link_vars = variables("build bin/mdslc-region:")
    sanitized = "-fsanitize=address,undefined" in shlex.split(compile_vars["FLAGS"])
    optimization = next((flag for flag in reversed(shlex.split(compile_vars["FLAGS"]))
                         if flag.startswith("-O")), "-O0") if sanitized else "-O0"
    sanitizer_flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if sanitized else []
    configured = (build / "lib/regions/mdslc-region/region_driver_paths.h").read_text()
    if ("#define REGION_SANITIZED 1" in configured) != sanitized:
        parser.error("configured compiler and source-driver sanitizer profiles differ")
    # Use one physical copy of pragma-once private headers: this worktree's
    # sources, plus only the configured build's generated headers/toolchain.
    includes = ["-I" + str(repo / "compiler/lib" / name)
                for name in ("codegen", "frontend", "mlir", "ir", "support")]
    includes += shlex.split(compile_vars["INCLUDES"])
    libraries = shlex.split(link_vars["LINK_LIBRARIES"])
    libraries = [str(build / item) if item.startswith("lib/") else item for item in libraries]
    pinned = {str(Path(item)): digest(item) for item in libraries
              if Path(item).is_file()}
    for name in ("region_driver_paths.h", "region_driver_identity.h"):
        path = build / "lib/regions/mdslc-region" / name
        pinned[str(path)] = digest(path)
    for path in (source / "driver.cpp", repo / "compiler/lib/frontend/ClosedRegionHostAdmission.cpp",
                 repo / "compiler/lib/codegen/AuthenticatedHostThunk.cpp",
                 repo / "compiler/tools/mdslc-region/main.cpp"):
        pinned[str(path)] = digest(path)
    for path in (repo / "compiler").rglob("*.h"):
        pinned[str(path)] = digest(path)
    pinned[str(source / "build.py")] = digest(source / "build.py")
    pinned["/usr/bin/clang++-21"] = digest("/usr/bin/clang++-21")
    command = ["/usr/bin/clang++-21", "-std=c++20", optimization, "-g", *sanitizer_flags, *includes,
               str(source / "driver.cpp"), *libraries, "-o", str(output / "multi-tu")]
    result = subprocess.run(command, text=True, capture_output=True)
    changed = [path for path, sha in pinned.items() if digest(path) != sha]
    record = {"base": "8fbdc61258173c46f199f9142092fc1bbec6cb04",
              "sanitizer_profile": "address,undefined" if sanitized else "none",
              "research_compiler_optimization": optimization,
              "original_h1_base": "486d3e5d74e17fbf305c73fca3e3760ef7da59b6",
              "linked_build_head": subprocess.check_output(["git", "-C", build.parent, "rev-parse", "HEAD"], text=True).strip(),
              "research_head": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
              "command": command, "exit": result.returncode, "stdout": result.stdout,
              "stderr": result.stderr, "linked_input_sha256": pinned,
              "changed_inputs": changed, "driver_source_sha256": pinned[str(source / "driver.cpp")]}
    if result.returncode == 0:
        record["executable_sha256"] = digest(output / "multi-tu")
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode or changed:
        raise SystemExit(result.returncode or 1)
    print("BUILT", output / "multi-tu", record["executable_sha256"])


if __name__ == "__main__":
    main()
