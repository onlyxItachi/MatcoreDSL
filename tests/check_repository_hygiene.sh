#!/usr/bin/env bash

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

declare -a violations=()

declare -a ignore_sentinels=(
  "build-hygiene/CMakeCache.txt"
  "build_debug/CMakeCache.txt"
  "build.review/build.ninja"
  "install-hygiene/lib/libmatcore_runtime.so.1"
  "install_debug/lib/libmatcore_runtime.so.1"
  "benchmark_reports/hygiene/test.log"
  ".matcore_cache-hygiene/key/metadata.json"
  "custom-cache-root/key/metadata.json"
  "memory/session.md"
  "compiler/CMakeFiles/hygiene"
  "CMakeCache.txt"
  "compiler/compile_commands.json"
  "compile_commands.json"
  "Testing/Temporary/LastTest.log"
  ".deps/hygiene"
  ".matcore_debug/hygiene"
  ".matcore_ir_dump/hygiene"
  "hygiene.host.cpp"
  "hygiene.host-overlay.yaml"
  "hygiene.matcore.json"
  "hygiene.sites.h"
  "hygiene.stubs.cpp"
  "hygiene.backend.cpp"
  ".pytest_cache/hygiene"
)

for path in "${ignore_sentinels[@]}"; do
  if ! git check-ignore --quiet --no-index "$path"; then
    violations+=("$path (required ignore policy is missing)")
  fi
done

# This catches force-added files covered by the repository's ignore policy.
while IFS= read -r path; do
  [[ -n "$path" ]] && violations+=("$path (tracked despite .gitignore)")
done < <(git ls-files --cached --ignored --exclude-standard)

# Keep an explicit denylist so removing an ignore rule cannot admit generated
# repository-local output silently.
while IFS= read -r path; do
  case "$path" in
    build/* | build[-_.]*/* | cmake-build[-_.]*/* | \
    install/* | install[-_.]*/* | \
    benchmark_reports/* | .matcore_cache*/* | custom-cache-root/* | \
    memory/* | CMakeFiles/* | */CMakeFiles/* | \
    CMakeCache.txt | */CMakeCache.txt | cmake_install.cmake | \
    */cmake_install.cmake | build.ninja | */build.ninja | \
    .ninja_deps | */.ninja_deps | .ninja_log | */.ninja_log | \
    compile_commands.json | */compile_commands.json | \
    CTestTestfile.cmake | */CTestTestfile.cmake | Testing/* | */Testing/* | \
    .deps/* | */.deps/* | .matcore_debug/* | */.matcore_debug/* | \
    .matcore_ir_dump/* | */.matcore_ir_dump/* | \
    *.o | *.a | *.so | *.so.* | *.dylib | *.dll | *.bin | *.cubin | \
    *.fatbin | *.ptx | *.pyc | *.pyo | *.log)
      violations+=("$path (forbidden generated path)")
      ;;
    *.host.cpp | *.host-overlay.yaml | *.matcore.json | *.sites.h | \
    *.stubs.cpp | *.backend.cpp | .pytest_cache/* | */.pytest_cache/* | \
    .mypy_cache/* | */.mypy_cache/* | .ruff_cache/* | */.ruff_cache/* | \
    .coverage | */.coverage | htmlcov/* | */htmlcov/*)
      violations+=("$path (forbidden generated path)")
      ;;
  esac
done < <(git ls-files)

if (( ${#violations[@]} != 0 )); then
  printf 'repository hygiene check failed; generated files are tracked:\n' >&2
  printf '  %s\n' "${violations[@]}" >&2
  printf 'remove them from the index and keep generated output in an ignored build directory\n' >&2
  exit 1
fi

echo "repository hygiene check passed"
