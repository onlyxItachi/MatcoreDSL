#!/usr/bin/env bash
set -euo pipefail
exe=$1 mode=$2 work=$3
mkdir -p -- "$work"
export DEBUGINFOD_URLS=
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:exitcode=1
export UBSAN_OPTIONS=halt_on_error=1
case "$mode" in
  corrupt) option=--corrupt-output ;;
  strict) option=--require-strict ;;
  read-a) option=--asan-invalid-capacity; width=4 ;;
  read-b) option=--asan-invalid-b-capacity; width=32 ;;
  *) exit 2 ;;
esac
set +e
"$exe" "$option" > "$work/stdout" 2> "$work/stderr"
status=$?
set -e
if [[ $status == 77 ]]; then exit 77; fi
if [[ $status != 1 ]]; then echo "Wrong negative-control outcome: $status" >&2; exit 1; fi
if [[ $mode == corrupt || $mode == strict ]]; then
  grep -q 'FAIL: chosen legal realization' "$work/stderr"
else
  grep -Eq '^==[0-9]+==ERROR: AddressSanitizer: heap-buffer-overflow on address ' "$work/stderr"
  grep -Eq "^READ of size $width at 0x[[:xdigit:]]+ thread T0$" "$work/stderr"
  grep -Eq '^    #0 0x[[:xdigit:]]+ in __matcore_reassociate_gemm_f32_avx2_v1([[:space:]]|$)' "$work/stderr"
fi
printf 'Issued reassociate negative %s: exact exit1 and required diagnostic\n' "$mode"
