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
  # Bind the primary error, its access, and its FIRST stack frame. A generated
  # entry in a later allocation stack is not evidence of the fault owner.
  awk -v width="$width" '
    /^==[0-9]+==ERROR: AddressSanitizer:/ {
      if (phase != 0 || $0 !~ /^==[0-9]+==ERROR: AddressSanitizer: heap-buffer-overflow on address /)
        exit 1
      phase = 1
      next
    }
    phase == 1 && /^(READ|WRITE) of size / {
      if ($0 !~ ("^READ of size " width " at 0x[[:xdigit:]]+ thread T0$"))
        exit 1
      phase = 2
      next
    }
    phase != 0 && /^[[:space:]]+#[0-9]+ / {
      accepted = phase == 2 && $0 ~ /^    #0 0x[[:xdigit:]]+ in __matcore_reassociate_gemm_f32_avx2_v1([[:space:]]|$)/
      exit
    }
    END { exit !accepted }
  ' "$work/stderr"
fi
printf 'Issued reassociate negative %s: exact exit1 and required diagnostic\n' "$mode"
