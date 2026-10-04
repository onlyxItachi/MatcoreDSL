cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED DRIVER)
  message(FATAL_ERROR "Missing strict fused pair driver")
endif()
if(NOT NM)
  find_program(NM NAMES llvm-nm-21 nm REQUIRED)
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${CMAKE_CURRENT_BINARY_DIR}/strict-fused-pair-${suffix}")
file(MAKE_DIRECTORY "${root}")
file(READ "${CMAKE_CURRENT_LIST_DIR}/strict_fused_pair.mdsl" source)
if(PROGRAM)
  string(REPLACE "int main()" "int math_main()" source "${source}")
  file(WRITE "${root}/main.cpp" "extern int math_main(); int main(){return math_main();}\n")
endif()
file(WRITE "${root}/source.mdsl" "${source}")
if(PROGRAM)
  set(invocation --program --host "${root}/main.cpp" --region "${root}/source.mdsl" strict_fused_pipeline)
else()
  set(invocation "${root}/source.mdsl" --region strict_fused_pipeline)
endif()
foreach(optimization IN ITEMS none strict-fused-pair)
  execute_process(COMMAND "${DRIVER}" ${invocation} --candidate generated-strict
    --optimization "${optimization}" -o "${root}/${optimization}"
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 120)
  if(NOT status EQUAL 0 OR NOT EXISTS "${root}/${optimization}")
    message(FATAL_ERROR "Strict pair ${optimization} compilation failed: ${stdout}\n${stderr}")
  endif()
  execute_process(COMMAND "${NM}" --undefined-only --demangle "${root}/${optimization}"
    RESULT_VARIABLE status OUTPUT_VARIABLE symbols ERROR_VARIABLE stderr)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Cannot inspect compiled strict pair connection: ${stderr}")
  endif()
  if(optimization STREQUAL "strict-fused-pair")
    if(NOT symbols MATCHES "closed_host_v1::SessionAbiV2::gemmStrictFusedPair")
      message(FATAL_ERROR "Optimization flag did not connect checked strict pair runtime")
    endif()
  elseif(symbols MATCHES "closed_host_v1::SessionAbiV2::gemmStrictFusedPair")
    message(FATAL_ERROR "Baseline unexpectedly invokes fused pair runtime")
  endif()
  # Huge legal empty dimensions must finish, not traverse INT64_MAX row panels.
  execute_process(COMMAND "${root}/${optimization}"
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 20)
  if(NOT status EQUAL 0 OR NOT stdout MATCHES
      "^Strict fused pair source: [1-9][0-9]* checks; 0 failures; 22 executed cases\n$")
    message(FATAL_ERROR "Strict pair ${optimization} exact source oracle failed: ${stdout}\n${stderr}")
  endif()
  if(optimization STREQUAL "none")
    set(baseline "${stdout}")
  else()
    set(optimized "${stdout}")
  endif()
  message(STATUS "strict pair ${optimization}: ${stdout}")
endforeach()
if(NOT baseline STREQUAL optimized)
  message(FATAL_ERROR "Same-source fused/unfused math/status/effect/FP outcomes differ")
endif()

# Neither automatic selection nor an unrelated forced route may silently
# dispatch this recipe. GPU combined source is qualified separately.
foreach(policy IN ITEMS automatic native-strict generated-strict-avx
    generated-strict-avx2 generated-strict-avx512f generated-reassociate
    existing-native openblas)
  execute_process(COMMAND "${DRIVER}" ${invocation} --candidate "${policy}"
    --optimization strict-fused-pair -o "${root}/incompatible-${policy}"
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 60)
  if(status EQUAL 0 OR EXISTS "${root}/incompatible-${policy}" OR
      NOT stderr MATCHES "strict-fused-pair requires explicit")
    message(FATAL_ERROR "Incompatible ${policy} acquired pair authority: ${stdout}\n${stderr}")
  endif()
endforeach()
execute_process(COMMAND "${DRIVER}" ${invocation} --optimization strict-fused-pair
  -o "${root}/automatic-default" RESULT_VARIABLE status OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr TIMEOUT 60)
if(status EQUAL 0 OR EXISTS "${root}/automatic-default" OR
    NOT stderr MATCHES "strict-fused-pair requires explicit")
  message(FATAL_ERROR "Default automatic policy dispatched explicit-only pair: ${stdout}\n${stderr}")
endif()

# The selected original Program still admits this dead-C source, but the opt-in
# pair must fail closed when its only consumer is no longer the exact lhs chain.
string(REPLACE "auto e = md::gemm(c, d, md::Numerics::strict_f32);"
  "auto e = md::gemm(a, b, md::Numerics::strict_f32);" no_pair "${source}")
file(WRITE "${root}/no-pair.mdsl" "${no_pair}")
execute_process(COMMAND "${DRIVER}" "${root}/no-pair.mdsl" --region strict_fused_pipeline
  --candidate generated-strict --optimization strict-fused-pair -o "${root}/no-pair"
  RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr TIMEOUT 60)
if(status EQUAL 0 OR EXISTS "${root}/no-pair" OR NOT stderr MATCHES "eligible adjacent")
  message(FATAL_ERROR "Zero-eligible pair request fell back: ${stdout}\n${stderr}")
endif()
execute_process(COMMAND "${DRIVER}" ${invocation} --candidate generated-strict
  --optimization none --optimization strict-fused-pair -o "${root}/duplicate"
  RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(status EQUAL 0 OR EXISTS "${root}/duplicate" OR NOT stderr MATCHES "duplicate --optimization")
  message(FATAL_ERROR "Duplicate optimization choice was accepted: ${stdout}\n${stderr}")
endif()
