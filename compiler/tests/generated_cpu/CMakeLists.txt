# Tests consume the production issuer and issued objects from lib/regions.
if(NOT TARGET matcore-cpu-gemm-candidate)
  return()
endif()

include(StrictCpuIsaTests.cmake)

if(TARGET mdslc-region)
  foreach(case IN ITEMS source same_value)
    add_test(NAME generated_cpu.source_schedule.${case} COMMAND "${CMAKE_COMMAND}"
      "-DDRIVER=$<TARGET_FILE:mdslc-region>" "-DCASE=${case}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/schedule_source_contract.cmake")
  endforeach()
  # Retained from the rejected cache schedule: these are source-level shape,
  # numerical and ordered-effect oracles, valid for every issued schedule.
  add_test(NAME generated_cpu.source_schedule.boundary_matrix COMMAND "${CMAKE_COMMAND}"
    "-DDRIVER=$<TARGET_FILE:mdslc-region>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cache_schedule_source_contract.cmake")
endif()
add_executable(matcore_cpu_gemm_candidate_test candidate_test.cpp)
# Match the integrated compiler's one shared LLVM registry, including its parser.
target_link_libraries(matcore_cpu_gemm_candidate_test PRIVATE matcore_cpu_gemm_candidate MLIRLinalgDialect LLVM)
mdslc_target_match_llvm_rtti(matcore_cpu_gemm_candidate_test)
add_test(NAME generated_cpu.strict_candidate COMMAND matcore_cpu_gemm_candidate_test)
add_executable(matcore_cpu_target_contract_test cpu_target_contract_test.cpp)
target_link_libraries(matcore_cpu_target_contract_test PRIVATE matcore_cpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_cpu_target_contract_test)
add_test(NAME generated_cpu.target_contract COMMAND matcore_cpu_target_contract_test)
add_test(NAME generated_cpu.target_cli_contract COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-cpu-gemm-candidate>"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/cpu_target_cli_contract.cmake")
if(MDSLC_CLOSED_CPU_X86)
add_executable(matcore_cpu_reassociate_candidate_test reassociate_candidate_test.cpp)
target_link_libraries(matcore_cpu_reassociate_candidate_test PRIVATE matcore_cpu_gemm_candidate
  MLIRLinalgDialect MLIRVectorDialect LLVM)
mdslc_target_match_llvm_rtti(matcore_cpu_reassociate_candidate_test)
add_test(NAME generated_cpu.reassociate_candidate COMMAND matcore_cpu_reassociate_candidate_test)
add_subdirectory(reassociate_independent)
add_test(NAME generated_cpu.reassociate_cli_contract COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-cpu-gemm-candidate>"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/reassociate_cli_contract.cmake")

foreach(mode IN ITEMS normal asan)
  string(TOUPPER "${mode}" mode_upper)
  set(object "${MDSLC_CLOSED_REASSOCIATE_${mode_upper}_OBJECT}")
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_executable(matcore_generated_reassociate_${mode}_test reassociate_execution_test.cpp "${object}")
  target_compile_features(matcore_generated_reassociate_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_reassociate_${mode}_test PRIVATE
    -ffp-contract=off -frounding-math -ftrapping-math)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_reassociate_${mode}_test PRIVATE -fsanitize=address,undefined)
    target_link_options(matcore_generated_reassociate_${mode}_test PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME generated_cpu.reassociate_execution.${mode} COMMAND matcore_generated_reassociate_${mode}_test)
  set_tests_properties(generated_cpu.reassociate_execution.${mode} PROPERTIES SKIP_RETURN_CODE 77)
  foreach(negative IN ITEMS corrupt strict)
    add_test(NAME generated_cpu.reassociate_${negative}.${mode} COMMAND bash
      "${CMAKE_CURRENT_SOURCE_DIR}/expect_reassociate.sh"
      "$<TARGET_FILE:matcore_generated_reassociate_${mode}_test>" "${negative}"
      "${CMAKE_CURRENT_BINARY_DIR}/reassociate-${negative}-${mode}")
    set_tests_properties(generated_cpu.reassociate_${negative}.${mode} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endforeach()
foreach(input IN ITEMS a b)
  add_test(NAME generated_cpu.reassociate_sanitizer_read_${input} COMMAND bash
    "${CMAKE_CURRENT_SOURCE_DIR}/expect_reassociate.sh"
    "$<TARGET_FILE:matcore_generated_reassociate_asan_test>" "read-${input}"
    "${CMAKE_CURRENT_BINARY_DIR}/reassociate-read-${input}")
  set_tests_properties(generated_cpu.reassociate_sanitizer_read_${input} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
endif()
foreach(mode IN ITEMS normal asan)
  string(TOUPPER "${mode}" mode_upper)
  set(object "${MDSLC_CLOSED_GEMM_${mode_upper}_OBJECT}")
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_executable(matcore_generated_cpu_${mode}_test execution_test.cpp "${object}")
  target_compile_features(matcore_generated_cpu_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_cpu_${mode}_test PRIVATE -ffp-contract=off)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_cpu_${mode}_test PRIVATE -fsanitize=address)
    target_link_options(matcore_generated_cpu_${mode}_test PRIVATE -fsanitize=address)
  endif()
  add_test(NAME generated_cpu.execution.${mode} COMMAND matcore_generated_cpu_${mode}_test)
  add_executable(matcore_generated_cpu_independent_${mode}_test independent_execution_test.cpp "${object}")
  target_compile_features(matcore_generated_cpu_independent_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_cpu_independent_${mode}_test PRIVATE -ffp-contract=off -frounding-math)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_cpu_independent_${mode}_test PRIVATE -fsanitize=address)
    target_link_options(matcore_generated_cpu_independent_${mode}_test PRIVATE -fsanitize=address)
  endif()
  add_test(NAME generated_cpu.independent_execution.${mode} COMMAND matcore_generated_cpu_independent_${mode}_test)
  add_executable(matcore_generated_cpu_input_alias_${mode}_test input_alias_execution_test.cpp "${object}")
  target_compile_features(matcore_generated_cpu_input_alias_${mode}_test PRIVATE cxx_std_20)
  target_compile_options(matcore_generated_cpu_input_alias_${mode}_test PRIVATE -ffp-contract=off -frounding-math)
  if(mode STREQUAL "asan")
    target_compile_options(matcore_generated_cpu_input_alias_${mode}_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(matcore_generated_cpu_input_alias_${mode}_test PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME generated_cpu.input_alias.${mode} COMMAND matcore_generated_cpu_input_alias_${mode}_test)
  add_test(NAME generated_cpu.input_alias_corruption.${mode} COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLE=$<TARGET_FILE:matcore_generated_cpu_input_alias_${mode}_test>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_input_alias_corruption.cmake")
endforeach()
find_program(MDSLC_CANDIDATE_NM NAMES llvm-nm HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(MDSLC_CANDIDATE_OBJDUMP NAMES llvm-objdump HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
if(MDSLC_CLOSED_CPU_X86)
foreach(mode IN ITEMS normal asan)
  string(TOUPPER "${mode}" mode_upper)
  set(sanitized OFF)
  if(mode STREQUAL "asan" OR MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
    set(sanitized ON)
  endif()
  add_test(NAME generated_cpu.reassociate_object.${mode} COMMAND "${CMAKE_COMMAND}"
    "-DNM=${MDSLC_CANDIDATE_NM}" "-DOBJDUMP=${MDSLC_CANDIDATE_OBJDUMP}"
    "-DOBJECT=${MDSLC_CLOSED_REASSOCIATE_${mode_upper}_OBJECT}"
    "-DSANITIZED=${sanitized}" -P "${CMAKE_CURRENT_SOURCE_DIR}/reassociate_object_contract.cmake")
endforeach()
endif()
add_test(NAME generated_cpu.object_contract COMMAND "${CMAKE_COMMAND}"
  "-DNM=${MDSLC_CANDIDATE_NM}" "-DOBJDUMP=${MDSLC_CANDIDATE_OBJDUMP}"
  "-DOBJECT=${MDSLC_CLOSED_GEMM_NORMAL_OBJECT}"
  "-DCPU_X86=${MDSLC_CLOSED_CPU_X86}"
  "-DSANITIZED=${MDSLC_CLOSED_REGION_CXX_FLAGS}" -P "${CMAKE_CURRENT_SOURCE_DIR}/object_contract.cmake")
add_test(NAME generated_cpu.sanitizer_negative_control COMMAND "${CMAKE_COMMAND}"
  "-DEXECUTABLE=$<TARGET_FILE:matcore_generated_cpu_asan_test>" -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
if(MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE STREQUAL "row-contiguous")
  add_test(NAME generated_cpu.simd_sanitizer_negative_control COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLE=$<TARGET_FILE:matcore_generated_cpu_asan_test>" "-DMODE=simd"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
endif()

# Instrumented builds need not retain the same auto-vectorization decision.
# Assert actual SIMD only on the unsanitized realization that claims it.
if(MDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE STREQUAL "row-contiguous" AND
   NOT MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=")
  string(REGEX REPLACE "\\.o$" ".ll.manifest" row_manifest
    "${MDSLC_CLOSED_GEMM_NORMAL_OBJECT}")
  add_test(NAME generated_cpu.row_contiguous_simd COMMAND "${CMAKE_COMMAND}"
    "-DOBJDUMP=${MDSLC_CANDIDATE_OBJDUMP}"
    "-DOBJECT=${MDSLC_CLOSED_GEMM_NORMAL_OBJECT}"
    "-DMANIFEST=${row_manifest}"
    "-DCPU_X86=${MDSLC_CLOSED_CPU_X86}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/row_contiguous_object.cmake")
endif()
