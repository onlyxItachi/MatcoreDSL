# The same closed IR pattern is instantiated on demand by the production issuer.
# These are qualification points, not a precomputed catalogue or tuning policy.
add_executable(matcore_output_pattern_test output_pattern_test.cpp)
target_link_libraries(matcore_output_pattern_test PRIVATE matcore_cpu_gemm_candidate
  MLIRParser MLIRLinalgDialect MLIRSCFDialect LLVM)
mdslc_target_match_llvm_rtti(matcore_output_pattern_test)
add_test(NAME generated_cpu.output_pattern.issuer COMMAND matcore_output_pattern_test)
add_test(NAME generated_cpu.output_pattern.cli COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-cpu-gemm-candidate>"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/output_pattern_cli.cmake")

set(patterns baseline_3_5 baseline_4_16 baseline_1_1 baseline_64_64)
if(MDSLC_CLOSED_CPU_X86)
  list(APPEND patterns avx2_4_8 avx512f_4_16)
endif()
foreach(pattern IN LISTS patterns)
  string(REPLACE "_" ";" fields "${pattern}")
  list(GET fields 0 isa)
  list(GET fields 1 tile_m)
  list(GET fields 2 tile_n)
  foreach(mode IN ITEMS normal asan)
    set(issuer_args "--output-tiles=${tile_m},${tile_n}" "--isa=${isa}")
    set(flags -O3 -ffp-contract=off)
    if(NOT MDSLC_CLOSED_CPU_X86)
      list(APPEND issuer_args --target=linux-aarch64)
      list(APPEND flags --target=aarch64-unknown-linux-gnu -march=armv8-a)
    endif()
    if(mode STREQUAL "asan")
      list(APPEND issuer_args --asan)
      list(APPEND flags -g -fsanitize=address)
    endif()
    set(ir "${CMAKE_CURRENT_BINARY_DIR}/pattern-${pattern}-${mode}.ll")
    set(object "${CMAKE_CURRENT_BINARY_DIR}/pattern-${pattern}-${mode}.o")
    add_custom_command(OUTPUT "${ir}"
      BYPRODUCTS "${ir}.manifest" "${ir}.semantic.mlir" "${ir}.structured.mlir"
        "${ir}.bufferized.mlir" "${ir}.scheduled.mlir" "${ir}.transform.mlir"
      COMMAND matcore-cpu-gemm-candidate --output "${ir}" ${issuer_args}
      DEPENDS matcore-cpu-gemm-candidate VERBATIM)
    add_custom_command(OUTPUT "${object}"
      COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -c -x ir "${ir}" ${flags} -o "${object}"
      DEPENDS "${ir}" VERBATIM)
    set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    set(target "matcore_output_pattern_${pattern}_${mode}_test")
    add_executable(${target} output_pattern_execution_test.cpp "${object}"
      ../../lib/platform/closed_cpu_isa_v1.cpp)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -ffp-contract=off -frounding-math)
    target_compile_definitions(${target} PRIVATE
      "MDSLC_PATTERN_TILE_M=${tile_m}" "MDSLC_PATTERN_TILE_N=${tile_n}")
    if(NOT isa STREQUAL "baseline")
      set(isa_index 2)
      if(isa STREQUAL "avx512f")
        set(isa_index 3)
      endif()
      target_compile_definitions(${target} PRIVATE "MDSLC_TEST_STRICT_ISA=${isa_index}"
        "MDSLC_TEST_STRICT_SYMBOL=_mlir_ciface___matcore_strict_gemm_f32_${isa}_v1")
    endif()
    if(mode STREQUAL "asan")
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    add_test(NAME generated_cpu.output_pattern.${pattern}.${mode} COMMAND ${target})
    set_tests_properties(generated_cpu.output_pattern.${pattern}.${mode}
      PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME generated_cpu.output_pattern.${pattern}.${mode}.corruption
      COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLE=$<TARGET_FILE:${target}>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_output_pattern_corruption.cmake")
    set_tests_properties(generated_cpu.output_pattern.${pattern}.${mode}.corruption
      PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP strict ISA")
    if(mode STREQUAL "asan")
      set(symbol __matcore_strict_gemm_f32_v1)
      if(NOT isa STREQUAL "baseline")
        set(symbol "__matcore_strict_gemm_f32_${isa}_v1")
      endif()
      add_test(NAME generated_cpu.output_pattern.${pattern}.asan_oob
        COMMAND "${CMAKE_COMMAND}"
          "-DEXECUTABLE=$<TARGET_FILE:${target}>" "-DSYMBOL=${symbol}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
      set_tests_properties(generated_cpu.output_pattern.${pattern}.asan_oob
        PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP strict ISA")
    endif()
  endforeach()
endforeach()
