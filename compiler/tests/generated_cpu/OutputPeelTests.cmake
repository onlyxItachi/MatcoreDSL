# Separate compiler-private realization; no source-driver/default selection.
add_executable(matcore_output_peel_test output_peel_test.cpp)
target_link_libraries(matcore_output_peel_test PRIVATE matcore_cpu_gemm_candidate
  MLIRParser MLIRLinalgDialect MLIRSCFDialect LLVM)
mdslc_target_match_llvm_rtti(matcore_output_peel_test)
add_test(NAME generated_cpu_output_peel.issuer COMMAND matcore_output_peel_test)
add_test(NAME generated_cpu_output_peel.cli COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-cpu-gemm-candidate>"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/output_peel_cli.cmake")

# The 5x3 case is the mechanism witness, not an optimal tile. Unit, rectangular,
# non-power-of-two and maximum tiles qualify the same bounded derivation.
foreach(pattern IN ITEMS 5_3 3_5 2_2 1_1 1_64 64_1 64_64)
  string(REPLACE "_" ";" fields "${pattern}")
  list(GET fields 0 tile_m)
  list(GET fields 1 tile_n)
  foreach(mode IN ITEMS normal asan)
    set(issuer_args "--output-tiles-peel-n=${tile_m},${tile_n}")
    set(flags -O2 -ffp-contract=off)
    if(NOT MDSLC_CLOSED_CPU_X86)
      list(APPEND issuer_args --target=linux-aarch64)
      list(APPEND flags --target=aarch64-unknown-linux-gnu -march=armv8-a)
    endif()
    if(mode STREQUAL "asan")
      list(APPEND issuer_args --asan)
      list(APPEND flags -g -fsanitize=address)
    endif()
    set(ir "${CMAKE_CURRENT_BINARY_DIR}/peel-${pattern}-${mode}.ll")
    set(object "${CMAKE_CURRENT_BINARY_DIR}/peel-${pattern}-${mode}.o")
    add_custom_command(OUTPUT "${ir}"
      BYPRODUCTS "${ir}.manifest" "${ir}.semantic.mlir" "${ir}.structured.mlir"
        "${ir}.bufferized.mlir" "${ir}.scheduled.mlir" "${ir}.transform.mlir"
      COMMAND matcore-cpu-gemm-candidate --output "${ir}" ${issuer_args}
      DEPENDS matcore-cpu-gemm-candidate VERBATIM)
    add_custom_command(OUTPUT "${object}"
      COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -c -x ir "${ir}" ${flags} -o "${object}"
      DEPENDS "${ir}" VERBATIM)
    set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
    set(target "matcore_output_peel_${pattern}_${mode}_test")
    add_executable(${target} output_peel_execution_test.cpp "${object}"
      ../../lib/platform/closed_cpu_isa_v1.cpp)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE -ffp-contract=off -frounding-math)
    target_compile_definitions(${target} PRIVATE
      "MDSLC_PATTERN_TILE_M=${tile_m}" "MDSLC_PATTERN_TILE_N=${tile_n}")
    if(mode STREQUAL "asan")
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    add_test(NAME generated_cpu_output_peel.${pattern}.${mode} COMMAND ${target})
    add_test(NAME generated_cpu_output_peel.${pattern}.${mode}.corruption
      COMMAND "${CMAKE_COMMAND}" "-DEXECUTABLE=$<TARGET_FILE:${target}>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_output_pattern_corruption.cmake")
    if(mode STREQUAL "asan")
      add_test(NAME generated_cpu_output_peel.${pattern}.asan_oob
        COMMAND "${CMAKE_COMMAND}"
          "-DEXECUTABLE=$<TARGET_FILE:${target}>"
          "-DSYMBOL=__matcore_strict_gemm_f32_v1"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
      add_test(NAME generated_cpu_output_peel.${pattern}.asan_full_oob
        COMMAND "${CMAKE_COMMAND}"
          "-DEXECUTABLE=$<TARGET_FILE:${target}>"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_output_peel_full_asan.cmake")
    endif()
  endforeach()
endforeach()
