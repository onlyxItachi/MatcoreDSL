add_executable(matcore_strict_isa_contract_test strict_isa_contract_test.cpp)
target_link_libraries(matcore_strict_isa_contract_test PRIVATE matcore_cpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_strict_isa_contract_test)
add_test(NAME generated_cpu.strict_isa.issuer COMMAND matcore_strict_isa_contract_test)
add_test(NAME generated_cpu.strict_isa.cli COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-cpu-gemm-candidate>" -P "${CMAKE_CURRENT_SOURCE_DIR}/strict_isa_cli_contract.cmake")
add_executable(matcore_strict_isa_facts_test strict_isa_facts_test.cpp ../../lib/platform/closed_cpu_isa_v1.cpp)
target_compile_features(matcore_strict_isa_facts_test PRIVATE cxx_std_20)
add_test(NAME generated_cpu.strict_isa.facts COMMAND matcore_strict_isa_facts_test)
if(MDSLC_CLOSED_CPU_X86)
  string(REGEX REPLACE "\\.o$" ".ll" avx512_ir "${MDSLC_CLOSED_STRICT_AVX512F_NORMAL_OBJECT}")
  add_test(NAME generated_cpu.strict_isa.object_fma_negative COMMAND "${CMAKE_COMMAND}"
    "-DORIGINAL_IR=${avx512_ir}" "-DCLANG=${MDSLC_CLANGXX_EXECUTABLE}"
    "-DNM=${MDSLC_STRICT_ISA_NM}" "-DOBJDUMP=${MDSLC_STRICT_ISA_OBJDUMP}"
    "-DCHECKER=${CMAKE_CURRENT_SOURCE_DIR}/../../lib/regions/strict_isa_object.cmake"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/strict_isa_object_negative.cmake")
  set(isa_index 0)
  foreach(isa IN ITEMS avx avx2 avx512f)
    math(EXPR isa_index "${isa_index}+1")
    foreach(mode IN ITEMS normal asan)
      string(TOUPPER "${isa}_${mode}" suffix)
      set(object "${MDSLC_CLOSED_STRICT_${suffix}_OBJECT}")
      set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
      foreach(oracle IN ITEMS execution independent_execution input_alias_execution)
        set(target matcore_strict_${isa}_${mode}_${oracle}_test)
        add_executable(${target} "${oracle}_test.cpp" "${object}" ../../lib/platform/closed_cpu_isa_v1.cpp)
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_compile_options(${target} PRIVATE -ffp-contract=off -frounding-math)
        target_compile_definitions(${target} PRIVATE "MDSLC_TEST_STRICT_ISA=${isa_index}"
          "MDSLC_TEST_STRICT_SYMBOL=_mlir_ciface___matcore_strict_gemm_f32_${isa}_v1")
        add_dependencies(${target} matcore_generated_strict_${isa}_${mode}_object)
        if(mode STREQUAL "asan")
          target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
          target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
        add_test(NAME generated_cpu.strict_isa.${isa}.${mode}.${oracle} COMMAND ${target})
        set_tests_properties(generated_cpu.strict_isa.${isa}.${mode}.${oracle} PROPERTIES SKIP_RETURN_CODE 77)
      endforeach()
      add_test(NAME generated_cpu.strict_isa.${isa}.${mode}.corruption COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLE=$<TARGET_FILE:matcore_strict_${isa}_${mode}_input_alias_execution_test>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_input_alias_corruption.cmake")
      set_tests_properties(generated_cpu.strict_isa.${isa}.${mode}.corruption PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP strict ISA")
    endforeach()
    foreach(fault IN ITEMS scalar simd)
      set(extra)
      if(fault STREQUAL "simd")
        set(bytes 32)
        if(isa STREQUAL "avx512f")
          set(bytes 64)
        endif()
        set(extra -DMODE=simd "-DWIDTH_BYTES=${bytes}")
      endif()
      add_test(NAME generated_cpu.strict_isa.${isa}.asan.${fault}_fault COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLE=$<TARGET_FILE:matcore_strict_${isa}_asan_execution_test>"
        "-DSYMBOL=__matcore_strict_gemm_f32_${isa}_v1" ${extra}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_asan.cmake")
      set_tests_properties(generated_cpu.strict_isa.${isa}.asan.${fault}_fault PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP strict ISA")
    endforeach()
    foreach(case IN ITEMS source same_value)
      add_test(NAME generated_cpu.strict_isa.${isa}.source.${case} COMMAND "${CMAKE_COMMAND}"
        "-DDRIVER=$<TARGET_FILE:mdslc-region>" "-DCASE=${case}" "-DPOLICY=generated-strict-${isa}"
        "-DPROBE=$<TARGET_FILE:matcore_strict_${isa}_normal_execution_test>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/schedule_source_contract.cmake")
      set_tests_properties(generated_cpu.strict_isa.${isa}.source.${case} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP strict ISA")
    endforeach()
    add_test(NAME generated_cpu.strict_isa.${isa}.source.ownership COMMAND "${CMAKE_COMMAND}"
      "-DDRIVER=$<TARGET_FILE:mdslc-region>" "-DISA=${isa}"
      "-DPROBE=$<TARGET_FILE:matcore_strict_${isa}_normal_execution_test>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/strict_isa_ownership.cmake")
    set_tests_properties(generated_cpu.strict_isa.${isa}.source.ownership PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP strict ISA")
  endforeach()
endif()
