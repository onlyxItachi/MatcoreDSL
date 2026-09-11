# Closed strict variants are isolated compiler-issued objects. The issuer owns
# their LLVM target attributes; no target flags leak into compiler/runtime TUs.
if(MDSLC_CLOSED_CPU_X86)
  find_program(MDSLC_STRICT_ISA_NM NAMES llvm-nm HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_program(MDSLC_STRICT_ISA_OBJDUMP NAMES llvm-objdump HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
  foreach(isa IN ITEMS avx avx2 avx512f)
    foreach(mode IN LISTS region_leaf_modes)
      set(issuer_args --schedule=row-contiguous "--isa=${isa}")
      set(kernel_flags -O3 -ffp-contract=off)
      set(sanitized OFF)
      if(mode STREQUAL "asan" OR MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
        list(APPEND issuer_args --asan)
        list(APPEND kernel_flags -g -fsanitize=address)
        set(sanitized ON)
      endif()
      set(ir "${CMAKE_CURRENT_BINARY_DIR}/strict-${isa}-${mode}.ll")
      set(object "${CMAKE_CURRENT_BINARY_DIR}/strict-${isa}-${mode}.o")
      add_custom_command(OUTPUT "${ir}"
        BYPRODUCTS "${ir}.manifest" "${ir}.semantic.mlir" "${ir}.structured.mlir"
          "${ir}.bufferized.mlir" "${ir}.scheduled.mlir" "${ir}.transform.mlir"
        COMMAND matcore-cpu-gemm-candidate --output "${ir}" ${issuer_args}
        DEPENDS matcore-cpu-gemm-candidate VERBATIM)
      add_custom_command(OUTPUT "${object}"
        COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -c -x ir "${ir}" ${kernel_flags} -o "${object}"
        COMMAND "${CMAKE_COMMAND}" "-DNM=${MDSLC_STRICT_ISA_NM}" "-DOBJDUMP=${MDSLC_STRICT_ISA_OBJDUMP}"
          "-DOBJECT=${object}" "-DIR=${ir}" "-DISA=${isa}" "-DSANITIZED=${sanitized}"
          -P "${CMAKE_CURRENT_LIST_DIR}/strict_isa_object.cmake"
        DEPENDS "${ir}" "${CMAKE_CURRENT_LIST_DIR}/strict_isa_object.cmake" VERBATIM)
      set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
      add_custom_target(matcore_generated_strict_${isa}_${mode}_object DEPENDS "${object}")
      string(TOUPPER "${isa}_${mode}" suffix)
      set(MDSLC_CLOSED_STRICT_${suffix}_OBJECT "${object}")
      set(MDSLC_CLOSED_STRICT_${suffix}_OBJECT "${object}" PARENT_SCOPE)
    endforeach()
    add_dependencies(matcore_closed_candidates_leaf matcore_generated_strict_${isa}_normal_object)
  endforeach()
endif()
