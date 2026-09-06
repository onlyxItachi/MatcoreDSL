# Shared production definitions. Tests consume these targets; BUILD_TESTING is
# not an execution capability switch. Installation is separately opt-in.
get_filename_component(region_compiler_dir "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_library(matcore_closed_region_semantics STATIC ../mlir/MatcoreClosedRegion.cpp)
target_compile_features(matcore_closed_region_semantics PUBLIC cxx_std_20)
target_include_directories(matcore_closed_region_semantics PUBLIC ../mlir)
target_link_libraries(matcore_closed_region_semantics PUBLIC
  matcore_mlir_semantics PRIVATE MLIRParser MLIRArithDialect MLIRFuncDialect)
mdslc_target_enable_warnings(matcore_closed_region_semantics)
mdslc_target_match_llvm_rtti(matcore_closed_region_semantics)

add_library(matcore_closed_region_admission STATIC
  ../frontend/ClosedRegionAdmission.cpp
  ../frontend/ClosedRegionHostAdmission.cpp
  ../frontend/ClosedRegionHostInputs.cpp)
set(region_include "${region_compiler_dir}/include/matcore")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${region_include}/region.h" "${region_include}/detail/region_storage.h")
file(READ "${region_include}/region.h" MDSLC_EXPERIMENTAL_REGION_SOURCE)
file(READ "${region_include}/detail/region_storage.h" MDSLC_EXPERIMENTAL_REGION_STORAGE_SOURCE)
configure_file(../frontend/ExperimentalRegionHeaders.inc.in
  "${CMAKE_CURRENT_BINARY_DIR}/ExperimentalRegionHeaders.inc" @ONLY)
target_compile_features(matcore_closed_region_admission PUBLIC cxx_std_20)
target_include_directories(matcore_closed_region_admission PUBLIC ../frontend
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(matcore_closed_region_admission SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
  OUTPUT_VARIABLE MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR
  "${MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR}" PARENT_SCOPE)
set(region_clang_runtime "")
if(TARGET clang-cpp)
  set(region_clang_runtime "$<TARGET_FILE:clang-cpp>")
endif()
target_compile_definitions(matcore_closed_region_admission PRIVATE
  ${LLVM_DEFINITIONS}
  "MDSLC_HOST_CLANG_EXECUTABLE=\"${MDSLC_CLANGXX_EXECUTABLE}\""
  "MDSLC_HOST_CLANG_RESOURCE_DIRECTORY=\"${MDSLC_CLOSED_REGION_HOST_RESOURCE_DIR}\""
  "MDSLC_HOST_TOOLCHAIN_VERSION=\"${MDSLC_REQUIRED_TOOLCHAIN_VERSION}\""
  "MDSLC_HOST_CLANG_CPP_RUNTIME=\"${region_clang_runtime}\"")
target_link_libraries(matcore_closed_region_admission PUBLIC
  matcore_closed_region_semantics PRIVATE ${MDSLC_CLANG_TOOLING_LIBRARIES}
  matcore_frontend_runtime_identity MatcoreDSL::PlatformSupportV1)
mdslc_target_enable_warnings(matcore_closed_region_admission)
mdslc_target_match_llvm_rtti(matcore_closed_region_admission)

add_library(matcore_closed_host_emitter STATIC ../codegen/ClosedHostEmitter.cpp)
target_compile_features(matcore_closed_host_emitter PUBLIC cxx_std_20)
target_include_directories(matcore_closed_host_emitter PUBLIC ../codegen)
target_link_libraries(matcore_closed_host_emitter PUBLIC matcore_closed_region_admission)
mdslc_target_enable_warnings(matcore_closed_host_emitter)
mdslc_target_match_llvm_rtti(matcore_closed_host_emitter)

# Existing compatibility experiments may build the semantic inspection surface
# on other tuples. This executable issuer is certified only on this native tuple.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT MDSLC_REQUIRED_TOOLCHAIN_VERSION STREQUAL "21.1.8" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  return()
endif()
add_library(matcore_cpu_gemm_candidate STATIC ../mlir/MatcoreCpuGemmCandidate.cpp)
target_include_directories(matcore_cpu_gemm_candidate PUBLIC ../mlir)
target_include_directories(matcore_cpu_gemm_candidate SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(matcore_cpu_gemm_candidate PUBLIC matcore_closed_region_semantics
  PRIVATE matcore_mlir_bufferized_handoff MLIRLinalgTransforms MLIRArithToLLVM
    MLIRAffineToStandard MLIRSCFToControlFlow MLIRMemRefToLLVM MLIRFuncToLLVM
    MLIRControlFlowToLLVM MLIRReconcileUnrealizedCasts MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport)
mdslc_target_enable_warnings(matcore_cpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_cpu_gemm_candidate)
add_executable(matcore-cpu-gemm-candidate
  ../../tools/matcore-cpu-gemm-candidate/main.cpp)
target_link_libraries(matcore-cpu-gemm-candidate PRIVATE matcore_cpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore-cpu-gemm-candidate)

set(region_leaf_modes normal)
if(BUILD_TESTING)
  list(APPEND region_leaf_modes asan)
endif()
foreach(mode IN LISTS region_leaf_modes)
  set(issuer_args)
  set(kernel_flags -O2)
  # A sanitized compiler build also instruments its production leaf; linking a
  # sanitized wrapper alone is not instrumentation of the emitted computation.
  if(mode STREQUAL "asan" OR MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
    list(APPEND issuer_args --asan)
    list(APPEND kernel_flags -fsanitize=address)
  endif()
  set(ir "${CMAKE_CURRENT_BINARY_DIR}/strict-${mode}.ll")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/strict-${mode}.o")
  add_custom_command(OUTPUT "${ir}"
    BYPRODUCTS "${ir}.manifest" "${ir}.semantic.mlir" "${ir}.structured.mlir" "${ir}.bufferized.mlir"
    COMMAND matcore-cpu-gemm-candidate --output "${ir}" ${issuer_args}
    DEPENDS matcore-cpu-gemm-candidate VERBATIM)
  add_custom_command(OUTPUT "${object}"
    COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -c -x ir "${ir}" ${kernel_flags} -o "${object}"
    DEPENDS "${ir}" VERBATIM)
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  # Keep the rule owned here even when its only object consumers live in another
  # directory (notably the always-ASan negative-control executables).
  add_custom_target(matcore_generated_cpu_${mode}_object DEPENDS "${object}")
  string(TOUPPER "${mode}" mode_upper)
  set(MDSLC_CLOSED_GEMM_${mode_upper}_OBJECT "${object}" PARENT_SCOPE)
endforeach()
set(MDSLC_CLOSED_GEMM_NORMAL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/strict-normal.o")
add_custom_target(matcore_closed_candidates_leaf)
add_dependencies(matcore_closed_candidates_leaf matcore_generated_cpu_normal_object)

get_target_property(backend_definitions matcore_cpu_backends_v1 INTERFACE_COMPILE_DEFINITIONS)
set(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS OFF)
if("MATCORE_MDSLC_HAS_OPENBLAS=1" IN_LIST backend_definitions)
  set(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS ON)
endif()
set(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS "${MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS}" PARENT_SCOPE)

# Test variants use exactly this definition, with injection enabled only in the
# dedicated archive. The default invocation produces the installed candidate set.
function(mdslc_add_closed_candidate_runtime target)
  cmake_parse_arguments(REGION "TESTING;NATIVE_ONLY" "" "" ${ARGN})
  if(REGION_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown closed candidate runtime option: ${REGION_UNPARSED_ARGUMENTS}")
  endif()
  set(compiler_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../..")
  add_library(${target} STATIC "${compiler_dir}/lib/runtime/closed_host_v1.cpp")
  target_include_directories(${target} PUBLIC
    "${compiler_dir}/lib/runtime" "${compiler_dir}/include")
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_compile_options(${target} PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
  target_link_libraries(${target} PRIVATE m)
  mdslc_target_enable_warnings(${target})
  if(NOT REGION_NATIVE_ONLY)
    target_compile_definitions(${target} PRIVATE
      MDSLC_CLOSED_HOST_GENERATED_STRICT MDSLC_CLOSED_HOST_LEGACY_CANDIDATES)
    if(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS)
      target_compile_definitions(${target} PRIVATE MDSLC_CLOSED_HOST_OPENBLAS)
    endif()
    set_source_files_properties("${MDSLC_CLOSED_GEMM_NORMAL_OBJECT}" PROPERTIES
      GENERATED TRUE EXTERNAL_OBJECT TRUE)
    target_sources(${target} PRIVATE "${MDSLC_CLOSED_GEMM_NORMAL_OBJECT}")
    add_dependencies(${target} matcore_closed_candidates_leaf)
    # The shared legacy runtime remains the sole OpenBLAS policy-state owner.
    # Never embed the private CPU backend archive into this archive.
    target_link_libraries(${target} PRIVATE MatcoreDSL::Runtime)
  endif()
  if(REGION_TESTING)
    target_compile_definitions(${target} PUBLIC MDSLC_CLOSED_HOST_TESTING)
  endif()
endfunction()
mdslc_add_closed_candidate_runtime(matcore_closed_candidates_production_v1)

if(MDSLC_ENABLE_EXPERIMENTAL_REGIONS)
  install(TARGETS matcore_closed_candidates_production_v1
    ARCHIVE DESTINATION "${MDSLC_EXPERIMENTAL_REGION_PRIVATE_LIBDIR}")
  install(FILES "${region_compiler_dir}/lib/runtime/closed_host_v1.h"
    DESTINATION "${MDSLC_EXPERIMENTAL_REGION_PRIVATE_INCLUDEDIR}")
endif()
