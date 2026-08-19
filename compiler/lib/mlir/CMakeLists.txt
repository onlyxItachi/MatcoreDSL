set(LLVM_TARGET_DEFINITIONS MatcoreDialect.td)
mlir_tablegen(MatcoreDialect.h.inc -gen-dialect-decls -dialect=mdsl
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
mlir_tablegen(MatcoreDialect.cpp.inc -gen-dialect-defs -dialect=mdsl
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
add_public_tablegen_target(MatcoreDialectIncGen)

set(LLVM_TARGET_DEFINITIONS MatcoreOps.td)
mlir_tablegen(MatcoreOps.h.inc -gen-op-decls
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
mlir_tablegen(MatcoreOps.cpp.inc -gen-op-defs
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
add_public_tablegen_target(MatcoreOpsIncGen)

add_library(
  matcore_mlir_semantics
  STATIC
    MatcoreDialect.cpp
    MatcoreGemmSemanticBuilder.cpp
    MatcoreOps.cpp
    MatcoreV1Bridge.cpp
    MatcoreCpuRuntimeLowering.cpp
    MatcoreStaticGemmLowering.cpp
)
add_dependencies(
  matcore_mlir_semantics
  MatcoreDialectIncGen
  MatcoreOpsIncGen
)
target_compile_features(matcore_mlir_semantics PUBLIC cxx_std_20)
target_include_directories(
  matcore_mlir_semantics
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
target_include_directories(
  matcore_mlir_semantics
  SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
    "$<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>"
  PRIVATE
    "${LLVM_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
)
target_link_libraries(
  matcore_mlir_semantics
  PUBLIC
    matcore_ir_v1
  PRIVATE
    MLIRIR
    MLIRFuncDialect
    MLIRDestinationStyleOpInterface
    MLIRSideEffectInterfaces
)
mdslc_target_enable_warnings(matcore_mlir_semantics)

if(MDSLC_ENABLE_NATIVE_FRONTEND AND NOT WIN32)
  # Recovery analysis is kept in a separate internal library so the general
  # semantic dialect and explicit-v1 bridge never depend on Clang frontend
  # implementation details.
  add_library(
    matcore_mlir_recovered_analysis
    STATIC
      MatcoreRecoveredGemmBridge.cpp
  )
  target_compile_features(matcore_mlir_recovered_analysis PUBLIC cxx_std_20)
  target_include_directories(
    matcore_mlir_recovered_analysis
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  )
  target_include_directories(
    matcore_mlir_recovered_analysis
    SYSTEM PRIVATE
      "${LLVM_INCLUDE_DIRS}"
      "${MLIR_INCLUDE_DIRS}"
  )
  target_link_libraries(
    matcore_mlir_recovered_analysis
    PUBLIC
      matcore_mlir_semantics
      matcore_frontend_native
  )
  mdslc_target_enable_warnings(matcore_mlir_recovered_analysis)
endif()

if(BUILD_TESTING)
  add_executable(
    matcore_mlir_semantics_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_mlir_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_semantics_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
      MLIRTransforms
  )
  target_include_directories(
    matcore_mlir_semantics_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_semantics_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_semantics_tests)
  add_test(NAME mlir.semantic.core COMMAND matcore_mlir_semantics_tests)

  add_executable(
    matcore_mlir_map_domain_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_map_domain_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_map_domain_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_map_domain_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_map_domain_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_map_domain_tests)
  add_test(NAME mlir.semantic.map-domain COMMAND matcore_mlir_map_domain_tests)

  add_executable(
    matcore_mlir_cpu_runtime_lowering_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_cpu_runtime_lowering_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_cpu_runtime_lowering_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_cpu_runtime_lowering_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_cpu_runtime_lowering_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_cpu_runtime_lowering_tests)
  add_test(NAME mlir.cpu.runtime_dispatch_lowering_v1
    COMMAND matcore_mlir_cpu_runtime_lowering_tests
  )

  add_executable(
    matcore_mlir_static_gemm_lowering_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_static_gemm_lowering_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_static_gemm_lowering_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_static_gemm_lowering_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_static_gemm_lowering_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_static_gemm_lowering_tests)
  add_test(NAME mlir.cpu.static_specialization_lowering_v1
    COMMAND matcore_mlir_static_gemm_lowering_tests
  )

  if(MDSLC_ENABLE_NATIVE_FRONTEND AND NOT WIN32)
    execute_process(
      COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
      OUTPUT_VARIABLE MDSLC_RECOVERED_TEST_RESOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE MDSLC_RECOVERED_TEST_RESOURCE_STATUS
    )
    if(NOT MDSLC_RECOVERED_TEST_RESOURCE_STATUS EQUAL 0 OR
       NOT IS_DIRECTORY "${MDSLC_RECOVERED_TEST_RESOURCE_DIR}")
      message(FATAL_ERROR
        "recovered MLIR tests require the configured Clang resource directory")
    endif()
    add_executable(
      matcore_mlir_recovered_gemm_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_recovered_gemm_bridge_test.cpp"
    )
    target_link_libraries(
      matcore_mlir_recovered_gemm_tests
      PRIVATE
        matcore_mlir_recovered_analysis
    )
    target_include_directories(
      matcore_mlir_recovered_gemm_tests
      SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
    )
    target_compile_definitions(
      matcore_mlir_recovered_gemm_tests
      PRIVATE
        MDSLC_RECOVERED_TEST_CLANG="${MDSLC_CLANGXX_EXECUTABLE}"
        MDSLC_RECOVERED_TEST_RESOURCE_DIR="${MDSLC_RECOVERED_TEST_RESOURCE_DIR}"
        MDSLC_RECOVERED_TEST_PUBLIC_HEADER="${CMAKE_CURRENT_SOURCE_DIR}/../../include/matcore/mdsl.h"
        MDSLC_RECOVERED_TEST_FRONTEND_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/frontend/recovered"
        MDSLC_RECOVERED_TEST_IR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
    )
    mdslc_target_enable_warnings(matcore_mlir_recovered_gemm_tests)
    add_test(NAME mlir.semantic.recovered-gemm
      COMMAND matcore_mlir_recovered_gemm_tests)
  endif()
endif()
