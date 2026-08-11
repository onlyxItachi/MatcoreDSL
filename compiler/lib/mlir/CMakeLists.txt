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
    MatcoreOps.cpp
    MatcoreV1Bridge.cpp
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
endif()
