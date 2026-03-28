cmake_minimum_required(VERSION 3.24)

set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE FILEPATH "C++ compiler" FORCE)

project(MatCore LANGUAGES C CXX)

if(NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake" AND
   EXISTS "/usr/lib/llvm-18/lib/cmake/mlir/MLIRConfig.cmake")
  set(MLIR_DIR "/usr/lib/llvm-18/lib/cmake/mlir" CACHE PATH "MLIR CMake package" FORCE)
endif()

if(NOT EXISTS "${nanobind_DIR}/nanobind-config.cmake" AND
   EXISTS "/usr/share/nanobind/cmake/nanobind-config.cmake")
  set(nanobind_DIR "/usr/share/nanobind/cmake" CACHE PATH "nanobind CMake package" FORCE)
endif()

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(WARNING "MatCore prefers Ninja; configure with: cmake -B build -G Ninja")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

find_package(MLIR 18.1.3 REQUIRED CONFIG)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED CONFIG)

include_directories(SYSTEM
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
)
add_definitions(${LLVM_DEFINITIONS})

set(MATCORE_NATIVE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_lowering.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_mapping.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_nvvm_lowering.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_runtime_symbols.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_tiling.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/lowering_pipeline.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mlir_engine.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/jit_runner.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_capabilities.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/target_registry.cpp"
)

nanobind_add_module(_matcore_native ${MATCORE_NATIVE_SOURCES})

target_include_directories(_matcore_native
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if(EXISTS "/usr/local/cuda/include")
  target_include_directories(_matcore_native PRIVATE "/usr/local/cuda/include")
endif()

target_link_libraries(_matcore_native
  PRIVATE
    ${CMAKE_DL_LIBS}
    MLIR
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRBuiltinToLLVMIRTranslation
    MLIRControlFlowDialect
    MLIRControlFlowToLLVM
    MLIRExecutionEngine
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRGPUDialect
    MLIRGPUPipelines
    MLIRGPUToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    MLIRGPUTransformOps
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRLinalgDialect
    MLIRLinalgTransformOps
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRNVGPUDialect
    MLIRNVGPUTransformOps
    MLIRNVGPUToNVVM
    MLIRNVGPUTransforms
    MLIRNVVMDialect
    MLIRNVVMTarget
    MLIRNVVMToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRParser
    MLIRPass
    MLIRROCDLDialect
    MLIRROCDLTarget
    MLIRROCDLToLLVMIRTranslation
    MLIRSCFDialect
    MLIRSCFToGPU
    MLIRSCFToControlFlow
    MLIRTargetLLVMIRExport
    MLIRTransformDialect
    MLIRTransformDebugExtension
    MLIRTransformDialectTransforms
    MLIRTransformDialectUtils
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorToLLVMPass
    MLIRX86VectorDialect
    MLIRX86VectorToLLVMIRTranslation
    MLIRX86VectorTransforms
)

set(MATCORE_PY_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/matcore")
file(MAKE_DIRECTORY "${MATCORE_PY_OUTPUT_DIR}")

set_target_properties(_matcore_native PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${MATCORE_PY_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${MATCORE_PY_OUTPUT_DIR}"
)

foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${cfg}" cfg_upper)
  set_target_properties(_matcore_native PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${cfg_upper}" "${MATCORE_PY_OUTPUT_DIR}"
    "RUNTIME_OUTPUT_DIRECTORY_${cfg_upper}" "${MATCORE_PY_OUTPUT_DIR}"
  )
endforeach()
