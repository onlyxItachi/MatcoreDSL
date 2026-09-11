# Optional correctness-first, forced, synchronous staged GPU candidates.
# No hardware discovery changes the default CPU/provider policy.
function(mdslc_add_closed_gpu_runtime target)
  if(NOT TARGET matcore_gpu_gemm_candidate OR
     (NOT MDSLC_ENABLE_EXPERIMENTAL_NVVM AND NOT MDSLC_ENABLE_EXPERIMENTAL_ROCDL))
    return()
  endif()
  foreach(property IN ITEMS SOURCES LIBRARIES INCLUDES DEFINITIONS)
    get_target_property(gpu_${property} matcore_gpu_gemm_candidate MDSLC_GPU_${property})
  endforeach()
  target_sources(${target} PRIVATE ${gpu_SOURCES})
  target_link_libraries(${target} PRIVATE ${gpu_LIBRARIES} Threads::Threads)
  target_include_directories(${target} PRIVATE ${gpu_INCLUDES})
  target_compile_definitions(${target} PRIVATE ${gpu_DEFINITIONS})
  add_dependencies(${target} matcore_closed_gpu_leaf)
endfunction()

if(NOT MDSLC_ENABLE_EXPERIMENTAL_GPU_ISSUER AND
   NOT MDSLC_ENABLE_EXPERIMENTAL_NVVM AND NOT MDSLC_ENABLE_EXPERIMENTAL_ROCDL)
  return()
endif()
if(NOT MDSLC_ENABLE_EXPERIMENTAL_REGIONS OR
   NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" OR
   NOT MDSLC_REQUIRED_TOOLCHAIN_VERSION STREQUAL "21.1.8")
  message(FATAL_ERROR "Staged GPU candidates require the native Linux x86-64 experimental-region 21.1.8 tuple")
endif()

find_package(Threads REQUIRED)
find_program(MDSLC_GPU_LLC NAMES llc HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND "${MDSLC_GPU_LLC}" --version
  OUTPUT_VARIABLE gpu_llc_version COMMAND_ERROR_IS_FATAL ANY)
if(NOT gpu_llc_version MATCHES "LLVM version 21\\.1\\.8")
  message(FATAL_ERROR "GPU machine lowering requires LLVM 21.1.8")
endif()
add_library(matcore_gpu_gemm_candidate STATIC ../mlir/MatcoreGpuGemmCandidate.cpp)
target_compile_features(matcore_gpu_gemm_candidate PUBLIC cxx_std_20)
target_include_directories(matcore_gpu_gemm_candidate PUBLIC ../mlir)
target_include_directories(matcore_gpu_gemm_candidate SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(matcore_gpu_gemm_candidate PUBLIC matcore_cpu_gemm_candidate
  PRIVATE MLIRGPUTransforms MLIRSCFToGPU MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms MLIRNVVMToLLVMIRTranslation MLIRROCDLToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation MLIRTransforms MLIRTargetLLVMIRExport LLVM)
mdslc_target_enable_warnings(matcore_gpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_gpu_gemm_candidate)
add_executable(matcore-gpu-gemm-candidate ../../tools/matcore-gpu-gemm-candidate/main.cpp)
target_link_libraries(matcore-gpu-gemm-candidate PRIVATE matcore_gpu_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore-gpu-gemm-candidate)
add_custom_target(matcore_closed_gpu_leaf)
set(gpu_sources)
set(gpu_libraries)
set(gpu_definitions)
set(gpu_includes "${region_compiler_dir}/lib/runtime")

foreach(kind IN ITEMS nvvm rocdl)
  string(TOUPPER "${kind}" upper)
  if(NOT MDSLC_ENABLE_EXPERIMENTAL_${upper})
    continue()
  endif()
  if(kind STREQUAL "nvvm")
    set(target_name nvvm-sm89)
    find_path(MDSLC_CUDA_INCLUDE cuda.h HINTS /usr/local/cuda/include REQUIRED)
    file(STRINGS "${MDSLC_CUDA_INCLUDE}/cuda.h" gpu_cuda_version
      REGEX "^#define CUDA_VERSION[ \t]+13030$")
    if(NOT gpu_cuda_version)
      message(FATAL_ERROR "Initial CUDA driver adapter requires CUDA 13.3 headers")
    endif()
    find_library(MDSLC_CUDA_DRIVER NAMES cuda
      PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib REQUIRED)
    file(REAL_PATH "${MDSLC_CUDA_DRIVER}" MDSLC_CUDA_DRIVER_IDENTITY)
    if(MDSLC_CUDA_DRIVER_IDENTITY MATCHES "[/]stubs[/]" OR
       NOT MDSLC_CUDA_DRIVER_IDENTITY MATCHES "[/]libcuda\\.so\\.[0-9.]+$")
      message(FATAL_ERROR "NVVM requires the actual versioned NVIDIA driver DSO, never CUDA stubs")
    endif()
    find_program(MDSLC_PTXAS NAMES ptxas HINTS /usr/local/cuda/bin REQUIRED)
    execute_process(COMMAND "${MDSLC_PTXAS}" --version
      OUTPUT_VARIABLE gpu_ptxas_version COMMAND_ERROR_IS_FATAL ANY)
    if(NOT gpu_ptxas_version MATCHES "V13\\.3\\.73")
      message(FATAL_ERROR "Initial sm89 recipe is qualified only with ptxas 13.3.73")
    endif()
    list(APPEND gpu_sources "${region_compiler_dir}/lib/runtime/closed_cuda_candidate_v1.cpp")
    list(APPEND gpu_libraries "${MDSLC_CUDA_DRIVER_IDENTITY}")
    list(APPEND gpu_includes "${MDSLC_CUDA_INCLUDE}")
    list(APPEND gpu_definitions MDSLC_CLOSED_HOST_GENERATED_NVVM)
    set(MDSLC_CUDA_DRIVER_IDENTITY "${MDSLC_CUDA_DRIVER_IDENTITY}" PARENT_SCOPE)
  else()
    set(target_name rocdl-gfx1150)
    find_path(MDSLC_HIP_INCLUDE hip/hip_runtime_api.h HINTS /opt/rocm/include REQUIRED)
    find_library(MDSLC_HIP_RUNTIME NAMES amdhip64 HINTS /opt/rocm/lib REQUIRED)
    file(REAL_PATH "${MDSLC_HIP_RUNTIME}" MDSLC_HIP_RUNTIME_IDENTITY)
    if(NOT MDSLC_HIP_RUNTIME_IDENTITY MATCHES "[/]libamdhip64\\.so\\.7\\.2\\.70201$")
      message(FATAL_ERROR "Initial gfx1150 adapter is qualified only with the actual HIP 7.2.70201 DSO")
    endif()
    find_program(MDSLC_GPU_LLD NAMES ld.lld HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
    execute_process(COMMAND "${MDSLC_GPU_LLD}" --version
      OUTPUT_VARIABLE gpu_lld_version COMMAND_ERROR_IS_FATAL ANY)
    if(NOT gpu_lld_version MATCHES "LLD 21\\.1\\.8")
      message(FATAL_ERROR "ROCDL code-object linking requires LLD 21.1.8")
    endif()
    list(APPEND gpu_sources "${region_compiler_dir}/lib/runtime/closed_rocdl_candidate_v1.cpp")
    list(APPEND gpu_libraries "${MDSLC_HIP_RUNTIME_IDENTITY}")
    list(APPEND gpu_includes "${MDSLC_HIP_INCLUDE}")
    list(APPEND gpu_definitions MDSLC_CLOSED_HOST_GENERATED_ROCDL __HIP_PLATFORM_AMD__)
    set(MDSLC_HIP_RUNTIME_IDENTITY "${MDSLC_HIP_RUNTIME_IDENTITY}" PARENT_SCOPE)
  endif()
  set(prefix "${CMAKE_CURRENT_BINARY_DIR}/${kind}-strict")
  add_custom_command(OUTPUT "${prefix}.fill.ll" "${prefix}.gemm.ll"
    BYPRODUCTS "${prefix}.semantic.mlir" "${prefix}.structured.mlir"
      "${prefix}.bufferized.mlir" "${prefix}.outlined.mlir" "${prefix}.manifest"
    COMMAND matcore-gpu-gemm-candidate --output-prefix "${prefix}" --target "${target_name}"
    DEPENDS matcore-gpu-gemm-candidate VERBATIM)
  foreach(stage IN ITEMS fill gemm)
    if(kind STREQUAL "nvvm")
      set(image "${prefix}.${stage}.cubin")
      add_custom_command(OUTPUT "${image}"
        BYPRODUCTS "${prefix}.${stage}.ptx"
        COMMAND "${MDSLC_GPU_LLC}" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89
          -mattr=+ptx80 -fp-contract=off "${prefix}.${stage}.ll" -o "${prefix}.${stage}.ptx"
        COMMAND "${MDSLC_PTXAS}" --gpu-name sm_89 --fmad=false "${prefix}.${stage}.ptx" -o "${image}"
        DEPENDS "${prefix}.${stage}.ll" "${MDSLC_GPU_LLC}" "${MDSLC_PTXAS}" VERBATIM)
    else()
      set(image "${prefix}.${stage}.hsaco")
      add_custom_command(OUTPUT "${image}"
        BYPRODUCTS "${prefix}.${stage}.o"
        COMMAND "${MDSLC_GPU_LLC}" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150
          -fp-contract=off -denormal-fp-math=ieee -denormal-fp-math-f32=ieee
          -filetype=obj "${prefix}.${stage}.ll" -o "${prefix}.${stage}.o"
        COMMAND "${MDSLC_GPU_LLD}" -shared "${prefix}.${stage}.o" -o "${image}"
        DEPENDS "${prefix}.${stage}.ll" "${MDSLC_GPU_LLC}" "${MDSLC_GPU_LLD}" VERBATIM)
    endif()
    set(${stage}_image "${image}")
  endforeach()
  set(embedded "${prefix}.images.cpp")
  set_property(TARGET matcore_gpu_gemm_candidate PROPERTY MDSLC_${upper}_EMBEDDED_SOURCE "${embedded}")
  add_custom_command(OUTPUT "${embedded}"
    BYPRODUCTS "${embedded}.manifest"
    COMMAND "${CMAKE_COMMAND}" "-DFILL_IMAGE=${fill_image}" "-DGEMM_IMAGE=${gemm_image}"
      "-DTARGET_KIND=${kind}" "-DOUTPUT=${embedded}"
      -P "${CMAKE_CURRENT_LIST_DIR}/embed_gpu_images.cmake"
    DEPENDS "${fill_image}" "${gemm_image}" "${CMAKE_CURRENT_LIST_DIR}/embed_gpu_images.cmake" VERBATIM)
  add_custom_target(matcore_${kind}_images DEPENDS "${embedded}")
  add_dependencies(matcore_closed_gpu_leaf matcore_${kind}_images)
  list(APPEND gpu_sources "${embedded}")
endforeach()
set_target_properties(matcore_gpu_gemm_candidate PROPERTIES
  MDSLC_GPU_SOURCES "${gpu_sources}" MDSLC_GPU_LIBRARIES "${gpu_libraries}"
  MDSLC_GPU_INCLUDES "${gpu_includes}" MDSLC_GPU_DEFINITIONS "${gpu_definitions}")
