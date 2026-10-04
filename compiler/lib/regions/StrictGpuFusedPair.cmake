# Closed compiler-private pair issuer. No runtime/source/default policy here.
if(NOT TARGET matcore_gpu_gemm_candidate OR NOT TARGET matcore_cpu_fused_gemm_candidate)
  return()
endif()
add_library(matcore_gpu_fused_gemm_candidate STATIC ../mlir/MatcoreGpuFusedGemmCandidate.cpp)
target_compile_features(matcore_gpu_fused_gemm_candidate PUBLIC cxx_std_20)
target_include_directories(matcore_gpu_fused_gemm_candidate PUBLIC ../mlir)
target_include_directories(matcore_gpu_fused_gemm_candidate SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(matcore_gpu_fused_gemm_candidate PUBLIC matcore_cpu_fused_gemm_candidate
  PRIVATE MLIRGPUTransforms MLIRGPUToNVVMTransforms MLIRGPUToROCDLTransforms
    MLIRNVVMToLLVMIRTranslation MLIRROCDLToLLVMIRTranslation MLIRGPUToLLVMIRTranslation
    MLIRTransforms MLIRParser MLIRTargetLLVMIRExport LLVM)
mdslc_target_enable_warnings(matcore_gpu_fused_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore_gpu_fused_gemm_candidate)
add_executable(matcore-gpu-fused-gemm-candidate ../../tools/matcore-gpu-fused-gemm-candidate/main.cpp)
target_link_libraries(matcore-gpu-fused-gemm-candidate PRIVATE matcore_gpu_fused_gemm_candidate)
mdslc_target_match_llvm_rtti(matcore-gpu-fused-gemm-candidate)
find_program(MDSLC_FUSED_PAIR_NM NAMES llvm-nm HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(MDSLC_FUSED_PAIR_OBJDUMP NAMES llvm-objdump HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(MDSLC_FUSED_PAIR_READELF NAMES llvm-readelf HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
set(gpu_fused_pair_definition_dir "${CMAKE_CURRENT_LIST_DIR}")
function(matcore_define_strict_gpu_fused_pair_images kind output_embedded)
  if(NOT kind MATCHES "^(nvvm|rocdl)$")
    message(FATAL_ERROR "Unknown strict GPU pair target")
  endif()
  string(TOUPPER "${kind}" upper)
  set(prefix "${CMAKE_CURRENT_BINARY_DIR}/${kind}-strict-fused-pair")
  if(kind STREQUAL "nvvm")
    set(target_name nvvm-sm89)
    set(image "${prefix}.pair.cubin")
    find_program(MDSLC_FUSED_PAIR_CUOBJDUMP NAMES cuobjdump HINTS /usr/local/cuda/bin REQUIRED)
  else()
    set(target_name rocdl-gfx1150)
    set(image "${prefix}.pair.hsaco")
  endif()
  add_custom_command(OUTPUT "${prefix}.pair.ll"
    BYPRODUCTS "${prefix}.semantic.mlir" "${prefix}.structured.mlir" "${prefix}.scheduled.mlir"
      "${prefix}.bufferized.mlir" "${prefix}.outlined.mlir" "${prefix}.manifest"
    COMMAND matcore-gpu-fused-gemm-candidate --output-prefix "${prefix}" --target "${target_name}"
    DEPENDS matcore-gpu-fused-gemm-candidate VERBATIM)
  if(kind STREQUAL "nvvm")
    add_custom_command(OUTPUT "${image}"
      BYPRODUCTS "${prefix}.pair.ptx"
      COMMAND "${MDSLC_GPU_LLC}" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 -mattr=+ptx80
        -fp-contract=off "${prefix}.pair.ll" -o "${prefix}.pair.ptx"
      COMMAND "${MDSLC_PTXAS}" --gpu-name sm_89 --fmad=false "${prefix}.pair.ptx" -o "${image}"
      DEPENDS "${prefix}.pair.ll" "${MDSLC_GPU_LLC}" "${MDSLC_PTXAS}" VERBATIM)
  else()
    add_custom_command(OUTPUT "${image}"
      BYPRODUCTS "${prefix}.pair.o"
      COMMAND "${MDSLC_GPU_LLC}" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -fp-contract=off
        -denormal-fp-math=ieee -denormal-fp-math-f32=ieee -filetype=obj "${prefix}.pair.ll" -o "${prefix}.pair.o"
      COMMAND "${MDSLC_GPU_LLD}" -shared "${prefix}.pair.o" -o "${image}"
      DEPENDS "${prefix}.pair.ll" "${MDSLC_GPU_LLC}" "${MDSLC_GPU_LLD}" VERBATIM)
  endif()
  set(embedded "${prefix}.images.cpp")
  add_custom_command(OUTPUT "${embedded}"
    BYPRODUCTS "${embedded}.manifest" "${image}.object-contract" "${image}.assembly"
    COMMAND "${CMAKE_COMMAND}" "-DIMAGE=${image}" "-DTARGET_KIND=${kind}"
      "-DNM=${MDSLC_FUSED_PAIR_NM}" "-DOBJDUMP=${MDSLC_FUSED_PAIR_OBJDUMP}"
      "-DREADELF=${MDSLC_FUSED_PAIR_READELF}" "-DCUOBJDUMP=${MDSLC_FUSED_PAIR_CUOBJDUMP}"
      -P "${gpu_fused_pair_definition_dir}/strict_gpu_fused_pair_object.cmake"
    COMMAND "${CMAKE_COMMAND}" "-DIMAGE=${image}" "-DTARGET_KIND=${kind}" "-DOUTPUT=${embedded}"
      -P "${gpu_fused_pair_definition_dir}/embed_gpu_fused_pair.cmake"
    DEPENDS "${image}" "${gpu_fused_pair_definition_dir}/strict_gpu_fused_pair_object.cmake"
      "${gpu_fused_pair_definition_dir}/embed_gpu_fused_pair.cmake" VERBATIM)
  add_custom_target(matcore_${kind}_fused_pair_images DEPENDS "${embedded}")
  set_property(TARGET matcore_gpu_fused_gemm_candidate PROPERTY MDSLC_${upper}_FUSED_PAIR_PREFIX "${prefix}")
  set_property(TARGET matcore_gpu_fused_gemm_candidate PROPERTY MDSLC_${upper}_FUSED_PAIR_IMAGE "${image}")
  set_property(TARGET matcore_gpu_fused_gemm_candidate PROPERTY MDSLC_${upper}_FUSED_PAIR_EMBEDDED_SOURCE "${embedded}")
  set(${output_embedded} "${embedded}" PARENT_SCOPE)
  set(MDSLC_FUSED_PAIR_CUOBJDUMP "${MDSLC_FUSED_PAIR_CUOBJDUMP}" PARENT_SCOPE)
endfunction()
