if(NOT TARGET matcore_gpu_fused_gemm_candidate)
  return()
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(matcore_gpu_fused_gemm_candidate_test candidate_test.cpp)
target_compile_features(matcore_gpu_fused_gemm_candidate_test PRIVATE cxx_std_20)
target_link_libraries(matcore_gpu_fused_gemm_candidate_test PRIVATE matcore_gpu_fused_gemm_candidate LLVM)
mdslc_target_match_llvm_rtti(matcore_gpu_fused_gemm_candidate_test)
add_test(NAME generated_gpu_fused_pair.issuer COMMAND matcore_gpu_fused_gemm_candidate_test)
add_test(NAME generated_gpu_fused_pair.cli COMMAND "${CMAKE_COMMAND}"
  "-DISSUER=$<TARGET_FILE:matcore-gpu-fused-gemm-candidate>" -P "${CMAKE_CURRENT_SOURCE_DIR}/cli_contract.cmake")
foreach(kind IN ITEMS nvvm rocdl)
  string(TOUPPER "${kind}" upper)
  if(NOT MDSLC_ENABLE_EXPERIMENTAL_${upper})
    continue()
  endif()
  get_target_property(image matcore_gpu_fused_gemm_candidate MDSLC_${upper}_FUSED_PAIR_IMAGE)
  if(kind STREQUAL "nvvm")
    set(MDSLC_NVVM_FUSED_PAIR_TEST_IMAGE "${image}")
  endif()
  add_test(NAME generated_gpu_fused_pair.${kind}.object COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/object_contract.py" --image "${image}" --target "${kind}"
    --gate "${PROJECT_SOURCE_DIR}/lib/regions/strict_gpu_fused_pair_object.cmake"
    --build "${CMAKE_CURRENT_BINARY_DIR}/${kind}-object-falsifiers")
  # Real vendor DSOs with host ASan remain unqualified. Device instrumentation
  # is separately exercised by CUDA Compute Sanitizer below; adapter API mocks
  # own their independent host-ASan/UBSan scope in RuntimeTests.cmake.
  set(physical_modes normal)
  foreach(mode IN LISTS physical_modes)
    set(executable matcore_gpu_fused_pair_${kind}_${mode}_test)
    add_executable(${executable} physical_test.cpp)
    target_compile_features(${executable} PRIVATE cxx_std_20)
    target_compile_options(${executable} PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
    target_link_libraries(${executable} PRIVATE Threads::Threads)
    if(kind STREQUAL "nvvm")
      target_compile_definitions(${executable} PRIVATE RESEARCH_CUDA)
      target_include_directories(${executable} PRIVATE "${MDSLC_CUDA_INCLUDE}")
      target_link_libraries(${executable} PRIVATE "${MDSLC_CUDA_DRIVER_IDENTITY}")
    else()
      target_compile_definitions(${executable} PRIVATE __HIP_PLATFORM_AMD__)
      target_include_directories(${executable} PRIVATE "${MDSLC_HIP_INCLUDE}")
      target_link_libraries(${executable} PRIVATE "${MDSLC_HIP_RUNTIME_IDENTITY}")
    endif()
    if(mode STREQUAL "asan")
      target_compile_options(${executable} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${executable} PRIVATE -fsanitize=address,undefined)
    endif()
    add_dependencies(${executable} matcore_${kind}_fused_pair_images)
    add_test(NAME generated_gpu_fused_pair.${kind}.physical.${mode} COMMAND ${executable} "${image}")
    set_tests_properties(generated_gpu_fused_pair.${kind}.physical.${mode} PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endforeach()
endforeach()
if(MDSLC_ENABLE_EXPERIMENTAL_NVVM)
  find_program(MDSLC_FUSED_PAIR_COMPUTE_SANITIZER NAMES compute-sanitizer HINTS /usr/local/cuda/bin REQUIRED)
  foreach(mode IN ITEMS normal negative)
    add_test(NAME generated_gpu_fused_pair.nvvm.memcheck.${mode} COMMAND "${CMAKE_COMMAND}"
      "-DEXECUTABLE=$<TARGET_FILE:matcore_gpu_fused_pair_nvvm_normal_test>"
      "-DIMAGE=${MDSLC_NVVM_FUSED_PAIR_TEST_IMAGE}" "-DMODE=${mode}"
      "-DSANITIZER=${MDSLC_FUSED_PAIR_COMPUTE_SANITIZER}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/memcheck.cmake")
    set_tests_properties(generated_gpu_fused_pair.nvvm.memcheck.${mode} PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
  endforeach()
endif()
