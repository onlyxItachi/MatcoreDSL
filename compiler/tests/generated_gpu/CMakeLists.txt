if(NOT TARGET mdslc-region)
  return()
endif()
add_executable(matcore_gpu_qualification_test qualification_test.cpp
  ../../lib/runtime/closed_host_v1.cpp
  ../../lib/platform/closed_fp_environment_v1.cpp)
target_compile_features(matcore_gpu_qualification_test PRIVATE cxx_std_20)
target_compile_definitions(matcore_gpu_qualification_test PRIVATE
  MDSLC_CLOSED_HOST_TESTING MDSLC_CLOSED_HOST_GENERATED_ROCDL)
target_include_directories(matcore_gpu_qualification_test PRIVATE ../../lib/runtime ../../include)
target_compile_options(matcore_gpu_qualification_test PRIVATE
  -fno-fast-math -ffp-contract=off -frounding-math)
add_test(NAME generated_gpu.qualification COMMAND matcore_gpu_qualification_test)
foreach(kind IN ITEMS nvvm rocdl)
  string(TOUPPER "${kind}" upper)
  set(enabled)
  if(MDSLC_ENABLE_EXPERIMENTAL_${upper})
    list(APPEND enabled --enabled)
    add_executable(matcore_${kind}_candidate_identity candidate_identity.cpp)
    target_compile_features(matcore_${kind}_candidate_identity PRIVATE cxx_std_20)
    target_compile_options(matcore_${kind}_candidate_identity PRIVATE
      -fno-fast-math -ffp-contract=off -frounding-math)
    target_include_directories(matcore_${kind}_candidate_identity PRIVATE
      ../../lib/runtime ../../include)
    target_link_libraries(matcore_${kind}_candidate_identity PRIVATE matcore_closed_candidates_isolated_v1)
    add_test(NAME generated_gpu.${kind}.registry COMMAND matcore_${kind}_candidate_identity ${kind})
    add_test(NAME generated_gpu.${kind}.installed_source COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/source_execution.py"
      --driver "$<TARGET_FILE:mdslc-region>" --build "${CMAKE_CURRENT_BINARY_DIR}"
      --target "${kind}" --enabled --install-build "${CMAKE_BINARY_DIR}"
      --install-bindir "${CMAKE_INSTALL_BINDIR}")
    set_tests_properties(generated_gpu.${kind}.installed_source PROPERTIES
      TIMEOUT 900 RUN_SERIAL TRUE)
  endif()
  add_test(NAME generated_gpu.${kind}.source COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/source_execution.py"
    --driver "$<TARGET_FILE:mdslc-region>" --build "${CMAKE_CURRENT_BINARY_DIR}"
    --target "${kind}" ${enabled})
  set_tests_properties(generated_gpu.${kind}.source PROPERTIES
    TIMEOUT 900 RUN_SERIAL TRUE)
endforeach()
if(TARGET matcore_gpu_gemm_candidate)
  add_executable(matcore_gpu_issuer_test gpu_issuer_test.cpp)
  target_compile_features(matcore_gpu_issuer_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_gpu_issuer_test PRIVATE matcore_gpu_gemm_candidate LLVM)
  mdslc_target_match_llvm_rtti(matcore_gpu_issuer_test)
  add_test(NAME generated_gpu.issuer COMMAND matcore_gpu_issuer_test)
endif()

if(MDSLC_ENABLE_EXPERIMENTAL_ROCDL)
  foreach(mode IN ITEMS normal asan)
    add_executable(matcore_rocdl_adapter_${mode}_test
      ../closed_candidates/rocdl_adapter_test.cpp ../../lib/runtime/closed_rocdl_candidate_v1.cpp)
    target_compile_features(matcore_rocdl_adapter_${mode}_test PRIVATE cxx_std_20)
    target_include_directories(matcore_rocdl_adapter_${mode}_test PRIVATE
      ../../lib/runtime ../../include "${MDSLC_HIP_INCLUDE}")
    target_compile_definitions(matcore_rocdl_adapter_${mode}_test PRIVATE __HIP_PLATFORM_AMD__)
    target_compile_options(matcore_rocdl_adapter_${mode}_test PRIVATE
      -fno-fast-math -ffp-contract=off -frounding-math)
    target_link_libraries(matcore_rocdl_adapter_${mode}_test PRIVATE Threads::Threads)
    if(mode STREQUAL "asan")
      target_compile_options(matcore_rocdl_adapter_${mode}_test PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(matcore_rocdl_adapter_${mode}_test PRIVATE -fsanitize=address,undefined)
    endif()
    set(controls good unavailable shape oversize output-alias null environment
      unknown-completion output-limit zero-K-limit work-limit)
    foreach(index RANGE 1 23)
      list(APPEND controls "fault-${index}")
    endforeach()
    foreach(control IN LISTS controls)
      add_test(NAME generated_gpu.rocdl.mock.${mode}.${control}
        COMMAND matcore_rocdl_adapter_${mode}_test "${control}")
    endforeach()
  endforeach()
endif()

if(MDSLC_ENABLE_EXPERIMENTAL_NVVM)
  foreach(mode IN ITEMS normal asan)
    add_executable(matcore_cuda_adapter_${mode}_test
      cuda_adapter_test.cpp ../../lib/runtime/closed_cuda_candidate_v1.cpp)
    target_compile_features(matcore_cuda_adapter_${mode}_test PRIVATE cxx_std_20)
    target_include_directories(matcore_cuda_adapter_${mode}_test PRIVATE
      ../../lib/runtime ../../include "${MDSLC_CUDA_INCLUDE}")
    target_compile_options(matcore_cuda_adapter_${mode}_test PRIVATE
      -fno-fast-math -ffp-contract=off -frounding-math)
    target_link_libraries(matcore_cuda_adapter_${mode}_test PRIVATE Threads::Threads)
    if(mode STREQUAL "asan")
      target_compile_options(matcore_cuda_adapter_${mode}_test PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(matcore_cuda_adapter_${mode}_test PRIVATE -fsanitize=address,undefined)
    endif()
    set(controls 0 pending pending-download)
    foreach(index RANGE 1 31)
      list(APPEND controls "${index}")
    endforeach()
    foreach(control IN LISTS controls)
      add_test(NAME generated_gpu.nvvm.mock.${mode}.${control}
        COMMAND matcore_cuda_adapter_${mode}_test "${control}")
    endforeach()
  endforeach()
endif()

foreach(kind IN ITEMS nvvm rocdl)
  string(TOUPPER "${kind}" upper)
  if(NOT MDSLC_ENABLE_EXPERIMENTAL_${upper})
    continue()
  endif()
  get_target_property(embedded matcore_gpu_gemm_candidate MDSLC_${upper}_EMBEDDED_SOURCE)
  if(kind STREQUAL "nvvm")
    set(adapter cuda)
    set(fixture "${CMAKE_CURRENT_SOURCE_DIR}/cuda_adapter_physical_test.cpp")
    set(sdk_include "${MDSLC_CUDA_INCLUDE}")
    set(sdk_library "${MDSLC_CUDA_DRIVER_IDENTITY}")
    find_program(MDSLC_CUOBJDUMP NAMES cuobjdump HINTS /usr/local/cuda/bin REQUIRED)
    set(object_options --cuobjdump "${MDSLC_CUOBJDUMP}")
  else()
    set(adapter rocdl)
    set(fixture "${CMAKE_CURRENT_SOURCE_DIR}/../closed_candidates/rocdl_adapter_physical_test.cpp")
    set(sdk_include "${MDSLC_HIP_INCLUDE}")
    set(sdk_library "${MDSLC_HIP_RUNTIME_IDENTITY}")
    set(object_options)
  endif()
  get_filename_component(image_directory "${embedded}" DIRECTORY)
  add_test(NAME generated_gpu.${kind}.object COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/object_contract.py"
    --target "${kind}" --prefix "${image_directory}/${kind}-strict"
    --llvm-bin "${LLVM_TOOLS_BINARY_DIR}" ${object_options}
    --report "${CMAKE_CURRENT_BINARY_DIR}/${kind}-object-contract.json")
  add_executable(matcore_${kind}_adapter_physical_test "${fixture}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/runtime/closed_${adapter}_candidate_v1.cpp" "${embedded}")
  target_compile_features(matcore_${kind}_adapter_physical_test PRIVATE cxx_std_20)
  target_include_directories(matcore_${kind}_adapter_physical_test PRIVATE
    ../../lib/runtime ../../include "${sdk_include}")
  target_compile_options(matcore_${kind}_adapter_physical_test PRIVATE
    -fno-fast-math -ffp-contract=off -frounding-math)
  if(kind STREQUAL "rocdl")
    target_compile_definitions(matcore_${kind}_adapter_physical_test PRIVATE __HIP_PLATFORM_AMD__)
  endif()
  target_link_libraries(matcore_${kind}_adapter_physical_test PRIVATE "${sdk_library}" Threads::Threads)
  add_dependencies(matcore_${kind}_adapter_physical_test matcore_${kind}_images)
  add_test(NAME generated_gpu.${kind}.physical_adapter COMMAND matcore_${kind}_adapter_physical_test)
  set_tests_properties(generated_gpu.${kind}.physical_adapter PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
endforeach()
