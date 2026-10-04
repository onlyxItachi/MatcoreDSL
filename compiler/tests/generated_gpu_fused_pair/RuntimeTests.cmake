# Real Session state transitions with independent mocked device functions; this
# target requires no SDK/device and must run even when GPU issuers are disabled.
if(NOT TARGET mdslc-region OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT MDSLC_CLOSED_CPU_X86)
  return()
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(matcore_gpu_fused_frontier_test frontier_test.cpp
  ../../lib/runtime/closed_host_v1.cpp ../../lib/platform/closed_fp_environment_v1.cpp)
target_compile_features(matcore_gpu_fused_frontier_test PRIVATE cxx_std_20)
target_include_directories(matcore_gpu_fused_frontier_test PRIVATE ../../lib/runtime ../../include)
target_compile_definitions(matcore_gpu_fused_frontier_test PRIVATE MDSLC_CLOSED_HOST_TESTING
  MDSLC_CLOSED_HOST_GENERATED_NVVM MDSLC_CLOSED_HOST_GENERATED_NVVM_FUSED_PAIR
  MDSLC_CLOSED_HOST_GENERATED_ROCDL MDSLC_CLOSED_HOST_GENERATED_ROCDL_FUSED_PAIR)
target_compile_options(matcore_gpu_fused_frontier_test PRIVATE
  -fno-fast-math -ffp-contract=off -frounding-math)
target_link_libraries(matcore_gpu_fused_frontier_test PRIVATE Threads::Threads)
add_test(NAME generated_gpu_fused_pair.frontiers COMMAND matcore_gpu_fused_frontier_test)

foreach(kind IN ITEMS nvvm rocdl)
  string(TOUPPER "${kind}" upper)
  set(enabled)
  if(MDSLC_ENABLE_EXPERIMENTAL_${upper})
    list(APPEND enabled --enabled)
  endif()
  foreach(route IN ITEMS source program)
    set(program)
    if(route STREQUAL "program")
      list(APPEND program --program)
    endif()
    add_test(NAME generated_gpu_fused_pair.${kind}.${route}
      COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/source_execution.py"
        --driver "$<TARGET_FILE:mdslc-region>" --build "${CMAKE_CURRENT_BINARY_DIR}"
        --nm "${CMAKE_NM}" --target "${kind}" ${enabled} ${program})
    set_tests_properties(generated_gpu_fused_pair.${kind}.${route}
      PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
  endforeach()
  if(NOT MDSLC_ENABLE_EXPERIMENTAL_${upper})
    continue()
  endif()
  add_test(NAME generated_gpu_fused_pair.${kind}.installed_source
    COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/source_execution.py"
      --driver "$<TARGET_FILE:mdslc-region>" --build "${CMAKE_CURRENT_BINARY_DIR}"
      --nm "${CMAKE_NM}" --target "${kind}" --enabled --program
      --install-build "${CMAKE_BINARY_DIR}" --install-bindir "${CMAKE_INSTALL_BINDIR}")
  set_tests_properties(generated_gpu_fused_pair.${kind}.installed_source
    PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
  if(kind STREQUAL "nvvm")
    set(adapter cuda)
    set(fixture "${CMAKE_CURRENT_SOURCE_DIR}/../generated_gpu/cuda_adapter_test.cpp")
    set(sdk_include "${MDSLC_CUDA_INCLUDE}")
    set(faults 0 pending pending-download pending-D pending-launch zero-N)
    foreach(index RANGE 1 30)
      list(APPEND faults "${index}")
    endforeach()
  else()
    set(adapter rocdl)
    set(fixture "${CMAKE_CURRENT_SOURCE_DIR}/../closed_candidates/rocdl_adapter_test.cpp")
    set(sdk_include "${MDSLC_HIP_INCLUDE}")
    set(faults good unavailable shape oversize output-alias null environment
      unknown-completion unknown-D unknown-launch unknown-download
      output-limit zero-K-limit work-limit zero-N)
    foreach(index RANGE 1 24)
      list(APPEND faults "fault-${index}")
    endforeach()
  endif()
  foreach(mode IN ITEMS normal asan)
    set(target matcore_${kind}_fused_pair_mock_${mode})
    add_executable(${target} "${fixture}" ../../lib/runtime/closed_${adapter}_candidate_v1.cpp)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_include_directories(${target} PRIVATE ../../lib/runtime ../../include "${sdk_include}")
    target_compile_definitions(${target} PRIVATE MDSLC_TEST_GPU_FUSED_PAIR
      "MDSLC_CLOSED_HOST_GENERATED_${upper}_FUSED_PAIR")
    if(kind STREQUAL "rocdl")
      target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__)
    endif()
    target_compile_options(${target} PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    if(mode STREQUAL "asan")
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
    foreach(fault IN LISTS faults)
      add_test(NAME generated_gpu_fused_pair.${kind}.mock.${mode}.${fault}
        COMMAND ${target} "${fault}")
    endforeach()
  endforeach()
endforeach()
