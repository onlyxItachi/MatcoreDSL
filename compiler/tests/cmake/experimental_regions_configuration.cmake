foreach(mode IN ITEMS native_off mlir_off newer_toolchain cross_compile)
  set(extra)
  if(mode STREQUAL "native_off")
    set(extra -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF)
  elseif(mode STREQUAL "mlir_off")
    set(extra -DMDSLC_ENABLE_MATCORE_MLIR=OFF)
  elseif(mode STREQUAL "newer_toolchain")
    set(extra -DMDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION=22.1.8)
  else()
    set(extra -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64)
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}"
    -B "${BINARY_DIR}/experimental-region-rejection-${mode}" -G Ninja
    "-DCMAKE_C_COMPILER=${CC}" "-DCMAKE_CXX_COMPILER=${CXX}"
    -DBUILD_TESTING=OFF -DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON
    -DMDSLC_ENABLE_NATIVE_FRONTEND=ON -DMDSLC_ENABLE_MATCORE_MLIR=ON ${extra}
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT "${output}${error}" MATCHES
      "MDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON requires native Linux x86_64")
    message(FATAL_ERROR "Unsupported ${mode} tuple was not rejected at the feature gate: ${output}\n${error}")
  endif()
endforeach()
message(STATUS "Four unsupported experimental-region configurations rejected at the feature gate")
