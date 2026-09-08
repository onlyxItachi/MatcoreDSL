# The experimental source driver uses this intrinsically bound implementation.
# The private archive remains a separate internal test/package contract; it does
# not establish the driver's protection against host implementation replacement.
mdslc_add_closed_candidate_runtime(matcore_closed_candidates_isolated_v1 SHARED)
set(region_private_exports "${CMAKE_CURRENT_LIST_DIR}/private_runtime_v2.exports")
set_property(TARGET matcore_closed_candidates_isolated_v1 APPEND PROPERTY
  LINK_DEPENDS "${region_private_exports}")
target_link_options(matcore_closed_candidates_isolated_v1 PRIVATE
  "LINKER:--version-script=${region_private_exports}"
  "LINKER:-Bsymbolic")
# Clang sanitizer runtimes are supplied by the final executable, not by each
# instrumented shared object. --no-undefined is valid only without that contract.
if(NOT MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=")
  target_link_options(matcore_closed_candidates_isolated_v1 PRIVATE "LINKER:--no-undefined")
endif()

if(BUILD_TESTING)
  add_subdirectory("${region_compiler_dir}/tests/private_runtime" "private_runtime_tests")
endif()
if(MDSLC_ENABLE_EXPERIMENTAL_REGIONS)
  install(TARGETS matcore_closed_candidates_isolated_v1
    LIBRARY DESTINATION "${MDSLC_EXPERIMENTAL_REGION_PRIVATE_LIBDIR}")
endif()
