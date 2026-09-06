# Experimental intrinsic ownership prototype. The private archive remains the
# existing test/package contract until the isolated driver integration is proven.
mdslc_add_closed_candidate_runtime(matcore_closed_candidates_isolated_v1 SHARED)
set(region_private_exports "${CMAKE_CURRENT_LIST_DIR}/private_runtime_v2.exports")
set_property(TARGET matcore_closed_candidates_isolated_v1 APPEND PROPERTY
  LINK_DEPENDS "${region_private_exports}")
target_link_options(matcore_closed_candidates_isolated_v1 PRIVATE
  "LINKER:--version-script=${region_private_exports}"
  "LINKER:-Bsymbolic" "LINKER:--no-undefined")

if(BUILD_TESTING)
  add_subdirectory("${region_compiler_dir}/tests/private_runtime" "private_runtime_tests")
endif()
