if(NOT DEFINED MDSLC_SOURCE_DIR)
  message(FATAL_ERROR "MDSLC_SOURCE_DIR is required")
endif()

include("${MDSLC_SOURCE_DIR}/cmake/MatcoreDSLToolchainVersion.cmake")

foreach(mdslc_valid_output IN ITEMS
    "clang version 22.1.8"
    "Ubuntu clang version 22.1.8 (vendor build)"
    "clang version 22.1.8\nTarget: x86_64-unknown-linux-gnu")
  mdslc_extract_clang_version_token(
    "${mdslc_valid_output}" mdslc_reported_version)
  if(NOT mdslc_reported_version STREQUAL "22.1.8")
    message(FATAL_ERROR
      "exact Clang version token was not extracted: ${mdslc_valid_output}")
  endif()
endforeach()

foreach(mdslc_forged_output IN ITEMS
    "clang version 22.1.80"
    "clang version 122.1.8"
    "clang version 22.1.8-rc1"
    "not a Clang version response")
  mdslc_extract_clang_version_token(
    "${mdslc_forged_output}" mdslc_reported_version)
  if(mdslc_reported_version STREQUAL "22.1.8")
    message(FATAL_ERROR
      "near-version forgery was admitted: ${mdslc_forged_output}")
  endif()
endforeach()
