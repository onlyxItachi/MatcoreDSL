add_executable(mdslc-region main.cpp)
target_compile_features(mdslc-region PRIVATE cxx_std_20)
target_link_libraries(mdslc-region PRIVATE matcore_experimental_region_compiler
  MatcoreDSL::PlatformSupportV1 LLVM)
target_include_directories(mdslc-region PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
mdslc_target_enable_warnings(mdslc-region WERROR)
mdslc_target_match_llvm_rtti(mdslc-region)
set(region_sanitized 0)
if(MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=.*address")
  set(region_sanitized 1)
endif()
# Canonical Runtime has no installed RPATH. The driver pins both its actual
# build bytes and the exact CMake-installed byte image, not an ELF-normalized
# digest or an editable hash manifest supplied by a consumer.
get_target_property(region_runtime_install_rpath matcore_runtime INSTALL_RPATH)
if(region_runtime_install_rpath)
  message(FATAL_ERROR "Experimental region driver requires the canonical empty Runtime INSTALL_RPATH")
endif()
# No external configuration file or source option can choose the owning runtime.
file(RELATIVE_PATH region_prefix_from_bin "/${CMAKE_INSTALL_BINDIR}" "/")
configure_file(config.h.in region_driver_paths.h.in @ONLY)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/region_driver_paths.h"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/region_driver_paths.h.in")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/region_driver_identity.h"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/region_driver_identity.h.runtime-install-image"
  COMMAND "${CMAKE_COMMAND}"
    "-DARCHIVE=$<TARGET_FILE:matcore_closed_candidates_production_v1>"
    "-DRUNTIME=$<TARGET_FILE:matcore_runtime>"
    "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/region_driver_identity.h"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/installation_identity.cmake"
  DEPENDS matcore_closed_candidates_production_v1 matcore_runtime
    "${CMAKE_CURRENT_SOURCE_DIR}/installation_identity.cmake" VERBATIM)
target_sources(mdslc-region PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/region_driver_identity.h")
install(TARGETS mdslc-region RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
