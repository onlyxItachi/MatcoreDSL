# Source-to-host linkage libraries are production compiler components. The
# separately installed driver is opt-in; tests never confer execution authority.
add_library(matcore_authenticated_host_thunk STATIC ../codegen/AuthenticatedHostThunk.cpp)
target_compile_features(matcore_authenticated_host_thunk PUBLIC cxx_std_20)
target_include_directories(matcore_authenticated_host_thunk PUBLIC ../codegen)
target_include_directories(matcore_authenticated_host_thunk SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
# Clang already uses the shared LLVM runtime. Mixing static LLVM components can
# duplicate APFloat semantic singletons and corrupt even ordinary float IR.
target_link_libraries(matcore_authenticated_host_thunk PUBLIC LLVM)
mdslc_target_enable_warnings(matcore_authenticated_host_thunk WERROR)
mdslc_target_match_llvm_rtti(matcore_authenticated_host_thunk)

add_library(matcore_frozen_host_codegen STATIC ../codegen/FrozenHostCodegen.cpp)
target_compile_features(matcore_frozen_host_codegen PUBLIC cxx_std_20)
target_include_directories(matcore_frozen_host_codegen PUBLIC ../codegen)
target_include_directories(matcore_frozen_host_codegen SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_frozen_host_codegen PUBLIC matcore_closed_region_admission)
mdslc_target_enable_warnings(matcore_frozen_host_codegen)
mdslc_target_match_llvm_rtti(matcore_frozen_host_codegen)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${region_compiler_dir}/lib/runtime/closed_host_v1.h")
file(READ "${region_compiler_dir}/lib/runtime/closed_host_v1.h" MDSLC_PRIVATE_REGION_RUNTIME_HEADER)
configure_file(../codegen/ExperimentalRegionRuntimeHeader.inc.in
  "ExperimentalRegionRuntimeHeader.inc" @ONLY)
add_library(matcore_experimental_region_compiler STATIC
  ../codegen/ExperimentalRegionEmitter.cpp ../codegen/ExperimentalRegionCompiler.cpp)
target_compile_features(matcore_experimental_region_compiler PUBLIC cxx_std_20)
target_include_directories(matcore_experimental_region_compiler PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(matcore_experimental_region_compiler SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_experimental_region_compiler PUBLIC
  matcore_frozen_host_codegen matcore_closed_host_emitter matcore_authenticated_host_thunk
  PRIVATE LLVM)
mdslc_target_enable_warnings(matcore_experimental_region_compiler WERROR)
mdslc_target_match_llvm_rtti(matcore_experimental_region_compiler)

if(MDSLC_ENABLE_EXPERIMENTAL_REGIONS)
  add_subdirectory("${region_compiler_dir}/tools/mdslc-region" "mdslc-region")
endif()
