include_guard(GLOBAL)

function(matcoredsl_add_executable target_name)
  cmake_parse_arguments(
    PARSE_ARGV 1
    MDSLC
    ""
    "SOURCE;CXX_STANDARD;MATCORE_TARGET;FRONTEND"
    "COMPILE_OPTIONS;LINK_LIBRARIES"
  )

  if(MDSLC_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}): unrecognized arguments: "
      "${MDSLC_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT MDSLC_SOURCE)
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}) requires SOURCE")
  endif()
  if(TARGET "${target_name}")
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}): target already exists")
  endif()

  if(NOT MDSLC_CXX_STANDARD)
    set(MDSLC_CXX_STANDARD 20)
  endif()
  if(NOT MDSLC_MATCORE_TARGET)
    set(MDSLC_MATCORE_TARGET cpu)
  endif()
  if(NOT MDSLC_FRONTEND)
    set(MDSLC_FRONTEND native)
  endif()
  if(NOT MDSLC_MATCORE_TARGET STREQUAL "cpu")
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}): bootstrap v0 supports only "
      "MATCORE_TARGET cpu")
  endif()
  if(NOT MDSLC_FRONTEND STREQUAL "native" AND
     NOT MDSLC_FRONTEND STREQUAL "ast-json-bootstrap")
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}): FRONTEND must be native or "
      "ast-json-bootstrap; no automatic fallback is performed")
  endif()

  get_filename_component(
    mdsl_source "${MDSLC_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${mdsl_source}")
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}): source does not exist: "
      "${mdsl_source}")
  endif()
  get_filename_component(mdsl_extension "${mdsl_source}" LAST_EXT)
  if(NOT mdsl_extension STREQUAL ".mdsl")
    message(FATAL_ERROR
      "matcoredsl_add_executable(${target_name}): SOURCE must have a .mdsl "
      "extension")
  endif()

  string(SHA256 mdsl_source_hash "${mdsl_source}")
  string(SUBSTRING "${mdsl_source_hash}" 0 12 mdsl_source_hash)
  get_filename_component(mdsl_stem "${mdsl_source}" NAME_WE)
  set(mdsl_object_dir
      "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${target_name}.mdsl")
  if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # One MDSLC translation unit produces host, stub, and backend objects.
    # Windows has no safe analogue of the ELF partial link, so mdslc++ archives
    # those ordinary COFF objects into one normal static .lib. Individual .obj
    # files remain visible only through --save-temps/artifact validation.
    set(mdsl_object_extension ".lib")
    set(mdsl_standard_option "/std:c++${MDSLC_CXX_STANDARD}")
    set(mdsl_compile_only_option "/c")
    set(mdsl_output_option -o
      "${mdsl_object_dir}/${mdsl_stem}-${mdsl_source_hash}.lib")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(mdsl_runtime_option /MDd)
    else()
      set(mdsl_runtime_option /MD)
    endif()
    set(mdsl_frontend_options /TP /EHsc /Zc:__cplusplus ${mdsl_runtime_option})
  else()
    set(mdsl_object_extension ".o")
    set(mdsl_standard_option "-std=c++${MDSLC_CXX_STANDARD}")
    set(mdsl_compile_only_option -c)
    set(mdsl_output_option -o
      "${mdsl_object_dir}/${mdsl_stem}-${mdsl_source_hash}.o")
    set(mdsl_frontend_options)
  endif()
  set(mdsl_object
      "${mdsl_object_dir}/${mdsl_stem}-${mdsl_source_hash}${mdsl_object_extension}")
  set(mdsl_depfile "${mdsl_object}.d")
  # These are public mdslc++ options rather than clang-cl escape hatches.  The
  # driver scans the original and generated translation units, authenticates
  # that their dependency closures agree, and atomically publishes this one
  # CMake/Ninja DEPFILE without temporary generated paths.
  set(mdsl_dependency_options "--matcore-depfile=${mdsl_depfile}")

  add_custom_command(
    OUTPUT "${mdsl_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${mdsl_object_dir}"
    COMMAND
      "$<TARGET_FILE:MatcoreDSL::Compiler>"
      "${mdsl_standard_option}"
      "--matcore-target=${MDSLC_MATCORE_TARGET}"
      "--frontend=${MDSLC_FRONTEND}"
      ${mdsl_frontend_options}
      ${MDSLC_COMPILE_OPTIONS}
      ${mdsl_dependency_options}
      ${mdsl_compile_only_option} "${mdsl_source}"
      ${mdsl_output_option}
    DEPENDS
      "${mdsl_source}"
      MatcoreDSL::Compiler
      MatcoreDSL::Extractor
    DEPFILE "${mdsl_depfile}"
    COMMENT "Compiling MDSLC source ${MDSLC_SOURCE}"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  set_source_files_properties(
    "${mdsl_object}"
    PROPERTIES
      GENERATED TRUE
      EXTERNAL_OBJECT TRUE
  )
  add_executable("${target_name}" "${mdsl_object}")
  set_property(TARGET "${target_name}" PROPERTY LINKER_LANGUAGE CXX)
  if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # This archive is the complete output of one MDSLC translation unit.  In
    # particular, it can contain the executable entry point.  link.exe cannot
    # discover that entry point through the usual unresolved-symbol archive
    # extraction cycle when no ordinary object precedes the archive, so retain
    # every constituent COFF object explicitly.  The archive remains listed as
    # a GENERATED source above to preserve the custom-command dependency.
    target_link_options(
      "${target_name}" PRIVATE "LINKER:/WHOLEARCHIVE:${mdsl_object}")
  endif()
  # Linker-generated dependency files do not reliably escape whitespace in
  # imported shared-library paths. Imported package runtimes have no build
  # rule in the consumer project, so retain target-level ordering while
  # avoiding a malformed file dependency and perpetual no-op relinks.
  set_property(TARGET "${target_name}" PROPERTY LINK_DEPENDS_NO_SHARED TRUE)
  target_link_libraries(
    "${target_name}"
    PRIVATE
      MatcoreDSL::Runtime
      ${MDSLC_LINK_LIBRARIES}
  )
endfunction()
