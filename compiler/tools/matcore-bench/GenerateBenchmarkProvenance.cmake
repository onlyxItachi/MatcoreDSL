cmake_minimum_required(VERSION 3.24)

foreach(required_variable
    MATCORE_BENCH_PROVENANCE_OUTPUT
    MATCORE_BENCH_SOURCE_ROOT)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

set(source_commit "unknown")
set(source_dirty 0)
set(source_state "unknown")
set(source_origin "unavailable")

set(has_commit_override FALSE)
set(has_state_override FALSE)
if(DEFINED MATCORE_BENCH_SOURCE_COMMIT_OVERRIDE AND
   NOT MATCORE_BENCH_SOURCE_COMMIT_OVERRIDE STREQUAL "")
  set(has_commit_override TRUE)
endif()
if(DEFINED MATCORE_BENCH_SOURCE_STATE_OVERRIDE AND
   NOT MATCORE_BENCH_SOURCE_STATE_OVERRIDE STREQUAL "")
  set(has_state_override TRUE)
endif()

if(has_commit_override OR has_state_override)
  if(NOT has_commit_override OR NOT has_state_override)
    message(FATAL_ERROR
      "benchmark provenance overrides require both commit and state")
  endif()
  string(LENGTH "${MATCORE_BENCH_SOURCE_COMMIT_OVERRIDE}" override_length)
  if((NOT override_length EQUAL 40 AND NOT override_length EQUAL 64) OR
     NOT MATCORE_BENCH_SOURCE_COMMIT_OVERRIDE MATCHES "^[0-9A-Fa-f]+$")
    message(FATAL_ERROR
      "benchmark provenance commit override must be an exact Git object ID")
  endif()
  if(NOT MATCORE_BENCH_SOURCE_STATE_OVERRIDE MATCHES
      "^(clean|dirty|unknown)$")
    message(FATAL_ERROR
      "benchmark provenance state override must be clean, dirty, or unknown")
  endif()
  string(TOLOWER "${MATCORE_BENCH_SOURCE_COMMIT_OVERRIDE}" source_commit)
  set(source_state "${MATCORE_BENCH_SOURCE_STATE_OVERRIDE}")
  if(source_state STREQUAL "dirty")
    set(source_dirty 1)
  endif()
  set(source_origin "explicit-override")
elseif(DEFINED MATCORE_BENCH_GIT_EXECUTABLE AND
       NOT MATCORE_BENCH_GIT_EXECUTABLE STREQUAL "" AND
       EXISTS "${MATCORE_BENCH_GIT_EXECUTABLE}")
  file(REAL_PATH "${MATCORE_BENCH_SOURCE_ROOT}" expected_root)
  execute_process(
    COMMAND "${MATCORE_BENCH_GIT_EXECUTABLE}" -C
            "${MATCORE_BENCH_SOURCE_ROOT}" rev-parse --show-toplevel
    OUTPUT_VARIABLE discovered_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE root_result
  )
  if(root_result EQUAL 0)
    file(REAL_PATH "${discovered_root}" discovered_root)
  endif()
  if(root_result EQUAL 0 AND discovered_root STREQUAL expected_root)
    execute_process(
      COMMAND "${MATCORE_BENCH_GIT_EXECUTABLE}" -C "${expected_root}"
              rev-parse --verify HEAD
      OUTPUT_VARIABLE discovered_commit
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE commit_result
    )
    execute_process(
      COMMAND "${MATCORE_BENCH_GIT_EXECUTABLE}" -C "${expected_root}"
              status --porcelain=v1 --untracked-files=no
      OUTPUT_VARIABLE tracked_status
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE status_result
    )
    string(LENGTH "${discovered_commit}" discovered_commit_length)
    if(commit_result EQUAL 0 AND status_result EQUAL 0 AND
       (discovered_commit_length EQUAL 40 OR
        discovered_commit_length EQUAL 64) AND
       discovered_commit MATCHES "^[0-9A-Fa-f]+$")
      string(TOLOWER "${discovered_commit}" source_commit)
      if(tracked_status STREQUAL "")
        set(source_state "clean")
      else()
        set(source_dirty 1)
        set(source_state "dirty")
      endif()
      set(source_origin "git-worktree")
    endif()
  endif()
endif()

get_filename_component(output_directory
  "${MATCORE_BENCH_PROVENANCE_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(temporary_output "${MATCORE_BENCH_PROVENANCE_OUTPUT}.tmp")
file(WRITE "${temporary_output}"
  "#ifndef MATCORE_BENCH_PROVENANCE_GENERATED_H\n"
  "#define MATCORE_BENCH_PROVENANCE_GENERATED_H\n\n"
  "#define MATCORE_BENCH_SOURCE_COMMIT \"${source_commit}\"\n"
  "#define MATCORE_BENCH_SOURCE_WORKTREE_DIRTY ${source_dirty}\n"
  "#define MATCORE_BENCH_SOURCE_PROVENANCE_STATE \"${source_state}\"\n"
  "#define MATCORE_BENCH_SOURCE_PROVENANCE_ORIGIN \"${source_origin}\"\n\n"
  "#endif\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${temporary_output}" "${MATCORE_BENCH_PROVENANCE_OUTPUT}"
  COMMAND_ERROR_IS_FATAL ANY
)
file(REMOVE "${temporary_output}")
