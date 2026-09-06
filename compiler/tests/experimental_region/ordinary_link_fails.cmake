file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT_DIR}/ordinary.cpp" [=[
#include <matcore/region.h>
using namespace matcore::mdsl;
MATCORE_REGION Result computation(Storage a) noexcept {
  auto value=read(a,1,1);
  publish(value,a);
  observe(a);
  return complete();
}
int main() {
  float a=1;
  return computation({&a,1,1,1,Access::read_write}) ? 0 : 1;
}
]=])
separate_arguments(flags NATIVE_COMMAND "${CXX_FLAGS}")
separate_arguments(link_flags NATIVE_COMMAND "${LINK_FLAGS}")
execute_process(COMMAND "${CXX}" -std=c++20 ${flags} "-I${INCLUDE_DIR}"
  -c "${OUTPUT_DIR}/ordinary.cpp" -o "${OUTPUT_DIR}/ordinary.o"
  RESULT_VARIABLE compiled OUTPUT_VARIABLE out ERROR_VARIABLE error)
if(NOT compiled EQUAL 0)
  message(FATAL_ERROR "Experimental source stopped being ordinary valid C++: ${out}${error}")
endif()
execute_process(COMMAND "${CXX}" ${link_flags} "${OUTPUT_DIR}/ordinary.o"
  "${RUNTIME}" -lm -o "${OUTPUT_DIR}/ordinary"
  RESULT_VARIABLE linked OUTPUT_VARIABLE out ERROR_VARIABLE error)
if(linked EQUAL 0 OR NOT "${out}${error}" MATCHES "complete")
  message(FATAL_ERROR "Uncompiled source did not fail at the canonical intrinsic: ${out}${error}")
endif()
