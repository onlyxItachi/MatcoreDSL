// All modes of this client remain ASan+UBSan instrumented. Negative modes are
// intentional sanitizer controls, not production paths or expected successes.
#include "mlir/IR/MLIRContext.h"
#include <cstdio>
#include <cstring>
#include <sanitizer/asan_interface.h>

#if !__has_feature(address_sanitizer)
#error "allocator sanitizer control requires an instrumented client"
#endif

void loadAllocatorControlDialect(mlir::MLIRContext &context);

int main(int argc, char **argv) {
  mlir::MLIRContext context;
  std::puts("builtin context constructed");
  std::fflush(stdout);
  loadAllocatorControlDialect(context);
  std::puts("custom singleton registered");
  std::fflush(stdout);
  if (argc == 1)
    return 0;

  auto *allocation = new unsigned char[64]();
  if (std::strcmp(argv[1], "heap_oob") == 0) {
    volatile unsigned char *outside = allocation + 64;
    *outside = 42;
  } else if (std::strcmp(argv[1], "manual_poison") == 0) {
    __asan_poison_memory_region(allocation + 32, 8);
    volatile unsigned char *poisoned = allocation + 32;
    *poisoned = 42;
  } else {
    delete[] allocation;
    return 2;
  }
  delete[] allocation;
  return 0;
}
