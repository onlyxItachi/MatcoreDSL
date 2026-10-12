// Noninstalled negative control for the *actual source-issued DSO*. No public
// dynamic export is added for a private leaf: its local ELF symbol value is an
// independently inspected test input, not production execution authority.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <type_traits>

struct MemRef {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
static_assert(std::is_standard_layout_v<MemRef> && sizeof(MemRef) == 56 &&
              alignof(MemRef) == 8 && offsetof(MemRef, sizes) == 24 &&
              offsetof(MemRef, strides) == 40);
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  char *end = nullptr;
  const auto offset = std::strtoull(argv[2], &end, 16);
  if (!offset || !end || *end) return 2;
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) { std::fprintf(stderr, "probe dlopen: %s\n", dlerror()); return 2; }
  auto *marker = dlsym(library, "matcore_closed_host_private_value_abi_v2");
  Dl_info owner{};
  if (!marker || !dladdr(marker, &owner) || !owner.dli_fbase || !owner.dli_fname ||
      std::filesystem::canonical(owner.dli_fname) != std::filesystem::canonical(argv[1])) return 2;
  using Leaf = void (*)(MemRef *, MemRef *, MemRef *);
  auto leaf = reinterpret_cast<Leaf>(reinterpret_cast<std::uintptr_t>(owner.dli_fbase) + offset);
  auto *a = new float[2]{1, 1}, *b = new float[1]{1}, *c = new float[1]{0};
  MemRef ad{a, a, 0, {1, 2}, {2, 1}}, bd{b, b, 0, {2, 1}, {1, 1}},
      cd{c, c, 0, {1, 1}, {1, 1}};
  leaf(&ad, &bd, &cd); // The issued leaf must diagnose B[1], not the harness.
  delete[] a; delete[] b; delete[] c;
  dlclose(library);
  return 0;
}
