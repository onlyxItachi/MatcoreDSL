// Reuse exact ABI/loader ownership vocabulary; only vary the violating N width.
// Noninstalled test-only local-symbol invocation, never a source authority.
#define main retainedSourcePatternAsanProbe
#include "../source_pattern/raw_asan_probe.cpp"
#undef main
#include <algorithm>

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  char *offset_end=nullptr,*width_end=nullptr;
  const auto offset=std::strtoull(argv[2],&offset_end,16);
  const auto n=std::strtoul(argv[3],&width_end,10);
  if(!offset || !offset_end || *offset_end || !width_end || *width_end || n<1 || n>64) return 2;
  void *library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!library) { std::fprintf(stderr,"probe dlopen: %s\n",dlerror()); return 2; }
  auto *marker=dlsym(library,"matcore_closed_host_private_value_abi_v2");
  Dl_info owner{};
  if(!marker || !dladdr(marker,&owner) || !owner.dli_fbase || !owner.dli_fname ||
      std::filesystem::canonical(owner.dli_fname)!=std::filesystem::canonical(argv[1])) return 2;
  using Leaf=void (*)(MemRef *,MemRef *,MemRef *);
  auto leaf=reinterpret_cast<Leaf>(reinterpret_cast<std::uintptr_t>(owner.dli_fbase)+offset);
  auto *a=new float[2]{1,1}, *b=new float[2*n-1], *c=new float[n];
  std::fill_n(b,2*n-1,1.0F); std::fill_n(c,n,0.0F);
  const auto width=static_cast<std::int64_t>(n);
  MemRef ad{a,a,0,{1,2},{2,1}}, bd{b,b,0,{2,width},{width,1}}, cd{c,c,0,{1,width},{width,1}};
  leaf(&ad,&bd,&cd); // Only the last B element is absent, in chosen full/tail N.
  delete[] a; delete[] b; delete[] c;
  dlclose(library);
  return 0;
}
