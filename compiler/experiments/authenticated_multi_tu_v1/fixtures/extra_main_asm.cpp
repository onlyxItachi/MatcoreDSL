#include "api.h"
extern "C" int alternative_entry() asm("main");
extern "C" int alternative_entry() { return 23; }
