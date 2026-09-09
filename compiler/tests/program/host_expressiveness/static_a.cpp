#include <matcore/region.h>
static int local_value() { return 31; }
int first_host_value() { return local_value(); }
