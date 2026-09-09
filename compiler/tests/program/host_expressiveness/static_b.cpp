#include <matcore/region.h>
static int local_value() { return 47; }
int second_host_value() { return local_value(); }
