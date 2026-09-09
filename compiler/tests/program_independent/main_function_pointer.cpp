#include "api.h"
// Non-const owning host globals deliberately retain indirect calls in original
// Clang IR. The source ABI and original region symbol identity must survive.
auto first_pointer = &first;
auto second_pointer = &second;
#define first first_pointer
#define second second_pointer
#include "main_body.h"
