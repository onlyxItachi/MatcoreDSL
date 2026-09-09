static int local_value() { return 9; }
int host_helper(int value) { return value + local_value(); }
