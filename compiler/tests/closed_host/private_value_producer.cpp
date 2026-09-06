#include "closed_host_v1.h"
namespace h=matcore::mdslc::runtime::closed_host_v1;
h::Value opaque_value_from_normal_configuration() {
  h::Session session;h::Value value;float input[4]={1,2,3,4};
  (void)session.read(1,{input,2,2,4},value);
  return value;
}
