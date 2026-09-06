#include "closed_host_v1.h"
namespace h=matcore::mdslc::runtime::closed_host_v1;
int main() {
  h::Session session;h::Value value;float input=7;
  if(!session.read(1,{&input,1,1,1},value))return 1;
  return value.valid() && value.data()[0]==7?0:2;
}
