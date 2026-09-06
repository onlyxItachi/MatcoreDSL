#include "closed_host_v1.h"
#include <utility>
namespace h=matcore::mdslc::runtime::closed_host_v1;
namespace m=matcore::mdsl;
m::Result issue(bool twice) {
  h::Session s; h::Value v;
  float input=7, output=0;
  if(!s.read(1,{&input,1,1,1},v)) return std::move(s).takeResult({"read",1,1});
  if(!s.publish(2,v,{&output,1,1,1,h::Access::read_write})) return std::move(s).takeResult({"publish",2,1});
  if(!s.observe(3,{&output,1,1,1})) return std::move(s).takeResult({"observe",3,1});
  if(twice) {
    if(!s.publish(4,v,{&output,1,1,1,h::Access::read_write})) return std::move(s).takeResult({"publish",4,1});
    if(!s.observe(5,{&output,1,1,1})) return std::move(s).takeResult({"observe",5,1});
  }
  s.complete(twice?6:4);
  return std::move(s).takeResult();
}
