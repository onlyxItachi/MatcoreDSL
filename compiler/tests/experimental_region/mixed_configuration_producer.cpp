#include "closed_host_v1.h"
#include <utility>
namespace mdsl = matcore::mdsl;
namespace h = matcore::mdslc::runtime::closed_host_v1;
mdsl::Result produce_observation(float *output, bool fail) noexcept {
  h::Session session;
  h::Value value;
  float input[4]{1, 2, 3, 4};
  session.read(1, {input, 2, 2, 4}, value);
  session.publish(2, value, {output, 2, 2, 4, h::Access::read_write});
  session.observe(3, {output, 2, 2, 4});
  if (fail) session.read(4, {input, 2, 2, 3}, value);
  session.complete(5);
  return std::move(session).takeResult({"mixed.mdsl", 7, 9});
}
