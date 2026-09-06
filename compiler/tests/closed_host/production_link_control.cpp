#include "closed_host_v1.h"

int main() {
  matcore::mdslc::runtime::closed_host_v1::Session session;
  return session.complete(1) ? 0 : 1;
}
