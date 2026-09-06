// Models a previously generated object's weak inline constructor. It is never
// executed: the adversary asks whether selecting this old COMDAT definition can
// discard the only reference to the new private runtime revision marker.
namespace matcore::mdslc::runtime::closed_host_v1 {
class Session { public: Session() noexcept {} };
}
void emit_stale_constructor() {
  matcore::mdslc::runtime::closed_host_v1::Session unused;
}
