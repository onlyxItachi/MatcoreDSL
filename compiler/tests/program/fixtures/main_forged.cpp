// Same spelling is not the canonical physical declaration owner.
namespace matcore::mdsl {
struct Storage { float *data; unsigned long long rows, columns, capacity; unsigned char access; };
struct Result {};
using Shape = unsigned long long;
}
namespace md = matcore::mdsl;
md::Result first(md::Storage, md::Storage, md::Storage, md::Shape, md::Shape, md::Shape) noexcept;
int main() { return 0; }
