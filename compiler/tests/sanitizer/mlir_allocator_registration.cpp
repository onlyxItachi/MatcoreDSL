// Build this fixture twice: once with the client's ASan flags (the rejected
// mixed allocator protocol), once with only ASan disabled for this TU (the
// protocol of the pinned prebuilt MLIR). Keep UBSan enabled in both versions.
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace {
class AllocatorControlType
    : public mlir::Type::TypeBase<AllocatorControlType, mlir::Type,
                                  mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "allocator_control.singleton";
};
class AllocatorControlDialect : public mlir::Dialect {
public:
  explicit AllocatorControlDialect(mlir::MLIRContext *context)
      : mlir::Dialect(getDialectNamespace(), context,
                      mlir::TypeID::get<AllocatorControlDialect>()) {
    addTypes<AllocatorControlType>();
  }
  static llvm::StringRef getDialectNamespace() { return "allocator_control"; }
};
} // namespace

void loadAllocatorControlDialect(mlir::MLIRContext &context) {
  context.getOrLoadDialect<AllocatorControlDialect>();
}
