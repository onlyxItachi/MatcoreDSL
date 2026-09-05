include "mlir/IR/OpBase.td"

def Matcore_Dialect : Dialect {
  let name = "mdsl";
  let cppNamespace = "::matcore::mdslc::mlir_dialect";
  let summary = "Matcore target-independent semantic operations";
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
  let description = [{
    The Matcore dialect is the compositional optimizer representation reached
    from the verified Matcore IR v1 capture boundary. It represents semantic
    operations and their legality contracts, not a selected CPU or accelerator
    implementation.
  }];
}
