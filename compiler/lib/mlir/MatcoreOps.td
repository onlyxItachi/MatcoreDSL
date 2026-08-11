include "MatcoreDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Matcore_Op<string mnemonic, list<Trait> traits = []>
    : Op<Matcore_Dialect, mnemonic, traits>;

def Matcore_GemmOp : Matcore_Op<"gemm", [
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "destination-aware semantic matrix multiplication";
  let description = [{
    Computes a rank-two matrix contraction from `lhs` and `rhs` into the
    explicit `output` destination and returns the post-overwrite tensor SSA
    value tied to that destination, not an independently allocated tensor.
    Attributes retain the verified capture, numerical, alias, effect, policy,
    and provenance contracts until a later legalizer structurally consumes
    them. Alignment and no-alias attributes are required preconditions, not
    optimizer facts; a later lowering must prove or guard them before mutating
    the destination.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    AnyRankedTensor:$output,
    StrAttr:$site_id,
    DictionaryAttr:$origin,
    TypeAttr:$accumulation_type,
    DictionaryAttr:$lhs_semantics,
    DictionaryAttr:$rhs_semantics,
    DictionaryAttr:$output_semantics,
    ArrayAttr:$semantic_requirements,
    ArrayAttr:$aliasing,
    DictionaryAttr:$effects,
    StrAttr:$synchronization,
    DictionaryAttr:$policy,
    DictionaryAttr:$numerical,
    DictionaryAttr:$provenance
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `outs` `(` $output `:` type($output) `)` attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputMutable();
    }
  }];
  let hasVerifier = 1;
}
