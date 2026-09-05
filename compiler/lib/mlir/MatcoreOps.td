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

def Matcore_MapOp : Matcore_Op<"map", [
    IsolatedFromAbove,
    Pure,
    SingleBlock
  ]> {
  let summary = "functional elementwise mapping over a closed tensor domain";
  let description = [{
    Applies the scalar computation in `body` to active elements of `input`.
    The result is a distinct functional tensor value; inactive elements are
    preserved from `input` for partial domains. The exact, versioned `domain`
    dictionary is one of `all`, `slice`, `indices`, or `mask`. A mask domain
    alone carries the optional ranked i1 tensor operand.

    The single body block has one scalar f32 argument and must contain only
    pure Matcore scalar operations followed by `mdsl.yield`. The operation is
    isolated from above so arbitrary captures cannot smuggle effects or
    ordering into the elementwise region.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Optional<AnyRankedTensor>:$mask,
    DictionaryAttr:$domain,
    StrAttr:$outside_domain,
    DictionaryAttr:$tensor_contract,
    DictionaryAttr:$effects,
    DictionaryAttr:$numerical,
    DictionaryAttr:$provenance
  );
  let results = (outs AnyRankedTensor:$result);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def Matcore_SinOp : Matcore_Op<"sin", [Pure]> {
  let summary = "scalar f32 sine with an exact numerical contract";
  let description = [{
    Computes sine for one active scalar element. Version 1 requires an exact
    correctly-rounded, non-approximate f32 contract, including NaN, infinity,
    signed-zero, rounding, exception, and subnormal behavior.
  }];

  let arguments = (ins F32:$input, DictionaryAttr:$numerical,
                       DictionaryAttr:$provenance);
  let results = (outs F32:$result);
  let hasVerifier = 1;
}

def Matcore_YieldOp : Matcore_Op<"yield", [
    Pure,
    Terminator,
    HasParent<"MapOp">
  ]> {
  let summary = "yield the active scalar result of mdsl.map";
  let arguments = (ins F32:$value);
  let hasVerifier = 1;
}

// These operations describe only the bounded source-connected two-GEMM
// inspection. Their unscoped effects deliberately make no pointer-disjointness
// promise from distinct descriptor SSA values. No executable lowering exists.
def Matcore_RegionBeginOp : Matcore_Op<"region_begin", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let results = (outs AnyType:$order);
  let hasVerifier = 1;
}

def Matcore_RegionGuardOp : Matcore_Op<"region_guard", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyType:$order, AnyType:$lhs, AnyType:$rhs,
                       AnyType:$output, I64Attr:$stage,
                       DictionaryAttr:$semantic_contract,
                       ArrayAttr:$required_guards);
  let results = (outs AnyType:$checked);
  let hasVerifier = 1;
}

def Matcore_RegionReadOp : Matcore_Op<"region_read", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyType:$descriptor, AnyType:$checked,
                       I64Attr:$stage, StrAttr:$role);
  let results = (outs AnyRankedTensor:$value);
  let hasVerifier = 1;
}

def Matcore_RegionCommitOp : Matcore_Op<"region_commit", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyRankedTensor:$value, AnyType:$descriptor,
                       AnyType:$checked, I64Attr:$stage,
                       StrAttr:$failure_behavior);
  let results = (outs AnyRankedTensor:$committed, AnyType:$order);
  let hasVerifier = 1;
}

def Matcore_RegionEndOp : Matcore_Op<"region_end", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyType:$order);
  let hasVerifier = 1;
}
