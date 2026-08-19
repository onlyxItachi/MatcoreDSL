# LLVM 20 ↔ 21 ↔ 22 Decision Interface Stability

**Investigation Scope**: Comparative source and interface analysis across LLVM/Clang 20.1.8, 21.1.8, and 22.1.8  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Interface Stability Classification Matrix

| Subsystem & Interface | Semantic Contract | Stability Status (20 ↔ 21 ↔ 22) | Architectural Implication for MatcoreDSL |
| :--- | :--- | :--- | :--- |
| **`TTI::getRegisterBitWidth`** | Returns hardware vector width for register class | **100% Stable** (Exact same signature & semantics) | Safe for long-term target querying. |
| **`TTI::getNumberOfRegisters`**| Returns physical register count | **100% Stable** (Exact same signature & semantics) | Safe for computing register tile unroll upper bounds. |
| **`TTI::getInterleaveCount`**  | Returns execution port interleave factor | **100% Stable** (Exact same signature & semantics) | Safe for estimating instruction-level parallelism. |
| **`RecurrenceDescriptor`**     | Evaluates reduction PHI and `reassoc` flags | **Stable** (Underlying `FastMathFlags` contract intact) | Semantic contract for floating-point reordering is invariant. |
| **`SelectionDAG` Memory Lowering** | Maps `align` to `vmovaps` vs `vmovups` | **Stable** (Target lowering rules intact) | Alignment metadata reliably translates to hardware aligned moves. |
| **`VPlanTransforms`**          | Internal loop vectorization cost recipe | **Internal Churn** (Migrated recipes from legacy cost model) | Internal optimizer machinery; delegate to LLVM without coupling. |
| **Clang LibTooling AST APIs**   | C++ frontend AST matchers & visitors | **High Churn** (Frequent header reorganizations & API deprecations) | **Isolate Frontend**: Keep Clang AST parsing behind an explicit, versioned boundary. |
