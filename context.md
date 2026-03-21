# cuTile / Tile IR Context for MatCore

## What cuTile shows

cuTile is documented as a Python-based DSL for NVIDIA GPUs. The key pattern is that a kernel is declared with a decorator, and the host launches that kernel separately. The kernel body is not normal host Python control flow; it is source that the compiler/runtime uses to build GPU work.

The docs describe `@ct.kernel` as the kernel entry-point marker, and state that kernels cannot be called directly from host code. The host queues execution with `ct.launch(...)`. That means the decorator is used to identify and capture kernel intent, while `ct.launch` is the actual execution boundary.

Example pattern from the docs:

```python
import cuda.tile as ct
import cupy

TILE_SIZE = 16

@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

def vector_add(a: cupy.ndarray, b: cupy.ndarray, result: cupy.ndarray):
    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
```

This same execution model is reinforced in the execution-model docs, which define tile kernels as entry points for tile code and describe execution spaces such as host and tile code. That is the main precedent MatCore can borrow from.

## Load, store, and matmul

cuTile uses explicit array/tile movement through `ct.load()` and `ct.store()`. The documentation also shows that tile operations can be combined with arithmetic and matrix multiply.

Load/store example from the performance docs:

```python
import cuda.tile as ct

TILE_SIZE = 16

@ct.kernel
def load_store_with_hints_kernel(x, y):
    bid = ct.bid(0)
    tx = ct.load(
        x,
        index=(bid,),
        shape=(TILE_SIZE,),
        latency=8,
    )
    ct.store(
        y,
        index=(bid,),
        tile=tx,
        latency=2,
        allow_tma=False,
    )
```

Matmul example from the generated operation docs:

```python
import cuda.tile as ct

tx = ct.full((2, 4), 3, dtype=ct.float32)
ty = ct.full((4, 8), 4, dtype=ct.float32)

tz = ct.matmul(tx, ty)
tz2 = tx @ ty
```

The same page also documents `ct.mma(x, y, acc)` for fused matrix-multiply-accumulate. This is useful for MatCore because it shows that cuTile distinguishes between a plain matmul and a fused accumulator form:

```python
acc = ct.full((2, 8), 0, dtype=ct.float32)
tz = ct.mma(tx, ty, acc)
```

## Why MatCore should parse AST instead of using a string DSL

MatCore should follow the cuTile pattern of source-first kernel definition, but implement its own front end by parsing Python source into an AST rather than inventing a separate string language.

The practical reasons are:

1. Python decorators already mark kernel entry points cleanly, so MatCore can preserve the normal Python function shape and use `@mc.kernel` as the capture boundary.
2. AST parsing keeps the syntax aligned with ordinary Python, which means the kernel author writes real Python control flow and expressions instead of inventing a second syntax for loops, indexing, and tensor operations.
3. A string DSL would throw away Python structure too early. With AST nodes, MatCore can preserve loop bounds, call sites, loads, stores, and matmul expressions as compiler input, which is a better fit for MLIR lowering.
4. The cuTile docs already separate the host launch boundary from kernel definition. MatCore can keep that model while replacing cuTile’s backend with a source-to-IR pipeline tailored to MLIR.

In short, the AST approach keeps the user-facing syntax Pythonic while giving MatCore a structured intermediate representation that is much easier to lower, validate, and optimize than free-form strings.

## Sources

- https://docs.nvidia.com/cuda/cutile-python
- https://docs.nvidia.com/cuda/cutile-python/execution.html
- https://docs.nvidia.com/cuda/cutile-python/performance.html
- https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.matmul.html
- https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.mma.html
- https://github.com/NVIDIA/cutile-python
