# Einsum Overview

## What is Einsum?

Einsum (Einstein summation) is a compact notation for expressing tensor operations. It comes from physics but is incredibly useful in machine learning and scientific computing.

The key insight: **indices that appear on the input but not the output get summed over**.

## The Einsum String

An einsum string has two parts separated by `->`:
- **Left side**: indices for each input array, comma-separated
- **Right side**: indices for the output

```python
jnp.einsum('ij,jk->ik', A, B)  # Matrix multiplication
#           ^^  ^^  ^^
#           A   B   output
```

## Rules

1. **Each index is a single letter** (usually i, j, k, etc.)
2. **Matching letters must have matching dimensions**
3. **Letters on input but not output → summed away**
4. **Letters on output → preserved in result**

## Common Patterns

### Vector Operations

| Operation | Einsum | Explanation |
|-----------|--------|-------------|
| Sum all | `'i->'` | i disappears → sum over i |
| Element-wise multiply | `'i,i->i'` | same i, keep i → element-wise |
| Dot product | `'i,i->'` | same i, no output → sum products |
| Outer product | `'i,j->ij'` | different indices → 2D grid |

### Matrix Operations

| Operation | Einsum | Explanation |
|-----------|--------|-------------|
| Transpose | `'ij->ji'` | swap index order |
| Matrix-vector | `'ij,j->i'` | j summed, i kept |
| Matrix-matrix | `'ij,jk->ik'` | j summed (contracted) |
| Trace | `'ii->'` | diagonal elements summed |
| Extract diagonal | `'ii->i'` | diagonal elements kept |
| Row sums | `'ij->i'` | j summed for each i |
| Column sums | `'ij->j'` | i summed for each j |

### Batch Operations

| Operation | Einsum | Explanation |
|-----------|--------|-------------|
| Batch sum | `'bi->b'` | sum each vector in batch |
| Batch dot | `'bi,bi->b'` | pairwise dots |
| All-pairs dot | `'id,jd->ij'` | every i with every j |
| Batch matrix-vec | `'ij,bj->bi'` | same matrix, many vectors |
| Batch outer | `'bi,bj->bij'` | outer product per batch |

## The Mental Model

Think of einsum as a recipe:
1. **Align** inputs by matching indices
2. **Multiply** aligned elements
3. **Sum** over indices not in output

Example: `'ij,jk->ik'`
- For each output position (i, k):
  - Multiply A[i, j] * B[j, k] for all j
  - Sum over j

## Why JAX?

We use `jax.numpy.einsum` because:
- Clean syntax (identical to numpy)
- Forces explicit thinking about operations
- No hidden broadcasting surprises
- JIT compilation for performance
- Same patterns work in numpy/torch

## Common Mistakes

1. **Shape mismatch**: Indices with same letter must have same size
2. **Missing contraction**: Forgot to remove an index from output
3. **Wrong order**: Output indices in wrong order
4. **Loops**: Don't write loops — einsum handles the iteration

## Debugging Tips

1. **Print shapes**: `print(a.shape, b.shape)`
2. **Try small examples**: 2×2 matrices, length-3 vectors
3. **Work by hand**: Write out what the operation does
4. **Check dimensions**: Count indices on each side
