# Chapter 3: The Slaughter Batch — Teaching Guide

## Overview

This is the crucial chapter. Students learn to **think in batches** — the key insight for efficient numerical computing. No loops. Process all data simultaneously.

## The Key Insight

Traditional thinking: "For each item, do operation X"
Batch thinking: "Do operation X to all items at once"

Einsum makes this natural: the batch index just becomes another letter.

## Functions

### 3.1 batch_vector_sum

**What it teaches**: Batch operations preserve the batch dimension.

**The pattern**: `'bi->b'`

**Explanation**:
- `b` is the batch dimension (which vector)
- `i` is the element dimension (position in vector)
- `i` disappears → sum each vector
- `b` remains → keep results separate per batch

**Common mistakes**:
- `'bi->'` (sums everything into one number)
- Writing a loop (defeats the purpose!)

**Solution**:
```python
def batch_vector_sum(batch):
    return jnp.einsum('bi->b', batch)
```

### 3.2 batch_dot_pairwise

**What it teaches**: Pairwise operations within batches.

**The pattern**: `'bi,bi->b'`

**Explanation**:
- Same `b` on both inputs → corresponding batches paired
- Same `i` on both inputs → element-wise alignment
- Only `b` in output → sum over `i` for each batch

**Common mistakes**:
- Confusing with all-pairs (next function)

**Solution**:
```python
def batch_dot_pairwise(a, b):
    return jnp.einsum('bi,bi->b', a, b)
```

### 3.3 batch_magnitude_sq

**What it teaches**: Self-pairing for norms.

**The pattern**: `'bi,bi->b'`

**Explanation**: Same as batch_dot_pairwise, but with the same array twice. Computes ||v||² for each vector.

**Solution**:
```python
def batch_magnitude_sq(v):
    return jnp.einsum('bi,bi->b', v, v)
```

### 3.4 all_pairs_dot

**What it teaches**: The "outer product" of batches — every combination.

**The pattern**: `'id,jd->ij'`

**Explanation**:
- Different batch indices (`i` vs `j`) → no pairing, all combinations
- Same feature index (`d`) → contract over features
- Output `ij` → n×m result matrix

This is the foundation of attention mechanisms, distance matrices, etc.

**Common mistakes**:
- Using same index for both batches (gives pairwise)
- Getting output shape wrong

**Solution**:
```python
def all_pairs_dot(a, b):
    return jnp.einsum('id,jd->ij', a, b)
```

### 3.5 batch_matrix_vector

**What it teaches**: Broadcasting a matrix over a batch of vectors.

**The pattern**: `'ij,bj->bi'`

**Explanation**:
- Matrix `M` has no batch dimension — applies same to all
- Vectors have batch `b`
- Contract over `j` (matrix columns, vector elements)
- Result: transformed vectors, still batched

**Common mistakes**:
- Index order confusion (`'ij,bj->bi'` not `'ij,jb->ib'`)

**Solution**:
```python
def batch_matrix_vector(M, batch):
    return jnp.einsum('ij,bj->bi', M, batch)
```

### 3.6 batch_outer

**What it teaches**: Batched outer products.

**The pattern**: `'bi,bj->bij'`

**Explanation**:
- Same `b` on both → corresponding batches paired
- Different vector indices (`i`, `j`) → outer product
- All three in output → 3D result

**Solution**:
```python
def batch_outer(a, b):
    return jnp.einsum('bi,bj->bij', a, b)
```

## Completion Message

*"BATCH PROCESSING UNLOCKED. You see the world differently now. Not one at a time. All at once. A player appears on the map. Rays emanate into the void. Chapter 3 complete."*

## Teaching Tips

1. **The loop-free mindset**: Emphasize that loops are the old way. Batches are the way.
2. **Shape intuition**: Always predict output shape before running
3. **all_pairs is key**: This pattern (`'id,jd->ij'`) is everywhere in ML

## Visual Aid

```
Batch of 3 vectors, length 4:
batch = [[a, b, c, d],     # batch 0
         [e, f, g, h],     # batch 1
         [i, j, k, l]]     # batch 2

'bi->b' gives: [a+b+c+d, e+f+g+h, i+j+k+l]

All-pairs between 2 sets of 2D vectors:
a = [[1, 0],     b = [[0, 1],
     [0, 1]]          [1, 1]]

'id,jd->ij' gives:
[[1*0+0*1, 1*1+0*1],    = [[0, 1],
 [0*0+1*1, 0*1+1*1]]       [1, 1]]
```

## Why This Matters for Sum Eternal

After this chapter, they can process **all rays at once**:
- 320 rays, 8 walls → compute all 2560 intersections simultaneously
- No for loops, no iteration, pure einsum
