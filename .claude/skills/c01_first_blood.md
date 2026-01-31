# Chapter 1: First Blood — Teaching Guide

## Overview

This chapter introduces the fundamental einsum patterns. Students learn that:
- Indices on input but not output → summed
- Matching indices → element-wise alignment
- Output indices determine result shape

## Functions

### 1.1 vector_sum

**What it teaches**: The simplest contraction — summing away an index.

**The pattern**: `'i->'`

**Explanation**: Index `i` appears on input but not output, so all elements along `i` get summed into a scalar.

**Socratic questions**:
- "What happens to an index that appears on the left but not on the right of the arrow?"
- "If your input has shape (5,) and your output is a scalar, what must have happened to those 5 elements?"

**Common mistakes**:
- Trying `'i->i'` (keeps the vector, doesn't sum)
- Using `jnp.sum()` instead of einsum (works but misses the point)

**Solution**:
```python
def vector_sum(v):
    return jnp.einsum('i->', v)
```

### 1.2 element_multiply

**What it teaches**: Element-wise operations with matching indices.

**The pattern**: `'i,i->i'`

**Explanation**: Same index `i` on both inputs means they're aligned element-wise. Index `i` in output means we keep the result for each position.

**Socratic questions**:
- "Both inputs have index `i` and the output has `i`. What does that tell you about alignment?"
- "What would `'i,j->ij'` produce instead? Why is that different?"

**Common mistakes**:
- `'i,j->ij'` (outer product, not element-wise)
- Not realizing both inputs need the same index

**Solution**:
```python
def element_multiply(a, b):
    return jnp.einsum('i,i->i', a, b)
```

### 1.3 dot_product

**What it teaches**: Contraction — multiply aligned elements, then sum.

**The pattern**: `'i,i->'`

**Explanation**: Same index `i` on both inputs (aligned), no output index (summed). This is the fundamental pattern of contraction.

**Socratic questions**:
- "What if we kept the i in the output? What would we get?"
- "How is this different from element_multiply?"

**Solution**:
```python
def dot_product(a, b):
    return jnp.einsum('i,i->', a, b)
```

### 1.4 outer_product

**What it teaches**: Creating higher-dimensional outputs from lower-dimensional inputs.

**The pattern**: `'i,j->ij'`

**Explanation**: Different indices mean no alignment. Both indices in output mean we create a 2D result with every combination.

**Socratic questions**:
- "If we have two vectors of lengths 3 and 4, what shape should the outer product be?"
- "Why do we use different letters (i, j) instead of the same letter for both inputs?"

**Common mistakes**:
- Using same index for both inputs
- Output index order (doesn't matter for this case, but builds intuition)

**Solution**:
```python
def outer_product(a, b):
    return jnp.einsum('i,j->ij', a, b)
```

### 1.5 matrix_vector_mul

**What it teaches**: Matrix operations as contractions along shared dimensions.

**The pattern**: `'ij,j->i'`

**Explanation**:
- `ij` for matrix (rows i, columns j)
- `j` for vector (elements indexed by j)
- `->i` output has rows only (columns contracted)

**Socratic questions**:
- "Which dimension of the matrix must match the vector length?"
- "The `j` index appears in both inputs but not the output. What happens to it?"

**Common mistakes**:
- Getting row/column indices backward
- `'ij,i->j'` (would contract over rows, not columns)

**Solution**:
```python
def matrix_vector_mul(M, v):
    return jnp.einsum('ij,j->i', M, v)
```

### 1.6 matrix_matrix_mul

**What it teaches**: The classic matrix multiplication pattern.

**The pattern**: `'ij,jk->ik'`

**Explanation**:
- First matrix: rows `i`, columns `j`
- Second matrix: rows `j`, columns `k`
- Shared index `j` gets contracted (summed over)
- Result: rows `i`, columns `k`

**Socratic questions**:
- "For matrix multiplication A @ B, which dimension of A must match which dimension of B?"
- "If A is 3x4 and B is 4x5, what shape is the result and why?"

**Common mistakes**:
- Index order confusion
- Forgetting which index is shared

**Solution**:
```python
def matrix_matrix_mul(A, B):
    return jnp.einsum('ij,jk->ik', A, B)
```

## Completion Message

*"SYSTEMS ONLINE. The notation recognizes you... barely. Chapter 1 complete."*

## Teaching Tips

1. **Start with shapes**: Always have them state input/output shapes first
2. **Build intuition**: Relate to familiar operations (sum, multiply, @)
3. **Use small examples**: 2-element vectors, 2×2 matrices
4. **Celebrate the aha moments**: The contraction insight is key

## If They're Struggling

Ask them to trace through a tiny example by hand:
```
a = [1, 2], b = [3, 4]
einsum('i,i->', a, b) = ?

For i=0: 1 * 3 = 3
For i=1: 2 * 4 = 8
Sum: 3 + 8 = 11
```
