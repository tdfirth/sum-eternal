# Chapter 2: Knee Deep in the Indices — Teaching Guide

## Overview

This chapter covers matrix rearrangements and reductions. Students learn:
- Using the same index twice for diagonal access
- Partial reductions (sum rows vs columns)
- The Frobenius norm as self-contraction

## Functions

### 2.1 transpose

**What it teaches**: Output index order determines result arrangement.

**The pattern**: `'ij->ji'`

**Explanation**: Simply swap the order of indices in the output. No contraction, just rearrangement.

**Socratic questions**:
- "If the input has shape (3, 5), what shape should the transpose have?"
- "What happens if you just swap the order of indices on the right side of the arrow?"

**Common mistakes**:
- Overcomplicating it — this is the simplest possible pattern

**Solution**:
```python
def transpose(M):
    return jnp.einsum('ij->ji', M)
```

### 2.2 trace

**What it teaches**: Using the same index twice to select diagonal elements.

**The pattern**: `'ii->'`

**Explanation**: When the same index appears twice in an input, it selects diagonal elements (where row index = column index). No output index means sum them all.

**Socratic questions**:
- "What does M[i,i] mean for different values of i?"
- "If we used 'ij->', what would happen?"

**Solution**:
```python
def trace(M):
    return jnp.einsum('ii->', M)
```

### 2.3 diag_extract

**What it teaches**: Keeping the diagonal as a vector.

**The pattern**: `'ii->i'`

**Explanation**: Same diagonal selection as trace, but keeping the index in output preserves the elements as a vector instead of summing.

**Socratic questions**:
- "What's the difference between `'ii->'` and `'ii->i'`?"
- "If the matrix is 5x5, what shape is the extracted diagonal?"

**Common mistakes**:
- Confusing with trace (which sums)

**Solution**:
```python
def diag_extract(M):
    return jnp.einsum('ii->i', M)
```

### 2.4 sum_rows

**What it teaches**: Selective reduction — sum one dimension, keep the other.

**The pattern**: `'ij->i'`

**Explanation**:
- `j` disappears → sum over columns
- `i` remains → keep rows separate
- Result: one value per row

**Socratic questions**:
- "If `j` is the column index and it disappears, what are we summing over?"
- "For a 3x4 matrix, what shape would `'ij->i'` produce?"

**Common mistakes**:
- Confusing row vs column (common!)
- `'ij->j'` would be column sums

**Solution**:
```python
def sum_rows(M):
    return jnp.einsum('ij->i', M)
```

### 2.5 sum_cols

**What it teaches**: The complement to sum_rows.

**The pattern**: `'ij->j'`

**Explanation**:
- `i` disappears → sum over rows
- `j` remains → keep columns separate
- Result: one value per column

**Socratic questions**:
- "How is this different from sum_rows? Which index survives?"
- "If you wanted one number per column, which index should stay in the output?"

**Solution**:
```python
def sum_cols(M):
    return jnp.einsum('ij->j', M)
```

### 2.6 frobenius_norm_sq

**What it teaches**: Self-contraction for computing norms.

**The pattern**: `'ij,ij->'`

**Explanation**:
- Same indices on both inputs → element-wise alignment (M with itself)
- No output indices → sum everything
- Result: sum of M[i,j]² for all i,j

**Socratic questions**:
- "What do you get when you element-wise multiply a matrix with itself?"
- "After squaring each element, what happens when all indices disappear from the output?"

**Common mistakes**:
- Trying to do element-wise square differently
- Forgetting that einsum can take the same array twice

**Solution**:
```python
def frobenius_norm_sq(M):
    return jnp.einsum('ij,ij->', M, M)
```

## Completion Message

*"MATRIX OPERATIONS ONLINE. A 2D grid flickers into existence. You're starting to see the shape of things. Chapter 2 complete."*

## Teaching Tips

1. **The diagonal insight**: `ii` is the key pattern — make sure they understand this
2. **Row vs column confusion**: Use concrete examples: "sum_rows means each row becomes one number"
3. **Frobenius as warm-up**: This pattern (`'...,ij,ij->...'`) appears a lot in ML

## Visual Aid

For a 3×3 matrix:
```
    j=0  j=1  j=2
i=0 [a    b    c ]  ← sum_rows[0] = a+b+c
i=1 [d    e    f ]  ← sum_rows[1] = d+e+f
i=2 [g    h    i ]  ← sum_rows[2] = g+h+i
     ↓    ↓    ↓
    sum_cols = [a+d+g, b+e+h, c+f+i]

diagonal (ii): [a, e, i]
trace: a + e + i
```
