# Chapter 4: Rip and Trace — Teaching Guide

## Overview

This chapter generates the rays for raycasting. Students learn to convert angles to directions, apply rotations, and normalize vectors — all in batches.

Not everything is pure einsum here. Some operations combine einsum with regular operations.

## Functions

### 4.1 angles_to_directions

**What it teaches**: Building 2D vectors from angles.

**Not pure einsum**: This uses `jnp.cos`, `jnp.sin`, and `jnp.stack`.

**Socratic questions**:
- "If an angle is 0 (pointing east), what are the x and y components of the unit vector?"
- "What about angle π/2 (pointing north)?"
- "Which trigonometric function gives the x component? Which gives y?"

**The approach**:
```python
cos_vals = jnp.cos(angles)  # shape (r,)
sin_vals = jnp.sin(angles)  # shape (r,)
# Stack into (r, 2) array
```

**Solution**:
```python
def angles_to_directions(angles):
    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    # or: jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
```

### 4.2 rotate_vectors

**What it teaches**: Applying a rotation matrix to a batch of vectors.

**The pattern**: `'ij,nj->ni'`

**Explanation**:
- Build 2×2 rotation matrix from angle
- Apply to each of n vectors
- This is batch matrix-vector multiplication

**Socratic questions**:
- "A 2D rotation matrix is 2x2. The vectors are 2D. How does this relate to `'ij,j->i'` from Chapter 1?"
- "Why is the einsum `'ij,nj->ni'` and not `'ij,jn->in'`? What does the index order mean?"

**The rotation matrix**:
```
R = [[cos(θ), -sin(θ)],
     [sin(θ),  cos(θ)]]
```

**Common mistakes**:
- Building the rotation matrix incorrectly (sign of sin term)
- Index order in einsum

**Solution**:
```python
def rotate_vectors(vecs, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([[c, -s], [s, c]])
    return jnp.einsum('ij,nj->ni', R, vecs)
```

### 4.3 normalize_vectors

**What it teaches**: Computing magnitudes and scaling vectors.

**The approach**:
1. Compute squared magnitudes: `'ni,ni->n'`
2. Take square root
3. Divide each component by magnitude

**Socratic questions**:
- "How do you compute the magnitude (length) of a vector using a dot product?"
- "Once you have the magnitudes, how do you scale each vector to length 1?"
- "What happens if a vector has length 0? How do we handle that?"

**Common mistakes**:
- Forgetting to handle zero-length vectors
- Wrong broadcasting when dividing

**Watch out for**: Zero-length vectors (add epsilon to avoid division by zero).

**Solution**:
```python
def normalize_vectors(v):
    mag_sq = jnp.einsum('ni,ni->n', v, v)
    mag = jnp.sqrt(mag_sq + 1e-10)  # epsilon for safety
    return v / mag[:, None]  # broadcast division
    # or: return jnp.einsum('ni,n->ni', v, 1.0 / mag)
```

### 4.4 scale_vectors

**What it teaches**: Per-vector scaling.

**The pattern**: `'nd,n->nd'`

**Explanation**:
- Each vector has its own scale factor
- Broadcast the scale to each component
- This is element-wise multiplication with broadcasting

**Socratic questions**:
- "If each of 10 vectors needs its own scale factor, what shape is the scales array?"
- "How does `'nd,n->nd'` broadcast the scalar to both components of each vector?"

**Common mistakes**:
- Shape mismatch between vectors and scales

**Solution**:
```python
def scale_vectors(v, scales):
    return jnp.einsum('nd,n->nd', v, scales)
    # or: return v * scales[:, None]
```

## Completion Message

*"RAYS ONLINE. Your vision extends outward. The void awaits your perception. Watch the minimap — rays now emanate from your position. Chapter 4 complete."*

## Teaching Tips

1. **Not everything is einsum**: These functions mix einsum with regular JAX ops
2. **Rotation is important**: The 2D rotation matrix is fundamental
3. **Broadcasting alternative**: Some of these can be done with broadcasting alone
4. **Numerical stability**: Always watch for division by zero

## Why This Matters

After this chapter, they can:
- Generate directions for all rays in one operation
- Rotate the ray fan with player turning
- Normalize direction vectors for consistent ray lengths

The rays now appear on the minimap. The raycaster is taking shape.

## Math Reference

### 2D Rotation Matrix

To rotate vector (x, y) counter-clockwise by angle θ:
```
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]
```

### Unit Vectors

A unit vector has length 1:
- Given v = (x, y)
- Magnitude: ||v|| = √(x² + y²)
- Unit vector: v̂ = v / ||v|| = (x/||v||, y/||v||)
