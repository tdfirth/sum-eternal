# Chapter 5: Total Intersection — Teaching Guide

## Overview

This is the heart of the raycaster. Students learn to compute ray-wall intersections using Cramer's rule, processing all rays against all walls simultaneously.

**Completing this chapter unlocks the 3D view.**

## The Math

### Ray-Line Intersection

Ray: `P + t*D` (point P, direction D, parameter t)
Line: `A + s*W` (point A, direction W, parameter s)

At intersection: `P + t*D = A + s*W`

Rearranging as a linear system:
```
D_x * t - W_x * s = A_x - P_x
D_y * t - W_y * s = A_y - P_y
```

### Cramer's Rule

For system `[a b; c d] * [t; s] = [e; f]`:
```
det = ad - bc
t = (ed - bf) / det
s = (af - ec) / det
```

For our system:
```
det = D_x * W_y - D_y * W_x  (2D cross product!)
t = ((A - P) × W) / det
s = ((A - P) × D) / det
```

### Valid Intersection

- `t > 0`: Wall is in front of player
- `0 ≤ s ≤ 1`: Intersection is within wall segment
- If `det ≈ 0`: Ray and wall are parallel (no intersection)

## Functions

### 5.1 cross_2d

**What it teaches**: The 2D cross product (foundation for det).

**Not einsum**: Simple arithmetic.

**Solution**:
```python
def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]
```

### 5.2 batch_cross_2d

**What it teaches**: Batched 2D cross products for paired vectors.

**The approach**: Component-wise operations on arrays.

**Solution**:
```python
def batch_cross_2d(a, b):
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
```

### 5.3 all_pairs_cross_2d

**What it teaches**: Cross product for every (ray, wall) pair.

**The key insight**: Use einsum's broadcasting for outer-product-like structure.

**Solution**:
```python
def all_pairs_cross_2d(a, b):
    # a is (r, 2), b is (w, 2)
    # result is (r, w) where result[i,j] = cross(a[i], b[j])
    return jnp.einsum('r,w->rw', a[:, 0], b[:, 1]) - jnp.einsum('r,w->rw', a[:, 1], b[:, 0])
    # or using broadcasting:
    # return a[:, 0:1] * b[:, 1:2].T - a[:, 1:2] * b[:, 0:1].T
```

### 5.4 ray_wall_determinants

**What it teaches**: The determinants are just cross products of directions.

**Same as all_pairs_cross_2d**:
```python
def ray_wall_determinants(ray_dirs, wall_dirs):
    return all_pairs_cross_2d(ray_dirs, wall_dirs)
```

### 5.5 ray_wall_t_values

**What it teaches**: Computing t (distance along ray) for all intersections.

**The formula**: `t = ((A - P) × W) / det`

**Solution**:
```python
def ray_wall_t_values(player, ray_dirs, wall_starts, wall_dirs):
    # (A - P) for each wall: shape (w, 2)
    diff = wall_starts - player

    # cross((A-P), W) for each wall: shape (w,)
    numerator = diff[:, 0] * wall_dirs[:, 1] - diff[:, 1] * wall_dirs[:, 0]

    # det for each (ray, wall): shape (r, w)
    det = ray_wall_determinants(ray_dirs, wall_dirs)

    # Broadcast numerator to (r, w) and divide
    # Add small epsilon to avoid division by zero
    return numerator[None, :] / (det + 1e-10)
```

### 5.6 ray_wall_s_values

**What it teaches**: Computing s (position along wall) for all intersections.

**The formula**: `s = ((A - P) × D) / det`

**Solution**:
```python
def ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs):
    # (A - P) for each wall: shape (w, 2)
    diff = wall_starts - player

    # cross((A-P), D) for each (ray, wall): shape (r, w)
    # This requires einsum for the all-pairs structure
    numerator = (jnp.einsum('r,w->rw', ray_dirs[:, 0], diff[:, 1]) -
                 jnp.einsum('r,w->rw', ray_dirs[:, 1], diff[:, 0]))

    # det for each (ray, wall): shape (r, w)
    det = ray_wall_determinants(ray_dirs, wall_dirs)

    return numerator / (det + 1e-10)
```

## Completion Message

*"3D RENDERING ONLINE. You see. For the first time, you truly see. The walls rise before you. The arena takes shape. This is the heart of raycasting. Chapter 5 complete."*

## Teaching Tips

1. **Work through the math**: Don't skip Cramer's rule explanation
2. **Start with single ray/wall**: Before batching, understand one intersection
3. **Draw pictures**: Ray from player, wall segment, intersection point
4. **Test edge cases**: Parallel ray/wall (det=0), ray pointing away (t<0)

## Why This Matters

After this chapter:
- The 3D view activates
- Walls render based on computed distances
- The raycaster is functionally complete

This is the payoff for all the einsum learning. The 2D arena becomes a 3D world.

## Debugging Tips

If walls look wrong:
1. Check t values are positive (in front of player)
2. Check s values are in [0, 1] (within wall segment)
3. Check det isn't too close to zero (parallel case)
4. Verify wall coordinate system matches expected layout
