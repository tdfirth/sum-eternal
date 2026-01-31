# Chapter 7: Spooky Action at a Distance — Teaching Guide

## Overview

This chapter adds the Einsteins (sprites) to the world. Students learn to compute distances, angles, and screen positions for point objects.

**Completing this chapter makes the Einsteins visible.**

## Functions

### 7.1 point_distances

**What it teaches**: Euclidean distance from origin to multiple points.

**Socratic questions**:
- "What's the first step in computing distance between two points?"
- "How does `'ni,ni->n'` compute squared magnitudes for a batch of vectors?"
- "Why compute squared distance first, then sqrt, instead of component-wise?"

**The approach**:
1. Compute difference vectors: `points - origin`
2. Squared magnitudes: `'ni,ni->n'`
3. Square root

**Solution**:
```python
def point_distances(origin, points):
    diff = points - origin
    sq_dist = jnp.einsum('ni,ni->n', diff, diff)
    return jnp.sqrt(sq_dist)
```

### 7.2 all_pairs_distances

**What it teaches**: Distance matrix between two sets of points.

**Socratic questions**:
- "Why use `||a-b||² = ||a||² + ||b||² - 2(a·b)` instead of computing all differences directly?"
- "What einsum patterns do you need for ||a||², ||b||², and a·b?"
- "How do you combine these three terms with the right broadcasting?"

**The approach**: Use the identity `||a-b||² = ||a||² + ||b||² - 2(a·b)`

**Solution**:
```python
def all_pairs_distances(a, b):
    # ||a||² for each point in a: shape (n,)
    a_sq = jnp.einsum('nd,nd->n', a, a)
    # ||b||² for each point in b: shape (m,)
    b_sq = jnp.einsum('md,md->m', b, b)
    # a·b for all pairs: shape (n, m)
    ab = jnp.einsum('nd,md->nm', a, b)
    # ||a-b||² = ||a||² + ||b||² - 2(a·b)
    sq_dist = a_sq[:, None] + b_sq[None, :] - 2 * ab
    return jnp.sqrt(jnp.maximum(sq_dist, 0))  # max to handle numerical issues
```

### 7.3 points_to_angles

**What it teaches**: Converting direction vectors to angles.

**Uses atan2**: Handles all quadrants correctly.

**Socratic questions**:
- "Why use atan2(y, x) instead of atan(y/x)?"
- "What angle does atan2 return for a point directly to the east? North? West?"
- "What's the first step before calling atan2?"

**Solution**:
```python
def points_to_angles(origin, points):
    diff = points - origin
    return jnp.arctan2(diff[:, 1], diff[:, 0])
```

### 7.4 angle_in_fov

**What it teaches**: FOV checking with angle wraparound.

**The tricky part**: Angles wrap at ±π. Need to normalize differences.

**Socratic questions**:
- "If player_angle is 170 degrees and target_angle is -170 degrees, what's the actual angular distance?"
- "How do you normalize an angle difference to [-π, π]?"
- "What does 'within FOV' mean mathematically?"

**Solution**:
```python
def angle_in_fov(angles, player_angle, fov):
    # Compute angle difference
    diff = angles - player_angle
    # Normalize to [-π, π]
    diff = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi
    # Check if within FOV
    return jnp.abs(diff) <= fov / 2
```

### 7.5 project_to_screen_x

**What it teaches**: Converting world angles to screen coordinates.

**The formula**: `x = width/2 + (angle_diff / (fov/2)) * (width/2)`

**Socratic questions**:
- "If something is directly ahead (angle_diff = 0), what's its screen x coordinate?"
- "If something is at the left edge of FOV (angle_diff = -fov/2), what x does it map to?"
- "How does this formula map [-fov/2, +fov/2] to [0, width]?"

**Solution**:
```python
def project_to_screen_x(angles, player_angle, fov, width):
    # Compute normalized angle difference
    diff = angles - player_angle
    diff = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi
    # Map [-fov/2, fov/2] to [0, width]
    return width / 2 + (diff / (fov / 2)) * (width / 2)
```

### 7.6 sprite_scale

**What it teaches**: Size scales inversely with distance.

**Socratic questions**:
- "If base_size is 100 and distance is 2, how big should the sprite appear?"
- "This is the same formula as distance_to_height. Why?"

**Solution**:
```python
def sprite_scale(dists, base_size):
    return base_size / (dists + 1e-10)
```

## Completion Message

*"THE EINSTEINS MANIFEST. They see you. Tongue out, eyes knowing. They are not enemies. They are trials. Prove yourself worthy. Chapter 7 complete."*

## Teaching Tips

1. **atan2 vs atan**: Always use atan2 for direction angles
2. **Angle normalization**: Critical for wraparound at ±π
3. **all_pairs_distances trick**: The identity avoids explicit loops

## The Distance Matrix Trick

For computing `||a - b||²`:
```
||a - b||² = (a - b)·(a - b)
           = a·a - 2(a·b) + b·b
           = ||a||² - 2(a·b) + ||b||²
```

This is faster than computing all pairwise differences explicitly.

## Why This Matters

After this chapter:
- Einsteins appear in the 3D view
- They're rendered at correct positions and sizes
- They move toward the player
- The game feels alive
