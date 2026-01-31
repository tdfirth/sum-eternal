# Chapter 5: Total Intersection — Additional Notes

## Alternative Implementations

### Using Pure Einsum for Cross Products

The all_pairs_cross_2d can be thought of as:
```python
# cross(a, b) = a_x * b_y - a_y * b_x
# For all pairs, we need a_x[r] * b_y[w] - a_y[r] * b_x[w]

def all_pairs_cross_2d(a, b):
    term1 = jnp.einsum('r,w->rw', a[:, 0], b[:, 1])  # a_x * b_y
    term2 = jnp.einsum('r,w->rw', a[:, 1], b[:, 0])  # a_y * b_x
    return term1 - term2
```

### Using Broadcasting Instead

Many of these operations can use NumPy broadcasting:
```python
def all_pairs_cross_2d(a, b):
    # a is (r, 2), b is (w, 2)
    # a[:, 0:1] is (r, 1), b[:, 1] is (w,)
    # Broadcasting: (r, 1) * (w,) = (r, w)
    return a[:, 0:1] * b[:, 1] - a[:, 1:2] * b[:, 0]
```

Both approaches are valid. Einsum makes the intent clearer; broadcasting is more concise.

## Handling Edge Cases

### Parallel Rays and Walls

When det ≈ 0, the ray and wall are parallel. Handle with:
```python
# Instead of dividing by det directly:
t = jnp.where(jnp.abs(det) > 1e-10, numerator / det, jnp.inf)
```

### Numerical Stability

For very large or very small values:
```python
# Use sign-safe division
sign = jnp.sign(det)
safe_det = jnp.maximum(jnp.abs(det), 1e-10) * jnp.where(sign == 0, 1, sign)
t = numerator / safe_det
```

## Testing Strategy

### Unit Tests for Single Intersection
```python
def test_perpendicular_intersection():
    # Ray from (0, 0) pointing east
    # Wall at x=5, vertical
    player = jnp.array([0.0, 0.0])
    ray_dirs = jnp.array([[1.0, 0.0]])
    wall_starts = jnp.array([[5.0, -10.0]])
    wall_dirs = jnp.array([[0.0, 20.0]])

    t = ray_wall_t_values(...)
    assert t[0, 0] == 5.0  # Distance to wall

    s = ray_wall_s_values(...)
    assert s[0, 0] == 0.5  # Hits middle of wall
```

### Batch Tests
```python
def test_multiple_rays_multiple_walls():
    # Fan of rays against box of walls
    # Verify shape of output
    # Verify values for known configurations
```

## Visual Verification

After implementing, the game should show:
- Walls at correct distances
- No walls behind player
- Proper perspective (closer = taller)

If something looks wrong, add debug output:
```python
print(f"Ray 160 (center): min_t = {distances[160]:.2f}")
print(f"Number of valid intersections: {jnp.sum(valid)}")
```
