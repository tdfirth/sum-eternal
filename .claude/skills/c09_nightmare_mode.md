# Chapter 9: Nightmare Mode — Teaching Guide

## Overview

Optional advanced content. These functions add textures to the raycaster. They're significantly harder and not required for completion.

## Functions

### 9.1 texture_column_lookup

**What it teaches**: Mapping continuous coordinates to discrete texture indices.

**The approach**: Simple index calculation with bounds checking.

**Solution**:
```python
def texture_column_lookup(hit_s, tex_width):
    # Map [0, 1] to [0, tex_width-1]
    col = jnp.floor(hit_s * tex_width).astype(int)
    # Clamp to valid range
    return jnp.clip(col, 0, tex_width - 1)
```

### 9.2 bilinear_sample

**What it teaches**: Interpolating between four texel values.

**The math**: Bilinear interpolation uses weights based on fractional position.

**This is complex**:
```
(1-fx)(1-fy) * top_left + fx(1-fy) * top_right +
(1-fx)fy * bottom_left + fx*fy * bottom_right
```

**Solution**:
```python
def bilinear_sample(texture, coords):
    h, w, c = texture.shape
    y, x = coords[:, 0], coords[:, 1]

    # Integer and fractional parts
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    x1 = jnp.minimum(x0 + 1, w - 1)
    y1 = jnp.minimum(y0 + 1, h - 1)

    fx = x - x0
    fy = y - y0

    # Clamp indices
    x0 = jnp.clip(x0, 0, w - 1)
    y0 = jnp.clip(y0, 0, h - 1)

    # Sample four corners
    tl = texture[y0, x0]  # (n, 3)
    tr = texture[y0, x1]
    bl = texture[y1, x0]
    br = texture[y1, x1]

    # Interpolate
    # Using einsum for weighted sum
    w_tl = (1 - fx) * (1 - fy)
    w_tr = fx * (1 - fy)
    w_bl = (1 - fx) * fy
    w_br = fx * fy

    result = (jnp.einsum('nc,n->nc', tl, w_tl) +
              jnp.einsum('nc,n->nc', tr, w_tr) +
              jnp.einsum('nc,n->nc', bl, w_bl) +
              jnp.einsum('nc,n->nc', br, w_br))

    return result
```

### 9.3 floor_cast_coords

**What it teaches**: Computing world coordinates for floor rendering.

**This is the hardest function**. It requires:
1. Perspective projection math
2. Coordinate system transformations
3. Batch operations over entire screen

**Simplified approach**:
```python
def floor_cast_coords(screen_y, player_pos, player_angle):
    # This is a simplified version
    # Full implementation requires more careful handling

    # Assume screen height 480, horizon at 240
    screen_h = 480
    horizon = screen_h // 2

    # Distance to floor at each row (inverse relationship)
    row_offset = screen_y - horizon + 1  # +1 to avoid div by zero
    floor_dist = screen_h / (2 * row_offset)

    # For now, just return placeholder
    # Full implementation would compute actual world coords
    # based on player position, angle, and FOV

    # ... complex trigonometry here ...

    raise NotImplementedError("Full floor casting is very complex")
```

## Completion Message

*"NIGHTMARE CONQUERED. There is nothing more to teach. You have seen the depths of the notation. Go forth. Einsum everything. Master of the notation, you are truly worthy."*

## Teaching Notes

1. **These are genuinely hard**: Don't feel bad about providing more help
2. **Bilinear is common**: Used in all texture sampling
3. **Floor casting is advanced**: Many raycasters skip this entirely

## If They Get Stuck

For bilinear sampling:
- Start with nearest neighbor (no interpolation)
- Then add x interpolation only
- Then add y interpolation

For floor casting:
- Focus on understanding the geometry
- Draw diagrams of the perspective projection
- Accept that this is optional content

## Why Include This?

- Shows einsum scales to complex problems
- Introduces texture mapping concepts
- Provides challenge for advanced users
- Demonstrates bilinear interpolation (ML relevance)
