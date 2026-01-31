# Chapter 6: Infernal Projection — Teaching Guide

## Overview

This chapter transforms raw ray distances into the rendered 3D view. Students learn about fisheye correction, perspective projection, and distance-based shading.

## Functions

### 6.1 fisheye_correct

**What it teaches**: Why raw raycasting looks distorted and how to fix it.

**The problem**: Rays at the edges of the FOV travel further to hit walls at the same perpendicular distance, causing a "fisheye" or barrel distortion.

**The fix**: Multiply distance by cos(ray_angle - player_angle).

**Socratic questions**:
- "If a wall is 10 units in front of you, but you measure along a diagonal ray, is the measured distance longer or shorter than 10?"
- "What's cos(0)? What does that mean for the center ray?"
- "What's cos(30 degrees)? How does that affect edge rays?"

**Not einsum**: Simple element-wise operations.

**Solution**:
```python
def fisheye_correct(dists, angles, player_angle):
    return dists * jnp.cos(angles - player_angle)
```

**Why it works**: The correction projects the distance onto the view plane perpendicular to the player's facing direction.

### 6.2 distance_to_height

**What it teaches**: Perspective projection — closer = taller.

**The formula**: `height = screen_height / distance`

**Socratic questions**:
- "If a wall is twice as far, how tall should it appear on screen?"
- "What happens as distance approaches zero? How do we handle that?"
- "Why is it screen_height / distance and not distance / screen_height?"

**Solution**:
```python
def distance_to_height(dists, screen_h):
    return screen_h / (dists + 1e-10)  # epsilon to avoid div by zero
```

### 6.3 shade_by_distance

**What it teaches**: Distance-based fog effect using einsum for broadcasting.

**The pattern**: `'rc,r->rc'` (or use broadcasting)

**Socratic questions**:
- "If something is at max_dist, what should its shade factor be? What about at distance 0?"
- "Why do we clamp to [0.2, 1.0] instead of [0, 1]?"
- "Colors have shape (r, 3) and shades have shape (r,). How do we multiply them?"

**The approach**:
1. Compute shade factor: `1 - (dist / max_dist)`, clamped to [0.2, 1.0]
2. Multiply each color component by the factor

**Solution**:
```python
def shade_by_distance(colors, dists, max_dist):
    # Compute shade factor for each ray
    shade = jnp.clip(1.0 - dists / max_dist, 0.2, 1.0)
    # Apply to all color channels
    return jnp.einsum('rc,r->rc', colors, shade)
    # or: return colors * shade[:, None]
```

### 6.4 build_column_masks

**What it teaches**: Building 2D masks with broadcasting.

**Not einsum**: Uses comparison broadcasting.

**Socratic questions**:
- "If a wall has height 100 and the screen is 480 pixels tall, what are the top and bottom row numbers?"
- "How do we create a boolean array where each cell says 'is this row within the wall height'?"
- "Why use broadcasting instead of a loop?"

**The approach**:
1. Compute top and bottom row for each column
2. Create row indices
3. Compare to create boolean mask

**Solution**:
```python
def build_column_masks(heights, screen_h):
    # Clamp heights to screen bounds
    heights = jnp.clip(heights, 0, screen_h)

    # Top and bottom rows for each column
    half_heights = heights / 2
    top = (screen_h / 2 - half_heights).astype(int)
    bottom = (screen_h / 2 + half_heights).astype(int)

    # Row indices: (screen_h, 1) for broadcasting
    rows = jnp.arange(screen_h)[:, None]

    # Mask: True where top <= row < bottom
    mask = (rows >= top) & (rows < bottom)

    return mask
```

## Completion Message

*"PROJECTION COMPLETE. Light and shadow bow to your indices. Walls have depth. Distance fades into darkness. This is starting to look like a game. Chapter 6 complete."*

## Teaching Tips

1. **Fisheye is subtle**: Show before/after if possible
2. **Perspective is intuitive**: "Closer things look bigger"
3. **Shading adds atmosphere**: The fog effect makes the 3D feel real

## Visual Reference

```
Without fisheye correction:    With correction:
   _____                          |   |
  /     \                         |   |
 /       \                        |   |
|         |                       |   |
Walls curve outward               Walls are straight
```

## Why This Matters

After this chapter:
- Walls render with correct perspective
- No more fisheye distortion
- Distance fog adds depth perception
- The raycaster looks polished
