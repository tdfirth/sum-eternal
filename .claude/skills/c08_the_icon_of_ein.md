# Chapter 8: The Icon of Ein — Teaching Guide

## Overview

This chapter implements combat — the ability to "sum" Einsteins. Students learn ray-target intersection and movement toward a point.

**Completing this chapter enables combat and completes the main game.**

## Functions

### 8.1 project_points_onto_ray

**What it teaches**: Projecting points onto a line (signed distance along ray).

**The math**: For point P and ray from O with direction D:
`t = (P - O) · D`

**The pattern**: `'ni,i->n'`

**Socratic questions**:
- "What does the dot product of two vectors tell you geometrically?"
- "Why do we compute (P - O) first? What does this vector represent?"
- "If the result is negative, what does that mean about the point's position?"

**Solution**:
```python
def project_points_onto_ray(points, origin, direction):
    diff = points - origin
    return jnp.einsum('ni,i->n', diff, direction)
```

### 8.2 perpendicular_distance_to_ray

**What it teaches**: Distance from points to a ray (for hit detection).

**The math**: Use 2D cross product magnitude:
`perp_dist = |cross((P - O), D)|`

**Socratic questions**:
- "The dot product gives distance along the ray. What does the cross product magnitude give?"
- "Why do we take the absolute value of the cross product?"
- "If perpendicular distance is 0, what does that mean geometrically?"

**Solution**:
```python
def perpendicular_distance_to_ray(points, origin, direction):
    diff = points - origin
    # 2D cross product: diff_x * dir_y - diff_y * dir_x
    cross = diff[:, 0] * direction[1] - diff[:, 1] * direction[0]
    return jnp.abs(cross)
```

### 8.3 ray_hits_target

**What it teaches**: Combining conditions for hit detection.

**Three conditions**:
1. Target in front: `proj_dist > 0`
2. Target before wall: `proj_dist < wall_dist`
3. Close enough to ray: `perp_dist < radii`

**Socratic questions**:
- "Why check proj_dist > 0? What would happen if we didn't?"
- "Why does the target need to be closer than the wall?"
- "What does it mean for perpendicular distance to be less than the radius?"

**Solution**:
```python
def ray_hits_target(proj_dist, perp_dist, radii, wall_dist):
    in_front = proj_dist > 0
    before_wall = proj_dist < wall_dist
    close_enough = perp_dist < radii
    return in_front & before_wall & close_enough
```

### 8.4 move_toward_point

**What it teaches**: AI movement using normalization and scaling.

**Socratic questions**:
- "Why do we normalize the direction vector before scaling?"
- "How do speed and dt combine to determine movement distance?"
- "What einsum pattern scales each direction vector by its own speed?"

**The approach**:
1. Compute direction to target
2. Normalize each direction
3. Scale by speed and dt
4. Add to positions

**Solution**:
```python
def move_toward_point(positions, target, speeds, dt):
    # Direction to target for each position
    direction = target - positions  # (n, 2)

    # Normalize
    mag = jnp.sqrt(jnp.einsum('ni,ni->n', direction, direction) + 1e-10)
    normalized = direction / mag[:, None]

    # Scale by speed and dt
    movement = jnp.einsum('nd,n->nd', normalized, speeds * dt)

    return positions + movement
```

## Completion Message

*"THE NOTATION IS YOURS. You have proven yourself against the trials. Einstein nods approvingly, tongue still out. You are worthy. Rip and tensor, complete."*

## Teaching Tips

1. **Projection is fundamental**: Used everywhere in graphics
2. **Hit detection is simple**: Just three boolean conditions
3. **Movement combines skills**: Uses normalization from Chapter 4

## The Hit Detection Geometry

```
        perpendicular distance
        <---->
   O ----+-----> ray direction
         |
         | (proj_dist along ray)
         |
         P (target)

Hit if:
- proj_dist > 0 (P is in front of O)
- proj_dist < wall_dist (P is not behind a wall)
- perp_dist < radius (P is close to ray)
```

## Victory!

After this chapter:
- Space bar sums Einsteins in crosshair
- Einsteins move toward player
- When all are summed: victory screen

The player has mastered einsum through building a complete raycaster.

## Chapter 9 Teaser

*"Nightmare mode awaits. Textures. Bilinear interpolation. Floor casting. These are optional trials for those who seek true mastery."*
