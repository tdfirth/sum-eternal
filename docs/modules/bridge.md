# Module: bridge.py

## Purpose

Connects student solutions to the game engine. Handles NotImplementedError gracefully and provides progress detection.

## Location

`src/sum_eternal/bridge.py`

## Public Interface

### Enums

```python
class Progress(IntEnum):
    """Tracks how far the student has progressed."""
    NOTHING = 0
    CHAPTER_1_COMPLETE = 1  # Basic ops - debug view
    CHAPTER_2_COMPLETE = 2  # Matrix ops - 2D map
    CHAPTER_3_COMPLETE = 3  # Batch ops - player + rays
    CHAPTER_4_COMPLETE = 4  # Ray generation - ray fan
    CHAPTER_5_COMPLETE = 5  # Ray-wall intersection - 3D!
    CHAPTER_6_COMPLETE = 6  # Projection - full shading
    CHAPTER_7_COMPLETE = 7  # Einstein math - sprites
    CHAPTER_8_COMPLETE = 8  # Combat - game complete
    CHAPTER_9_COMPLETE = 9  # Nightmare - textures
```

### Functions

```python
def check_progress() -> Progress:
    """
    Determine progress by testing each function.

    Runs smoke tests with minimal inputs.
    Returns the highest complete chapter.
    """

def has_raycasting_functions() -> bool:
    """Check if all raycasting functions are available."""

def cast_all_rays(
    player_pos: tuple[float, float],
    player_angle: float,
    fov: float,
    num_rays: int,
    wall_data: dict
) -> tuple[list[float], list[tuple[int, int, int]]]:
    """
    Cast all rays using student implementations.

    Returns: (distances, colors) for each ray
    """
```

## Implementation Approach

### Progress Detection

```python
def check_progress() -> Progress:
    # Chapter 1
    try:
        from solutions.c01_first_blood import vector_sum, ...
        if _safe_call(vector_sum, jnp.array([1.0, 2.0, 3.0])) is None:
            return Progress.NOTHING
        # ... test each function
    except (ImportError, Exception):
        return Progress.NOTHING

    # Chapter 2
    try:
        from solutions.c02_knee_deep_in_the_indices import transpose, ...
        # ... test each function
    except (ImportError, Exception):
        return Progress.CHAPTER_1_COMPLETE

    # ... continue for each chapter

    return Progress.CHAPTER_9_COMPLETE  # All complete
```

### Safe Function Calls

```python
def _safe_call(func: Callable, *args, **kwargs) -> Any | None:
    """Call function, return None if it fails."""
    try:
        result = func(*args, **kwargs)
        if result is None:
            return None
        return result
    except NotImplementedError:
        return None
    except Exception:
        return None
```

### Raycasting Pipeline

```python
def cast_all_rays(player_pos, player_angle, fov, num_rays, wall_data):
    try:
        from solutions.c04_rip_and_trace import angles_to_directions
        from solutions.c05_total_intersection import ray_wall_t_values, ray_wall_s_values

        # Generate ray directions
        ray_angles = jnp.linspace(player_angle - fov/2, player_angle + fov/2, num_rays)
        ray_dirs = angles_to_directions(ray_angles)

        # Compute intersections
        player = jnp.array(player_pos)
        t_values = ray_wall_t_values(player, ray_dirs, wall_starts, wall_dirs)
        s_values = ray_wall_s_values(player, ray_dirs, wall_starts, wall_dirs)

        # Find nearest valid intersection
        valid = (t_values > 0.001) & (s_values >= 0) & (s_values <= 1)
        t_values = jnp.where(valid, t_values, jnp.inf)
        nearest_idx = jnp.argmin(t_values, axis=1)
        distances = t_values[jnp.arange(num_rays), nearest_idx]

        # Get wall colors
        colors = [wall_colors[int(idx)] for idx, dist in zip(nearest_idx, distances)]

        return list(distances), colors

    except Exception:
        # Fallback
        return [10.0] * num_rays, [(100, 100, 100)] * num_rays
```

## Smoke Test Inputs

Each function has a minimal test input:

```python
# Chapter 1
vector_sum: jnp.array([1.0, 2.0, 3.0]) -> 6.0
element_multiply: (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])) -> [3.0, 8.0]
dot_product: (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])) -> 11.0
# ... etc

# Chapter 5
cross_2d: (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])) -> 1.0
# ... etc
```

## Dependencies

- `jax.numpy`: Array operations for smoke tests
- `solutions.*`: Student implementations

## Error Handling

| Error | Handling |
|-------|----------|
| NotImplementedError | Return None, progress limited |
| ImportError | Return current progress level |
| Any Exception | Caught, return None |
| Raycasting failure | Return fallback distances/colors |

## Design Decisions

1. **Smoke tests over inspection**: Actually calling functions is more reliable than checking if they raise NotImplementedError
2. **Linear progression**: Can't skip chapters; must complete in order
3. **Fail-safe defaults**: Game always renders something, even if solutions fail
4. **Fresh imports**: Modules cleared before each check to pick up changes

## Testing Strategy

```python
def test_progress_nothing():
    """Verify NOTHING returned when no functions implemented."""

def test_progress_incremental():
    """Verify progress advances as functions are implemented."""

def test_safe_call_handles_errors():
    """Verify _safe_call catches all expected exceptions."""

def test_cast_all_rays_fallback():
    """Verify fallback works when student code fails."""
```
