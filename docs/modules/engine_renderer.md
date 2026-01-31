# Module: engine/renderer.py

## Purpose

All pygame rendering for every game state. Transforms game data into pixels.

## Location

`src/sum_eternal/engine/renderer.py`

## Public Interface

```python
class Renderer:
    """Main renderer for Sum Eternal."""

    def __init__(self, screen: Surface, assets: Assets) -> None:
        """
        Initialize renderer with display surface and assets.

        Pre-computes ray angles, initializes fonts.
        """

    def render(self, data: GameData, game_map: Map) -> None:
        """
        Render the current frame based on game state.

        Dispatches to state-specific render methods.
        Always renders error overlay if present.
        """
```

## Implementation Approach

### Render Dispatch

```python
def render(self, data: GameData, game_map: Map):
    self.screen.fill(BLACK)

    if data.state == GameState.TITLE:
        self._render_title(data)
    elif data.state == GameState.VICTORY:
        self._render_victory(data)
    else:  # TUTORIAL or NIGHTMARE
        self._render_gameplay(data, game_map)

    if data.error_message:
        self._render_error_overlay(data.error_message)
```

### Gameplay Rendering (Progressive)

```python
def _render_gameplay(self, data: GameData, game_map: Map):
    if data.progress < CHAPTER_1_COMPLETE:
        self._render_waiting(data)
    elif data.progress < CHAPTER_3_COMPLETE:
        self._render_debug_view(data)
    elif data.progress < CHAPTER_5_COMPLETE:
        self._render_2d_map(data, game_map)
    else:
        self._render_3d_view(data, game_map)

    self._render_progress_bar(data)
```

### Debug View (Chapters 1-2)

```
┌─────────────────────────────────────────┐
│  SUM ETERNAL - SYSTEMS INITIALIZING     │
│  ═══════════════════════════════════════│
│                                         │
│  VECTOR OPS: ONLINE                     │
│    sum(v) = 6.0           ✓             │
│    a·b = 32.0             ✓             │
│                                         │
│  MATRIX OPS: ONLINE                     │
│    trace(M) = 15.0        ✓             │
└─────────────────────────────────────────┘
```

### 2D Map View (Chapters 3-4)

```python
def _render_2d_map(self, data: GameData, game_map: Map):
    # Scale world to screen
    scale = min(width, height - 50) / 40
    offset = (width // 2, height // 2)

    # Draw walls as lines
    for wall in game_map.walls:
        pygame.draw.line(screen, wall.color, world_to_screen(wall.start), world_to_screen(wall.end))

    # Draw player
    pygame.draw.circle(screen, CYAN, world_to_screen(player.position), 8)

    # Draw direction indicator
    dir_end = player.position + direction * 20
    pygame.draw.line(screen, CYAN, world_to_screen(player.position), world_to_screen(dir_end))

    # Draw ray fan (if Chapter 4)
    if data.progress >= CHAPTER_4:
        for angle in ray_angles:
            pygame.draw.line(screen, DARK_GRAY, player_screen, ray_end_screen)
```

### 3D Raycasting View (Chapters 5+)

```python
def _render_3d_view(self, data: GameData, game_map: Map):
    # Get distances from bridge (uses student code)
    distances, colors = bridge.cast_all_rays(
        player_pos, player_angle, fov, num_rays, game_map.wall_data
    )

    # Render wall columns
    for i, (dist, color) in enumerate(zip(distances, colors)):
        height = screen_height / dist
        top = (screen_height - height) // 2
        pygame.draw.rect(screen, shaded_color, (i * col_width, top, col_width, height))

    # Render minimap in corner
    self._render_minimap(data, game_map)

    # Render crosshair
    pygame.draw.line(screen, WHITE, (cx - 10, cy), (cx + 10, cy))
    pygame.draw.line(screen, WHITE, (cx, cy - 10), (cx, cy + 10))

    # Render Einsteins (if Chapter 7+)
    if data.progress >= CHAPTER_7:
        self._render_einsteins(data, distances)
```

### Einstein Sprite Rendering

```python
def _render_einsteins(self, data: GameData, wall_distances: list[float]):
    for einstein in data.einsteins:
        if not einstein.active:
            continue

        # Calculate screen position
        dx, dy = einstein.x - player.x, einstein.y - player.y
        dist = sqrt(dx*dx + dy*dy)
        angle_to_einstein = atan2(dy, dx)
        angle_diff = normalize_angle(angle_to_einstein - player_angle)

        # Check if in FOV
        if abs(angle_diff) > fov / 2:
            continue

        # Check if behind wall
        ray_idx = angle_to_ray_index(angle_diff)
        if wall_distances[ray_idx] < dist:
            continue

        # Calculate screen position and size
        screen_x = width / 2 + (angle_diff / (fov / 2)) * (width / 2)
        sprite_size = height / dist

        # Draw sprite
        sprite = assets.get_einstein_sprite(sprite_size)
        screen.blit(sprite, sprite.get_rect(center=(screen_x, height // 2)))
```

## Dependencies

- `pygame`: All rendering
- `sum_eternal.bridge`: Raycasting via student code
- `sum_eternal.engine.assets.Assets`: Einstein sprite

## Performance Considerations

- Ray count: 320 default (one per 2 horizontal pixels)
- Sprite cache: Scaled sprites are cached by size
- Minimap: Simplified rendering (fewer details than main view)
- Target: 60 FPS

## Error Handling

- Missing student functions: Fall back to _render_3d_fallback()
- Raycasting errors: Catch and show simple gradient view
- Asset loading errors: Use placeholder sprites

## Testing Strategy

Renderer is primarily tested manually/visually. Unit tests for:
- `_format_time()` helper
- Coordinate transformation functions
- State dispatch logic (mock pygame)
