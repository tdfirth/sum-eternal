# Module: main.py

## Purpose

Entry point for Sum Eternal. Initializes pygame, creates the game, starts the hot reloader, and runs the main game loop.

## Location

`src/sum_eternal/main.py`

## Public Interface

### Functions

```python
def main() -> None:
    """
    Main entry point. Called by 'uv run sum-eternal'.

    1. Initialize pygame
    2. Create display surface
    3. Create Game instance
    4. Start HotReloader
    5. Run game loop until quit
    6. Clean up and exit
    """
```

## Implementation Approach

```python
def main():
    pygame.init()

    # Resolution from env or default 640x480
    resolution = parse_resolution(os.environ.get("SUM_ETERNAL_RESOLUTION", "640x480"))
    screen = pygame.display.set_mode(resolution)
    pygame.display.set_caption("SUM ETERNAL")

    game = Game(screen)
    reloader = HotReloader(game)
    reloader.start()

    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    game.handle_event(event)

            dt = clock.tick(60) / 1000.0
            game.update(dt)
            game.render()
            pygame.display.flip()
    finally:
        reloader.stop()
        pygame.quit()
```

## Dependencies

- `pygame`: Display and event handling
- `sum_eternal.engine.game.Game`: Core game logic
- `sum_eternal.engine.hot_reload.HotReloader`: File watching

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUM_ETERNAL_RESOLUTION` | `640x480` | Screen resolution (e.g., `800x600`) |

## Error Handling

- Invalid resolution format: Fall back to 640x480
- Pygame init failure: Let exception propagate
- HotReloader failure: Log warning, continue without hot reload

## Testing Strategy

Integration test only:
- Verify module imports without error
- Verify main() can be called (mock pygame)
