# Module: engine/game.py

## Purpose

Core game state management and update loop. Coordinates all game systems and maintains the canonical game state.

## Location

`src/sum_eternal/engine/game.py`

## Public Interface

### Enums

```python
class GameState(Enum):
    """Top-level game states."""
    TITLE = auto()      # Title screen, waiting for progress
    TUTORIAL = auto()   # Active gameplay
    VICTORY = auto()    # Game complete
    NIGHTMARE = auto()  # Post-victory bonus content
```

### Data Classes

```python
@dataclass
class Player:
    """Player state."""
    x: float = 0.0
    y: float = -15.0
    angle: float = pi/2  # Facing north

    def move(self, forward: float, strafe: float, dt: float) -> None: ...
    def turn(self, amount: float, dt: float) -> None: ...
    @property
    def position(self) -> tuple[float, float]: ...

@dataclass
class Einstein:
    """An Einstein trial (enemy)."""
    x: float
    y: float
    active: bool = False
    health: float = 1.0
    summed: bool = False

    @property
    def position(self) -> tuple[float, float]: ...

@dataclass
class GameData:
    """All mutable game data."""
    state: GameState = GameState.TITLE
    progress: Progress = Progress.NOTHING
    player: Player = field(default_factory=Player)
    einsteins: list[Einstein] = field(default_factory=list)
    error_message: str | None = None
    error_timeout: float = 0.0
    total_time: float = 0.0
    einsteins_summed: int = 0
```

### Classes

```python
class Game:
    """Main game controller."""

    def __init__(self, screen: Surface) -> None:
        """
        Initialize game with pygame surface.

        Creates: Renderer, Map, Assets, initial Einsteins
        Checks initial progress from solutions.
        """

    def refresh_progress(self) -> None:
        """
        Re-check progress from solution files.
        Called after hot reload detects changes.
        Updates self.data.progress and game state accordingly.
        """

    def show_error(self, message: str, duration: float = 5.0) -> None:
        """Display an error message overlay."""

    def clear_error(self) -> None:
        """Clear the error message."""

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Handle a pygame event.

        Routes events based on current game state:
        - TITLE: Any key advances if progress > NOTHING
        - TUTORIAL: Movement and action keys
        - VICTORY: N for nightmare, Q for quit
        """

    def update(self, dt: float) -> None:
        """
        Update game state.

        - Update total time
        - Handle continuous input (movement)
        - Update Einstein AI
        - Update error timeout
        """

    def render(self) -> None:
        """Render the current frame via Renderer."""
```

## Implementation Approach

### State Transitions

```
TITLE -> TUTORIAL: When progress > NOTHING
TUTORIAL -> VICTORY: When Chapter 8 complete AND all Einsteins summed
VICTORY -> NIGHTMARE: When user presses N
VICTORY -> TITLE: When user presses Q
```

### Movement System

```python
def _handle_movement(self, dt: float):
    # Read held keys
    forward = 0.0
    if K_w in self._keys_held:
        forward += 1.0
    if K_s in self._keys_held:
        forward -= 1.0

    # Similar for strafe and turn
    self.data.player.move(forward, strafe, dt)
    self.data.player.turn(turn, dt)
    self._apply_collision()

def _apply_collision(self):
    # Clamp to outer bounds
    x = clamp(x, -19.5 + margin, 19.5 - margin)
    y = clamp(y, -19.5 + margin, 19.5 - margin)

    # Push out of inner pillar
    if in_pillar_bounds(x, y):
        push_to_nearest_edge(x, y)
```

### Einstein AI

```python
def _update_einsteins(self, dt: float):
    if progress < CHAPTER_7:
        return  # Not active yet

    for einstein in self.data.einsteins:
        if einstein.active and not einstein.summed:
            # Move toward player at slow speed
            direction = normalize(player_pos - einstein_pos)
            einstein.pos += direction * speed * dt
```

## Dependencies

- `sum_eternal.bridge`: Progress checking, raycasting
- `sum_eternal.engine.renderer.Renderer`: Rendering
- `sum_eternal.engine.map.Map`: Level data
- `sum_eternal.engine.assets.Assets`: Sprites

## State Management

All mutable state lives in `GameData`. This makes it easy to:
- Save/load progress
- Reset game state
- Test in isolation

## Error Handling

- Solution import errors: Caught in bridge, return NOTHING progress
- Rendering errors: Show error overlay, continue running
- Input errors: Log and ignore

## Testing Strategy

```python
def test_state_transitions():
    """Verify TITLE -> TUTORIAL -> VICTORY flow."""

def test_player_movement():
    """Verify player moves correctly with dt scaling."""

def test_collision_detection():
    """Verify player can't walk through walls."""

def test_einstein_activation():
    """Verify Einsteins become active at Chapter 7."""

def test_victory_condition():
    """Verify victory triggers at correct conditions."""
```
