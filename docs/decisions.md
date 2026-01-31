# Architectural Decisions

This document records all design decisions made during architecture. Implementing agents should not face ambiguity on these points.

---

## Decision 1: Einstein Asset

**Decision**: Use a simple pixel art sprite (32x32 or 64x64) depicting Einstein's iconic tongue-out pose in a ghostly/spectral style.

**Rationale**:
- Pixel art fits the retro DOOM aesthetic
- Avoids copyright concerns with actual photos
- Can be easily scaled for distance effects
- Simple to create as a placeholder

**Specification**:
- File: `assets/einstein.png`
- Size: 64x64 pixels, RGBA
- Style: Ghostly blue/white tint, visible tongue, wild hair silhouette
- Placeholder: A simple colored circle with "E=mc²" text is acceptable for initial development
- The sprite should have transparency for compositing over the 3D view

**Implementation Note**: For v1, we'll include a placeholder sprite. The asset can be improved later without code changes.

---

## Decision 2: Debug Visualization (Chapters 1-2)

**Decision**: Before 3D activates, render a "terminal-style" debug view showing operations as they're computed.

**Specification**:

### Chapter 1 Complete (Debug View Level 1):
```
┌─────────────────────────────────────────┐
│  SUM ETERNAL - SYSTEMS INITIALIZING     │
│  ═══════════════════════════════════════│
│                                         │
│  VECTOR OPS: ONLINE                     │
│  ┌─────────────────────────────────────┐│
│  │ v = [1.0, 2.0, 3.0]                ││
│  │ sum(v) = 6.0           ✓           ││
│  │                                     ││
│  │ a·b = 32.0             ✓           ││
│  │ a⊗b = [[...]]          ✓           ││
│  └─────────────────────────────────────┘│
│                                         │
│  Progress: Chapter 1 Complete           │
│  Next: Knee Deep in the Indices         │
└─────────────────────────────────────────┘
```

### Chapter 2 Complete (Debug View Level 2):
```
┌─────────────────────────────────────────┐
│  SUM ETERNAL - MATRIX SYSTEMS ONLINE    │
│  ═══════════════════════════════════════│
│                                         │
│  ┌─────────────────┐  ┌─────────────┐  │
│  │ 1  2  3 │ T    │  │ 1  4  7 │   │  │
│  │ 4  5  6 │ ───► │  │ 2  5  8 │   │  │
│  │ 7  8  9 │      │  │ 3  6  9 │   │  │
│  └─────────────────┘  └─────────────┘  │
│                                         │
│  trace(M) = 15.0  ✓                     │
│  diag(M) = [1, 5, 9]  ✓                 │
│                                         │
│  [2D MAP PREVIEW LOADING...]            │
└─────────────────────────────────────────┘
```

**Animation**:
- Matrices pulse when operations complete
- Checkmarks appear with a brief flash
- Smooth transitions between states

---

## Decision 3: Map Design

**Decision**: A single, well-designed level that's small enough to be comprehensible but large enough to feel like DOOM.

**Specification**:

### Coordinate System:
- World coordinates: floating point, Y increases upward
- Map origin: (0, 0) at center
- Scale: 1 unit = 1 "meter" (roughly player height)

### Level Layout:
```
    Wall coordinates (forming a simple arena):

         (0,20)
           │
    ┌──────┴──────┐
    │             │
    │    ┌───┐    │    Inner pillar creates
    │    │   │    │    interesting shadows
(-20,0)──┤   ├──(20,0)
    │    │   │    │
    │    └───┘    │
    │             │
    └──────┬──────┘
           │
         (0,-20)

    Player starts at (0, -15), facing north (angle = π/2)
```

### Wall Data Structure:
```python
# Each wall is (start_x, start_y, end_x, end_y, color)
WALLS = [
    # Outer walls
    (-20, -20, 20, -20, (100, 100, 100)),   # South
    (20, -20, 20, 20, (120, 120, 120)),     # East
    (20, 20, -20, 20, (100, 100, 100)),     # North
    (-20, 20, -20, -20, (120, 120, 120)),   # West

    # Inner pillar (4x4, centered at origin)
    (-2, -2, 2, -2, (80, 80, 80)),          # South face
    (2, -2, 2, 2, (90, 90, 90)),            # East face
    (2, 2, -2, 2, (80, 80, 80)),            # North face
    (-2, 2, -2, -2, (90, 90, 90)),          # West face
]
```

### Einstein Spawn Points:
```python
EINSTEIN_SPAWNS = [
    (10, 10),    # Northeast corner
    (-10, 10),   # Northwest corner
    (10, -10),   # Southeast corner
    (-10, -10),  # Southwest corner (final boss position)
]
```

### Player Configuration:
```python
PLAYER_START = (0.0, -15.0)
PLAYER_START_ANGLE = math.pi / 2  # Facing north
PLAYER_SPEED = 5.0  # Units per second
PLAYER_TURN_SPEED = 2.0  # Radians per second
FOV = math.pi / 3  # 60 degrees
```

---

## Decision 4: Hot Reload Mechanism

**Decision**: Use a shared flag file for IPC between the file watcher and pygame.

**Rationale**:
- Simplest possible mechanism
- No external dependencies beyond watchdog
- Works reliably across platforms
- Easy to debug

**Specification**:

### Flag File Location:
```
/tmp/sum_eternal_reload_<pid>
```

### Protocol:
1. **Watcher process** (runs in separate thread):
   - Monitors `solutions/` for .py file changes
   - On change:
     - Runs `pytest tests/test_<chapter>.py -x -q` (fail fast, quiet)
     - If tests pass: write chapter name to flag file
     - If tests fail: do nothing (console shows errors)

2. **Game loop** (main thread):
   - Each frame: check if flag file exists
   - If exists:
     - Read chapter name
     - Delete flag file
     - Call `reload_solutions()`
     - Recalculate progress
     - Update game state

### Module Reloading:
```python
import importlib
import sys

def reload_solutions():
    """Reload all solution modules."""
    # Remove cached modules
    to_remove = [k for k in sys.modules if k.startswith('solutions.')]
    for k in to_remove:
        del sys.modules[k]

    # Re-import bridge (which re-imports solutions)
    import sum_eternal.bridge
    importlib.reload(sum_eternal.bridge)
```

### Thread Safety:
- Flag file operations are atomic (write to temp, rename)
- Game state updates happen only in main thread
- No shared mutable state between threads

---

## Decision 5: Progress Detection

**Decision**: Try/except on actual function calls with test inputs.

**Rationale**:
- Simple and reliable
- Actually verifies the implementation works
- Matches what tests do

**Specification**:

```python
def check_progress() -> Progress:
    """Determine progress by testing each function."""
    import jax.numpy as jnp

    # Test Chapter 1
    try:
        from solutions.c01_first_blood import vector_sum
        result = vector_sum(jnp.array([1.0, 2.0, 3.0]))
        if result is None or not jnp.isclose(result, 6.0):
            return Progress.NOTHING
    except (NotImplementedError, ImportError, Exception):
        return Progress.NOTHING

    # ... continue for each function in each chapter
    # Return the highest complete chapter
```

### Test Inputs:
Each function has a minimal "smoke test" input defined in the bridge:
```python
SMOKE_TESTS = {
    'vector_sum': (jnp.array([1.0, 2.0, 3.0]),),
    'element_multiply': (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
    # ... etc
}
```

---

## Decision 6: Victory Condition

**Decision**: Victory triggers when all Chapter 8 functions pass. Chapter 9 is optional bonus content.

**Specification**:

### Victory Trigger:
- `check_progress()` returns `Progress.CHAPTER_8_COMPLETE`
- All Einsteins have been "summed" (defeated) at least once

### Victory Flow:
1. Final Einstein defeated
2. Screen fades to white (0.5s)
3. Victory screen displays:
   ```
   ╔═══════════════════════════════════════╗
   ║                                       ║
   ║        YOU ARE WORTHY                 ║
   ║                                       ║
   ║   The notation recognizes you.        ║
   ║   Einstein nods approvingly.          ║
   ║                                       ║
   ║   Functions mastered: 42/42           ║
   ║   Einsteins summed: 4                 ║
   ║   Time: 3h 42m                        ║
   ║                                       ║
   ║   [N] Nightmare Mode unlocked         ║
   ║   [Q] Quit                            ║
   ║                                       ║
   ╚═══════════════════════════════════════╝
   ```
4. If user presses N: transition to Chapter 9 content
5. If user presses Q: return to title screen with "WORTHY" badge

### Nightmare Mode Completion:
- Separate victory screen with "NIGHTMARE CONQUERED"
- Additional stats for texture functions

---

## Decision 7: Progress Persistence

**Decision**: Yes, persist progress to a local file.

**Rationale**:
- Users may close and reopen over multiple sessions
- Prevents frustration of losing progress
- Simple to implement

**Specification**:

### File Location:
```
~/.sum_eternal/progress.json
```

### File Format:
```json
{
    "version": 1,
    "functions_completed": [
        "c01_first_blood.vector_sum",
        "c01_first_blood.element_multiply",
        ...
    ],
    "einsteins_summed": 2,
    "total_time_seconds": 3600,
    "nightmare_unlocked": false,
    "nightmare_completed": false,
    "last_session": "2024-01-15T10:30:00Z"
}
```

### Behavior:
- Load on game start
- Save after each function completion
- Save on clean exit
- If file missing or corrupted: start fresh (with warning)

---

## Decision 8: Error Display

**Decision**: Errors display in both console AND a pygame overlay.

**Rationale**:
- Console captures full traceback for debugging
- Pygame overlay ensures user sees there's a problem
- Students shouldn't have to switch contexts constantly

**Specification**:

### Console Output:
Full pytest output with colors, including:
- Which test failed
- Expected vs actual
- Full traceback

### Pygame Overlay:
Semi-transparent red banner at top of screen:
```
┌─────────────────────────────────────────────────────────────┐
│ ⚠ TEST FAILED: c01_first_blood.py::TestVectorSum::test_basic│
│   Expected: 6.0  Got: None                                  │
│   Check console for details. The notation demands precision.│
└─────────────────────────────────────────────────────────────┘
```

### Behavior:
- Overlay appears on test failure
- Overlay dismisses on next successful save
- Game continues running underneath (doesn't freeze)
- Overlay has slight transparency so game is visible

---

## Decision 9: Sound

**Decision**: No sound for v1.

**Rationale**:
- Adds complexity without core value
- Sound assets require additional work
- Can be added in v2 without architectural changes

**Future Consideration**:
- Reserve `assets/sounds/` directory
- Document sound events for future implementation:
  - `test_pass.wav` - function implemented correctly
  - `chapter_complete.wav` - chapter milestone
  - `einstein_spawn.wav` - Einstein appears
  - `einstein_summed.wav` - Einstein defeated
  - `victory.wav` - game complete

---

## Decision 10: Frame Rate / Performance

**Decision**: Target 60 FPS with graceful degradation.

**Specification**:

### Target:
- 60 FPS for smooth gameplay
- Minimum acceptable: 30 FPS

### Ray Count:
- Default: 320 rays (one per 2 pixels on 640-wide screen)
- Can be reduced to 160 if performance issues detected

### Performance Monitoring:
```python
# Track frame times
frame_times = collections.deque(maxlen=60)

def check_performance():
    avg_frame_time = sum(frame_times) / len(frame_times)
    if avg_frame_time > 33:  # Below 30 FPS
        reduce_ray_count()
        show_performance_warning()
```

### Student Code Performance:
- If student einsum is slow, it's educational
- No automatic optimization of student code
- Display frame time in debug mode so they can see impact

### Screen Resolution:
- Default: 640x480 (classic DOOM feel)
- Configurable via environment variable: `SUM_ETERNAL_RESOLUTION=800x600`

---

## Summary Table

| # | Decision | Choice |
|---|----------|--------|
| 1 | Einstein asset | 64x64 pixel art, ghostly style |
| 2 | Debug visualization | Terminal-style matrix display |
| 3 | Map design | 40x40 arena with central pillar |
| 4 | Hot reload | Flag file IPC, pytest on change |
| 5 | Progress detection | Try/except with smoke tests |
| 6 | Victory condition | Chapter 8 complete + all Einsteins summed |
| 7 | Progress persistence | ~/.sum_eternal/progress.json |
| 8 | Error display | Console + pygame overlay |
| 9 | Sound | No sound for v1 |
| 10 | Performance | 60 FPS target, 320 rays default |

---

*All decisions are final. Implementing agents should not revisit these choices.*
