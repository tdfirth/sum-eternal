# Sum Eternal — Orchestration Plan

This document enables a coordinating agent to build Sum Eternal efficiently, spawning subagents for parallelizable work and sequencing dependent work correctly.

## Work Packages Overview

| ID | Package | Est. Size | Dependencies | Parallelizable? |
|----|---------|-----------|--------------|-----------------|
| PKG-01 | Solution stubs + tests validation | Small | None | Yes |
| PKG-02 | Claude skills files completion | Medium | None | Yes |
| PKG-03 | Hot reload system | Small | None | Yes |
| PKG-04 | Bridge module completion | Small | PKG-01 (smoke tests defined) | After PKG-01 |
| PKG-05 | Map and assets | Small | None | Yes |
| PKG-06 | Game state machine | Medium | PKG-04 (progress enum) | After PKG-04 |
| PKG-07 | Renderer - Title/Victory | Small | PKG-06 (game states) | After PKG-06 |
| PKG-08 | Renderer - Debug views | Medium | PKG-06 | After PKG-06 |
| PKG-09 | Renderer - 2D map | Medium | PKG-06, PKG-05 | After PKG-06 |
| PKG-10 | Renderer - 3D raycasting | Large | PKG-06, PKG-04 | After PKG-06 |
| PKG-11 | Renderer - Einsteins | Medium | PKG-10, PKG-05 | After PKG-10 |
| PKG-12 | Integration & polish | Medium | All | Last |

## Dependency Graph

```
Phase 1 (Parallel):
  PKG-01 ─┐
  PKG-02  ├─ Can run simultaneously
  PKG-03  │
  PKG-05 ─┘

Phase 2 (After PKG-01):
  PKG-04 (bridge needs smoke tests from PKG-01)

Phase 3 (After PKG-04):
  PKG-06 (game needs bridge/progress)

Phase 4 (After PKG-06, Parallel):
  PKG-07 ─┐
  PKG-08  ├─ Can run simultaneously
  PKG-09 ─┘

Phase 5 (After PKG-06):
  PKG-10 (3D rendering, complex)

Phase 6 (After PKG-10):
  PKG-11 (Einstein sprites need 3D)

Phase 7:
  PKG-12 (integration)
```

## Optimal Execution Strategy

1. **Start Phase 1**: Spawn 4 agents for PKG-01, PKG-02, PKG-03, PKG-05
2. **When PKG-01 completes**: Start PKG-04
3. **When PKG-04 completes**: Start PKG-06
4. **When PKG-06 completes**: Spawn 3 agents for PKG-07, PKG-08, PKG-09 + start PKG-10
5. **When PKG-10 completes**: Start PKG-11
6. **When all complete**: Start PKG-12

---

## PKG-01: Solution Stubs + Tests Validation

### Context
The solution stubs and test files have been created. This package validates they're correct.

### Task
1. Verify all solution files exist with correct function signatures
2. Run all tests and confirm they fail with NotImplementedError
3. Implement one function from Chapter 1 and verify tests pass
4. Revert to stub and confirm tests fail again

### Files
- `solutions/c01_first_blood.py` through `c09_nightmare_mode.py`
- `tests/test_c01_first_blood.py` through `test_c09_nightmare_mode.py`

### Acceptance Criteria
- [ ] All 45 functions have stubs raising NotImplementedError
- [ ] All tests run without import errors
- [ ] Tests fail as expected (NotImplementedError or assertion)
- [ ] No syntax errors in any file

### Commands
```bash
uv sync
uv run pytest tests/ --collect-only  # Verify test discovery
uv run pytest tests/ -x  # Should fail on first test
```

---

## PKG-02: Claude Skills Files Completion

### Context
Teaching materials for Claude Code have been created. Verify and enhance.

### Task
1. Verify all 9 chapter skill files exist
2. Check each file has: overview, function explanations, solutions, common mistakes
3. Add any missing details based on the curriculum spec
4. Ensure consistency in tone and format

### Files
- `.claude/skills/overview.md`
- `.claude/skills/c01_first_blood.md` through `c09_nightmare_mode.md`

### Acceptance Criteria
- [ ] All skill files exist and are complete
- [ ] Each function has solution code documented
- [ ] Teaching tips are practical
- [ ] Socratic questions are included

---

## PKG-03: Hot Reload System

### Context
File watching needs to detect changes and run tests.

### Task
1. Implement `SolutionFileHandler` in `hot_reload.py`
2. Implement debouncing (0.5s between triggers)
3. Implement module reloading via `reload_solution_modules()`
4. Test with actual file changes

### Files
- `src/sum_eternal/engine/hot_reload.py`

### Interface
```python
class HotReloader:
    def __init__(self, game: Game) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

def reload_solution_modules() -> None: ...
```

### Acceptance Criteria
- [ ] File changes in solutions/ trigger handler
- [ ] Pytest runs for correct chapter on change
- [ ] Rapid saves don't trigger multiple reloads
- [ ] Module cache is properly cleared
- [ ] Game.refresh_progress() is called on success

### Testing
```bash
# Start game, edit a solution file, verify console shows test output
```

---

## PKG-04: Bridge Module Completion

### Context
The bridge connects solutions to the game engine. Core structure exists.

### Task
1. Complete `check_progress()` with all chapter checks
2. Implement `cast_all_rays()` using student functions
3. Add `has_raycasting_functions()` check
4. Test with mock implementations

### Files
- `src/sum_eternal/bridge.py`

### Interface
```python
class Progress(IntEnum): ...  # Already defined

def check_progress() -> Progress: ...
def has_raycasting_functions() -> bool: ...
def cast_all_rays(...) -> tuple[list[float], list[tuple[int,int,int]]]: ...
```

### Acceptance Criteria
- [ ] `check_progress()` correctly identifies all 10 progress levels
- [ ] NotImplementedError is caught gracefully
- [ ] Import errors don't crash the game
- [ ] `cast_all_rays()` returns fallback on failure
- [ ] Smoke tests use correct inputs for each function

### Testing
```python
# Test with no implementations
assert check_progress() == Progress.NOTHING

# Test with Chapter 1 implemented
# (temporarily implement vector_sum etc.)
assert check_progress() == Progress.CHAPTER_1_COMPLETE
```

---

## PKG-05: Map and Assets

### Context
Level geometry and sprite loading.

### Task
1. Finalize wall coordinates in `map.py`
2. Implement `Assets` class for Einstein sprite
3. Create placeholder Einstein sprite (or use pixel art)
4. Test sprite scaling

### Files
- `src/sum_eternal/engine/map.py`
- `src/sum_eternal/engine/assets.py`
- `assets/einstein.png` (create or placeholder)

### Acceptance Criteria
- [ ] Map has outer walls and central pillar
- [ ] Einstein spawn points are defined
- [ ] `Assets.get_einstein_sprite(size)` returns scaled surface
- [ ] Sprite cache works correctly
- [ ] Placeholder works if real asset missing

---

## PKG-06: Game State Machine

### Context
Game manages states, player, Einsteins, and coordinates systems.

### Task
1. Complete `Game.__init__()` setup
2. Implement state transitions (TITLE → TUTORIAL → VICTORY)
3. Implement `handle_event()` for all states
4. Implement `update()` with movement and collision
5. Implement Einstein AI (move toward player)

### Files
- `src/sum_eternal/engine/game.py`

### Interface
```python
class Game:
    def __init__(self, screen: Surface) -> None: ...
    def refresh_progress(self) -> None: ...
    def handle_event(self, event: Event) -> None: ...
    def update(self, dt: float) -> None: ...
    def render(self) -> None: ...
```

### Acceptance Criteria
- [ ] Game starts in TITLE state
- [ ] Transitions to TUTORIAL when progress > NOTHING
- [ ] Player moves with WASD, turns with arrows
- [ ] Collision keeps player inside bounds
- [ ] Einsteins activate at Chapter 7
- [ ] Victory triggers at Chapter 8 + all summed

### Testing
```bash
uv run sum-eternal  # Should show title screen
# Implement one function, save, verify state changes
```

---

## PKG-07: Renderer - Title/Victory

### Context
Static screens for game start and end.

### Task
1. Implement `_render_title()`
2. Implement `_render_victory()`
3. Style with DOOM aesthetic
4. Show progress info on title

### Files
- `src/sum_eternal/engine/renderer.py`

### Acceptance Criteria
- [ ] Title shows "SUM ETERNAL" prominently
- [ ] Progress level displayed
- [ ] Victory shows stats (time, functions, Einsteins)
- [ ] Nightmare mode option shown
- [ ] Colors match DOOM aesthetic (red, gray, green)

---

## PKG-08: Renderer - Debug Views

### Context
Pre-3D visualization for Chapters 1-2.

### Task
1. Implement `_render_debug_view()`
2. Show matrix operations with results
3. Add checkmarks for completed functions
4. Style as terminal/console

### Files
- `src/sum_eternal/engine/renderer.py`

### Acceptance Criteria
- [ ] Shows "SYSTEMS INITIALIZING" style text
- [ ] Lists vector/matrix operations
- [ ] Checkmarks appear for working functions
- [ ] Progress bar at bottom

---

## PKG-09: Renderer - 2D Map

### Context
Top-down map view for Chapters 3-4.

### Task
1. Implement `_render_2d_map()`
2. Draw walls as lines
3. Draw player with direction indicator
4. Draw ray fan (Chapter 4)
5. Scale world to screen

### Files
- `src/sum_eternal/engine/renderer.py`

### Acceptance Criteria
- [ ] Walls render in correct positions
- [ ] Player position updates with movement
- [ ] Direction indicator shows facing
- [ ] Ray fan appears at Chapter 4
- [ ] Controls hint displayed

---

## PKG-10: Renderer - 3D Raycasting

### Context
The core 3D view that unlocks at Chapter 5.

### Task
1. Implement `_render_3d_view()`
2. Call `bridge.cast_all_rays()`
3. Render wall columns based on distances
4. Implement `_render_wall_columns()`
5. Implement `_render_floor_ceiling()` (gradients)
6. Implement `_render_minimap()`
7. Implement fallback for missing functions

### Files
- `src/sum_eternal/engine/renderer.py`

### Interface (uses bridge)
```python
distances, colors = bridge.cast_all_rays(
    player_pos, player_angle, fov, num_rays, map.wall_data
)
```

### Acceptance Criteria
- [ ] 3D view activates at Chapter 5
- [ ] Walls render at correct heights
- [ ] Distance shading works (Chapter 6)
- [ ] Minimap shows in corner
- [ ] Crosshair rendered
- [ ] Fallback renders gradient when functions missing

### Testing
Implement Chapter 5 functions, verify walls render correctly.

---

## PKG-11: Renderer - Einsteins

### Context
Sprite rendering for Einstein trials.

### Task
1. Implement `_render_einsteins()`
2. Calculate screen position from world position
3. Check FOV visibility
4. Check wall occlusion
5. Scale sprite by distance

### Files
- `src/sum_eternal/engine/renderer.py`

### Acceptance Criteria
- [ ] Einsteins appear at Chapter 7
- [ ] Position correctly on screen
- [ ] Occluded by walls
- [ ] Scale with distance
- [ ] Only active (not summed) Einsteins render

---

## PKG-12: Integration & Polish

### Context
Final integration, bug fixes, and polish.

### Task
1. Full playthrough test
2. Fix any integration issues
3. Verify hot reload works end-to-end
4. Test all state transitions
5. Verify victory condition
6. Performance check (60 FPS)
7. Update README with final instructions

### Acceptance Criteria
- [ ] Game starts and shows title
- [ ] Can progress through all chapters
- [ ] Hot reload works on all solution files
- [ ] 3D view renders correctly
- [ ] Einsteins can be summed
- [ ] Victory screen shows
- [ ] No crashes or hangs
- [ ] README has clear setup instructions

---

## Root Orchestration Prompt

Use this prompt to coordinate the build:

```markdown
# Sum Eternal Build Orchestration

You are coordinating the build of Sum Eternal. The architecture is complete in docs/.
Your job is to spawn subagents to implement each package efficiently.

## Current Status
[Track which packages are complete/in-progress/pending]

## Execution Order

### Phase 1: Start immediately (parallel)
- PKG-01: Solution validation
- PKG-02: Skills files
- PKG-03: Hot reload
- PKG-05: Map and assets

### Phase 2: After PKG-01 completes
- PKG-04: Bridge module

### Phase 3: After PKG-04 completes
- PKG-06: Game state machine

### Phase 4: After PKG-06 completes (parallel)
- PKG-07: Title/Victory renderer
- PKG-08: Debug view renderer
- PKG-09: 2D map renderer
- PKG-10: 3D raycasting renderer

### Phase 5: After PKG-10 completes
- PKG-11: Einstein renderer

### Phase 6: After all complete
- PKG-12: Integration

## For Each Package

1. Read the package spec from docs/orchestration.md
2. Read relevant module specs from docs/modules/
3. Spawn a subagent with the package task
4. Verify acceptance criteria when complete
5. Update status and proceed to next phase

## Verification Commands
```bash
uv sync                     # Install dependencies
uv run pytest tests/ -v     # Run all tests
uv run sum-eternal          # Run the game
```

## Success Criteria
- All tests pass (when solutions implemented)
- Game runs without errors
- Can progress through all chapters
- Hot reload works
- Victory achievable
```

---

## Subagent Prompts

### PKG-03 Subagent Prompt Example

```markdown
# Task: Implement Hot Reload System

## Context
You're implementing the hot reload system for Sum Eternal, a DOOM-style raycaster tutorial.

## Your Files
- `src/sum_eternal/engine/hot_reload.py` (edit this)

## Requirements
1. Use watchdog to monitor `solutions/` directory
2. On .py file change:
   - Debounce (ignore changes within 0.5s)
   - Run pytest for corresponding test file
   - If pass: reload modules and signal game
   - If fail: show error message
3. Module reloading must clear sys.modules cache

## Interface
```python
class HotReloader:
    def __init__(self, game: Game) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

def reload_solution_modules() -> None: ...
```

## Solution-to-Test Mapping
```python
SOLUTION_TO_TEST = {
    "c01_first_blood.py": "test_c01_first_blood.py",
    ...
}
```

## Testing
1. Start the game
2. Edit a solution file
3. Verify console shows test output
4. Verify rapid edits don't trigger multiple runs

## Done When
- [ ] File changes detected
- [ ] Tests run automatically
- [ ] Debouncing works
- [ ] Modules reload correctly
```

---

## Notes for Orchestrating Agent

1. **Don't wait unnecessarily**: Start Phase 1 packages immediately
2. **Parallelize within phases**: Multiple agents can work simultaneously
3. **Verify before proceeding**: Check acceptance criteria before moving to dependent phases
4. **Integration is last**: PKG-12 catches any issues from parallel work
5. **Test frequently**: Run `uv run sum-eternal` to catch integration issues early
