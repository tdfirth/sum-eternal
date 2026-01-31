# Sum Eternal — System Architecture

## Overview

Sum Eternal is an interactive tutorial that teaches einsum notation by having users progressively implement the core functions of a DOOM-style raycaster. As functions are implemented, the game hot-reloads and users watch the game materialize from nothing.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                          Entry Point                            │
│                         (main.py)                               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                            Game                                 │
│                          (game.py)                              │
│   - Manages game state (title, tutorial, victory)               │
│   - Handles player input and movement                           │
│   - Coordinates renderer, hot reload, and bridge                │
└─────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│    Renderer     │  │   Hot Reload    │  │       Bridge        │
│  (renderer.py)  │  │ (hot_reload.py) │  │     (bridge.py)     │
│                 │  │                 │  │                     │
│ - Title screen  │  │ - File watcher  │  │ - Progress check    │
│ - Debug view    │  │ - Test runner   │  │ - Safe imports      │
│ - 2D map view   │  │ - Module reload │  │ - Smoke tests       │
│ - 3D raycasting │  │ - Signal game   │  │ - Raycasting        │
│ - Sprites       │  │                 │  │                     │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
                                                   │
                                                   ▼
                               ┌─────────────────────────────────┐
                               │       Solutions Directory       │
                               │         (solutions/)            │
                               │                                 │
                               │  c01_first_blood.py             │
                               │  c02_knee_deep_in_the_indices.py│
                               │  c03_the_slaughter_batch.py     │
                               │  ...                            │
                               └─────────────────────────────────┘
```

## Data Flow

### Startup
1. `main.py` initializes pygame and creates Game instance
2. Game creates Renderer and starts HotReloader in background thread
3. Game checks initial progress via Bridge
4. Render loop begins

### Hot Reload Cycle
1. User edits a solution file
2. Watchdog detects file change
3. HotReloader runs pytest for that chapter
4. If tests pass: reload modules, signal Game
5. Game refreshes progress and updates state
6. Renderer displays new visuals

### Raycasting Pipeline (when unlocked)
1. Bridge.cast_all_rays() is called each frame
2. Uses student's angles_to_directions() to generate ray directions
3. Uses student's ray_wall_t/s_values() to find intersections
4. Returns distances and colors to Renderer
5. Renderer draws wall columns based on distances

## State Machine

```
                    ┌─────────────┐
                    │    TITLE    │
                    │   Screen    │
                    └──────┬──────┘
                           │ Progress > NOTHING
                           ▼
                    ┌─────────────┐
                    │  TUTORIAL   │◄────────────┐
                    │  (Active)   │             │
                    └──────┬──────┘             │
                           │ Chapter 8 complete │
                           │ + all Einsteins    │
                           │   summed           │
                           ▼                    │
                    ┌─────────────┐             │
                    │   VICTORY   │─── Q ───────┘
                    │             │
                    └──────┬──────┘
                           │ N (Nightmare)
                           ▼
                    ┌─────────────┐
                    │  NIGHTMARE  │
                    │   (Ch 9)    │
                    └─────────────┘
```

## Progress Levels

| Level | Visual State | Unlocked By |
|-------|-------------|-------------|
| NOTHING | Title screen only | Initial state |
| CHAPTER_1_COMPLETE | Debug view | 6 basic contractions |
| CHAPTER_2_COMPLETE | 2D map appears | 6 matrix operations |
| CHAPTER_3_COMPLETE | Player + rays | 6 batch operations |
| CHAPTER_4_COMPLETE | Ray fan rotates | 4 ray generation |
| CHAPTER_5_COMPLETE | **3D VIEW** | 6 intersection functions |
| CHAPTER_6_COMPLETE | Full shading | 4 projection functions |
| CHAPTER_7_COMPLETE | Einsteins visible | 6 Einstein math functions |
| CHAPTER_8_COMPLETE | Combat works | 4 combat functions |
| CHAPTER_9_COMPLETE | Textures | 3 advanced functions |

## File Structure

```
sum-eternal/
├── pyproject.toml          # uv project config
├── README.md               # User-facing documentation
├── CLAUDE.md               # Claude Code tutor instructions
├── .gitignore
│
├── src/sum_eternal/        # Game engine
│   ├── __init__.py
│   ├── main.py             # Entry point
│   ├── bridge.py           # Solution interface
│   └── engine/
│       ├── game.py         # Game state and loop
│       ├── renderer.py     # Pygame rendering
│       ├── hot_reload.py   # File watcher
│       ├── map.py          # Level data
│       └── assets.py       # Sprite loading
│
├── solutions/              # Student workspace
│   ├── c01_first_blood.py
│   ├── c02_knee_deep_in_the_indices.py
│   └── ...
│
├── tests/                  # Pytest tests
│   ├── conftest.py
│   ├── test_c01_first_blood.py
│   └── ...
│
├── .claude/skills/         # Claude teaching materials
│   ├── overview.md
│   ├── c01_first_blood.md
│   └── ...
│
├── assets/                 # Game assets
│   └── einstein.png
│
└── docs/                   # Architecture docs
    ├── overview.md         # This file
    ├── decisions.md        # Design decisions
    ├── modules/            # Module specs
    └── orchestration.md    # Build plan
```

## Thread Model

```
Main Thread                 Background Thread
───────────────────────    ───────────────────────
│                     │    │                     │
│  Pygame event loop  │    │  Watchdog observer  │
│         │           │    │         │           │
│         ▼           │    │         ▼           │
│  Game.update()      │    │  File modified      │
│         │           │    │         │           │
│         ▼           │    │         ▼           │
│  Game.render()      │    │  Run pytest         │
│         │           │    │         │           │
│         ▼           │    │         ▼           │
│  Check reload flag ◄├────┤► Set reload flag    │
│         │           │    │                     │
│         ▼           │    │                     │
│  Refresh progress   │    │                     │
│                     │    │                     │
───────────────────────    ───────────────────────
```

## Key Design Principles

1. **Graceful Degradation**: Every NotImplementedError is caught; game always runs
2. **Fast Feedback**: Tests run on file save, results within seconds
3. **Visual Progress**: Each chapter completion unlocks new game features
4. **No Magic**: Students write real einsum, see real results
5. **Socratic Teaching**: Claude guides without giving answers
