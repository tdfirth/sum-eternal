# Sum Eternal

**Learn einsum by building DOOM.**

Sum Eternal is an interactive tutorial that teaches JAX einsum notation by having you progressively implement the core functions of a DOOM-style raycaster. As you complete each function, the game hot-reloads and you watch DOOM materialize from nothing.

The Einsteins are not enemies. They are trials. Prove your einsum mastery, and you will be deemed worthy.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/sum-eternal
cd sum-eternal

# Install dependencies
uv sync

# Run the game
uv run sum-eternal
```

Then open Claude Code in this directory. Claude will guide you through implementing each function.

## How It Works

1. **Run the game** — initially you'll see just a title screen
2. **Open Claude Code** — Claude acts as your Socratic tutor
3. **Implement functions** — write einsum code in `solutions/`
4. **Watch the game evolve** — each chapter unlocks new visuals

## The Journey

| Chapter | File | Unlocks |
|---------|------|---------|
| 1 | `c01_first_blood.py` | Debug view |
| 2 | `c02_knee_deep_in_the_indices.py` | 2D map |
| 3 | `c03_the_slaughter_batch.py` | Player + rays |
| 4 | `c04_rip_and_trace.py` | Ray fan |
| 5 | `c05_total_intersection.py` | **3D WALLS** |
| 6 | `c06_infernal_projection.py` | Full shading |
| 7 | `c07_spooky_action_at_a_distance.py` | Einsteins visible |
| 8 | `c08_the_icon_of_ein.py` | Combat |
| 9 | `c09_nightmare_mode.py` | Textures (optional) |

## Controls

- **WASD** — Move
- **Arrow keys** — Turn
- **Space** — Sum (fire at Einsteins)
- **Q** — Quit (from victory screen)
- **N** — Nightmare mode (from victory screen)

## What You'll Learn

- Basic einsum contractions (`'i->'`, `'i,i->'`)
- Matrix operations (`'ij->ji'`, `'ii->'`)
- Batch processing (`'bi->b'`, `'bi,bi->b'`)
- All-pairs operations (`'id,jd->ij'`)
- Ray-wall intersection math
- Perspective projection
- 2D cross products

## Requirements

- Python 3.11+
- uv (package manager)
- Claude Code (for the tutorial experience)

## Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific chapter
uv run pytest tests/test_c01_first_blood.py -v

# Run with watch mode (re-run on changes)
uv run pytest-watch
```

## Project Structure

```
sum-eternal/
├── solutions/          # YOUR CODE GOES HERE
├── tests/              # Tests for your solutions
├── src/sum_eternal/    # Game engine (don't modify)
├── .claude/skills/     # Teaching materials
└── assets/             # Game assets
```

## Terminology

- **Einsteins** — The spectral trials (tongue-out Einstein heads)
- **Summing** — Defeating an Einstein through correct implementation
- **The Notation** — Einsum knowledge
- **Rip and Tensor** — The battle cry

## Troubleshooting

**Game won't start?**
```bash
uv sync  # Make sure dependencies are installed
```

**Tests not running?**
```bash
uv run pytest --collect-only  # Check test discovery
```

**Hot reload not working?**
- Make sure you're saving the file
- Check console for test output
- Tests must pass for reload to trigger

## Credits

Inspired by DOOM (1993) and the joy of learning linear algebra through building things.

*Rip and tensor.*
