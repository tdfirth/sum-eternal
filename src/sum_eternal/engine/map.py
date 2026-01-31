"""
Map - Level data and wall definitions.

The arena:
- 40x40 unit square (-20 to 20 on both axes)
- Central 4x4 pillar for interesting shadows
- Four Einstein spawn points in the corners
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp


# Player configuration
PLAYER_START = (0.0, -15.0)
PLAYER_START_ANGLE = math.pi / 2  # Facing north
PLAYER_SPEED = 5.0  # Units per second
PLAYER_TURN_SPEED = 2.0  # Radians per second
FOV = math.pi / 3  # 60 degrees


@dataclass
class Wall:
    """A wall segment defined by start and end points."""
    x1: float
    y1: float
    x2: float
    y2: float
    color: tuple[int, int, int] = (128, 128, 128)

    @property
    def start(self) -> tuple[float, float]:
        return (self.x1, self.y1)

    @property
    def end(self) -> tuple[float, float]:
        return (self.x2, self.y2)

    @property
    def direction(self) -> tuple[float, float]:
        """Normalized direction vector."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return (0.0, 0.0)
        return (dx / length, dy / length)

    @property
    def length(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return math.sqrt(dx * dx + dy * dy)


# Wall definitions
# Each wall: (x1, y1, x2, y2, color)
WALLS: list[Wall] = [
    # Outer walls (slightly darker on N/S vs E/W for visual distinction)
    Wall(-20, -20, 20, -20, (100, 100, 100)),   # South wall
    Wall(20, -20, 20, 20, (120, 120, 120)),     # East wall
    Wall(20, 20, -20, 20, (100, 100, 100)),     # North wall
    Wall(-20, 20, -20, -20, (120, 120, 120)),   # West wall

    # Central pillar (4x4, centered at origin)
    Wall(-2, -2, 2, -2, (80, 80, 80)),          # South face
    Wall(2, -2, 2, 2, (90, 90, 90)),            # East face
    Wall(2, 2, -2, 2, (80, 80, 80)),            # North face
    Wall(-2, 2, -2, -2, (90, 90, 90)),          # West face
]


# Einstein spawn points (corners of the arena)
EINSTEIN_SPAWNS: list[tuple[float, float]] = [
    (12.0, 12.0),    # Northeast corner
    (-12.0, 12.0),   # Northwest corner
    (12.0, -5.0),    # Southeast area
    (-12.0, -5.0),   # Southwest area
]


class Map:
    """Game map containing wall data."""

    def __init__(self, walls: Sequence[Wall] | None = None) -> None:
        self.walls = list(walls) if walls else list(WALLS)

    @property
    def wall_data(self) -> dict:
        """Return wall data in a format suitable for raycasting.

        Returns dict with JAX arrays:
            - wall_starts: (num_walls, 2) array of wall start points
            - wall_dirs: (num_walls, 2) array of wall direction vectors
            - wall_colors: list of (r, g, b) tuples
        """
        num_walls = len(self.walls)

        wall_starts = jnp.array([[w.x1, w.y1] for w in self.walls])
        wall_ends = jnp.array([[w.x2, w.y2] for w in self.walls])
        wall_dirs = wall_ends - wall_starts
        wall_colors = [w.color for w in self.walls]

        return {
            "wall_starts": wall_starts,
            "wall_dirs": wall_dirs,
            "wall_colors": wall_colors,
            "num_walls": num_walls,
        }

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounds of the map."""
        min_x = min(min(w.x1, w.x2) for w in self.walls)
        min_y = min(min(w.y1, w.y2) for w in self.walls)
        max_x = max(max(w.x1, w.x2) for w in self.walls)
        max_y = max(max(w.y1, w.y2) for w in self.walls)
        return (min_x, min_y, max_x, max_y)
