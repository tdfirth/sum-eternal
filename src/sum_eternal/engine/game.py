"""
Game - Core game state and update loop.

Manages:
- Current game state (title, tutorial, victory)
- Player position and angle
- Einstein positions and states
- Progress tracking
- State transitions
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import pygame

from sum_eternal.bridge import Progress, check_progress
from sum_eternal.engine.map import Map, PLAYER_START, PLAYER_START_ANGLE, PLAYER_SPEED, PLAYER_TURN_SPEED
from sum_eternal.engine.renderer import Renderer
from sum_eternal.engine.assets import Assets

if TYPE_CHECKING:
    from pygame import Surface


class GameState(Enum):
    """Top-level game states."""
    TITLE = auto()
    TUTORIAL = auto()  # Active gameplay with tutorial
    VICTORY = auto()
    NIGHTMARE = auto()  # Post-victory bonus content


@dataclass
class Player:
    """Player state."""
    x: float = PLAYER_START[0]
    y: float = PLAYER_START[1]
    angle: float = PLAYER_START_ANGLE  # Radians, 0 = east, π/2 = north

    def move(self, forward: float, strafe: float, dt: float) -> None:
        """Move player based on input."""
        # Forward/backward
        self.x += math.cos(self.angle) * forward * PLAYER_SPEED * dt
        self.y += math.sin(self.angle) * forward * PLAYER_SPEED * dt
        # Strafe (perpendicular to facing)
        self.x += math.cos(self.angle + math.pi / 2) * strafe * PLAYER_SPEED * dt
        self.y += math.sin(self.angle + math.pi / 2) * strafe * PLAYER_SPEED * dt

    def turn(self, amount: float, dt: float) -> None:
        """Turn player."""
        self.angle += amount * PLAYER_TURN_SPEED * dt
        # Normalize to [0, 2π)
        self.angle = self.angle % (2 * math.pi)

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Einstein:
    """An Einstein trial."""
    x: float
    y: float
    active: bool = False  # Only visible when progress reaches Chapter 7
    health: float = 1.0   # Summed when health reaches 0
    summed: bool = False

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class GameData:
    """All mutable game data."""
    state: GameState = GameState.TITLE
    progress: Progress = Progress.NOTHING
    player: Player = field(default_factory=Player)
    einsteins: list[Einstein] = field(default_factory=list)
    error_message: str | None = None
    error_timeout: float = 0.0
    total_time: float = 0.0  # Total play time in seconds
    einsteins_summed: int = 0


class Game:
    """Main game controller."""

    def __init__(self, screen: Surface) -> None:
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()

        # Load assets
        self.assets = Assets()

        # Initialize renderer
        self.renderer = Renderer(screen, self.assets)

        # Initialize map
        self.map = Map()

        # Initialize game data
        self.data = GameData()

        # Initialize Einsteins at spawn points
        from sum_eternal.engine.map import EINSTEIN_SPAWNS
        self.data.einsteins = [
            Einstein(x=pos[0], y=pos[1])
            for pos in EINSTEIN_SPAWNS
        ]

        # Check initial progress
        self.refresh_progress()

        # Input state
        self._keys_held: set[int] = set()

    def refresh_progress(self) -> None:
        """Re-check progress from solution files."""
        old_progress = self.data.progress
        self.data.progress = check_progress()

        # Activate Einsteins when Chapter 7 is reached
        if self.data.progress >= Progress.CHAPTER_7_COMPLETE:
            for einstein in self.data.einsteins:
                if not einstein.summed:
                    einstein.active = True

        # Check for victory
        if (self.data.progress >= Progress.CHAPTER_8_COMPLETE and
            all(e.summed for e in self.data.einsteins)):
            self.data.state = GameState.VICTORY

        # Transition from title to tutorial when any progress made
        if (self.data.state == GameState.TITLE and
            self.data.progress > Progress.NOTHING):
            self.data.state = GameState.TUTORIAL

    def show_error(self, message: str, duration: float = 5.0) -> None:
        """Display an error message overlay."""
        self.data.error_message = message
        self.data.error_timeout = duration

    def clear_error(self) -> None:
        """Clear the error message."""
        self.data.error_message = None
        self.data.error_timeout = 0.0

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle a pygame event."""
        if event.type == pygame.KEYDOWN:
            self._keys_held.add(event.key)

            # State-specific key handling
            if self.data.state == GameState.TITLE:
                # Any key starts if there's progress
                if self.data.progress > Progress.NOTHING:
                    self.data.state = GameState.TUTORIAL

            elif self.data.state == GameState.VICTORY:
                if event.key == pygame.K_n:
                    # Enter nightmare mode
                    self.data.state = GameState.NIGHTMARE
                elif event.key == pygame.K_q:
                    # Return to title
                    self.data.state = GameState.TITLE

            elif self.data.state == GameState.TUTORIAL:
                # Space to "fire" (sum Einsteins)
                if event.key == pygame.K_SPACE:
                    self._try_sum_einstein()

        elif event.type == pygame.KEYUP:
            self._keys_held.discard(event.key)

    def _try_sum_einstein(self) -> None:
        """Attempt to sum an Einstein in the crosshair."""
        if self.data.progress < Progress.CHAPTER_8_COMPLETE:
            return  # Combat not unlocked

        # TODO: Implement ray-based hit detection
        # For now, sum the nearest active Einstein within range
        player_pos = self.data.player.position
        for einstein in self.data.einsteins:
            if einstein.active and not einstein.summed:
                dx = einstein.x - player_pos[0]
                dy = einstein.y - player_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 15.0:  # Within range
                    einstein.summed = True
                    einstein.active = False
                    self.data.einsteins_summed += 1
                    break

    def update(self, dt: float) -> None:
        """Update game state."""
        # Update total time
        if self.data.state == GameState.TUTORIAL:
            self.data.total_time += dt

        # Update error timeout
        if self.data.error_timeout > 0:
            self.data.error_timeout -= dt
            if self.data.error_timeout <= 0:
                self.clear_error()

        # Handle continuous input (movement)
        if self.data.state == GameState.TUTORIAL:
            self._handle_movement(dt)

        # Update Einsteins (move toward player)
        if self.data.state == GameState.TUTORIAL:
            self._update_einsteins(dt)

    def _handle_movement(self, dt: float) -> None:
        """Handle continuous movement input."""
        forward = 0.0
        strafe = 0.0
        turn = 0.0

        if pygame.K_w in self._keys_held or pygame.K_UP in self._keys_held:
            forward += 1.0
        if pygame.K_s in self._keys_held or pygame.K_DOWN in self._keys_held:
            forward -= 1.0
        if pygame.K_a in self._keys_held:
            strafe -= 1.0
        if pygame.K_d in self._keys_held:
            strafe += 1.0
        if pygame.K_LEFT in self._keys_held:
            turn += 1.0
        if pygame.K_RIGHT in self._keys_held:
            turn -= 1.0

        self.data.player.move(forward, strafe, dt)
        self.data.player.turn(turn, dt)

        # Collision detection with walls
        self._apply_collision()

    def _apply_collision(self) -> None:
        """Keep player inside the map bounds."""
        # Simple bounding box collision
        margin = 0.5
        x, y = self.data.player.x, self.data.player.y

        # Outer walls
        x = max(-19.5 + margin, min(19.5 - margin, x))
        y = max(-19.5 + margin, min(19.5 - margin, y))

        # Inner pillar collision
        if -2.5 < x < 2.5 and -2.5 < y < 2.5:
            # Push out to nearest edge
            dx_left = x - (-2.5)
            dx_right = 2.5 - x
            dy_bottom = y - (-2.5)
            dy_top = 2.5 - y

            min_dist = min(dx_left, dx_right, dy_bottom, dy_top)
            if min_dist == dx_left:
                x = -2.5 - margin
            elif min_dist == dx_right:
                x = 2.5 + margin
            elif min_dist == dy_bottom:
                y = -2.5 - margin
            else:
                y = 2.5 + margin

        self.data.player.x = x
        self.data.player.y = y

    def _update_einsteins(self, dt: float) -> None:
        """Update Einstein positions (they slowly approach the player)."""
        if self.data.progress < Progress.CHAPTER_7_COMPLETE:
            return

        player_pos = self.data.player.position
        einstein_speed = 1.0  # Units per second

        for einstein in self.data.einsteins:
            if einstein.active and not einstein.summed:
                dx = player_pos[0] - einstein.x
                dy = player_pos[1] - einstein.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 2.0:  # Don't get too close
                    einstein.x += (dx / dist) * einstein_speed * dt
                    einstein.y += (dy / dist) * einstein_speed * dt

    def render(self) -> None:
        """Render the current frame."""
        self.renderer.render(self.data, self.map)
