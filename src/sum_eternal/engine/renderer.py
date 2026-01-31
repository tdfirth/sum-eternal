"""
Renderer - Pygame rendering for all game states.

Handles:
- Title screen
- Debug visualization (Chapters 1-2)
- 2D map view (Chapters 3-4)
- 3D raycasting view (Chapters 5+)
- Einstein sprites
- Error overlays
- Victory screen
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from sum_eternal.bridge import Progress

if TYPE_CHECKING:
    from pygame import Surface
    from sum_eternal.engine.game import GameData, GameState
    from sum_eternal.engine.map import Map
    from sum_eternal.engine.assets import Assets


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (100, 100, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)


class Renderer:
    """Main renderer for Sum Eternal."""

    def __init__(self, screen: Surface, assets: Assets) -> None:
        self.screen = screen
        self.assets = assets
        self.width = screen.get_width()
        self.height = screen.get_height()

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)

        # Pre-calculate ray angles for raycasting
        self.num_rays = 320
        self.fov = math.pi / 3  # 60 degrees

    def render(self, data: GameData, game_map: Map) -> None:
        """Render the current frame based on game state."""
        from sum_eternal.engine.game import GameState

        self.screen.fill(BLACK)

        if data.state == GameState.TITLE:
            self._render_title(data)
        elif data.state == GameState.VICTORY:
            self._render_victory(data)
        elif data.state in (GameState.TUTORIAL, GameState.NIGHTMARE):
            self._render_gameplay(data, game_map)

        # Render error overlay if present
        if data.error_message:
            self._render_error_overlay(data.error_message)

    def _render_title(self, data: GameData) -> None:
        """Render the title screen."""
        # Title
        title = self.font_large.render("SUM ETERNAL", True, RED)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.font_medium.render("The Notation Awaits", True, GRAY)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, self.height // 3 + 50))
        self.screen.blit(subtitle, subtitle_rect)

        # Instructions
        if data.progress == Progress.NOTHING:
            instructions = [
                "Open Claude Code in this project directory",
                "to begin your training.",
                "",
                "The Einsteins are not enemies.",
                "They are trials.",
            ]
        else:
            instructions = [
                f"Progress: {data.progress.name}",
                "",
                "Press any key to continue",
            ]

        y = self.height // 2 + 50
        for line in instructions:
            text = self.font_small.render(line, True, WHITE)
            text_rect = text.get_rect(center=(self.width // 2, y))
            self.screen.blit(text, text_rect)
            y += 30

    def _render_victory(self, data: GameData) -> None:
        """Render the victory screen."""
        # Victory message
        title = self.font_large.render("YOU ARE WORTHY", True, GREEN)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title, title_rect)

        subtitle = self.font_medium.render("The notation recognizes you.", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, self.height // 4 + 50))
        self.screen.blit(subtitle, subtitle_rect)

        # Stats
        stats = [
            f"Functions mastered: 42/42",
            f"Einsteins summed: {data.einsteins_summed}",
            f"Time: {self._format_time(data.total_time)}",
        ]

        y = self.height // 2
        for line in stats:
            text = self.font_small.render(line, True, CYAN)
            text_rect = text.get_rect(center=(self.width // 2, y))
            self.screen.blit(text, text_rect)
            y += 30

        # Options
        options = [
            "[N] Nightmare Mode",
            "[Q] Return to Title",
        ]

        y = self.height * 3 // 4
        for line in options:
            text = self.font_small.render(line, True, YELLOW)
            text_rect = text.get_rect(center=(self.width // 2, y))
            self.screen.blit(text, text_rect)
            y += 30

    def _render_gameplay(self, data: GameData, game_map: Map) -> None:
        """Render the main gameplay view."""
        if data.progress < Progress.CHAPTER_1_COMPLETE:
            self._render_waiting(data)
        elif data.progress < Progress.CHAPTER_3_COMPLETE:
            self._render_debug_view(data)
        elif data.progress < Progress.CHAPTER_5_COMPLETE:
            self._render_2d_map(data, game_map)
        else:
            self._render_3d_view(data, game_map)

        # Always show progress bar at bottom
        self._render_progress_bar(data)

    def _render_waiting(self, data: GameData) -> None:
        """Render waiting for first implementation."""
        text = self.font_medium.render("SYSTEMS INITIALIZING...", True, GREEN)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, text_rect)

        hint = self.font_small.render("Complete c01_first_blood.py to begin", True, GRAY)
        hint_rect = hint.get_rect(center=(self.width // 2, self.height // 2 + 40))
        self.screen.blit(hint, hint_rect)

    def _render_debug_view(self, data: GameData) -> None:
        """Render the debug visualization for Chapters 1-2."""
        # Header
        header = self.font_medium.render("SUM ETERNAL - SYSTEMS INITIALIZING", True, GREEN)
        self.screen.blit(header, (20, 20))

        # Draw a border
        pygame.draw.rect(self.screen, GREEN, (10, 10, self.width - 20, self.height - 60), 2)

        # Show vector/matrix operations
        y = 70
        operations = []

        if data.progress >= Progress.CHAPTER_1_COMPLETE:
            operations.extend([
                "VECTOR OPS: ONLINE",
                "  sum(v) = 6.0           [OK]",
                "  a . b = 32.0           [OK]",
                "  a x b = [[...]]        [OK]",
            ])

        if data.progress >= Progress.CHAPTER_2_COMPLETE:
            operations.extend([
                "",
                "MATRIX OPS: ONLINE",
                "  transpose(M)           [OK]",
                "  trace(M) = 15.0        [OK]",
                "  diag(M) = [1, 5, 9]    [OK]",
            ])

        for line in operations:
            color = GREEN if "[OK]" in line else WHITE
            text = self.font_small.render(line, True, color)
            self.screen.blit(text, (30, y))
            y += 25

    def _render_2d_map(self, data: GameData, game_map: Map) -> None:
        """Render the 2D top-down map view."""
        # Scale: map is 40x40, fit to screen with margin
        margin = 50
        scale = min((self.width - 2 * margin) / 40, (self.height - 2 * margin - 50) / 40)
        offset_x = self.width // 2
        offset_y = (self.height - 50) // 2

        def world_to_screen(x: float, y: float) -> tuple[int, int]:
            return (int(offset_x + x * scale), int(offset_y - y * scale))

        # Draw walls
        for wall in game_map.walls:
            start = world_to_screen(wall.x1, wall.y1)
            end = world_to_screen(wall.x2, wall.y2)
            pygame.draw.line(self.screen, wall.color, start, end, 3)

        # Draw player
        px, py = world_to_screen(data.player.x, data.player.y)
        pygame.draw.circle(self.screen, CYAN, (px, py), 8)

        # Draw player direction
        dir_len = 20
        dir_x = px + int(math.cos(data.player.angle) * dir_len)
        dir_y = py - int(math.sin(data.player.angle) * dir_len)
        pygame.draw.line(self.screen, CYAN, (px, py), (dir_x, dir_y), 2)

        # Draw ray fan if Chapter 4 complete
        if data.progress >= Progress.CHAPTER_4_COMPLETE:
            num_rays = 32  # Reduced for 2D view
            for i in range(num_rays):
                angle = data.player.angle - self.fov / 2 + (i / num_rays) * self.fov
                ray_len = 100
                ray_x = px + int(math.cos(angle) * ray_len)
                ray_y = py - int(math.sin(angle) * ray_len)
                pygame.draw.line(self.screen, (50, 50, 50), (px, py), (ray_x, ray_y), 1)

        # Header
        header = self.font_small.render("2D MAP VIEW - WASD to move, arrows to turn", True, WHITE)
        self.screen.blit(header, (20, 20))

    def _render_3d_view(self, data: GameData, game_map: Map) -> None:
        """Render the 3D raycasted view."""
        # This is where the actual raycasting happens
        # Uses student-implemented functions from the bridge

        from sum_eternal import bridge

        # Check if we have the necessary functions
        if not bridge.has_raycasting_functions():
            self._render_3d_fallback(data, game_map)
            return

        # Generate rays
        player_pos = (data.player.x, data.player.y)
        player_angle = data.player.angle

        # Cast rays and render walls
        try:
            distances, wall_colors = bridge.cast_all_rays(
                player_pos, player_angle, self.fov, self.num_rays, game_map.wall_data
            )

            # Render wall columns
            self._render_wall_columns(distances, wall_colors, data.progress)

            # Render floor and ceiling
            if data.progress >= Progress.CHAPTER_6_COMPLETE:
                self._render_floor_ceiling()

            # Render Einsteins as sprites
            if data.progress >= Progress.CHAPTER_7_COMPLETE:
                self._render_einsteins(data, distances)

        except Exception as e:
            # If raycasting fails, show fallback
            self._render_3d_fallback(data, game_map)

        # Render minimap in corner
        self._render_minimap(data, game_map)

        # Render crosshair
        cx, cy = self.width // 2, self.height // 2
        pygame.draw.line(self.screen, WHITE, (cx - 10, cy), (cx + 10, cy), 2)
        pygame.draw.line(self.screen, WHITE, (cx, cy - 10), (cx, cy + 10), 2)

    def _render_3d_fallback(self, data: GameData, game_map: Map) -> None:
        """Render a simple 3D view when student functions aren't working."""
        # Simple gradient sky/floor
        self.screen.fill((20, 20, 40), (0, 0, self.width, self.height // 2))
        self.screen.fill((40, 40, 40), (0, self.height // 2, self.width, self.height // 2))

        text = self.font_medium.render("3D VIEW - Complete Chapter 5 functions", True, WHITE)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, text_rect)

    def _render_wall_columns(self, distances: list[float], wall_colors: list[tuple[int, int, int]], progress: Progress) -> None:
        """Render wall columns based on ray distances."""
        col_width = self.width // len(distances)
        max_dist = 30.0

        for i, (dist, color) in enumerate(zip(distances, wall_colors)):
            if dist <= 0 or dist > max_dist:
                continue

            # Calculate wall height
            wall_height = int(self.height / (dist + 0.1))
            wall_height = min(wall_height, self.height)

            # Apply distance shading if Chapter 6 complete
            if progress >= Progress.CHAPTER_6_COMPLETE:
                shade = max(0.2, 1.0 - dist / max_dist)
                color = tuple(int(c * shade) for c in color)

            # Draw column
            top = (self.height - wall_height) // 2
            x = i * col_width
            pygame.draw.rect(self.screen, color, (x, top, col_width + 1, wall_height))

    def _render_floor_ceiling(self) -> None:
        """Render floor and ceiling gradients."""
        # Simple gradient for now
        mid = self.height // 2
        for y in range(mid):
            # Ceiling
            shade = int(20 + (y / mid) * 20)
            pygame.draw.line(self.screen, (shade, shade, shade + 20), (0, y), (self.width, y))

        for y in range(mid, self.height):
            # Floor
            shade = int(40 + ((y - mid) / mid) * 20)
            pygame.draw.line(self.screen, (shade, shade, shade), (0, y), (self.width, y))

    def _render_einsteins(self, data: GameData, wall_distances: list[float]) -> None:
        """Render Einstein sprites."""
        from sum_eternal import bridge

        for einstein in data.einsteins:
            if not einstein.active or einstein.summed:
                continue

            # Calculate Einstein's screen position and size
            dx = einstein.x - data.player.x
            dy = einstein.y - data.player.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.5 or dist > 25.0:
                continue

            # Calculate angle to Einstein
            angle_to_einstein = math.atan2(dy, dx)
            angle_diff = angle_to_einstein - data.player.angle

            # Normalize angle difference
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Check if in FOV
            if abs(angle_diff) > self.fov / 2:
                continue

            # Calculate screen X position
            screen_x = int(self.width / 2 + (angle_diff / (self.fov / 2)) * (self.width / 2))

            # Calculate sprite size based on distance
            sprite_size = int(self.height / (dist + 0.1) * 0.8)
            sprite_size = min(sprite_size, self.height)

            # Check if Einstein is behind a wall
            ray_index = int((screen_x / self.width) * len(wall_distances))
            ray_index = max(0, min(ray_index, len(wall_distances) - 1))
            if wall_distances[ray_index] < dist:
                continue  # Behind wall

            # Draw Einstein sprite
            sprite = self.assets.get_einstein_sprite(sprite_size)
            if sprite:
                sprite_rect = sprite.get_rect(center=(screen_x, self.height // 2))
                self.screen.blit(sprite, sprite_rect)
            else:
                # Fallback: draw a circle
                pygame.draw.circle(self.screen, (255, 200, 100), (screen_x, self.height // 2), sprite_size // 4)

    def _render_minimap(self, data: GameData, game_map: Map) -> None:
        """Render a small minimap in the corner."""
        map_size = 100
        margin = 10
        scale = map_size / 40

        # Background
        pygame.draw.rect(self.screen, (20, 20, 20), (margin, margin, map_size, map_size))
        pygame.draw.rect(self.screen, GRAY, (margin, margin, map_size, map_size), 1)

        def world_to_minimap(x: float, y: float) -> tuple[int, int]:
            return (int(margin + map_size // 2 + x * scale),
                    int(margin + map_size // 2 - y * scale))

        # Draw walls
        for wall in game_map.walls:
            start = world_to_minimap(wall.x1, wall.y1)
            end = world_to_minimap(wall.x2, wall.y2)
            pygame.draw.line(self.screen, GRAY, start, end, 1)

        # Draw player
        px, py = world_to_minimap(data.player.x, data.player.y)
        pygame.draw.circle(self.screen, CYAN, (px, py), 3)

        # Draw direction
        dir_len = 8
        dir_x = px + int(math.cos(data.player.angle) * dir_len)
        dir_y = py - int(math.sin(data.player.angle) * dir_len)
        pygame.draw.line(self.screen, CYAN, (px, py), (dir_x, dir_y), 1)

        # Draw Einsteins
        for einstein in data.einsteins:
            if einstein.active and not einstein.summed:
                ex, ey = world_to_minimap(einstein.x, einstein.y)
                pygame.draw.circle(self.screen, YELLOW, (ex, ey), 2)

    def _render_progress_bar(self, data: GameData) -> None:
        """Render progress bar at bottom of screen."""
        bar_height = 40
        y = self.height - bar_height

        # Background
        pygame.draw.rect(self.screen, DARK_GRAY, (0, y, self.width, bar_height))

        # Progress text
        progress_name = data.progress.name.replace("_", " ").title()
        text = self.font_tiny.render(f"Progress: {progress_name}", True, WHITE)
        self.screen.blit(text, (10, y + 10))

        # Controls hint
        controls = "WASD: Move | Arrows: Turn | Space: Sum"
        hint = self.font_tiny.render(controls, True, GRAY)
        self.screen.blit(hint, (self.width - hint.get_width() - 10, y + 10))

    def _render_error_overlay(self, message: str) -> None:
        """Render error message overlay."""
        # Semi-transparent red background
        overlay = pygame.Surface((self.width, 80))
        overlay.fill((100, 0, 0))
        overlay.set_alpha(200)
        self.screen.blit(overlay, (0, 0))

        # Error icon and message
        icon = self.font_medium.render("!", True, YELLOW)
        self.screen.blit(icon, (20, 25))

        # Truncate message if too long
        if len(message) > 80:
            message = message[:77] + "..."

        text = self.font_small.render(message, True, WHITE)
        self.screen.blit(text, (50, 15))

        hint = self.font_tiny.render("Check console for details. The notation demands precision.", True, GRAY)
        self.screen.blit(hint, (50, 45))

    def _format_time(self, seconds: float) -> str:
        """Format seconds as h:mm:ss or m:ss."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        else:
            return f"{minutes}m {secs:02d}s"
