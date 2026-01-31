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
DARK_GREEN = (0, 150, 0)
RED = (255, 0, 0)
DARK_RED = (150, 0, 0)
BLOOD_RED = (120, 0, 0)
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
        self.font_title = pygame.font.Font(None, 72)   # Title screen main
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
        """Render the title screen with DOOM aesthetic."""
        # Dark gradient background (blood red to black)
        for y in range(self.height):
            shade = int(30 * (1 - y / self.height))
            color = (shade, 0, 0)
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

        # Decorative border lines (DOOM style)
        border_color = DARK_RED
        pygame.draw.rect(self.screen, border_color, (10, 10, self.width - 20, self.height - 20), 3)
        pygame.draw.rect(self.screen, RED, (15, 15, self.width - 30, self.height - 30), 1)

        # Main title with shadow effect
        title_text = "SUM ETERNAL"
        # Shadow
        shadow = self.font_title.render(title_text, True, BLOOD_RED)
        shadow_rect = shadow.get_rect(center=(self.width // 2 + 3, self.height // 4 + 3))
        self.screen.blit(shadow, shadow_rect)
        # Main title
        title = self.font_title.render(title_text, True, RED)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 4))
        self.screen.blit(title, title_rect)

        # Decorative line under title
        line_y = self.height // 4 + 40
        pygame.draw.line(self.screen, DARK_RED, (self.width // 4, line_y), (3 * self.width // 4, line_y), 2)

        # Subtitle
        subtitle = self.font_medium.render("The Notation Awaits", True, GRAY)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, self.height // 4 + 70))
        self.screen.blit(subtitle, subtitle_rect)

        # Progress level display
        if data.progress != Progress.NOTHING:
            progress_name = data.progress.name.replace("_", " ").title()
            progress_text = f"[ {progress_name} ]"
            progress_render = self.font_medium.render(progress_text, True, GREEN)
            progress_rect = progress_render.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(progress_render, progress_rect)

        # Instructions
        if data.progress == Progress.NOTHING:
            instructions = [
                ("Open Claude Code in this project directory", GRAY),
                ("to begin your training.", GRAY),
                ("", GRAY),
                ("The Einsteins are not enemies.", WHITE),
                ("They are trials.", WHITE),
                ("", GRAY),
                ("RIP AND TENSOR", RED),
            ]
        else:
            instructions = [
                ("Press any key to continue", WHITE),
                ("", GRAY),
                ("WASD - Move   Arrows - Turn   Space - Sum", GRAY),
            ]

        y = self.height // 2 + 60
        for line, color in instructions:
            if line:
                text = self.font_small.render(line, True, color)
                text_rect = text.get_rect(center=(self.width // 2, y))
                self.screen.blit(text, text_rect)
            y += 28

        # Footer
        footer = self.font_tiny.render("An einsum tutorial in DOOM style", True, DARK_GRAY)
        footer_rect = footer.get_rect(center=(self.width // 2, self.height - 40))
        self.screen.blit(footer, footer_rect)

    def _render_victory(self, data: GameData) -> None:
        """Render the victory screen with DOOM aesthetic."""
        # Dark gradient background (green tint for victory)
        for y in range(self.height):
            shade = int(20 * (1 - y / self.height))
            color = (0, shade, 0)
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

        # Decorative border lines (victory green)
        pygame.draw.rect(self.screen, DARK_GREEN, (10, 10, self.width - 20, self.height - 20), 3)
        pygame.draw.rect(self.screen, GREEN, (15, 15, self.width - 30, self.height - 30), 1)

        # Main title with shadow effect
        title_text = "YOU ARE WORTHY"
        # Shadow
        shadow = self.font_title.render(title_text, True, DARK_GREEN)
        shadow_rect = shadow.get_rect(center=(self.width // 2 + 3, self.height // 5 + 3))
        self.screen.blit(shadow, shadow_rect)
        # Main title
        title = self.font_title.render(title_text, True, GREEN)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 5))
        self.screen.blit(title, title_rect)

        # Subtitle
        subtitle = self.font_medium.render("The notation recognizes you.", True, WHITE)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, self.height // 5 + 60))
        self.screen.blit(subtitle, subtitle_rect)

        # Decorative line
        line_y = self.height // 5 + 90
        pygame.draw.line(self.screen, DARK_GREEN, (self.width // 4, line_y), (3 * self.width // 4, line_y), 2)

        # Stats header
        stats_header = self.font_medium.render("- FINAL STATISTICS -", True, GRAY)
        stats_header_rect = stats_header.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(stats_header, stats_header_rect)

        # Stats
        stats = [
            ("Functions mastered:", "42/42", GREEN),
            ("Einsteins summed:", str(data.einsteins_summed), CYAN),
            ("Total time:", self._format_time(data.total_time), CYAN),
        ]

        y = self.height // 2
        for label, value, color in stats:
            # Label
            label_text = self.font_small.render(label, True, GRAY)
            label_rect = label_text.get_rect(midright=(self.width // 2 - 10, y))
            self.screen.blit(label_text, label_rect)
            # Value
            value_text = self.font_small.render(value, True, color)
            value_rect = value_text.get_rect(midleft=(self.width // 2 + 10, y))
            self.screen.blit(value_text, value_rect)
            y += 30

        # Options header
        options_line_y = self.height * 3 // 4 - 40
        pygame.draw.line(self.screen, DARK_GRAY, (self.width // 3, options_line_y), (2 * self.width // 3, options_line_y), 1)

        # Options
        options = [
            ("[N] Enter Nightmare Mode", RED),
            ("[Q] Return to Title", YELLOW),
        ]

        y = self.height * 3 // 4
        for line, color in options:
            text = self.font_small.render(line, True, color)
            text_rect = text.get_rect(center=(self.width // 2, y))
            self.screen.blit(text, text_rect)
            y += 30

        # Footer - battle cry
        footer = self.font_medium.render("RIP AND TENSOR", True, RED)
        footer_rect = footer.get_rect(center=(self.width // 2, self.height - 50))
        self.screen.blit(footer, footer_rect)

    def _render_gameplay(self, data: GameData, game_map: Map) -> None:
        """Render the main gameplay view."""
        # Show debug view for chapters 1-4 (until 3D unlocks)
        if data.progress < Progress.CHAPTER_5_COMPLETE:
            self._render_debug_view(data, game_map)
        else:
            self._render_3d_view(data, game_map)

        # Always show progress bar at bottom
        self._render_progress_bar(data)

    def _render_debug_view(self, data: GameData, game_map: Map) -> None:
        """Render the debug visualization for Chapters 1-4.

        Shows a terminal/console aesthetic with operation results and
        checkmarks for completed functions. Only shows the current chapter.
        """
        # Terminal-style colors
        TERM_GREEN = (0, 255, 128)
        TERM_AMBER = (255, 191, 0)
        TERM_DIM = (0, 128, 64)
        TERM_BG = (10, 20, 15)

        # Draw terminal background
        margin = 20
        term_rect = (margin, margin, self.width - 2 * margin, self.height - 80)
        pygame.draw.rect(self.screen, TERM_BG, term_rect)
        pygame.draw.rect(self.screen, TERM_GREEN, term_rect, 2)

        # Determine current chapter
        if data.progress < Progress.CHAPTER_1_COMPLETE:
            current_chapter = 1
        elif data.progress < Progress.CHAPTER_2_COMPLETE:
            current_chapter = 2
        elif data.progress < Progress.CHAPTER_3_COMPLETE:
            current_chapter = 3
        else:
            current_chapter = 4

        # Header
        header_y = margin + 15
        title = f"SUM ETERNAL - CHAPTER {current_chapter}"
        title_color = TERM_AMBER
        header_text = self.font_medium.render(title, True, title_color)
        header_rect = header_text.get_rect(centerx=self.width // 2, y=header_y)
        self.screen.blit(header_text, header_rect)

        # Draw separator line
        sep_y = header_y + 35
        sep_line = "=" * (len(title) + 10)
        sep_text = self.font_small.render(sep_line, True, TERM_GREEN)
        sep_rect = sep_text.get_rect(centerx=self.width // 2, y=sep_y)
        self.screen.blit(sep_text, sep_rect)

        # Starting Y position for operations
        y = sep_y + 40
        left_margin = margin + 30

        # Get chapter status from unified test system
        from solutions.test_cases import get_chapter_status

        # Display names for functions
        DISPLAY_NAMES = {
            # Chapter 1
            "vector_sum": "sum(v)",
            "element_multiply": "a * b",
            "dot_product": "a . b",
            "outer_product": "a (x) b",
            "matrix_vector_mul": "M @ v",
            "matrix_matrix_mul": "M @ M",
            # Chapter 2
            "transpose": "M^T",
            "trace": "trace(M)",
            "diag_extract": "diag(M)",
            "sum_rows": "sum rows",
            "sum_cols": "sum cols",
            "frobenius_norm_sq": "||M||^2",
            # Chapter 3
            "batch_vector_sum": "batch sum",
            "batch_dot_pairwise": "batch dot",
            "batch_magnitude_sq": "batch |v|²",
            "all_pairs_dot": "all pairs",
            "batch_matrix_vector": "batch M@v",
            "batch_outer": "batch outer",
            # Chapter 4
            "angles_to_directions": "angles->dirs",
            "rotate_vectors": "rotate vecs",
            "normalize_vectors": "normalize",
            "scale_vectors": "scale vecs",
        }

        SECTION_NAMES = {
            1: "VECTOR OPS",
            2: "MATRIX OPS",
            3: "BATCH OPS",
            4: "RAY GENERATION",
        }

        # Get status and render
        section_header = self.font_small.render(SECTION_NAMES.get(current_chapter, ""), True, TERM_AMBER)
        self.screen.blit(section_header, (left_margin, y))
        y += 30

        status = get_chapter_status(current_chapter)
        self._render_function_status(status, DISPLAY_NAMES, y, left_margin, TERM_GREEN, TERM_DIM)

    def _render_function_status(
        self,
        status: dict[str, tuple[int, int, bool | None]],
        display_names: dict[str, str],
        y: int,
        left_margin: int,
        color_ok: tuple[int, int, int],
        color_pending: tuple[int, int, int]
    ) -> int:
        """Render function status from unified test system.

        Args:
            status: dict mapping func_name -> (passed, total, is_implemented)
            display_names: dict mapping func_name -> display string
            y: starting Y position
            left_margin: X position
            color_ok: color for passing functions
            color_pending: color for not-yet-implemented functions

        Returns the new Y position after rendering.
        """
        FAIL_COLOR = (255, 80, 80)  # Red for errors

        for func_name, (passed, total, is_impl) in status.items():
            display = display_names.get(func_name, func_name)

            if is_impl is None:
                # Not implemented yet
                line = f"  {display:16}   {'---':8}  [ ? ]"
                color = color_pending
            elif is_impl is True:
                # All tests passing
                line = f"  {display:16}   {passed}/{total:5}  [OK]"
                color = color_ok
            else:
                # Some tests failing
                line = f"  {display:16}   {passed}/{total:5}  [FAIL]"
                color = FAIL_COLOR

            text = self.font_small.render(line, True, color)
            self.screen.blit(text, (left_margin, y))
            y += 22

        return y

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

        # Draw player direction (20 units in world space)
        world_dir_len = 20
        dir_end = world_to_screen(
            data.player.x + math.cos(data.player.angle) * world_dir_len,
            data.player.y + math.sin(data.player.angle) * world_dir_len
        )
        pygame.draw.line(self.screen, CYAN, (px, py), dir_end, 2)

        # Draw ray fan if Chapter 4 complete (rays drawn behind player)
        if data.progress >= Progress.CHAPTER_4_COMPLETE:
            num_rays = 32  # Reduced for 2D view
            ray_world_len = 30  # Length in world units
            for i in range(num_rays):
                angle = data.player.angle - self.fov / 2 + (i / num_rays) * self.fov
                ray_end = world_to_screen(
                    data.player.x + math.cos(angle) * ray_world_len,
                    data.player.y + math.sin(angle) * ray_world_len
                )
                pygame.draw.line(self.screen, DARK_GRAY, (px, py), ray_end, 1)

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
        else:
            # Generate rays
            player_pos = (data.player.x, data.player.y)
            player_angle = data.player.angle

            # Cast rays and render walls
            try:
                distances, wall_colors = bridge.cast_all_rays(
                    player_pos, player_angle, self.fov, self.num_rays, game_map.wall_data
                )

                # Render floor and ceiling first (background)
                self._render_floor_ceiling()

                # Render wall columns on top of floor/ceiling
                self._render_wall_columns(distances, wall_colors, data.progress)

                # Render Einsteins as sprites
                if data.progress >= Progress.CHAPTER_7_COMPLETE:
                    self._render_einsteins(data, distances)

            except Exception:
                # If raycasting fails, show fallback
                self._render_3d_fallback(data, game_map)

        # Always render minimap in corner
        self._render_minimap(data, game_map)

        # Always render crosshair
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
        """Render Einstein sprites.

        Renders Einstein sprites in the 3D view when progress >= Chapter 7.
        - Calculates screen position from world position relative to player
        - Checks if Einstein is within player's FOV (60 degrees)
        - Applies wall occlusion (don't render behind walls)
        - Scales sprite based on distance from player
        - Sorts Einsteins back-to-front for correct rendering order
        """
        # First pass: collect visible Einsteins with their render data
        visible_einsteins: list[tuple[float, int, int, int]] = []  # (dist, screen_x, sprite_size, idx)

        for idx, einstein in enumerate(data.einsteins):
            if not einstein.active or einstein.summed:
                continue

            # Calculate Einstein's position relative to player
            dx = einstein.x - data.player.x
            dy = einstein.y - data.player.y
            dist = math.sqrt(dx * dx + dy * dy)

            # Skip if too close or too far
            if dist < 0.5 or dist > 25.0:
                continue

            # Calculate angle to Einstein relative to player facing direction
            angle_to_einstein = math.atan2(dy, dx)
            angle_diff = angle_to_einstein - data.player.angle

            # Normalize angle difference to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Check if within FOV (60 degrees = pi/3 radians)
            if abs(angle_diff) > self.fov / 2:
                continue

            # Calculate screen X position
            # angle_diff / (fov/2) gives us a value in [-1, 1]
            # Map to screen: center + offset
            screen_x = int(self.width / 2 + (angle_diff / (self.fov / 2)) * (self.width / 2))

            # Calculate sprite size based on distance (closer = larger)
            sprite_size = int(self.height / (dist + 0.1) * 0.8)
            sprite_size = max(8, min(sprite_size, self.height))

            # Check if Einstein is behind a wall (wall occlusion)
            ray_index = int((screen_x / self.width) * len(wall_distances))
            ray_index = max(0, min(ray_index, len(wall_distances) - 1))
            if wall_distances[ray_index] < dist:
                continue  # Einstein is occluded by wall

            # Add to visible list for sorting
            visible_einsteins.append((dist, screen_x, sprite_size, idx))

        # Sort back-to-front (farthest first, so closer Einsteins render on top)
        visible_einsteins.sort(key=lambda x: -x[0])

        # Second pass: render in sorted order
        for dist, screen_x, sprite_size, idx in visible_einsteins:
            sprite = self.assets.get_einstein_sprite(sprite_size)
            if sprite:
                sprite_rect = sprite.get_rect(center=(screen_x, self.height // 2))
                self.screen.blit(sprite, sprite_rect)
            else:
                # Fallback: draw a placeholder circle
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

        # Get function-level progress for current chapter
        func_progress = self._get_function_progress(data)

        # Left side: chapter progress
        progress_name = data.progress.name.replace("_", " ").title()
        text = self.font_tiny.render(f"Progress: {progress_name}", True, WHITE)
        self.screen.blit(text, (10, y + 5))

        # Function progress bar below the text
        if func_progress:
            completed, total, chapter_name = func_progress
            bar_width = 150
            bar_x = 10
            bar_y = y + 22
            bar_h = 12

            # Background bar
            pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_width, bar_h))

            # Fill bar
            fill_width = int((completed / total) * bar_width)
            if fill_width > 0:
                fill_color = GREEN if completed == total else YELLOW
                pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, bar_h))

            # Border
            pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_h), 1)

            # Count text
            count_text = self.font_tiny.render(f"{chapter_name}: {completed}/{total}", True, WHITE)
            self.screen.blit(count_text, (bar_x + bar_width + 10, bar_y - 2))

        # Controls hint (right side)
        controls = "WASD: Move | Arrows: Turn | Space: Sum"
        hint = self.font_tiny.render(controls, True, GRAY)
        self.screen.blit(hint, (self.width - hint.get_width() - 10, y + 12))

    def _get_function_progress(self, data: GameData) -> tuple[int, int, str] | None:
        """Get current chapter's function progress using unified test system."""
        from solutions.test_cases import get_chapter_status

        # Determine which chapter to show progress for
        if data.progress < Progress.CHAPTER_1_COMPLETE:
            chapter = 1
        elif data.progress < Progress.CHAPTER_2_COMPLETE:
            chapter = 2
        elif data.progress < Progress.CHAPTER_3_COMPLETE:
            chapter = 3
        elif data.progress < Progress.CHAPTER_4_COMPLETE:
            chapter = 4
        elif data.progress < Progress.CHAPTER_5_COMPLETE:
            chapter = 5
        elif data.progress < Progress.CHAPTER_6_COMPLETE:
            chapter = 6
        elif data.progress < Progress.CHAPTER_7_COMPLETE:
            chapter = 7
        elif data.progress < Progress.CHAPTER_8_COMPLETE:
            chapter = 8
        else:
            return None

        status = get_chapter_status(chapter)
        if not status:
            return None

        # Count functions with all tests passing
        completed = sum(1 for _, (_, _, is_impl) in status.items() if is_impl is True)
        total = len(status)

        return (completed, total, f"Ch{chapter}")

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
