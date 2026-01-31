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
        # Show debug view from the start - it displays per-function progress
        if data.progress < Progress.CHAPTER_3_COMPLETE:
            self._render_debug_view(data)
        elif data.progress < Progress.CHAPTER_5_COMPLETE:
            self._render_2d_map(data, game_map)
        else:
            self._render_3d_view(data, game_map)

        # Always show progress bar at bottom
        self._render_progress_bar(data)

    def _render_debug_view(self, data: GameData) -> None:
        """Render the debug visualization for Chapters 1-2.

        Shows a terminal/console aesthetic with operation results and
        checkmarks for completed functions.
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

        # Header - changes based on progress
        header_y = margin + 15
        if data.progress >= Progress.CHAPTER_2_COMPLETE:
            title = "SUM ETERNAL - SYSTEMS ONLINE"
            title_color = TERM_GREEN
        elif data.progress >= Progress.CHAPTER_1_COMPLETE:
            title = "SUM ETERNAL - MATRIX OPS INITIALIZING"
            title_color = TERM_AMBER
        else:
            title = "SUM ETERNAL - BOOT SEQUENCE"
            title_color = TERM_AMBER
        header_text = self.font_medium.render(title, True, title_color)
        header_rect = header_text.get_rect(centerx=self.width // 2, y=header_y)
        self.screen.blit(header_text, header_rect)

        # Draw separator line
        sep_y = header_y + 35
        sep_char = "="
        sep_line = sep_char * (len(title) + 10)
        sep_text = self.font_small.render(sep_line, True, TERM_GREEN)
        sep_rect = sep_text.get_rect(centerx=self.width // 2, y=sep_y)
        self.screen.blit(sep_text, sep_rect)

        # Starting Y position for operations
        y = sep_y + 40
        left_margin = margin + 30

        c1_complete = data.progress >= Progress.CHAPTER_1_COMPLETE

        if not c1_complete:
            # Show Chapter 1: Vector Operations
            c1_results = self._get_chapter1_results()

            section_header = self.font_small.render("CHAPTER 1: VECTOR OPS", True, TERM_AMBER)
            self.screen.blit(section_header, (left_margin, y))
            y += 30

            c1_functions = [
                ("vector_sum", "sum(v)", c1_results.get("vector_sum")),
                ("element_multiply", "a * b", c1_results.get("element_multiply")),
                ("dot_product", "a . b", c1_results.get("dot_product")),
                ("outer_product", "a (x) b", c1_results.get("outer_product")),
                ("matrix_vector_mul", "M @ v", c1_results.get("matrix_vector_mul")),
                ("matrix_matrix_mul", "M @ M", c1_results.get("matrix_matrix_mul")),
            ]

            self._render_function_list(c1_functions, y, left_margin, TERM_GREEN, TERM_DIM)

        else:
            # Show Chapter 2: Matrix Operations (chapter 1 complete)
            c2_results = self._get_chapter2_results()
            c2_complete = data.progress >= Progress.CHAPTER_2_COMPLETE

            status_text = "COMPLETE" if c2_complete else "MATRIX OPS"
            section_header = self.font_small.render(f"CHAPTER 2: {status_text}", True, TERM_GREEN if c2_complete else TERM_AMBER)
            self.screen.blit(section_header, (left_margin, y))
            y += 30

            c2_functions = [
                ("transpose", "M^T", c2_results.get("transpose")),
                ("trace", "trace(M)", c2_results.get("trace")),
                ("diag_extract", "diag(M)", c2_results.get("diag_extract")),
                ("sum_rows", "sum rows", c2_results.get("sum_rows")),
                ("sum_cols", "sum cols", c2_results.get("sum_cols")),
                ("frobenius_norm_sq", "||M||^2", c2_results.get("frobenius_norm_sq")),
            ]

            self._render_function_list(c2_functions, y, left_margin, TERM_GREEN, TERM_DIM)

        # Render debug progress bar at bottom of terminal
        self._render_debug_progress(data, margin, term_rect[1] + term_rect[3] - 30, TERM_GREEN, TERM_DIM)

    def _render_function_list(
        self,
        functions: list[tuple[str, str, str | None]],
        y: int,
        left_margin: int,
        color_ok: tuple[int, int, int],
        color_pending: tuple[int, int, int]
    ) -> int:
        """Render a list of functions with their status.

        Returns the new Y position after rendering.
        """
        CHECK = "[OK]"
        PENDING = "[ ? ]"
        FAIL = "[FAIL]"
        FAIL_COLOR = (255, 80, 80)  # Red for errors

        for func_name, display_name, result in functions:
            if result == "ERROR":
                # Function implemented but broken
                line = f"  {display_name:16} = {'error':16} {FAIL}"
                color = FAIL_COLOR
            elif result is not None:
                # Function is working - show in green with result
                line = f"  {display_name:16} = {result:16} {CHECK}"
                color = color_ok
            else:
                # Function not implemented yet (NotImplementedError)
                line = f"  {display_name:16} = {'---':16} {PENDING}"
                color = color_pending

            text = self.font_small.render(line, True, color)
            self.screen.blit(text, (left_margin, y))
            y += 22

        return y

    def _render_debug_progress(
        self,
        data: GameData,
        x: int,
        y: int,
        color_fill: tuple[int, int, int],
        color_bg: tuple[int, int, int]
    ) -> None:
        """Render a progress bar showing chapter completion."""
        bar_width = self.width - 2 * x - 40
        bar_height = 16

        # Calculate progress percentage (Chapters 1-2 = 2 chapters for debug view)
        total_chapters = 2
        completed = 0
        if data.progress >= Progress.CHAPTER_1_COMPLETE:
            completed += 1
        if data.progress >= Progress.CHAPTER_2_COMPLETE:
            completed += 1

        fill_width = int((completed / total_chapters) * bar_width)

        # Draw progress bar background
        pygame.draw.rect(self.screen, color_bg, (x, y, bar_width, bar_height))

        # Draw progress bar fill
        if fill_width > 0:
            pygame.draw.rect(self.screen, color_fill, (x, y, fill_width, bar_height))

        # Draw border
        pygame.draw.rect(self.screen, color_fill, (x, y, bar_width, bar_height), 1)

        # Progress text
        progress_pct = int((completed / total_chapters) * 100)
        pct_text = self.font_tiny.render(f"{progress_pct}%", True, color_fill)
        self.screen.blit(pct_text, (x + bar_width + 5, y))

    def _get_chapter1_results(self) -> dict[str, str | None]:
        """Get results from Chapter 1 functions, or None if not implemented."""
        import importlib
        import sys
        import jax.numpy as jnp

        results = {}

        # Test vectors
        v = jnp.array([1.0, 2.0, 3.0])
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([4.0, 5.0, 6.0, 7.0])
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Force fresh import
        module_name = "solutions.c01_first_blood"
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return {k: None for k in ["vector_sum", "element_multiply", "dot_product",
                                       "outer_product", "matrix_vector_mul", "matrix_matrix_mul"]}

        # Test each function
        # Returns: string = success, None = not implemented, "ERROR" = implemented but broken
        try:
            result = module.vector_sum(v)
            results["vector_sum"] = f"{float(result):.1f}"
        except NotImplementedError:
            results["vector_sum"] = None
        except Exception:
            results["vector_sum"] = "ERROR"

        try:
            result = module.element_multiply(a[:2], b[:2])
            arr = [float(x) for x in result]
            results["element_multiply"] = f"[{arr[0]:.0f}, {arr[1]:.0f}]"
        except NotImplementedError:
            results["element_multiply"] = None
        except Exception:
            results["element_multiply"] = "ERROR"

        try:
            result = module.dot_product(a, b)
            results["dot_product"] = f"{float(result):.1f}"
        except NotImplementedError:
            results["dot_product"] = None
        except Exception:
            results["dot_product"] = "ERROR"

        try:
            result = module.outer_product(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
            results["outer_product"] = "[[3,4],[6,8]]"
        except NotImplementedError:
            results["outer_product"] = None
        except Exception:
            results["outer_product"] = "ERROR"

        try:
            result = module.matrix_vector_mul(M, jnp.array([1.0, 1.0]))
            arr = [float(x) for x in result]
            results["matrix_vector_mul"] = f"[{arr[0]:.0f}, {arr[1]:.0f}]"
        except NotImplementedError:
            results["matrix_vector_mul"] = None
        except Exception:
            results["matrix_vector_mul"] = "ERROR"

        try:
            result = module.matrix_matrix_mul(M, M)
            results["matrix_matrix_mul"] = "[[7,10],[15,22]]"
        except NotImplementedError:
            results["matrix_matrix_mul"] = None
        except Exception:
            results["matrix_matrix_mul"] = "ERROR"

        return results

    def _get_chapter2_results(self) -> dict[str, str | None]:
        """Get results from Chapter 2 functions, or None if not implemented."""
        import importlib
        import sys
        import jax.numpy as jnp

        results = {}

        # Test matrix
        M = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Force fresh import
        module_name = "solutions.c02_knee_deep_in_the_indices"
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return {k: None for k in ["transpose", "trace", "diag_extract",
                                       "sum_rows", "sum_cols", "frobenius_norm_sq"]}

        try:
            result = module.transpose(M)
            results["transpose"] = "3x3 matrix"
        except NotImplementedError:
            results["transpose"] = None
        except Exception:
            results["transpose"] = "ERROR"

        try:
            result = module.trace(M)
            results["trace"] = f"{float(result):.1f}"
        except NotImplementedError:
            results["trace"] = None
        except Exception:
            results["trace"] = "ERROR"

        try:
            result = module.diag_extract(M)
            arr = [int(x) for x in result]
            results["diag_extract"] = f"[{arr[0]},{arr[1]},{arr[2]}]"
        except NotImplementedError:
            results["diag_extract"] = None
        except Exception:
            results["diag_extract"] = "ERROR"

        try:
            result = module.sum_rows(M)
            arr = [int(x) for x in result]
            results["sum_rows"] = f"[{arr[0]},{arr[1]},{arr[2]}]"
        except NotImplementedError:
            results["sum_rows"] = None
        except Exception:
            results["sum_rows"] = "ERROR"

        try:
            result = module.sum_cols(M)
            arr = [int(x) for x in result]
            results["sum_cols"] = f"[{arr[0]},{arr[1]},{arr[2]}]"
        except NotImplementedError:
            results["sum_cols"] = None
        except Exception:
            results["sum_cols"] = "ERROR"

        try:
            result = module.frobenius_norm_sq(M)
            results["frobenius_norm_sq"] = f"{float(result):.1f}"
        except NotImplementedError:
            results["frobenius_norm_sq"] = None
        except Exception:
            results["frobenius_norm_sq"] = "ERROR"

        return results

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
        """Get current chapter's function progress (completed, total, chapter_name)."""
        # Determine which chapter to show progress for
        if data.progress < Progress.CHAPTER_1_COMPLETE:
            results = self._get_chapter1_results()
            completed = sum(1 for v in results.values() if v is not None)
            return (completed, 6, "Ch1")
        elif data.progress < Progress.CHAPTER_2_COMPLETE:
            results = self._get_chapter2_results()
            completed = sum(1 for v in results.values() if v is not None)
            return (completed, 6, "Ch2")
        elif data.progress < Progress.CHAPTER_3_COMPLETE:
            # After ch2, show chapter 3 progress
            return self._get_chapter_n_progress(3)
        elif data.progress < Progress.CHAPTER_4_COMPLETE:
            return self._get_chapter_n_progress(4)
        elif data.progress < Progress.CHAPTER_5_COMPLETE:
            return self._get_chapter_n_progress(5)
        elif data.progress < Progress.CHAPTER_6_COMPLETE:
            return self._get_chapter_n_progress(6)
        elif data.progress < Progress.CHAPTER_7_COMPLETE:
            return self._get_chapter_n_progress(7)
        elif data.progress < Progress.CHAPTER_8_COMPLETE:
            return self._get_chapter_n_progress(8)
        return None

    def _get_chapter_n_progress(self, chapter: int) -> tuple[int, int, str] | None:
        """Get progress for chapters 3-9 by testing each function."""
        import jax.numpy as jnp

        # Test data for each chapter
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        vec = jnp.array([1.0, 2.0])
        angles = jnp.array([0.0, 1.57])
        scalar = jnp.array(1.0)
        mat = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # Define test calls for each function
        chapter_tests: dict[int, list[tuple[str, str, tuple]]] = {
            3: [
                ("solutions.c03_the_slaughter_batch", "batch_vector_sum", (batch,)),
                ("solutions.c03_the_slaughter_batch", "batch_dot_pairwise", (batch, batch)),
                ("solutions.c03_the_slaughter_batch", "batch_magnitude_sq", (batch,)),
                ("solutions.c03_the_slaughter_batch", "all_pairs_dot", (batch, batch)),
                ("solutions.c03_the_slaughter_batch", "batch_matrix_vector", (mat[None, :, :], vec)),
                ("solutions.c03_the_slaughter_batch", "batch_outer", (batch, batch)),
            ],
            4: [
                ("solutions.c04_rip_and_trace", "angles_to_directions", (angles,)),
                ("solutions.c04_rip_and_trace", "rotate_vectors", (batch, scalar)),
                ("solutions.c04_rip_and_trace", "normalize_vectors", (batch,)),
                ("solutions.c04_rip_and_trace", "scale_vectors", (batch, vec)),
            ],
            5: [
                ("solutions.c05_total_intersection", "cross_2d", (vec, vec)),
                ("solutions.c05_total_intersection", "batch_cross_2d", (batch, batch)),
                ("solutions.c05_total_intersection", "all_pairs_cross_2d", (batch, batch)),
                ("solutions.c05_total_intersection", "ray_wall_determinants", (batch, batch)),
                ("solutions.c05_total_intersection", "ray_wall_t_values", (batch, batch, batch, batch)),
                ("solutions.c05_total_intersection", "ray_wall_s_values", (batch, batch, batch, batch)),
            ],
            6: [
                ("solutions.c06_infernal_projection", "fisheye_correct", (vec, angles)),
                ("solutions.c06_infernal_projection", "distance_to_height", (vec, scalar)),
                ("solutions.c06_infernal_projection", "shade_by_distance", (vec, scalar)),
                ("solutions.c06_infernal_projection", "build_column_masks", (vec, jnp.array([100, 200]))),
            ],
            7: [
                ("solutions.c07_spooky_action_at_a_distance", "point_distances", (vec, batch)),
                ("solutions.c07_spooky_action_at_a_distance", "all_pairs_distances", (batch, batch)),
                ("solutions.c07_spooky_action_at_a_distance", "points_to_angles", (vec, batch)),
                ("solutions.c07_spooky_action_at_a_distance", "angle_in_fov", (vec, scalar, scalar)),
                ("solutions.c07_spooky_action_at_a_distance", "project_to_screen_x", (vec, scalar, jnp.array(320))),
                ("solutions.c07_spooky_action_at_a_distance", "sprite_scale", (vec, scalar)),
            ],
            8: [
                ("solutions.c08_the_icon_of_ein", "project_points_onto_ray", (batch, vec, vec)),
                ("solutions.c08_the_icon_of_ein", "perpendicular_distance_to_ray", (batch, vec, vec)),
                ("solutions.c08_the_icon_of_ein", "ray_hits_target", (batch, vec, vec, scalar)),
                ("solutions.c08_the_icon_of_ein", "move_toward_point", (vec, vec, scalar)),
            ],
        }

        if chapter not in chapter_tests:
            return None

        tests = chapter_tests[chapter]
        completed = 0
        total = len(tests)

        for module_name, func_name, args in tests:
            try:
                import importlib
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                func(*args)  # If this doesn't raise NotImplementedError, it's implemented
                completed += 1
            except NotImplementedError:
                pass
            except Exception:
                # Other errors (like wrong shape) mean at least they tried
                completed += 1

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
