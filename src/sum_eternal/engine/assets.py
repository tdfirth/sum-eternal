"""
Assets - Load and manage game assets.

Currently:
- Einstein sprite (placeholder or real)
- Fonts (using pygame defaults for now)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from pygame import Surface


def get_assets_dir() -> Path:
    """Get the assets directory path."""
    # Walk up from this file to find the assets/ directory
    current = Path(__file__).parent
    while current != current.parent:
        assets_dir = current / "assets"
        if assets_dir.exists():
            return assets_dir
        current = current.parent

    # Fallback: create assets dir in expected location
    expected = Path(__file__).parent.parent.parent.parent / "assets"
    expected.mkdir(exist_ok=True)
    return expected


class Assets:
    """Manages game assets."""

    def __init__(self) -> None:
        self.assets_dir = get_assets_dir()
        self._einstein_base: Surface | None = None
        self._einstein_cache: dict[int, Surface] = {}
        self._load_einstein()

    def _load_einstein(self) -> None:
        """Load or create the Einstein sprite."""
        einstein_path = self.assets_dir / "einstein.png"

        if einstein_path.exists():
            try:
                self._einstein_base = pygame.image.load(str(einstein_path)).convert_alpha()
                return
            except pygame.error as e:
                print(f"[Assets] Failed to load Einstein sprite: {e}")

        # Create placeholder sprite
        self._einstein_base = self._create_placeholder_einstein()

    def _create_placeholder_einstein(self) -> Surface:
        """Create a placeholder Einstein sprite."""
        size = 64
        surface = pygame.Surface((size, size), pygame.SRCALPHA)

        # Draw a ghostly circle with face
        center = size // 2
        radius = size // 2 - 4

        # Outer glow (cyan/blue)
        for i in range(3):
            alpha = 50 - i * 15
            color = (100, 200, 255, alpha)
            pygame.draw.circle(surface, color, (center, center), radius + 3 - i)

        # Main head (off-white, ghostly)
        pygame.draw.circle(surface, (220, 220, 230, 200), (center, center), radius)

        # Wild hair (white lines radiating from top)
        hair_color = (240, 240, 250, 180)
        for angle in [-0.8, -0.4, 0, 0.4, 0.8]:
            import math
            start_x = center + int(math.sin(angle) * (radius - 5))
            start_y = center - int(math.cos(angle) * (radius - 5))
            end_x = center + int(math.sin(angle) * (radius + 8))
            end_y = center - int(math.cos(angle) * (radius + 8))
            pygame.draw.line(surface, hair_color, (start_x, start_y), (end_x, end_y), 2)

        # Eyes (dark)
        eye_y = center - 5
        pygame.draw.circle(surface, (30, 30, 40, 255), (center - 10, eye_y), 4)
        pygame.draw.circle(surface, (30, 30, 40, 255), (center + 10, eye_y), 4)

        # Tongue sticking out (pink)
        tongue_color = (255, 150, 150, 255)
        tongue_rect = pygame.Rect(center - 4, center + 8, 8, 12)
        pygame.draw.ellipse(surface, tongue_color, tongue_rect)

        return surface

    def get_einstein_sprite(self, size: int) -> Surface | None:
        """Get Einstein sprite scaled to the given size."""
        if self._einstein_base is None:
            return None

        # Clamp size to reasonable bounds
        size = max(8, min(size, 512))

        # Check cache
        if size in self._einstein_cache:
            return self._einstein_cache[size]

        # Scale and cache
        scaled = pygame.transform.smoothscale(self._einstein_base, (size, size))
        self._einstein_cache[size] = scaled
        return scaled

    def clear_cache(self) -> None:
        """Clear the sprite cache."""
        self._einstein_cache.clear()
