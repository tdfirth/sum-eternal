"""
Sum Eternal - Entry Point

The game loop and initialization.
Run with: uv run sum-eternal
"""

import sys
import os

# Ensure solutions directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pygame

from sum_eternal.engine.game import Game
from sum_eternal.engine.hot_reload import HotReloader


def main() -> None:
    """Main entry point for Sum Eternal."""
    pygame.init()

    # Parse resolution from environment or use default
    resolution_str = os.environ.get("SUM_ETERNAL_RESOLUTION", "640x480")
    try:
        width, height = map(int, resolution_str.split("x"))
    except ValueError:
        width, height = 640, 480

    # Create display
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SUM ETERNAL")

    # Initialize game
    game = Game(screen)

    # Start hot reloader in background thread
    reloader = HotReloader(game)
    reloader.start()

    # Main game loop
    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    game.handle_event(event)

            # Update game state
            dt = clock.tick(60) / 1000.0  # Delta time in seconds
            game.update(dt)

            # Render
            game.render()
            pygame.display.flip()

    finally:
        reloader.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
