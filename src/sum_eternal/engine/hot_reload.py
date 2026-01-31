"""
Hot Reload - File watching and module reloading.

Watches the solutions/ directory for changes.
When a file changes:
1. Run pytest for that chapter
2. If tests pass, signal the game to reload
3. If tests fail, do nothing (console shows errors)
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

if TYPE_CHECKING:
    from sum_eternal.engine.game import Game


# Map solution files to test files
SOLUTION_TO_TEST = {
    "c01_first_blood.py": "test_c01_first_blood.py",
    "c02_knee_deep_in_the_indices.py": "test_c02_knee_deep_in_the_indices.py",
    "c03_the_slaughter_batch.py": "test_c03_the_slaughter_batch.py",
    "c04_rip_and_trace.py": "test_c04_rip_and_trace.py",
    "c05_total_intersection.py": "test_c05_total_intersection.py",
    "c06_infernal_projection.py": "test_c06_infernal_projection.py",
    "c07_spooky_action_at_a_distance.py": "test_c07_spooky_action_at_a_distance.py",
    "c08_the_icon_of_ein.py": "test_c08_the_icon_of_ein.py",
    "c09_nightmare_mode.py": "test_c09_nightmare_mode.py",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def reload_solution_modules() -> None:
    """Reload all solution modules to pick up changes."""
    # Remove cached solution modules
    to_remove = [k for k in sys.modules if k.startswith("solutions.")]
    for key in to_remove:
        del sys.modules[key]

    # Also reload the bridge
    if "sum_eternal.bridge" in sys.modules:
        del sys.modules["sum_eternal.bridge"]


class SolutionFileHandler(FileSystemEventHandler):
    """Handles file system events for solution files."""

    def __init__(self, game: Game, project_root: Path) -> None:
        self.game = game
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self._last_modified: dict[str, float] = {}
        self._debounce_seconds = 0.5

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Only handle Python files in solutions/
        if path.suffix != ".py":
            return

        if path.name.startswith("__"):
            return

        # Debounce: ignore rapid successive changes
        now = time.time()
        if path.name in self._last_modified:
            if now - self._last_modified[path.name] < self._debounce_seconds:
                return
        self._last_modified[path.name] = now

        # Find corresponding test file
        test_file = SOLUTION_TO_TEST.get(path.name)
        if not test_file:
            return

        test_path = self.tests_dir / test_file
        if not test_path.exists():
            print(f"[Hot Reload] No test file found: {test_file}")
            return

        print(f"\n[Hot Reload] Detected change in {path.name}")
        print(f"[Hot Reload] Running tests: {test_file}")

        # Run pytest for this chapter
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-x", "-q", "--tb=short"],
            cwd=self.project_root,
            capture_output=False,  # Let output go to console
        )

        if result.returncode == 0:
            print(f"[Hot Reload] Tests passed! Reloading...")
            self._trigger_reload()
        else:
            print(f"[Hot Reload] Tests failed. Fix the errors and save again.")
            self.game.show_error(f"Tests failed in {path.name}")

    def _trigger_reload(self) -> None:
        """Trigger a game reload."""
        reload_solution_modules()
        self.game.refresh_progress()
        self.game.clear_error()
        print(f"[Hot Reload] Progress: {self.game.data.progress.name}")


class HotReloader:
    """Manages file watching for hot reload."""

    def __init__(self, game: Game) -> None:
        self.game = game
        self.project_root = get_project_root()
        self.solutions_dir = self.project_root / "solutions"
        self.observer: Observer | None = None
        self._running = False

    def start(self) -> None:
        """Start the file watcher in a background thread."""
        if self._running:
            return

        if not self.solutions_dir.exists():
            print(f"[Hot Reload] Solutions directory not found: {self.solutions_dir}")
            return

        self._running = True

        handler = SolutionFileHandler(self.game, self.project_root)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.solutions_dir), recursive=False)
        self.observer.start()

        print(f"[Hot Reload] Watching {self.solutions_dir}")

    def stop(self) -> None:
        """Stop the file watcher."""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=1.0)
            self.observer = None
        self._running = False
        print("[Hot Reload] Stopped")
