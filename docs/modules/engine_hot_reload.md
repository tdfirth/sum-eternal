# Module: engine/hot_reload.py

## Purpose

File watching and module reloading. Detects changes to solution files, runs tests, and signals the game to refresh.

## Location

`src/sum_eternal/engine/hot_reload.py`

## Public Interface

```python
class HotReloader:
    """Manages file watching for hot reload."""

    def __init__(self, game: Game) -> None:
        """
        Initialize with reference to game instance.

        Locates project root and solutions directory.
        """

    def start(self) -> None:
        """
        Start file watcher in background thread.

        Uses watchdog to monitor solutions/ directory.
        """

    def stop(self) -> None:
        """
        Stop file watcher gracefully.

        Joins observer thread with timeout.
        """
```

### Internal Classes

```python
class SolutionFileHandler(FileSystemEventHandler):
    """Handles file system events for solution files."""

    def on_modified(self, event: FileModifiedEvent) -> None:
        """
        Handle file modification.

        1. Debounce rapid changes
        2. Run pytest for changed chapter
        3. If pass: reload modules, signal game
        4. If fail: show error message
        """
```

### Module Functions

```python
def get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""

def reload_solution_modules() -> None:
    """
    Clear cached solution modules and bridge.

    Forces fresh import on next access.
    """
```

## Implementation Approach

### File Watcher Setup

```python
class HotReloader:
    def start(self):
        handler = SolutionFileHandler(self.game, self.project_root)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.solutions_dir), recursive=False)
        self.observer.start()
```

### Change Detection

```python
class SolutionFileHandler:
    def on_modified(self, event):
        path = Path(event.src_path)

        # Filter to Python files only
        if path.suffix != ".py":
            return

        # Debounce (ignore changes within 0.5s)
        if self._recently_modified(path):
            return

        # Map solution file to test file
        test_file = SOLUTION_TO_TEST.get(path.name)
        if not test_file:
            return

        # Run tests
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-x", "-q"],
            cwd=self.project_root,
        )

        if result.returncode == 0:
            self._trigger_reload()
        else:
            self.game.show_error(f"Tests failed in {path.name}")
```

### Module Reloading

```python
def reload_solution_modules():
    # Remove all cached solution modules
    to_remove = [k for k in sys.modules if k.startswith("solutions.")]
    for key in to_remove:
        del sys.modules[key]

    # Also clear bridge (it imports solutions)
    if "sum_eternal.bridge" in sys.modules:
        del sys.modules["sum_eternal.bridge"]

def _trigger_reload(self):
    reload_solution_modules()
    self.game.refresh_progress()
    self.game.clear_error()
```

## Solution-to-Test Mapping

```python
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
```

## Thread Safety

- File watcher runs in background thread
- Game state updates only happen in main thread via game.refresh_progress()
- No shared mutable state between threads
- Module reloading is synchronized (happens before game update)

## Dependencies

- `watchdog`: File system events
- `subprocess`: Running pytest
- `importlib`: Module reloading
- Game instance (for refresh_progress(), show_error())

## Error Handling

| Error | Handling |
|-------|----------|
| Solutions dir missing | Log warning, don't start watcher |
| Test failure | Show error message, don't reload |
| Import error | Caught in bridge, progress stays unchanged |
| Observer crash | Log error, game continues without hot reload |

## Testing Strategy

```python
def test_debouncing():
    """Verify rapid saves don't trigger multiple reloads."""

def test_module_reloading():
    """Verify modules are actually reloaded after clear."""

def test_test_file_mapping():
    """Verify all solution files have test file mappings."""
```
