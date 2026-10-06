---
inclusion: always
---

# Technology Stack

## Language & Runtime
- Python 3.12 (required range `>=3.12,<3.13`, enforced in `pyproject.toml`)
- Project name in `pyproject.toml`: `maestro`

## Core Dependencies
| Library | Purpose |
|---|---|
| OpenCV (`cv2`) | Webcam frame capture |
| MediaPipe | Hand landmark detection (runs fully locally) |
| Tkinter | Floating HUD overlay (standard library) |
| PyAutoGUI | Browser action adapter (scrolling, tab switching, etc.) |
| pytest | Unit and integration tests |
| Hypothesis | Property-based tests |

## Dependency Management
- Use `uv` to add, remove, and sync dependencies — do **not** use `pip` directly.
- Example: `uv add mediapipe`, `uv run pytest`
- Keep `pyproject.toml` as the single source of truth for dependencies.

## Architecture Rules
- Maintain strict separation between pipeline stages: **landmark detection → gesture classification → intent mapping → browser action execution**
- Gesture detection and browser control modules must remain decoupled from each other.
- Store configuration (gesture mappings, sensitivity thresholds) in `config/`; the active file format is TOML (`config/config.toml`).
- Place all core application code under `src/`.
- Place all tests under `tests/`.

## Code Style
- Type hints are required on all function signatures.
- Docstrings are required for all modules, classes, and public functions.
- Follow PEP 8 conventions.
- Prefer small, focused functions over large monolithic blocks.

## Testing
- Write unit tests with `pytest`; use `Hypothesis` for property-based tests on gesture classifiers and intent mappers.
- Design modules for testability — mock webcam frames and MediaPipe landmark outputs rather than requiring real hardware in tests.
- Run tests via: `uv run pytest` (or the project virtual environment's Python executable when `uv` is unavailable)

## General Constraints
- Prefer standard-library solutions where practical.
- Do **not** introduce additional frameworks or libraries without a clear architectural justification.
- No cloud APIs — all gesture recognition must run locally.
