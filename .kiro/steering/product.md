---
inclusion: always
---

# Conductor

## Overview

Conductor is a local, privacy-first gesture interface for browser navigation. Users interact with their browser using hand gestures captured through a standard webcam. All gesture recognition runs locally—no cloud APIs required.

**Core Pipeline:** Webcam → MediaPipe landmarks → gesture interpreter → intent → browser action

## Product Principles

- **Privacy-first**: All gesture recognition runs locally. No webcam frames or gesture inference should require a cloud API.
- **Low latency**: The system must respond quickly to maintain a fluid user experience.
- **Predictable behavior**: Gestures should trigger consistent, expected actions.
- **False-positive prevention**: The system must avoid accidental triggers from unintended movements.
- **Clear visual feedback**: Users should always understand what gesture was detected and what action will be taken.

## Supported Actions

- Scrolling (up/down)
- Tab switching (next/previous)
- Browser navigation (back/forward)
- Tab closing

## Current Gesture Contract

- Victory held: enter navigation mode
- Open palm held while navigation mode is active: release navigation mode
- Open palm held while idle: close the current tab
- Thumb up: browser back
- Thumb down: browser forward
- Horizontal swipe: browser back/forward fallback

The live recognizer does not emit closed-fist candidates. Gesture recognition assumes the palm is broadly facing the webcam, and ambiguous closed-hand poses remain inactive rather than triggering an additional action.

## Technical Stack

- **Language**: Python 3.12
- **Gesture Detection**: MediaPipe for hand landmark detection
- **Project Name**: `maestro` (in pyproject.toml)

## Architecture Guidelines

- Keep gesture detection and browser control modules decoupled
- Design for testability—mock webcam/MediaPipe inputs for unit tests
- Prefer clear separation between: landmark detection, gesture classification, intent mapping, and browser action execution
- Store configuration (gesture mappings, sensitivity settings) in the `config/` directory, using the active TOML file `config/config.toml`
- Place core application code in `src/`
- Write tests in `tests/`

## Code Style

- Use type hints for all function signatures
- Write docstrings for modules, classes, and public functions
- Follow PEP 8 conventions
- Prefer small, focused functions over large monolithic blocks
