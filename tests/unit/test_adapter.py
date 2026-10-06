"""Unit tests for the BrowserActionAdapter.

Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6
"""

from __future__ import annotations

from typing import Any

import pytest

from src.actions.adapter import BrowserActionAdapter


class _FakePyAutoGUI:
    """Captures calls without touching real input devices."""

    def __init__(self) -> None:
        self.scroll_calls: list[int] = []
        self.hotkey_calls: list[tuple[str, ...]] = []

    def scroll(self, amount: int) -> None:
        self.scroll_calls.append(amount)

    def hotkey(self, *keys: str) -> None:
        self.hotkey_calls.append(tuple(keys))


# ---------------------------------------------------------------- scroll


def test_scroll_up_calls_pyautogui_with_positive_amount() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)

    adapter.scroll_up(120)

    assert fake.scroll_calls == [120]


def test_scroll_down_calls_pyautogui_with_negative_amount() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)

    adapter.scroll_down(80)

    assert fake.scroll_calls == [-80]


def test_scroll_with_zero_or_negative_amount_is_noop() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)

    adapter.scroll_up(0)
    adapter.scroll_down(-10)

    assert fake.scroll_calls == []


# ---------------------------------------------------------------- platform shortcuts


def test_next_tab_uses_ctrl_tab_on_linux() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "other"

    adapter.next_tab()

    assert fake.hotkey_calls == [("ctrl", "tab")]


def test_previous_tab_uses_ctrl_shift_tab() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "other"

    adapter.previous_tab()

    assert fake.hotkey_calls == [("ctrl", "shift", "tab")]


def test_browser_back_uses_alt_left_on_non_macos() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "other"

    adapter.browser_back()

    assert fake.hotkey_calls == [("alt", "left")]


def test_browser_back_uses_command_left_on_macos() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "darwin"

    adapter.browser_back()

    assert fake.hotkey_calls == [("command", "left")]


def test_browser_forward_uses_alt_right_on_non_macos() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "other"

    adapter.browser_forward()

    assert fake.hotkey_calls == [("alt", "right")]


def test_browser_forward_uses_command_right_on_macos() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "darwin"

    adapter.browser_forward()

    assert fake.hotkey_calls == [("command", "right")]


def test_close_tab_uses_ctrl_w_on_non_macos() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "other"

    adapter.close_tab()

    assert fake.hotkey_calls == [("ctrl", "w")]


def test_close_tab_uses_command_w_on_macos() -> None:
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "darwin"

    adapter.close_tab()

    assert fake.hotkey_calls == [("command", "w")]


def test_mac_shortcuts_are_browser_tab_shortcuts_not_app_switcher() -> None:
    """Confirm we never use ``Command+Tab`` (OS app switcher)."""
    fake = _FakePyAutoGUI()
    adapter = BrowserActionAdapter(pyautogui_module=fake)
    adapter.os_name = "darwin"

    adapter.next_tab()
    adapter.previous_tab()

    for call in fake.hotkey_calls:
        assert ("command", "tab") not in (call,)
