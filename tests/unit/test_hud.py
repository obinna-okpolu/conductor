"""Unit tests for the HUD.

The HUD uses Tkinter, which is not desirable in CI. Tests construct the
HUD with a stub ``tk_module`` so they can run headless.

Requirements: 15.2, 15.3, 15.4, 15.5, 15.6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.config import HUDConfig
from src.ui.hud import HUD


@dataclass
class _StubLabel:
    text: str = ""
    configs: list[str] = field(default_factory=list)

    def pack(self, **_kwargs: Any) -> None:
        return None

    def config(self, text: str | None = None, **_kwargs: Any) -> None:
        if text is not None:
            self.text = text
            self.configs.append(text)


@dataclass
class _StubRoot:
    title_text: str = ""
    geometry_text: str = ""
    overrideredirect_called: bool = False
    topmost_called: bool = False
    alpha: float | None = None
    destroyed: bool = False
    state_label: _StubLabel | None = None
    feedback_label: _StubLabel | None = None

    def title(self, text: str) -> None:
        self.title_text = text

    def overrideredirect(self, flag: bool) -> None:
        self.overrideredirect_called = flag

    def attributes(self, *args: Any, **kwargs: Any) -> None:
        if not args:
            return
        key = args[0]
        if key == "-topmost":
            self.topmost_called = bool(args[1]) if len(args) > 1 else False
        elif key == "-alpha":
            self.alpha = float(args[1]) if len(args) > 1 else None

    def geometry(self, geom: str) -> None:
        self.geometry_text = geom

    def update_idletasks(self) -> None:
        return None

    def update(self) -> None:
        return None

    def destroy(self) -> None:
        self.destroyed = True


@dataclass
class _StubTk:
    roots: list[_StubRoot] = field(default_factory=list)
    labels: list[_StubLabel] = field(default_factory=list)

    def Tk(self) -> _StubRoot:
        root = _StubRoot()
        self.roots.append(root)
        return root

    def Label(self, _root: Any, text: str = "", **_kw: Any) -> _StubLabel:
        label = _StubLabel(text=text)
        self.labels.append(label)
        return label


# ---------------------------------------------------------------- tests


def test_start_creates_root_and_sets_window_flags() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)

    hud.start()

    assert hud._started is True
    assert len(tk_stub.roots) == 1
    root = tk_stub.roots[0]
    assert root.overrideredirect_called is True
    assert root.topmost_called is True
    assert root.title_text == "Conductor"
    assert root.geometry_text.endswith("+10+10")


def test_start_is_idempotent() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)

    hud.start()
    hud.start()

    assert len(tk_stub.roots) == 1


def test_update_state_changes_label() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)
    hud.start()

    hud.update_state("clutch_engaged")

    assert hud._state_label.text == "Clutch Engaged"

    hud.update_state("clutch_active")
    assert hud._state_label.text == "Clutch Active"


def test_show_feedback_sets_message_and_expiry() -> None:
    tk_stub = _StubTk()
    clock = [0.0]
    hud = HUD(
        HUDConfig(feedback_duration=2.0), tk_module=tk_stub, clock=lambda: clock[0]
    )
    hud.start()

    hud.show_feedback("Next Tab")

    assert hud._feedback_message == "Next Tab"
    assert hud._feedback_expiry == 2.0
    assert hud._feedback_label.text == "Next Tab"


def test_feedback_expires_after_duration() -> None:
    tk_stub = _StubTk()
    clock = [0.0]
    hud = HUD(
        HUDConfig(feedback_duration=1.0), tk_module=tk_stub, clock=lambda: clock[0]
    )
    hud.start()

    hud.show_feedback("Back")
    assert hud._feedback_label.text == "Back"

    clock[0] = 1.5
    hud.process_events()
    assert hud._feedback_label.text == ""


def test_process_events_pumps_event_loop() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)
    hud.start()

    hud.process_events()

    root = tk_stub.roots[0]
    assert root.destroyed is False  # Update should not destroy.


def test_stop_destroys_root() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)
    hud.start()

    hud.stop()
    hud.stop()  # idempotent

    assert tk_stub.roots[0].destroyed is True
    assert hud._started is False


def test_process_events_is_safe_when_not_started() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)
    hud.process_events()  # Should not raise.


def test_hud_state_reflects_internal_state() -> None:
    tk_stub = _StubTk()
    hud = HUD(HUDConfig(), tk_module=tk_stub)
    hud.start()

    hud.update_state("clutch_engaged")
    hud.show_feedback("Forward")

    snapshot = hud.get_state()
    assert snapshot.interaction_state == "clutch_engaged"
    assert snapshot.feedback_message == "Forward"
    assert snapshot.feedback_expiry is not None
