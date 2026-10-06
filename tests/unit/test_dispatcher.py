"""Unit tests for the ActionDispatcher.

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 19.5
"""

from __future__ import annotations

from typing import Any

from src.actions.adapter import BrowserActionAdapter
from src.actions.dispatcher import ActionDispatcher
from src.models import BrowserIntent, IntentType


class _RecordingAdapter(BrowserActionAdapter):
    """Records every method call without performing real actions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        # Skip super().__init__ to avoid needing a pyautogui module.

    def scroll_up(self, amount: float) -> None:
        self.calls.append(("scroll_up", (amount,)))

    def scroll_down(self, amount: float) -> None:
        self.calls.append(("scroll_down", (amount,)))

    def next_tab(self) -> None:
        self.calls.append(("next_tab", ()))

    def previous_tab(self) -> None:
        self.calls.append(("previous_tab", ()))

    def browser_back(self) -> None:
        self.calls.append(("browser_back", ()))

    def browser_forward(self) -> None:
        self.calls.append(("browser_forward", ()))

    def close_tab(self) -> None:
        self.calls.append(("close_tab", ()))


def _intent(intent_type: IntentType, magnitude: float | None = None) -> BrowserIntent:
    return BrowserIntent(
        intent_type=intent_type, magnitude=magnitude, frame_id=1, timestamp=0.0
    )


def test_dispatch_scroll_up() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.SCROLL_UP, magnitude=120))

    assert adapter.calls == [("scroll_up", (120,))]


def test_dispatch_scroll_down() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.SCROLL_DOWN, magnitude=80))

    assert adapter.calls == [("scroll_down", (80,))]


def test_dispatch_next_tab() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.NEXT_TAB))

    assert adapter.calls == [("next_tab", ())]


def test_dispatch_previous_tab() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.PREVIOUS_TAB))

    assert adapter.calls == [("previous_tab", ())]


def test_dispatch_browser_back() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.BROWSER_BACK))

    assert adapter.calls == [("browser_back", ())]


def test_dispatch_browser_forward() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.BROWSER_FORWARD))

    assert adapter.calls == [("browser_forward", ())]


def test_dispatch_close_tab() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.CLOSE_TAB))

    assert adapter.calls == [("close_tab", ())]


def test_dispatch_with_zero_magnitude_falls_back_to_zero() -> None:
    """If magnitude is None the dispatcher coerces to 0 for scroll intents."""
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.SCROLL_UP, magnitude=None))

    assert adapter.calls == [("scroll_up", (0.0,))]


def test_dispatcher_only_calls_adapter_once_per_intent() -> None:
    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.NEXT_TAB))

    assert len(adapter.calls) == 1


def test_unknown_intent_is_silently_ignored() -> None:
    """Adding new intent types must not crash older builds."""
    from src.models import IntentType as _IT

    # Construct a sentinel value outside the enum so the match falls through.
    class _Sentinel:
        pass

    adapter = _RecordingAdapter()
    dispatcher = ActionDispatcher(adapter)

    dispatcher.dispatch(_intent(IntentType.SCROLL_UP))  # baseline
    bogus = BrowserIntent(
        intent_type=_Sentinel(), magnitude=None, frame_id=2, timestamp=0.0
    )  # type: ignore[arg-type]
    dispatcher.dispatch(bogus)

    # Only the first call should have landed.
    assert adapter.calls == [("scroll_up", (0.0,))]
