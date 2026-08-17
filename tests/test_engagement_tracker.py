"""Hysteresis tests for EngagementTracker.

The tracker takes its timestamp as an argument, so the whole state machine is
testable without a camera, a clock, or a simulator. Run:

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attention import EngagementState, EngagementTracker  # noqa: E402

FRAME_PERIOD = 1.0 / 15.0  # matches attention.DETECT_HZ


def feed(
    tracker: EngagementTracker,
    attending: bool,
    seconds: float,
    clock: List[float],
) -> List[Optional[EngagementState]]:
    """Play *seconds* of frames all reporting *attending*; return transitions."""
    transitions = []
    for _ in range(int(round(seconds / FRAME_PERIOD))):
        clock[0] += FRAME_PERIOD
        transitions.append(tracker.update(attending, clock[0]))
    return [t for t in transitions if t is not None]


def test_starts_idle_and_stays_idle_without_a_face():
    tracker, clock = EngagementTracker(), [0.0]
    assert tracker.state is EngagementState.IDLE
    assert feed(tracker, False, 5.0, clock) == []
    assert tracker.state is EngagementState.IDLE


def test_engages_once_after_sustained_attention():
    tracker, clock = EngagementTracker(), [0.0]
    assert feed(tracker, True, 3.0, clock) == [EngagementState.ENGAGED]
    assert tracker.state is EngagementState.ENGAGED


def test_a_glance_shorter_than_the_hold_does_not_engage():
    tracker, clock = EngagementTracker(), [0.0]
    assert feed(tracker, True, 0.5, clock) == []  # < ENGAGE_HOLD_SECONDS
    assert tracker.state is EngagementState.IDLE


def test_streak_resets_after_a_long_gap_so_glances_do_not_accumulate():
    tracker, clock = EngagementTracker(), [0.0]
    for _ in range(4):
        feed(tracker, True, 0.5, clock)
        feed(tracker, False, 1.0, clock)  # > MISS_TOLERANCE_SECONDS
    assert tracker.state is EngagementState.IDLE


def test_single_dropped_frame_does_not_break_the_engage_streak():
    tracker, clock = EngagementTracker(), [0.0]
    feed(tracker, True, 0.4, clock)
    feed(tracker, False, FRAME_PERIOD, clock)  # one missed detection
    assert feed(tracker, True, 0.6, clock) == [EngagementState.ENGAGED]


def test_brief_detection_loss_while_engaged_does_not_disengage():
    tracker, clock = EngagementTracker(), [0.0]
    feed(tracker, True, 3.0, clock)
    assert feed(tracker, False, 1.5, clock) == []  # < DISENGAGE_HOLD_SECONDS
    assert tracker.state is EngagementState.ENGAGED


def test_sustained_absence_disengages_once():
    tracker, clock = EngagementTracker(), [0.0]
    feed(tracker, True, 3.0, clock)
    assert feed(tracker, False, 5.0, clock) == [EngagementState.IDLE]
    assert tracker.state is EngagementState.IDLE


def test_full_cycle_repeats():
    tracker, clock = EngagementTracker(), [0.0]
    for _ in range(3):
        assert feed(tracker, True, 3.0, clock) == [EngagementState.ENGAGED]
        assert feed(tracker, False, 5.0, clock) == [EngagementState.IDLE]


def test_disengaging_does_not_immediately_re_engage_on_a_stale_streak():
    tracker, clock = EngagementTracker(), [0.0]
    feed(tracker, True, 3.0, clock)
    feed(tracker, False, 5.0, clock)
    # A single attending frame right after disengaging must not re-trigger.
    clock[0] += FRAME_PERIOD
    assert tracker.update(True, clock[0]) is None
