"""End-to-end wiring test: scripted attention -> real LampController motion.

Drives ``engagement_demo.run`` headlessly with a fake sensor instead of the
webcam, and checks the lamp actually ends up in the engage / neutral poses.
This covers the part the camera-free unit tests cannot: that the state machine
is wired to the body layer in the right direction.

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import engagement_demo  # noqa: E402
from attention import AttentionReading, DETECT_HZ  # noqa: E402
from robot_controller import POSES  # noqa: E402


class ScriptedSensor:
    """Replays a fixed attention pattern, then stops the loop."""

    def __init__(self, script):
        #: list of (attending, seconds) played at DETECT_HZ
        self._frames = [
            attending
            for attending, seconds in script
            for _ in range(int(round(seconds * DETECT_HZ)))
        ]
        self._index = 0

    def read(self) -> AttentionReading:
        if self._index >= len(self._frames):
            raise StopIteration
        attending = self._frames[self._index]
        self._index += 1
        return AttentionReading(attending=attending, faces=(), frame=None)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _run(script, capsys):
    engagement_demo.run(preview=False, headless=True, sensor=ScriptedSensor(script))
    return capsys.readouterr().out


@pytest.mark.parametrize(
    "script, expected_transitions",
    [
        # Never attending -> the lamp is never disturbed from neutral.
        ([(False, 4.0)], []),
        # Sustained attention engages exactly once, and stays engaged.
        ([(False, 1.0), (True, 4.0)], ["IDLE -> ENGAGED"]),
        # Brief loss while engaged is absorbed; the long absence disengages.
        (
            [(True, 2.0), (False, 1.0), (True, 1.0), (False, 4.0)],
            ["IDLE -> ENGAGED", "ENGAGED -> IDLE"],
        ),
        # The cycle repeats.
        (
            [(True, 2.0), (False, 4.0), (True, 2.0), (False, 4.0)],
            ["IDLE -> ENGAGED", "ENGAGED -> IDLE", "IDLE -> ENGAGED", "ENGAGED -> IDLE"],
        ),
    ],
    ids=["never-attends", "engages-once", "absorbs-brief-loss", "repeats"],
)
def test_transitions(script, expected_transitions, capsys):
    out = _run(script, capsys)
    logged = [line.split("] ")[-1] for line in out.splitlines() if " -> " in line]
    assert logged == expected_transitions


def test_lamp_reaches_the_engage_pose(capsys, monkeypatch):
    captured = {}
    original = engagement_demo.LampController

    class Spy(original):  # type: ignore[misc, valid-type]
        def engage(self, *a, **k):
            super().engage(*a, **k)
            captured["engage"] = dict(self._targets)

        def neutral(self, *a, **k):
            super().neutral(*a, **k)
            captured["neutral"] = dict(self._targets)

    monkeypatch.setattr(engagement_demo, "LampController", Spy)
    _run([(True, 2.0), (False, 4.0)], capsys)

    for pose_name in ("engage", "neutral"):
        for joint, angle in POSES[pose_name].items():
            assert captured[pose_name][joint] == pytest.approx(angle, abs=1e-6)
