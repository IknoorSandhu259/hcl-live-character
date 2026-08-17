"""Focused camera failure regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import engagement_demo  # noqa: E402
from attention import AttentionSensor, CameraError  # noqa: E402


class _FailingCapture:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def read(self):
        if self.error is not None:
            raise self.error
        return self.result


def _sensor_with_capture(capture):
    sensor = AttentionSensor.__new__(AttentionSensor)
    sensor._capture = capture
    return sensor


def test_runtime_failed_read_raises_camera_error_instead_of_no_attention():
    sensor = _sensor_with_capture(_FailingCapture(result=(False, None)))

    with pytest.raises(CameraError, match="no frame returned"):
        sensor.read()


def test_runtime_opencv_read_error_is_wrapped_as_camera_error():
    sensor = _sensor_with_capture(_FailingCapture(error=cv2.error("capture failed")))

    with pytest.raises(CameraError, match="Camera read failed"):
        sensor.read()


def test_valid_frame_without_face_remains_not_attending():
    sensor = _sensor_with_capture(_FailingCapture(result=(True, object())))
    sensor._detect_faces = lambda _frame: []

    reading = sensor.read()

    assert reading.attending is False
    assert reading.faces == ()


def test_camera_is_closed_if_pybullet_connect_raises(monkeypatch):
    class Sensor:
        closed = False

        def close(self):
            self.closed = True

    sensor = Sensor()

    def fail_connect(*_args, **_kwargs):
        raise RuntimeError("GUI initialization failed")

    monkeypatch.setattr(engagement_demo.p, "connect", fail_connect)

    with pytest.raises(RuntimeError, match="GUI initialization failed"):
        engagement_demo.run(sensor=sensor, preview=False, headless=False)

    assert sensor.closed
