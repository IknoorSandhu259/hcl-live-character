"""Perception layer: "is someone paying attention to the character?"

This module owns the laptop camera and nothing else. It never touches
PyBullet, the URDF, or LampController -- it only answers a boolean question
per frame and smooths that answer over time into a two-state engagement
signal. The body layer (robot_controller.LampController) decides what to do
about it. Keeping the split here means the attention proxy can be swapped for
a real gaze estimator later without touching motion code.

Attention proxy
---------------
We use *approximately forward-facing face presence*, measured with OpenCV's
bundled Haar frontal-face cascade:

* The cascade is trained on frontal faces, so it degrades sharply once the
  head yaws away from the lens (roughly beyond +/-30 deg). "Detected by the
  frontal cascade" is therefore already a coarse "facing the camera" test --
  we get a cheap head-pose gate for free without running a gaze model.
* The camera sits where the character's eye is, so facing the camera *is*
  facing the character.
* A minimum face size gate (:data:`MIN_FACE_FRACTION`) rejects faces that are
  far away, so someone crossing the back of the room does not trigger
  engagement.

This is deliberately not gaze estimation. It cannot tell "looking at the lamp"
from "looking just past the lamp", and that is an accepted limitation for this
milestone -- see the known limitations in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple
import sys
import time

import cv2

# --------------------------------------------------------------------------
# Tunable constants -- every threshold in the engagement pipeline lives here.
# --------------------------------------------------------------------------

#: Capture resolution requested from the webcam. 640x480 is enough for face
#: detection at conversational distance and keeps one CPU core well clear of
#: saturation on the 4-core Ubuntu target.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

#: Target perception rate. The detector is much faster than this; we pace the
#: loop so it does not spin the CPU for no extra information.
DETECT_HZ = 15.0

# -- detector tuning --------------------------------------------------------

#: Haar pyramid step. Larger = faster, coarser.
SCALE_FACTOR = 1.2

#: How many overlapping detections are required to accept a face. Higher
#: values reject more false positives at the cost of a few missed frames --
#: which the temporal filter below is there to absorb.
MIN_NEIGHBORS = 6

#: Minimum face height as a fraction of frame height. At 640x480 and a typical
#: laptop webcam FOV, 0.18 corresponds to roughly "within ~1.5 m of the
#: laptop", i.e. plausibly here to interact rather than walking past.
MIN_FACE_FRACTION = 0.18

# -- temporal filtering / hysteresis ----------------------------------------
#
# Three separate constants, because engaging and disengaging should not be
# symmetric: engaging should require conviction, disengaging should be
# forgiving of dropped frames but still deliberate.

#: Attention must be held this long, continuously, before IDLE -> ENGAGED.
#: Short enough to feel responsive, long enough that a head turning *through*
#: the camera does not trigger it.
ENGAGE_HOLD_SECONDS = 0.8

#: Gaps up to this long do not break the engage streak. One or two dropped
#: detections at 15 Hz (blink, motion blur, hand across the face) stay inside
#: this window, so the streak survives them.
MISS_TOLERANCE_SECONDS = 0.4

#: Attention must be absent this long, continuously, before ENGAGED -> IDLE.
#: This is the anti-twitch guard: brief detection loss while engaged is
#: ignored, but genuinely looking away or leaving is honoured.
DISENGAGE_HOLD_SECONDS = 2.5


class EngagementState(Enum):
    """The only two states this milestone has."""

    IDLE = "IDLE"
    ENGAGED = "ENGAGED"


#: A detected face as (x, y, width, height) in frame pixels.
FaceBox = Tuple[int, int, int, int]


# --------------------------------------------------------------------------
# Camera + detector
# --------------------------------------------------------------------------


def _candidate_backends() -> List[int]:
    """Capture backends to try, in order.

    On Windows the default MSMF backend can take several seconds to open a
    webcam, so DirectShow is tried first -- but not all Windows machines can
    capture by index through DSHOW, hence the CAP_ANY fallback. Everywhere
    else (macOS dev, the Ubuntu 24.04 target with V4L2) OpenCV's automatic
    choice is already correct.
    """
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class AttentionSensor:
    """Wraps the webcam and answers "is a person facing us right now?".

    Use as a context manager so the camera is always released::

        with AttentionSensor() as sensor:
            reading = sensor.read()
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None
        if not hasattr(cv2, "CascadeClassifier"):
            raise CameraError(
                f"This OpenCV build ({cv2.__version__}) has no CascadeClassifier. "
                "OpenCV 5 removed the Haar cascade API; install the pinned 4.x "
                "line from requirements.txt (pip install 'opencv-python>=4.8,<5')."
            )
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise CameraError(
                f"Could not load the Haar frontal-face cascade from {cascade_path}. "
                "Install the 'opencv-python' wheel from requirements.txt; a custom "
                "OpenCV build may omit the bundled cascade data."
            )

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "AttentionSensor":
        for backend in _candidate_backends():
            capture = cv2.VideoCapture(self.camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            # Keep the driver queue shallow so read() returns a *current*
            # frame rather than a backlog one after a blocking robot motion.
            # Not every backend honours this; it is advisory.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # isOpened() alone is not proof of a working device -- some
            # backends only fail on the first grab.
            ok, _ = capture.read()
            if not ok:
                capture.release()
                continue

            self._capture = capture
            return self

        raise CameraError(
            f"Could not open camera index {self.camera_index}. Check that a webcam "
            "is connected, that no other application is holding it, and that this "
            "terminal has camera permission (macOS: System Settings > Privacy & "
            "Security > Camera; Linux: the user must be in the 'video' group for "
            "/dev/video*; sandboxed shells often block camera access entirely). "
            "Try a different device with --camera-index."
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "AttentionSensor":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- perception --------------------------------------------------------

    def read(self) -> "AttentionReading":
        """Grab one frame and report whether someone is facing the camera."""
        if self._capture is None:
            raise CameraError("AttentionSensor.read() called before open()")

        try:
            ok, frame = self._capture.read()
        except cv2.error as exc:
            raise CameraError(f"Camera read failed: {exc}") from exc
        if not ok or frame is None:
            raise CameraError("Camera read failed: no frame returned")

        faces = self._detect_faces(frame)
        return AttentionReading(attending=bool(faces), faces=tuple(faces), frame=frame)

    def flush(self) -> None:
        """Drop any frames the driver queued while we were busy elsewhere.

        A robot behaviour blocks for ~1.5 s. On backends that ignore
        CAP_PROP_BUFFERSIZE the next ``read()`` would otherwise return a frame
        from *before* the motion, so the state machine would be reasoning
        about the past.
        """
        if self._capture is not None:
            for _ in range(2):
                self._capture.grab()

    def _detect_faces(self, frame) -> List[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Flattens the histogram so the cascade behaves under uneven room
        # lighting (e.g. a bright window behind the user).
        gray = cv2.equalizeHist(gray)
        min_side = int(MIN_FACE_FRACTION * frame.shape[0])
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=(min_side, min_side),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


@dataclass(frozen=True)
class AttentionReading:
    """One frame's worth of perception output."""

    #: True if at least one near, approximately forward-facing face was found.
    attending: bool
    #: Accepted face boxes, for the preview overlay.
    faces: Sequence[FaceBox]
    #: The BGR frame, or None if the grab failed. Preview only.
    frame: object


# --------------------------------------------------------------------------
# Temporal filtering
# --------------------------------------------------------------------------


class EngagementTracker:
    """Turns noisy per-frame attention into a stable IDLE / ENGAGED state.

    Pure logic: no camera, no clock of its own (the caller passes the
    timestamp), no robot. That makes the hysteresis directly unit-testable.

    Hysteresis, in words:

    * A run of attending frames builds a *streak*. Gaps shorter than
      :data:`MISS_TOLERANCE_SECONDS` do not break it.
    * IDLE -> ENGAGED once that streak reaches :data:`ENGAGE_HOLD_SECONDS`.
    * ENGAGED -> IDLE only after :data:`DISENGAGE_HOLD_SECONDS` with no
      attending frame at all.

    Because the engage and disengage thresholds are different and separated by
    the miss tolerance, the state cannot chatter on a single bad frame.
    """

    def __init__(
        self,
        engage_hold: float = ENGAGE_HOLD_SECONDS,
        disengage_hold: float = DISENGAGE_HOLD_SECONDS,
        miss_tolerance: float = MISS_TOLERANCE_SECONDS,
    ):
        self.engage_hold = engage_hold
        self.disengage_hold = disengage_hold
        self.miss_tolerance = miss_tolerance

        self.state = EngagementState.IDLE
        #: Start of the current uninterrupted attention streak, or None.
        self._streak_start: Optional[float] = None
        #: Timestamp of the most recent attending frame, or None.
        self._last_attending: Optional[float] = None

    def update(self, attending: bool, now: Optional[float] = None) -> Optional[EngagementState]:
        """Feed one frame's verdict.

        Returns the new state if this frame caused a transition, else None.
        """
        if now is None:
            now = time.monotonic()

        if attending:
            if self._streak_start is None:
                self._streak_start = now
            self._last_attending = now
        elif self._last_attending is None or now - self._last_attending > self.miss_tolerance:
            # The gap is too long to call it a dropped frame: the streak ends.
            self._streak_start = None

        if self.state is EngagementState.IDLE:
            # Only an *attending* frame may engage. Without this guard a short
            # glance followed by a tolerated gap would keep ageing the streak
            # and engage after the person had already looked away.
            if (
                attending
                and self._streak_start is not None
                and now - self._streak_start >= self.engage_hold
            ):
                return self._transition_to(EngagementState.ENGAGED)
        else:
            absent_for = float("inf") if self._last_attending is None else now - self._last_attending
            if absent_for >= self.disengage_hold:
                return self._transition_to(EngagementState.IDLE)
        return None

    def _transition_to(self, state: EngagementState) -> EngagementState:
        self.state = state
        if state is EngagementState.ENGAGED:
            # Entering ENGAGED consumes the streak; the next engage needs a
            # fresh one. Prevents an immediate re-trigger after disengaging.
            self._streak_start = None
        else:
            self._last_attending = None
        return state
