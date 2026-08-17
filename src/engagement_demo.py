"""Camera-driven engagement demo for the dummy five-DOF character lamp.

Run:  python src/engagement_demo.py
      python src/engagement_demo.py --no-preview      (no camera window)
      python src/engagement_demo.py --camera-index 1  (pick another webcam)

The character starts neutral. Look at the laptop camera and it sits up
(``LampController.engage``); look away or leave for a few seconds and it
settles back down (``LampController.neutral``).

This file is the *wiring* only, and is deliberately thin:

    attention.AttentionSensor      camera -> "is a face facing us?"  (perception)
    attention.EngagementTracker    noisy booleans -> IDLE / ENGAGED  (policy)
    robot_controller.LampController   IDLE / ENGAGED -> joint motion (body)

Perception never issues a joint command, and the body layer never sees a
pixel. Every joint command still goes through LampController, which owns the
joint limits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import (  # noqa: E402
    DETECT_HZ,
    AttentionSensor,
    CameraError,
    EngagementState,
    EngagementTracker,
)
from robot_controller import LampController, load_lamp  # noqa: E402

#: Simulation time advanced per perception tick while no behaviour is playing.
#: The lamp is holding a pose here, so this only has to keep the position
#: controller ticking over; the frame grab dominates the loop period anyway.
IDLE_SIM_STEP = 1.0 / 60.0

PREVIEW_WINDOW = "lamp attention"


def _draw_preview(reading, state: EngagementState) -> None:
    """Show what the perception layer actually sees. Debug aid, not a UI."""
    frame = reading.frame
    if frame is None:
        return
    colour = (0, 200, 0) if state is EngagementState.ENGAGED else (0, 165, 255)
    for (x, y, w, h) in reading.faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(
        frame,
        f"{state.value}   attending={reading.attending}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
    )
    cv2.imshow(PREVIEW_WINDOW, frame)


def run(
    camera_index: int = 0,
    preview: bool = True,
    headless: bool = False,
    sensor=None,
) -> None:
    """Drive the lamp from the camera until interrupted.

    *sensor* exists so the loop can be exercised headlessly against a scripted
    fake reader; leave it None to use the real webcam.
    """
    # Open the camera before spending time on the simulator, so a permission
    # or device problem surfaces immediately instead of behind a GUI window.
    if sensor is None:
        sensor = AttentionSensor(camera_index=camera_index).open()

    client = None
    try:
        client = p.connect(p.DIRECT if headless else p.GUI)
        if client < 0:
            raise RuntimeError("Could not connect to PyBullet")

        p.setGravity(0, 0, -9.81, physicsClientId=client)
        if not headless:
            # Same face-on viewpoint as demo.py.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25,
                cameraYaw=272,
                cameraPitch=-10,
                cameraTargetPosition=[-0.10, 0.0, 0.50],
                physicsClientId=client,
            )

        lamp = LampController(load_lamp(client), client)
        tracker = EngagementTracker()

        lamp.neutral()
        print(f"[{time.strftime('%H:%M:%S')}] ready in {tracker.state.value}; "
              "look at the camera to engage. Ctrl-C (or 'q' in the preview) to quit.")

        period = 1.0 / DETECT_HZ
        while True:
            tick = time.monotonic()
            try:
                reading = sensor.read()
            except StopIteration:
                break  # a scripted fake sensor ran out of frames
            transition = tracker.update(reading.attending, tick)

            if preview:
                _draw_preview(reading, tracker.state)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if transition is EngagementState.ENGAGED:
                print(f"[{time.strftime('%H:%M:%S')}] IDLE -> ENGAGED")
                lamp.engage()
                sensor.flush()
            elif transition is EngagementState.IDLE:
                print(f"[{time.strftime('%H:%M:%S')}] ENGAGED -> IDLE")
                lamp.neutral()
                sensor.flush()
            else:
                # No transition: hold the pose and keep the physics ticking.
                lamp.step(IDLE_SIM_STEP)

            # Pace the perception loop to DETECT_HZ so it does not spin a core
            # for information the camera cannot supply any faster. A behaviour
            # that just played already overran this budget, so the sleep
            # collapses to zero.
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        sensor.close()
        if preview:
            cv2.destroyAllWindows()
        if client is not None and p.isConnected(client):
            p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--no-preview", action="store_true", help="hide the camera preview window")
    parser.add_argument("--headless", action="store_true", help="run PyBullet without a window")
    args = parser.parse_args()
    run(camera_index=args.camera_index, preview=not args.no_preview, headless=args.headless)


if __name__ == "__main__":
    try:
        main()
    except CameraError as exc:
        print(f"camera error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
