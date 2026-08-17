"""Motion demo for the dummy five-DOF character lamp.

Run:  python src/demo.py           (PyBullet GUI)
      python src/demo.py --headless (no window, for validation)

Plays: neutral -> engage -> look left -> look right -> nod -> neutral.
Every joint command goes through LampController, which owns the joint limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_controller import LampController, load_lamp  # noqa: E402


def run(headless: bool = False) -> None:
    client = p.connect(p.DIRECT if headless else p.GUI)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet")

    try:
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        if not headless:
            # Stand the viewer in front of the lamp. The head faces world -X,
            # so yaw 272 deg is very nearly face on, which keeps the left and
            # right glances symmetric on screen. Note that, as with a person
            # facing you, the robot's own left appears on the viewer's right.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25,
                cameraYaw=272,
                cameraPitch=-10,
                cameraTargetPosition=[-0.10, 0.0, 0.50],
                physicsClientId=client,
            )

        robot = load_lamp(client)
        lamp = LampController(robot, client)
        print(lamp.describe())

        script = [
            ("neutral", lambda: lamp.neutral()),
            ("engage", lambda: lamp.engage()),
            ("look left", lambda: lamp.look_left()),
            ("look right", lambda: lamp.look_right()),
            # Face front again before nodding, so the nod is not read as part
            # of the glance to the right.
            ("face front, then nod", lambda: (lamp.engage(), lamp.nod())),
            ("neutral", lambda: lamp.neutral()),
        ]

        for label, action in script:
            print(f"-> {label}")
            action()
            lamp.step(1.0)  # settle, and give the eye time to read the pose

        print("done; holding the final pose")
        lamp.step(2.0)
    finally:
        if p.isConnected(client):
            p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without the GUI window")
    args = parser.parse_args()
    run(headless=args.headless)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
