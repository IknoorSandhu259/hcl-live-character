from pathlib import Path
import time

import pybullet as p


def main():
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet GUI")

    try:
        robot_path = Path(__file__).resolve().parent.parent / "robot" / "dummy_lamp_5dof.urdf"
        robot = p.loadURDF(str(robot_path), useFixedBase=True, physicsClientId=client)

        joint_types = {
            p.JOINT_REVOLUTE: "revolute",
            p.JOINT_PRISMATIC: "prismatic",
            p.JOINT_SPHERICAL: "spherical",
            p.JOINT_PLANAR: "planar",
            p.JOINT_FIXED: "fixed",
        }

        for index in range(p.getNumJoints(robot, physicsClientId=client)):
            info = p.getJointInfo(robot, index, physicsClientId=client)
            print(
                f"joint {index}: name={info[1].decode()}, "
                f"type={joint_types.get(info[2], info[2])}, "
                f"lower={info[8]}, upper={info[9]}, "
                f"max_force={info[10]}, max_velocity={info[11]}"
            )

        while p.isConnected(client):
            p.stepSimulation(physicsClientId=client)
            time.sleep(1.0 / 240.0)
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
