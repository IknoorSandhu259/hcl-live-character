"""Body / execution layer for the dummy five-DOF character lamp.

This module is the *only* place that is allowed to issue joint commands to the
simulated robot. Higher level components (perception, language, planning) are
expected to call the named behaviours below ("engage", "look_left", "nod", ...)
or to hand this layer a named pose. They must not talk to PyBullet directly,
because every physical safety rule -- joint limits, force limits, velocity
limits -- lives here.

Coordinate notes for the supplied URDF (verified against the loaded model):

* At the zero pose the arm is vertical and the lamp head faces world -X,
  because ``head_pitch_joint`` carries an ``rpy="0 0 pi"`` origin.
* Positive ``base_yaw_joint`` / ``neck_yaw_joint`` therefore turn the head
  toward the robot's own LEFT (world -Y).
* Positive ``head_pitch_joint`` pitches the head DOWN.
* Positive ``shoulder_pitch_joint`` / ``elbow_pitch_joint`` lean the arm
  backwards (away from the direction the head faces).

All angles are radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional
import time

import pybullet as p

# --------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------

URDF_PATH = Path(__file__).resolve().parent.parent / "robot" / "dummy_lamp_5dof.urdf"

#: The joints this layer is allowed to actuate, in kinematic order. Everything
#: else in the URDF (lamp head, light emitter, camera, speaker) is a fixed
#: joint and is never commanded.
ACTUATED_JOINTS = (
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
)

#: Massless "semantic frame" links in the URDF. They exist only to mark where
#: the light, camera and speaker are; they have no <inertial> block, so
#: PyBullet gives each of them a default 1 kg. See _zero_semantic_frame_masses.
SEMANTIC_FRAME_LINKS = ("light_emitter_link", "camera_link", "speaker_link")

#: Keep commanded targets this far away from the URDF hard stops. The URDF also
#: declares ``safety_controller`` soft limits 0.10-0.15 rad inside the hard
#: limits; PyBullet does not expose those, so we apply our own margin in the
#: same spirit. Never larger than 10% of the joint's range.
LIMIT_MARGIN_RAD = 0.05

#: Simulation step. 240 Hz is PyBullet's default and keeps the position
#: controller stable at these gains.
TIME_STEP = 1.0 / 240.0

#: Fraction of each joint's URDF ``velocity`` limit we are willing to use.
#: Motion that stays well below the limit looks deliberate rather than snappy.
VELOCITY_SCALE = 0.5


@dataclass(frozen=True)
class JointSpec:
    """Physical constraints of one actuated joint, read from the loaded model."""

    name: str
    index: int
    lower: float
    upper: float
    max_force: float
    max_velocity: float

    def clamp(self, position: float) -> float:
        """Clamp *position* into the safe band inside the URDF limits."""
        margin = min(LIMIT_MARGIN_RAD, 0.1 * (self.upper - self.lower))
        return max(self.lower + margin, min(self.upper - margin, position))


# --------------------------------------------------------------------------
# Named poses
# --------------------------------------------------------------------------
#
# Poses are plain dictionaries of joint name -> angle, kept deliberately small
# and readable. Values were chosen inside the URDF limits (printed by
# LampController.describe()) and are clamped again at execution time.

Pose = Mapping[str, float]

POSES: Dict[str, Dict[str, float]] = {
    # Relaxed, slightly slumped, head tipped down: "idle / not paying attention".
    "neutral": {
        "base_yaw_joint": 0.0,
        "shoulder_pitch_joint": 0.25,
        "elbow_pitch_joint": -0.55,
        "neck_yaw_joint": 0.0,
        "head_pitch_joint": 0.35,
    },
    # Leaning in toward the person with the head brought back to level:
    # "I see you". head_pitch cancels the arm's forward tilt, because the head
    # elevation is (shoulder + elbow) - head_pitch.
    "engage": {
        "base_yaw_joint": 0.0,
        "shoulder_pitch_joint": -0.30,
        "elbow_pitch_joint": -0.15,
        "neck_yaw_joint": 0.0,
        "head_pitch_joint": -0.40,
    },
}

#: Head yaw used by look_left / look_right, applied on top of the engage pose.
#: Well inside neck_yaw_joint's +/-1.35 rad limit so the gesture reads as a
#: glance rather than a full turn.
LOOK_YAW = 0.85

#: Nod amplitude (down from the current head pitch) and number of dips.
NOD_AMPLITUDE = 0.28
NOD_CYCLES = 2


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_lamp(client_id: int) -> int:
    """Load the supplied lamp URDF with a fixed base and return its body id.

    Also repairs one loader artefact: see :data:`SEMANTIC_FRAME_LINKS`.
    """
    body_id = p.loadURDF(str(URDF_PATH), useFixedBase=True, physicsClientId=client_id)
    _zero_semantic_frame_masses(body_id, client_id)
    return body_id


def _zero_semantic_frame_masses(body_id: int, client_id: int) -> None:
    """Remove the phantom mass PyBullet invents for inertia-less links.

    ``light_emitter_link``, ``camera_link`` and ``speaker_link`` are pure
    semantic frames in the URDF and carry no ``<inertial>`` block. PyBullet
    substitutes mass = 1.0 kg for such links, which hangs 2 kg of phantom
    payload off the lamp head -- roughly 12 N*m at the shoulder, i.e. the whole
    of ``shoulder_pitch_joint``'s 12 N*m effort budget, so the arm collapses to
    its lower stop under gravity.

    We fix this at load time rather than editing the supplied URDF.
    """
    for index in range(p.getNumJoints(body_id, physicsClientId=client_id)):
        link_name = p.getJointInfo(body_id, index, physicsClientId=client_id)[12].decode()
        if link_name in SEMANTIC_FRAME_LINKS:
            p.changeDynamics(
                body_id,
                index,
                mass=0.0,
                localInertiaDiagonal=[0.0, 0.0, 0.0],
                physicsClientId=client_id,
            )


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


class LampController:
    """Joint-space execution boundary for the five-DOF lamp.

    Discovers the actuated joints by *name* (never by hardcoded index), reads
    their limits from the loaded model, and refuses to command anything else.
    """

    def __init__(self, body_id: int, client_id: int, realtime: Optional[bool] = None):
        self.body_id = body_id
        self.client_id = client_id
        self.joints = self._discover_joints()
        # Last commanded target per joint; the starting point of every motion.
        self._targets: Dict[str, float] = {
            name: spec.clamp(self._measured_position(spec)) for name, spec in self.joints.items()
        }
        if realtime is None:
            info = p.getConnectionInfo(physicsClientId=client_id)
            realtime = info.get("connectionMethod") == p.GUI
        self.realtime = bool(realtime)

    # -- discovery ---------------------------------------------------------

    def _discover_joints(self) -> Dict[str, JointSpec]:
        """Map the expected joint names onto the actually loaded model."""
        found: Dict[str, JointSpec] = {}
        for index in range(p.getNumJoints(self.body_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.body_id, index, physicsClientId=self.client_id)
            name = info[1].decode()
            if name not in ACTUATED_JOINTS:
                continue
            joint_type = info[2]
            if joint_type != p.JOINT_REVOLUTE:
                raise ValueError(
                    f"joint '{name}' is type {joint_type}, expected revolute; "
                    "refusing to actuate it"
                )
            lower, upper = info[8], info[9]
            if lower >= upper:
                raise ValueError(f"joint '{name}' has no usable range: [{lower}, {upper}]")
            found[name] = JointSpec(
                name=name,
                index=index,
                lower=lower,
                upper=upper,
                max_force=info[10],
                max_velocity=info[11],
            )

        missing = [name for name in ACTUATED_JOINTS if name not in found]
        if missing:
            raise ValueError(f"URDF is missing expected actuated joints: {missing}")
        # Preserve the declared order for readable logging.
        return {name: found[name] for name in ACTUATED_JOINTS}

    def _measured_position(self, spec: JointSpec) -> float:
        return p.getJointState(self.body_id, spec.index, physicsClientId=self.client_id)[0]

    def describe(self) -> str:
        lines = ["discovered actuated joints:"]
        for spec in self.joints.values():
            lines.append(
                f"  [{spec.index}] {spec.name}: "
                f"limits [{spec.lower:+.3f}, {spec.upper:+.3f}] rad, "
                f"max_force {spec.max_force:.1f} N*m, max_velocity {spec.max_velocity:.2f} rad/s"
            )
        return "\n".join(lines)

    # -- low level ---------------------------------------------------------

    def set_joint_target(self, name: str, position: float) -> float:
        """Command one joint. Unknown joints are rejected, targets are clamped.

        Returns the target that was actually sent.
        """
        if name not in self.joints:
            raise KeyError(
                f"'{name}' is not an actuated joint; allowed: {list(self.joints)}"
            )
        spec = self.joints[name]
        target = spec.clamp(float(position))
        p.setJointMotorControl2(
            bodyUniqueId=self.body_id,
            jointIndex=spec.index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=spec.max_force,
            maxVelocity=spec.max_velocity * VELOCITY_SCALE,
            physicsClientId=self.client_id,
        )
        self._targets[name] = target
        return target

    def step(self, seconds: float) -> None:
        """Advance the simulation, pacing to wall clock when a GUI is attached."""
        for _ in range(max(0, int(round(seconds / TIME_STEP)))):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.realtime:
                time.sleep(TIME_STEP)

    # -- motion ------------------------------------------------------------

    def move_to_pose(self, pose: Pose, duration: float = 1.2) -> None:
        """Interpolate from the current targets to *pose* over *duration*.

        Joints absent from *pose* hold their current target. The duration is
        stretched if any joint would otherwise have to exceed its usable
        velocity, so the URDF velocity limits are respected by construction as
        well as by the position controller.
        """
        start = dict(self._targets)
        goal = dict(start)
        for name, value in pose.items():
            if name not in self.joints:
                raise KeyError(
                    f"'{name}' is not an actuated joint; allowed: {list(self.joints)}"
                )
            goal[name] = self.joints[name].clamp(float(value))

        duration = max(duration, self._minimum_duration(start, goal))
        steps = max(1, int(round(duration / TIME_STEP)))
        for i in range(1, steps + 1):
            blend = _smoothstep(i / steps)
            for name in self.joints:
                self.set_joint_target(name, start[name] + blend * (goal[name] - start[name]))
            p.stepSimulation(physicsClientId=self.client_id)
            if self.realtime:
                time.sleep(TIME_STEP)

    def _minimum_duration(self, start: Mapping[str, float], goal: Mapping[str, float]) -> float:
        """Shortest duration that keeps every joint under its velocity budget.

        Smoothstep peaks at 1.5x the average rate, hence the factor.
        """
        needed = 0.0
        for name, spec in self.joints.items():
            travel = abs(goal[name] - start[name])
            budget = spec.max_velocity * VELOCITY_SCALE
            if budget > 0:
                needed = max(needed, 1.5 * travel / budget)
        return needed

    # -- named behaviours --------------------------------------------------

    def neutral(self, duration: float = 1.6) -> None:
        """Relaxed idle posture."""
        self.move_to_pose(POSES["neutral"], duration)

    def engage(self, duration: float = 1.2) -> None:
        """Sit up and face forward: the character has noticed someone."""
        self.move_to_pose(POSES["engage"], duration)

    def look_left(self, duration: float = 1.0) -> None:
        """Glance to the robot's own left (world -Y)."""
        self.move_to_pose({"neck_yaw_joint": LOOK_YAW}, duration)

    def look_right(self, duration: float = 1.0) -> None:
        """Glance to the robot's own right (world +Y)."""
        self.move_to_pose({"neck_yaw_joint": -LOOK_YAW}, duration)

    def nod(self, cycles: int = NOD_CYCLES, amplitude: float = NOD_AMPLITUDE) -> None:
        """Dip the head a few times around its current pitch, then restore it.

        Deliberately a small hand-written gesture rather than a general
        keyframe/animation system.
        """
        base = self._targets["head_pitch_joint"]
        for _ in range(max(1, cycles)):
            self.move_to_pose({"head_pitch_joint": base + amplitude}, 0.28)
            self.move_to_pose({"head_pitch_joint": base - amplitude * 0.4}, 0.28)
        self.move_to_pose({"head_pitch_joint": base}, 0.25)


def _smoothstep(t: float) -> float:
    """Ease-in / ease-out on [0, 1]; zero velocity at both ends."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
